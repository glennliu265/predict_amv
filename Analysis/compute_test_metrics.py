#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Compute Test Metrics

  - For a given experiment and variable, compute the test metrics
  - Save to an output file...

 Copied from test_cesm_witheld on Tue Jun 13 11:20AM

@author: gliu
"""


import numpy as np
import sys
import glob
import importlib
import copy
import xarray as xr

import torch
from torch import nn

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from tqdm import tqdm
import time
import os

from torch.utils.data import DataLoader, TensorDataset,Dataset
#%% Load some functions

#% Load custom packages and setup parameters
# Import general utilities from amv module
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import proc,viz


# Import packages specific to predict_amv
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import train_cesm_params as train_cesm_params
import amv_dataloader as dl
import amvmod as am


# Import LRP package
lrp_path = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/ml_demo/Pytorch-LRP-master/"
sys.path.append(lrp_path)
from innvestigator import InnvestigateModel

#%% User Edits


# Shared Information
varname            = "SST" # Testing variable
detrend            = False
leads              = np.arange(0,26,3)
region_name        = "NAT"
nsamples           = "ALL"
shuffle_trainsplit = False

# CESM1-trained model information
expdir             = "FNN4_128_SingleVar"
modelname          = "FNN4_128"
nmodels            = 50 # Specify manually how much to do in the analysis
eparams            = train_cesm_params.train_params_all[expdir] # Load experiment parameters
ens_wh             = [40,41] # Withheld member#eparams['ens']
runids             = np.arange(0,nmodels)



# Load parameters from [oredict_amv_param.py]
datpath            = pparams.datpath
figpath            = pparams.figpath
figpath            = pparams.figpath
nn_param_dict      = pparams.nn_param_dict
class_colors       = pparams.class_colors
classes            = pparams.classes
bbox               = pparams.bbox

#eparams['shuffle_trainsplit'] = False # Turn off shuffling

# Reanalysis dataset information
dataset_name       = "HadISST"
regrid             = "CESM1"


# LRP Parameters
innexp         = 2
innmethod      ='b-rule'
innbeta        = 0.1

# Other toggles
debug              = False
checkgpu           = True
darkmode           = False


if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
    transparent      = True
else:
    plt.style.use('default')
    dfcol = "k"
    transparent      = False


#%% Load the datasets

# Load reanalysis datasets [channel x ensemble x year x lat x lon]
re_data,re_lat,re_lon=dl.load_data_reanalysis(dataset_name,varname,bbox,
                        detrend=detrend,regrid=regrid,return_latlon=True)

# Load the target dataset
re_target = dl.load_target_reanalysis(dataset_name,region_name,detrend=detrend)
re_target = re_target[None,:] # ens x year

# Do further preprocessing and get dimensions sizes
re_data[np.isnan(re_data)]     = 0                      # NaN Points to Zero

inputsize                      = nchannels*nlat*nlon

#%% Load regular data... (as a comparison for debugging, can remove later)

# Loads that that has been preprocessed by: ___

# Load predictor and labels, lat/lon, cut region
target         = dl.load_target_cesm(detrend=eparams['detrend'],region=eparams['region'])
data,lat,lon   = dl.load_data_cesm([varname,],eparams['bbox'],detrend=eparams['detrend'],return_latlon=True)

# Subset predictor by ensemble, remove NaNs, and get sizes
data                           = data[:,ens_wh,...]      # Limit to Ens
target = target[ens_wh,:]
data[np.isnan(data)]           = 0                      # NaN Points to Zero
nchannels,nens,ntime,nlat,nlon = data.shape
#%% Make the classes from reanalysis data

# Set exact threshold value
std1         = target.std(1).mean() * eparams['thresholds'][1] # Multiple stdev by threshold value 
if eparams['quantile'] is False:
    thresholds_in = [-std1,std1]
else:
    thresholds_in = eparams['thresholds']
    
thresholds_in  = [-.36,.36]

# Classify AMV Events
target_class = am.make_classes(target.flatten()[:,None],thresholds_in,
                               exact_value=True,reverse=True,quantiles=eparams['quantile'])
target_class = target_class.reshape(target.shape)

# Get necessary dimension sizes/values
nclasses     = len(eparams['thresholds'])+1
nlead        = len(leads)

"""
# Output: 
    predictors :: [channel x ens x year x lat x lon]
    labels     :: [ens x year]
"""     

# ----------------------------------------------------
# %% Retrieve a consistent sample if the option is set
# ----------------------------------------------------


if shuffle_trainsplit is False:
    print("Pre-selecting indices for consistency")
    output_sample = am.consistent_sample(data,target_class,leads,nsamples,leadmax=leads.max(),
                          nens=len(ens_wh),ntime=ntime,
                          shuffle_class=eparams['shuffle_class'],debug=False)
    
    target_indices,target_refids,predictor_indices,predictor_refids = output_sample
else:
    target_indices     = None
    predictor_indices  = None
    target_refids      = None
    predictor_refids   = None


"""
Output

shuffidx_target  = [nsamples*nclasses,]        - Indices of target
predictor_refids = [nlead][nsamples*nclasses,] - Indices of predictor at each leadtime

tref --> array of the target years
predictor_refids --> array of the predictor refids
"""



#%% Load model weights 

# Get the model weights
modweights_lead,modlist_lead=am.load_model_weights(datpath,expdir,leads,varname)

# Get list of metric files
search = "%s%s/Metrics/%s" % (datpath,expdir,"*%s*" % varname)
flist  = glob.glob(search)
flist  = [f for f in flist if "of" not in f]
flist.sort()

print("Found %i files per lead for %s using searchstring: %s" % (len(flist),varname,search))
#%% 



# ------------------------------------------------------------
# %% Looping for runid
# ------------------------------------------------------------

# Print Message


# ------------------------
# 04. Loop by predictor...
# ------------------------
vt                    = time.time()
predictors            = data[[0],...] # Get selected predictor
total_acc_all         = np.zeros((nmodels,nlead))
class_acc_all         = np.zeros((nmodels,nlead,3)) # 

relevances_all        = []
predictor_all         = []

if shuffle_trainsplit:
    
    y_actual_all      = []
else:
    
    nsample_total     = len(target_indices)
    y_predicted_all   = np.zeros((nmodels,nlead,nsample_total))
    y_actual_all      = np.zeros((nlead,nsample_total))

# --------------------
# 05. Loop by runid...
# --------------------
for nr,runid in enumerate(runids):
    rt = time.time()
    
    # Preallocate Evaluation Metrics...
    # -----------------------
    # 07. Loop by Leadtime...
    # -----------------------
    outname = "/leadtime_testing_%s_%s_ALL.npz" % (varname,dataset_name)
    
    predictor_lead = []
    relevances_lead  = []
    for l,lead in enumerate(leads):
        
        if target_indices is None:
            # --------------------------
            # 08. Apply lead/lag to data
            # --------------------------
            # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
            X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=ens,tstep=ntime)
            
            # ----------------------
            # 09. Select samples
            # ----------------------
            if shuffle_trainsplit is False:
                if eparams['nsamples'] is None: # Default: nsamples = smallest class
                    threscount = np.zeros(nclasses)
                    for t in range(nclasses):
                        threscount[t] = len(np.where(y_class==t)[0])
                    eparams['nsamples'] = int(np.min(threscount))
                    print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
                y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle_class'])
            else:
                print("Select the sample samples")
                shuffidx = sampled_idx[l-1]
                y_class  = y_class[shuffidx,...]
                X        = X[shuffidx,...]
                am.count_samples(eparams['nsamples'],y_class)
            shuffidx = shuffidx.astype(int)
        else:
            print("Using preselected indices")
            pred_indices = predictor_indices[l]
            nchan        = predictors.shape[0]
            y_class      = target_class.reshape((ntime*nens,1))[target_indices,:]
            X            = predictors.reshape((nchan,nens*ntime,nlat,nlon))[:,pred_indices,:,:]
            X            = X.transpose(1,0,2,3) # [sample x channel x lat x lon]
            shuffidx     = target_indices    
        
        # ----------------------
        # Flatten inputs for FNN
        # ----------------------
        if "FNN" in eparams['netname']:
            ndat,nchannels,nlat,nlon = X.shape
            inputsize                = nchannels*nlat*nlon
            outsize                  = nclasses
            X_in                     = X.reshape(ndat,inputsize)
        
        # -----------------------------
        # Place data into a data loader
        # -----------------------------
        # Convert to Tensors
        X_torch = torch.from_numpy(X_in.astype(np.float32))
        y_torch = torch.from_numpy(y_class.astype(np.compat.long))
        
        # Put into pytorch dataloaders
        test_loader = DataLoader(TensorDataset(X_torch,y_torch), batch_size=eparams['batch_size'])
        
        
        #
        # Rebuild the model
        #
        # Get the models (now by leadtime)
        modweights = modweights_lead[l][nr]
        modlist    = modlist_lead[l][nr]
        
        # Rebuild the model
        pmodel = am.recreate_model(modelname,nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
        
        # Load the weights
        pmodel.load_state_dict(modweights)
        pmodel.eval()
        
        # ------------------------------------------------------
        # Test the model separately to get accuracy by class
        # ------------------------------------------------------
        y_predicted,y_actual,test_loss = am.test_model(pmodel,test_loader,eparams['loss_fn'],
                                                       checkgpu=checkgpu,debug=False)
        
        lead_acc,class_acc = am.compute_class_acc(y_predicted,y_actual,nclasses,debug=True,verbose=False)
        
        
        
        total_acc_all[nr,l]   = lead_acc
        class_acc_all[nr,l,:] = class_acc
        y_predicted_all[nr,l,:]   = y_predicted
        y_actual_all[l,:] = y_actual
        
        #
        # Perform LRP
        #
        nsamples_lead = len(shuffidx)
        inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
                                          method=innmethod,
                                          beta=innbeta)
        model_prediction, sample_relevances = inn_model.innvestigate(in_tensor=X_torch)
        model_prediction = model_prediction.detach().numpy().copy()
        sample_relevances = sample_relevances.detach().numpy().copy()
        if "FNN" in eparams['netname']:
            predictor_test    = X_torch.detach().numpy().copy().reshape(nsamples_lead,nlat,nlon)
            sample_relevances = sample_relevances.reshape(nsamples_lead,nlat,nlon) # [test_samples,lat,lon] 
        predictor_lead.append(predictor_test)
        relevances_lead.append(sample_relevances)
        
                
        
        # Clear some memory
        del pmodel
        torch.cuda.empty_cache()  # Save some memory
        
        print("\nCompleted training for %s lead %i of %i" % (varname,lead,leads[-1]))
        # End Lead Loop >>>
    predictor_all.append(predictor_lead)
    relevances_all.append(relevances_lead)
    print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
    # End Runid Loop >>>
#print("\nPredictor %s finished in %.2fs" % (varname,time.time()-vt))
# End Predictor Loop >>>

#print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))


#%% Perform LRP

#%% Prepare to do some visualization

# Load baselines
persleads,pers_class_acc,pers_total_acc = dl.load_persistence_baseline("CESM1",
                                                                        return_npfile=False,region=None,quantile=False,
                                                                        detrend=detrend,limit_samples=False,nsamples=nsamples,repeat_calc=1)


    
    
# persleads,pers_class_acc,pers_total_acc = dl.load_persistence_baseline(dataset_name,
#                                                                         return_npfile=False,region="NAT",quantile=False,
#                                                                         detrend=False,limit_samples=True,nsamples=None,repeat_calc=1)

# Load results from CESM1
#%%

fig,ax = plt.subplots(1,1)
for nr in range(nmodels):
    ax.plot(leads,total_acc_all[nr,:],alpha=0.1,color="g")
    
ax.plot(leads,total_acc_all.mean(0),color="green",label="CESM1-trained NN (SST)")
ax.plot(persleads,pers_total_acc,color="k",ls="dashed",label="Persistence Baseline")
ax.axhline([.33],color="gray",ls="dashed",lw=0.75,label="Random Chance Baseline")

ax.legend()
ax.grid(True,ls="dotted")
ax.set_xticks(persleads[::3])
ax.set_xlim([0,24])
ax.set_yticks(np.arange(0,1.25,0.25))
ax.set_xlabel("Prediction Lead (Years)")
ax.set_ylabel("Accuracy")
ax.set_title("Total Accuracy (HadISST Testing, %i samples per class)" % (nsample_total/3))
# 
figname = "%sReanalysis_Test_%s_Total_Acc.png" % (figpath,dataset_name)
plt.savefig(figname,dpi=150)
#%% 

fig,axs = plt.subplots(1,3,constrained_layout=True,figsize=(16,4))
for c in range(3):
    ax = axs[c]
    for nr in range(nmodels):
        ax.plot(leads,class_acc_all[nr,:,c],alpha=0.1,color=class_colors[c])
    ax.plot(leads,class_acc_all.mean(0)[...,c],color=class_colors[c],label="CESM1-trained NN (SST)")
    
    ax.plot(persleads,pers_class_acc[:,c],color="k",ls="dashed",label="Persistence Baseline")
    ax.axhline([.33],color="gray",ls="dashed",lw=2,label="Random Chance Baseline")
    
    if c == 1:
        ax.legend()
    ax.grid(True,ls="dotted")
    ax.set_xticks(persleads[::3])
    ax.set_xlim([0,24])
    ax.set_xlabel("Prediction Lead (Years)")
    ax.set_ylabel("Accuracy")
    ax.set_yticks(np.arange(0,1.25,0.25))
    ax.set_title(classes[c])
    ax.minorticks_on()
figname = "%sReanalysis_Test_%s_Class_Acc.png" % (figpath,dataset_name)
plt.savefig(figname,dpi=150)

#%% Visualizet he class distribution

idx_by_class,count_by_class = am.count_samples(None,target_class)

class_str = "Class Count: AMV+ (%i) | Neutral (%i) | AMV- (%i)" % tuple(count_by_class)

timeaxis = np.arange(0,re_target.shape[1]) + 1870
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))

ax.plot(timeaxis,target.squeeze(),color="k",lw=2.5)
ax.grid(True,ls="dashed")
ax.minorticks_on()

for th in thresholds_in:
    ax.axhline([th],color="k",ls="dashed")
ax.axhline([0],color="k",ls="solid",lw=0.5)
ax.set_xlim([timeaxis[0],timeaxis[-1]])
ax.set_title("CESM1WH NASST Index (1870-2022) \n%s" % (class_str))
plt.savefig("%sCESM1WH_NASST.png" %(figpath),dpi=150,bbox_inches='tight')

#%% Get correct indices for each class


# y_predicted_all = [runs,lead,sample]
# y_actual_all    = [lead,sample]

correct_mask = []
for l in range(len(leads)):
    lead = leads[l]
    y_preds   = y_predicted_all[:,l,:] # [runs lead sample]
    i_correct = (y_preds == y_actual_all[l,:][None,:]) # Which Predictions are correct
    correct_mask_lead = []
    for c in range(3):
        i_class = (y_actual_all[l,:] == c)
        correct_mask_lead.append(i_correct*i_class)
    correct_mask.append(correct_mask_lead)

#%% Visualize relevance maps

relevances_all = np.array(relevances_all)
predictor_all = np.array(predictor_all)
nruns,nleads,nsamples_lead,nlat,nlon = relevances_all.shape

plotleads        = [24,18,12,6,0]
normalize_sample = 2

plot_bbox        = [-80,0,0,60]

cmax  = 1
clvl = np.arange(-2.2,2.2,0.2)

fsz_title        = 20
fsz_axlbl        = 18
fsz_ticks        = 16


fig,axs  = plt.subplots(3,len(plotleads),constrained_layout=True,figsize=(18,10),
                        subplot_kw={'projection':ccrs.PlateCarree()})


ii = 0
for c in range(3):
    for l in range(len(plotleads)):
        
        ax = axs.flatten()[ii]
        lead  = plotleads[l]
        ilead = list(leads).index(lead) 
        
        # Axis Formatting
        blabel = [0,0,0,0]
        if c == 0:
            ax.set_title("%s-Year Lead" % (plotleads[l]),fontsize=fsz_title)
        if l == 0:
            blabel[0] = 1
            ax.text(-0.15, 0.55, classes[c], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_axlbl)
        ax = viz.add_coast_grid(ax,bbox=plot_bbox,blabels=blabel,fill_color="k")
        ax = viz.label_sp(ii,ax=ax,fig=fig,alpha=0.8,fontsize=fsz_axlbl)
            
        # Get correct predictions
        cmask = correct_mask[l][c].flatten()
        relevances_in = relevances_all[:,ilead,:,:,:]
        newshape      = (np.prod(relevances_in.shape[:2]),) + (nlat,nlon)
        # Apprently using cmask[:,...] brocasts, while cmask[:,None,None] doesn't
        relevances_sel = relevances_in.reshape(newshape)[cmask[:,...]] # [Samples x Lat x Lon]
        
        predictor_in   = predictor_all[:,ilead,:,:,:]
        predictor_sel = predictor_in.reshape(newshape)[cmask[:,...]] # [Samples x Lat x Lon]
        if normalize_sample == 1:
            relevances_sel = relevances_sel / np.abs(relevances_sel.max(0))[None,...]
        
        
        # Plot the results
        plotrel = relevances_sel.mean(0)
        plotvar = predictor_sel.mean(0)
        if normalize_sample == 2:
            plotrel = plotrel/np.max(np.abs(plotrel))
            
        # Set Land Points to Zero
        plotrel[plotrel==0] = np.nan
        plotvar[plotrel==0] = np.nan
        
            
        # Do the plotting
        pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r")
        cl = ax.contour(lon,lat,plotvar,levels=clvl,colors="k",linewidths=0.75)
        ax.clabel(cl,clvl[::2])
        
            
        ii+=1
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.025,pad=0.01)
cb.set_label("Normalized Relevance",fontsize=fsz_axlbl)
cb.ax.tick_params(labelsize=fsz_ticks)


savename = "%sHadISSTClassComposites_LRP_%s_normalize%i_Outline.png" % (figpath,expdir,normalize_sample)
if darkmode:
    savename = proc.addstrtoext(savename,"_darkmode")
plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=transparent)

#%% Make a scatterplot of the event distribution and 

imodel = 6
ilead  = 8
msize  = 100
timeaxis = np.arange(0,re_target.shape[1]) + 1870

for imodel in range(50):
    # Select the model
    y_predicted_in = y_predicted_all[imodel,ilead,:]
    y_actual_in    = y_actual_all[ilead,:]
    re_target_in   = re_target[:,leads[ilead]:].squeeze()
    id_correct     = (y_predicted_in == y_actual_in)
    
    
    timeaxis_in = np.arange(leads[ilead],re_target.shape[1]) + 1870
    
    
    
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))
    
    # Plot the amv classes
    for c in range(3):
        
        # Get the id for the class
        id_class = (y_actual_in == c)
        
        id_right = id_class * id_correct
        id_wrong = id_class * ~id_correct
        
        # Plot the correct ones
        ax.scatter(timeaxis_in[id_right],re_target_in[id_right],s=msize,marker="o",color=class_colors[c],facecolors="None")
        ax.scatter(timeaxis_in[id_wrong],re_target_in[id_wrong],s=msize,marker="x",color=class_colors[c])
        
    
    # Plot the actual AMV Index
    #ax.plot(timeaxis,re_target.squeeze(),color="k",lw=0.75,zorder=-9)
    ax.grid(True,ls="dashed")
    ax.minorticks_on()
    
    # Plot the Thresholds
    for th in thresholds_in:
        ax.axhline([th],color="k",ls="dashed")
    ax.axhline([0],color="k",ls="solid",lw=0.5)
    ax.set_xlim([timeaxis[0],timeaxis[-1]])
    
    class_str = "Class Acc: AMV+ (%.2f), Neutral (%.2f), AMV- (%.2f)" % (class_acc_all[imodel,ilead,0],
                                                                         class_acc_all[imodel,ilead,1],
                                                                         class_acc_all[imodel,ilead,2])
    ax.set_title("HadISST NASST Index and Prediction Results (1870-2022) \nNetwork #%i, Lead = %i years \n %s" % (imodel+1,leads[ilead],class_str))
    plt.savefig("%sHadISST_NASST_lead%02i_imodel%03i.png" %(figpath,leads[ilead],imodel,),dpi=150,bbox_inches='tight')




#%% Function version of above
def plot_scatter_predictions(imodel,ilead,y_predicted_all,y_actual_all,re_target,class_acc_all,msize=100,
                             figsize=(12,4),class_colors=('salmon', 'gray', 'cornflowerblue')):
    
    
    # Select the model
    y_predicted_in = y_predicted_all[imodel,ilead,:]
    y_actual_in    = y_actual_all[ilead,:]
    re_target_in   = re_target[:,leads[ilead]:].squeeze()
    id_correct     = (y_predicted_in == y_actual_in)
    
    timeaxis_in = np.arange(leads[ilead],re_target.shape[1]) + 1870
    
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))
    
    # Plot the amv classes
    for c in range(3):
        
        # Get the id for the class
        id_class = (y_actual_in == c)
        
        id_right = id_class * id_correct
        id_wrong = id_class * ~id_correct
        
        # Plot the correct ones
        ax.scatter(timeaxis_in[id_right],re_target_in[id_right],s=msize,marker="o",color=class_colors[c],facecolors="None")
        ax.scatter(timeaxis_in[id_wrong],re_target_in[id_wrong],s=msize,marker="x",color=class_colors[c])
        
    
    # Plot the actual AMV Index
    #ax.plot(timeaxis,re_target.squeeze(),color="k",lw=0.75,zorder=-9)
    ax.grid(True,ls="dashed")
    ax.minorticks_on()
    
    # Plot the Thresholds
    for th in thresholds_in:
        ax.axhline([th],color="k",ls="dashed")
    ax.axhline([0],color="k",ls="solid",lw=0.5)
    ax.set_xlim([timeaxis[0],timeaxis[-1]])
    
    class_str = "Class Acc: AMV+ (%.2f), Neutral (%.2f), AMV- (%.2f)" % (class_acc_all[imodel,ilead,0],
                                                                         class_acc_all[imodel,ilead,1],
                                                                         class_acc_all[imodel,ilead,2])
    return fig,ax

    
#%% MAKE A PLOT OF ABOVE, BUT WITH THE BEST performing model

ilead   = -1
id_best = total_acc_all[:,ilead].argmax()


fig,ax = plot_scatter_predictions(id_best,ilead,y_predicted_all,y_actual_all,re_target,class_acc_all,msize=100,
                             figsize=(12,4))

ax.set_ylim([-1.5,1.5])
ax.set_xlim([1890,2025])
ax.set_title("HadISST NASST Index and Prediction Results (1870-2022) \nNetwork #%i, Lead = %i years \n %s" % (id_best+1,leads[ilead],class_str))
plt.savefig("%sHadISST_NASST_lead%02i_imodel%03i.png" %(figpath,leads[ilead],imodel,),dpi=150,bbox_inches='tight')



#%% Make a histogram


# Visualize prediction count by year

# Select the model
#y_predicted_in = y_predicted_all[imodel,ilead,:]
#y_actual_in    = y_actual_all[ilead,:]
#re_target_in   = re_target[:,leads[ilead]:].squeeze()
#id_correct     = (y_predicted_in == y_actual_in)


count_by_year = np.zeros((ntime-leads[-1],nclasses))
timeaxis_in   = np.arange(leads[ilead],target.shape[1])

y_predicted_all = y_predicted_all.reshape(y_predicted_all.shape[0],nlead,nens,ntime-leads[-1])

# Assumes leads are not shuffled
for e in range(nens):
    for y in range(ntime-leads[ilead]):
        y_pred_year = y_predicted_all[...,e,y]
        
        for c in range(3):
            
            count_by_year[y,c] = (y_pred_year == c).sum()

#%% Barplot of Year vs Prediction frequency
# for c in range(3):
#     y_predicted_all == 
#     y_predicted_all
    

fig,ax       = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))

for c in range(3):
    label = classes[c]
    #label = "%s (Test Acc = %.2f" % (classes[c],class_acc[c]*100)+"%)"
    
    ax.bar(timeaxis_in+1920,count_by_year[:,c],bottom=count_by_year[:,:c].sum(1),
           label=label,color=class_colors[c],alpha=0.75,edgecolor="white")

ax.set_ylabel("Frequency of Predicted Class")
ax.set_xlabel("Year")

ax.minorticks_on()
ax.grid(True,ls="dotted")
ax.set_xlim([1920+20,2010])
#ax.set_ylim([0,450])

ax2    = ax.twinx()
ls_ens = ["solid","dashdot"]
for e in range(nens):
    ax2.plot(timeaxis+1920,target[e,:].squeeze(),color="k",ls=ls_ens[e],label="NASST Index, ens%02i"%(e+1))
ax2.set_ylabel("NASST Index ($\degree C$)")
ax2.set_ylim([-1.3,1.3])
for th in thresholds_in:
    ax2.axhline([th],color="k",ls="dashed")
ax2.axhline([0],color="k",ls="solid",lw=0.5)
ax.legend()
ax2.legend(ncol=2)
plt.savefig("%sCESMWH_Prediction_Count_AllLeads.png"%figpath,dpi=150,bbox_inches="tight")
#%% Try the above, but get prediction count for selected leadtimes
# Q : Is there a systematic shift towards the selected leadtimes?
selected_leads      = [0,6,12,18,24]
nleads_sel          = len(selected_leads)

count_by_year_leads = np.zeros((ntime-leads[-1],nclasses,nleads_sel))

# Assumes leads are not shuffled
for y in range(ntime-leads[ilead]):
    
    for ll in range(nleads_sel):
        sel_lead_index = list(leads).index(selected_leads[ll])
        y_pred_year = y_predicted_all[...,sel_lead_index,y]
    
        for c in range(3):
            
            count_by_year_leads[y,c,ll] = (y_pred_year == c).sum()


#%% 
fig,axs       = plt.subplots(3,1,constrained_layout=True,figsize=(16,8))



lead_colors = ["lightsteelblue","cornflowerblue","royalblue","mediumblue","midnightblue"]
for c in range(3):
    ax = axs[c]
    
    for ll in range(nleads_sel):
        ax.plot(timeaxis_in+1870,count_by_year_leads[:,c,ll],label="%02i-yr Lead" % selected_leads[ll],lw=1.5,c=lead_colors[ll])
        
    if c == 0:
        ax.legend()
    ax.set_title(classes[c])
    
    ax.set_xlabel("Year")
    ax.minorticks_on()
    ax.grid(True,ls="dashed")
    
    # label = "%s (Test Acc = %.2f" % (classes[c],class_acc[c]*100)+"%)"
    # ax.bar(timeaxis_in+1870,count_by_year[:,c],bottom=count_by_year[:,:c].sum(1),
    #        label=label,color=class_colors[c],alpha=0.75,edgecolor="k")
    
    ax.set_ylabel("Predicted Class Count")

plt.savefig("%sHadISST_Class_Prediction_Frequency_byYear.png"%(figpath),dpi=150,bbox_inches="tight")


#%% Remake barplot. but for the selected leadtimes
def make_count_barplot(count_by_year,lead,leadmax=24,classes=['AMV+', 'Neutral', 'AMV-'],
                       class_colors=('salmon', 'gray', 'cornflowerblue')
                       ):
    
    timeaxis      = np.arange(0,len(re_target.squeeze()))
    timeaxis_in   = np.arange(leadmax,re_target.shape[1])
    
    fig,ax       = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))
    for c in range(3):
        label = classes[c]
        ax.bar(timeaxis_in+1870,count_by_year[:,c],bottom=count_by_year[:,:c].sum(1),
               label=label,color=class_colors[c],alpha=0.75,edgecolor="white")
    
    ax.set_ylabel("Frequency of Predicted Class")
    ax.set_xlabel("Year")
    ax.legend()
    ax.minorticks_on()
    ax.grid(True,ls="dotted")
    ax.set_xlim([1880,2025])
    ax.set_ylim([0,450])

    ax2 = ax.twinx()
    ax2.plot(timeaxis,re_target.squeeze(),color="k",label="HadISST NASST Index")
    ax2.set_ylabel("NASST Index ($\degree C$)")
    ax2.set_ylim([-1.3,1.3])
    for th in thresholds_in:
        ax2.axhline([th],color="k",ls="dashed")
    ax2.axhline([0],color="k",ls="solid",lw=0.5)
    axs = [ax,ax2]
    return fig,axs


for ll in range(nleads_sel):
    lead = selected_leads[ll]
    ilead = list(leads).index(lead)
    

    fig,axs = make_count_barplot(count_by_year_leads[:,:,ll],lead,re_target,)

    plt.savefig("%sHadISST_Prediction_Count_Lead%02i.png"% (figpath,lead),dpi=150,bbox_inches="tight")
    


#%%

ax.set_title(title)
ax.set_ylim([0,10])
plot_mode = 0

for plot_mode in range(2):
    ax = axs[plot_mode]
    ax = format_axis(ax,x=timeaxis)
    if plot_mode == 0:
        title = "Actual Class"
    elif plot_mode == 1:
        title = "Predicted Class"
    testc = np.arange(0,3)
    for c in range(3):
        label = "%s (Test Acc = %.2f" % (class_names[c],class_acc[c]*100)+"%)"
        if debug:
            print("For c %i, sum of prior values is %s" % (c,testc[:c]))
        ax.bar(timeaxis,count_by_year[:,c,plot_mode],bottom=count_by_year[:,:c,plot_mode].sum(1),
               label=label,color=class_colors[c],alpha=0.75,edgecolor="k")
    ax.set_title(title)
    ax.set_ylim([0,10])
    if plot_mode == 0:
        ax.legend()
plt.suptitle("AMV Class Distribution by Year (%s) \n %s" % (modelname,exp_titlestr))
if savefig:
    plt.savefig("%sClass_Distr_byYear_%s_lead%02i_nepochs%02i.png" % (figpath,varnames[v],lead,epoch_axis[-1]),dpi=150)
