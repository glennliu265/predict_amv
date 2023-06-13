#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Compute Test Metrics

 - Test Accuracy
 - Loss by Epoch (Test)
 - 


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

#%% Load custom packages and setup parameters

machine = 'stormtrack' # Indicate machine (see module packages section in pparams)

# Import packages specific to predict_amv
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import train_cesm_params as train_cesm_params
import amv_dataloader as dl
import amvmod as am

# Load Predictor Information
bbox          = pparams.bbox

# Import general utilities from amv module
pkgpath = pparams.machine_paths[machine]['amv_path']
sys.path.append(pkgpath)
from amv import proc

# Import LRP package
lrp_path = pparams.machine_paths[machine]['lrp_path']
sys.path.append(lrp_path)
from innvestigator import InnvestigateModel

# Load ML architecture information
nn_param_dict      = pparams.nn_param_dict


# ============================================================
#%% User Edits vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ============================================================

# Set machine and import corresponding paths

# Set experiment directory/key used to retrieve params from [train_cesm_params.py]
expdir              = "FNN4_128_SingleVar_PaperRun_detrended"
eparams             = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Processing Options
even_sample         = False

# Get some paths
datpath             = pparams.datpath
dataset_name        = "CESM1"

# Set some looping parameters and toggles
varnames            = ["SSH","SST"]       # Names of predictor variables
leads               = np.arange(0,26,1)    # Prediction Leadtimes
runids              = np.arange(0,100,1)    # Which runs to do


# LRP Parameters
innexp         = 2
innmethod      ='b-rule'
innbeta        = 0.1


# Other toggles
checkgpu            = True                 # Set to true to check if GPU is availabl
debug               = True                 # Set verbose outputs
savemodel           = True                 # Set to true to save model weights

# Save looping parameters into parameter dictionary
eparams['varnames'] = varnames
eparams['leads']    = leads
eparams['runids']   = runids



# ============================================================
#%% Load the data 
# ============================================================
# Copied segment from train_NN_CESM1.py

# Load data + target
load_dict                      = am.prepare_predictors_target(varnames,eparams,return_nfactors=True,load_all_ens=True)
data                           = load_dict['data']
target_class                   = load_dict['target_class']


# Pick just the testing set
data                           = data[:,ens_test,...]
target_class                   = target_class[ens_test,:]

# Get necessary sizes
nchannels,nens,ntime,nlat,nlon = data.shape             
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN
nclasses                       = len(eparams['thresholds'])+1
nlead                          = len(leads)

# Count Samples...
am.count_samples(None,target_class)


# -----------------------------------
# %% Get some other needed parameters
# -----------------------------------

# Ensemble members
ens_all        = np.arange(0,42)
ens_train_val  = ens_all[:eparams['ens']]
ens_test       = ens_all[eparams['ens']:]
nens_test      = len(ens_test)

#%% 

"""

General Procedure

 1. Load data and subset to test set
 2. Looping by variable...
     3. Load the model weights and metrics
     4. 
     
"""

nvars = len(varnames)


#for v in range(nvars):
v       = 0
nr      = 0
runid   = runids[nr]
l       = -1
lead    = leads[l]
# -------------------- Loop, but debug first

vt      = time.time()
varname = varnames[v]

# ~~~~~~~
#% 1. Load model weights + Metrics
# ~~~~~~~
# Get the model weights [lead][run]
modweights_lead,modlist_lead=am.load_model_weights(datpath,expdir,leads,varname)
nmodels = len(modweights_lead[0])

# Get list of metric files
search = "%s%s/Metrics/%s" % (datpath,expdir,"*%s*" % varname)
flist  = glob.glob(search)
flist  = [f for f in flist if "of" not in f]
flist.sort()
print("Found %i files per lead for %s using searchstring: %s" % (len(flist),varname,search))

#
#% 2. Retrieve predictor and preallocate
#
lt = time.time()
predictors            = data[[v],...] # Get selected predictor

# Preallocate
total_acc_all         = np.zeros((nmodels,nlead))
class_acc_all         = np.zeros((nmodels,nlead,3)) # 

relevances_all        = [] # [nlead][nmodel][sample x lat x lon]
predictor_all         = [] # [nlead][sample x lat x lon]

predictions_all       = [] # [nlead][nmodel][sample]
targets_all           = [] # [nlead][sample]
#
#%% Loop by lead
# Note: Since the testing sample is the same withheld set for the experiment, we can use leadtime as the outer loop.

for l,lead in enumerate(leads):
    
    # -----------------------
    # Loop by Leadtime...
    # -----------------------
    outname = "/Test_Metrics_%s_%s_evensample%i.npz" % (dataset_name,varname,even_sample)
    
    # ===================================
    # I. Data Prep
    # ===================================
    
    # IA. Apply lead/lag to data
    # --------------------------
    # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
    X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=nens_test,tstep=ntime)
    
    # ----------------------
    # IB. Select samples
    # ----------------------
    _,class_count = am.count_samples(None,y_class)
    if even_sample:
        eparams['nsamples'] = int(np.min(class_count))
        print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
        y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle_class'])
    
    
    # ----------------------
    # IC. Flatten inputs for FNN
    # ----------------------
    if "FNN" in eparams['netname']:
        ndat,nchannels,nlat,nlon = X.shape
        inputsize                = nchannels*nlat*nlon
        outsize                  = nclasses
        X_in                     = X.reshape(ndat,inputsize)
    
    # -----------------------------
    # ID. Place data into a data loader
    # -----------------------------
    # Convert to Tensors
    X_torch = torch.from_numpy(X_in.astype(np.float32))
    y_torch = torch.from_numpy(y_class.astype(np.compat.long))
    
    # Put into pytorch dataloaders
    test_loader = DataLoader(TensorDataset(X_torch,y_torch), batch_size=eparams['batch_size'])
    
    # Preallocate
    predictor_lead   = []
    relevances_lead  = []
    
    predictions_lead = []
    targets_lead     = []
    
    # --------------------
    # 05. Loop by runid...
    # --------------------
    for nr,runid in tqdm(enumerate(runids)):
        rt = time.time()
        
        # =====================
        # II. Rebuild the model
        # =====================
        # Get the models (now by leadtime)
        modweights = modweights_lead[l][nr]
        modlist    = modlist_lead[l][nr]
        
        # Rebuild the model
        pmodel = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
        
        # Load the weights
        pmodel.load_state_dict(modweights)
        pmodel.eval()
        
        # =======================================================
        # III. Test the model separately to get accuracy by class
        # =======================================================
        y_predicted,y_actual,test_loss = am.test_model(pmodel,test_loader,eparams['loss_fn'],
                                                       checkgpu=checkgpu,debug=False)
        lead_acc,class_acc = am.compute_class_acc(y_predicted,y_actual,nclasses,debug=debug,verbose=False)
        
        # Save variables
        total_acc_all[nr,l]   = lead_acc
        class_acc_all[nr,l,:] = class_acc
        predictions_lead.append(y_predicted)
        if nr == 0:
            targets_all.append(y_actual)
        
        # ===========================
        # IV. Perform LRP
        # ===========================
        nsamples_lead = len(y_actual)
        inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
                                          method=innmethod,
                                          beta=innbeta)
        model_prediction, sample_relevances = inn_model.innvestigate(in_tensor=X_torch)
        model_prediction                    = model_prediction.detach().numpy().copy()
        sample_relevances                   = sample_relevances.detach().numpy().copy()
        if "FNN" in eparams['netname']:
            predictor_test    = X_torch.detach().numpy().copy().reshape(nsamples_lead,nlat,nlon)
            sample_relevances = sample_relevances.reshape(nsamples_lead,nlat,nlon) # [test_samples,lat,lon] 
        
        # Save Variables
        if nr == 0:
            predictor_all.append(predictor_test) # Predictors are the same across model runs
        relevances_lead.append(sample_relevances)
        
        # Clear some memory
        del pmodel
        torch.cuda.empty_cache()  # Save some memory
        
        #print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
        # End Lead Loop >>>
    relevances_all.append(relevances_lead)
    predictions_all.append(predictions_lead)
    print("\nCompleted training for %s lead %i of %i in %.2fs" % (varname,lead,leads[-1],time.time()-lt))
    


#%% Composite the relevances (can look at single events later, it might actually be easier to write a separate script for that)
# the purpose here is to get some quick, aggregate metrics

# Need to add option to cull models...

relevance_composites = np.zeros((nlead,nmodels,3,nlat,nlon)) * np.nan # [lead x model x class x lat x lon]
relevance_variances  = relevance_composites.copy()                    # [lead x model x class x lat x lon]
relevance_range      = relevance_composites.copy()                    # [lead x model x class x lat x lon]
predictor_composites = np.zeros((nlead,3,nlat,nlon)) * np.nan         # [lead x class x lat x lon]
predictor_variances  = predictor_composites.copy()                    # [lead x class x lat x lon]
ncorrect_byclass     = np.zeros((nlead,nmodels,3))                # [lead x model x class

for l in range(nlead):
    
    for nr in tqdm(range(nmodels)):
        
        predictions_model = predictions_all[l][nr] # [sample]
        relevances_model  = relevances_all[l][nr]  # [sample x lat x lon]
        
        for c in range(3):
            
            # Get correct indices
            class_indices                   = np.where(targets_all[l] == c)[0] # Sample indices of a particular class
            correct_ids                     = np.where(targets_all[l][class_indices] == predictions_model[class_indices])
            correct_pred_id                 = class_indices[correct_ids] # Correct predictions to composite over
            ncorrect                        = len(correct_pred_id)
            ncorrect_byclass[l,nr,c]        = ncorrect
            
            if ncorrect == 0:
                continue # Set NaN to model without any results
            # Make Composite
            correct_relevances               =  relevances_model[correct_pred_id,...]
            relevance_composites[l,nr,c,:,:] =  correct_relevances.mean(0)
            relevance_variances[l,nr,c,:,:]  =  correct_relevances.var(0)
            relevance_range[l,nr,c,:,:]      =  correct_relevances.max(0) - correct_relevances.min(0)
            
            # Make Corresponding predictor composites
            correct_predictors               = predictor_all[l][correct_pred_id,...]
            predictor_composites[l,c,:,:]    = correct_predictors.mean(0)
            predictor_variances[l,c,:,:]     = correct_predictors.var(0)
            

#%% Save output

"""
Save as a dataset
"""

lat = load_dict['lat']
lon = load_dict['lon']

# Save variables
save_vars      = [relevance_composites,relevance_variances,relevance_range,predictor_composites,predictor_variances,ncorrect_byclass]
save_vars_name = ['relevance_composites','relevance_variances','relevance_range','predictor_composites','predictor_variances',
                  'ncorrect_byclass']

# Make Coords
coords_relevances = {"lead":leads,"runid":runids,"class":pparams.classes,"lat":lat,"lon":lon}
coords_preds      = {"lead":leads,"class":pparams.classes,"lat":lat,"lon":lon}
coords_counts     = {"lead":leads,"runid":runids,"class":pparams.classes}

# Convert to dataarray and make encoding dictionaries
ds_all    = []
encodings = {}
for sv in range(len(save_vars)):
    
    svname = save_vars_name[sv]
    if "relevance" in svname:
        coord_in = coords_relevances
    elif "predictor" in svname:
        coord_in = coords_preds
    elif "ncorrect" in svname:
        coord_in = coords_counts
    
    da = xr.DataArray(save_vars[sv],dims=coord_in,coords=coord_in,name=svname)
    encodings[svname] = {'zlib':True}
    ds_all.append(da)
    
# Merge into dataset
ds_all = xr.merge(ds_all)

# Save Relevance data
outname    = "%s%s/Metrics/Test_Metrics_%s_%s_evensample%i_relevance_maps.nc" % (datpath,expdir,dataset_name,varname,even_sample)
ds_all.to_netcdf(outname,encoding=encodings)

#%% Try saving

"""

-rw-rw-r-- 1 glliu glliu  82K Jun 13 16:39 Test_Metrics_CESM1_SSH_evensample0_ncorrect_byclass.npy
-rw-rw-r-- 1 glliu glliu 2.7M Jun 13 16:39 Test_Metrics_CESM1_SSH_evensample0_predictor_variances.npy
-rw-rw-r-- 1 glliu glliu 2.7M Jun 13 16:39 Test_Metrics_CESM1_SSH_evensample0_predictor_composites.npy
-rw-rw-r-- 1 glliu glliu 267M Jun 13 16:39 Test_Metrics_CESM1_SSH_evensample0_relevance_range.npy
-rw-rw-r-- 1 glliu glliu 267M Jun 13 16:39 Test_Metrics_CESM1_SSH_evensample0_relevance_variances.npy
-rw-rw-r-- 1 glliu glliu 267M Jun 13 16:39 Test_Metrics_CESM1_SSH_evensample0_relevance_composites.npy

"""
save_vars      = [relevance_composites,relevance_variances,relevance_range,predictor_composites,predictor_variances,ncorrect_byclass]
save_vars_name = ['relevance_composites','relevance_variances','relevance_range','predictor_composites','predictor_variances',
                  'ncorrect_byclass']
for sv in range(len(save_vars)):
    outname    = "%s%s/Metrics/Test_Metrics_%s_%s_evensample%i_%s.npy" % (datpath,expdir,dataset_name,varname,even_sample,save_vars_name[sv])
    np.save(outname,save_vars[sv],allow_pickle=True)
#test_name = proc.addstrtoext(outname,"_"+save_vars_name[v])

#%% Try saving


"""

predictors            = data[[v],...] # Get selected predictor
total_acc_all         = np.zeros((nmodels,nlead))
class_acc_all         = np.zeros((nmodels,nlead,3)) # 

relevances_all        = []# [nmodel][nlead]
predictor_all         = []# [nmodel][nlead]

predictions_all       = []
targets_all           = []


Sizes:
-rw-rw-r-- 1 glliu glliu  62K Jun 13 13:20 Test_Metrics_CESM1_SSH_evensample0_class_acc.npy
-rw-rw-r-- 1 glliu glliu  15M Jun 13 13:23 Test_Metrics_CESM1_SSH_evensample0_predictions.npy
-rw-rw-r-- 1 glliu glliu  32G Jun 13 13:23 Test_Metrics_CESM1_SSH_evensample0_predictors.npy
-rw-rw-r-- 1 glliu glliu  32G Jun 13 13:21 Test_Metrics_CESM1_SSH_evensample0_relevances.npy
-rw-rw-r-- 1 glliu glliu  15M Jun 13 13:23 Test_Metrics_CESM1_SSH_evensample0_targets.npy
-rw-rw-r-- 1 glliu glliu  21K Jun 13 13:20 Test_Metrics_CESM1_SSH_evensample0_total_acc.npy

"""


save_vars      = [total_acc_all,class_acc_all,relevances_all,predictor_all,predictions_all,targets_all]
save_vars_name = ["total_acc","class_acc","relevances","predictors","predictions","targets"]


for sv in range(len(save_vars)):
    outname    = "%s%s/Metrics/Test_Metrics_%s_%s_evensample%i_%s.npy" % (datpath,expdir,dataset_name,varname,even_sample,save_vars_name[sv])
    np.save(outname,save_vars[sv],allow_pickle=True)
#test_name = proc.addstrtoext(outname,"_"+save_vars_name[v])



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





#%% 



# ------------------------------------------------------------
# %% Looping for runid
# ------------------------------------------------------------

# Print Message

# ------------------------
# 04. Loop by predictor...
# ------------------------


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
