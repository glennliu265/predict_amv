#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

LRP by Predictor

Visualize LRP Maps by Predictor for a given experiment. Made for the GRL Outline.

Copied [viz_reigonal_predictability]

Created on Wed Apr 12 06:09:48 2023

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

#%% Load custom packages

# LRP Methods
sys.path.append("/Users/gliu/Downloads/02_Research/03_Code/github/Pytorch-LRP-master/")
from innvestigator import InnvestigateModel

# Load modules (LRPutils by Peidong)
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/scrap/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/predict_amv/")
import LRPutils as utils

# Load visualization module
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz,proc

# Load parameter files
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import amvmod as am
import amv_dataloader as dl
import train_cesm_params as train_cesm_params
import pamv_visualizer as pviz

# Load relevant variables from parameter files
bboxes  = pparams.bboxes
regions = pparams.regions
rcolors = pparams.rcolors

classes = pparams.classes
proj    = pparams.proj
bbox    = pparams.bbox

datpath = pparams.datpath
figpath = pparams.figpath
proc.makedir(figpath)

# Load model_dict
nn_param_dict = pparams.nn_param_dict

#%% # User Edits

# Indicate settings (Network Name)

# Data and variable settings
varnames       = pparams.varnames #("SST","SSH","SSS","SLP")
varnames_plot  = pparams.varnames #("SST","SSH","SSS","SLP")
expdir         = "FNN4_128_SingleVar_PaperRun"
eparams        = train_cesm_params.train_params_all[expdir]

leads          = np.arange(0,26,1)

no_val         = True # Set to True for old training set, which had no validation
nmodels        = 100 # Specify manually how much to do in the analysis

# Compositing options
topN           = 50 # Top models to include

#modelname     = "FNN4_128"
#leads         = np.arange(0,25,3)
#nleads        = len(leads)
#detrend       = False
#region_subset = ["NAT",]

# LRP Settings (note, this currently uses the innvestigate package from LRP-Pytorch)
gamma          = 0.1
epsilon        = 0.1
innexp         = 1
innmethod      ='e-rule'
innbeta        = 0.5
innepsi        = 1e-2

# Labeling for plots and output files
ge_label       = "method=%s, exp=%i, $beta$=%.02f, $epsilon$=%.2e" % (innmethod,innexp,innbeta,innepsi)
ge_label_fn    = "%s_exp%.2e_beta%.02f_epsi%.2i" % (innmethod[0],innexp,innbeta,innepsi)

use_shuffidx   = True  # Use shuffidx to index correct testing set
old_shuffling  = True  # Use old script method (just split directly)

# Other Toggles
darkmode  = False
debug     = True

# Data Settings
#regrid         = None
#quantile       = False
#ens            = 40
#tstep          = 86
#percent_train  = 0.8              # Percentage of data to use for training (remaining for testing)
#detrend        = 0
#bbox           = [-80,0,0,65]
#thresholds     = [-1,1]
#outsize        = len(thresholds) + 1

# # Region Settings
    
# Plotting Settings
#classes   = ["AMV+","Neutral","AMV-"] # [Class1 = AMV+, Class2 = Neutral, Class3 = AMV-]
#proj      = ccrs.PlateCarree()

# Dark mode settings

if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
else:
    plt.style.use('default')
    dfcol = "k"
    


# Other settings
nclasses = 3

#%% Load the data and target (copied from [test_predictor_uncertainty.py] on 2023.04.12)

# Load predictor and labels, lat/lon, cut region
target                          = dl.load_target_cesm(detrend=eparams['detrend'],region=eparams['region'],newpath=True)
data_all,lat,lon                = dl.load_data_cesm(varnames,eparams['bbox'],detrend=eparams['detrend'],return_latlon=True,newpath=True)

# Apply Preprocessing
target_all                      = target[:eparams['ens'],:]
data_all                        = data_all[:,:eparams['ens'],:,:,:]
nchannels,nens,ntime,nlat,nlon  = data_all.shape

# Make land mask
data_mask = np.sum(data_all,(0,1,2))
data_mask[~np.isnan(data_mask)] = 1
if debug:
    plt.pcolormesh(data_mask),plt.colorbar()

# Remove all NaN points
data_all[np.isnan(data_all)]    = 0

# Get Sizes
nchannels                       = 1 # Change to 1, since we are just processing 1 variable at a time
inputsize                       = nchannels*nlat*nlon    # Compute inputsize to remake FNN
nclasses                        = len(eparams['thresholds']) + 1
nlead                           = len(leads)

# Create Classes
std1         = target.std(1).mean() * eparams['thresholds'][1] # Multiple stdev by threshold value 
if eparams['quantile'] is False:
    thresholds_in = [-std1,std1]
else:
    thresholds_in = eparams['thresholds']

# Classify AMV Events
target_class = am.make_classes(target.flatten()[:,None],thresholds_in,exact_value=True,reverse=True,quantiles=eparams['quantile'])
target_class = target_class.reshape(target.shape)



#%% Load the relevance composites, from [compute_test_metrics.py]
nvars       = len(varnames)
nleads      = len(leads)
metrics_dir = "%s%s/Metrics/Test_Metrics/" % (datpath,expdir)
pcomps   = []
rcomps   = []
ds_all   = []
acc_dict = []
for v in range(nvars):
    # Load the composites
    varname = varnames[v]
    ncname = "%sTest_Metrics_CESM1_%s_evensample0_relevance_maps.nc" % (metrics_dir,varname)
    ds     = xr.open_dataset(ncname)
    #ds_all.append(ds)
    rcomps.append(ds['relevance_composites'].values)
    pcomps.append(ds['predictor_composites'].values)
    
    # Load the accuracies
    ldname  = "%sTest_Metrics_CESM1_%s_evensample0_accuracy_predictions.npz" % (metrics_dir,varname)
    npz     = np.load(ldname,allow_pickle=True)
    expdict = proc.npz_to_dict(npz)
    acc_dict.append(expdict)

nleads,nruns,nclasses,nlat,nlon=rcomps[v].shape
lon = ds.lon.values
lat = ds.lat.values


#%% Composite topN composites

# Get accuracy by class [var][run x lead x class]
class_accs  = [acc_dict[v]['class_acc'] for v in range(nvars)]
rcomps_topN = np.zeros((nvars,nleads,nclasses,nlat,nlon))

for v in range(nvars):
    for l in tqdm(range(nleads)):
        for c in range(nclasses):
            
            # Get ranking of models by test accuracy
            acc_list = class_accs[v][:,l,c] # [runs]
            id_hi2lo  = np.argsort(acc_list)[::-1] # Reverse to get largest value first
            id_topN   = id_hi2lo[:topN]
            
            # Make composite 
            rcomp_in  = rcomps[v][l,id_topN,c,:,:] # [runs x lat x lon]
            rcomps_topN[v,l,c,:,:] = rcomp_in.mean(0) # Mean along run dimension


#%% Make plot of selected leadtimes (copied from below), note this only plots the FIRST 4 Variables

# Set darkmode
darkmode = False
if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
    transparent      = True
else:
    plt.style.use('default')
    dfcol = "k"
    transparent      = False

#Same as above but reduce the number of leadtimes
plot_bbox        = [-80,0,0,60]
leadsplot        = [25,20,10,5,0]

normalize_sample = 2 # 0=None, 1=samplewise, 2=after composite
absval           = False
cmax             = 1
cmin             = 1
clvl             = np.arange(-2.1,2.1,0.3)
no_sp_label      = True
fsz_title        = 20
fsz_axlbl        = 18
fsz_ticks        = 16
cmap='cmo.balance'

for c in range(3): # Loop for class
    ia = 0
    fig,axs = plt.subplots(4,5,figsize=(24,16),
                           subplot_kw={'projection':proj},constrained_layout=True)
    # Loop for variable
    for v,varname in enumerate(varnames):
        # Loop for leadtime
        for l,lead in enumerate(leadsplot):
            
            # Get lead index
            id_lead    = list(leads).index(lead)
            
            if debug:
                print("Lead %02i, idx=%i" % (lead,id_lead))
            
            # Axis Formatting
            ax = axs[v,l]
            blabel = [0,0,0,0]
            
            #ax.set_extent(plot_bbox)
            #ax.coastlines()
            
            if v == 0:
                ax.set_title("Lead %02i Years" % (leads[id_lead]),fontsize=fsz_title)
            if l == 0:
                blabel[0] = 1
                ax.text(-0.15, 0.55, varnames_plot[v], va='bottom', ha='center',rotation='vertical',
                        rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_axlbl)
            if v == (len(varnames)-1):
                blabel[-1]=1
            
            ax = viz.add_coast_grid(ax,bbox=plot_bbox,blabels=blabel,fill_color="k")
            if no_sp_label is False:
                ax = viz.label_sp(ia,ax=ax,fig=fig,alpha=0.8,fontsize=fsz_axlbl)
            # -----------------------------
            
            # --------- Composite the Relevances and variables
            plotrel = rcomps_topN[v,id_lead,c,:,:]
            if normalize_sample == 2:
                plotrel = plotrel/np.max(np.abs(plotrel))
            plotvar = pcomps[v][id_lead,c,:,:]
            #plotvar = plotvar/np.max(np.abs(plotvar))
            
            
            # Set Land Points to Zero
            plotrel[plotrel==0] = np.nan
            plotvar[plotrel==0] = np.nan
            
            # Do the plotting
            pcm=ax.pcolormesh(lon,lat,plotrel*data_mask,vmin=-cmin,vmax=cmax,cmap=cmap)
            cl = ax.contour(lon,lat,plotvar*data_mask,levels=clvl,colors="k",linewidths=0.75)
            ax.clabel(cl,clvl[::2])
            ia += 1
            # Finish Leadtime Loop (Column)
        # Finish Variable Loop (Row)
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.025,pad=0.01)
    cb.set_label("Normalized Relevance",fontsize=fsz_axlbl)
    cb.ax.tick_params(labelsize=fsz_ticks)
    
    #plt.suptitle("Mean LRP Maps for Predicting %s using %s, \n Composite of Top %02i FNNs per leadtime" % (classes[c],varname,topN,))
    savename = "%sPredictorComparison_LRP_%s_%s_top%02i_normalize%i_abs%i_%s_Draft2.png" % (figpath,expdir,classes[c],topN,normalize_sample,absval,ge_label_fn)
    if darkmode:
        savename = proc.addstrtoext(savename,"_darkmode")
    plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=transparent)


#%% Make a plot for a specific variable

v = varnames.index('SSS')

# Set darkmode
darkmode = False
if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
    transparent      = True
else:
    plt.style.use('default')
    dfcol = "k"
    transparent      = False

#Same as above but reduce the number of leadtimes
plot_bbox        = [-80,0,0,60]
leadsplot        = np.arange(25,-1,-5)

normalize_sample = 2 # 0=None, 1=samplewise, 2=after composite
absval           = False
cmax             = 1
cmin             = 1
clvl             = np.arange(-2.1,2.1,0.3)
no_sp_label      = True
fsz_title        = 20
fsz_axlbl        = 18
fsz_ticks        = 16
cmap='cmo.balance'


fig,axs = plt.subplots(3,6,figsize=(24,12),
                       subplot_kw={'projection':proj},constrained_layout=True)
    
for c in range(3): # Loop for class
    ia = 0
    
    varname = varnames[v]


    # Loop for leadtime
    for l,lead in enumerate(leadsplot):
        
        # Get lead index
        id_lead    = list(leads).index(lead)
        
        if debug:
            print("Lead %02i, idx=%i" % (lead,id_lead))
        
        # Axis Formatting
        ax = axs[c,l]
        blabel = [0,0,0,0]
        
        #ax.set_extent(plot_bbox)
        #ax.coastlines()
        
        if c == 0:
            ax.set_title("Lead %02i Years" % (leads[id_lead]),fontsize=fsz_title)
        if l == 0:
            blabel[0] = 1
            ax.text(-0.15, 0.55, classes[c], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_axlbl)
        if c == (len(varnames)-1):
            blabel[-1]=1
        
        ax = viz.add_coast_grid(ax,bbox=plot_bbox,blabels=blabel,fill_color="k")
        if no_sp_label is False:
            ax = viz.label_sp(ia,ax=ax,fig=fig,alpha=0.8,fontsize=fsz_axlbl)
        # -----------------------------
        
        # --------- Composite the Relevances and variables
        plotrel = rcomps_topN[v,id_lead,c,:,:]
        if normalize_sample == 2:
            plotrel = plotrel/np.max(np.abs(plotrel))
        plotvar = pcomps[v][id_lead,c,:,:]
        #plotvar = plotvar/np.max(np.abs(plotvar))
        
        
        # Set Land Points to Zero
        plotrel[plotrel==0] = np.nan
        plotvar[plotrel==0] = np.nan
        
        # Do the plotting
        pcm=ax.pcolormesh(lon,lat,plotrel*data_mask,vmin=-cmin,vmax=cmax,cmap=cmap)
        cl = ax.contour(lon,lat,plotvar*data_mask,levels=clvl,colors="k",linewidths=0.75)
        ax.clabel(cl,clvl[::2])
        ia += 1
        # Finish Leadtime Loop (Column)
        
# Finish Variable Loop (Row)
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.025,pad=0.01)
cb.set_label("Normalized Relevance",fontsize=fsz_axlbl)
cb.ax.tick_params(labelsize=fsz_ticks)
plt.suptitle("Relevance Maps for %s" % varname,fontsize=fsz_title)
#plt.suptitle("Mean LRP Maps for Predicting %s using %s, \n Composite of Top %02i FNNs per leadtime" % (classes[c],varname,topN,))
savename = "%sPredictorComparison_LRP_AllClasses_%s_%s_top%02i_normalize%i_abs%i_%s_Draft2.png" % (figpath,expdir,varname,topN,normalize_sample,absval,ge_label_fn)
if darkmode:
    savename = proc.addstrtoext(savename,"_darkmode")
plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=transparent)



#%% Section below is the old script, where the relevance is explicilty calculated-----------------

#%% Quick sanity check

if debug:
    
    fig,axs = pviz.init_classacc_fig(leads)
    for c in range(3):
        ax = axs[c]
        for v,varname in enumerate(varnames):
            plotacc = np.array(classacc[v])[:,:,c].mean(0) # nrun, nlead, nclass
            ax.plot(leads,plotacc,label=varname,lw=2)
    ax.legend()

#%%

# A simple wrapper function
def prep_traintest_classification_new(predictor,target_class,lead,eparams,
                                      nens=42,ntime=86):
    
       # Apply Lead Lag (Might have to move this further within the loop...)
       X,y_class = am.apply_lead(predictor,target_class,lead,reshape=True,ens=nens,tstep=ntime)
       
       # Flatten input data for FNN
       # if "FNN" in eparams['netname']:
       #     ndat,nchannels,nlat,nlon = X.shape
       #     inputsize                = nchannels*nlat*nlon
       #     X                        = X.reshape(ndat,inputsize)
               
       
       # ------------------------
       # 10. Train/Test/Val Split
       # ------------------------
       X_subsets,y_subsets = am.train_test_split(X,y_class,eparams['percent_train'],
                                                      percent_val=eparams['percent_val'],
                                                      debug=False,offset=eparams['cv_offset'])
       
       # Convert to Tensors
       X_subsets = [torch.from_numpy(X.astype(np.float32)) for X in X_subsets]
       y_subsets = [torch.from_numpy(y.astype(np.compat.long)) for y in y_subsets]
       
       
       return X_subsets,y_subsets


    
    


#%% Obtain validation LRP Maps for each Region [Copied many sections from test_predictor_uncertainty.py]
# Currently takes ~74.51s to run...

st               = time.time()
# Preallocate
relevances_all   = {} # [variable][lead][model x sample x inputsize ]
factivations_all = {} # [variable][lead][model x sample x class]
idcorrect_all    = {} # [variable][lead][class][model][ids]
modelacc_all     = {} # [variable][lead][model x class]
labels_all       = {} # [variable][lead][samplesize]

for v,varname in enumerate(varnames): # Training data does not change for region
    
    # Get the data
    predictor         = data_all[[v],...]
    
    # Preallocate
    relevances_lead   = []
    factivations_lead = []
    idcorrect_lead    = []
    modelacc_lead     = []
    labels_lead       = []
    print("Computing Relevances for %s" % varname)
    
    for l,lead in enumerate(leads_sel): # Training data does chain with leadtime
        
        id_lead = list(leads).index(lead)
        if debug:
            print("\nLead is %i, idx = %i" % (leads[id_lead],id_lead))
        
        # Apply Lead Lag (Might have to move this further within the loop...)
        X,y_class = am.apply_lead(predictor,target_class,lead,reshape=True,ens=nens,tstep=ntime)
        
        # ----------------------------
        # Flatten input data for FNN
        if "FNN" in eparams['netname']:
            ndat,nchannels,nlat,nlon = X.shape
            inputsize                = nchannels*nlat*nlon
            outsize                  = nclasses
            X                        = X.reshape(ndat,inputsize)
                
        
        # ------------------------
        # 10. Train/Test/Val Split
        # ------------------------
        X_subsets,y_subsets = am.train_test_split(X,y_class,eparams['percent_train'],
                                                       percent_val=eparams['percent_val'],
                                                       debug=False,offset=eparams['cv_offset'])
        
        # --------------------------
        
        # Convert to Tensors
        X_subsets = [torch.from_numpy(X.astype(np.float32)) for X in X_subsets]
        y_subsets = [torch.from_numpy(y.astype(np.compat.long)) for y in y_subsets]
        
        if eparams['percent_val'] > 0:
            X_train,X_test,X_val = X_subsets
            y_train,y_test,y_val = y_subsets
        else:
            X_train,X_test       = X_subsets
            y_train,y_test       = y_subsets
        
        # # Put into pytorch dataloaders
        # data_loaders = [DataLoader(TensorDataset(X_subsets[iset],y_subsets[iset]), batch_size=eparams['batch_size']) for iset in range(len(X_subsets))]
        # if eparams['percent_val'] > 0:
        #     train_loader,test_loader,val_loader = data_loaders
        #     val_id = 1
        # else:
        #     train_loader,test_loader, = data_loaders
        #     val_id = 1
        
        # Preallocate, compute relevances
        valsize      = X_test.shape[0] # Take last element (test set ...)
        relevances   = np.zeros((nmodels,valsize,inputsize))*np.nan # [model x sample x inputsize ]
        factivations = np.zeros((nmodels,valsize,3))*np.nan         # [model x sample x 3]
        for m in range(nmodels):
            
            # Get List of Models
            modlist = modlist_byvar[v][id_lead][m]
            modweights = modweights_byvar[v][id_lead][m]
            
            # Get sampled indices for entire set
            #sampleids = (shuffids[v][nm][l]).astype(int)
            
            # ----------------- Section from LRP_LENs
            # Rebuild the model
            pmodel = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
            
            # Load the weights
            pmodel.load_state_dict(modweights)
            pmodel.eval()
            # ----------------- ----------------------
            
            # Investigate
            inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,epsilon=innepsi,
                                  method=innmethod,
                                  beta=innbeta)
            
            input_data                       = X_test.squeeze().float()
            model_prediction, true_relevance = inn_model.innvestigate(in_tensor=input_data)
            relevances[m,:,:]                = true_relevance.detach().numpy().copy()
            factivations[m,:,:]              = model_prediction.detach().numpy().copy()
        
        # Reshape Output
        relevances = relevances.reshape(nmodels,valsize,nchannels,nlat,nlon)
        y_pred     = np.argmax(factivations,2)
        y_test     = y_test.numpy()
        
        # Compute accuracy
        modelacc  = np.zeros((nmodels,3)) # [model x class]
        modelnum  = np.arange(nmodels)+1 
        idcorrect = []
        for c in range(3):
            
            # Compute accuracy
            class_id           = np.where(y_test == c)[0]
            pred               = y_pred[:,class_id]
            targ               = y_test[class_id,:].squeeze()
            correct            = (targ[None,:] == pred)
            num_correct        = correct.sum(1)
            num_total          = correct.shape[1]
            modelacc[:,c]      = num_correct/num_total
            meanacc            = modelacc.mean(0)[c]
            
            # Get indices of correct predictions
            corrid = []
            for zz in range(nmodels):
                corrid.append(class_id[correct[zz,:]])
            idcorrect.append(corrid)
        
        # Append for each leadtime
        relevances_lead.append(relevances)
        factivations_lead.append(factivations)
        idcorrect_lead.append(idcorrect)
        modelacc_lead.append(modelacc)
        labels_lead.append(y_test)
        
    # Assign to dictionary
    relevances_all[varname]   = relevances_lead
    factivations_all[varname] = factivations_lead
    idcorrect_all[varname]    = idcorrect_lead
    modelacc_all[varname]     = modelacc_lead
    labels_all[varname]       = labels_lead

print("Computed relevances in %.2fs" % (time.time()-st))
# XXXXXXXXXXXx XXXXXXXXXXXx XXXXXXXXXXXx XXXXXXXXXXXx XXXXXXXXXXXx XXXXXXXXXXXx
  
#%% Plot reduced nubber of leadtimes, for several variables (GRL Outline plot version)

# Set darkmode
darkmode = False
if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
    transparent      = True
else:
    plt.style.use('default')
    dfcol = "k"
    transparent      = False

#Same as above but reduce the number of leadtimes
plot_bbox        = [-80,0,0,60]
leadsplot        = [24,18,12,6,0]
topN             = 25
normalize_sample = 2 # 0=None, 1=samplewise, 2=after composite
absval           = False

cmax             = 1

clvl             = np.arange(-2.2,2.2,0.2)

fsz_title        = pparams.fsz_title
fsz_axlbl        = pparams.fsz_axlbl
fsz_ticks        = pparams.fsz_ticks

#cmax            = 0.5

# Loop for each class
for c in range(3):
    ia = 0
    fig,axs = plt.subplots(4,5,figsize=(24,16),
                           subplot_kw={'projection':proj},constrained_layout=True)
    # Loop for variable
    for v,varname in enumerate(varnames):
        # Loop for leadtime
        for l,lead in enumerate(leadsplot):
            
            # Get lead index
            id_lead    = list(leads_sel).index(lead)
            if debug:
                print("Lead %02i, idx=%i" % (lead,id_lead))
            
            # Axis Formatting
            ax = axs[v,l]
            blabel = [0,0,0,0]
            
            #ax.set_extent(plot_bbox)
            #ax.coastlines()
            
            if v == 0:
                ax.set_title("Lead %02i Years" % (leads_sel[id_lead]),fontsize=fsz_title)
            if l == 0:
                blabel[0] = 1
                ax.text(-0.15, 0.55, varnames_plot[v], va='bottom', ha='center',rotation='vertical',
                        rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_axlbl)
            if v == (len(varnames)-1):
                blabel[-1]=1
            ax = viz.add_coast_grid(ax,bbox=plot_bbox,blabels=blabel,fill_color="k")
            ax = viz.label_sp(ia,ax=ax,fig=fig,alpha=0.8,fontsize=fsz_axlbl)
            # -----------------------------
            
            # --------- Composite the Relevances and variables
            
            # Get variable
            X_subsets,y_subsets = prep_traintest_classification_new(data_all[[v],...],
                                                                    target_class,lead,
                                                                    eparams,nens=nens,ntime=ntime)
            
            if eparams['percent_val'] > 0:
                X_train,X_test,X_val = X_subsets
                y_train,y_test,y_val = y_subsets
            else:
                X_train,X_test       = X_subsets
                y_train,y_test       = y_subsets
            X_test = X_test.numpy()
    
            
            # Get indices of the top 10 models
            acc_in = modelacc_all[varname][id_lead][:,c] # [model x class]
            idtopN = am.get_topN(acc_in,topN,sort=True)
            
            # Get the plotting indices
            id_plot = np.array(idcorrect_all[varname][id_lead][c])[idtopN] # Indices to composite
            
            # Composite the relevances/variables
            plotrel = np.zeros((nlat,nlon)) # Relevances
            plotvar = np.zeros((nlat,nlon)) # Predictor
            
            for NN in range(topN):
                
                relevances_sel = relevances_all[varname][id_lead][idtopN[NN],id_plot[NN],:,:,:].squeeze()
                var_sel        = X_test[id_plot[NN],:,:,:].squeeze()
                
                if (relevances_sel.shape[0] == 0) or (var_sel.shape[0]==0):
                    continue
                
                if normalize_sample == 1: # Normalize each sample
                    relevances_sel = relevances_sel / np.max(np.abs(relevances_sel),0)[None,...]
                
                if absval:
                    relevances_sel = np.abs(relevances_sel)
                    
                plotrel += relevances_sel.mean(0) # Take mean of samples and add to relevance
                plotvar += np.nanmean(var_sel,0)
            
            plotrel /= topN # Divide by # of models considered (top N)
            plotvar /= topN
            if normalize_sample == 2:
                plotrel = plotrel/np.max(np.abs(plotrel))
            plotvar = plotvar/np.max(np.abs(plotvar))
                
            # Set Land Points to Zero
            plotrel[plotrel==0] = np.nan
            plotvar[plotrel==0] = np.nan
            
            # Do the plotting
            pcm=ax.pcolormesh(lon,lat,plotrel*data_mask,vmin=-cmax,vmax=cmax,cmap="RdBu_r")
            cl = ax.contour(lon,lat,plotvar*data_mask,levels=clvl,colors="k",linewidths=0.75)
            ax.clabel(cl,clvl[::2])
            ia += 1
            # Finish Leadtime Loop (Column)
        # Finish Variable Loop (Row)
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.025,pad=0.01)
    cb.set_label("Normalized Relevance",fontsize=fsz_axlbl)
    cb.ax.tick_params(labelsize=fsz_ticks)
    
    #plt.suptitle("Mean LRP Maps for Predicting %s using %s, \n Composite of Top %02i FNNs per leadtime" % (classes[c],varname,topN,))
    savename = "%sPredictorComparison_LRP_%s_%s_top%02i_normalize%i_abs%i_%s_Outline.png" % (figpath,expdir,classes[c],topN,normalize_sample,absval,ge_label_fn)
    if darkmode:
        savename = proc.addstrtoext(savename,"_darkmode")
    plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=transparent)
    # Finish class loop (Fig)
#%% Write a function to retrieve the year and ensemble of labels, given shuffidx from the old script

v    = 0
nmod = 0
l    = 8

# Required inputs (debugging)
lead          = leads[l]
percent_train = eparams['percent_train']
percent_val   = eparams['percent_val']
shuffid_in    = shuffids[v][nmod][l,:] # Samples (all)
offset        = 0
ens           = eparams['ens']
nyr           = 86

# Inputs: lead, shuffid, percent_train, percent_val, offset
# make some dummy variables



def retrieve_ensyr_shuffid(lead,shuffid_in,percent_train,percent_val=0,
                           offset=0,ens=42,nyr=86):
    """
    Retrieve the linear indices, ensemble, and year labels given a set
    of shuffled indices from select_sample, accounting for offset and leadtimes.
    

    Parameters
    ----------
    lead : INT. Leadtime applied in units of provided time axis.
    shuffid_in : ARRAY. Shuffled indices to subset, select, and retrieve indices from.
    percent_train : NUMERIC. % data used for training
    percent_val : NUMERIC. % data used for validation. optional, default is 0.
    offset : NUMERIC. Offset to shift train.test.val split optional, default is 0.
    ens : INT. Number of ensemble members to include optional, The default is 42.
    nyr : INT, Numer of years to include. optional, the default is 86.

    Returns
    -------
    shuffid_split : LIST of ARRAYS [train/test/val][samples]
        Shuffids partitioned into each set
    refids_linear_split : LIST of ARRAYS [train/test/val][samples]
        Corresponding linear indices to unlagged data (ens x year)
    refids_label_split : LIST of ARRAYS  [train/test/val][samples,[ens,yr]]
        Ensemble and Year arrays for each of the corresponding splits

    """
    
    # Apply Train/Test/Validation Split, accounting for offset
    dummyX                  = shuffid_in[:,None]
    dumX,dumY,split_indices = am.train_test_split(dummyX,dummyX,percent_train,percent_val=percent_val,debug=True,offset=offset,return_indices=True)
    shuffid_split           = [shuffid_in[split_id] for split_id in split_indices]

    # Get the actual ens and year (and corresponding linear id)
    refids_label_split  = []
    refids_linear_split = []
    for ii in range(len(shuffid_split)):
        shuffi = shuffid_split[ii].astype(int)
        
        ref_linearids,refids_label=am.get_ensyr_linear(lead,shuffi,
                      reflead=0,nens=ens,nyr=nyr,
                      apply_lead=True,ref_lead=True,
                      return_labels=True,debug=True,return_counterpart=False)
        
        # Convert to Array
        refids_label = np.array([[rf[0],rf[1]] for rf in refids_label]) # [sample, [ens,yr]]
        
        # Append
        refids_linear_split.append(ref_linearids)
        refids_label_split.append(refids_label)
    return shuffid_split,refids_linear_split,refids_label_split

#%% Make Histogram Plots of predictions vs. year, similar to analysis with HadISST for AMV Outline
# Copied from test_reanalysis.py on 2023.05.04

nvar           = len(factivations_all.keys())
test_refids    =  np.empty((nvar,nlead,nmod),dtype='object')# [predictor x model x lead x class]
test_ensyr     =  test_refids.copy() # 

train_refids   = test_refids.copy()
train_ensyr    = test_ensyr.copy()

# Assumes leads are not shuffled, loop for each variable
for v,varname in enumerate(varnames):
    
    #refids_lead = []
    #ensyr_lead  = []
    # Loop for leadtime
    for l,lead in tqdm(enumerate(leads)):
        
        ilead         = list(leads).index(lead)
        
        # This is for the processing above
       # y_predictions = np.argmax(factivations_all[varname][l],2) # [Model, Sample]
        #y_actual      = labels_all[varname][l] # [Sample x 1]
        
        
        # This is not for the processing above
        
        
        for nm in range(nmod):
            
            y_predictions = ypred[v][nm][ilead,:]
            y_actual      = ylabs[v][nm][ilead,:]
            
            
            # Get sample indexes and ensemble/year information ---------------
            shuffid_in      = shuffids[v][nm][ilead,:] # [Samples]
            
            # Retrieve Linear Indices and Ens/Year of testing set...
            output_split = am.retrieve_ensyr_shuffid(lead,shuffid_in,eparams['percent_train'],
                                                  percent_val=eparams['percent_val'],
                                                  offset=eparams['cv_offset'],ens=eparams['ens'],nyr=ntime)
            shuffid_split,refids_linear,refids_label = output_split
            
            # Save information from the testing set
            test_refids[v,l,nm] = refids_linear[1]
            test_ensyr[v,l,nm]  = refids_label[1]
            
            train_refids[v,l,m] = refids_linear[0]
            train_ensyr[v,l,nm] = refids_label[0]

            #refids_mod.append(refids_linear[1]) # [sample,]
            #ensyr_mod.append(refids_label[1]) # [sample,(ens,yr)]
            # ----------------------------------------------------------------
            
            
 
            
            # End model loop
        #refids_lead.append(refids_mod)
        #ensyr_mod.append()
        # End leadtime loop
        

#%% Compute the years 


# assume for now that you can covert everyhting into an array
#test_ensyr = np.array(test_ensyr.flatten())

num_peryear         = np.zeros((nvar,ntime)) # [predictor time]
count_by_year       = np.zeros((nvar,ntime,nclasses)) # [predictor x time x class]
count_by_year_label = count_by_year.copy()

timeaxis_in    = np.arange(leads[-1],target.shape[1])
timeaxis       = np.arange(0,target.shape[1])

count_by_year_ens       = np.zeros((nvar,ntime,nclasses,nens))
count_by_year_ens_label = count_by_year_ens.copy()
num_peryear_ens         = np.zeros((nvar,ntime,nens))

# Do a stupid loop...
for v in range(nvar):
    for l in range(nlead):
        for nm in tqdm(range(nmod)):
            
            # Get predictions for that model/leadtime/predictor
            test_years       = test_ensyr[v,l,m][:,1] # [Sample]
            test_ens         = test_ensyr[v,l,m][:,0] # [Sample]
            test_predictions = ypred[v][nm][l]
            test_labels      = ylabs[v][nm][l]
            
            # Loop for each year
            for y in range(len(timeaxis)):
                
                y_sel   = timeaxis[y]
                # Get boolean indices and restrict predictions
                id_year = (test_years == y_sel)
                
                # Do the ensemble aggregate
                predictions_ysel = test_predictions[id_year].astype(int)
                labels_ysel      = test_labels[id_year].astype(int)
                num_peryear[v,y] += len(predictions_ysel)
                for c in range(3): # Tally up the predicted classes
                    count_by_year[v,y,c]       += (predictions_ysel == c).sum()
                    count_by_year_label[v,y,c] += (labels_ysel == c).sum()
                    # End Class Loop
                    
                # Count individuall for ensemble member
                for e in range(nens):
                    id_ens = (test_ens == e)
                    predictions_y_ens_sel = test_predictions[id_year*id_ens].astype(int)
                    predictions_y_ens_sel_label = test_labels[id_year*id_ens].astype(int)
                    num_peryear_ens[v,y,e] += len(predictions_y_ens_sel)
                    for c in range(3): # Tally up the predicted classes
                        count_by_year_ens[v,y,c,e] += (predictions_y_ens_sel == c).sum()
                        count_by_year_ens_label[v,y,c,e] += (predictions_y_ens_sel_label == c).sum()
                        # End Class Loop (ens)
                    # End Ens Loop
                # End Year Loop
            # End Model Loop
        # End Lead Loop
    # End Prediction Loop
                
#%% Make the plot

plot_ens        = 8 #Non # Set to "ALL"" to plot all "Actual" to plot real data
normalize_count = True

if plot_ens == "ALL":
    num_peryear_in   = num_peryear
    count_by_year_in = count_by_year
    plot_target      = target.mean(0)
elif plot_ens == "Actual":
    num_peryear_in   = num_peryear
    count_by_year_in = count_by_year_label
    plot_target      = target.mean(0)
else:
    num_peryear_in   = num_peryear_ens[:,:,plot_ens]
    count_by_year_in = count_by_year_ens[:,:,:,plot_ens]
    plot_target      = target[plot_ens,:]
    
if normalize_count:
    ylab = "Percentage of Predictions"
    count_in = count_by_year_in / num_peryear_in[:,:,None]
    #ylim = [0,1]
else:
    ylab = "Frequency of Predicted Class"
    count_in = count_by_year_in.copy()
    ylim = [0,2500]
for v in range(nvar):
    fig,axs=pviz.make_count_barplot(count_in[v,...],lead,target,thresholds_in,leadmax=0,classes=classes,
                           class_colors=pparams.class_colors,startyr=1920)
    
    ax,ax2=axs
    ax.set_xlim([1915,2010])
    #ax.plot(timeaxis+1920,num_peryear[v,:],color="magenta")
    
    #ax2 = ax.twinx()
    ax2.plot(np.arange(0,ntime)+1920,plot_target,c="k")
    for th in range(2):
        ax2.axhline([thresholds_in[th]],ls="dashed",color="k",lw=0.95)
    ax2.axhline([0],ls="solid",color="k",lw=0.75,label="Ensemble Average NASST")
    ax2.legend(loc="upper right")
    
    #ax.set_ylim(ylim)
    ax.set_title("Class Prediction Count by Year for %s, Experiment: %s" % (varnames[v],expdir))
    ax.set_ylabel(ylab)
    savename = "%sCount_by_year_%s_%s_normalize%i_ens%s.png" % (figpath,expdir,varnames[v],
                                                          normalize_count,plot_ens+1)
    plt.savefig(savename,dpi=150)
    
#%% Debug check to make sure full proportion is equivalent

if debug:
    test1 = count_by_year.sum(-1)
    test2 = test1/num_peryear
    print(test2) # Should be 1...


#%% Check the testing dataset



#%%
#%% Plot Colorbar

fig,axs = plt.subplots(4,5,figsize=(8,6.5),
                       subplot_kw={'projection':proj},constrained_layout=True)

for r,region in enumerate(regions):
    
    for i in range(len(leadsplot)):
        
        lead = leadsplot[i]
        print(lead)
        l    = list(leads).index(lead)
        print(l)
        
        ### Leads are all wrong need to fix it
        ax = axs[r,i]
        for r,region in enumerate(regions):
            
            for i in range(len(leadsplot)):
                
                lead = leadsplot[i]
                print(lead)
                l    = list(leads).index(lead)
                print(l)
                
                ### Leads are all wrong need to fix it
                ax = axs[r,i]
                
                ax.set_extent(bbox)
                

fig.colorbar(pcm,ax=axs[0,-1])
savename = "%sRegional_LRP_AGU_%s_Colorbar.png" % (figpath,varname,)
plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=True)

#%% Visualize test accuracy by class and by region

fig,axs = plt.subplots(1,3,figsize=(16,4),constrained_layout=True)

for c in range(3):
    ax = axs[c]
    ax.set_title(classes[c])
    
    for v,varname in enumerate(varnames):
        
        accsbylead = np.array(modelacc_all[region]) # lead x model x class
        
        for m in range(nmodels):
            ax.plot(leads,accsbylead[:,m,c],color=rcolors[r],alpha=0.05,label="")
        ax.plot(leads,accsbylead[:,:,c].mean(1),color=rcolors[r],alpha=1,label=region)
        ax.legend()
    
    ax.set_xticks(leads)
    ax.set_xlim([leads[0],leads[-1]])
    ax.grid(True,ls='dotted')
    ax.set_ylim(0,1.0)

plt.suptitle("Mean Accuracy for Regional Prediction Target, \n Model: %s,Predictor: %s" % (modelname,varname))
savename = "%s%s_%s_meanacc_byregion.png" %(figpath,modelname,varname,)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%%

# -----------------------------------------------------------
# Visualize mean relevance for top N samples of a given class
# -----------------------------------------------------------

c                = 0  # Class
topN             = 25 # Top 10 models
normalize_sample = 2 # 0=None, 1=samplewise, 2=after composite
absval           = False
cmax             = 1

for topN in np.arange(5,55,5):
    fig,axs = plt.subplots(4,9,figsize=(16,6.5),
                           subplot_kw={'projection':proj},constrained_layout=True)
    for r,region in enumerate(regions):
        
        for l,lead in enumerate(lead_sel):
            ax = axs[r,l]
            
            # Get indices of the top 10 models
            acc_in = modelacc_all[region][l][:,c] # [model x class]
            idtopN = am.get_topN(acc_in,topN,sort=True)
            
            # Get the plotting variables
            id_plot = np.array(idcorrect_all[region][l][c])[idtopN] # Indices to composite
            
            plotrel = np.zeros((nlat,nlon))
            for NN in range(topN):
                relevances_sel = relevances_all[region][l][idtopN[NN],id_plot[NN],:,:,:].squeeze()
                
                if normalize_sample == 1:
                    relevances_sel = relevances_sel / np.max(np.abs(relevances_sel),0)[None,...]
                
                if absval:
                    relevances_sel = np.abs(relevances_sel)
                plotrel += relevances_sel.mean(0)
            plotrel /= topN
                
            if normalize_sample == 2:
                plotrel = plotrel/np.max(np.abs(plotrel))
                
            pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r")
            
            # Plot bounding box
            viz.plot_box(bboxes[r],ax=ax,linewidth=0.5)
            
            # Do Plotting Business and labeling
            if r == 0:
                ax.set_title("Lead %i" % (lead))
            if l == 0:
                ax.text(-0.05, 0.55, region, va='bottom', ha='center',rotation='vertical',
                        rotation_mode='anchor',transform=ax.transAxes)
            ax.set_extent(bbox)
            ax.coastlines()
            
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05)
    cb.set_label("Normalized Relevance")
    plt.suptitle("Mean LRP Maps for predicting %s using %s, %s, \n Top %02i Models (%s)" % (classes[c],varname,modelname,topN,ge_label))
    savename = "%sRegional_LRP_%s_%s_top%02i_normalize%i_abs%i_%s.png" % (figpath,varname,classes[c],topN,normalize_sample,absval,ge_label_fn)
    plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Visualize mean relevances for topN models, looping by leadtime
c                = 0
normalize_sample = 2# 0=None, 1=samplewise, 2=after composite

absval           = False
cmax             = 0.50

for l,lead in enumerate(leads):
    
    
    fig,axs = plt.subplots(4,5,figsize=(9,6.5),
                           subplot_kw={'projection':proj},constrained_layout=True)
    for r,region in enumerate(regions):
        
        # Get indices of the top 10 models
        acc_in = modelacc_all[region][l][:,c] # [model x class]
        idtopN = am.get_topN(acc_in,5,sort=True)
        
        # Get the plotting variables
        id_plot = np.array(idcorrect_all[region][l][c])[idtopN] # Indices to composite
        
        for nn in range(5):
            ax = axs[r,nn]
            
            accplot = acc_in[idtopN[nn]] * 100
            
            # Get relevances for selected model and take sample mean
            relevances_sel = relevances_all[region][l][idtopN[nn],id_plot[nn],:,:,:].squeeze()
            if normalize_sample == 1:
                relevances_sel = relevances_sel / np.max(np.abs(relevances_sel),0)[None,...]
            if absval:
                relevances_sel = np.abs(relevances_sel)
            
            plotrel = relevances_sel.mean(0)
            if normalize_sample == 2:
                plotrel = plotrel/np.max(np.abs(plotrel))
            
            pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r")
            
            # Labeling
            viz.label_sp("%.1f" % accplot + "%",ax=ax,labelstyle="%s",
                         usenumber=True,fontsize=9,x=-.05,alpha=0.5)
            
            # Plot bounding box
            viz.plot_box(bboxes[r],ax=ax,linewidth=0.5)
                     
            if r == 0:
                ax.set_title("Rank %i" % (nn+1))
            if nn == 0:
                ax.text(-0.05, 0.55, region, va='bottom', ha='center',rotation='vertical',
                        rotation_mode='anchor',transform=ax.transAxes)
            ax.set_extent(bbox)
            ax.coastlines()
            
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05)
    cb.set_label("Normalized Relevance")
    
    plt.suptitle("Mean LRP Maps for predicting %s using %s, %s, Lead %02i \n Top 5 Models (%s)" % (classes[c],varname,modelname,lead,ge_label))
    savename = "%sRegional_LRP_%s_%s_normalize%i_abs%i_%s_lead%02i.png" % (figpath,varname,classes[c],normalize_sample,absval,ge_label_fn,lead)
    plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Visualize a set of leadtimes, for the AGU presentation

c                = 0  # Class
topN             = 25 # Top 10 models
normalize_sample = 2 # 0=None, 1=samplewise, 2=after composite
absval           = False
cmax             = 0.50

pcount = 0

fig,axs = plt.subplots(4,9,figsize=(16,6.5),
                       subplot_kw={'projection':proj},constrained_layout=True)
for r,region in enumerate(regions):
    
    for l,lead in enumerate(leads):
        ax = axs[r,l]
        
        # Get indices of the top 10 models
        acc_in = modelacc_all[region][l][:,c] # [model x class]
        idtopN = am.get_topN(acc_in,topN,sort=True)
        
        # Get the plotting variables
        id_plot = np.array(idcorrect_all[region][l][c])[idtopN] # Indices to composite
        
        plotrel = np.zeros((nlat,nlon))
        for NN in range(topN):
            relevances_sel = relevances_all[region][l][idtopN[NN],id_plot[NN],:,:,:].squeeze()
            
            if normalize_sample == 1:
                relevances_sel = relevances_sel / np.max(np.abs(relevances_sel),0)[None,...]
            
            if absval:
                relevances_sel = np.abs(relevances_sel)
            plotrel += relevances_sel.mean(0)
        plotrel /= topN
            
        if normalize_sample == 2:
            plotrel = plotrel/np.max(np.abs(plotrel))
        
        plotrel[plotrel==0] = np.nan
        pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r")
        
        # Plot bounding box
        viz.plot_box(bboxes[r],ax=ax,linewidth=0.5)
        
        # Do Plotting Business and labeling
        if r == 0:
            ax.set_title("Lead %i" % (lead))
        if l == 0:
            ax.text(-0.05, 0.55, region, va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
        ax.set_extent(bbox)
        ax.coastlines()
        
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05)
cb.set_label("Normalized Relevance")
plt.suptitle("Mean LRP Maps for predicting %s using %s, %s, \n Top %02i Models (%s)" % (classes[c],varname,modelname,topN,ge_label))
savename = "%sRegional_LRP_%s_%s_top%02i_normalize%i_abs%i_%s_AGU%02i.png" % (figpath,varname,classes[c],topN,normalize_sample,absval,ge_label_fn,pcount)
plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=True)

#%% Same as above but reduce the number of leadtimes

leadsplot = [24,18,12,6,0]
#cmax      = 0.5

fig,axs = plt.subplots(4,5,figsize=(8,6.5),
                       subplot_kw={'projection':proj},constrained_layout=True)
for r,region in enumerate(regions):
    
    for i in range(len(leadsplot)):
        
        lead = leadsplot[i]
        print(lead)
        l    = list(leads).index(lead)
        print(l)
        
        ### Leads are all wrong need to fix it
        ax = axs[r,i]
        
        # Get indices of the top 10 models
        acc_in = modelacc_all[region][l][:,c] # [model x class]
        idtopN = am.get_topN(acc_in,topN,sort=True)
        
        # Get the plotting variables
        id_plot = np.array(idcorrect_all[region][l][c])[idtopN] # Indices to composite
        
        plotrel = np.zeros((nlat,nlon))
        for NN in range(topN):
            relevances_sel = relevances_all[region][l][idtopN[NN],id_plot[NN],:,:,:].squeeze()
            
            if normalize_sample == 1:
                relevances_sel = relevances_sel / np.max(np.abs(relevances_sel),0)[None,...]
            
            if absval:
                relevances_sel = np.abs(relevances_sel)
            plotrel += relevances_sel.mean(0)
        plotrel /= topN
            
        if normalize_sample == 2:
            plotrel = plotrel/np.max(np.abs(plotrel))
        
        plotrel[plotrel==0] = np.nan
        pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r")
        
        # Plot bounding box
        if r !=0:
            viz.plot_box(bboxes[r],ax=ax,color="yellow",
                         linewidth=2.5)
        
        # Do Plotting Business and labeling
        if r == 0:
            ax.set_title("Lead %i" % (lead))
        if i == 0:
            ax.text(-0.05, 0.55, region, va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
        ax.set_extent(bbox)
        ax.coastlines()
        
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05)
cb.set_label("Normalized Relevance")
plt.suptitle("Mean LRP Maps for Predicting %s using %s, \n Composite of Top %02i FNNs per leadtime" % (classes[c],varname,topN,))
savename = "%sRegional_LRP_%s_%s_top%02i_normalize%i_abs%i_%s_AGU%02i_smaller.png" % (figpath,varname,classes[c],topN,normalize_sample,absval,ge_label_fn,pcount)
plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=True)

#%% Plot Colorbar

fig,axs = plt.subplots(4,5,figsize=(8,6.5),
                       subplot_kw={'projection':proj},constrained_layout=True)

for r,region in enumerate(regions):
    
    for i in range(len(leadsplot)):
        
        lead = leadsplot[i]
        print(lead)
        l    = list(leads).index(lead)
        print(l)
        
        ### Leads are all wrong need to fix it
        ax = axs[r,i]
        for r,region in enumerate(regions):
            
            for i in range(len(leadsplot)):
                
                lead = leadsplot[i]
                print(lead)
                l    = list(leads).index(lead)
                print(l)
                
                ### Leads are all wrong need to fix it
                ax = axs[r,i]
                
                ax.set_extent(bbox)
                

fig.colorbar(pcm,ax=axs[0,-1])
savename = "%sRegional_LRP_AGU_%s_Colorbar.png" % (figpath,varname,)
plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=True)

#%% For a given region, do composites by lead for positive and negative AMV

# TZ Focus
focus_region     = None#"TZ" # Set to None to plot the whole basin
focus_bbox       = [-80,0,30,58]
focus_figsize    = (28,3.5)

if focus_region is None:
    bbox_plot        = [-80,0,0,65]
    figsize          = (16,4)
else:
    bbox_plot    = focus_bbox
    figsize      = focus_figsize
    

topN             = 25 # Top 10 models
normalize_sample = 2 # 0=None, 1=samplewise, 2=after composite
absval           = False
cmax             = 0.5
region           = "NAT"
r                = regions.index(region)
clvl             = np.arange(-2.2,2.2,0.2)


fig,axs = plt.subplots(2,9,figsize=figsize,
                       subplot_kw={'projection':proj},constrained_layout=True)

for row,c in enumerate([0,2]):
    for i in range(nleads):
        
        l = nleads-1-i
        
        lead = leads[l]
        print(l)
        ax   = axs[row,i]
        
        # Get topN Models
        acc_in = modelacc_all[region][l][:,c] # [model x class]
        idtopN = am.get_topN(acc_in,topN,sort=True)
        id_plot = np.array(idcorrect_all[region][l][c])[idtopN]
        
        plotrel = np.zeros((nlat,nlon)) # Relevances
        plotvar = np.zeros((nlat,nlon)) # Variable Value
        
        # Get data
        _,X_val,_,y_val = am.prep_traintest_classification(data,region_targets[0],lead,thresholds,percent_train,
                                                                       ens=ens,tstep=tstep,quantile=quantile)
            
        for NN in range(topN):
            relevances_sel = relevances_all[region][l][idtopN[NN],id_plot[NN],:,:,:].squeeze()
            var_sel        = X_val[id_plot[NN],:,:,:].squeeze()
            
            if (relevances_sel.shape[0] == 0) or (var_sel.shape[0]==0):
                continue
            
            if normalize_sample == 1:
                relevances_sel = relevances_sel / np.max(np.abs(relevances_sel),0)[None,...]
            if absval:
                relevances_sel = np.abs(relevances_sel)
            plotrel += relevances_sel.mean(0)
            plotvar += np.nanmean(var_sel,0)
            
        plotrel /= topN
        plotvar /= topN
        
        if normalize_sample == 2:
            plotrel = plotrel/np.nanmax(np.abs(plotrel))
        
        cl = ax.contour(lon,lat,plotvar,levels=clvl,colors="k",linewidths=0.75)
        pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r",alpha=0.8)
        ax.clabel(cl,clvl[::2])
        
        # Plot Region
        viz.plot_box(bboxes[r],ax=ax,linewidth=0.5)
        
        # Set Labels
        if row == 0:
            ax.set_title("Lead %i Years" % (lead))
        if i == 0:
            ax.text(-0.05, 0.55, classes[c], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
        ax.set_extent(bbox_plot)
        ax.coastlines()
cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.015,pad=0.01)

cb.set_label("Normalized Relevance")
plt.suptitle("Composite Relevance for Predicting AMV using %s, %s, \n Top %02i Models (%s)" % (varname,modelname,topN,ge_label))
savename = "%sComposite_LRP_%s_bylead_%s_top%02i_normalize%i_abs%i_%s_detrend%i.png" % (figpath,region,varname,topN,normalize_sample,absval,ge_label_fn,detrend)

if focus_region is not None:
    savename = proc.addstrtoext(savename,"_focus%s" % focus_region,)
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Same as above, but in separate figures

for i in range(nleads):
    
    l = nleads-1-i
    
    lead = leads[l]
    
    print(l)
    
    fig,axs = plt.subplots(2,1,figsize=(4,6),
                           subplot_kw={'projection':proj},constrained_layout=True)
    
    
    for row,c in enumerate([0,2]):
        ax   = axs[row]
        # Get topN Models
        acc_in = modelacc_all[region][l][:,c] # [model x class]
        idtopN = am.get_topN(acc_in,topN,sort=True)
        id_plot = np.array(idcorrect_all[region][l][c])[idtopN]
        
        plotrel = np.zeros((nlat,nlon)) # Relevances
        plotvar = np.zeros((nlat,nlon)) # Variable Value
        
        # Get data
        _,X_val,_,y_val = am.prep_traintest_classification(data,region_targets[0],lead,thresholds,percent_train,
                                                                       ens=ens,tstep=tstep,quantile=quantile)
            
        for NN in range(topN):
            relevances_sel = relevances_all[region][l][idtopN[NN],id_plot[NN],:,:,:].squeeze()
            var_sel        = X_val[id_plot[NN],:,:,:].squeeze()
            
            if (relevances_sel.shape[0] == 0) or (var_sel.shape[0]==0):
                continue
            
            if normalize_sample == 1:
                relevances_sel = relevances_sel / np.max(np.abs(relevances_sel),0)[None,...]
            if absval:
                relevances_sel = np.abs(relevances_sel)
            plotrel += relevances_sel.mean(0)
            plotvar += var_sel.mean(0)
            
        plotrel /= topN
        plotvar /= topN
        
        if normalize_sample == 2:
            plotrel = plotrel/np.max(np.abs(plotrel))
        
        cl = ax.contour(lon,lat,plotvar,levels=clvl,colors="k",linewidths=0.75)
        pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r",alpha=0.8)
        ax.clabel(cl,clvl[::2])
        
        # Plot Region
        viz.plot_box(bboxes[r],ax=ax,linewidth=0.5)
        
        # Set Labels
        if row == 0:
            ax.set_title("Lead %i Years" % (lead))
        ax.text(-0.05, 0.55, classes[c], va='bottom', ha='center',rotation='vertical',
                rotation_mode='anchor',transform=ax.transAxes)
        ax.set_extent(bbox)
        ax.coastlines()
    
    cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.045,)

    cb.set_label("Normalized Relevance")
    plt.suptitle("Composite Relevance for predicting using %s, %s, \n Top %02i Models (%s)" % (varname,modelname,topN,ge_label))
    savename = "%sComposite_LRP_%s_%s_top%02i_normalize%i_abs%i_%s_lead%02i.png" % (figpath,region,varname,topN,normalize_sample,absval,ge_label_fn,lead)
    plt.savefig(savename,dpi=150,bbox_inches="tight")



# ===========================
#%% Try Clustering the Output
# Based on: https://www.linkedin.com/pulse/identify-north-atlantic-winter-weather-regimes-k-means-chonghua-yin
# ===========================

from sklearn.cluster import KMeans

nclusts         = 6
norm_samplewise = False


rname           = "NAT"
il              = -1
iclass          = 0
rtest = relevances_all[rname][il]

# Restrict to particular class and samples
# Reassemble

for nm in range(nmodels):
    
    ids_model = idcorrect_all[rname][il][iclass][nm]
    relevances_nm = rtest[nm,ids_model,:,:,:].squeeze()
    if nm == 0:
        rtest_use = relevances_nm
    else:
        rtest_use = np.concatenate([rtest_use,relevances_nm],axis=0)

# [region][lead][class][model][ids]


if iclass is None:
    nmod,nsamples,_,_,_ = rtest.shape
    rtest = rtest.squeeze().reshape(nmod*nsamples,nlat*nlon)
else:
    nsamples,_,_ = rtest_use.shape
    nmod = 1
    rtest = rtest_use.reshape(nsamples,nlat*nlon)
    

if norm_samplewise:
    normconst = (np.max(np.abs(rtest),0))
    normconst[np.isnan(normconst)] = 1 # Set NaNs to 1
    normconst[normconst == 0] = 1 # Set zeros to 1
    rtest = rtest / normconst

st = time.time()
mk       = KMeans(n_clusters=nclusts, random_state=0,).fit(rtest)
ccenters = mk.cluster_centers_.reshape(nclusts,nlat,nlon)
clabels  = mk.labels_#.reshape(nmod*nsamples)
print("Completed clustering in %.2fs" % (time.time()-st))

# Reshape (sample x lat x lon)
rtest = rtest.reshape(nmod*nsamples,nlat,nlon)


#%% Plot Cluster Centers

fig,axs = plt.subplots(2,3,constrained_layout=True,figsize=(8,4),
                       subplot_kw={'projection':ccrs.PlateCarree()})

for n in range(nclusts):
    ax = axs.flatten()[n]
    
    # if norm_samplewise:
    #     pcm = ax.pcolormesh(lon,lat,ccenters[n,:,:],vmin=0,vmax=0.25,cmap="inferno")
    # else:
    #     pcm = ax.pcolormesh(lon,lat,ccenters[n,:,:],cmap="inferno")
        
    pcm = ax.pcolormesh(lon,lat,ccenters[n,:,:],cmap="inferno")
    
    ax.coastlines()
    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.025)
    ax.set_title("Cluster %i" % (n+1))
    
plt.suptitle("Cluster Centers (%s, %s, Lead %02i)" % (classes[iclass],rname,leads[il]))

savename = "%sCluster_result_%s_nclusts%i_region%s_lead%02i_cluster_means_normsmp%i.png" % (figpath,classes[iclass],nclusts,rname,leads[il],norm_samplewise)
plt.savefig(savename,dpi=150)

#%% Plot within cluster stdev

fig,axs = plt.subplots(2,3,constrained_layout=True,figsize=(8,4),
                       subplot_kw={'projection':ccrs.PlateCarree()})

for n in range(nclusts):
    ax = axs.flatten()[n]
    
    id_clust = np.where(clabels == n)[0]
    clustrel = rtest[id_clust,:,:].std(0)
    
    pcm = ax.pcolormesh(lon,lat,clustrel,cmap="inferno")
    ax.coastlines()
    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.025)
    ax.set_title("Cluster %i" % (n+1))
    
plt.suptitle("Cluster Stdev (%s, %s, Lead %02i)" % (classes[iclass],rname,leads[il]))
savename = "%sCluster_result_%s_nclusts%i_region%s_lead%02i_cluster_std_normsmp%i.png" % (figpath,classes[iclass],nclusts,rname,leads[il],norm_samplewise)
plt.savefig(savename,dpi=150)


#%% Plot samples from each cluster



clustsizes = []

for ic in range(6):
    
    id_clust = np.where(clabels == ic)[0]
    clustsizes.append(len(id_clust))
    
    fig,axs = plt.subplots(5,5,constrained_layout=True,figsize=(8,10),
                           subplot_kw={'projection':ccrs.PlateCarree()})
    
    for a in range(5*5):
        
        # Set up plot
        ax = axs.flatten()[a]
        ax.coastlines()
        ax.set_extent(bbox)
        ax.set_title("Sample = %i" % (id_clust[a]))
        
        # Plot 1 Sample
        pcm = ax.pcolormesh(lon,lat,rtest[id_clust[a],:,:],cmap="inferno")
        fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.025)
        
    plt.suptitle("Cluster %i, nsamples=%i (%s, %s, Lead %02i)" % (ic+1,clustsizes[ic],classes[iclass],rname,leads[il])
)    
    savename = "%sCluster_result_%s_nclusts%i_region%s_lead%02i_cluster%02i_normsmp%i..png" % (figpath,classes[iclass],nclusts,rname,leads[il],ic+1,norm_samplewise)
    plt.savefig(savename,dpi=150)
    

# ======================
#%% Event Based Analysis
# ======================

#%% Compute the "Accuracy" of each AMV event

rname = "NAT"

# Compute accuracy for each leadtime
amvacc_bylead = []
for il in range(nleads):
    
    # Get Label, Activation, Prediction
    labels_in = labels_all[rname][il] # [Sample x 1]
    actin     = factivations_all[rname][il]
    pred_in   = np.argmax(actin,2)
    
    # Get accuracy for the events
    correct = labels_in.T == pred_in
    leadacc = correct.sum(0)/len(correct)
    
    # Get the accuracy for that leadtime
    amvacc_bylead.append(leadacc)
    
    

# Average Accuracy for a given event
all_labels_acc = np.zeros((ens,tstep,nleads)) # Mean accuracy for each lead (of 50 models)
all_labels_cnt = np.zeros((ens,tstep))        # Count of leads for each label
all_labels_val = np.zeros((ens,tstep,))       # Class of label Last dimension should be redundant
all_labels_cls = np.zeros((ens,tstep,nleads))
all_labels_pre = np.zeros((ens,tstep,nleads,nmodels)) # Predictions for each model
all_labels_rel = np.zeros((ens,tstep,nleads,nmodels,nlat,nlon)) # Corresponding relevance Map

all_labels_slab = np.zeros((ens,tstep),dtype="object")

for il in range(nleads):
    
    # Get lead and label value
    leadaccs  = amvacc_bylead[il]
    labels_in = labels_all[rname][il]
    
    nsamples  = len(leadaccs)
    leadinds  = am.get_ensyr(np.arange(0,nsamples),leads[il])
    
    # Get predictions
    actin     = factivations_all[rname][il]
    pred_in   = np.argmax(actin,2)
    
    # Get relevances
    rel_in    = relevances_all[rname][il].squeeze()
    
    
    for n in tqdm(range(nsamples)):
        e,y = leadinds.squeeze()[n]
        if il == 0:
            start_iens = e
            start_iyr = y
        all_labels_acc[e,y,il] = leadaccs[n]
        all_labels_cnt[e,y] += 1
        all_labels_cls[e,y,il] = labels_in[n] 
        all_labels_val[e,y] = target[e,y]
        all_labels_slab[e,y] = (e,y)
        all_labels_pre[e,y,il,:] = pred_in[:,n]
        all_labels_rel[e,y,il,:,:,:] = rel_in[:,n,:,:]
        
all_labels_acc_avg = all_labels_acc.sum(2)/all_labels_cnt


# Get rid of zero points
for e in range(ens):
    for y in range(tstep):
        if all_labels_cnt[e,y] == 0:
            all_labels_acc[e,y,il] = np.nan

#%% Scatterplot selected leads (leadtime vs. AMV Index value)

plotabs = True

fig,ax = plt.subplots(1,1,figsize=(8,8))

plotx = all_labels_val[start_iens:,:]
xlab  = "AMV Index Value"
if plotabs:
    plotx = np.abs(plotx)
    xlab  = "|AMV Index Value|"

ax.scatter(plotx,
           all_labels_acc_avg[start_iens:,:],
           c=all_labels_cls[start_iens:,:,0],
           alpha=0.7,marker="o")
ax.set_xlabel(xlab)
ax.set_ylabel("Average Accuracy (All Leads)")
ax.grid(True,ls='dotted')

for n in range(len(plotx.flatten())):
    txt = str((all_labels_slab[start_iens:,:].flatten()[n]))
    ax.annotate(txt,
                (plotx.flatten()[n],all_labels_acc_avg[start_iens:,:].flatten()[n]),
                fontsize=8)



plt.savefig("%sAMVIdx_vs_TestAcc_LeadAvg_%s_plotabs%i.png" % (figpath,rname,plotabs),dpi=150)
#%% Make Scatterplot by Leadtime

fig,axs = plt.subplots(1,nleads,figsize=(16,4),constrained_layout=True)

for il in range(nleads):
    ax = axs[il]
    
    plotx = all_labels_val[start_iens:,:]
    xlab  = "AMV Index Value"
    if plotabs:
        plotx = np.abs(plotx)
        xlab  = "|AMV Index Value|"
    
    ax.scatter(plotx,
               all_labels_acc[start_iens:,:,il],
               c=all_labels_cls[start_iens:,:,0],
               alpha=0.7,marker="o",s=4)
    if il == 0:
        ax.set_xlabel("AMV Index Value")
        ax.set_ylabel("Average Accuracy (All Leads)")
    ax.grid(True,ls='dotted')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Lead %02i" % leads[il])
    
    
    # for n in range(len(plotx.flatten())):
    #     txt = str((all_labels_slab[start_iens:,:].flatten()[n]))
    #     ax.annotate(txt,
    #                 (plotx.flatten()[n],all_labels_acc_avg[start_iens:,:].flatten()[n]),
    #                 fontsize=8)
    
plt.savefig("%sAMVIdx_vs_TestAcc_ByLead_%s_plotabs%i.png" % (figpath,rname,plotabs),dpi=150)
#%% Select and plot event (spatial volution of the predictor)

e = 39
y = 5


plotsst= False


cints = np.arange(-10,11,1)


fig,axs = plt.subplots(1,nleads,subplot_kw={'projection':proj},figsize=(27,6),
                       constrained_layout=True)

for il in range(nleads):
    
    # Index Backwards from 24, 21, ..., 0
    kl   = nleads-il-1
    lead = leads[kl] 
    print("Lead index for interation %i is %i, l=%02i" % (il,kl,leads[kl]))
    
    ax = axs.flatten()[il]
    ax.coastlines()
    ax.set_extent(bbox)
    ax.set_title("Lead %02i" % (leads[kl]))
    
    if plotsst:
        plotdata = datasst[0,e,y-lead,:,:]
    else:
        plotdata = data[0,e,y-lead,:,:]
    cf       = ax.contourf(lon,lat,plotdata,cmap="RdBu_r",levels=cints)
    cl       = ax.contour(lon,lat,plotdata,colors="k",linewidths=0.5,levels=cints)
    ax.clabel(cl)
    
    fig.colorbar(cf,ax=ax,orientation='horizontal',fraction=0.026)
    if il == 0:
        if plotsst:
            ylab = "SST"
        else:
            ylab = varname
        ax.text(-0.05, 0.55, "%s, ens%02i" % (ylab,e+1), va='bottom', ha='center',rotation='vertical',
                rotation_mode='anchor',transform=ax.transAxes)
        
savename = "%s%s_Plot_%s_Prediction_e%02i_y%02i_plotsst%i.png" % (figpath,varname,rname,e,y,plotsst)
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Plot the timeseries for this period
yrs   = np.arange(0,tstep) + 1920
yrtks = yrs[::4] 

restrict_range=True # Restrict to prediction period (w/ 2 year buffer)

plot_idx    = target[e,:]
plot_idx_lp = proc.lp_butter(plot_idx,10,5) 

fig,ax = plt.subplots(1,1,figsize=(16,4))

ax.plot(yrs,plot_idx,color="gray",label="NASST")
ax.plot(yrs,plot_idx_lp,color="k",label="AMV Index")


ax.plot(yrs[(y-leads[-1]):(y+1)][::3],
        target[e,(y-leads[-1]):(y+1)][::3],
        color='magenta',marker="d",linestyle="",label="Prediction Leads for Target y=%04i" % (yrs[y]))


ax.axhline([0],ls='dashed',color="k",lw=0.75)
ax.axhline([0.3625],ls='dashed',color="gray",lw=0.75)
ax.axhline([-0.3625],ls='dashed',color="gray",lw=0.75)

ax.grid(True,ls='dotted')
if restrict_range:
    ax.set_xlim([yrs[y-leads[-1]-2],yrs[y+2]])
    ax.set_xticks(np.arange(yrs[y-leads[-1]-2],yrs[y+2]))
else:
    ax.set_xlim([yrs[0],yrs[-1]])
    ax.set_xticks(yrtks)
    
ax.set_title("AMV Index for Ens%02i" % (e+1))
ax.legend()

savename = "%sAMVIndex_Plot_%s_Prediction_e%02i_y%02i_plotsst%i_restrict%i.png" % (figpath,rname,e,y,plotsst,restrict_range)
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Copying from the above code, make a plot for each model



for n in tqdm(range(nmodels)):
    
    fig,axs = plt.subplots(1,nleads,subplot_kw={'projection':proj},figsize=(27,6),
                           constrained_layout=True)
    
    for il in range(nleads):
        
        # Index Backwards from 24, 21, ..., 0
        kl   = nleads-il-1
        lead = leads[kl] 
        print("Lead index for interation %i is %i, l=%02i" % (il,kl,leads[kl]))
        
        # Get predictions for the given model
        predlag   = all_labels_pre[e,y,kl,n]
        chk       = (predlag == all_labels_cls[e,y,kl])
        
        # Compute Plotting variables and normalize relevance
        plotdata  = data[0,e,y-lead,:,:]
        plotrel   = all_labels_rel[e,y,kl,n,:,:]
        normfactor = np.nanmax(np.abs(plotrel.flatten()))
        plotrel   = plotrel / normfactor
        
        
        ax = axs.flatten()[il]
        ax.coastlines()
        ax.set_extent(bbox)
        ax.set_title("l=%02i, Pred: %s (%s)\n normfac=%.02e" % (leads[kl],classes[int(predlag)],
                                                                chk,normfactor))
        
        
        pcm      = ax.pcolormesh(lon,lat,plotrel,cmap="RdBu_r",vmin=-1,vmax=1)
        #pcm      = ax.pcolormesh(lon,lat,plotrel,cmap="RdBu_r")
        cl       = ax.contour(lon,lat,plotdata,colors="k",linewidths=0.5,levels=cints)
        
        
        ax.clabel(cl)
        fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.026)
        if il == 0:
            if plotsst:
                ylab = "SST"
            else:
                ylab = varname
            ax.text(-0.05, 0.55, "%s, run%02i" % ("Relevance",n+1), va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
        
        
    savename = "%s%s_Plot_%s_Prediction_e%02i_y%02i_Relevances_model%02i.png" % (figpath,varname,rname,e,y,n)
    plt.savefig(savename,dpi=150,bbox_inches="tight")



#%% Compute mean and stdev of a particular prediction (intermodel)



intermodel_std = np.zeros((nleads,nlat,nlon)) * np.nan
intermodel_avg = intermodel_std.copy()
intermodel_acc = np.zeros((nleads)) * np.nan
intermodel_cnt = intermodel_acc.copy()


for moment in range(2):
    fig,axs = plt.subplots(1,nleads,subplot_kw={'projection':proj},figsize=(27,6),
                           constrained_layout=True)
    
    for il in range(nleads):
        
        # Index Backwards from 24, 21, ..., 0
        kl   = nleads-il-1
        lead = leads[kl] 
        print("Lead index for interation %i is %i, l=%02i" % (il,kl,leads[kl]))
        
        
        # Get indices of CORRECT predictions
        predlag   = all_labels_pre[e,y,kl,:]
        targlag   = all_labels_cls[e,y,il]
        correctid = np.where(predlag == int(targlag))[0]
        acclag    = len(correctid)/nmodels
        intermodel_acc[il] = acclag
        intermodel_cnt[il] = len(correctid)
        
        # Compute Plotting variables and normalize relevance
        plotdata  = data[0,e,y-lead,:,:]
        plotrel   = all_labels_rel[e,y,kl,correctid,:,:]
        
        
        # Compute Mean/Stdev (between N iterations of correct models)
        if moment == 0:
            mode = "Mean"
            plotrel                = np.mean(plotrel,0)
            intermodel_avg[il,:,:] = plotrel.copy()
            plotrng                = [-1,1]
            cmap                   = 'RdBu_r'
        elif moment == 1:
            mode = "Std. Dev."
            plotrel = np.std(plotrel,0)
            intermodel_std[il,:,:] = plotrel.copy()
            plotrng                = [0,1]
            cmap                   = "Greens"
        
        # Normalize for plotting
        #if moment == 0:
        normfactor = np.nanmax(np.abs(plotrel.flatten()))
        plotrel   = plotrel / normfactor
        # else:
        #     normfactor = "NA"
        
        # Start Plotting
        ax = axs.flatten()[il]
        ax.coastlines()
        ax.set_extent(bbox)
        ax.set_title("l=%02i, Acc=%.02f" % (leads[kl],acclag))
        
        # Contour and Pcolor
        pcm      = ax.pcolormesh(lon,lat,plotrel,cmap=cmap,vmin=plotrng[0],vmax=plotrng[1])
        cl       = ax.contour(lon,lat,plotdata,colors="k",linewidths=0.5,levels=cints)
        ax.clabel(cl)
        fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.026)
        
        # More Labeling
        if il == 0:
            
            if plotsst:
                ylab = "SST"
            else:
                ylab = varname
            ax.text(-0.05, 0.55, "%s, %s" % ("Relevance",mode,), va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
        
        
    savename = "%s%s_Plot_%s_Prediction_e%02i_y%02i_Relevances_%s.png" % (figpath,varname,rname,e,y,mode)
    plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Save the mean and stdev
savename = "%sEvent_based_intermodel_relevance_composites_e%02i_y%02i.npz" % (figpath,e,y)
np.savez(savename,**{
    'intermodel_avg':intermodel_avg,
    'intermodel_std':intermodel_std,
    'intermodel_acc':intermodel_acc,},allow_pickle=True)
# Testing the Load
#ld = np.load(savename,allow_pickle=True)

#%%
