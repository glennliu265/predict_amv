#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Testing Predictor Uncertainty

Examine if training on a particular predictor actually has any effect...

Steps:
    1) Load in data for 2 selected predictors
    2) Load in model weights for 2 selected predictors
    3) Do ablation study (predict one for each)
    4) Compare/visualize accuracies
    5) Compare/visualize LRP patterns

Notes:
    - Copied upper section from viz_regional_predictability

Created on Fri Mar 24 14:49:04 2023

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
import proc


# Import packages specific to predict_amv
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import train_cesm_params as train_cesm_params
import amv_dataloader as dl
import amvmod as am



#%% User Edits

expdir             = "FNN4_128_SingleVar"
modelname          = "FNN4_128"
varnames           = ["PSL","SSH"] 


nmodels            = 50 # Specify manually how much to do in the analysis
leads              = np.arange(0,26,3)


# Load parameters from [oredict_amv_param.py]
datpath            = pparams.datpath
figpath            = pparams.figpath
nn_param_dict      = pparams.nn_param_dict
class_colors       = pparams.class_colors
classes            = pparams.classes

# Load some relevant parameters from [train_cesm1_params.py]
eparams            = train_cesm_params.train_params_all[expdir] # Load experiment parameters
ens                = eparams['ens']

checkgpu           = True

#%% Load data (taken from train_NN_CESM1.py)

# Load some variables for ease

# Load predictor and labels, lat/lon, cut region
target         = dl.load_target_cesm(detrend=eparams['detrend'],region=eparams['region'])
data,lat,lon   = dl.load_data_cesm(varnames,eparams['bbox'],detrend=eparams['detrend'],return_latlon=True)

# Create classes 
# Set exact threshold value
std1         = target.std(1).mean() * eparams['thresholds'][1] # Multiple stdev by threshold value 
if eparams['quantile'] is False:
    thresholds_in = [-std1,std1]
else:
    thresholds_in = eparams['thresholds']

# Classify AMV Events
target_class = am.make_classes(target.flatten()[:,None],thresholds_in,exact_value=True,reverse=True,quantiles=eparams['quantile'])
target_class = target_class.reshape(target.shape)

# Subset predictor by ensemble, remove NaNs, and get sizes
data                           = data[:,0:ens,...]      # Limit to Ens
data[np.isnan(data)]           = 0                      # NaN Points to Zero
nchannels,nens,ntime,nlat,nlon = data.shape             # Ignore year and ens for now...
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN
nclasses = len(eparams['thresholds']) + 1
nlead    = len(leads)

#%% Load model weights (taken from LRP_LENS.py and viz_acc_byexp)

modweights_byvar = [] # [variable][lead][runid][?]
modlist_byvar    = []
flists           = []
for v,varname in enumerate(varnames):
    
    # Get the model weights
    modweights_lead,modlist_lead=am.load_model_weights(datpath,expdir,leads,varname)
    modweights_byvar.append(modweights_lead)
    modlist_byvar.append(modlist_lead)
    
    
    # Get list of metric files
    search = "%s%s/Metrics/%s" % (datpath,expdir,"*%s*" % varname)
    flist  = glob.glob(search)
    flist  = [f for f in flist if "of" not in f]
    flist.sort()
    flists.append(flist)
    print("Found %i files for %s using searchstring: %s" % (len(flist),varname,search))
    
# Get the shuffled indices
expdict = am.make_expdict(flists,leads)

# Unpack Dictionary
totalacc,classacc,ypred,ylabs,shuffids = am.unpack_expdict(expdict)
# shuffids [predictor][run][lead][nsamples]

#%% An aside, examining class distribution for each run

sample_counts = np.zeros((len(varnames),nlead,nmodels,3,nclasses)) # [predictor,lead,run,set,class]
sample_accs   = np.zeros((len(varnames),nlead,nmodels,nclasses)) # [predictor,lead,run,class]

for tvar, train_name in enumerate(varnames): # Training Variable
    for l,lead in enumerate(leads):
        
        # ---------------------
        # 08. Apply Lead
        # ---------------------
        X,y_class = am.apply_lead(predictor,target_class,lead,reshape=True,ens=ens,tstep=ntime)
        
        
        for nm in range(nmodels):

        
            # ------------------------------------------------------------------
            # 09. Select samples recorded in the shuffled indices (nsamples x 1)
            # ------------------------------------------------------------------
            sampleids = (shuffids[tvar][nm][l]).astype(int)
            X_in      = X[sampleids,...] 
            y_in      = y_class[sampleids,...]
            
            # ------------------------
            # 10. Train/Test/Val Split
            # ------------------------
            X_subsets,y_subsets = am.train_test_split(X_in,y_in,eparams['percent_train'],
                                                           percent_val=eparams['percent_val'],
                                                           debug=False,offset=eparams['cv_offset'])
    
            
            # Loop for set
            nsets = len(y_subsets)
            for s in range(nsets):
                y_subset = y_subsets[s]#.detach().numpy
                for c in range(nclasses):
                    
                    count = (y_subset == c).sum()
                    sample_counts[tvar,l,nm,s,c] = count
                    # End class loop >>>
                # End set loop >>>

            # Can do a test here...
            # Get class accuracy
            run_acc = classacc[tvar][nm][l] # [class]
            
            sample_accs[tvar,l,nm,:] = run_acc.copy()
            # End run loop >>>
        # End lead loop >>>
    # End predictor loop >>> 

#%% Visualize relationship between Sample Size and Accuracy

setnames = ["Train","Test","Val"]
fig,axs = plt.subplots(1,3,constrained_layout=True,figsize=(12,4))

for s in range(3):
    ax = axs[s]
    
    for c in range(3):
        ax.scatter(sample_counts[:,:,:,s,c].flatten(),
                   sample_accs[...,c].flatten(),25,class_colors[c],
                   label=classes[c],alpha=0.5)
        ax.legend()
        ax.set_title(setnames[s])
        ax.set_xlabel("Sample Count")
        ax.set_ylabel("Class Accuracy")
        #ax.axis('equal')

plt.savefig("%sAccuracy_vs_SampleCount_%s.png" % (figpath,expdir,),dpi=150)

#%% Recalculate some statistics... (ablation study)

# 

st                = time.time()

# 
relevances_all   = []
factivations_all = []
idcorrect_all    = []
modelacc_all     = []
labels_all       = []
ablation_idx     = []


modelaccs_all    = np.zeros((2,2,nlead,nmodels,nclasses)) # [predictor, trainvar, lead, run, class]
totalacc_all     = np.zeros((2,2,nlead,nmodels))          # [predictor, trainvar, lead, run,]


for pvar, predictor_name in enumerate(varnames): # Predictor Variable
    
    # Get the predictor
    predictor = data[[pvar],:,:,:,:] # [channel x ens x time x lat x lon]
    
    for tvar, train_name in enumerate(varnames): # Training Variable
        
        # Preallocate
        relevances_lead   = []
        factivations_lead = []
        idcorrect_lead    = []
        labels_lead       = []
        ypred_lead        = []
        
        for l,lead in enumerate(leads):
            
            # Get the models (now by variable and by leadtime)
            modweights = modweights_byvar[tvar][l]
            modlist    = modlist_byvar[tvar][l]
            
            # ---------------------
            # 08. Apply Lead
            # ---------------------
            X,y_class = am.apply_lead(predictor,target_class,lead,reshape=True,ens=ens,tstep=ntime)
            
            # Loop by model..
            for nm in range(nmodels):
                
            
                # ------------------------------------------------------------------
                # 09. Select samples recorded in the shuffled indices (nsamples x 1)
                # ------------------------------------------------------------------
                sampleids = shuffids[tvar][nm][l].astype(int)
                X_in      = X[sampleids,...] 
                y_in      = y_class[sampleids,...]
                
                # Flatten input data for FNN
                if "FNN" in eparams['netname']:
                    ndat,nchannels,nlat,nlon = X_in.shape
                    inputsize                = nchannels*nlat*nlon
                    outsize                  = nclasses
                    X_in                     = X_in.reshape(ndat,inputsize)
                
                # ------------------------
                # 10. Train/Test/Val Split
                # ------------------------
                X_subsets,y_subsets      = am.train_test_split(X_in,y_in,eparams['percent_train'],
                                                               percent_val=eparams['percent_val'],
                                                               debug=True,offset=eparams['cv_offset'])
                # Convert to Tensors
                X_subsets = [torch.from_numpy(X.astype(np.float32)) for X in X_subsets]
                y_subsets = [torch.from_numpy(y.astype(np.compat.long)) for y in y_subsets]
                
                # # Put into pytorch dataloaders
                data_loaders = [DataLoader(TensorDataset(X_subsets[iset],y_subsets[iset]), batch_size=eparams['batch_size']) for iset in range(len(X_subsets))]
                if eparams['percent_val'] > 0:
                    train_loader,test_loader,val_loader = data_loaders
                else:
                    train_loader,test_loader, = data_loaders
                
                # ----------------- Section from LRP_LENs
                # Rebuild the model
                pmodel = am.recreate_model(modelname,nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
                
                # Load the weights
                pmodel.load_state_dict(modweights[nm])
                pmodel.eval()
                # ----------------- ----------------------
                
                # ------------------------------------------------------
                # 12. Test the model separately to get accuracy by class
                # ------------------------------------------------------
                y_predicted,y_actual,test_loss = am.test_model(pmodel,test_loader,eparams['loss_fn'],
                                                               checkgpu=checkgpu,debug=False)
                lead_acc,class_acc = am.compute_class_acc(y_predicted,y_actual,nclasses,debug=True,verbose=False)
                
                
                modelaccs_all[pvar,tvar,l,nm,:] = class_acc
                totalacc_all[pvar,tvar,l,nm]    = lead_acc

            # End lead loop >>>
        ablation_idx.append("predictor%s_train%s" % (predictor_name,train_name))
        # End Training Variable loop >>>
    # End Predictor Variable loop >>>


#%% Visualize the accuracy Differences

acolors = ("red","blue",
           "orange","magenta")

# First check total accuracy
ii = 0
fig,ax = plt.subplots(1,1,constrained_layout=True)
for pvar in range(2):
    for tvar in range(2):
        plotacc_model = totalacc_all[pvar,tvar,:,:]
        
        for r in range(nmodels):
            ax.plot(leads,plotacc_model[:,r],color=acolors[ii],alpha=0.05)
        mu    = plotacc_model.mean(1)
        sigma = plotacc_model.std(1)
        ax.plot(leads,mu,color=acolors[ii],label=ablation_idx[ii])
        ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.15,color=acolors[ii])
        ii+=1

ax.legend()
ax.set_xlabel("Prediction Lead (Years)")
ax.set_ylabel("Accuracy")
ax.set_xticks(leads)
ax.grid(True,ls="dotted")
ax.set_xlim([0,24])
ax.set_title("Total Accuracy for Predicting AMV (Predictor Uncertainty Test)")
savename = "%sPredictor_Ablation_Test_TotalAcc_%s.png" % (figpath,expdir)
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Visualize accuracy differences by class


fig,axs = plt.subplots(1,3,figsize=(18,4),constrained_layout=True)
for c in range(3):
    # Initialize plot
    ax = axs[c]
    ax.set_title("%s" %(classes[c]),fontsize=16,)
    ax.set_xlim([0,24])
    ax.set_xticks(leads)
    ax.set_ylim([0,1])
    ax.set_yticks(np.arange(0,1.25,.25))
    ax.grid(True,ls='dotted')
    
    # Do the plotting
    ii = 0
    for pvar in range(2):
        for tvar in range(2):
            plotacc_model = modelaccs_all[pvar,tvar,:,:,c]
            
            for r in range(nmodels):
                ax.plot(leads,plotacc_model[:,r],color=acolors[ii],alpha=0.05)
            mu    = plotacc_model.mean(1)
            sigma = plotacc_model.std(1)
            ax.plot(leads,mu,color=acolors[ii],label=ablation_idx[ii])
            ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.15,color=acolors[ii],zorder=-9)
            ii+=1
    if c == 1:
        ax.legend()
        ax.set_xlabel("Prediction Lead (Years)")
    if c == 0:
        ax.set_ylabel("Accuracy")
        

ax.set_title("Total Accuracy for Predicting AMV (Predictor Uncertainty Test)")
savename = "%sPredictor_Ablation_Test_ClassAcc_%s.png" % (figpath,expdir)
plt.savefig(savename,dpi=150,bbox_inches="tight")
