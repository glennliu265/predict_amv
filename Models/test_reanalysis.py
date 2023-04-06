#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test selected networks on reanalysis

- Works with reanalysis dataset preprocessed in 
- 


    Copied upper section from test_predictor_uncertainty

Created on Tue Apr  4 11:20:44 2023

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


# Shared Information
varname            = "SST" # Testing variable
detrend            = False
leads              = np.arange(0,26,3)
region_name        = "NAT"
eparams['shuffle_trainsplit'] = False # Turn off shuffling

# CESM1-trained model information
expdir             = "FNN4_128_SingleVar"
modelname          = "FNN4_128"
nmodels            = 50 # Specify manually how much to do in the analysis
eparams            = train_cesm_params.train_params_all[expdir] # Load experiment parameters
ens                = 0#eparams['ens']
runids             = np.arange(0,nmodels)

# Load parameters from [oredict_amv_param.py]
datpath            = pparams.datpath
figpath            = pparams.figpath
figpath            = pparams.figpath
nn_param_dict      = pparams.nn_param_dict
class_colors       = pparams.class_colors
classes            = pparams.classes
bbox               = pparams.bbox


# Reanalysis dataset information
dataset_name       = "HadISST"
regrid             = "CESM1"


# Other toggles
debug    = False
checkgpu = True


#%% Load the datasets

# Load reanalysis datasets [channel x ensemble x year x lat x lon]
re_data,re_lat,re_lon=dl.load_data_reanalysis(dataset_name,varname,bbox,
                        detrend=detrend,regrid=regrid,return_latlon=True)

# Load the target dataset
re_target = dl.load_target_reanalysis(dataset_name,region_name,detrend=detrend)
re_target = re_target[None,:] # ens x year

# Do further preprocessing and get dimensions sizes
re_data[np.isnan(re_data)]     = 0                      # NaN Points to Zero
nchannels,nens,ntime,nlat,nlon = re_data.shape
inputsize                      = nchannels*nlat*nlon

#%% Load regular data... (as a comparison for debugging, can remove later)

# Loads that that has been preprocessed by: ___

# Load predictor and labels, lat/lon, cut region
target         = dl.load_target_cesm(detrend=eparams['detrend'],region=eparams['region'])
data,lat,lon   = dl.load_data_cesm([varname,],eparams['bbox'],detrend=eparams['detrend'],return_latlon=True)

# Subset predictor by ensemble, remove NaNs, and get sizes
data                           = data[:,0:ens,...]      # Limit to Ens
data[np.isnan(data)]           = 0                      # NaN Points to Zero

#%% Make the classes from reanalysis data

# Set exact threshold value
std1         = re_target.std(1).mean() * eparams['thresholds'][1] # Multiple stdev by threshold value 
if eparams['quantile'] is False:
    thresholds_in = [-std1,std1]
else:
    thresholds_in = eparams['thresholds']

# Classify AMV Events
target_class = am.make_classes(re_target.flatten()[:,None],thresholds_in,
                               exact_value=True,reverse=True,quantiles=eparams['quantile'])
target_class = target_class.reshape(re_target.shape)

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


if eparams["shuffle_trainsplit"] is False:
    print("Pre-selecting indices for consistency")
    output_sample=am.consistent_sample(re_data,target_class,leads,None,leadmax=leads.max(),
                          nens=1,ntime=ntime,
                          shuffle_class=eparams['shuffle_class'],debug=True)
    
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

vt = time.time()
predictors = re_data[[0],...] # Get selected predictor


nsample_total = len(target_indices)
total_acc_all = np.zeros((nmodels,nlead))
class_acc_all = np.zeros((nmodels,nlead,3)) # 
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
            if eparams['shuffle_trainsplit'] is False:
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
        
        #
        # Flatten inputs for FNN
        #
        if "FNN" in eparams['netname']:
            ndat,nchannels,nlat,nlon = X.shape
            inputsize                = nchannels*nlat*nlon
            outsize                  = nclasses
            X_in                     = X.reshape(ndat,inputsize)
        
        #
        # Place data into a data loader
        #
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
        
        
        
        # Clear some memory
        del pmodel
        torch.cuda.empty_cache()  # Save some memory
        
        print("\nCompleted training for %s lead %i of %i" % (varname,lead,leads[-1]))
        # End Lead Loop >>>
    print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
    # End Runid Loop >>>
#print("\nPredictor %s finished in %.2fs" % (varname,time.time()-vt))
# End Predictor Loop >>>

#print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))
#%%
fig,ax = plt.subplots(1,1)

for nr in range(nmodels):
    ax.plot(total_acc_all[nr,:],alpha=0.2)
ax.plot(total_acc_all.mean(0))

#%%

fig,axs = plt.subplots(1,3,constrained_layout=True,figsize=(12,4))

for c in range(3):
    ax = axs[c]
    for nr in range(nmodels):
        ax.plot(class_acc_all[nr,:,c],alpha=0.2)
    ax.plot(class_acc_all.mean(0)[...,c])
    ax.set_title(classes[c])

