#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Copied from LRP_LENS.py

Load model weights for networks trained on a given CMIP6 LENS.
Copied sections from LRP_LENS.py

Test on another large ensemble and:
    - Compute Accuracy by Class
    - Get indices of correct predictions
    - Compute Composite LRP for correct predictions by class
    - Compute Variance of LRP for correct predictions by class

Created on Wed Feb 15 14:56:08 2023

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

#%%

# Model Information
cmipver              = 6
varname              = "sst"
modelname            = "FNN4_128"
leads                = np.arange(0,26,1)
model_dataset_name   = "ACCESS-ESM1-5"
expdir               = "%s_SingleVar_%s_Train" % (modelname,model_dataset_name)
nmodels              = 50 # Specify manually how much to do in the analysis

print("Loading Models from %s" % (expdir))

# Dataset Test Information
use_train     = True # Set to True to use training data for testing for other LENs
skip_dataset  = ('MPI-ESM1-2-LR',)
limit_time    = [1850,2014] # Set Dates here to limit the range of the variable
restrict_ens  = 25 # Set to None to use all ensemble members

# Composite Options
composite_topNs  = (1,5,10,25,50)
absval           = False
normalize_sample = False

# LRP Settings (note, this currently uses the innvestigate package from LRP-Pytorch)
gamma                = 0.1
epsilon              = 0.1
innexp               = 2
innmethod            ='b-rule'
innbeta              = 0.1

# Labeling for plots and output files
ge_label       = "exp=%i, method=%s, $beta$=%.02f" % (innexp,innmethod,innbeta)
ge_label_fn    = "innexp%i_%s_innbeta%.02f" % (innexp,innmethod,innbeta)

# lrp methods
sys.path.append("/Users/gliu/Downloads/02_Research/03_Code/github/Pytorch-LRP-master/")
from innvestigator import InnvestigateModel

# Load modules (LRPutils by Peidong)
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/scrap/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/predict_amv/")
import LRPutils as utils
import amvmod as am

# Load visualization module
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz,proc

#%% IMport params
# Note; Need to set script into current working directory (need to think of a better way)
import os
cwd = os.getcwd()

sys.path.append(cwd+"/../")
import predict_amv_params as pparams

classes         = pparams.classes
proj            = pparams.proj
figpath         = pparams.figpath
proc.makedir(figpath)

bbox            = pparams.bbox
nn_param_dict   = pparams.nn_param_dict

leadticks25     = pparams.leadticks25
leadticks24     = pparams.leadticks24

dataset_names   = pparams.cmip6_names
cmip6_dict      = pparams.cmip6_dict

#%% Load some other things

# Set Paths based on CMIP version
if cmipver == 5:
    datpath        = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/processed/"
    modepath       = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/LENS_30_1950/"
elif cmipver == 6:
    datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/CMIP6_LENS/processed/"
    modpath        = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/CMIP6_LENS/models/"

# Set Outpath
outpath            = "%s../LRP/" % datpath

# Compute some dimensions
nleads         = len(leads)

# Set preprocessing options based on cmip version

if cmipver == 5:
    dataset_names = pparams.dataset_names
    ystarts       = pparams.dataset_starts
    limit_time    = [1950,2005] # Set Dates here to limit the range of the variable
    ens           = 30
    regrid        = 3
    
elif cmipver == 6:
    varname       = varname.lower()
    #dataset_names = pparams.cmip6_names[1:-1]
    ystarts       = (1850,)*len(dataset_names)
    varnames      = ("sst","ssh","sss")
    regrid        = None

quantile      = True
thresholds    = [1/3,2/3]
tstep         = limit_time[1] - limit_time[0] + 1
percent_train = 0.8
detrend       = 0
outsize       = 3
lowpass       = 0
ystart        = 1850
yend          = 2014
save_latlon   = False # Set to True to save latlon for plotting
#%% Load Model Weights

modweights_lead,modlist_lead=am.load_model_weights(modpath,expdir,leads,varname)

#%% Looping for each dataset...

for d in range(len(dataset_names)):
    st_dataset=time.time()
    
    dataset_name = dataset_names[d]
    print("Now Doing %s" % dataset_name)
    if dataset_name in skip_dataset: # Skip this dataset
        print("Skipping %s" % dataset_name)
        continue
    
    #%Load Predictors (works just for CMIP6 for now)
    
    # Load predictor and labels,lat,lon, cut region
    data,target,lat,lon = am.load_cmip6_data(dataset_name,varname,bbox,datpath=datpath,
                                     detrend=detrend,regrid=regrid,
                                     ystart=ystart,yend=yend,lowpass=lowpass,
                                     return_latlon=True)
    
    # Subset predictor by ensemble, remove NaNs, and get sizes
    if restrict_ens is None:
        ens = data.shape[1] # Set ens to maximum number of members
    else:
        ens = restrict_ens
    data                           = data[:,0:ens,...]      # Limit to Ens
    data[np.isnan(data)]           = 0                      # NaN Points to Zero
    nchannels,nens,ntime,nlat,nlon = data.shape             # Ignore year and ens for now...
    inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN
    
    #% Calculate the Relevance by leadtime
    
    st                = time.time()
    
    
    ncomposites = len(composite_topNs)
    
    # List for each leadtime
    relevances_lead   = [] 
    factivations_lead = [] #
    idcorrect_lead    = [] # Save Indices
    modelacc_lead     = [] # Save Accuracy by Model
    labels_lead       = [] 
    composites_lead   = [] # Save Composites by Class
    variances_lead    = [] # Relevance variances
    
    for l,lead in enumerate(leads): # Training data does chain with leadtime
        
        # Get List of Models
        modlist = modlist_lead[l]
        modweights = modweights_lead[l]
        
        # Prepare data
        X_train,X_val,y_train,y_val = am.prep_traintest_classification(data,target,lead,thresholds,percent_train,
                                                                       ens=ens,tstep=tstep,quantile=quantile,)
        
        # Use all data when dataset is not the same
        if (dataset_name is not model_dataset_name) and (use_train):
            print("Using all data!")
            X_val = np.concatenate([X_train,X_val],axis=0)
            y_val = np.concatenate([y_train,y_val],axis=0)
        
        # Make land/ice mask
        xsum = np.sum(np.abs(X_val),(0,1))
        limask = np.zeros(xsum.shape) * np.nan
        limask[np.abs(xsum)>1e-4] = 1
        
        # Preallocate, compute relevances
        valsize      = X_val.shape[0]
        relevances   = np.zeros((nmodels,valsize,inputsize))*np.nan # [model x sample x inputsize ]
        factivations = np.zeros((nmodels,valsize,3))*np.nan         # [model x sample x 3]
        for m in tqdm(range(nmodels)): # Loop for each model
            
            # Rebuild the model
            pmodel = am.recreate_model(modelname,nn_param_dict,inputsize,outsize,nlon=nlon,nlat=nlat)
            
            # Load the weights
            pmodel.load_state_dict(modweights[m])
            pmodel.eval()
            
            # Investigate
            inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
                                  method=innmethod,
                                  beta=innbeta)
            input_data = torch.from_numpy(X_val.reshape(X_val.shape[0],1,inputsize)).squeeze().float()
            model_prediction, true_relevance = inn_model.innvestigate(in_tensor=input_data)
            relevances[m,:,:]   = true_relevance.detach().numpy().copy()
            factivations[m,:,:] = model_prediction.detach().numpy().copy()
        
        # Reshape Output
        relevances = relevances.reshape(nmodels,valsize,nchannels,nlat,nlon)
        y_pred     = np.argmax(factivations,2)
        
        # Compute accuracy
        composites   = np.zeros((ncomposites,3,nchannels,nlat,nlon))*np.nan # [compositeN,class,channels,lat,lon]
        modelacc     = np.zeros((nmodels,3)) # [model x class]
        modelnum     = np.arange(nmodels)+1  
        idcorrect    = []                    # [class][model][correct_samples]
        #variances   = np.zeros((3,nchannels,nlat,nlon)) * np.nan #[class,channels,lat,lon]
        for c in range(3):
            
            # Compute accuracy
            class_id           = np.where(y_val == c)[0]     # [class_samples]
            pred               = y_pred[:,class_id]          # [models x class_samples]
            targ               = y_val[class_id,:].squeeze() # [class_samples]
            correct            = (targ[None,:] == pred)      # [models x class_samples]
            num_correct        = correct.sum(1)              # [models]
            num_total          = correct.shape[1]            # [models]
            modelacc[:,c]      = num_correct/num_total       # [models]
            meanacc            = modelacc.mean(0)[c]         # 1
            
            # Get indices of correct predictions
            correct_id = [] # [model][correct_samples]
            for zz in range(nmodels):
                correct_id.append(class_id[correct[zz,:]])
            idcorrect.append(correct_id)
            
            # Make composites of the correct predictions
            for NC in range(ncomposites):
                topN   = composite_topNs[NC]
                composite_rel = am.compute_LRP_composites(topN,modelacc[:,c],correct_id,relevances,
                                                       absval=absval,normalize_sample=normalize_sample)
                # Save
                composites[NC,c,0,:,:] = composite_rel.copy()
                # End Composite Loop
            #End Class Loop
        
        # Append for each leadtime
        composites_lead.append(composites)
        #relevances_lead.append(relevances)
        #factivations_lead.append(factivations)
        idcorrect_lead.append(idcorrect)
        modelacc_lead.append(modelacc)
        #labels_lead.append(y_val)
        del relevances
        del factivations
        del y_val
    print("\nComputed relevances in %.2fs" % (time.time()-st))
    
    savename = "%sLRP_Output_%s_TEST_%s_ens%03i.npz" % (outpath,expdir,dataset_name,nens)
    np.savez(savename,**{
        "composites_lead" : composites_lead, # [lead][compositeN,class,channels,lat,lon]
        "idcorrect_lead"  : idcorrect_lead,  # [lead][class][model][correct_samples]
        "modelacc_lead"   : modelacc_lead,},   # [lead][model x class]
        allow_pickle=True)
    
    # End Dataset Loop
    print("Finished %s in %.2fs" % (model_dataset_name,time.time()-st_dataset))


#%%
if save_latlon:
    savename="%sLRP_Output_LATLON.npz" % (outpath)
    np.savez(savename,**{"lat":lat,"lon":lon},allow_pickle=True)
