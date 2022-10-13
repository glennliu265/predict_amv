#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute Relevance using Layerwise-Relevance Propagation for a simple
2-layer FNN

Created on Thu Oct  6 15:28:13 2022, copied code from LRP_Test.ipynb

@author: gliu

"""

import numpy as np
import sys
import glob
import importlib
import torch
from torch import nn
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tqdm import trange, tqdm

import time

#%% User Edits

# Indicate regridding
regrid = None

# Indicate paths
datpath = "../../CESM_data/"
if regrid is None:
    modpath = datpath + "Models/FNN2_quant0_resNone/"
else:
    modpath = datpath + "Models/FNN2_quant0_res224/"
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20221007/"
outpath = datpath + "Metrics/"
# Indicate settings
detrend = 0

# Load modules (LRPutils by Peidong)
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/scrap/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/predict_amv/")
import LRPutils as utils
import amvmod as am

# Plotting Settings
vnames     = ("SST","SSS","SLP")
thresnames = ("AMV+","Neutral","AMV-",)

# Model Parameters (copied from NN_test_lead_ann_ImageNet_classification.py)
thresholds  = [-1,1]


outsize     = len(thresholds) + 1
nlayers     = 2
nunits      = [20,20]
activations = [nn.ReLU(),nn.ReLU()]
dropout     = 0.5

# Data Preparation
nruns       = 10
leads       = np.arange(0,24+3,3)
ens         = 40
tstep       = 86
quantile    = False

# Settings for LRP
gamma       = 0.25
epsilon     = 0.25 * 0.36 # 0.36 is 1 stdev of the AMV Index

#%%

# Load in input and labels 
data   = np.load(datpath+ "CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i_regrid%s.npy" % (detrend,regrid)) # [variable x ensemble x year x lon x lat]
target = np.load(datpath+ "CESM_label_amv_index_detrend%i_regrid%s.npy" % (detrend,regrid))

#%%
nleads = len(leads)


# This is saving for all files, but it might be too slow...
# relevances = np.empty((nleads,nruns,outsize),dtype='object') # [lead x run x threshold][samples]
# ids        = relevances.copy()
# y_pred     = relevances.copy()
# y_targ     = np.empty((nleads,),dtype='object')
# y_val      = y_targ.copy()


# Loop for each lead
for l in range(nleads):
    st = time.time()
    # Prepare the data by applying lead
    lead = leads[l]
    nchannels,nens,ntime,nlat,nlon = data.shape
    y    = target[:ens,lead:].reshape(ens*(tstep-lead),1)
    X    = (data[:,:ens,:tstep-lead,:,:]).reshape(3,ens*(tstep-lead),nlat,nlon).transpose(1,0,2,3)
    nsamples,nchannels,nlon,nlat = X.shape
    x_in = X.reshape(nsamples,nchannels*nlon*nlat) # Flatten for processing
    
    inputsize   = nchannels*nlat*nlon


        
    # Make the labels
    y_class = am.make_classes(y,thresholds,reverse=True,quantiles=quantile)
    if quantile == True:
        thresholds = y_class[1].T[0]
        y_class   = y_class[0]
    if (nsamples is None) or (quantile is True):
        nthres = len(thresholds) + 1
        threscount = np.zeros(nthres)
        for t in range(nthres):
            threscount[t] = len(np.where(y_class==t)[0])
        nsamples = int(np.min(threscount))
    y_targ = y_class.copy()
    y_val  = y.copy()
    print("Preproc data in %.2fs"%(time.time()-st))
    
    # Saving for each leadtime
    relevances = np.empty((nruns,outsize),dtype='object') # [lead x run x threshold][samples]
    ids        = relevances.copy()
    y_pred     = relevances.copy()
    
    
    # Loop for each run
    for r in range(nruns):
        
        # Load the model state dict
        modelname = "AMVClass3_FNN2_nepoch20_nens40_maxlead24_detrend0_noise0_unfreeze_allTrue_run%i_unfreezeall_quant0_res%s_ALL_lead%i_classify.pt" % (r,regrid,lead)
        mod    = torch.load(modpath+modelname)
        
        # Reconstruct the model and load in the weights (only needed for acc. evaluation)
        layers = am.build_FNN_simple(inputsize,outsize,nlayers,nunits,activations,dropout=0.5)
        pmodel = nn.Sequential(*layers)
        pmodel.load_state_dict(mod)
        pmodel.eval()
        
        
        # Get the weights for the model
        Ws,Bs = utils.get_weight(mod)
        print("Load model in %.2fs"%(time.time()-st))
        
        # Loop for each class
        for th in range(outsize):
            
            # Select only the samples for given class
            ithres  = np.where(y_class==th)[0]
            x_class = x_in[ithres,:]
            x_size  = x_class.shape[0]
            
            # Preallocate to store relevances
            rel_x   = np.zeros(x_class.shape)
            # Calculate relevance for each sample for the selected class
            for x in tqdm(range(x_size)):
                
                rel = utils.LRP_single_sample(x_class[x],Ws,Bs,epsilon,gamma)
                rel_x[x,:] = rel.copy() # 
            
            # Reshape output to [sample x channel x lon x lat] and store
            rel_x = rel_x.reshape(x_size,nchannels,nlon,nlat)
            
            # Reevaluate the model and obtain the predictions
            y_out = pmodel(torch.from_numpy(x_class).float())
            _,y_out = torch.max(y_out,1) # Take maximum along the class dimension, and output the indices
            
            # Store output
            #relevances[l,r,th] = rel_x.copy()
            #ids[l,r,th] = ithres.copy()
            #y_pred[l,r,th] = y_out
            
            relevances[r,th] = rel_x.copy()
            ids[r,th] = ithres.copy()
            y_pred[r,th] = y_out
    
    # Save the data for a given leadtime
    savename = "%sLRPout_lead%02i_gamma%.3f_epsilon%.3f.npz" % (outpath,lead,gamma,epsilon)
    np.savez(savename,**{
        'relevances':relevances,
        'ids':ids,
        'y_pred':y_pred,
        'y_targ':y_targ,
        'y_val':y_val,
        'lead':lead,
        'gamma':gamma,
        'epsilon':epsilon
        },allow_pickle=True)
#%% Save the output

savename = "%sLRPout_gamma%.3f_epsilon%.3f.npz" % (outpath,gamma,epsilon)


# relevances = np.empty((nleads,nruns,outsize),dtype='object') # [lead x run x threshold][samples]
# ids        = relevances.copy()
# y_pred     = relevances.copy()
# y_targ     = np.empty((nleads,),dtype='object')
# y_val      = y_targ.copy()