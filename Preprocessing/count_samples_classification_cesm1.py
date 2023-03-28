#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script for counting samples for CESM1 Training

Copied upper section of train_NN_CESM1

Created on Mon Mar 27 22:36:05 2023

@author: gliu
"""

import sys
import numpy as np
import os
import time
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset,Dataset
import matplotlib.pyplot as plt

#%% Load custom packages and setup parameters
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

# Load Predictor Information
bbox          = pparams.bbox

# ============================================================
#%% User Edits vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ============================================================

# Set experiment directory/key used to retrieve params from [train_cesm_params.py]
expdir             = "FNN4_128_SingleVar_Rerun100"
eparams            = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Set some looping parameters and toggles
varnames           = ["SST",]       # Names of predictor variables
leads              = np.arange(0,26,1)    # Prediction Leadtimes
runids             = np.arange(0,1,1)    # Which runs to do

# Other toggles
checkgpu           = True                 # Set to true to check if GPU is availabl
debug              = True                 # Set verbose outputs
savemodel          = True                 # Set to true to save model weights

# Save looping parameters into parameter dictionary
eparams['varnames'] = varnames
eparams['leads']    = leads
eparams['runids']   = runids

# ============================================================
# End User Edits ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ============================================================
# ------------------------------------------------------------
# %% 01. Check for existence of experiment directory and create it
# ------------------------------------------------------------
allstart = time.time()

proc.makedir("../../CESM_data/"+expdir)
for fn in ("Metrics","Models","Figures"):
    proc.makedir("../../CESM_data/"+expdir+"/"+fn)
    
    
# Check if there is gpu
if checkgpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

# ----------------------------------------------
#%% 02. Data Loading...
# ----------------------------------------------

# Load some variables for ease
ens            = eparams['ens']

# Loads that that has been preprocessed by: ___

# Load predictor and labels, lat/lon, cut region
target         = dl.load_target_cesm(detrend=eparams['detrend'],region=eparams['region'])
data,lat,lon   = dl.load_data_cesm(varnames,eparams['bbox'],detrend=eparams['detrend'],return_latlon=True)

# Subset predictor by ensemble, remove NaNs, and get sizes
data                           = data[:,0:ens,...]      # Limit to Ens
data[np.isnan(data)]           = 0                      # NaN Points to Zero
nchannels,nens,ntime,nlat,nlon = data.shape             # Ignore year and ens for now...
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN

# ------------------------------------------------------------
# %% 03. Determine the AMV Classes
# ------------------------------------------------------------

# Set exact threshold value
std1         = target.std(1).mean() * eparams['thresholds'][1] # Multiple stdev by threshold value 
if eparams['quantile'] is False:
    thresholds_in = [-std1,std1]
else:
    thresholds_in = eparams['thresholds']

# Classify AMV Events
target_class = am.make_classes(target.flatten()[:,None],thresholds_in,exact_value=True,reverse=True,quantiles=eparams['quantile'])
target_class = target_class.reshape(target.shape)

# Get necessary dimension sizes/values
nclasses     = len(eparams['thresholds'])+1
nlead        = len(leads)

"""
# Output: 
    predictors :: [channel x ens x year x lat x lon]
    labels     :: [ens x year]
"""
     
#%% Add option to load existing runid?



# Do some dummy selections
v = 0
predictors = data[[v],...] # Get selected predictor
k_offset = 0
# Preallocate
nruns = len(runids)
nleads = len(leads)
nsamples = eparams['nsamples']
varname  = "SST"



y_subsets_all    = []
shuffid_all      = []
sample_size_all  = []
idx_byclass_all  = []
total_count_byclass = np.zeros((nruns,nleads,nclasses))

# --------------------
# 05. Loop by runid...
# --------------------
for nr,runid in enumerate(runids):
    rt = time.time()
    
    # Preallocate Evaluation Metrics...
    sampled_idx          = []
    sample_sizes         = []
    y_subsets_lead       = []
    idx_byclass_lead     = []
    # -----------------------
    # 07. Loop by Leadtime...
    # -----------------------
    for l,lead in enumerate(leads):
        
        # --------------------------
        # 08. Apply lead/lag to data
        # --------------------------
        # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
        X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=ens,tstep=ntime)
        
        
        idx_by_class,count_by_class=am.count_samples(nsamples,y_class)
        total_count_byclass[nr,l,:] = count_by_class
        
        # ----------------------
        # 09. Select samples
        # ----------------------
        if eparams['nsamples'] is None: # Default: nsamples = smallest class
            threscount = np.zeros(nclasses)
            for t in range(nclasses):
                threscount[t] = len(np.where(y_class==t)[0])
            eparams['nsamples'] = int(np.min(threscount))
            print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
       
        y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle'])
        

        
        
        # --------------------------
        # 10. Train Test Split
        # --------------------------
        X_subsets,y_subsets      = am.train_test_split(X,y_class,eparams['percent_train'],
                                                       percent_val=eparams['percent_val'],
                                                       debug=True,offset=k_offset)
        
        
        sampled_idx.append(shuffidx) # Save the sample indices
        sample_sizes.append(eparams['nsamples'])
        y_subsets_lead.append(y_subsets)
        idx_byclass_lead.append(idx_by_class)
        print("\nCompleted counting for %s lead %i of %i" % (varname,lead,leads[-1]))
    
    
    shuffid_all.append(sampled_idx)
    sample_size_all.append(sample_size_all)
    y_subsets_all.append(y_subsets_lead)
    idx_byclass_all.append(idx_byclass_lead)
    print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
    # End Runid Loop >>>

#%% Plot total class counts by leadtime
fig,axs = plt.subplots(1,2,figsize=(8,3))

for c in range(3):
    if c == 1:
        ax = axs[1]
    else:
        ax = axs[0]
    for nr in range(nruns):
        clabel = "%s (Run %i,n=%i-%i)" % (pparams.classes[c],runids[nr],total_count_byclass[nr,:,c].min(),total_count_byclass[nr,:,c].max())
        ax.plot(leads,total_count_byclass[nr,:,c],label=clabel,color=pparams.class_colors[c])
        
    ax.grid(True)
    ax.legend()
    
#%% Plot total classes by 
    

             