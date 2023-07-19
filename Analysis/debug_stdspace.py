#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Just checking to make sure the predictors are properly standardized...


Copied upper sectoin of train_NN_CESM1.py on 2023.07.05

Created on Wed Jul  5 09:01:51 2023

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

#%% Load custom packages and setup parameters

machine = 'stormtrack' # Indicate machine (see module packages section in pparams)

# Import packages specific to predict_amv
cwd     = os.getcwd()
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

# ============================================================
#%% User Edits vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ============================================================

# Set machine and import corresponding paths

# Set experiment directory/key used to retrieve params from [train_cesm_params.py]
expdir              = "FNN4_128_SingleVar_PaperRun"
eparams             = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Set some looping parameters and toggles
varnames            = ["SST","SSS","SLP","NHFLX","SSH",]#]       # Names of predictor variables
leads               = np.arange(0,26,3)    # Prediction Leadtimes
runids              = np.arange(0,100,1)    # Which runs to do

# Other toggles
checkgpu            = True                 # Set to true to check if GPU is availabl
debug               = True                 # Set verbose outputs
savemodel           = True                 # Set to true to save model weights

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
#%% 02. Data Loading, Classify Targets
# ----------------------------------------------

loaded_data  = []
loaded_class = []
for i in range(2):
    
    eparams['stdspace'] = i
    
    # Load data + target
    load_dict                      = am.prepare_predictors_target(varnames,eparams,return_nfactors=True)
    data                           = load_dict['data']
    target_class                   = load_dict['target_class']
    
    # Get necessary sizes
    nchannels,nens,ntime,nlat,nlon = data.shape             
    inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN
    nclasses                       = len(eparams['thresholds'])+1
    nlead                          = len(leads)
    
    # Debug messages
    if debug:
        print("Loaded data of size: %s" % (str(data.shape)))
    
    loaded_data.append(data)
    loaded_class.append(target_class)
    """
    # Output: 
        predictors       :: [channel x ens x year x lat x lon]
        target_class     :: [ens x year]
    """

#%% Check the data to make sure that standardization occured properly...

lon,lat=load_dict['lon'],load_dict['lat']
fig,axs = plt.subplots(2,nchannels,constrained_layout=True,figsize=(12,4.5))
for i in range(2):
    for v in range(nchannels):
        ax  = axs[i,v]
        if i == 1:# Reduce color range to check if there are artifacts (by ensemble)...
            pcm = ax.pcolormesh(lon,lat,np.std(loaded_data[i][v,...],(0,1)),vmin=.95,vmax=1.05) 
        else:
            pcm = ax.pcolormesh(lon,lat,np.std(loaded_data[i][v,...],(0,1)))
        fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.025)
        if i == 0:
            ax.set_title(varnames[v])
figname = "%sSpatial_Standardization_Output.png" % (figpath,)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%%


        

