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

# CESM1-trained model information
expdir             = "FNN4_128_SingleVar"
modelname          = "FNN4_128"
nmodels            = 50 # Specify manually how much to do in the analysis
eparams            = train_cesm_params.train_params_all[expdir] # Load experiment parameters
ens                = eparams['ens']

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



#%% Load the datasets

# Load reanalysis datasets [channel x ensemble x year x lat x lon]
re_data,re_lat,re_lon=dl.load_data_reanalysis(dataset_name,varname,bbox,
                        detrend=detrend,regrid=regrid,return_latlon=True)

# Preprocess
re_target = dl.load_target_reanalysis(dataset_name,region_name,detrend=detrend)


#%% Make the classes



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







