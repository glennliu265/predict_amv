#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Dummy Fix to renormalize variables based on ocean points rather than land points

Created on Wed Jun  7 00:39:21 2023

@author: gliu
"""

import glob
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys

varname      = "SST"
datpath      = "../../CESM_data/Predictors/"
datpathr     = "../../CESM_data/Predictors_Renormalized/"
target_path  = "../../CESM_data/Target/" 
target_pathr = "../../CESM_data/Target_Renormalized/" 


#%% Get Project parameters
sys.path.append("../")
import predict_amv_params as pparams
import amv_dataloader as dl


#%%

def normalize_ds(ds):
    std,mu = ds.std(),ds.mean() 
    print("Standard deviation is %.2e, Mean is %.2e"% (std,mu))
    return std,mu
    

#%%

limask = dl.load_limask()


#%%

nc         = glob.glob(datpath+"*%s*.nc" % varname)[0]
npy        = glob.glob(datpath+"*%s*.npy" % varname)[0]

# Check current values
ds         = xr.open_dataset(nc).load()
mu,std     = np.load(npy)
print("Current Standard deviation is %.2e, Mean is %.2e"% (ds[varname].std(),ds[varname].mean()))
print("Removed Standard deviation is %.2e, Mean is %.2e"% (std,mu))

# Check masked values
ds_mask        = ds[varname] * limask[None,None,:,:]
std_ocn,mu_ocn = ds_mask.std(),ds_mask.mean() 
print("Masked Standard deviation is %.2e, Mean is %.2e"% (ds_mask.std(),ds_mask.mean()))

# Normalize "After" applying landicemask and initial erroneous normalization
ds_nafter            = (ds_mask - mu_ocn)/std_ocn
nafter_std,nafter_mu = normalize_ds(ds_nafter)


# Correct normalization, then renormalize
ds_correct      = (ds.SST * std) + mu
ori_std,ori_mu  = normalize_ds(ds_correct)
ds_remask       = ds_correct * limask[None,None,:,:]
std_new,mu_new  = normalize_ds(ds_remask)
ds_nbefore       = (ds_correct - mu_new) / std_new
nbefore_std,nbefore_mu = normalize_ds(ds_nbefore)







