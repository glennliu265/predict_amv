#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:55:34 2022


Adopted sections from:
    - build_plot_linear_regression_at_lags.ipynb
    
    

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

from tqdm import tqdm_notebook as tqdm
import time

from numpy.linalg import inv

#%% # User Edits

# Indicate settings (Network Name)
#expdir    = "FNN4_128_SingleVar"
#modelname = "FNN4_128"

expdir     = "baseline_linreg"
modelname  = "linreg"

datpath   = "../../CESM_data/"
figpath   = datpath + expdir + "/Figures/"

# lrp methods
sys.path.append("/Users/gliu/Downloads/02_Research/03_Code/github/Pytorch-LRP-master/")
from innvestigator import InnvestigateModel


# Load modules (LRPutils by Peidong)
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/scrap/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/predict_amv/")

import LRPutils as utils
import amvmod as am

# Load my own custom modules
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz,proc

leads         = np.arange(0,27,3)
detrend       = 0
regrid        = None
varnames      = ("SST","SSS","PSL","SSH","BSF","HMXL",)
bbox          = [-80,0,0,65]
ens           = 40
tstep         = 86
thresholds    = [-1,1]
thres         = np.abs(thresholds[0])
percent_train = 0.8
quantile      = False

#%%

def reform_class(labels,preds,thres):
    # Get Number of samples and classes
    nsamp  = preds.shape[0]
    y_lab_class  = am.make_classes(labels,[-thres,thres],exact_value=True,reverse=True)
    y_pred_class = am.make_classes(preds,[-thres,thres],exact_value=True,reverse=True)

    correct = np.array([0,0,0])
    total   = np.array([0,0,0])

    n = 0
    for n in tqdm(range(nsamp)):
        lab = int(y_lab_class[n])
        prd = int(y_pred_class[n])
        if lab == prd:
            correct[lab] += 1
        total[lab] += 1

    acc = correct/total
    print(acc)
    
    return acc,correct,total,y_pred_class,y_lab_class

#%% Load the data
st = time.time()

all_data = []
for v,varname in enumerate(varnames):
    # Load in input and labels 
    ds   = xr.open_dataset(datpath+"CESM1LE_%s_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc" % (varname,detrend,regrid) )
    ds   = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3])).isel(ensemble=np.arange(0,ens))
    data = ds[varname].values[None,...]
    all_data.append(data)
all_data = np.array(all_data).squeeze() # [variable x ens x yr x lat x lon]
#[print(d.shape) for d in all_data]


# Load the target
target = np.load(datpath+ "CESM_label_amv_index_detrend%i_regrid%s.npy" % (detrend,regrid))


# region_targets = []
# region_targets.append(target)
# # Load Targets for other regions
# for region in regions[1:]:
#     index = np.load(datpath+"CESM_label_%s_amv_index_detrend%i_regrid%s.npy" % (region,detrend,regrid))
#     region_targets.append(index)

# Apply Land Mask
# Apply a landmask based on SST, set all NaN points to zero
msk = xr.open_dataset(datpath+'CESM1LE_SST_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc'% (detrend,regrid))
msk = msk.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
msk = msk["SST"].values
msk[~np.isnan(msk)] = 1
msk[np.isnan(msk)] = 0
# Limit to input to ensemble member and apply mask
all_data = all_data[:,:,...] * msk[None,0:ens,...]
all_data[np.isnan(all_data)] = 0

nchannels,nens,ntime,nlat,nlon = data.shape # Ignore year and ens for now...
inputsize                      = nchannels*nlat*nlon # Compute inputsize to remake FNN

#%% # Fit linear regression models

# def remove_zeros(X,axis=0):
#     X = X.sum(axis)
    
for l,lead in enumerate(leads):
    
    for v,varname in enumerate(varnames):
        
        # Train/Test Split
        indata = all_data[[v],...]
        X_train,X_val,y_train,y_val = am.prep_traintest_classification(indata,target,lead,thresholds,percent_train,
                                                                       ens=ens,tstep=tstep,quantile=quantile)
        
        trainsize = X_train.shape[0]
        valsize  = X_val.shape[0]
        
        # Flatten and remove nan points
        okdata_train,knan_train,okpts_train = proc.find_nan(X_train.reshape(trainsize,nlat*nlon),0,val=0)
        okdata_val,knan_val,okpts_val       = proc.find_nan(X_val.reshape(valsize,nlat*nlon),0,val=0)
        
        # Fit Theta
        okdata_train = okdata_train.T # [sample x input] --> [input x sample]
        okdata_val   = okdata_val.T
        
        #theta = (inv(X_train @ X_train.T) @ X_train ) @ y_train
        theta = (inv(okdata_train @ okdata_train.T) @ okdata_train) @ y_train
        y_pred_train = (theta.T@okdata_train).T
        y_pred_val   = (theta.T@okdata_val).T
        
        acc,correct,total,y_pred_class,y_lab_class = reform_class(y_val,y_pred_val,thres)
        
        # Replace theta in variables
        theta_reshape = np.zeros((nlat*nlon)) * np.nan
        theta_reshape[okpts_val] = theta.squeeze()
        theta_reshape = theta_reshape.reshape(nlat,nlon)
        
        # Fit linear model on the training set 
        #beta,b=proc.regress_2d(okdata_train,y_train) # beta is [space x 1]
        
        # Predict...
        #y_pred = beta * okdata_train.T + b[:,None]
        
        #coeffs = np.polyfit(y_train.squeeze(),okdata_train,1)
        
        # Fit the linear model
        
        # Compute regression results for test set
        
        # Convert to class predictions, evaluate accuracy

#%%

