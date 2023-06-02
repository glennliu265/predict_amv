#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute the map or persistence of each predictor


    - Copied section from  viz_LRP_by_predictor

Created on Fri Apr 21 13:01:16 2023

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
import cmocean as cmo

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
varnames       = ("SST","SSH","SSS","PSL")
varnames_plot  = ("SST","SSH","SSS","SLP")


leads          = np.arange(0,25,3)
leads_sel      = [0,6,12,18,24] # Subset the leads for processing

# Other Toggles
darkmode       = False
debug          = True

# Data Settings
regrid         = None
region         = None
quantile       = False
detrend        = 1
ens            = 40
bbox           = [-80,0,0,65]
thresholds     = [-1,1]
outsize        = len(thresholds) + 1

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

#%% Load the predictors

target                          = dl.load_target_cesm(detrend=detrend,region=region)
data_all,lat,lon                = dl.load_data_cesm(varnames,bbox,detrend=detrend,return_latlon=True)

# Apply Preprocessing
target_all                      = target[:ens,:]
data_all                        = data_all[:,:ens,:,:,:]
nvars,nens,ntime,nlat,nlon  = data_all.shape

# Make land mask
data_mask = np.sum(data_all,(0,1,2))
data_mask[~np.isnan(data_mask)] = 1
if debug:
    plt.pcolormesh(data_mask),plt.colorbar()

#%% Compuge lag correlations

lags       = np.arange(0,25,1)
detrendopt = 0
dim_time   = 2
lag_corr_all,window_lengths = proc.calc_lag_covar_ann(data_all,data_all,lags,dim_time,detrendopt,)

#%% Look at lag correlations at 1 point

klon,klat = proc.find_latlon(-30,50,lon,lat)
fig,ax    = plt.subplots(1,1,constrained_layout=True)
for v in range(nvars):
    ax.plot(lags,lag_corr_all[:,v,:,klat,klon].mean(1),label=varnames_plot[v])
ax.legend()

#%% Plot T2 for each variable

cmap   = cmo.cm.solar
vlims  = [0,6]
pcolor = False
if detrend:
    clvls = np.arange(0,4.25,.25)
else:
    clvls = np.arange(0,6.5,0.5)
fig,axs = plt.subplots(1,4,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(12,4))

for a in range(4):
    ax = axs[a]
    #ax.coastlines()
    #ax.set_extent(bbox)
    blabel = [0,0,0,1]
    if a ==0:
        blabel[0] = 1
    
    viz.add_coast_grid(ax,bbox=bbox,fill_color="k",blabels=blabel)
    ax.set_title(varnames[a])
    
    plotvar = lag_corr_all[:,a,:,:,:].sum(0).mean(0)
    if pcolor:
        pcm = ax.pcolormesh(lon,lat,plotvar*data_mask,cmap=cmap,vmin=vlims[0],vmax=vlims[1])
    else:
        pcm = ax.contourf(lon,lat,plotvar*data_mask,cmap=cmap,levels=clvls)
        cl  = ax.contour(lon,lat,plotvar*data_mask,colors="k",linewidths=0.75,levels=clvls)
        ax.clabel(cl,levels=clvls[::2])
    cb = fig.colorbar(pcm,ax=ax,orientation='horizontal')

plt.savefig("%sPersistence_T2_by_Predictor_detrend%i_EnsAvg.png" % (figpath,detrend),
            bbox_inches="tight",dpi=150)
    

