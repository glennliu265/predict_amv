#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Analyze Predictors for AMV Project

Created on Wed Dec  7 12:07:15 2022

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

#%% # User Edits

# Indicate settings (Network Name)
#expdir    = "FNN4_128_SingleVar"
#modelname = "FNN4_128"

expdir     = "baseline_linreg"
modelname  = "linreg"

datpath   = "../../CESM_data/"
figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20221209/"

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
varcolors = ("r","limegreen","pink","darkblue","purple","cyan")
bbox          = [-80,0,0,65]
ens           = 40
tstep         = 86
thresholds    = [-1,1]
percent_train = 0.8
quantile      = False



# Plotting
proj = ccrs.PlateCarree()
plotbbox          = [-80,0,0,62]
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

nvars = all_data.shape[0]


lon = ds.lon.values
lat = ds.lat.values
#%% Get the Regression Maps of Trends for each variable

rmaps = np.zeros((nvars,nens,nlat,nlon))
target_ensavg   = target.mean(0)
for v in range(nvars):
    for e in range(ens):
        invar = all_data[v,e,:,:,:].transpose(2,1,0) # lon x lat x time
        beta = proc.regress2ts(invar,target_ensavg,verbose=False)
        rmaps[v,e,:,:] = beta.T

#%% Plot Ensemble average regression maps

fig,axs = plt.subplots(2,3,subplot_kw={'projection':proj},
                       constrained_layout=True,figsize=(6,5))

cmax = 2

for v in range(nvars):
    ax = axs.flatten()[v]
    
    plotvar = rmaps[v,:,:,:].mean(0)
    cmax    = np.std(plotvar)*3
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-cmax,vmax=cmax,
                        cmap="RdBu_r")
    
    ax.set_title(varnames[v])
    ax.set_extent(plotbbox)
    ax.coastlines()
    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.04)
plt.suptitle("Ens. Avg. Regression Maps for each Predictor (to AMV Index Trend)")
savename = "%sEnsAvg_PredictorRegressionMaps_toAMVTrend.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')
#plt.savefig("")

#%% Check how it looks for individual members

cmax = 2
for v in range(nvars):
    
    fig,axs = plt.subplots(5,8,subplot_kw={'projection':proj},
                           constrained_layout=True,figsize=(20,14))
    
    plotvar = rmaps[v,:,:,:].mean(0)
    cmax    = np.std(plotvar)*3
    
    for e in range(ens):
        ax = axs.flatten()[e] 
        plotvar = rmaps[v,e,:,:]
    
        pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-cmax,vmax=cmax,
                            cmap="RdBu_r")
        
        viz.label_sp("ens%02i" % (e+1),ax=ax,alpha=0.8,labelstyle="%s",usenumber=True,
                     fontsize=14)

        ax.set_extent(plotbbox)
        ax.coastlines()
    fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.025,pad=0.01)
    plt.suptitle("Regression Maps for each Predictor (to AMV Index Trend)",fontsize=16,y=0.99)
    
    savename = "%sAllEns_%s_RegressionMaps_toAMVTrend.png" % (figpath,varnames[v])
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% See if there is a correspondence to just variability of each variable

fig,axs = plt.subplots(2,3,subplot_kw={'projection':proj},
                       constrained_layout=True,figsize=(6,5))

cmax = 2

for v in range(nvars):
    ax = axs.flatten()[v]
    
    plotvar = np.std(all_data[v,:,:,:,:],(1)).mean(0)
    cmax    = np.std(plotvar)*3
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=0,vmax=cmax,
                        cmap="inferno")
    
    ax.set_title(varnames[v])
    ax.set_extent(plotbbox)
    ax.coastlines()
    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.04)
plt.suptitle("Ens. Avg. $\sigma$ Maps for each Predictor")
savename = "%sEnsAvg_PredictorStdevMaps.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')
#plt.savefig("")

#%% Examine the persistence of each predictor (basinwide)

latweight= np.cos(np.pi*lat/180)

# Take areaavg
all_data_weighted = all_data * latweight[None,None,None,:,None]
all_data_aa       = np.nanmean(all_data*msk[None,:ens,...],(3,4))
lagcorrs = np.zeros((nvars,nens,len(leads)))
for v in range(nvars):
    for e in range(ens):
        for i,l in enumerate(leads):
            ts = all_data_aa[v,e,:]
            rr = np.corrcoef(ts[l:],ts[:ts.shape[0]-l])
            print(rr)
            lagcorrs[v,e,i] = rr[0,1]

#%% Plot persistence of each variable

fig,ax= plt.subplots(1,1)
for v in range(nvars):
    eavg = 0
    for e in range(ens):
        
        plotvar = lagcorrs[v,e,:]
        if e == 0:
            eavg = plotvar.copy()
        else:
            eavg = eavg + plotvar
            
        ax.plot(leads,plotvar,color=varcolors[v],alpha=0.05,label="")
    ax.plot(leads,eavg/ens,color=varcolors[v],label=varnames[v])
ax.set_xticks(leads)
ax.set_xlim([leads[0],leads[-1]])
ax.legend()
ax.grid(True,ls="dotted")
ax.set_title("NAT-Averaged Persistence by Predictor")
savename = "%sPersistence_NAT_ByPredictor.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')