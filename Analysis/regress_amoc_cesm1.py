#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regress the maximum AMOC Index to a given CESM1 Predictor

Created on Tue Mar  7 02:38:23 2023

@author: gliu
"""

import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import sys
import os

from tqdm import tqdm
import cartopy.crs as ccrs


#%% Add custom modules and packages
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import proc,viz


cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import amvmod as am
import amv_dataloader as dl 

#%%
# Set path to the data 
mocpath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/01_Data/AMOC/"
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20230308/"

# Time_Period
startyr = 1920
endyr   = 2005
ntime   = (endyr-startyr+1)*12

# Select MOC component and region
icomp        = 0 # 0=Eulerian Mean; 1=Eddy-Induced (Bolus); 2=Submeso
iregion      = 1 # 0=Global Mean - Marginal Seas; 1= Altantic Ocean + Mediterranean Sea + Labrador Sea + GIN Sea + Arctic Ocean + Hudson Bay
savename_moc = "%sCESM1_LENS_AMO_%sto%s_comp%i_region%i.npz" % (mocpath,startyr,endyr,icomp,iregion)


# Set predictor options
varnames    = ["SSH","SST",]
detrend     = 0
bbox        = pparams.bbox

debug        = True


#%% Load MOC data for the given ensemble member

ld      = np.load(savename_moc,allow_pickle=True)
max_moc = ld['max_moc'] # {Ens x Time}


# Take the Annual averages
max_moc_annavg = proc.ann_avg(max_moc,1,)

#%% Load the predictors

data,lat,lon   = dl.load_data_cesm(varnames,bbox,detrend=detrend,return_latlon=True) # {Channel x ens x twE, X OLQ }


#%% Make the regression maps

nvars,nens,nyrs,nlat,nlon = data.shape
regr_maps = np.zeros([nvars,nens,nlat,nlon])

for v in range(nvars):
    for e in range(nens):
        in_predictor = data[v,e,:,:,:]
        in_moc       = max_moc_annavg[e,:] #- max_moc_annavg.mean(0)
        
        
        regr_maps[v,e,:,:] = proc.regress2ts(in_predictor.transpose(2,1,0),in_moc,).T
        

#%% Examine the AMOC regression patterns (ensemble mean)



fig,axs = plt.subplots(1,nvars,figsize=(10,4),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True,)



for v in range(nvars):
    ax      = axs.flatten()[v]
    plotvar = regr_maps[v,...].mean(0)
    ax      = viz.add_coast_grid(ax,bbox=bbox,proj=ccrs.PlateCarree(),fill_color="k")
    
    pcm = ax.pcolormesh(lon,lat,plotvar,cmap="RdBu_r",vmin=-2.5,vmax=2.5)
    cb=fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
    ax.set_title(varnames[v])
    cb.set_label("AMOC Regression ([Fluctuation per Sv of iAMOC])")

plt.savefig("%sAMOC_Regression_2var.png" % (figpath),dpi=200,bbox_inches="tight")
    
    
    

        
        
        

