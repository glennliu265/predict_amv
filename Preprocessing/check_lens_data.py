#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check LENs data preprocessed by prep_data_lens.py

Created on Tue Jan 24 14:02:41 2023

@author: gliu
"""

import time
import numpy as np
import xarray as xr
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

import cartopy.crs as ccrs

#%% User Edits

# I/O, dataset, paths
regrid         = 3
dataset_names  = ("canesm2_lens" ,"csiro_mk36_lens","gfdl_esm2m_lens","mpi_lens"  ,"CESM1")
dataset_long   = ("CCCma-CanESM2","CSIRO-MK3.6"    ,"GFDL-ESM2M"     ,"MPI-ESM-LR","NCAR-CESM1")
dataset_colors = ("r"            ,"b"              ,"magenta"        ,"gold" ,"limegreen")
dataset_starts = (1950           ,1920             ,1950             ,1920        ,1920)
varname        = "sst"

# Preprocessing and Cropping Options
detrend        = False
bbox           = [-90,20,0,90] # Crop Selection
bbox_fn        = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])
amvbbox        = [-80,0,0,65]  # AMV Index Calculation

# Paths
machine = "gliu_mbp"
if machine == "stormtrack":
    lenspath       = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/ts/"
    datpath        = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/processed/"
elif machine == "gliu_mbp":
    datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/LENS_other/processed/"
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
    import viz,proc
#%% Import packages and universal variables
import predict_amv_params as pparams

# Import paths
figpath         = pparams.figpath
proc.makedir(figpath)



#%% Load datasets in to dataarrays

# Load datasets for each
ndata  = len(dataset_names)
ds_all = []
for d in range(ndata):
    savename       = "%s%s_%s_NAtl_%sto2005_detrend%i_regrid%sdeg.nc" % (datpath,
                                                                         dataset_names[d],
                                                                         varname,
                                                                         dataset_starts[d],
                                                                         detrend,regrid)
    ds = xr.open_dataset(savename).load()
    ds_all.append(ds)
dataset_enssize = [len(ds.ensemble) for ds in ds_all] # Get Ensemble Sizes
samplesize      = dataset_enssize * (2005-np.array(dataset_starts)) # Rough Calculation of sample size

print([len(ds.lon) for ds in ds_all])
print([len(ds.lat) for ds in ds_all])

print([ds.lon for ds in ds_all])
# Get lat/lon
lon = ds_all[0].lon.values
lat = ds_all[0].lat.values
#%% Compute NASST from each dataset

amvids = []
for d in range(ndata):
    ds        = ds_all[d].sel(lon=slice(amvbbox[0],amvbbox[1]),lat=slice(amvbbox[2],amvbbox[3]))
    dsidx = (np.cos(np.pi*ds.lat/180) * ds).mean(dim=('lat','lon'))
    amvids.append(dsidx)
    
    # Save the labels
    savename       = "%s%s_nasst_label_%sto2005_detrend%i_regrid%sdeg.npy" % (datpath,
                                                                         dataset_names[d],
                                                                         dataset_starts[d],
                                                                         detrend,regrid)
    np.save(savename,dsidx.values)
    print("Saved Target to %s"%savename)
    
#%% Visualize the ens-avg timeseries for each large ensemble

fig,ax = plt.subplots(1,1,figsize=(12,4),constrained_layout=True)

for d in range(ndata):
    t = np.arange(dataset_starts[d],2005+1,1)
    label = "%s (N=%i)" % (dataset_long[d],dataset_enssize[d])
    ax.plot(t,amvids[d].sst.mean('ensemble'),label=label,color=dataset_colors[d],lw=2.5)

ax.axhline([0],ls="dashed",lw=0.75,color="k")
ax.set_ylim([-1.25,1.25])
ax.set_xlim([1920,2005])
ax.grid(True,ls="dotted")
ax.set_title("Ensemble Average NASST")
ax.legend()

savename = "%sNASST_%s_EnsAvg_Lens.png" % (figpath,bbox_fn)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Check out what is happening for particular datasets (map)

# ------------------------------------------------------------
# Plot Ann Mean Ens Avg SST Anomalies for a selected <<YEAR>>
# ------------------------------------------------------------
y=1950
fig,axs = plt.subplots(1,ndata,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True)
for d in range(ndata):
    ax = axs.flatten()[d]
    ax.coastlines()
    ax.set_extent(amvbbox)
    ax.set_title("%s" % (dataset_long[d]))
    
    plotvar = ds_all[d].sst.sel(year=y).mean("ensemble")
    print(plotvar)
    
    pcm     = ax.pcolormesh(lon,lat,plotvar,cmap="RdBu_r",vmin=-2,vmax=2)
    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.025)
plt.suptitle("Ensemble Mean SST ($\degree C$) for y=%i"%y,y=0.75)
    
savename = "%sAnnMeanSST_EnsAvg_LENS_y%i.png" % (figpath,y)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Plot first 5 ensemble members for a given model

t = 0
for d in range(ndata):
    
    fig,axs = plt.subplots(1,ndata,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()},
                           constrained_layout=True)
    
    for e in range(5):
        ax = axs.flatten()[e]
        
        ax.coastlines()
        ax.set_extent(amvbbox)
        ax.set_title("%s ens %i" % (dataset_long[d],e+1))

        plotvar = ds_all[d].sst.isel(year=t,ensemble=e)
        pcm     = ax.pcolormesh(lon,lat,plotvar)
        fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.025)

#%% Visualize the distribution of + and - AMV events

fig,axs =  plt.subplots(ndata,1,figsize=(4,12),constrained_layout=True,
                        sharex=True,sharey=False)

binedges = np.arange(-1.5,1.6,.1)
for d in range(ndata):
    
    ax = axs[d]
    plotdata = amvids[d].sst.values.flatten()
    mu    = np.mean(plotdata)
    stdev = np.std(plotdata)
    
    ax.hist(plotdata,bins=binedges,edgecolor="k",alpha=0.60,color=dataset_colors[d])
    
    ax.axvline([mu]      ,ls="solid",lw=0.7,color="k")
    ax.axvline([mu+stdev],ls="dashed",lw=0.7,color="k")
    ax.axvline([mu-stdev],ls="dashed",lw=0.7,color="k")
    

    cntpos = np.sum(plotdata > mu+stdev)
    cntneg = np.sum(plotdata < mu-stdev)
    cntneu = np.sum( (plotdata < mu+stdev) * (plotdata > mu-stdev) )
    
    pcts   = np.array([cntneg,cntneu,cntpos])/len(plotdata)
    title = "%s (N=%i) \n $\mu=%.2e$, $\sigma=%.2f$" % (dataset_long[d],
                                                        dataset_enssize[d],
                                                        mu,
                                                        stdev)
    
    ax.text(0.05,.7,"AMV-\n %i \n%.2f" % (cntneg,pcts[0]),transform=ax.transAxes,
            bbox=dict(facecolor='w', alpha=0.2))
    ax.text(0.40,.7,"Neutral\n %i \n%.2f" % (cntneu,pcts[1]),transform=ax.transAxes,
            bbox=dict(facecolor='w', alpha=0.2))
    ax.text(0.75,.7,"AMV+\n %i \n%.2f" % (cntpos,pcts[2]),transform=ax.transAxes,
            bbox=dict(facecolor='w', alpha=0.2))
    
    ax.set_title(title)
    ax.grid(True,ls="dotted")
    
savename = "%sNASST_%s_Histogram_Lens.png" % (figpath,bbox_fn)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Compute Power Spectra, AMV, Etc
