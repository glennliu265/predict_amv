#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare labels (AMV/NASST) indices for prediction...

Works with output from [prep_data_lens.py]
Copied setions from [check_lens_data.py]

Created on Tue Feb  7 17:28:43 2023

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
varname        = "sst" # (tos, sos, zos)
detrend        = False
ystart         = 1850
yend           = 2014
regrid         = None
debug          = True # Set to true to do some debugging

# Preprocessing and Cropping Options

# Paths
machine = "Astraeus"
if machine == "stormtrack":
    lenspath       = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/ts/"
    datpath        = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/processed/"
elif machine == "gliu_mbp":
    datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/LENS_other/processed/"
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
    import viz,proc
elif machine == "Astraeus":
    datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/CMIP6_LENS/processed/"
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
    import viz,proc


plt.style.use('default')
## Load some global information (lat.lon) <Note, I need to customize this better..
#%% Import packages and universal variables

# Note; Need to set script into current working directory (need to think of a better way)
import os
cwd = os.getcwd()

sys.path.append(cwd+"/../")
import predict_amv_params as pparams

# Import paths
figpath         = pparams.figpath
proc.makedir(figpath)

# Import class information
classes         = pparams.classes
class_colors    = pparams.class_colors

# Import dataset inforation
dataset_names   = pparams.cmip6_names
dataset_long    = pparams.cmip6_names
dataset_colors  = pparams.cmip6_colors
dataset_starts  =(1850,) * len(dataset_names)

# AMV related information
amvbbox         = pparams.amvbbox

#%% Get the data

d = 1

ds_all        = []
ndata         = len(dataset_names)


nasst_all     = []
amvid_all     = []
amvpats_all   = []
nasstpats_all = []

for d in range(ndata):
    # Open the dataset
    ncsearch  = "%s%s_%s_NAtl_%sto%s_detrend%i_regrid%sdeg.nc" % (datpath,dataset_names[d],
                                                                 varname,ystart,yend,
                                                                 detrend,regrid)
    ds        = xr.open_dataset(ncsearch).load()
    ds_all.append(ds)
    
    # Compute the mean over the area
    ds        = ds.sel(lon=slice(amvbbox[0],amvbbox[1]),lat=slice(amvbbox[2],amvbbox[3]))
    dsidx     = (np.cos(np.pi*ds.lat/180) * ds).mean(dim=('lat','lon'))
    nasst     = dsidx.sst.values #[ensemble x time]
    
    # Compute the low-passed version [ensemble x time]
    amvid_lp  = proc.lp_butter(nasst.T[...,None],10,order=6).squeeze().T
    
    # Save the labels (normal and low-pass filtered)
    savename  = "%s%s_%s_label_%sto%s_detrend%i_regrid%sdeg_lp0.npy" % (datpath,dataset_names[d],
                                                                         varname,ystart,yend,
                                                                         detrend,regrid)
    savename_lp = savename.replace("lp0","lp1")
    
    np.save(savename,nasst)
    np.save(savename_lp,amvid_lp)
    
    print(savename)
    print(savename_lp)
    
    # Do some plotting
    if debug:
        yrs  = np.arange(ystart,yend+1,1)
        nens = nasst.shape[0]
        
        fig,axs = plt.subplots(2,1,figsize=(12,6),
                               constrained_layout=True,sharex=True,sharey=True)
        
        for a in range(2):
            ax = axs[a]
            
            if a == 0:
                plotidx = nasst
                idx_name = "NASST"
                ax.set_title("NASST (top) and AMV (bot.) Indices (%s, %i Member Ensemble)" % (dataset_names[d],nens))
            else:
                plotidx = amvid_lp
                idx_name = "AMV"
                ax.set_xlabel("Years")
            
            for e in range(nens):
                ax.plot(yrs,plotidx[e,:],label="",alpha=0.1,color="k")
            ax.plot(yrs,plotidx[1,:],label="Individual Member",alpha=0.1,color="k")
            ax.plot(yrs,plotidx[:,:].mean(0),label="Ensemble Average.",alpha=1,color="k")
            ax.grid(True,ls='dotted',color="k")
            ax.set_xlim([ystart,yend])
            ax.legend()
            ax.set_ylabel("%s ($\degree C$)" % (idx_name))
            ax.axhline([0],ls="dashed",lw=0.75,color="k")
            
        plt.savefig("%s%s_AMV_Indices.png" % (figpath,dataset_names[d]),dpi=150,bbox_inches="tight",transparent=True)
    
    # Plot AMV
    indata   = ds.sst.values#ds_all[d].sst.values
    nens,ntime,nlat,nlon = indata.shape
    indata   = indata.transpose(0,3,2,1) #  --> [ens x lon x lat x time]
    
    # Preallocate and regress (need to remove NaNs or write a functions)
    amvpats   = np.zeros((nlon,nlat,nens))
    nasstpats = amvpats.copy()
    inidx     = nasst
    inidx_lp  = amvid_lp
    for e in range(nens):
        
        nasstpats[:,:,e] = proc.regress2ts(indata[e,...],inidx[e,:]/inidx[e,:].std())
        amvpats[:,:,e]   = proc.regress2ts(indata[e,...],inidx_lp[e,:]/inidx_lp[e,:].std())
    
    amvpats_all.append(amvpats.transpose(2,1,0)) # [ens x lat x lon]
    nasstpats_all.append(nasstpats.transpose(2,1,0))
    
    amvid_all.append(amvid_lp)
    nasst_all.append(nasst)

#%% Visualize the [ensemble mean] AMVs for each large ensemble

lon = ds.lon.values
lat = ds.lat.values
plotdatasets = np.arange(0,ndata)

ylabelnames = ("NASST","AMV")

cints       = np.arange(-2,2.1,0.1)
fig,axs = plt.subplots(2,ndata,figsize=(14,5.5),subplot_kw={'projection':ccrs.PlateCarree()},
                           constrained_layout=True)
for d in range(len(plotdatasets)):
    print(d)
    for ii in range(2):
        ax = axs[ii,d]
        ax.coastlines()
        ax.set_extent(amvbbox)
        
        if ii == 0:   # Plot the NASST Pattern
            plotpat = nasstpats_all[d].mean(0)
            ax.set_title("%s, nens=%i" % (dataset_long[d],nasstpats_all[d].shape[0]))
        elif ii == 1: # Plot the AMV Pattern
            plotpat = amvpats_all[d].mean(0)
            
        cf = ax.contourf(lon,lat,plotpat,levels=cints,cmap="RdBu_r",extend="both")
        cl = ax.contour(lon,lat,plotpat,levels=cints,colors="k",linewidths=0.45)
        ax.clabel(cl,cints[::2],fontsize=8)
        
        if d == 0:
            ax.text(-0.05, 0.55, ylabelnames[ii], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
fig.colorbar(cf,ax=axs.flatten(),orientation='horizontal',fraction=.045)
            
            
savename = "%sAMV_NASST_Patterns_EnsAvg_CMIP6_LENS.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Plot AMVs for 10 members of each large ensemble

plotnums    = np.arange(1,11)
idxmode     = "AMV"

cints       = np.arange(-2.6,2.8,0.2)


fig,axs  = plt.subplots(len(plotdatasets),10,figsize=(20,8.0),subplot_kw={'projection':ccrs.PlateCarree()},
                           constrained_layout=True)

for d in range(len(plotdatasets)):
    
    if idxmode == "AMV":
        inpats = amvpats_all[d]
    elif idxmode == "NASST":
        inpats = nasstpats_all[d]
    
    for e in range(10):
        
        ax = axs[d,e]
        ax.coastlines()
        ax.set_extent(amvbbox)
        
        if d == 0:
            ax.set_title("Member %i" % (plotnums[e]))
            
        if e == 0:
            ax.text(-0.05, 0.55, dataset_names[d], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
        

        plotpat = inpats[e,...]
        cf = ax.contourf(lon,lat,plotpat,levels=cints,cmap="RdBu_r",extend="both")
        cl = ax.contour(lon,lat,plotpat,levels=cints,colors="k",linewidths=0.45)
        ax.clabel(cl,cints[::2],fontsize=8)
        
            
fig.colorbar(cf,ax=axs.flatten(),orientation='horizontal',fraction=.035)
plt.suptitle("%s Regression Patterns ($\degree$C per 1$\sigma_{%s}$)" % (idxmode,idxmode),fontsize=16)


savename = "%sAMV_NASST_Patterns_10mem_CMIP6_LENS.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches="tight")
#%%



