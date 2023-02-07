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

ds_all = []
ndata  = len(dataset_names)
for d in range(1,ndata):
    # Open the dataset
    ncsearch = "%s%s_%s_NAtl_%sto%s_detrend%i_regrid%sdeg.nc" % (datpath,dataset_names[d],
                                                                 varname,ystart,yend,
                                                                 detrend,regrid)
    ds       = xr.open_dataset(ncsearch).load()
    ds_all.append(ds)
    
    # Compute the mean over the area
    ds        = ds.sel(lon=slice(amvbbox[0],amvbbox[1]),lat=slice(amvbbox[2],amvbbox[3]))
    dsidx     = (np.cos(np.pi*ds.lat/180) * ds).mean(dim=('lat','lon'))
    nasst     = dsidx.sst.values #[ensemble x time]
    
    # Compute the low-passed version [ensemble x time]
    amvid_lp  = proc.lp_butter(nasst.T[...,None],10,order=6).squeeze().T
    
    # Save the labels (normal and low-pass filtered)
    savename       = "%s%s_%s_label_%sto%s_detrend%i_regrid%sdeg_lp0.npy" % (datpath,dataset_names[d],
                                                                         varname,ystart,yend,
                                                                         detrend,regrid)
    
    savename_lp    = savename.replace("lp0","lp1")
    
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


#%% Plot the AMV pattern for each large ensemble


nasstpats_all = []
amvpats_all   = []

for d in range(ndata):

    indata   = ds_all_nomask[d].sst.values
    nens,ntime,nlat,nlon = indata.shape
    indata   = indata.transpose(0,3,2,1) # [ens x lon x lat x time]
    inidx    = amvids[d].sst.values
    inidx_lp = amvids_lp[d]
    
    amvpats = np.zeros((nlon,nlat,nens))
    nasstpats = amvpats.copy()
    for e in range(nens):
        
        nasstpats[:,:,e] = proc.regress2ts(indata[e,...],inidx[e,:]/inidx[e,:].std())
        amvpats[:,:,e]   = proc.regress2ts(indata[e,...],inidx_lp[:,e]/inidx_lp[:,e].std())
        
    amvpats_all.append(amvpats.transpose(2,1,0)) # [ens x lat x lon]
    nasstpats_all.append(nasstpats.transpose(2,1,0))




