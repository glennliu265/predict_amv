#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check CMIP6 LENS

- copied from check_lens_data.py, but for cmip6 rather than cmip5 mmle

Created on Tue Feb  7 13:52:22 2023

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
# dataset_names  = ("canesm2_lens" ,"csiro_mk36_lens","gfdl_esm2m_lens","mpi_lens"  ,"CESM1")
# dataset_long   = ("CCCma-CanESM2","CSIRO-MK3.6"    ,"GFDL-ESM2M"     ,"MPI-ESM-LR","NCAR-CESM1")
# dataset_colors = ("r"            ,"b"              ,"magenta"        ,"gold" ,"limegreen")
# dataset_starts = (1950           ,1920             ,1950             ,1920        ,1920)
varname        = "sst"

# Preprocessing and Cropping Options
detrend        = False
bbox           = [-90,20,0,90] # Crop Selection
bbox_fn        = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])
amvbbox        = [-80,0,0,65]  # AMV Index Calculation
apply_limasks  = False

# Paths
machine = "gliu_mbp"
if machine == "stormtrack":
    lenspath       = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/ts/"
    datpath        = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/processed/"
elif machine == "gliu_mbp":
    datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/LENS_other/processed/"
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
    import viz,proc

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
dataset_names = pparams.dataset_names
dataset_long  = pparams.dataset_long
dataset_colors= pparams.dataset_colors
dataset_starts= pparams.dataset_starts

#%% Load datasets in to dataarrays

# Load datasets for each
ndata  = len(dataset_names)
ds_all = []
ds_all_nomask = []
ds_landmask = []
ds_icemask  = []

for d in range(ndata):
    
    
    # Load masked data
    savename       = "%s%s_%s_NAtl_%sto2005_detrend%i_regrid%sdeg.nc" % (datpath,
                                                                         dataset_names[d],
                                                                         varname,
                                                                         dataset_starts[d],
                                                                         detrend,regrid)
    ds = xr.open_dataset(savename).load()
    ds_all.append(ds)
    
    # Load unmasked data
    savename       = "%s/../processed_nomask/%s_%s_NAtl_%sto2005_detrend%i_regrid%sdeg.nc" % (datpath,
                                                                         dataset_names[d],
                                                                         varname,
                                                                         dataset_starts[d],
                                                                         detrend,regrid)
    ds = xr.open_dataset(savename).load()
    ds_all_nomask.append(ds)
    
    # Load masks
    mmnames = ("land", "ice")
    mmds    = (ds_landmask, ds_icemask)
    for mm in range(2):
        savename       = "%s/../processed_nomask/%s_mask_%s_byens_regrid%sdeg.npy" % (datpath,
                                                                                   mmnames[mm],
                                                                                   dataset_names[d],
                                                                                   regrid
                                                                                   )
        
        msk = np.load(savename)
        # Load global lat/lon for selection
        if "CESM1" in dataset_names[d]:
            ds = xr.open_dataset("%s../ensAVG/%s_htr_ts_regrid%ideg_ensAVG_nomask.nc" % (datpath,dataset_names[d],regrid))
        else:
            ds = xr.open_dataset("%s../ensAVG/%s_ts_regrid%ideg_ensAVG_nomask.nc" % (datpath,dataset_names[d],regrid))
        longlob = ds.lon.values
        latglob = ds.lat.values
        # Quickly select the target region
        mskreg,lonr,latr = proc.sel_region(msk.transpose(2,1,0),longlob,latglob,bbox)
        mmds[mm].append(mskreg.transpose(2,1,0))
        
        
dataset_enssize = [len(ds.ensemble) for ds in ds_all] # Get Ensemble Sizes
samplesize      = dataset_enssize * (2005-np.array(dataset_starts)) # Rough Calculation of sample size

print([len(ds.lon) for ds in ds_all])
print([len(ds.lat) for ds in ds_all])

print([ds.lon for ds in ds_all])
# Get lat/lon
lon = ds_all[0].lon.values
lat = ds_all[0].lat.values

#%% Visualize land/ice masks

fig,axs = plt.subplots(2,ndata,figsize=(14,5.5),subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True)
for d in range(ndata):
    
    for ii in range(2):
        ax = axs[ii,d]
        ax.coastlines()
        ax.set_extent(amvbbox)
        
        
        plotpat = mmds[ii][d].mean(0)
        if ii == 0: # Plot the Land Mask
            ax.set_title("%s" % (dataset_long[d]))
        
            
        pcm = ax.pcolormesh(lon,lat,plotpat,cmap="RdBu_r")
        # ax.clabel(cl,cints[::2],fontsize=8)
        
        # if d == 0:
        #     ax.text(-0.05, 0.55, ylabelnames[ii], va='bottom', ha='center',rotation='vertical',
        #             rotation_mode='anchor',transform=ax.transAxes)
fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=.045)
#%% Decide whether or not to use land ice masks

ds_masked = ds_all.copy()
if apply_limasks:
    ds_in = ds_all.copy()
else:
    ds_in = ds_all_nomask.copy()
    
ds_all = []
for d in range(ndata): # Apply landmask
    # Apply landmask
    ds_all.append(ds_in[d] * ds_landmask[d][:,None,:,:])

#%% Compute NASST from each dataset

amvids    = []
amvids_lp = []
for d in range(ndata):
    ds        = ds_masked[d].sel(lon=slice(amvbbox[0],amvbbox[1]),lat=slice(amvbbox[2],amvbbox[3]))
    dsidx     = (np.cos(np.pi*ds.lat/180) * ds).mean(dim=('lat','lon'))
    amvids.append(dsidx)
    
    amvid_lp = proc.lp_butter(dsidx.sst.values.T[...,None],10,order=6).squeeze()
    amvids_lp.append(amvid_lp)
    
    # Save the labels
    savename       = "%s%s_nasst_label_%sto2005_detrend%i_regrid%sdeg.npy" % (datpath,
                                                                         dataset_names[d],
                                                                         dataset_starts[d],
                                                                         detrend,regrid)
    np.save(savename,dsidx.sst.values)
    print("Saved Target to %s"%savename)
    
#%% Visualize the ens-avg timeseries for each large ensemble

fig,ax = plt.subplots(1,1,figsize=(12,4),constrained_layout=True)

for d in range(ndata):
    t = np.arange(dataset_starts[d],2005+1,1)
    label = "%s (N=%i)" % (dataset_long[d],dataset_enssize[d])
    ax.plot(t,amvids[d].sst.mean('ensemble'),label="",color=dataset_colors[d],lw=2.5,alpha=0.5)
    ax.plot(t,amvids_lp[d].mean(1),label=label,color=dataset_colors[d],lw=1.5)
    

ax.axhline([0],ls="dashed",lw=0.75,color="k")
ax.set_ylim([-1.25,1.25])
ax.set_xlim([1920,2005])
ax.grid(True,ls="dotted")
ax.set_title("Ensemble Average AMV and NASST Index")
ax.set_xlabel("Years")
ax.set_ylabel("Index Value ($\degree$C)")
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

#%% Calculate and visualize the AMV pattern

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

#%% Visualize the AMV Patterns

ylabelnames = ("NASST","AMV")

cints   = np.arange(-1,1.1,0.1)
fig,axs = plt.subplots(2,ndata,figsize=(14,5.5),subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True)
for d in range(ndata):
    
    for ii in range(2):
        ax = axs[ii,d]
        ax.coastlines()
        ax.set_extent(amvbbox)
        
        if ii == 0:   # Plot the NASST Pattern
            plotpat = nasstpats_all[d].mean(0)
            ax.set_title("%s" % (dataset_long[d]))
        elif ii == 1: # Plot the AMV Pattern
            plotpat = amvpats_all[d].mean(0)
            
        cf = ax.contourf(lon,lat,plotpat,levels=cints,cmap="RdBu_r",extend="both")
        cl = ax.contour(lon,lat,plotpat,levels=cints,colors="k",linewidths=0.45)
        ax.clabel(cl,cints[::2],fontsize=8)
        
        if d == 0:
            ax.text(-0.05, 0.55, ylabelnames[ii], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
fig.colorbar(cf,ax=axs.flatten(),orientation='horizontal',fraction=.045)
            
            
savename = "%sAMV_NASST_Patterns_EnsAvg_LENS.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Visualize the distribution of + and - AMV events

# Some visualization toggles
makepie = False # If false, include a barplot instead
addtxt  = True # If true, include count in each class

if makepie:
    fig,axs =  plt.subplots(ndata,2,figsize=(8,12),constrained_layout=True,
                            sharex=True,sharey=False)
else:
    
    fig,axs =  plt.subplots(ndata,1,figsize=(8,12),constrained_layout=True,
                            sharex=True,sharey=False)
    
binedges = np.arange(-1.5,1.6,.1)
for d in range(ndata):
    
    # Make the Bar Plot
    if makepie:
        ax = axs[d,0]
    else:
        ax = axs[d]
    plotdata = amvids[d].sst.values.flatten()
    mu    = np.mean(plotdata)
    stdev = np.std(plotdata)
    
    ax.hist(plotdata,bins=binedges,edgecolor="k",alpha=0.60,color=dataset_colors[d])
    
    ax.axvline([mu]      ,ls="solid",lw=0.7,color="k")
    ax.axvline([mu+stdev],ls="dashed",lw=0.7,color="k")
    ax.axvline([mu-stdev],ls="dashed",lw=0.7,color="k")
    

    cntpos       = np.sum(plotdata > mu+stdev)
    cntneg       = np.sum(plotdata < mu-stdev)
    cntneu       = np.sum( (plotdata < mu+stdev) * (plotdata > mu-stdev) )
    class_counts = [cntpos,cntneu,cntneg]
    
    
    title = "%s (N=%i) \n $\mu=%.2e$, $\sigma=%.2f$" % (dataset_long[d],
                                                        dataset_enssize[d],
                                                        mu,
                                                        stdev)
    
    # Text Labels (too messy, but works for single panel..)
    if addtxt:
        ax.text(0.05,.7,"AMV-\n%i" % (cntneg),transform=ax.transAxes,
                bbox=dict(facecolor='w', alpha=0.2))
        ax.text(0.45,.7,"Neutral\n%i" % (cntneu),transform=ax.transAxes,
                bbox=dict(facecolor='w', alpha=0.2))
        ax.text(0.75,.7,"AMV+\n%i" % (cntpos),transform=ax.transAxes,
                bbox=dict(facecolor='w', alpha=0.2))
        
    # pcts   = np.array([cntneg,cntneu,cntpos])/len(plotdata)
    # ax.text(0.05,.7,"AMV-\n %i \n%.2f" % (cntneg,pcts[0]),transform=ax.transAxes,
    #         bbox=dict(facecolor='w', alpha=0.2))
    # ax.text(0.40,.7,"Neutral\n %i \n%.2f" % (cntneu,pcts[1]),transform=ax.transAxes,
    #         bbox=dict(facecolor='w', alpha=0.2))
    # ax.text(0.75,.7,"AMV+\n %i \n%.2f" % (cntpos,pcts[2]),transform=ax.transAxes,
    #         bbox=dict(facecolor='w', alpha=0.2))
    ax.set_title(title)
    ax.grid(True,ls="dotted")
    
    
    # Make pie plot
    if makepie:
        ax =axs[d,1]
        labels = ["%s\n %.2f" % (classes[i],class_counts[i]/len(plotdata)*100)+"%" for i in range(3)]
        ax.pie(class_counts,colors=class_colors,
                labels=labels,labeldistance=1)
savename = "%sNASST_%s_Histogram_Lens_makepie%i.png" % (figpath,bbox_fn,makepie)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%%

fig,ax = plt.subplots(1,1)
labels = ["%s\n(%.2f)" % (classes[i],class_counts[i]/len(plotdata)*100)+"%" for i in range(3)]
ax.pie(class_counts,colors=class_colors,
       labels=labels,labeldistance=0.4)


#%% Compute Power Spectra, AMV, Etc
