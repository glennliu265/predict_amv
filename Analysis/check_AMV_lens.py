#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check AMV LENS

Examine AMV Patterns for CMIP6 MMLE
    Works with output calculated from [prep_data_lens.py]
    Copies sectinos from [prep_labels_lens.py]

Explores the following effects
    - Detrending/Ens Mean Removal (Index AND Variable) [x]
    - LP Filtering 
    - Normalization [x]
    - Time Period
    - AMV Bounding Box [x]

Created on Fri Feb 10 10:09:22 2023

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
    ice_nc         = "%s../other/siconc_mon_clim_ensavg_CMIP6_10ice_re1x1.nc" % (datpath) # Full path and file of ice mask
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

# Load the bounding box
bbox            = pparams.bbox

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
print(amvbbox)

#%% Load the data

ds_all        = []
ndata         = len(dataset_names)
for d in range(ndata):
    # Open the dataset
    ncsearch  = "%s%s_%s_NAtl_%sto%s_detrend%i_regrid%sdeg.nc" % (datpath,dataset_names[d],
                                                                 varname,ystart,yend,
                                                                 detrend,regrid)
    ds        = xr.open_dataset(ncsearch).load()
    ds_all.append(ds)
#%% Load Normalization Factors

nfactors      = []
ndata         = len(dataset_names)
for d in range(ndata):
    # Open the dataset
    npfile = "%s%s_nfactors_sst_detrend0_regridNonedeg_lon-90to20_lat0to90_1850to2014.npy" % (datpath,dataset_names[d])
    ld = np.load(npfile,allow_pickle=True)
    nfactors.append(ld)

# Get Lat.Lon for later (plotting, cropping, etc)
lon        = ds_all[0].lon.values
lat        = ds_all[0].lat.values

#%% Load universal ice mask (created using the following line from terminal:)
# cdo remapbil,regrid_re1x1.nc siconc_mon_clim_ensavg_CMIP6_10ice.nc siconc_mon_clim_ensavg_CMIP6_10ice_re1x1.nc

# Load ice concentrations
ds_ice = xr.open_dataset(ice_nc).load()

# Flip longitude
ds_ice = proc.lon360to180_ds(ds_ice,lonname="lon")

# Match the region
ds_ice_reg = ds_ice.sel(lon=slice(lon[0],lon[-1]),lat=slice(lat[0],lat[-1])).load()

# Change all fractional values on ice-free points (<10%) to 1 for masking
ice_mask           = ds_ice_reg.siconc.values
ice_shape          = ice_mask.shape
ice_free           = ~np.isnan(ice_mask)
ice_mask[ice_free] = 1 # [month x lat x lon]

#%% Set some computation options

apply_nfactor   = True
normalize_idx   = True
detrend_idx     = True
detrend_var     = True
maskice         = True

# Set bounding box options
print(amvbbox) # Print the default
amvbbox = [-80, 0, 0, 65]
bbstr_fn,bbstr_title=proc.make_locstring_bbox(amvbbox)

# Set time crop option
cropstart = 1920
cropend   = 2005
cropstr   = "%ito%i" % (cropstart,cropend)

# Expstr
expstr = "%s_%s_nfactor%i_normidx%i_icemask%i_dtidx%i_dtvar%i" % (bbstr_fn,cropstr,
                                                                  apply_nfactor,
                                                               normalize_idx,maskice,
                                                               detrend_idx,detrend_var)

#%% Compute indices


# Crop time for each ds
ds_all = [ds.sel(year=slice(cropstart,cropend)) for ds in ds_all]

nasst_all     = []
amvid_all     = []
for d in range(ndata):
    # Compute the mean over the area
    ds        = ds_all[d]
    
    if maskice:
        ds = ds * ice_mask.sum(0) # Apply Ice Mask, only including entirely ice free points in the index calculation
    
    ds        = ds.sel(lon=slice(amvbbox[0],amvbbox[1]),lat=slice(amvbbox[2],amvbbox[3]))
    dsidx     = (np.cos(np.pi*ds.lat/180) * ds).mean(dim=('lat','lon'))
    nasst     = dsidx.sst.values #[ensemble x time]
    
    # Compute the low-passed version [ensemble x time]
    amvid_lp  = proc.lp_butter(nasst.T[...,None],10,order=6).squeeze().T
    
    amvid_all.append(amvid_lp)
    nasst_all.append(nasst)

#%% Compute patterns



# Set some options
amvpats_all   = []
nasstpats_all = []
for d in tqdm(range(ndata)):
    
    # Plot AMV
    indata   = ds_all[d].sst.values
    nens,ntime,nlat,nlon = indata.shape
    indata   = indata.transpose(0,3,2,1) #  --> [ens x lon x lat x time]
    
    # Preallocate and regress (need to remove NaNs or write a functions)
    amvpats   = np.zeros((nlon,nlat,nens))
    nasstpats = amvpats.copy()
    idx     = nasst_all[d].copy()
    idx_lp  = amvid_all[d].copy()
    
    if apply_nfactor:
        mu,sigma=nfactors[d]
        idx    = idx * sigma + mu
        idx_lp = idx_lp * sigma + mu
        indata = indata * sigma + mu
        
    
    if normalize_idx:
        idx    = idx      / idx.std(1)[:,None]
        idx_lp   = idx_lp   / idx_lp.std(1)[:,None]
        
    
    if detrend_idx:
        idx    = idx - idx.mean(0)[None,:]
        idx_lp = idx_lp - idx_lp.mean(0)[None,:]
        
    
    if detrend_var:
        indata = indata - indata.mean(0)[None,...]
        
    for e in range(nens):
        nasstpats[:,:,e] = proc.regress2ts(indata[e,...],idx[e,:],nanwarn=0,verbose=False)
        amvpats[:,:,e]   = proc.regress2ts(indata[e,...],idx_lp[e,:],nanwarn=0,verbose=False)
    
    amvpats_all.append(amvpats.transpose(2,1,0)) # [ens x lat x lon]
    nasstpats_all.append(nasstpats.transpose(2,1,0))

#%% Examine the indices


plot_allens = False
yrs         = np.arange(cropstart,cropend+1,1)

    
fig,axs = plt.subplots(2,1,figsize=(12,6),
                        constrained_layout=True,sharex=True,sharey=True)
for d in range(ndata):
    
    for a in range(2):
        
        ax = axs[a]
        
        if a == 0:
            plotidx = nasst_all[d]
            idx_name = "NASST"
            ax.set_title("NASST (top) and AMV (bot.) Indices (%s, %i Member Ensemble)" % (dataset_names[d],nens))
        else:
            plotidx = amvid_all[d]
            idx_name = "AMV"
            ax.set_xlabel("Years")
        
        nens = plotidx.shape[0]
        
        # Compute mean and stdv
        mu    = plotidx.mean(0)
        if plot_allens:
            for e in range(nens):
                ax.plot(yrs,plotidx[e,:],label="",alpha=0.01,color=dataset_colors[d],zorder=-9)
        else:
            sigma = plotidx.std(0)
            ax.fill_between(yrs,mu-sigma,mu+sigma,alpha=.1,color=dataset_colors[d],zorder=1)
        
        
        #ax.plot(yrs,plotidx[1,:],label="Individual Member",alpha=0.1,color=dataset_colors[d])
        leglab="%s (n=%i)" % (dataset_names[d],nens)
        ax.plot(yrs,mu,label=leglab,alpha=1,color=dataset_colors[d])
        ax.grid(True,ls='dotted',color="k")
        ax.set_xlim([ystart,yend])
        if a == 0:
            ax.legend(ncol=3,loc="upper center")
        ax.set_ylabel("%s ($\degree C$)" % (idx_name))
        ax.axhline([0],ls="dashed",lw=0.75,color="k")
        
plt.savefig("%sAllEns_AMV_Indices.png" % (figpath),dpi=150,bbox_inches="tight",transparent=True)

#TODO (Testing..)
#%% Visualize the [ensemble mean] AMVs for each large ensemble

plotdatasets = np.arange(0,ndata)
ylabelnames  = ("NASST","AMV")

cints        = np.arange(-1.0,1.1,.1) # np.arange(-1.6,1.8,.2)  # np.arange(-.50,0.55,0.05)#)  #
fig,axs      = plt.subplots(2,ndata,figsize=(14,5.5),subplot_kw={'projection':ccrs.PlateCarree()},
                           constrained_layout=True)
for d in range(len(plotdatasets)):
    
    print(d)
    for ii in range(2):
        ax   = axs[ii,d]
        ax.coastlines()
        ax.set_extent(amvbbox)
        
        if ii == 0:   # Plot the NASST Pattern
            plotpat = nasstpats_all[d].mean(0)
            ax.set_title("%s, nens=%i" % (dataset_long[d],nasstpats_all[d].shape[0]))
        elif ii == 1: # Plot the AMV Pattern
            plotpat = amvpats_all[d].mean(0)
            
        cf   = ax.contourf(lon,lat,plotpat,levels=cints,cmap="RdBu_r",extend="both")
        cl   = ax.contour(lon,lat,plotpat,levels=cints,colors="k",linewidths=0.45)
        ax.clabel(cl,cints[::2],fontsize=8)
        
        if d == 0:
            ax.text(-0.05, 0.55, ylabelnames[ii], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
fig.colorbar(cf,ax=axs.flatten(),orientation='horizontal',fraction=.045)

savename = "%sAMV_NASST_Patterns_EnsAvg_CMIP6_LENS_%s.png" % (figpath,expstr)
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Plot AMVs for 10 members of each large ensemble

plotnums = np.arange(1,11)
idxmode  = "AMV"

#cints   = np.arange(-4.0,4.4,0.4)

fig,axs  = plt.subplots(len(plotdatasets),10,figsize=(20,12),subplot_kw={'projection':ccrs.PlateCarree()},
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
    
