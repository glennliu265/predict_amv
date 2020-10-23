#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate AMV Index from HadISST Dataset

General Procedure:
    1 - Load data and calculate monthly anomalies
    2 - Detrend data at each point
    3 - Calculate AMV Index
        i)  Take Area-weighted Average
        ii) Apply Low-pass butterworth filter



@author: gliu
"""


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys

## User Specific Edits <START> ----

# Path settings ...
# Path to directory containing data, downloaded from link below
# https://drive.google.com/drive/u/0/folders/1o0R4RSj34HNInR9ehZ9Yw2pCiZGRPo-s
datpath = "/Users/gliu/Downloads/2020_Fall/6.862/Project/Data/"

# Path to module
modpath = "/Users/gliu/Downloads/2020_Fall/6.862/Project/predict_amv/"

# Output Path
outpath = "/Users/gliu/Downloads/2020_Fall/6.862/Project/Data/proc/"


# AMV Calculation settings ... 
# Indicate start and ending year
start = "1920-01-01"
end   = "2005-12-01"

# Indicate the lat/lon bounds (lon is degrees West, -180-180)
lonW = -80
lonE = 0
latS = 0
latN = 65

# AMV Index Filtering Options
cutofftime = 120 # In units of months
awgt       = 1   # Area weighting (0=none,1=cos(lat),2=sqrt(cos(lat)))
order      = 5   # Order of butterworth filter

# Degree of polynomial for detrending
deg = 4

# Debug mode, makes some plots to check detrending, amv, etc
debug = 1

## User Edits <END> ----
#%% 1) Load Data and calculate monthly anomalies ----

# Import module with functions
sys.path.append(modpath)
import amvmod as amv

# Open dataset and slice to time period
sstname ="CESM1LE_sst_NAtl_19200101_20051201.nc"
ds = xr.open_dataset(datpath+sstname)
ds = ds.sel(time=slice(start,end))

# Load data from DataArray to numpy array
sst = ds['sst'].values #[1740 x 180 x 360]
lon = ds['lon'].values
lat = ds['lat'].values   
mon = ds['time'].values
nlat,nlon,nmon,nens = sst.shape


# Calculate monthly anomalies
sst  = sst.transpose(2,3,0,1)  # [ntime,nens,nlat,nlon]
sstm = sst.reshape(int(np.ceil(nmon/12)),12,nens*nlat*nlon) # Separate mon/year, combine lat/lon/ens
ssta = sstm - sstm.mean(0)[None,:,:] # [86 x 12 x 358848]
ssta = ssta.reshape(np.prod(ssta.shape[:2]),nens*nlat*nlon) #Recombine mon/year with ensemble [1032, 358848]

# %% 2) Perform detrending at each point ----

if detrend_method == 1:
    x = np.arange(0,nmon)
    okdata,knan,okpts = amv.find_nan(ssta.T,1) # Find Non-Nan Points
    okdt,model = amv.detrend_poly(x,okdata,deg)
    okdt = okdt.T # [space x time]
    
    # Test visualize detrending
    if debug == 1:
        pt = 4026
        fig,ax=plt.subplots(1,1)
        plt.style.use('seaborn')
        ax.scatter(x,okdata[pt,:],label='raw')
        ax.plot(x,model[pt,:],label='fit',color='k')
        ax.scatter(x,okdt[pt,:],label='detrended')
        ax.legend()
        ax.set_title("SST Detrended , %i Deg. Polynomial"%deg)
    
    # Refill NaN values
    sstdt = np.ones((nens*nlat*nlon,nmon)) * np.nan
    sstdt[okpts,:] = okdt
    sstdt = sstdt.reshape(nens,nlat,nlon,nmon)
    if debug == 1:
        plt.pcolormesh(lon,lat,sstdt[0,:,:,0]),plt.colorbar(),plt.title("SST Detrended")
else: # Detrend by moving ensemble average
    ssta = ssta.reshape(nmon,nens,nlat,nlon)
    sstdt = ssta - ssta.mean(1)[:,None,:,:]
    
    if debug == 1:
        fig,ax=plt.subplots(1,1)
        ax.plot(ssta[:,0,50,53],label='raw')
        ax.plot(sstdt[:,0,50,53],label='detrended')
        ax.legend()
    
    sstdt = sstdt.transpose(1,2,3,0)


# %% Calculate AMV Index ----

amvidx = np.ones((nens,nmon)) * np.nan
for e in range(nens):
    
    insst = sstdt[e,:,:,:].squeeze()
    amvidx[e,:],_   = amv.calc_AMV(lon,lat,insst,[lonW,lonE,latS,latN],order,cutofftime,1)
    print("Completed ensemble %i"%(e+1))
    

if debug == True:
    e = 12
    
    xtks = np.arange(0,nmon,120)
    xlb = np.arange(int(start[0:4]),int(end[0:4]),10)
    fig,ax=plt.subplots(1,1)
    ax = amv.plot_AMV(amvidx[e,:].squeeze(),ax=ax)
    ax.set_xticks(xtks)
    ax.set_xticklabels(xlb)
    ax.set_xlabel("Year")
    ax.set_ylabel("AMV Index")
    ax.set_title("AMV Index, HadISST %i-%s" % (xlb[0],end[0:4]))

# Save data
np.save("%sCESM1LE_AMVIndex_%s-%s.npy" % (outpath,start[0:4],end[0:4]),amvidx)
