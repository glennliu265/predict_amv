#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Get atmospheric variable

Took upper section from [preproc_CESM1_LENS.py]

Created on Thu May 25 12:16:28 2023

@author: gliu

"""

import numpy as np
import cartopy as crs
import xarray as xr

from tqdm import tqdm

import time
import sys
import glob

#machine   = "stormtrack"

datpath   = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/atm/proc/tseries/monthly/"
modpath   = sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")

from amv import loaders,proc,viz

#%% 

# General Information
nens = 42

# Part 1 (Land/Ice Mask Creation)
mask_sep    = True
vnames      = ("LANDFRAC","ICEFRAC") # Variables
mthres      = (0.30,0.05) # Mask out if grid ever exceeds this value

# Part 2 ()
maskmode = "enssum"
mconfig  = 'htr' # ['rcp85','htr']

if mconfig == "rcp85":
    outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_RCP85/01_PREPROC/"
    mnum    = np.concatenate([np.arange(1,36),np.arange(101,106)])
    ntime = 1140
elif mconfig == "htr":
    outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_HTR/"
    mnum    = np.concatenate([np.arange(1,36),np.arange(101,108)])
    ntime = 1032


def load_atmvar(vname,mnum,datpath,preproc=None): # Just load everything, and preprocess if necessary.
    nens = len(mnum)
    for e in tqdm(range(nens)):
        N = mnum[e]
        
        if mconfig =='rcp85':
            ds =loaders.load_rcp85(vname,N,datpath=datpath)
        elif mconfig == 'htr':
            ds = loaders.load_htr(vname,N,datpath=datpath)
        if preproc is not None:
            ds = preproc(ds)
        invar = ds.values # [Time x Lat x Lon]
        
        if e == 0:
            ntime,nlat,nlon = invar.shape
            var_allens = np.zeros((nens,ntime,nlat,nlon))
            times = ds.time.values
            lon   = ds.lon.values
            lat   = ds.lat.values
        var_allens[e,...] = invar.copy()
    return var_allens,times,lat,lon


#%% Load the atmospheric variables

vnames   = ["ICEFRAC","LANDFRAC"]
maskvars = []
for v in range(len(vnames)):
    
    vname = vnames[v]
    outvar,times,lat,lon = load_atmvar(vname,mnum,datpath)
    maskvars.append(outvar)



#%% Save the mean ice concentration

icepath  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/ICEFRAC/"
landpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/LANDFRAC/"

meanicefrac = maskvars[0].mean(1)
savename    = "%sCESM1LE_HTR_ICEFRAC_mean.nc" % icepath
proc.numpy_to_da(meanicefrac,np.arange(1,43),lat,lon,"ICEFRAC",savenetcdf=savename)

#%% Save the maxice concentration

maxicefrac = maskvars[0].max(1)
savename    = "%sCESM1LE_HTR_ICEFRAC_max.nc" % icepath
proc.numpy_to_da(maxicefrac,np.arange(1,43),lat,lon,"ICEFRAC",savenetcdf=savename)
print(savename)

#%% Make and check the masks (for ICEFRAC and LANDFRAC)

mthres = (0.05,0.30)
mask = [] #np.ones((nens,192,288))
if mask_sep:
    lmasks = []
    imasks = []
# Loop for each ensemble member
for e in tqdm(range(nens)):
    N = mnum[e] # Get ensemble member
    emask = np.ones((192,288)) * np.nan # Preallocate
    if mask_sep:
        imask = np.ones((192,288)) 
        lmask = np.ones((192,288)) * np.nan
    for v in range(2): 
        
        # Load dataset
        vname = vnames[v]
        invar = maskvars[v][e,...]
        inthres = mthres[v]
        # Mask (set 0 where it ever exceeds, propagate in time)
        
        if v == 1: # Landmask
            
            maskpts       = ((invar <= inthres).prod(0)) # [Lat x Lon]
            emask[maskpts==1] = 1 # 1 means it is ocean point
            if mask_sep:
                lmask[maskpts==1] = 1
        elif v == 0:
            maskpts       = ((invar <= inthres).prod(0)) # [Lat x Lon]
            emask[maskpts==0] = np.nan # 0 means it has sea ice
            if mask_sep:
                imask[maskpts==0] = np.nan # All points 1, ice points NaN
    # Save masks separately
    if mask_sep:
        imasks.append(imask)
        lmasks.append(lmask)
    mask.append(emask.copy())
# Make into array
mask = np.array(mask)  # [ENS x LAT x LON]
imasks = np.array(imasks)
lmasks = np.array(lmasks)

#%% Save each array

savename    = "%sCESM1LE_HTR_icemask_allens.nc" % icepath
da = proc.numpy_to_da(imasks,np.arange(1,43),lat,lon,"ICEMASK",savenetcdf=savename)
print(savename)

savename    = "%sCESM1LE_HTR_landmask_allens.nc" % landpath
da = proc.numpy_to_da(lmasks,np.arange(1,43),lat,lon,"LANDMASK",savenetcdf=savename)
print(savename)

savename    = "%sCESM1LE_HTR_limask_allens.nc" % icepath
da = proc.numpy_to_da(mask,np.arange(1,43),lat,lon,"MASK",savenetcdf=savename)
print(savename)

#%%

#%% Make the mask

# Initialize Mask
mask = [] #np.ones((nens,192,288))
if mask_sep:
    lmasks = []
    imasks = []
# Loop for each ensemble member
for e in tqdm(range(nens)):
    N = mnum[e] # Get ensemble member
    emask = np.ones((192,288)) * np.nan # Preallocate
    if mask_sep:
        imask = np.ones((192,288))
        lmask = emask.copy()
    for v in range(2):
    
        # Load dataset
        vname = vnames[v]
        if mconfig =='rcp85':
            ds =loaders.load_rcp85(vname,N,datpath=datpath)
        elif mconfig == 'htr':
            ds = loaders.load_htr(vname,N,datpath=datpath)
        invar = ds.values # [Time x Lat x Lon]
        
        # Mask (set 0 where it ever exceeds, propagate in time)
        inthres       = mthres[v]
        if v == 0: # Landmask
            maskpts       = ((invar <= inthres).prod(0)) # [Lat x Lon]
            emask[maskpts==1] = 1 # 1 means it is ocean point
            if mask_sep:
                lmask[maskpts==1] = 1
        elif v == 1:
            maskpts       = ((invar <= inthres).prod(0)) # [Lat x Lon]
            emask[maskpts==0] = np.nan # 0 means it has sea ice
            if mask_sep:
                imask[maskpts==0] = np.nan # All points 1, ice points NaN
    # Save masks separately
    if mask_sep:
        imasks.append(imask)
        lmasks.append(lmask)
    mask.append(emask.copy())
# Make into array
mask = np.array(mask)  # [ENS x LAT x LON]
