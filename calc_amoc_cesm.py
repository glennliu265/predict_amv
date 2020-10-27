#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate AMOC Index from CESM1.1 Large Ensemble

General Procedure:
    1 - Load Data
    2 - Cut to region 
    3 - sum selected components
    3 - calculate index by taking depth and latitude maximum
    
    

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

# NAO Calculation settings ... 
# Indicate start and ending year
start = "1920-01-01"
end   = "2005-12-01"

# Indicate the lat bounds for AMOC Index
latS = 26
latN = 27

# Detrending Options
# For deg = 0, Detrend by removing the ensemble average 
# For deg > 0, Specify degree of polynomial for detrending 
deg = 0
        
# Debug mode, makes some plots to check detrending, etc
debug = 1

# MOC Components to include in sum
# 0 - Eulerian Mean
# 1 - Eddy-Induced (Bolus)
# 2 - Submeso
comps = [0,1,2]

## User Edits <END> ----
#%% 1) Load Data and calculate monthly anomalies ----

# Import module with functions
sys.path.append(modpath)
import amvmod as amv

mocname ="CESM1LE_MOC_NAtl_20N-50N_19200101_20051201.nc"
ds = xr.open_dataset(datpath+mocname)
ds = ds.sel(time=slice(start,end))

#%% 2) Select region for calculation
ds = ds.sel(lat_aux_grid=slice(latS,latN))
if debug == 1:
    ds.MOC.isel(ensemble=0,time=0,moc_comp=0).plot()

# Load data from DataArray to numpy array
moc = ds['MOC'].values #[ens x time x moc_component x depth x lat]
lat = ds['lat_aux_grid'].values #[96]  
dz   = ds['moc_z']
mon = ds['time'].values#[1032]
nens,nmon,comp,nz,nlat =moc.shape

# 3 -- Sum the components
moc = moc[:,:,comps,:,:].sum(2) # [ens x time x depth x lat]
if len(comps)>1:
    compname = "all"
else:
    compname = str(comps[:])
    

# # Calculate monthly anomalies
# moc = moc.tranpose(1,0,2,3) # [time x ens x depth x lat]
# mocm = moc.reshape(int(np.ceil(nmon/12)),12,nens*nz*nlat) # Separate mon/year, combine lat/depth/ens
# moca = mocm - mocm.mean(0)[None,:,:] # Calculate monthly anomalies
# moca = moca.reshape(nmon,nens*nz*nlat) # Recombine mon x year

# 4 -- Take the maximum from the selected region
amocidx = moc.reshape(nens,nmon,nz*nlat).max(2)


#%% 5 - Save Output
np.save("%sCESM1LE_AMOCIndex_%s-%s_region%ito%i_component%s.npy" % (outpath,start[0:4],end[0:4],latS,latN,compname),amocidx)
