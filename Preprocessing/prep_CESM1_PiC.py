#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regrid, detrend, etc for CESM1 PiC data


For predictors other than SST:
    - Works with data regridded using cdo, "regridding_cdo_cesm1.sh".
    - Copies section from "regrid_reanalysis_cesm1.py"
    
    
For SST (uses data preprocessed by the stochmod data)


Created on Mon Apr 24 13:49:31 2023

@author: gliu
"""

import xarray as xr
import glob
import sys
import time
import numpy as np
import scipy

#%% User edits and shared variables

# Import necessary modules
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import proc,viz

# Import stochastic model processing packages (PiC)
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
import scm

# Indicate variable
varname = "SST" # SSH

# Indicate paths
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/CESM1-PIC"

# Indicate crop selection
bbox          = [-90,20,0,90] # Crop Selection
bbox_amv      = [-80,0,0,65]
region_name   = "NAT"
regrid_data   = None
detrend       = False
debug         = True

# Other naming things
dataset_name = "CESM1-PIC"
ystart       = "0400"
yend         = "2200"

# Define Preprocessing Function
def preprocess(ds):
    ds        = proc.lon360to180_xr(ds)
    ds        = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
    return ds


def preprocess_SST(sst,lat,lon,bbox):
    
    ntime,nlat,nlon = sst.shape
    
    # Remove mean climatological cycle
    st = time.time()
    climavg,sst_yrmon=proc.calc_clim(sst,0,returnts=True)
    sst_anom = sst_yrmon - climavg[None,:,:,:]
    sst_anom = sst_anom.reshape(sst.shape)
    print("Deseasoned in %.2fs" % (time.time()-st))
    
    # Flip dimensions
    sst_flipped = proc.flipdims(sst_anom) # {Lon x Lat x Time}
    
    # Flip Lat/lon
    if np.any(lon > 180):
        st = time.time()
        print("Flipping Longitude!")
        lon,sst_anom=proc.lon360to180(lon,sst_flipped)
        print("Flipped lon in %.2fs" % (time.time()-st))
    
    # Crop to region
    print("Cropping Region!")
    sst_region,lonr,latr = proc.sel_region(sst_flipped,lon,lat,bbox)
    print("Cropped in %.2fs." % (time.time()-st))
    
    # Upflip dimensions
    sst_region = proc.flipdims(sst_region) # {Time x Lat x Lon}
    return sst_region,lonr,latr

# -----------------------------------------------------------------------------
#%% Part 1: Process all other variables (except for SST)
# -----------------------------------------------------------------------------

# Indicate paths for cropping/selecting


# Indicate regridding method based on the input variable
if varname in ["SSH","SSS","BSF"]:
    method = "bilinear"
else:
    method = None
    
    
#%% Find and load the data

if varname == "SST":
    
    datpath_pic = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
    
    # Load CESM1 lat lon
    lon,lat=scm.load_latlon(lon360=True)

    # Load the CESM PiC Data
    #st = time.time()
    ssts_pic        = scm.load_cesm_pt(datpath_pic,loadname='full',grabpoint=None,ensorem=0)
    sst_pic         = ssts_pic[0] # Time x lat x lon
    #print("Loaded PiC Data in %.2fs" % (time.time()-st))

    # Preprocess and crop CESM PiC (deseason, flip dimension, crop to NAtl)
    sst_region,lonr,latr = preprocess_SST(sst_pic,lat,lon,bbox)
    sst_anom_pic    = sst_region
    
    # Take annual average (can remove later if I can find my monthly anomalized Htr Data)
    sst_anom_pic    = proc.ann_avg(sst_anom_pic,0)
    ntime,nlat,nlon = sst_anom_pic.shape
    
    # Replace back into a dataframe
    times           = np.arange(0,ntime)
    ds_all_anom     = proc.numpy_to_da(sst_anom_pic,times,latr,lonr,varname,)
    
    data_array_flag = True # Set to true so that .to_array() can be avoided later...
    
    
    
elif varname == "SSH":
    
    datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LE/proc/SSH/"
    
    globstr = "%s*.nc" % (datpath)
    nclist  = glob.glob(globstr)
    nclist.sort()
    ds_all = xr.open_mfdataset(globstr,preprocess=preprocess,combine="nested",
                               concat_dim="time").load()
    
    # Load all variables into a dataset with time x lat x lon, flipped to lonW and
    # cropped to bbox

    #% Now perform preprocessing, as was done in "regrid_reanalysis_cesm1.py"
    
    # <6> Compute Monthly Anomalies and Annual Average
    st = time.time() #387 sec
    ds_all_anom = (ds_all.groupby('time.month') - ds_all.groupby('time.month').mean('time')).groupby('time.year').mean('time')
    print("Deseasoned in %.2fs!" % (time.time()-st))
    if debug:
        ds_all_anom.isel(year=0)[varname].plot(vmin=-2,vmax=2,cmap="RdBu_r")
        
        
    data_array_flag = False

#%% Now perform additional preprocessing steps

# <7> Detrending (Skip for now)

# <8> Normalize and Standardize Data
mu            = ds_all_anom.mean()
sigma         = ds_all_anom.std()
ds_normalized = (ds_all_anom - mu)/sigma
if data_array_flag:
    np.save('%s%s_nfactors_%s_detrend%i_regridCESM.npy' % (outpath,dataset_name,varname,detrend),(mu.values,sigma.values))
else:
    np.save('%s%s_nfactors_%s_detrend%i_regridCESM.npy' % (outpath,dataset_name,varname,detrend),(mu.to_array().values,sigma.to_array().values))

# <9> Save the Output
st = time.time() #387 sec
if varname != varname.upper():
    print("Capitalizing variable name.")
    ds_normalized_out = ds_normalized.rename({varname:varname.upper()})
    varname = varname.upper()
else:
    ds_normalized_out = ds_normalized
encoding_dict = {varname : {'zlib': True}} 
outname       = "%s%s_%s_NAtl_%s_%s_%s_detrend%i_regridCESM1.nc" % (outpath,dataset_name,varname,
                                                                         ystart.replace("-",""),yend.replace("-","")
                                                                         ,method,detrend)
ds_normalized_out.to_netcdf(outname,encoding=encoding_dict)
print("Saved data in %s" % outname)

#%% Compute Monthly Anomalies and Annual Averag

