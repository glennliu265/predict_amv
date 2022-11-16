#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Prepare data by variable (for ML Prediction)

- Crops the data to a given region/time period
- Merges by ensemble and computes ensemble average

For the given large ensemble dataset/variable, perform the following:
    1. Regrid (if necessary) to the specified resolution
    
    <Note, I have outsourced the steps above to prep_mld_PIC.py>
    
    <THIS is what the script actually does...>
    2. Crop the Data in Time (1920 - 2005)
    3. Crop to Region (*****NOTE: assumes degrees West = neg!)
    4. Concatenate each ensemble member
    5. Output in array ['ensemble','year','lat','lon']

For Single variable case...
    Based on procedure in prepare_training_validation_data.py
    ** does not apply land mask!
    
    6 . Calculate Monthly Anomalies + Annual Average
    7 . Remove trend (if specified)
    8 . Normalize data
    9 . Perform regridding (if option is set)
    10. Output in array ['ensemble','year','lat','lon']
    
Copies sections from:
    - prep_mld_PIC.py (stochmod repo)
    

Created on Thu Oct 20 16:35:33 2022

@author: gliu

"""

import numpy as np
import xarray as xr
import xesmf as xe
import glob
import time
from tqdm import tqdm

stall         = time.time()

varname       = "HMXL"

detrend       = False # Detrending is currently not applied
regrid        = None  # Set to desired resolution. Set None for no regridding.

bbox          = [-90,20,0,90] # Crop Selection
ystart        = 1920
yend          = 2005

mconfig       = "FULL_HTR"
use_xesmf     = True # Use xESMF for regridding. False = box average
method        = "bilinear" # regridding method for POP ocean data

datpath       = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/"
outpath       = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/"

debug         = True # Set to True for debugging flat

#%% Necessary Functions
def fix_febstart(ds):
    # Copied from preproc_CESM.py on 2022.11.15
    if ds.time.values[0].month != 1:
        print("Warning, first month is %s"% ds.time.values[0])
        # Get starting year, must be "YYYY"
        startyr = str(ds.time.values[0].year)
        while len(startyr) < 4:
            startyr = '0' + startyr
        nmon = ds.time.shape[0] # Get number of months
        # Corrected Time
        correctedtime = xr.cftime_range(start=startyr,periods=nmon,freq="MS",calendar="noleap")
        ds = ds.assign_coords(time=correctedtime) 
    return ds

def lon360to180_xr(ds,lonname='lon'):
    # Based on https://stackoverflow.com/questions/53345442/about-changing-longitude-array-from-0-360-to-180-to-180-with-python-xarray
    # Copied from amv.proc on 2022.11.15
    ds.coords[lonname] = (ds.coords[lonname] + 180) % 360 - 180
    ds = ds.sortby(ds[lonname])
    return ds

def xrdeseason(ds):
    """ Remove seasonal cycle, given an Dataarray with dimension 'time'"""
    # Copied from amv.proc on 2022.11.15
    return ds.groupby('time.month') - ds.groupby('time.month').mean('time')
#%% Get the data

# Ex: SSH_FULL_HTR_bilinear_num01.nc
ncsearch = "%s%s/%s_%s_%s*.nc" % (datpath,varname,varname,mconfig,method)
nclist   = glob.glob(ncsearch)
nclist.sort()
nens     = len(nclist)
print("Found %i files!" % (nens))
if debug:
    print(*nclist,sep="\n")

#%%
"""
-----------------------
The Preprocessing Steps
-----------------------
    2. Crop the Data in Time (1920 - 2005)
    3. Crop to Region (*****NOTE: assumes degrees West = neg!)
    4. Calculate Monthly Anomalies
    5. Compute average (annual or seasonal)
    6. Concatenate each ensemble member
    6. Remove Trend (if specified)
    7. Output in array ['ensemble','year','lat','lon']
    
    
    
    2. Crop the Data in Time (1920 - 2005)
    3. Crop to Region (*****NOTE: assumes degrees West = neg!)
    4. Concatenate each ensemble member
    5. Output in array ['ensemble','year','lat','lon'] 

# The following is done at another step...
    4. Calculate Monthly Anomalies
    5. Compute average (annual or seasonal)
    6. Remove Trend (if specified)
    7. Output in array ['ensemble','year','lat','lon']
    
"""

for e in tqdm(range(nens)):
    # Get the data
    ds = xr.open_dataset(nclist[e])
    
    # Correct time if needed, then crop to range
    ds = fix_febstart(ds)
    ds = ds.sel(time=slice("%s-01-01"%(ystart),"%s-12-31"%(yend)))
    
    # Crop to region
    if np.any(ds.lon.values > 180):
        ds = lon360to180_xr(ds)
        print("Flipping Longitude (includes degrees west)")
    dsreg = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
    
    # Compute monthly anomalies (takes 14m with this inside the loop)
    #dsreg = xrdeseason(dsreg)
    
    # Concatenate to ensemble
    if e == 0:
        ds_all = dsreg.copy()
    else:
        ds_all = xr.concat([ds_all,dsreg],dim="ensemble")
    
    
# # Compute monthly anomalies
# st = time.time()
# ds_all_anom = xrdeseason(ds_all)
# print("Deseasoned in %.2fs!" % (time.time()-st))

# Compute and remove trend
#if detrend:
    
# Set encoding dictionary
encoding_dict = {varname : {'zlib': True}} 

# Compute and save the ensemble average
#ensavg  = ds_all_anom.mean('ensemble')
ensavg = ds_all.mean('ensemble')
outname = "%s%s_%s_%s_ensavg_%sto%s.nc" % (outpath,varname,mconfig,method,ystart,yend)
ensavg.to_netcdf(outname,encoding=encoding_dict)
    
    
# Save the data (detrend, calculate anomalies, later)
    
    # Remove the ensemble average, save the output
    #ds_all_anom = ds_all_anom - ensavg

ds_all  = ds_all.transpose('ensemble','time','lat','lon')
outname = "%sCESM1LE_%s_NAtl_%s0101_%s0101_%s.nc" % (outpath,varname,ystart,yend,method)
ds_all.to_netcdf(outname,encoding=encoding_dict)
#outname = "%s%s_%s_%s_%sto%s.nc" % (outpath,varname,mconfig,method,ystart,yend)
    
    
print("Merged data in %.2fs" % (time.time()-stall))
    
# --------
#%% Part 2
# --------
"""
Based on procedure in prepare_training_validation_data.py
** does not apply land mask!

    4. Calculate Monthly Anomalies + Annual Average
    5. Remove trend (if specified)
    6. Normalize data
    7. Perform regridding (if option is set)
    8. Output in array ['ensemble','year','lat','lon']
"""

for varname in ("SSH","sst","sss","psl"):
    # -------------------
    # Load in the dataset
    # -------------------
    if varname.lower() in ("sst","sss","psl"):
        varname = varname.lower()
    outname = "%sCESM1LE_%s_NAtl_%s0101_%s0101_%s.nc" % (outpath,varname,ystart,yend,method)
    ds_all  = xr.open_dataset(outname)
    
    # --------------------------------
    # Deseason and take annual average
    # --------------------------------
    st = time.time() #387 sec
    ds_all_anom = (ds_all.groupby('time.month') - ds_all.groupby('time.month').mean('time')).groupby('time.year').mean('time')
    print("Deseasoned in %.2fs!" % (time.time()-st))
    
    # -------
    # Detrend
    # -------
    if detrend:
        ds_all_anom = ds_all_anom - ds_all_anom.mean('ensemble')
    
    # -------------------------
    # Normalize and standardize
    # -------------------------
    mu            = ds_all_anom.mean()
    sigma         = ds_all_anom.std()
    ds_normalized = (ds_all_anom - mu)/sigma
    np.save('%sCESM1LE_nfactors_%s_detrend%i_regrid%s.npy' % (outpath,varname,detrend,regrid),(mu,sigma))
    
    # ------------------------
    # Regrid, if option is set <Add this section later...>
    # ------------------------
    if regrid is not None:
        print("Data will be regridded to %i degree resolution." % regrid)
        # Prepare Latitude/Longitude
        lat = ds_normalized.lat
        lon = ds_normalized.lon
        lat_out = np.linspace(lat[0],lat[-1],regrid)
        lon_out = np.linspace(lon[0],lon[-1],regrid)
        
        # Make Regridder
        ds_out    = xr.Dataset({'lat': (['lat'], lat_out), 'lon': (['lon'], lon_out) })
        regridder = xe.Regridder(ds_normalized, ds_out, 'nearest_s2d')
    
        # Regrid
        ds_normalized_out = regridder( ds_normalized.transpose('ensemble','year','lat','lon') )
    else:
        print("Data will not be regridded.")
        ds_normalized_out = ds_normalized.transpose('ensemble','year','lat','lon')
        
        
    # ---------------
    # Save the output
    # ---------------
    st = time.time() #387 sec
    if varname != varname.upper():
        print("Capitalizing variable name.")
        ds_normalized_out = ds_normalized_out.rename({varname:varname.upper()})
        varname = varname.upper()
    encoding_dict = {varname : {'zlib': True}} 
    outname       = "%sCESM1LE_%s_NAtl_%s0101_%s0101_%s_detrend%i_regrid%s.nc" % (outpath,varname,ystart,yend,method,detrend,regrid)
    ds_normalized_out.to_netcdf(outname,encoding=encoding_dict)
    print("Saved output tp %s in %.2fs!" % (outname,time.time()-st))

#%%


# #%% Get list of netCDFs

# # Set up variables to keep for preprocessing script
# varkeep   = [varname,"TLONG","TLAT","time"]

# # Adjust data path on stormtrack
# if varname == "SSS":
#     datpath   = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/processed/ocn/proc/tseries/monthly/SSS/"
# else:
#     datpath   = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/ocn/proc/tseries/monthly/%s/" % varname

# # Set ncsearch string on stormtrack based on input dataset
# catdim  = 'time'
# savesep = False # Save all files together
# if mconfig == "FULL_PIC":
#     ncsearch      = "b.e11.B1850C5CN.f09_g16.*.pop.h.%s.*.nc" % varname
# elif mconfig == "SLAB_PIC":
#     ncsearch      = "e.e11.B1850C5CN.f09_g16.*.pop.h.%s.*.nc" % varname
# elif mconfig == "FULL_HTR":
#     ncsearch      = "b.e11.B20TRC5CNBDRD.f09_g16.*.pop.h.%s.*.nc" % varname
#     catdim  = 'ensemble'
#     savesep = True # Save each ensemble member separately
#     use_mfdataset = False

# # Open the files
# if "HTR" in mconfig: # Concatenate by ensemble

#     #% Get the filenames
#     # EX: SSS_FULL_HTR_bilinear_num00.nc
#     globstr       = "%s%s_%s_%s_num*.nc" % (datpath,varname,mconfig,method)
#     nclist        = glob.glob(globstr)
#     nclist.sort()
#     print("Found %i files!" % (len(nclist)))
    
#     # Ensemble x Time x z x Lat x Lon
#     ds_all = xr.open_mfdataset(nclist,concat_dim='ensemble',combine="nested",parallel=True)
    
# else: # Just open it
    
#     # Just open 1 file
#     nc = "%s%s_%s_%s.nc" % (datpath,varname,mconfig,method)
#     print("Opening %s" % (nc))

#     # Time x z x Lat x Lon
#     ds_all = xr.open_dataset(nc)
    
    

#%% Steps 1-6

# Get list of netCDFs



#%%
