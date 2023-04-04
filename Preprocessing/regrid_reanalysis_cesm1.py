#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Regrid the reanalysis datasets to CESM1 resolution using xesmf.

This follows a similar procedure to prep_data_byvariable, as described below:
 
    "
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
    "

Note: Regridding section requires xesmf_env.
Script currently written to run on stormtrack. Need to add astraeus paths.

Created on Mon Apr  3 14:13:23 2023

@author: gliu
"""

import numpy as np
import xarray as xr

import glob
import time
from tqdm import tqdm
import sys

import matplotlib.pyplot as plt

#%% import some packages and preloaded parameters (current)
machine = "Astraeus"
# Note: need to install xesmf for Astraeaus
if machine == "stormtrack":
    import xesmf as xe


    
    sys.path.append("../")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    
    from amv import proc
    #import predict_amv_params as pparams
    
    #reanalysis_dict = pparams.reanalysis_dict
#elif machine == "Astraeus":
    

stall         = time.time()


#%% User Edits

# File Information
datpath       = "/stormtrack/data4/glliu/01_Data/Reanalysis/"
dataset_name  = "HadISST"
ncname        = "HadISST_sst_18700115_20230115.nc"

latlon_name   = "cesm_latlon360.npz"
latlon_path   = '/stormtrack/home/glliu/01_Data/'
varname       = "sst"
outpath       = "../../CESM_data/Reanalysis/regridded/"

# (Move to predict_amv_params after torch  module is removed...)
had_dict = {
    'dataset_name' : 'HadISST',
    'sst'          : 'sst',
    'lat'          : 'latitude',
    'lon'          : 'longitude',
    'ystart'       : 1870,
    'drop_dims'    : ["nv",],
    }

indicts = (had_dict,)
reanalysis_names = [d['dataset_name'] for d in indicts]
reanalysis_dict  = dict(zip(reanalysis_names,indicts))


# Load CESM Lat/Lon


# Preprocessing Information
detrend       = False
bbox          = [-90,20,0,90] # Crop Selection
ystart        = '1870-01-01'
yend          = '2022-12-31'
method        = "bilinear" # regridding method for POP ocean data

# -----------------------------
#%% Load in the data and regrid
# -----------------------------
# Load Lat/lon
ll_ld             = np.load(latlon_path+latlon_name,allow_pickle=True)
cesm_lon,cesm_lat = ll_ld['lon'],ll_ld['lat']
dummyvar          = np.zeros([len(cesm_lon),len(cesm_lat),1])
cesm_lon180,_     = proc.lon360to180(cesm_lon,dummyvar,)

# Get data
ds = xr.open_dataset(datpath+ncname).load()

# Crop to time
ds_slice = ds.sel(time=slice(ystart,yend))

# Rename and drop dimensions
dim_names_original = reanalysis_dict[dataset_name]
rename_dict = {dim_names_original['lon'] : 'lon',
               dim_names_original['lat'] : 'lat',
               }
ds_slice = ds_slice.rename_dims(rename_dict)

dropdims = dim_names_original['drop_dims']
ds_slice = ds_slice.drop_dims(dropdims)

# Regrid
ds_out    = xr.Dataset({'lat': (['lat'], cesm_lat), 'lon': (['lon'], cesm_lon180) })
regridder = xe.Regridder(ds_slice, ds_out, 'nearest_s2d')

# Regrid
ds_regridded = regridder( ds_slice.transpose('time','lat','lon') )

# Save CESM Version (NOTE getting weird output permission error on stormtrack....)
encoding_dict = {varname : {'zlib': True}} 
outname       = "%s%s_%s_regridCESM1_%s.nc" % (outpath,dataset_name,varname,method)
ds_regridded.to_netcdf(outname,encoding=encoding_dict)

# Debugging plots
#ds_regridded.isel(time=0).sst.plot(vmin=0,vmax=34),plt.show()

# ---------------------
#%% Preprocess and Crop
# ---------------------
debug = True

# Load data if necessary
savename     = "%s%s_%s_regridCESM1_%s.nc" % (outpath,dataset_name,varname,method)
ds_regridded = xr.open_dataset(savename)

# Crop data to region
ds_reg = ds_regridded.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
if debug:
    ds_reg.isel(time=0).sst.plot(vmin=0,vmax=32)
# Make into [ensemble x time x lat x lon]

# Calculate monthly anomalies and annual average (taken from prep-data_byvariable.py)
# --------------------------------
# Deseason and take annual average
# --------------------------------
st = time.time() #387 sec
ds_all_anom = (ds_reg.groupby('time.month') - ds_reg.groupby('time.month').mean('time')).groupby('time.year').mean('time')
print("Deseasoned in %.2fs!" % (time.time()-st))
if debug:
    ds_all_anom.isel(year=0).sst.plot(vmin=-2,vmax=2,cmap="RdBu_r")
    
    
# -------
# Detrend
# -------
#%% NOTE: Need to rewrite this section...
if detrend:
    print("WARNING, Detrending section still needs to be written.")
    break
    sst = ds_all_anom.sst.values #[time x lat x lon]

    #ds_all_anom = ds_all_anom - ds_all_anom.mean('ensemble')
        
        
#%% Normalize data
# -------------------------
# Normalize and standardize
# -------------------------
mu            = ds_all_anom.mean()
sigma         = ds_all_anom.std()
ds_normalized = (ds_all_anom - mu)/sigma
np.save('%s%s_nfactors_%s_detrend%i_regridCESM.npy' % (outpath,dataset_name,varname,detrend),(mu.to_array().values,sigma.to_array().values))

# ---------------
# Save the output
# ---------------
st = time.time() #387 sec
if varname != varname.upper():
    print("Capitalizing variable name.")
    ds_normalized_out = ds_normalized.rename({varname:varname.upper()})
    varname = varname.upper()
encoding_dict = {varname : {'zlib': True}} 
outname       = "%s%s_%s_NAtl_%s_%s_%s_detrend%i_regridCESM.nc" % (outpath,dataset_name,varname,
                                                                         ystart.replace("-",""),yend.replace("-","")
                                                                         ,method,detrend)
ds_normalized_out.to_netcdf(outname,encoding=encoding_dict)


