        #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 21:10:05 2020

@author: gliu
"""

import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
import xarray as xr
import xesmf as xe

detrend = True # Set to True to detrend data by removing ensemble average

# Load the data into dataarrays
sst_ds = xr.open_dataset('../../CESM_data/CESM1LE_sst_NAtl_19200101_20051201.nc',chunks={'ensemble':1})['sst'][0:69,8:-16,:,:40].astype(np.float32)
sss_ds = xr.open_dataset('../../CESM_data/CESM1LE_sss_NAtl_19200101_20051201.nc',chunks={'ensemble':1})['sss'][0:69,8:-32,:,:40].astype(np.float32)
psl_ds = xr.open_dataset('../../CESM_data/CESM1LE_psl_NAtl_19200101_20051201.nc',chunks={'ensemble':1})['psl'][0:69,8:-32,:,:40].astype(np.float32)

# Deseason the data
sst_deseason = (sst_ds.groupby('time.month') - sst_ds.groupby('time.month').mean('time')).groupby('time.year').mean('time')
sss_deseason = (sss_ds.groupby('time.month') - sss_ds.groupby('time.month').mean('time')).groupby('time.year').mean('time')
psl_deseason = (psl_ds.groupby('time.month') - psl_ds.groupby('time.month').mean('time')).groupby('time.year').mean('time')
print("Deseasoned Data")

# Apply land mask to PSL
landmask = ~np.isnan( sst_ds[:,:,0,0].values )
psl_deseason *= landmask[:,:,None,None]
# for ilat in range(len(psl_ds.lat)):
#     for ilon in range(len(psl_ds.lon)):
#         if landmask[ilat,ilon] == True:
#             psl_deseason[ilat,ilon,:,:] = np.nan
print("Applied Land Mask")

# Detrend the data if option is set
if detrend:
    sst_deseason = sst_deseason - sst_deseason.mean('ensemble')
    sss_deseason = sss_deseason - sss_deseason.mean('ensemble')
    psl_deseason = psl_deseason - psl_deseason.mean('ensemble')

# Normalize the data
sst_normalized = (sst_deseason - sst_deseason.mean())/sst_deseason.std()
sss_normalized = (sss_deseason - sss_deseason.mean())/sss_deseason.std()
psl_normalized = (psl_deseason - psl_deseason.mean())/psl_deseason.std()
print("Normalized Data")

# Retrieve lat/lon and downscale
lat = sst_ds.lat
lon = sst_ds.lon
#lat_out = np.linspace(lat[0],lat[-1],244)
#lon_out = np.linspace(lon[0],lon[-1],244)
lat_out = np.linspace(lat[0],lat[-1],224)
lon_out = np.linspace(lon[0],lon[-1],224)
ds_out = xr.Dataset({'lat': (['lat'], lat_out), 'lon': (['lon'], lon_out) })

# Set up regridder
regridder = xe.Regridder(sst_ds, ds_out, 'nearest_s2d')
#regridder

# Apply Regridder
sst_out = regridder( sst_normalized.transpose('ensemble','year','lat','lon').astype(np.float32) )
sss_out = regridder( sss_normalized.transpose('ensemble','year','lat','lon').astype(np.float32) )
psl_out = regridder( psl_normalized.transpose('ensemble','year','lat','lon').astype(np.float32) )
print("Regridding Data")

# Calculate AMV Index
amv_index = (np.cos(np.pi*sst_out.lat/180) * sst_out).mean(dim=('lat','lon'))
print("Calculated AMV Index")

# Read out values
sst_out_values = sst_out.astype(np.float32).values
sss_out_values = sss_out.astype(np.float32).values
psl_out_values = psl_out.astype(np.float32).values
print("Read out data")

# Set NaN values to zero
sst_out_values[np.isnan(sst_out_values)] = 0
sss_out_values[np.isnan(sss_out_values)] = 0
psl_out_values[np.isnan(psl_out_values)] = 0
print("Set NaNs to zeros")

# Save values
data_out = np.array([sst_out_values[:,:,:,:],sss_out_values[:,:,:,:],psl_out_values[:,:,:,:]])
np.save('CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i.npy' % detrend,data_out)
np.save('CESM_label_amv_index_detrend%i.npy' % detrend,amv_index[:,:])
print("Saved Data")
