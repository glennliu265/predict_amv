#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Copied from prepare_training_validation.py
Essentially only does the SST component and averages over particular regions

Created on Fri Dec  2 11:48:40 2022

@author: gliu
"""


import numpy as np
import xarray as xr
import xesmf as xe

detrend = False # Detrending is currently not applied
regrid  = None # Set to desired resolution. Set None for no regridding.

# -----------------------------------------------------------------------------
# Copied from sm_paper_stylesheet.py on 2022.12.02
# removed NNAT
bbox_SP     = [-60,-15,40,65]
bbox_ST     = [-80,-10,20,40]
bbox_TR     = [-75,-15,10,20]
bbox_NA     = [-80,0 ,0,65]
bbox_NA_new = [-80,0,10,65]
bbox_ST_w   = [-80,-40,20,40]
bbox_ST_e   = [-40,-10,20,40]
regions     = ("SPG","STG","TRO","NAT","STGe","STGw")        # Region Names
bboxes      = (bbox_SP,bbox_ST,bbox_TR,bbox_NA,bbox_ST_e,bbox_ST_w) # Bounding Boxes
regionlong  = ("Subpolar","Subtropical","Tropical","North Atlantic","Subtropical (East)","Subtropical (West)",)

# -----------------------------------------------------------------------------

# --------------------------------
# Select Box in the North Atlantic
# --------------------------------
sst_ds = xr.open_dataset('../../CESM_data/CESM1LE_sst_NAtl_19200101_20051201.nc')['sst'][:,:,:,:]

# ----------------------------------------------------------------
# Calculate Monhtly Anomalies (Remove mean seasonal cycle)
# ----------------------------------------------------------------
print('begin deseason')
sst_deseason = (sst_ds.groupby('time.month') - sst_ds.groupby('time.month').mean('time')).groupby('time.year').mean('time')


# --------------------------------
# Detrend the data if option is set
# --------------------------------
if detrend:
    sst_deseason = sst_deseason - sst_deseason.mean('ensemble')

# -------------------------
# Normalize and standardize
# -------------------------
sst_normalized = (sst_deseason - sst_deseason.mean())/sst_deseason.std()
means  = (sst_deseason.mean(),)
stdevs = (sst_deseason.std(),)

# -------------------------
# Regrid Variables
# -------------------------
if regrid is not None:
    print("Data will be regridded to %i degree resolution"%regrid)
    # Prepare Latitude/Longitude
    lat = sst_ds.lat
    lon = sst_ds.lon
    lat_out = np.linspace(lat[0],lat[-1],regrid)
    lon_out = np.linspace(lon[0],lon[-1],regrid)
    
    # Make Regridder
    ds_out    = xr.Dataset({'lat': (['lat'], lat_out), 'lon': (['lon'], lon_out) })
    regridder = xe.Regridder(sst_ds, ds_out, 'nearest_s2d')

    # Regrid
    sst_out = regridder( sst_normalized.transpose('ensemble','year','lat','lon') )
else:
    print("Data will not be regridded")
    sst_out = sst_normalized.transpose('ensemble','year','lat','lon') 

# -----------------------------------------------
# Calculate the AMV Index (Area weighted average)
# -----------------------------------------------
for b,bb in enumerate(bboxes):
    sst_out_reg = sst_out.sel(lon=slice(bb[0],bb[1]),lat=slice(bb[2],bb[3]))
    amv_index   = (np.cos(np.pi*sst_out_reg.lat/180) * sst_out_reg).mean(dim=('lat','lon'))
    savename = 'CESM_label_%s_amv_index_detrend%i_regrid%s.npy' % (regions[b],detrend,regrid)
    np.save(savename,amv_index[:,:])
    print("Saved Region %s ([%s]) as %s" % (regionlong[b],bb,savename))





