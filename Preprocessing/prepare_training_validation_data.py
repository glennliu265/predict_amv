"""
prepare training validation data

Regrids sea surface temperature (SST), sea surface salinity (SSS),
and sea level pressure (PSL) to 224 x 224 (lat x lon) for input into
ImageNet style models. Also calculates AMV Index, the prediction
objective

Assumes data is located in ../../CESM_data/

Outputs data to the same directory.
    - Predictors: CESM_data_sst_sss_psl_deseason_normalized_resized_detrend0.npy
    - Labels: CESM_label_amv_index_detrend0.npy
    - Normalization Factors : CESM_nfactors_detrend0.npy
"""

import numpy as np
import xarray as xr
import xesmf as xe

detrend = True # Detrending is currently not applied
regrid  = None # Set to desired resolution. Set None for no regridding.


# --------------------------------
# Select Box in the North Atlantic
# --------------------------------
sst_ds = xr.open_dataset('../../CESM_data/CESM1LE_sst_NAtl_19200101_20051201.nc')['sst'][0:69,8:-16,:,:]
sss_ds = xr.open_dataset('../../CESM_data/CESM1LE_sss_NAtl_19200101_20051201.nc')['sss'][0:69,8:-32,:,:]
psl_ds = xr.open_dataset('../../CESM_data/CESM1LE_psl_NAtl_19200101_20051201.nc')['psl'][0:69,8:-32,:,:]


# ----------------------------------------------------------------
# Calculate Monhtly Anomalies (Remove mean seasonal cycle)
# ----------------------------------------------------------------
print('begin deseason')
sst_deseason = (sst_ds.groupby('time.month') - sst_ds.groupby('time.month').mean('time')).groupby('time.year').mean('time')
print('finished SST deseason')
sss_deseason = (sss_ds.groupby('time.month') - sss_ds.groupby('time.month').mean('time')).groupby('time.year').mean('time')
print('finished SSS deseason')
psl_deseason = (psl_ds.groupby('time.month') - psl_ds.groupby('time.month').mean('time')).groupby('time.year').mean('time')
print('finished PSL deseason')

# --------------------------------
# Apply Land/Ice Mask to PSL
# --------------------------------
landmask = ~np.isnan( sst_ds[:,:,0,0].values )
psl_deseason *= landmask[:,:,None,None]

# --------------------------------
# Detrend the data if option is set
# --------------------------------
if detrend:
    print("Detrending data!")
    sst_deseason = sst_deseason - sst_deseason.mean('ensemble')
    sss_deseason = sss_deseason - sss_deseason.mean('ensemble')
    psl_deseason = psl_deseason - psl_deseason.mean('ensemble')

# -------------------------
# Normalize and standardize
# -------------------------
sst_normalized = (sst_deseason - sst_deseason.mean())/sst_deseason.std()
sss_normalized = (sss_deseason - sss_deseason.mean())/sss_deseason.std()
psl_normalized = (psl_deseason - psl_deseason.mean())/psl_deseason.std()

means  = (sst_deseason.mean(),sss_deseason.mean(),psl_deseason.mean())
stdevs = (sst_deseason.std(),sss_deseason.std(),psl_deseason.std())
np.save('CESM_nfactors_detrend%i_regrid%s.npy' % (detrend,regrid),(means,stdevs))


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
    sss_out = regridder( sss_normalized.transpose('ensemble','year','lat','lon') )
    psl_out = regridder( psl_normalized.transpose('ensemble','year','lat','lon') )
else:
    print("Data will not be regridded")
    sst_out = sst_normalized.transpose('ensemble','year','lat','lon') 
    sss_out = sss_normalized.transpose('ensemble','year','lat','lon')
    psl_out = psl_normalized.transpose('ensemble','year','lat','lon')

# -----------------------------------------------
# Calculate the AMV Index (Area weighted average)
# -----------------------------------------------
amv_index = (np.cos(np.pi*sst_out.lat/180) * sst_out).mean(dim=('lat','lon'))

# ----------------------------------
# Load data to numpy arrays and save
# ----------------------------------
sst_out_values = sst_out.values
sss_out_values = sss_out.values
psl_out_values = psl_out.values

sst_out_values[np.isnan(sst_out_values)] = 0
sss_out_values[np.isnan(sss_out_values)] = 0
psl_out_values[np.isnan(psl_out_values)] = 0

data_out = np.array([sst_out_values,sss_out_values,psl_out_values])
np.save('CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i_regrid%s.npy' % (detrend,regrid),data_out)
np.save('CESM_label_amv_index_detrend%i_regrid%s.npy' % (detrend,regrid),amv_index[:,:])
