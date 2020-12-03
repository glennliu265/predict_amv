import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import xesmf as xe

sst_ds = xr.open_dataset('../../CESM_data/CESM1LE_sst_NAtl_19200101_20051201.nc')['sst'][0:69,8:-16,:,:]
sss_ds = xr.open_dataset('../../CESM_data/CESM1LE_sss_NAtl_19200101_20051201.nc')['sss'][0:69,8:-32,:,:]
psl_ds = xr.open_dataset('../../CESM_data/CESM1LE_psl_NAtl_19200101_20051201.nc')['psl'][0:69,8:-32,:,:]

print('begin deseason')
sst_deseason = (sst_ds.groupby('time.month') - sst_ds.groupby('time.month').mean('time')).groupby('time.year').mean('time')
print('finished SST deseason')
sss_deseason = (sss_ds.groupby('time.month') - sss_ds.groupby('time.month').mean('time')).groupby('time.year').mean('time')
print('finished SSS deseason')
psl_deseason = (psl_ds.groupby('time.month') - psl_ds.groupby('time.month').mean('time')).groupby('time.year').mean('time')
print('finished PSL deseason')


landmask = np.isnan( sst_ds[:,:,0,0].values )

for ilat in range(len(psl_ds.lat)):
    for ilon in range(len(psl_ds.lon)):
        if landmask[ilat,ilon] == True:
            psl_deseason[ilat,ilon,:,:] = np.nan
            
sst_normalized = (sst_deseason - sst_deseason.mean())/sst_deseason.std()
sss_normalized = (sss_deseason - sss_deseason.mean())/sss_deseason.std()
psl_normalized = (psl_deseason - psl_deseason.mean())/psl_deseason.std()

lat = sst_ds.lat
lon = sst_ds.lon

lat_out = np.linspace(lat[0],lat[-1],244)
lon_out = np.linspace(lon[0],lon[-1],244)

ds_out = xr.Dataset({'lat': (['lat'], lat_out), 'lon': (['lon'], lon_out) })

regridder = xe.Regridder(sst_ds, ds_out, 'nearest_s2d')


sst_out = regridder( sst_normalized.transpose('ensemble','year','lat','lon') )
sss_out = regridder( sss_normalized.transpose('ensemble','year','lat','lon') )
psl_out = regridder( psl_normalized.transpose('ensemble','year','lat','lon') )

amv_index = (np.cos(np.pi*sst_out.lat/180) * sst_out).mean(dim=('lat','lon'))

sst_out_values = sst_out.values
sss_out_values = sss_out.values
psl_out_values = psl_out.values

sst_out_values[np.isnan(sst_out_values)] = 0
sss_out_values[np.isnan(sss_out_values)] = 0
psl_out_values[np.isnan(psl_out_values)] = 0

data_out = np.array([sst_out_values,sss_out_values,psl_out_values])
np.save('CESM_data_sst_sss_psl_deseason_normalized_resized.npy',data_out)
np.save('CESM_label_amv_index.npy',amv_index)
