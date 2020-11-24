#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coarsen Data

Coarsen datasets to the same resolution so that observations
and model output can be input into the Neural Network

Created on Fri Nov 20 21:57:57 2020

@author: gliu
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature



#%% User Edits

# Path to directory containing data, downloaded from link below
# https://drive.google.com/drive/u/0/folders/1o0R4RSj34HNInR9ehZ9Yw2pCiZGRPo-s
datpath = "/Users/gliu/Downloads/2020_Fall/6.862/Project/CESM_Data/"

# Output Path
outpath = "/Users/gliu/Downloads/2020_Fall/6.862/Project/CESM_Data/proc/"

# Path to module
modpath = "/Users/gliu/Downloads/2020_Fall/6.862/Project/predict_amv/"

# Variables to process
vnames  = ["sst","sss","psl","NHFLX"]

# Coarsen Resolution
deg  = 2
tol  = 0.75 # Search Resolution
bbox = [-90,20,0,90]
# Debug option
debug = True
#%% Part 1: CESM Data

st = time.time()

# Import module with functions
sys.path.append(modpath)
import amvmod as amv


for vname in vnames:
    
    # Open dataset and slice to time period
    ncname ="CESM1LE_%s_NAtl_19200101_20051201.nc" % vname
    ds = xr.open_dataset(datpath+ncname)
    
    # Read out the dataset
    lon = ds.lon.values
    lat = ds.lat.values
    var = ds[vname].values
    times = ds.time.values
    
    # Combine ensemble and time dimensions and transpose for input
    nlat,nlon,ntime,nens = var.shape
    var = var.reshape(nlat,nlon,ntime*nens) # [lat x lon x otherdims]
    var = var.transpose(2,0,1) # [otherdims x lat x lon]
    
    # Regrid (With Area Weights)
    cvar2,lat2,lon2 = amv.coarsen_byavg(var,lat,lon,deg,tol,bboxnew=bbox,latweight=False,verbose=True)
    
    if debug:
        
        cvar1,lat2,lon2 = amv.coarsen_byavg(var,lat,lon,deg,tol,bboxnew=bbox,latweight=True,verbose=True)
        plt.pcolormesh(lon2,lat2,cvar2[0,:,:]),plt.title(vname+"NotArea Weighted"),plt.colorbar(),plt.show()
        plt.pcolormesh(lon2,lat2,cvar1[0,:,:]),plt.title(vname+"Area Weighted"),plt.colorbar(),plt.show()
        plt.pcolormesh(lon,lat,var[0,:,:]),plt.title(vname+"Original"),plt.colorbar(),plt.show()
    
    # Uncombine lat/lon
    outvar = cvar2.transpose(1,2,0)
    outvar = outvar.reshape(outvar.shape[0],outvar.shape[1],ntime,nens)
    
    # Save output
    dsnew = xr.DataArray(outvar[:,:,:,0:40],
                    coords={'lat':lat2,'lon':lon2,'time':times,"ensemble":np.arange(1,41,1)},
                    dims={'lat':lat2,'lon':lon2,"time":times,"ensemble":np.arange(1,41,1)},
                    name=vname)
    
    dsnew.to_netcdf(outpath+"CESM1LE_%s_NAtl_19200101_20051201_Regridded%ideg.nc"%(vname,deg),encoding={vname: {'zlib': True}})   
    
    print("Completed %s (t=%.2fs)"% (vname,time.time()-st))


#%% Make a land/ice mask from CESM Data

dsm = xr.open_dataset(outpath+"CESM1LE_%s_NAtl_19200101_20051201_Regridded%ideg.nc"%('sst',deg))
msk = dsm.sst.values
msk = msk.sum((2,3))
msk[~np.isnan(msk)] = 1


if debug:
    plt.pcolormesh(lon2,lat2,msk),plt.colorbar()

#%% Coarsen HadISST Data
st = time.time()

ds = xr.open_dataset(datpath+"hadisst.1870-01-01_2018-12-01.nc")

# Select Region
lonW,lonE,latS,latN = bbox
dsr = ds.sel(lat=slice(latS,latN))
dsr = dsr.sel(lon=slice(lonW,lonE))

# Read out variables
lon = dsr.lon.values
lat = dsr.lat.values
var = dsr.sst.values
times = dsr.time.values

# Coarsen
# Regrid (With Area Weights)
cvar2,lat2,lon2 = amv.coarsen_byavg(var,lat,lon,deg,tol,bboxnew=bbox,latweight=False,verbose=True)

# Apply Mask
cvar2 *= msk

if debug:
    
    cvar1,lat2,lon2 = amv.coarsen_byavg(var,lat,lon,deg,tol,bboxnew=bbox,latweight=True,verbose=True)
    plt.pcolormesh(lon2,lat2,cvar2[0,:,:]),plt.title(vname+"NotArea Weighted"),plt.colorbar(),plt.show()
    plt.pcolormesh(lon2,lat2,cvar1[0,:,:]),plt.title(vname+"Area Weighted"),plt.colorbar(),plt.show()
    plt.pcolormesh(lon,lat,var[0,:,:]),plt.title(vname+"Original"),plt.colorbar(),plt.show()

# Transpose to lat x lon x time
cvar2 = cvar2.transpose(1,2,0)    

# Save output
dsnew = xr.DataArray(cvar2,
                coords={'lat':lat2,'lon':lon2,'time':times},
                dims={'lat':lat2,'lon':lon2,"time":times},
                name='sst')

dsnew.to_netcdf(outpath+"HadISST_%s_NAtl_18700101_20181201_Regridded%ideg.nc"%('sst',deg),encoding={'sst': {'zlib': True}})   

print("Completed %s (t=%.2fs)"% ('HadISST',time.time()-st))

#%% Coarsen 20C Re-anaysis
st = time.time()

# Read in the data
ds = xr.open_dataset(datpath+"prmsl.mon.mean.nc")
var = ds.prmsl.values
lon = ds.lon.values
lat = ds.lat.values
times = ds.time.values


# Coarsen
# Regrid (With Area Weights)
cvar2,lat2,lon2 = amv.coarsen_byavg(var,lat,lon,deg,tol,bboxnew=bbox,latweight=False,verbose=True)

# Apply Mask
cvar2 *= msk

if debug:
    
    cvar1,lat2,lon2 = amv.coarsen_byavg(var,lat,lon,deg,tol,bboxnew=bbox,latweight=True,verbose=True)
    plt.pcolormesh(lon2,lat2,cvar2[0,:,:]),plt.title(vname+"NotArea Weighted"),plt.colorbar(),plt.show()
    plt.pcolormesh(lon2,lat2,cvar1[0,:,:]),plt.title(vname+"Area Weighted"),plt.colorbar(),plt.show()
    plt.pcolormesh(lon,lat,var[0,:,:]),plt.title(vname+"Original"),plt.colorbar(),plt.show()

# Transpose to lat x lon x time
cvar2 = cvar2.transpose(1,2,0)    

# Note values are in Pa


# Save output
dsnew = xr.DataArray(cvar2,
                coords={'lat':lat2,'lon':lon2,'time':times},
                dims={'lat':lat2,'lon':lon2,"time":times},
                name='psl')

dsnew.to_netcdf(outpath+"NOAA20CR_%s_NAtl_18510101_20141201_Regridded%ideg.nc"%('psl',deg),encoding={'psl': {'zlib': True}})   

print("Completed %s (t=%.2fs)"% ('NOAA 20CR',time.time()-st))


#%% CGLORS salinity

dsn = "/Users/gliu/Downloads/06_School/Unsorted/F2019/12860/TermProject/tp_p1/Data/data_f/C-GLORSv5_sea_surface_salinity_1980_2015.nc"

ds = xr.open_dataset(dsn,decode_times=False)
ds = xr.open_dataset(dsn)

# Time dimension
times = xr.cftime_range(start='1980-01-15',periods=432,freq="MS")

# Read out variables
var = ds.sea_surface_salinity.values #[time x lon x lat]
lon = ds.lon
lat = ds.lat

# Coarsen
# Regrid (With Area Weights)
cvar2,lat2,lon2 = amv.coarsen_byavg(var,lat,lon,deg,tol,bboxnew=bbox,latweight=False,verbose=True)

# Apply Mask
cvar2 *= msk

if debug:
    
    cvar1,lat2,lon2 = amv.coarsen_byavg(var,lat,lon,deg,tol,bboxnew=bbox,latweight=True,verbose=True)
    plt.pcolormesh(lon2,lat2,cvar2[0,:,:]),plt.title(vname+"NotArea Weighted"),plt.colorbar(),plt.show()
    plt.pcolormesh(lon2,lat2,cvar1[0,:,:]),plt.title(vname+"Area Weighted"),plt.colorbar(),plt.show()
    plt.pcolormesh(lon,lat,var[0,:,:]),plt.title(vname+"Original"),plt.colorbar(),plt.show()

# Transpose to lat x lon x time
cvar2 = cvar2.transpose(1,2,0)    

# Save output
dsnew = xr.DataArray(cvar2,
                coords={'lat':lat2,'lon':lon2,'time':times},
                dims={'lat':lat2,'lon':lon2,"time":times},
                name='sss')

dsnew.to_netcdf(outpath+"CGLORSv5_%s_NAtl_19800115_20160101_Regridded%ideg.nc"%('sss',deg),encoding={'sss': {'zlib': True}})   

print("Completed %s (t=%.2fs)"% ('CGLORSv5',time.time()-st))
