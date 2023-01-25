#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Based on calc_uohc, compute the upper ocean quantity of a selected variable

General Procedure

   1. Locate HMXL and [ocnvar] file
   2. For each timestep/point... 
       a. index the mixed-layer depth and find the corresponding z_top index
       b. store these indices somewhere
       c. select this point and add temperatures above this value for [ocnvar]
       d. divide by the depth/thickness
       c. Append/save to a file [time x lat x lon]
       
Copied upper section from /reemergence/preprocess_data.py

Created on Fri Jan 20 14:08:53 2023

@author: gliu
"""

import time
import numpy as np
import xarray as xr
import glob
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

#from scipy.io import loadmat
#%%

# Variables I downloaded from NCAR
# /stormtrack/data3/glliu/01_Data/02_AMV_Project/00_Commons/CESM1_LE/%s/" % varname

# Variables downloaded by Young-Oh
# /vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/ocn/proc/tseries/monthly/ % varname

#%% User Edits

# Import module
sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
from amv import proc

# Data Information
scenario       = "20TR" 
varname        = "SALT"
mldname        = "HMXL"
if varname == "TEMP":
    datpath_var    = "/stormtrack/data4/share/deep_learning/data_yuchiaol/cesm_le/"
else:
    datpath_var    = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/00_Commons/CESM1_LE/"
datpath_mld    = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/ocn/proc/tseries/monthly/"

# Output Information
outname        = "UOSC"
outpath        = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/%s/" % outname
proc.makedir(outpath)

# Set Bounding Box/Lat/Lon Info
bbox          = [-80,0,0,65] # Set Bounding Box
bboxfn        = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])
ldz           = np.load("/stormtrack/home/glliu/01_Data/cesm_latlon360.npz",allow_pickle=True)
longlob       = ldz['lon']
latglob       = ldz['lat']

# Set Bounding Time
tstart        = "1920-02-01"
tend          = "2006-01-01"

# Other Preprocessing Options/Toggles
ens           = 40 # Number of ensemble members to include
save_netcdf   = True  # Set to True to save as netcdf rather than .npy
debug         = False

#%% Set up coordinates

dx  = (longlob[1:] - longlob[:-1]).mean()
dy  = (latglob[1:] - latglob[:-1]).mean()
lon = np.arange(bbox[0],bbox[1]+dx,dx)
lat = np.arange(bbox[2],bbox[3]+dy,dy)
nlon,nlat = len(lon),len(lat)

#%% Get List of nc files

# Find nclists
nclists = []
for ii in range(2):
        
    if ii == 0:
        search_path = datpath_var
        search_var  = varname
    elif ii == 1:
        search_path = datpath_mld
        search_var  = mldname
        
    nclist = glob.glob("%s%s/*%s*.nc" % (search_path,search_var,scenario))
    nclist = [nc for nc in nclist if "OIC" not in nc]
    print("Found %i files for %s" % (len(nclist),search_var))
    
    # Keep only up to included ensemble members
    nclist.sort()
    nclist = nclist[:ens]
    nclists.append(nclist)
    
#%%

# def select_depth(ds_in,zlimit,zname="z_t"):
#     # zbounds  = ds_in[zname]
#     # zmax     = np.argmax(zbounds > zlimit)+1
#     # idz      = np.argmax(dztop > mldpt)+1 
#     # zbounds  = zbounds.where(zbounds < zlimit,other=zlimit)
#     # dz       = np.abs(zbounds-np.roll(zbounds,-1))
#     # dz       
#     return
    

# Preallocate
for e in range(ens): # Looping for each ensemble member NOTE I HAVE CHANGED THIS, switch it back later
#for e in np.arange(1,ens+1):
    st_e = time.time()
    
    # Open DataArray
    stld = time.time()
    ds_mld = xr.open_dataset(nclists[1][e])
    ds_var = xr.open_dataset(nclists[0][e])
    
    # Get depth levels (Cell Top)
    z_w    = ds_var.z_w.values
    
    # Restrict time time point
    
    ds_mld = ds_mld[mldname].sel(time=slice(tstart,tend)).load()
    ds_var = ds_var[varname].sel(time=slice(tstart,tend)).load()
    times  = ds_mld.time.values
    print("Data loaded in %.2fs for Ens. %02i" % (time.time()-stld,e+1))
    
    # Preallocate for ensemble member
    ntime     = len(ds_mld.time)
    intgr_var = np.zeros((ntime,nlat,nlon)) * np.nan # [ens x time x lat x lon]
    mld_depth = np.zeros((ntime,nlat,nlon)) * np.nan
    
    # Looping for each point :(
    for o in range(nlon):
        for a in tqdm(range(nlat)):
            
            lonf = lon[o]
            if lonf < 0:
                lonf += 360
            latf = lat[a]
            mld_pt = proc.find_tlatlon(ds_mld,lonf,latf,verbose=False)
            if np.any(np.isnan(mld_pt)): # Skip NaN Points
                continue
            var_pt = proc.find_tlatlon(ds_var,lonf,latf,verbose=False)
            
            for t in range(ntime):
                
                # Get MLD information for the timestep
                zlimit       = mld_pt.isel(time=t).values                          # Get MLD [zlimit] for that timestep
                k_zmax       = np.argmax(z_w > zlimit)                             # Find first level deeper than MLD 
                if debug:
                    print("The first level deeper than %.2f is %.2f, index %i" % (zlimit,z_w[k_zmax],k_zmax))
                
                # Compute dz
                dz_pt          = z_w.copy()
                dz_pt[k_zmax:] = zlimit                                            # Assign zlimit to all levels greater than the target MLD
                dz_pt          = np.abs(dz_pt - np.roll(dz_pt,-1))[:k_zmax]        # Take difference and restrict values
                
                # Get and sum variable values, multiplying by depth of each level
                var_z          = var_pt.isel(time=t).sel(z_t=slice(0,z_w[k_zmax])) # Get values down to first level deeper than MLD 
                var_sum        = (var_z * dz_pt).sum()
                dz             = dz_pt.sum()
                
                # Record to an array
                intgr_var[t,a,o] = var_sum
                mld_depth[t,a,o]  = dz
                # End Time Loop
            # End Lat Loop
        # End Lon Loop
    
    # Save information for variable (integrated variable)
    if save_netcdf:
        # Save variable
        savename_var = "%s%s_%s_%s_ens%02i.nc" % (outpath,outname,scenario,bboxfn,e+1)
        da = proc.numpy_to_da(intgr_var,times,lat,lon,outname) # NOTE: funky error when saving within function having to do with zlib on stormtrack....
        da.to_netcdf(savename_var,
                 encoding={outname: {'zlib': True}})
        
        # Save Integrated depth
        savename_var = "%s%s_%s_%s_ens%02i.nc" % (outpath,outname+"_dz",scenario,bboxfn,e+1)
        da = proc.numpy_to_da(mld_depth,times,lat,lon,outname+"_dz")
        da.to_netcdf(savename_var,
                 encoding={outname+"_dz": {'zlib': True}})
    else:
        # Save variable # (units degC*h)
        savename_var = "%s%s_%s_%s_ens%02i.npy" % (outpath,outname,scenario,bboxfn,e+1)
        np.save(savename_var,intgr_var,)
        
        # Save Integrated depth 
        savename_var = "%s%s_%s_%s_ens%02i.npy" % (outpath,outname+"_dz",scenario,bboxfn,e+1)
        np.save(savename_var,mld_depth,)
    
    print("Completed calculations for Ens %02i in %.2fs" % (e+1,time.time()-st_e))
    # For each timestep, add up to the surface
    
    
    
    
    
    
    
    
