#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Prepare Detrended data

    - Uses output from [prep_data_byvariable].
    - For each predictor, removes ensemble average from each point

Created on Mon Jun 12 10:02:01 2023

@author: gliu
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys

# ------------
#%% User Edits
# ------------

machine       = "stormtrack"
varnames      = ["SLP","NHFLX","BSF","HMXL"] # "SST","SSH","SSS",
datpath       = "../../CESM_data/Predictors/" # Path to SST data processed by prep_data_byvariable.py

# Information to access file "[datpath]CESM1LE_[varname]_NAtl_[ystart]0101_[yend]1201_[method]_detrend0_regrid[regrid].nc"
ystart        = 1920 # Start year
yend          = 2005 # End year
method        = "bilinear" # regridding method for POP ocean data
regrid        = None # Set to desired resolution. Set None for no regridding.

# Other settings
save_ensavg   = False # If True, save recalculated ensemble average to : "[datpath]CESM1LE_[varname]_FULL_HTR_[method]_ensavg_[ystart]to[yend].nc}
debug         = True

# -----------------------------------------------------------------------------
#%% Import Packages + Paths based on machine
# -----------------------------------------------------------------------------

# Get Project parameters
sys.path.append("../")
import predict_amv_params as pparams
import amv_dataloader as dl

# Get paths based on machine
machine_paths = pparams.machine_paths[machine]

# Import custom modules
sys.path.append(machine_paths['amv_path'])
from amv import loaders,proc

# -----------------------------------------------------------------------------
#%% Main Script
# -----------------------------------------------------------------------------

nvars = len(varnames)
for v in range(nvars):
    
    # Load in predictor variable # [ensemble x year x lat x lon]
    varname = varnames[v]
    ncname  = "%sCESM1LE_%s_NAtl_%i0101_%i1201_%s_detrend0_regrid%s.nc" % (datpath,varname,ystart,yend,method,regrid)
    ds      = xr.open_dataset(ncname).load() 
    
    # Recalculate + optionally save the ensemble average
    print("Calculating the Ensemble Average for %s!" % (varname))
    ds_ensavg     = ds.mean('ensemble')
    if save_ensavg:
        ncname_ensavg = "%sCESM1LE_%s_FULL_HTR_%s_ensavg_%sto%s.nc" % (datpath,varname,method,ystart,yend)
        encoding_dict = {varname:{'zlib':True}}
        ds_ensavg.to_netcdf(ncname_ensavg,encoding=encoding_dict)
        print("Saved new ensavg calculation as %s." % ncname_ensavg)
    
    # Remove ensemble average and resave
    ds_detrended = ds[varname] - ds_ensavg[varname]
    
    # Check differences
    #assert np.all( (ds[varname] - ds_ensavg[varname]) == ds_detrended)
    if debug:
        lonf,latf = -30,50 # Chose random latlon to check
        iyear,iens = 2,2
        print("Checking values for %s" % varname)
        print("\tOriginal value @ %.2f, %.2f      : %.3f" % (lonf,latf,ds[varname].sel(lon=lonf,lat=latf,method='nearest').isel(year=iyear,ensemble=iens)))
        print("\tRemoved ensemble average was     : %.3f" % (ds_ensavg[varname].sel(lon=lonf,lat=latf,method='nearest').isel(year=iyear)))
        print("\tFinal value is                   : %.3f" % (ds_detrended.sel(lon=lonf,lat=latf,method='nearest').isel(year=iyear,ensemble=iens)))
        print("\tMaximum ensavg of new dataset is : %.3e" % (np.nanmax(ds_detrended.mean('ensemble'))))
        
    # Save file
    ncname_out    = "%sCESM1LE_%s_NAtl_%i0101_%i1201_%s_detrend1_regrid%s.nc" % (datpath,varname,ystart,yend,method,regrid)
    encoding_dict = {varname:{'zlib':True}}
    ds_detrended.to_netcdf(ncname_out,encoding=encoding_dict)