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

machine  = "stormtrack"
varnames = ["SST","SSH","SSS","PSL","NHFLX","BSF","HMXL"]
datpath  = "../../CESM_data/Predictors/" # Path to SST data processed by prep_data_byvariable.py

# Information to access file "[datpath]CESM1LE_[varname]_NAtl_[ystart]0101_[yend]1201_[method]_detrend0_regrid[regrid].nc"
ystart   = 1920 # Start year
yend     = 2005 # End year
method   = "bilinear" # regridding method for POP ocean data
regrid   = None # Set to desired resolution. Set None for no regridding.


# Other settings
recalc_ensavg = True # If False, looks for file: "[datpath]CESM1LE_[varname]_FULL_HTR_[method]_ensavg_[ystart]to[yend].nc}
save_ensavg   = False # If True, save recalculated ensemble average

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

# Get experiment bounding box for preprocessing
bbox_crop   = pparams.bbox_crop
bbox_SP     = pparams.bbox_SP#[-60,-15,40,65]
bbox_ST     = pparams.bbox_ST#[-80,-10,20,40]
bbox_TR     = pparams.bbox_TR#[-75,-15,10,20]
bbox_NA     = pparams.bbox_NA#[-80,0 ,0,65]
regions     = pparams.regions
print(regions)
bboxes      = (bbox_NA,bbox_SP,bbox_ST,bbox_TR,) # Bounding Boxes
debug       = False

#
# %%
#

nvars = len(varnames)
for v in range(nvars):
    
    # Load in predictor variable # [ensemble x year x lat x lon]
    varname = varnames[v]
    ncname  = "%sCESM1LE_%s_NAtl_%i0101_%i1201_%s_detrend0_regrid%s.nc" % (datpath,varname,ystart,yend,method,regrid)
    ds      = xr.open_dataset(ncname).load() 
    
    # Load in (or recalculate + optionally save) the ensemble average
    ncname_ensavg = "%sCESM1LE_%s_FULL_HTR_%s_ensavg_%sto%s.nc" % (datpath,varname,method,ystart,yend)
    if recalc_ensavg:
        print("Re-calculculating the Ensemble Average for %s!" % (varname))
        ds_ensavg = ds.mean('ensemble')
        if save_ensavg:
            exists_flag = proc.checkfile(ncname_ensavg)
            if exists_flag:
                print("Since file exists new ensavg calculation will not be saved.")
            else:
                encoding_dict = {varname:{'zlib':True}}
                ds_ensavg.to_netcdf(ncname_ensavg,encoding=encoding_dict)
                print("Saved new ensavg calculation.")
    else:
        ds_ensavg = xr.open_dataset(ncname_ensavg)
    