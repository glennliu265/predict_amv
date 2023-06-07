#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Copied from prepare_training_validation.py
Essentially only does the SST component and averages over particular regions



Created on Fri Dec  2 11:48:40 2022

Update: 2023.06.06 -> Works with new output from prep_data_byvariable

   - Loads in SST preprocessed by [prep_data_byvariable] in ../../CESM_data/Predictors/
         Input naming style  : CESM1LE_SST_NAtl_19200101_20051201_bilinear_detrend0_regridNone.nc
   - Outputs Indices averaged over each region enumerated from [predict_amv_params]
     Files are located in ../../CESM_data/Targets/
         
         Output naming style : CESM1LE_label_NAT_NASST_index_detrend0_regridNone.npy
     

@author: gliu
"""

import numpy as np
import xarray as xr
import xesmf as xe
import sys

# ------------
#%% User Edits
# ------------

machine = "stormtrack"
detrend = False # Detrending is currently not applied
regrid  = None # Set to desired resolution. Set None for no regridding.

datpath  = "../../CESM_data/Predictors/" # Path to SST data processed by prep_data_byvariable.py
maskpath = "../../CESM_data/Masks/"
outpath  = "../../CESM_data/Targets/"
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
bboxes      = (bbox_SP,bbox_ST,bbox_TR,bbox_NA,) # Bounding Boxes
   # regionlong  = pparams.regionlong
    #regionlong  = ("Subpolar","Subtropical","Tropical","North Atlantic","Subtropical (East)","Subtropical (West)",)
    
    
#
#%%

# --------------------------------
# Load the DataArray
# --------------------------------
ncname = "%sCESM1LE_SST_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc" % (datpath,detrend,regrid)
sst_ds = xr.open_dataset(ncname)['SST'][:,:,:,:] # [Ensemble x Year x Lat x Lon]


# ----------------------------
# Load land/ice mask and apply
# ----------------------------
limask = dl.load_limask(datpath=maskpath,bbox=bbox_crop) #[lat x lon]
sst_ds = sst_ds * limask[None,None,...]

# -----------------------------------------------
# Calculate the AMV Index (Area weighted average)
# -----------------------------------------------
for b,bb in enumerate(bboxes):
    sst_out_reg = sst_ds.sel(lon=slice(bb[0],bb[1]),lat=slice(bb[2],bb[3]))
    amv_index   = (np.cos(np.pi*sst_out_reg.lat/180) * sst_out_reg).mean(dim=('lat','lon'))
    savename = '%sCESM1LE_label_%s_NASST_index_detrend%i_regrid%s.npy' % (outpath,regions[b],detrend,regrid)
    np.save(savename,amv_index[:,:])
    print("Saved Region %s ([%s]) as %s" % (regions[b],bb,savename))





