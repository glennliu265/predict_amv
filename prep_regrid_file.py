#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Prepare Regrid File

# Prepare regridding file for CDO

Created on Tue Apr 25 10:35:15 2023

@author: gliu
"""


#%%

import xarray as xr
import numpy as np


#%%

# Open sample regridding file provided by Ray
sample_ds = xr.open_dataset("~/regrid_re1x1.nc")
print(sample_ds)


# Open some CESM1 CAM5 data

datpath           = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
regrid_name       = "TS_clim_PIC_FULL.nc"
ds = xr.open_dataset("%s%s"%(datpath,regrid_name))

# Save the dataset
savename = "~/regrid_CESM1CAM.nc"
ds.to_netcdf(savename)

