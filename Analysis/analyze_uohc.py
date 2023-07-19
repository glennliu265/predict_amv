#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze UOHC and UOSC calculated via ____

Created on Mon Feb 13 15:30:08 2023

@author: gliu
"""

import time
import numpy as np
import xarray as xr
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import cartopy.crs as ccrs



# Copied sectionb elow from check_AMV_lens, no edits made yet..

#%% User Edits

# I/O, dataset, paths
varname        = "sst" # (tos, sos, zos)
detrend        = False
ystart         = 1850
yend           = 2014
regrid         = None
debug          = True # Set to true to do some debugging

# Paths
machine = "Astraeus"
if machine == "stormtrack":
    lenspath       = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/ts/"
    datpath        = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/processed/"
elif machine == "gliu_mbp":
    datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/LENS_other/processed/"
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
    import viz,proc
elif machine == "Astraeus":
    datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/CMIP6_LENS/processed/"
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
    ice_nc         = "%s../other/siconc_mon_clim_ensavg_CMIP6_10ice_re1x1.nc" % (datpath) # Full path and file of ice mask
    import viz,proc

plt.style.use('default')
## Load some global information (lat.lon) <Note, I need to customize this better..
#%% Import packages and universal variables

# Note; Need to set script into current working directory (need to think of a better way)
import os
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams

# Import paths
figpath         = pparams.figpath
proc.makedir(figpath)

# Load the bounding box
bbox            = pparams.bbox

# Import class information
classes         = pparams.classes
class_colors    = pparams.class_colors

# Import dataset inforation
dataset_names   = pparams.cmip6_names
dataset_long    = pparams.cmip6_names
dataset_colors  = pparams.cmip6_colors
dataset_starts  =(1850,) * len(dataset_names)

# AMV related information
amvbbox         = pparams.amvbbox
print(amvbbox)

#%% Load the data

ds_all        = []
ndata         = len(dataset_names)
for d in range(ndata):
    # Open the dataset
    ncsearch  = "%s%s_%s_NAtl_%sto%s_detrend%i_regrid%sdeg.nc" % (datpath,dataset_names[d],
                                                                 varname,ystart,yend,
                                                                 detrend,regrid)
    ds        = xr.open_dataset(ncsearch).load()
    ds_all.append(ds)