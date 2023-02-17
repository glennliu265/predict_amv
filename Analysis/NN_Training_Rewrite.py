#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempts to Re-write NN Training Script, Starting from the Bare Bones

Created on Wed Feb 15 13:50:14 2023

@author: gliu
"""

import sys
import numpy as np
import os

# -----------------------------------------------------------------------------
# Think about moving the section between this into another setup script VVV
#%% Load packages and parameter spaces

cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams

# Load Predictor Information
bbox          = pparams.bbox

#%%




dataset_name    =
varname         = 

datpath         = 


# Predictor and Target Information (Likely Fixed by Loop)
detrend         = 0    # True if the target was detrended
regrid          = None # Regrid option of data
ystart          = 1850 # Start year of processed dataset
yend            = 2014 # End year of processed dataset
lowpass         = 0    # True if the target was low-pass filtered

ens             = 30   # Number of ensemble members to limit to
bbox            = pparams.bbox

#%% # Think about moving the section above this into another setup script ^^^
# -----------------------------------------------------------------------------



#%% Data Loading...

# Load predictor and labels,lat,lon, cut region
data,target,lat,lon = am.load_cmip6_data(dataset_name,varname,bbox,datpath=datpath,
                                 detrend=detrend,regrid=regrid,
                                 ystart=ystart,yend=yend,lowpass=lowpass,
                                 return_latlon=True)

# Subset predictor by ensemble, remove NaNs, and get sizes
data                           = data[:,0:ens,...]      # Limit to Ens
data[np.isnan(data)]           = 0                      # NaN Points to Zero
nchannels,nens,ntime,nlat,nlon = data.shape             # Ignore year and ens for now...
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN

"""
# Output: 
    predictors :: [channel x ens x year x lat x lon]
    labels     :: [ens x year]
"""
#%%



