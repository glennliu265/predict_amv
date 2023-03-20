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
import amv_dataloader as dl
import amvmod as am

# Load Predictor Information
bbox          = pparams.bbox

#%%

"""
Thinking of 3 major categories

---------------------------------
(1) Data Preprocessing Parameters
---------------------------------
Decisions made in the data preprocessing step
    - detrend
    - regrid
    - region of index
    - ice masking
    - predictor (varnames)
    - dataset
    - season
    - lowpass
    - dataset/cmipver

---------------------------------
(2) Subsetting
---------------------------------
Determining how the data is subsetting for training
    - Thresholding Type (quantile, stdev)
    - Test/Train/Val Split Percentage
    - Crossfold Offset
    - # of Ensemble Members to Use
    - Time Period of Trainining (ystart, yend)
    - Bounding Box
    - Training Sample Size (nsamples)
    
    
---------------------------------
(3) Machine Learning Parameters
---------------------------------
    - epochs
    - assorted hyperparameters
    - early stop
    - architecture


"""

# Predictor and Target Information (Likely Fixed by Loop)

# ---------------------------------
# (1) Data Preprocessing Parameters
# ---------------------------------
detrend         = 0        # True if the target was detrended
varnames        = ["SST",]

region          = None     # Region of AMV Index (not yet implemented)
season          = None     # Season of AMV Index (not yet implemented)
cmipver         = "cesm1"  # "5","6","cesm1","reanalysis"

lowpass         = False    # True if the target was low-pass filtered
regrid          = None     # Regrid option of data
mask            = True     # True for land-ice masking

# ---------------------------------
# (2) Subsetting Parameters
# ---------------------------------
ens             = 30   # Number of ensemble members to limit to
ystart          = 1850 # Start year of processed dataset
yend            = 2014 # End year of processed dataset
nsamples        = 300  # Number of samples from each class to train with
bbox            = pparams.bbox

# ---------------------------------
# (2) ML Parameters
# ---------------------------------


#%% # Think about moving the section above this into another setup script ^^^
# -----------------------------------------------------------------------------



#%% Data Loading...

# Load predictor and labels,lat,lon, cut region
if cmipver == "6":
    data,target,lat,lon = am.load_cmip6_data(dataset_name,varname,bbox,datpath=datpath,
                                     detrend=detrend,regrid=regrid,
                                     ystart=ystart,yend=yend,lowpass=lowpass,
                                     return_latlon=True)
elif cmipver == "cesm1":
    target         = dl.load_target_cesm(detrend=detrend,region=region)
    data,lat,lon   = dl.load_data_cesm(varnames,bbox,detrend=detrend,return_latlon=True)
    




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



