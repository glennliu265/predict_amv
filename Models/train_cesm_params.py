#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Training Params

File containing input training parameters for the training script

Created on Thu Mar 23 09:21:09 2023

@author: gliu
"""

# Load my own custom modules

import sys
import os


from torch import nn

import predict_amv_params as pparams
import numpy as np

# Import Custom Packages
cwd = os.getcwd()
sys.path.append(cwd+"/../")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
sys.path.append("../")
import predict_amv_params as pparams

# Dictionary Containing All Parameters
train_params_all = {}

# --------------------------
#%% Experiment dictionaries 
# =============================
# Note: Copy section below and change parameters. These are stored in a dict [expdict]
#       that can be accessed via a unique key of your choosing [expname]

"""

FNN4_128_SingleVar_Rewrite

Single Variable Training for 4 Layer Fully-Connected NN

"""

# # Create Experiment Directory
expname                    = "FNN4_128_SingleVar_Rewrite"
expdict                    = {}

# ---------------------------------
# (1) Data Preprocessing Parameters
# ---------------------------------
expdict['detrend']         = 0        # True if the target was detrended
expdict['varnames']        = ["SST","SSH",] # Names of predictor variables
expdict['region']          = None     # Region of AMV Index (not yet implemented)
expdict['season']          = None     # Season of AMV Index (not yet implemented)
expdict['lowpass']         = False    # True if the target was low-pass filtered
expdict['regrid']          = None     # Regrid option of data
expdict['mask']            = True     # True for land-ice masking

# ---------------------------------
# (2) Subsetting Parameters
# ---------------------------------
# Data subsetting
expdict['ens']             = 42      # Number of ensemble members to limit to
expdict['ystart']          = 1920    # Start year of processed dataset
expdict['yend']            = 2005    # End year of processed dataset
expdict['bbox']            = pparams.bbox

# Label Determination
expdict['quantile']        = False   # Set to True to use quantiles
expdict['thresholds']      = [-1,1]  # Thresholds (standard deviations, or quantile values) 

# Test/Train/Validate and Sampling
expdict['nsamples']        = 300     # Number of samples from each class to train with
expdict['percent_train']   = 0.60    # Training percentage
expdict['percent_val']     = 0.10    # Validation Percentage

# Cross Validation Options
expdict['cv_loop']         = True    # Repeat for cross-validation
expdict['cv_offset']       = 0       # Set cv option. Default is test size chunk

# ---------------------------------
# (2) ML Parameters
# ---------------------------------

# Network Hyperparameters
expdict['netname']       = "FNN4_128"           # Key for Architecture Hyperparameters
expdict['loss_fn']       = nn.CrossEntropyLoss()# Loss Function (nn.CrossEntropyLoss())
expdict['opt']           = ['Adam',1e-3,0]      # [Optimizer Name, Learning Rate, Weight Decay]
expdict['use_softmax']   = False                # Set to true to change final layer to softmax
expdict['reduceLR']      = False                # Set to true to use LR scheduler
expdict['LRpatience']    = False                # Set patience for LR scheduler

# Regularization and Training
expdict['early_stop']    = 3                    # Number of epochs where validation loss increases before stopping
expdict['max_epochs']    = 20                   # Maximum # of Epochs to train for
expdict['batch_size']    = 16                   # Pairs of predictions
expdict['unfreeze_all']  = True                 # Set to true to unfreeze all layers, false to only unfreeze last layer


# ---------------------------------
# (3) Other Parameters
# ---------------------------------
expdict['runids']         = np.arange(0,21,1)    # Which runs to do
expdict['leads']          = np.arange(0,26,3)    # Prediction Leadtimes
expdict['debug']          = True                 # Set to true for debugging/verbose outputs
expdict['checkgpu']       = True                 # Set to true to check if GPU is available
expdict['savemodel']      = True                 # Set to true to save model weights

train_params_all[expname] = expdict.copy()