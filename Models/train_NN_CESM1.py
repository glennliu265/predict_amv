#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train Neural Networks (NN) for CESM1 Large Ensemble Simulations

 - Copied introductory section from NN_Training_Rewrite.py on 2023.03.20
 - Based on NN_test_lead_ann_ImageNet_classification_singlevar.py


Updated Procedure
    1) Create Experiment Directory
    2) Load Data
    3) Determine (and make) AMV Classes



General Procedure (Old)
    1) Load Data
    2) Create Experiment Directory
    3) Determine AMV Class Threshold (for exact threshold)
    4) Loop for runid/weight initialization...
        5) Set output name, preallocate arrays
        6) Looping by leadtime...

             7) Apply lead/lag to data
             8) Make AMV classes

             9) Select samples
             10) Test/Train Split, Set up Dataloader
             11) Initialize model
             12) Train Model
             13) Test the model again?
             14) Convert and store predicted values
             15) Save the model
             16) Compute & Save success metrics

Created on Mon Mar 20 21:34:32 2023
@author: gliu


"""

import sys
import numpy as np
import os
import time
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset,Dataset

# -----------------------------------------------------------------------------
# Think about moving the section between this into another setup script VVV
#%% Load packages and parameter spaces

# Import general utilities from amv module
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import proc


# Import packages specific to predict_amv
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

Specific Machine Learning Parameters (regularization, training options, etc)

    - epochs
    - assorted hyperparameters
    - early stop
    - architecture
    - batch size
    - 


"""
# # Create Experiment Directory
expdir             = "FNN4_128_SingleVar_Rewrite"


# ---------------------------------
# (1) Data Preprocessing Parameters
# ---------------------------------
detrend         = 0        # True if the target was detrended
varnames        = ["SST",] # Names of predictor variables
region          = None     # Region of AMV Index (not yet implemented)
season          = None     # Season of AMV Index (not yet implemented)
lowpass         = False    # True if the target was low-pass filtered
regrid          = None     # Regrid option of data
mask            = True     # True for land-ice masking

# ---------------------------------
# (2) Subsetting Parameters
# ---------------------------------
# Data subsetting
ens             = 42      # Number of ensemble members to limit to
ystart          = 1850    # Start year of processed dataset
yend            = 2014    # End year of processed dataset
bbox            = pparams.bbox

# Label Determination
quantile        = False   # Set to True to use quantiles
thresholds      = [-1,1]  # Thresholds (standard deviations, or quantile values) 

# Test/Train/Validate and Sampling
nsamples        = 300     # Number of samples from each class to train with
percent_train   = 0.60    # Training percentage
percent_val     = 0.10    # Validation Percentage

# Cross Validation Options
cv_loop         = True    # Repeat for cross-validation
cv_offset       = 0       # Set cv option. Default is test size chunk

# ---------------------------------
# (2) ML Parameters
# ---------------------------------

# Network Hyperparameters
netname       = "FNN4_128"           # Key for Architecture Hyperparameters
loss_fn       = nn.CrossEntropyLoss()# Loss Function (nn.CrossEntropyLoss())
opt           = ['Adam',1e-3,0]      # [Optimizer Name, Learning Rate, Weight Decay]
use_softmax   = False                # Set to true to change final layer to softmax
reduceLR      = False                # Set to true to use LR scheduler
LRpatience    = False                # Set patience for LR scheduler

# Regularization and Training
early_stop    = 3                    # Number of epochs where validation loss increases before stopping
max_epochs    = 20                   # Maximum # of Epochs to train for
batch_size    = 16                   # Pairs of predictions
unfreeze_all  = True                 # Set to true to unfreeze all layers, false to only unfreeze last layer


# ---------------------------------
# (3) Other Parameters
# ---------------------------------
runids         = np.arange(0,11,1)    # Which runs to do
leads          = np.arange(0,26,1)    # Prediction Leadtimes
debug          = True                 # Set to true for debugging/verbose outputs
checkgpu       = True                 # Set to true to check if GPU is available
savemodel      = True                 # Set to true to save model weights
#% # Think about moving the section above this into another setup script ^^^
# -----------------------------------------------------------------------------

# >> Add Loop by Predictor >> >>

# ------------------------------------------------------------
# %% Check for existence of experiment directory and create it
# ------------------------------------------------------------
proc.makedir("../../CESM_data/"+expdir)
for fn in ("Metrics","Models","Figures"):
    proc.makedir("../../CESM_data/"+expdir+"/"+fn)

# ----------------------------------------------
#%% Data Loading...
# ----------------------------------------------
# Loads that that has been preprocessed by: ___

# Load predictor and labels, lat/lon, cut region
target         = dl.load_target_cesm(detrend=detrend,region=region)
data,lat,lon   = dl.load_data_cesm(varnames,bbox,detrend=detrend,return_latlon=True)

# Subset predictor by ensemble, remove NaNs, and get sizes
data                           = data[:,0:ens,...]      # Limit to Ens
data[np.isnan(data)]           = 0                      # NaN Points to Zero
nchannels,nens,ntime,nlat,nlon = data.shape             # Ignore year and ens for now...
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN

# ------------------------------------------------------------
# %% Determine the AMV Classes
# ------------------------------------------------------------

# Set exact threshold value
std1         = target.std(1).mean() * thresholds[1] # Multiple stdev by threshold value 
if quantile is False:
    thresholds_in = [-std1,std1]
else:
    thresholds_in = thresholds

# Classify AMV Events
target_class = am.make_classes(target.flatten()[:,None],thresholds_in,exact_value=True,reverse=True,quantiles=quantile)
target_class = target_class.reshape(target.shape)

# Get necessary dimension sizes/values
nclasses     = len(thresholds)+1
nlead        = len(leads)

"""
# Output: 
    predictors :: [channel x ens x year x lat x lon]
    labels     :: [ens x year]
"""
# ------------------------------------------------------------
# %% Looping for runid
# ------------------------------------------------------------

for v,varname in varnames: #...
    
    
    for nr,runid in enumerate(runids):
        rt = time.time()
        
        # Save data (ex: Ann2deg_NAT_CNN2_nepoch5_nens_40_lead24 )
        expname = ("AMVClass%i_%s_nepoch%02i_" \
                   "nens%02i_maxlead%02i_"\
                   "detrend%i_run%02i_"\
                   "quant%i_res%s" % (nclasses,netname,max_epochs,
                                         ens,leads[-1],detrend,runid,
                                         quantile,regrid))
        
        # Preallocate Evaluation Metrics...
        corr_grid_train = np.zeros((nlead))
        corr_grid_test  = np.zeros((nlead))
        
        train_loss_grid = []#np.zeros((max_epochs,nlead))
        test_loss_grid  = []#np.zeros((max_epochs,nlead))
        val_loss_grid   = []
        
        train_acc_grid  = []
        #test_acc_grid   = []
        val_acc_grid    = []
        
        acc_by_class    = []
        total_acc       = []
        yvalpred        = []
        yvallabels      = []
        sampled_idx     = []
        thresholds_all  = []
        
        if checkgpu:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        
        # -------------
        # Print Message
        # -------------
        print("Running [train_NN_CESM1.py] with the following settings:")
        print("\tNetwork Type   : "+netname)
        print("\tPredictor(s)   : "+str(varnames))
        print("\tLeadtimes      : %i to %i" % (leads[0],leads[-1]))
        print("\tRunids         : %i to %i" % (runids[0],runids[-1]))
        print("\tMax Epochs     : " + str(max_epochs))
        print("\tEarly Stop     : " + str(early_stop))
        print("\t# Ens. Members : "+ str(ens))
        print("\tDetrend        : "+ str(detrend))
        
        for l,lead in enumerate(leads):
            
            # ---------------------------------
            # Set names for intermediate saving
            # ---------------------------------
            if (lead == leads[-1]) and (len(leads)>1): # Output all files together
                outname = "/leadtime_testing_%s_%s_ALL.npz" % (varname,expname)
            else: # Output individual lead times while training
                outname = "/leadtime_testing_%s_%s_lead%02dof%02d.npz" % (varname,expname,lead,leads[-1])
            
            # ----------------------
            # Apply lead/lag to data
            # ----------------------
            # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
            X,y_class = am.apply_lead(data,target_class,lead,reshape=True,ens=ens,tstep=ntime)
            
            # ----------------------
            # Select samples
            # ----------------------
            if nsamples is None: # Default: nsamples = smallest class
                threscount = np.zeros(nclasses)
                for t in range(nclasses):
                    threscount[t] = len(np.where(y_class==t)[0])
                nsamples = int(np.min(threscount))
            y_class,X,shuffidx = am.select_samples(nsamples,y_class,X,verbose=debug)
            lead_nsamples      = y_class.shape[0]
            sampled_idx.append(shuffidx) # Save the sample indices
            
            # --------------------------
            # Flatten input data for FNN
            # --------------------------
            if "FNN" in netname:
                ndat,nchan,nlat,nlon = X.shape
                inputsize            = nchan*nlat*nlon
                outsize              = nclasses
                X                    = X.reshape(ndat,inputsize)
            
            # --------------------------
            # Train Test Split
            # --------------------------
            X_subsets,y_subsets      = am.train_test_split(X,y_class,percent_train,
                                                           percent_val=percent_val,
                                                           debug=debug,offset=cv_offset)
            # Convert to Tensors
            X_subsets = [torch.from_numpy(X.astype(np.float32)) for X in X_subsets]
            y_subsets = [torch.from_numpy(y.astype(np.compat.long)) for y in y_subsets]
            
            
            # # Put into pytorch dataloaders
            data_loaders = [DataLoader(TensorDataset(X_subsets[iset],y_subsets[iset]), batch_size=batch_size) for iset in range(len(X_subsets))]
            train_loader,test_loader,val_loader = data_loaders
            
            # ---------------
            # Train the model
            # ---------------
            nn_params = pparams.nn_param_dict[netname] # Get corresponding param dict for network
            
            # Initialize model
            if "FNN" in netname:
                layers = am.build_FNN_simple(inputsize,outsize,nn_params['nlayers'],nn_params['nunits'],nn_params['activations'],
                                          dropout=nn_params['dropout'],use_softmax=use_softmax)
                pmodel = nn.Sequential(*layers)
                
            else:
                # Note: Currently not supported due to issues with timm model. Need to rewrite later...
                pmodel = am.transfer_model(netname,nclasses,cnndropout=nn_params['cnndropout'],unfreeze_all=unfreeze_all,
                                        nlat=nlat,nlon=nlon,nchannels=nchannels)
            # Train/Validate Model
            model,trainloss,valloss,trainacc,valacc = am.train_ResNet(pmodel,loss_fn,opt,train_loader,val_loader,max_epochs,
                                                                     early_stop=early_stop,verbose=debug,
                                                                     reduceLR=reduceLR,LRpatience=LRpatience,checkgpu=checkgpu)
            
            # Save train/validation loss
            train_loss_grid.append(trainloss)
            val_loss_grid.append(valloss)
            train_acc_grid.append(trainacc)
            val_acc_grid.append(valacc)
            
            # --------------
            # Test the model
            # --------------
            y_predicted,y_actual,test_loss = am.test_model(model,test_loader,loss_fn,
                                                           checkgpu=checkgpu,debug=False)
            lead_acc,class_acc = am.compute_class_acc(y_predicted,y_actual,nclasses,debug=True,verbose=False)
            
            yvalpred.append(y_predicted)
            yvallabels.append(y_actual)
            test_loss_grid.append(test_loss)
            acc_by_class.append(class_acc)
            total_acc.append(lead_acc)
            
            
            
            # --------------
            # Save the model
            # --------------
            if savemodel:
                modout = "../../CESM_data/%s/Models/%s_%s_lead%02i_classify.pt" %(expdir,expname,varname,lead)
                torch.save(model.state_dict(),modout)
            
             


