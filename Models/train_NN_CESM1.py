#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train Neural Networks (NN) for CESM1 Large Ensemble Simulations

 - Copied introductory section from NN_Training_Rewrite.py on 2023.03.20
 - Based on NN_test_lead_ann_ImageNet_classification_singlevar.py


Current Structure:
    - Indicate CESM1 training parameters in [train_cesm_parameters]
    - Functions are mostly contained in [amvmod.py]
    - Universal Variables + Architectures are in [predict_amv_params.py]
    - Additional helper function from [amv] module [proc] and [viz]

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
import train_cesm_params as train_cesm_params
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

# Load experiment parameters
eparams            = train_cesm_params.train_params_all[expdir]

# Set some looping parameters and toggles
varnames           = ["SST","SSH",]       # Names of predictor variables
leads              = np.arange(0,26,3)    # Prediction Leadtimes
runids             = np.arange(0,21,1)    # Which runs to do
checkgpu           = True                 # Set to true to check if GPU is availabl
debug              = True                 # Set verbose outputs
savemodel          = True                 # Set to true to save model weights
#
# -----------------------------------------------------------------------------

# >> Add Loop by Predictor >> >>
allstart = time.time()

# ------------------------------------------------------------
# %% Check for existence of experiment directory and create it
# ------------------------------------------------------------
proc.makedir("../../CESM_data/"+expdir)
for fn in ("Metrics","Models","Figures"):
    proc.makedir("../../CESM_data/"+expdir+"/"+fn)

# ----------------------------------------------
#%% Data Loading...
# ----------------------------------------------

# Load some variables for ease
ens            = eparams['ens']

# Loads that that has been preprocessed by: ___

# Load predictor and labels, lat/lon, cut region
target         = dl.load_target_cesm(detrend=eparams['detrend'],region=eparams['region'])
data,lat,lon   = dl.load_data_cesm(varnames,eparams['bbox'],detrend=eparams['detrend'],return_latlon=True)

# Subset predictor by ensemble, remove NaNs, and get sizes
data                           = data[:,0:ens,...]      # Limit to Ens
data[np.isnan(data)]           = 0                      # NaN Points to Zero
nchannels,nens,ntime,nlat,nlon = data.shape             # Ignore year and ens for now...
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN

# ------------------------------------------------------------
# %% Determine the AMV Classes
# ------------------------------------------------------------

# Set exact threshold value
std1         = target.std(1).mean() * eparams['thresholds'][1] # Multiple stdev by threshold value 
if eparams['quantile'] is False:
    thresholds_in = [-std1,std1]
else:
    thresholds_in = eparams['thresholds']

# Classify AMV Events
target_class = am.make_classes(target.flatten()[:,None],thresholds_in,exact_value=True,reverse=True,quantiles=eparams['quantile'])
target_class = target_class.reshape(target.shape)

# Get necessary dimension sizes/values
nclasses     = len(eparams['thresholds'])+1
nlead        = len(leads)

"""
# Output: 
    predictors :: [channel x ens x year x lat x lon]
    labels     :: [ens x year]
"""     
# ------------------------------------------------------------
# %% Looping for runid
# ------------------------------------------------------------

# Print Message
print("Running [train_NN_CESM1.py] with the following settings:")
print("\tNetwork Type   : "+ eparams['netname'])
print("\tPredictor(s)   : "+str(varnames))
print("\tLeadtimes      : %i to %i" % (leads[0],leads[-1]))
print("\tRunids         : %i to %i" % (runids[0],runids[-1]))
print("\tMax Epochs     : " + str(eparams['max_epochs']))
print("\tEarly Stop     : " + str(eparams['early_stop']))
print("\t# Ens. Members : "+ str(ens))
print("\tDetrend        : "+ str(eparams['detrend']))

for v,varname in enumerate(varnames): 
    
    predictors = data[[v],...]
    
    for nr,runid in enumerate(runids):
        rt = time.time()
        
        # Save data (ex: Ann2deg_NAT_CNN2_nepoch5_nens_40_lead24 )
        expname = ("AMVClass%i_%s_nepoch%02i_" \
                   "nens%02i_maxlead%02i_"\
                   "detrend%i_run%02i_"\
                   "quant%i_res%s" % (nclasses,eparams['netname'],eparams['max_epochs'],
                                         ens,leads[-1],eparams['detrend'],runid,
                                         eparams['quantile'],eparams['regrid']))
        
        # Preallocate Evaluation Metrics...
        train_loss_grid = [] #np.zeros((max_epochs,nlead))
        test_loss_grid  = [] #np.zeros((max_epochs,nlead))
        val_loss_grid   = [] 
        
        train_acc_grid  = []
        test_acc_grid   = [] # This is total_acc
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
            X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=ens,tstep=ntime)
            
            # ----------------------
            # Select samples
            # ----------------------
            if eparams['nsamples'] is None: # Default: nsamples = smallest class
                
                threscount = np.zeros(nclasses)
                for t in range(nclasses):
                    threscount[t] = len(np.where(y_class==t)[0])
                nsamples = int(np.min(threscount))
                print("Using %i samples, the size of the smallest class" % (nsamples))
            y_class,X,shuffidx = am.select_samples(nsamples,y_class,X,verbose=debug)
            lead_nsamples      = y_class.shape[0]
            sampled_idx.append(shuffidx) # Save the sample indices
            
            # --------------------------
            # Flatten input data for FNN
            # --------------------------
            if "FNN" in eparams['netname']:
                ndat,nchan,nlat,nlon = X.shape
                inputsize            = nchan*nlat*nlon
                outsize              = nclasses
                X                    = X.reshape(ndat,inputsize)
            
            # --------------------------
            # Train Test Split
            # --------------------------
            X_subsets,y_subsets      = am.train_test_split(X,y_class,eparams['percent_train'],
                                                           percent_val=eparams['percent_val'],
                                                           debug=debug,offset=eparams['cv_offset'])
            # Convert to Tensors
            X_subsets = [torch.from_numpy(X.astype(np.float32)) for X in X_subsets]
            y_subsets = [torch.from_numpy(y.astype(np.compat.long)) for y in y_subsets]
            
            
            # # Put into pytorch dataloaders
            data_loaders = [DataLoader(TensorDataset(X_subsets[iset],y_subsets[iset]), batch_size=eparams['batch_size']) for iset in range(len(X_subsets))]
            train_loader,test_loader,val_loader = data_loaders
            
            # ---------------
            # Train the model
            # ---------------
            nn_params = pparams.nn_param_dict[eparams['netname']] # Get corresponding param dict for network
            
            # Initialize model
            if "FNN" in eparams['netname']:
                layers = am.build_FNN_simple(inputsize,outsize,nn_params['nlayers'],nn_params['nunits'],nn_params['activations'],
                                          dropout=nn_params['dropout'],use_softmax=eparams['use_softmax'])
                pmodel = nn.Sequential(*layers)
                
            else:
                # Note: Currently not supported due to issues with timm model. Need to rewrite later...
                pmodel = am.transfer_model(eparams['netname'],nclasses,cnndropout=nn_params['cnndropout'],unfreeze_all=eparams['unfreeze_all'],
                                        nlat=nlat,nlon=nlon,nchannels=nchannels)
            # Train/Validate Model
            model,trainloss,testloss,valloss,trainacc,testacc,valacc = am.train_ResNet(pmodel,eparams['loss_fn'],eparams['opt'],
                                                                                       train_loader,test_loader,val_loader,
                                                                                       eparams['max_epochs'],early_stop=eparams['early_stop'],
                                                                                       verbose=debug,reduceLR=eparams['reduceLR'],
                                                                                       LRpatience=eparams['LRpatience'],checkgpu=checkgpu)
            
            # Save train/validation loss
            train_loss_grid.append(trainloss)
            val_loss_grid.append(valloss)
            test_loss_grid.append(testloss)
            train_acc_grid.append(trainacc)
            val_acc_grid.append(valacc)
            test_acc_grid.append(testacc)
            
            # --------------------------------------------------
            # Test the model separately to get accuracy by class
            # --------------------------------------------------
            y_predicted,y_actual,test_loss = am.test_model(model,test_loader,eparams['loss_fn'],
                                                           checkgpu=checkgpu,debug=False)
            lead_acc,class_acc = am.compute_class_acc(y_predicted,y_actual,nclasses,debug=True,verbose=False)
            
            
            yvalpred.append(y_predicted)
            yvallabels.append(y_actual)
            acc_by_class.append(class_acc)
            total_acc.append(lead_acc)
            
            # --------------
            # Save the model
            # --------------
            if savemodel:
                modout = "../../CESM_data/%s/Models/%s_%s_lead%02i_classify.pt" %(expdir,expname,varname,lead)
                torch.save(model.state_dict(),modout)
            
            print("\nCompleted training for %s lead %i of %i" % (varname,lead,leads[-1]))
            
            # Clear some memory
            del model
            torch.cuda.empty_cache()  # Save some memory
            
            # -----------------
            # Save Eval Metrics
            # -----------------
            savename = "../../CESM_data/"+expdir+"/"+"Metrics"+outname
            np.savez(savename,**{
                     'train_loss'     : train_loss_grid,
                     'test_loss'      : test_loss_grid,
                     'train_acc'      : train_acc_grid,
                     'test_acc'       : test_acc_grid,
                     'total_acc'      : total_acc,
                     'acc_by_class'   : acc_by_class,
                     'yvalpred'       : yvalpred,
                     'yvallabels'     : yvallabels,
                     'sampled_idx'    : sampled_idx,
                     'thresholds_all' : thresholds_all
                     }
                     )
            #print("Saved data to %s%s. Finished variable %s in %ss"%(outpath,outname,varname,time.time()-start))
        #print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))
             


