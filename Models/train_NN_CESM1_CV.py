#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train Neural Networks (NN) for CESM1, testing cross validation

 - Copied train_NN_CESM1.py.

Current Structure:
    - Indicate CESM1 training parameters in [train_cesm_parameters.py]
    - Functions are mostly contained in [amvmod.py]
    - Universal Variables + Architectures are in [predict_amv_params.py]
    - Additional helper function from [amv] module [proc] and [viz]

Updated Procedure:
    01) Create Experiment Directory
    02) Load Data
    03) Determine (and make) AMV Classes based on selected thresholds
    04) Loop by Predictor...
        05) Loop by runid (train [nr] networks)...
            06) Preallocate variables and set experiment output name
            07) Loop by Leadtime...
                08) Apply Lead/Lag to predictors+target
                09) Select N samples from each class
                ---- moved to function amvmod.train_NN_lead (10-12)
                10) Perform Train/Test Split, place into dataloaders
                11) Initialize and train the model
                12) Test the model, compute accuracy by class
                ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- -
                13) Save the model and output

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

#% Load custom packages and setup parameters
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

# ============================================================
#%% User Edits vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ============================================================

# Set experiment directory/key used to retrieve params from [train_cesm_params.py]
expdir             = "FNN4_128_SingleVar_CV_consistent"
eparams            = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Set some looping parameters and toggles
varnames           = ["SST","SSH",]#"SSS","PSL"]       # Names of predictor variables
leads              = np.arange(0,26,3)    # Prediction Leadtimes
runids             = np.arange(0,21,1)    # Which runs to do

# Other toggles
checkgpu           = True                 # Set to true to check if GPU is availabl
debug              = False                 # Set verbose outputs
savemodel          = True                 # Set to true to save model weights

# CV Options
eparams['cv_loop']   = True    # Repeat for cross-validation
percent_test         = 1-(eparams['percent_train']+eparams['percent_val'])
eparams['cv_offset'] = percent_test      # Set cv option. Default is test size chunk


# Save looping parameters into parameter dictionary
eparams['varnames'] = varnames
eparams['leads']    = leads
eparams['runids']   = runids

# ============================================================
# End User Edits ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ============================================================
# ------------------------------------------------------------
# %% 01. Check for existence of experiment directory and create it
# ------------------------------------------------------------
allstart = time.time()

proc.makedir("../../CESM_data/"+expdir)
for fn in ("Metrics","Models","Figures"):
    proc.makedir("../../CESM_data/"+expdir+"/"+fn)
    
# Check if there is gpu
if checkgpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

# ----------------------------------------------
#%% 02. Data Loading...
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
# %% 03. Determine the AMV Classes
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


#
# %% Retrieve a consistent sample if the option is set
#


if eparams["shuffle_trainsplit"] is False:
    print("Pre-selecting indices for consistency")
    output_sample=am.consistent_sample(data,target_class,leads,eparams['nsamples'],leadmax=leads.max(),
                          nens=None,ntime=None,
                          shuffle_class=eparams['shuffle_class'],debug=False)
    
    target_indices,target_refids,predictor_indices,predictor_refids = output_sample
else:
    target_indices     = None
    predictor_indices  = None
    target_refids      = None
    predictor_refids   = None
    
    
"""
Output

shuffidx_target  = [nsamples*nclasses,]        - Indices of target
predictor_refids = [nlead][nsamples*nclasses,] - Indices of predictor at each leadtime

tref --> array of the target years
predictor_refids --> array of the predictor refids
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

# ------------------------
# 04. Loop by predictor...
# ------------------------
for v,varname in enumerate(varnames): 
    vt = time.time()
    predictors = data[[v],...] # Get selected predictor
    
    # ---------------------
    # Do Cross Validation Loop
    nfolds = int(np.ceil(1/percent_test))
    for k in range(nfolds):
        ft = time.time()
        
        k_offset = k*percent_test
        eparams["cv_offset"] = k_offset
        print(k_offset)
        
        if debug: # Try to check k-fold splitting
            lead = 0
            l    = 0
            
            if target_indices is None: # Target Indices not consistent
                # --------------------------
                # 08. Apply lead/lag to data
                # --------------------------
                # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
                X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=ens,tstep=ntime)
                
                # ----------------------
                # 09. Select samples
                # ----------------------
                if eparams['nsamples'] is None: # Default: nsamples = smallest class
                    threscount = np.zeros(nclasses)
                    for t in range(nclasses):
                        threscount[t] = len(np.where(y_class==t)[0])
                    eparams['nsamples'] = int(np.min(threscount))
                    print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
               
                y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle_class'])
            else:
                print("Using preselected indices")
                pred_indices = predictor_indices[l]
                nchan        = predictors.shape[0]
                y_class      = target_class.reshape((ntime*nens,1))[target_indices,:]
                X            = predictors.reshape((nchan,nens*ntime,nlat,nlon))[:,pred_indices,:,:]
                X            = X.transpose(1,0,2,3) # [sample x channel x lat x lon]
                shuffidx     = target_indices
            
            # --------------------------
            # 10. Train Test Split
            # --------------------------
            X_subsets,y_subsets      = am.train_test_split(X,y_class,eparams['percent_train'],
                                                           percent_val=eparams['percent_val'],
                                                           debug=True,offset=k_offset)
            #print(y_subsets[1][:10].T)
            print("\nFor fold  k=%i" % k)
            ibc,ccounts = am.count_samples(eparams['nsamples'],y_subsets[1])
            print("")
        
        
            
        
        # --------------------
        # 05. Loop by runid...
        # --------------------
        for nr,runid in enumerate(runids):
            rt = time.time()
            
            # ---------------------------------------
            # 06. Set experiment name and preallocate
            # ---------------------------------------
            # Set experiment save name (ex: Ann2deg_NAT_CNN2_nepoch5_nens_40_lead24 )
            expname = ("AMVClass%i_%s_nepoch%02i_" \
                       "nens%02i_maxlead%02i_"\
                       "detrend%i_run%02i_"\
                       "quant%i_res%s_kfold%02iot%02i" % (nclasses,eparams['netname'],eparams['max_epochs'],
                                             ens,leads[-1],eparams['detrend'],runid,
                                             eparams['quantile'],eparams['regrid'],k,nfolds))
            
    
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
            sample_sizes    = []
            
            # -----------------------
            # 07. Loop by Leadtime...
            # -----------------------
            for l,lead in enumerate(leads):
                
                # Set names for intermediate saving, based on leadtime
                if (lead == leads[-1]) and (len(leads)>1): # Output all files together
                    outname = "/leadtime_testing_%s_%s_ALL.npz" % (varname,expname)
                else: # Output individual lead times while training
                    outname = "/leadtime_testing_%s_%s_lead%02dof%02d.npz" % (varname,expname,lead,leads[-1])
                
                if target_indices is None: # Target Indices not consistent
                    # --------------------------
                    # 08. Apply lead/lag to data
                    # --------------------------
                    # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
                    X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=ens,tstep=ntime)
                    
                    # ----------------------
                    # 09. Select samples
                    # ----------------------
                    if eparams['nsamples'] is None: # Default: nsamples = smallest class
                        threscount = np.zeros(nclasses)
                        for t in range(nclasses):
                            threscount[t] = len(np.where(y_class==t)[0])
                        eparams['nsamples'] = int(np.min(threscount))
                        print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
                   
                    y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle_class'])
                else:
                    print("Using preselected indices")
                    pred_indices = predictor_indices[l]
                    nchan        = predictors.shape[0]
                    y_class      = target_class.reshape((ntime*nens,1))[target_indices,:]
                    X            = predictors.reshape((nchan,nens*ntime,nlat,nlon))[:,pred_indices,:,:]
                    X            = X.transpose(1,0,2,3) # [sample x channel x lat x lon]
                    shuffidx     = target_indices
                    
                # --------------------------------------------------------------------------------
                # Steps 10-12 (Split Data, Train/Test/Validate Model, Calculate Accuracy by Class)
                # --------------------------------------------------------------------------------
                #_=am.train_NN_lead(X,y_class,eparams,pparams,debug=True,checkgpu=checkgpu)
                output = am.train_NN_lead(X,y_class,eparams,pparams,debug=debug,checkgpu=checkgpu)
                model,trainloss,valloss,testloss,trainacc,valacc,testacc,y_predicted,y_actual,class_acc,lead_acc = output
                
                # Append outputs for the leadtime
                train_loss_grid.append(trainloss)
                val_loss_grid.append(valloss)
                test_loss_grid.append(testloss)
                
                train_acc_grid.append(trainacc)
                val_acc_grid.append(valacc)
                test_acc_grid.append(testacc)
                
                acc_by_class.append(class_acc)
                total_acc.append(lead_acc)
                yvalpred.append(y_predicted)
                yvallabels.append(y_actual)
                sampled_idx.append(shuffidx) # Save the sample indices
                sample_sizes.append(eparams['nsamples'])
                
                # ------------------------------
                # 13. Save the model and metrics
                # ------------------------------
                if savemodel:
                    modout = "../../CESM_data/%s/Models/%s_%s_lead%02i_classify.pt" %(expdir,expname,varname,lead)
                    torch.save(model.state_dict(),modout)
                
                # Save Metrics
                if lead == leads[-1]: # Only output the last
                    savename = "../../CESM_data/"+expdir+"/"+"Metrics"+outname
                    np.savez(savename,**{
                             'train_loss'     : train_loss_grid,
                             'test_loss'      : test_loss_grid,
                             'val_loss'       : val_loss_grid,
                             'train_acc'      : train_acc_grid,
                             'val_acc'        : val_acc_grid,
                             'test_acc'       : test_acc_grid,
                             'total_acc'      : total_acc,
                             'acc_by_class'   : acc_by_class,
                             'yvalpred'       : yvalpred,
                             'yvallabels'     : yvallabels,
                             'sampled_idx'    : sampled_idx,
                             'thresholds_all' : thresholds_all,
                             'exp_params'     : eparams,
                             'sample_sizes'   : sample_sizes
                             }
                             )
                
                # Clear some memory
                del model
                torch.cuda.empty_cache()  # Save some memory
                
                print("\nCompleted training for %s lead %i of %i" % (varname,lead,leads[-1]))
                # End Lead Loop >>>
            print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
            # End Runid Loop >>>
        print("\nFold %i of %i finished in %.2fs" % (k,nfolds,time.time()-ft))
        # End k-fold loop >>>
    print("\nPredictor %s finished in %.2fs" % (varname,time.time()-vt))
    # End Predictor Loop >>>
print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))
             


