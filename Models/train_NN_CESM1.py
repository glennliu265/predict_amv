#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train Neural Networks (NN) for CESM1 Large Ensemble Simulations

 - Copied introductory section from NN_Training_Rewrite.py on 2023.03.20
 - Based on NN_test_lead_ann_ImageNet_classification_singlevar.py

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

#%% Load custom packages and setup parameters

machine = 'Astraeus' # Indicate machine (see module packages section in pparams)

# Import packages specific to predict_amv
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import train_cesm_params as train_cesm_params
import amv_dataloader as dl
import amvmod as am

# Load Predictor Information
bbox          = pparams.bbox

# Import general utilities from amv module
pkgpath = pparams.machine_paths[machine]['amv_path']
sys.path.append(pkgpath)
from amv import proc

# ============================================================
#%% User Edits vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ============================================================

# Set machine and import corresponding paths

# Set experiment directory/key used to retrieve params from [train_cesm_params.py]
expdir             = "FNN4_128_SingleVar_Rerun100"
eparams            = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Set some looping parameters and toggles
varnames           = ["PSL","SSS",]       # Names of predictor variables
leads              = np.arange(0,26,1)    # Prediction Leadtimes
runids             = np.arange(0,100,1)    # Which runs to do

# Other toggles
checkgpu           = True                 # Set to true to check if GPU is availabl
debug              = True                 # Set verbose outputs
savemodel          = True                 # Set to true to save model weights

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

# ---------------------------------------------
# %% If option is set, presample for everything
# ---------------------------------------------

if eparams["shuffle_trainset"] is True:
    # am.consistent_sample(data,target_class,leads,nsamples,leadmax=None,
    #                       nens=None,ntime=None,shuffle_class=shuffle_class,debug=False)
    
# leadmax      = leads.max()
# X,y_class    = am.apply_lead(data[[0],...],target_class,leadmax,reshape=True,ens=ens,tstep=ntime)

# if eparams['nsamples'] is None: # Default: nsamples = smallest class
#     threscount = np.zeros(nclasses)
#     for t in range(nclasses):
#         threscount[t] = len(np.where(y_class==t)[0])
#     eparams['nsamples'] = int(np.min(threscount))
#     print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))

# # Select samples based on the longest leadtime. 
# y_class,X,shuffidx_max = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle_class'])
# shuffidx_max           = shuffidx_max.astype(int) # There indices are w.r.t. the lagged data

# # Get the absolute indices
# apply_lead = True # Lead is applied to current
# ref_lead   = True # Lead is applied to reference

# # Get [absolute] linear indices for reference lead [lead], based on applied lead [leadmax]
# shuffidx_target_abs,tref = am.get_ensyr_linear(leadmax,shuffidx_max,
#             reflead=0,nens=nens,nyr=ntime,
#             apply_lead=True,ref_lead=True,
#             return_labels=True,debug=True)

# # Convert to numpy array of [sample x 2] where 0 = ens, 1 = yr
# tref              = np.array([[a[0],a[1]] for a in tref],dtype='int')

# # Get indices for predictors
# predictor_indices = []
# predictor_refids  = []
# for l,lead in enumerate(leads):
#     # Get the references 
#     pref      = np.array([[a[0],a[1]-lead] for a in tref],dtype="int")
#     if debug:
#         plt.hist(pref[:,1]),plt.title("lead %i (predictor years %i to %i)"% (lead,pref[:,1].min(),pref[:,1].max())),plt.show()
        
#     target_linearids = am.select_ensyr_linearids(pref,target_lead=0,lag=False,nens=nens,
#                                                  nyr=ntime,)
#     predictor_indices.append(target_linearids)
#     predictor_refids.append(pref)
    
# ii = 22
# for l in range(nlead):
#     print("Lead %02i, target is (e=%02i,y=%02i, idx=%i), predictor is (e=%02i,y=%02i, idx%i)" % (leads[l],
#                                                                         tref[ii,0],tref[ii,1],shuffidx_target_abs[ii],
#                                                                         predictor_refids[l][ii,0],predictor_refids[l][ii,1],
#                                                                         predictor_indices[l][ii]))
"""
Output

shuffidx_target  = [nsamples*nclasses,]        - Indices of target
predictor_refids = [nlead][nsamples*nclasses,] - Indices of predictor at each leadtime

tref --> array of the target years
predictor_refids --> array of the predictor refids
"""


# shuffidx_all_target    = []
# shuffidx_all_predictor = []
# target_refs            = []
# predictor_refs         = []
# for l,lead in enumerate(leads):
#     print(lead)
    
#     reflead    = 0
#     apply_lead = True # Lead is applied to current
#     ref_lead   = True # Lead is applied to reference

#     # Get linear indices for reference lead [lead], based on applied lead [leadmax]
#     shuffidx_target,tref = am.get_ensyr_linear(leadmax,shuffidx_max,
#                   reflead=lead,nens=nens,nyr=ntime,
#                   apply_lead=True,ref_lead=True,
#                   return_labels=True,debug=False)
    
#     shuffidx_predictor,pref = am.get_ensyr_linear(leadmax,shuffidx_max,
#                   reflead=lead,nens=nens,nyr=ntime,
#                   apply_lead=True,ref_lead=False,
#                   return_labels=True,debug=False)
    
#     shuffidx_all_target.append(shuffidx_target)
#     shuffidx_all_predictor.append(shuffidx_predictor)
#     target_refs.append(tref)
#     predictor_refs.append(pref)
    


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
            
            # --------------------------
            # 08. Apply lead/lag to data
            # --------------------------
            # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
            X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=ens,tstep=ntime)
            
            # ----------------------
            # 09. Select samples
            # ----------------------
            if eparams['shuffle_trainsplit'] is False:
                if eparams['nsamples'] is None: # Default: nsamples = smallest class
                    threscount = np.zeros(nclasses)
                    for t in range(nclasses):
                        threscount[t] = len(np.where(y_class==t)[0])
                    eparams['nsamples'] = int(np.min(threscount))
                    print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
                y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle_class'])
            else:
                print("Select the sample samples")
                shuffidx = sampled_idx[l-1]
                y_class  = y_class[shuffidx,...]
                X        = X[shuffidx,...]
                am.count_samples(eparams['nsamples'],y_class)
            # # --------------------------------------------------------------------------------
            # # Steps 10-12 (Split Data, Train/Test/Validate Model, Calculate Accuracy by Class)
            # # --------------------------------------------------------------------------------
            # output = am.train_NN_lead(X,y_class,eparams,pparams,debug=debug,checkgpu=checkgpu)
            # model,trainloss,valloss,testloss,trainacc,valacc,testacc,y_predicted,y_actual,class_acc,lead_acc = output
            
            # # Append outputs for the leadtime
            # train_loss_grid.append(trainloss)
            # val_loss_grid.append(valloss)
            # test_loss_grid.append(testloss)
            
            # train_acc_grid.append(trainacc)
            # val_acc_grid.append(valacc)
            # test_acc_grid.append(testacc)
            
            # acc_by_class.append(class_acc)
            # total_acc.append(lead_acc)
            # yvalpred.append(y_predicted)
            # yvallabels.append(y_actual)
            sampled_idx.append(shuffidx.astype(int)) # Save the sample indices
            # sample_sizes.append(eparams['nsamples'])
            
            # # ------------------------------
            # # 13. Save the model and metrics
            # # ------------------------------
            # if savemodel:
            #     modout = "../../CESM_data/%s/Models/%s_%s_lead%02i_classify.pt" %(expdir,expname,varname,lead)
            #     torch.save(model.state_dict(),modout)
            
            # # Save Metrics
            # savename = "../../CESM_data/"+expdir+"/"+"Metrics"+outname
            # np.savez(savename,**{
            #          'train_loss'     : train_loss_grid,
            #          'test_loss'      : test_loss_grid,
            #          'train_acc'      : train_acc_grid,
            #          'test_acc'       : test_acc_grid,
            #          'total_acc'      : total_acc,
            #          'acc_by_class'   : acc_by_class,
            #          'yvalpred'       : yvalpred,
            #          'yvallabels'     : yvallabels,
            #          'sampled_idx'    : sampled_idx,
            #          'thresholds_all' : thresholds_all,
            #          'exp_params'     : eparams,
            #          'sample_sizes'   : sample_sizes,
            #          }
            #          )
            
            # # Clear some memory
            # del model
            # torch.cuda.empty_cache()  # Save some memory
            
            print("\nCompleted training for %s lead %i of %i" % (varname,lead,leads[-1]))
            # End Lead Loop >>>
        print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
        # End Runid Loop >>>
    print("\nPredictor %s finished in %.2fs" % (varname,time.time()-vt))
    # End Predictor Loop >>>
print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))
             


