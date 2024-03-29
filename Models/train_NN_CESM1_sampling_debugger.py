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

machine = 'stormtrack' # Indicate machine (see module packages section in pparams)

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
expdir             = "FNN4_128_SingleVar_Rerun100_consistent"
eparams            = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Set some looping parameters and toggles
varnames           = ["SST","SSS",]       # Names of predictor variables
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

# ----------------------------------------------------
# %% Retrieve a consistent sample if the option is set
# ----------------------------------------------------


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

#
# %%  First, check that we are actually sampling the right classes
#

# Informmation
import matplotlib.pyplot as plt
class_colors=pparams.class_colors


y_class = target_class.reshape((ntime*nens,1))[target_indices,:]
xtks    = np.arange(0,86,20)
xlabs   = xtks + 1920 
xlm     = [0,85]


# fig,axs = plt.subplots(2,3,constrained_layout=True,sharey=True,
#                        figsize=(16,4))

# for ens in range(nens):
    
#     ax = axs.flatten()[ens%6]
#     ax.plot(target[ens,:],label="Ens %02i" % (ens+1))
#     # print(ens%6)

# for a,ax in enumerate(axs.flatten()):
#     ax.legend()
#     ax.set_xticks(xtks)
#     ax.set_xticklabels(xlabs)
#     ax.set_xlim(xlm)

# indicate indices
ens = 0
l   = 4
"""
Inputs: target, thresholds_in, ens, l,

"""

found_ids        = []
found_events     = []
found_predictors = []
# Locate events
for tt,targ in enumerate(target_refids):
    print(targ)
    targ_ens,targ_yr = targ
    if targ_ens == ens: # Record information
        found_ids.append(tt)
        found_events.append(targ)
        found_predictors.append(predictor_refids[l][tt])



# Initialize Plot
fig,ax = plt.subplots(1,1,figsize=(10,3))
ax.legend()
ax.set_xticks(xtks)
ax.set_xticklabels(xlabs)
ax.set_xlim(xlm)
ax.minorticks_on()
ax.grid(True,ls='dotted')
ax.set_xlabel("Year")
ax.set_ylabel("AMV Index ($\degree$C)")


# Plot data and thresholds
ax.plot(target[ens,:],label="Ens %02i" % (ens+1),color='limegreen')
for th in thresholds_in:
    ax.axhline(th,ls="dashed",color="gray",lw=0.75)
ax.axhline(0,ls="solid",color="gray",lw=0.75)
ax.plot(target.mean(0),label="Ens Avg.",color="gray",lw=0.75)
ax.set_title("Selected Events for Ens %02i, Lead=%i Years" % (ens+1,leads[l]))


# Plot the identified events
for ii in found_ids:
    targ_ens,targ_yr = target_refids[ii]
    targ_class       = int(y_class[ii,0])
    pred_ens,pred_yr = predictor_refids[l][ii]
    
    event_info     = (targ_yr,target[targ_ens,targ_yr])
    predictor_info = (pred_yr,target[pred_ens,pred_yr])
    
    ax.plot([predictor_info[0],event_info[0]],
            [predictor_info[1],event_info[1]],color=class_colors[targ_class],
            linestyle="solid",lw=0.8)
    ax.scatter(targ_yr,target[targ_ens,targ_yr],c=class_colors[targ_class])


    
#%%



    
    
# ------------------------------------------------------------
# %% Looping for runid
# ------------------------------------------------------------

v  = 0
nr = 0

varname = varnames[v]
runid   = runids[nr]


# Variable Loop
vt = time.time()
predictors = data[[v],...] # Get selected predictor
    
# Runid Loop
rt = time.time()



nexps         = 2
exp_dicts_all = []
    

for exp in range(nexps):
    # Preallocate Evaluation Metrics...
    # train_loss_grid = [] #np.zeros((max_epochs,nlead))
    # test_loss_grid  = [] #np.zeros((max_epochs,nlead))
    # val_loss_grid   = [] 
    
    # train_acc_grid  = []
    # test_acc_grid   = [] # This is total_acc
    # val_acc_grid    = []
    
    # acc_by_class    = []
    # total_acc       = []
    # yvalpred        = []
    # yvallabels      = []
    # sampled_idx     = []
    # thresholds_all  = []
    # sample_sizes    = []
    
    predictors_lead = []
    targets_lead    = []
    
    # -----------------------
    # 07. Loop by Leadtime...
    # -----------------------
    for l,lead in enumerate(leads):
        
        if target_indices is None:
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
            shuffidx = shuffidx.astype(int)
        else:
            print("Using presesected indices")
            pred_indices = predictor_indices[l]
            nchan        = predictors.shape[0]
            y_class      = target_class.reshape((ntime*nens,1))[target_indices,:]
            X            = predictors.reshape((nchan,nens*ntime,nlat,nlon))[:,pred_indices,:,:]
            X            = X.transpose(1,0,2,3) # [sample x channel x lat x lon]
            shuffidx     = target_indices    
        
        # Get the predictor and target sets for that given leadtime
        predictors_lead.append(X)    # [lead] [sample x 1 x lat x lon]
        targets_lead.append(y_class) # [lead] [sample x 1]
        
        # Count the given leadtimes
        am.count_samples(eparams['nsamples'],y_class,)
        
        
        # # --------------------------------------------------------------------------------
        # # Steps 10-12 (Split Data, Train/Test/Validate Model, Calculate Accuracy by Class)
        # # --------------------------------------------------------------------------------
        output = am.train_NN_lead(X,y_class,eparams,pparams,debug=debug,checkgpu=checkgpu)
        model,trainloss,valloss,testloss,trainacc,valacc,testacc,y_predicted,y_actual,class_acc,lead_acc = output
        
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
        # sampled_idx.append(shuffidx) # Save the sample indices
        # sample_sizes.append(eparams['nsamples'])
        
        # # ------------------------------
        # # 13. Save the model and metrics
        # # ------------------------------
        # if savemodel:
        #     modout = "../../CESM_data/%s/Models/%s_%s_lead%02i_classify.pt" %(expdir,expname,varname,lead)
        #     torch.save(model.state_dict(),modout)
        
        # # Save Metrics
        # savename = "../../CESM_data/"+expdir+"/"+"Metrics"+outname
        # savedict = {
        #     'train_loss'      : train_loss_grid,
        #     'test_loss'       : test_loss_grid,
        #     'val_loss'        : val_loss_grid,
        #     'train_acc'       : train_acc_grid,
        #     'test_acc'        : test_acc_grid,
        #     'val_acc'         : val_acc_grid,
        #     'total_acc'       : total_acc,
        #     'acc_by_class'    : acc_by_class,
        #     'yvalpred'        : yvalpred,
        #     'yvallabels'      : yvallabels,
        #     'sampled_idx'     : sampled_idx,
        #     'thresholds_all'  : thresholds_all,
        #     'exp_params'      : eparams,
        #     'sample_sizes'    : sample_sizes,
        #     'predictors_lead' : predictors_lead,
        #     'targets_lead'    : targets_lead,
            
        #     }
        
        # np.savez(savename,**{
        #           'train_loss'     : train_loss_grid,
        #           'test_loss'      : test_loss_grid,
        #           'val_loss'       : val_loss_grid,
        #           'train_acc'      : train_acc_grid,
        #           'test_acc'       : test_acc_grid,
        #           'val_acc'        : val_acc_grid,
        #           'total_acc'      : total_acc,
        #           'acc_by_class'   : acc_by_class,
        #           'yvalpred'       : yvalpred,
        #           'yvallabels'     : yvallabels,
        #           'sampled_idx'    : sampled_idx,
        #           'thresholds_all' : thresholds_all,
        #           'exp_params'     : eparams,
        #           'sample_sizes'   : sample_sizes,
        #           }
        #           )
        
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
    
    savedict = {
        'predictors_lead' : predictors_lead,
        'targets_lead'    : targets_lead,}
                 


