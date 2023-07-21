#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Events based analysis

Created on Sat Jul  1 14:06:45 2023
Copied beginning of [compute_test_metrics.py] on 2023.07.19

- Load in data, model weights
- Examine accuracy by event
- Select an event and examine heatmaps, composites, etc.

@author: gliu

"""

import numpy as np
import sys
import glob

import xarray as xr

import torch
from torch import nn

import matplotlib.pyplot as plt

from tqdm import tqdm
import time
import os

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
from amv import proc,viz

# Import LRP package
lrp_path = pparams.machine_paths[machine]['lrp_path']
sys.path.append(lrp_path)
from innvestigator import InnvestigateModel

# Load ML architecture information
nn_param_dict      = pparams.nn_param_dict

# ============================================================
#%% User Edits vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ============================================================

# Set machine and import corresponding paths

# Set experiment directory/key used to retrieve params from [train_cesm_params.py]
expdir              = "FNN4_128_SingleVar_PaperRun"
eparams             = train_cesm_params.train_params_all[expdir] # Load experiment parameters
figpath             = pparams.figpath

# Processing Options
even_sample         = False # Note this does not support if even_sample is True due to shuffling that occurs during that case...
standardize_input   = False # Set to True to standardize variance at each point

# Get some paths
datpath             = pparams.datpath
dataset_name        = "CESM1"

# Set some looping parameters and toggles
varnames            = ["SST","SSH","SSS","SLP","NHFLX","TAUX","TAUY"]       # Names of predictor variables
leads               = np.arange(0,26,1)    # Prediction Leadtimes
runids              = np.arange(0,100,1)    # Which runs to do

# LRP Parameters
calc_lrp       = False # Set to True to calculate relevance composites
innexp         = 2
innmethod      ='b-rule'
innbeta        = 0.1
innepsilon     = 1e-2

# Other toggles
save_all_relevances = True                # True to save all relevances (~33G per file...)
checkgpu            = True                 # Set to true to check if GPU is availabl
debug               = False                 # Set verbose outputs
savemodel           = True                 # Set to true to save model weights

# Save looping parameters into parameter dictionary
eparams['varnames'] = varnames
eparams['leads']    = leads
eparams['runids']   = runids

# -----------------------------------
# %% Get some other needed parameters
# -----------------------------------

# Ensemble members
ens_all        = np.arange(0,42)
ens_train_val  = ens_all[:eparams['ens']]
ens_test       = ens_all[eparams['ens']:]
nens_test      = len(ens_test)

# ============================================================
#%% Load the data 
# ============================================================
# Copied segment from train_NN_CESM1.py

# Load data + target
load_dict                      = am.prepare_predictors_target(varnames,eparams,return_nfactors=True,load_all_ens=False,return_test_set=True)
data                           = load_dict['data']
target_class                   = load_dict['target_class']

# Pick just the testing set
data                           = load_dict['data_test']#data[:,ens_test,...]
target_class                   = load_dict['target_class_test']#target_class[ens_test,:]
target                         = load_dict['target_test']

# Get necessary sizes
nchannels,nens,ntime,nlat,nlon = data.shape             
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN
nclasses                       = len(eparams['thresholds'])+1
nlead                          = len(leads)

# Count Samples...
am.count_samples(None,target_class)

# Get ens year numbers
yrs,ensnums = am.make_ensyr()
yrs     = yrs[ens_test,:]
ensnums = ensnums[ens_test,:] 
# --------------------------------------------------------
#%% Option to standardize input to test effect of variance
# --------------------------------------------------------

if standardize_input:
    # Compute standardizing factor (and save)
    std_vars = np.std(data,(1,2)) # [variable x lat x lon]
    for v in range(nchannels):
        savename = "%s%s/Metrics/%s_standardizing_factor_ens%02ito%02i.npy" % (datpath,expdir,varnames[v],ens_test[0],ens_test[-1])
        np.save(savename,std_vars[v,:,:])
    # Apply standardization
    data = data / std_vars[:,None,None,:,:] 
    data[np.isnan(data)] = 0
    std_vars_after = np.std(data,(1,2))
    check =  np.all(np.nanmax(np.abs(std_vars_after)) < 2)
    assert check, "Standardized values are not below 2!"

# --------------------------------------------------------
#%% Get computed test accuracy
# --------------------------------------------------------

flist,npz_list = dl.load_test_accuracy(expdir,varnames,evensample=even_sample)

# --------------------------------------------------------
#%% Compute the event-wise accuracy and count
# --------------------------------------------------------


nvars   = len(varnames)
nleads  = len(leads) 
nmodels = len(runids)


acc_by_event   = np.zeros((nvars,nleads,nens,ntime)) * np.nan          # [var x lead x ens x time], Accuracy for each event across 100 networks
count_by_event = np.zeros((nvars,nleads,nclasses,nens,ntime)) * np.nan # [var x lead x class x ens x time], Prediction count for each event
preds_by_event = np.zeros((nvars,nleads,nmodels,nens,ntime)) * np.nan  # [var x lead x model x ens x time], Actual Predictions for each event


for v in range(nvars):
    
    varname = varnames[v]
    npz     = npz_list[v]
    
    for l in tqdm(range(nleads)):
        
        lead  = leads[l]
        
        # Retrieve predictions
        preds = npz['predictions'][l,:] # [lead x model][sample]
        preds = np.array([pred.astype(int) for pred in preds]) # [model x sample]
        
        # Retrieve labels
        labs  = npz['targets'][l] # [lead][sample]
        labs  = np.array(labs).astype(int) # [sample]
        
        # Double check with the loaded target_class
        labs_targ = target_class[:,lead:]
        assert np.all(labs_targ.flatten() == labs),"Label classes do not resemble the loaded target classes!"
        
        # Try reshaping and check again
        labs_rs = labs.reshape(nens,ntime-lead) # [ens x year]
        assert np.all(labs_targ == labs_rs),"Reshaped values do not resemble the target values!"
        
        # Reshape the predictors and check against the computed total acccuracy
        preds   = preds.reshape(nmodels,nens,ntime-lead) # [model x ens x year]
        correct = np.array(preds == labs_targ[None,:,:]) # Check that the test accuracy matches
        testacc = np.sum(correct,(1,2)) /  labs.shape[0]
        assert np.all(npz['total_acc'][:,l] == testacc),"Computed test accuracy does not resemble the loaded test accuracy!"
        
        # Now compute the accuracy by event summing along the network/model dimension
        percent_correct_byevent   = correct.sum(0)/nmodels # [ens x year]
        acc_by_event[v,l,:,lead:] = percent_correct_byevent.copy()
        preds_by_event[v,l,:,:,lead:] = preds.copy()
        
        # Get the count by class
        for c in range(nclasses):
            class_preds = (preds == c)
            class_preds = class_preds.sum(0) # [ens x year]
            count_by_event[v,l,c,:,lead:] = class_preds.copy()
            # <End Class Loop>
        # <End Lead Loop>
    # <End Predictor Loop>

#%% Look at the accuracy of events versus the actual amv target value (based on similar analysis from viz_LRP_predictor.py)

varcolors   = pparams.varcolors
varmarker   = pparams.varmarker
fig,ax      = plt.subplots(1,1,figsize=(10,8),constrained_layout=True)
for v in range(nvars):
    acc_in =np.nanmean(acc_by_event[v,...],(0)) # Average across predictors and leadtimes
    sc = ax.scatter(np.abs(target.flatten()),acc_in.flatten(),c=varcolors[v]
                    ,alpha=0.4,
                    label=varnames[v],marker=varmarker[v])
ax.axvline([0.37],color='k',ls='dashed',label="Class Threshold = 0.37")
ax.legend()
ax.set_xlabel("abs(NASST Index Value)")
ax.set_ylabel("Mean Test Accuracy")
ax.set_aspect('equal', adjustable='box')
ax      = viz.add_ticks(ax)
figname = "%s%s_TestAcc_v_NASSTValue_ByPredictor.png" % (figpath,expdir)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Look at 1 leadtime averaged across variables, and separate by class

l            = 25
for l in range(26):
    class_colors = pparams.class_colors
    classes      = pparams.classes
    
    varcolors    = pparams.varcolors
    fig,ax       = plt.subplots(1,1,figsize=(16,8),constrained_layout=True)
    for c in range(3):
        acc_in =np.nanmean(acc_by_event[:,l,...],(0)) # Average across predictors and leadtimes
        
        idx_class = target_class.flatten() == c
        
        plotx = np.abs(target.flatten())
        ploty = acc_in.flatten()
        sc    = ax.scatter(plotx[idx_class],ploty[idx_class],c=class_colors[c],
                           label=classes[c],alpha=0.8,s=55)
        
        if c != 1:
            cids   =np.where(idx_class)[0]
            for ii in range(len(cids)):
                cid = cids[ii]
                
                txt = "(%s,%s)" % (1+ensnums.flatten()[cid],1920+yrs.flatten()[cid])
                ax.annotate(txt,
                            (plotx[cid],ploty[cid]),
                            fontsize=8)
            
    
    ax.axvline([0.37],color='k',ls='dashed',label="Class Threshold = 0.37")
    ax.legend()
    ax.set_xlabel("abs(NASST Index Value)")
    ax.set_ylabel("Mean Test Accuracy")
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Prediction Leadtime: %i-years"%leads[l])
    
    figname = "%s%s_TestAcc_v_NASSTValue_lead%i.png" % (figpath,expdir,leads[l])
    plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Look at the mean predictability of each event across leadtimes

acc_byevent_varmean = acc_by_event.mean(0).reshape(nleads,nens*ntime)

fig,ax = plt.subplots(1,1)
for s in range(ntime*nens):
    ax.plot(leads,acc_byevent_varmean[:,s],alpha=0.1)
ax.plot(leads,np.nanmean(acc_byevent_varmean[:,:],1),color="k")

idmax = np.argmax(np.nansum(acc_byevent_varmean,0))
idmin = np.argmin(np.nansum(acc_byevent_varmean,0))
ax.plot(leads,acc_byevent_varmean[:,idmax],color="r")
ax.plot(leads,acc_byevent_varmean[:,idmin],color="b") # Doesn't show up due to mostly being NaN values 

# Get Indices of Event with largest cumulative accuracy across leadtimes
ensmax,ymax = np.unravel_index(idmax,(nens,ntime))


#%% Make composites of the predictor for that event



predictors = data[[0],ensmax,ymax,:,:,:] # Predictor x Year x Lat x Lon
