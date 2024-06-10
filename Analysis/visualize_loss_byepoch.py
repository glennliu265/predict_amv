#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Epochs and Loss during Training

Copied upper setion of "compute_test_metrics.py" On 31 Oct 2023

Created on Tue Oct 31 13:21:03 2023

@author: gliu
"""

import numpy as np
import sys
import glob

import xarray as xr

import torch
from torch import nn

from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt

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


# Processing Options
even_sample         = False
standardize_input   = False # Set to True to standardize variance at each point
calc_lrp            = True # Set to True to calculate relevance composites

# Get some paths
datpath             = pparams.datpath
dataset_name        = "CESM1"

figpath             = pparams.figpath
proc.makedir(figpath)

# Set some looping parameters and toggles
varnames            = ["SSH","SST","SLP","SSS","NHFLX"]       # Names of predictor variables
varcolors           = ["dodgerblue","r","gold","violet","cyan"]
leads               = np.arange(0,26,1)    # Prediction Leadtimes
runids              = np.arange(0,100,1)    # Which runs to do

# LRP Parameters
innexp         = 2
innmethod      ='b-rule'
innbeta        = 0.1
innepsilon     = 1e-2

# Other toggles
save_all_relevances = False                # True to save all relevances (~33G per file...)
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

# Get necessary sizes
nchannels,nens,ntime,nlat,nlon = data.shape             
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN
nclasses                       = len(eparams['thresholds'])+1
nlead                          = len(leads)

# Count Samples...
am.count_samples(None,target_class)

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
        

#%% 

"""

General Procedure


Items in Metrics File

['train_loss',
 'test_loss',
 'val_loss',
 'train_acc',
 'test_acc',
 'val_acc',
 'total_acc',
 'acc_by_class',
 'yvalpred',
 'yvallabels',
 'sampled_idx',
 'thresholds_all',
 'exp_params',
 'sample_sizes']

"""


outvar_names = ['train_loss','test_loss']

nvars        = len(varnames)
flists_var   = {}

outvars_var       = {}

for v in range(nvars):
    vt      = time.time()
    varname = varnames[v]
    
    # ================================
    #% 1. Load Metrics
    # ================================
    # Get list of metric files
    search = "%s%s/Metrics/%s" % (datpath,expdir,"*%s*" % varname)
    flist  = glob.glob(search)
    flist  = [f for f in flist if "of" not in f]
    flist.sort()
    print("Found %i files per lead for %s using searchstring: %s" % (len(flist),varname,search))
    flists_var[varname] = flist.copy()
    
    # -----------------------------
    # 2. Retrieve Selected Variables
    # -----------------------------
    outvars_all = {}
    for ov in range(len(outvar_names)):
        ovname = outvar_names[ov]
        outvars_f = []
        for f in range(len(flist)):
            outvar = np.load(flist[f],allow_pickle=True)[ovname]
            outvars_f.append(outvar)
        outvars_all[ovname] = np.array(outvars_f) # Files x Lead x Epoch
    outvars_var[varname] = outvars_all.copy()
    
#%% Visualize the loss by epoch plots...

ovname  = 'train_loss'
varname = "NHFLX"

losses  = outvars_var[varname][ovname] # [Run, Lead, Epoch]
nruns   = losses.shape[0]
nepochs = losses.shape[2]
xepochs = np.arange(1,nepochs+1)

# Do a test plot
plotleads = [0,5,15,25]
fig,axs = plt.subplots(4,1,figsize=(8,10),constrained_layout=True,sharex=True)
for a in range(4):
    ax = axs[a]
    
    plotlosses = losses[:,plotleads[a],:] # [Run x Epoch]
    for nm in range(nruns):
        ax.plot(xepochs,plotlosses[nm,:],alpha=0.25)
    ax.set_title("Lead = %i" % (plotleads[a]))
    
    if a == 3:
        ax.set_xlabel("Epoch")
    ax.grid(True,ls='dashed')
    if a == 0:
        ax.set_ylabel("Training Loss")
    
plt.suptitle("Loss by Epoch, Predictor = %s" % (varname))

#%%

def find_threscross(invar,thres,axis):
    # in var = [epochs, runs] where axis = dim of epochs
    # thres = threshold to check
    nmax               = invar.shape[axis]
    kcross             = np.argmax(invar<thres,axis=axis)
    id_pastmax         = kcross == 0
    kcross[id_pastmax] = nmax - 1 
    return kcross


def check_early_loss(invar,debug=True):
    # invar = [epochs,]
    invar_out  = invar.copy()
    nepochs    = len(invar)
    diff       = invar[1:] - invar[:(nepochs-1)] # [pos for increasing loss]
    losscount  = 0
    stop_epoch = nepochs-1
    incr_epochs = []
    for e in range(nepochs-1):
        if diff[e] > 0:
            losscount += 1 
            if debug:
                incr_epochs.append(e)
                print("Loss Increased at epoch %i, count=%i" % (e,losscount))
        else:
            losscount = 0 
        if losscount > 5:
            invar_out[e:] = 0 # Zero Out
            stop_epoch = e 
    return invar_out,stop_epoch,incr_epochs

#%% Check when it passes a threshold
#nruns = 20

thres      = 1e-3
irun       = 9
ilead      = 25
varname    = "SLP"

plotlosses = outvars_var[varname]['train_loss'][:,ilead,:] # Run x Lead x Epochs

for irun in range(nruns):
    fig,ax = plt.subplots(1,1,figsize=(8,4),constrained_layout=True)
    # Find Index 
    #kend            = np.argmax(plotlosses<thres,axis=1) # Indices before threshold
    #id_past50       = kend == 0                          # Which ones were past 50
    kend            = find_threscross(plotlosses,thres,1) # Changed to function version
    num_earlystop   = nepochs - (kend == nepochs-1).sum()
    
    # Apply early stopping
    inloss = plotlosses[irun,:]
    inloss_out,stop_epoch,incr_epochs=check_early_loss(inloss)
    
    ax.set_title("Run = %i (Predictor %s, Lead %s)" % (irun,varname,leads[ilead]))
    ax.plot(xepochs,inloss,alpha=1,color="k",marker=".",markersize=5)
    
    ax.scatter(xepochs[incr_epochs],inloss[incr_epochs],color='red',label="Increasing",zorder=9,alpha=0.5)
    
    ax.axvline([xepochs[kend[irun]]],label="Below Threshold %.2e" % (thres))
    ax.axvline([xepochs[stop_epoch]],label="Early Stop",color='darkred')
    ax.legend()
    
    figname = "%sLossByEpoch_%s_lead%02i_run%03i.png" % (figpath,varname,leads[ilead],irun+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    # for nm in range(nruns):
    #     ax.plot(xepochs,plotlosses[nm,:],alpha=.25)

#%% Check Percentage of Runs Past a Threshold
nleads        = len(leads)
thres         = 1e-3
earlystop_num = np.zeros([nvars,nleads])
for v in range(nvars):
    varname = varnames[v]
    for l in range(nleads):
        losses_in = outvars_var[varname]['train_loss'][:,l,:] # [Run Lead Epoch]
        kend            = find_threscross(losses_in,thres,1) # Changed to function version
        nearlystop      = nruns - (kend == nepochs-1).sum()
        earlystop_num[v,l] = nearlystop.copy()

#%% Visualize Leadtime x Percentage for each variable

nruns = len(runids)
fig,ax = plt.subplots(1,1,figsize=(10,4),constrained_layout=True)

for v in range(5):
    ax.plot(leads,earlystop_num[v,:]/nruns,label=varnames[v],
            c=pparams.varcolors[v])
ax.legend()
ax.set_xlabel("Prediction Leadtime (Years)")
ax.set_ylabel("Percentage of Networks Below Threshold")
ax.set_title("Percentage of Networks\nBelow Threshold %.1e at 50 Epochs" % thres)


