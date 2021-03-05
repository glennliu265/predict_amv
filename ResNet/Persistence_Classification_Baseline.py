#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Persistence Baseline

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset,Dataset
import os
import copy
import timm

# -------------
#%% User Edits
# -------------

# Data preparation settings
leads          = np.arange(0,25,3)    # Time ahead (in years) to forecast AMV
season         = 'Ann'                # Season to take mean over ['Ann','DJF','MAM',...]
indexregion    = 'NAT'                # One of the following ("SPG","STG","TRO","NAT")
resolution     = '224pix'             # Resolution of dataset ('2deg','224pix')
detrend        = False                # Set to true to use detrended data
usenoise       = False                # Set to true to train the model with pure noise
thresholds     = [-1,1]               # Thresholds (standard deviations, determines number of classes) 
num_classes    = len(thresholds)+1    # Set up number of classes for prediction (current supports)
nsamples       = 300                  # Number of samples for each class

# Training/Testing Subsets
percent_train = 0.8   # Percentage of data to use for training (remaining for testing)
ens           = 40   # Ensemble members to use
tstep         = 86    # Size of time dimension (in years)
numruns       = 1    # Number of times to train each run

# Model training settings
tstep         = 86
outpath       = ''

# Options
debug         = True # Visualize training and testing loss
verbose       = True # Print loss for each epoch
savemodel     = True # Set to true to save model dict.

# -----------
#%% Functions
# -----------

def make_classes(y,thresholds,exact_value=False,reverse=False):
    """
    Makes classes based on given thresholds. 

    Parameters
    ----------
    y : ARRAY
        Labels to classify
    thresholds : ARRAY
        1D Array of thresholds to partition the data
    exact_value: BOOL, optional
        Set to True to use the exact value in thresholds (rather than scaling by
                                                          standard deviation)

    Returns
    -------
    y_class : ARRAY [samples,class]
        Classified samples, where the second dimension contains an integer
        representing each threshold

    """
    nthres = len(thresholds)
    if ~exact_value: # Scale thresholds by standard deviation
        y_std = np.std(y) # Get standard deviation
        thresholds = np.array(thresholds) * y_std
    y_class = np.zeros((y.shape[0],1))
    
    if nthres == 1: # For single threshold cases
        thres = thresholds[0]
        y_class[y<=thres] = 0
        y_class[y>thres] = 1
        
        print("Class 0 Threshold is y <= %.2f " % (thres))
        print("Class 0 Threshold is y > %.2f " % (thres))
        return y_class
    
    for t in range(nthres+1):
        if t < nthres:
            thres = thresholds[t]
        else:
            thres = thresholds[-1]
        
        if reverse: # Assign class 0 to largest values
            tassign = nthres-t
        else:
            tassign = t
        
        if t == 0: # First threshold
            y_class[y<=thres] = tassign
            print("Class %i Threshold is y <= %.2f " % (tassign,thres))
        elif t == nthres: # Last threshold
            y_class[y>thres] = tassign
            print("Class %i Threshold is y > %.2f " % (tassign,thres))
        else: # Intermediate values
            thres0 = thresholds[t-1]
            y_class[(y>thres0) * (y<=thres)] = tassign
            print("Class %i Threshold is %.2f < y <= %.2f " % (tassign,thres0,thres))
    return y_class


def select_samples(nsamples,y_class,X):
    """
    Sample even amounts from each class

    Parameters
    ----------
    nsample : INT
        Number of samples to get from each class
    y_class : ARRAY [samples x 1]
        Labels for each sample
    X : ARRAY [samples x channels x height x width]
        Input data for each sample
    
    Returns
    -------
    
    y_class_sel : ARRAY [samples x 1]
        Subsample of labels with equal amounts for each class
    X_sel : ARRAY [samples x channels x height x width]
        Subsample of inputs with equal amounts for each class
    idx_sel : ARRAY [samples x 1]
        Indices of selected arrays
    
    """
    
    allsamples,nchannels,H,W = X.shape
    classes    = np.unique(y_class)
    nclasses   = len(classes)
    

    # Sort input by classes
    label_by_class  = []
    input_by_class  = []
    idx_by_class    = []
    
    y_class_sel = np.zeros([nsamples*nclasses,1])#[]
    X_sel       = np.zeros([nsamples*nclasses,nchannels,H,W])#[]
    idx_sel     = np.zeros([nsamples*nclasses]) 
    for i in range(nclasses):
        
        # Sort by Class
        inclass = classes[i]
        idx = (y_class==inclass).squeeze()
        sel_label = y_class[idx,:]
        sel_input = X[idx,:,:,:]
        sel_idx = np.where(idx)[0]
        
        label_by_class.append(sel_label)
        input_by_class.append(sel_input)
        idx_by_class.append(sel_idx)
        classcount = sel_input.shape[0]
        print("%i samples found for class %i" % (classcount,inclass))
        
        # Shuffle and select first nsamples
        shuffidx = np.arange(0,classcount,1)
        np.random.shuffle(shuffidx)
        shuffidx = shuffidx[0:nsamples]
        
        # Select Shuffled Indices
        y_class_sel[i*nsamples:(i+1)*nsamples,:] = sel_label[shuffidx,:]
        X_sel[i*nsamples:(i+1)*nsamples,...]     = sel_input[shuffidx,...]
        idx_sel[i*nsamples:(i+1)*nsamples]       = sel_idx[shuffidx]
    
    # Shuffle samples again before output (so they arent organized by class)
    shuffidx = np.arange(0,nsamples*nclasses,1)
    np.random.shuffle(shuffidx)
    
    return y_class_sel[shuffidx,...],X_sel[shuffidx,...],idx_sel[shuffidx,...]

# ----------------------------------------
# %% Set-up
# ----------------------------------------
allstart = time.time()

# Load the data for whole North Atlantic
if usenoise:
    # Make white noise time series
    data   = np.random.normal(0,1,(3,40,tstep,224,224))
    
    ## Load latitude
    #lat = np.linspace(0.4712,64.55497382,224)
    
    # Apply land mask
    dataori   = np.load('../../CESM_data/CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i.npy'%detrend)[:,:40,...]
    data[dataori==0] = 0 # change all ocean points to zero
    target = np.load('../../CESM_data/CESM_label_amv_index_detrend%i.npy'%detrend)
    
    #data[dataori==0] = np.nan
    #target = np.nanmean(((np.cos(np.pi*lat/180))[None,None,:,None] * data[0,:,:,:,:]),(2,3)) 
    #data[np.isnan(data)] = 0
else:
    data   = np.load('../../CESM_data/CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i.npy'%detrend)
    target = np.load('../../CESM_data/CESM_label_amv_index_detrend%i.npy'%detrend)
data   = data[:,0:ens,:,:,:]
target = target[0:ens,:]

# %% Some more user edits
nbefore = 'variable'

# Preallocate
nlead    = len(leads)
channels = 3
start    = time.time()
varname  = 'ALL'
#subtitle = "\n %s = %i; detrend = %s"% (testname,testvalues[i],detrend)
subtitle="\nPersistence Baseline, averaging %s years before" % (str(nbefore))

# Save data (ex: Ann2deg_NAT_CNN2_nepoch5_nens_40_lead24 )
expname = "AMVClass%i_PersistenceBaseline_%sbefore_nens%02i_maxlead%02i_detrend%i_noise%i" % (num_classes,str(nbefore),ens,leads[-1],detrend,usenoise)
outname = "/leadtime_testing_%s_%s_ALL.npz" % (varname,expname)

#%%
# Preallocate Evaluation Metrics...
#train_loss_grid = []#np.zeros((max_epochs,nlead))
#test_loss_grid  = []#np.zeros((max_epochs,nlead))
total_acc       = [] # [lead]
acc_by_class    = [] # [lead x class]
yvalpred        = [] # [lead x ensemble x time]
yvallabels      = [] # [lead x ensemble x time]

nvarflag = False

# -------------
# Print Message
# -------------
print("Calculate Persistence Baseline with the following settings:")
print("\tLeadtimes      : %i to %i" % (leads[0],leads[-1]))
print("\t# Ens. Members : "+ str(ens))
print("\t# Years Before : "+ str(nbefore))
print("\tDetrend        : "+ str(detrend))
print("\tUse Noise      : " + str(usenoise))

for l,lead in enumerate(leads):
    
    
    if nvarflag or nbefore=='variable':
        print("Using Variable nbefore")
        nvarflag=True
        nbefore =lead 
    
    # ----------------------
    # Apply lead/lag to data
    # ----------------------
    y = target[:ens,:].reshape(ens*tstep,1)
    #y_class_pred = target[:ens,:lead].reshape(ens*(tstep-lead),1)
    #X = (data[:,:ens,:tstep-lead,:,:]).reshape(3,ens,(tstep-lead),224,224).transpose(1,0,2,3)
    y_class = make_classes(y,thresholds,reverse=True)
    y_class = y_class.reshape(ens,(tstep)) # Reshape to ens x lead
    
    y_class_predictor = y_class[:,:(tstep-lead)]
    y_class_label     = y_class[:,lead:]
    
    # ----------------------
    # Make predictions
    # ----------------------
    y_pred = np.zeros((ens,tstep-lead))
    classval = [0,1,2]
    classname = ['AMV+','NEUTRAL','AMV-']
    correct  = np.array([0,0,0])
    total    = np.array([0,0,0])
    for e in range(ens):
        for t in range(tstep-lead):
            
            # Get index before
            idstart = t-nbefore
            if idstart < 0:
                idstart = 0
            
            valbefore = y_class_predictor[e,idstart:t]
            #print(t)
            if len(valbefore)==0: # Don't make prediction if there is no data before
                y_pred[e,t] = np.nan
                continue
            
            # Average values and select nearest class
            avgclass    = int(np.round(valbefore.mean()))
            y_pred[e,t] = avgclass
            
            # Add to counter
            actual = int(y_class_label[e,t])
            if avgclass == actual:
                correct[actual] += 1
            total[actual] += 1
    
    # ----------------------------------
    # Calculate and save overall results
    # ----------------------------------
    accbyclass   = correct/total
    totalacc     = correct.sum()/total.sum() 
    
    # Append Results
    acc_by_class.append(accbyclass)
    total_acc.append(totalacc)
    yvalpred.append(y_pred)
    yvallabels.append(y_class_label)
    
    # Report Results
    print("**********************************")
    print("Results for lead %i" % lead + "...")
    print("\t Total Accuracy is %.3f " % (totalacc*100) + "%")
    print("\t Accuracy by Class is...")
    for i in range(3):
        print("\t\t Class %i : %.3f " % (classval[i],accbyclass[i]*100) + "%")
    print("**********************************")

# -----------------
# Save Eval Metrics
# -----------------
outvars = {
         'total_acc'   : total_acc,
         'acc_by_class': acc_by_class,
         'yvalpred'    : yvalpred,
         'yvallabels'  : yvallabels}
np.savez("../../CESM_data/Metrics"+outname,outvars)
print("Saved data to %s%s. Finished variable %s in %ss"%(outpath,outname,varname,time.time()-start))
print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))
