#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Count Samples for Classification. Eventually make this into a
function to set samples.

Use this to evaluate the characteristics of each sample.


- Copied from NN_test_lead_ann_singlevar_lens.py


Created on Tue Feb 14 13:49:58 2023

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,Dataset
import os
import copy
import xarray as xr
import sys

#%% Import some parameters (add more here eventually)

cmipver = 6
do_only = ["CESM2",]

sys.path.append("../")
import predict_amv_params as pparams
import amvmod as am

# Added for purposes of plotting
classes      = pparams.classes
class_colors = pparams.class_colors


if cmipver == 5:
    dataset_names = pparams.dataset_names
    ystarts       = pparams.dataset_starts
    datdir        =  "../../CESM_data/LENS_other/processed"
    limit_time    = [1950,2005] # Set Dates here to limit the range of the variable
    ens           = 30
    regrid        = 3
    
elif cmipver == 6:
    dataset_names = pparams.cmip6_names
    ystarts       = (1850,)*len(dataset_names)
    datdir        = "../../CESM_data/CMIP6_LENS/processed/" 
    varnames      = ("sst","ssh","sss")
    limit_time    = [1850,2014] # Set Dates here to limit the range of the variable
    ens           = 25
    regrid        = None


# Special Settings for double checking 1920-2005/CESM1 differences
limit_time = [1920,2005]
ens        = 40

# -------------
#%% User Edits
# -------------

# Other Things...
yend                = 2014
lp                  = 0
d                   = 5
v                   = 0

# What was within the loops initially...
varname        = varnames[v]
datasetname    = dataset_names[d]
expdir         = "CMIP6_LENS/processed/"
ystart         = ystarts[d]
    
# Data preparation settings
bbox           = [-80,0,0,65]               # Bounding box of predictor
leads          = np.arange(0,25,3)#[a for a in np.arange(0,26,1) if a not in np.arange(0,27,3)]#np.arange(0,27,3)#(0,)     # np.arange(0,25,3)   # Time ahead (in years) to forecast AMV
thresholds     = [-1,1]#[1/3,2/3]      #     # Thresholds (standard deviations, or quantile values) 
quantile       = False                      # Set to True to use quantiles

usefakedata    = None                       # Set to None, or name of fake dataset.
region         = None                       # Set region of analysis (None for basinwide)
allpred        = ("SST","SSS","PSL","SSH")
detrend        = False                      # Set to true to use detrended data

# Training/Testing Subsets
nsamples       = 300                        # Number of samples for each class. Set to None to use all
percent_train  = 0.8              # Percentage of data to use for training (remaining for testing)
#numruns       = 10    # Number of times to train for each leadtime

# Additional Hyperparameters (CNN)
batch_size     = 16                   # Pairs of predictions

# Toggle Options
# --------------
debug         = False # Visualize training and testing loss
verbose       = True # Print loss for each epoch

# -----------------------------------------------------------------
#%% Additional (Legacy) Variables (modify for future customization)
# -----------------------------------------------------------------

# Data Preparation names
num_classes    = len(thresholds)+1    # Set up number of classes for prediction (current supports)
season         = 'Ann'                # Season to take mean over ['Ann','DJF','MAM',...]
usenoise       = False                # Set to true to train the model with pure noise

# -----------
#%% Functions
# -----------

def make_classes(y,thresholds,exact_value=False,reverse=False,
                 quantiles=False):
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
    
    if quantiles is False:
        if ~exact_value: # Scale thresholds by standard deviation
            y_std = np.std(y) # Get standard deviation
            thresholds = np.array(thresholds) * y_std
    else: # Determine Thresholds from quantiles
        thresholds = np.quantile(y,thresholds,axis=0) # Replace Thresholds with quantiles
    
    nthres  = len(thresholds)
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
    if quantiles is True:
        return y_class,thresholds
    return y_class

def select_samples(nsamples,y_class,X):
    """
    Sample even amounts from each class

    Parameters
    ----------
    nsample (INT)                   : Number of samples to get from each class
    y_class (ARRAY) [samples x 1]   : Labels for each sample
    X       (ARRAY) [samples x channels x height x width] Input data for each sample

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
        inclass   = classes[i]
        idx       = (y_class==inclass).squeeze()
        sel_label = y_class[idx,:]
        sel_input = X[idx,:,:,:]
        sel_idx   = np.where(idx)[0]
        
        label_by_class.append(sel_label)
        input_by_class.append(sel_input)
        idx_by_class.append(sel_idx)
        classcount = sel_input.shape[0]
        print("%i samples found for class %i" % (classcount,inclass))
        
        # Check
        assert nsamples <= classcount,"Number of samples found for class %s (n=%i) is less than the selection amount (n=%i)!" % (inclass,classcount,nsamples)
        
        
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

def count_samples(nsamples,y_class):
    """
    Simplified version of select_samples that only counts the classes
    and returns the indices/counts
    """
    classes    = np.unique(y_class)
    nclasses   = len(classes)
    idx_by_class    = [] 
    count_by_class  = []
    for i in range(nclasses):
        
        # Sort by Class
        inclass   = classes[i]
        idx       = (y_class==inclass).squeeze()
        sel_idx   = np.where(idx)[0]
        
        idx_by_class.append(sel_idx)
        classcount = sel_idx.shape[0]
        count_by_class.append(classcount)
        print("%i samples found for class %i" % (classcount,inclass))
    return idx_by_class,count_by_class
    
    

#%% 

# ----------------------------------------
# %% Set-up
# ----------------------------------------
allstart = time.time()

# -------------
# Load the data
# -------------
fname = "%s/%s_%s_NAtl_%ito%i_detrend%i_regrid%sdeg.nc" % (datdir,
                                                           datasetname,
                                                           varname,
                                                           ystart,
                                                           yend,
                                                           detrend,
                                                           regrid)

ds      = xr.open_dataset(fname)
allflag = False
        
if allflag is False:
    ds     = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
    # Crop to a time period if the option is set
    if limit_time is not None:
        ds = ds.sel(year=slice(limit_time[0],limit_time[1]))
        tstep = limit_time[1]-limit_time[0] +1
        print("Limiting dataset to specified time %s" % limit_time)
    data   = ds[varname].values[None,...] # [channel x ens x yr x lat x lon]
    data[np.isnan(data)] = 0
    
if region is None:
    if cmipver == 5:
        targname = "%s/%s_nasst_label_%ito%i_detrend%i_regrid%sdeg.npy" % (datdir,datasetname,ystart,yend,detrend,regrid)
    elif cmipver ==6:
        targname = "%s/%s_sst_label_%ito%i_detrend%i_regrid%sdeg_lp%i.npy" % (datdir,datasetname,ystart,yend,detrend,regrid,lp)
    target = np.load(targname,allow_pickle=True)
else:
    print("WARNING, region currently not supported")

# Limit target to ensemble member and time
target     = target[0:ens,:]
if limit_time is not None:
    yrvalues = np.arange(ystart,ystart+target.shape[1])
    istart   = np.argwhere(yrvalues==limit_time[0])[0][0]
    iend     = np.argwhere(yrvalues==limit_time[1])[0][0]
    target   = target[:,istart:iend+1]

testvalues=[True]
testname='unfreeze_all'

    
# Set experiment names ----
nlead    = len(leads)
channels = data.shape[0]
start    = time.time()

# -------------
# Print Message
# -------------
print("Running CNN_test_lead_ann.py with the following settings:")
print("\tLeadtimes      : %i to %i" % (leads[0],leads[-1]))
print("\t# Ens. Members : "+ str(ens))
print("\tDetrend        : "+ str(detrend))
print("\tUse Noise      :" + str(usenoise))

#%%


thresholds_all     = [] # Thresholds
sampled_idx        = [] # Sample Indices
lead_nsamples_all  = [] # Number of samples for that leadtime

count_by_class_all = [] # [lead][class]
idx_by_class_all   = [] # [lead][class][sample]

inputs_all         = []
labels_all         = []


nr = 0
i  = 0

# Looping for each leadtime
for l,lead in enumerate(leads):
    print("Lead %i" % lead)
    # ----------------------
    # Apply lead/lag to data
    # ----------------------
    if (i == 0) and (nr ==0):
        thresholds_old = thresholds.copy() # Copy Original Thresholds (Hack Fix)
    thresholds = thresholds_old.copy()
    nchannels,nens,ntime,nlat,nlon=data.shape
    y = target[:ens,lead:].reshape(ens*(tstep-lead),1)
    X = (data[:,:ens,:tstep-lead,:,:]).reshape(nchannels,ens*(tstep-lead),nlat,nlon).transpose(1,0,2,3)
    
    # Test current and old make_classes function
    #y_class      = make_classes(y,thresholds,reverse=True,quantiles=quantile)
    #y_class_func = am.make_classes(y,thresholds,reverse=True,quantiles=quantile)
    #assert np.all(y_class == y_class_func)
    
    std1    = target.std(1).mean() # Take stdev in time, mean across ensembles
    print(std1)
    y_class = am.make_classes(y,[-std1,std1],exact_value=True,reverse=True,quantiles=quantile)
    
    if quantile == True:
        thresholds = y_class[1].T[0]
        print(y_class.shape)
        y_class    = y_class[0]
        print(y_class.shape)
    thresholds_all.append(thresholds) # Save Thresholds
    
    if (nsamples is None) or (quantile is True):
        nthres = len(thresholds) + 1
        threscount = np.zeros(nthres)
        for t in range(nthres):
            threscount[t] = len(np.where(y_class==t)[0])
        nsamples = int(np.min(threscount))
    
    idx_by_class,count_by_class=count_samples(nsamples,y_class) 
    idx_by_class_all.append(idx_by_class)
    count_by_class_all.append(count_by_class)
    
    y_class,X,shuffidx = select_samples(nsamples,y_class,X)
    lead_nsamples      = y_class.shape[0]
    lead_nsamples_all.append(lead_nsamples)
    sampled_idx.append(shuffidx) # Save the sample indices
    
    # ---------------------------------
    # Split into training and test sets
    # ---------------------------------
    X_train = torch.from_numpy( X[0:int(np.floor(percent_train*lead_nsamples)),...].astype(np.float32) )
    X_val   = torch.from_numpy( X[int(np.floor(percent_train*lead_nsamples)):,...].astype(np.float32) )
    y_train = torch.from_numpy( y_class[0:int(np.floor(percent_train*lead_nsamples)),:].astype(np.compat.long)  )
    y_val   = torch.from_numpy( y_class[int(np.floor(percent_train*lead_nsamples)):,:].astype(np.compat.long)  )
    inputs_all.append((X_train,X_val))
    labels_all.append((y_train,y_val))
    
    # Put into pytorch DataLoader
    #train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    #val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    
#%% Make some visualizations

counts_bylead = np.array(count_by_class_all)
fig,ax        = plt.subplots(1,1,constrained_layout=True)
for c in range(3):
    if c == 1:
        continue
    else:
        ax.plot(leads,counts_bylead[:,c],
                label=classes[c],color=class_colors[c],
                lw=2.5,marker="o")
ax.legend(fontsize=12)
ax.grid(True,ls="dotted")
ax.set_xlim([leads[0],leads[-1]])
ax.set_xticks(leads)
ax.set_xlabel("Leadtime (Years)")
ax.set_ylabel("# Samples")
ax.set_title("%s Class Samples by Leadtime\n(n=%i, percent train=%.2f)" % (datasetname,ens,percent_train))

#%%


#%%

#%% Examine characteristics of the selected indices



    
    


