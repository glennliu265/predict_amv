#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Persistence Baseline

Script to calculate the persistence baseline.

Uses data preprocessed by "prepare_training_validation_data.py"
    Assumes data is placed in "../../CESM_data/"

Output is saved to "../../CESM_data/Metrics/"
"""

import numpy as np
import time

# -------------
#%% User Edits
# -------------

# Adjustable settings
leads          = np.arange(0,25,3)    # Time ahead (in years) to forecast AMV
thresholds     = [-1,1]               # Thresholds (standard deviations, determines number of classes) 
nsamples       = 300                  # Number of samples for each class

# -----------------------------------------------------------------
#%% Additional (Legacy) Variables (modify for future customization)
# -----------------------------------------------------------------

# Data information
season         = 'Ann'                # Season to take mean over ['Ann','DJF','MAM',...]
indexregion    = 'NAT'                # One of the following ("SPG","STG","TRO","NAT")
resolution     = '224pix'             # Resolution of dataset ('2deg','224pix')
detrend        = False                # Set to true to use detrended data
limitsamples   = True                 # Set to true to only evaluate first [nsamples] for each class
usenoise       = False                # Set to true to train the model with pure noise
ens           = 40                   # Ensemble members to use
tstep         = 86    # Size of time dimension (in years)

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
    if exact_value is False: # Scale thresholds by standard deviation
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

# Set number of classes (# thresholds +1)
num_classes    = len(thresholds)+1    # Set up number of classes for prediction

# Load the data for whole North Atlantic
data   = np.load('../../CESM_data/CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i.npy'%detrend)
target = np.load('../../CESM_data/CESM_label_amv_index_detrend%i.npy'%detrend)
data   = data[:,0:ens,:,:,:]
target = target[0:ens,:]

# %% Some more user edits
nbefore = 1#'variable'

# Preallocate
nlead    = len(leads)
channels = 3
start    = time.time()
varname  = 'ALL'
#subtitle = "\n %s = %i; detrend = %s"% (testname,testvalues[i],detrend)
subtitle="\nPersistence Baseline, averaging %s years before" % (str(nbefore))

# Save data (ex: Ann2deg_NAT_CNN2_nepoch5_nens_40_lead24 )
expname = "AMVClass%i_PersistenceBaseline_%sbefore_nens%02i_maxlead%02i_detrend%i_noise%i_nsample%i_limitsamples%i" % (num_classes,str(nbefore),ens,leads[-1],detrend,usenoise,nsamples,limitsamples)
outname = "/leadtime_testing_%s_%s_ALL_nsamples1.npz" % (varname,expname)

#%%

# Preallocate Evaluation Metrics...
total_acc       = [] # [lead]
acc_by_class    = [] # [lead x class]
yvalpred        = [] # [lead x ensemble x time]
yvallabels      = [] # [lead x ensemble x time]
nvarflag        = False

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
    
    # -------------
    # Make classes
    # -------------
    y = target[:ens,:].reshape(ens*tstep,1)
    y_class = make_classes(y,thresholds,reverse=True)
    y_class = y_class.reshape(ens,(tstep)) # Reshape to ens x lead
    
    # -------------------------------------
    # Randomly sample same # for each class
    # -------------------------------------
    X = y_class[:,:(tstep-lead)].flatten()[:,None,None,None] # Expand dimensions to accomodate function
    y_class_label = y_class[:,lead:].flatten()[:,None]
    y_class_label,y_class_predictor,shuffidx = select_samples(nsamples,y_class_label,X)
    y_class_predictor = y_class_predictor.squeeze()
    
    # ----------------------
    # Make predictions
    # ----------------------
    allsamples = y_class_predictor.shape[0]
    classval = [0,1,2]
    classname = ['AMV+','NEUTRAL','AMV-']
    correct  = np.array([0,0,0])
    total    = np.array([0,0,0])
    for n in range(allsamples):
        actual = int(y_class_label[n,0])
        y_pred = int(y_class_predictor[n])
        
        # Add to Counter
        if actual == y_pred:
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
    #yvalpred.append(y_pred)
    yvalpred.append(y_class_predictor)
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
np.savez("../../CESM_data/Metrics"+outname,outvars,allow_pickle=True)
print("Saved data to %s. Finished variable %s in %ss"%(outname,varname,time.time()-start))
print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))