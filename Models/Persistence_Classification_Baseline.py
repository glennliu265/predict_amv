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
import matplotlib.pyplot as plt
import sys
import os

# -------------------------------
# %% Import Experiment Parameters
# -------------------------------

import os
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import amvmod as am
import amv_dataloader as dl 

# -------------
#%% User Edits
# -------------

# Adjustable settings
leads          = np.arange(0,25,3)    # Time ahead (in years) to forecast AMV
nsamples       = 300                  # Number of samples for each class
ens            = 40
bbox           = pparams.bbox
thresholds     = pparams.thresholds   # Thresholds (standard deviations, determines number of classes)   
classes        = pparams.classes      # Name of classes
quantile       = False  # Set to True to use quantile thresholds
percent_train  = 1.00
use_train      = False
usenoise       = False
varname        = "SST"

# Other Toggles
detrend        = 1                    # Set to True to use detrended data
save_baseline  = True                 # Set to True to save baseline
ccai_names     = False                # Set to True to use CCAI naming conventions (will likely become dead code)

# Other Toggles
datpath        = pparams.datpath

# Additional user edits (Erase, possibly)
#nbefore = 1 #'variable'
#subtitle = "\n %s = %i; detrend = %s"% (testname,testvalues[i],detrend)
#subtitle = "\nPersistence Baseline, averaging %s years before" % (str(nbefore))

# -----------------------------------------------------------------
#%% Additional (Legacy) Variables (modify for future customization)
# -----------------------------------------------------------------

# Saving Information
num_classes    = len(thresholds)+1    # Set up number of classes for prediction
nlead          = len(leads)
leadstep       = leads[1]-leads[0]

# Data information
if ccai_names:
    season         = 'Ann'                # Season to take mean over ['Ann','DJF','MAM',...]
    indexregion    = 'NAT'                # One of the following ("SPG","STG","TRO","NAT")
    resolution     = '224pix'             # Resolution of dataset ('2deg','224pix')
    detrend        = False                # Set to true to use detrended data
    limitsamples   = True                 # Set to true to only evaluate first [nsamples] for each class
    usenoise       = False                # Set to true to train the model with pure noise
    tstep          = 86    # Size of time dimension (in years)
    channels       = 3
    varname        = 'ALL'
    quantile       = False
    
    # Set saving names
    nbefore        = 1 #"variable"
    expname        = "AMVClass%i_PersistenceBaseline_%sbefore_nens%02i_maxlead%02i_detrend%i_noise%i_nsample%s_limitsamples%i" % (num_classes,str(nbefore),ens,leads[-1],detrend,usenoise,nsamples,limitsamples)
    outname        = "/leadtime_testing_%s_%s_ALL_nsamples1.npz" % (varname,expname)

else:
    region = None
    varnames = ["SST",]
    
    # Set saving names
    expname  = "AMVClassification_Persistence_Baseline_ens%02i_Region%s_maxlead%02i_step%i_nsamples%s_detrend%i_%03ipctdata" % (ens,region,leads[-1],leadstep,nsamples,detrend,np.round((1.00-percent_train)*100,0))
    outname  = "%s%s.npz" % (datpath,expname)

if save_baseline:
    print("Data will be saved to: %s" % (outname))

# ----------------------------------------
# %% Set-up
# ----------------------------------------
allstart = time.time()

# Load the data for whole North Atlantic
if ccai_names:
    data   = np.load('../../CESM_data/CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i.npy'%detrend)
    target = np.load('../../CESM_data/CESM_label_amv_index_detrend%i.npy'%detrend)
else:
    target = dl.load_target_cesm(detrend=detrend,region=region)
    data,lat,lon   = dl.load_data_cesm(varnames,bbox,detrend=detrend,return_latlon=True)

"""
At this point, have:
    data   : [channel x ens x yr x lat x lon]
    target : [ens x yr]
"""
# Limit to # of ensemble members
data   = data[:,0:ens,:,:,:]
target = target[0:ens,:]
tstep  = data.shape[2]
# -------------
#%% Make classes
# -------------

# Get Standard Deviation Threshold
print("Original thresholds are %i stdev" % thresholds[0])
std1   = target.std(1).mean() * thresholds[1] # Multiple stdev by threshold value 
if quantile is False:
    thresholds = [-std1,std1]
    print(r"Setting Thresholds to +/- %.2f" % (std1))

# Convert target to class
y       = target[:ens,:].reshape(ens*tstep,1)
y_class = am.make_classes(y,thresholds,reverse=True,exact_value=True,quantiles=quantile)
y_class = y_class.reshape(ens,(tstep)) # Reshape to [ens x lead]


#%% Quickly Count the Classes....
for l,lead in enumerate(leads):
    y_class_in = y_class[:,lead:]
    print("Lead %i" % lead)
    idx_by_class,count_by_class=am.count_samples(nsamples,y_class_in)

#%%

# Get some dimension sizes, etc

start    = time.time()

# Preallocate Evaluation Metrics...
total_acc       = [] # [lead]
acc_by_class    = [] # [lead x class]
yvalpred        = [] # [lead x ensemble x time]
yvallabels      = [] # [lead x ensemble x time]
samples_counts  = [] # [lead x class]
nvarflag        = False

# -------------
# Print Message
# -------------
print("Calculate Persistence Baseline with the following settings:")
print("\tLeadtimes        \t: %i to %i (step=%i)" % (leads[0],leads[-1],leadstep))
print("\tClass Threshold  \t: [%.2f, %.2f] (quantile=%i)" % (thresholds[0],thresholds[1],quantile))
print("\t# Ens. Members   \t: "+ str(ens))
print("\t# Training Samples \t: "+ str(nsamples))
#print("\t# Years Before : "+ str(nbefore))
print("\tDetrend          \t: "+ str(detrend))
print("\tUse Noise        \t: " + str(usenoise))
print("\tCCAI Names       \t: " + str(ccai_names))
#%%

for l,lead in enumerate(leads):
    
    # if nvarflag or nbefore=='variable':
    #     print("Using Variable nbefore")
    #     nvarflag=True
    #     nbefore =lead 
    
    # -------------------------------------------------------
    # Set [predictor] to the [target] but at the initial time
    # -------------------------------------------------------
    X                 = y_class[:,:(tstep-lead)].flatten()[:,None,None,None] # Expand dimensions to accomodate function
    y                 = y_class[:,lead:].flatten()[:,None] # Note, overwriting y again ...
    
    #
    # Subsample prior to the split
    #
    if nsamples is not None:
        y_class_label,y_class_predictor,shuffidx = am.select_samples(nsamples,y,X)
        y_class_predictor                        = y_class_predictor.squeeze()
    else:
        y_class_label     = y
        y_class_predictor = X.squeeze()
    
    
    # ----------------
    # Train/Test Split
    # ----------------
    X_subset,y_subset = am.train_test_split(y_class_predictor,y_class_label,
                                            percent_train=percent_train,
                                            debug=True)
    if percent_train < 1:
        X_train,X_val     = X_subset
        y_train,y_val     = y_subset
        if use_train:
            y_class_label     = y_train
            y_class_predictor = X_train
        else:
            y_class_label     = y_val
            y_class_predictor = X_val
    
    # -------------------------------------
    # Randomly sample same # for each class
    # -------------------------------------

    
    # Output : y_class_predictor 
    
    # ----------------------
    # Make predictions
    # ----------------------
    allsamples = y_class_predictor.shape[0]
    classval   = [0,1,2]
    correct    = np.array([0,0,0])
    total      = np.array([0,0,0])
    for n in range(allsamples):
        actual = int(y_class_label[n,0])
        y_pred = int(y_class_predictor[n])
        
        #print("For sample %i, predicted %i, actual %i" % (n,y_pred,actual))
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
    samples_counts.append(total)
    
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
         'yvallabels'  : yvallabels,
         'samples_counts': samples_counts}
if save_baseline:
    np.savez("../../CESM_data/Metrics/"+outname,outvars,allow_pickle=True)
    print("Saved data to %s. Finished variable %s in %ss"%(outname,varname,time.time()-start))
    print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))
else:
    print("Persistsence baseline is not saved!")

#%% Do functionized version and compare
out_dict = am.compute_persistence_baseline(leads,y_class,nsamples=nsamples,percent_train=percent_train)


shared_keys = [k for k in out_dict.keys() if k in outvars.keys()]
#check       = [np.all(out_dict[k] == outvars[k]) for k in shared_keys]

# -----------------------  
#%% Do some visualization
# -----------------------



fig,ax = plt.subplots(1,1)
ax.plot(leads,outvars['total_acc'],label="In Script Version")
ax.plot(leads,out_dict['total_acc'],label="Function Version")
ax.set_xlabel("Prediction Lead (Years)")
ax.set_ylabel("Accuracy")
ax.legend()



#%%


    
    