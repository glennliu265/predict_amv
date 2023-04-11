#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Analyze the experiment output for train_NN_CESM1.py


Borrows the upper section from the eperiment 

Created on Mon Apr  3 08:01:55 2023

@author: gliu
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

# Load my own custom modules
import os
cwd = os.getcwd()
sys.path.append(cwd+"/../")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
sys.path.append("../")

import viz,proc
import amvmod as am # Import amv module for predict amv
#%% Import common parameters from file

import predict_amv_params as pparams

# Import data paths
datpath             = pparams.datpath
figpath             = pparams.figpath
proc.makedir(figpath)

# Import class information
classes             = pparams.classes
class_colors        = pparams.class_colors

# Import other information
leads               = np.arange(0,26,1)#pparams.leads

cmip6_names         = pparams.cmip6_names
cmip6_colors        = pparams.cmip6_colors
cmip6_markers       = pparams.cmip6_markers

# import Predictor information
varnames            = pparams.varnames
varnames_long       = pparams.varnames_long
varcolors           = pparams.varcolors
varmarker           = pparams.varmarker
cmip6_varnames_long = pparams.cmip6_varnames_long
cmip6_varcolors     = pparams.varcolors

#%% User Edits (PASTE BELOW HERE)

# ----------
# [TEMPLATE]
# ----------

"""
# Copy this template and enter the experiment information

exp1 = {"expdir"        :  , # Directory of the experiment
        "searchstr"     :  , # Search/Glob string used for pulling files
        "expname"       :  , # Name of the experiment (Short)
        "expname_long"  :  , # Long name of the experiment (for labeling on plots)
        "c"             :  , # Color for plotting
        "marker"        :  , # Marker for plotting
        "ls"            :  , # Linestyle for plotting
        
        }
"""

# ---------------------------------------------------
# %% Updated Cross Validation with consistent samples
# ---------------------------------------------------
nfolds = 4
inexps = []

vcolors = ["r","b"]
markers = ["d","o","x","+"]
lss     = ["dashed","solid","dotted","dashdot"]

for v,vname in enumerate(['SST','SSH']):
    for k in range(nfolds):
        exp = {"expdir"         : "FNN4_128_Singlevar_CV_consistent"      , # Directory of the experiment
                "searchstr"     :  "*%s*kfold%02i*" % (vname,k), # Search/Glob string used for pulling files
                "expname"       : "%s_fold%02i" % (vname,k)    , # Name of the experiment (Short)
                "expname_long"  : "%s (fold=%02i)" % (vname,k)   , # Long name of the experiment (for labeling on plots)
                "c"             : vcolors[v]                    , # Color for plotting
                "marker"        : markers[k]                   , # Marker for plotting
                "ls"            : lss[k]               , # Linestyle for plotting
                }
        inexps.append(exp)
compname                        = "FNN4_128_CV_consistent_SSH_SST"# CHANGE THIS for each new comparison

quartile = False
leads    = np.arange(0,26,3)
detrend  = False


# --------------------------------------------------
# %% Compare particular predictor across experiments
# --------------------------------------------------


exp2 = {"expdir"        : "FNN4_128_SingleVar"   , # Directory of the experiment
        "searchstr"     :  "*SSH*"               , # Search/Glob string used for pulling files
        "expname"       : "SSH_Original"       , # Name of the experiment (Short)
        "expname_long"  : "SSH (Original Script)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "b"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        "no_val"        : True  # Whether or not there is a validation dataset
        }

exp3 = {"expdir"        : "FNN4_128_Singlevar_Rewrite" , # Directory of the experiment
        "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
        "expname"       : "SSH_Rewrite"           , # Name of the experiment (Short)
        "expname_long"  : "SSH (Rewrite)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "orange"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "dashed"               , # Linestyle for plotting
        "no_val"        : True  # Whether or not there is a validation dataset
        }


exp4 = {"expdir"        : "FNN4_128_SingleVar_debug1_shuffle_all" , # Directory of the experiment
        "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
        "expname"       : "SSH_Rewrite_newest"           , # Name of the experiment (Short)
        "expname_long"  : "SSH (Rewrite Newest)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "r"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "dashed"               , # Linestyle for plotting
        "no_val"        : False  # Whether or not there is a validation dataset
        }

exp5 = {"expdir"        : "FNN4_128_SingleVar_debug1_shuffle_all_20ep_3ES_32bs" , # Directory of the experiment
        "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
        "expname"       : "SSH_Rewrite_newest_redEp"           , # Name of the experiment (Short)
        "expname_long"  : "SSH (Rewrite Newest, Reduce Epochs)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "magenta"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "dashed"               , # Linestyle for plotting
        "no_val"        : False  # Whether or not there is a validation dataset
        }


exp6 = {"expdir"        : "FNN4_128_SingleVar_debug1_shuffle_all_20ep_3ES_16bs" , # Directory of the experiment
        "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
        "expname"       : "SSH_Rewrite_newest_redEp_redBS"           , # Name of the experiment (Short)
        "expname_long"  : "SSH (Rewrite Newest, Reduce Epochs and Batch Size)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "limegreen"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "dashed"               , # Linestyle for plotting
        "no_val"        : False  # Whether or not there is a validation dataset
        }


inexps   = (exp2,exp3,exp4,exp5,exp6)
compname = "Rewrite"
quartile = False
leads    = np.arange(0,26,3)
detrend  = False
no_vals  = [d['no_val'] for d in inexps]

#%% [X] --------------- E N D    U S E R    I N P U T-------------------------------

#%% Locate the files
nexps = len(inexps)
flists = []
for ex in range(nexps):
    
    search = "%s%s/Metrics/%s" % (datpath,inexps[ex]["expdir"],inexps[ex]["searchstr"])
    flist  = glob.glob(search)
    flist  = [f for f in flist if "of" not in f]
    flist.sort()
    
    print("Found %i files for %s using searchstring: %s" % (len(flist),inexps[ex]["expname"],search))
    flists.append(flist)
#%% Load the data
"""
    Contents of expdict: 
        totalacc = [] # Accuracy for all classes combined [exp x run x leadtime]
        classacc = [] # Accuracy by class                 [exp x run x leadtime x class]
        ypred    = [] # Predictions                       [exp x run x leadtime x sample]
        ylabs    = [] # Labels                            [exp x run x leadtime x sample]
        shuffids = [] # Indices                           [exp x run x leadtime x sample]
"""

# Make the experiment dictionary
expdict = am.make_expdict(flists,leads,no_val=no_vals)

# Gather some dimension information for plotting
# if isinstance(quartile,list): #Different Threshold Types
#     quartile_first = quartile.index(True) # Get first non-quartile index
# else:
#     if quartile is True: # Nested Array [exp][run][lead][____]
#         nruns    = expdict['classacc'].shape[1]
#         nleads   = expdict['classacc'][0][0].shape[0]
#         nclasses = expdict['classacc'][0][0].shape[1]
#     else:
#         _,nruns,nleads,nclasses      = expdict['classacc'].shape


# Get Dimensions: classacc = [experiment][runs][lead x class]
nruns     = len(expdict['classacc'][0])      # Number of runs
nleads    = len(leads)                       # This will differ by experiment, so best to use leads above
nclasses  = len(expdict['classacc'][0][0][0]) # Thresholds of classification

# Convert classacc to np.array as it should now be uniform due to same leadtimes
if isinstance(expdict['classacc'],list):
    expdict['classacc'] = np.array(expdict['classacc'])

# Unpack Dictionary
totalacc,classacc,ypred,ylabs,shuffids = am.unpack_expdict(expdict)



#%% Load ALL the data




"""
'train_loss', (9,)
'test_loss', (9,)
'train_acc', (9,)
'test_acc', (9,)
'total_acc', (9,)
'acc_by_class', (9, 3)
'yvalpred', (9, 270)
'yvallabels', (9, 270)
'sampled_idx', (9, 900)
'thresholds_all', (0,)
'exp_params', ()           
'sample_sizes' (9,)     
"""

#%% Let's look at one of the experiments.

# First, load in all outputs
expdicts = [] # [experiment][run][output]
for iexp in range(len(inexps)):
    
    flist = flists[iexp]
    all_outputs = []
    all_vnames  = []
    for fn in flist:
        output,vnames = am.load_result(fn,load_dict=True)
        all_outputs.append(output)
        all_vnames.append(vnames)
    
    expdicts.append(all_outputs)

#%% First, let's check that the lead accuracies are looking... weird...

nexps         = len(inexps)
nruns         = 20 # Just load the first 20...

class_acc_all = np.zeros([nexps,nruns,nleads,3])

for iexp in range(nexps):
    #print(expdicts[iexp][0].files)
    
    for nr in range(nruns):
        
        # Load the Accuracy by Class
        class_acc_run = expdicts[iexp][nr]['acc_by_class'] # [lead x 3]
        if class_acc_run.shape[0] > 9:
            class_acc_run = class_acc_run[leads,:]
        class_acc_all[iexp,nr,:,:] = class_acc_run.copy()

#%% Quickly plot the results (for class accuracy)


def init_classacc_fig(leads,sp_titles=None):
    fig,axs=plt.subplots(1,3,constrained_layout=True,figsize=(18,4),sharey=True)
    if sp_titles is None:
        sp_titles=["AMV+","Neutral","AMV-"]
    for a,ax in enumerate(axs):
        ax.set_xlim([leads[0],leads[-1]])
        if len(leads) == 9:
            ax.set_xticks(leads)
        else:
            ax.set_xticks(leads[::3])
        ax.set_ylim([0,1])
        ax.set_yticks(np.arange(0,1.25,.25))
        ax.grid(True,ls='dotted')
        ax.minorticks_on()
        ax.set_title(sp_titles[a],fontsize=20)
        if a == 0:
            ax.set_ylabel("Accuracy")
        if a == 1:
            ax.set_xlabel("Prediction Leadtime (Years)")
    return fig,axs


fig,axs=init_classacc_fig(leads)

for iexp in range(nexps):
    inexpdict = inexps[iexp]
    
    for c in range(3):
        ax = axs[c]
        plotacc = class_acc_all[iexp,:,:,c].mean(0)
        ax.plot(leads,plotacc,label=inexpdict["expname_long"],lw=2.5,
                color=inexpdict["c"],marker=inexpdict["marker"])
        

ax = axs[1]
ax.legend()
        
#%% Lets look at the train and test loss for each experiment
nexps         = len(inexps)
nruns         = 20 # Just load the first 20...
l             = 0
#class_acc_all = np.zeros([nexps,nruns,nleads,3])

modes      = ["train","test","val"]
modecolors = ["blue","red","orange"]

fig,axs = plt.subplots(1,5,constrained_layout=True,figsize=(18,4),sharey=True)

for iexp in range(nexps):
    inexpdict = inexps[iexp]
    #print(expdicts[iexp][0].files)
    ax  = axs[iexp]
    for nr in range(nruns):
        
        # Load the losses
        losses = []
        for mode in modes:
            if (no_vals[iexp] is True) and mode == "val":
                continue
            losses.append(expdicts[iexp][nr]['%s_loss'%mode][l])
        
        for mm in range(len(losses)):
            plotloss = losses[mm]
            plotepochs = np.arange(1,len(plotloss)+1)
            ax.plot(plotepochs,plotloss,color=modecolors[mm],alpha=0.5,label="")
            
    ax.legend()
    ax.set_title(inexpdict["expname_long"])
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.grid(True,ls='dotted')
    ax.minorticks_on()
    ax.set_xlim([1,20])







    
    
    

        
        


#%% Get some necessary values
nleads   = len(leads)


#%% Plot train loss and test loss by EPOCH for each experiment

#output = all_outputs[0]

for iexp in range(len(inexps)):
    
    flist = flists[iexp]
    all_outputs = []
    for fn in flist:
        output,vnames = am.load_result(fn,load_dict=False)
        all_outputs.append(output)
    
    nruns    = len(all_outputs)
    nleads   = len(leads)
    
    
    fig,axs = plt.subplots(3,3,figsize=(16,8),sharey=True,
                          constrained_layout=True)
    
    for l in range(nleads):
        ax = axs.flatten()[l]
    
        if l == 7:
            ax.set_xlabel("Prediction Lead (Years)")
        if l == 3:
            ax.set_ylabel("Loss")
        
        lossavg = np.zeros((2,20)) # [train/test,epoch]
        losscnt = np.zeros((20))
        
        for nr in range(nruns):
            # Get values
            nepochs   = len(all_outputs[nr][0][l])
            trainloss = all_outputs[nr][0][l]
            testloss  = all_outputs[nr][1][l]
            # Track for Average
            lossavg[0,:nepochs] += trainloss
            lossavg[1,:nepochs] += testloss
            losscnt[:nepochs] += 1
            # Plot
            ax.plot(np.arange(1,nepochs+1),trainloss,color="cornflowerblue",label="",alpha=0.55)
            ax.plot(np.arange(1,nepochs+1),testloss,color="orange",label="",alpha=0.55)
        
        ax.plot(np.arange(1,21),lossavg[0,:]/losscnt,color="k",label="Train Loss Avg.",alpha=1,ls='dashed')
        ax.plot(np.arange(1,21),lossavg[1,:]/losscnt,color="k",label="Test Loss Avg.",alpha=1)
        
        
        ax.set_xticks(np.arange(1,21,1))
        ax.set_xlim([1,20])
        ax.grid(True,ls="dotted",alpha=0.75)
        ax.set_title("Lead %02i" % leads[l])
        
        if l == 0:
            ax.legend()
    
    plt.suptitle("%s Experiment: %s" % (compname,inexps[iexp]['expname_long']))
    savename = "%sTrainTestLoss_%s_%s.png" % (figpath,compname,inexps[iexp]['expname'])
    plt.savefig(savename,dpi=150)

#%% Plot the distribution of classes


iexp        = 0
flist       = flists[iexp]
all_outputs = []
for fn in flist:
    all_outputs.append(np.load(fn,allow_pickle=True))
nruns    = len(all_outputs)

# ------------------------------
# Compare histograms across runs
# ------------------------------
fig,axs = plt.subplots(3,3,figsize=(16,8),sharey=True,
                      constrained_layout=True)
for l in range(nleads):
    ax = axs.flatten()[l]
    
    ax.set_title("Lead %02i" % leads[l])
    if l == 7:
        ax.set_xlabel("Run")
    if l == 3:
        ax.set_ylabel("Counts")
    
    for nr in range(nruns):
        
        runsamples = all_outputs[nr]['yvallabels'][l,:]
        ibc,counts = am.count_samples(len(runsamples),runsamples)
        
        for c in range(3):
            ax.plot(nr,counts[c],color=class_colors[c],marker="o",alpha=0.5)
    if l == 0:
        ax.legend()
plt.suptitle("Class Count by lead (run intercomparison)")
                


# -----------------------------------
# Compare histogram runs across folds
# -----------------------------------
nruns             = len(flists[0])
nexps             = len(inexps)
class_counts_exps = np.zeros((nexps,nruns,nleads,3)) # [experiment, run, lead, class]
for iexp in range(nexps):
    
    flist       = flists[iexp]
    all_outputs = []
    for fn in flist:
        all_outputs.append(np.load(fn,allow_pickle=True))
    
    for l in range(nleads):
        for nr in range(nruns):
            
            runsamples = all_outputs[nr]['yvallabels'][l,:]
            ibc,counts = am.count_samples(len(runsamples),runsamples)
            
            class_counts_exps[iexp,nr,l,:] = counts
            

fig,axs = plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
for c in range(3):
    ax = axs[c]
    countavgs = []
    labels    = []
    for iexp in range(nexps):
        countavg = np.mean(class_counts_exps[iexp,:,:,c],(0,1))
        countavgs.append(countavg)
        labels.append(inexps[iexp]['expname_long'])
    ax.bar(labels,countavgs)
    ax.legend()

    ax.set_xticklabels(labels,rotation=45)
#%%





    
#%%
    





#%%

