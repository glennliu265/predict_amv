#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General NN accuracy visualization and evaluation.

   - Copied output based on viz_acc_by_predictor.py


Created on Wed Jan 25 10:19:36 2023

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
import amv_dataloader as dl
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

# -----------------------
#%% SingleVar Comparison (version agnostic)
# -----------------------


# MIROC6, Exact Thresholds, Limit 1920 - 2005
# expname     = "CESM2_1920to2005"
# expdir      = "CMIP6_LENS/models/Limit_1920to2005/FNN4_128_SingleVar_CESM2_Train"
# cmipver     = 6
# quartile    = True
# leads       = np.arange(0,26,3)
# var_include = ["ssh","sst","sss"]

# expname     = "MIROC6_ExactThres"
# expdir      = "CMIP6_LENS/models/Limit_1920to2005/FNN4_128_SingleVar_MIROC6_Train"
# cmipver     = 6
# quartile    = True
# leads       = np.arange(0,26,3)
# var_include = ["ssh","sst","sss"]


expname     = "CESM1_Detrend"
expdir      = "FNN4_128_detrend"
cmipver     = 5
quartile    = False
leads       = np.arange(0,25,3)
var_include = ["SSH","SST","SSS"]

if cmipver == 5:
    varnames         = pparams.varnames
    varnames_long_in = varnames_long
    varcolors_in     = varcolors

elif cmipver == 6:
    varnames         = pparams.cmip6_varnames_remap
    varnames_long_in = cmip6_varnames_long
    varcolors_in     = cmip6_varcolors

inexps = []
for v,varname in enumerate(varnames):
    print(varname)
    print(v)
    if cmipver == 5:
        if varname not in var_include:
            print("Skipping %s" % varname)
            continue

    
    exp = {"expdir"         : expdir                , # Directory of the experiment
            "searchstr"     :  "*%s*" % varname     , # Search/Glob string used for pulling files
            "expname"       : varname               , # Name of the experiment (Short)
            "expname_long"  : varnames_long_in[v]      , # Long name of the experiment (for labeling on plots)
            "c"             : varcolors_in[v]          , # Color for plotting
            "marker"        : varmarker[v]          , # Marker for plotting
            "ls"            : "solid"               , # Linestyle for plotting
        }
    inexps.append(exp)

compname = "%s_SingleVar_comparison" % expname# CHANGE THIS for each new comparison

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
# %% Compare particular predictor across experiments for wrtiten version
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


exp7 = {"expdir"        : "FNN4_128_SingleVar_debug1_shuffle_all_no_val" , # Directory of the experiment
        "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
        "expname"       : "SSH_Rewrite_newest_no_val"           , # Name of the experiment (Short)
        "expname_long"  : "SSH (Rewrite Newest, No Validation)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "cyan"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        "no_val"        : False  # Whether or not there is a validation dataset
        }

exp8 = {"expdir"        : "FNN4_128_SingleVar_debug1_shuffle_all_no_val_8020" , # Directory of the experiment
        "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
        "expname"       : "SSH_Rewrite_newest_no_val_8020"           , # Name of the experiment (Short)
        "expname_long"  : "SSH (Rewrite Newest, No Validation 80-20)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "yellow"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        "no_val"        : False  # Whether or not there is a validation dataset
        }

exp9 = {"expdir"        : "FNN4_128_Singlevar_Rewrite_June" , # Directory of the experiment
        "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
        "expname"       : "SSH_Rewrite_June"           , # Name of the experiment (Short)
        "expname_long"  : "SSH (Rewrite June)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "cornflowerblue"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "dashed"               , # Linestyle for plotting
        "no_val"        : False  # Whether or not there is a validation dataset
        }

inexps   = (exp2,exp3,exp4,exp5,exp6,exp7,exp8,exp9)
compname = "Rewrite"
quartile = False
leads    = np.arange(0,26,3)
detrend  = False
no_vals  = [d['no_val'] for d in inexps]

#%% PIC vs HTR


inexps   = []
vcolors  = ["r","b"]
markers  = ["d","o","d","o"]
lss      = ["dashed","solid","dashed","solid"]
exps     = ["FNN4_128_detrend","FNN4_128_SingleVar_PIC"]
expnames = ["Historical Detrended","PiControl"]

for v,vname in enumerate(['SST','SSH']):
    for exp in range(2):
        
        
        exp = {"expdir"         : exps[exp]     , # Directory of the experiment
                "searchstr"     :  "*%s*" % (vname), # Search/Glob string used for pulling files
                "expname"       : "%s_%s" % (exps,vname), # Name of the experiment (Short)
                "expname_long"  : "%s (%s)" % (expnames[exp],vname)   , # Long name of the experiment (for labeling on plots)
                "c"             : vcolors[v]                    , # Color for plotting
                "marker"        : markers[exp]                   , # Marker for plotting
                "ls"            : lss[exp]               , # Linestyle for plotting
                "no_val"        : False,  # Whether or not there is a validation dataset
                }
        
        inexps.append(exp)
        
compname                        = "FNN4_128_HTR_v_PiC"# CHANGE THIS for each new comparison
leads                           = np.arange(0,26,3)
quartile                        = False
detrend                         = True
no_vals                         = [True,False,True,False]


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
#%% Load the persistence baseline

# fpath = "../Data/Metrics/"
# fnp ="leadtime_testing_ALL_AMVClass3_PersistenceBaseline_1before_nens40_maxlead24_detrend%i_noise0_nsample400_limitsamples1_ALL_nsamples1.npz" % detrend

# ldp = np.load(fpath+fnp,allow_pickle=True)#.f#.arr_0

# persaccclass = np.array(ldp['arr_0'][None][0]['acc_by_class']) # [Lead x Class]}
# persacctotal = np.array(ldp['arr_0'][None][0]['total_acc'])

# persleads    = np.arange(0,26,3)


persleads,persaccclass,persacctotal = dl.load_persistence_baseline("CESM1",datpath=None,return_npfile=False,region=None,quantile=False,
                              detrend=True,limit_samples=True,nsamples=None,repeat_calc=1,ens=42)


# ------------------------
#%% Do some visualizations
# ------------------------

# General plotting options
lwall      = 2.5
darkmode   = False
alpha      = 0.05

if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
else:
    plt.style.use('default')
    dfcol = "k"

# Visualize Accuracy by Class
plotmax    = False # Set to True to plot maximum
add_conf   = True  # Add confidence intervals
plotconf   = 0.95  # Select which intervals to plot

fig,axs = plt.subplots(1,3,figsize=(18,4))
for c in range(3):
    
    # Initialize plot
    ax = axs[c]
    ax.set_title("%s" %(classes[c]),fontsize=16,)
    ax.set_xlim([0,24])
    ax.set_xticks(persleads)
    ax.set_ylim([0,1])
    ax.set_yticks(np.arange(0,1.25,.25))
    ax.grid(True,ls='dotted')
    
    for i in range(nexps):

        col = inexps[i]['c']
        lbl = inexps[i]['expname_long']
        mrk = inexps[i]['marker']
        ls  = inexps[i]['ls']
        
        if quartile:
            
            if plotmax:
                plotacc = classacc[i,:,:,c].max(0)
            else:
                plotacc = classacc[i,:,:,c].mean(0)
                
        else:
        
            if plotmax:
                plotacc = classacc[i,:,:,c].max(0)
            else:
                plotacc = classacc[i,:,:,c].mean(0)
            
        # Calculate some statistics
        mu        = classacc[i,:50,:,c].mean(0)
        sigma     = classacc[i,:50,:,c].std(0)
        sortacc  = np.sort(classacc[i,:,:,c],0)
        idpct    = sortacc.shape[0] * plotconf
        lobnd   = np.floor(idpct).astype(int)
        hibnd   = np.ceil(sortacc.shape[0]-idpct).astype(int)
        
        # Plot things
        ax.plot(leads,mu,color=col,marker=mrk,alpha=1.0,lw=2.5,label=lbl,zorder=9,ls=ls)
        if add_conf:
            if plotconf:
                ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=alpha,color=col,zorder=-9,label="")
            else:
                ax.fill_between(leads,mu-sigma,mu+sigma,alpha=alpha,color=col,zorder=1)
    ax.plot(persleads,persaccclass[:,c],color=dfcol,label="Persistence",ls="dashed")
    ax.axhline(.33,color=dfcol,label="Random Chance",ls="dotted")
    
    ax.hlines([0.33],xmin=-1,xmax=25,ls="dashed",color='k')
    
    if c == 0:
        ax.set_ylabel("Accuracy")
    if c == 1:
        ax.legend(ncol=2,fontsize=8)
        ax.set_xlabel("Prediction Lead (Years)")

#plt.suptitle(expname)
savename = "%sExperiment_Intercomparison_byclass_plotmax%i_%s.png"% (figpath,plotmax,compname)
plt.savefig(savename,dpi=200,bbox_inches="tight",transparent=True)

#%% Get counts of each class...

classcounts_all = np.zeros((nexps,nleads,nclasses,nruns)) # [experiment, lead, class, run]
predcounts_all  = np.zeros((nexps,nleads,nclasses,nruns))
for d in range(nexps):
    for l in range(nleads):
         # [runs]
        for r in range(nruns):
            
            samples      = ylabs[d][r][l] # The actual counts
            samples_pred = ypred[d][r][l] # The predicted counts
            for c in range(nclasses):
                classcounts_all[d,l,c,r] += (samples == c).sum()
                predcounts_all[d,l,c,r] += (samples_pred == c).sum()

#%% Visualize distributions by class and leadtime

# General plotting options
lwall      = 2.5
darkmode   = False
if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
else:
    plt.style.use('default')
    dfcol = "k"

# Visualize Accuracy by Class
add_conf   = True  # Add confidence intervals
plotconf   = 0.95  # Select which intervals to plot

fig,axs = plt.subplots(1,3,figsize=(18,4))
for c in range(3):
    
    # Initialize plot
    ax = axs[c]
    ax.set_title("%s" %(classes[c]),fontsize=16,)
    ax.set_xlim([0,24])
    ax.set_xticks(leads)
    #ax.set_ylim([125,40])
    #ax.set_ylim([0,1])
    #ax.set_yticks(np.arange(0,1.25,.25))
    ax.grid(True,ls='dotted')
    
    for i in range(nexps):

        col = inexps[i]['c']
        lbl = inexps[i]['expname_long']
        mrk = inexps[i]['marker']
        ls  = inexps[i]['ls']
        
        
        # Calculate some statistics
        mu        = classcounts_all[i,:,c,:].mean(1) # Mean by Run
        sigma     = classcounts_all[i,:,c,:].std(1) # Stdev by run
        
        sortacc   = np.sort(classcounts_all[i,:,c,:],1)
        sortacc   = sortacc.T
        idpct     = sortacc.shape[0] * plotconf
        lobnd     = np.floor(idpct).astype(int)
        hibnd     = np.ceil(sortacc.shape[0]-idpct).astype(int)
        
        # Plot things
        ax.plot(leads,mu,color=col,marker=mrk,alpha=1.0,lw=2.5,label=lbl,zorder=9,ls=ls)
        if add_conf:
            if plotconf:
                ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=.2,color=col,zorder=1,label="")
            else:
                ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.4,color=col,zorder=1)
        
    #ax.plot(leads,persacctotal,color=dfcol,label="Persistence",ls="dashed")
    #ax.axhline(.33,color=dfcol,label="Random Chance",ls="dotted")
    #ax.hlines([0.33],xmin=-1,xmax=25,ls="dashed",color='k')
    
    if c == 0:
        ax.set_ylabel("Accuracy")
    if c == 1:
        ax.legend(ncol=2,fontsize=10)
        ax.set_xlabel("Prediction Lead (Years)")
    
savename = "%sExperiment_Intercomparison_ClassCount_%s.png"% (figpath,compname)
plt.savefig(savename,dpi=200,bbox_inches="tight",transparent=True)

#%%

#%% Unorganized Below --------------------------------------------------------------------------
#%% For a given predictor, visualize the distribution in accuracies

# Leadtime x Variable Plots
binedges = np.arange(0,1.05,0.05)
v        = 0

for c in range(3):
    fig,axs  = plt.subplots(nvar,nleads,figsize=(24,10),sharex=True)
    
    for v in range(nvar):
        for l in range(nleads):
            
            ax = axs[v,l]
            if v == 0:
                ax.set_title("Lead %02i Years" % (leads[l]))
            
            if l == 0:
                ax.text(-0.2, 0.55, varnames[v], va='bottom', ha='center',rotation='vertical',
                                 rotation_mode='anchor',transform=ax.transAxes)
                
            plotvar = classacc[v,:,l,c]
            #h = ax.hist(plotvar)
            
            h = ax.hist(plotvar,binedges,color=varcolors[v],alpha=0.6,
                        label="")
            ax.axvline(plotvar.mean(),color="k",ls='dashed',lw=0.9,
                       label="%.2f" %(plotvar.mean()*100)+"%")
            ax.legend()
            
            ax.set_xticks(binedges[::4])
            ax.set_ylim([0,20])
            ax.grid(True,ls='dotted')
    
    savename = "%sHistograms_By_Leadtime_Variable_Class%s.png" % (figpath,classes[c])
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
#%% Visualize the total accuracy relative to a few baselines

fig,ax = plt.subplots(1,1,figsize=(6,4),sharex=True)
for v in range(nvar):
    
    plotacc   = totalacc[v,:,:]
    mu        = plotacc.mean(0)
    sigma     = plotacc.std(0)
    
    ax.plot(leads,mu,color=varcolors[v],marker="o",alpha=1.0,lw=2,label=varnames[v])
    #ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.05,color=varcolors[v])

ax.plot(leads,persacctotal,color='k',label="Persistence",ls="dashed")
ax.axhline(.33,color='k',label="Random Chance",ls="dotted")

ax.set_xlim([0,24])
ax.set_xticks(leads)
ax.set_ylim([.25,1])
ax.set_yticks(np.arange(.30,1.1,.1))
ax.set_yticklabels((np.arange(.30,1.1,.1)*100).astype(int))
ax.set_ylabel("Total Accuracy (%)")
ax.set_xlabel("Prediction Lead Time (years)")
ax.grid(True,ls='dotted')
ax.legend()

savename = "%sTotal_Accuracy_%s.png" % (figpath,expdirs[expnum])
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot CNN vs NN for a selected variable

v = 0

justbaseline = True

plotconf = 0.05

fsz   = 14
fszt  = 12
fszb  = 16


for v in range(nvar):
    fig,ax = plt.subplots(1,1,figsize=(8,5.5),sharex=True,constrained_layout=True)
    
    # Plotting for each experiment
    if justbaseline is False:
        for ex in range(2):
            totalacc,classacc,ypred,ylabs,shuffids=unpack_expdict(alloutputs[ex])
            
            
            plotacc   = totalacc[v,:,:]
            mu        = plotacc.mean(0)
            sigma     = plotacc.std(0)
            
            
            sortacc  = np.sort(plotacc,0)
            idpct    = sortacc.shape[0] * plotconf
            lobnd   = np.floor(idpct).astype(int)
            hibnd   = np.ceil(sortacc.shape[0]-idpct).astype(int)
            
            
            ax.plot(leads,mu,color=expcolors[ex],marker="o",alpha=1.0,lw=2.5,label=expnames[ex] + " (mean)",zorder=9)
            if plotconf:
                ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=.3,color=expcolors[ex],zorder=1,label=expnames[ex]+" (95% conf.)")
            else:
                ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.4,color=expcolors[ex],zorder=1)
        
    ax.plot(leads,persacctotal,color=dfcol,label="Persistence",ls=":")
    ax.axhline(.33,color=dfcol,label="Random Chance",ls="-")
    
    
    ax.set_xlim([0,24])
    ax.set_xticks(leads,fontsize=fszt)
    ax.set_ylim([.25,1])
    ax.set_yticks(np.arange(.30,1.1,.1))
    ax.set_yticklabels((np.arange(.30,1.1,.1)*100).astype(int),fontsize=fszt)
    ax.set_ylabel("Accuracy (%)",fontsize=fsz)
    ax.set_xlabel("Prediction Lead Time (Years)",fontsize=fsz)
    ax.grid(True,ls='dotted')
    ax.legend(fontsize=fsz)
    ax.set_title("Total Accuracy, Predictor: %s" % (varnames[v]),fontsize=fszb)
    
    if justbaseline:
        savename = "%sTotalAcc_CNNvFNN_conf%03i_baselineonly.png" % (figpath,plotconf*100)
    else:
        savename = "%sTotalAcc_CNNvFNN_%s_conf%03i.png" % (figpath,varnames[v],plotconf*100)
    
    print(savename)
    plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=True)
    #ax.set_title("")
    
    
#%% Same as above for for each class


fsz   = 10
fszt  = 8
fszb  = 14

ylower = 0.2
add_conf = False
incl_title = False

for v in range(nvar):
    fig,axs = plt.subplots(3,1,figsize=(3,10),sharex=True,)
    
    # Plotting for each experiment
    for c in range(3):
        ax = axs[c]
        viz.label_sp(classes[c],labelstyle="%s",usenumber=True,ax=ax,alpha=0,fontcolor=threscolors[c])
        for ex in range(2):
            totalacc,classacc,ypred,ylabs,shuffids=unpack_expdict(alloutputs[ex])
            
            plotacc   = classacc[v,:,:,c]
            # ------
            mu        = plotacc.mean(0)
            sigma     = plotacc.std(0)
            
            sortacc  = np.sort(plotacc,0)
            idpct    = sortacc.shape[0] * plotconf
            lobnd   = np.floor(idpct).astype(int)
            hibnd   = np.ceil(sortacc.shape[0]-idpct).astype(int)
            
            ax.plot(leads,mu,color=expcolors[ex],marker="o",alpha=1.0,lw=2.5,label=expnames[ex] + " (mean)",zorder=9)
            if add_conf:
                if plotconf:
                    ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=.3,color=expcolors[ex],zorder=1,label=expnames[ex]+" (95% conf.)")
                else:
                    ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.4,color=expcolors[ex],zorder=1)
            
            
        ax.plot(leads,persacctotal,color=dfcol,label="Persistence",ls="dashed")
        ax.axhline(.33,color=dfcol,label="Random Chance",ls="dotted")
        
        
        ax.set_xlim([0,24])
        ax.set_xticks(leads,fontsize=fszt)
        ax.set_ylim([ylower,1])
        ax.set_yticks(np.arange(ylower,1.1,.1))
        ax.set_yticklabels((np.arange(ylower,1.1,.1)*100).astype(int),fontsize=fszt)
        
        ax.grid(True,ls='dotted')
        
        if c == 0:
            if incl_title:
                ax.set_title("Total Accuracy, Predictor: %s" % (varnames[v]),fontsize=fszb)
        if c == 1:
            #ax.legend(fontsize=fszt,ncol=3)
            ax.set_ylabel("Accuracy (%)",fontsize=fsz)
        if c == 2:
            ax.set_xlabel("Prediction Lead Time (Years)",fontsize=fsz)
    
    savename = "%sClassAcc_CNNvFNN_%s_conf%03i.png" % (figpath,varnames[v],plotconf*100)
    print(savename)
    plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=True)
    #ax.set_title("")

#%% Cross Cross Validation Experiments. Quantify the cross-fold variance



#%% In this section below, we make some comparisons of skill by experiment...

#%% Sort by accuracies

#%% Get (in order) 

plotvar = classacc[v,:,l,c]
fig,ax = plt.subplots(1,1)



#%% Invetigating shuffled indices

shuffids_pred = shuffids[0] # [predictor][run][lead,sample]

# It seems that for a given leadtime, the indices of samples used are the same
r = 4
id_run0 = shuffids_pred[r][0,:].astype(int)
id_run1 = shuffids_pred[r][0,:].astype(int)
plt.plot(id_run0-id_run1)

# However, for the train/test/val split, the indices are different...
