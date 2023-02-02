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
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
sys.path.append("../")
import viz,proc
import amvmod as am # Import amv module for predict amv
#%% Import common parameters from file

import predict_amv_params as pparams

# Import data paths
datpath = pparams.datpath
figpath = pparams.figpath
proc.makedir(figpath)

# Import class information
classes         = pparams.classes
class_colors    = pparams.class_colors

# Import other information
leads           = np.arange(0,26,1)#pparams.leads

#%% User Edits

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

# ---- CESM1_res_1v3 ----

exp0 = {"expdir"        : "FNN4_128_SingleVar"   , # Directory of the experiment
        "searchstr"     :  "*SST*"               , # Search/Glob string used for pulling files
        "expname"       : "CESM1_Original"       , # Name of the experiment (Short)
        "expname_long"  : "CESM1 (1-deg. SST)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "b"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }

exp1 = {"expdir"        : "FNN4_128_ALL_CESM1_Train" , # Directory of the experiment
        "searchstr"     :  "*", # Search/Glob string used for pulling files
        "expname"       : "CESM1_3deg"           , # Name of the experiment (Short)
        "expname_long"  : "CESM1 (3-deg. SST)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "r"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "dashed"               , # Linestyle for plotting
        }

inexps   = [exp0,exp1] # Put in experiments here...
compname = "CESM1_res_1v3"# CHANGE THIS for each new comparison


#%%

exp0 = {"expdir"        : "LENS_30_1950/FNN4_128_ALL_canesm2_lens_Train/"   , # Directory of the experiment
        "searchstr"     :  "*"               , # Search/Glob string used for pulling files
        "expname"       : "canesm2_lens"       , # Name of the experiment (Short)
        "expname_long"  : "CCCma-CanESM2"   , # Long name of the experiment (for labeling on plots)
        "c"             : "r"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }

exp1 = {"expdir"        : "LENS_30_1950/FNN4_128_ALL_csiro_mk36_lens_Train/"   , # Directory of the experiment
        "searchstr"     :  "*"               , # Search/Glob string used for pulling files
        "expname"       : "csiro_mk36_lens"       , # Name of the experiment (Short)
        "expname_long"  : "CSIRO-MK3.6"   , # Long name of the experiment (for labeling on plots)
        "c"             : "b"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }

exp2 = {"expdir"        : "LENS_30_1950/FNN4_128_ALL_gfdl_esm2m_lens_Train/"   , # Directory of the experiment
        "searchstr"     :  "*"               , # Search/Glob string used for pulling files
        "expname"       : "gfdl_esm2m_lens"       , # Name of the experiment (Short)
        "expname_long"  : "GFDL-ESM2M"   , # Long name of the experiment (for labeling on plots)
        "c"             : "magenta"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }


exp3 = {"expdir"        : "LENS_30_1950/FNN4_128_ALL_mpi_lens_Train/"   , # Directory of the experiment
        "searchstr"     :  "*"               , # Search/Glob string used for pulling files
        "expname"       : "mpi_lens"       , # Name of the experiment (Short)
        "expname_long"  : "MPI-ESM-LR"   , # Long name of the experiment (for labeling on plots)
        "c"             : "gold"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }


exp4 = {"expdir"        : "LENS_30_1950/FNN4_128_ALL_CESM1_Train/"   , # Directory of the experiment
        "searchstr"     :  "*"               , # Search/Glob string used for pulling files
        "expname"       : "CESM1"       , # Name of the experiment (Short)
        "expname_long"  : "NCAR-CESM1"   , # Long name of the experiment (for labeling on plots)
        "c"             : "limegreen"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }

inexps   = [exp0,exp1,exp2,exp3,exp4] # Put in experiments here...
compname = "CMIP5_Lens_30_1950"# CHANGE THIS for each new comparison
quartile = True

#%% Locate the files

nexps = len(inexps)
flists = []
for ex in range(nexps):
    
    search = "%s%s/Metrics/%s" % (datpath,inexps[ex]["expdir"],inexps[ex]["searchstr"])
    flist  = glob.glob(search)
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
expdict = am.make_expdict(flists,leads)

# Gather some dimension information for plotting
if quartile is True:
    nruns    = expdict['classacc'].shape[1]
    nleads   = expdict['classacc'][0][0].shape[0]
    nclasses = expdict['classacc'][0][0].shape[1]
else:
    _,nruns,nleads,nclasses      = expdict['classacc'].shape

# Unpack Dictionary
totalacc,classacc,ypred,ylabs,shuffids = am.unpack_expdict(expdict)

#%% Load the persistence baseline

fpath = "../Data/Metrics/"
fnp ="leadtime_testing_ALL_AMVClass3_PersistenceBaseline_1before_nens40_maxlead24_detrend0_noise0_nsample400_limitsamples1_ALL_nsamples1.npz"

ldp = np.load(fpath+fnp,allow_pickle=True)#.f#.arr_0

persaccclass = np.array(ldp['arr_0'][None][0]['acc_by_class']) # [Lead x Class]}
persacctotal = np.array(ldp['arr_0'][None][0]['total_acc'])
# ------------------------
#%% Do some visualizations
# ------------------------

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
plotmax    = False # Set to True to plot maximum
add_conf   = True  # Add confidence intervals
plotconf   = 0.95  # Select which intervals to plot

fig,axs = plt.subplots(1,3,figsize=(18,4))
for c in range(3):
    
    # Initialize plot
    ax = axs[c]
    ax.set_title("%s" %(classes[c]),fontsize=16,)
    ax.set_xlim([0,24])
    ax.set_xticks(leads)
    ax.set_ylim([0,1])
    ax.set_yticks(np.arange(0,1.25,.25))
    ax.grid(True,ls='dotted')
    
    for i in range(nexps):

        col = inexps[i]['c']
        lbl = inexps[i]['expname_long']
        mrk = inexps[i]['marker']
        
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
        ax.plot(leads,mu,color=col,marker=mrk,alpha=1.0,lw=2.5,label=lbl,zorder=9)
        if add_conf:
            if plotconf:
                ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=.2,color=col,zorder=1,label="")
            else:
                ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.4,color=col,zorder=1)
        
    #ax.plot(leads,persacctotal,color=dfcol,label="Persistence",ls="dashed")
    ax.axhline(.33,color=dfcol,label="Random Chance",ls="dotted")
    
    ax.hlines([0.33],xmin=-1,xmax=25,ls="dashed",color='k')
    
    if c == 0:
        ax.set_ylabel("Accuracy")
    if c == 1:
        ax.legend(ncol=2,fontsize=10)
        ax.set_xlabel("Prediction Lead (Years)")
    
savename = "%sExperiment_Intercomparison_byclass_plotmax%i_%s.png"% (figpath,plotmax,compname)
plt.savefig(savename,dpi=200,bbox_inches="tight",transparent=True)
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

#%% In this section below, we make some comparisons of skill by experiment...

#%% Sort by accuracies

#%% Get (in order) 

plotvar = classacc[v,:,l,c]
fig,ax = plt.subplots(1,1)
