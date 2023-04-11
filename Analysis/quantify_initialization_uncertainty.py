#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Quantify uncertainty due to network weight intialization...

Upper section taken from [test_predictor_uncertainty.py]

Created on Fri Apr  7 07:21:19 2023

@author: gliu
"""

import numpy as np
import sys
import glob
import importlib
import copy
import xarray as xr

import torch
from torch import nn

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from tqdm import tqdm
import time
import os

from torch.utils.data import DataLoader, TensorDataset,Dataset
#%% Load some functions

#% Load custom packages and setup parameters
# Import general utilities from amv module
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import proc


# Import packages specific to predict_amv
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import train_cesm_params as train_cesm_params
import amv_dataloader as dl
import amvmod as am



#%% User Edits

expdir             = "FNN4_128_SingleVar_Rerun100_consistent"
modelname          = "FNN4_128"
varnames           = ["SST","SSH","PSL"] 


nmodels            = 100 # Specify manually how much to do in the analysis
leads              = np.arange(0,26,1)


# Load parameters from [oredict_amv_param.py]
datpath            = pparams.datpath
figpath            = pparams.figpath
nn_param_dict      = pparams.nn_param_dict
class_colors       = pparams.class_colors
classes            = pparams.classes

# Load some relevant parameters from [train_cesm1_params.py]
eparams            = train_cesm_params.train_params_all[expdir] # Load experiment parameters
ens                = eparams['ens']

checkgpu           = True

#%% Load data (taken from train_NN_CESM1.py)

# Load some variables for ease


# Load predictor and labels, lat/lon, cut region
target         = dl.load_target_cesm(detrend=eparams['detrend'],region=eparams['region'])
data,lat,lon   = dl.load_data_cesm(varnames,eparams['bbox'],detrend=eparams['detrend'],return_latlon=True)

# Create classes 
# Set exact threshold value
std1         = target.std(1).mean() * eparams['thresholds'][1] # Multiple stdev by threshold value 
if eparams['quantile'] is False:
    thresholds_in = [-std1,std1]
else:
    thresholds_in = eparams['thresholds']

# Classify AMV Events
target_class = am.make_classes(target.flatten()[:,None],thresholds_in,exact_value=True,reverse=True,quantiles=eparams['quantile'])
target_class = target_class.reshape(target.shape)

# Subset predictor by ensemble, remove NaNs, and get sizes
data                           = data[:,0:ens,...]      # Limit to Ens
data[np.isnan(data)]           = 0                      # NaN Points to Zero
nchannels,nens,ntime,nlat,nlon = data.shape             # Ignore year and ens for now...
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN
nclasses = len(eparams['thresholds']) + 1
nlead    = len(leads)

#%% Load model weights (taken from LRP_LENS.py and viz_acc_byexp)

modweights_byvar = [] # [variable][lead][runid][?]
modlist_byvar    = []
flists           = []
for v,varname in enumerate(varnames):
    
    # Get the model weights
    modweights_lead,modlist_lead=am.load_model_weights(datpath,expdir,leads,varname)
    modweights_byvar.append(modweights_lead)
    modlist_byvar.append(modlist_lead)
    
    
    # Get list of metric files
    search = "%s%s/Metrics/%s" % (datpath,expdir,"*%s*" % varname)
    flist  = glob.glob(search)
    flist  = [f for f in flist if "of" not in f]
    flist.sort()
    flists.append(flist)
    print("Found %i files for %s using searchstring: %s" % (len(flist),varname,search))
    
# Get the shuffled indices
expdict = am.make_expdict(flists,leads)

# Unpack Dictionary
totalacc,classacc,ypred,ylabs,shuffids = am.unpack_expdict(expdict)
# shuffids [predictor][run][lead][nsamples]


totalacc = np.array(totalacc)
classacc = np.array(classacc)
#%% Load the baselines
persleads,pers_class_acc,pers_total_acc = dl.load_persistence_baseline("CESM1",
                                                                        return_npfile=False,region=eparams['region'],quantile=eparams['quantile'],
                                                                        detrend=eparams['detrend'],limit_samples=True,nsamples=eparams['nsamples'],repeat_calc=1)


#%% First, let's briefly check the accuracy

acolors = ("red","blue","orange")

# First check total accuracy
ii = 0
fig,ax = plt.subplots(1,1,constrained_layout=True)
for pvar in range(3):

    plotacc_model = totalacc[pvar,:,:]
    
    for r in range(nmodels):
        ax.plot(leads,plotacc_model[r,:],color=acolors[ii],alpha=0.05)
        
    mu    = plotacc_model.mean(0)
    sigma = plotacc_model.std(0)
    ax.plot(leads,mu,color=acolors[pvar],label=varnames[pvar])
    ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.15,color=acolors[pvar])
    ii+=1

ax.plot(persleads,pers_total_acc,color="k",ls="dashed",label="Persistence Baseline")
ax.axhline([.33],color="gray",ls="dashed",lw=0.75,label="Random Chance Baseline")

ax.legend()
ax.set_xlabel("Prediction Lead (Years)")
ax.set_ylabel("Accuracy")
ax.set_xticks(leads)
ax.grid(True,ls="dotted")
ax.set_xlim([0,25])
ax.set_yticks(np.arange(0,1.25,.25))
ax.set_xticks(np.arange(0,25,3))
ax.set_title("Total Accuracy for Predicting AMV (Consistent Samples)")
savename = "%sNN_InitWeight_Uncertainty_Consistent_Sample_Test_TotalAcc_%s.png" % (figpath,expdir)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Plot accuracy by class

plotint = 3

acolors = ("red","blue","orange")

# First check total accuracy

fig,axs = plt.subplots(1,3,constrained_layout=True,figsize=(16,4))
for c in range(3):
    
    ax = axs[c]
    ii = 0
    for pvar in range(3):
    
        plotacc_model = classacc[pvar,:,:,c]
        
        for r in range(nmodels):
            ax.plot(leads[::plotint],plotacc_model[r,::plotint],color=acolors[ii],alpha=0.05)
            
        mu    = plotacc_model.mean(0)
        sigma = plotacc_model.std(0)
        ax.plot(leads[::plotint],mu[::plotint],color=acolors[pvar],label=varnames[pvar])
        ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.15,color=acolors[pvar])
        ii+=1
    
    ax.plot(persleads,pers_class_acc[...,c],color="k",ls="dashed",label="Persistence Baseline")
    ax.axhline([.33],color="gray",ls="dashed",lw=0.75,label="Random Chance Baseline")
    
    if c == 1:
        ax.legend()
    ax.set_xlabel("Prediction Lead (Years)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(leads)
    ax.grid(True,ls="dotted")
    ax.set_xlim([0,25])
    ax.set_yticks(np.arange(0,1.25,.25))
    ax.set_xticks(np.arange(0,25,3))
    ax.set_title(classes[c])
    savename = "%sNN_InitWeight_Uncertainty_Consistent_Sample_Test_ClassAcc_%s.png" % (figpath,expdir)
    plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Do monte-carlo sampling of data and compute resultant accuracy


network_numbers = [10,20,30,40,50,60,70,80,90,100]
N_mc            = 1000
N_numbers       = len(network_numbers)

mc_variance     = np.zeros((len(varnames),3,N_numbers,N_mc,nlead)) # [predictor,class,network_number,iteration]

for c in range(3):
    
    for pvar in range(3):
        
        inacc = classacc[pvar,:,:,c] # [nrun,nlead]
        
        for netnum in tqdm(range(len(network_numbers))):
            
            for N in range(N_mc):
            
                # Initialize indices
                nn_idx = np.arange(0,nmodels,1)
                
                # Shuffle and select
                np.random.shuffle(nn_idx)
                sel_indices = nn_idx[:network_numbers[netnum]]
                
                # Compute variances
                inacc_sel = inacc[sel_indices,:]
                mc_variance[pvar,c,netnum,N,:] = inacc_sel.var(0)
                
                
#%% Plot variance in accuracy vs. # of networks included for a GIVEN LEADTIME

varcolors = ["red","blue","orange"]
l         = -1 # Indicate selected leadtime

fig,ax = plt.subplots(1,1)

for pvar in range(3):
    
    plot_uncerts = mc_variance[pvar,:,:,:,l] # [class,netnum,iteration]
    plot_uncerts = plot_uncerts.mean(0)      # Take class mean [netnum,iteration]
    
    
    for N in range(N_mc):
        ax.plot(network_numbers,plot_uncerts[:,N],alpha=0.01,color=varcolors[pvar],label="")
    mu    = plot_uncerts.mean(1)
    sigma = plot_uncerts.std(1)
    
    ax.plot(network_numbers,mu,color=varcolors[pvar],label=varnames[pvar])
    ax.fill_between(network_numbers,mu-sigma,mu+sigma,alpha=.15,color=varcolors[pvar],zorder=-9)

ax.set_title("Number of Networks vs. Variance of Accuracy (lead = % i)" % (leads[l]))
ax.set_xlim([5,105])
ax.set_ylim([0,0.025])

ax.set_xlabel("Number of Networks Included")
ax.set_ylabel("Variance in Accuracy")

ax.legend()

plt.savefig("%sNetwork_Initialization_Error_%s.png" % (figpath,expdir),dpi=150)


#%%
varcolors = ["red","blue","orange"]

l         = -1

fig,ax = plt.subplots(1,1)

for pvar in range(3):
    
    plot_uncerts = mc_variance[pvar,:,:,:,l] # [class,netnum,iteration]
    plot_uncerts = plot_uncerts.mean(0)      # Take class mean [netnum,iteration]
    meanval      = plot_uncerts.mean() # Mean or "true" value of the variance
    
    plot_uncerts = np.abs(plot_uncerts - meanval)
    
    
    # for N in range(N_mc):
    #     ax.plot(network_numbers,plot_uncerts[:,N],alpha=0.01,color=varcolors[pvar],label="")
    mu    = plot_uncerts.mean(1)
    sigma = plot_uncerts.std(1)
    
    ax.plot(network_numbers,mu,color=varcolors[pvar],label=varnames[pvar] + " ($\mu$ = %.2e)" % (meanval))
    ax.fill_between(network_numbers,mu-sigma,mu+sigma,alpha=.10,color=varcolors[pvar],zorder=-9)

ax.set_title("Number of Networks vs. Variance of Accuracy (lead = % i)" % (leads[l]))
ax.set_xlim([5,105])
ax.set_ylim([0,0.004])
ax.legend()

ax.set_xlabel("Number of Networks Included")
ax.set_ylabel("Abs. Deviation from Mean Variance in Accuracy")





                
            



