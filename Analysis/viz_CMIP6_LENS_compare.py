#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Output from LRP_CMIP6_LENS_compare.py

Created on Thu Feb 16 06:14:27 2023

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

import cmocean as cmo
import os

#%% Load modules (LRPutils by Peidong)
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/scrap/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/predict_amv/")
import LRPutils as utils
import amvmod as am

# Load visualization module
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz,proc

#%% Import params
# Note; Need to set script into current working directory (need to think of a better way)
import os
cwd = os.getcwd()

sys.path.append(cwd+"/../")
import predict_amv_params as pparams

classes         = pparams.classes
proj            = pparams.proj
figpath         = pparams.figpath
proc.makedir(figpath)

bbox            = pparams.bbox
nn_param_dict   = pparams.nn_param_dict
proj            = pparams.proj


leadticks25     = pparams.leadticks25
leadticks24     = pparams.leadticks24

dataset_names   = pparams.cmip6_names
dataset_colors  = pparams.cmip6_colors
dataset_markers = pparams.cmip6_markers
cmip6_dict      = pparams.cmip6_dict

#%% Define some functions

def load_lrp(lrp_dict):
    keys = ("composites_lead","idcorrect_lead","modelacc_lead")
    #    "composites_lead" : composites_lead,  # [lead][compositeN,class,channels,lat,lon]
    #    "idcorrect_lead"  : idcorrect_lead,   # [lead][class][model][correct_samples]
    #    "modelacc_lead"   : modelacc_lead ,   # [lead][model x class]
    return [lrp_dict[key] for key in keys]

#%% User Edits

datpath          = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/CMIP6_LENS/LRP/"
restrict_ens     = 25
skip_dataset     = ('MPI-ESM1-2-LR',)
varname          = "ssh"
composite_topNs  = (1,5,10,25,50)
leads            = np.arange(0,26,1)





#%% Load LRP ... 
# Delete skipped datasets and get count
dataset_names   = [name for name in dataset_names if name not in skip_dataset]
ndata           =  len(dataset_names)

# Load Latlon
npz_latlon = np.load("%sLRP_Output_LATLON.npz" % datpath,allow_pickle=True)
lon = npz_latlon['lon']
lat = npz_latlon['lat']

# Load Lat/Lon
lrp_dicts       = {}
for id_train in range(ndata):
    train_name  = dataset_names[id_train]
    trained_by_data_dict = {} # Dictionary containing results of NN trained by "train_data"
    for id_test in range(ndata):
        
        test_name = dataset_names[id_test]
        ld = np.load("%s%s/LRP_Output_FNN4_128_SingleVar_%s_Train_TEST_%s_ens%03i.npz" % (datpath,varname,
                                                                                      train_name,test_name,
                                                                                      restrict_ens),allow_pickle=True)
        trained_by_data_dict[test_name] = ld
        # End loop by testing datasets
    # Store in overall dict by train_name
    lrp_dicts[train_name] = trained_by_data_dict.copy()

#%% Plot Settings

id_topN  = 3  # Top 25 Models
lead     = 25 # Leadtime to plot
id_class = 0  #

clims          = [0,1e-2]
normalize_maps = False
label_norm     = True

cmap = cmo.cm.amp # "RdBu_r"
plotname  = ""
#%% Make LRP Composites

for lead in leads:
    if clims is None:
        figsize = (20,16)
    else:
        figsize = (20,17.5)
    
    # Make a bunch of plots
    fig,axs = plt.subplots(5,5,subplot_kw={"projection":proj},
                           figsize=figsize,constrained_layout=True,)
    
    # Loop n plot
    for id_train in range(ndata):
        
        train_name = dataset_names[id_train]
        for id_test in range(ndata):
            
            
            test_name = dataset_names[id_test]
            
            # Get Axis
            ax = axs[id_train,id_test]
            
            # Do some labeling
            blabels = [0,0,0,0]
            if id_test == 0:  # First Column
                ax.text(-0.15, 0.55, train_name, va='bottom', ha='center',rotation='vertical',
                        rotation_mode='anchor',transform=ax.transAxes)
                blabels[0] = 1 # Add Left Latitude Label
            if id_train == 0: # First row, add titles
                ax.set_title(test_name)
            if id_train == (ndata-1):
                blabels[-1] = 1 # Add Bottom Longitude Label
            
            # Make the PLot
            plotlrp = lrp_dicts[train_name][test_name]["composites_lead"] # [Lead x TopN, Class, Channel, Lat, Lon]
            plotlrp = plotlrp[lead,id_topN,id_class,0,:,:]
            nfactor = np.nanmax(np.abs(plotlrp).flatten())
            if normalize_maps:
                plotlrp = plotlrp / np.nanmax(np.abs(plotlrp).flatten())
            if clims is None:
                pcm     = ax.pcolormesh(lon,lat,plotlrp,cmap=cmap)
                fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.035)
            else:
                pcm     = ax.pcolormesh(lon,lat,plotlrp,vmax=clims[1],vmin=clims[0],cmap=cmap)
            
            # Label with the accuracy
            plotacc   = lrp_dicts[train_name][test_name]["modelacc_lead"]  # [Lead, Model, Class]
            plotmu    = plotacc[lead,:,id_class].mean(0)
            plotstd   = plotacc[lead,:,id_class].std(0)
            if label_norm:
                splabel   = "%.02f$\pm$%.02f" % (plotmu*100,plotstd*100) + "%"
                splabel += ", nfactor=%.02e" % nfactor
            else:
                splabel   = "%.02f$\pm$%.02f" % (plotmu*100,plotstd*100) + "%"
            ax = viz.label_sp(splabel,ax=ax,alpha=0.75,labelstyle="%s",usenumber=True)
            
            ax = viz.add_coast_grid(ax,bbox=bbox,proj=proj,
                                    blabels=blabels,fill_color="lightgray",ignore_error=True)
    
    if clims is not None:
        fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.020,pad=0.01)
    plt.suptitle("%s %s LRP Composites for Top %i Models, Prediction Lead = %02i Years" % (classes[id_class],varname,composite_topNs[id_topN],lead),fontsize=26)
    savename = "%sCMIP6_LENS_Intercomparison_%s_LRP_%s_lead%02i_top%02i_normalize%i.png" % (figpath,varname,classes[id_class],lead,composite_topNs[id_topN],normalize_maps)
    plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Plot model accuracies (for each test dataset)

add_conf  = True
plotconf  = 0.95

for id_test in range(ndata):
    test_name = dataset_names[id_test]
            
    # Initialize plot
    fig,axs = plt.subplots(1,3,figsize=(18,4))
    
    for c in range(3):
        ax = axs[c]
        ax.set_title("%s" %(classes[c]),fontsize=16,)
        ax.set_xlim([0,24])
        ax.set_xticks(leadticks24)
        ax.set_ylim([0,1])
        ax.set_yticks(np.arange(0,1.25,.25))
        ax.grid(True,ls='dotted')
        
        # Loop n plot
        for id_train in range(ndata):
            train_name = dataset_names[id_train]
            
            # Set plotting colors
            mrk = cmip6_dict[train_name]['mrk']
            col =cmip6_dict[train_name]['col']
            
            # Get the plotting accuracies
            plotacc   = lrp_dicts[train_name][test_name]["modelacc_lead"] # [Lead, Model, Class]
            plotacc   = plotacc[:,:,id_class]
            
            # Compute mean, stdev, low and hi bounds
            mu    = plotacc.mean(1)
            sigma = plotacc.std(1)
            sortacc  = np.sort(plotacc,1)
            idpct    = sortacc.shape[0] * plotconf
            lobnd   = np.floor(idpct).astype(int)
            hibnd   = np.ceil(sortacc.shape[0]-idpct).astype(int)
            
            # Plot things
            lbl = train_name
            ls  = "solid"
            
            ax.plot(leads,mu,color=col,marker=mrk,alpha=1.0,lw=2.5,label=lbl,zorder=9,ls=ls)
            if add_conf:
                if plotconf:
                    ax.fill_between(leads,sortacc[:,lobnd],sortacc[:,hibnd],alpha=.2,color=col,zorder=1,label="")
                else:
                    ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.4,color=col,zorder=1)
            # End Train Loop
        if c == 0:
            ax.legend()
        # End Class Loop
    plt.suptitle("Accuracy by Class (Test Dataset=%s, Predictor=%s)" % (test_name,varname.upper()),fontsize=22,y=1.05)
    savename = "%sAccByClass_Test_%s_%s.png" % (figpath,test_name,varname)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
            
            
#%% Plot overall model performance (across all datasets):

plotconf          = False # Somethin going funky with this...
mean_bydata_first = True
nleads,nmodels,nclasses         = lrp_dicts[train_name][test_name]["modelacc_lead"].shape

# Reassign
mean_acc_all  = np.zeros((nleads,nmodels,nclasses,ndata,ndata)) # [lead,model,class,train_set,test_set]
#mean_acc_bytrain = mean_acc_bytest.copy()            # Accuracy by Training Set
for id_test in range(ndata):
    test_name = dataset_names[id_test]
    for id_train in range(ndata):
        train_name = dataset_names[id_train]
        getacc     = lrp_dicts[train_name][test_name]["modelacc_lead"]
        mean_acc_all[...,id_train,id_test] = getacc.copy()

            
# Initialize plot
fig,axs = plt.subplots(2,3,figsize=(18,6.5),constrained_layout=True)
for row in range(2): # Looop for groupin by train and grouping by test

    if row == 0: # Plot by train on first row
        if mean_bydata_first:
            meanacc = mean_acc_all.mean(4) # Average over testing sets to group results by training
        ylab    = "Acc. by Trmean_bydata_firstaining Set"
    else: # Plot by test on second row
        if mean_bydata_first:
            meanacc = mean_acc_all.mean(3) # Average over training sets to group results by testing
        ylab    = "Acc. by Testing Set"

    for c in range(3):
        
        ax = axs[row,c]
        if row == 0:
            ax.set_title("%s" %(classes[c]),fontsize=16,)
        if c == 0:
            ax.text(-0.15, 0.55, ylab, va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
        ax.set_xlim([0,24])
        ax.set_xticks(leadticks24)
        ax.set_ylim([0,1])
        ax.set_yticks(np.arange(0,1.25,.25))
        ax.grid(True,ls='dotted')    
        
        # Loop n plot
        for id_data in range(ndata):
            
            data_name = dataset_names[id_data]
            
            # Set plotting colors
            mrk = cmip6_dict[data_name]['mrk']
            col = cmip6_dict[data_name]['col']
            
            if mean_bydata_first:
                # Get the plotting accuracies
                plotacc   = meanacc[:,:,c,id_data]
                
                # Compute mean, stdev, low and hi bounds
                mu        = plotacc.mean(1)
                sigma     = plotacc.std(1)
                sortacc   = np.sort(plotacc,1)
                
                idpct    = sortacc.shape[0] * plotconf
                lobnd    = np.floor(idpct).astype(int)
                hibnd    = np.ceil(sortacc.shape[0]-idpct).astype(int)
            else: # take mean over datasets last
                if row == 0:
                    plotacc = mean_acc_all[:,:,c,id_data,:] # [lead model acc_by_train_set] (for a given train set)
                else:
                    plotacc = mean_acc_all[:,:,c,:,id_data] # [lead model acc_by_test_set] (for a given test set)
                
                # Compute mean, stdev, low and hi bounds
                mu        = plotacc.mean(1).mean(1) # [lead acc_by_[row]_set]
                sigma     = plotacc.std(1).mean(1)  
                sortacc   = np.sort(plotacc,1)
                
                idpct    = sortacc.shape[0] * plotconf
                lobnd    = np.floor(idpct).astype(int)
                hibnd    = np.ceil(sortacc.shape[0]-idpct).astype(int)
                
            # Plot things
            lbl = data_name
            ls  = "solid"
            
            ax.plot(leads,mu,color=col,marker=mrk,alpha=1.0,lw=2.5,label=lbl,zorder=9,ls=ls)
            if add_conf:
                if plotconf:
                    ax.fill_between(leads,sortacc[:,lobnd],sortacc[:,hibnd],alpha=.2,color=col,zorder=1,label="")
                else:
                    ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.2,color=col,zorder=1)
            # End Train Loop
        if c == 0:
            ax.legend()
        # End Class Loop
    savename = "%sAccByClass_TrainTestSetMean_%s.png" % (figpath,varname)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
