#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Perform LRP for LENS data...

(1) Load in predictors and labels for the given model


Copied sections from viz_regional_predictability
Currently runs on [Astraeus] data paths

Created on Fri Feb  3 09:05:15 2023

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


#%%

cmipver        = 6
varname        = "SSH"
modelname      = "FNN4_128"
leads          = np.arange(0,26,1)
dataset_name   = "IPSL-CM6A-LR"

# LRP Settings (note, this currently uses the innvestigate package from LRP-Pytorch)
gamma          = 0.1
epsilon        = 0.1
innexp         = 2
innmethod      ='b-rule'
innbeta        = 0.1

# Labeling for plots and output files
ge_label       = "exp=%i, method=%s, $beta$=%.02f" % (innexp,innmethod,innbeta)
ge_label_fn    = "innexp%i_%s_innbeta%.02f" % (innexp,innmethod,innbeta)

# lrp methods
sys.path.append("/Users/gliu/Downloads/02_Research/03_Code/github/Pytorch-LRP-master/")
from innvestigator import InnvestigateModel

# Load modules (LRPutils by Peidong)
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/scrap/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/predict_amv/")
import LRPutils as utils
import amvmod as am

# Load visualization module
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz,proc

#%% IMport params
# Note; Need to set script into current working directory (need to think of a better way)
import os
cwd = os.getcwd()

sys.path.append(cwd+"/../")
import predict_amv_params as pparams

classes    = pparams.classes
proj       = pparams.proj
figpath    = pparams.figpath
proc.makedir(figpath)


bbox          = pparams.bbox
nn_param_dict = pparams.nn_param_dict

#%% Load some other things

# Set Paths based on CMIP version
if cmipver == 5:
    datpath        = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/processed/"
    modepath       = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/LENS_30_1950/"
elif cmipver == 6:
    datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/CMIP6_LENS/processed/"
    modpath        = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/CMIP6_LENS/models/"

# Compute some dimensions
nleads         = len(leads)

# Set preprocessing options based on cmip version

if cmipver == 5:
    dataset_names = pparams.dataset_names
    ystarts       = pparams.dataset_starts
    limit_time    = [1950,2005] # Set Dates here to limit the range of the variable
    ens           = 30
    regrid        = 3
    
elif cmipver == 6:
    varname       = varname.lower()
    dataset_names = pparams.cmip6_names[1:-1]
    ystarts       = (1850,)*len(dataset_names)
    varnames      = ("sst","ssh","sss")
    limit_time    = [1850,2014] # Set Dates here to limit the range of the variable
    ens           = 25
    regrid        = None
    

quantile      = True
thresholds    = [1/3,2/3]
tstep         = limit_time[1] - limit_time[0] + 1
percent_train = 0.8
detrend       = 0
outsize       = 3
lp            = 0

#%% Load Predictors (works just for CMIP6 for now)

# Load predictor
ncname  = "%s/%s_%s_NAtl_%ito%i_detrend%i_regrid%sdeg.nc" % (datpath,dataset_name,
                                                                       varname,
                                                                       1850,2014,
                                                                       detrend,regrid)
ds      = xr.open_dataset(ncname)
ds      = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3])) 
data    = ds[varname].values[None,...] # [echannel x ensemble x year x lat x lon]

# Load labels
lblname = "%s/%s_sst_label_%ito%i_detrend%i_regrid%sdeg_lp%i.npy" % (datpath,dataset_name, #Mostly compied from NN_traiing script
                                                                     1850,2014,
                                                                     detrend,regrid,lp)
target  = np.load(lblname) # [ensemble x year]



# Limit to input to ensemble member
data = data[:,0:ens,...] 
data[np.isnan(data)] = 0
nchannels,nens,ntime,nlat,nlon = data.shape # Ignore year and ens for now...
inputsize               = nchannels*nlat*nlon # Compute inputsize to remake FNN

# Load Lat/Lon
lat = ds.lat.values
lon = ds.lon.values
print(nlon),print(nlat)


#%% Get list of Model Weights

# Make the experiment directory
expdir = "%s_SingleVar_%s_Train" % (modelname,dataset_name)

# Pull model list
modlist_lead = []
modweights_lead = []
for lead in leads:
    # Get Model Names
    modlist = glob.glob("%s%s/Models/*%s*.pt" % (modpath,expdir,varname))
    modlist.sort()
    print("Found %i models for %s, Lead %i" % (len(modlist),dataset_name,lead))
    
    # Cull the list (only keep files with the specified leadtime)
    str1 = "_lead%i_" % (lead)   # ex. "..._lead2_..."
    str2 = "_lead%02i_" % (lead) # ex. "..._lead02_..."
    if np.any([str2 in f for f in modlist]):
        modlist = [fname for fname in modlist if str2 in fname]
    else:
        modlist = [fname for fname in modlist if str1 in fname]
    nmodels = len(modlist)
    print("\t %i models remain for lead %i" % (len(modlist),lead))
    
    modlist_lead.append(modlist)
    
    modweights = []
    for m in range(nmodels):
        mod    = torch.load(modlist[m])
        modweights.append(mod)
    
    modweights_lead.append(modweights)

#%% 


nmodels          = 50 # Specify manually how much to do in the analysis
st               = time.time()

# List for each leadtime
relevances_lead   = []
factivations_lead = []
idcorrect_lead    = []
modelacc_lead     = []
labels_lead       = []
for l,lead in enumerate(leads): # Training data does chain with leadtime
    
    # Get List of Models
    modlist = modlist_lead[l]
    modweights = modweights_lead[l]
    
    # Prepare data
    X_train,X_val,y_train,y_val = am.prep_traintest_classification(data,target,lead,thresholds,percent_train,
                                                                   ens=ens,tstep=tstep,quantile=quantile)
    
    # Make land/ice mask
    xsum = np.sum(np.abs(X_val),(0,1))
    limask = np.zeros(xsum.shape) * np.nan
    limask[np.abs(xsum)>1e-4] = 1
    
    # Preallocate, compute relevances
    valsize      = X_val.shape[0]
    relevances   = np.zeros((nmodels,valsize,inputsize))*np.nan # [model x sample x inputsize ]
    factivations = np.zeros((nmodels,valsize,3))*np.nan         # [model x sample x 3]
    for m in tqdm(range(nmodels)): # Loop for each model
        
        # Rebuild the model
        pmodel = am.recreate_model(modelname,nn_param_dict,inputsize,outsize,nlon=nlon,nlat=nlat)
        
        # Load the weights
        pmodel.load_state_dict(modweights[m])
        pmodel.eval()
        
        # Investigate
        inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
                              method=innmethod,
                              beta=innbeta)
        input_data = torch.from_numpy(X_val.reshape(X_val.shape[0],1,inputsize)).squeeze().float()
        model_prediction, true_relevance = inn_model.innvestigate(in_tensor=input_data)
        relevances[m,:,:]   = true_relevance.detach().numpy().copy()
        factivations[m,:,:] = model_prediction.detach().numpy().copy()
    
    # Reshape Output
    relevances = relevances.reshape(nmodels,valsize,nchannels,nlat,nlon)
    y_pred     = np.argmax(factivations,2)
    
    # Compute accuracy
    modelacc  = np.zeros((nmodels,3)) # [model x class]
    modelnum  = np.arange(nmodels)+1 
    top3mod   = []                    # [class]
    idcorrect = []
    for c in range(3):
        
        # Compute accuracy
        class_id           = np.where(y_val == c)[0]
        pred               = y_pred[:,class_id]
        targ               = y_val[class_id,:].squeeze()
        correct            = (targ[None,:] == pred)
        num_correct        = correct.sum(1)
        num_total          = correct.shape[1]
        modelacc[:,c]      = num_correct/num_total
        meanacc            = modelacc.mean(0)[c]
        
        # Get indices of correct predictions
        corrid = []
        for zz in range(nmodels):
            corrid.append(class_id[correct[zz,:]])
        idcorrect.append(corrid)
    
    # Append for each leadtime
    relevances_lead.append(relevances)
    factivations_lead.append(factivations)
    idcorrect_lead.append(idcorrect)
    modelacc_lead.append(modelacc)
    labels_lead.append(y_val)
print("Computed relevances in %.2fs" % (time.time()-st))

#%% Plot composites for some of the relevances

plotleads        = [0,6,12,18,24]
c                = 1
topN             = 50
normalize_sample = 2 # 0=None, 1=samplewise, 2=after composite
absval           = False
cmax             = 0.5
pcount           = 0

fig,axs   = plt.subplots(1,len(plotleads),figsize=(16,4.25),
                       subplot_kw={'projection':proj},constrained_layout=True) 
for l,lead in enumerate(plotleads):
    ax = axs[l]
    
    # Get indices of the top 10 models
    acc_in = modelacc_lead[l][:,c] # [model x class]
    idtopN = am.get_topN(acc_in,topN,sort=True)
    
    # Get the plotting variables
    id_plot = np.array(idcorrect_lead[l][c])[idtopN] # Indices to composite
    
    plotrel = np.zeros((nlat,nlon))
    for NN in range(topN):
        relevances_sel = relevances_lead[l][idtopN[NN],id_plot[NN],:,:,:].squeeze()
        
        if normalize_sample == 1:
            relevances_sel = relevances_sel / np.max(np.abs(relevances_sel),0)[None,...]
        
        if absval:
            relevances_sel = np.abs(relevances_sel)
        plotrel += relevances_sel.mean(0)
    plotrel /= topN
        
    if normalize_sample == 2:
        plotrel = plotrel/np.max(np.abs(plotrel))
    
    plotrel[plotrel==0] = np.nan
    pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r")
    
    
    ax.set_title("Lead %i" % (lead))
    if l == 0:
        ax.text(-0.05, 0.55, dataset_name, va='bottom', ha='center',rotation='vertical',
                rotation_mode='anchor',transform=ax.transAxes)
    ax.set_extent(bbox)
    ax.coastlines()
        
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05)
cb.set_label("Normalized Relevance")
plt.suptitle("Mean LRP Maps for predicting %s using %s, %s, \n Top %02i Models (%s)" % (classes[c],varname,modelname,topN,ge_label))
savename = "%sLRP_%s_%s_top%02i_normalize%i_abs%i_%s_AGU%02i.png" % (figpath,varname,classes[c],topN,normalize_sample,absval,ge_label_fn,pcount)
plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=True)


#%% Plot some of the correct relevances


# ---
c                = 0  # Class
topN             = 25 # Top 10 models
normalize_sample = 2 # 0=None, 1=samplewise, 2=after composite
absval           = False
cmax             = 1
# 

pcount = 0

fig,axs = plt.subplots(1,9,figsize=(16,6.5),
                       subplot_kw={'projection':proj},constrained_layout=True) 
for l,lead in enumerate(leads):
    ax = axs[l]
    
    # Get indices of the top 10 models
    acc_in = modelacc_lead[l][:,c] # [model x class]
    idtopN = am.get_topN(acc_in,topN,sort=True)
    
    # Get the plotting variables
    id_plot = np.array(idcorrect_lead[l][c])[idtopN] # Indices to composite
    
    plotrel = np.zeros((nlat,nlon))
    for NN in range(topN):
        relevances_sel = relevances_lead[l][idtopN[NN],id_plot[NN],:,:,:].squeeze()
        
        if normalize_sample == 1:
            relevances_sel = relevances_sel / np.max(np.abs(relevances_sel),0)[None,...]
        
        if absval:
            relevances_sel = np.abs(relevances_sel)
        plotrel += relevances_sel.mean(0)
    plotrel /= topN
        
    if normalize_sample == 2:
        plotrel = plotrel/np.max(np.abs(plotrel))
    
    plotrel[plotrel==0] = np.nan
    pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r")
    
    
    ax.set_title("Lead %i" % (lead))
    if l == 0:
        ax.text(-0.05, 0.55, dataset_name, va='bottom', ha='center',rotation='vertical',
                rotation_mode='anchor',transform=ax.transAxes)
    ax.set_extent(bbox)
    ax.coastlines()
        
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05)
cb.set_label("Normalized Relevance")
plt.suptitle("Mean LRP Maps for predicting %s using %s, %s, \n Top %02i Models (%s)" % (classes[c],varname,modelname,topN,ge_label))
savename = "%sLRP_%s_%s_top%02i_normalize%i_abs%i_%s_AGU%02i.png" % (figpath,varname,classes[c],topN,normalize_sample,absval,ge_label_fn,pcount)
plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=True)



#%% General Procedure

# 1.   >> Load Predictors and Labels <<

# 2.   >> Load Model Weights + Reconstruct <<

# 3.   >> Reproject and save indices of correct predictions for test set <<

# 4.   >> Perform LRP on this subset <<





