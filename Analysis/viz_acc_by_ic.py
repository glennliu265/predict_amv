#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Accuracy by Initial Condition

   Is AMV more predictable from certain initial states?
   
   
Copied base from evaluate_relative_relevance (2023.01.04)

Created on Wed Jan  4 12:04:09 2023

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

#%% # User Edits

# Indicate settings (Network Name)

# Data and variable settings
expdir    = "FNN4_128_ALL"
modelname  = "FNN4_128"
allpred    = ("SST","SSS","PSL","SSH")
apcolors   = ("r","limegreen","pink","darkblue")

leads      = np.arange(0,27,3)
nleads     = len(leads)

datpath    = "../../CESM_data/"
figpath    = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20230106/"
#datpath + expdir + "/Figures/"

# lrp methods
sys.path.append("/Users/gliu/Downloads/02_Research/03_Code/github/Pytorch-LRP-master/")
from innvestigator import InnvestigateModel

# Load modules (LRPutils by Peidong)
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/scrap/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/predict_amv/")

import LRPutils as utils
import amvmod as am

# Load visualization module
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/viz")
import viz

# LRP Settings (note, this currently uses the innvestigate package from LRP-Pytorch)
gamma          = 0.1
epsilon        = 0.1
innexp         = 2
innmethod      ='b-rule'
innbeta        = 0.1
# Labeling for plots and output files
ge_label     = "exp=%i, method=%s, $beta$=%.02f" % (innexp,innmethod,innbeta)
ge_label_fn  = "innexp%i_%s_innbeta%.02f" % (innexp,innmethod,innbeta)

# Data Settings
regrid         = None
quantile       = False
ens            = 40
tstep          = 86
percent_train  = 0.8              # Percentage of data to use for training (remaining for testing)
detrend        = 0
bbox           = [-80,0,0,65]
thresholds     = [-1,1]
outsize        = len(thresholds) + 1

# Region Settings
regions = ("NAT","SPG","STG","TRO")#("NAT","SPG","STG","TRO")
rcolors = ("k","b",'r',"orange")
bbox_SP     = [-60,-15,40,65]
bbox_ST     = [-80,-10,20,40]
bbox_TR     = [-75,-15,10,20]
bbox_NA     = [-80,0 ,0,65]
bbox_NA_new = [-80,0,10,65]
bbox_ST_w   = [-80,-40,20,40]
bbox_ST_e   = [-40,-10,20,40]
bboxes      = (bbox_NA,bbox_SP,bbox_ST,bbox_TR,) # Bounding Boxes

if modelname == "FNN2":
    nlayers     = 2
    nunits      = [20,20]
    activations = [nn.ReLU(),nn.ReLU()]
    dropout     = 0.5
elif "FNN4_120" in modelname:
    nlayers     = 4
    nunits      = [120,120,120,120]
    activations = [nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.ReLU()]
    dropout     = 0.5
elif "FNN4_128" in modelname:
    nlayers     = 4
    nunits      = [128,128,128,128]
    activations = [nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.ReLU()]
    dropout     = 0.5
elif modelname == "simplecnn":
    cnndropout     = True
    num_classes    = 3 # 3 AMV States
    num_inchannels = 1 # Single Predictor
if "nodropout" in modelname:
    dropout = 0
    
    
# Plotting Settings
classes   = ["AMV+","Neutral","AMV-"] # [Class1 = AMV+, Class2 = Neutral, Class3 = AMV-]
class_colors = ("red","gray","cornflowerblue")
proj      = ccrs.PlateCarree()


# ----------------------
#%% Load Data and Labels
# ----------------------
st = time.time()

varnames = allpred
# Load in input and labels 
all_data = []
for v,varname in enumerate(varnames):
    # Load in input and labels 
    ds   = xr.open_dataset(datpath+"CESM1LE_%s_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc" % (varname,detrend,regrid) )
    ds   = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3])).isel(ensemble=np.arange(0,ens))
    data = ds[varname].values[None,...]
    all_data.append(data)
all_data = np.array(all_data).squeeze() # [variable x ens x yr x lat x lon]

target = np.load(datpath+ "CESM_label_amv_index_detrend%i_regrid%s.npy" % (detrend,regrid))

# Apply Land Mask
# Apply a landmask based on SST, set all NaN points to zero
msk = xr.open_dataset(datpath+'CESM1LE_SST_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc'% (detrend,regrid))
msk = msk.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
msk = msk["SST"].values
msk[~np.isnan(msk)] = 1
msk[np.isnan(msk)] = 0
# Limit to input to ensemble member and apply mask
all_data = all_data[:,0:ens,...] * msk[None,0:ens,...]
all_data[np.isnan(all_data)] = 0

nchannels,nens,ntime,nlat,nlon = all_data.shape # Ignore year and ens for now...
inputsize               = nchannels*nlat*nlon # Compute inputsize to remake FNN


# Get indices based on training size

# Load Lat/Lon
lat = ds.lat.values
lon = ds.lon.values
print(lon.shape),print(lat.shape)

#%% Get model weights
varname = "ALL"
    
# Pull model list
modlist_lead = []
modweights_lead = []
for lead in leads:
    # Get Model Names
    modpath = "%s%s/Models/" % (datpath,expdir)
    modlist = glob.glob("%s*%s*.pt" % (modpath,varname))
    modlist.sort()
    print("Found %i models for Lead %i" % (len(modlist),lead))
    
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

#%% Use LRP to compute the relevance

nmodels = 50 # Specify manually how much to do in the analysis
st = time.time()

# Preallocate
relevances_all   = {} # [region][lead][model x sample x inputsize ]
factivations_all = {} # [region][lead][model x sample x class]
idcorrect_all    = {} # [region][lead][class][model][ids]

modelacc_all     = {} # [region][lead][model x class]
labels_all       = {} # [region][lead][samplesize]


# List for each leadtime
relevances_lead   = []
factivations_lead = []
idcorrect_lead    = []
modelacc_lead     = []
labels_lead       = []

labels_train_ic         = []
labels_val_ic           = []

for l,lead in enumerate(leads): # Training data does chain with leadtime
    
    # Get List of Models
    modlist = modlist_lead[l]
    modweights = modweights_lead[l]
    
    # Prepare data
    X_train,X_val,y_train,y_val,y_train_ic,y_val_ic = am.prep_traintest_classification(all_data,target,lead,thresholds,percent_train,
                                                                   ens=ens,tstep=tstep,
                                                                   quantile=quantile,return_ic=True)
    
    
    # Make land/ice mask
    xsum = np.sum(np.abs(X_val),(0,1))
    limask = np.zeros(xsum.shape) * np.nan
    limask[np.abs(xsum)>1e-4] = 1
    
    # Preallocate, compute relevances
    valsize      = X_val.shape[0]
    relevances   = np.zeros((nmodels,valsize,inputsize))*np.nan # [model x sample x inputsize ]
    factivations = np.zeros((nmodels,valsize,3))*np.nan         # [model x sample x 3]
    for m in tqdm(range(nmodels)): # Loop for each model
        
        # Build model 
        if "FNN" in modelname:
            layers = am.build_FNN_simple(inputsize,outsize,nlayers,nunits,activations,dropout=dropout)
            pmodel = nn.Sequential(*layers)
        elif modelname == "simplecnn":
            pmodel = am.build_simplecnn(num_classes,cnndropout=cnndropout,unfreeze_all=True,
                                nlat=nlat,nlon=nlon,num_inchannels=num_inchannels)
        
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
    
    labels_train_ic.append(y_train_ic)
    labels_val_ic.append(y_val_ic)

print("\nComputed relevances in %.2fs" % (time.time()-st))

#%% Using Activations (predictions), Labels, and Indices, try to recalculate accuracies
# based on the initial state

acc_by_ic       = np.zeros((nleads,nmodels,3))
classcount_ic   = np.zeros((nleads,3))

relevances_ic = np.zeros((nleads,3,4,nlat,nlon)) * np.nan
variables_ic  = np.zeros((nleads,3,4,nlat,nlon)) * np.nan


for l,lead in tqdm(enumerate(leads)): # Training data does chain with leadtime
    
    # Get variables for the given leadtime
    ic_class     = labels_val_ic[l]          # Initial Condition Class
    y_val        = labels_lead[l]            # Actual Class
    factivations = factivations_lead[l]      # Activations
    y_pred       = np.argmax(factivations,2) # Predicted Class
    
    relevancesin = relevances_lead[l]
    
    # Prepare data
    _,X_val,_,_,_,_ = am.prep_traintest_classification(all_data,target,lead,thresholds,percent_train,
                                                                   ens=ens,tstep=tstep,
                                                                   quantile=quantile,return_ic=True)
    
    # Compute the accuracy by ic_class
    
    for c in range(3):
        
        # Get predictions made for a given starting class
        idsel = np.where(ic_class.squeeze() == c)[0]
        correctmat = (y_val[idsel,0][None,:] == y_pred[:,idsel]) # [Model x Sample]
        
        # Compute percentage of correct predictions
        nsamples_c = idsel.shape[0]
        acc_by_ic[l,:,c] = correctmat.sum(1)/nsamples_c
        classcount_ic[l,c] = nsamples_c
        
        # Make composites of relevances (where predictions were correct)
        relcomp = relevancesin[:,idsel,:,:,:]
        relcomp = relcomp.reshape(50*nsamples_c,nchannels,nlat,nlon)
        relevances_ic[l,c,:,:,:] = relcomp[correctmat.flatten(),...].mean(0) # {var x lat x lon}
        variables_ic[l,c,:,:,:] = X_val[idsel,:,:,:].mean(0)
        
    

#%% Visualize accuracy by leadtime

plotindv=False
fig,ax = plt.subplots(1,1,constrained_layout=True,)

for c in range(3):
    
    if plotindv:
        for m in range(nmodels):
            
            ax.plot(leads,acc_by_ic[:,m,c],color=class_colors[c],label="",alpha=0.05)
            
            #$\overline{}$
    
    label="Lead 0=%s ($n$=%i)" % (classes[c],classcount_ic[l,c])
    mu    = acc_by_ic[:,:,c].mean(1)
    sigma = acc_by_ic[:,:,c].std(1)
    ax.plot(leads,mu,color=class_colors[c],label=label)
    if plotindv is False:
        ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.2,color=class_colors[c],zorder=1)

ax.set_title("Accuracy by Initial AMV State \n %s Trained with %s" % (modelname,str(allpred)))
ax.legend()
ax.grid(True,ls='dotted')
ax.set_xlim([0,24])

ax.set_yticks(np.arange(.25,1.1,.1))
ax.set_xticks(leads)
ax.set_ylim([.25,1])

ax.set_xlabel("Prediction Leadtime (Years)")
ax.set_ylabel("Accuracy")

savename = "%sAccuracy_by_IC_%s_plotindv%i.png" % (figpath,modelname,plotindv)
plt.savefig(savename,transparent=True,dpi=150) 

#%% Compute composites of correct predictions

# Write Script to visualize relevances (using relevances and variables_ic)

nv   = 3
clvl = np.arange(-3,3.25,.25)

fig,axs = plt.subplots(3,9,constrained_layout=True,figsize=(16,5),
                       subplot_kw={'projection':ccrs.PlateCarree()})

for c in range(3):
    for l in range(nleads):
        
        plotrel = relevances_ic[l,c,nv,:,:]
        plotrel = plotrel / np.nanmax(np.abs(plotrel.flatten()))
        
        plotvar = variables_ic[l,c,nv,:,:]
        
        
        pcm = ax.pcolormesh(lon,lat,plotrel,vmin=-1,vmax=1,cmap="RdBu_r",alpha=0.5)
        cl  = ax.contour(lon,lat,plotvar,levels=clvl,colors="k",linewidths=0.75)
        
        
        ax = axs[c,l]
        ax.set_extent(bbox)
        ax.coastlines()
        
        if c == 0:
            ax.set_title(leads[l])
            
        if l == 0:
            lbl = "%s" % (classes[c])
            ax.text(-0.05, 0.55, lbl, va='bottom', ha='center',rotation='vertical',
                rotation_mode='anchor',transform=ax.transAxes)
        
plt.suptitle("Relevances and %s by Initial Class" % (varnames[nv]))

savename = "%sRelevances_and_%s_byIC_%s.png" % (figpath,varnames[nv],modelname)
plt.savefig(savename,dpi=150,bbox_tight='inches')
