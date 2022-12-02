#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Analyze output of synthetic datset created from
make_synthetic_dataset.py and networks trained with
NN_test_lead_ann_ImageNet_classification_singlevar.py

Created on Wed Nov 30 14:33:36 2022

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

from tqdm import tqdm_notebook as tqdm
import time

# Load modules (LRPutils by Peidong)
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/scrap/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/predict_amv/")

import LRPutils as utils
import amvmod as am
#%%

# Indicate settings (Network Name)
datpath   = "../../CESM_data/"
detrend = 0
regrid  = None
bbox    = [-80,0,0,65]
usefakedata = "fakedata_1Neg1Pos1Random_3box_fixval.nc"# Set to None, or name of fake dataset.

# Load in input and labels 
ds   = xr.open_dataset(datpath+usefakedata)
ds   = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
data = ds['fakedata'].values[None,...]
target = np.load(datpath+ "CESM_label_amv_index_detrend%i_regrid%s.npy" % (detrend,regrid))
nchannels,nens,ntime,nlat,nlon = data.shape

lat = ds.lat.values
lon = ds.lon.values

#%% Add LRP Package
sys.path.append("/Users/gliu/Downloads/02_Research/03_Code/github/Pytorch-LRP-master/")
from innvestigator import InnvestigateModel




#%% Some other setttings (copied from viz_LRP_by_predictor)

modelname = "FNN4_128"
expdir    = "fakedata_3reg_fixval"
varname   = "fakedata"


# Plotting Settings
thresnames  = ("AMV+","Neutral","AMV-",)
threscolors = ('r',"gray","b") 
# Model Parameters (copied from NN_test_lead_ann_ImageNet_classification.py)
#inputsize   = 3*224*224
bbox        = [-80,0,0,65]
thresholds  = [-1,1]
outsize     = len(thresholds) + 1

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
if "nodropout" in modelname:
    dropout = 0

# Data Infor,ation
lead          = 0
ens           = 40
tstep         = 86
thresholds    = [-1,1]
quantile      = False
percent_train = 0.8

#%%% Preprocess data. train/test split. 

# Restrict to select lead time and train.test
y                            = target[:ens,lead:].reshape(ens*(tstep-lead),1)
X                            = (data[:,:ens,:tstep-lead,:,:]).reshape(nchannels,ens*(tstep-lead),nlat,nlon).transpose(1,0,2,3)
nsamples,_,_,_ = X.shape
x_in                         = X.reshape(nsamples,nchannels*nlat*nlon) # Flatten for processing
inputsize                    = nchannels*nlat*nlon

# Make the labels
y_class = am.make_classes(y,thresholds,reverse=True,quantiles=quantile)
if quantile == True:
    thresholds = y_class[1].T[0]
    y_class   = y_class[0]
if (nsamples is None) or (quantile is True):
    nthres = len(thresholds) + 1
    threscount = np.zeros(nthres)
    for t in range(nthres):
        threscount[t] = len(np.where(y_class==t)[0])
    nsamples = int(np.min(threscount))
y_targ = y_class.copy()
y_val  = y.copy()

# Test/Train Split
X_train = X[0:int(np.floor(percent_train*nsamples)),...]
X_val   = X[int(np.floor(percent_train*nsamples)):,...]
y_train = y_class[0:int(np.floor(percent_train*nsamples)),:]
y_val   = y_class[int(np.floor(percent_train*nsamples)):,:]

#%% Get Model Names


# Get Model Names
modpath = "%s%s/Models/" % (datpath,expdir)

modlist = glob.glob("%s*%s*.pt" % (modpath,varname))
modlist.sort()
print("Found %i models" % (len(modlist)))

# Cull the list (only keep files with the specified leadtime)
str1 = "_lead%i_" % (lead)   # ex. "..._lead2_..."
str2 = "_lead%02i_" % (lead) # ex. "..._lead02_..."
if np.any([str2 in f for f in modlist]):
    modlist = [fname for fname in modlist if str2 in fname]
else:
    modlist = [fname for fname in modlist if str1 in fname]
nmodels = len(modlist)
print("%i models remain for lead %i" % (len(modlist),lead))

#print([m[-len(modpath):] for m in modlist])
# Load each model
modweights = []
for m in range(nmodels):
    mod    = torch.load(modlist[m])
    modweights.append(mod)
    #print("Loaded %s" % (modlist[m][-len(modpath):]))

#%% Do LRP

lrpmethod = 1
innexp    = 2
innmethod ='b-rule'
innbeta   = 0.1

strel        = time.time()

gamma        = 0.0
epsilon      = 0.0

ge_label     = "$\gamma$=%.02f, $\epsilon$=%.02f" % (gamma,epsilon)
ge_label_fn  = "gamma%.02f_epsilon%.02f" % (gamma,epsilon)

valsize      = X_val.shape[0]
relevances   = np.zeros((nmodels,valsize,inputsize))*np.nan # [model x sample x inputsize ]
factivations = np.zeros((nmodels,valsize,3))*np.nan         # [model x sample x 3]

# Loop for each model
for m in tqdm(range(nmodels)):
    
    # Build model and load the weights
    layers = am.build_FNN_simple(inputsize,outsize,nlayers,nunits,activations,dropout=dropout)
    pmodel = nn.Sequential(*layers)
    
    pmodel.load_state_dict(modweights[m])
    pmodel.eval()
    
    if lrpmethod == 0:
    
        # Evaluate things...
        for s in range(valsize):
            
            input_data = torch.from_numpy(X_val[s,:,:,:].reshape(1,inputsize)).float()
            
            # Calculate the predicted values
            val                = pmodel(input_data)
            factivations[m,s,:] = val.detach().numpy().squeeze() # [1 x 3]
            
            # Backpropagate for the relevance
            Ws,Bs  = utils.get_weight(modweights[m])
            rel    = utils.LRP_single_sample(X_val[s,:,:,:].reshape(1,inputsize).squeeze(),Ws,Bs,epsilon,gamma)
            relevances[m,s,:] = rel
    elif lrpmethod == 1:
        
        inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
                              method=innmethod,
                              beta=innbeta)
        input_data = torch.from_numpy(X_val.reshape(X_val.shape[0],1,inputsize)).squeeze().float()
        model_prediction, true_relevance = inn_model.innvestigate(in_tensor=input_data)
        
        relevances[m,:,:] = true_relevance.detach().numpy().copy()
        factivations[m,:,:] = model_prediction.detach().numpy().copy()
        
        

relevances = relevances.reshape(nmodels,valsize,nchannels,nlat,nlon)
y_pred     = np.argmax(factivations,2)

print("Calculated relevances in %.2fs!"% (time.time()-strel))


#%% Examine Model Accuracy for that leadtime

figpath = "%s%s/Figures/" % (datpath,expdir)

fig,axs  = plt.subplots(3,1,figsize=(12,8),constrained_layout=True)
modelacc  = np.zeros((nmodels,3)) # [model x class]
modelnum  = np.arange(nmodels)+1 
top3mod   = []                    # [class]
idcorrect = []

for c in range(3):
    
    # Compute accuracy
    class_id = np.where(y_val == c)[0]
    
    pred     = y_pred[:,class_id]
    targ     = y_val[class_id,:].squeeze()
    correct            = (targ[None,:] == pred)
    num_correct        = correct.sum(1)
    print(pred)
    num_total          = correct.shape[1]
    modelacc[:,c]      = num_correct/num_total
    meanacc            = modelacc.mean(0)[c]
    
    # Get indices of correct predictions
    corrid = []
    for zz in range(nmodels):
        corrid.append(class_id[correct[zz,:]])
    idcorrect.append(corrid)
    
    
    # Plot it
    ax = axs[c]
    ax.bar(modelnum,modelacc[:,c]*100,color=threscolors[c],alpha=0.75,edgecolor='k')
    ax.set_title("%s ($\mu=%.2f$)" % (thresnames[c],meanacc*100)+"%")
    ax.set_ylim([90,100])
    ax.grid(True,ls='dotted')
    ax.set_ylabel("Test Set Accuracy (%)")
    ax.axhline([33],color="k",ls='dotted',label="Random Chance (33%)",lw=1)
    ax.axhline([meanacc*100],ls='dashed',label="Mean Acc. ($\mu$)")
    ax.set_xticks(modelnum)
    if c == 0:
        ax.legend()
    if c == 2:
        ax.set_xlabel("Model Number")
    
    # Label Bars
    rects = ax.patches
    labels = ["%.2f" % (m*100) for m in modelacc[:,c]]
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 1, label+"%", ha="center", va="bottom"
        )
        
    # Mark the top 3
    top3=am.get_topN(modelacc[:,c],3,sort=True)
    ax.vlines(np.array([top3])+1,ymin=0,ymax=100,color='k')
    top3mod.append(top3)

plt.suptitle("Test Set Accuracy for %i-Year Lead AMV Prediction" % (lead))
savename = "%s%s_TestAcc_byModel_%s_lead%02i.png" % (figpath,modelname,varname,lead)
plt.savefig(savename,dpi=150)

#%% Check how the relevances look like for a particular class

isample = 199
fig,axs = plt.subplots(3,4,figsize=(16,12))
for a in range(nmodels):
    ax = axs.flatten()[a]
    
    pcm = ax.pcolormesh(lon,lat,relevances[a,isample,0,:,:],
                        cmap="RdBu_r",vmin=-1,vmax=1)
    
    #title = "Class: %i" % ()
    
#%%
N = 5

for c in [0,2]: # Look for AMV+ and AMV-

    imodel = top3mod[c][0]# 0 # Model Number
    
    cints = (np.arange(-3,3.4,0.4),
             np.arange(-2.0,2.4,0.4),
             np.arange(-4,4.5,0.5)
             )
    
    print(thresnames[c])
    
    
    fig,axs = plt.subplots(1,N,subplot_kw={'projection':ccrs.PlateCarree()},
                           constrained_layout=True,figsize=(16,4))
    
    # Get indices for correct predictions of selected class
    class_id = np.where((y_val == c).squeeze() * (y_pred[imodel,:]==c))[0]
    
    # Get indices for top N activation values for that class
    cactivations = factivations[imodel,class_id,c]
    idx_topact  = class_id[am.get_topN(cactivations,N,sort=True,)] # Get top activations index, and select from class id indices
    
    for n in range(N):
        
        idact        = idx_topact[n]
        n_relevance  = relevances[imodel,idact,:,:,:] # {varianble x lat x lon}
        n_relevance  = n_relevance / np.max(np.abs(n_relevance))  # Normalize by sample

            
        ax    = axs[n]
        
        
        title = "$a_{%i}$=%.2f" % (idact,factivations[imodel,idact,c]) 
        ax.set_title(title)
        
        # # Plot the variable
        plotrel = n_relevance[0,:,:] 
        pcm = ax.pcolormesh(lon,lat,plotrel,
                            cmap="RdBu_r",vmin=-1,vmax=1,alpha=0.7)
        
        #cl = ax.contour(lon,lat,X_val[idact,0,:,:],linewidths=0.75,colors="k",levels=cints[0])
        #ax.clabel(cl)
        #fig.colorbar(pcm,ax=ax)
        
        
        #ax.set_extent(bbox)
        ax.coastlines()
        
    fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.025,pad=0.01)
    plt.suptitle("Top %i %s Relevance Maps with Highest Activation for %s (Lead %02i, Model %i, %s)" % (N,varname,thresnames[c],lead,imodel+1,ge_label),
                 fontsize=20)
    savename = "%sTop%i_%s_Relevances_%s_lead%02i_model%s_%s.png" % (figpath,N,varname,thresnames[c],lead,imodel+1,ge_label_fn)
    plt.savefig(savename,dpi=150)
    
    # Look at all samples (stdev and abs value)
    #c       = 0 # Class
    #imodel  = top3mod[c][0] # Model Number
    
    for ii in range(3):
        imodel=top3mod[c][ii]
        
        # Retrieve the relevance and normalize, samplewise
        n_relevance  = relevances[imodel,:,:,:,:] # {sample x varianble x lat x lon}
        n_relevance  = n_relevance / np.max(np.abs(n_relevance),(1,2,3))[:,None,None,None] # Normalize across channel, lat, lon
        
        fig,axs = plt.subplots(2,1,subplot_kw={'projection':ccrs.PlateCarree()},
                               constrained_layout=True,figsize=(8,5.5))
        
        for a in range(2): # Stdev, then Absval
            sel_id = idcorrect[c][imodel]
            if a  == 0: # Plot Stdev
                plotrel = np.nanstd(n_relevance[sel_id,:,:,:],0)
                lbl     = "$\sigma$(relevances)"
                vlims   = [0,1.0]
                cmap    = 'magma'
            else: # Plot mean
                plotrel = np.nanmean(n_relevance[sel_id,:,:,:],0)
                lbl     = "$\mu$(relevances)"
                vlims   =  [0,1.0]
                cmap    = 'gist_heat'
            v=0
                
                
            ax = axs[a]
            
            pcm = ax.pcolormesh(lon,lat,plotrel[v,...],cmap=cmap,
                                vmin=vlims[0],vmax=vlims[1])
            
            
            if v == 0:
                ax.text(-0.05, 0.55, lbl, va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
            fig.colorbar(pcm,ax=ax,fraction=0.035)
            ax.coastlines()
                
        ax = axs[0]
        title = "Samplewise Summary Statistics of %s for %s \n (Lead %02i, Model %i [%.2f" % (varname,thresnames[c],lead,imodel+1,modelacc[imodel,c]*100) + "%" + "], %s)" % (ge_label)
        ax.set_title(title,fontsize=12,y=1)
        
        savename = "%sStatSummary_Relevances_%s_lead%02i_model%s_%s.png" % (figpath,thresnames[c],lead,imodel+1,ge_label_fn)
        plt.savefig(savename,dpi=150,bbox_inches='tight')
        
        
#%% Plot Relevances for 12 models


