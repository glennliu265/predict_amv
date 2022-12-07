#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize the predictability by different regions
for a selected predictor

Copied upper section from viz_LRP_by_predictor

Created on Mon Dec  5 14:49:17 2022

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
#expdir    = "FNN4_128_SingleVar"
modelname  = "FNN4_128"
varname    = "SSS" 
leads      = np.arange(0,27,3)


datpath    = "../../CESM_data/"
figpath    = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20221209/"
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
proj      = ccrs.PlateCarree()
#%% Convenience functions

def get_prediction(factivations):
    # factivations  [model x sample x class]
    return np.argmax(factivations,2)

    
# ----------------------
#%% Load Data and Labels
# ----------------------
st = time.time()

# Load in input and labels 
ds   = xr.open_dataset(datpath+"CESM1LE_%s_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc" % (varname,detrend,regrid) )
ds   = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
data = ds[varname].values[None,...]
target = np.load(datpath+ "CESM_label_amv_index_detrend%i_regrid%s.npy" % (detrend,regrid))



region_targets = []
region_targets.append(target)
# Load Targets for other regions
for region in regions[1:]:
    index = np.load(datpath+"CESM_label_%s_amv_index_detrend%i_regrid%s.npy" % (region,detrend,regrid))
    region_targets.append(index)

# Apply Land Mask
# Apply a landmask based on SST, set all NaN points to zero
msk = xr.open_dataset(datpath+'CESM1LE_SST_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc'% (detrend,regrid))
msk = msk.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
msk = msk["SST"].values
msk[~np.isnan(msk)] = 1
msk[np.isnan(msk)] = 0
# Limit to input to ensemble member and apply mask
data = data[:,0:ens,...] * msk[None,0:ens,...]
data[np.isnan(data)] = 0

nchannels,nens,ntime,nlat,nlon = data.shape # Ignore year and ens for now...
inputsize               = nchannels*nlat*nlon # Compute inputsize to remake FNN


# Get indices based on training size




# Load Lat/Lon
lat = ds.lat.values
lon = ds.lon.values
print(lon.shape),print(lat.shape)


# ---------------------
#%% Get Model Weights
# ---------------------

modlist_all = {}
modweights_all = {}
for region in regions:
    
    # Get experiment Directory
    if region == "NAT":
        expdir = "%s_SingleVar" % modelname
    else:
        expdir = "FNN4_128_singlevar_regional/%s_%s" % (modelname,region)
    
    # Pull model list
    modlist_lead = []
    modweights_lead = []
    for lead in leads:
        # Get Model Names
        modpath = "%s%s/Models/" % (datpath,expdir)
        modlist = glob.glob("%s*%s*.pt" % (modpath,varname))
        modlist.sort()
        print("Found %i models for %s, Lead %i" % (len(modlist),region,lead))
        
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
        
    modlist_all[region] = modlist_lead
    modweights_all[region] = modweights_lead

#%% Obtain validation LRP Maps for each Region


nmodels = 50 # Specify manually how much to do in the analysis
st = time.time()

# Preallocate
relevances_all   = {} # [region][lead][model x sample x inputsize ]
factivations_all = {} # [region][lead][model x sample x class]
idcorrect_all    = {} # [region][lead][class][model][ids]

modelacc_all     = {} # [region][lead][model x class]
labels_all       = {} # [region][lead][samplesize]


for r,region in enumerate(regions): # Training data does not change for region
    

    # List for each leadtime
    relevances_lead   = []
    factivations_lead = []
    idcorrect_lead    = []
    modelacc_lead     = []
    labels_lead       = []
    for l,lead in enumerate(leads): # Training data does chain with leadtime
        
        # Get List of Models
        modlist = modlist_all[region][l]
        modweights = modweights_all[region][l]
        
        # Prepare data
        X_train,X_val,y_train,y_val = am.prep_traintest_classification(data,region_targets[r],lead,thresholds,percent_train,
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
        
    # Assign to dictionary
    relevances_all[region]   = relevances_lead
    factivations_all[region] = factivations_lead
    idcorrect_all[region]    = idcorrect_lead
    modelacc_all[region]     = modelacc_lead
    labels_all[region]       = labels_lead

print("Computed relevances in %.2fs" % (time.time()-st))
#%% Do a quick save

# savename = "%sLRP_results_regional_%s.npz" % (datpath,ge_label_fn)
# np.savez(savename,**{
#     'relevances_all'  :   relevances_all, # [region][lead][model x sample x inputsize ]
#     'factivations_all': factivations_all,  # [region][lead][model x sample x class]
#     'idcorrect_all'   : idcorrect_all, # [region][lead][class][model][ids]
#     'modelacc_all'    : modelacc_all, # [region][lead][model x class]
#     'labels_all'      : labels_all, # [region][lead][samplesize]
#     },allow_pickle=True)

#%% Do some visualizations

#%% Visualize test accuracy by class and by region

fig,axs = plt.subplots(1,3,figsize=(16,4),constrained_layout=True)

for c in range(3):
    ax = axs[c]
    ax.set_title(classes[c])
    
    for r,region in enumerate(regions):
        
        accsbylead = np.array(modelacc_all[region]) # lead x model x class
        
        for m in range(nmodels):
            ax.plot(leads,accsbylead[:,m,c],color=rcolors[r],alpha=0.05,label="")
        ax.plot(leads,accsbylead[:,:,c].mean(1),color=rcolors[r],alpha=1,label=region)
        ax.legend()
    
    ax.set_xticks(leads)
    ax.set_xlim([leads[0],leads[-1]])
    ax.grid(True,ls='dotted')
    ax.set_ylim(0,1.0)

plt.suptitle("Mean Accuracy for Regional Prediction Target, \n Model: %s,Predictor: %s" % (modelname,varname))
savename = "%s%s_%s_meanacc_byregion.png" %(figpath,modelname,varname,)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%%


# -----------------------------------------------------------
# Visualize mean relevance for top N samples of a given class
# -----------------------------------------------------------

c                = 0  # Class
topN             = 50 # Top 10 models
normalize_sample = 2 # 0=None, 1=samplewise, 2=after composite
absval           = False
cmax             = 1
# 
for topN in np.arange(5,55,5):
    fig,axs = plt.subplots(4,9,figsize=(16,6.5),
                           subplot_kw={'projection':proj},constrained_layout=True)
    for r,region in enumerate(regions):
        
        for l,lead in enumerate(leads):
            ax = axs[r,l]
            
            # Get indices of the top 10 models
            acc_in = modelacc_all[region][l][:,c] # [model x class]
            idtopN = am.get_topN(acc_in,topN,sort=True)
            
            # Get the plotting variables
            id_plot = np.array(idcorrect_all[region][l][c])[idtopN] # Indices to composite
            
            plotrel = np.zeros((nlat,nlon))
            for NN in range(topN):
                relevances_sel = relevances_all[region][l][idtopN[NN],id_plot[NN],:,:,:].squeeze()
                
                if normalize_sample == 1:
                    relevances_sel = relevances_sel / np.max(np.abs(relevances_sel),0)[None,...]
                
                if absval:
                    relevances_sel = np.abs(relevances_sel)
                plotrel += relevances_sel.mean(0)
            plotrel /= topN
                
            if normalize_sample == 2:
                plotrel = plotrel/np.max(np.abs(plotrel))
                
            pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r")
            
            # Plot bounding box
            viz.plot_box(bboxes[r],ax=ax,linewidth=0.5)
            
            # Do Plotting Business and labeling
            if r == 0:
                ax.set_title("Lead %i" % (lead))
            if l == 0:
                ax.text(-0.05, 0.55, region, va='bottom', ha='center',rotation='vertical',
                        rotation_mode='anchor',transform=ax.transAxes)
            ax.set_extent(bbox)
            ax.coastlines()
            
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05)
    cb.set_label("Normalized Relevance")
    plt.suptitle("Mean LRP Maps for predicting %s using %s, %s, \n Top %02i Models (%s)" % (classes[c],varname,modelname,topN,ge_label))
    savename = "%sRegional_LRP_%s_%s_top%02i_normalize%i_abs%i_%s.png" % (figpath,varname,classes[c],topN,normalize_sample,absval,ge_label_fn)
    plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Visualize mean relevances for topN models, looping by leadtime

c                = 0
normalize_sample = 2# 0=None, 1=samplewise, 2=after composite

absval           = False
cmax             = 1

for l,lead in enumerate(leads):
    
    
    fig,axs = plt.subplots(4,5,figsize=(9,6.5),
                           subplot_kw={'projection':proj},constrained_layout=True)
    for r,region in enumerate(regions):
        
        # Get indices of the top 10 models
        acc_in = modelacc_all[region][l][:,c] # [model x class]
        idtopN = am.get_topN(acc_in,5,sort=True)
        
        # Get the plotting variables
        id_plot = np.array(idcorrect_all[region][l][c])[idtopN] # Indices to composite
        
        for nn in range(5):
            ax = axs[r,nn]
            
            accplot = acc_in[idtopN[nn]] * 100
            
            # Get relevances for selected model and take sample mean
            relevances_sel = relevances_all[region][l][idtopN[nn],id_plot[nn],:,:,:].squeeze()
            if normalize_sample == 1:
                relevances_sel = relevances_sel / np.max(np.abs(relevances_sel),0)[None,...]
            if absval:
                relevances_sel = np.abs(relevances_sel)
            
            plotrel = relevances_sel.mean(0)
            if normalize_sample == 2:
                plotrel = plotrel/np.max(np.abs(plotrel))
            
            pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r")
            
            # Labeling
            viz.label_sp("%.1f" % accplot + "%",ax=ax,labelstyle="%s",
                         usenumber=True,fontsize=9,x=-.05,alpha=0.5)
            
            # Plot bounding box
            viz.plot_box(bboxes[r],ax=ax,linewidth=0.5)
                     
            if r == 0:
                ax.set_title("Rank %i" % (nn+1))
            if nn == 0:
                ax.text(-0.05, 0.55, region, va='bottom', ha='center',rotation='vertical',
                        rotation_mode='anchor',transform=ax.transAxes)
            ax.set_extent(bbox)
            ax.coastlines()
            
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05)
    cb.set_label("Normalized Relevance")
    
    plt.suptitle("Mean LRP Maps for predicting %s using %s, %s, Lead %02i \n Top 5 Models (%s)" % (classes[c],varname,modelname,lead,ge_label))
    savename = "%sRegional_LRP_%s_%s_normalize%i_abs%i_%s_lead%02i.png" % (figpath,varname,classes[c],normalize_sample,absval,ge_label_fn,lead)
    plt.savefig(savename,dpi=150,bbox_inches="tight")


            
        
        
        

        
    
    


#%%


for r,region in enumerate(regions):
    
    for l,lead in enumerate(leads):



#%%
