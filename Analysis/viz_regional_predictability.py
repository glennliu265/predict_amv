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
varname    = "SSH" 
leads      = np.arange(0,27,3)
nleads     = len(leads)

#datpath    = "../../CESM_data/"
#figpath    = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20221209/"
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
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz,proc

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

# # Region Settings
# regions     = ("NAT","SPG","STG","TRO")#("NAT","SPG","STG","TRO")
# rcolors     = ("k","b",'r',"orange")
# bbox_SP     = [-60,-15,40,65]
# bbox_ST     = [-80,-10,20,40]
# bbox_TR     = [-75,-15,10,20]
# bbox_NA     = [-80,0 ,0,65]
# bbox_NA_new = [-80,0,10,65]
# bbox_ST_w   = [-80,-40,20,40]
# bbox_ST_e   = [-40,-10,20,40]
# bboxes      = (bbox_NA,bbox_SP,bbox_ST,bbox_TR,) # Bounding Boxes

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
#classes   = ["AMV+","Neutral","AMV-"] # [Class1 = AMV+, Class2 = Neutral, Class3 = AMV-]
#proj      = ccrs.PlateCarree()

# Dark mode settings
darkmode  = False
if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
else:
    plt.style.use('default')
    dfcol = "k"
    
#%% Convenience functions

def get_prediction(factivations):
    # factivations  [model x sample x class]
    return np.argmax(factivations,2)

#%% Load parameters to workspace

import predict_amv_params as pparams

regions = pparams.regions
bboxes  = pparams.bboxes
classes = pparams.classes
proj    = pparams.proj

rcolors = pparams.rcolors

datpath = pparams.datpath
figpath = pparams.figpath
proc.makedir(figpath)



    
# ----------------------
#%% Load Data and Labels
# ----------------------
st = time.time()




# Load in input and labels 
ds   = xr.open_dataset(datpath+"CESM1LE_%s_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc" % (varname,detrend,regrid) )
ds   = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
data = ds[varname].values[None,...]
target = np.load(datpath+ "CESM_label_amv_index_detrend%i_regrid%s.npy" % (detrend,regrid))


# Load in SST just for reference
ds   = xr.open_dataset(datpath+"CESM1LE_%s_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc" % ("SST",detrend,regrid) )
ds   = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
datasst = ds["SST"].values[None,...]


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

nmodels          = 50 # Specify manually how much to do in the analysis
st               = time.time()

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

#%% Visualize a set of leadtimes, for the AGU presentation

c                = 0  # Class
topN             = 25 # Top 10 models
normalize_sample = 2 # 0=None, 1=samplewise, 2=after composite
absval           = False
cmax             = 1
# 

pcount = 0

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
        
        plotrel[plotrel==0] = np.nan
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
savename = "%sRegional_LRP_%s_%s_top%02i_normalize%i_abs%i_%s_AGU%02i.png" % (figpath,varname,classes[c],topN,normalize_sample,absval,ge_label_fn,pcount)
plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=True)



#%% Same as above but reduce the number of leadtimes

leadsplot = [24,18,12,6,0]
#cmax      = 0.5

fig,axs = plt.subplots(4,5,figsize=(8,6.5),
                       subplot_kw={'projection':proj},constrained_layout=True)
for r,region in enumerate(regions):
    
    for i in range(len(leadsplot)):
        
        lead = leadsplot[i]
        print(lead)
        l    = list(leads).index(lead)
        print(l)
        
        ### Leads are all wrong need to fix it
        ax = axs[r,i]
        
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
        
        plotrel[plotrel==0] = np.nan
        pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r")
        
        # Plot bounding box
        if r !=0:
            viz.plot_box(bboxes[r],ax=ax,color="yellow",
                         linewidth=2.5)
        
        # Do Plotting Business and labeling
        if r == 0:
            ax.set_title("Lead %i" % (lead))
        if i == 0:
            ax.text(-0.05, 0.55, region, va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
        ax.set_extent(bbox)
        ax.coastlines()
        
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05)
cb.set_label("Normalized Relevance")
plt.suptitle("Mean LRP Maps for Predicting %s using %s, \n Composite of Top %02i FNNs per leadtime" % (classes[c],varname,topN,))
savename = "%sRegional_LRP_%s_%s_top%02i_normalize%i_abs%i_%s_AGU%02i_smaller.png" % (figpath,varname,classes[c],topN,normalize_sample,absval,ge_label_fn,pcount)
plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=True)

#%% Plot COlorbar

fig,axs = plt.subplots(4,5,figsize=(8,6.5),
                       subplot_kw={'projection':proj},constrained_layout=True)

for r,region in enumerate(regions):
    
    for i in range(len(leadsplot)):
        
        lead = leadsplot[i]
        print(lead)
        l    = list(leads).index(lead)
        print(l)
        
        ### Leads are all wrong need to fix it
        ax = axs[r,i]
        for r,region in enumerate(regions):
            
            for i in range(len(leadsplot)):
                
                lead = leadsplot[i]
                print(lead)
                l    = list(leads).index(lead)
                print(l)
                
                ### Leads are all wrong need to fix it
                ax = axs[r,i]
                
                ax.set_extent(bbox)
                

fig.colorbar(pcm,ax=axs[0,-1])
savename = "%sRegional_LRP_AGU_%s_Colorbar.png" % (figpath,varname,)
plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=True)

#%% For a given region, do composites by lead for positive and negative AMV


topN             = 50 # Top 10 models
normalize_sample = 2 # 0=None, 1=samplewise, 2=after composite
absval           = False
cmax             = 0.75
region           = "TRO"
r = regions.index(region)
clvl             = np.arange(-2,2.4,0.4)

fig,axs = plt.subplots(2,9,figsize=(16,4),
                       subplot_kw={'projection':proj},constrained_layout=True)

for row,c in enumerate([0,2]):
    for i in range(nleads):
        
        l = nleads-1-i
        
        lead = leads[l]
        print(l)
        ax   = axs[row,i]
        
        # Get topN Models
        acc_in = modelacc_all[region][l][:,c] # [model x class]
        idtopN = am.get_topN(acc_in,topN,sort=True)
        id_plot = np.array(idcorrect_all[region][l][c])[idtopN]
        
        plotrel = np.zeros((nlat,nlon)) # Relevances
        plotvar = np.zeros((nlat,nlon)) # Variable Value
        
        # Get data
        _,X_val,_,y_val = am.prep_traintest_classification(data,region_targets[0],lead,thresholds,percent_train,
                                                                       ens=ens,tstep=tstep,quantile=quantile)
            
        for NN in range(topN):
            relevances_sel = relevances_all[region][l][idtopN[NN],id_plot[NN],:,:,:].squeeze()
            var_sel        = X_val[id_plot[NN],:,:,:].squeeze()
            
            if (relevances_sel.shape[0] == 0) or (var_sel.shape[0]==0):
                continue
            
            if normalize_sample == 1:
                relevances_sel = relevances_sel / np.max(np.abs(relevances_sel),0)[None,...]
            if absval:
                relevances_sel = np.abs(relevances_sel)
            plotrel += relevances_sel.mean(0)
            plotvar += np.nanmean(var_sel,0)
            
        plotrel /= topN
        plotvar /= topN
        
        if normalize_sample == 2:
            plotrel = plotrel/np.nanmax(np.abs(plotrel))
        
        cl = ax.contour(lon,lat,plotvar,levels=clvl,colors="k",linewidths=0.75)
        pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r",alpha=0.8)
        ax.clabel(cl,clvl[::2])
        
        # Plot Region
        viz.plot_box(bboxes[r],ax=ax,linewidth=0.5)
        
        # Set Labels
        if row == 0:
            ax.set_title("Lead %i Years" % (lead))
        if i == 0:
            ax.text(-0.05, 0.55, classes[c], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
        ax.set_extent(bbox)
        ax.coastlines()
cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.01)

cb.set_label("Normalized Relevance")
plt.suptitle("Composite Relevance for predicting using %s, %s, \n Top %02i Models (%s)" % (varname,modelname,topN,ge_label))
savename = "%sComposite_LRP_%s_bylead_%s_top%02i_normalize%i_abs%i_%s.png" % (figpath,region,varname,topN,normalize_sample,absval,ge_label_fn)
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Same as above, but in separate figures

for i in range(nleads):
    
    l = nleads-1-i
    
    lead = leads[l]
    
    print(l)
    
    fig,axs = plt.subplots(2,1,figsize=(4,6),
                           subplot_kw={'projection':proj},constrained_layout=True)
    
    
    for row,c in enumerate([0,2]):
        ax   = axs[row]
        # Get topN Models
        acc_in = modelacc_all[region][l][:,c] # [model x class]
        idtopN = am.get_topN(acc_in,topN,sort=True)
        id_plot = np.array(idcorrect_all[region][l][c])[idtopN]
        
        plotrel = np.zeros((nlat,nlon)) # Relevances
        plotvar = np.zeros((nlat,nlon)) # Variable Value
        
        # Get data
        _,X_val,_,y_val = am.prep_traintest_classification(data,region_targets[0],lead,thresholds,percent_train,
                                                                       ens=ens,tstep=tstep,quantile=quantile)
            
        for NN in range(topN):
            relevances_sel = relevances_all[region][l][idtopN[NN],id_plot[NN],:,:,:].squeeze()
            var_sel        = X_val[id_plot[NN],:,:,:].squeeze()
            
            if (relevances_sel.shape[0] == 0) or (var_sel.shape[0]==0):
                continue
            
            if normalize_sample == 1:
                relevances_sel = relevances_sel / np.max(np.abs(relevances_sel),0)[None,...]
            if absval:
                relevances_sel = np.abs(relevances_sel)
            plotrel += relevances_sel.mean(0)
            plotvar += var_sel.mean(0)
            
        plotrel /= topN
        plotvar /= topN
        
        if normalize_sample == 2:
            plotrel = plotrel/np.max(np.abs(plotrel))
        
        cl = ax.contour(lon,lat,plotvar,levels=clvl,colors="k",linewidths=0.75)
        pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r",alpha=0.8)
        ax.clabel(cl,clvl[::2])
        
        # Plot Region
        viz.plot_box(bboxes[r],ax=ax,linewidth=0.5)
        
        # Set Labels
        if row == 0:
            ax.set_title("Lead %i Years" % (lead))
        ax.text(-0.05, 0.55, classes[c], va='bottom', ha='center',rotation='vertical',
                rotation_mode='anchor',transform=ax.transAxes)
        ax.set_extent(bbox)
        ax.coastlines()
    
    cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.045,)

    cb.set_label("Normalized Relevance")
    plt.suptitle("Composite Relevance for predicting using %s, %s, \n Top %02i Models (%s)" % (varname,modelname,topN,ge_label))
    savename = "%sComposite_LRP_%s_%s_top%02i_normalize%i_abs%i_%s_lead%02i.png" % (figpath,region,varname,topN,normalize_sample,absval,ge_label_fn,lead)
    plt.savefig(savename,dpi=150,bbox_inches="tight")



# ===========================
#%% Try Clustering the Output
# Based on: https://www.linkedin.com/pulse/identify-north-atlantic-winter-weather-regimes-k-means-chonghua-yin
# ===========================

from sklearn.cluster import KMeans

nclusts         = 6
norm_samplewise = False


rname           = "NAT"
il              = -1
iclass          = 0
rtest = relevances_all[rname][il]

# Restrict to particular class and samples
# Reassemble

for nm in range(nmodels):
    
    ids_model = idcorrect_all[rname][il][iclass][nm]
    relevances_nm = rtest[nm,ids_model,:,:,:].squeeze()
    if nm == 0:
        rtest_use = relevances_nm
    else:
        rtest_use = np.concatenate([rtest_use,relevances_nm],axis=0)

# [region][lead][class][model][ids]


if iclass is None:
    nmod,nsamples,_,_,_ = rtest.shape
    rtest = rtest.squeeze().reshape(nmod*nsamples,nlat*nlon)
else:
    nsamples,_,_ = rtest_use.shape
    nmod = 1
    rtest = rtest_use.reshape(nsamples,nlat*nlon)
    

if norm_samplewise:
    normconst = (np.max(np.abs(rtest),0))
    normconst[np.isnan(normconst)] = 1 # Set NaNs to 1
    normconst[normconst == 0] = 1 # Set zeros to 1
    rtest = rtest / normconst

st = time.time()
mk       = KMeans(n_clusters=nclusts, random_state=0,).fit(rtest)
ccenters = mk.cluster_centers_.reshape(nclusts,nlat,nlon)
clabels  = mk.labels_#.reshape(nmod*nsamples)
print("Completed clustering in %.2fs" % (time.time()-st))

# Reshape (sample x lat x lon)
rtest = rtest.reshape(nmod*nsamples,nlat,nlon)


#%% Plot Cluster Centers

fig,axs = plt.subplots(2,3,constrained_layout=True,figsize=(8,4),
                       subplot_kw={'projection':ccrs.PlateCarree()})

for n in range(nclusts):
    ax = axs.flatten()[n]
    
    # if norm_samplewise:
    #     pcm = ax.pcolormesh(lon,lat,ccenters[n,:,:],vmin=0,vmax=0.25,cmap="inferno")
    # else:
    #     pcm = ax.pcolormesh(lon,lat,ccenters[n,:,:],cmap="inferno")
        
    pcm = ax.pcolormesh(lon,lat,ccenters[n,:,:],cmap="inferno")
    
    ax.coastlines()
    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.025)
    ax.set_title("Cluster %i" % (n+1))
    
plt.suptitle("Cluster Centers (%s, %s, Lead %02i)" % (classes[iclass],rname,leads[il]))

savename = "%sCluster_result_%s_nclusts%i_region%s_lead%02i_cluster_means_normsmp%i.png" % (figpath,classes[iclass],nclusts,rname,leads[il],norm_samplewise)
plt.savefig(savename,dpi=150)

#%% Plot within cluster stdev

fig,axs = plt.subplots(2,3,constrained_layout=True,figsize=(8,4),
                       subplot_kw={'projection':ccrs.PlateCarree()})

for n in range(nclusts):
    ax = axs.flatten()[n]
    
    id_clust = np.where(clabels == n)[0]
    clustrel = rtest[id_clust,:,:].std(0)
    
    pcm = ax.pcolormesh(lon,lat,clustrel,cmap="inferno")
    ax.coastlines()
    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.025)
    ax.set_title("Cluster %i" % (n+1))
    
plt.suptitle("Cluster Stdev (%s, %s, Lead %02i)" % (classes[iclass],rname,leads[il]))
savename = "%sCluster_result_%s_nclusts%i_region%s_lead%02i_cluster_std_normsmp%i.png" % (figpath,classes[iclass],nclusts,rname,leads[il],norm_samplewise)
plt.savefig(savename,dpi=150)


#%% Plot samples from each cluster



clustsizes = []

for ic in range(6):
    
    id_clust = np.where(clabels == ic)[0]
    clustsizes.append(len(id_clust))
    
    fig,axs = plt.subplots(5,5,constrained_layout=True,figsize=(8,10),
                           subplot_kw={'projection':ccrs.PlateCarree()})
    
    for a in range(5*5):
        
        # Set up plot
        ax = axs.flatten()[a]
        ax.coastlines()
        ax.set_extent(bbox)
        ax.set_title("Sample = %i" % (id_clust[a]))
        
        # Plot 1 Sample
        pcm = ax.pcolormesh(lon,lat,rtest[id_clust[a],:,:],cmap="inferno")
        fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.025)
        
    plt.suptitle("Cluster %i, nsamples=%i (%s, %s, Lead %02i)" % (ic+1,clustsizes[ic],classes[iclass],rname,leads[il])
)    
    savename = "%sCluster_result_%s_nclusts%i_region%s_lead%02i_cluster%02i_normsmp%i..png" % (figpath,classes[iclass],nclusts,rname,leads[il],ic+1,norm_samplewise)
    plt.savefig(savename,dpi=150)
    

# ======================
#%% Event Based Analysis
# ======================

#%% Compute the "Accuracy" of each AMV event

rname = "NAT"

# Compute accuracy for each leadtime
amvacc_bylead = []
for il in range(nleads):
    
    # Get Label, Activation, Prediction
    labels_in = labels_all[rname][il] # [Sample x 1]
    actin     = factivations_all[rname][il]
    pred_in   = np.argmax(actin,2)
    
    # Get accuracy for the events
    correct = labels_in.T == pred_in
    leadacc = correct.sum(0)/len(correct)
    
    # Get the accuracy for that leadtime
    amvacc_bylead.append(leadacc)
    
    

# Average Accuracy for a given event
all_labels_acc = np.zeros((ens,tstep,nleads)) # Mean accuracy for each lead (of 50 models)
all_labels_cnt = np.zeros((ens,tstep))        # Count of leads for each label
all_labels_val = np.zeros((ens,tstep,))       # Class of label Last dimension should be redundant
all_labels_cls = np.zeros((ens,tstep,nleads))
all_labels_pre = np.zeros((ens,tstep,nleads,nmodels)) # Predictions for each model
all_labels_rel = np.zeros((ens,tstep,nleads,nmodels,nlat,nlon)) # Corresponding relevance Map

all_labels_slab = np.zeros((ens,tstep),dtype="object")

for il in range(nleads):
    
    # Get lead and label value
    leadaccs  = amvacc_bylead[il]
    labels_in = labels_all[rname][il]
    
    nsamples  = len(leadaccs)
    leadinds  = am.get_ensyr(np.arange(0,nsamples),leads[il])
    
    # Get predictions
    actin     = factivations_all[rname][il]
    pred_in   = np.argmax(actin,2)
    
    # Get relevances
    rel_in    = relevances_all[rname][il].squeeze()
    
    
    for n in tqdm(range(nsamples)):
        e,y = leadinds.squeeze()[n]
        if il == 0:
            start_iens = e
            start_iyr = y
        all_labels_acc[e,y,il] = leadaccs[n]
        all_labels_cnt[e,y] += 1
        all_labels_cls[e,y,il] = labels_in[n] 
        all_labels_val[e,y] = target[e,y]
        all_labels_slab[e,y] = (e,y)
        all_labels_pre[e,y,il,:] = pred_in[:,n]
        all_labels_rel[e,y,il,:,:,:] = rel_in[:,n,:,:]
        
all_labels_acc_avg = all_labels_acc.sum(2)/all_labels_cnt


# Get rid of zero points
for e in range(ens):
    for y in range(tstep):
        if all_labels_cnt[e,y] == 0:
            all_labels_acc[e,y,il] = np.nan

#%% Scatterplot selected leads (leadtime vs. AMV Index value)

plotabs = True

fig,ax = plt.subplots(1,1,figsize=(8,8))

plotx = all_labels_val[start_iens:,:]
xlab  = "AMV Index Value"
if plotabs:
    plotx = np.abs(plotx)
    xlab  = "|AMV Index Value|"

ax.scatter(plotx,
           all_labels_acc_avg[start_iens:,:],
           c=all_labels_cls[start_iens:,:,0],
           alpha=0.7,marker="o")
ax.set_xlabel(xlab)
ax.set_ylabel("Average Accuracy (All Leads)")
ax.grid(True,ls='dotted')

for n in range(len(plotx.flatten())):
    txt = str((all_labels_slab[start_iens:,:].flatten()[n]))
    ax.annotate(txt,
                (plotx.flatten()[n],all_labels_acc_avg[start_iens:,:].flatten()[n]),
                fontsize=8)



plt.savefig("%sAMVIdx_vs_TestAcc_LeadAvg_%s_plotabs%i.png" % (figpath,rname,plotabs),dpi=150)
#%% Make Scatterplot by Leadtime

fig,axs = plt.subplots(1,nleads,figsize=(16,4),constrained_layout=True)

for il in range(nleads):
    ax = axs[il]
    
    plotx = all_labels_val[start_iens:,:]
    xlab  = "AMV Index Value"
    if plotabs:
        plotx = np.abs(plotx)
        xlab  = "|AMV Index Value|"
    
    ax.scatter(plotx,
               all_labels_acc[start_iens:,:,il],
               c=all_labels_cls[start_iens:,:,0],
               alpha=0.7,marker="o",s=4)
    if il == 0:
        ax.set_xlabel("AMV Index Value")
        ax.set_ylabel("Average Accuracy (All Leads)")
    ax.grid(True,ls='dotted')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Lead %02i" % leads[il])
    
    
    # for n in range(len(plotx.flatten())):
    #     txt = str((all_labels_slab[start_iens:,:].flatten()[n]))
    #     ax.annotate(txt,
    #                 (plotx.flatten()[n],all_labels_acc_avg[start_iens:,:].flatten()[n]),
    #                 fontsize=8)
    
plt.savefig("%sAMVIdx_vs_TestAcc_ByLead_%s_plotabs%i.png" % (figpath,rname,plotabs),dpi=150)
#%% Select and plot event (spatial volution of the predictor)

e = 39
y = 5


plotsst= False


cints = np.arange(-10,11,1)


fig,axs = plt.subplots(1,nleads,subplot_kw={'projection':proj},figsize=(27,6),
                       constrained_layout=True)

for il in range(nleads):
    
    # Index Backwards from 24, 21, ..., 0
    kl   = nleads-il-1
    lead = leads[kl] 
    print("Lead index for interation %i is %i, l=%02i" % (il,kl,leads[kl]))
    
    ax = axs.flatten()[il]
    ax.coastlines()
    ax.set_extent(bbox)
    ax.set_title("Lead %02i" % (leads[kl]))
    
    if plotsst:
        plotdata = datasst[0,e,y-lead,:,:]
    else:
        plotdata = data[0,e,y-lead,:,:]
    cf       = ax.contourf(lon,lat,plotdata,cmap="RdBu_r",levels=cints)
    cl       = ax.contour(lon,lat,plotdata,colors="k",linewidths=0.5,levels=cints)
    ax.clabel(cl)
    
    fig.colorbar(cf,ax=ax,orientation='horizontal',fraction=0.026)
    if il == 0:
        if plotsst:
            ylab = "SST"
        else:
            ylab = varname
        ax.text(-0.05, 0.55, "%s, ens%02i" % (ylab,e+1), va='bottom', ha='center',rotation='vertical',
                rotation_mode='anchor',transform=ax.transAxes)
        
savename = "%s%s_Plot_%s_Prediction_e%02i_y%02i_plotsst%i.png" % (figpath,varname,rname,e,y,plotsst)
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Plot the timeseries for this period
yrs   = np.arange(0,tstep) + 1920
yrtks = yrs[::4] 

restrict_range=True # Restrict to prediction period (w/ 2 year buffer)

plot_idx    = target[e,:]
plot_idx_lp = proc.lp_butter(plot_idx,10,5) 

fig,ax = plt.subplots(1,1,figsize=(16,4))

ax.plot(yrs,plot_idx,color="gray",label="NASST")
ax.plot(yrs,plot_idx_lp,color="k",label="AMV Index")


ax.plot(yrs[(y-leads[-1]):(y+1)][::3],
        target[e,(y-leads[-1]):(y+1)][::3],
        color='magenta',marker="d",linestyle="",label="Prediction Leads for Target y=%04i" % (yrs[y]))


ax.axhline([0],ls='dashed',color="k",lw=0.75)
ax.axhline([0.3625],ls='dashed',color="gray",lw=0.75)
ax.axhline([-0.3625],ls='dashed',color="gray",lw=0.75)

ax.grid(True,ls='dotted')
if restrict_range:
    ax.set_xlim([yrs[y-leads[-1]-2],yrs[y+2]])
    ax.set_xticks(np.arange(yrs[y-leads[-1]-2],yrs[y+2]))
else:
    ax.set_xlim([yrs[0],yrs[-1]])
    ax.set_xticks(yrtks)
    
ax.set_title("AMV Index for Ens%02i" % (e+1))
ax.legend()

savename = "%sAMVIndex_Plot_%s_Prediction_e%02i_y%02i_plotsst%i_restrict%i.png" % (figpath,rname,e,y,plotsst,restrict_range)
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Copying from the above code, make a plot for each model



for n in tqdm(range(nmodels)):
    
    fig,axs = plt.subplots(1,nleads,subplot_kw={'projection':proj},figsize=(27,6),
                           constrained_layout=True)
    
    for il in range(nleads):
        
        # Index Backwards from 24, 21, ..., 0
        kl   = nleads-il-1
        lead = leads[kl] 
        print("Lead index for interation %i is %i, l=%02i" % (il,kl,leads[kl]))
        
        # Get predictions for the given model
        predlag   = all_labels_pre[e,y,kl,n]
        chk       = (predlag == all_labels_cls[e,y,kl])
        
        # Compute Plotting variables and normalize relevance
        plotdata  = data[0,e,y-lead,:,:]
        plotrel   = all_labels_rel[e,y,kl,n,:,:]
        normfactor = np.nanmax(np.abs(plotrel.flatten()))
        plotrel   = plotrel / normfactor
        
        
        ax = axs.flatten()[il]
        ax.coastlines()
        ax.set_extent(bbox)
        ax.set_title("l=%02i, Pred: %s (%s)\n normfac=%.02e" % (leads[kl],classes[int(predlag)],
                                                                chk,normfactor))
        
        
        pcm      = ax.pcolormesh(lon,lat,plotrel,cmap="RdBu_r",vmin=-1,vmax=1)
        #pcm      = ax.pcolormesh(lon,lat,plotrel,cmap="RdBu_r")
        cl       = ax.contour(lon,lat,plotdata,colors="k",linewidths=0.5,levels=cints)
        
        
        ax.clabel(cl)
        fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.026)
        if il == 0:
            if plotsst:
                ylab = "SST"
            else:
                ylab = varname
            ax.text(-0.05, 0.55, "%s, run%02i" % ("Relevance",n+1), va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
        
        
    savename = "%s%s_Plot_%s_Prediction_e%02i_y%02i_Relevances_model%02i.png" % (figpath,varname,rname,e,y,n)
    plt.savefig(savename,dpi=150,bbox_inches="tight")



#%% Compute mean and stdev of a particular prediction (intermodel)



intermodel_std = np.zeros((nleads,nlat,nlon)) * np.nan
intermodel_avg = intermodel_std.copy()
intermodel_acc = np.zeros((nleads)) * np.nan
intermodel_cnt = intermodel_acc.copy()


for moment in range(2):
    fig,axs = plt.subplots(1,nleads,subplot_kw={'projection':proj},figsize=(27,6),
                           constrained_layout=True)
    
    for il in range(nleads):
        
        # Index Backwards from 24, 21, ..., 0
        kl   = nleads-il-1
        lead = leads[kl] 
        print("Lead index for interation %i is %i, l=%02i" % (il,kl,leads[kl]))
        
        
        # Get indices of CORRECT predictions
        predlag   = all_labels_pre[e,y,kl,:]
        targlag   = all_labels_cls[e,y,il]
        correctid = np.where(predlag == int(targlag))[0]
        acclag    = len(correctid)/nmodels
        intermodel_acc[il] = acclag
        intermodel_cnt[il] = len(correctid)
        
        # Compute Plotting variables and normalize relevance
        plotdata  = data[0,e,y-lead,:,:]
        plotrel   = all_labels_rel[e,y,kl,correctid,:,:]
        
        
        # Compute Mean/Stdev (between N iterations of correct models)
        if moment == 0:
            mode = "Mean"
            plotrel                = np.mean(plotrel,0)
            intermodel_avg[il,:,:] = plotrel.copy()
            plotrng                = [-1,1]
            cmap                   = 'RdBu_r'
        elif moment == 1:
            mode = "Std. Dev."
            plotrel = np.std(plotrel,0)
            intermodel_std[il,:,:] = plotrel.copy()
            plotrng                = [0,1]
            cmap                   = "Greens"
        
        # Normalize for plotting
        #if moment == 0:
        normfactor = np.nanmax(np.abs(plotrel.flatten()))
        plotrel   = plotrel / normfactor
        # else:
        #     normfactor = "NA"
        
        # Start Plotting
        ax = axs.flatten()[il]
        ax.coastlines()
        ax.set_extent(bbox)
        ax.set_title("l=%02i, Acc=%.02f" % (leads[kl],acclag))
        
        # Contour and Pcolor
        pcm      = ax.pcolormesh(lon,lat,plotrel,cmap=cmap,vmin=plotrng[0],vmax=plotrng[1])
        cl       = ax.contour(lon,lat,plotdata,colors="k",linewidths=0.5,levels=cints)
        ax.clabel(cl)
        fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.026)
        
        # More Labeling
        if il == 0:
            
            if plotsst:
                ylab = "SST"
            else:
                ylab = varname
            ax.text(-0.05, 0.55, "%s, %s" % ("Relevance",mode,), va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
        
        
    savename = "%s%s_Plot_%s_Prediction_e%02i_y%02i_Relevances_%s.png" % (figpath,varname,rname,e,y,mode)
    plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Save the mean and stdev
savename = "%sEvent_based_intermodel_relevance_composites_e%02i_y%02i.npz" % (figpath,e,y)
np.savez(savename,**{
    'intermodel_avg':intermodel_avg,
    'intermodel_std':intermodel_std,
    'intermodel_acc':intermodel_acc,},allow_pickle=True)
# Testing the Load
#ld = np.load(savename,allow_pickle=True)

#%%
