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
dataset_name   = "IPSL-CM6A-LR"#"CanESM5"

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

#%% Calculate the Relevance by leadtime

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
topN             = 25
normalize_sample = 2 # 0=None, 1=samplewise, 2=after composite
absval           = False
cmax             = 1.0
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
plt.suptitle("Mean LRP Maps for predicting %s using %s, %s, \n Top %02i Models (%s), %s" % (classes[c],varname,modelname,topN,ge_label,dataset_name))
savename = "%sLRP_%s_%s_%s_top%02i_normalize%i_abs%i_%s.png" % (figpath,varname,classes[c],dataset_name,topN,normalize_sample,absval,ge_label_fn)
plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=True)

#%% Select particular events

# ======================
#%% Event Based Analysis
# ======================

#%% Compute the "Accuracy" of each AMV event

# def calc_acc_bylead(labels_lead,factivations_lead):
#     """
#     Calculates the mean accuracy by lag
    
#     Parameters
#     ----------
#     labels_lead       : List of ARRAYs [lead][sample x 1]
#     factivations_lead : List of ARRAYs [lead][model x sample x activation]

#     Returns
#     -------
#     amvacc_bylead     : List of ACCURACIES [lead][sample] 

#     """
#     # Compute accuracy for each leadtime
#     nleads        = len(labels_lead)
#     amvacc_bylead = []
#     for il in range(nleads):
        
#         # Get Label, Activation, Prediction
#         labels_in = labels_lead[il] # [Sample x 1]
#         actin     = factivations_lead[il]
#         pred_in   = np.argmax(actin,2)
        

        
#         # Get the accuracy for that leadtime
#         amvacc_bylead.append(leadacc)
#     return amvacc_bylead

# amvacc_bylead = calc_acc_bylead(labels_lead,factivations_lead)

#%% Compute accuracy and other parameters by event


def calc_acc_byevent(labels_lead,leads,
                     factivations_lead,
                     relevances_lead,
                     target,
                     ens,tstep,percent_train=0.8):
    """
    Sort event accuracies and relevances by AMV event, reindexing sample to event..

    Parameters
    ----------
    labels_lead         : List of ARRAYs [lead][sample x 1] - Labels by leadtime
    leads               : LIST [lead]                       - leadtimes
    factivations_lead   : List of ARRAYs [lead][model x sample x activation] - activation values
    relevances_lead     : List of ndARRAYS [lead][model x sample x channel x lat x lon] - relevance maps
    ens                 : INT - # of ensemble members 
    tstep               : INT - # of years (timesteps)
    percent_train       : FLOAT - Percentage of data used in training

    Returns
    -------
    event_dict : dict with following keys/entires:
        "acc"      :: accuracy by event  --> [ens x yr x lead]
        "acc_avg"  :: mean acc by lead   --> [ens x yr]
        "cnt"      :: count of events    --> [ens x yr]
        "val"      :: value of amv idx   --> [ens x yr]
        "cls"      :: class of amv idx   --> [ens x yr x lead]
        "pre"      :: predicted class    --> [ens x yr x lead x model]
        "rel"      :: relevance of model --> [ens x yr x lead x model x lat x lon]
        "slab"     :: honestly not sure  --> [ens x yr]
        "start_iens" :: index of starting ensemble
        "start_iyr"  :: index of starting year

    """
    
    # Get dimension sizes
    nmodels,_,_,nlat,nlon = relevances_lead[0].shape
    nleads = len(leads)
    
    # Average Accuracy for a given event
    all_labels_acc = np.zeros((ens,tstep,nleads))                   # Mean accuracy for each lead (of 50 models)
    all_labels_cnt = np.zeros((ens,tstep))                          # Count of leads for each label
    all_labels_val = np.zeros((ens,tstep,))                         # Class of label Last dimension should be redundant
    all_labels_cls = np.zeros((ens,tstep,nleads))                   # Class of each label
    all_labels_pre = np.zeros((ens,tstep,nleads,nmodels))           # Predictions for each model
    all_labels_rel = np.zeros((ens,tstep,nleads,nmodels,nlat,nlon)) # Corresponding relevance Map
    all_labels_slab = np.zeros((ens,tstep),dtype="object")          # ...

    for il in range(nleads):
        print(il)
        # Get label value
        labels_in = labels_lead[il]
        
        # Get predictions
        actin     = factivations_lead[il]
        pred_in   = np.argmax(actin,2)
        
        # Get relevances
        rel_in    = relevances_lead[il].squeeze()
        
        # Compute accuracy for the leadtime
        correct = labels_in.T == pred_in
        leadaccs = correct.sum(0)/len(correct)
        
        # Get the indices
        nsamples  = len(leadaccs)
        print(nsamples)
        leadinds  = am.get_ensyr(np.arange(0,nsamples),leads[il],ens=ens,tstep=tstep,percent_train=percent_train,)
        print(leadinds.shape)
        # Looping for each sample
        for n in tqdm(range(nsamples)):
            e,y = leadinds.squeeze()[n] # Recover the ensemble and year
            if il == 0:
                start_iens = e
                start_iyr  = y
            
            # Assign
            all_labels_acc[e,y,il]       = leadaccs[n]     # Record the accuracy
            all_labels_cnt[e,y]          += 1              # Add to count
            all_labels_cls[e,y,il]       = labels_in[n]    # Record the class
            all_labels_val[e,y]          = target[e,y]     # Record the value (AMV wise)
            all_labels_slab[e,y]         = (e,y)           # Record the ...
            all_labels_pre[e,y,il,:]     = pred_in[:,n]    # Record predicted value 
            all_labels_rel[e,y,il,:,:,:] = rel_in[:,n,:,:] # Record the relevance
    
    # Get the mean accuracy (across leadtimes)
    all_labels_acc_avg = all_labels_acc.sum(2)/all_labels_cnt # A
    
    # Get rid of zero points
    for e in range(ens):
        for y in range(tstep):
            if all_labels_cnt[e,y] == 0:
                all_labels_acc[e,y,il] = np.nan
    
    event_dict = {
        "acc"           : all_labels_acc,
        "acc_avg"       : all_labels_acc_avg,
        "cnt"           : all_labels_cnt,
        "val"           : all_labels_val,
        "cls"           : all_labels_cls,
        "pre"           : all_labels_pre,
        "rel"           : all_labels_rel,
        "slab"          : all_labels_slab,
        "istart_ens"    : start_iens,
        "istart_yr"     : start_iyr
        }
    
    return event_dict


event_dict = calc_acc_byevent(labels_lead,leads,
                     factivations_lead,
                     relevances_lead,
                     target,
                     ens,tstep)

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
#%%




#%% General Procedure

# 1.   >> Load Predictors and Labels <<

# 2.   >> Load Model Weights + Reconstruct <<

# 3.   >> Reproject and save indices of correct predictions for test set <<

# 4.   >> Perform LRP on this subset <<





