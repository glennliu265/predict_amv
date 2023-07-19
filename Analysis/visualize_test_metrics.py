#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Test Metrics

Moved chunks from compute_test_metrics (original from test_cesm_withheld)
Created on Tue Jun 13 17:20:11 2023

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

#%% Load custom packages and setup parameters

machine = 'Astraeus' # Indicate machine (see module packages section in pparams)

# Import packages specific to predict_amv
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import train_cesm_params as train_cesm_params
import amv_dataloader as dl
import amvmod as am

# Load Predictor Information
bbox          = pparams.bbox

# Import general utilities from amv module
pkgpath = pparams.machine_paths[machine]['amv_path']
sys.path.append(pkgpath)
from amv import proc

# Import LRP package
lrp_path = pparams.machine_paths[machine]['lrp_path']
sys.path.append(lrp_path)
from innvestigator import InnvestigateModel

# Load ML architecture information
nn_param_dict      = pparams.nn_param_dict

# Load plotting information
classes   = pparams.classes
varcolors = pparams.varcolors
varmarker = pparams.varmarker
figpath   = pparams.figpath
# ============================================================
#%% User Edits vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ============================================================

# Set machine and import corresponding paths

# Set experiment directory/key used to retrieve params from [train_cesm_params.py]
expdir              = "FNN4_128_SingleVar_PaperRun"
eparams             = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Processing Options
even_sample         = False

# Get some paths
datpath             = pparams.datpath
dataset_name        = "CESM1"

# Set some looping parameters and toggles
varnames            = ["SSH","SST","SLP","SSS","NHFLX"]       # Names of predictor variables
varcolors           = ["cornflowerblue","red","gold","limegreen","magenta"]
leads               = np.arange(0,26,1)    # Prediction Leadtimes
runids              = np.arange(0,100,1)    # Which runs to do

# LRP Parameters
innexp         = 2
innmethod      ='b-rule'
innbeta        = 0.1

# Other toggles
save_all_relevances = False                # True to save all relevances (~33G per file...)
checkgpu            = True                 # Set to true to check if GPU is availabl
debug               = False                 # Set verbose outputs
savemodel           = True                 # Set to true to save model weights

# Save looping parameters into parameter dictionary
eparams['varnames'] = varnames
eparams['leads']    = leads
eparams['runids']   = runids

#%% Load the results

datpath_metrics     = "%s/%s/Metrics/Test_Metrics/" % (datpath,expdir)

#%% Dummy Section for loading variables (hashing things out)

"""
nvars = len(varnames)

for v in range(nvars):
    
v       = 0
varname = varnames[v]

# Relevances
fn_relevance = "%sTest_Metrics_CESM1_%s_evensample%i_relevance_maps.nc" % (datpath_metrics,varname,even_sample,)
ds_rel       = xr.open_dataset(fn_relevance)

# Metrics
fn_metrics   = "%sTest_Metrics_CESM1_%s_evensample%i_accuracy_predictions.npz" % (datpath_metrics,varname,even_sample,)
ld_metrics   = np.load(fn_metrics,allow_pickle=True)
"""


#%% Load the test accuracies

nleads  = len(leads)
nruns   = len(runids)
nvars   = len(varnames)
metrics_byvar = []
for v in range(nvars):
    fn_metrics   = "%sTest_Metrics_CESM1_%s_evensample%i_accuracy_predictions.npz" % (datpath_metrics,varnames[v],even_sample,)
    ld_metrics   = np.load(fn_metrics,allow_pickle=True)
    metrics_byvar.append(ld_metrics)
    
"""
['total_acc',
 'class_acc',
 'predictions',
 'targets',
 'ensemble',
 'leads',
 'runids',
 'allow_pickle']

'predictions' : [leadtime x model], samples
'targets'     : [leadtime], samples


"""
    
#%% Updated load of persistence baseline (copied from viz_acc_bypredictor on 2023.06.16)
persaccclass = []
persacctotal = []
persleads   = []
for detrend in [False,True]:
    pers_leads,pers_class_acc,pers_total_acc = dl.load_persistence_baseline("CESM1",
                                                                            return_npfile=False,region=None,quantile=eparams['quantile'],
                                                                            detrend=eparams['detrend'],limit_samples=False,nsamples=eparams['nsamples'],repeat_calc=1)

    persaccclass.append(pers_class_acc)
    persacctotal.append(pers_total_acc)
    persleads.append(pers_leads)

chance_baseline = [.33,.66,.33]
#%% Visualize test accuracy by class


add_conf   = True
plotconf   = 0.68
plotmax    = False # Set to True to plot maximum
alpha      = 0.25

xtks       = np.arange(0,26,5)


dfcol = "k"

fig,axs    = plt.subplots(1,3,figsize=(18,4))
for c in range(3):
    
    # Initialize plot
    ax = axs[c]
    ax.set_title("%s" %(classes[c]),fontsize=16,)
    ax.set_xlim([0,24])
    ax.set_xticks(leads)
    ax.set_ylim([0,1])
    ax.set_yticks(np.arange(0,1.25,.25))
    ax.grid(True,ls='dotted')
    
    for v in range(nvars):
            
        plotacc = metrics_byvar[v]['class_acc'][...,c] #classacc[i,:,:,c].mean(0)
        #ax.plot(leads,plotacc,color=varcolors[i],alpha=1,lw=lwall,label=varnames[i])
        
        mu        = plotacc.mean(0)
        sigma     = plotacc.std(0)
        
        sortacc  = np.sort(plotacc,0)
        idpct    = sortacc.shape[0] * plotconf
        lobnd    = np.floor(idpct).astype(int)
        hibnd    = np.ceil(sortacc.shape[0]-idpct).astype(int)
        
        ax.plot(leads,mu,color=varcolors[v],marker=varmarker[v],alpha=1.0,lw=2.5,label=varnames[v],zorder=9)
        if add_conf:
            if plotconf:
                ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=alpha,color=varcolors[v],zorder=1,label="")
            else:
                ax.fill_between(leads,mu-sigma,mu+sigma,alpha=alpha,color=varcolors[v],zorder=1)
        
    ax.plot(persleads[0],persaccclass[detrend][:,c],color=dfcol,label="Persistence",ls="dashed")
    ax.axhline(chance_baseline[c],color=dfcol,label="Random Chance",ls="dotted")
    
        # Add max/min predictability dots (removed this b/c it looks messy)
        # ax.scatter(leads,classacc[i,:,:,c].max(0),color=varcolors[i])
        # ax.scatter(leads,classacc[i,:,:,c].min(0),color=varcolors[i])
    
    #ax.plot(leads,autodat[::3,c],color='k',ls='dotted',label="AutoML",lw=lwall)
    #ax.plot(leads,persaccclass[:,c],color='k',label="Persistence",lw=lwall)

    #ax.hlines([0.33],xmin=-1,xmax=25,ls="dashed",color=dfcol)
        
    if c == 0:
        ax.legend(ncol=2,fontsize=10)
        ax.set_ylabel("Accuracy")
    if c == 1:
        ax.set_xlabel("Prediction Lead (Years)")
    
    ax.set_xticks(xtks)
plt.savefig("%sPredictor_Intercomparison_byclass_UPDATED_detrend%i_plotmax%i_%s_AGUver.png"% (figpath,detrend,plotmax,expdir),
            dpi=200,bbox_inches="tight",transparent=True)


#%% Get count of each class prediction by variable, netc
preds_byvar    = np.array([metrics_byvar[v]['predictions'] for v in range(nvars)]) # [variable, lead, network][samples]
count_by_class = np.zeros((nvars,nleads,nruns,3))
for v in range(nvars):
    for l in range(nleads):
        for r in range(nruns):
            for c in range(3):
                count_by_class[v,l,r,c] = len(np.where(preds_byvar[v,l,r] == c)[0])
                
#%% Plot the results

fig,axs    = plt.subplots(1,3,figsize=(18,4))
for c in range(3):
    
    # Initialize plots
    ax = axs[c]
    ax.set_title("%s" %(classes[c]),fontsize=16,)
    ax.set_xlim([0,25])
    ax.set_xticks(leads)
    ax.grid(True,ls='dotted')
    ax.set_xticks(xtks)
    
    
    for v in range(nvars):
        plotvar = count_by_class[v,:,:,c].mean(1)
        ax.plot(leads,plotvar,color=varcolors[v],label=varnames[v])
    if c == 0:
        ax.legend()
        ax.set_ylabel("Mean Prediction Count")
    
        
plt.savefig("%sMeanPredictionCount_byClass_detrend%i_plotmax%i_%s_AGUver.png"% (figpath,detrend,plotmax,expdir),
            dpi=200,bbox_inches="tight",transparent=True)
plt.show()


#%% Look s


relevances_byvar = []
predictors_byvar = []
for v in range(nvars):
    
    varname = varnames[v]
    
    # Relevances
    fn_relevance = "%sTest_Metrics_CESM1_%s_evensample%i_relevance_maps.nc" % (datpath_metrics,varname,even_sample,)
    ds_rel       = xr.open_dataset(fn_relevance)
    ds_rel       = ds_rel.rename({'class': 'classes'}) # Rename...
    
    # Load the stuff
    relevances_byvar.append(ds_rel['relevance_composites'].load())
    predictors_byvar.append(ds_rel['predictor_composites'].load())

#%% Plot relevance composites

v         = 1
c         = 0
plotleads = [25,20,15,10,5,0]

cint = np.arange(-2,2.2,.2)
vmax = 1

for v in range(nvars):
    varname = varnames[v]
    for c in range(3):
        fig,axs = plt.subplots(1,6,constrained_layout=True,figsize=(16,4),
                               subplot_kw={'projection':ccrs.PlateCarree()})
        
        
        for a in range(len(plotleads)):
            
            ax = axs.flatten()[a]
            ax.coastlines(color="gray")
            ax.set_extent(bbox)
            
            ax.set_title("Lead %i years" % plotleads[a])
            
            plotrel = relevances_byvar[v]
            plotrel = plotrel.sel(lead=plotleads[a]).isel(classes=c).mean('runid')
            plotrel = plotrel / np.nanmax(np.abs(plotrel))
            
            plotvar = predictors_byvar[v].sel(lead=plotleads[a]).isel(classes=c)
            
            pcm     = ax.pcolormesh(plotrel.lon,plotrel.lat,plotrel,vmax=vmax,vmin=-vmax,
                                    cmap="RdBu_r")
            cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,colors="k",linewidths=0.75,
                                 levels=cint)
            ax.clabel(cl)
            fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05)
        plt.suptitle("Relevance Composites for predicting %s using %s" % (classes[c],varname),
                     y=0.80)
        
        savename = "%sRelevanceComp_%s_%s_%s.png" % (figpath,expdir,varname,classes[c])
        plt.savefig(savename,dpi=150)
        print(savename)


#%% Recompute class prediction count by year


for v in range(nvars):
    for c in range(3):
        
        # Get indices
        # get the predicted values
        #
        
        


#%% User Edits


# Shared Information
varname            = "SST" # Testing variable
detrend            = False
leads              = np.arange(0,26,3)
region_name        = "NAT"
nsamples           = "ALL"
shuffle_trainsplit = False

# CESM1-trained model information
expdir             = "FNN4_128_SingleVar"
modelname          = "FNN4_128"
nmodels            = 50 # Specify manually how much to do in the analysis
eparams            = train_cesm_params.train_params_all[expdir] # Load experiment parameters
ens_wh             = [40,41] # Withheld member#eparams['ens']
runids             = np.arange(0,nmodels)



# Load parameters from [oredict_amv_param.py]
datpath            = pparams.datpath
figpath            = pparams.figpath
figpath            = pparams.figpath
nn_param_dict      = pparams.nn_param_dict
class_colors       = pparams.class_colors
classes            = pparams.classes
bbox               = pparams.bbox


# Other toggles
debug              = False
checkgpu           = True
darkmode           = False


if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
    transparent      = True
else:
    plt.style.use('default')
    dfcol = "k"
    transparent      = False





#%% 



# ------------------------------------------------------------
# %% Looping for runid
# ------------------------------------------------------------

# Print Message

# ------------------------
# 04. Loop by predictor...
# ------------------------


    # End Runid Loop >>>
#print("\nPredictor %s finished in %.2fs" % (varname,time.time()-vt))
# End Predictor Loop >>>

#print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))


#%% Perform LRP

#%% Prepare to do some visualization

# Load baselines
persleads,pers_class_acc,pers_total_acc = dl.load_persistence_baseline("CESM1",
                                                                        return_npfile=False,region=None,quantile=False,
                                                                        detrend=detrend,limit_samples=False,nsamples=nsamples,repeat_calc=1)


    
    
# persleads,pers_class_acc,pers_total_acc = dl.load_persistence_baseline(dataset_name,
#                                                                         return_npfile=False,region="NAT",quantile=False,
#                                                                         detrend=False,limit_samples=True,nsamples=None,repeat_calc=1)

# Load results from CESM1
#%%

fig,ax = plt.subplots(1,1)
for nr in range(nmodels):
    ax.plot(leads,total_acc_all[nr,:],alpha=0.1,color="g")
    
ax.plot(leads,total_acc_all.mean(0),color="green",label="CESM1-trained NN (SST)")
ax.plot(persleads,pers_total_acc,color="k",ls="dashed",label="Persistence Baseline")
ax.axhline([.33],color="gray",ls="dashed",lw=0.75,label="Random Chance Baseline")

ax.legend()
ax.grid(True,ls="dotted")
ax.set_xticks(persleads[::3])
ax.set_xlim([0,24])
ax.set_yticks(np.arange(0,1.25,0.25))
ax.set_xlabel("Prediction Lead (Years)")
ax.set_ylabel("Accuracy")
ax.set_title("Total Accuracy (HadISST Testing, %i samples per class)" % (nsample_total/3))
# 
figname = "%sReanalysis_Test_%s_Total_Acc.png" % (figpath,dataset_name)
plt.savefig(figname,dpi=150)
#%% 

fig,axs = plt.subplots(1,3,constrained_layout=True,figsize=(16,4))
for c in range(3):
    ax = axs[c]
    for nr in range(nmodels):
        ax.plot(leads,class_acc_all[nr,:,c],alpha=0.1,color=class_colors[c])
    ax.plot(leads,class_acc_all.mean(0)[...,c],color=class_colors[c],label="CESM1-trained NN (SST)")
    
    ax.plot(persleads,pers_class_acc[:,c],color="k",ls="dashed",label="Persistence Baseline")
    ax.axhline([.33],color="gray",ls="dashed",lw=2,label="Random Chance Baseline")
    
    if c == 1:
        ax.legend()
    ax.grid(True,ls="dotted")
    ax.set_xticks(persleads[::3])
    ax.set_xlim([0,24])
    ax.set_xlabel("Prediction Lead (Years)")
    ax.set_ylabel("Accuracy")
    ax.set_yticks(np.arange(0,1.25,0.25))
    ax.set_title(classes[c])
    ax.minorticks_on()
figname = "%sReanalysis_Test_%s_Class_Acc.png" % (figpath,dataset_name)
plt.savefig(figname,dpi=150)

#%% Visualizet he class distribution

idx_by_class,count_by_class = am.count_samples(None,target_class)

class_str = "Class Count: AMV+ (%i) | Neutral (%i) | AMV- (%i)" % tuple(count_by_class)

timeaxis = np.arange(0,re_target.shape[1]) + 1870
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))

ax.plot(timeaxis,target.squeeze(),color="k",lw=2.5)
ax.grid(True,ls="dashed")
ax.minorticks_on()

for th in thresholds_in:
    ax.axhline([th],color="k",ls="dashed")
ax.axhline([0],color="k",ls="solid",lw=0.5)
ax.set_xlim([timeaxis[0],timeaxis[-1]])
ax.set_title("CESM1WH NASST Index (1870-2022) \n%s" % (class_str))
plt.savefig("%sCESM1WH_NASST.png" %(figpath),dpi=150,bbox_inches='tight')

#%% Get correct indices for each class


# y_predicted_all = [runs,lead,sample]
# y_actual_all    = [lead,sample]

correct_mask = []
for l in range(len(leads)):
    lead = leads[l]
    y_preds   = y_predicted_all[:,l,:] # [runs lead sample]
    i_correct = (y_preds == y_actual_all[l,:][None,:]) # Which Predictions are correct
    correct_mask_lead = []
    for c in range(3):
        i_class = (y_actual_all[l,:] == c)
        correct_mask_lead.append(i_correct*i_class)
    correct_mask.append(correct_mask_lead)

#%% Visualize relevance maps

relevances_all = np.array(relevances_all)
predictor_all = np.array(predictor_all)
nruns,nleads,nsamples_lead,nlat,nlon = relevances_all.shape

plotleads        = [24,18,12,6,0]
normalize_sample = 2

plot_bbox        = [-80,0,0,60]

cmax  = 1
clvl = np.arange(-2.2,2.2,0.2)

fsz_title        = 20
fsz_axlbl        = 18
fsz_ticks        = 16


fig,axs  = plt.subplots(3,len(plotleads),constrained_layout=True,figsize=(18,10),
                        subplot_kw={'projection':ccrs.PlateCarree()})


ii = 0
for c in range(3):
    for l in range(len(plotleads)):
        
        ax = axs.flatten()[ii]
        lead  = plotleads[l]
        ilead = list(leads).index(lead) 
        
        # Axis Formatting
        blabel = [0,0,0,0]
        if c == 0:
            ax.set_title("%s-Year Lead" % (plotleads[l]),fontsize=fsz_title)
        if l == 0:
            blabel[0] = 1
            ax.text(-0.15, 0.55, classes[c], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_axlbl)
        ax = viz.add_coast_grid(ax,bbox=plot_bbox,blabels=blabel,fill_color="k")
        ax = viz.label_sp(ii,ax=ax,fig=fig,alpha=0.8,fontsize=fsz_axlbl)
            
        # Get correct predictions
        cmask = correct_mask[l][c].flatten()
        relevances_in = relevances_all[:,ilead,:,:,:]
        newshape      = (np.prod(relevances_in.shape[:2]),) + (nlat,nlon)
        # Apprently using cmask[:,...] brocasts, while cmask[:,None,None] doesn't
        relevances_sel = relevances_in.reshape(newshape)[cmask[:,...]] # [Samples x Lat x Lon]
        
        predictor_in   = predictor_all[:,ilead,:,:,:]
        predictor_sel = predictor_in.reshape(newshape)[cmask[:,...]] # [Samples x Lat x Lon]
        if normalize_sample == 1:
            relevances_sel = relevances_sel / np.abs(relevances_sel.max(0))[None,...]
        
        
        # Plot the results
        plotrel = relevances_sel.mean(0)
        plotvar = predictor_sel.mean(0)
        if normalize_sample == 2:
            plotrel = plotrel/np.max(np.abs(plotrel))
            
        # Set Land Points to Zero
        plotrel[plotrel==0] = np.nan
        plotvar[plotrel==0] = np.nan
        
            
        # Do the plotting
        pcm=ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r")
        cl = ax.contour(lon,lat,plotvar,levels=clvl,colors="k",linewidths=0.75)
        ax.clabel(cl,clvl[::2])
        
            
        ii+=1
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.025,pad=0.01)
cb.set_label("Normalized Relevance",fontsize=fsz_axlbl)
cb.ax.tick_params(labelsize=fsz_ticks)


savename = "%sHadISSTClassComposites_LRP_%s_normalize%i_Outline.png" % (figpath,expdir,normalize_sample)
if darkmode:
    savename = proc.addstrtoext(savename,"_darkmode")
plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=transparent)

#%% Make a scatterplot of the event distribution and 

imodel = 6
ilead  = 8
msize  = 100
timeaxis = np.arange(0,re_target.shape[1]) + 1870

for imodel in range(50):
    # Select the model
    y_predicted_in = y_predicted_all[imodel,ilead,:]
    y_actual_in    = y_actual_all[ilead,:]
    re_target_in   = re_target[:,leads[ilead]:].squeeze()
    id_correct     = (y_predicted_in == y_actual_in)
    
    
    timeaxis_in = np.arange(leads[ilead],re_target.shape[1]) + 1870
    
    
    
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))
    
    # Plot the amv classes
    for c in range(3):
        
        # Get the id for the class
        id_class = (y_actual_in == c)
        
        id_right = id_class * id_correct
        id_wrong = id_class * ~id_correct
        
        # Plot the correct ones
        ax.scatter(timeaxis_in[id_right],re_target_in[id_right],s=msize,marker="o",color=class_colors[c],facecolors="None")
        ax.scatter(timeaxis_in[id_wrong],re_target_in[id_wrong],s=msize,marker="x",color=class_colors[c])
        
    
    # Plot the actual AMV Index
    #ax.plot(timeaxis,re_target.squeeze(),color="k",lw=0.75,zorder=-9)
    ax.grid(True,ls="dashed")
    ax.minorticks_on()
    
    # Plot the Thresholds
    for th in thresholds_in:
        ax.axhline([th],color="k",ls="dashed")
    ax.axhline([0],color="k",ls="solid",lw=0.5)
    ax.set_xlim([timeaxis[0],timeaxis[-1]])
    
    class_str = "Class Acc: AMV+ (%.2f), Neutral (%.2f), AMV- (%.2f)" % (class_acc_all[imodel,ilead,0],
                                                                         class_acc_all[imodel,ilead,1],
                                                                         class_acc_all[imodel,ilead,2])
    ax.set_title("HadISST NASST Index and Prediction Results (1870-2022) \nNetwork #%i, Lead = %i years \n %s" % (imodel+1,leads[ilead],class_str))
    plt.savefig("%sHadISST_NASST_lead%02i_imodel%03i.png" %(figpath,leads[ilead],imodel,),dpi=150,bbox_inches='tight')




#%% Function version of above
def plot_scatter_predictions(imodel,ilead,y_predicted_all,y_actual_all,re_target,class_acc_all,msize=100,
                             figsize=(12,4),class_colors=('salmon', 'gray', 'cornflowerblue')):
    
    
    # Select the model
    y_predicted_in = y_predicted_all[imodel,ilead,:]
    y_actual_in    = y_actual_all[ilead,:]
    re_target_in   = re_target[:,leads[ilead]:].squeeze()
    id_correct     = (y_predicted_in == y_actual_in)
    
    timeaxis_in = np.arange(leads[ilead],re_target.shape[1]) + 1870
    
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))
    
    # Plot the amv classes
    for c in range(3):
        
        # Get the id for the class
        id_class = (y_actual_in == c)
        
        id_right = id_class * id_correct
        id_wrong = id_class * ~id_correct
        
        # Plot the correct ones
        ax.scatter(timeaxis_in[id_right],re_target_in[id_right],s=msize,marker="o",color=class_colors[c],facecolors="None")
        ax.scatter(timeaxis_in[id_wrong],re_target_in[id_wrong],s=msize,marker="x",color=class_colors[c])
        
    
    # Plot the actual AMV Index
    #ax.plot(timeaxis,re_target.squeeze(),color="k",lw=0.75,zorder=-9)
    ax.grid(True,ls="dashed")
    ax.minorticks_on()
    
    # Plot the Thresholds
    for th in thresholds_in:
        ax.axhline([th],color="k",ls="dashed")
    ax.axhline([0],color="k",ls="solid",lw=0.5)
    ax.set_xlim([timeaxis[0],timeaxis[-1]])
    
    class_str = "Class Acc: AMV+ (%.2f), Neutral (%.2f), AMV- (%.2f)" % (class_acc_all[imodel,ilead,0],
                                                                         class_acc_all[imodel,ilead,1],
                                                                         class_acc_all[imodel,ilead,2])
    return fig,ax

    
#%% MAKE A PLOT OF ABOVE, BUT WITH THE BEST performing model

ilead   = -1
id_best = total_acc_all[:,ilead].argmax()


fig,ax = plot_scatter_predictions(id_best,ilead,y_predicted_all,y_actual_all,re_target,class_acc_all,msize=100,
                             figsize=(12,4))

ax.set_ylim([-1.5,1.5])
ax.set_xlim([1890,2025])
ax.set_title("HadISST NASST Index and Prediction Results (1870-2022) \nNetwork #%i, Lead = %i years \n %s" % (id_best+1,leads[ilead],class_str))
plt.savefig("%sHadISST_NASST_lead%02i_imodel%03i.png" %(figpath,leads[ilead],imodel,),dpi=150,bbox_inches='tight')



#%% Make a histogram


# Visualize prediction count by year

# Select the model
#y_predicted_in = y_predicted_all[imodel,ilead,:]
#y_actual_in    = y_actual_all[ilead,:]
#re_target_in   = re_target[:,leads[ilead]:].squeeze()
#id_correct     = (y_predicted_in == y_actual_in)


count_by_year = np.zeros((ntime-leads[-1],nclasses))
timeaxis_in   = np.arange(leads[ilead],target.shape[1])

y_predicted_all = y_predicted_all.reshape(y_predicted_all.shape[0],nlead,nens,ntime-leads[-1])

# Assumes leads are not shuffled
for e in range(nens):
    for y in range(ntime-leads[ilead]):
        y_pred_year = y_predicted_all[...,e,y]
        
        for c in range(3):
            
            count_by_year[y,c] = (y_pred_year == c).sum()

#%% Barplot of Year vs Prediction frequency
# for c in range(3):
#     y_predicted_all == 
#     y_predicted_all
    

fig,ax       = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))

for c in range(3):
    label = classes[c]
    #label = "%s (Test Acc = %.2f" % (classes[c],class_acc[c]*100)+"%)"
    
    ax.bar(timeaxis_in+1920,count_by_year[:,c],bottom=count_by_year[:,:c].sum(1),
           label=label,color=class_colors[c],alpha=0.75,edgecolor="white")

ax.set_ylabel("Frequency of Predicted Class")
ax.set_xlabel("Year")

ax.minorticks_on()
ax.grid(True,ls="dotted")
ax.set_xlim([1920+20,2010])
#ax.set_ylim([0,450])

ax2    = ax.twinx()
ls_ens = ["solid","dashdot"]
for e in range(nens):
    ax2.plot(timeaxis+1920,target[e,:].squeeze(),color="k",ls=ls_ens[e],label="NASST Index, ens%02i"%(e+1))
ax2.set_ylabel("NASST Index ($\degree C$)")
ax2.set_ylim([-1.3,1.3])
for th in thresholds_in:
    ax2.axhline([th],color="k",ls="dashed")
ax2.axhline([0],color="k",ls="solid",lw=0.5)
ax.legend()
ax2.legend(ncol=2)
plt.savefig("%sCESMWH_Prediction_Count_AllLeads.png"%figpath,dpi=150,bbox_inches="tight")
#%% Try the above, but get prediction count for selected leadtimes
# Q : Is there a systematic shift towards the selected leadtimes?
selected_leads      = [0,6,12,18,24]
nleads_sel          = len(selected_leads)

count_by_year_leads = np.zeros((ntime-leads[-1],nclasses,nleads_sel))

# Assumes leads are not shuffled
for y in range(ntime-leads[ilead]):
    
    for ll in range(nleads_sel):
        sel_lead_index = list(leads).index(selected_leads[ll])
        y_pred_year = y_predicted_all[...,sel_lead_index,y]
    
        for c in range(3):
            
            count_by_year_leads[y,c,ll] = (y_pred_year == c).sum()


#%% 
fig,axs       = plt.subplots(3,1,constrained_layout=True,figsize=(16,8))



lead_colors = ["lightsteelblue","cornflowerblue","royalblue","mediumblue","midnightblue"]
for c in range(3):
    ax = axs[c]
    
    for ll in range(nleads_sel):
        ax.plot(timeaxis_in+1870,count_by_year_leads[:,c,ll],label="%02i-yr Lead" % selected_leads[ll],lw=1.5,c=lead_colors[ll])
        
    if c == 0:
        ax.legend()
    ax.set_title(classes[c])
    
    ax.set_xlabel("Year")
    ax.minorticks_on()
    ax.grid(True,ls="dashed")
    
    # label = "%s (Test Acc = %.2f" % (classes[c],class_acc[c]*100)+"%)"
    # ax.bar(timeaxis_in+1870,count_by_year[:,c],bottom=count_by_year[:,:c].sum(1),
    #        label=label,color=class_colors[c],alpha=0.75,edgecolor="k")
    
    ax.set_ylabel("Predicted Class Count")

plt.savefig("%sHadISST_Class_Prediction_Frequency_byYear.png"%(figpath),dpi=150,bbox_inches="tight")


#%% Remake barplot. but for the selected leadtimes
def make_count_barplot(count_by_year,lead,leadmax=24,classes=['AMV+', 'Neutral', 'AMV-'],
                       class_colors=('salmon', 'gray', 'cornflowerblue')
                       ):
    
    timeaxis      = np.arange(0,len(re_target.squeeze()))
    timeaxis_in   = np.arange(leadmax,re_target.shape[1])
    
    fig,ax       = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))
    for c in range(3):
        label = classes[c]
        ax.bar(timeaxis_in+1870,count_by_year[:,c],bottom=count_by_year[:,:c].sum(1),
               label=label,color=class_colors[c],alpha=0.75,edgecolor="white")
    
    ax.set_ylabel("Frequency of Predicted Class")
    ax.set_xlabel("Year")
    ax.legend()
    ax.minorticks_on()
    ax.grid(True,ls="dotted")
    ax.set_xlim([1880,2025])
    ax.set_ylim([0,450])

    ax2 = ax.twinx()
    ax2.plot(timeaxis,re_target.squeeze(),color="k",label="HadISST NASST Index")
    ax2.set_ylabel("NASST Index ($\degree C$)")
    ax2.set_ylim([-1.3,1.3])
    for th in thresholds_in:
        ax2.axhline([th],color="k",ls="dashed")
    ax2.axhline([0],color="k",ls="solid",lw=0.5)
    axs = [ax,ax2]
    return fig,axs


for ll in range(nleads_sel):
    lead = selected_leads[ll]
    ilead = list(leads).index(lead)
    

    fig,axs = make_count_barplot(count_by_year_leads[:,:,ll],lead,re_target,)

    plt.savefig("%sHadISST_Prediction_Count_Lead%02i.png"% (figpath,lead),dpi=150,bbox_inches="tight")
    


#%%

ax.set_title(title)
ax.set_ylim([0,10])
plot_mode = 0

for plot_mode in range(2):
    ax = axs[plot_mode]
    ax = format_axis(ax,x=timeaxis)
    if plot_mode == 0:
        title = "Actual Class"
    elif plot_mode == 1:
        title = "Predicted Class"
    testc = np.arange(0,3)
    for c in range(3):
        label = "%s (Test Acc = %.2f" % (class_names[c],class_acc[c]*100)+"%)"
        if debug:
            print("For c %i, sum of prior values is %s" % (c,testc[:c]))
        ax.bar(timeaxis,count_by_year[:,c,plot_mode],bottom=count_by_year[:,:c,plot_mode].sum(1),
               label=label,color=class_colors[c],alpha=0.75,edgecolor="k")
    ax.set_title(title)
    ax.set_ylim([0,10])
    if plot_mode == 0:
        ax.legend()
plt.suptitle("AMV Class Distribution by Year (%s) \n %s" % (modelname,exp_titlestr))
if savefig:
    plt.savefig("%sClass_Distr_byYear_%s_lead%02i_nepochs%02i.png" % (figpath,varnames[v],lead,epoch_axis[-1]),dpi=150)