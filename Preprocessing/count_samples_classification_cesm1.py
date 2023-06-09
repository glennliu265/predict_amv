#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script for counting samples for CESM1 Training

Copied upper section of train_NN_CESM1

Created on Mon Mar 27 22:36:05 2023

@author: gliu
"""

import sys
import numpy as np
import os
import time
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset,Dataset
import matplotlib.pyplot as plt

import cartopy.crs as ccrs

#%% Load custom packages and setup parameters
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

# Load Predictor Information
bbox          = pparams.bbox
figpath       = pparams.figpath
proc.makedir(figpath)
# ============================================================
#%% User Edits vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ============================================================

# Set experiment directory/key used to retrieve params from [train_cesm_params.py]
expdir             = "FNN4_128_SingleVar_PaperRun"#"FNN4_128_SingleVar_Rerun100"
eparams            = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Set some looping parameters and toggles
varnames           = ["SST","SSS","SLP","SSH"]       # Names of predictor variables
leads              = np.arange(0,26,1)    # Prediction Leadtimes
runids             = np.arange(0,1,1)    # Which runs to do

# Other toggles
checkgpu           = True                 # Set to true to check if GPU is availabl
debug              = True                 # Set verbose outputs
savemodel          = True                 # Set to true to save model weights

# Save looping parameters into parameter dictionary
eparams['varnames'] = varnames
eparams['leads']    = leads
eparams['runids']   = runids

# ============================================================
# End User Edits ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ============================================================
# ------------------------------------------------------------
# %% 01. Check for existence of experiment directory and create it
# ------------------------------------------------------------
allstart = time.time()

proc.makedir("../../CESM_data/"+expdir)
for fn in ("Metrics","Models","Figures"):
    proc.makedir("../../CESM_data/"+expdir+"/"+fn)
    
    
# Check if there is gpu
if checkgpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

# ----------------------------------------------
#%% 02. Data Loading...
# ----------------------------------------------

# Load some variables for ease
ens            = eparams['ens']
norm           = eparams['norm']

# Loads that that has been preprocessed by: ___

# Load predictor and labels, lat/lon, cut region
target         = dl.load_target_cesm(detrend=eparams['detrend'],region=eparams['region'],newpath=True,norm=norm)
data,lat,lon   = dl.load_data_cesm(varnames,eparams['bbox'],detrend=eparams['detrend'],return_latlon=True,newpath=True)

# Make a mask and apply
limask                         = dl.load_limask(bbox=eparams['bbox'])
data                           = data * limask[None,None,None,:,:]  # NaN Points to Zero

# Normalize data
nchannels = data.shape[0]
# *** Note, doing this for each channel, but in reality, need to do for all channels
for ch in range(nchannels):
    std_var = np.nanstd(data[ch,...])
    mu_var = np.nanmean(data[ch,...])
    data[ch,...] = (data[ch,...] - mu_var)/std_var
[print(am.normalize_ds(data[d,...])) for d in range(4)]

# Change nan points to zero
data[np.isnan(data)] = 0 
# Subset predictor by ensemble, remove NaNs, and get sizes

# Subset predictor, get dimensions
data                           = data[:,0:ens,...]                  # Limit to Ens
target                         = target[0:ens,:]
nchannels,nens,ntime,nlat,nlon = data.shape                         # Ignore year and ens for now...
inputsize                      = nchannels*nlat*nlon                # Compute inputsize to remake FNN

for i in range(4):
    plt.figure(),plt.pcolormesh(data[i,1,2,...]),plt.colorbar()

# ------------------------------------------------------------
# %% 03. Determine the AMV Classes
# ------------------------------------------------------------

# Set exact threshold value
std1         = target.std(1).mean() * eparams['thresholds'][1] # Multiple stdev by threshold value 
if eparams['quantile'] is False:
    thresholds_in = [-std1,std1]
else:
    thresholds_in = eparams['thresholds']

# Classify AMV Events
target_class = am.make_classes(target.flatten()[:,None],thresholds_in,exact_value=True,reverse=True,quantiles=eparams['quantile'])
target_class = target_class.reshape(target.shape)

# Get necessary dimension sizes/values
nclasses     = len(eparams['thresholds'])+1
nlead        = len(leads)

"""
# Output: 
    predictors :: [channel x ens x year x lat x lon]
    labels     :: [ens x year]
"""

#%% Check counts for longest leadtime

target_class = am.make_classes(target[:,25:].flatten()[:,None],thresholds_in,exact_value=True,reverse=True,quantiles=eparams['quantile'])
am.count_samples(None,target_class)

#%% Make a plot for AMV Prediction Draft

e       = 36

lead0   = 1940
N       = 25

startyr = 1920
yrs     = np.arange(startyr,startyr+target.shape[1])

# Get lead0 and leadN indices
ipred   = np.where(yrs==lead0)[0][0]
ilabel  = ipred+N
lead1   = lead0+N
xtk     = [lead0,lead1]
xtklabs = (str(lead0) + "\n(lead=0)",str(lead1)+"\n(lead=%i)"%N) 



fig,ax= plt.subplots(1,1,figsize=(8,3),constrained_layout=True)
ax.plot(yrs,target[e,:],c="gray",label="Ensemble member %02i" % (e+1),lw=2.5)

ax.spines[['right', 'top']].set_visible(False)

ax.axhline([0],ls="dotted",color="k",lw=0.55)
ax.axhline([-std1],ls="dashed",color="cornflowerblue",lw=0.75,label="")
ax.axhline([std1],ls="dashed",color="red",lw=0.75,label="1$\sigma$")
ax.legend(loc="lower center")
ax.minorticks_on()
ax.set_ylim([-1.25,1.25])
ax.set_xlim([1935,1970])
ax.set_ylabel("NASST Index ($\degree$C)")
ax.set_xlabel("Time (Years)")

# Plot sample leadtimes for prediction
ax.set_xticks(xtk,labels=xtklabs) # Label lead 0 and lead N
ax.scatter(lead0,target[e,ipred],marker=".",color="k",label="",
        s=255,edgecolors="k",zorder=3,alpha=0.8) # Plot lead0
ax.axvline([lead0],color="k",ls="dashed",lw=0.75,zorder=1)
ax.scatter(lead1,target[e,ilabel],marker="X",color="cornflowerblue",label="",
        s=230,edgecolors="k",zorder=3) # Plot lead0
ax.axvline([lead1],color="k",ls="dashed",lw=0.75,zorder=1)
     
savename = "%sNASST_Prediction_Example_Ens%i_Start%04i_lead%02i.svg" % (figpath,e+1,lead0,N)
plt.savefig(savename,transparent=True,dpi=300)

#%% Plot predictors at that time

consistent_mask = np.sum(data,(0,1,2)) == 0

use_contour = True
nvars = len(varnames)
vlims = [-2,2]
vcmap = ["cmo.balance","cmo.delta","cmo.curl","PuOr_r"]
varunits = [("normalized"),] * nvars
for v in range(nvars):
    
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree(0)},
                          constrained_layout=True,figsize=(4,3))
    
    ax.set_extent(bbox)
    plotdata = data[v,e,ipred,:,:].copy() * limask
    if use_contour:
        pcm = ax.contourf(lon,lat,plotdata,levels=np.arange(-2.0,2.2,0.2),cmap=vcmap[v],extend="both")
    else:
        pcm = ax.pcolormesh(lon,lat,plotdata,vmin=vlims[0],vmax=vlims[1],cmap=vcmap[v])
    viz.add_coast_grid(ax=ax,bbox=bbox,blabels=[0,0,0,0],fill_color="k")
    ax.set_title("%s (normalized)" % (varnames[v]))
    cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.035,pad=0.02,
                      ticks=[-2,-1,0,1,2])
    #cb.ax.set_yticklabels(['< -1', '0', '> 1'])
    #cb.set_label("%s (%s)" % (varnames[v],varunits[v]))
    savename = "%sNASST_Prediction_Example_Ens%i_Start%04i_predictor%s.svg" % (figpath,e+1,lead0,varnames[v])
    plt.savefig(savename,transparent=True,dpi=300)
    
    
    
    
    



#%% Add option to load existing runid?



# Do some dummy selections
v = 0
predictors = data[[v],...] # Get selected predictor
k_offset = 0
# Preallocate
nruns = len(runids)
nleads = len(leads)
nsamples = eparams['nsamples']
varname  = "SST"



y_subsets_all    = []
shuffid_all      = []
sample_size_all  = []
idx_byclass_all  = []
total_count_byclass = np.zeros((nruns,nleads,nclasses))

# --------------------
# 05. Loop by runid...
# --------------------
for nr,runid in enumerate(runids):
    rt = time.time()
    
    # Preallocate Evaluation Metrics...
    sampled_idx          = []
    sample_sizes         = []
    y_subsets_lead       = []
    idx_byclass_lead     = []
    # -----------------------
    # 07. Loop by Leadtime...
    # -----------------------
    for l,lead in enumerate(leads):
        
        # --------------------------
        # 08. Apply lead/lag to data
        # --------------------------
        # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
        X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=ens,tstep=ntime)
        
        
        idx_by_class,count_by_class=am.count_samples(nsamples,y_class)
        total_count_byclass[nr,l,:] = count_by_class
        
        # ----------------------
        # 09. Select samples
        # ----------------------
        if eparams['nsamples'] is None: # Default: nsamples = smallest class
            threscount = np.zeros(nclasses)
            for t in range(nclasses):
                threscount[t] = len(np.where(y_class==t)[0])
            eparams['nsamples'] = int(np.min(threscount))
            print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
       
        y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle'])
        

        
        
        # --------------------------
        # 10. Train Test Split
        # --------------------------
        X_subsets,y_subsets      = am.train_test_split(X,y_class,eparams['percent_train'],
                                                       percent_val=eparams['percent_val'],
                                                       debug=True,offset=k_offset)
        
        
        sampled_idx.append(shuffidx) # Save the sample indices
        sample_sizes.append(eparams['nsamples'])
        y_subsets_lead.append(y_subsets)
        idx_byclass_lead.append(idx_by_class)
        print("\nCompleted counting for %s lead %i of %i" % (varname,lead,leads[-1]))
    
    
    shuffid_all.append(sampled_idx)
    sample_size_all.append(sample_size_all)
    y_subsets_all.append(y_subsets_lead)
    idx_byclass_all.append(idx_byclass_lead)
    print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
    # End Runid Loop >>>

#%% Plot total class counts by leadtime
fig,axs = plt.subplots(1,2,figsize=(8,3))

for c in range(3):
    if c == 1:
        ax = axs[1]
    else:
        ax = axs[0]
    for nr in range(nruns):
        clabel = "%s (Run %i,n=%i-%i)" % (pparams.classes[c],runids[nr],total_count_byclass[nr,:,c].min(),total_count_byclass[nr,:,c].max())
        ax.plot(leads,total_count_byclass[nr,:,c],label=clabel,color=pparams.class_colors[c])
        
    ax.grid(True)
    ax.legend()
    
#%% Plot total classes by 
    

             