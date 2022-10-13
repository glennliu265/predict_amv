#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get Confusion matrix indices for given predictions
- Currently only supports FNN
- Copies over section from viz_LRP_FNN

Created on Thu Oct 13 15:16:32 2022

@author: gliu
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import time
import sys

from tqdm import tqdm
import matplotlib.ticker as tick

#%% User Edits

# -------------------
# Data settings
# -------------------
regrid      = None
detrend     = 0
ens         = 40
tstep       = 86

# -------------------
# Indicate paths
# -------------------
datpath = "../../CESM_data/"
if regrid is None:
    modpath = datpath + "Models/FNN2_quant0_resNone/"
else:
    modpath = datpath + "Models/FNN2_quant0_res224/"
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20221014/"
outpath = datpath + "Metrics/"

# -------------------
# Modules
# -------------------
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import viz
# Load modules (LRPutils by Peidong)
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/scrap/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/predict_amv/")
import LRPutils as utils
import amvmod as am

# ----------------
# Settings for LRP
# ----------------
gamma       = 0.25
epsilon     = 0.25 * 0.36 # 0.36 is 1 stdev of the AMV Index

# -------------------
# Training Parameters
# -------------------
runs        = np.arange(1,11,1)
leads       = np.arange(0,24+3,3)
thresholds  = [-1,1]
nruns,nleads,nthres = len(runs),len(leads),len(thresholds)+1,

# -------------------
# Plot Settings
# -------------------
proj            = ccrs.PlateCarree()
vnames          = ("SST","SSS","SLP")
thresnames      = ("AMV+","Neutral","AMV-",)
cmnames_long    = ("True Positive","False Positive","False Negative","True Positive")
scale_relevance = True # Set to True to scale relevance values for each sample to be betweeen -1 and 1 after compositing
plot_composites = False


# -------------------
# Load Lat/Lon
# -------------------
lat2 = np.load("%slat_2deg_NAT.npy"% (datpath))
lon2 = np.load("%slon_2deg_NAT.npy"% (datpath))

# -------------------------------------------------------------------------
# Load data normalization factors (see prepare_training_validation_data.py)
# -------------------------------------------------------------------------
vmeans,vstdevs = np.load(datpath+"CESM_nfactors_detrend0_regridNone.npy") # [Mean,Stdev][SST,SSS,SLP]
#%% Load the data, get conf matrix indices

cmids_lead = []
for l,lead in tqdm(enumerate(np.arange(0,24+3,3))): # (0,24+3,3)
    
    # Load the relevance data for a given leadtime...
    st       = time.time()
    savename = "%sLRPout_lead%02i_gamma%.3f_epsilon%.3f.npz" % (outpath,lead,gamma,epsilon)
    npz      = np.load(savename,allow_pickle=True)
    ndict    = [npz[z] for z in npz.files]
    relevances,ids,y_pred,y_targ,y_val,lead,gamma,epsilon,allow_pickle=ndict
    print("Loaded data in %.2fs"% (time.time()-st))
    
    # Load y predictions back into an array of the same size
    """
    ** NOTE: Modify LRP processing script to this step can be avoided...
    """
    y_pred_new = np.zeros( (nruns,)+y_val.shape) * np.nan # [Model x Samples x 1]
    for r in range(nruns):
        for th in range(3):
            ithres               = ids[r,th]
            y_pred_new[r,ithres,0] = y_pred[r,th].numpy()
    nsamples = y_targ.shape[0]
    
    #%% Compute confusion matrices
    cmids_all    = np.zeros([nruns,nthres,4,nsamples]) * np.nan # [run][Class,Confmat_quadrant,Indices]
    #cmcounts_all = [] # Get thus by running cmids_all[r].sum(-1)
    #cmtotals_all = [] # This is the nsamples for the given lead 
    for r in range(nruns):
        cm_ids,cm_counts,cm_totals,cm_acc,cm_names = am.calc_confmat_loop(y_pred_new[r,...],y_targ)
        cmids_all[r,:,:,:] = cm_ids
    
    # These indices are relative to y_targ, which has been shifted by [:ens,lead:] ...
    cmids_lead.append(cmids_all)

# Convert to a numpy array
cmids_lead_arr = np.empty(nleads,dtype='object') # [lead][run x class x confmat quadrant x ids (ens*{tstep-lead})]
for l in range(nleads):
    cmids_lead_arr[l] = cmids_lead[l]

savename = outpath+"../FNN2_confmatids_detrend%i_regrid%s.npy" % (detrend,regrid)
np.save(savename,cmids_lead_arr,allow_pickle=True)
#np.savez(savename,**{'cmids_lead':cmids_lead},allow_pickle=True)
