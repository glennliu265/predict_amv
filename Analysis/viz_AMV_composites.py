#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMV Composites

Visualize AMV composites at different lead times for comparison with 
the LRP maps.
User Edits and upper sections copied from viz_LRP_FNN

Created on Thu Oct 13 11:27:05 2022

@author: gliu
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import time
import sys

from tqdm import tqdm
import matplotlib.ticker as tick
import cmoean as cmo

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
quantile    = False
nruns,nleads,nthres = len(runs),len(leads),len(thresholds)+1,

# -------------------
# Plot Settings
# -------------------
proj            = ccrs.PlateCarree()
vnames          = ("SST","SSS","SLP")
thresnames      = ("AMV+","Neutral","AMV-",)
cmnames_long    = ("True Positive","False Positive","False Negative","True Positive")
scale_relevance = True # Set to True to scale relevance values for each sample to be betweeen -1 and 1 after compositing
cmbal = cmo.cm.balance.copy()
cmbal.set_under('b')
cmbal.set_over('r')


cm_names = ['TP', 'FP', 'FN', 'TN']
cmnames_long    = ("True Positive","False Positive","False Negative","True Positive")

# -------------------
# Load Lat/Lon
# -------------------
lat2 = np.load("%slat_2deg_NAT.npy"% (datpath))
lon2 = np.load("%slon_2deg_NAT.npy"% (datpath))

#%% Load the data


# Load in input and labels 
if regrid is None:
    data   = np.load(datpath+ "CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i_regridNone.npy" % detrend) # [variable x ensemble x year x lon x lat]
    target = np.load(datpath+ "CESM_label_amv_index_detrend%i_regridNone.npy" % detrend)
else:
    data   = np.load(datpath+ "CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i_regrid%i.npy" % (detrend,regrid)) # [variable x ensemble x year x lon x lat]
    target = np.load(datpath+ "CESM_label_amv_index_detrend%i_regrid%i.npy" % (detrend,regrid))


data   = data[:,:ens,:tstep,:,:] # [Channel, Ens, Year, Lat, Lon]
target = target[:ens,:tstep] # [Ens x Year]
nvar,nens,ntime,nlat,nlon = data.shape
print(data.shape)
print(target.shape)

#%% Load normalization factors

mu,sigma = np.load("%sCESM_nfactors_detrend%i_regrid%s.npy" % (datpath,detrend,regrid))




#%% Compute composites

vcomposites = np.zeros((nleads,nthres,nvar,nlat,nlon)) * np.nan
vvariances  = vcomposites.copy()
vcounts     = np.zeros((nleads,nthres))
for l in tqdm(range(nleads)):
    lead = leads[l]
    print(lead)
    
    # Restrict data X --> [sample x channel x lat x lon]
    X = (data[:,:ens,:tstep-lead,:,:]).reshape(3,ens*(tstep-lead),nlat,nlon).transpose(1,0,2,3)
    y = target[:ens,lead:].reshape(ens*(tstep-lead),1) # [sample x 1]
    nsamples = X.shape[0]
    
    # Calculate classes, adjust for quantile choice
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
    print(y_class.shape)
    
    # Looping for each threshold
    for th in range(nthres):
        
        # Select that variable
        ithres  = np.where(y_class==th)[0]
        
        vcomposites[l,th,:,:,:] = X[ithres,:,:,:].mean(0)
        vvariances[l,th,:,:,:]  = X[ithres,:,:,:].var(0)
        vcounts[l,th] = len(ithres)
        
# Set lat/lon variables (for plotting)
lat  =  np.linspace(lat2[0],lat2[-1],nlat)
lon  = np.linspace(lon2[0],lon2[-1],nlon)
    
#%% Visualize these composites (by class)

th          = 0
set_clvls   = True
normalized  = True

if normalized:
    var_clvls   = (np.arange(-1.3,1.4,0.1),)*3
else:
    var_clvls   = (
        np.arange(-0.40,0.41,0.01),
        np.arange(-0.12,0.130,0.010),
        np.arange(-50,55,5)
        )

for th in range(3):
    fig,axs=plt.subplots(3,nleads,figsize=(18,6),
                         subplot_kw={'projection':proj},constrained_layout=True)
    for v in range(nvar):
        for l in range(nleads):
            ax = axs[v,l]
            ax.coastlines()
            ax.set_extent([-80,0,0,65])
            
            if v == 0:
                ax.set_title("Lead=%02i yrs" % (leads[l]))
            if l == 0:
                ax.text(-0.05, 0.55, vnames[v], va='bottom', ha='center',rotation='vertical',
                                 rotation_mode='anchor',transform=ax.transAxes)
                
            plotvar = vcomposites[l,th,v,:,:]
            if normalized is False:
                plotvar = plotvar * sigma[v] + mu[v]
            if set_clvls:
                cf = ax.contourf(lon,lat,plotvar,cmap='RdBu_r',levels=var_clvls[v],extend='both')
                cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75,levels=var_clvls[v])
                ax.clabel(cl,levels=var_clvls[v][::2])
            else:
                cf = ax.contourf(lon,lat,plotvar,cmap='RdBu_r',extend='both')
                cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75)
                ax.clabel(cl)
        
        fig.colorbar(cf,ax=axs[v,:].flatten(),fraction=0.025,pad=0.01)
    plt.suptitle("%s Lead Patterns" % (thresnames[th]))
    plt.savefig("%sLeadtime_Composites_%s_clvls%i_normalize%i.png" % (figpath,thresnames[th],set_clvls,normalized),dpi=150,bbox_inches='tight')
        

#%% Make equivalent figures, but now for the model predictions

# Load indices
cm_savename = outpath+"../FNN2_confmatids_detrend%i_regrid%s.npy" % (detrend,regrid)
cmids_lead  = np.load(cm_savename,allow_pickle=True) # [lead][run x class x confmat quadrant x ids (ens*{tstep-lead})]


#%% For each model, recreate the above figure

c               = 0
plot_relevances = True # Set to true to load in relevances and plot it.
normalize_rel   = True
vmax_rel        = 0.001

#%% Load the relevance data for a given leadtime...
if plot_relevances:
    st       = time.time()
    ids_class = []
    rels = []
    for l in tqdm(range(nleads)):
        lead = leads[l]
        savename = "%sLRPout_lead%02i_gamma%.3f_epsilon%.3f.npz" % (outpath,lead,gamma,epsilon)
        npz      = np.load(savename,allow_pickle=True)
        ndict    = [npz[z] for z in npz.files]
        relevances,ids,y_pred,y_targ,y_val,lead,gamma,epsilon,allow_pickle=ndict
        rels.append(relevances)
        ids_class.append(ids)
    print("Loaded data in %.2fs"% (time.time()-st))

# rels : [lead][run,class][sample x variable x lat x lon]

#%%
for r in range(nruns):
    for th in range(3):
        fig,axs=plt.subplots(3,nleads,figsize=(18,6),
                             subplot_kw={'projection':proj},constrained_layout=True)
        
        for l in range(nleads):
            lead = leads[l]
            
            # Make the composite
            ids  = cmids_lead[l][r,th,c,:].astype(bool) # [samples,]
            X    = (data[:,:ens,:tstep-lead,:,:]).reshape(3,ens*(tstep-lead),nlat,nlon).transpose(1,0,2,3)
            x_in = X[ids,:,:,:].mean(0) # [variable x lat x lon]
            
            # Calculate True Positive Rate
            cmcounts = cmids_lead[l][r,th,:,:].sum(1)
            TP,FP,FN,TN = cmcounts
            plotacc     = TP / (TP+FN)
            
            # Composite the relevances
            
            #relevances[r,th][id_sel[ids[r,th]],v,:,:].mean(0)
            
            id_confm       = cmids_lead[l][r,th,c,:].astype(bool) # Indices from full variable corresponding to quadrant
            id_class_confm = id_confm[ids_class[l][r,th]] # Select the class indices from those
            rel_lead        = rels[l][r,th][:,:,:,:][id_class_confm,:,:,:].mean(0) # Make composite
            
            for v in range(nvar): # Reverse the order compared to above
                
                ax = axs[v,l]
                ax.coastlines()
                ax.set_extent([-80,0,0,65])
                
                if v == 0:
                    ax.set_title("Lead=%02i yrs" % (leads[l]))
                if l == 0:
                    ax.text(-0.05, 0.55, vnames[v], va='bottom', ha='center',rotation='vertical',
                                     rotation_mode='anchor',transform=ax.transAxes)
                
                plotvar = x_in[v,:,:]
                
                if plot_relevances:
                    
                    plotrel = rel_lead[v,:,:]
                    if normalize_rel:
                        plotrel = plotrel/np.max(np.abs(plotrel.flatten()))
                        vmax_rel = 1
                    
                    if normalized is False:
                        plotvar = plotvar * sigma[v] + mu[v]
                    
                    if set_clvls:
                        cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75,levels=var_clvls[v])
                        ax.clabel(cl,levels=var_clvls[v][::2])
                        cf = ax.pcolormesh(lon,lat,plotrel,cmap='RdBu_r',vmin=-vmax_rel,vmax=vmax_rel,alpha=0.75)
                        
                    else:
                        cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75)
                        ax.clabel(cl)
                        cf = ax.pcolormesh(lon,lat,plotrel,cmap='RdBu_r',vmin=-vmax_rel,vmax=vmax_rel,alpha=0.75)
                        
                else:
                    if normalized is False:
                        plotvar = plotvar * sigma[v] + mu[v]
                    if set_clvls:
                        cf = ax.contourf(lon,lat,plotvar,cmap='RdBu_r',levels=var_clvls[v],extend='both')
                        cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75,levels=var_clvls[v])
                        ax.clabel(cl,levels=var_clvls[v][::2])
                    else:
                        cf = ax.contourf(lon,lat,plotvar,cmap='RdBu_r',extend='both')
                        cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75)
                        ax.clabel(cl)
                
                if lead == 24:
                    fig.colorbar(cf,ax=axs[v,:].flatten(),fraction=0.025,pad=0.01)
        plt.suptitle("%s %s Lead Patterns (run %02i, TPR=%.02f" % (cmnames_long[c],thresnames[th],r,plotacc*100)+"%)")
        savename = "%sLeadtime_Composites_%s_clvls%i_normalize%i_%s_run%02i.png" % (figpath,thresnames[th],set_clvls,normalized,cm_names[c],r)
        if plot_relevances:
            savename = "%sLeadtime_Composites_Relevances_%s_clvls%i_normalize%i_%s_run%02i.png" % (figpath,thresnames[th],set_clvls,normalized,cm_names[c],r)
        plt.savefig(savename,dpi=150,bbox_inches='tight')
            

