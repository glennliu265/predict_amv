#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize the AMV spectra from the data

Created on Fri Oct 14 11:13:05 2022

@author: gliu
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import time
import sys

from tqdm import tqdm
import matplotlib.ticker as tick
import cmocean as cmo


#%%

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
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20221021/"
outpath = datpath + "Metrics/"

# -------------------
# Modules
# -------------------
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm
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
cmbal           = cmo.cm.balance.copy()
cmbal.set_under('b')
cmbal.set_over('r')


cm_names        = ['TP', 'FP', 'FN', 'TN']
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

# Set lat/lon variables (for plotting)
lat  =  np.linspace(lat2[0],lat2[-1],nlat)
lon  = np.linspace(lon2[0],lon2[-1],nlon)

#%%


y1 = target[0,:]
x1 = np.mean(data[0,0,:,:,:],(1,2))


xper = np.array([75,50,30,25,20,15,10,5])
xtks = 1/xper


nsmooths = np.ones(ens)*3
pct      = 0.10
dtplot   = 3600*24*365
dt       = 3600*24*365


amvids         = [target[s,:] for s in range(40)]
ensmean        = target.mean(0)
amvids_dt      = [target[s,:]-ensmean for s in range(40)]  # Detrended by subtracting ens_mean from index, NOT individual points...

specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(amvids,nsmooths,pct,dt=dt)
specs_dt,freqs,CCs,dofs,r1s = scm.quick_spectrum(amvids_dt,nsmooths,pct,dt=dt)

#%%
fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(8,6))

for e in tqdm(range(ens)):
    
    ax = axs[0]
    ax.plot(freqs[e]*dtplot,specs[e]/dtplot,
            marker=".",color='gray',alpha=0.5,label="")
    
    ax = axs[1]
    ax.plot(freqs[e]*dtplot,specs_dt[e]/dtplot,
            marker=".",color='cornflowerblue',alpha=0.9,label="")
    
    
    if e == 0:
        #freqsmean    = freqs[e]
        specsmean    = specs[e]
        specsmean_dt = specs_dt[e]
    else:
        #freqsmean    = freqsmean + freqs[e]
        specsmean    = specsmean + specs[e]
        specsmean_dt = specsmean_dt + specs_dt[e]


ax = axs[0]
ax.plot(freqs[e]*dtplot,specs[e]/dtplot,
        marker=".",color='gray',alpha=0.5,label="Indv. Member (Undetrended)")
ax.plot(freqs[e]*dtplot,specsmean/ens/dtplot,color='k',alpha=1.0,marker=".",
        label="Ens. Mean (Undetrended)")

ax = axs[1]

ax.plot(freqs[e]*dtplot,specs_dt[e]/dtplot,
        marker=".",color='cornflowerblue',alpha=0.9,label="Indv. Member (Detrended)")
ax.plot(freqs[e]*dtplot,specsmean_dt/ens/dtplot,color='b',alpha=1.0,marker=".",
        label="Ens. Mean (Detrended)")

for a,ax in enumerate(axs):
    ax.set_xticks(xtks)
    ax.set_xlim([xtks[0],xtks[-1]])
    ax.set_xticklabels(xper.astype(int))
    ax.grid(True,ls='dotted')
    ax.legend()
    
    
    ax.set_ylabel("Power ($\degree C^{2} cpy^{-1}$)")
    
    if a == 0:
        ax.set_ylim([0,4.0])
    else:
        ax.set_ylim([0,1.25])
        ax.set_xlabel("Period (Years)")
    
plt.suptitle("AMV Index Power Spectra (CESM1-LE, 1920-2005)")
plt.savefig("%sAMV_Index_Power_Spectra_Detrend_Undetrend.png" % (figpath),dpi=200)
#%% Quick scatter comparing forced and unforced AMV variances

# Convert from list to nparray
amvids_tr  = np.array(amvids)    # [ens x time]
amvids_ntr = np.array(amvids_dt) # [ens x time]

# Set some plotting params
ensnum = np.arange(1,41)

# Compute variance and scatter
fig,ax = plt.subplots(1,1)
sc = ax.scatter(amvids_tr.var(1),amvids_ntr.var(1))


for i, txt in enumerate(ensnum):
    ax.annotate(txt, (amvids_tr.var(1)[i], amvids_ntr.var(1)[i]))


ax.grid(True,ls='dotted')
ax.set_xlabel("AMVi Variance (With Trend)")
ax.set_ylabel("AMVi Variance (Detrended)")

# Need to think of what the "standard" is, or how to draw the line of larger
# or smaller external forcing 
# Maybe a simple barplot of the difference would be more informative
# With lines indicating the average difference...
ax.set_ylim([.04,0.2])
ax.set_xlim([.04,0.2])

#%% Quick Barplot of the above

percent_diff = True

# Compute variance and scatter
fig,ax = plt.subplots(1,1,figsize=(12,4),constrained_layout=True)


amvi_vardiff = amvids_tr.var(1)-amvids_ntr.var(1) # [ens,]
if percent_diff: # Express as percentage of original variance
    amvi_vardiff = amvi_vardiff/amvids_tr.var(1)
meanvar      = (amvi_vardiff).mean()




ax.bar(ensnum,amvi_vardiff,color='darkviolet',alpha=0.5,edgecolor="k")

# Get top and bottom values 5
topid        = proc.get_topN(amvi_vardiff,5,sort=True)
botid        = proc.get_topN(amvi_vardiff,5,sort=True,bot=True)




ax.axhline(meanvar,ls='dashed',color='k',label="Mean Difference = %.03f $\degree C^2$" % (meanvar))


if percent_diff:
    ax.vlines(np.array(topid)+1,ymin=0,ymax=0.45,color="r")
    ax.vlines(np.array(botid)+1,ymin=0,ymax=0.45,color="b")
    ax.set_ylabel("% Decrease in AMVi Variance after detrending")
    plt.suptitle("AMV Index (AMVi) Variance Difference $\frac{With Trend - Detrended}{With Trend}$")
else:
    ax.vlines(np.array(topid)+1,ymin=0,ymax=0.1,color="r")
    ax.vlines(np.array(botid)+1,ymin=0,ymax=0.1,color="b")
    ax.set_ylabel("Decrease in AMVi Variance after detrending ($\degree C^2$)")
    plt.suptitle("AMV Index (AMVi) Variance Difference (With Trend - Detrended)")
ax.set_xlabel("Ensemble Member")

ax.set_xticks(ensnum)
ax.set_xlim([0,41])
ax.legend()
ax.grid(True,ls='dotted')




plt.savefig("%sAMV_Index_Variance_Difference_Barplot_percdiff%i.png" % (figpath,percent_diff),dpi=200,bbox_inches='tight')


#%% Rexamine Spectra, but for top/bottom 5

fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(8,6))



for e in tqdm(range(ens)):
    
    ax = axs[0]
    ax.plot(freqs[e]*dtplot,specs[e]/dtplot,
            marker=".",color='gray',alpha=0.5,label="")
    
    ax = axs[1]
    ax.plot(freqs[e]*dtplot,specs_dt[e]/dtplot,
            marker=".",color='cornflowerblue',alpha=0.9,label="")
    
    if e == 0:
        #freqsmean    = freqs[e]
        specsmean    = specs[e]
        specsmean_dt = specs_dt[e]
    else:
        #freqsmean    = freqsmean + freqs[e]
        specsmean    = specsmean + specs[e]
        specsmean_dt = specsmean_dt + specs_dt[e]


ax = axs[0]
ax.plot(freqs[e]*dtplot,specs[e]/dtplot,
        marker=".",color='gray',alpha=0.5,label="Indv. Member (Undetrended)")
ax.plot(freqs[e]*dtplot,specsmean/ens/dtplot,color='k',alpha=1.0,marker=".",
        label="Ens. Mean (Undetrended)")

ax = axs[1]
ax.plot(freqs[e]*dtplot,specs_dt[e]/dtplot,
        marker=".",color='cornflowerblue',alpha=0.9,label="Indv. Member (Detrended)")
ax.plot(freqs[e]*dtplot,specsmean_dt/ens/dtplot,color='b',alpha=1.0,marker=".",
        label="Ens. Mean (Detrended)")

#Plot Top/Bottom 5
ax = axs[0]
ax.plot(freqs[e]*dtplot,np.array(specs)[topid,:].mean(0)/dtplot,
        marker=".",color='red',label="Top5",ls='dashed')
ax.plot(freqs[e]*dtplot,np.array(specs)[botid,:].mean(0)/dtplot,
        marker=".",color='red',label="Bot5",ls='dotted')
ax = axs[1]
ax.plot(freqs[e]*dtplot,np.array(specs_dt)[topid,:].mean(0)/dtplot,
        marker=".",color='red',label="Top5",ls='dashed')
ax.plot(freqs[e]*dtplot,np.array(specs_dt)[botid,:].mean(0)/dtplot,
        marker=".",color='red',label="Bot5",ls='dotted')



for a,ax in enumerate(axs):
    ax.set_xticks(xtks)
    ax.set_xlim([xtks[0],xtks[-1]])
    ax.set_xticklabels(xper.astype(int))
    ax.grid(True,ls='dotted')
    ax.legend()
    
    
    ax.set_ylabel("Power ($\degree C^{2} cpy^{-1}$)")
    
    if a == 0:
        ax.set_ylim([0,4.0])
    else:
        ax.set_ylim([0,1.25])
        ax.set_xlabel("Period (Years)")
    
plt.suptitle("AMV Index Power Spectra (CESM1-LE, 1920-2005)")
plt.savefig("%sAMV_Index_Power_Spectra_Detrend_Undetrend_withTop.png" % (figpath),dpi=200)

#%% Examine differences in the forced and unforced patterns

amvpats_all = np.zeros([2,nvar,ens,nlat,nlon])

for v in range(3):
    for e in tqdm(range(ens)):
        
        # Detrended
        idx_sel = amvids_tr[e,:] / np.std(amvids_tr[e,:])
        amvpats = proc.regress2ts(data[v,e,:,:,:].transpose(2,1,0),idx_sel)
        amvpats_all[0,v,e,:,:] = amvpats.T.copy()
        
        # Undetrended
        idx_sel = amvids_ntr[e,:] / np.std(amvids_ntr[e,:])
        amvpats_dt = proc.regress2ts(data[v,e,:,:,:].transpose(2,1,0),idx_sel)
        amvpats_all[1,v,e,:,:] = amvpats_dt.T.copy()
        
        
#%% Plot AMV Patterns
alabels   = ("Ens. Mean","Top 5","Bottom 5")
dlabels   = ("Detrended","With Trend")
clvls_var = (np.arange(-1.2,1.3,0.1),
             np.arange(-.75,.80,.05),
             np.arange(-1.2,1.3,0.1)
             
             )
v         = 2 
clvls   = clvls_var[v]
fig,axs = plt.subplots(2,3,subplot_kw={'projection':proj},
                       constrained_layout=True,figsize=(12,6))
for d in range(2):
    for a in range(3):
        ax = axs[d,a]
        if a==0: # Ensemble Mean Pattern
            plotvar = amvpats_all[d,v,:,:,:].mean(0)
        elif a==1: # Top 5 Pattern
            plotvar = amvpats_all[d,v,topid,:,:].mean(0)
        elif a==2:
            plotvar = amvpats_all[d,v,botid,:,:].mean(0)
            
        if d == 0:
            ax.set_title(alabels[a])
        if a == 0:
            ax.text(-0.05, 0.55, dlabels[d], va='bottom', ha='center',rotation='vertical',
                rotation_mode='anchor',transform=ax.transAxes)
        
        pcm = ax.contourf(lon,lat,plotvar,cmap="RdBu_r",levels=clvls)
        ax.coastlines()
    
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',pad=0.05,fraction=0.045)
cb.set_label("Normalized AMV Pattern for %s" % (vnames[v]))
plt.savefig("%sAMV_Pattern_Composites_%s.png" % (figpath,vnames[v]),dpi=200,bbox_inches='tight')