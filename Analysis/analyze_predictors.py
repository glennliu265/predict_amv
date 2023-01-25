#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Analyze Predictors for AMV Project

Created on Wed Dec  7 12:07:15 2022

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

import cmocean as cmo
#%% # User Edits

# Indicate settings (Network Name)
#expdir    = "FNN4_128_SingleVar"
#modelname = "FNN4_128"

expdir     = "baseline_linreg"
modelname  = "linreg"

# lrp methods
sys.path.append("/Users/gliu/Downloads/02_Research/03_Code/github/Pytorch-LRP-master/")
from innvestigator import InnvestigateModel


# Load modules (LRPutils by Peidong)
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/scrap/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/predict_amv/")

import LRPutils as utils
import amvmod as am

# Load my own custom modules
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz,proc

examine_dt    = True # Set to True to also load detrended versions
darkmode      = True

#%% Import Parameters
import predict_amv_params as pparams

# Region Names
regions         = pparams.regions
bboxes          = pparams.bboxes
classes         = pparams.classes
rcolors         = pparams.rcolors

# Mapping/Plotting
bbox            = pparams.bbox
plotbbox        = pparams.plotbbox
proj            = pparams.proj

# Variable Names
varnames        = pparams.varnames
varnamesplot    = pparams.varnamesplot
varnames_long   = pparams.varnames_long
vunits          = pparams.vunits
varcolors       = pparams.varcolors

# Import Paths
datpath         = pparams.datpath
figpath         = pparams.figpath
proc.makedir(figpath)

# ML Training Parameters
detrend         = pparams.detrend
leads           = pparams.leads
regrid          = pparams.regrid
tstep           = pparams.tstep
ens             = pparams.ens
thresholds      = pparams.thresholds 
quantile        = pparams.quantile
percent_train   = pparams.percent_train

#%% Load the data
st = time.time()

all_data = []
for v,varname in enumerate(varnames):
    # Load in input and labels 
    ds   = xr.open_dataset(datpath+"CESM1LE_%s_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc" % (varname,detrend,regrid) )
    ds   = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3])).isel(ensemble=np.arange(0,ens))
    data = ds[varname].values[None,...]
    all_data.append(data)
all_data = np.array(all_data).squeeze() # [variable x ens x yr x lat x lon]
#[print(d.shape) for d in all_data]

if examine_dt:
    all_data_dt = []
    for v,varname in enumerate(varnames):
        # Load in input and labels 
        ds   = xr.open_dataset(datpath+"CESM1LE_%s_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc" % (varname,1,regrid) )
        ds   = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3])).isel(ensemble=np.arange(0,ens))
        data = ds[varname].values[None,...]
        all_data_dt.append(data)
    all_data_dt = np.array(all_data_dt).squeeze()

# Load the target
target = np.load(datpath+ "CESM_label_amv_index_detrend%i_regrid%s.npy" % (detrend,regrid))
target_dt = np.load(datpath+ "CESM_label_amv_index_detrend%i_regrid%s.npy" % (1,regrid))

# region_targets = []
# region_targets.append(target)
# # Load Targets for other regions
# for region in regions[1:]:
#     index = np.load(datpath+"CESM_label_%s_amv_index_detrend%i_regrid%s.npy" % (region,detrend,regrid))
#     region_targets.append(index)

# Apply Land Mask
# Apply a landmask based on SST, set all NaN points to zero
msk = xr.open_dataset(datpath+'CESM1LE_SST_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc'% (detrend,regrid))
msk = msk.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
msk = msk["SST"].values
msk[~np.isnan(msk)] = 1
msk[np.isnan(msk)] = 0
# Limit to input to ensemble member and apply mask
all_data = all_data[:,:,...] * msk[None,0:ens,...]
all_data[np.isnan(all_data)] = 0

nchannels,nens,ntime,nlat,nlon = data.shape # Ignore year and ens for now...
inputsize                      = nchannels*nlat*nlon # Compute inputsize to remake FNN

nvars = all_data.shape[0]

lon = ds.lon.values
lat = ds.lat.values
#%% Get the Regression Maps of Trends for each variable

rmaps = np.zeros((nvars,nens,nlat,nlon))
target_ensavg   = target.mean(0)
for v in range(nvars):
    for e in range(ens):
        invar = all_data[v,e,:,:,:].transpose(2,1,0) # lon x lat x time
        beta = proc.regress2ts(invar,target_ensavg,verbose=False)
        rmaps[v,e,:,:] = beta.T

if examine_dt:
    rmaps_dt = np.zeros((nvars,nens,nlat,nlon))
    target_ensavg   = target_dt.mean(0)
    for v in range(nvars):
        for e in range(ens):
            invar = all_data_dt[v,e,:,:,:].transpose(2,1,0) # lon x lat x time
            beta = proc.regress2ts(invar,target_ensavg,verbose=False)
            rmaps_dt[v,e,:,:] = beta.T

#%% Plot Ensemble average regression maps

fig,axs = plt.subplots(2,3,subplot_kw={'projection':proj},
                       constrained_layout=True,figsize=(6,5))

cmax = 2

for v in range(nvars):
    ax = axs.flatten()[v]
    
    plotvar = rmaps[v,:,:,:].mean(0)
    cmax    = np.std(plotvar)*3
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-cmax,vmax=cmax,
                        cmap="RdBu_r")
    
    ax.set_title(varnames[v])
    ax.set_extent(plotbbox)
    ax.coastlines()
    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.04)
plt.suptitle("Ens. Avg. Regression Maps for each Predictor (to AMV Index Trend)")
savename = "%sEnsAvg_PredictorRegressionMaps_toAMVTrend.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')
#plt.savefig("")

#%% Check how it looks for individual members

cmax = 2
for v in range(nvars):
    
    fig,axs = plt.subplots(5,8,subplot_kw={'projection':proj},
                           constrained_layout=True,figsize=(20,14))
    
    plotvar = rmaps[v,:,:,:].mean(0)
    cmax    = np.std(plotvar)*3
    
    for e in range(ens):
        ax = axs.flatten()[e] 
        plotvar = rmaps[v,e,:,:]
    
        pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-cmax,vmax=cmax,
                            cmap="RdBu_r")
        
        viz.label_sp("ens%02i" % (e+1),ax=ax,alpha=0.8,labelstyle="%s",usenumber=True,
                     fontsize=14)

        ax.set_extent(plotbbox)
        ax.coastlines()
    fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.025,pad=0.01)
    plt.suptitle("Regression Maps for each Predictor (to AMV Index Trend)",fontsize=16,y=0.99)
    
    savename = "%sAllEns_%s_RegressionMaps_toAMVTrend.png" % (figpath,varnames[v])
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% See if there is a correspondence to just variability of each variable

fig,axs = plt.subplots(2,3,subplot_kw={'projection':proj},
                       constrained_layout=True,figsize=(6,5))

cmax = 2

for v in range(nvars):
    ax = axs.flatten()[v]
    
    plotvar = np.std(all_data[v,:,:,:,:],(1)).mean(0)
    cmax    = np.std(plotvar)*3
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=0,vmax=cmax,
                        cmap="inferno")
    
    ax.set_title(varnames[v])
    ax.set_extent(plotbbox)
    ax.coastlines()
    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.04)
plt.suptitle("Ens. Avg. $\sigma$ Maps for each Predictor")
savename = "%sEnsAvg_PredictorStdevMaps.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')
#plt.savefig("")

#%% Examine the persistence of each predictor (basinwide)

latweight= np.cos(np.pi*lat/180)

# Take areaavg
all_data_weighted = all_data * latweight[None,None,None,:,None]
all_data_aa       = np.nanmean(all_data*msk[None,:ens,...],(3,4))
lagcorrs = np.zeros((nvars,nens,len(leads)))
for v in range(nvars):
    for e in range(ens):
        for i,l in enumerate(leads):
            ts = all_data_aa[v,e,:]
            rr = np.corrcoef(ts[l:],ts[:ts.shape[0]-l])
            print(rr)
            lagcorrs[v,e,i] = rr[0,1]

#%% Plot persistence of each variable

fig,ax= plt.subplots(1,1)
for v in range(nvars):
    eavg = 0
    for e in range(ens):
        
        plotvar = lagcorrs[v,e,:]
        if e == 0:
            eavg = plotvar.copy()
        else:
            eavg = eavg + plotvar
            
        ax.plot(leads,plotvar,color=varcolors[v],alpha=0.05,label="")
    ax.plot(leads,eavg/ens,marker="o",color=varcolors[v],label=varnames[v])
ax.set_xticks(leads)
ax.set_xlim([leads[0],leads[-1]])
ax.legend()
ax.grid(True,ls="dotted")
ax.set_title("NAT-Averaged Persistence by Predictor")
savename = "%sPersistence_NAT_ByPredictor.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Quickly remake mask

vizmsk = msk[0,0,...]
vizmsk[vizmsk==0] = np.nan


#%%
v    = 0
varcmaps      = ("cmo.thermal","cmo.haline","cmo.curl","cmo.ice","cmo.tempo","cmo.deep",)
varcints      = (2.5,)*nvars
add_contours  = False

iens = 0
iyr  = 0

levels = np.arange(-2,2.2,.2)

for v in range(nvars):
    if darkmode:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    
    
    fig,ax = plt.subplots(1,1,figsize=(8,4),subplot_kw={'projection':proj},)
    plotvar = all_data[v,iens,iyr,:,:] * vizmsk
    pcm     = ax.pcolormesh(lon,lat,plotvar,cmap=varcmaps[v],
                            vmin=-varcints[v],vmax=varcints[v])
    if add_contours:
        ax.contour(lon,lat,plotvar,levels=levels,colors="k",linewidths=0.5)
    ax.set_title("%s (%s, %s)"% (varnamesplot[v],varnames_long[v],vunits[v]))
    fig.colorbar(pcm,ax=ax,fraction=0.045,pad=0.05,orientation='horizontal')
    savename = "%sPredictor_example_%s_ens%02i_yr%02i_contour%i.png" % (figpath,varnames[v],iens,iyr,add_contours)
    #ax.coastlines(color="w")
    plt.savefig(savename,dpi=150,transparent=True,bbox_inches="tight")
    print(savename)

#%% Compare histograms of detrended and undetrended indices

fig,axs = plt.subplots(1,2,constrained_layout=True,sharey=True)

binedges = np.arange(-1.2,1.4,0.2)
dtlab    = ("Raw","Detrended")
dtcol    = ("salmon","cyan")

plotvars = [a.flatten() for a in [target, target_dt]]
for i in range(2):
    
    ax = axs[i]
    ax.hist(plotvars[i],bins=binedges,edgecolor="k",color=dtcol[i],alpha=0.9)
    ax.set_title("%s (1$\sigma$=%.3f$\degree C$)" % (dtlab[i],plotvars[i].std()))
    
    ax.grid(True,ls='dotted')
    ax.set_xlim([-1.2,1.2])

plt.suptitle("AMV Index Histograms, CESM1-LE (1920-2005)")
savename = "%sAMV_Index_Histograms_DetrendComparison.png" % (figpath,varnames[v],iens,iyr,add_contours)
plt.savefig(savename,dpi=150,transparent=True,bbox_inches="tight")
#%% Compare regression maps of detrended and undetrended variables...


#%% Examine Lag Covariance of each variable with the AMV Index


v       = 4
nleads  = len(leads)

# Subset variables
selvar  = all_data[v,:,:,:,:]
seltarg = target[:ens,:] 


# Loop by ensemble member
cov_bylag = np.zeros((nleads,nens,nlat,nlon)) * np.nan

for e in range(nens):
    A = selvar[e,:,:,:] # [time x lat x lon ]
    A = A.reshape(ntime,nlat*nlon) # [time x space]
    B = target[[e],:].T # [time x 1]
    
    for il in range(nleads):
        
        # Index and calculate covariance
        print("Lead %i" %leads[il])
        print("Indexing Predictor from %i to %i" % (0,tstep-leads[il]))
        print("Indexing Target from %i to %i" % (leads[il],tstep))
        print("\n")
        
        # Apply lead
        Ain = A[:(tstep-leads[il]),:]
        Bin = B[leads[il]:,:]
        
        # Compute Anomaly'
        Aanom = Ain - Ain.mean(0)
        Banom = Bin - Bin.mean(0)
        N     = Aanom.shape[0] - 1
        
        # Broadcast B and take the Expectation
        cov =  (np.sum(Aanom * Banom,0))/N # Sum Along Time
        cov = cov.reshape(nlat,nlon)
        cov_bylag[il,e,:,:] = cov.copy()
        
        
        
        
        
#%% Sanity Check (plot timeseries, covariance, and covariance-by-lag)

klon = 45
klat = 22
il   = 7
e    = 22

fig,axs = plt.subplots(2,1,figsize=(8,4),constrained_layout=True)

idx   = target[e,leads[il]:]
varts = selvar[e,:(tstep-leads[il]),klat,klon]

cov   = np.cov(idx,varts)[0,1]
rho   = np.corrcoef(idx,varts)[0,1]

ax = axs[0]
ax.plot(idx,label="NASST")
ax.plot(varts,label="%s Value" % (varnames[v]))

title = "Lon %i, Lat %i. estcov=%.5f, np.cov=%.5f, Corr=%.2f" % (lon[klon],lat[klat],
                                                                 cov_bylag[il,e,klat,klon,],cov,rho)
ax.set_title(title)

ax.legend()
ax.grid(True,ls="dotted")

ax = axs[1]
ax.plot(leads,cov_bylag[:,e,klat,klon],marker="o")
ax.grid(True,ls="dotted")
ax.set_title("Covariance By Lag")
ax.axhline(0,ls="dashed",color="k")


#%% Now plot ensemble average covariance

vmn  = 0.2
step = 0.05

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
    
    plotdata = cov_bylag[kl,:,:,:].mean(0)
    cf       = ax.pcolormesh(lon,lat,plotdata,cmap="RdBu_r",vmax=vmn,vmin=-vmn)
    cl       = ax.contour(lon,lat,plotdata,colors="k",linewidths=0.5,levels=np.arange(-vmn,vmn+step,step))
    ax.clabel(cl)
    
    fig.colorbar(cf,ax=ax,orientation='horizontal',fraction=0.026)
    if il == 0:
        ylab = varnames[v]
        ax.text(-0.05, 0.55, "%s-NASST Covariance" % (ylab), va='bottom', ha='center',rotation='vertical',
                rotation_mode='anchor',transform=ax.transAxes)
        
savename = "%sLag_Covariance_AMVIndex-%s_ensavg.png" % (figpath,varnames[v])
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Plot Covariance for individual ensemble members

vmn  = 0.5
step = 0.1

for e in range(ens):
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
            
            plotdata = cov_bylag[kl,e,:,:]
            cf       = ax.pcolormesh(lon,lat,plotdata,cmap="RdBu_r",vmax=vmn,vmin=-vmn)
            cl       = ax.contour(lon,lat,plotdata,colors="k",linewidths=0.5,levels=np.arange(-vmn,vmn+step,step))
            ax.clabel(cl)
            
            fig.colorbar(cf,ax=ax,orientation='horizontal',fraction=0.026)
            if il == 0:
                ylab = varnames[v]
                ax.text(-0.05, 0.55, "%s-NASST Covariance" % (ylab), va='bottom', ha='center',rotation='vertical',
                        rotation_mode='anchor',transform=ax.transAxes)
                
        savename = "%sLag_Covariance_AMVIndex-%s_ens%02i.png" % (figpath,varnames[v],e+1)
        plt.savefig(savename,dpi=150,bbox_inches="tight")

