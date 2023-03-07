#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate AMOC Maximum

- Script based on exploratory version drafted in viz_moc

Created on Mon Mar  6 15:39:22 2023

@author: gliu
"""

import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import sys
import os

from tqdm import tqdm

#%% Add custom modules and packages
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import proc

# Set path to the data 
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LE/"
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/01_Data/AMOC/"
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20230308/"

# Time_Period
startyr = 1920
endyr   = 2005
ntime   = (endyr-startyr+1)*12

# Select MOC component and region
icomp   = 0 # 0=Eulerian Mean; 1=Eddy-Induced (Bolus); 2=Submeso
iregion = 1 # 0=Global Mean - Marginal Seas; 1= Altantic Ocean + Mediterranean Sea + Labrador Sea + GIN Sea + Arctic Ocean + Hudson Bay

debug=True
#%% Make the function


def plot_moc(moc,lat,z,idz,idlat):
    fig,ax      = plt.subplots(1,1)
    plotmoc     = moc.mean(0)
    
    cf          = ax.contourf(lat,z,plotmoc)
    cl          = ax.contour(lat,z,plotmoc,colors="k",linewidths=0.7)
    ax.clabel(cl,)
    
    ax.plot(lat[idlat],z[idz],marker="x",color='k',ls="",markersize=10,label="$\psi_{max}$")       
    ax.legend()
    ax.invert_yaxis()
    plt.colorbar(cf)
    ax.set_title("Mean AMOC Streamfunction\n$z_{max}$: %.2fm, Latitude$_{max}$: %.2f$\degree$" % (z[idz],lat[idlat]))
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Depth (m)")
    return fig,ax

#%% Get File Names
nclist = glob.glob(datpath+"*.nc")
nclist.sort()
nens   = len(nclist)
print("Found %i Files!" % (nens))


#%% Compute moc at the maximum streamfunction and save it
# Preallocate
max_lats = np.zeros((nens))
max_zs   = max_lats.copy()
max_moc  = np.zeros((nens,ntime)) # [ensemble x time]
for e in tqdm(range(nens)):
    
    # Get the file- name and load to dataset
    fn = nclist[e]
    ds = xr.open_dataset(fn)
    
    # Restrict to time period
    ds = ds.sel(time=slice(str(startyr)+'-02-01',str(endyr+1)+'-01-01'))
    ds = ds.isel(transport_reg=iregion,moc_comp=icomp)
    
    # Load data
    moc = ds.MOC.values # [time x z x lat]
    lat = ds.lat_aux_grid.values
    z   = ds.moc_z.values/100
    
    if e == 0:
        moc_means    = np.zeros((nens,)+moc.shape[1:]) * np.nan
    
    # Get the indices
    idz,idlat        = proc.maxid_2d(moc.mean(0))
    # Record desired data
    max_moc[e,:]     = moc[:,idz,idlat]
    max_zs[e]        = z[idz]
    max_lats[e]      = lat[idlat] 
    moc_means[e,:,:] = moc.mean(0)
    
    if debug:
        fig,ax = plot_moc(moc,lat,z,idz,idlat)
        plt.savefig("%sAMOC_Maximum_ens%02i.png" % (figpath,e),dpi=150)
        
#%%
savename = "%sCESM1_LENS_AMO_%sto%s_comp%i_region%i.npz" % (outpath,startyr,endyr,icomp,iregion)
np.savez(savename,**{
    'max_moc'   : max_moc,
    'max_zs'    : max_zs,
    'max_lats'  : max_lats,
    'moc_means' : moc_means,
    'lat'       : lat,
    'z'         : z,
    },allow_pickle=True)
#%% Compute the AMV Index, which I will then use to regress to something

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))

cutoff_mon=24

for e in range(nens):
    
    ax.plot(proc.lp_butter(max_moc[e,:],cutoff_mon,6),label="",alpha=0.10,color="k")

ax.plot(proc.lp_butter(max_moc[e,:],cutoff_mon,6),label="Indv. Ens",alpha=0.10,color="k")
ax.plot(proc.lp_butter(max_moc.mean(0),cutoff_mon,6),label="Mean AMOC",color="r")
#ax.plot(max_moc[1,:],label="Ens1")

ax.legend()

times   = ds.time.values
times   = [str(t) for t in times]
yrs     = [t[:4] for t in times]
xtks        = np.arange(0,len(times)+1,120)
xtk_labels  = np.array(yrs)[xtks]
ax.set_title("AMOC Strength (%i-month LP-Filter) \n$z_{max}$: %.2fm, Latitude$_{max}$: %.2f$\degree$" % (cutoff_mon,z[idz],lat[idlat]))
ax.set_xlabel("Time (months)")
ax.set_ylabel("AMOC Strength at Maximum Streamfunction (Sv)")
ax.grid(True,ls='dotted')

ax.set_xticks(xtks)
ax.set_xticklabels(xtk_labels)
ax.set_xlim([0,1032])
plt.savefig("%sAMOC_Strength_LPFilter%03i.png" % (figpath,cutoff_mon),dpi =150)





