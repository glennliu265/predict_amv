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
import proc,viz



# Time_Period
startyr         = 1920                  # Starting year
endyr           = 2005                  # Ending year
ntime           = (endyr-startyr+1)*12  # Number of months
coordinate      = "depth"             # "depth" or "density"

# Set path to the data 
outpath         = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/01_Data/AMOC/"
figpath         = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20230505/"
if coordinate == "depth":
    datpath     = "/Users/gliu/Globus_File_Transfer/CESM1_LE/Historical/MOC/"
    moc_name = "MOC"
elif coordinate == "density":
    datpath     = "/Users/gliu/Globus_File_Transfer/CESM1_LE/AMOCrho/"
    moc_name = "AMOCsigsum"
else:
    print("coordinate must be [depth] or [density]")

# Select MOC component and region
icomp           = 0 # 0=Eulerian Mean; 1=Eddy-Induced (Bolus); 2=Submeso
iregion         = 1 # 0=Global Mean - Marginal Seas; 1= Altantic Ocean + Mediterranean Sea + Labrador Sea + GIN Sea + Arctic Ocean + Hudson Bay

debug           = True
save_output     = False 

savename        = "%sCESM1_LENS_AMO_%sto%s_comp%i_region%i_%s.npz" % (outpath,startyr,endyr,icomp,iregion,coordinate)
print("%s-coordinate AMOC calculation output will will saved to:\n\t%s" % (coordinate,savename))
#%% Make the function


def plot_moc(moc,lat,z,idz,idlat,coordinate):
    fig,ax      = plt.subplots(1,1)
    plotmoc     = moc.mean(0)
    
    cf          = ax.contourf(lat,z,plotmoc)
    cl          = ax.contour(lat,z,plotmoc,colors="k",linewidths=0.7)
    ax.clabel(cl,)
    
    ax.plot(lat[idlat],z[idz],marker="x",color='k',ls="",markersize=10,label="$\psi_{max}$")       
    ax.legend()
    ax.invert_yaxis()
    plt.colorbar(cf)
    if coordinate == "depth":
        ax.set_title("Mean AMOC Streamfunction\n$z_{max}$: %.2fm, Latitude$_{max}$: %.2f$\degree$" % (z[idz],lat[idlat]))
        ax.set_ylabel("Depth (m)")
    else:
        ax.set_title("Mean AMOC Streamfunction\n"+r"$\rho_{max}$: %.2f kg $m^{-3}$, Latitude$_{max}$: %.2f$\degree$" % (z[idz],lat[idlat]))
        ax.set_ylabel("Depth (kg/m$^3$)")
    
    ax.set_xlabel("Latitude")
    
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
    if coordinate == "depth":
        ds = ds.isel(transport_reg=iregion,moc_comp=icomp)
    
    # Load data
    moc = ds[moc_name].values # [time x z x lat]
    lat = ds.lat_aux_grid.values
    if coordinate == "depth":
        z   = ds.moc_z.values/100
    else:
        z   = ds.moc_s.values
    
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
        fig,ax = plot_moc(moc,lat,z,idz,idlat,coordinate)
        plt.savefig("%sAMOC_Maximum_%s_ens%02i.png" % (figpath,coordinate,e),dpi=150)
        

#%% Save File
if save_output:
    np.savez(savename,**{
        'max_moc'   : max_moc,
        'max_zs'    : max_zs,
        'max_lats'  : max_lats,
        'moc_means' : moc_means,
        'lat'       : lat,
        'z'         : z,
        },allow_pickle=True)
#%% Load File

ld = np.load(savename,allow_pickle=True)
max_moc     = ld['max_moc']
max_zs      = ld['max_zs']
max_lats    = ld['max_lats']
moc_means   = ld['moc_means']
lat         = ld['lat']
z           = ld['z']

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
if coordinate == "depth":
    ax.set_title("AMOC Strength (%i-month LP-Filter) \n$z_{max}$: %.2fm, Latitude$_{max}$: %.2f$\degree$" % (cutoff_mon,z[idz],lat[idlat]))
else:
    ax.set_title("AMOC Strength (%i-month LP-Filter) \n" % (cutoff_mon) + r"$\rho_{max}$: %.2f kg m$^{-3}$, Latitude$_{max}$: %.2f$\degree$" % (z[idz],lat[idlat]))

ax.set_ylabel("AMOC Strength at Maximum Streamfunction (Sv)")
ax.grid(True,ls='dotted')

ax.set_xticks(xtks)
ax.set_xticklabels(xtk_labels)
ax.set_xlim([0,1032])

plt.savefig("%sAMOC_Strength_LPFilter%03i.png" % (figpath,cutoff_mon),dpi =150)


#%% Plot the mean AMOC streamfunction

fig,ax = plot_moc(moc_means,lat,z,idz,idlat,coordinate)
plt.savefig("%sAMOC_Maximum_%s_ensMEAN.png" % (figpath,coordinate),dpi=150)

#%% Compute the autocorrelation timescales, both detrended and undetrended

dropyear = 10

lags  = np.arange(0,25,1)
nlags = len(lags)

amocid_ann = proc.ann_avg(max_moc,dim=1)

amocid_ann = amocid_ann[:,dropyear:]
ac_amoc    = np.zeros((nlags,nens,2)) * np.nan # [lag,ens,[not detrended, detrended]]

for ii in range(2):
    if ii == 0:
        input_indices = amocid_ann
    elif ii == 1:
        input_indices = amocid_ann - amocid_ann.mean(0)[None,:]
    acs,win_len = proc.calc_lag_covar_ann(input_indices,input_indices,lags,1,0)
    ac_amoc[:,:,ii] = acs.copy()
        



#%% Compute rho critical value and set up plotting labels
dtcol = ["r","k"]
dtlab = ["Undetrended","Detrended"]



p     = 0.05
tails = 1
dof   = np.min(win_len) 
rhocrit = proc.ttest_rho(p,tails,dof)




#%% Examine the Distribution

first_year_under = np.zeros([nens,2])
exact_year_under = np.zeros([nens,2])


for ii in range(2):
    
    var_in      = ac_amoc[:,:,ii]
    first_under = np.argmax(var_in < rhocrit, axis=0)
    first_year_under[:,ii] = first_under
    
    for e in range(nens):
        kyear   = first_under[e]
        rho_ens = var_in[:,e]
        
        rho_interval  =[rho_ens[kyear],rho_ens[kyear-1]] #[rho_ens[kyear-1],rho_ens[kyear]]
        year_interval =[kyear,kyear-1] #[kyear-1,kyear]
        
        ycrit = np.interp(rhocrit,rho_interval,year_interval)
        
        exact_year_under[e,ii] = ycrit
        
        
#%% Plot Distribution of Predictability Limit

bin_edges = np.arange(0,24,1)
fig,ax    = plt.subplots(1,1,constrained_layout=True,figsize=(6,4))

ax = viz.add_ticks(ax,grid_lw=0.5)

for ii in range(2):
    ax.hist(exact_year_under[:,ii],bins=bin_edges,label=dtlab[ii],color=dtcol[ii],edgecolor="k",alpha=0.7)
ax.legend()
ax.set_title("Histogram of Predictability Limits for AMOC Autocorrelation\n" + "%i-tailed T-Test, dof=%i, p=%.02d, corr=%.2f" % (tails,dof,p,rhocrit))
ax.set_ylabel("Count (Ens. Members)")
ax.set_xlabel("Predictability Limit (First Year Below Threshold corr=%.2f)" % (rhocrit))
ax.set_xlim(0,12)
savename = "%sAMOC_Strength_Autocorrelation_rhocrit%03i_Histograms.png" % (figpath,rhocrit*100)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Plot the autocorrelation function

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,6))
ax = viz.add_ticks(ax,grid_lw=0.5)

for ii in range(2):
    for e in range(nens):
        if e == np.argmax(exact_year_under[:,ii]):
            label="Ens %02i" % (e+1)
            alpha=0.5
        else:
            label=""
            alpha=0.05
        
        ax.plot(lags,ac_amoc[:,e,ii],color=dtcol[ii],label=label,alpha=alpha)
    mu  = ac_amoc[:,:,ii].mean(1)
    std = ac_amoc[:,:,ii].std(1)
    ax.plot(lags,mu,color=dtcol[ii],label=dtlab[ii],marker="d")
    ax.fill_between(lags,mu-std,mu+std,color=dtcol[ii],alpha=0.20,zorder=-9)

ax.axhline([0],ls="dashed",color="k",lw=0.75)
ax.axhline([rhocrit],ls="dotted",color="k",lw=0.75)

ax.axvline([exact_year_under[:,1].mean()],ls="solid",color="k",lw=0.75,label="year=%.2f"% (exact_year_under[:,1].mean()))
ax.axvline([exact_year_under[:,0].mean()],ls="solid",color="r",lw=0.75,label="year=%.2f"% (exact_year_under[:,0].mean()))

ax.set_title("AMOC Index Autocorrelation\n" + "%i-tailed T-Test, dof=%i, p=%.02d, corr=%.2f" % (tails,dof,p,rhocrit))
ax.set_ylabel("Correlation")
ax.set_xlabel("Lag (Years)")
ax.set_xlim(0,24)

ax.legend()

savename = "%sAMOC_Strength_Autocorrelation_rhocrit%03i.png" % (figpath,rhocrit*100)
plt.savefig(savename,dpi=150,bbox_inches="tight")

    
#%% Re-Plot AMOC Indices 

ii = 1

iens_max = np.argmax(exact_year_under[:,ii])

if ii == 0:
    plot_moc = max_moc.copy()
else:
    plot_moc = max_moc - max_moc.mean(0)[None,:]
    

fig,ax  = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))

cutoff_mon=24

for e in range(nens):
    
    ax.plot(proc.lp_butter(plot_moc[e,:],cutoff_mon,6),label="",alpha=0.05,color="k")

ax.plot(proc.lp_butter(plot_moc[e,:],cutoff_mon,6),label="Indv. Ens",alpha=0.10,color="k")
ax.plot(proc.lp_butter(plot_moc.mean(0),cutoff_mon,6),label="Mean AMOC",color="blue")
ax.plot(proc.lp_butter(plot_moc[iens_max,:],cutoff_mon,6),label="Ens %i" % (iens_max+1),alpha=0.8,color=dtcol[ii])
#ax.plot(max_moc[1,:],label="Ens1")
ax.legend()

# Get the file- name and load to dataset
ds = xr.open_dataset(nclist[0])
times   = ds.time.values
times   = [str(t) for t in times]
yrs     = [t[:4] for t in times]
xtks        = np.arange(0,len(times)+1,120)
xtk_labels  = np.array(yrs)[xtks]
if coordinate == "depth":
    ax.set_title("AMOC Strength (%i-month LP-Filter, Detrend %i) \n$z_{max}$: %.2fm, Latitude$_{max}$: %.2f$\degree$" % (cutoff_mon,ii,z[idz],lat[idlat]))
else:
    ax.set_title("AMOC Strength (%i-month LP-Filter, Detrend %i) \n" % (cutoff_mon,ii) + r"$\rho_{max}$: %.2f kg m$^{-3}$, Latitude$_{max}$: %.2f$\degree$" % (z[idz],lat[idlat]))

ax.set_ylabel("AMOC Strength at Maximum Streamfunction (Sv)")

ax.set_xticks(xtks)
ax.set_xticklabels(xtk_labels)
ax.set_xlim([0,1032])

ax = viz.add_ticks(ax)

plt.savefig("%sAMOC_Strength_LPFilter%03i_Replot_detrend%i.png" % (figpath,cutoff_mon,ii),dpi =150)


