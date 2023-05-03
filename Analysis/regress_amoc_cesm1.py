#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regress the maximum AMOC Index to a given CESM1 Predictor

Created on Tue Mar  7 02:38:23 2023

@author: gliu
"""

import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import sys
import os

from tqdm import tqdm
import cartopy.crs as ccrs


#%% Add custom modules and packages
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import proc,viz


cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import amvmod as am
import amv_dataloader as dl 

#%%
# Set path to the data 
mocpath      = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/01_Data/AMOC/"
figpath      = pparams.figpath

# Time_Period
startyr      = 1920
endyr        = 2005
ntime        = (endyr-startyr+1)*12
coordinate   = "depth"

# Select MOC component and region
icomp        = 0 # 0=Eulerian Mean; 1=Eddy-Induced (Bolus); 2=Submeso
iregion      = 1 # 0=Global Mean - Marginal Seas; 1= Altantic Ocean + Mediterranean Sea + Labrador Sea + GIN Sea + Arctic Ocean + Hudson Bay
savename_moc = "%sCESM1_LENS_AMO_%sto%s_comp%i_region%i_%s.npz" % (mocpath,startyr,endyr,icomp,iregion,coordinate)
leads        = np.arange(0,26,6)



# Set predictor options
varnames     = ["SSH","SST","SSS","PSL"]
detrend      = 0
bbox         = pparams.bbox

# Other toggles
debug        = True


#%% Load MOC data for the given ensemble member

ld      = np.load(savename_moc,allow_pickle=True)
max_moc = ld['max_moc'] # {Ens x Time}


# Take the Annual averages
max_moc_annavg = proc.ann_avg(max_moc,1,)

#%% Load the predictors

data,lat,lon   = dl.load_data_cesm(varnames,bbox,detrend=detrend,return_latlon=True) # {Channel x ens x twE, X OLQ }


#%% Make the regression maps

nvars,nens,nyrs,nlat,nlon = data.shape
nleads                    = len(leads)
regr_maps                 = np.zeros([nleads,nvars,nens,nlat,nlon])
amoc_lead                 = False
if coordinate == "density":
    nens = 40 # Reduce to 40 members for density space AMOC

for l in range(nleads):
    lead = leads[l]
    for v in range(nvars):
        for e in range(nens):
            
            if amoc_lead:
                in_predictor = data[v,e,lead:,:,:]
                in_moc       = max_moc_annavg[e,:(nyrs-lead)] #- max_moc_annavg.mean(0)
            else:
                in_predictor = data[v,e,:(nyrs-lead),:,:]
                in_moc       = max_moc_annavg[e,lead:] #- max_moc_annavg.mean(0)
            
            
            regr_maps[l,v,e,:,:] = proc.regress2ts(in_predictor.transpose(2,1,0),in_moc,).T
        

#%% Examine the AMOC regression patterns (ensemble mean)

proj  = ccrs.PlateCarree()
bbox_plot = [-80,0,20,63]
l = 0
mesh=True
clvl=np.arange(-1.8,2,0.2)
fig,axs = plt.subplots(1,nvars,figsize=(16,4),
                       subplot_kw={'projection':proj},constrained_layout=True,)



for v in range(nvars):
    ax      = axs.flatten()[v]
    plotvar = regr_maps[l,v,...].mean(0)
    
    blabel=[0,0,0,1]
    if v == 0:
        blabel[0] = 1
    ax      = viz.add_coast_grid(ax,bbox=bbox_plot,proj=ccrs.PlateCarree(),fill_color="k",blabels=blabel)
    if mesh:
        pcm = ax.pcolormesh(lon,lat,plotvar,cmap="RdBu_r",vmin=clvl[0],vmax=clvl[-1])
    else:
        pcm = ax.contourf(lon,lat,plotvar,cmap="RdBu_r",levels=clvl)
    
    cb=fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
    ax.set_title(varnames[v])
    cb.set_label("AMOC Regression \n ([Fluctuation per Sv of iAMOC])")

plt.savefig("%sAMOC_Regression_2var_amoclead%i_%s_Lead%02i.png" % (figpath,amoc_lead,coordinate,leads[l]),dpi=200,bbox_inches="tight")
#%% Copied from viz_regional_predictability
cmax    = 1
fig,axs = plt.subplots(2,5,figsize=(14,6.5),
                       subplot_kw={'projection':proj},constrained_layout=True)
for v in range(2):
    for i in range(len(leads)):
        
        lead = leads[i]
        print(lead)
        l    = list(leads).index(lead)
        print(l)
        
        ### Leads are all wrong need to fix it
        ax = axs[v,i]
        
        plotvar = regr_maps[l,v,...].mean(0)
        pcm     = ax.pcolormesh(lon,lat,plotvar,vmin=-cmax,vmax=cmax,cmap="RdBu_r")
        
        # Do Plotting Business and labeling
        if v == 0:
            if amoc_lead:
                ax.set_title("AMOC Lead %i yrs" % (lead))
            else:
                ax.set_title("Predictor Lead %i yrs" % (lead))
        if i == 0:
            ax.text(-0.05, 0.55, varnames[v], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
        ax.set_extent(bbox)
        ax.coastlines()
    
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05)
cb.set_label("AMOC Regression Coefficient ([Fluctuation per Sv of iAMOC])")

plt.suptitle("Ensemble Average Predictor Maps regressed to AMOC Index (iAMOC in %s-space), %i to %i" % (coordinate,startyr, endyr))
figname  = "%siAMOC_Predictor_LeadRegression_%ito%i_amooclead%i_detrend%i_%s.png" % (figpath,startyr,endyr,amoc_lead,detrend,coordinate)
# savename = "%s.png" % (figpath,varname,classes[c],topN,normalize_sample,absval,ge_label_fn,pcount)
plt.savefig(figname,dpi=150,bbox_inches="tight",transparent=True)
    
    

