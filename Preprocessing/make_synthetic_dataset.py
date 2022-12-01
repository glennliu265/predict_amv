#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a synthetic dataset to test LRP

Created on Wed Nov 30 11:23:39 2022

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

#%%

# Indicate settings (Network Name)
datpath   = "../../CESM_data/"
varname   = "SST"
detrend = 0
regrid  = None
bbox    = [-80,0,0,65]

# Load in input and labels 
ds   = xr.open_dataset(datpath+"CESM1LE_%s_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc" % (varname,detrend,regrid) )
ds   = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
data = ds[varname].values[None,...]
target = np.load(datpath+ "CESM_label_amv_index_detrend%i_regrid%s.npy" % (detrend,regrid))

lat = ds.lat.values
lon = ds.lon.values


nvar,nens,nyr,nlat,nlon = data.shape

#%%

# Make a dummy set
#fakedata = np.zeros((nlat,nlon))
fakedata = np.zeros(data.shape)

fake_bboxes=(
    [-60,-20,40,60],
    [-40,-5,10,30],
    [-75,-60,10,35]
    )

mask = np.abs(target) > np.std(target)

# Set Positive/negative
for bb in range(len(fake_bboxes)-1):
    
    # Get Indices
    bbin = fake_bboxes[bb]
    klats = np.where((lat> bbin[2]) & (lat<=bbin[3]))[0]
    klons = np.where((lon>bbin[0]) & (lon<=bbin[1]))[0]
    
    # Get neutral timesteps
    #mask = np.where(np.abs(target)<np.std(target),) # Get neutral Cases
    
    
    # Set indices
    if bb == 0:
        
        fill_val = target[...,None,None]/np.abs(target[...,None,None])*5
    
    elif bb == 1:
        fill_val = target[...,None,None]/np.abs(target[...,None,None])*-5#target[...,None,None] * -1
        
    elif bb == 2:
        
        fill_val = np.random.normal(0,1,(len(klats),len(klons))) 
    
    
    # if bb < 2:
    #     fakedata[:,:,:,:,:] = fakedata * mask[None,:,:,None,None] # Set neutral times to zero
        
        
    fakedata[:,:,:,klats[:,None],klons] = fill_val
    
    
# Set Neutral Values to zero
fakedata = fakedata.reshape(1,nens*nyr,nlat,nlon)
fakedata[:,mask.flatten(),:,:] = 0
fakedata = fakedata.reshape(1,nens,nyr,nlat,nlon)


# Set a random box


#%% Test Viz
iens = 31
iyr  = 31

fig,ax = plt.subplots(1,1)
pcm=ax.pcolormesh(lon,lat,fakedata[0,iens,iyr,:,:],vmin=-.5,vmax=.5,cmap='RdBu_r')
fig.colorbar(pcm,ax=ax)
ax.set_title("Ens=%02i, Yr=%02i, Target iAMV=%.2f" % (iens+1,iyr+1920,target[iens,iyr]))

#%% Save the data

fn = "%sfakedata_1Neg1Pos1Random_3box_fixval.nc" % (datpath)

dsfake = ds.copy()

dsfake['fakedata'] = (("ensemble","year","lat","lon"),fakedata[0,...])
dsfake = dsfake.drop("SST")
dsfake.to_netcdf()


dsfake.to_netcdf(fn,
         encoding={'fakedata': {'zlib': True}})

#%% Train a network to make the prediction (Take from NN_test_lead.py)






