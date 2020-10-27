#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate NAO Index from CESM1.1 Large Ensemble

General Procedure:
    1 - Load data and calculate monthly anomalies
    2 - Detrend data at each point
    3 - Apply Area Weights
    4 - Calculate NAO Index
        i   - Perform EOF
        ii  - Standardize PC and regress back to SLP
        iii - Flip signs for consistency (low over Iceland)
    5- Save Data

@author: gliu
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys

## User Specific Edits <START> ----

# Path settings ...
# Path to directory containing data, downloaded from link below
# https://drive.google.com/drive/u/0/folders/1o0R4RSj34HNInR9ehZ9Yw2pCiZGRPo-s
datpath = "/Users/gliu/Downloads/2020_Fall/6.862/Project/Data/"

# Path to module
modpath = "/Users/gliu/Downloads/2020_Fall/6.862/Project/predict_amv/"

# Output Path
outpath = "/Users/gliu/Downloads/2020_Fall/6.862/Project/Data/proc/"

# NAO Calculation settings ... 
# Indicate start and ending year
start = "1920-01-01"
end   = "2005-12-01"

# Indicate the lat/lon bounds (lon is degrees West, -180-180)
lonW = -90
lonE = 40
latS = 20
latN = 80

# Detrending Options
# For deg = 0, Detrend by removing the ensemble average 
# For deg > 0, Specify degree of polynomial for detrending 
deg = 0
        
# Debug mode, makes some plots to check detrending, etc
debug = 1


# EOF options
N_mode = 1 # Number of EOFs to calculate (just 1 for NAO)
lonLisbon = -9.1398
latLisbon = 38.7223
lonReykjavik = -21.9426
latReykjavik = 64.146621
tol = 5 # in degrees

## User Edits <END> ----
#%% 1) Load Data and calculate monthly anomalies ----

# Import module with functions
sys.path.append(modpath)
import amvmod as amv

pslname ="CESM1LE_psl_NAtl_19200101_20051201.nc"
ds = xr.open_dataset(datpath+pslname)
ds = ds.sel(time=slice(start,end))

# Selection region for calculation
ds = ds.where((ds.lon>lonW)|(ds.lon<lonE),drop=True).sel(lat=slice(latS,latN))
if debug == 1:
    ds.psl.isel(ensemble=0,time=0).plot()

# Load data from DataArray to numpy array
psl = ds['psl'].values/100 #[96 x 105 x 1032 x 42] # Convert Pa --> hPa
lon = ds['lon'].values #[105]
lat = ds['lat'].values #[96]  
mon = ds['time'].values#[1032]
nlat,nlon,nmon,nens = psl.shape

# Calculate monthly anomalies
psl  = psl.transpose(2,3,0,1)  # [ntime,nens,nlat,nlon]
pslm = psl.reshape(int(np.ceil(nmon/12)),12,nens*nlat*nlon) # Separate mon/year, combine lat/lon/ens
psla = pslm - pslm.mean(0)[None,:,:] # [86 x 12 x 282240]
psla = psla.reshape(np.prod(psla.shape[:2]),nens*nlat*nlon) #Recombine mon/year with ensemble [1032, 282240]


# %% 2) Perform detrending at each point ----

if deg > 0: # Detrend by removing fitted polynomial
    x = np.arange(0,nmon)
    psldt,model = amv.detrend_poly(x,psla,deg)
    
    # Test visualize detrending
    if debug == 1:
        pt = 4026
        fig,ax=plt.subplots(1,1)
        plt.style.use('seaborn')
        ax.scatter(x,psla[:,pt],label='raw',color='r')
        ax.plot(x,model[pt,:],label='fit',color='k')
        ax.scatter(x,psldt[:,pt],label='detrended',color='b')
        ax.legend()
        ax.set_title("PSL Detrended , %i Deg. Polynomial"%deg)
    
    
    # Separate out ensemble dimensions
    psldt = psldt.reshape(nmon,nens,nlat*nlon)  #[1032 x 42 x 10080]
    
    if debug == 1: # Test plot map of detrended SLP
        test = psldt.reshape(nmon,nens,nlat,nlon)
        fig,ax = plt.subplots(1,1)
        ax.pcolormesh(lon,lat,test[0,0,:,:]),plt.colorbar(),plt.title("PSL Detrended")
else: # Detrend by moving ensemble average

    # Separate out ensemble dimensions
    psla = psla.reshape(nmon,nens,nlat*nlon)  #[1032 x 42 x 10080]
    psldt = psla - psla.mean(1)[:,None,:]
    
    # Test visualize detrending
    if debug == 1:
        ensavg = psla.mean(1)[:,2230]
        fig,ax=plt.subplots(1,1)
        ax.plot(x,psla[:,0,2230],label='raw',color='r')
        ax.plot(x,psldt[:,0,2230],label='detrended',color='b')
        ax.plot(ensavg,label='ensavg',color='k')
        ax.legend()
    

#%% 3 - Apply area weights
_,Y = np.meshgrid(lon,lat)
wgt = np.sqrt(np.cos(np.radians(Y))) # [lat x lon]

pslwgt = psla * wgt.reshape(nlat*nlon)[None,None,:] #[time x ens x lat x lon]
if debug == 1: # Visualize Area Weighting
    e=0
    t=0
    
    fig,axs = plt.subplots(1,3,sharex=True,sharey=True)
    pcm = axs[0].pcolormesh(lon,lat,wgt)
    axs[0].set_title("Area Weights")
    fig.colorbar(pcm,ax=axs[0])
    
    p1 = axs[1].pcolormesh(lon,lat,psla.reshape(nmon,nens,nlat,nlon)[t,e,:,:],
                           vmin=-10,vmax=10)
    fig.colorbar(p1,ax=axs[1])
    axs[1].set_title("Unweighted PSL")
    
    p2 = axs[2].pcolormesh(lon,lat,pslwgt.reshape(nmon,nens,nlat,nlon)[t,e,:,:],
                            vmin=-10,vmax=10)
    fig.colorbar(p2,ax=axs[2])
    axs[2].set_title("Area-weighted PSL")
    
    axs[0].set_ylabel("Lat")
    axs[1].set_xlabel("Lon")

# %% 4 - Calculate NAO Index ----

# Separate month and ensemble
#pslwgt = pslwgt.reshape(int(np.ceil(nmon/12)),12,nens,nlat*nlon) 

naoidx      = np.zeros((nens,nmon,N_mode)) # [ens x time x pc]
naopattern  = np.zeros((nens,nlat,nlon,N_mode)) # [ens x space x pc]
varexp      = np.zeros((nens,N_mode)) # [ens x pc]
for e in range(nens):
    # Select ensemble and transpose to space x time
    pslin = pslwgt[:,e,:].T
    
    # i - Perform EOF 
    _,pcs,varexp[e,:] = amv.eof_simple(pslin,N_mode,0)
    
    # ii - Standardize PC
    pcstd = pcs/np.std(pcs,0)
    
    # Looping by PC (NAO is just pc1, so this is technically unneeded)
    for n in range(N_mode):        
        # iii - Regress and reshape to lat x lon
        eof,_ = amv.regress_2d(pcstd[:,n],pslin,nanwarn=0)
        eof = eof.reshape(nlat,nlon)
        
        # iv -- Check signs (negative anomaly should be over iceland)
        rboxlon = np.where((lon >= lonReykjavik-tol) & (lon <= lonReykjavik+tol))[0]
        rboxlat = np.where((lat >= latReykjavik-tol) & (lat <= latReykjavik+tol))[0]
        
        chksum = np.sum(eof[rboxlat[:,None],rboxlon[None,:]],(0,1))
        if chksum > 0:
            #print("\t Flipping sign based on Reykjavik, Ens%i" % (e+1))
            eof *= -1
            pcs *= -1
        
         # Double Check with Lisbon
        lboxlon = np.where((lon >= lonLisbon-tol) & (lon <= lonLisbon+tol))[0]
        lboxlat = np.where((lat >= latLisbon-tol) & (lat <= latLisbon+tol))[0]
                
        chksum = np.nansum(eof[lboxlat[:,None],lboxlon[None,:]],(0,1))
        if chksum < 0:
            #print("\t Flipping sign based on Lisbon,Ens%i" % (e+1))
            eof *= -1
            pcs *= -1
    
        naoidx[e,:,:] = pcs.copy()
        naopattern[e,:,:,n] = eof.copy()
        print("\rCalculated EOFs for ens %02d" % (e+1),end="\r",flush=True)

# Visualize NAO Index and Pattern
if debug == 1:
    e = 3
    cints =np.arange(-5,5.5,0.5)
    
    fig,ax = plt.subplots(2,1)
    
    # Plot NAO Index
    ax[0].plot(naoidx[e,:,0])
    ax[0].set_title("NAO Index, Ensemble %i,Variance Explained:%.2f%s" % (e+1,varexp[e,0]*100,"%",))
    ax[0].set_ylim([-6,6])
    ax[0].set_xlabel("Time (Months) ")
    
    # Plot Spatial Pattern
    pcm = ax[1].contourf(lon,lat,naopattern[e,:,:,0],levels=cints,cmap='seismic')
    fig.colorbar(pcm,ax=ax[1])
    ax[1].set_title("NAO Spatial Pattern (Sea-Level Pressure Anomaly, hPa), Ensemble %i" % (e+1))
    ax[1].set_xlabel("Longitude")
    ax[1].set_ylabel("Latitude")
    
    plt.tight_layout()

#%% 5 - Save Output
np.save("%sCESM1LE_NAOIndex_%s-%s_detrend%i.npy" % (outpath,start[0:4],end[0:4],deg),naoidx)
np.save("%sCESM1LE_NAOPattern_%s-%s_detrend%i.npy" % (outpath,start[0:4],end[0:4],deg),naopattern)
np.save("%sCESM1LE_NAOVarExp_%s-%s_detrend%i.npy" % (outpath,start[0:4],end[0:4],deg),varexp)
