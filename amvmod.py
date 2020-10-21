#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amvmod

Module containing functions for predict_amv
Working on updating documentation...


@author: gliu
"""

from scipy.signal import butter,filtfilt

import numpy as np
import matplotlib.pyplot as plt

## Processing/Analysis ----
def find_nan(data,dim):
    """
    For a 2D array, remove any point if there is a nan in dimension [dim]
    
    Inputs:
        1) data: 2d array, which will be summed along last dimension
        2) dim: dimension to sum along. 0 or 1
    Outputs:
        1) okdata: data with nan points removed
        2) knan: boolean array with indices of nan points
        

    """
    
    # Sum along select dimension
    if len(data.shape) > 1:
        datasum = np.sum(data,axis=dim)
    else:
        datasum = data.copy()
    
    
    # Find non nan pts
    knan  = np.isnan(datasum)
    okpts = np.invert(knan)
    
    if len(data.shape) > 1:
        if dim == 0:
            okdata = data[:,okpts]
        elif dim == 1:    
            okdata = data[okpts,:]
    else:
        okdata = data[okpts]
        
    return okdata,knan,okpts


def eof_simple(pattern,N_mode,remove_timemean):
    """
    Simple EOF function based on script by Yu-Chiao
    
    
    Inputs:
        1) pattern: Array of Space x Time [MxN], no NaNs
        2) N_mode:  Number of Modes to output
        3) remove_timemean: Set 1 to remove mean along N
    
    Outputs:
        1) eof: EOF patterns   [M x N_mode]
        2) pcs: PC time series [N x N_mode]
        3) varexp: % Variance explained [N_mode]
    
    Dependencies:
        import numpy as np
    
    """
    pattern1 = pattern.copy()
    nt = pattern1.shape[1] # Get time dimension size
    ns = pattern1.shape[0] # Get space dimension size
    
    # Preallocate
    eofs = np.zeros((ns,N_mode))
    pcs  = np.zeros((nt,N_mode))
    varexp = np.zeros((N_mode))
    
    # Remove time mean if option is set
    if remove_timemean == 1:
        pattern1 = pattern1 - pattern1.mean(axis=1)[:,None] # Note, the None adds another dimension and helps with broadcasting
    
    # Compute SVD
    [U, sigma, V] = np.linalg.svd(pattern1, full_matrices=False)
    
    # Compute variance (total?)
    norm_sq_S = (sigma**2).sum()
    
    for II in range(N_mode):
        
        # Calculate explained variance
        varexp[II] = sigma[II]**2/norm_sq_S
        
        # Calculate PCs
        pcs[:,II] = np.squeeze(V[II,:]*np.sqrt(nt-1))
        
        # Calculate EOFs and normalize
        eofs[:,II] = np.squeeze(U[:,II]*sigma[II]/np.sqrt(nt-1))
    return eofs, pcs, varexp


def sel_region(var,lon,lat,bbox,reg_avg=0,reg_sum=0,warn=1):
    """
    
    Select Region
    
    Inputs
        1) var: ARRAY, variable with dimensions [lon x lat x otherdims]
        2) lon: ARRAY, Longitude values
        3) lat: ARRAY, Latitude values
        4) bbox: ARRAY, bounding coordinates [lonW lonE latS latN]
        5) reg_avg: BOOL, set to 1 to return regional average
        6) reg_sum: BOOL, set to 1 to return regional sum
        7) warn: BOOL, set to 1 to print warning text for region selection
    Outputs:
        1) varr: ARRAY: Output variable, cut to region
        2+3), lonr, latr: ARRAYs, new cut lat/lon
    
    
    """    
        
    # Find indices
    klat = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]
    if bbox[0] < bbox[1]:
        klon = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]
    elif bbox[0] > bbox[1]:
        if warn == 1:
            print("Warning, crossing the prime meridian!")
        klon = np.where((lon <= bbox[1]) | (lon >= bbox[0]))[0]
    
    
    lonr = lon[klon]
    latr = lat[klat]
    
    #print("Bounds from %.2f to %.2f Latitude and %.2f to %.2f Longitude" % (latr[0],latr[-1],lonr[0],lonr[-1]))
        
    
    # Index variable
    varr = var[klon[:,None],klat[None,:],...]
    
    if reg_avg==1:
        varr = np.nanmean(varr,(0,1))
        return varr
    elif reg_sum == 1:
        varr = np.nansum(varr,(0,1))
        return varr
    return varr,lonr,latr

def calc_AMV(lon,lat,sst,bbox,order,cutofftime,awgt):
    """
    Calculate AMV Index for detrended/anomalized SST data [LON x LAT x Time]
    given bounding box [bbox]. Applies area weight based on awgt
    
    Parameters
    ----------
    lon : ARRAY [LON]
        Longitude values
    lat : ARRAY [LAT]
        Latitude Values
    sst : ARRAY [LON x LAT x TIME]
        Sea Surface Temperature
    bbox : ARRAY [LonW,LonE,LonS,LonN]
        Bounding Box for Area Average
    order : INT
        Butterworth Filter Order
    cutofftime : INT
        Filter Cutoff, expressed in same timesteps as input data
    awgt : INT
        0 = No weight, 1 = cos(lat), 2 = sqrt(cos(lat))
        
    Returns
    -------
    amv: ARRAY [TIME]
        AMV Index (Not Standardized)
    
    aa_sst: ARRAY [TIME]
        Area Averaged SST

    # Dependencies
    functions
        area_avg
    modules
        numpy as np
        from scipy.signal import butter,filtfilt
    """
    
    # Take the weighted area average
    aa_sst = area_avg(sst,bbox,lon,lat,awgt)

    # Design Butterworth Lowpass Filter
    filtfreq = len(aa_sst)/cutofftime
    nyquist  = len(aa_sst)/2
    cutoff = filtfreq/nyquist
    b,a    = butter(order,cutoff,btype="lowpass")
    
    # Compute AMV Index
    amv = filtfilt(b,a,aa_sst)
    
    return amv,aa_sst


def detrend_poly(x,y,deg):
    """
    Matrix for of polynomial detrend
    # Based on :https://stackoverflow.com/questions/27746297/detrend-flux-time-series-with-non-linear-trend
    
    Inputs:
        1) x --> independent variable
        2) y --> 2D Array of dependent variables
        3) deg --> degree of polynomial to fit
    
    """
    # Transpose to align dimensions for polyfit
    if len(y) != len(x):
        y = y.T
    
    # Get the fit
    fit = np.polyfit(x,y,deg=deg)
    
    # Prepare matrix (x^n, x^n-1 , ... , x^0)
    #inputs = np.array([np.power(x,d) for d in range(len(fit))])
    inputs = np.array([np.power(x,d) for d in reversed(range(len(fit)))])
    # Calculate model
    model = fit.T.dot(inputs)
    # Remove trend
    ydetrend = y - model.T
    return ydetrend,model

def lon360to180(lon360,var):
    """
    Convert Longitude from Degrees East to Degrees West 
    Inputs:
        1. lon360 - array with longitude in degrees east
        2. var    - corresponding variable [lon x lat x time]
    """
    kw = np.where(lon360 >= 180)[0]
    ke = np.where(lon360 < 180)[0]
    lon180 = np.concatenate((lon360[kw]-360,lon360[ke]),0)
    var = np.concatenate((var[kw,...],var[ke,...]),0)
    
    return lon180,var


def area_avg(data,bbox,lon,lat,wgt):
    
    """
    Function to find the area average of [data] within bounding box [bbox], 
    based on wgt type (see inputs)
    
    Inputs:
        1) data: target array [lat x lon x otherdims]
        2) bbox: bounding box [lonW, lonE, latS, latN]
        3) lon:  longitude coordinate
        4) lat:  latitude coodinate
        5) wgt:  number to indicate weight type
                    0 = no weighting
                    1 = cos(lat)
                    2 = sqrt(cos(lat))
    
    Output:
        1) data_aa: Area-weighted array of size [otherdims]
        
    Dependencies:
        numpy as np
    

    """
        
    # Find lat/lon indices 
    kw = np.abs(lon - bbox[0]).argmin()
    ke = np.abs(lon - bbox[1]).argmin()
    ks = np.abs(lat - bbox[2]).argmin()
    kn = np.abs(lat - bbox[3]).argmin()
    
        
    # Select the region
    sel_data = data[kw:ke+1,ks:kn+1,:]
    
    # If wgt == 1, apply area-weighting 
    if wgt != 0:
        
        # Make Meshgrid
        _,yy = np.meshgrid(lon[kw:ke+1],lat[ks:kn+1])
        
        
        # Calculate Area Weights (cosine of latitude)
        if wgt == 1:
            wgta = np.cos(np.radians(yy)).T
        elif wgt == 2:
            wgta = np.sqrt(np.cos(np.radians(yy))).T
        
        # Remove nanpts from weight, ignoring any pt with nan in otherdims
        nansearch = np.sum(sel_data,2) # Sum along otherdims
        wgta[np.isnan(nansearch)] = 0
        
        # Apply area weights
        #data = data * wgtm[None,:,None]
        sel_data  = sel_data * wgta[:,:,None]

    
    # Take average over lon and lat
    if wgt != 0:

        # Sum weights to get total area
        sel_lat  = np.sum(wgta,(0,1))
        
        # Sum weighted values
        data_aa = np.nansum(sel_data/sel_lat,axis=(0,1))
    else:
        # Take explicit average
        data_aa = np.nanmean(sel_data,(0,1))
    
    return data_aa

## Plotting ----
def plot_AMV(amv,ax=None):
    
    """
    Plot amv time series
    
    Dependencies:
        
    matplotlib.pyplot as plt
    numpy as np
    """
    if ax is None:
        ax = plt.gca()
    
    
    htimefull = np.arange(len(amv))
    
    ax.plot(htimefull,amv,color='k')
    ax.fill_between(htimefull,0,amv,where=amv>0,facecolor='red',interpolate=True,alpha=0.5)
    ax.fill_between(htimefull,0,amv,where=amv<0,facecolor='blue',interpolate=True,alpha=0.5)

    return ax