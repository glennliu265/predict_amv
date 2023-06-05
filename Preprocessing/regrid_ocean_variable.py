#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Regrid MLD (or selected variable) to cartesian lat/lon grid
and prepare seasonal means

Uses xesmf environment. Files are large--need to figure out compression
or datatype change.

Created on Tue Feb  2 21:24:45 2021

@author: gliu
"""

import time
import numpy as np
import xarray as xr
import glob
from scipy.io import loadmat
from tqdm import tqdm
import xesmf as xe

#%% User Edits

varname       = "BSF" # "HMXL"
mconfig       = "FULL_HTR" # [FULL_PIC, SLAB_PIC, FULL_HTR]

use_xesmf     = True # Use xESMF for regridding. False = box average
method        = "bilinear" # regridding method

use_mfdataset = False # Open all datasets at once (currently only works for PIC, boxavg)
outpath       = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/%s/" % varname

# Set ncsearch string on stormtrack based on input dataset
catdim  = 'time'
savesep = False # Save all files together
if mconfig == "FULL_PIC":
    ncsearch      = "b.e11.B1850C5CN.f09_g16.*.pop.h.%s.*.nc" % varname
elif mconfig == "SLAB_PIC":
    ncsearch      = "e.e11.B1850C5CN.f09_g16.*.pop.h.%s.*.nc" % varname
elif mconfig == "FULL_HTR":
    ncsearch      = "b.e11.B20TRC5CNBDRD.f09_g16.*.pop.h.%s.*.nc" % varname
    catdim  = 'ensemble'
    savesep = True # Save each ensemble member separately
    use_mfdataset = False

# Adjust data path on stormtrack
if varname == "SSS":
    datpath   = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/processed/ocn/proc/tseries/monthly/SSS/"
else:
    datpath   = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/ocn/proc/tseries/monthly/%s/" % varname

# Set up variables to keep for preprocessing script
varkeep   = [varname,"TLONG","TLAT","time"]

#%% Functions

# Define preprocessing variable
def preprocess(ds,varlist=varkeep):
    """"preprocess dataarray [ds],dropping variables not in [varlist] and 
    selecting surface variables at [lev=-1]"""
    # Drop unwanted dimension
    dsvars = list(ds.variables)
    remvar = [i for i in dsvars if i not in varlist]
    ds = ds.drop(remvar)
    
    # # Correct first month (Note this isn't working)
    if ds.time.values[0].month != 1:
         startyr = "%04i-01-01" % ds.time.values[0].year
         endyr = "%04i-12-01" % (ds.time.values[-1].year-1)
         correctedtime = xr.cftime_range(start=startyr,end=endyr,freq="MS",calendar="noleap") 
         ds = ds.assign_coords(time=correctedtime) 
         print("Corrected time to be from %s to %s"% (startyr,endyr))
    return ds


def getpt_pop_array(lonf,latf,invar,tlon,tlat,searchdeg=0.75,printfind=True,verbose=False):
    
    """
    IMPT: assumes input variable is of the shape [lat x lon x otherdims]
    tlon = ARRAY [lat x lon]
    tlat = ARRAY [lat x lon]
    """
    
    if lonf < 0:# Convet longitude to degrees East
        lonf += 360
    
    # Query Points
    quer = np.where((lonf-searchdeg < tlon) & (tlon < lonf+searchdeg) & (latf-searchdeg < tlat) & (tlat < latf+searchdeg))
    latid,lonid = quer
    
    if printfind:
        print("Closest LAT to %.1f was %s" % (latf,tlat[quer]))
        print("Closest LON to %.1f was %s" % (lonf,tlon[quer]))
        
    if (len(latid)==0) | (len(lonid)==0):
        if verbose:
            print("Returning NaN because no points were found for LAT%.1f LON%.1f"%(latf,lonf))
        return np.nan
        
    # Locate points on variable
    if invar.shape[:2] != tlon.shape:
        print("Warning, dimensions do not line up. Make sure invar is Lat x Lon x Otherdims")
        
    return invar[latid,lonid,:].mean(0) # Take mean along first dimension


#%% Load in data

if not use_xesmf:
    method = "boxAVG"

savename = "%s%s_%s_%s.nc" % (outpath,varname,mconfig,method)

# Get file names
globby = datpath+ncsearch
nclist =glob.glob(globby)
nclist = [nc for nc in nclist if "OIC" not in nc]
nclist.sort()
print("Found %i items" % (len(nclist)))

# Set up target latitude and longitude
latlonmat = "/home/glliu/01_Data/CESM1_LATLON.mat"
ll   = loadmat(latlonmat)
lat  = ll['LAT'].squeeze()
lon  = ll["LON"].squeeze()
lon1 = np.hstack([lon[lon>=180]-360,lon[lon<180]])
#lon1,_ = proc.lon360to180(lon,np.zeros((288,192,1)))

# Read in variables (all at once)
if use_mfdataset or not use_xesmf:
    ds = xr.open_mfdataset(nclist,concat_dim='time',
                       preprocess=preprocess,
                       combine='nested',
                       parallel="True",
                      )

# Define new grid (note: seems to support flipping longitude)
ds_out = xr.Dataset({'lat':lat,
                     'lon':lon1})
if use_xesmf:

    start = time.time()

    # Get Data Array, and rename coordinates to lon/lat
    if ~use_mfdataset:
        
        ds_rgrd = [] # Loop thru each array (mfdataset doesn't seem to be working?)
        for nc in tqdm(range(len(nclist))):
            ds = xr.open_dataset(nclist[nc])
            
            ds = ds.rename({"TLONG": "lon", "TLAT": "lat"})
            da = ds
            #da = ds[varname] # Using dataarray seems to throw an error
            
            # Initialize Regridder
            regridder = xe.Regridder(da,ds_out,method,periodic=True)
            #print(regridder)
                        
            # Regrid
            daproc = regridder(da[varname]) # Need to input dataarray
            
            print("Finished regridding in %f seconds" % (time.time()-start))
            ds_rgrd.append(daproc)
            
            # Save each ensemble member separately (or time period)
            if savesep: 
                savename = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/%s_%s_%s_num%02i.nc" % (varname,mconfig,method,nc)
                daproc.to_netcdf(savename,
                                 encoding={varname: {'zlib': True}})
                
        # Concatenate along selected dimension
        if not savesep:
            dsproc = xr.concat(ds_rgrd,dim=catdim)
        
    else: # Do all at once (seems to be failing due to cf compliant issue)
        
        ds = ds.rename({"TLONG": "lon", "TLAT": "lat"})
        #ds = ds.rename_dims({"nlon": "longitude", "nlat": "latitude"})
        da = ds#[varname]
        
        # Define new grid (note: seems to support flipping longitude)
        ds_out = xr.Dataset({'lat':lat,
                             'lon':lon1})
    
        # Initialize Regridder
        regridder = xe.Regridder(da,ds_out,method,periodic=True)
        print(regridder)
    
    
        # Regrid
        dsproc = regridder(da[varname])
        print("Finished regridding in %f seconds" % (time.time()-start))  
    
    # Save the data
    if not savesep:
        dsproc.to_netcdf(savename,
                         encoding={varname: {'zlib': True}})
    
else:

    # Load variables in 
    st    = time.time()
    hmxl  = ds[varname].values
    tlon  = ds.TLONG.values
    tlat  = ds.TLAT.values
    times = ds.time.values
    print("Read out data in %.2fs"%(time.time()-st))
    

    # Transpose the data
    h = hmxl.transpose(1,2,0) # [384,320,time]
    h.shape
    
    # Loop time
    
    start = time.time()
    icount= 0
    stol  = 0.75
    hclim = np.zeros((lon1.shape[0],lat.shape[0],h.shape[2]))
    for o in tqdm(range(lon1.shape[0])):
        
        # Get Longitude Value
        lonf = lon1[o]
        
        # Convert to degrees Easth
        if lonf < 0:
            lonf = lonf + 360
        
        for a in range(0,lat.shape[0]):
            
            
            # Get latitude indices
            latf = lat[a]
            
            # Get point
            value = getpt_pop_array(lonf,latf,h,tlon,tlat,searchdeg=stol,printfind=False)
            if np.any(np.isnan(value)):
                msg = "Land Point @ lon %f lat %f" % (lonf,latf)
                hclim[o,a,:] = np.ones(h.shape[2])*np.nan
                
            else:
                hclim[o,a,:] = value.copy()
            icount +=1
            #print("Completed %i of %i" % (icount,lon1.shape[0]*lat.shape[0]))
            
            
    print("Finished in %f seconds" % (time.time()-start))  
    
    dsproc = xr.DataArray(hclim,
                      dims={'lon':lon1,'lat':lat,'time':times},
                      coords={'lon':lon1,'lat':lat,'time':times},
                      name = varname
                      )
    dsproc.to_netcdf("/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/%s_PIC.nc" % (varname),
                     encoding={varname: {'zlib': True}})

