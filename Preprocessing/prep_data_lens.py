#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Prepare datasets for ML training
    
    
Works with output from the hfcalc package with [pred_prep=True]:
    preproc_damping_lens.py
    preproc_CESM1_LENS.py
    
    [time x lat x lon], 1 nc file per ens. member
    land-ice masked & regridded
    
Performs the following preprocessing steps based on:
    prep_data_byvariable
    prepare_training_validation_data
    
<Section 1: Finalize PostProc>
1. Concatenate each ensembl member
2. Crop to time period (post-1920)
3. Crop to region ([-90,20,0,90])

<Section 2: Normalize, Detrend, Deseason>
4. Calculate Monthly Anomalies + Annual Averages
5. Remove trend (if specified)
6. Normalize data
10. Output in array ['ensemble','year','lat','lon']
    
Created on Mon Jan 23 11:55:25 2023
@author: gliu
"""

import time
import numpy as np
import xarray as xr
import glob
from scipy.io import loadmat
from tqdm import tqdm
import matplotlib.pyplot as plt
#%% User Edits

# indicate CMIP version
cmipver        = 6 # 5 or 6


# I/O, dataset, paths
regrid         = 3
dataset_names  = ("canesm2_lens","csiro_mk36_lens","gfdl_esm2m_lens","mpi_lens","CESM1")
varnames       = ("ts","ts","ts","ts","ts")
apply_limask   = False
varname_out    = "sst"

# Preprocessing and Cropping Options
#apply_limask   = False
detrend        = False
start          = "1920-01-01"
end            = "2005-12-31"
bbox           = [-90,20,0,90] # Crop Selection
bbox_fn        = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])

if apply_limask:
    lenspath       = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/ts/" # limkased
    outpath        = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/processed/"
else:
    lenspath       = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/nomask/"
    outpath       = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/nomask/processed/"

#%% Get list of files (last one is ensemble average)

ndata = len(dataset_names)
nclists = []
for d in range(ndata):
    ncsearch = "%s%s*.nc" % (lenspath,dataset_names[d])
    nclist   = glob.glob(ncsearch)
    nclist.sort()
    print("Found %02i files for %s!" % (len(nclist),dataset_names[d]))
    nclists.append(nclist)
    
#%% Section 1 (Finish Postprocessing for each dataset)

for d in range(len(dataset_names)):
    
    st_s1 = time.time()
    
    # <1> Concatenate Ensemble Members
    # Read in data [ens x time x lat x lon]
    varname = varnames[d] # Get variable name
    dsall   = xr.open_mfdataset(nclists[d][:-1],concat_dim="ensemble",combine="nested")
    
    # <2> Crop to Time
    dssel      = dsall.sel(time=slice(start,end))
    start_crop = str(dssel.time[0].values)[:10]
    end_crop   = str(dssel.time[-1].values)[:10]
    print("Time dimension is size %i from %s to %s" % (len(dssel.time),start_crop,end_crop))
    
    # <3> Crop to Region
    # First, fix an issue with longitude
    # Double counted longitude values by checking first and last longitude
    if np.all(dssel.isel(lon=0,ensemble=0,lat=50).ts.values == dssel.isel(lon=-1,ensemble=0,lat=50).ts.values):
        # Drop the last
        dssel = dssel.sel(lon=slice(dssel.lon[0],dssel.lon[-2]))
    
    if not np.any((dssel.lon.values)<0): # Longitude not flipped
        print("Correcting longitude values for %s because no negative one was found..." % (dataset_names[d]))
        dssel.coords['lon'] = (dssel.coords['lon'] + 180) % 360 - 180
        dssel = dssel.sortby(dssel['lon'])
    dssel = dssel.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
    
    # Load out data
    ds_all = dssel.load()
    print("Finished Section 1 for %s in %.2fs" % (dataset_names[d],time.time()-st_s1))
    #%% Section 2
    
    # --------------------------------
    # Deseason and take annual average
    # --------------------------------
    st = time.time() #387 sec
    ds_all_anom = (ds_all.groupby('time.month') - ds_all.groupby('time.month').mean('time')).groupby('time.year').mean('time')
    print("Deseasoned in %.2fs!" % (time.time()-st))
    
    # -------
    # Detrend
    # -------
    if detrend:
        ds_all_anom = ds_all_anom - ds_all_anom.mean('ensemble')
    
    # -------------------------
    # Normalize and standardize
    # -------------------------
    mu            = ds_all_anom.mean()
    sigma         = ds_all_anom.std()
    ds_normalized = (ds_all_anom - mu)/sigma
    savename      = '%s%s_nfactors_%s_detrend%i_regrid%sdeg_%s_%sto%s.npy' % (outpath,dataset_names[d],varname_out,
                                                         detrend,regrid,bbox_fn,start_crop[:4],end_crop[:4])
    np.save(savename,
            (mu.to_array().values,sigma.to_array().values))
    
    # ---------------
    # Save the output
    # ---------------
    st = time.time() #387 sec
    ds_normalized_out = ds_normalized.transpose('ensemble','year','lat','lon') # Transpose
    ds_normalized_out = ds_normalized_out.rename({varname:varname_out})        # Rename
    encoding_dict = {varname_out : {'zlib': True}}
    outname       = "%s%s_%s_NAtl_%sto%s_detrend%i_regrid%sdeg.nc" % (outpath,
                                                                             dataset_names[d],
                                                                             varname_out,start_crop[:4],end_crop[:4],detrend,regrid)
    ds_normalized_out.to_netcdf(outname,encoding=encoding_dict)
    print("Saved output to %s in %.2fs!" % (outname,time.time()-st))

#%%


#%%

# #%% Do some sanity checks with a dataset

# d = 0

# stest = time.time()
# dsall = xr.open_mfdataset(nclists[d][:-1],concat_dim="ensemble",combine="nested").load()
# print("Loaded data for %s in %.fs" % (dataset_names[d],time.time()-stest))

# dsall.ts.mean('time').std('ensemble').plot(),plt.show()


# # Load 2 members
# ds1 = xr.open_dataset(nclists[d][0])
# ds4 = xr.open_dataset(nclists[d][4])

