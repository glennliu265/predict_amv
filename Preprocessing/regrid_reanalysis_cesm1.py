#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Regrid the reanalysis datasets to CESM1 resolution using xesmf.
The following steps are performed in this order


    1. Load data and target lat/lon
    2. Crop data to time-period and standardize dimensions (names, etc.)
    3. Apply ice mask
    4. Perform regridding (xesmf)
    
    --- Intermediate Saving  ---
    --- Intermediate Loading ---
    
    5. Crop data to region
    6. Calculate monthly anomalies (deseason) and annual average
    7. Detrend (if option is set)
    8. Normalize/Standardize
    9. Save the output (predictor)
    10. Calculate and save the index (target)

This follows a similar procedure to prep_data_byvariable, as described below:
 
    ""
    
    For the given large ensemble dataset/variable, perform the following:
        1. Regrid (if necessary) to the specified resolution
        
        <Note, I have outsourced the steps above to prep_mld_PIC.py>
        
        <THIS is what the script actually does...>
        2. Crop the Data in Time (1920 - 2005)
        3. Crop to Region (*****NOTE: assumes degrees West = neg!)
        4. Concatenate each ensemble member
        5. Output in array ['ensemble','year','lat','lon']

    For Single variable case...
        Based on procedure in prepare_training_validation_data.py
        ** does not apply land mask!
        
        6 . Calculate Monthly Anomalies + Annual Average
        7 . Remove trend (if specified)
        8 . Normalize data
        9 . Perform regridding (if option is set)
        10. Output in array ['ensemble','year','lat','lon']
    
    ""

Note: Regridding section requires xesmf_env.
Script currently written to run on stormtrack. Need to add astraeus paths.

Created on Mon Apr  3 14:13:23 2023

@author: gliu
"""

import numpy as np
import xarray as xr

import glob
import time
from tqdm import tqdm
import sys

import matplotlib.pyplot as plt

#%% import some packages and preloaded parameters (current)
machine = "Astraeus"
# Note: need to install xesmf for Astraeaus
if machine == "stormtrack":
    import xesmf as xe


    
    sys.path.append("../")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    
    
    from amv import proc
    #import predict_amv_params as pparams
    
    #latlon_path   = '/stormtrack/home/glliu/01_Data/'
    datpath       = "/stormtrack/data4/glliu/01_Data/Reanalysis/"
    
else:
    
    sys.path.append("../")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
    from amv import proc
    datpath       = "/Users/gliu/Globus_File_Transfer/Reanalysis/HadISST/"
    #reanalysis_dict = pparams.reanalysis_dict
#elif machine == "Astraeus":
    

stall         = time.time()

#%% User Edits

# File Information

dataset_name  = "HadISST"
ncname        = "HadISST_sst_18700115_20230115.nc"
icename       = "HadISST_ice_18700115_20230115.nc"
latlon_name   = "cesm_latlon360.npz"

varname       = "sst"
latlon_path   = "../../CESM_data/"
outpath       = "../../CESM_data/Reanalysis/regridded/"



# (Move to predict_amv_params after torch  module is removed...)
had_dict = {
    'dataset_name' : 'HadISST',
    'sst'          : 'sst',
    'ice'          : 'sic',
    'lat'          : 'latitude',
    'lon'          : 'longitude',
    'ystart'       : 1870,
    'drop_dims'    : ["nv",],
    }

indicts = (had_dict,)
reanalysis_names = [d['dataset_name'] for d in indicts]
reanalysis_dict  = dict(zip(reanalysis_names,indicts))



# Preprocessing Information (predictor)
regrid_data   = False # Set to true to rerun regridding section
detrend       = True
detrend_order = 2     # Order of polynomial fit
bbox          = [-90,20,0,90] # Crop Selection
ystart        = '1870-01-01'
yend          = '2022-12-31'
method        = "bilinear" # regridding method for POP ocean data
icethresperc  = 0.05       # Mask out if ice concenration exceeds this percent...
apply_limask  = True # Set to True to apply ice mask

# Target information
bbox_amv      = [-80,0,0,65]
region_name   = "NAT"

# Other toggles
debug         = True

in_dict = had_dict
if apply_limask is False:
    icethresperc = 1
    
# -----------------------------
#%% Load in the data and regrid
# -----------------------------
if regrid_data:
    
    # < 1 > Load data ---------------------------------------------------------
    # Load Lat/lon
    ll_ld             = np.load(latlon_path+latlon_name,allow_pickle=True)
    cesm_lon,cesm_lat = ll_ld['lon'],ll_ld['lat']
    dummyvar          = np.zeros([len(cesm_lon),len(cesm_lat),1])
    cesm_lon180,_     = proc.lon360to180(cesm_lon,dummyvar,)
    
    # < 2 > Crop time and trim dimensions -------------------------------------
    ncnames  = [ncname,icename]
    varnames = [in_dict['sst'],in_dict['ice']] 
    ds_all   = []
    for v in range(2):
        
        # Get data
        ds = xr.open_dataset(datpath+ncnames[v]).load()
        
        
        ds_slice = ds.sel(time=slice(ystart,yend))
        
        # Rename and drop dimensions
        dim_names_original = reanalysis_dict[dataset_name]
        rename_dict = {dim_names_original['lon'] : 'lon',
                       dim_names_original['lat'] : 'lat',
                       }
        ds_slice = ds_slice.rename_dims(rename_dict)
        
        dropdims = dim_names_original['drop_dims']
        ds_slice = ds_slice.drop_dims(dropdims)
        
        ds_all.append(ds_slice)
    ds_slice,ds_ice = ds_all
    
    # < 3 > Apply an icemask --------------------------------------------------
    if apply_limask:
        print("Applying Ice Mask where values exceed %f" % (icethresperc))
        # Make the Mask
        ds_ice_max   = ds_ice[in_dict['ice']].max('time')
        ocean_mask   = ds_ice_max.where(ds_ice_max < icethresperc)
        ocean_mask   = ocean_mask.where(ocean_mask == 0).values
        ocean_mask[ocean_mask == 0] = 1
        
        # Apply the mask
        ds_slice      = ds_slice * ocean_mask
        applied_mask  = ds_slice.sst.values
        savename_mask = "%s%s_icemask_perc%i.npy" % (outpath,dataset_name,icethresperc*100)
        np.save(savename_mask,applied_mask)
    else:
        icethresperc = 1
            
    
        
    # < 4 > Perform Regridding ------------------------------------------------
    ds_out    = xr.Dataset({'lat': (['lat'], cesm_lat), 'lon': (['lon'], cesm_lon180) })
    regridder = xe.Regridder(ds_slice, ds_out, 'nearest_s2d')
    
    # Regrid
    ds_regridded = regridder( ds_slice.transpose('time','lat','lon') )
    
    # Save CESM Version (NOTE getting weird output permission error on stormtrack....)
    encoding_dict = {varname : {'zlib': True}} 
    outname       = "%s%s_%s_regridCESM1_%s_icemask%03i.nc" % (outpath,dataset_name,varname,method,icethresperc*100)
    ds_regridded.to_netcdf(outname,encoding=encoding_dict)
        
        # Debugging plots
        #ds_regridded.isel(time=0).sst.plot(vmin=0,vmax=34),plt.show()

# ---------------------
#%% Preprocess and Crop
# ---------------------


# Re-load data (if necessary) -------------------------------------------------
if regrid_data is False:
    savename     = "%s%s_%s_regridCESM1_%s_icemask%03i.nc" % (outpath,dataset_name,varname,method,icethresperc*100)
    ds_regridded = xr.open_dataset(savename)

#%% < 5 > Crop data to region -----------------------------------------------------
ds_reg = ds_regridded.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3])).load()
if debug:
    ds_reg.isel(time=0).sst.plot(vmin=0,vmax=32)

#%% < 6 > Calculate monthly anomalies and annual average --------------------------
# Calculate monthly anomalies and annual average (taken from prep-data_byvariable.py)

st = time.time() #387 sec
ds_all_anom = (ds_reg.groupby('time.month') - ds_reg.groupby('time.month').mean('time')).groupby('time.year').mean('time')
print("Deseasoned in %.2fs!" % (time.time()-st))
if debug:
    ds_all_anom.isel(year=0).sst.plot(vmin=-2,vmax=2,cmap="RdBu_r")

#%% < 7 > Detrend (if option is set) --------------------------------------------
# NOTE: Need to rewrite this section...
if detrend:
    
    # Load out data and combine dimensions
    sst_raw         = ds_all_anom.sst.values
    ntime,nlat,nlon = sst_raw.shape # [time x lat x lon]
    sst_in          = sst_raw.reshape(ntime,nlat*nlon)
    
    # Remove NaNs by summing along axis 0 (time)
    okdata,knan,okpts = proc.find_nan(sst_in,0)
    
    # Detrend
    x               = np.arange(0,ntime,1)
    okdt,model      = proc.detrend_poly(x,okdata,detrend_order)
    sst_dt          = np.zeros((ntime,nlat*nlon))*np.nan
    sst_dt[:,okpts] = okdt
    sst_dt          = sst_dt.reshape(ntime,nlat,nlon)
    sst_model          = np.zeros((ntime,nlat*nlon))*np.nan
    sst_model[:,okpts] =  model.T
    sst_model          = sst_model.reshape(ntime,nlat,nlon)
    
    # Debug Check
    if debug:
        
        klon,klat = proc.find_latlon(-30,50,ds_all_anom.lon.values,ds_all_anom.lat.values)
        fig,ax = plt.subplots(1,1,figsize=(8,3))
        ax.plot(x,sst_raw[:,klat,klon],label="Raw")
        ax.plot(x,sst_dt[:,klat,klon] + sst_model[:,klat,klon],label="Detrended",color="orange",alpha=0.7)
        #ax.plot(x,sst_model[:,klon,klat],label="Poly Fit (%i-order)" % (detrend_order),color="k")
        ax.legend()
        #ax.plot(x,)
    
    # Replace variable
    coords         = {'year':ds_all_anom.year,'lat':ds_all_anom.lat,'lon':ds_all_anom.lon}
    ds_all_anom_dt = xr.DataArray(sst_dt,coords=coords,dims=coords,name='sst')
    
    # Detrending (copied from amv_hadisst.py)
    # print("WARNING, Detrending section still needs to be written.")
    # break
    # sst = ds_all_anom.sst.values #[time x lat x lon]

    # #ds_all_anom = ds_all_anom - ds_all_anom.mean('ensemble')
else:
    ds_all_anom_dt = ds_all_anom.copy()
        
#%% < 8 > Normalize and standardize data --------------------------------------------

mu            = ds_all_anom_dt.mean()
sigma         = ds_all_anom_dt.std()
ds_normalized = (ds_all_anom_dt - mu)/sigma
np.save('%s%s_nfactors_%s_detrend%i_regridCESM.npy' % (outpath,dataset_name,varname,detrend),(mu.values,sigma.values))

#%% < 9 > Save the output (predictor) --------------------------------------------

st = time.time() #387 sec
if varname != varname.upper():
    print("Capitalizing variable name.")
    ds_normalized_out = ds_normalized.rename(varname.upper())
    varname = varname.upper()
encoding_dict = {varname : {'zlib': True}} 
outname       = "%s%s_%s_NAtl_%s_%s_%s_detrend%i_regridCESM1.nc" % (outpath,dataset_name,varname,
                                                                         ystart.replace("-",""),yend.replace("-","")
                                                                         ,method,detrend)
ds_normalized_out.to_netcdf(outname,encoding=encoding_dict)

# --------------------------
#%% < 10 > Compute the target--------------------------------------------
# --------------------------
#%Calculate the Index (with detrend)
ds_amv_reg = ds_normalized_out.sel(lon=slice(bbox_amv[0],bbox_amv[1]),lat=slice(bbox_amv[2],bbox_amv[3]))
amv_index  = (np.cos(np.pi*ds_amv_reg.lat/180) * ds_amv_reg).mean(dim=('lat','lon'))

if detrend:
    amv_index,_ = proc.detrend_poly(x,amv_index,3)

outname    = "%s%s_label_%s_amv_index_detrend%i_regridCESM1.npy" % (outpath,dataset_name,region_name,detrend)
np.save(outname,amv_index)
print("Saved target to %s" % outname)

if debug:
    
    fig,ax=plt.subplots(1,1)

    ax.plot(x,amv_index)