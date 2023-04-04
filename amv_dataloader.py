#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Some data loading scripts


    load_target_cesm : Load Target for CESM1-LENS training/testing
    

Created on Thu Mar  2 21:40:28 2023

@author: gliu
"""


import numpy as np
import xarray as xr

def load_target_cesm(datpath=None,region=None,detrend=False,regrid=None):
    """
    Load target for AMV prediction, as calculated from the script:
        
    Inputs:
        datpath [STR]  : Path to the dataset. Default is "../../CESM_data/"
        region  [STR]  : Region over which Index was calculated over (3-letter code). Default is None, whole basin
        detrend [BOOL] : Set to True if data was detrended. Default is False
        regrid  [STR]  : Regridding Option. Default is the default grid.
    Output:
        target  [ARRAY: ENS x Year] : Target index values
    """
    if datpath is None:
        datpath = "../../CESM_data/"
    # Load CESM Target
    if region is None:
        target = np.load('../../CESM_data/CESM_label_amv_index_detrend%i_regrid%s.npy'% (detrend,regrid))
    else:
        target = np.load('../../CESM_data/CESM_label_%s_amv_index_detrend%i_regrid%s.npy'% (region,detrend,regrid))
    return target

def load_data_cesm(varnames,bbox,datpath=None,detrend=False,regrid=None,return_latlon=False):
    """
    Load inputs for AMV prediction, as calculated from the script:
        
    Inputs:
        varnames [LIST]  : Name of variable in CESM
        bbox     [LIST]  : Bounding Box in the order [LonW,lonE,latS,latN]
        datpath  [STR]   : Path to the dataset. Default is "../../CESM_data/"
        detrend  [BOOL]  : Set to True if data was detrended. Default is False
        regrid   [STR]   : Regridding Option. Default is the default grid.
    Output:
        data     [ARRAY: channel x ens x yr x lat x lon] : Target index values
    
    """
    if datpath is None:
        datpath = "../../CESM_data/"
    for v,varname in enumerate(varnames):
        ds        = xr.open_dataset('../../CESM_data/CESM1LE_%s_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc'% (varname,detrend,regrid))
        ds        = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
        outdata   = ds[varname].values[None,...] # [channel x ens x yr x lat x lon]
        if v == 0:
            data = outdata.copy()
        else:
            data = np.concatenate([data,outdata],axis=0)
    if return_latlon:
        return data, ds.lat.values,ds.lon.values
    return data

def load_data_reanalysis(dataset_name,varname,bbox,datpath=None,detrend=False,regrid="CESM1",return_latlon=False):
    """
    Load predictors for a selected reanalysis dataset, preprocessed by [regrid_reanalysis_cesm1.py].
    
    """
    if datpath is None:
        datpath    = "../../CESM_data/Reanalysis/regridded/"
    if dataset_name == "HadISST":
        date_range = "18700101_20221231"
    ncname    = "%s%s_%s_NAtl_%s_bilinear_detrend%i_regrid%s.nc" % (datpath,dataset_name,varname,date_range,detrend,regrid) 
    ds        = xr.open_dataset(ncname)
    ds        = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3])) # [yr x lat x lon]
    data      = ds[varname].values[None,None,...]                             # [channel x ens x yr x lat x lon]
    if return_latlon:
        return data, ds.lat.values,ds.lon.values
    return data
    








