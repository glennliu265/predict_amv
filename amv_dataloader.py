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
         [prepare_regional_targets.py]
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
        ds        = xr.open_dataset('%sCESM1LE_%s_NAtl_19200101_20051201_bilinear_detrend%i_regrid%s.nc'% (datpath,varname,detrend,regrid))
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

def load_target_reanalysis(dataset_name,region_name,datpath=None,detrend=False,):
    """
    Load target for a selected reanalysis dataset, preprocessed by [regrid_reanalysis_cesm1.py].
    
    """
    if datpath is None:
        datpath    = "../../CESM_data/Reanalysis/regridded/"
    fn     = "%s%s_label_%s_amv_index_detrend%i_regridCESM1.npy" % (datpath,dataset_name,region_name,detrend)
    target = np.load(fn)
    return target
    

def load_persistence_baseline(dataset_name,datpath=None,return_npfile=False,region=None,quantile=False,
                              detrend=False,limit_samples=True,nsamples=None,repeat_calc=1,ens=42):
    
    if datpath is None:
        datpath = "../Data/Metrics/"
    if dataset_name == "CESM1":
        # Taken from viz_acc_byexp, generated using [Persistence_Classification_Baseline.py]
        datpath = "../../CESM_data/Metrics/"
        
        #fn_base   = "leadtime_testing_ALL_AMVClass3_PersistenceBaseline_1before_nens40_maxlead24_"
        #fn_extend = "detrend%i_noise0_nsample400_limitsamples1_ALL_nsamples1.npz" % (detrend)
        #ldp       = np.load(datpath+fn_base+fn_extend,allow_pickle=True)
        
        fn_base   = "Classification_Persistence_Baseline_ens%02i_RegionNone_maxlead24_step3_" % ens
        fn_extend = "nsamples%s_detrend%i_100pctdata.npz" % (nsamples,detrend)
        
        ldp       = np.load(datpath+fn_base+fn_extend,allow_pickle=True)
        class_acc = np.array(ldp['arr_0'][None][0]['acc_by_class']) # [Lead x Class]}
        total_acc = np.array(ldp['arr_0'][None][0]['total_acc'])
        
        if len(total_acc) == 9:
            persleads = np.arange(0,25,3)
        else:
            persleads = np.arange(0,26,1)
    elif dataset_name == "HadISST":
        # Based on output from [calculate_persistence_baseline.py]
        savename      = "%spersistence_baseline_%s_%s_detrend%i_quantile%i_nsamples%s_repeat%i.npz" % (datpath,dataset_name,
                                                                                                    region,detrend,
                                                                                                    quantile,nsamples,repeat_calc)
        ldp = np.load(savename,allow_pickle=True)
        class_acc = ldp['acc_by_class']
        total_acc = ldp['total_acc']
        persleads    = ldp['leads']
    else:
        print("Currently, only CESM1 and HadISST are supported")
    if return_npfile:
        return ldp
    else:
        return persleads,class_acc,total_acc
        
    
    

def load_nfactors(varnames,datpath=None,detrend=0,regrid=None):
    """Load normalization factors for data"""
    if datpath is None:
        datpath = "../../CESM_data/"
    vardicts = []
    for v,varname in enumerate(varnames):
        np_fn = "%sCESM1LE_nfactors_%s_detrend%i_regrid%s.npy" % (datpath,varname,detrend,regrid)
        ld    = np.load(np_fn,allow_pickle=True)
        vdict = {
            "mean" : ld[0].copy(),
            "stdev": ld[1].copy()}
        vardicts.append(vdict)
    return vardicts

    








