#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Investigate files which have the regridding error (scrap script)

Created on Tue Feb  7 14:03:08 2023

@author: gliu
"""

import time
import numpy as np
import xarray as xr
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import cartopy.crs as ccrs

#%% User Edits

datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/CMIP6_LENS/regridded/"
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz,proc

probnc = ("sos_ACCESS-ESM1-5_historical_r18i1p1f1_185001-201412.nc",
          "tos_ACCESS-ESM1-5_historical_r20i1p1f1_185001-201412.nc",
          "sos_ACCESS-ESM1-5_historical_r19i1p1f1_185001-201412.nc",
          "sos_ACCESS-ESM1-5_historical_r8i1p1f1_185001-201412.nc",
          "sos_ACCESS-ESM1-5_historical_r11i1p1f1_185001-201412.nc",
          "sos_ACCESS-ESM1-5_historical_r15i1p1f1_185001-201412.nc",
          "tos_ACCESS-ESM1-5_historical_r14i1p1f1_185001-201412.nc",
          "tos_ACCESS-ESM1-5_historical_r11i1p1f1_185001-201412.nc"
          )
#%% Load the data

nprob  = len(probnc)
ds_all = []
for n in range(nprob):
    ds_all.append(xr.open_dataset(datpath+probnc[n]))
    
    
    
#%% They all seem to open ok..

n=0

ds = ds_all[n].load()

tends = []
tlims = []
for n in range(nprob):
    ds = ds_all[n]
    print("Time Range for %s is size %i, with bnds %s to %s \n" % (probnc[n],len(ds.time),ds.time[0].values,ds.time[-1].values))
    tends.append(ds.time[-1].values)
    tlims.append(len(ds.time))
    
#%% Check the raw data

reals = ("r18i1p1f1","r20i1p1f1","r19i1p1f1","r8i1p1f1","r11i1p1f1","r15i1p1f1","r14i1p1f1","r11i1p1f1")
dvars = ("sos"      ,"tos"      ,"sos"      ,"sos"     ,"sos"      ,"sos"      ,"tos"      ,"tos"      )
dmods = ("ACCESS-ESM1-5",) * len(reals)

datpath_raw = "/Users/gliu/Globus_File_Transfer/CMIP6/"
rawncs = []
rawds  = []
for r,real in enumerate(reals):
    # Reconstruct the paths
    rpath = "%s%s/%s/%s/%s_Omon_%s_historical_%s_gn_185001-201412.nc" % (datpath_raw,dvars[r],dmods[r],real,dvars[r],dmods[r],real,)
    rawncs.append(rpath)
    rawds.append(xr.open_dataset(rpath))


for n in range(nprob):
    ds = rawds[n]
    print("Time Range for %s is size %i, with bnds %s to %s \n" % (probnc[n],len(ds.time),ds.time[0].values,ds.time[-1].values))
    
#%% Check if means are ok



dsm = []
for n in range(nprob):
    ds = rawds[n]
    dsm.append(ds.load())
    #dsm[n][dmods[n]].plot()
    #plt.show()



#%% Print info to send to Ray

rpath_globus = [path[33:] for path in rawncs]




#%% It seems that the time bounds might be th eissue

