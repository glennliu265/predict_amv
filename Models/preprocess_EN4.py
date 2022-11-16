#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Preprocess EN4 Data


Created on Fri Oct 21 16:06:27 2022

@author: gliu
"""

#%%

import numpy as np
import xarray as xr
import sys
import glob
from tqdm import tqdm

#%% User Edits

datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/EN4/"
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/EN4/proc/"

sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm


startyr  = 1900
endyr    = 2021
#start_time = '1900-01-01'
#end_time   = '2021-12-01'

#searchstr = "EN.4.2.2.f.analysis.c14.%04i%02i.nc" % (yr,m)

# Cropping Options
yrs  = np.arange(startyr,endyr+1,1)
bbox = [-80,0,0,65]

#%%

nclist = glob.glob(datpath+"*.nc")
nclist.sort()


tsize = (endyr-startyr+1)*12

it = 0
for y,yr in tqdm(enumerate(yrs)):
    
    for im in range(12):
        
        m=im+1
        ncname = "%sEN.4.2.2.f.analysis.c14.%04i%02i.nc" % (datpath,yr,m)
        ds     = xr.open_dataset(ncname)
        
        
        # Flip the longitude
        ds     = proc.lon360to180_xr(ds,lonname='lon')
        
        
        ds = ds.isel(depth=0)
        
        
        ds = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
        
        
        if y == 0 and im == 0: # Preallocate
            ds_all = ds.copy()
        else:
            ds_all = xr.concat([ds_all,ds],dim='time')

# Save the output
savename = "%sEN4_concatenate_%sto%s_lon%02ito%02i_lat%02ito%02i.nc" % (outpath,startyr,endyr,
                                                                        bbox[0],bbox[1],bbox[2],bbox[3])
ds_all.to_netcdf(savename)