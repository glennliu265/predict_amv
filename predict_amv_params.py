#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict AMV, Parameter File

Created on Mon Jan 16 13:32:37 2023

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


#%% Project paths

datpath = "../../CESM_data/"
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20230210/"





#%% Regions (Bounding Boxes and Names)
regions       = ("NAT","SPG","STG","TRO")#("NAT","SPG","STG","TRO")
rcolors       = ("k","b",'r',"orange")
bbox_SP       = [-60,-15,40,65]
bbox_ST       = [-80,-10,20,40]
bbox_TR       = [-75,-15,10,20]
bbox_NA       = [-80,0 ,0,65]
bbox_NA_new   = [-80,0,10,65]
bbox_ST_w     = [-80,-40,20,40]
bbox_ST_e     = [-40,-10,20,40]
bboxes        = (bbox_NA,bbox_SP,bbox_ST,bbox_TR,) # Bounding Boxes

# Variables (allpred)
allpred       = ("SST","SSS","PSL","SSH")
apcolors      = ("r","limegreen","pink","darkblue")

# Variables (all, old pre 2022.12.09)
varnames      = ("SST","SSS","PSL","SSH","BSF","HMXL",)
varcolors     = ("r","limegreen","pink","darkblue","purple","cyan")
threscolors   = ("r","gray","cornflowerblue")

# Variables (all, new since 2022.12.09, updated 2023.02.07 adding UOHC/UOSC)
varnames      = ("SST","SSS","PSL","BSF","SSH","HMXL","UOHC","UOSC")
varnamesplot  = ("SST","SSS","SLP","BSF","SSH","MLD","UOHC","UOSC")
varnames_long = ("Temperature","Salinity","Pressure","Surface Height",
                 "Barotropic Streamfunction","Mixed-Layer Depth",
                 "Upper Ocean Heat Content","Upper Ocean Salt Content")
vunits        = ("$\degree$C","psu","mb","cm","Sv","cm","$J\,m^{-2}$","$J\,m^{-2}$")
varcolors     = ("r","violet","yellow","darkblue","dodgerblue","cyan","lightcoral","orchid")
varmarker     = ("o","d","x","v","^","*","1","2")

# Class Names and colors

classes       = ["AMV+","Neutral","AMV-"] # [Class1 = AMV+, Class2 = Neutral, Class3 = AMV-]
class_colors  = ("salmon","gray","cornflowerblue")

# Plotting
proj     = ccrs.PlateCarree()
bbox     = [-80,0,0,65]
plotbbox = [-80,0,0,62]



# ML Training Parameters
detrend       = 0
leads         = np.arange(0,27,3)
regrid        = None
tstep         = 86
ens           = 40
thresholds    = [-1,1] 
quantile      = False
percent_train = 0.8


#%% LENs Parameters

# CMIP5
dataset_names  = ("canesm2_lens" ,"csiro_mk36_lens","gfdl_esm2m_lens","mpi_lens"  ,"CESM1")
dataset_long   = ("CCCma-CanESM2","CSIRO-MK3.6"    ,"GFDL-ESM2M"     ,"MPI-ESM-LR","NCAR-CESM1")
dataset_colors = ("r"            ,"b"              ,"magenta"        ,"gold" ,"limegreen")
dataset_starts = (1950           ,1920             ,1950             ,1920        ,1920)

# CMIP6
cmip6_varnames       = ("tos","sos","zos")
cmip6_varnames_remap = ("SST","SSS","SSH")
cmip6_names          = ("ACCESS-ESM1-5","CanESM5","IPSL-CM6A-LR","MIROC6","MPI-ESM1-2-LR")
cmip6_colors         = ("orange"       ,"r"      ,"magenta"     ,"b"     ,"gold"        )

#%%
# # Darkmode Settings
# darkmode  = True
# if darkmode:
#     plt.style.use('dark_background')
#     dfcol = "w"
# else:
#     plt.style.use('default')
#     dfcol = "k"

# # ==========
# #%% Exp 1
# # ==========

# expdir         = "CNN2_singlevar"
# allpred        = ("SST","SSS","PSL","SSH")

# #%%
# #%% Simple CNN
# modelname = "simplecnn"

# nchannels     = [32,64]
# filtersizes   = [[2,3],[3,3]]
# filterstrides = [[1,1],[1,1]]
# poolsizes     = [[2,3],[2,3]]
# poolstrides   = [[2,3],[2,3]]