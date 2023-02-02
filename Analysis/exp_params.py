#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Experiment/Model Params and Information.
Moved to predict_amv_params.py. Delete this script....

Created on Mon Dec  5 13:11:53 2022

@author: gliu
"""

import matplotlib.pyplot as plt


#%% Plotting and other information


# Regions
regions     = ("NAT","SPG","STG","TRO")#("NAT","SPG","STG","TRO")
rcolors     = ("k","b",'r',"orange")
bbox_SP     = [-60,-15,40,65]
bbox_ST     = [-80,-10,20,40]
bbox_TR     = [-75,-15,10,20]
bbox_NA     = [-80,0 ,0,65]
bbox_NA_new = [-80,0,10,65]
bbox_ST_w   = [-80,-40,20,40]
bbox_ST_e   = [-40,-10,20,40]
bboxes      = (bbox_NA,bbox_SP,bbox_ST,bbox_TR,) # Bounding Boxes


# Variables (allpred)
allpred       = ("SST","SSS","PSL","SSH")
apcolors      = ("r","limegreen","pink","darkblue")

# Variables (all, old pre 2022.12.09)
varnames      = ("SST","SSS","PSL","SSH","BSF","HMXL",)
varcolors     = ("r","limegreen","pink","darkblue","purple","cyan")
threscolors   = ("r","gray","cornflowerblue")

# Variables (all, new since 2022.12.09)
varnames     = ("SST","SSS","PSL","BSF","SSH","HMXL")
varnamesplot = ("SST","SSS","SLP","BSF","SSH","MLD")
varcolors    = ("r","violet","yellow","darkblue","dodgerblue","cyan")
varmarker     = ("o","d","x","v","^","*")


# Darkmode Settings
darkmode  = True
if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
else:
    plt.style.use('default')
    dfcol = "k"

# ==========
#%% Exp 1
# ==========

expdir         = "CNN2_singlevar"
allpred        = ("SST","SSS","PSL","SSH")


#%%



#%% Simple CNN
modelname = "simplecnn"

nchannels     = [32,64]
filtersizes   = [[2,3],[3,3]]
filterstrides = [[1,1],[1,1]]
poolsizes     = [[2,3],[2,3]]
poolstrides   = [[2,3],[2,3]]