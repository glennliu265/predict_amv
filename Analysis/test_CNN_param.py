#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:38:28 2023

@author: gliu
"""


import numpy as np
import sys
import glob
import importlib
import copy
import xarray as xr

import torch
from torch import nn

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from tqdm import tqdm
import time
import os

from torch.utils.data import DataLoader, TensorDataset,Dataset

#%% Load custom packages

# LRP Methods
sys.path.append("/Users/gliu/Downloads/02_Research/03_Code/github/Pytorch-LRP-master/")
from innvestigator import InnvestigateModel

# Load modules (LRPutils by Peidong)
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/scrap/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/predict_amv/")
import LRPutils as utils

# Load visualization module
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz,proc

# Load parameter files
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import amvmod as am
import amv_dataloader as dl
import train_cesm_params as train_cesm_params
import pamv_visualizer as pviz


# Load relevant variables from parameter files
bboxes  = pparams.bboxes
regions = pparams.regions
rcolors = pparams.rcolors

classes = pparams.classes
proj    = pparams.proj
bbox    = pparams.bbox

datpath = pparams.datpath
figpath = pparams.figpath
proc.makedir(figpath)

# Load model_dict
nn_param_dict = pparams.nn_param_dict



#%%
expdir         = "FNN4_128_SingleVar"
eparams        = train_cesm_params.train_params_all[expdir]

leads          = np.arange(0,26,1)
varnames = ["SST",]


# LRP Settings (note, this currently uses the innvestigate package from LRP-Pytorch)
gamma          = 0.1
epsilon        = 0.1
innexp         = 2
innmethod      ='b-rule'
innbeta        = 0.1
#eparams  = 

# ============================================================
#%% Load the data 
# ============================================================
# Copied segment from train_NN_CESM1.py

# Load data + target
load_dict                      = am.prepare_predictors_target(varnames,eparams,return_nfactors=True,load_all_ens=True)
data                           = load_dict['data']
target_class                   = load_dict['target_class']



# Get necessary sizes
nchannels,nens,ntime,nlat,nlon = data.shape             
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN
nclasses                       = len(eparams['thresholds'])+1
nlead                          = len(leads)

# Count Samples...
am.count_samples(None,target_class)

#%% Load CNN

"""
Trying to build a CNN as a class following the example here:
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
netname    = "cnn_lrp"#'simplecnn'
param_dict = nn_param_dict['simplecnn']

print(param_dict)

#nchannels = [3,]

channels    = 1
num_classes = 3

# Trying to make a new cnn
nchannels     = [32,64]

filtersizes   = [[2,3],[3,3]]
filterstrides = [[1,1],[1,1]]
poolsizes     = [[2,3],[2,3]]
poolstrides   = [[2,3],[2,3]]

firstlineardim = am.calc_layerdims(nlat,nlon,filtersizes,filterstrides,poolsizes,poolstrides,nchannels)

import torch.nn.functional as F

class CNN2(nn.Module):
    
    def __init__(self,channels,nchannels,filtersizes,poolsizes,firstlineardim,num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels,out_channels=nchannels[0] ,kernel_size=filtersizes[0])
        self.pool  = nn.MaxPool2d(kernel_size=poolsizes[0])
        self.conv2 = nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1])
        self.fc1   = nn.Linear(in_features=firstlineardim,out_features=64)
        self.fc2   = nn.Linear(64, num_classes)
        
    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

pmodel = CNN2(channels,nchannels,filtersizes,poolsizes,firstlineardim,num_classes)

inn_model_new = InnvestigateModel(pmodel, lrp_exponent=innexp,
                      method=innmethod,
                      beta=innbeta)


#%% 

"""
Explicitly declare each layer, as is done in the function version
"""

layers = [
        nn.Conv2d(in_channels=channels, out_channels=nchannels[0], kernel_size=filtersizes[0]),
        #nn.Tanh(),
        nn.ReLU(),
        #nn.Sigmoid(),
        nn.MaxPool2d(kernel_size=poolsizes[0]),

        nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1]),
        #nn.Tanh(),
        nn.ReLU(),
        #nn.Sigmoid(),
        nn.MaxPool2d(kernel_size=poolsizes[1]),
        
        nn.Flatten(),
        nn.Linear(in_features=firstlineardim,out_features=64),
        #nn.Tanh(),
        nn.ReLU(),
        #nn.Sigmoid(),
        
        nn.Dropout(p=0.5),
        nn.Linear(in_features=64,out_features=num_classes)
        ]

pmodel_old = nn.Sequential(*layers) # Set up model

inn_model_old = InnvestigateModel(pmodel, lrp_exponent=innexp,
                      method=innmethod,
                      beta=innbeta)


#%% Test the model (cppied from viz_LRP_predictor.py)
"""
This section prepares the inputs for testing the model
"""

v    = 0
lead = 0
predictors = data[[v],...] # Get selected predictor
X,y_class  = am.apply_lead(predictors,target_class,lead,reshape=True,ens=eparams['ens'],tstep=ntime)


        
# ------------------------
# 10. Train/Test/Val Split
# ------------------------
X_subsets,y_subsets = am.train_test_split(X,y_class,eparams['percent_train'],
                                               percent_val=eparams['percent_val'],
                                               debug=False,offset=eparams['cv_offset'])

# --------------------------

# Convert to Tensors
X_subsets = [torch.from_numpy(X.astype(np.float32)) for X in X_subsets]
y_subsets = [torch.from_numpy(y.astype(np.compat.long)) for y in y_subsets]

if eparams['percent_val'] > 0:
    X_train,X_test,X_val = X_subsets
    y_train,y_test,y_val = y_subsets
else:
    X_train,X_test       = X_subsets
    y_train,y_test       = y_subsets


#%% Pass thru model


"""
This didn't work. It appears that the INNvestigate package has some trouble with this.
It doesn;t support one of the features

"""
input_data                       = X_test.float()[[0],...]
pred_new,rel_new                 = inn_model_new.innvestigate(in_tensor=input_data)
pred_old,rel_old                 = inn_model_old.innvestigate(in_tensor=input_data)



        
#%% Let's try captum....
import captum


# A bunch of scra below I should delete...
# pmodel = am.transfer_model(netname,3,cnndropout=param_dict['cnndropout'],unfreeze_all=True,
#                         nlat=nlat,nlon=nlon,nchannels=nchannels)

# inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
#                       method=innmethod,
#                       beta=innbeta)




lrp       = captum.attr.LRP(pmodel)
result    = lrp.attribute(input_data,target=(2,3))

"""

# Ok this seems to be not working. I get the error: RuntimeError: 
    Module MaxPool2d(kernel_size=[2, 3], stride=[2, 3], padding=0, dilation=1, ceil_mode=False) 
    is being used more than once in the network, which is not supported by LRP. 
    Please ensure that module is being used only once in the network. lets try again

"""

#%% Try making CNN with only 1 max pool...

channels    = 1
num_classes = 3

# Trying to make a new cnn
nchannels     = [32,]

filtersizes    = [[3,3]]
filterstrides  = [[1,1]]
poolsizes      = [[2,3]]
poolstrides    = [[2,3]]

firstlineardim = am.calc_layerdims(nlat,nlon,filtersizes,filterstrides,poolsizes,poolstrides,nchannels)

import torch.nn.functional as F

class CNN2(nn.Module):
    
    def __init__(self,channels,nchannels,filtersizes,poolsizes,firstlineardim,num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels,out_channels=nchannels[0] ,kernel_size=filtersizes[0])
        self.pool  = nn.MaxPool2d(kernel_size=poolsizes[0])
        #self.conv2 = nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1])
        self.fc1   = nn.Linear(in_features=firstlineardim,out_features=64)
        self.fc2   = nn.Linear(64, num_classes)
        
    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.pool(self.conv1(x)))
        #x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

pmodel_1pool = CNN2(channels,nchannels,filtersizes,poolsizes,firstlineardim,num_classes)


lrp       = captum.attr.LRP(pmodel_1pool)
result    = lrp.attribute(input_data,target=1)

plt.pcolormesh(result.detach().numpy().squeeze(),vmin=-.02,vmax=.02,cmap="RdBu_r"),plt.colorbar()
"""
After removing the max pool, it appears that captum now works. But at what cost?
It seems that captum only supports 1 max pool layer which is inconvenient
flatten is not supported either.

Some of the errors may come in reusing functional nonlinearities
https://captum.ai/docs/faq#can-my-model-use-functional-non-linearities-eg-nnfunctionalrelu-or-can-reused-modules-be-used-with-captum


"""
#%% Try rewriting the function

class CNN2(nn.Module):
    
    def __init__(self,channels,nchannels,filtersizes,poolsizes,firstlineardim,num_classes):
        super().__init__()
        self.conv1  = nn.Conv2d(in_channels=channels,out_channels=nchannels[0] ,kernel_size=filtersizes[0])
        self.pool1  = nn.MaxPool2d(kernel_size=poolsizes[0])
        self.activ1 = nn.ReLU()
        self.conv2  = nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1])
        self.pool2  = nn.MaxPool2d(kernel_size=poolsizes[1])
        self.activ2 = nn.ReLU()
        self.fc1    = nn.Linear(in_features=firstlineardim,out_features=64)
        self.activ3 = nn.ReLU()
        self.fc2    = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.activ1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.activ2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.activ3(x)
        x = self.fc2(x)
        return x

pmodel_old = CNN2(channels,nchannels,filtersizes,poolsizes,firstlineardim,num_classes)


lrp       = captum.attr.LRP(pmodel_old)
result    = lrp.attribute(input_data,target=0)
plt.pcolormesh(result.detach().numpy().squeeze(),vmin=-.02,vmax=.02,cmap="RdBu_r"),plt.colorbar()

"""
Ok it kinda seems to be working, so i guess it is important to epxlicitly define each part of the model....

I also need to figure out what the "target" argument result option does....

"""
#%% Tryh some class definitions




