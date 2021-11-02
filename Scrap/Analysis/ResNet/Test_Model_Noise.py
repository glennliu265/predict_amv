#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


# Load Model Weights and test

Created on Fri Feb  5 02:50:08 2021

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset,Dataset
import os
import copy
import timm



# -------------
#%% User Edits
# -------------

# Data preparation settings
leads          = [24,]#np.arange(0,25,3)    # Time ahead (in years) to forecast AMV
season         = 'Ann'                # Season to take mean over ['Ann','DJF','MAM',...]
indexregion    = 'NAT'                # One of the following ("SPG","STG","TRO","NAT")
resolution     = '224pix'             # Resolution of dataset ('2deg','224pix')
detrend        = False                 # Set to true to use detrended data
usenoise       = False                # Set to true to train the model with pure noise

# Training/Testing Subsets
percent_train = 0.8   # Percentage of data to use for training (remaining for testing)
ens           = 40    # Ensemble members to use
tstep         = 86    # Size of time dimension (in years)
numruns       = 1    # Number of times to train each run

# Model training settings
unfreeze_all  = False # Set to true to unfreeze all layers, false to only unfreeze last layer
early_stop    = 1200                    # Number of epochs where validation loss increases before stopping
max_epochs    = 1200                    # Maximum number of epochs
batch_size    = 32                   # Pairs of predictions
loss_fn       = nn.MSELoss()          # Loss Function
opt           = ['Adam',1e-4,0]     # Name optimizer
reduceLR      = False                 # Set to true to use LR scheduler
LRpatience    = 3                     # Set patience for LR scheduler
netname       = 'resnet50'            #'simplecnn'           # Name of network ('resnet50','simplecnn')
tstep         = 86
outpath       = ''
cnndropout    = True                  # Set to 1 to test simple CN with dropout layer

# Options
debug     = True # Visualize training and testing loss
verbose   = False # Print loss for each epoch
checkgpu  = True # Set to true to check for GPU otherwise run on CPU
savemodel = True # Set to true to save model dict.
# -----------
#%% Functions
# -----------

def calc_layerdims(nx,ny,filtersizes,filterstrides,poolsizes,poolstrides,nchannels):
    """
    For a series of N convolutional layers, calculate the size of the first fully-connected
    layer

    Inputs:
        nx:           x dimensions of input
        ny:           y dimensions of input
        filtersize:   [ARRAY,length N] sizes of the filter in each layer [(x1,y1),[x2,y2]]
        poolsize:     [ARRAY,length N] sizes of the maxpooling kernel in each layer
        nchannels:    [ARRAY,] number of out_channels in each layer
    output:
        flattensize:  flattened dimensions of layer for input into FC layer

    """
    N = len(filtersizes)
    xsizes = [nx]
    ysizes = [ny]
    fcsizes  = []
    for i in range(N):
        xsizes.append(np.floor((xsizes[i]-filtersizes[i][0])/filterstrides[i][0])+1)
        ysizes.append(np.floor((ysizes[i]-filtersizes[i][1])/filterstrides[i][1])+1)

        xsizes[i+1] = np.floor((xsizes[i+1] - poolsizes[i][0])/poolstrides[i][0]+1)
        ysizes[i+1] = np.floor((ysizes[i+1] - poolsizes[i][1])/poolstrides[i][1]+1)

        fcsizes.append(np.floor(xsizes[i+1]*ysizes[i+1]*nchannels[i]))
    return int(fcsizes[-1])


def transfer_model(modelname,cnndropout=False,unfreeze_all=False):
    if 'resnet' in modelname: # Load from torchvision
        #model = models.resnet50(pretrained=True) # read in resnet model
        model = timm.create_model(modelname,pretrained=True)
        if unfreeze_all is False:
            # Freeze all layers except the last
            for param in model.parameters():
                param.requires_grad = False
        else:
            print("Warning: All weights are unfrozen!")
        model.fc = nn.Linear(model.fc.in_features, 1)                    # freeze all layers except the last one
    elif modelname == 'simplecnn': # Use Simple CNN from previous testing framework
        channels = 3
        nlat = 224
        nlon = 224

        # 2 layer CNN settings
        nchannels     = [32,64]
        filtersizes   = [[2,3],[3,3]]
        filterstrides = [[1,1],[1,1]]
        poolsizes     = [[2,3],[2,3]]
        poolstrides   = [[2,3],[2,3]]

        firstlineardim = calc_layerdims(nlat,nlon,filtersizes,filterstrides,poolsizes,poolstrides,nchannels)
        
        if cnndropout: # Include Dropout
            layers = [
                    nn.Conv2d(in_channels=channels, out_channels=nchannels[0], kernel_size=filtersizes[0]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=poolsizes[0]),
    
                    nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=poolsizes[1]),
    
                    nn.Flatten(),
                    nn.Linear(in_features=firstlineardim,out_features=64),
                    nn.ReLU(),
    
                    nn.Dropout(p=0.5),
                    nn.Linear(in_features=64,out_features=1)
                    ]
        else:
            layers = [
                    nn.Conv2d(in_channels=channels, out_channels=nchannels[0], kernel_size=filtersizes[0]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=poolsizes[0]),
    
                    nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=poolsizes[1]),
    
                    nn.Flatten(),
                    nn.Linear(in_features=firstlineardim,out_features=64),
                    nn.ReLU(),
    
                    #nn.Dropout(p=0.5),
                    nn.Linear(in_features=64,out_features=1)
                    ]
        model = nn.Sequential(*layers) # Set up model

    else: # Load from timm
        model = timm.create_model(modelname,pretrained=True)
        if unfreeze_all is False:
            # Freeze all layers except the last
            for param in model.parameters():
                param.requires_grad = False
        else:
            print("Warning: All weights are unfrozen!")
        model.classifier=nn.Linear(model.classifier.in_features,1)
    return model


#%%
modpath = "/Users/gliu/Downloads/2020_Fall/6.862/Project/CESM_data/Models/lead24s/"
PATH = modpath + "HPT_simplecnn_nepoch20_nens40_maxlead24_detrend0_noise0_cnndropoutTrue_run0_ALL_lead24.pt"
#Set up the model
model = transfer_model(netname,cnndropout=cnndropout,unfreeze_all=unfreeze_all)

#model = torch.load(PATH,map_location=torch.device('cpu'))


# Initialize some random predictions
nsamples = 20
xin = torch.from_numpy(np.random.normal(0,1,(nsamples*batch_size*2,3,224,224)).astype(np.float32))


model.eval()
xout = np.zeros((batch_size,nsamples))
for i in tqdm(range(nsamples)):
    
    xout[:,i] = model(xin[batch_size*i:(batch_size*(i+1)),:,:,:]).detach().numpy().squeeze()
    
xout = xout.reshape(np.prod(xout.shape))




plt.hist(xout)
plt.title("Predictions for Simple CNN at leadtime 24, given random noise as inputs")
plt.grid(True,ls='dotted')




