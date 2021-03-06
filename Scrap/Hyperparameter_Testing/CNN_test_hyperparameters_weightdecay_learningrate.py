#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN_test_hyperparameters_lr_wd

Testing learning rate and weight decay for a CNN

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import time
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,Dataset
import os


#%% Functions
def train_CNN(layers,loss_fn,optimizer,trainloader,testloader,max_epochs,verbose=True):
    """
    inputs:
        layers      - tuple of NN layers
        loss_fn     - (torch.nn) loss function
        opt         - tuple of [optimizer_name, learning_rate, weight_decay] for updating the weights
                      currently supports "Adadelta" and "SGD" optimizers
        trainloader - (torch.utils.data.DataLoader) for training dataset
        testloader  - (torch.utils.data.DataLoader) for testing dataset
        max_epochs  - number of training epochs
        verbose     - set to True to display training messages
    
    output:
    
    dependencies:
        from torch import nn,optim
        
    """
    model = nn.Sequential(*layers) # Set up model
    
    # Set optimizer
    if optimizer[0] == "Adadelta":
        opt = optim.Adadelta(model.parameters(),lr=optimizer[1],weight_decay=optimizer[2])
    elif optimizer[0] == "SGD":
        opt = optim.SGD(model.parameters(),lr=optimizer[1],weight_decay=optimizer[2])
        
    
    train_loss,test_loss = [],[]   # Preallocate tuples to store loss
    for epoch in tqdm(range(max_epochs)): # loop by epoch
        for mode,data_loader in [('train',trainloader),('test',testloader)]: # train/test for each epoch
    
            if mode == 'train':  # Training, update weights
                model.train()
            elif mode == 'eval': # Testing, freeze weights
                model.eval()
                
            runningloss = 0
            for i,data in enumerate(data_loader):
                
                # Get mini batch
                batch_x, batch_y = data
                
                # Set gradients to zero
                opt.zero_grad()
                
                # Forward pass
                pred_y = model(batch_x).squeeze()
                
                # Calculate loss
                loss = loss_fn(pred_y,batch_y.squeeze())
                
                # Update weights
                if mode == 'train':
                    loss.backward() # Backward pass to calculate gradients w.r.t. loss
                    opt.step()      # Update weights using optimizer
                runningloss += loss.item()
                
            if verbose: # Print message
                print('{} Set: Epoch {:02d}. loss: {:3f}'.format(mode, epoch+1, \
                                                runningloss/len(data_loader)))
            # Save running loss values for the epoch
            if mode == 'train':
                train_loss.append(runningloss/len(data_loader))
            else:
                test_loss.append(runningloss/len(data_loader))
                
    return model,train_loss,test_loss         

# -------------
#%% User Edits
# -------------

# Set Paths
machine='local-glenn'
if machine == 'local-glenn':
    os.chdir('/Users/gliu/Downloads/2020_Fall/6.862/Project/predict_amv/')
    outpath = '/Users/gliu/Downloads/2020_Fall/6.862/Project'
else:
    outpath = os.getcwd()

t/'
# Data preparation settings
lead          = 12    # Time ahead (in months) to forecast AMV
tstep         = 1032  # Total number of months

percent_train = 0.8   # Percentage of data to use for training (remaining for testing)
ens           = 1    # Ensemble members to use


# Select variable
channels   = 3     # Number of variables to include
varname    = 'SST+SSS+PSL'
sst_normed = np.load('../CESM_data/CESM_SST_normalized_lat_weighted.npy').astype(np.float32)
sss_normed = np.load('../CESM_data/CESM_SSS_normalized_lat_weighted.npy').astype(np.float32)
psl_normed = np.load('../CESM_data/CESM_PSL_normalized_lat_weighted.npy').astype(np.float32)
invars = [sst_normed,sss_normed,psl_normed]

# Model training settings
max_epochs    = 15 
batch_size    = 32                    # Pairs of predictions
loss_fn       = nn.MSELoss()          # Loss Function
optname       = 'Adadelta'    # Name optimizer
layers        = [
                nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=5),

                nn.Flatten(),
                nn.Linear(in_features=5*17*32,out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128,out_features=64),
                nn.ReLU(),

                nn.Dropout(p=0.5),
                nn.Linear(in_features=64,out_features=1)
                ]

# ---------------------
#%% Load and prep data
# ---------------------

allstart = time.time()

# Apply lead/lag to data
y = np.mean(sst_normed[:ens,lead:,:,:],axis=(2,3)).reshape((tstep-lead)*ens,1) # Take area average for SST 
X = np.transpose(
    np.array(invars)[:,:ens,0:tstep-lead,:,:].reshape(channels,(tstep-lead)*ens,33,89),
    (1,0,2,3))

# Print shapes
print("y is size: %s" % str(y.shape)) # [ data (ensemble*time) x 1 ]
print("X is size: %s" % str(X.shape)) # [ data (ensemble*time) x variable x lat x lon]

# Split into training and test sets
X_train = torch.from_numpy( X[0:int(np.floor(percent_train*(tstep-lead)*ens)),:,:,:] )
y_train = torch.from_numpy( y[0:int(np.floor(percent_train*(tstep-lead)*ens)),:] )

X_val = torch.from_numpy( X[int(np.floor(percent_train*(tstep-lead)*ens)):,:,:,:] )
y_val = torch.from_numpy( y[int(np.floor(percent_train*(tstep-lead)*ens)):,:] )

# Print shapes
print("Training set is size: %i" % y_train.shape[0]) 
print("Validation set is size: %i" %y_val.shape[0])  

# Put into pytorch DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)


# ----------------------------------------
# %% Manual Hyperparameter Testing: Set-up
# ----------------------------------------
# Setup: Indicate hyperparameter values and set up grid. Preallocate output variables and prepare data
# Currently, only works with varying eta and wd (need to think of more generalizable code...)

# User edits start here -----------------------
# Input some hyperparameters and their values, as well as choices for the data

maxorder = 0
minorder = -7
eta = [10**i for i in range(minorder,maxorder+1)]
wd  = eta.copy()

# Set the hyperparameter grid
param1 = eta.copy()
param2 = wd.copy()

# Save data
expname = "wd_eta_1e%i_1e%i_nens%i_lead%i_%s" % (minorder,maxorder,ens,lead,varname)
outname = "hyperparameter_testing_%s.npz" % (expname)

# Dimensions of target parameters
np1 = len(param1)
np2 = len(param2)
gridsize = np.array([np1,np2]) 

# Preallocate Relevant Variables...
corr_grid_train = np.zeros(gridsize)
corr_grid_test  = np.zeros(gridsize)
train_loss_grid = np.zeros(np.concatenate([gridsize,[max_epochs,]]))
test_loss_grid  = np.zeros(train_loss_grid.shape)

# ----------------------------------------------
# %% Manual Hyperparameter Testing: Testing Loop
# ----------------------------------------------


for i in range(np1): # Loop for Learning Rate
    eta_in = param1[i]
    for j in range(np2): # Loop for Weight Decay
        start = time.time()
        wd_in = param2[j]
        
        # Set Optimizer
        opt = [optname, eta_in, wd_in]
        
        # Train the model
        model,trainloss,testloss = train_CNN(layers,loss_fn,opt,train_loader,val_loader,max_epochs,verbose=False)
        
        # Save train/test loss
        train_loss_grid[i,j,...] = np.array(trainloss)
        test_loss_grid[i,j,...]  = np.array(testloss)
        
        # Evalute the model
        y_pred_val   = model(X_val).detach().numpy()
        y_valdt        = y_val.detach().numpy()
        y_pred_train = model(X_train).detach().numpy()
        y_traindt      = y_train.detach().numpy()
        
        # Get the correlation (save these)
        traincorr = np.corrcoef( y_pred_train.T[0,:], y_traindt.T[0,:])[0,1]
        testcorr  = np.corrcoef( y_pred_val.T[0,:], y_valdt.T[0,:])[0,1]
        
        # Calculate Correlation and RMSE
        corr_grid_test[i,j]    = np.corrcoef( y_pred_val.T[0,:], y_valdt.T[0,:])[0,1]
        corr_grid_train[i,j]   = np.corrcoef( y_pred_train.T[0,:], y_traindt.T[0,:])[0,1]
        
        print("\nCompleted training for learning rate %i of %i, weight decay %i of %i in %.2fs" % (i+1,np1,j+1,np2,time.time()-start))
        
# Save Data
np.savez(outpath+outname,
         train_loss_grid,
         test_loss_grid,
         corr_grid_test,
         corr_grid_train
        )
print("Saved data to %s%s. Script ran to completion in %ss"%(outpath,outname,time.time()-allstart))



# -------------
# %% Make Plots
# -------------


import matplotlib.pyplot as plt


# Plot the Correlation grid
data = corr_grid_test.copy()**2
gsize = data.shape[0]
cmap = plt.get_cmap("pink",20)
cmap.set_bad(np.array([0,255,0])/255)
fig,ax = plt.subplots(1,1,figsize=(8,8))
im = ax.imshow(data,vmin=0,vmax=1,cmap=cmap)
ax.set_title("Correlation $(R^{2})$"+"(CESM - CNN Output); Predictor = %s \n Weight Decay vs Learning Rate"%varname)
ax.set_xticks(np.arange(0,gsize))
ax.set_yticks(np.arange(0,gsize))
ax.set_xticklabels(param1)
ax.set_yticklabels(param2)
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Weight Decay")
plt.gca().invert_yaxis()
plt.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
# Loop over data dimensions and create text annotations.
for i in range(np1):
    for j in range(np2):
        # Set color to black if above threshold, white otherwise
        if data[i,j] > 0.6:
            usecolor='k'
        else:
            usecolor='w'
        
        if data[i,j] == np.nanmax(data): # Max in Red
            usecolor='r'
        elif data[i,j] == np.nanmin(data): # Min in Blue
            usecolor= np.array([0,202,231])/255
        
        text = ax.text(j, i, "%.1e"%data[i, j],
                       ha="center", va="center", color=usecolor)
        
        #text.set_path_effects([path_effects.Stroke(linewidth=0.25,foreground='k')])
plt.savefig("%sCorr_%s.png"% (outpath,expname),dpi=200)
plt.show()


# Plot the RMSE grid
data = test_loss_grid.min(2).copy()
gsize = data.shape[0]
cmap = plt.get_cmap("pink",20)
cmap.set_bad(np.array([0,255,0])/255)
fig,ax = plt.subplots(1,1,figsize=(8,8))
im = ax.imshow(data,vmin=0,vmax=1,cmap=cmap)
ax.set_title("MSE (CESM - CNN Output); Predictor %s \n Weight Decay vs Learning Rate"%varname)
ax.set_xticks(np.arange(0,gsize))
ax.set_yticks(np.arange(0,gsize))
ax.set_xticklabels(param1)
ax.set_yticklabels(param2)
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Weight Decay")
plt.gca().invert_yaxis()
plt.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
# Loop over data dimensions and create text annotations.
for i in range(np1):
    for j in range(np2):
        # Set color to black if above threshold, white otherwise
        if data[i,j] > 0.6:
            usecolor='k'
        else:
            usecolor='w'
        
        if data[i,j] == np.nanmax(data): # Max in Red
            usecolor='r'
        elif data[i,j] == np.nanmin(data): # Min in Blue
            usecolor= np.array([0,202,231])/255
        
        text = ax.text(j, i, "%.1e"%data[i, j],
                       ha="center", va="center", color=usecolor)
        
        #text.set_path_effects([path_effects.Stroke(linewidth=0.25,foreground='k')])
plt.savefig("%sMSE_%s.png"%(outpath,expname),dpi=200)
plt.show()

