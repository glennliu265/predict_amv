#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet50 Test Lead, Annual

Train ResNet50 to forecast AMV Index at a set of lead times, given
normalized input from the CESM Large Ensemble

Can also indicate the region over which to predict the AMV Index

"""

import numpy as np

from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from torchvision import datasets, transforms as T

import os
import time
import copy
import matplotlib.pyplot as plt
 
## -------------
#%% User Edits
# -------------
allstart = time.time()

# Indicate machine to set path
machine='stormtrack'

# Set directory and load data depending on machine
if machine == 'local-glenn':
    os.chdir('/Users/gliu/Downloads/2020_Fall/6.862/Project/predict_amv/CNN/')
    outpath = '/Users/gliu/Downloads/2020_Fall/6.862/Project/'

else:
    outpath = os.getcwd()
    sst_normed = np.load('../../CESM_data/CESM_SST_normalized_lat_weighted.npy').astype(np.float32)
    
# Data preparation settings
leads          = np.arange(0,25,1)    # Time ahead (in years) to forecast AMV
resolution     = '2deg'               # Resolution of input (2deg or full)
season         = 'Ann'                # Season to take mean over
indexregion    = 'NAT'                # One of the following ("SPG","STG","TRO","NAT")

# Training/Testing Subsets
percent_train = 0.8   # Percentage of data to use for training (remaining for testing)
ens           = 1   # Ensemble members to use

# Model training settings
max_epochs    = 10
batch_size    = 32                    # Pairs of predictions
loss_fn       = nn.MSELoss()          # Loss Function
opt           = ['Adadelta',0.1,0]    # Name optimizer

# Set model architecture
netname = 'RN18'
resnet50 = models.resnet18(pretrained=True)
# model = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=(1,1),padding=(95,67)),
#                       resnet50,
#                       nn.Linear(in_features=1000,out_features=1))
#

# Options
debug= True # Visualize training and testing loss
verbose = False # Print loss for each epoch

#%% Functions
#
def train_CNN(layers,loss_fn,optimizer,trainloader,testloader,max_epochs,verbose=True):
    """
    inputs:
        layers      - tuple of NN layers
        loss_fn     - (torch.nn) loss function
        opt         - tuple of [optimizer_name, learning_rate, weight_decay] for updating the weights
                      currently supports "Adadelta" and "SGD" optimizers
        trainloader - (torch.utils.data.DataLoader) for training datasetmo
        testloader  - (torch.utils.data.DataLoader) for testing dataset
        max_epochs  - number of training epochs
        verbose     - set to True to display training messages
    
    output:
    
    dependencies:
        from torch import nn,optim
        
    """
    model = nn.Sequential(*layers) # Set up model
    bestloss = np.infty
    
    # Set optimizer
    if optimizer[0] == "Adadelta":
        opt = optim.Adadelta(model.parameters(),lr=optimizer[1],weight_decay=optimizer[2])
    elif optimizer[0] == "SGD":
        opt = optim.SGD(model.parameters(),lr=optimizer[1],weight_decay=optimizer[2])
    elif optimizer[0] == 'Adam':
        opt = optim.Adam(model.parameters(),lr=optimizer[1],weight_decay=optimizer[2])
    
    train_loss,test_loss = [],[]   # Preallocate tuples to store loss
    #for epoch in tqdm(range(max_epochs)): # loop by epoch
    for epoch in range(max_epochs):
        for mode,data_loader in [('train',trainloader),('eval',testloader)]: # train/test for each epoch
    
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
                
                # Calculate losslay
                loss = loss_fn(pred_y,batch_y.squeeze())
                
                # Update weights
                if mode == 'train':
                    loss.backward() # Backward pass to calculate gradients w.r.t. loss
                    opt.step()      # Update weights using optimizer
                    
                    
                    ## Investigate need for model.eval() in calculating train loss
                    # model.eval()
                    
                    # # Forward pass
                    # pred_y = model(batch_x).squeeze()
                    
                    # # Calculate loss
                    # loss = loss_fn(pred_y,batch_y.squeeze())
                    
                
                runningloss += loss.item()
                #print("Runningloss %.2f"%runningloss)
                
            if verbose: # Print message
                print('{} Set: Epoch {:02d}. loss: {:3f}'.format(mode, epoch+1, \
                                                runningloss/len(data_loader)))
            
            if (runningloss/len(data_loader) < bestloss) and (mode == 'eval'):
                bestloss = runningloss/len(data_loader)
                bestmodel = copy.deepcopy(model)
                if verbose:
                    print("Best Loss of %f at epoch %i"% (bestloss,epoch+1))
                
            # Save running loss values for the epoch
            if mode == 'train':
                train_loss.append(runningloss/len(data_loader))
            else:
                test_loss.append(runningloss/len(data_loader))
    return bestmodel,train_loss,test_loss     

def calc_AMV_index(region,invar,lat,lon):
    """
    Select bounding box for a given AMV region for an input variable
        "SPG" - Subpolar Gyre
        "STG" - Subtropical Gyre
        "TRO" - Tropics
        "NAT" - North Atlantic
    
    Parameters
    ----------
    region : STR
        One of following the 3-letter combinations indicating selected region
        ("SPG","STG","TRO","NAT")
        
    var : ARRAY [Ensemble x time x lat x lon]
        Input Array to select from
    lat : ARRAY
        Latitude values
    lon : ARRAY
        Longitude values    

    Returns
    -------
    amv_index [ensemble x time]
        AMV Index for a given region/variable

    """
    
    # Select AMV Index region
    bbox_SP = [-60,-15,40,65]
    bbox_ST = [-80,-10,20,40]
    bbox_TR = [-75,-15,0,20]
    bbox_NA = [-80,0 ,0,65]
    regions = ("SPG","STG","TRO","NAT")        # Region Names
    bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA) # Bounding Boxes
    
    # Get bounding box
    bbox = bboxes[regions.index(region)]
    
    # Select Region
    selvar = invar.copy()
    klon = np.where((lon>=bbox[0]) & (lon<=bbox[1]))[0]
    klat = np.where((lat>=bbox[2]) & (lat<=bbox[3]))[0]
    selvar = selvar[:,:,klat[:,None],klon[None,:]]
    
    # Take mean ove region
    amv_index = np.nanmean(selvar,(2,3))
    
    return amv_index

# ----------------------------------------
# %% Set-up
# ----------------------------------------

# Set experiment names ----
nvar  = 4 # Combinations of variables to test
nlead = len(leads)

# Save data (ex: Ann2deg_NAT_CNN2_nepoch5_nens_40_lead24 )
expname = "%s%s_%s_%s_nepoch%02i_nens%02i_lead%02i" % (season,resolution,indexregion,netname,max_epochs,ens,len(leads)-1)

# Load the data for whole North Atlantic
sst_normed = np.load('../../CESM_data/CESM_sst_normalized_lat_weighted_%s_NAT_%s.npy' % (resolution,season)).astype(np.float32)
sss_normed = np.load('../../CESM_data/CESM_sss_normalized_lat_weighted_%s_NAT_%s.npy' % (resolution,season)).astype(np.float32)
psl_normed = np.load('../../CESM_data/CESM_psl_normalized_lat_weighted_%s_NAT_%s.npy' % (resolution,season)).astype(np.float32)

# Load lat/lon
lon = np.load("../../CESM_data/lon_%s_NAT.npy"%(resolution))
lat = np.load("../../CESM_data/lat_%s_NAT.npy"%(resolution))
nens,tstep,nlat,nlon = sst_normed.shape


# Preallocate Relevant Variables...
corr_grid_train = np.zeros((nlead))
corr_grid_test  = np.zeros((nlead))
train_loss_grid = np.zeros((max_epochs,nlead))
test_loss_grid  = np.zeros((max_epochs,nlead))


# ----------------------------------------------
# %% Train for each variable combination and lead time
# ----------------------------------------------

channels = 1
for v in range(nvar): # Loop for each variable
    start = time.time()
    if v == 0:
        varname = 'SST'
        invars = [sst_normed]
    elif v == 1:
        varname = 'SSS'
        invars = [sss_normed]
    elif v == 2:
        varname = 'PSL'
        invars = [psl_normed]
    elif v == 3:
        channels = 3
        varname = 'ALL'
        invars = [sst_normed,sss_normed,psl_normed]
    
    outname = "/leadtime_testing_%s_%s.npz" % (varname,expname)
    
    
    # Begin Loop
    for l,lead in enumerate(leads):
        start = time.time()
        
        # Apply lead/lag to data
        y = calc_AMV_index(indexregion,sst_normed[:ens,lead:,:,:],lat,lon)
        y = y.reshape((y.shape[0]*y.shape[1]))[:,None]
        X = np.transpose(np.array(invars)[:,:ens,0:tstep-lead,:,:].reshape(channels,(tstep-lead)*ens,nlat,nlon),
            (1,0,2,3))
                
        
        # Split into training and test sets
        X_train = torch.from_numpy( X[0:int(np.floor(percent_train*(tstep-lead)*ens)),:,:,:] )
        y_train = torch.from_numpy( y[0:int(np.floor(percent_train*(tstep-lead)*ens)),:] )
            
        X_val = torch.from_numpy( X[int(np.floor(percent_train*(tstep-lead)*ens)):,:,:,:] )
        y_val = torch.from_numpy( y[int(np.floor(percent_train*(tstep-lead)*ens)):,:] )
        
        # Put into pytorch DataLoader
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
        val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
                
        # Load resnet
        resnet50 = models.resnet50(pretrained=True)
        
        layers = [nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=(1,1),padding=(95,67)),
                          resnet50,
                          nn.Linear(in_features=1000,out_features=1)]
        
        # Train CNN
        model,trainloss,testloss = train_CNN(layers,loss_fn,opt,train_loader,val_loader,max_epochs,verbose=verbose)
        
        # Save train/test loss
        train_loss_grid = np.array(trainloss)# Take minum of each epoch
        test_loss_grid  = np.array(testloss)
        
        # Evalute the model
        model.eval()
        y_pred_val     = model(X_val).detach().numpy()
        y_valdt        = y_val.detach().numpy()
        y_pred_train   = model(X_train).detach().numpy()
        y_traindt      = y_train.detach().numpy()
            
        # Get the correlation (save these)
        traincorr = np.corrcoef( y_pred_train.T[0,:], y_traindt.T[0,:])[0,1]
        testcorr  = np.corrcoef( y_pred_val.T[0,:], y_valdt.T[0,:])[0,1]
        
        if np.isnan(traincorr) | np.isnan(testcorr):
            if debug:
                fig,ax=plt.subplots(1,1)
                plt.style.use('seaborn')
                ax.plot(trainloss[1:],label='train loss')
                ax.plot(testloss[1:],label='test loss')
                ax.legend()
                ax.set_title("Losses for Predictor %s Leadtime %i"%(varname,lead))
                plt.show()
                
                
                fig,ax=plt.subplots(1,1)
                plt.style.use('seaborn')
                #ax.plot(y_pred_train,label='train corr')
                ax.plot(y_pred_val,label='test corr')
                ax.plot(y_valdt,label='truth')
                ax.legend()
                ax.set_title("Correlation for Predictor %s Leadtime %i"%(varname,lead))
                plt.show()
            print("Warning, NaN Detected for lead %i of %i. Stopping!" % (lead,len(leads)))
            break
        
        # Calculate Correlation and RMSE
        corr_grid_test    = np.corrcoef( y_pred_val.T[0,:], y_valdt.T[0,:])[0,1]
        corr_grid_train   = np.corrcoef( y_pred_train.T[0,:], y_traindt.T[0,:])[0,1]
        
        # Save the model
        modout = "%s%s_lead%i.pt" %(outpath,expname,lead)
        torch.save(model.state_dict(),modout)
        
        # Save Data
        outname = "/leadtime_testing_%s_%s_lead%02i.npz" % (varname,expname,lead)
        np.savez(outpath+outname,**{
                 'train_loss': train_loss_grid,
                 'test_loss': test_loss_grid,
                 'test_corr': corr_grid_test,
                 'train_corr': corr_grid_train}
                )
        
        if debug:
            fig,ax=plt.subplots(1,1)
            plt.style.use('seaborn')
            ax.plot(trainloss[1:],label='train loss')
            ax.plot(testloss[1:],label='test loss')
            ax.legend()
            ax.set_title("Losses for Predictor %s Leadtime %i"%(varname,lead))
            plt.show()
            
            
            fig,ax=plt.subplots(1,1)
            plt.style.use('seaborn')
            #ax.plot(y_pred_train,label='train corr')
            ax.plot(y_pred_val,label='test corr')
            ax.plot(y_valdt,label='truth')
            ax.legend()
            ax.set_title("Correlation for Predictor %s Leadtime %i"%(varname,lead))
            plt.show()
        print("\nCompleted training for lead %i of %i in %.2fs" % (lead,len(leads),time.time()-start))

print("Saved data to %s%s. Script ran to completion in %ss"%(outpath,outname,time.time()-start))