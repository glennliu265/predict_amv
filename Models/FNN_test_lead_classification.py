
"""
FNN_test_lead_time_classification

Testing different lead times for a feed-forward neural network.
Copied from FNN_test_lead.py with adaptations from NN_test_lead_ann_ImageNet_classification.py

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
def train_FNN(layers,loss_fn,optimizer,trainloader,testloader,max_epochs,verbose=True):
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


def build_FNN_simple(inputsize,outsize,nlayers,nunits,activations,dropout=0.5):
    """
    Build a Feed-foward neural network with N layers, each with corresponding
    number of units indicated in nunits and activations. 
    
    A dropbout layer is included at the end
    
    inputs:
        inputsize:  INT - size of the input layer
        outputsize: INT  - size of output layer
        nlayers:    INT - number of hidden layers to include 
        nunits:     Tuple of units in each layer
        activations: Tuple of pytorch.nn activations
        --optional--
        dropout: percentage of units to dropout before last layer
        
    outputs:
        Tuple containing FNN layers
        
    dependencies:
        from pytorch import nn
        
    """
    layers = []
    for n in range(nlayers+1):
        #print(n)
        if n == 0:
            #print("First Layer")
            layers.append(nn.Linear(inputsize,nunits[n]))
            layers.append(activations[n])
            
        elif n == (nlayers):
            #print("Last Layer")
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(nunits[n-1],outsize))
            
        else:
            #print("Intermediate")
            layers.append(nn.Linear(nunits[n-1],nunits[n]))
            layers.append(activations[n])
    return layers

# -------------
#%% User Edits
# -------------

allstart=time.time()

# Indicate machine to set path
# ----------------------------
machine='stormtrack'

# Set directory and load data depending on machine
# ------------------------------------------------


# Load Old Data (Remove this)
# Note that these are 40 x 86 x 33 x 41
# outpath    = '../../CESM_data/'
# sst_normed = np.load('../../CESM_data/CESM_sst_normalized_lat_weighted_2deg_NAT_Ann.npy').astype(np.float32)
# sss_normed = np.load('../../CESM_data/CESM_sss_normalized_lat_weighted_2deg_NAT_Ann.npy').astype(np.float32)
# psl_normed = np.load('../../CESM_data/CESM_psl_normalized_lat_weighted_2deg_NAT_Ann.npy').astype(np.float32)

# Data preparation settings
# -------------------------
leads          = np.arange(0,25,1)    # Time ahead (in tunits) to forecast AMV
thresholds     = [1/3,2/3]            # Thresholds (standard deviations, or quantile values) 
quantile       = True                 # Set to True to use quantiles
nsamples       = None                 # Number of samples for each class. Set to None to use all


# Training/Testing Subsets
# ------------------------
percent_train  = 0.8   # Percentage of data to use for training (remaining for testing)
numruns        = 10    # Number of times to train for each leadtime

early_stop    = 3
max_epochs    = 20 
batch_size    = 16                    # Pairs of predictions
loss_fn       = nn.MSELoss()          # Loss Function
opt           = ['Adadelta',0.1,0]    # Name optimizer

# FNN Architecture
# ----------------
nlayers     = 2
nunits      = [20,20]
activations = [nn.ReLU(),nn.ReLU()]
outsize     = 1
netname     = "FNN2"

# Toggle Options
debug         = True # Visualize training and testing loss
verbose       = True # Print loss for each epoch
checkgpu      = True # Set to true to check for GPU otherwise run on CPU
savemodel     = True # Set to true to save model dict.

# -----------------------------------------------------------------
#%% Additional (Legacy) Variables (modify for future customization)
# -----------------------------------------------------------------

# Data Preparation names
num_classes    = len(thresholds)+1    # Set up number of classes for prediction (current supports)
season         = 'Ann'                # Season to take mean over ['Ann','DJF','MAM',...]
indexregion    = 'NAT'                # One of the following ("SPG","STG","TRO","NAT")
resolution     = '224pix'             # Resolution of dataset ('2deg','224pix')
detrend        = False                # Set to true to use detrended data
usenoise       = False                # Set to true to train the model with pure noise
tstep          = 86                   # Size of time dimension (in years)
ens            = 40                   # Ensemble members (climate model output) to use 

# -------------
#%% Load the data
# -------------
"""
Section below here is copied from NN_test_lead_ann ... .py

data   : [ 3 x 40 x 86 x 224 x 224] (variable x ens x year x lon(?) x lat(?))
target : [40 x 86] (ens x year)

"""
# Load the data for whole North Atlantic
if usenoise:
    # Make white noise time series
    data   = np.random.normal(0,1,(3,40,tstep,224,224))
    
    ## Load latitude
    #lat = np.linspace(0.4712,64.55497382,224)
    
    # Apply land mask
    dataori   = np.load('../../CESM_data/CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i.npy'%detrend)[:,:40,...]
    data[dataori==0] = 0 # change all ocean points to zero
    target = np.load('../../CESM_data/CESM_label_amv_index_detrend%i.npy'%detrend)
    
else:
    data   = np.load('../../CESM_data/CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i.npy'%detrend)
    target = np.load('../../CESM_data/CESM_label_amv_index_detrend%i.npy'%detrend)
data   = data[:,0:ens,:,:,:]
target = target[0:ens,:]



for nr in range(numruns):
    rt = time.time()
    
    # Set experiment names ---- (Will need to adjust later when testing individual variables)
    channels = 3
    start    = time.time()
    varname  = 'ALL'
    nlead    = len(leads)
    
    
    # Save data (ex: Ann2deg_NAT_CNN2_nepoch5_nens_40_lead24 )
    expname = "AMVClass%i_%s_nepoch%02i_nens%02i_maxlead%02i_detrend%i_noise%i_run%i_quant%i" % (num_classes,netname,max_epochs,ens,
                                                                              leads[-1],detrend,usenoise,
                                                                              nr,quantile)
    
    # Preallocate Evaluation Metrics...
    corr_grid_train = np.zeros((nlead))
    corr_grid_test  = np.zeros((nlead))
    train_loss_grid = []#np.zeros((max_epochs,nlead))
    test_loss_grid  = []#np.zeros((max_epochs,nlead))
    train_acc_grid  = []
    test_acc_grid   = []
    acc_by_class    = []
    total_acc       = []
    yvalpred        = []
    yvallabels      = []
    sampled_idx     = []
    thresholds_all  = []
    
    if checkgpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
        
    
    
    
    


# ----------------------------------------
# %% Experiment storage setup
# ----------------------------------------

nvar  = 4 # Combinations of variables to test
nlead = len(leads)

# Save data
expname = "%s_nepoch%02i_nens%02i_lead%02i" % (netname,max_epochs,ens,len(leads)-1)


# Preallocate Relevant Variables...
corr_grid_train = np.zeros((nlead))
corr_grid_test  = np.zeros((nlead))
train_loss_grid = np.zeros((max_epochs,nlead))
test_loss_grid  = np.zeros((max_epochs,nlead))

# ----------------------------------------------
# %% Manual Hyperparameter Testing: Testing Loop
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
    
    for l,lead in enumerate(leads):

        # Apply lead/lag to data
        y = np.mean(sst_normed[:ens,lead:,:,:],axis=(2,3)).reshape((tstep-lead)*ens,1) # Take area average for SST 
        X = np.transpose(
            np.array(invars)[:,:ens,0:tstep-lead,:,:].reshape(channels,(tstep-lead)*ens,33,89),
            (1,0,2,3))
        
        
        # Make the classes based on Y (copied from NN_test_lead..._classification.py)
        
        
        
        
        
        # Reshape to combine variable/lat/lon for X
        ndat,nchan,nlat,nlon = X.shape # Get latitude and longitude sizes for dimension calculation
        inputsize = nchan*nlat*nlon
        X = X.reshape(ndat,inputsize)
        
        # Split into training and test sets
        X_train = torch.from_numpy( X[0:int(np.floor(percent_train*(tstep-lead)*ens)),:] )
        y_train = torch.from_numpy( y[0:int(np.floor(percent_train*(tstep-lead)*ens)),:] )
        
        X_val = torch.from_numpy( X[int(np.floor(percent_train*(tstep-lead)*ens)):,:] )
        y_val = torch.from_numpy( y[int(np.floor(percent_train*(tstep-lead)*ens)):,:] )
        
        # Put into pytorch DataLoader
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
        val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
            
        # Set up FNN
        layers = build_FNN_simple(inputsize,outsize,nlayers,nunits,activations,dropout=0.5)
        
        
        # Train the model
        model,trainloss,testloss = train_FNN(layers,loss_fn,opt,train_loader,val_loader,max_epochs,verbose=False)
            
        # Save train/test loss
        train_loss_grid[:,l] = np.array(trainloss).min().squeeze() # Take minum of each epoch
        test_loss_grid[:,l]  = np.array(testloss).min().squeeze()
        
        # Evalute the model
        y_pred_val     = model(X_val).detach().numpy()
        y_valdt        = y_val.detach().numpy()
        y_pred_train   = model(X_train).detach().numpy()
        y_traindt      = y_train.detach().numpy()
            
        # Get the correlation (save these)
        traincorr = np.corrcoef( y_pred_train.T[0,:], y_traindt.T[0,:])[0,1]
        testcorr  = np.corrcoef( y_pred_val.T[0,:], y_valdt.T[0,:])[0,1]
        
        if np.isnan(traincorr) | np.isnan(testcorr):
            print("Warning, NaN Detected for %s lead %i of %i in %.2fs. Stopping!" % (varname,lead,len(leads)))
            break
        
        # Calculate Correlation and RMSE
        corr_grid_test[l]    = np.corrcoef( y_pred_val.T[0,:], y_valdt.T[0,:])[0,1]
        corr_grid_train[l]   = np.corrcoef( y_pred_train.T[0,:], y_traindt.T[0,:])[0,1]
        
        print("\nCompleted training for %s lead %i of %i in %.2fs" % (varname,lead,len(leads),time.time()-start))
    # Save Data
    np.savez(outpath+outname,**{
             'train_loss': train_loss_grid,
             'test_loss': test_loss_grid,
             'test_corr': corr_grid_test,
             'train_corr': corr_grid_train}
            )
    print("Saved data to %s%s. Script ran to completion in %ss"%(outpath,outname,time.time()-start))

        
    
    



