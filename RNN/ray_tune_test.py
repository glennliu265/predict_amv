#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 01:52:29 2021

Source: 

    https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
@author: gliu
"""


from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader, TensorDataset,Dataset
import timm
from tqdm import tqdm
import time

#%% 

# User Edits
detrend       = False
lead          =  0
ens           =  10
percent_train =  0.8
test_size     =  2       # Number of ensemble members to use for testing
seq_len       =  5

# Additional network selections
netname        = 'resnet50'
rnnname        = "GRU"
rnn_out        = 1
rnn_activation = False
cnn_dropout    = False

# other settings
num_workers = 2


#%% Utilities

class Combine(nn.Module):
    """
    Model that combines a feature extractor, RNN (LSTM or GRU), and linear classifier

    Inputs
    ------
        1) feature_extractor [nn.Module] - pretrained CNN with last layer unfrozen
        2) rnn [nn.Module] - RNN unit that takes feature extractor output
        3) classifier [nn.Linear] - Fully connected layer for classification
        4) activation [function[ - Activation function for final output
    """
    def __init__(self,feature_extractor,rnn,classifier,activation=False):
        super(Combine, self).__init__()
        self.cnn        = feature_extractor # Pretrained CNN (last layer unfrozen)
        self.rnn        = rnn               # RNN unit (LSTM or GRU)
        self.linear     = classifier        # Classifier Layer
        self.activation = activation        # activation funcion
    
    
    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()        # Get dimension sizes
        c_in = x.view(batch_size * timesteps, C, H, W)   # Combine batch + time
        c_out = self.cnn(c_in)                           # Extract features
        r_in = c_out.view(batch_size, timesteps, -1)     # Separate batch + time
        #r_out, (h_n, h_c) = self.rnn(r_in)
        self.rnn.flatten_parameters()                    # Suppress warning
        r_out,_ = self.rnn(r_in)                         # Pass through RNN
        r_out2 = self.linear(r_out[:, -1, :])             # Classify
        if ~self.activation:
            return r_out2
        else:
            return self.activation(r_out2)

def make_sequences(X,y,seq_len):

        """
        Prepares inputs and labels for input in RNN. Splits in sequences of
        length [sequence length] and combines the ensemble and sample dimensions

        Inputs
        ------
            1. X [ndarray: ens x time x channel x lat x lon]: Input 2d maps (predictors)
            2. y [ndarray: ens x time]: Labels for actual values

        Outputs
        -------
            1. Xseq [ndarray: ens*sample x time x channel x lat x lon]
            2. yseq [ndarray: ens*sample]

        """
        
        nens,ntime,nchan,nlat,nlon = X.shape

        nsamples= ntime- seq_len

        Xseq = np.zeros([nens,nsamples,seq_len,nchan,nlat,nlon],dtype=np.float32)
        yseq = np.zeros([nens,nsamples],dtype=np.float32)

        for i in range(ntime):
            # Find end of pattern
            end_ix = i+seq_len

            # Check if index is at end of timeseries
            if end_ix > ntime-1: # leave 1 for the label
                #print(i)
                break

            # Gather input/output:
            seqx,seqy = X[:,i:end_ix,...],y[:,end_ix]

            # Save in output
            Xseq[:,i,...] = seqx
            yseq[:,i] = seqy

        # Combine the n_samples and n_ens  dimensions
        Xseq = Xseq.reshape(nens*nsamples,seq_len,nchan,nlat,nlon)
        yseq = yseq.reshape(nens*nsamples)

        return Xseq,yseq

def transfer_model(modelname,cnn_out,cnndropout=False):
    if 'resnet' in modelname: # Load from torchvision
        model = timm.create_model(modelname,pretrained=True)
        
        # Freeze all layers except the last
        for param in model.parameters():
            param.requires_grad = False
        
        model.fc = nn.Linear(model.fc.in_features, cnn_out)                    # freeze all layers except the last one
    
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
                    nn.Linear(in_features=64,out_features=cnn_out)
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
                    nn.Linear(in_features=64,out_features=cnn_out)
                    ]
        model = nn.Sequential(*layers) # Set up model
    else: # Load from timm
        model = timm.create_model(modelname,pretrained=True)
        # Freeze all layers except the last
        for param in model.parameters():
            param.requires_grad = False
        model.classifier=nn.Linear(model.classifier.in_features,cnn_out)
    return model

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

#%% New Functions

def load_data(lead,data_dir,detrend=False,ens=1,seq_len=5,percent_train=0.8,test_size=2,test_mode=False):
    """
    Load and prepare data in the following steps:
    
    1. Load in climate model data
    2. Apply lead/lag in time
    3. Proprocess into sequences
    4. Split into train/test

    Parameters
    ----------
    lead : INT
        Lead time between the predictor/predictand
    data_dir : STR
        String of the path to the working directory
    detrend : BOOL, optional
        Use detrended AMV Index as the target. Default = False
    ens : INT, optional
        Number of ensembles (climate model output) to use (1-40). Default = 40
    seq_len : INT, optional
        Number of timesteps to include in each sample. The default is 5.
    percent_train : TYPE, optional
        Percentage of data used for training. The default is 0.8.
    test_size : INT, optional
        Number of ensemble members to withhold for testing, default is 2.
    test_mode : BOOL, optional
        Set to true to output just the testing dataset (based on testsize)

    Returns
    -------
    X_train : ARRAY [samples, sequence, channel, latitude, longitude]
        Training dataset
    y_train : ARRAY [samples, sequence, channel, latitude, longitude]
        Train labels
    X_val : ARRAY [samples]
        Validation dataset
    y_val : ARRAY [samples]
        Validation labels
    if test_mode, outputs X_test, y_test (same dimensions as above)
    """
    # ---------
    # Load data
    # ---------
    print(ens)
    #print("CWD is %s"%(os.getcwd()))
    data   = np.load('%s/../../CESM_data/CESM_data_sst_sss_psl_deseason_normalized_resized_detrend0.npy' % (data_dir))
    target = np.load('%s/../../CESM_data/CESM_label_amv_index_detrend%i.npy' % (data_dir,detrend))
    _,_,tstep,_,_ = data.shape # Get size of time axis
    
    if test_size > ens:
        print("Warning, # ens members withheld from test size (%i) is larger than ens (%i)" % (test_size,ens))
        print(ens-test_size)
        test_size=0
    
    
    #print("Data found!")
    # ----------------------------------------------------------------------
    # Apply lead/lag, transpose inputs to [ens x time x channel x lat x lon]
    # ----------------------------------------------------------------------
    if test_mode == True: # Just take last n ensemble members, where n = test_size
        y = target[-1*test_size:,lead:].astype(np.float32)
        X = (data[:,-1*test_size:,:tstep-lead,:,:].transpose(1,2,0,3,4)).astype(np.float32)
    else:
        y = target[:ens-test_size,lead:].astype(np.float32)
        X = (data[:,:ens-test_size,:tstep-lead,:,:].transpose(1,2,0,3,4)).astype(np.float32)
    
    # -------------------------
    # Preprocess into sequences
    # -------------------------
    Xseq,yseq = make_sequences(X,y,int(seq_len))
    nsamples = Xseq.shape[0]
    
    # ---------------------------------
    # Split into training and test sets
    # ---------------------------------
    if test_mode == True:
        X_test   = torch.from_numpy( Xseq.astype(np.float32))
        y_test   = torch.from_numpy( yseq[:,None].astype(np.float32))
        return X_test,y_test
    
    X_train = torch.from_numpy( Xseq[0:int(np.floor(percent_train*nsamples)),...].astype(np.float32))
    X_val = torch.from_numpy( Xseq[int(np.floor(percent_train*nsamples)):,...].astype(np.float32))

    y_train = torch.from_numpy(  yseq[0:int(np.floor(percent_train*nsamples)),None].astype(np.float32))
    y_val = torch.from_numpy( yseq[int(np.floor(percent_train*nsamples)):,None].astype(np.float32))
    
    print("X_train is size %s" % (str(X_train.shape)))
    return X_train,y_train,X_val,y_val

def setup_model(config,netname='resnet50',rnnname="GRU",
                rnn_out=1,rnn_activation=None,
                cnn_dropout=False):
    """
    
    Things to consider removing from config
    netname
    cnn_dropout
    rnnname
    rnn_activation
    rnn_out
    
    Things to definitely keep
    cnn_out
    rnn_hiddensize
    rnn_layers
    

    Parameters
    ----------
    config : DICT
        Contains tunable parameters...

    Returns
    -------
    seqmodel : NN.module

    """
    
    # Set pretrained CNN as feature extractor
    pmodel = transfer_model(netname,config['cnn_out'],cnndropout=cnn_dropout)
    
    # Set either a LTSM or GRU unit
    if rnnname == 'LSTM':
        rnn = nn.LSTM(
                input_size =config['cnn_out'],
                hidden_size=config['rnn_hiddensize'],
                num_layers =config['rnn_layers'],
                batch_first=True # Input is [batch,seq,feature]
                )
    elif rnnname == 'GRU':
        rnn = nn.GRU(
                input_size =config['cnn_out'],
                hidden_size=config['rnn_hiddensize'],
                num_layers =config['rnn_layers'],
                batch_first=True # Input is [batch,seq,feature]
                )
    
    # Set fully-connected layer for classification
    classifier = nn.Linear(config['rnn_hiddensize'],rnn_out)
    
    # Combine all into sequence model
    seqmodel = Combine(pmodel,rnn,classifier,rnn_activation)
    
    return seqmodel

def train_cesm(config, checkpoint_dir=None, data_dir=None):
    """
    Follow parameters need to potentially remove from config/need to be
    specified globally
    
    lead
    detrend
    ens
    percent_train
    criterion
    test_size
    
    Keep the following in config
    
    batch_size
    seq_len
    optimizer
    lr
    
    """
    
    # Set up the model
    st = time.time()
    net = setup_model(config,netname=netname,rnnname=rnnname,
                rnn_out=rnn_out,rnn_activation=rnn_activation,
                cnn_dropout=cnn_dropout)
    print("Model intialized in %s" % (time.time()-st))
    
    # Check for GPU
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)
    
    # Set optimizer and criterion
    criterion = nn.MSELoss()
    #opt = optim.Adam
    optimizer = optim.Adam(net.parameters(), lr=config["lr"])
    
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # Load data and set up loaders
    st = time.time()
    X_train,y_train,X_val,y_val = load_data(lead,data_dir,
                                        detrend=detrend,
                                        ens=ens,
                                        seq_len=seq_len,
                                        percent_train=percent_train,
                                        test_size=test_size
                                        )
    trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=config['batch_size'],num_workers=num_workers)
    valloader   = DataLoader(TensorDataset(X_val, y_val), batch_size=config['batch_size'],num_workers=num_workers)
    print("Loaded data in %s"%(time.time()-st))
    
    for epoch in tqdm(range(10)):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            
            loss = criterion(outputs, labels)
                             
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        predicted_value = np.asarray([])
        actual_value    = np.asarray([])
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                
                #print("inputs are of size %s" % str(inputs))
                #print("labels are of size %s" % str(labels))
                 # forward + backward + optimize
                outputs = net(inputs)
            
                # Calculate Loss
                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

                # Make prediction and concatenate for each batch
                batch_pred = outputs.squeeze()
                if i == 0:
                    predicted_value=batch_pred.detach().cpu().numpy().squeeze()
                    actual_value = labels.detach().cpu().numpy().squeeze()
                else:
                    predicted_value = np.hstack([predicted_value,batch_pred.detach().cpu().numpy().squeeze()])
                    actual_value = np.hstack([actual_value,labels.detach().cpu().numpy().squeeze()])
                
        # Calculate correlation between prediction+label
        testcorr = np.corrcoef( predicted_value[:].T, actual_value[:].T)[0,1]
        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), correlation=testcorr)
    print("Finished Training")

def test_correlation(net, config, data_dir,device="cpu"):
    X_test,y_test= load_data(lead,data_dir,
                                detrend=detrend,
                                ens=ens,
                                seq_len=seq_len,
                                percent_train=percent_train,
                                test_size=test_size,
                                test_mode=True
                                )
    
    testloader   = DataLoader(TensorDataset(X_test, y_test), batch_size=config['batch_size'],num_workers=num_workers)
    
    predicted_value = np.asarray([])
    actual_value    = np.asarray([])
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Make prediction and concatenate for each batch
            batch_pred = net(inputs).squeeze()
            if i == 0:
                predicted_value=batch_pred.detach().cpu().numpy().squeeze()
                actual_value = labels.detach().cpu().numpy().squeeze()
            else:
                predicted_value = np.hstack([predicted_value,batch_pred.detach().cpu().numpy().squeeze()])
                actual_value = np.hstack([actual_value,labels.detach().cpu().numpy().squeeze()])
                
        # Calculate correlation between prediction+label
        testcorr = np.corrcoef( predicted_value[:].T, actual_value[:].T)[0,1]

    return testcorr

#%%

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    
    # Set test hyperparameters
    config = {
        #"optimizer" : tune.grid_search([optim.Adam,optim.Adadelta]),
        "lr" : tune.grid_search([1e-4,1e-2,1e-1]),
        'batch_size' : tune.grid_search([8,16,32]),
        "cnn_out": tune.grid_search([1,100,1000]),
        "rnn_hiddensize" : tune.grid_search([10,100]),
        "rnn_layers": tune.grid_search([1,2,3])
        }
    print(config)
    
    #data_dir = os.path.abspath("./data")
    data_dir = os.getcwd()
    load_data(lead,data_dir,detrend=detrend,ens=ens,
              seq_len=seq_len,
              percent_train=percent_train,
              test_size=test_size,test_mode=False)
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "correlation", "training_iteration"])
    
    result = tune.run(
        partial(train_cesm, data_dir=data_dir),
        name = "DEFAULT_2021-01-31_20-07-40",
        resources_per_trial={"cpu": 16, "gpu": gpus_per_trial},
        #local_dir="/home/glennliu/ray_results/DEFAULT_2021-01-31_20-07-40",
        resume="PROMPT",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation correlation: {}".format(
        best_trial.last_result["correlation"]))
    
    testparams = ["lr","batch_size","cnn_out","rnn_hiddensize","rnn_layers"]
    bestconfig = {param:best_trial.config[param] for param in testparams}
    
    best_trained_model = setup_model(bestconfig,netname=netname,rnnname=rnnname,
                rnn_out=rnn_out,rnn_activation=rnn_activation,
                cnn_dropout=cnn_dropout)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)
    
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_corr = test_correlation(best_trained_model, bestconfig, data_dir, device)
    print("Best trial test set correlation: {}".format(test_corr))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=1, max_num_epochs=10, gpus_per_trial=1)