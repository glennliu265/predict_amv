#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save Ensemble Predictions

See user edits below for further specifications.

"""
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
#import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset,Dataset
import os
import copy
import timm

# -------------
#%% User Edits
# -------------

# Data preparation settings
leads          = np.arange(0,25,1)    # Time ahead (in years) to forecast AMV
season         = 'Ann'                # Season to take mean over ['Ann','DJF','MAM',...]
indexregion    = 'NAT'                # One of the following ("SPG","STG","TRO","NAT")
resolution     = '224pix'             # Resolution of dataset ('2deg','224pix')
detrend        = False                # Set to true to use detrended data
usenoise       = False                # Set to true to train the model with pure noise
thresholds     = [-1,1]               # Thresholds (standard deviations, determines number of classes) 
num_classes    = len(thresholds)+1    # Set up number of classes for prediction (current supports)
nsamples       = 300                  # Number of samples for each class
runids         = [0,2,3,4,5,6,7,8,9] # Runids to process


# Training/Testing Subsets
percent_train = 0.8   # Percentage of data to use for training (remaining for testing)
ens           = 40   # Ensemble members to use
tstep         = 86    # Size of time dimension (in years)
numruns       = 10    # Number of times to train each run

# Model training settings
unfreeze_all  = True               # Set to true to unfreeze all layers, false to only unfreeze last layer
early_stop    = 3                  # Number of epochs where validation loss increases before stopping
max_epochs    = 20                  # Maximum number of epochs
batch_size    = 16                   # Pairs of predictions
loss_fn       = nn.CrossEntropyLoss() # Loss Function
#max_fn       = nn.LogSoftmax(dim=1)
opt           = ['Adam',1e-3,0]       # Name optimizer
reduceLR      = False                 # Set to true to use LR scheduler
LRpatience    = 3                     # Set patience for LR scheduler
netname       = 'resnet50'           #'simplecnn'           # Name of network ('resnet50','simplecnn')
tstep         = 86
outpath       = ''
cnndropout    = True                  # Set to 1 to test simple CN with dropout layer

# Options
debug         = True # Visualize training and testing loss
verbose       = True # Print loss for each epoch
checkgpu      = True # Set to true to check for GPU otherwise run on CPU
savemodel     = True # Set to true to save model dict.

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

def transfer_model(modelname,num_classes,cnndropout=False,unfreeze_all=False):
    """
    Load pretrained weights and architectures based on [modelname]
    
    Parameters
    ----------
    modelname : STR
        Name of model (currently supports 'simplecnn',or any resnet/efficientnet from timms)
    num_classes : INT
        Dimensions of output (ex. number of classes)
    cnndropout : BOOL, optional
        Include dropout layer in simplecnn. The default is False.
    unfreeze_all : BOOL, optional
        Set to True to unfreeze all weights in the model. Otherwise, just
        the last layer is unfrozen. The default is False.
    
    Returns
    -------
    model : PyTorch Model
        Returns loaded Pytorch model
    """
    if 'resnet' in modelname: # Load ResNet
        model = timm.create_model(modelname,pretrained=True)
        if unfreeze_all is False: # Freeze all layers except the last
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes) # Set last layer size
        
    elif modelname == 'simplecnn': # Use Simple CNN from previous testing framework
        # 2 layer CNN settings
        channels = 3
        nlat = 224
        nlon = 224
        nchannels     = [32,64]
        filtersizes   = [[2,3],[3,3]]
        filterstrides = [[1,1],[1,1]]
        poolsizes     = [[2,3],[2,3]]
        poolstrides   = [[2,3],[2,3]]
        firstlineardim = calc_layerdims(nlat,nlon,filtersizes,filterstrides,poolsizes,poolstrides,nchannels)
        if cnndropout: # Include Dropout
            layers = [
                    nn.Conv2d(in_channels=channels, out_channels=nchannels[0], kernel_size=filtersizes[0]),
                    nn.Tanh(),
                    #nn.ReLU(),
                    #nn.Sigmoid(),
                    nn.MaxPool2d(kernel_size=poolsizes[0]),
    
                    nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1]),
                    nn.Tanh(),
                    #nn.ReLU(),
                    #nn.Sigmoid(),
                    nn.MaxPool2d(kernel_size=poolsizes[1]),
    
                    nn.Flatten(),
                    nn.Linear(in_features=firstlineardim,out_features=64),
                    nn.Tanh(),
                    #nn.ReLU(),
                    #nn.Sigmoid(),
    
                    nn.Dropout(p=0.5),
                    nn.Linear(in_features=64,out_features=num_classes)
                    ]
        else: # Do not include dropout
            layers = [
                    nn.Conv2d(in_channels=channels, out_channels=nchannels[0], kernel_size=filtersizes[0]),
                    nn.Tanh(),
                    #nn.ReLU(),
                    #nn.Sigmoid(),
                    nn.MaxPool2d(kernel_size=poolsizes[0]),
    
                    nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1]),
                    nn.Tanh(),
                    #nn.ReLU(),
                    #nn.Sigmoid(),
                    nn.MaxPool2d(kernel_size=poolsizes[1]),
    
                    nn.Flatten(),
                    nn.Linear(in_features=firstlineardim,out_features=64),
                    nn.Tanh(),
                    #nn.ReLU(),
                    #nn.Sigmoid(),

                    nn.Linear(in_features=64,out_features=num_classes)
                    ]
        model = nn.Sequential(*layers) # Set up model
    else: # Load Efficientnet from Timmm
        model = timm.create_model(modelname,pretrained=True)
        if unfreeze_all is False: # Freeze all layers except the last
            for param in model.parameters():
                param.requires_grad = False
        model.classifier=nn.Linear(model.classifier.in_features,num_classes)
    return model

def train_ResNet(model,loss_fn,optimizer,trainloader,testloader,max_epochs,early_stop=False,verbose=True,
                 reduceLR=False,LRpatience=3):
    """
    inputs:
        model       - Resnet model
        loss_fn     - (torch.nn) loss function
        opt         - tuple of [optimizer_name, learning_rate, weight_decay] for updating the weights
                      currently supports "Adadelta" and "SGD" optimizers
        trainloader - (torch.utils.data.DataLoader) for training datasetmo
        testloader  - (torch.utils.data.DataLoader) for testing dataset
        max_epochs  - number of training epochs
        early_stop  - BOOL or INT, Stop training after N epochs of increasing validation error
                     (set to False to stop at max epoch, or INT for number of epochs)
        verbose     - set to True to display training messages
        reduceLR    - BOOL, set to true to use LR scheduler
        LRpatience  - INT, patience for LR scheduler

    output:

    dependencies:
        from torch import nn,optim

    """
    # Check if there is GPU
    if checkgpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    # Get list of params to update
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            # if verbose:
            #     print("Params to learn:")
            #     print("\t",name)

    # Set optimizer
    if optimizer[0] == "Adadelta":
        opt = optim.Adadelta(model.parameters(),lr=optimizer[1],weight_decay=optimizer[2])
    elif optimizer[0] == "SGD":
        opt = optim.SGD(model.parameters(),lr=optimizer[1],weight_decay=optimizer[2])
    elif optimizer[0] == 'Adam':
        opt = optim.Adam(model.parameters(),lr=optimizer[1],weight_decay=optimizer[2])

    # Add Scheduler
    if reduceLR:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=LRpatience)
    
    # Set early stopping threshold and counter
    if early_stop is False:
        i_thres = max_epochs
    else:
        i_thres = early_stop
    i_incr    = 0 # Number of epochs for which the validation loss increases
    bestloss  = np.infty

    # Main Loop
    train_acc,test_acc = [],[] # Preallocate tuples to store accuracy
    train_loss,test_loss = [],[]   # Preallocate tuples to store loss
    bestloss = np.infty
    
    for epoch in tqdm(range(max_epochs)): # loop by epoch
        for mode,data_loader in [('train',trainloader),('eval',testloader)]: # train/test for each epoch
            if mode == 'train':  # Training, update weights
                model.train()
            elif mode == 'eval': # Testing, freeze weights
                model.eval()

            runningloss = 0
            correct     = 0
            total       = 0
            for i,data in enumerate(data_loader):
                # Get mini batch
                batch_x, batch_y = data
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                # Set gradients to zero
                opt.zero_grad()
                
                # Forward pass
                pred_y = model(batch_x)
                
                # Calculate loss
                loss = loss_fn(pred_y,batch_y[:,0])
                
                # Track accuracy
                _,predicted = torch.max(pred_y.data,1)
                total   += batch_y.size(0)
                correct += (predicted == batch_y[:,0]).sum().item()
                #print("Total is now %.2f, Correct is now %.2f" % (total,correct))
                
                # Update weights
                if mode == 'train':
                    loss.backward() # Backward pass to calculate gradients w.r.t. loss
                    opt.step()      # Update weights using optimizer
                elif mode == 'eval':  # update scheduler after 1st epoch
                    if reduceLR:
                        scheduler.step(loss)
                    
                runningloss += float(loss.item())
            
            if verbose: # Print progress message
                print('{} Set: Epoch {:02d}. loss: {:3f}. acc: {:.3f}%'.format(mode, epoch+1, \
                                                runningloss/len(data_loader),correct/total*100))

            # Save model if this is the best loss
            if (runningloss/len(data_loader) < bestloss) and (mode == 'eval'):
                bestloss = runningloss/len(data_loader)
                bestmodel = copy.deepcopy(model)
                if verbose:
                    print("Best Loss of %f at epoch %i"% (bestloss,epoch+1))

            # Save running loss values for the epoch
            if mode == 'train':
                train_loss.append(runningloss/len(data_loader))
                train_acc.append(correct/total)
            else:
                test_loss.append(runningloss/len(data_loader))
                test_acc.append(correct/total)

                # Evaluate if early stopping is needed
                if epoch == 0: # Save previous loss
                    lossprev = runningloss/len(data_loader)
                else: # Add to counter if validation loss increases
                    if runningloss/len(data_loader) > lossprev:
                        i_incr += 1 # Add to counter
                        if verbose:
                            print("Validation loss has increased at epoch %i, count=%i"%(epoch+1,i_incr))
                        
                    else:
                        i_incr = 0 # Zero out counter
                    lossprev = runningloss/len(data_loader)

                if (epoch != 0) and (i_incr >= i_thres):
                    print("\tEarly stop at epoch %i "% (epoch+1))
                    return bestmodel,train_loss,test_loss,train_acc,test_acc

            # Clear some memory
            #print("Before clearing in epoch %i mode %s, memory is %i"%(epoch,mode,torch.cuda.memory_allocated(device)))
            del batch_x
            del batch_y
            torch.cuda.empty_cache()
            #print("After clearing in epoch %i mode %s, memory is %i"%(epoch,mode,torch.cuda.memory_allocated(device)))

    #bestmodel.load_state_dict(best_model_wts)
    return bestmodel,train_loss,test_loss,train_acc,test_acc

def make_classes(y,thresholds,exact_value=False,reverse=False):
    """
    Makes classes based on given thresholds. 

    Parameters
    ----------
    y : ARRAY
        Labels to classify
    thresholds : ARRAY
        1D Array of thresholds to partition the data
    exact_value: BOOL, optional
        Set to True to use the exact value in thresholds (rather than scaling by
                                                          standard deviation)

    Returns
    -------
    y_class : ARRAY [samples,class]
        Classified samples, where the second dimension contains an integer
        representing each threshold

    """
    nthres = len(thresholds)
    if ~exact_value: # Scale thresholds by standard deviation
        y_std = np.std(y) # Get standard deviation
        thresholds = np.array(thresholds) * y_std
    y_class = np.zeros((y.shape[0],1))
    
    if nthres == 1: # For single threshold cases
        thres = thresholds[0]
        y_class[y<=thres] = 0
        y_class[y>thres] = 1
        
        print("Class 0 Threshold is y <= %.2f " % (thres))
        print("Class 0 Threshold is y > %.2f " % (thres))
        return y_class
    
    for t in range(nthres+1):
        if t < nthres:
            thres = thresholds[t]
        else:
            thres = thresholds[-1]
        
        if reverse: # Assign class 0 to largest values
            tassign = nthres-t
        else:
            tassign = t
        
        if t == 0: # First threshold
            y_class[y<=thres] = tassign
            print("Class %i Threshold is y <= %.2f " % (tassign,thres))
        elif t == nthres: # Last threshold
            y_class[y>thres] = tassign
            print("Class %i Threshold is y > %.2f " % (tassign,thres))
        else: # Intermediate values
            thres0 = thresholds[t-1]
            y_class[(y>thres0) * (y<=thres)] = tassign
            print("Class %i Threshold is %.2f < y <= %.2f " % (tassign,thres0,thres))
    return y_class


def select_samples(nsamples,y_class,X):
    """
    Sample even amounts from each class

    Parameters
    ----------
    nsample : INT
        Number of samples to get from each class
    y_class : ARRAY [samples x 1]
        Labels for each sample
    X : ARRAY [samples x channels x height x width]
        Input data for each sample
    
    Returns
    -------
    
    y_class_sel : ARRAY [samples x 1]
        Subsample of labels with equal amounts for each class
    X_sel : ARRAY [samples x channels x height x width]
        Subsample of inputs with equal amounts for each class
    idx_sel : ARRAY [samples x 1]
        Indices of selected arrays
    
    """
    
    allsamples,nchannels,H,W = X.shape
    classes    = np.unique(y_class)
    nclasses   = len(classes)
    

    # Sort input by classes
    label_by_class  = []
    input_by_class  = []
    idx_by_class    = []
    
    y_class_sel = np.zeros([nsamples*nclasses,1])#[]
    X_sel       = np.zeros([nsamples*nclasses,nchannels,H,W])#[]
    idx_sel     = np.zeros([nsamples*nclasses]) 
    for i in range(nclasses):
        
        # Sort by Class
        inclass = classes[i]
        idx = (y_class==inclass).squeeze()
        sel_label = y_class[idx,:]
        sel_input = X[idx,:,:,:]
        sel_idx = np.where(idx)[0]
        
        label_by_class.append(sel_label)
        input_by_class.append(sel_input)
        idx_by_class.append(sel_idx)
        classcount = sel_input.shape[0]
        print("%i samples found for class %i" % (classcount,inclass))
        
        # Shuffle and select first nsamples
        shuffidx = np.arange(0,classcount,1)
        np.random.shuffle(shuffidx)
        shuffidx = shuffidx[0:nsamples]
        
        # Select Shuffled Indices
        y_class_sel[i*nsamples:(i+1)*nsamples,:] = sel_label[shuffidx,:]
        X_sel[i*nsamples:(i+1)*nsamples,...]     = sel_input[shuffidx,...]
        idx_sel[i*nsamples:(i+1)*nsamples]       = sel_idx[shuffidx]
    
    # Shuffle samples again before output (so they arent organized by class)
    shuffidx = np.arange(0,nsamples*nclasses,1)
    np.random.shuffle(shuffidx)
    
    return y_class_sel[shuffidx,...],X_sel[shuffidx,...],idx_sel[shuffidx,...]

# ----------------------------------------
# %% Set-up
# ----------------------------------------
allstart = time.time()

# Load the data
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
    
    #data[dataori==0] = np.nan
    #target = np.nanmean(((np.cos(np.pi*lat/180))[None,None,:,None] * data[0,:,:,:,:]),(2,3)) 
    #data[np.isnan(data)] = 0
else:
    data   = np.load('../../CESM_data/CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i.npy'%detrend)
    target = np.load('../../CESM_data/CESM_label_amv_index_detrend%i.npy'%detrend)
data   = data[:,0:ens,:,:,:]
target = target[0:ens,:]
    
#testvalues = [1e-3,1e-2,1e-1,1,2]
#testname = "LR"

#testvalues = [False]
#testname   = "cnndropout" # Note need to manually locate variable and edit
testvalues=[True]
testname='unfreeze_all'

for n in range(len(runids)):
    rt = time.time()
    
    nr = runids[n]
    
    for i in range(len(testvalues)):
        
        # ********************************************************************
        # NOTE: Manually assign value here (will implement automatic fix later)
        unfreeze_all = testvalues[i]
        
        print("Testing %s=%s"% (testname,str(testvalues[i])))
        # ********************************************************************
        
        # Set experiment names ----
        nlead    = len(leads)
        channels = 3
        start    = time.time()
        varname  = 'ALL'
        #subtitle = "\n %s = %i; detrend = %s"% (testname,testvalues[i],detrend)
        subtitle="\n%s=%s" % (testname, str(testvalues[i]))
        
        # Save data (ex: Ann2deg_NAT_CNN2_nepoch5_nens_40_lead24 )
        expname = "AMVClass%i_%s_nepoch%02i_nens%02i_maxlead%02i_detrend%i_noise%i_%s%s_run%i_unfreezeall" % (num_classes,netname,max_epochs,ens,
                                                                                  leads[-1],detrend,usenoise,
                                                                                  testname,testvalues[i],nr)
        # Preallocate Evaluation Metrics...
        y_pred_prob = []
        y_label     = []

        if checkgpu:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        
        # -------------
        # Print Message
        # -------------
        print("Running CNN_test_lead_ann.py with the following settings:")
        print("\tNetwork Type   : "+netname)
        print("\tLeadtimes      : %i to %i" % (leads[0],leads[-1]))
        print("\tMax Epochs     : " + str(max_epochs))
        print("\tEarly Stop     : " + str(early_stop))
        print("\t# Ens. Members : "+ str(ens))
        print("\t%" +testname +  " : "+ str(testvalues[i]))
        print("\tDetrend        : "+ str(detrend))
        print("\tUse Noise      :" + str(usenoise))
        
        for l,lead in enumerate(leads):
            if (lead == leads[-1]) and (len(leads)>1):
                outname = "/leadtime_testing_%s_%s_ALL.npz" % (varname,expname)
            else:
                outname = "/leadtime_testiang_%s_%s_lead%02dof%02d.npz" % (varname,expname,lead,leads[-1])
            
            # ----------------------
            # Apply lead/lag to data
            # ----------------------
            y = target[:ens,lead:].reshape(ens*(tstep-lead),1)
            X = (data[:,:ens,:tstep-lead,:,:]).reshape(3,ens*(tstep-lead),224,224).transpose(1,0,2,3)
            y_class = make_classes(y,thresholds,reverse=True)
            
            
            # ---------------------
            # Load shuffled indices
            # ---------------------
            ld = np.load("../../CESM_data/Metrics/leadtime_testing_%s_%s_ALL.npz" % (varname,expname))['sampled_idx']
            shuffidx = ld[l,:].astype('int')
            y_class = y_class[shuffidx,:]
            X = X[shuffidx,:]
            lead_nsamples      = y_class.shape[0]
            
            # ---------------------------------
            # Split into training and test sets
            # ---------------------------------
            X_val   = torch.from_numpy( X[int(np.floor(percent_train*lead_nsamples)):,:,:,:].astype(np.float32) )
            y_val   = torch.from_numpy( y_class[int(np.floor(percent_train*lead_nsamples)):,:].astype(np.long)  )
            
            # Put into pytorch DataLoader
            val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
            
            # --------------------------
            # Load Trained Model Weights
            # --------------------------
            model = transfer_model(netname,num_classes,cnndropout=cnndropout,unfreeze_all=unfreeze_all)
            MPATH  = "../../CESM_data/Models/%s_%s_lead%i_classify.pt" %(expname,varname,lead)
            model.load_state_dict(torch.load(MPATH,map_location=device))
            
            
            # #print("After train function memory is %i"%(torch.cuda.memory_allocated(device)))
            # -----------------------------------------------
            # Pass to GPU or CPU for evaluation of best model
            # -----------------------------------------------
            with torch.no_grad():
                X_val = X_val.to(device)
                model.eval()
                
                # -----------------
                # Evalute the model
                # -----------------
                y_pred_val = np.asarray([])
                y_valdt    = np.asarray([])

                for i,vdata in enumerate(val_loader):
                    #break
                    #print(i)
                    # Get mini batch
                    batch_x, batch_y = vdata
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    # Make prediction and concatenate
                    batch_pred = model(batch_x) # [nsample, class]
                    
                    # Convert predicted values
                    y_batch_pred = batch_pred.detach().cpu().numpy()
                    y_batch_lab  = batch_y.detach().cpu().numpy().squeeze()
                    
                    # Store Predictions
                    if i == 0:
                        y_pred_val = y_batch_pred
                    else:
                        y_pred_val = np.concatenate([y_pred_val,y_batch_pred])
                    y_valdt = np.concatenate([y_valdt,y_batch_lab])
            
            # Save the actual and predicted values
            y_pred_prob.append(y_pred_val) # [lead][nsample,class]
            y_label.append(y_valdt) # [lead][nsample]
            
            # Clear some memory
            del model
            del X_val
            del y_val
            torch.cuda.empty_cache()  # Save some memory
            
            # -----------------
            # Save Eval Metrics
            # -----------------
            np.savez("../../CESM_data/Metrics/Validation_Probabilities_"+outname,**{
                     'y_pred_prob': y_pred_prob,
                     'y_label': y_label
                     }
                     )
        print("Saved data to %s%s. Finished variable %s in %ss"%(outpath,outname,varname,time.time()-start))
    print("\nRun %i finished in %.2fs" % (nr,time.time()-rt))
print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))
