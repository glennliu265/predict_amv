
"""
NN Test Lead Annual

Train/test NN prediction skill at various leadtimes.  Currently supports
FNN, CNN, and ResNet.

Uses data that has been preprocessed by "output_normalized_data.ipynb"
in /Preprocessing

See user edits below for further specifications.

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
#%% User Edits (Default Settings)
# -------------
# NOTE: Edit specific hyperparameters for testing in the section below
# (look for sections marked IMPORTANT and bounded by ******************)
# Currently between lines 359-361 AND 372-377

# Data preparation settings
leads          = [24,]#np.arange(0,25,3)    # Time ahead (in years) to forecast AMV
season         = 'Ann'                # Season to take mean over ['Ann','DJF','MAM',...]
indexregion    = 'NAT'                # One of the following ("SPG","STG","TRO","NAT")
resolution     = '224pix'             # Resolution of dataset ('2deg','224pix')
detrend        = False                # Set to true to use detrended data
usenoise       = False                # Set to true to train the model with pure noise
num_classes    = 1                    # Set up number of classes for prediction

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
cnndropout    = False                  # Set to 1 to test simple CN with dropout layer

# Options
debug     = True  # Visualize training and testing loss
verbose   = False # Print loss for each epoch
checkgpu  = True  # Set to true to check for GPU otherwise run on CPU
savemodel = True  # Set to true to save model dict.
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
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=poolsizes[0]),
    
                    nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=poolsizes[1]),
    
                    nn.Flatten(),
                    nn.Linear(in_features=firstlineardim,out_features=64),
                    nn.ReLU(),
    
                    nn.Dropout(p=0.5),
                    nn.Linear(in_features=64,out_features=num_classes)
                    ]
        else: # Do not include dropout
            layers = [
                    nn.Conv2d(in_channels=channels, out_channels=nchannels[0], kernel_size=filtersizes[0]),
                    nn.ReLU(),,
                    nn.MaxPool2d(kernel_size=poolsizes[0]),
    
                    nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1]),
                    nn.ReLU(),,
                    nn.MaxPool2d(kernel_size=poolsizes[1]),
    
                    nn.Flatten(),
                    nn.Linear(in_features=firstlineardim,out_features=64),
                    nn.ReLU(),,

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
    bestloss = np.infty
    
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
            if verbose:
                print("Params to learn:")
                print("\t",name)
    
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
    train_loss,test_loss = [],[]   # Preallocate tuples to store loss
    for epoch in tqdm(range(max_epochs)): # loop by epoch
    #for epoch in range(max_epochs):
        for mode,data_loader in [('train',trainloader),('eval',testloader)]: # train/test for each epoch

            if mode == 'train':  # Training, update weights
                model.train()
            elif mode == 'eval': # Testing, freeze weights
                model.eval()

            runningloss = 0
            for i,data in enumerate(data_loader):

                # Get mini batch
                batch_x, batch_y = data
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # Set gradients to zero
                opt.zero_grad()

                # Forward pass
                pred_y = model(batch_x)

                # Calculate losslay
                loss = loss_fn(pred_y,batch_y)

                # Update weights
                if mode == 'train':
                    loss.backward() # Backward pass to calculate gradients w.r.t. loss
                    opt.step()      # Update weights using optimizer
                elif mode == 'eval':  # update scheduler after 1st epoch
                    if reduceLR:
                        scheduler.step(loss)
                    
                runningloss += float(loss.item())

            if verbose: # Print progress message
                print('{} Set: Epoch {:02d}. loss: {:3f}'.format(mode, epoch+1, \
                                                runningloss/len(data_loader)))

            # Save model if this is the best loss
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
                    return bestmodel,train_loss,test_loss

            # Clear some memory
            #print("Before clearing in epoch %i mode %s, memory is %i"%(epoch,mode,torch.cuda.memory_allocated(device)))
            del batch_x
            del batch_y
            torch.cuda.empty_cache()
            #print("After clearing in epoch %i mode %s, memory is %i"%(epoch,mode,torch.cuda.memory_allocated(device)))
    return bestmodel,train_loss,test_loss



def set_classes(labels,thresholds):
    
    return classified_labels
    
# ----------------------------------------
# %% Set-up
# ----------------------------------------
allstart = time.time()

# -------------
# Load the data
# -------------
if usenoise: # Put in input maps of gaussian noise
    # Make white noise time series
    data   = np.random.normal(0,1,(3,40,tstep,224,224))
    
    ## Load latitude
    #lat = np.linspace(0.4712,64.55497382,224)
    
    # Apply land mask
    dataori   = np.load('../../CESM_data/CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i.npy'%detrend)[:,:40,...]
    data[dataori==0] = 0 # change all ocean points to zero
    target = np.load('../../CESM_data/CESM_label_amv_index_detrend%i.npy'%detrend)
else: # Load actual data
    data   = np.load('../../CESM_data/CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i.npy'%detrend)
    target = np.load('../../CESM_data/CESM_label_amv_index_detrend%i.npy'%detrend)
data   = data[:,0:ens,:,:,:]
target = target[0:ens,:]

# *****************************************
# IMPORTANT: Set Hyperparameter values here
# *****************************************
# ex.
#     testvalues = [1e-3,1e-2,1e-1,1,2]
#     testname = "LR"
testvalues=[False]
testname='cnndropout'

for nr in range(numruns): # Initialize and train network for [numruns] iterations
    rt = time.time()
    for i in range(len(testvalues)): # Loop for each hyperparameter test value
        
        # ********************************************************************
        # IMPORTANT: Manually assign value here (will implement automatic fix later)
        # <variable_name> = testvalues[i] (ex. opt[1] = testvalues [i])
        cnndropout = testvalues[i]
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
        expname = "HPT_%s_nepoch%02i_nens%02i_maxlead%02i_detrend%i_noise%i_%s%s_run%i" % (netname,max_epochs,ens,
                                                                                  leads[-1],detrend,usenoise,
                                                                                  testname,testvalues[i],nr)
        # Preallocate Evaluation Metrics...
        corr_grid_train = np.zeros((nlead))
        corr_grid_test  = np.zeros((nlead))
        train_loss_grid = []#np.zeros((max_epochs,nlead))
        test_loss_grid  = []#np.zeros((max_epochs,nlead))
        
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
        
        yvalpred     = []
        yvallabels   = []
        for l,lead in enumerate(leads):
            if (lead == leads[-1]) and (len(leads)>1):
                outname = "/leadtime_testing_%s_%s_ALL.npz" % (varname,expname)
            else:
                outname = "/leadtime_testing_%s_%s_lead%02dof%02d.npz" % (varname,expname,lead,leads[-1])
            
            # ----------------------
            # Apply lead/lag to data
            # ----------------------
            y = target[:ens,lead:].reshape(ens*(tstep-lead),1)
            X = (data[:,:,:tstep-lead,:,:]).reshape(3,ens*(tstep-lead),224,224).transpose(1,0,2,3)
            
            # ---------------------------------
            # Split into training and test sets
            # ---------------------------------
            X_train = torch.from_numpy( X[0:int(np.floor(percent_train*(tstep-lead)*ens)),:,:,:].astype(np.float32) )
            X_val = torch.from_numpy( X[int(np.floor(percent_train*(tstep-lead)*ens)):,:,:,:].astype(np.float32) )
            y_train = torch.from_numpy(  y[0:int(np.floor(percent_train*(tstep-lead)*ens)),:].astype(np.float32)  )
            y_val = torch.from_numpy( y[int(np.floor(percent_train*(tstep-lead)*ens)):,:].astype(np.float32)  )
        
            # Put into pytorch DataLoader
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
            val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
        
            # ---------------
            # Train the model
            # ---------------
            pmodel = transfer_model(netname,num_classes,cnndropout=cnndropout,unfreeze_all=unfreeze_all)
            model,trainloss,testloss = train_ResNet(pmodel,loss_fn,opt,train_loader,val_loader,max_epochs,
                                                    early_stop=early_stop,verbose=verbose,
                                                    reduceLR=reduceLR,LRpatience=LRpatience)
            
            # Save train/test loss
            train_loss_grid.append(trainloss)
            test_loss_grid.append(testloss)
            
            #print("After train function memory is %i"%(torch.cuda.memory_allocated(device)))
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
                    #print(i)
                    # Get mini batch
                    batch_x, batch_y = vdata
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
        
                    # Make prediction and concatenate
                    batch_pred = model(batch_x).squeeze()
                    #print(batch_pred.detach().shape)
                    #print(y_pred_val.shape)
                    y_pred_val = np.concatenate([y_pred_val,batch_pred.detach().cpu().numpy()])
                    y_valdt = np.concatenate([y_valdt,batch_y.detach().cpu().numpy().squeeze()])
        
            # --------------
            # Save the model
            # --------------
            if savemodel:
                modout = "../../CESM_data/Models/%s_%s_lead%i.pt" %(expname,varname,lead)
                torch.save(model.state_dict(),modout)
        
            # Save the actual and predicted values
            yvalpred.append(y_pred_val)
            yvallabels.append(y_valdt)
        
            # Get the correlation (save these)
            testcorr  = np.corrcoef( y_pred_val.T[:], y_valdt.T[:])[0,1]
        
            # Stop if model is just predicting the same value (usually need to examine optimizer settings)
            if np.isnan(testcorr):
                print("Warning, NaN Detected for %s lead %i of %i. Stopping!" % (varname,lead,len(leads)))
                for param in model.parameters():
                    if np.any(np.isnan(param.data.cpu().numpy())):
                        print(param.data)
                if debug:
                    fig,ax=plt.subplots(1,1)
                    #plt.style.use('seaborn')
                    ax.plot(trainloss,label='train loss')
                    ax.plot(testloss,label='test loss')
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
            
        
            # Calculate Correlation and RMSE
            #if verbose:
            print("Correlation for lead %i was %f"%(lead,testcorr))
            corr_grid_test[l]    = testcorr
            
            # Visualize loss vs epoch for training/testing and correlation
            if debug:
                fig,ax=plt.subplots(1,1)
                plt.style.use('default')
                ax.plot(trainloss,label='train loss')
                ax.plot(testloss,label='test loss')
                ax.legend()
                ax.set_title("Losses for Predictor %s Leadtime %i %s"%(varname,lead,subtitle))
                ax.grid(True,linestyle="dotted")
                #plt.show()
                plt.savefig("../../CESM_data/Figures/%s_%s_leadnum%s_LossbyEpoch.png"%(expname,varname,lead))
        
                fig,ax=plt.subplots(1,1)
                #plt.style.use('seaborn')
                ax.scatter(y_pred_val,y_valdt,label="Test",marker='+',zorder=2)
                ax.legend()
                ax.set_ylim([-1.5,1.5])
                ax.set_xlim([-1.5,1.5])
                lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                        ]
                ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
                ax.legend()
                ax.set_ylabel("Actual AMV Index")
                ax.set_xlabel("Predicted AMV Index")
                ax.grid(True,linestyle="dotted")
                ax.set_title("Correlation %.2f for Predictor %s Leadtime %i %s"%(corr_grid_test[l],varname,lead,subtitle))
                #plt.show()
                plt.savefig("../../CESM_data/Figures/%s_%s_leadnum%s_ValidationScatter.png"%(expname,varname,lead))
        
            print("\nCompleted training for %s lead %i of %i" % (varname,lead,leads[-1]))
        
            # Clear some memory
            del model
            del X_val
            del y_val
            del X_train
            del y_train
            torch.cuda.empty_cache()  # Save some memory
            
            # -----------------
            # Save Eval Metrics
            # -----------------
    
            np.savez("../../CESM_data/Metrics"+outname,**{
                     'train_loss': train_loss_grid,
                     'test_loss': test_loss_grid,
                     'test_corr': corr_grid_test,
                     #'train_corr': corr_grid_train,
                     #'ytrainpred': ytrainpred,
                     #'ytrainlabels': ytrainlabels,
                     'yvalpred': yvalpred,
                     'yvallabels' : yvallabels
                     }
                    )
        print("Saved data to %s%s. Finished variable %s in %ss"%(outpath,outname,varname,time.time()-start))
    print("\nRun %i finished in %.2fs" % (nr,time.time()-rt))
print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))
