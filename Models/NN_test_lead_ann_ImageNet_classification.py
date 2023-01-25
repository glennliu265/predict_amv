
"""
NN Test Lead Annual (Classification Version)

Train/test NN prediction skill at various leadtimes.  Currently supports a
2-layer CNN, and ResNet (Transfer Learning and Fully-Retrained).

Uses data that has been preprocessed by "output_normalized_data.ipynb"
in /Preprocessing
    Assumes data is stored in ../../CESM_data/

Outputs are stored in 
    - ../../CESM_data/Metrics (Experimental Metrics (ex. Acc))
    - ../../CESM_data/Models (Model Dict/Weights)
    - ../../CESM_data/Figures (Loss Visualizations)

See user edits below for further specifications.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,Dataset
import os
import copy
import timm

# -------------
#%% User Edits
# -------------

# Data preparation settings
leads          = [1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23] #np.arange(0,25,3)    # Time ahead (in years) to forecast AMV
thresholds     = [-1,1]#[1/3,2/3]   # Thresholds (standard deviations, or quantile values) 
quantile       = False                # Set to True to use quantiles
nsamples       = 300                 # Number of samples for each class. Set to None to use all
usefakedata    = None#"fakedata_1Neg1Pos1Random_3box.nc"# Set to None, or name of fake dataset.

# Training/Testing Subsets
percent_train  = 0.8              # Percentage of data to use for training (remaining for testing)
runids         = np.arange(0,51,1) # Which runs to do

#numruns        = 10    # Number of times to train for each leadtime

# Model training settings
netname       = 'FNN4_128'           # Name of network ('resnet50','simplecnn','FNN2')
unfreeze_all  = True                 # Set to true to unfreeze all layers, false to only unfreeze last layer
use_softmax   = False                 # Set to true to end on softmax layer

# Additional Hyperparameters (CNN)
early_stop    = 3                    # Number of epochs where validation loss increases before stopping
max_epochs    = 20                   # Maximum number of epochs
batch_size    = 16                   # Pairs of predictions
loss_fn       = nn.CrossEntropyLoss()# Loss Function (nn.CrossEntropyLoss())
opt           = ['Adam',1e-3,0]      # [Optimizer Name, Learning Rate, Weight Decay]
reduceLR      = False                # Set to true to use LR scheduler
LRpatience    = 3                    # Set patience for LR scheduler
cnndropout    = True                 # Set to 1 to test simple CNN with dropout layer
fnndropout    = 0.5                  # 0.5

# Hyperparameters (FNN)
# ----------------
nlayers     = 4
nunits      = [128,128,128,128]
activations = [nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.ReLU()]
#netname     = "FNN2"

# Toggle Options
# --------------
debug         = False # Visualize training and testing loss
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
resolution     = '1deg'             # Resolution of dataset ('2deg','224pix')
regrid         = None
detrend        = False                # Set to true to use detrended data
usenoise       = False                # Set to true to train the model with pure noise
tstep          = 86                   # Size of time dimension (in years)
ens            = 40                   # Ensemble members (climate model output) to use
outpath        = ""
numruns        = len(runids)
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

def transfer_model(modelname,num_classes,cnndropout=False,unfreeze_all=False
                   ,nlat=224,nlon=224,nchannels=3):
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

def make_classes(y,thresholds,exact_value=False,reverse=False,
                 quantiles=False):
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
    
    if quantiles is False:
        if ~exact_value: # Scale thresholds by standard deviation
            y_std = np.std(y) # Get standard deviation
            thresholds = np.array(thresholds) * y_std
    else: # Determine Thresholds from quantiles
        thresholds = np.quantile(y,thresholds,axis=0) # Replace Thresholds with quantiles
    
    nthres  = len(thresholds)
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
    if quantiles is True:
        return y_class,thresholds
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

def build_FNN_simple(inputsize,outsize,nlayers,nunits,activations,dropout=0.5,
                     use_softmax=False):
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
        use_softmax : BOOL, True to end with softmax layer
        
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
            if use_softmax:
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.Linear(nunits[n-1],outsize))
                layers.append(nn.Softmax(dim=0))
            else:
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.Linear(nunits[n-1],outsize))
            
        else:
            #print("Intermediate")
            layers.append(nn.Linear(nunits[n-1],nunits[n]))
            layers.append(activations[n])
    return layers

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
    
    data   = np.load('../../CESM_data/CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i_regrid%s.npy'% (detrend,regrid))
    target = np.load('../../CESM_data/CESM_label_amv_index_detrend%i_regrid%s.npy'% (detrend,regrid))
data   = data[:,0:ens,:,:,:]
target = target[0:ens,:]
    
#testvalues = [1e-3,1e-2,1e-1,1,2]
#testname = "LR"

#testvalues = [False]
#testname   = "cnndropout" # Note need to manually locate variable and edit
testvalues=[True]
testname='unfreeze_all'

for nr,runid in enumerate(runids):
    rt = time.time()
    
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
        expname = "AMVClass%i_%s_nepoch%02i_nens%02i_maxlead%02i_detrend%i_noise%i_%s%s_run%i_unfreezeall_quant%i_res%s" % (num_classes,netname,max_epochs,ens,
                                                                                  leads[-1],detrend,usenoise,
                                                                                  testname,testvalues[i],runid,quantile,regrid)
        if use_softmax:
            expname += "_softmax"
        
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
            if (lead == leads[-1]) and (len(leads)>1): # Output all files together
                outname = "/leadtime_testing_%s_%s_ALL.npz" % (varname,expname)
            else: # Output individual lead times while training
                outname = "/leadtime_testing_%s_%s_lead%02dof%02d.npz" % (varname,expname,lead,leads[-1])
            
            # ----------------------
            # Apply lead/lag to data
            # ----------------------
            if (i == 0) and (nr ==0):
                thresholds_old = thresholds.copy() # Copy Original Thresholds (Hack Fix)
            thresholds = thresholds_old.copy()
            nchannels,nens,ntime,nlat,nlon=data.shape
            y = target[:ens,lead:].reshape(ens*(tstep-lead),1)
            X = (data[:,:ens,:tstep-lead,:,:]).reshape(3,ens*(tstep-lead),nlat,nlon).transpose(1,0,2,3)
            y_class = make_classes(y,thresholds,reverse=True,quantiles=quantile)
            
            if quantile == True:
                thresholds = y_class[1].T[0]
                y_class   = y_class[0]
            thresholds_all.append(thresholds) # Save Thresholds
            
            if (nsamples is None) or (quantile is True):
                nthres = len(thresholds) + 1
                threscount = np.zeros(nthres)
                for t in range(nthres):
                    threscount[t] = len(np.where(y_class==t)[0])
                nsamples = int(np.min(threscount))
            
            y_class,X,shuffidx = select_samples(nsamples,y_class,X)
            lead_nsamples      = y_class.shape[0]
            sampled_idx.append(shuffidx) # Save the sample indices
            

            
            # Visualize plot of variables that were selected
            
            ## Save shuffled data
            # np.save("y_class_lead0_nsample500.npy",y_class)
            # np.save("X_lead0_nsample500.npy",X)
            # np.save("shuffidx_lead0_nsample500.npy",shuffidx)
            
            # # save 1 sample
            # xsample = X[[0],:,:,:]
            # ysample = y_class[[0],:]
            # np.save("X.npy",xsample)
            # np.save("y.npy",ysample)
            
            # --------------------------
            # Flatten input data for FNN
            # --------------------------
            if "FNN" in netname:
                ndat,nchan,nlat,nlon = X.shape
                inputsize            = nchan*nlat*nlon
                outsize              = num_classes
                X = X.reshape(ndat,inputsize)
                
            """
            CNN
            X: time x channel x lat x lon    ex: [3438 x 3 x 224 x 224]
            y: time x 1                      ex: [3438 x 1]
            
            FNN
            X: time x inputsize     ex: [3438 x 15028]
            y: time x 1             ex: [3438 x 15028]
            """
            # ---------------------------------
            # Split into training and test sets
            # ---------------------------------
            X_train = torch.from_numpy( X[0:int(np.floor(percent_train*lead_nsamples)),...].astype(np.float32) )
            X_val   = torch.from_numpy( X[int(np.floor(percent_train*lead_nsamples)):,...].astype(np.float32) )
            y_train = torch.from_numpy( y_class[0:int(np.floor(percent_train*lead_nsamples)),:].astype(np.compat.long)  )
            y_val   = torch.from_numpy( y_class[int(np.floor(percent_train*lead_nsamples)):,:].astype(np.compat.long)  )
            
            
            # Put into pytorch DataLoader
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
            val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
        
            # ---------------
            # Train the model
            # ---------------
            if "FNN" in netname:
                layers = build_FNN_simple(inputsize,outsize,nlayers,nunits,activations,
                                          dropout=fnndropout,use_softmax=use_softmax)
                pmodel = nn.Sequential(*layers)
                
            else:
                pmodel = transfer_model(netname,num_classes,cnndropout=cnndropout,unfreeze_all=unfreeze_all,
                                        nlat=nlat,nlon=nlon,nchannels=nchannels)
            model,trainloss,testloss,trainacc,testacc = train_ResNet(pmodel,loss_fn,opt,train_loader,val_loader,max_epochs,
                                                                     early_stop=early_stop,verbose=verbose,
                                                                     reduceLR=reduceLR,LRpatience=LRpatience,)
            
            # Save train/test loss
            train_loss_grid.append(trainloss)
            test_loss_grid.append(testloss)
            train_acc_grid.append(trainacc)
            test_acc_grid.append(testacc)
            
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
                    batch_pred = model(batch_x)
                    
                    # Convert predicted values
                    y_batch_pred = np.argmax(batch_pred.detach().cpu().numpy(),axis=1)
                    y_batch_lab  = batch_y.detach().cpu().numpy().squeeze()
                    batch_acc    = np.sum(y_batch_pred==y_batch_lab)/y_batch_lab.shape[0]
                    #print("Acc. for batch %i is %.2f" % (i,batch_acc))
                    #print(y_batch_pred==y_batch_lab)
                    
                    # Store Predictions
                    y_pred_val = np.concatenate([y_pred_val,y_batch_pred])
                    y_valdt = np.concatenate([y_valdt,y_batch_lab])
                    
            # --------------
            # Save the model
            # --------------
            if savemodel:
                modout = "../../CESM_data/Models/%s_%s_lead%02i_classify.pt" %(expname,varname,lead)
                torch.save(model.state_dict(),modout)
            
            # Save the actual and predicted values
            yvalpred.append(y_pred_val)
            yvallabels.append(y_valdt)
            
            # -------------------------
            # Calculate Success Metrics
            # -------------------------
            # Calculate the total accuracy
            lead_acc = (yvalpred[l]==yvallabels[l]).sum()/ yvalpred[l].shape[0]
            total_acc.append(lead_acc)
            print("********Success rate********************")
            print("\t" +str(lead_acc*100) + r"%")
            
            # Calculate accuracy for each class
            class_total   = np.zeros([num_classes])
            class_correct = np.zeros([num_classes])
            val_size = yvalpred[l].shape[0]
            for i in range(val_size):
                class_idx  = int(yvallabels[l][i])
                check_pred = yvallabels[l][i] == yvalpred[l][i]
                class_total[class_idx]   += 1
                class_correct[class_idx] += check_pred 
                #print("At element %i, Predicted result for class %i was %s" % (i,class_idx,check_pred))
            class_acc = class_correct/class_total
            acc_by_class.append(class_acc)
            print("********Accuracy by Class***************")
            for  i in range(num_classes):
                print("\tClass %i : %03.3f" % (i,class_acc[i]*100) + "%\t" + "(%i/%i)"%(class_correct[i],class_total[i]))
            print("****************************************")
            
            # Visualize loss vs epoch for training/testing and correlation
            if debug:
                pepochs = np.arange(1,len(testloss)+1,1)
                
                # Loss by Epoch Plots
                fig,ax=plt.subplots(1,1)
                plt.style.use('default')
                ax.plot(pepochs,trainloss,label='train loss')
                ax.plot(pepochs,testloss,label='test loss')
                ax.set_xticks(np.arange(1,max_epochs+1,1))
                ax.legend()
                ax.set_title("Loss by Epoch for Leadtime %i %s"%(lead,subtitle))
                ax.set_ylabel("Loss")
                ax.set_xlabel("Epoch")
                ax.grid(True,linestyle="dotted")
                plt.savefig("../../CESM_data/Figures/%s_%s_leadnum%s_LossbyEpoch.png"%(expname,varname,lead))
                
                # Acc by Epoch Plots
                fig,ax=plt.subplots(1,1)
                plt.style.use('default')
                ax.plot(pepochs,trainacc,label='train accuracy')
                ax.plot(pepochs,testacc,label='test accuracy')
                ax.set_xticks(np.arange(1,max_epochs+1,1))
                ax.legend()
                ax.set_ylabel("Accuracy")
                ax.set_xlabel("Epoch")
                ax.set_title("Accuracy by Epoch for Leadtime %i %s"%(lead,subtitle))
                ax.grid(True,linestyle="dotted")
                plt.savefig("../../CESM_data/Figures/%s_%s_leadnum%s_AccbyEpoch.png"%(expname,varname,lead))
                
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
                     'train_acc' : train_acc_grid,
                     'test_acc' : test_acc_grid,
                     'total_acc': total_acc,
                     'acc_by_class': acc_by_class,
                     'yvalpred': yvalpred,
                     'yvallabels' : yvallabels,
                     'sampled_idx': sampled_idx,
                     'thresholds_all' : thresholds_all
                     }
                     )

        print("Saved data to %s%s. Finished variable %s in %ss"%(outpath,outname,varname,time.time()-start))
    print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))
