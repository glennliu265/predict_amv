
"""
CNN Test Lead Annual

testing different lead times for a specified CNN architecture

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

# -------------
#%% User Edits
# -------------

# Indicate machine to set path
machine='pdwang'

# Set directory and load data depending on machine
if machine == 'local-glenn':
    os.chdir('/Users/gliu/Downloads/2020_Fall/6.862/Project/predict_amv/CNN/')
    outpath = '/Users/gliu/Downloads/2020_Fall/6.862/Project/predict_amv/CNN/'

else:
    outpath = os.getcwd()
    
# Data preparation settings
leads          = np.arange(0,40,1)    # Time ahead (in years) to forecast AMV
resolution     = '2deg'               # Resolution of input (2deg or full)
season         = 'Ann'                # Season to take mean over ['Ann','DJF','MAM',...]
indexregion    = 'SPG'                # One of the following ("SPG","STG","TRO","NAT")

# Training/Testing Subsets
percent_train = 0.8   # Percentage of data to use for training (remaining for testing)
ens           = 40    # Ensemble members to use

# Model training settings
early_stop    = 3                     # Number of epochs where validation loss increases before stopping
max_epochs    = 10                    # Maximum number of epochs
batch_size    = 32                    # Pairs of predictions
loss_fn       = nn.MSELoss()          # Loss Function
opt           = ['Adadelta',0.1,0]    # Name optimizer
netname       = 'FNN2'                # See Choices under Network Settings below for strings that can be used

# Network Settings
if netname == 'CNN1':
    nchannels     = [32]                    # Number of out_channels for the first convolution
    filtersizes   = [[5,5]]                     # kernel size for first ConvLayer
    filterstrides = [[1,1]]
    poolsizes     = [[5,5]]                     # kernel size for first pooling layer
    poolstrides   = [[5,5]]
elif netname == 'CNN2':
    # 2 layer CNN settings 
    nchannels     = [32,64]
    filtersizes   = [[2,3],[3,3]]
    filterstrides = [[1,1],[1,1]]
    poolsizes     = [[2,3],[2,3]]
    poolstrides   = [[2,3],[2,3]]
elif netname == 'RN18': # ResNet18
    #resnet = models.resnet18(pretrained=True)
    inpadding = [95,67]
elif netname == 'RN50': # ResNet50
    inpadding = [95,67]
    #resnet = models.resnet50(pretrained=True)
elif netname == 'FNN2': # 2-layer Fully Connected NN
    nlayers = 2
    nunits  = [20,20]
    activations = [nn.ReLU(),nn.ReLU()]
    outsize = 1

# Options
debug   = False # Visualize training and testing loss
verbose = False # Print loss for each epoch

# -----------
#%% Functions
# -----------

def train_CNN(layers,loss_fn,optimizer,trainloader,testloader,max_epochs,early_stop=False,verbose=True):
    """
    inputs:
        layers      - tuple of NN layers
        loss_fn     - (torch.nn) loss function
        opt         - tuple of [optimizer_name, learning_rate, weight_decay] for updating the weights
                      currently supports "Adadelta" and "SGD" optimizers
        trainloader - (torch.utils.data.DataLoader) for training datasetmo
        testloader  - (torch.utils.data.DataLoader) for testing dataset
        max_epochs  - number of training epochs
        early_stop  - BOOL or INT, Stop training after N epochs of increasing validation error
                     (set to False to stop at max epoch, or INT for number of epochs)
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
    
    # Set early stopping threshold and counter
    if early_stop is False:
        i_thres = max_epochs
    else:
        i_thres = early_stop
    i_incr    = 0 # Number of epochs for which the validation loss increases
    prev_loss = 0 # Variable to store previous loss
    
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
                
                runningloss += loss.item()

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
                        if verbose:
                            print("Validation loss has increased at epoch %i"%(epoch+1))
                        i_incr += 1
                        lossprev = runningloss/len(data_loader)
                        
                if (epoch != 0) and (i_incr >= i_thres):
                    print("\tEarly stop at epoch %i "% (epoch+1))
                    return bestmodel,train_loss,test_loss  
                
                
    return bestmodel,train_loss,test_loss         

def calc_layerdim(nlat,nlon,filtersize,poolsize,nchannels):
    """
    For a 1 layer convolution, calculate the size of the next layer after flattening
    
    inputs:
        nlat:         latitude dimensions of input
        nlon:         longitude dimensions of input
        filtersize:   size of the filter in layer 1
        poolsize:     size of the maxpooling kernel
        nchannels:    number of out_channels in layer 1
    output:
        flattensize:  flattened dimensions of layer for input into FC layer
    
    """
    return int(np.floor((nlat-filtersize+1)/poolsize) * np.floor((nlon-filtersize+1)/poolsize) * nchannels)

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
    
    # # ## Debug entry
    # # 2 layer CNN settings 
    # nchannels     = [32,64]
    # filtersizes   = [[2,3],[3,3]]
    # filterstrides = [[1,1],[1,1]]
    # poolsizes     = [[2,3],[2,3]]
    # poolstrides   = [[2,3],[2,3]]
    # nx = 33
    # ny = 41
    # # # ----
    
    
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

def load_seq_model(layers,modpath):
    """
    Load a statedict into a model with the same architecture
    as specified in the layers tuple.
    
    Parameters
    ----------
    layers : TUPLE
        NN Layers for input into nn.Sequential()
    modpath : STR
        Path to saved state_dict()

    Returns
    -------
    model : Network (Pytorch)
        Loaded network with saved statedict.

    """
    model = nn.Sequential(**layers)
    model.load_state_dict(torch.load(modpath))
    return model

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

# ----------------------------------------
# %% Set-up
# ----------------------------------------
allstart = time.time()

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

# Preallocate Evaluation Metrics...
corr_grid_train = np.zeros((nlead))
corr_grid_test  = np.zeros((nlead))
train_loss_grid = np.zeros((max_epochs,nlead))
test_loss_grid  = np.zeros((max_epochs,nlead))

# Print Message
print("Running CNN_test_lead_ann.py with the following settings:")
print("\tNetwork Type   : "+netname)
print("\tPred. Region   : "+indexregion)
print("\tPred. Season   : "+season)
print("\tLeadtimes      : %i to %i" % (leads[0],leads[-1]))
print("\tMax Epochs     : " + str(max_epochs))
print("\tEarly Stop     : " + str(early_stop))
print("\t# Ens. Members : "+ str(ens))
print("\tOptimizer      : "+ opt[0])
# ----------------------------------------------
# %% Train for each variable combination and lead time
# ----------------------------------------------


for v in range(nvar): # Loop for each variable
    # -------------------
    # Set input variables
    # -------------------
    channels = 1
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
    
    # Set output path
    outname = "/leadtime_testing_%s_%s.npz" % (varname,expname)
    
    for l,lead in enumerate(leads):
        
        # ----------------------
        # Apply lead/lag to data
        # ----------------------
        y = calc_AMV_index(indexregion,sst_normed[:ens,lead:,:,:],lat,lon)
        y = y.reshape((y.shape[0]*y.shape[1]))[:,None]
        X = np.transpose(
            np.array(invars)[:,:ens,0:tstep-lead,:,:].reshape(channels,(tstep-lead)*ens,nlat,nlon),
            (1,0,2,3))
        
        # ---------------------------------
        # Split into training and test sets
        # ---------------------------------
        if netname == 'FNN2': # Flatten inputs for FNN 2
            ndat,nchan,nlat,nlon = X.shape # Get latitude and longitude sizes for dimension calculation
            inputsize = nchan*nlat*nlon
            X = X.reshape(ndat,inputsize)
            
            X_train = torch.from_numpy( X[0:int(np.floor(percent_train*(tstep-lead)*ens)),:] )
            X_val = torch.from_numpy( X[int(np.floor(percent_train*(tstep-lead)*ens)):,:] )
        else:
            X_train = torch.from_numpy( X[0:int(np.floor(percent_train*(tstep-lead)*ens)),:,:,:] )
            X_val = torch.from_numpy( X[int(np.floor(percent_train*(tstep-lead)*ens)):,:,:,:] )

        y_train = torch.from_numpy( y[0:int(np.floor(percent_train*(tstep-lead)*ens)),:] )
        y_val = torch.from_numpy( y[int(np.floor(percent_train*(tstep-lead)*ens)):,:] )
        
        # Put into pytorch DataLoader
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
        val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
        
        
        
        
        
        # -------------------------------
        # Initialize Network Architecture
        # -------------------------------
        
        if (netname == 'CNN1') | (netname == 'CNN2'):
            # Calculate dimensions of first FC layer (for CNNs)
            firstlineardim = calc_layerdims(nlat,nlon,filtersizes,filterstrides,poolsizes,poolstrides,nchannels)
        
        if netname == 'CNN1': # 1-layer CNN
            layers        = [
                            nn.Conv2d(in_channels=channels, out_channels=nchannels[0], kernel_size=filtersizes[0]),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=poolsizes[0]),
                            
                            nn.Flatten(),
                            nn.Linear(in_features=firstlineardim,out_features=128),
                            nn.ReLU(),
                            nn.Linear(in_features=128,out_features=64),
                            nn.ReLU(),
                            
                            #nn.Dropout(p=0.5),
                            nn.Linear(in_features=64,out_features=1)
                            ]
        elif netname == 'CNN2': # 2-layer CNN
            layers        = [
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
        elif netname == 'RN18':
            
            resnet18 = models.resnet18(pretrained=True)
            layers = [nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=(1,1),padding=inpadding),
                              resnet18,
                              nn.Linear(in_features=1000,out_features=1)]
        elif netname == 'RN50':
            
            resnet50 = models.resnet50(pretrained=True)
            layers = [nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=(1,1),padding=inpadding),
                              resnet50,
                              nn.Linear(in_features=1000,out_features=1)]
        elif netname == "FNN2":
            # Set up FNN
            layers = build_FNN_simple(inputsize,outsize,nlayers,nunits,activations,dropout=0)
        
        # ---------------
        # Train the model
        # ---------------
        model,trainloss,testloss = train_CNN(layers,loss_fn,opt,train_loader,val_loader,max_epochs,early_stop=early_stop,verbose=verbose)
            
        # Save train/test loss
        train_loss_grid[:,l] = np.array(trainloss).min().squeeze() # Take min of each epoch
        test_loss_grid[:,l]  = np.array(testloss).min().squeeze()
        
        # -----------------
        # Evalute the model
        # -----------------
        model.eval()
        y_pred_val     = model(X_val).detach().numpy()
        y_valdt        = y_val.detach().numpy()
        y_pred_train   = model(X_train).detach().numpy()
        y_traindt      = y_train.detach().numpy()
        
        # Get the correlation (save these)
        traincorr = np.corrcoef( y_pred_train.T[0,:], y_traindt.T[0,:])[0,1]
        testcorr  = np.corrcoef( y_pred_val.T[0,:], y_valdt.T[0,:])[0,1]
        
        # Stop if model is just predicting the same value (usually need to examine optimizer settings)
        if np.isnan(traincorr) | np.isnan(testcorr):
            print("Warning, NaN Detected for %s lead %i of %i. Stopping!" % (varname,lead,len(leads)))
            if debug:
                fig,ax=plt.subplots(1,1)
                plt.style.use('seaborn')
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
            break
        
        # Calculate Correlation and RMSE
        corr_grid_test[l]    = np.corrcoef( y_pred_val.T[0,:], y_valdt.T[0,:])[0,1]
        corr_grid_train[l]   = np.corrcoef( y_pred_train.T[0,:], y_traindt.T[0,:])[0,1]
        
        # Visualize loss vs epoch for training/testing and correlation
        if debug:
            fig,ax=plt.subplots(1,1)
            plt.style.use('seaborn')
            ax.plot(trainloss,label='train loss')
            ax.plot(testloss,label='test loss')
            ax.legend()
            ax.set_title("Losses for Predictor %s Leadtime %i"%(varname,lead))
            plt.show()
            
            
            fig,ax=plt.subplots(1,1)
            plt.style.use('seaborn')
            #ax.plot(y_pred_train,label='train corr')
            #ax.plot(y_pred_val,label='test corr')
            #ax.plot(y_valdt,label='truth')
            ax.scatter(y_pred_val,y_valdt,label="Test",marker='+',zorder=2)
            ax.scatter(y_pred_train,y_traindt,label="Train",marker='x',zorder=1,alpha=0.3)
            ax.legend()
            ax.set_ylim([-1.5,1.5])
            ax.set_xlim([-1.5,1.5])
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                    ]
            ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            ax.legend()
            ax.set_xlabel("Actual AMV Index")
            ax.set_ylabel("Predicted AMV Index")
            ax.set_title("Correlation %.2f for Predictor %s Leadtime %i"%(corr_grid_test[l],varname,lead))
            plt.show()
        
        # --------------
        # Save the model
        # --------------
        modout = "%s../../CESM_data/Models/%s_%s_lead%i.pt" %(outpath,expname,varname,lead)
        torch.save(model.state_dict(),modout)
        
        print("\nCompleted training for %s lead %i of %i" % (varname,lead,len(leads)))
    
    # -----------------
    # Save Eval Metrics
    # -----------------
    np.savez(outpath+"/../../CESM_data/Metrics"+outname,**{
             'train_loss': train_loss_grid,
             'test_loss': test_loss_grid,
             'test_corr': corr_grid_test,
             'train_corr': corr_grid_train}
            )
    print("Saved data to %s%s. Finished variable %s in %ss"%(outpath,outname,varname,time.time()-start))


print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))
#%%



# # -------------
# # %% Make Plots
# # -------------


# import matplotlib.pyplot as plt

# # Plot the Correlation grid
# data = corr_grid_test.copy()**2
# gsize = data.shape[0]
# cmap = plt.get_cmap("pink",20)
# cmap.set_bad(np.array([0,255,0])/255)
# fig,ax = plt.subplots(1,1,figsize=(8,8))
# im = ax.imshow(data,vmin=0,vmax=1,cmap=cmap)
# ax.set_title("Correlation $(R^{2})$"+"(CESM - CNN Output); Predictor = %s \n %s vs %s"% (varname,pr1name,pr2name))
# ax.set_xticks(np.arange(0,gsize))
# ax.set_yticks(np.arange(0,gsize))
# ax.set_xticklabels(param1)
# ax.set_yticklabels(param2)
# ax.set_xlabel(pr1name)
# ax.set_ylabel(pr2name)
# plt.gca().invert_yaxis()
# plt.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
# # Loop over data dimensions and create text annotations.
# for i in range(np1):
#     for j in range(np2):
#         # Set color to black if above threshold, white otherwise
#         if data[i,j] > 0.6:
#             usecolor='k'
#         else:
#             usecolor='w'
        
#         if data[i,j] == np.nanmax(data): # Max in Red
#             usecolor='r'
#         elif data[i,j] == np.nanmin(data): # Min in Blue
#             usecolor= np.array([0,202,231])/255
        
#         text = ax.text(j, i, "%.1e"%data[i, j],
#                        ha="center", va="center", color=usecolor)
        
#         #text.set_path_effects([path_effects.Stroke(linewidth=0.25,foreground='k')])
# plt.savefig("%sCorr_%s.png"% (outpath,expname),dpi=200)
# plt.show()


# # Plot the RMSE grid
# data = test_loss_grid
# gsize = data.shape[0]
# cmap = plt.get_cmap("pink",20)
# cmap.set_bad(np.array([0,255,0])/255)
# fig,ax = plt.subplots(1,1,figsize=(8,8))
# im = ax.imshow(data,vmin=0,vmax=1,cmap=cmap)
# ax.set_title("MSE (CESM - CNN Output); Predictor %s \n %s vs %s"% (varname,pr1name,pr2name))
# ax.set_xticks(np.arange(0,gsize))n
# ax.set_yticks(np.arange(0,gsize))
# ax.set_xticklabels(param1)
# ax.set_yticklabels(param2)
# ax.set_xlabel(pr1name)
# ax.set_ylabel(pr2name)
# plt.gca().invert_yaxis()
# plt.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
# # Loop over data dimensions and create text annotations.
# for i in range(np1):
#     for j in range(np2):
#         # Set color to black if above threshold, white otherwise
#         if data[i,j] > 0.6:
#             usecolor='k'
#         else:
#             usecolor='w'
        
#         if data[i,j] == np.nanmax(data): # Max in Red
#             usecolor='r'
#         elif data[i,j] == np.nanmin(data): # Min in Blue
#             usecolor= np.array([0,202,231])/255
        
#         text = ax.text(j, i, "%.1e"%data[i, j],
#                        ha="center", va="center", color=usecolor)
        
#         #text.set_path_effects([path_effects.Stroke(linewidth=0.25,foreground='k')])
# plt.savefig("%sMSE_%s.png"%(outpath,expname),dpi=200)
# plt.show()



