
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
# -------------------
#%% User Edits/Inputs
# -------------------

# Data preparation settings
leads          = np.arange(0,25,3)    # Time ahead (in years) to forecast AMV
season         = 'Ann'                # Season to take mean over ['Ann','DJF','MAM',...]
indexregion    = 'NAT'                # One of the following ("SPG","STG","TRO","NAT")
detrend        = False                # Predict undetrended data

# Training/Testing Subsets
percent_train = 0.8   # Percentage of data to use for training (remaining for testing)
ens           = 1    # Ensemble members to use
tstep         = 86    # Size of time dimension (in years)

# Model architecture settings
netname       = 'resnet18'            # Name of pretrained network (timm module)
rnnname       = 'GRU'                 # LSTM or GRU
hidden_size   = 30                    # The size of the hidden layers in the RNN
cnn_out       = 1000                  # Number of features to be extracted by CNN and input into RNN
rnn_layers    = 1                     # Number of rnn layers
outsize       = 1                     # Final output size
outactivation = False                 # Activation for final output
seq_len       = 10                     # Length of sequence (same units as data [years])
cnndropout    = False                  # Set to 1 to test simple CN with dropout layer

# Model training settings
early_stop    = 2                     # Number of epochs where validation loss increases before stopping
max_epochs    = 2                    # Maximum number of epochs
batch_size    = 4                     # Number of ensemble members to use per step
loss_fn       = nn.MSELoss()          # Loss Function
opt           = ['Adadelta',.01,0]    # Name optimizer
reduceLR      = False                 # Set to true to use LR scheduler
LRpatience    = 3                     # Set patience for LR scheduler
outpath       = ''



# Misc. saving options
resolution    = '224pix'
outpath       = ''

# Options
debug      = True  # Visualize training and testing loss
verbose    = True  # Print loss for each epoch
checkgpu   = True  # Set to true to check for GPU otherwise run on CPU
savemodel  = False # Set to true to save model dict.
freeze_all = False # Freeze all layers


#%%

# Set configurable network
config = {}

# Training Configurations
config['loss_fn']      = loss_fn
config['batch_size']   = batch_size
config['max_epochs']   = max_epochs
config['optname']      = opt[0]
config['lr']           = opt[1]
config['lr_scheduler'] = reduceLR
config['lr_patience']  = LRpatience
config['weight_decay'] = opt[2]
config['early_stop']   = early_stop


# Model Architecture
config['cnn_dropout'] = cnndropout
config['cnn_out'] = cnn_out

config['rnn_layers'] = rnn_layers
config['rnn_out']  = outsize




# -----------------------
#%% Functions and classes
# -----------------------

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
    # #model =   timm.create_model('tf_efficientnet_l2_ns') # read in resnet model
    # model = timm.create_model("tf_efficientnet_b7_ns")
    # for param in model.parameters():
    #     print(param)
    #     print(param.requires_grad)
    #     param.requires_grad = False

    # #model.classifier = nn.Linear(5504, 1) # freeze all layers except the last one l2-noisy student
    # model.classifier=nn.Linear(model.classifier.in_features,1)
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

    #bestmodel.load_state_dict(best_model_wts)
    return bestmodel,train_loss,test_loss

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

        nsamples= ntime-seq_len

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
# ----------------------------------------

# %% Set-up
# ----------------------------------------
allstart = time.time()

# Set experiment names ----
nvar    = 1 # Combinations of variables to test (currently set to test all 3 variables)
nlead   = len(leads)
varname = 'ALL'

# Save data (ex: Ann2deg_NAT_CNN2_nepoch5_nens_40_lead24 )
expname = "%s%s_%s_%s_nepoch%02i_nens%02i_lead%02i_%s_sqlen%i" % (season,resolution,indexregion,
                                                                  netname,max_epochs,ens,len(leads)-1,
                                                                  rnnname,hidden_size)
# Load the data for whole North Atlantic
# Data : [channel, ensemble, time, lat, lon]
# Label: [ensemble, time]
data   = np.load('../../CESM_data/CESM_data_sst_sss_psl_deseason_normalized_resized_detrend0.npy')
target = np.load('../../CESM_data/CESM_label_amv_index_detrend%i.npy'%detrend)
data   = data[:,0:ens,:,:,:]
target = target[0:ens,:]

# Preallocate Evaluation Metrics...
corr_grid_test  = []
train_loss_grid = np.zeros((max_epochs,nlead))
test_loss_grid  = np.zeros((max_epochs,nlead))
yvalpred        = []
yvallabels      = []

# Print Message
print("Running ENN_test_lead_ann.py with the following settings:")
print("\tNetwork Type       : "+netname)
print("\tRNN Type           : "+rnnname)
print("\tRNN hiddden states : "+str(hidden_size))
print("\tFeatures Extracted : "+str(cnn_out))
print("\tSequence Length    : "+str(seq_len))
print("\tLeadtimes          : %i to %i" % (leads[0],leads[-1]))
print("\tMax Epochs         : " + str(max_epochs))
print("\tEarly Stop         : " + str(early_stop))
print("\t# Ens. Members     : "+ str(ens))
print("\tOptimizer          : "+ opt[0])

# Check if device has GPU
if checkgpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
# ----------------------------------------------
# %% Train for each lead time
# ----------------------------------------------
start = time.time()

for l,lead in enumerate(leads):

    # Set output path
    outname = "/leadtime_testing_%s_%s_lead%02dof%02d.npz" % (varname,expname,lead,leads[-1])
    
    # ----------------------
    # Apply lead/lag to data
    # ----------------------
    y = target[:ens,lead:].astype(np.float32)
    X = (data[:,:,:tstep-lead,:,:].transpose(1,2,0,3,4)).astype(np.float32) # [Transpose to ens x time x channel x lat x lon]


    # -------------------------
    # Preprocess into sequences
    # -------------------------
    Xseq,yseq = make_sequences(X,y,seq_len)
    nsamples = Xseq.shape[0]


    # ---------------------------------
    # Split into training and test sets
    # ---------------------------------

    X_train = torch.from_numpy( Xseq[0:int(np.floor(percent_train*nsamples)),...].astype(np.float32))
    X_val = torch.from_numpy( Xseq[int(np.floor(percent_train*nsamples)):,...].astype(np.float32))

    y_train = torch.from_numpy(  yseq[0:int(np.floor(percent_train*nsamples)),None].astype(np.float32))
    y_val = torch.from_numpy( yseq[int(np.floor(percent_train*nsamples)):,None].astype(np.float32))

    # Put into pytorch DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size,num_workers=4)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size,num_workers=4)

    # -----------------------
    # Set up component models
    # -----------------------
    # Set pretrained CNN as feature extractor
    pmodel = transfer_model(netname,cnn_out,cnndropout=cnndropout)

    # Set either a LTSM or GRU unit
    if rnnname == 'LSTM':
        rnn = nn.LSTM(
                input_size=cnn_out,
                hidden_size=hidden_size,
                num_layers=rnn_layers,
                batch_first=True # Input is [batch,seq,feature]
                )
    elif rnnname == 'GRU':
        rnn = nn.GRU(
                input_size=cnn_out,
                hidden_size=hidden_size,
                num_layers=rnn_layers,
                batch_first=True # Input is [batch,seq,feature]
                )

    # Set fully-connected layer for classification
    classifier = nn.Linear(hidden_size,outsize)

    # Combine all into sequence model
    seqmodel = Combine(pmodel,rnn,classifier,outactivation)

    # ---------------
    # Train the model
    # ---------------
    model,trainloss,testloss = train_ResNet(seqmodel,loss_fn,opt,train_loader,val_loader,max_epochs,
                                            early_stop=early_stop,verbose=verbose,
                                            reduceLR=reduceLR,LRpatience=LRpatience)

                                                
    # Save train/test loss
    train_loss_grid[:,l] = np.array(trainloss).min().squeeze() # Take min of each epoch
    test_loss_grid[:,l]  = np.array(testloss).min().squeeze()

    #print("After train function memory is %i"%(torch.cuda.memory_allocated(device)))

    # -----------------------------------------------
    # Pass to GPU or CPU and Evaluate Model
    # -----------------------------------------------
    with torch.no_grad():
        # -----------------
        # Evalute the model
        # -----------------
        model.eval()
        for i,vdata in enumerate(val_loader):


            # Get mini batch
            batch_x, batch_y = vdata
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Make prediction and concatenate for each batch
            batch_pred = model(batch_x)

            if i == 0:
                y_pred_val=batch_pred.detach().cpu().numpy().squeeze()
                y_valdt = batch_y.detach().cpu().numpy().squeeze()
            else:
                y_pred_val = np.hstack([y_pred_val,batch_pred.detach().cpu().numpy().squeeze()])
                y_valdt = np.hstack([y_valdt,batch_y.detach().cpu().numpy().squeeze()])

    # Calculate correlation between prediction+label
    testcorr = np.corrcoef( y_pred_val[:].T, y_valdt[:].T)[0,1]

    if verbose:
        print("Correlation for lead %i was %f"%(lead,testcorr))
    corr_grid_test.append(testcorr)

    # --------------
    # Save the model
    # --------------
    if savemodel:
        modout = "../../CESM_data/Models/%s_%s_lead%i.pt" %(expname,varname,lead)
        torch.save(model.state_dict(),modout)


    # Stop if model is just predicting the same value (usually need to examine optimizer settings)
    if np.any(np.isnan(testcorr)):
        print("Warning, NaN Detected for %s lead %i of %i. Stopping!" % (varname,lead,len(leads)))
        if debug:
            fig,ax=plt.subplots(1,1)
            ax.plot(trainloss,label='train loss')
            ax.plot(testloss,label='test loss')
            ax.legend()
            ax.set_title("Losses for Predictor %s Leadtime %i"%(varname,lead))
            ax.grid(True,ls='dotted')
            plt.show()

            fig,ax=plt.subplots(1,1)
            #ax.plot(y_pred_train,label='train corr')
            ax.plot(y_pred_val,label='test corr')
            ax.plot(y_valdt,label='truth')
            ax.legend()
            ax.set_title("Correlation for Predictor %s Leadtime %i"%(varname,lead))
            ax.grid(True,ls='dotted')
            plt.show()
        break

    # Calculate Correlation and RMSE


    # Visualize loss vs epoch for training/testing and correlation
    if debug:

        # Train vs Test Loss Plot
        fig,ax=plt.subplots(1,1)
        ax.plot(trainloss,label='train loss')
        ax.plot(testloss,label='test loss')
        ax.legend()
        ax.set_title("Losses for Predictor %s Leadtime %i"%(varname,lead))
        ax.grid(True,ls='dotted')
        plt.savefig("../../CESM_data/Figures/%s_%s_leadnum%s_LossbyEpoch.png"%(expname,varname,lead))

        # Scatterplot of predictions vs Labels
        fig,ax=plt.subplots(1,1)
        ax.scatter(y_valdt,y_pred_val,label="Test",marker='+',zorder=2)
        ax.legend()
        ax.set_ylim([-1.5,1.5])
        ax.set_xlim([-1.5,1.5])
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.legend()
        ax.set_ylabel("Predicted AMV Index")
        ax.set_xlabel("Actual AMV Index")
        ax.set_title("Correlation %.2f for Predictor %s Leadtime %i"%(np.mean(corr_grid_test[l]),varname,lead))
        ax.grid(True,ls='dotted')
        plt.savefig("../../CESM_data/Figures/%s_%s_leadnum%s_ValidationScatter.png"%(expname,varname,lead))

    print("\nCompleted training for %s lead %i of %i" % (varname,lead,leads[-1]))

    # Clear some memory
    del model
    del X_val
    del y_val
    del X_train
    del y_train
    torch.cuda.empty_cache()  # Save some memory
    #print("After lead loop end for %i memory is %i"%(lead,torch.cuda.memory_allocated(device)))

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
             'y_pred_val': y_pred_val,
             'y_valdt' : y_valdt
             }
            )
print("Saved data to %s%s. Finished variable %s in %ss"%(outpath,outname,varname,time.time()-start))

fig,ax = plt.subplots(1,1)
ax.plot(leads,corr_grid_test)
ax.axhline(0,ls='dashed',color='k')
ax.set_ylim([-1,1])
ax.grid(True,ls='dotted')
plt.savefig("../../CESM_data/Figures/%s_%s_Prediction_v_Leadtime.png"%(expname,varname))

print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))
