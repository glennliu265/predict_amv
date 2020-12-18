
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
#%% User Edits
# -------------
    
# Data preparation settings
leads          = np.arange(0,25,1)    # Time ahead (in years) to forecast AMV
season         = 'Ann'                # Season to take mean over ['Ann','DJF','MAM',...]
indexregion    = 'NAT'                # One of the following ("SPG","STG","TRO","NAT")

# Training/Testing Subsets
percent_train = 0.8   # Percentage of data to use for training (remaining for testing)
ens           = 40    # Ensemble members to use

# Model training settings
early_stop    = 3                     # Number of epochs where validation loss increases before stopping
max_epochs    = 15                    # Maximum number of epochs
batch_size    = 32                    # Pairs of predictions
loss_fn       = nn.MSELoss()          # Loss Function
opt           = ['Adam']    # Name optimizer
netname       = 'EffNet-b7-ns'                # See Choices under Network Settings below for strings that can be used
resolution    = '244pix'
tstep         = 86
outpath       = ''

# Options
debug   = True # Visualize training and testing loss
verbose = False # Print loss for each epoch
checkgpu = True # Set to true to check for GPU otherwise run on CPU
# -----------
#%% Functions
# -----------

def train_ResNet(loss_fn,optimizer,trainloader,testloader,max_epochs,early_stop=False,verbose=True):
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
    
    output:
    
    dependencies:
        from torch import nn,optim
        
    """
    
    #model =   timm.create_model('tf_efficientnet_l2_ns') # read in resnet model
    model = timm.create_model("tf_efficientnet_b7_ns")
    for param in model.parameters():
        param.requires_grad = False
    
    #model.classifier = nn.Linear(5504, 1) # freeze all layers except the last one l2-noisy student
    model.classifier=nn.Linear(2560,1)
    bestloss = np.infty
    
    # Check if there is GPU
    if checkgpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model.to(device)
        
    
    # Set optimizer
    if optimizer[0] == "Adadelta":
        opt = optim.Adadelta(model.parameters())
    elif optimizer[0] == "SGD":
        opt = optim.SGD(model.parameters(),lr=optimizer[1],weight_decay=optimizer[2])
    elif optimizer[0] == 'Adam':
        opt = optim.Adam(model.parameters())
    
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

# ----------------------------------------
# %% Set-up
# ----------------------------------------
allstart = time.time()

# Set experiment names ----
nvar  = 1 # Combinations of variables to test
nlead = len(leads)

# Save data (ex: Ann2deg_NAT_CNN2_nepoch5_nens_40_lead24 )
expname = "%s%s_%s_%s_nepoch%02i_nens%02i_lead%02i" % (season,resolution,indexregion,netname,max_epochs,ens,len(leads)-1)

# Load the data for whole North Atlantic
data   = np.load('../../CESM_data/CESM_data_sst_sss_psl_deseason_normalized_resized.npy')
target = np.load('../../CESM_data/CESM_label_amv_index.npy')
data   = data[:,0:ens,:,:,:]
target = target[0:ens,:]

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
    channels = 3
    start = time.time()
    varname = 'ALL'
    
    # Set output path
    outname = "/leadtime_testing_%s_%s.npz" % (varname,expname)
    
    ytrainpred   = []
    ytrainlabels = []
    yvalpred     = []
    yvallabels   = []
    for l,lead in enumerate(leads):
        
        # ----------------------
        # Apply lead/lag to data
        # ----------------------
        y = target[:ens,lead:].reshape(ens*(tstep-lead),1)
        X = (data[:,:,:tstep-lead,:,:]).reshape(3,ens*(tstep-lead),244,244).transpose(1,0,2,3)
        
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
        model,trainloss,testloss = train_ResNet(loss_fn,opt,train_loader,val_loader,max_epochs,early_stop=early_stop,verbose=verbose)
            
        # Save train/test loss
        train_loss_grid[:,l] = np.array(trainloss).min().squeeze() # Take min of each epoch
        test_loss_grid[:,l]  = np.array(testloss).min().squeeze()
        
        
        # -----------------------------------------------
        # Pass to GPU or CPU for evaluation of best model
        # -----------------------------------------------
        if checkgpu:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        model.to(device)
        #X_train,X_val=X_train.to(device),X_val.to(device)
        X_val,y_val = X_val.to(device),y_val.to(device)
        #y_train,y_val=y_train.to(device),y_val.to(device)

        # -----------------
        # Evalute the model
        # -----------------
        model.eval()
        y_pred_val     = model(X_val).cpu().detach().numpy()
        y_valdt        = y_val.cpu().detach().numpy()
        #y_pred_train   = model(X_train).detach().numpy()
        #y_traindt      = y_train.detach().numpy()
        
        
        # Save the actual and predicted values
        #ytrainpred.append(y_pred_train)
        #ytrainlabels.append(y_traindt)
        yvalpred.append(y_pred_val)
        yvallabels.append(y_valdt)
        
        # Get the correlation (save these)
        #traincorr = np.corrcoef( y_pred_train.T[0,:], y_traindt.T[0,:])[0,1]
        testcorr  = np.corrcoef( y_pred_val.T[0,:], y_valdt.T[0,:])[0,1]
        
        # Stop if model is just predicting the same value (usually need to examine optimizer settings)
        #if np.isnan(traincorr) | np.isnan(testcorr):
        if np.isnan(testcorr):
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
        #corr_grid_train[l]   = np.corrcoef( y_pred_train.T[0,:], y_traindt.T[0,:])[0,1]
        
        # Visualize loss vs epoch for training/testing and correlation
        if debug:
            fig,ax=plt.subplots(1,1)
            plt.style.use('seaborn')
            ax.plot(trainloss,label='train loss')
            ax.plot(testloss,label='test loss')
            ax.legend()
            ax.set_title("Losses for Predictor %s Leadtime %i"%(varname,lead))
            plt.show()
            plt.savefig("../../CESM_data/Figures/%s_%s_leadnum%s_LossbyEpoch.png"%(expname,varname,lead))
            
            
            fig,ax=plt.subplots(1,1)
            plt.style.use('seaborn')
            #ax.plot(y_pred_train,label='train corr')
            #ax.plot(y_pred_val,label='test corr')
            #ax.plot(y_valdt,label='truth')
            ax.scatter(y_pred_val,y_valdt,label="Test",marker='+',zorder=2)
            #ax.scatter(y_pred_train,y_traindt,label="Train",marker='x',zorder=1,alpha=0.3)
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
            plt.savefig("../../CESM_data/Figures/%s_%s_leadnum%s_ValidationScatter.png"%(expname,varname,lead))
        
        # --------------
        # Save the model
        # --------------
        modout = "../../CESM_data/Models/%s_%s_lead%i.pt" %(expname,varname,lead)
        torch.save(model.state_dict(),modout)
        
        print("\nCompleted training for %s lead %i of %i" % (varname,lead,len(leads)))
    
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


print("Leadtesting ran to completion in %.2fs" % (time.time()-allstart))

