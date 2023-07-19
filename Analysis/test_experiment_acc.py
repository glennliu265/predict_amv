#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load model weights and check the differences in accuracy. 
Written for debugging issues with training script rewrite...

Copies upper sections of:
    - [test_predictor_uncertainty.py]
    - [check_experiment_output.py]
    
Created on Tue Apr 11 11:37:04 2023

@author: gliu
"""

import numpy as np
import sys
import glob
import importlib
import copy
import xarray as xr

import torch
from torch import nn

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from tqdm import tqdm
import time
import os

from torch.utils.data import DataLoader, TensorDataset,Dataset
#%% Load some functions

#% Load custom packages and setup parameters
# Import general utilities from amv module
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import proc

# Import packages specific to predict_amv
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import train_cesm_params as train_cesm_params
import amv_dataloader as dl
import amvmod as am


#%% User Edits


#%% Experiment Information (from viz_acc_byexp) # --------------------------------------------------

# % Compare particular predictor across experiments for wrtiten version


exp2 = {"expdir"        : "FNN4_128_SingleVar"   , # Directory of the experiment
        "searchstr"     :  "*SSH*"               , # Search/Glob string used for pulling files
        "expname"       : "SSH_Original"       , # Name of the experiment (Short)
        "expname_long"  : "SSH (Original Script)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "b"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        "no_val"        : True  # Whether or not there is a validation dataset
        }

# exp3 = {"expdir"        : "FNN4_128_SingleVar_Rewrite" , # Directory of the experiment
#         "searchstr"     :  "*SSH*",                      # Search/Glob string used for pulling files
#         "expname"       : "SSH_Rewrite"           ,      # Name of the experiment (Short)
#         "expname_long"  : "SSH (Rewrite)"   ,            # Long name of the experiment (for labeling on plots)
#         "c"             : "orange"                    , # Color for plotting
#         "marker"        : "d"                    , # Marker for plotting
#         "ls"            : "dashed"               , # Linestyle for plotting
#         "no_val"        : True  # Whether or not there is a validation dataset
#         }


# exp4 = {"expdir"        : "FNN4_128_SingleVar_debug1_shuffle_all" , # Directory of the experiment
#         "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
#         "expname"       : "SSH_Rewrite_newest"           , # Name of the experiment (Short)
#         "expname_long"  : "SSH (Rewrite Newest)"   , # Long name of the experiment (for labeling on plots)
#         "c"             : "r"                    , # Color for plotting
#         "marker"        : "d"                    , # Marker for plotting
#         "ls"            : "dashed"               , # Linestyle for plotting
#         "no_val"        : False  # Whether or not there is a validation dataset
#         }

# exp5 = {"expdir"        : "FNN4_128_SingleVar_debug1_shuffle_all_20ep_3ES_32bs" , # Directory of the experiment
#         "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
#         "expname"       : "SSH_Rewrite_newest_redEp"           , # Name of the experiment (Short)
#         "expname_long"  : "SSH (Rewrite Newest, Reduce Epochs)"   , # Long name of the experiment (for labeling on plots)
#         "c"             : "magenta"                    , # Color for plotting
#         "marker"        : "d"                    , # Marker for plotting
#         "ls"            : "dashed"               , # Linestyle for plotting
#         "no_val"        : False  # Whether or not there is a validation dataset
#         }


# exp6 = {"expdir"        : "FNN4_128_SingleVar_debug1_shuffle_all_20ep_3ES_16bs" , # Directory of the experiment
#         "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
#         "expname"       : "SSH_Rewrite_newest_redEp_redBS"           , # Name of the experiment (Short)
#         "expname_long"  : "SSH (Rewrite Newest, Reduce Epochs and Batch Size)"   , # Long name of the experiment (for labeling on plots)
#         "c"             : "limegreen"                    , # Color for plotting
#         "marker"        : "d"                    , # Marker for plotting
#         "ls"            : "dashed"               , # Linestyle for plotting
#         "no_val"        : False  # Whether or not there is a validation dataset
#         }


# exp7 = {"expdir"        : "FNN4_128_SingleVar_debug1_shuffle_all_no_val" , # Directory of the experiment
#         "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
#         "expname"       : "SSH_Rewrite_newest_no_val"           , # Name of the experiment (Short)
#         "expname_long"  : "SSH (Rewrite Newest, No Validation)"   , # Long name of the experiment (for labeling on plots)
#         "c"             : "cyan"                    , # Color for plotting
#         "marker"        : "d"                    , # Marker for plotting
#         "ls"            : "solid"               , # Linestyle for plotting
#         "no_val"        : False  # Whether or not there is a validation dataset
#         }

exp8 = {"expdir"        : "FNN4_128_SingleVar_debug1_shuffle_all_no_val_8020" , # Directory of the experiment
        "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
        "expname"       : "SSH_Rewrite_newest_no_val_8020"           , # Name of the experiment (Short)
        "expname_long"  : "SSH (Rewrite Newest, No Validation 80-20)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "gold"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        "no_val"        : False  # Whether or not there is a validation dataset
        }

ens_all  = [40,42]
inexps   = (exp2,exp8,)#exp3,exp4,exp5,exp6,exp7,exp8)
compname = "Rewrite"
quartile = False
leads    = np.arange(0,26,3)
detrend  = False
no_vals  = [d['no_val'] for d in inexps]



#%%


# -----------------------------------------------------------------------------


#%% User Edits 

# expdir             = "FNN4_128_SingleVar"
# modelname          = "FNN4_128"
varnames             = ["SSH",] 
ens                  = 42 # need to be aware that the number of ensemble member varies by experiment
nmodels              = 50 # Specify manually how much to do in the analysis

# Load parameters from [oredict_amv_param.py]
datpath              = pparams.datpath
figpath              = pparams.figpath
nn_param_dict        = pparams.nn_param_dict
class_colors         = pparams.class_colors
classes              = pparams.classes

checkgpu             = True
#%% Load the parameter dictionaries for each experiment

eparams_all = []
for expdict in inexps:
    
    expdir    = expdict['expdir']
    print("Loading parameters from %s" % (expdir))
    eparams   = train_cesm_params.train_params_all[expdir] # Load experiment parameters
    eparams_all.append(eparams)


#%% Load data (taken from train_NN_CESM1.py)

# Load some variables for ease

# Load predictor and labels, lat/lon, cut region
target         = dl.load_target_cesm(detrend=eparams['detrend'],region=eparams['region'])
data,lat,lon   = dl.load_data_cesm(varnames,eparams['bbox'],detrend=eparams['detrend'],return_latlon=True)

# Create classes 
# Set exact threshold value
std1         = target.std(1).mean() * eparams['thresholds'][1] # Multiple stdev by threshold value 
if eparams['quantile'] is False:
    thresholds_in = [-std1,std1]
else:
    thresholds_in = eparams['thresholds']

# Classify AMV Events
target_class = am.make_classes(target.flatten()[:,None],thresholds_in,exact_value=True,reverse=True,quantiles=eparams['quantile'])
target_class = target_class.reshape(target.shape)

# Subset predictor by ensemble, remove NaNs, and get sizes
data                           = data[:,0:ens,...]      # Limit to Ens
data[np.isnan(data)]           = 0                      # NaN Points to Zero
nchannels,nens,ntime,nlat,nlon = data.shape             # Ignore year and ens for now...
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN
nclasses = len(eparams['thresholds']) + 1
nlead    = len(leads)

#%% Load model weights (taken from LRP_LENS.py and viz_acc_byexp)


nexps = len(inexps)
varname = varnames[0]

modweights_byexp = [] # [exp][lead][runid][?]
modlist_byexp    = []
flists           = []
expdicts         = []
for exp in range(nexps):
    exp_info = inexps[exp]
    
    
    # Get the model weights
    modweights_lead,modlist_lead=am.load_model_weights(datpath,exp_info['expdir'],leads,varname)
    modweights_byexp.append(modweights_lead)
    modlist_byexp.append(modlist_lead)
    
    
    # Get list of metric files
    search = "%s%s/Metrics/%s" % (datpath,exp_info['expdir'],"*%s*" % varname)
    flist  = glob.glob(search)
    flist  = [f for f in flist if "of" not in f]
    flist.sort()
    flists.append(flist)
    print("Found %i files for %s using searchstring: %s" % (len(flist),varname,search))
    
    
    
    

# Get the shuffled indices


expdict = am.make_expdict(flists,leads,no_val=no_vals)

# Unpack Dictionary
totalacc,classacc,ypred,ylabs,shuffids = am.unpack_expdict(expdict,)

# shuffids [exp][run][lead][nsamples]

[print(len(shuffids[0][0]))]
#%% Test the models

nexps             = len(inexps)
st                = time.time()

# 
labels_all       = [] # [exp][lead]
predictions_all  = []


y_data = []

modelaccs_all    = np.zeros((nexps,nlead,nmodels,nclasses)) # [exp, lead, run, class]
totalacc_all     = np.zeros((nexps,nlead,nmodels))          # [exp, lead, run,]

totalacc_old     = totalacc_all.copy()
classacc_old     = modelaccs_all.copy()


for exp in range(nexps): # Predictor Variable
    
    eparams = eparams_all[exp]
    expdict = inexps[exp]
        
    ens = eparams['ens']
    

    # Get the predictor and restrict the label
    predictor = data[[0],:ens,:,:,:] # [channel x ens x time x lat x lon]
    target_in = target_class[:ens,:]
    
    
    # Preallocate
    idcorrect_lead    = []
    labels_lead       = []
    predictions_lead  = []
        
    y_data_lead =  []
    for l,lead in enumerate(leads):
        
        # Get the models (now by variable and by leadtime)
        modweights = modweights_byexp[exp][l]
        modlist    = modlist_byexp[exp][l]
        
        # ---------------------
        # 08. Apply Lead
        # ---------------------
        X,y_class = am.apply_lead(predictor,target_in,lead,reshape=True,ens=ens,tstep=ntime)
        
        
        #yvalpred        = []
        #yvallabels      = []
        
        y_data_run      = []
        # Loop by model..
        for nm in range(nmodels):
            
            
            # ------------------------------------------------------------------
            # 09. Select samples recorded in the shuffled indices (nsamples x 1)
            # ------------------------------------------------------------------
            sampleids = shuffids[exp][nm][l].astype(int)
            
            X_in      = X[:,...] #X[sampleids,...] 
            y_in      = y_class[:,...]#y_class[sampleids,...]
            
            # Flatten input data for FNN
            if "FNN" in eparams['netname']:
                ndat,nchannels,nlat,nlon = X_in.shape
                inputsize                = nchannels*nlat*nlon
                outsize                  = nclasses
                X_in                     = X_in.reshape(ndat,inputsize)
            
            # ------------------------
            # 10. Train/Test/Val Split
            # ------------------------
            X_subsets,y_subsets      = am.train_test_split(X_in,y_in,eparams['percent_train'],
                                                           percent_val=eparams['percent_val'],
                                                           debug=False,offset=eparams['cv_offset'])
            # Convert to Tensors
            X_subsets = [torch.from_numpy(X.astype(np.float32)) for X in X_subsets]
            y_subsets = [torch.from_numpy(y.astype(np.compat.long)) for y in y_subsets]
            
            # # Put into pytorch dataloaders
            data_loaders = [DataLoader(TensorDataset(X_subsets[iset],y_subsets[iset]), batch_size=eparams['batch_size']) for iset in range(len(X_subsets))]
            if eparams['percent_val'] > 0:
                train_loader,test_loader,val_loader = data_loaders
            else:
                train_loader,test_loader, = data_loaders
            
            # ----------------- Section from LRP_LENs
            # Rebuild the model
            
            pmodel = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
            
            # Load the weights
            pmodel.load_state_dict(modweights[nm])
            pmodel.eval()
            # ----------------- ----------------------
            
            # ------------------------------------------------------
            # 12. Test the model separately to get accuracy by class
            # ------------------------------------------------------
            y_predicted,y_actual,test_loss = am.test_model(pmodel,test_loader,eparams['loss_fn'],
                                                           checkgpu=checkgpu,debug=False)
            lead_acc,class_acc = am.compute_class_acc(y_predicted,y_actual,nclasses,debug=True,verbose=False)
            
            
            modelaccs_all[exp,l,nm,:] = class_acc
            totalacc_all[exp,l,nm]    = lead_acc
            
            
            
            
            # ------------------------------------------------------
            # Old way :(((
            # ------------------------------------------------------
            with torch.no_grad():
                debug  = False
                device = torch.device('cpu')
                X_val  = X_subsets[1]
                X_val  = X_val.to(device)
                pmodel.eval()
                
                # -----------------
                # Evalute the model
                # -----------------
                y_pred_val = np.asarray([])
                y_valdt    = np.asarray([])
                
                for i,vdata in enumerate(test_loader):
                    
                    # Get mini batch
                    batch_x, batch_y = vdata     # For debugging: vdata = next(iter(val_loader))
                    batch_x = batch_x.to(device) # [batch x input_size]
                    batch_y = batch_y.to(device) # [batch x 1]
                    
                    # Make prediction and concatenate
                    batch_pred = pmodel(batch_x)  # [batch x class activation]
                    
                    # Convert predicted values
                    y_batch_pred = np.argmax(batch_pred.detach().cpu().numpy(),axis=1) # [batch,]
                    y_batch_lab  = batch_y.detach().cpu().numpy()            # Removed .squeeze() as it fails when batch size is 1
                    y_batch_size = batch_y.detach().cpu().numpy().shape[0]
                    if y_batch_size == 1:
                        y_batch_lab = y_batch_lab[0,:] # Index to keep as array [1,] instead of collapsing to 0-dim value
                    else:
                        y_batch_lab = y_batch_lab.squeeze()
                    if debug:
                        print("Batch Shape on iter %i is %s" % (i,y_batch_size))
                        print("\t the shape wihout squeeze is %s" % (batch_y.detach().cpu().numpy().shape[0]))
                    batch_acc    = np.sum(y_batch_pred==y_batch_lab)/y_batch_lab.shape[0]
                    #print("Acc. for batch %i is %.2f" % (i,batch_acc))
                    #print(y_batch_pred==y_batch_lab)
                    
                    # Store Predictions
                    y_pred_val = np.concatenate([y_pred_val,y_batch_pred])
                    if debug:
                        print("\ty_valdt size is %s" % (y_valdt.shape))
                        print("\ty_batch_lab size is %s" % (y_batch_lab.shape))
                    y_valdt = np.concatenate([y_valdt,y_batch_lab],axis=0)
                    if debug:
                        print("\tFinal shape is %s" % y_valdt.shape)
                    
                    # Save the actual and predicted values
                    #yvalpred.append(y_pred_val)
                    #yvallabels.append(y_valdt)
                    
                # -------------------------
                # Calculate Success Metrics
                # -------------------------
                # Calculate the total accuracy
                lead_acc  = (y_pred_val == y_valdt).sum()/y_pred_val.shape[0]
                #lead_acc = (yvalpred[l]==yvallabels[l]).sum()/ yvalpred[l].shape[0]
                totalacc_old[exp,l,nm] = lead_acc
                #total_acc.append(lead_acc)
                print("********Success rate********************")
                print("\t" +str(lead_acc*100) + r"%")
                
                # Calculate accuracy for each class
                class_total   = np.zeros([nclasses])
                class_correct = np.zeros([nclasses])
                val_size = y_pred_val.shape[0] #yvalpred[l].shape[0]
                for i in range(val_size):
                    #class_idx  = int(yvallabels[l][i])
                    #check_pred = yvallabels[l][i] == yvalpred[l][i]
                    class_idx  = int(y_valdt[i])
                    check_pred = y_valdt[i] == y_pred_val[i]
                    class_total[class_idx]   += 1
                    class_correct[class_idx] += check_pred 
                    #print("At element %i, Predicted result for class %i was %s" % (i,class_idx,check_pred))
                class_acc = class_correct/class_total
                classacc_old[exp,l,nm,:] = class_acc
                #acc_by_class.append(class_acc)
                print("********Accuracy by Class***************")
                for  i in range(nclasses):
                    print("\tClass %i : %03.3f" % (i,class_acc[i]*100) + "%\t" + "(%i/%i)"%(class_correct[i],class_total[i]))
                    print("****************************************")
                
                ydata_loop = [y_predicted,y_actual,y_pred_val,y_valdt]
                y_data_run.append(ydata_loop)
                
                            
            
            # End run loop >>>
        y_data_lead.append(y_data_run)
        # End lead loop >>>
    y_data.append(y_data_lead)
    # End Experiment loop >>>

#%% Visualize recomputed accuracy

def init_classacc_fig(leads,sp_titles=None):
    fig,axs=plt.subplots(1,3,constrained_layout=True,figsize=(18,4),sharey=True)
    if sp_titles is None:
        sp_titles=["AMV+","Neutral","AMV-"]
    for a,ax in enumerate(axs):
        ax.set_xlim([leads[0],leads[-1]])
        if len(leads) == 9:
            ax.set_xticks(leads)
        else:
            ax.set_xticks(leads[::3])
        ax.set_ylim([0,1])
        ax.set_yticks(np.arange(0,1.25,.25))
        ax.grid(True,ls='dotted')
        ax.minorticks_on()
        ax.set_title(sp_titles[a],fontsize=20)
        if a == 0:
            ax.set_ylabel("Accuracy")
        if a == 1:
            ax.set_xlabel("Prediction Leadtime (Years)")
    return fig,axs


fig,axs=init_classacc_fig(leads)

for iexp in range(nexps):
    inexpdict = inexps[iexp]
    
    for c in range(3):
        ax = axs[c]
        plotacc     = modelaccs_all[iexp,:,:,c].mean(1)
        plotacc_old = classacc_old[iexp,:,:,c].mean(1)
        # ax.plot(leads,plotacc,label=inexpdict["expname_long"] + " (newcalc)",lw=2.5,
        #         color=inexpdict["c"],marker=inexpdict["marker"])
        ax.plot(leads,plotacc_old,label=inexpdict["expname_long"] + " (old)",lw=2.5,
                color=inexpdict["c"],marker=inexpdict["marker"],ls='dashed')
        

ax = axs[1]
ax.legend()
savename = "%sTest_ACC_ALLDATA_debug.png" % (figpath)
plt.savefig(savename,dpi=200)



