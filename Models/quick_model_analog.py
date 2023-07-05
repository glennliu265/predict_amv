#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Quick Model Analog

Baseline forecast using the model analog approach

Copied train_NN_CESM1.py on 2023.07.05

Created on Fri May 26 07:41:31 2023

@author: gliu
"""


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset,Dataset

import sys
import numpy as np
import os
import time
import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

#%% Load custom packages and setup parameters

machine = 'Astraeus' # Indicate machine (see module packages section in pparams)

# Import packages specific to predict_amv
cwd     = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import train_cesm_params as train_cesm_params
import amv_dataloader as dl
import amvmod as am

import pamv_visualizer as pviz

# Load Predictor Information
bbox          = pparams.bbox
# Import general utilities from amv module
pkgpath = pparams.machine_paths[machine]['amv_path']
sys.path.append(pkgpath)
from amv import proc

# ============================================================
#%% User Edits vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ============================================================

# Set machine and import corresponding paths

# Set experiment directory/key used to retrieve params from [train_cesm_params.py]
expdir              = "FNN4_128_SingleVar_PaperRun"
eparams             = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Set some looping parameters and toggles
varnames            = ['SSH',]#"SST","SSS","SLP","NHFLX",]       # Names of predictor variables
leads               = np.arange(0,26,1)    # Prediction Leadtimes

# Other toggles
debug               = False                 # Set verbose outputs

# Save looping parameters into parameter dictionary
eparams['varnames'] = varnames
eparams['leads']    = leads


# Model Analog Settings
# Indicate k 
selected_k                     =np.arange(1,11,1)
thres                          =.37
    
# ============================================================
# End User Edits ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ============================================================

# ------------------------------------------------------------
# %% 01. Check for existence of experiment directory and create it
# ------------------------------------------------------------
allstart = time.time()

proc.makedir("../../CESM_data/"+expdir)
for fn in ("Metrics","Models","Figures"):
    proc.makedir("../../CESM_data/"+expdir+"/"+fn)

# Check if there is gpu
if checkgpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

# ----------------------------------------------
#%% 02. Data Loading, Classify Targets
# ----------------------------------------------

# Load data + target
load_dict                      = am.prepare_predictors_target(varnames,eparams,return_target_values=True,
                                                              return_nfactors=True,return_test_set=True)
data                           = load_dict['data']
target_class                   = load_dict['target_class']
target                         = load_dict['target']

# Get separate testing set
data_test                      = load_dict['data_test']
target_test                    = load_dict['target_test']
target_class_test              = load_dict['target_class_test']
nens_test                      = data_test.shape[1]


# Get necessary sizes
nchannels,nens,ntime,nlat,nlon = data.shape             
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN
nclasses                       = len(eparams['thresholds'])+1
nlead                          = len(leads)



# Debug messages
if debug:
    print("Loaded data of size: %s" % (str(data.shape)))

"""
# Output: 
    predictors       :: [channel x ens x year x lat x lon]
    target_class     :: [ens x year]
"""
# ----------------------------------------------------
# %% Retrieve a consistent sample if the option is set
# ----------------------------------------------------

if eparams["shuffle_trainsplit"] is False:
    print("Pre-selecting indices for consistency")
    output_sample=am.consistent_sample(data,target_class,leads,eparams['nsamples'],leadmax=leads.max(),
                          nens=None,ntime=None,
                          shuffle_class=eparams['shuffle_class'],debug=False)
    
    target_indices,target_refids,predictor_indices,predictor_refids = output_sample
else:
    print("Indices will be shuffled for each training iteration")
    target_indices     = None
    predictor_indices  = None
    target_refids      = None
    predictor_refids   = None

"""
Output

shuffidx_target  = [nsamples*nclasses,]        - Indices of target
predictor_refids = [nlead][nsamples*nclasses,] - Indices of predictor at each leadtime

tref --> array of the target years
predictor_refids --> array of the predictor refids
"""

#%% 







#%%
# class model_analog_amv:
#     def __init__(self,predictor,target):
#         super().__init__()
#         self.library     = predictor # [sample x channel x lat x lon]
#         self.target      = target    # [sample x time]
        
    
#     def calc_distance(x,y):
#         # y is assumed to be [1 x channel x lat x lon]
#         # Compute domain-averaged standard deviation for each variable
#         std_i_x       = np.nanmean(np.nanstd(x,0),(1,2)) # [channel] # std across sample, then mean lat/lon
#         std_i_y       = np.nanmean(np.nanstd(y,0),(1,2)) # [channel] 
#         dist    = np.nansum((x/std_i_x[None,:,None,None] - y/std_i_x[None,:,None,None])**2,(1,2,3)) # sum across channel, lat, lon
#         # nsamples      = y.shape[0]
#         # dists         = []
#         # for n in range(nsamples):
#         #     dist    = np.nansum((x/std_i_x[None,:,None,None] - y[[n],...]/std_i_x[None,:,None,None])**2,(1,2,3)) # sum across channel, lat, lon
#         #     dists.append(dist)
#         # dists = np.array(dists) # [sample, distances to train samples]
#         return dist # [distance to train samples]
        
#     def find_match(self,sample,ensemble_size):
#         self.distances = self.calc_distance(self.library,self.sample) # Compute the distances
#         # Get the [K] closest distances
#         np.argsort(self.distances
        
        
        
#         # sample = [[sample] x channel x lat x lon]
        

#         return x
    


def calc_distance(x,y):
    # y is assumed to be [1 x channel x lat x lon]
    # Compute domain-averaged standard deviation for each variable (following Ding et al. 2018)
    std_i_x       = np.nanmean(np.nanstd(x,0),(1,2)) # [channel] # std across sample, then mean lat/lon
    std_i_y       = np.nanmean(np.nanstd(y,0),(1,2)) # [channel] 
    nsamples_y    = y.shape[0]
    dists         = []
    # Simple RMS distance normalized by amt above
    for n in tqdm(range(nsamples_y)):
        dist    = np.nansum((x/std_i_x[None,:,None,None] - y[[n],...]/std_i_x[None,:,None,None])**2,(1,2,3)) # sum across channel, lat, lon
        dists.append(dist)
    dists = np.array(dists) # [sample, distances to train samples]
    return dists # [distance to train samples]

def calc_rms_distance(x,y):
    #std_i_x       = np.nanmean(np.nanstd(x,0),(1) # [channel] # std across sample, then mean lat/lon
    #std_i_y       = np.nanmean(np.nanstd(y,0),(1)) # [channel] 
    #dist    = np.nansum((x/std_i_x[None,:,] - y/std_i_x[None,:,None,None])**2,(1,2,3)) 
    return np.nansum((x-y[[0],...])**2,1)

def std_predictor(x):
    std_i_x       = np.nanmean(np.nanstd(x,0),(1,2))
    return x/std_i_x[None,:,None,None]


def calc_distance_sklearn(x,y):
    x = (std_predictor(X)).reshape(X.shape[0],np.prod(X.shape[1:]))
    y = (std_predictor(X_test)).reshape(X_test.shape[0],np.prod(X_test.shape[1:]))
    d = pairwise_distances(y,x,metric='euclidean')
    return d
# x = (std_predictor(X)).reshape(X.shape[0],np.prod(X.shape[1:]))
# y = (std_predictor(X_test)).reshape(X_test.shape[0],np.prod(X_test.shape[1:]))
# d = pairwise_distances(y,x,metric='euclidean')
# d = pairwise_distances(y,x,metric=calc_rms_distance)

# ------------------------------------------------------------
# %% Looping for runid
# ------------------------------------------------------------

use_target_value = False

# Print Message
print("Running [train_NN_CESM1.py] with the following settings:")
print("\tNetwork Type   : "+ eparams['netname'])
print("\tPredictor(s)   : "+str(varnames))
print("\tLeadtimes      : %i to %i" % (leads[0],leads[-1]))
print("\tMax Epochs     : " + str(eparams['max_epochs']))
print("\tEarly Stop     : " + str(eparams['early_stop']))
print("\t# Ens. Members : " + str(eparams['ens']))
print("\tDetrend        : " + str(eparams['detrend']))
print("\tShuffle        : " + str(eparams['shuffle_trainsplit']))

# ------------------------
# 04. Loop by predictor...
# ------------------------

vt = time.time()
predictors      = data[:,...] # Get selected predictor
predictors_test = data_test[:,...]

    
# Preallocate Evaluation Metrics...
train_loss_grid = [] #np.zeros((max_epochs,nlead))
test_loss_grid  = [] #np.zeros((max_epochs,nlead))
val_loss_grid   = [] 

train_acc_grid  = []
test_acc_grid   = [] # This is total_acc
val_acc_grid    = []

acc_by_class    = []
total_acc       = []
yvalpred        = []
yvallabels      = []
sampled_idx     = []
thresholds_all  = []
sample_sizes    = []

nk     = len(selected_k)
nleads  = len(leads)

classacc_all = np.zeros((nleads,nk,3,)) # [lead,classes]
totalacc_all = np.zeros((nleads,nk))

# -----------------------
# 07. Loop by Leadtime...
# -----------------------
for l,lead in tqdm.tqdm(enumerate(leads)):
    
    # --------------------------
    # 08. Apply lead/lag to data
    # --------------------------
    # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
    X,y_class           = am.apply_lead(predictors,target_class,lead,reshape=True,ens=eparams['ens'],tstep=ntime)
    X_test,y_class_test = am.apply_lead(predictors_test,target_class_test,lead,reshape=True,ens=nens_test,tstep=ntime)
    _,y_lib_values = am.apply_lead(predictors,target,lead,reshape=True,ens=eparams['ens'],tstep=ntime)
    
    #X_test = X.copy()
    #y_class_test = y_class.copy()
    #
    # %
    #
    
    # Normalize and compute distance (replace with scikitlearn to speed up)
    #dists = calc_distance(X,X_test) # [test_sample, distance_to_train_sample]
    dists = calc_distance_sklearn(X,X_test)
    
    # Get indices to sort by distance (closest to furthest)
    id_sort = np.argsort(dists,axis=-1)
    if debug:
        iii=2 #Print to check if distances were actually sorted...
        print(dists[iii,id_sort[iii,:]])
    
    
    predictions_byk=[]
    classacc_byk=[]
    totalacc_byk=[]
    for ik,k_closest in enumerate(selected_k):
        
        nsamples_test       = X_test.shape[0]
        nearest_classes_all = []
        predicted_class     = []
        for n in range(nsamples_test):
            # Grab corresponding k-closest cases from the library
            id_sel          = id_sort[n,:]
            
            if use_target_value:
                # Get the value
                
                nearest_values  = y_lib_values[id_sel,0][:k_closest]
                predicted_value = nearest_values.mean()
                
                # Classify base on threshold
                if predicted_value <= -thres:
                    pred = 2
                elif predicted_value > thres:
                    pred = 0
                else:
                    pred = 1
                predicted_class.append(pred)
                
            else: # Use target class
                nearest_classes = y_class[id_sel,0][:k_closest]
                nearest_classes_all.append(nearest_classes)
                
                # Take mean of class and use as prediction
                predicted_class.append(np.round(nearest_classes.mean()).astype(int))
                #print(dists[n,id_sel])
        predicted_class = np.array(predicted_class)[:,None] # [sample x 1]
        predictions_byk.append(predicted_class)
        
        # Compute accuracy
        output = am.compute_class_acc(predicted_class,y_class_test,3,verbose=False,debug=False)
        classacc_all[l,ik,:] = output[1]
        totalacc_all[l,ik] = output[0]
    
    # Make plot to see how distribution of predictions changes with # of analogs
    if debug:
        fig,axs = plt.subplots(2,7,constrained_layout=True,figsize=(12,4.5))
        for a in range(nk):
            ax = axs.flatten()[a]
            ax.hist(predictions_byk[a],bins=np.arange(-1,4,1))
            ax.set_title("k=%i"% np.arange(1,15,1)[a])
            ax.set_xlim([-1,3])
        plt.suptitle("%02iyr-Lead Model Analog Forecast for Predictors: %s" % (lead,str(varnames)))
        plt.savefig("%sModel_Analog_Forecast_byk_lead%02i.png" % (pparams.figpath,lead),dpi=150,bbox_inches='tight')
    
    # Look at accuracy
    
    
    # # Testing....
    # dist = dists[0,:]
    
    # # Get indices to sort by distance (closest to furthest)
    # id_sort = np.argsort(dist)
    
    # #  Grab corresponding Indices
    
    # nearest_classes = y_class[id_sort,0][:k_closest]
    # predicted_class = np.nanmean(nearest_classes)
    
#%% Save the output

savename = "../../CESM_data/%s/Model_Analog_Forecast_euclidean_usevalues%i.npz" % (expdir,use_target_value)

savedict = {
    "selected_k" : selected_k,
    "leads"      : leads,
    "classacc_all" : classacc_all,
    "totalacc_all" : totalacc_all,
    }
np.savez(savename,**savedict,allow_pickle=True)

#%% Visualize accuracy changes by k


fig,axs = pviz.init_classacc_fig(leads,)

for a in range(3):
    ax = axs[a]
    for k in range(nk):
        ax.plot(classacc_all[:,k,a],label="k=%i"% selected_k[k])
    ax.legend(ncol=4)
    
plt.suptitle("Model Analog Forecast",fontsize=24)
figname = "%sModel_Analog_Forecast_allvars_euclidean_usetarget%i.png" % (pparams.figpath,use_target_value)
plt.savefig(figname,dpi=150,bbox_inches='tight')


    
    
    