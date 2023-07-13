#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load 2 Models

Check normalized predictors and see how they impact the LRP output

Created on Fri Jun 23 17:28:54 2023

@author: gliu
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import nitime

from torch.utils.data import DataLoader, TensorDataset,Dataset

#%% Load custom packages and setup parameters

machine = 'Astraeus' # Indicate machine (see module packages section in pparams)

# Import packages specific to predict_amv
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import train_cesm_params as train_cesm_params
import amv_dataloader as dl
import amvmod as am

# Load Predictor Information
bbox          = pparams.bbox

# Import general utilities from amv module
pkgpath = pparams.machine_paths[machine]['amv_path']
sys.path.append(pkgpath)
from amv import proc

# Import LRP package
lrp_path = pparams.machine_paths[machine]['lrp_path']
sys.path.append(lrp_path)
from innvestigator import InnvestigateModel

# Load ML architecture information
nn_param_dict      = pparams.nn_param_dict

# ============================================================
#%% User Edits vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ============================================================

# Set machine and import corresponding paths

# Set experiment directory/key used to retrieve params from [train_cesm_params.py]
expdir              = "FNN4_128_SingleVar_PaperRun"
eparams             = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Processing Options
even_sample         = False
#standardize_input   = True # Set to True to standardize variance at each point

# Get some paths
datpath             = pparams.datpath
figpath             = pparams.figpath
dataset_name        = "CESM1"

# Set some looping parameters and toggles
varnames            = ["SSH",]      # Names of predictor variables
leads               = [25,]         # Indicate which leads to look at 
runids              = np.arange(0,100,1)    # Which runs to do

# LRP Parameters
innexp         = 2
innmethod      ='b-rule'
innbeta        = 0.1
innepsi        = 1e-6

# Other toggles
save_all_relevances = False                # True to save all relevances (~33G per file...)
checkgpu            = True                 # Set to true to check if GPU is availabl
debug               = False                 # Set verbose outputs
savemodel           = True                 # Set to true to save model weights

# Save looping parameters into parameter dictionary
eparams['varnames'] = varnames
eparams['leads']    = leads
eparams['runids']   = runids

#%% Functions


def compute_relevances_lead(all_predictors,target_class,lead,eparams,modweights_lead,modlist_lead,
                            nn_param_dict,innexp,innmethod,innbeta,innepsi,
                            even_sample=False,debug=False,checkgpu=False,calculate_lrp=True):
    """
    Loop through a series of datasets in all_predictors and compute both the relevances and test accuracies
    
    all_predictors [dataset][channel x ens x time x lat x lon]
    target_class   [ens x time]
    modlist_lead   [lead][runs]
    modweights_lead [lead][runs]
    
    """
    
    # Get dimensions
    nloop               = len(all_predictors)
    nchannels,nens,ntime,nlat,nlon = all_predictors[0].shape
    nruns               = len(modlist_lead[0])
    nclasses            = len(eparams['thresholds']) + 1
    
    relevances_all      = []
    predictors_all_lead = []
    predictions_all     = []
    targets_all         = []
    test_acc_byclass    = np.zeros((nloop,nruns,nclasses)) # [experiment, runid, classes]
    for ii in range(nloop):
        vt = time.time()
        predictors= all_predictors[ii]
        
        # ===================================
        # I. Data Prep
        # ===================================
        
        # IA. Apply lead/lag to data
        # --------------------------
        # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
        X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=nens,tstep=ntime)
        
        # ----------------------
        # IB. Select samples
        # ----------------------
        _,class_count = am.count_samples(None,y_class)
        if even_sample:
            eparams['nsamples'] = int(np.min(class_count))
            print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
            y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle_class'])
        
        # ----------------------
        # IC. Flatten inputs for FNN
        # ----------------------
        if "FNN" in eparams['netname']:
            ndat,nchannels,nlat,nlon = X.shape
            inputsize                = nchannels*nlat*nlon
            X_in                     = X.reshape(ndat,inputsize)
        
        # -----------------------------
        # ID. Place data into a data loader
        # -----------------------------
        # Convert to Tensors
        X_torch = torch.from_numpy(X_in.astype(np.float32))
        y_torch = torch.from_numpy(y_class.astype(np.compat.long))
        
        # Put into pytorch dataloaders
        test_loader = DataLoader(TensorDataset(X_torch,y_torch), batch_size=eparams['batch_size'])
        
        # Preallocate
        relevances_byrun  = []
        predictions_byrun = []
        targets_byrun     = []
        
        # --------------------
        # 05. Loop by runid...
        # --------------------
        for nr in tqdm(range(nruns)):
            
            # =====================
            # II. Rebuild the model
            # =====================
            # Get the models (now by leadtime)
            modweights = modweights_lead[0][nr]
            modlist    = modlist_lead[0][nr]
            
            # Rebuild the model
            pmodel = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
            
            # Load the weights
            pmodel.load_state_dict(modweights)
            pmodel.eval()
            
            # =======================================================
            # III. Test the model separately to get accuracy by class
            # =======================================================
            y_predicted,y_actual,test_loss = am.test_model(pmodel,test_loader,eparams['loss_fn'],
                                                           checkgpu=checkgpu,debug=False)
            lead_acc,class_acc = am.compute_class_acc(y_predicted,y_actual,nclasses,debug=debug,verbose=False)
            
            test_acc_byclass[ii,nr,:] = class_acc.copy()
            
            # Save variables
            predictions_byrun.append(y_predicted)
            if nr == 0:
                targets_byrun.append(y_actual)
            
            # ===========================
            # IV. Perform LRP
            # ===========================
            if calculate_lrp:
                nsamples_lead = len(y_actual)
                inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
                                                  epsilon=innepsi,
                                                  method=innmethod,
                                                  beta=innbeta)
                model_prediction, sample_relevances = inn_model.innvestigate(in_tensor=X_torch)
                model_prediction                    = model_prediction.detach().numpy().copy()
                sample_relevances                   = sample_relevances.detach().numpy().copy()
                if "FNN" in eparams['netname']:
                    predictor_test    = X_torch.detach().numpy().copy().reshape(nsamples_lead,nlat,nlon)
                    sample_relevances = sample_relevances.reshape(nsamples_lead,nlat,nlon) # [test_samples,lat,lon] 
                
                # Save Variables
                if nr == 0:
                    predictors_all_lead.append(predictor_test) # Predictors are the same across model runs
                relevances_byrun.append(sample_relevances)
            
            # Clear some memory
            del pmodel
            torch.cuda.empty_cache()  # Save some memory
            
            # End Run Loop >>>
        relevances_all.append(relevances_byrun)
        predictions_all.append(predictions_byrun)
        print("\nCompleted training for lead of %i in %.2fs" % (lead,time.time()-vt))
        # End Data Loop >>>
    out_dict = {
        "relevances"    : relevances_all,
        "predictors"    : predictors_all_lead,
        "predictions"   : predictions_all,
        "targets"       : targets_byrun,
        "class_acc"     : test_acc_byclass
        }
    return out_dict

def composite_relevances_predictors(relevances_all,predictors_all,targets_all,nclasses=3):
    # relevances_all[dataset][nrun] (same for predictors_all)
    # targets_all[0][samples]
    # 
    nloop   = len(relevances_all)
    nmodels = len(relevances_all[0])
    
    
    st_rel_comp          = time.time()
    
    relevance_composites = np.zeros((nloop,nmodels,nclasses,nlat,nlon)) * np.nan     # [data x model x class x lat x lon]
    relevance_variances  = relevance_composites.copy()                    # [data x model x class x lat x lon]
    relevance_range      = relevance_composites.copy()                    # [data x model x class x lat x lon]
    predictor_composites = np.zeros((nloop,nclasses,nlat,nlon)) * np.nan             # [data x class x lat x lon]
    predictor_variances  = predictor_composites.copy()                    # [data x class x lat x lon]
    ncorrect_byclass     = np.zeros((nloop,nmodels,nclasses))                        # [data x model x class

    for l in range(nloop):
        for nr in tqdm(range(nmodels)):
            predictions_model = predictions_all[l][nr] # [sample]
            relevances_model  = relevances_all[l][nr]  # [sample x lat x lon]
            
            for c in range(nclasses):
                
                # Get correct indices
                class_indices                   = np.where(targets_all[0] == c)[0] # Sample indices of a particular class
                correct_ids                     = np.where(targets_all[0][class_indices] == predictions_model[class_indices])
                correct_pred_id                 = class_indices[correct_ids] # Correct predictions to composite over
                ncorrect                        = len(correct_pred_id)
                ncorrect_byclass[l,nr,c]        = ncorrect
                
                if ncorrect == 0:
                    continue # Set NaN to model without any results
                # Make Composite
                correct_relevances               =  relevances_model[correct_pred_id,...]
                relevance_composites[l,nr,c,:,:] =  correct_relevances.mean(0)
                relevance_variances[l,nr,c,:,:]  =  correct_relevances.var(0)
                relevance_range[l,nr,c,:,:]      =  correct_relevances.max(0) - correct_relevances.min(0)
                
                # Make Corresponding predictor composites
                correct_predictors               = predictors_all_lead[0][correct_pred_id,...]
                predictor_composites[l,c,:,:]    = correct_predictors.mean(0)
                predictor_variances[l,c,:,:]     = correct_predictors.var(0)
    print("Saved Relevance Composites in %.2fs" % (time.time()-st_rel_comp))
    
    out_composites = {
        "relevance_composites":relevance_composites,
        "relevance_variances" :relevance_variances,
        "relevance_range"     :relevance_range,
        "predictor_composites":predictor_composites,
        "predictor_variances" :predictor_variances,
        "ncorrect_byclas"     :ncorrect_byclass,
        }
    return out_composites

# -----------------------------------
# %% Get some other needed parameters
# -----------------------------------

# Ensemble members
ens_all        = np.arange(0,42)
ens_train_val  = ens_all[:eparams['ens']]
ens_test       = ens_all[eparams['ens']:]
nens_test      = len(ens_test)

# ============================================================
#%% Load the data 
# ============================================================
# Copied segment from train_NN_CESM1.py

# Load data + target
load_dict                      = am.prepare_predictors_target(varnames,eparams,return_nfactors=True,
                                                              return_test_set=True)
#data                           = load_dict['data']
#target_class                   = load_dict['target_class']

# Pick just the testing set
data                           = load_dict['data_test']
target_class                   = load_dict['target_class_test']

# Get necessary sizes
nchannels,nens,ntime,nlat,nlon = data.shape             
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN
nclasses                       = len(eparams['thresholds'])+1
nlead                          = len(leads)

# Count Samples...
am.count_samples(None,target_class)

# --------------------------------------------------------
#%% Option to standardize input to test effect of variance
# --------------------------------------------------------

"""
Modified original script so that we have data and data_std

"""

# Compute standardizing factor (and save)
std_vars = np.std(data,(1,2)) # [variable x lat x lon]
for v in range(nchannels):
    savename = "%s%s/%s_standardizing_factor_ens%02ito%02i.npy" % (datpath,expdir,varnames[v],ens_test[0],ens_test[-1])
    np.save(savename,std_vars[v,:,:])

# Apply standardization
data_std = data / std_vars[:,None,None,:,:] 
data_std[np.isnan(data_std)] = 0
std_vars_after = np.std(data_std,(1,2))
check =  np.all(np.nanmax(np.abs(std_vars_after)) < 2)
assert check, "Standardized values are not below 2!"


#%% 

"""

General Procedure

 1. Load data and subset to test set
 2. Looping by variable...
     3. Load the model weights and metrics
     4. 
     
"""

all_predictors = [data[[0],...],data_std[[0],...]]
data_names     = ("Raw","Temporally Standardized")

# Just take the first index, since we are only looking at one lead/variable
lead       = leads[0]
varname    = varnames[0]
predictors = data[[0],...]


# Indicate which leads to look at
vt      = time.time()

# ================================
#% 1. Load model weights + Metrics
# ================================
# Get the model weights [lead][run]
modweights_lead,modlist_lead=am.load_model_weights(datpath,expdir,leads,varname)
nmodels = len(modweights_lead[0])


# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
#%% Try just for one model first ( THIS SECTION IS WILL BE COPIED BELOW :(
# Need to turn into a function))

outdict_original = compute_relevances_lead(all_predictors,target_class,lead,eparams,modweights_lead,modlist_lead,
                            nn_param_dict,innexp,innmethod,innbeta,innepsi,
                            even_sample=even_sample,debug=debug,checkgpu=checkgpu,calculate_lrp=True)

relevances_all      = outdict_original['relevances']
predictors_all_lead = outdict_original['predictors']
predictions_all     = outdict_original['predictions']
targets_all         = outdict_original['targets']
test_acc_byclass    = outdict_original['class_acc']



#%% Examine if there is a difference

# =============================================================================================================================
#%% Composite the relevances (can look at single events later, it might actually be easier to write a separate script for that)
# =============================================================================================================================
# the purpose here is to get some quick, aggregate metrics

# Need to add option to cull models in visualization script...
st_rel_comp          = time.time()
composites_ori = composite_relevances_predictors(relevances_all,predictors_all_lead,targets_all,nclasses=3)
relevance_composites = composites_ori['relevance_composites']
predictor_composites  = composites_ori['predictor_composites']
print("Saved Relevance Composites in %.2fs" % (time.time()-st_rel_comp))
        
#%% Visualize relevance composites differences between normalized and unnormalized data

lon = load_dict['lon']
lat = load_dict['lat']


# Nneed to cimposite and relevan
fig,axs = plt.subplots(2,4,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,4.5))

for ii in range(2):
    for c in range(4):
        
        ax =axs[ii,c]
        ax.set_extent(bbox)
        ax.coastlines()
        if c < 3:
            plotvar = relevance_composites[ii,:,c,:,:].mean(0)
            title   = pparams.classes[c]
        else:
            plotvar = relevance_composites[ii,:,:,:,:].mean(1).mean(0)
            title   = "Class Mean"
        plotvar = plotvar / np.nanmax(np.abs(plotvar))
        pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-1,vmax=1,cmap="RdBu_r")
        
        if ii == 0:
            ax.set_title(title)
        
        if c == 0:
            ax.text(-0.05, 0.55, data_names[ii], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=12)
cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.05)
cb.set_label("Normalized Relevance")
plt.suptitle("Predicting AMV lead=%i years (%s Predictor)" % (lead,varname))

savename ="%sNormalizing_Effect_%s_lead%02iyears.png" % (figpath,varname,lead)
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Examine relevance histogram and select a threshold

thres_rel   = 0.6
fig,axs = plt.subplots(2,3,figsize=(8,4.5),constrained_layout=True)
bins    = np.arange(0,1.1,.1)

for ii in range(2):
    for c in range(3):
        
        ax =axs[ii,c]
        
        if ii == 0:
            ax.set_title(pparams.classes[c])
        
        if c == 0:
            ax.text(-0.25, 0.55, data_names[ii], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=12)

        plotvar = relevance_composites[ii,:,c,:,:].mean(0)
        plotvar = (plotvar / np.nanmax(np.abs(plotvar))).flatten()
        count_above = (plotvar > thres_rel).sum()
        
        ax.hist(plotvar,bins=bins,edgecolor="w")
        ax.axvline([thres_rel],ls='dashed',color="k")
        ax.set_title("Count Above %.2f: %i" % (thres_rel,count_above))


savename ="%sRelevanceAblation_Histogram_%s_lead%02iyears_thresrel%.02f.png" % (figpath,varname,lead,thres_rel)
plt.savefig(savename,dpi=150,bbox_inches="tight")
#%% Apply mask to data to see which regions remain

lon = load_dict['lon']
lat = load_dict['lat']


# Nneed to cimposite and relevance

fig,axs = plt.subplots(2,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,4.5))

for ii in range(2):
    for c in range(3):
        
        ax =axs[ii,c]
        ax.set_extent(bbox)
        ax.coastlines()
        plotvar = relevance_composites[ii,:,:,:,:].mean(1).mean(0)
        plotvar = plotvar / np.nanmax(np.abs(plotvar))
        
        #relevance_mask = np.where(plotvar.flatten()>thres_rel)[0]
        
        plotvar[plotvar < thres_rel] = 0
        pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-1,vmax=1,cmap="RdBu_r")
        
        
        if ii == 0:
            ax.set_title(pparams.classes[c])
        
        if c == 0:
            ax.text(-0.05, 0.55, data_names[ii], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=12)
cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.05)
cb.set_label("Normalized Relevance")
plt.suptitle("Predicting AMV lead=%i years (%s Predictor)" % (lead,varname))

savename ="%sRelevanceAblation_%s_lead%02iyears_thresrel%.02f.png" % (figpath,varname,lead,thres_rel)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Choose the dataset to use and replace the points with synthetic data

ii            = 0 # Let's use the un-normalized dataset
sel_c         = 2 # Select the positive class
select_random = True

# Get the predictor to use
synth_name   = ["zeros","white noise","red noise"]
synth_colors = ["gray","cornflowerblue","red"]
predictor_in = predictors.reshape(1,nens,ntime,nlat*nlon)

# Make mask based on selected dataset and class
plotvar = relevance_composites[ii,:,sel_c,:,:].mean(0)
plotvar = plotvar / np.nanmax(np.abs(plotvar))
sel_pts =  np.where(plotvar.flatten() > thres_rel)[0]
npts = len(sel_pts)
if select_random:
    sel_pts = np.random.choice(np.arange(nlat*nlon),size=npts) # Randomly select some points
    sel_pts_ori = np.where(plotvar.flatten() > thres_rel)[0]
    

# Create Synthetic Data
dropped_points = []
synthetic_data = np.zeros((3,nens,ntime,nlat*nlon,))
for pt in tqdm(range(npts)):
    
    idx = sel_pts[pt]
    
    for e in range(nens):
        
        ts_in     = predictor_in[0,e,:,idx] # Get the timeseries
        
        # Avoid land points
        if select_random:
            while np.all(predictor_in[0,e,:,idx]==0):
                idx = np.random.choice(np.arange(nlat*nlon),size=1)[0] #+=1
                # if idx > nlat*nlon-1:
                #     idx = 0
                ts_in     = predictor_in[0,e,:,idx]
        
            
        # Estimate AR1 coefficient using yule-walker
        coef,sigma=nitime.algorithms.AR_est_YW(ts_in,1)
        
        # Make red noise timeseries
        X_ar,noise,aph=nitime.utils.ar_generator(ntime,sigma=sigma,coefs=coef)
        
        synthetic_data[2,e,:,idx] = X_ar.copy()
        
        # Make white noise timeseries
        synthetic_data[1,e,:,idx] = np.random.normal(0,np.std(ts_in),ntime)
    dropped_points.append(idx)
        
if debug:
    pt = 22
    idx = sel_pts[pt]
    fig,ax = plt.subplots(1,1,figsize=(6,3),constrained_layout=True)
    ax.plot(predictor_in[0,e,:,idx],label="Original Timeseries",color="k")
    for zz in range(3):
        ax.plot(synthetic_data[zz,e,:,idx],label=synth_name[zz],color=synth_colors[zz])
    ax.legend(ncol=3)
    
    
    savename ="%sRelevanceAblation_%s_lead%02iyears_thresrel%.02f_sampletimeseries_pt%i.png" % (figpath,varname,lead,thres_rel,pt)
    plt.savefig(savename,dpi=150,bbox_inches="tight")
    

synthetic_data = synthetic_data.reshape(3,nens,ntime,nlat,nlon)

#%% Visualize which points where randomly dropped

if select_random:
    idlat,idlon=np.unravel_index(dropped_points,(nlat,nlon))
    
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,4.5))
    ax.set_extent(bbox)
    ax.coastlines()
    plotvar = relevance_composites[0,:,sel_c,:,:].mean(0)
    plotvar = plotvar / np.nanmax(np.abs(plotvar))
    ax.scatter(lon[idlon],lat[idlat])
            
    savename ="%sRelevanceAblation_Randompoints_%s_lead%02iyears_thresrel%.02f_sampletimeseries_pt%i.png" % (figpath,varname,lead,thres_rel,pt)
    plt.savefig(savename,dpi=150,bbox_inches="tight")

    pt_ids         = [dropped_points,sel_pts,sel_pts_ori]
    relevances_sel = []
    ptsel_names    = ["Random Dropped + Correction","Random Dropped","Original Relevant"]
    
    for zzz in range(3):
        idlat,idlon=np.unravel_index(pt_ids[zzz],(nlat,nlon))
        rzzz = relevance_composites[0,:,c,idlat,idlon]
        relevances_sel.append(rzzz)
        print("Mean relevance of %s points is %f" % (ptsel_names[zzz],rzzz.mean()))
    
    fig,axs = plt.subplots(3,1,sharey=False,sharex=True)
    for zzz in range(3):
        ax = axs[zzz]
        ax.set_title(ptsel_names[zzz])
        ax.hist(relevances_sel[zzz].flatten(),bins=10)
    

#%% Below this is copied code from above ^^^^^
# Need to turn into a function))

skipzero = True
# ===================================
# I. Data Prep
# ===================================

# IA. Apply lead/lag to data
# --------------------------
# X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
X,y_class = am.apply_lead(synthetic_data,target_class,lead,reshape=True,ens=nens_test,tstep=ntime)

# ----------------------
# IB. Select samples
# ----------------------
_,class_count = am.count_samples(None,y_class)
if even_sample:
    eparams['nsamples'] = int(np.min(class_count))
    print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
    y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle_class'])


# ----------------------
# IC. Flatten inputs for FNN
# ----------------------
if "FNN" in eparams['netname']:
    ndat,nchannels,nlat,nlon = X.shape
    inputsize                = nchannels*nlat*nlon
    outsize                  = nclasses
    X_in                     = X.reshape(ndat,inputsize)
        

# -----------------------------
# ID. Place data into a data loader
# -----------------------------
# Convert to Tensors
X_torch = torch.from_numpy(X_in.astype(np.float32))
y_torch = torch.from_numpy(y_class.astype(np.compat.long))

# Put into pytorch dataloaders
test_loader = DataLoader(TensorDataset(X_torch,y_torch), batch_size=eparams['batch_size'])

# ----------------------------------
#%Compute and composite relevances

relevances_all      = []
predictors_all_lead = []
predictions_all     = []
targets_all         = []

test_acc_byclass_synth = np.zeros((3,len(runids),3)) # [experiment, runid, classes]

for ii in range(3):
    if (ii == 0) and skipzero:
        continue # Skip zero
    
    predictors_input= synthetic_data[[ii],...]

    # ===================================
    # I. Data Prep
    # ===================================
    
    # IA. Apply lead/lag to data
    # --------------------------
    # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
    X,y_class = am.apply_lead(predictors_input,target_class,lead,reshape=True,ens=nens_test,tstep=ntime)
    
    # ----------------------
    # IB. Select samples
    # ----------------------
    _,class_count = am.count_samples(None,y_class)
    if even_sample:
        eparams['nsamples'] = int(np.min(class_count))
        print("Using %i samples, the size of the smallest class" % (eparams['nsamples']))
        y_class,X,shuffidx = am.select_samples(eparams['nsamples'],y_class,X,verbose=debug,shuffle=eparams['shuffle_class'])

    
    # ----------------------
    # IC. Flatten inputs for FNN
    # ----------------------
    if "FNN" in eparams['netname']:
        ndat,nchannels,nlat,nlon = X.shape
        inputsize                = nchannels*nlat*nlon
        outsize                  = nclasses
        X_in                     = X.reshape(ndat,inputsize)
    
    # -----------------------------
    # ID. Place data into a data loader
    # -----------------------------
    # Convert to Tensors
    X_torch = torch.from_numpy(X_in.astype(np.float32))
    y_torch = torch.from_numpy(y_class.astype(np.compat.long))
    
    # Put into pytorch dataloaders
    test_loader = DataLoader(TensorDataset(X_torch,y_torch), batch_size=eparams['batch_size'])
    
    # Preallocate
    relevances_byrun  = []
    predictions_byrun = []
    targets_byrun     = []
    
    # --------------------
    # 05. Loop by runid...
    # --------------------
    for nr,runid in tqdm(enumerate(runids)):
        rt = time.time()
        
        # =====================
        # II. Rebuild the model
        # =====================
        # Get the models (now by leadtime)
        modweights = modweights_lead[0][nr]
        modlist    = modlist_lead[0][nr]
        
        # Rebuild the model
        pmodel = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
        
        # Load the weights
        pmodel.load_state_dict(modweights)
        pmodel.eval()
        
        # =======================================================
        # III. Test the model separately to get accuracy by class
        # =======================================================
        y_predicted,y_actual,test_loss = am.test_model(pmodel,test_loader,eparams['loss_fn'],
                                                       checkgpu=checkgpu,debug=False)
        lead_acc,class_acc = am.compute_class_acc(y_predicted,y_actual,nclasses,debug=debug,verbose=False)
        
        
        test_acc_byclass_synth[ii,nr,:] = class_acc.copy()
        
        # Save variables
        predictions_byrun.append(y_predicted)
        if nr == 0:
            targets_byrun.append(y_actual)
        
        # ===========================
        # IV. Perform LRP
        # ===========================
        nsamples_lead = len(y_actual)
        inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
                                          method=innmethod,
                                          beta=innbeta)
        model_prediction, sample_relevances = inn_model.innvestigate(in_tensor=X_torch)
        model_prediction                    = model_prediction.detach().numpy().copy()
        sample_relevances                   = sample_relevances.detach().numpy().copy()
        if "FNN" in eparams['netname']:
            predictor_test    = X_torch.detach().numpy().copy().reshape(nsamples_lead,nlat,nlon)
            sample_relevances = sample_relevances.reshape(nsamples_lead,nlat,nlon) # [test_samples,lat,lon] 
        
        # Save Variables
        if nr == 0:
            predictors_all_lead.append(predictor_test) # Predictors are the same across model runs
        relevances_byrun.append(sample_relevances)
        
        # Clear some memory
        del pmodel
        torch.cuda.empty_cache()  # Save some memory
        
        #print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
        # End Lead Loop >>>
    
    relevances_all.append(relevances_byrun)
    predictions_all.append(predictions_byrun)
    print("\nCompleted training for lead of %i in %.2fs" % (lead,time.time()-vt))

#%% Look at the change in skill

method  = 2
remove_singleguesser = True
fig,axs = plt.subplots(3,1,constrained_layout=True)

for a in range(3):
    ax = axs[a]
    
    method_acc = test_acc_byclass_synth[method,:,a]
    
    perf_acc = np.where((method_acc == 0) | (method_acc == 1))[0]
    diff     = method_acc - test_acc_byclass[0,:,a]
    
    if remove_singleguesser:
        ax.bar(runids[perf_acc],diff[perf_acc],color="red")
        diff[perf_acc] = np.nan
        n_exclude = len(perf_acc)
    
    ax.bar(runids,diff)
    ax.axhline([0],ls='solid',color="k")
    ax.set_title("%s, Mean Diff: %.2f" % (pparams.classes[a],np.nanmean(diff)*100)+"%" + " (dropped=%i)"%n_exclude)
    
    ax.set_ylim([-.75,.75])

plt.suptitle("Change in Test Accuracy Using %s Data (Relevance Threshold %.2f)" % (synth_name[method],thres_rel))

figname = "%sRelevanceAblation_AccChange_%s_%s_relthres%.2f_class%s_selrand%i.png" % (figpath,expdir,synth_name[method].replace(" ",""),thres_rel,
                                                                                      pparams.classes[sel_c],select_random)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Visualize the accuracies as histograms
c = 0

fig,axs = plt.subplots(4,1,constrained_layout=True,sharex=True,figsize=(8,8))

bins = np.arange(0,1.05,.05)
for a in range(4):
    
    ax = axs[a]
    if a < 3:
        indata = test_acc_byclass_synth[a,:,c]
        title=synth_name[a]
    else:
        indata = test_acc_byclass[0,:,c]
        title="original"
        
    ax.hist(indata.flatten(),bins=bins)
    ax.axvline(indata.mean(),label="Mean=%.2f" % (indata.mean()*100)+"%",color="k")
    ax.legend()
    ax.set_title(title)

#%%

ii    = 2
c     = 0
pfm   = np.where(test_acc_byclass_synth[ii,:,c]==1)

#predictions_all[ii][pfm[0]] # [dataset][runs][predictions]


predictions_new = []
for pf in pfm[0]:
    predictions_new.append(predictions_all[ii][pf])

#%%



#%% Convert this into a loop



























# ===================================================
#%% Save Relevance Output
# ===================================================
#Save as relevance output as a dataset

st_rel = time.time()

lat = load_dict['lat']
lon = load_dict['lon']

# Save variables
save_vars      = [relevance_composites,relevance_variances,relevance_range,predictor_composites,predictor_variances,ncorrect_byclass]
save_vars_name = ['relevance_composites','relevance_variances','relevance_range','predictor_composites','predictor_variances',
                  'ncorrect_byclass']

# Make Coords
coords_relevances = {"lead":leads,"runid":runids,"class":pparams.classes,"lat":lat,"lon":lon}
coords_preds      = {"lead":leads,"class":pparams.classes,"lat":lat,"lon":lon}
coords_counts     = {"lead":leads,"runid":runids,"class":pparams.classes}

# Convert to dataarray and make encoding dictionaries
ds_all    = []
encodings = {}
for sv in range(len(save_vars)):
    
    svname = save_vars_name[sv]
    if "relevance" in svname:
        coord_in = coords_relevances
    elif "predictor" in svname:
        coord_in = coords_preds
    elif "ncorrect" in svname:
        coord_in = coords_counts
    
    da = xr.DataArray(save_vars[sv],dims=coord_in,coords=coord_in,name=svname)
    encodings[svname] = {'zlib':True}
    ds_all.append(da)
    
# Merge into dataset
ds_all = xr.merge(ds_all)

# Save Relevance data
outname    = "%s%s/Metrics/Test_Metrics_%s_%s_evensample%i_relevance_maps.nc" % (datpath,expdir,dataset_name,varname,even_sample)
if standardize_input:
    outname = proc.addstrtoext(outname,"_standardizeinput")
ds_all.to_netcdf(outname,encoding=encodings)
print("Saved Relevances to %s in %.2fs" % (outname,time.time()-st_rel))
# ===================================================
#%% Save accuracy and prediction data
# ===================================================
st_acc = time.time()

if save_all_relevances:
    print("Saving all relevances!")
    save_vars      = [relevances_all,predictor_all,]
    save_vars_name = ["relevances","predictors",]
    for sv in range(len(save_vars)):
        outname    = "%s%s/Metrics/Test_Metrics_%s_%s_evensample%i_%s.npy" % (datpath,expdir,dataset_name,varname,even_sample,save_vars_name[sv])
        np.save(outname,save_vars[sv],allow_pickle=True)
        print("Saved %s to %s in %.2fs" % (save_vars_name[sv],outname,time.time()-st_acc))

save_vars         = [total_acc_all,class_acc_all,predictions_all,targets_all,ens_test,leads,runids]
save_vars_name    = ["total_acc","class_acc","predictions","targets","ensemble","leads","runids"]
metrics_dict      = dict(zip(save_vars_name,save_vars))
outname           = "%s%s/Metrics/Test_Metrics_%s_%s_evensample%i_accuracy_predictions.npz" % (datpath,expdir,dataset_name,varname,even_sample)
if standardize_input:
    outname = proc.addstrtoext(outname,"_standardizeinput")
np.savez(outname,**metrics_dict,allow_pickle=True)
print("Saved Accuracy and Predictions to %s in %.2fs" % (outname,time.time()-st_acc))

print("Completed calculating metrics for %s in %.2fs" % (varname,time.time()-vt))







#%% Examine the relevance "gain"


norm_data = True
lon = load_dict['lon']
lat = load_dict['lat']


# Nneed to cimposite and relevan
fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,4.5))


for c in range(3):
    
    ax =axs[c]
    ax.set_extent(bbox)
    ax.coastlines()
    plotvar = relevance_composites[1,:,c,:,:].mean(0) - relevance_composites[0,:,c,:,:].mean(0)
    
    if norm_data:
        plotvar = plotvar / np.nanmax(np.abs(plotvar))
        pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-1,vmax=1,cmap="RdBu_r")
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-1e-2,vmax=1e-2,cmap="RdBu_r")
    
    if ii == 0:
        ax.set_title(pparams.classes[c])
    
    if c == 0:
        ax.text(-0.05, 0.55, data_names[ii], va='bottom', ha='center',rotation='vertical',
                rotation_mode='anchor',transform=ax.transAxes,fontsize=12)
cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.05)
cb.set_label("Relevance Gain")
plt.suptitle("Predicting AMV lead=%i years (%s Predictor)" % (lead,varname))

#savename ="%sNormalizing_Effect_%s_lead%02iyears.png" % (figpath,varname,lead)



#%% Check accuracy by cclass for both cases


fig,axs = plt.subplots(3,1,)


for c in range(3):
    for ii in range(2):
        plotvar = test_acc_byclass[ii,:,c].mean(0)
        print("Class Acc for predicting %s for %s is %.3f" % (pparams.classes[c],data_names[ii],plotvar))

#%% Some EOFs fun from paleocamp

from eofs.standard import Eof

# Select data
ii = 1
c  = 0
data_in = relevance_composites[ii,:,c,:,:] #Positive AMV

# Apply Latitude Weights
wgts    = np.cos(np.deg2rad(lat))
data_in_wgt = data_in * wgts[None,:,None]

# Create solver and extract PCs
solver = Eof(data_in_wgt)
pcs    = solver.pcs(npcs=10,pcscaling=1)
eof    = solver.eofsAsCorrelation(neofs=10)
expvar = solver.varianceFraction(neigs=10)
north  = solver.northTest(neigs=10,vfscaled="true") 


#%% Look at the eofs


fig,axs = plt.subplots(2,5,subplot_kw={'projection':ccrs.PlateCarree()},
                       figsize=(18,6),constrained_layout=True)
for a in range(10):
    ax = axs.flatten()[a]
    pcm = ax.pcolormesh(lon,lat,eof[a,...],vmin=0.5,vmax=1,cmap="jet")
    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.025)
    ax.set_title("EOF %i, VarExp=%.2f" % (a+1,expvar[a]*100) + "%")
plt.suptitle("EOF Across Network Composites")



fig,axs = plt.subplots(10,1,constrained_layout=True,figsize=(12,33))
for a in range(10):
    ax = axs.flatten()[a]
    ax.plot(np.arange(1,101),pcs[:,a],label="EOF %i" % (a+1))
    ax.legend()








