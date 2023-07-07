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
expdir              = "FNN4_128_SingleVar_PaperRun_stdspace"
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

# Other toggles
save_all_relevances = False                # True to save all relevances (~33G per file...)
checkgpu            = True                 # Set to true to check if GPU is availabl
debug               = False                 # Set verbose outputs
savemodel           = True                 # Set to true to save model weights

# Save looping parameters into parameter dictionary
eparams['varnames'] = varnames
eparams['leads']    = leads
eparams['runids']   = runids

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
data_names     = ("Raw","Spatially Standardized")

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



#%% Try just for one model first

# ===================================
# I. Data Prep
# ===================================

# IA. Apply lead/lag to data
# --------------------------
# X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=nens_test,tstep=ntime)

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



#%% Compute and composite relevances

relevances_all      = []
predictors_all_lead = []
predictions_all     = []
targets_all         = []

test_acc_byclass = np.zeros((2,len(runids),3)) # [experiment, runid, classes]

for ii in range(2):
    
    predictors= all_predictors[ii]

    # ===================================
    # I. Data Prep
    # ===================================
    
    # IA. Apply lead/lag to data
    # --------------------------
    # X -> [samples x channel x lat x lon] ; y_class -> [samples x 1]
    X,y_class = am.apply_lead(predictors,target_class,lead,reshape=True,ens=nens_test,tstep=ntime)

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
        
        
        test_acc_byclass[ii,nr,:] = class_acc.copy()
        
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
    
    
#%% Examine if ther eis a difference




# =============================================================================================================================
#%% Composite the relevances (can look at single events later, it might actually be easier to write a separate script for that)
# =============================================================================================================================
# the purpose here is to get some quick, aggregate metrics

# Need to add option to cull models in visualization script...
st_rel_comp          = time.time()
targets_all          = targets_byrun

relevance_composites = np.zeros((2,nmodels,3,nlat,nlon)) * np.nan # [lead x model x class x lat x lon]
relevance_variances  = relevance_composites.copy()                    # [lead x model x class x lat x lon]
relevance_range      = relevance_composites.copy()                    # [lead x model x class x lat x lon]
predictor_composites = np.zeros((2,3,nlat,nlon)) * np.nan         # [lead x class x lat x lon]
predictor_variances  = predictor_composites.copy()                    # [lead x class x lat x lon]
ncorrect_byclass     = np.zeros((2,nmodels,3))                # [lead x model x class

for l in range(2):
    
    for nr in tqdm(range(nmodels)):
        
        predictions_model = predictions_all[l][nr] # [sample]
        relevances_model  = relevances_all[l][nr]  # [sample x lat x lon]
        
        for c in range(3):
            
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
        
#%% Visualize relevance composites differences between normalized and unnormalized data

lon = load_dict['lon']
lat = load_dict['lat']


# Nneed to cimposite and relevan
fig,axs = plt.subplots(2,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,4.5))

for ii in range(2):
    for c in range(3):
        
        ax =axs[ii,c]
        ax.set_extent(bbox)
        ax.coastlines()
        plotvar = relevance_composites[ii,:,c,:,:].mean(0)
        plotvar = plotvar / np.nanmax(np.abs(plotvar))
        pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-1,vmax=1,cmap="RdBu_r")
        
        if ii == 0:
            ax.set_title(pparams.classes[c])
        
        if c == 0:
            ax.text(-0.05, 0.55, data_names[ii], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=12)
cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.05)
cb.set_label("Normalized Relevance")
plt.suptitle("Predicting AMV lead=%i years (%s Predictor)" % (lead,varname))

savename ="%sNormalizing_Effect_%s_lead%02iyears.png" % (figpath,varname,lead)
plt.savefig(savename,dpi=150,bbox_inches="tight")


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





#%% Below this is copied code...
#%%

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







