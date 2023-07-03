#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


For a given case, examine a host of possible XAI results
COpied section from compute_test_metrics.py


Created on Sat Jul  1 12:46:34 2023

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

import captum

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


figpath = pparams.figpath

# ============================================================
#%% User Edits vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ============================================================

# Set machine and import corresponding paths

# Set experiment directory/key used to retrieve params from [train_cesm_params.py]
expdir              = "FNN4_128_SingleVar_PaperRun"
eparams             = train_cesm_params.train_params_all[expdir] # Load experiment parameters

# Select Option
varname = ["SSH",]

# Processing Options
even_sample         = False
standardize_input   = False # Set to True to standardize variance at each point

# Get some paths
datpath             = pparams.datpath
dataset_name        = "CESM1"

# Set some looping parameters and toggles
#varnames            = ["SSH","SST","SLP","SSS","NHFLX"]       # Names of predictor variables
leads               = np.arange(0,26,1)    # Prediction Leadtimes
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
#eparams['varnames'] = varnames
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
load_dict                      = am.prepare_predictors_target(varname,eparams,return_nfactors=True,load_all_ens=True)
data                           = load_dict['data']
target_class                   = load_dict['target_class']

# Pick just the testing set
data                           = data[:,ens_test,...]
target_class                   = target_class[ens_test,:]

# Get necessary sizes
nchannels,nens,ntime,nlat,nlon = data.shape             
inputsize                      = nchannels*nlat*nlon    # Compute inputsize to remake FNN
nclasses                       = len(eparams['thresholds'])+1
nlead                          = len(leads)

# Count Samples...
am.count_samples(None,target_class)

# Load lat/lon
lon = load_dict['lon']
lat = load_dict['lat']

# --------------------------------------------------------
#%% Option to standardize input to test effect of variance
# --------------------------------------------------------

if standardize_input:
    # Compute standardizing factor (and save)
    std_vars = np.std(data,(1,2)) # [variable x lat x lon]
    # Apply standardization
    data = data / std_vars[:,None,None,:,:] 
    data[np.isnan(data)] = 0
    std_vars_after = np.std(data,(1,2))
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


vt      = time.time()
v       = 0
varname_in = varname[v]
l       = 25
lead    = leads[l]
nr      = 0




#%%
# ================================
#% 1. Load model weights + Metrics
# ================================
# Get the model weights [lead][run]
modweights_lead,modlist_lead=am.load_model_weights(datpath,expdir,leads,varname_in)
nmodels = len(modweights_lead[0])

# Get list of metric files
search = "%s%s/Metrics/%s" % (datpath,expdir,"*%s*" % varname_in)
flist  = glob.glob(search)
flist  = [f for f in flist if "of" not in f]
flist.sort()
print("Found %i files per lead for %s using searchstring: %s" % (len(flist),varname_in,search))

# ======================================
#% 2. Retrieve predictor and preallocate
# ======================================
lt = time.time()
predictors            = data[[v],...] # Get selected predictor

# Preallocate
total_acc_all         = np.zeros((nmodels,nlead))
class_acc_all         = np.zeros((nmodels,nlead,3)) # 

# Relevances
relevances_all        = [] # [nlead][nmodel][sample x lat x lon]
predictor_all         = [] # [nlead][sample x lat x lon]

# Predictions
predictions_all       = [] # [nlead][nmodel][sample]
targets_all           = [] # [nlead][sample]

# ==============
#%% Loop by lead
# ==============
# Note: Since the testing sample is the same withheld set for the experiment, we can use leadtime as the outer loop.

# -----------------------
# Loop by Leadtime...
# -----------------------
outname = "/Test_Metrics_%s_%s_evensample%i.npz" % (dataset_name,varname_in,even_sample)
if standardize_input:
    outname = proc.addstrtoext(outname,"_standardizeinput")

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

# =====================
# II. Rebuild the model
# =====================
# Get the models (now by leadtime)
modweights = modweights_lead[l][nr]
modlist    = modlist_lead[l][nr]

# Rebuild the model
pmodel = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)

# Load the weights
pmodel.load_state_dict(modweights)
pmodel.eval()


# ===========================
#%% IV. Perform LRP (Pytorch-LRP)
# ===========================

innexp         = 1
innmethod      ='b-rule'
innbeta        = 0.5
nsamples_lead = X_torch.shape[0]
innepsi        = 1e-6

inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,epsilon=innepsi,
                                  method=innmethod,
                                  beta=innbeta)
model_prediction, sample_relevances = inn_model.innvestigate(in_tensor=X_torch)
model_prediction                    = model_prediction.detach().numpy().copy()
sample_relevances                   = sample_relevances.detach().numpy().copy()

if "FNN" in eparams['netname']:
    predictor_test    = X_torch.detach().numpy().copy().reshape(nsamples_lead,nlat,nlon)
    sample_relevances = sample_relevances.reshape(nsamples_lead,nlat,nlon) # [test_samples,lat,lon] 

#  ===========================
# %% Compute relevance maps for captum 
#  ===========================

lrp                 = captum.attr.LRP(pmodel)
relevances_captum = []
for c in range(3):
    relevance_captum    = lrp.attribute(X_torch,target=c)
    relevances_captum.append(relevance_captum.detach().numpy().copy())
relevance_captum = np.array(relevances_captum).mean(0)
relevance_captum    = relevance_captum.reshape(nsamples_lead,nlat,nlon) 

# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
#%% PLOT: Check captum maps for each class 
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>

normalize_samplewise = True

vlm     = 4e-4
fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(12,4))

for a in range(3):
    ax = axs[a]
    plotvar = np.array(relevances_captum)[a,...].reshape(nsamples_lead,nlat,nlon)
    if normalize_samplewise:
        plotvar = plotvar / np.nanmax(np.abs(plotvar),(1,2))[:,None,None]
        print(np.nanmax(plotvar))
        plotvar = plotvar.mean(0)
        vlm_in  = 1e-1
    else:
        plotvar = plotvar.mean(0)
        vlm_in = vlm
        
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-vlm_in,vmax=vlm_in,cmap="cmo.balance")
    ax.coastlines()
    ax.set_title("Captum Class %i" % (a+1))

fig.colorbar(pcm,ax=axs.flatten(),orientation="horizontal",fraction=0.035,pad=0.01)
plt.suptitle("Relevance Maps for %s \n Predictor %s Lead %02i, Run %02i" % (expdir,varname_in,lead,runids[nr]))
        
        
figname = "%sCaptum_LRP_byclass_%s_%s_l%02i_run%s_standardize%i_samplenorm%i.png" % (figpath,expdir,varname_in,lead,runids[nr],standardize_input,normalize_samplewise)
plt.savefig(figname,dpi=250)

#  =================================
#%% Sort and composite by prediction
#  =================================
"""
Maybe the iNNvestigate model module is just doing the relevance maps for 
the predicted class.

In that case, I need to separate and re-composite the results accordingly.

"""

# Get indices of the predicted class
predicted_class = model_prediction.argmax(1)
predicted_class_indices = []
for c in range(3):
    cids = np.where(predicted_class == c)[0]
    predicted_class_indices.append(cids)

# Reorganize and composite by the above indices
relevances_captum_all = np.array(relevances_captum).reshape(3,nsamples_lead,nlat,nlon) 
relevances_captum_sorted = []
relevances_inn_sorted= []
for c in range(3):
    relevances_sel=relevances_captum_all[c,predicted_class_indices[c],:]
    relevances_captum_sorted.append(relevances_sel)
    
    # Also composite the Pytorch-LRP Output by Class
    relevances_inn_sorted.append(sample_relevances[predicted_class_indices[c],:,:])
    
    
composites_by_class_prediction = [relevances_captum_sorted[c].mean(0) for c in range(3)]
composites_by_class_prediction = np.array(composites_by_class_prediction)

composites_inn = np.array([relevances_inn_sorted[c].mean(0) for c in range(3)])

print(composites_by_class_prediction.shape)
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
#%% Composites by prediction, captum vs. pytorch-lrp
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
"""
Visualize the composites by class prediction
"""
normalize_samplewise=False

vlm     = 1e-3
fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(12,4))

for a in range(3):
    ax = axs[a]
    plotvar = composites_by_class_prediction[a,...]
    if normalize_samplewise:
        plotvar = plotvar / np.nanmax(np.abs(plotvar),(1,2))[:,None,None]
        print(np.nanmax(plotvar))
        plotvar = plotvar
        vlm_in  = 1e-1
    else:
        plotvar = plotvar
        vlm_in = vlm
        
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-vlm_in,vmax=vlm_in,cmap="cmo.balance")
    ax.coastlines()
    ax.set_title("Captum Class %i" % (a+1))

fig.colorbar(pcm,ax=axs.flatten(),orientation="horizontal",fraction=0.035,pad=0.01)
plt.suptitle("Relevance Maps for %s \n Predictor %s Lead %02i, Run %02i" % (expdir,varname_in,lead,runids[nr]))
        
        
figname = "%sCaptum_LRP_byPREDICTEDclass_%s_%s_l%02i_run%s_standardize%i_samplenorm%i.png" % (figpath,expdir,varname_in,lead,runids[nr],standardize_input,normalize_samplewise)
plt.savefig(figname,dpi=250)



#%% Compare Captum and Pytorch LRP



normalize_samplewise=False

vlm     = 1e-3
fig,axs = plt.subplots(2,3,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(12,8))


for mm in range(2):
    
    if mm == 0:
        composites_in = composites_by_class_prediction
        mlabel = "Captum"
    else:
        composites_in = composites_inn
        mlabel ="Pytorch-LRP"
    
    for a in range(3):
        ax = axs[mm,a]
        plotvar = composites_in[a,...]
        
        if normalize_samplewise:
            plotvar = plotvar / np.nanmax(np.abs(plotvar),(1,2))[:,None,None]
            print(np.nanmax(plotvar))
            plotvar = plotvar
            vlm_in  = 1e-1
        else:
            plotvar = plotvar
            vlm_in = vlm
        
        pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-vlm_in,vmax=vlm_in,cmap="cmo.balance")
        ax.coastlines()
        ax.set_title("Class %i (%s)" % (a+1,mlabel))

fig.colorbar(pcm,ax=axs.flatten(),orientation="horizontal",fraction=0.035,pad=0.01)

lrp_params = "LRP Method %s, $beta$: %.2f, $epsilon$: %.2e, exp: %i" % (innmethod,innbeta,innepsi,innexp)
lrp_out    = "meth%s_b%.2f_e%.2f_exp%.2f" % (innmethod[0],innbeta,innepsi,innexp)

plt.suptitle("Relevance Maps for %s \n Predictor %s Lead %02i, Run %02i\n%s" % (expdir,varname_in,lead,runids[nr],lrp_params))

        
figname = "%sCaptum_Comparison_LRP_byPREDICTEDclass_%s_%s_l%02i_run%s_standardize%i_samplenorm%i_%s.png" % (figpath,expdir,varname_in,lead,runids[nr],standardize_input,normalize_samplewise,lrp_out)
plt.savefig(figname,dpi=250)




#%% Compre the two outputs

vlm     = 4e-4
fig,axs = plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(8,4))

# Plot iNNvestigate output
ax = axs[0]
pcm = ax.pcolormesh(lon,lat,sample_relevances.mean(0),vmin=-vlm,vmax=vlm,cmap="cmo.balance")
fig.colorbar(pcm,ax=ax,orientation="horizontal",fraction=0.035,pad=0.01)
ax.set_title("LRP (iNNvestigate)")

ax = axs[1]
#plotrel = composites_by_class_prediction.mean(0) # Plot composites by predictor
plotrel = np.array(relevances_captum).reshape(3,nsamples_lead,nlat,nlon).mean(0).mean(0) # Plot blind composite of "everything"
pcm = ax.pcolormesh(lon,lat,plotrel,vmin=-vlm,vmax=vlm,cmap="cmo.balance")
fig.colorbar(pcm,ax=ax,orientation="horizontal",fraction=0.035,pad=0.01)
ax.set_title("LRP (captum)")

plt.suptitle("Relevance Maps for %s \n Predictor %s Lead %02i, Run %02i" % (expdir,varname_in,lead,runids[nr]))
    
#%% Lets test a specific event with a strong signal..

bins = np.arange(-2.5,2.7,0.2)

# Examine Histograms
fig,axs = plt.subplots(1,4,constrained_layout=True,figsize=(12,4))

ax = axs[0]
ax.hist(model_prediction.flatten(),bins=bins,edgecolor="w")
ax.set_title("Final Layer Activations (All Classes)")

for i in range(3):
    ax = axs[i+1]
    ax.hist(model_prediction[:,i],bins=bins,edgecolor="w")
    ax.set_title("Activations for class %i" % (i+1))

plt.suptitle("Final Layer Activation for %s \n Predictor %s Lead %02i, Run %02i" % (expdir,varname_in,lead,runids[nr]))

figname = "%sActivationsHistogram_%s_%s_l%02i_run%s.png" % (figpath,expdir,varname_in,lead,runids[nr])
plt.savefig(figname,dpi=250)



#%% Select an event (strong positive AMV)
idmaxs = np.argmax(np.abs(model_prediction),0)
print(model_prediction[idmaxs,:])
idmax = idmaxs[0]
print("Looking at sample %i" % idmax)

# Get information
predictor_input  = X_torch[[idmax],:]
answer           = y_class[idmax]
model_activation = model_prediction[idmax]



#%% Try a bunch of INNvestigate hyperparameters


innexps         = np.arange(1,11,1)
innmethods      =('e-rule','b-rule')
innbetas        = np.arange(.1,3.5,.5)

n_exps = len(innexps)
n_beta = len(innbetas)

output_relevances = np.zeros((2,n_exps,n_beta,1,nlat*nlon))
for meth in range(2):
    innmethod = innmethods[meth]
    for exp in range(n_exps):
        innexp = innexps[exp]
        for beta in range(n_beta):
            innbeta = innbetas[beta]
            
            inn_model = InnvestigateModel(pmodel, lrp_exponent=innexp,
                                              method=innmethod,
                                              beta=innbeta)
            model_prediction, sample_relevances = inn_model.innvestigate(in_tensor=predictor_input)
            sample_relevances                   = sample_relevances.detach().numpy().copy()
            
            output_relevances[meth,exp,beta,0,:] = sample_relevances.copy()
            
output_relevances= output_relevances.reshape(2,n_exps,n_beta,1,nlat,nlon)

#%% Save output
np.savez("%siNNvestigate_test_parameters.npz"%datpath,**{
    'lon':lon,
    'lat':lat,
    'output_relevances':output_relevances,
    'innexps':innexps,
    'innbetas':innbetas,
    'innmethods':innmethods,
    'predictor':predictor_input,
    'pmodel':pmodel    },allow_pickle=True)

# Make a grid plot
#%% Visualize the output

vlm_in = 2.5e-3#5e-3
meth = 1
fig,axs = plt.subplots(10,7,figsize=(25,25),constrained_layout=True,
                       subplot_kw={'projection':ccrs.PlateCarree()})

for e in range(10):
    idx_exp = e
    innexp = innexps[idx_exp]
    
    for b in range(7):
        idx_beta = b
        innbeta = innbetas[idx_beta]
        
        ax = axs[e,b]
        
        ax.set_extent(bbox)
        ax.coastlines()
        
        ax.set_title("b: %.2f, e: %.2f" % (innbeta,innexp))
        
        plotrel =output_relevances[meth,idx_exp,idx_beta,0,:,:]
        pcm     =ax.pcolormesh(lon,lat,plotrel,cmap='cmo.balance',vmin=-vlm_in,vmax=vlm_in)
        
fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.025,orientation='horizontal')


figname = "%sSample%i_LRP_Parameter_Test_%s_%s_l%02i_run%s_method%s.png" % (figpath,idmax,expdir,varname_in,lead,runids[nr],innmethods[meth])

plt.savefig(figname,dpi=150,bbox_inches='tight')
# Find where the LRP hyperparameter space best resembles the captum output


#%% Look at corresponding output for captum

vlm_in = 5e-3
fig,axs = plt.subplots(1,3,figsize=(12,4),constrained_layout=True,
                       subplot_kw={'projection':ccrs.PlateCarree()})


for c in range(3):
    ax = axs[c]
    ax.coastlines()
    plotrel =relevances_captum_all[c,idmax,:,:]
    pcm     =ax.pcolormesh(lon,lat,plotrel,cmap='cmo.balance',vmin=-vlm_in,vmax=vlm_in)
    
fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.025,orientation='horizontal')


figname = "%sSample%i_LRP_Captum_%s_%s_l%02i_run%s_method%s.png" % (figpath,idmax,expdir,varname_in,lead,runids[nr],innmethods[meth])
plt.savefig(figname,dpi=150,bbox_inches='tight')


#%% I'm Up to Here.... --------------------------------------------------------




#%%
# Preallocate
predictor_lead   = []
relevances_lead  = []

predictions_lead = []
targets_lead     = []

# --------------------
# 05. Loop by runid...
# --------------------
for nr,runid in tqdm(enumerate(runids)):
    rt = time.time()
    
    # =====================
    # II. Rebuild the model
    # =====================
    # Get the models (now by leadtime)
    modweights = modweights_lead[l][nr]
    modlist    = modlist_lead[l][nr]
    
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
    
    # Save variables
    total_acc_all[nr,l]   = lead_acc
    class_acc_all[nr,l,:] = class_acc
    predictions_lead.append(y_predicted)
    if nr == 0:
        targets_all.append(y_actual)
    
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
        predictor_all.append(predictor_test) # Predictors are the same across model runs
    relevances_lead.append(sample_relevances)
    
    # Clear some memory
    del pmodel
    torch.cuda.empty_cache()  # Save some memory
    
    #print("\nRun %i finished in %.2fs" % (runid,time.time()-rt))
    # End Lead Loop >>>
relevances_all.append(relevances_lead)
predictions_all.append(predictions_lead)
print("\nCompleted training for %s lead %i of %i in %.2fs" % (varname_in,lead,leads[-1],time.time()-lt))


# =============================================================================================================================
#%% Composite the relevances (can look at single events later, it might actually be easier to write a separate script for that)
# =============================================================================================================================
# the purpose here is to get some quick, aggregate metrics

# Need to add option to cull models in visualization script...
st_rel_comp          = time.time()

relevance_composites = np.zeros((nlead,nmodels,3,nlat,nlon)) * np.nan # [lead x model x class x lat x lon]
relevance_variances  = relevance_composites.copy()                    # [lead x model x class x lat x lon]
relevance_range      = relevance_composites.copy()                    # [lead x model x class x lat x lon]
predictor_composites = np.zeros((nlead,3,nlat,nlon)) * np.nan         # [lead x class x lat x lon]
predictor_variances  = predictor_composites.copy()                    # [lead x class x lat x lon]
ncorrect_byclass     = np.zeros((nlead,nmodels,3))                # [lead x model x class

for l in range(nlead):
    
    for nr in tqdm(range(nmodels)):
        
        predictions_model = predictions_all[l][nr] # [sample]
        relevances_model  = relevances_all[l][nr]  # [sample x lat x lon]
        
        for c in range(3):
            
            # Get correct indices
            class_indices                   = np.where(targets_all[l] == c)[0] # Sample indices of a particular class
            correct_ids                     = np.where(targets_all[l][class_indices] == predictions_model[class_indices])
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
            correct_predictors               = predictor_all[l][correct_pred_id,...]
            predictor_composites[l,c,:,:]    = correct_predictors.mean(0)
            predictor_variances[l,c,:,:]     = correct_predictors.var(0)
print("Saved Relevance Composites in %.2fs" % (time.time()-st_rel_comp))
        
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
outname    = "%s%s/Metrics/Test_Metrics_%s_%s_evensample%i_relevance_maps.nc" % (datpath,expdir,dataset_name,varname_in,even_sample)
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
        outname    = "%s%s/Metrics/Test_Metrics_%s_%s_evensample%i_%s.npy" % (datpath,expdir,dataset_name,varname_in,even_sample,save_vars_name[sv])
        np.save(outname,save_vars[sv],allow_pickle=True)
        print("Saved %s to %s in %.2fs" % (save_vars_name[sv],outname,time.time()-st_acc))

save_vars         = [total_acc_all,class_acc_all,predictions_all,targets_all,ens_test,leads,runids]
save_vars_name    = ["total_acc","class_acc","predictions","targets","ensemble","leads","runids"]
metrics_dict      = dict(zip(save_vars_name,save_vars))
outname           = "%s%s/Metrics/Test_Metrics_%s_%s_evensample%i_accuracy_predictions.npz" % (datpath,expdir,dataset_name,varname_in,even_sample)
if standardize_input:
    outname = proc.addstrtoext(outname,"_standardizeinput")
np.savez(outname,**metrics_dict,allow_pickle=True)
print("Saved Accuracy and Predictions to %s in %.2fs" % (outname,time.time()-st_acc))

print("Completed calculating metrics for %s in %.2fs" % (varname_in,time.time()-vt))





# Load the model weights
# Load the data (predictor), testing set
# 