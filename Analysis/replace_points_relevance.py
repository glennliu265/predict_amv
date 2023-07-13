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


outpath = "../../CESM_data/%s/Metrics/Test_Metrics/" % expdir
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
                            even_sample=False,debug=False,checkgpu=False,calculate_lrp=True,notqdm=False):
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
        #_,class_count = am.count_samples(None,y_class)
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
        for nr in tqdm(range(nruns),disable=notqdm):
            
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
        if notqdm is False:
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
    #print("Saved Relevance Composites in %.2fs" % (time.time()-st_rel_comp))
    
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

# Additional loads
lon = load_dict['lon']
lat = load_dict['lat']

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


#%% Set up

"""

General Procedure

 1. Load data and subset to test set
 2. Looping by variable...
     3. Load the model weights and metrics
     4. 
     
"""

all_predictors = [data[[0],...],]#data_std[[0],...]] # Just use unstandardized for now
data_names     = ("Raw","Temporally Standardized")

# Just take the first index, since we are only looking at one lead/variable
lead       = leads[0]
varname    = varnames[0]
predictors = data[[0],...]


for lead in range(26):
    # Indicate which leads to look at
    vt      = time.time()
    
    # ================================
    #% 1. Load model weights + Metrics
    # ================================
    # Get the model weights [lead][run]
    modweights_lead,modlist_lead=am.load_model_weights(datpath,expdir,leads,varname)
    nmodels = len(modweights_lead[0])
    
    
    # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    #%% Compute the relevance map with original data and composite
    
    outdict_original = compute_relevances_lead(all_predictors,target_class,lead,eparams,modweights_lead,modlist_lead,
                                nn_param_dict,innexp,innmethod,innbeta,innepsi,
                                even_sample=even_sample,debug=debug,checkgpu=checkgpu,calculate_lrp=True)
    
    relevances_all      = outdict_original['relevances']
    predictors_all_lead = outdict_original['predictors']
    predictions_all     = outdict_original['predictions']
    targets_all         = outdict_original['targets']
    test_acc_byclass    = outdict_original['class_acc']
    
    
    #%
    # Need to add option to cull models in visualization script...
    st_rel_comp          = time.time()
    composites_ori = composite_relevances_predictors(relevances_all,predictors_all_lead,targets_all,nclasses=3)
    relevance_composites = composites_ori['relevance_composites']
    predictor_composites  = composites_ori['predictor_composites']
    print("Computed Relevance Composites in %.2fs" % (time.time()-st_rel_comp))
            
    # #%% Visualize relevance composites differences between normalized and unnormalized data
    
    # lon = load_dict['lon']
    # lat = load_dict['lat']
    
    
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
    
    
    #%% Loop....
    
    
    sel_c = [0,2]
    
    # Loop for different relevance thresholds
    test_thresholds = np.arange(0.4,0.9,0.1)
    nthres_rel      = len(test_thresholds)
    inorm           = 0
    debug_plots     = True
    mc_iter         = 50
    
    random_dropped_points_bythres = []
    dropped_points_bythres = []
    random_test_acc        = []
    replacement_test_acc   = []
    random_predictions     = []
    replacement_predictions = []
    for r in range(nthres_rel):
        rtt = time.time()
        thres_rel   = test_thresholds[r]
        
        # Plot the histogram
        # ------------------
        if debug_plots:
            
            fig,axs = plt.subplots(1,4,figsize=(12,4),constrained_layout=True)
            bins    = np.arange(0,1.1,.1)
            for c in range(4):
                ax =axs[c]
                if c < 3:
                    plotvar = relevance_composites[inorm,:,c,:,:].mean(0)
                    title = pparams.classes[c]
                else:
                    plotvar = relevance_composites[inorm,:,[0,2],:,:].mean(0).mean(0)
                    title = "NASST+/NASST- Avg."
                if c == 0:
                    ax.text(-0.25, 0.55, data_names[inorm], va='bottom', ha='center',rotation='vertical',
                            rotation_mode='anchor',transform=ax.transAxes,fontsize=12)
                plotvar = (plotvar / np.nanmax(np.abs(plotvar))).flatten()
                count_above = (plotvar > thres_rel).sum()
                ax.hist(plotvar,bins=bins,edgecolor="w")
                ax.axvline([thres_rel],ls='dashed',color="k")
                ax.set_title("%s \n Count Above %.2f: %i" % (title,thres_rel,count_above))
                savename ="%sRelevanceAblation_Histogram_%s_lead%02iyears_thresrel%.02f.png" % (figpath,varname,lead,thres_rel)
                plt.savefig(savename,dpi=150,bbox_inches="tight")
        
        # Mask the points
        # ---------------
        predictor_in = predictors.reshape(1,nens,ntime,nlat*nlon)
        plotvar      = relevance_composites[inorm,:,sel_c,:,:].mean(0).mean(0) # Mean over class, then over run
        plotvar      = plotvar / np.nanmax(np.abs(plotvar))
        sel_pts      =  np.where(plotvar.flatten() > thres_rel)[0]
        npts = len(sel_pts)
        
        # Plot the relevance mask
        # -----------------------
        if debug_plots:
            fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,4.5),
                                   constrained_layout=True)
            for c in range(3):
                ax = axs[c]
                ax.set_extent(bbox)
                ax.coastlines()
                
                if c == 0:
                    plotvar_in = plotvar
                    pcm = ax.pcolormesh(lon,lat,plotvar_in,vmin=-1,vmax=1,cmap="RdBu_r")
                    title = "Relevance"
                    
                elif c == 1:
                    plotvar_in = plotvar.copy()
                    plotvar_in[plotvar<thres_rel] = 0
                    pcm = ax.pcolormesh(lon,lat,plotvar_in,vmin=-1,vmax=1,cmap="RdBu_r")
                    title = "Above Threshold"
                elif c == 2:
                    idlat,idlon=np.unravel_index(sel_pts,(nlat,nlon))
                    ax.scatter(lon[idlon],lat[idlat])
                    title = "Replaced Points"
                ax.set_title(title)
            cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.015,pad=0.05)
            cb.set_label("Normalized Relevance")
            plt.suptitle("Predicting AMV lead=%i years (%s Predictor)" % (lead,varname))
            
            savename ="%sRelevanceAblation_ReplacementSelection_%s_lead%02iyears_thresrel%.02f.png" % (figpath,varname,lead,thres_rel)
            plt.savefig(savename,dpi=150,bbox_inches="tight")
            
        # Create synthetic data with high relevance points
        # ------------------------------------------------
        synth_name     = ["white noise","red noise"]
        synthetic_data = np.zeros((2,nens,ntime,nlat*nlon,))
        dropped_points = []
        for pt in tqdm(range(npts)):
            idx = sel_pts[pt]
            for e in range(nens):
                
                ts_in     = predictor_in[0,e,:,idx] # Get the timeseries
                
                # Estimate AR1 coefficient using yule-walker
                coef,sigma=nitime.algorithms.AR_est_YW(ts_in,1)
                
                # Make red noise timeseries
                X_ar,noise,aph=nitime.utils.ar_generator(ntime,sigma=sigma,coefs=coef)
                
                synthetic_data[1,e,:,idx] = X_ar.copy()
                
                # Make white noise timeseries
                synthetic_data[0,e,:,idx] = np.random.normal(0,np.std(ts_in),ntime)
            dropped_points.append(idx)
            
        synthetic_data = [d[None,...].reshape(1,nens,ntime,nlat,nlon) for d in synthetic_data]
        
        # Recompute test accuracy
        # -----------------------
        outdict_replaced = compute_relevances_lead(synthetic_data,target_class,lead,eparams,modweights_lead,modlist_lead,
                                    nn_param_dict,innexp,innmethod,innbeta,innepsi,
                                    even_sample=even_sample,debug=debug,checkgpu=checkgpu,calculate_lrp=False,notqdm=True)
        test_acc_byclass_synth=outdict_replaced['class_acc'] # [data x run x class]
        rel_predictions = outdict_replaced['predictions']
        
        # Compute the output
        # -----------------------
        if debug_plots:
            for method in range(2):
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
                figname = "%sRelevanceAblation_AccChange_%s_%s_relthres%.2f_classPosNeg_selrand%i.png" % (figpath,expdir,synth_name[method].replace(" ",""),thres_rel,
                                                                                                      False)
                plt.savefig(figname,dpi=150,bbox_inches='tight')
        
        # Now try replacing with random points
        #
        dropped_points_mc = []
        predictions_mc    = []
        test_acc_mc       = np.zeros((mc_iter,2,nmodels,3))
        sel_pts_ori       = np.where(plotvar.flatten() > thres_rel)[0]
        npts              = len(sel_pts_ori)
        for mc in tqdm(range(mc_iter)):
            sel_pts = np.random.choice(np.arange(nlat*nlon),size=npts) # Randomly select some points
            synthetic_data_mc = np.zeros((2,nens,ntime,nlat*nlon,))
            dropped_points = []
            for pt in range(npts):
                idx = sel_pts[pt]
                for e in range(nens):
                    
                    ts_in     = predictor_in[0,e,:,idx] # Get the timeseries
                    
                    # Make sure land points not selected
                    while np.all(predictor_in[0,e,:,idx]==0):
                        idx = np.random.choice(np.arange(nlat*nlon),size=1)[0] #+=1
                        ts_in     = predictor_in[0,e,:,idx]
                    
                    # Estimate AR1 coefficient using yule-walker
                    coef,sigma=nitime.algorithms.AR_est_YW(ts_in,1)
                    
                    # Make red noise timeseries
                    X_ar,noise,aph=nitime.utils.ar_generator(ntime,sigma=sigma,coefs=coef)
                    synthetic_data_mc[1,e,:,idx] = X_ar.copy()
                    
                    # Make white noise timeseries
                    synthetic_data_mc[0,e,:,idx] = np.random.normal(0,np.std(ts_in),ntime)
                dropped_points.append(idx)
                # End loop e
            # End loop pt
            dropped_points_mc.append(dropped_points)
            synthetic_data_mc = [d[None,...].reshape(1,nens,ntime,nlat,nlon) for d in synthetic_data_mc]
            
            # Recompute test accuracy
            # -----------------------
            outdict_replaced = compute_relevances_lead(synthetic_data_mc,target_class,lead,eparams,modweights_lead,modlist_lead,
                                        nn_param_dict,innexp,innmethod,innbeta,innepsi,
                                        even_sample=even_sample,debug=debug,checkgpu=checkgpu,calculate_lrp=False,notqdm=True)
            test_acc_mc[mc,:,:,:]=outdict_replaced['class_acc'].copy() # [data x run x class]
            predictions_mc.append(outdict_replaced['predictions'].copy())
            # End MC Loop
        
        # Save output
        
        random_dropped_points_bythres.append(dropped_points_mc)
        random_test_acc.append(test_acc_mc.copy())
        random_predictions.append(predictions_mc)
        replacement_test_acc.append(test_acc_byclass_synth)
        dropped_points_bythres.append(sel_pts_ori)
        replacement_predictions.append(rel_predictions)
        
        # End relevance threshold loop
        # Save intermediate
        savename = "%sRelevance_Replacement_Test_%s_%s_lead%02i_thres%.2f_mciter%i.npz" % (outpath,expdir,varname,lead,thres_rel,mc_iter)
        np.savez(savename,**{
            'mc_droppedpoints'   : dropped_points_mc,
            'mc_test_acc'        : test_acc_mc,
            'mc_predictions'     : predictions_mc,
            
            'rel_dropped_points' : sel_pts_ori,
            'rel_test_acc'       : test_acc_byclass_synth,
            'rel_predictions'    : rel_predictions
            },allow_pickle=True)
        
        print("Completed time loop in %.2fs"% (time.time()-rtt))
    #%% Examine the output
    
    



# random_dropped_points_bythres = []
# dropped_points_bythres = []
# random_test_acc        = []
# replacement_test_acc   = []
# random_predictions     = []
# replacement_predictions = []

#%% DO some nan flagging


lessthan  = 2 # Check if model is predicting less than this # of classes
replaceSG = False # Replace with nan if so



rel_test_acc = np.array(replacement_test_acc) # [thres x data x run x class]
mc_test_acc  = np.array(random_test_acc)       # [thres x mc x data x run x class]

mc_singlepred_flag     = np.zeros((nthres_rel,mc_iter,2,nmodels))
mc_test_acc_flagged    = mc_test_acc.copy()

rel_singlepred_flag        = np.zeros((nthres_rel,2,nmodels))
rel_test_acc_flagged   = rel_test_acc.copy()
#nsamples = random_predictions[0][0][0][0].shape
# First, make the NaN flags
for th in range(nthres_rel):
    
    for d in range(2):
        for r in range(nmodels):
            for mc in range(mc_iter):
                
                mc_preds_in = random_predictions[th][mc][d][r]
                if len(np.unique(mc_preds_in)) < lessthan:
                    mc_singlepred_flag[th,mc,d,r] = True
                    mc_test_acc_flagged[th,mc,d,r,:] = np.nan
            
            rel_preds_in = replacement_predictions[th][d][r]
            if len(np.unique(rel_preds_in)) < lessthan:
                rel_singlepred_flag[th,d,r] = True
                rel_test_acc_flagged[th,d,r,:] = np.nan

#%% Compare with the differences



if replaceSG:
    mc_diff  = mc_test_acc_flagged  - test_acc_byclass[None,None,:,:] # []
    rel_diff = rel_test_acc_flagged - test_acc_byclass[None,:,:]
else:
    mc_diff  = mc_test_acc  - test_acc_byclass[None,None,:,:] # []
    rel_diff = rel_test_acc - test_acc_byclass[None,:,:]



#%% Plot the distribution of differences

method = 1

p      = 0.05
bins   = np.arange(-0.5,.22,0.02)
fig,axs = plt.subplots(5,3,figsize=(12,8),constrained_layout=True)

for th in range(nthres_rel):
    for c in range(3):
        ax = axs[th,c]
    
        if th == 0:
            ax.set_title(pparams.classes[c])
        if c == 0:
            ax.set_ylabel("%.2f"% (test_thresholds[th]))
        
        plothist_mc  = np.nanmean(mc_diff[th,:,method,:,c],1) # [mc x nmodels]
        plothist_rel = np.nanmean(rel_diff[th,method,:,c])
        
        mu_mc     = np.nanmean(plothist_mc)
        sort_acc = np.sort(plothist_mc)
        id_sel = p*mc_iter
        i0     = np.floor(id_sel).astype(int)
        sigthres = np.interp(id_sel,[i0,i0+1],[sort_acc[i0],sort_acc[i0+1]])
        
        
        ax.hist(plothist_mc,bins=bins,edgecolor='w',color='gray')
        ax.axvline([sigthres],color="red",label="%.2f sig.=%.2f"% (p,sigthres*100) + "%")
        ax.axvline([plothist_rel],color="k",label="Actual Diff.=%.2f"% (plothist_rel*100) + "%")
        ax.legend()
        
        if (c == 1) and (th == nthres_rel-1):
            ax.set_xlabel("Accuracy Loss")
            
figname = "%sRelevanceAblation_AccChangeHistogram_FULL_%s_%s_classPosNeg_replaceSG%i_lessthan%i.png" % (figpath,expdir,synth_name[method].replace(" ",""),
                                                                                                        replaceSG,lessthan)
plt.savefig(figname,dpi=150,bbox_inches='tight')
        
    
#%% Just look at total accuracy

method = 1

p      = 0.05
bins   = np.arange(-0.5,.22,0.02)
fig,axs = plt.subplots(5,1,figsize=(10,8),constrained_layout=True)

for th in range(nthres_rel):
    ax = axs[th]

    if th == 0:
        ax.set_title("Total Accuracy")
    ax.set_ylabel("%.2f"% (test_thresholds[th]))
    
    plothist_mc  = np.nanmean(mc_diff[th,:,method,:,:][...,[0,2]],(1,2)) # [mc x nmodels]
    plothist_rel = np.nanmean(rel_diff[th,method,:,:][...,[0,2]],(0,1))
    
    mu_mc     = np.nanmean(plothist_mc)
    sort_acc = np.sort(plothist_mc)
    id_sel = p*mc_iter
    i0     = np.floor(id_sel).astype(int)
    sigthres = np.interp(id_sel,[i0,i0+1],[sort_acc[i0],sort_acc[i0+1]])
    
    
    ax.hist(plothist_mc,bins=bins,edgecolor='w',color='gray')
    ax.axvline([sigthres],color="red",label="%.2f sig.=%.2f"% (p,sigthres*100) + "%")
    ax.axvline([plothist_rel],color="k",label="Actual Diff.=%.2f"% (plothist_rel*100) + "%")
    ax.legend()
    
    #ax.set_xlim([-.3,.1])
    if (th == nthres_rel-1):
        ax.set_xlabel("Accuracy Loss")
            
figname = "%sRelevanceAblation_AccChangeHistogram_FULL_%s_%s_classPosNeg_replaceSG%i_lessthan%i_TotalAcc.png" % (figpath,expdir,synth_name[method].replace(" ",""),
                                                                                                        replaceSG,lessthan)
plt.savefig(figname,dpi=150,bbox_inches='tight')


