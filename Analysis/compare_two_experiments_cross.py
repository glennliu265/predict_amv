#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Testing Predictor Uncertainty

Examine if training on a particular predictor actually has any effect...

Steps:
    1) Load in data for 2 selected predictors
    2) Load in model weights for 2 selected predictors
    3) Do ablation study (predict one for each)
    4) Compare/visualize accuracies
    5) Compare/visualize LRP patterns

Notes:
    - Copied upper section from viz_regional_predictability

Created on Fri Mar 24 14:49:04 2023

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
import proc,viz


# Import packages specific to predict_amv
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import train_cesm_params as train_cesm_params
import amv_dataloader as dl
import amvmod as am
import pamv_visualizer as pviz

# Load LRP Package
lrp_path = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/ml_demo/Pytorch-LRP-master/"
sys.path.append(lrp_path)
from innvestigator import InnvestigateModel

#%% User Edits

# varnames           = ["SST","SSH"]
# expdirs            = ["FNN4_128_detrend","FNN4_128_SingleVar_PIC"]
# no_vals            = [True,False]
#%% Part 1 SST Comparison
ed1 = {
       "varnames"      : ["SST",],
       "expdir"        : "FNN4_128_detrend",
       "expname"       : "HTR_SST",
       "expname_long"  : "Historical SST",
       "no_val"  : True
       }

ed2 = {
       "varnames" : ["SST",],
       "expdir"   : "FNN4_128_SingleVar_PIC",
       "expname"       : "PIC_SST",
       "expname_long"  : "PiControl SST",
       "no_val"  : False
       }
expnames_title = ("Historical","PiControl")
expinfo = (ed1,ed2)
nexps   = len(expinfo)
compare_name = "HTR_v_PIC_SST"


#%% Part 2 SSH Comparison
ed1 = {
       "varnames"      : ["SSH",],
       "expdir"        : "FNN4_128_detrend",
       "expname"       : "HTR_SSH",
       "expname_long"  : "Historical SSH",
       "no_val"  : True
       }

ed2 = {
       "varnames" : ["SSH",],
       "expdir"  : "FNN4_128_SingleVar_PIC",
       "expname"       : "PIC_SSH",
       "expname_long"  : "PiControl SSH",
       "no_val"  : False
       }
expnames_title = ("Historical","PiControl")
expinfo = (ed1,ed2)
nexps   = len(expinfo)
compare_name = "HTR_v_PIC_SSH"

#%%

# Indicate shared parameters
modelname          = "FNN4_128"
nmodels            = 50 # Specify manually how much to do in the analysis
leads              = np.arange(0,26,3)


# Load parameters from [oredict_amv_param.py]
datpath            = pparams.datpath
figpath            = pparams.figpath
nn_param_dict      = pparams.nn_param_dict
class_colors       = pparams.class_colors
classes            = pparams.classes

# LRP Parameters
innexp         = 2
innmethod      ='b-rule'
innbeta        = 0.1

# Load some relevant parameters from [train_cesm1_params.py]
eparams_all = []
for einfo in expinfo:
    eparams            = train_cesm_params.train_params_all[einfo['expdir']] # Load experiment parameters
    eparams_all.append(eparams)

ens_all            = [eparams['ens'] for eparams in eparams_all]
checkgpu           = True
debug              = False
#%% Load data (taken from train_NN_CESM1.py)


# Load data into dictionaries

indicts_all = []
for exp in range(2):
    
    einfo   = expinfo[exp]
    eparams = eparams_all[exp]
    ens     = ens_all[exp]
    
    # Load predictor and labels, lat/lon, cut region
    target         = dl.load_target_cesm(detrend=eparams['detrend'],region=eparams['region'],PIC=eparams['PIC'])
    data,lat,lon   = dl.load_data_cesm(einfo['varnames'],eparams['bbox'],detrend=eparams['detrend'],return_latlon=True,PIC=eparams['PIC'])
    
    # Create classes 
    # Set exact threshold value
    std1           = target.std(1).mean() * eparams['thresholds'][1] # Multiple stdev by threshold value 
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
    
    input_dict = {
        'target' : target,
        'data'   : data,
        'std1'   : std1,
        'thresholds_in' : thresholds_in,
        'target_class'  : target_class,
        }
    indicts_all.append(input_dict)
    
    
#%% Make a consistent ice mask
mask = np.ones((nlat,nlon))
for iexp in range(2):
    
    data = indicts_all[iexp]["data"]
    while len(data.shape) > 2:
        data = data.sum(0)
    limask       = (data == 0)
    plt.pcolormesh(limask),plt.colorbar(),plt.show()
    mask[limask] = np.nan
    

#%% Load model weights (taken from LRP_LENS.py and viz_acc_byexp)

modweights_all = []
modlist_all    = []
flists_all     = []
expdicts_all   = []
no_vals_all    = []
for exp in range(2):

    einfo   = expinfo[exp]
    expdir  = einfo['expdir']
    no_val  = einfo['no_val']
    varname = einfo['varnames'][0] # Just take the first variable name
    
    
    # Get the model weights
    modweights_lead,modlist_lead=am.load_model_weights(datpath,expdir,leads,varname)
    
    # Get list of metric files
    search = "%s%s/Metrics/%s" % (datpath,expdir,"*%s*" % varname)
    flist  = glob.glob(search)
    flist  = [f for f in flist if "of" not in f]
    flist.sort()
    print("Found %i files for %s using searchstring: %s" % (len(flist),varname,search))
    

    
    # Save in broader list
    flists_all.append(flist)
    modlist_all.append(modlist_lead)
    modweights_all.append(modweights_lead)
    no_vals_all.append(no_val)


# Get the shuffled indices
expdict_all = am.make_expdict(flists_all,leads,no_val=no_vals_all)

# Unpack Dictionary
totalacc,classacc,ypred,ylabs,shuffids = am.unpack_expdict(expdict_all)

# shuffids [exp][run][lead][nsamples]

#%% An aside, examining class distribution for each run

expnames = [einfo['expname'] for einfo in expinfo]

sample_counts = np.zeros((nexps,nlead,nmodels,3,nclasses)) # [predictor,lead,run,set,class]
sample_accs   = np.zeros((nexps,nlead,nmodels,nclasses)) # [predictor,lead,run,class]

for iexp, train_name in enumerate(expnames): # Training Variable
    
    predictor    = indicts_all[iexp]['data'][[0],...] # Just take first variable!!
    target_class = indicts_all[iexp]['target_class']
    ens          = ens_all[iexp]
    eparams      = eparams_all[iexp]
    
    nchannels,nens,ntime,nlat,nlon=predictor.shape
    
    
    for l,lead in enumerate(leads):
        
        # ---------------------
        # 08. Apply Lead
        # ---------------------
        X,y_class = am.apply_lead(predictor,target_class,lead,reshape=True,ens=ens,tstep=ntime)
        
        for nm in range(nmodels):

        
            # ------------------------------------------------------------------
            # 09. Select samples recorded in the shuffled indices (nsamples x 1)
            # ------------------------------------------------------------------
            sampleids = (shuffids[iexp][nm][l]).astype(int)
            X_in      = X[sampleids,...] 
            y_in      = y_class[sampleids,...]
            
            # ------------------------
            # 10. Train/Test/Val Split
            # ------------------------
            X_subsets,y_subsets = am.train_test_split(X_in,y_in,eparams['percent_train'],
                                                           percent_val=eparams['percent_val'],
                                                           debug=False,offset=eparams['cv_offset'])
    
            
            # Loop for set
            nsets = len(y_subsets)
            for s in range(nsets):
                y_subset = y_subsets[s]#.detach().numpy
                for c in range(nclasses):
                    
                    count = (y_subset == c).sum()
                    sample_counts[iexp,l,nm,s,c] = count
                    # End class loop >>>
                # End set loop >>>

            # Can do a test here...
            # Get class accuracy
            run_acc = classacc[iexp][nm][l] # [class]
            
            sample_accs[iexp,l,nm,:] = run_acc.copy()
            # End run loop >>>
        # End lead loop >>>
    # End predictor loop >>> 

#%% Visualize relationship between Sample Size and Accuracy

setnames = ["Train","Test","Val"]
fig,axs = plt.subplots(1,3,constrained_layout=True,figsize=(12,4))

for s in range(3):
    ax = axs[s]
    
    for c in range(3):
        ax.scatter(sample_counts[:,:,:,s,c].flatten(),
                   sample_accs[...,c].flatten(),25,class_colors[c],
                   label=classes[c],alpha=0.5)
        ax.legend()
        ax.set_title(setnames[s])
        ax.set_xlabel("Sample Count")
        ax.set_ylabel("Class Accuracy")
        #ax.axis('equal')

plt.savefig("%sAccuracy_vs_SampleCount_%s.png" % (figpath,expdir,),dpi=150)

#%% Recalculate some statistics... (ablation study)

# 
st                = time.time()

# 
relevances_all    = [] # [ipred*itrain][lead][model][sample x lat x lon]
predictions_all   = [] # [ipred*itrain][lead][model][sample x 3]
idtest_all        = [] # [ipred*itrain][lead][model][sample]
labels_all        = []
ablation_idx      = []

modelaccs_all     = np.zeros((2,2,nlead,nmodels,nclasses)) # [predictor, trainvar, lead, run, class]
totalacc_all      = np.zeros((2,2,nlead,nmodels))          # [predictor, trainvar, lead, run,]

for ipred in range(nexps): # Predictor Variable

    # Get the info for the predictor/target dataset
    predictor    = indicts_all[ipred]['data'][[0],...] # Just take first variable!! [channel x ens x time x lat x lon]
    target_class = indicts_all[ipred]['target_class']
    ens          = ens_all[ipred]
    eparams      = eparams_all[ipred]
    einfo        = expinfo[ipred]
    nchannels,nens,ntime,nlat,nlon=predictor.shape
    
    # Set the name for the predictor
    predictor_name = "Predictor - %s" % einfo['expname_long']
    
    for itrain in range(nexps): # Training Variable
        
        # Get information for the training set to retrieve the model
        eparams_train = eparams_all[itrain]
        
        # Preallocate
        relevances_lead   = []
        factivations_lead = []
        idtest_lead    = []
        
        labels_lead       = []
        ypred_lead        = []
        predictions_lead   = []
        
        # Looping by lead...
        for l,lead in enumerate(leads):
            
            # Get the models (now by variable and by leadtime)
            modweights = modweights_all[itrain][l]
            modlist    = modlist_all[itrain][l]
            
            # ---------------------
            # 08. Apply Lead
            # ---------------------
            X,y_class = am.apply_lead(predictor,target_class,lead,reshape=True,ens=ens,tstep=ntime)
            
            # Loop by model..
            relevances_model   = []
            factivations_model = []
            idtest_model    = []
            predictions_model  = []
            labels_model       = []
            
            for nm in tqdm(range(nmodels)):
                
                # ------------------------------------------------------------------
                # 09. Select samples recorded in the shuffled indices (nsamples x 1) OF THE PREDICTOR
                # ------------------------------------------------------------------
                sampleids = shuffids[ipred][nm][l].astype(int)
                X_in      = X[sampleids,...] 
                y_in      = y_class[sampleids,...]
                
                # Flatten input data for FNN
                if "FNN" in eparams['netname']:
                    ndat,nchannels,nlat,nlon = X_in.shape
                    inputsize                = nchannels*nlat*nlon
                    outsize                  = nclasses
                    X_in                     = X_in.reshape(ndat,inputsize)
                
                # ------------------------
                # 10. Train/Test/Val Split
                # ------------------------
                X_subsets,y_subsets,segment_indices = am.train_test_split(X_in,y_in,eparams['percent_train'],
                                                               percent_val=eparams['percent_val'],
                                                               debug=debug,offset=eparams['cv_offset'],return_indices=True)
                
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
                pmodel = am.recreate_model(modelname,nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
                
                # Load the weights
                pmodel.load_state_dict(modweights[nm])
                pmodel.eval()
                # ----------------- ----------------------
                
                # ------------------------------------------------------
                # 12. Test the model separately to get accuracy by class
                # ------------------------------------------------------
                y_predicted,y_actual,test_loss = am.test_model(pmodel,test_loader,eparams['loss_fn'],
                                                               checkgpu=checkgpu,debug=False)
                lead_acc,class_acc = am.compute_class_acc(y_predicted,y_actual,nclasses,debug=debug,verbose=False)
                
                # -----------------------------
                # 13. Perform LRP for the model
                # -----------------------------
                X_torch       = X_subsets[1]
                nsamples_lead = X_torch.shape[0]
                inn_model     = InnvestigateModel(pmodel, lrp_exponent=innexp,
                                                  method=innmethod,
                                                  beta=innbeta)
                model_prediction, sample_relevances = inn_model.innvestigate(in_tensor=X_torch)
                model_prediction = model_prediction.detach().numpy().copy()
                sample_relevances = sample_relevances.detach().numpy().copy()
                if "FNN" in eparams['netname']:
                    predictor_test    = X_torch.detach().numpy().copy().reshape(nsamples_lead,nlat,nlon)
                    sample_relevances = sample_relevances.reshape(nsamples_lead,nlat,nlon) # [test_samples,lat,lon] 
                
                # Get Test Set Indices
                test_ids = sampleids[segment_indices[1]]
                
                # Append the predictors
                assert np.all(y_predicted == model_prediction.argmax(1)) # Check to make sure they are equivalent
                predictions_model.append(model_prediction)
                relevances_model.append(sample_relevances)
                labels_model.append(y_actual)
                idtest_model.append(test_ids) # Corresponding indices (to lagged variables)
                
                # Save the needed variables (ACC)
                modelaccs_all[ipred,itrain,l,nm,:] = class_acc
                totalacc_all[ipred,itrain,l,nm]    = lead_acc
                
                # End Model Loop >>>
            
            # Save variables for the leadtime
            relevances_lead.append(relevances_model)
            predictions_lead.append(predictions_model)
            labels_lead.append(labels_model)
            idtest_lead.append(idtest_model)
            
            # End lead loop >>>
        
        # Save variables for the train/test set...
        predictions_all.append(predictions_lead)
        relevances_all.append(relevances_lead)
        labels_all.append(labels_lead)
        idtest_all.append(idtest_lead)
        ablation_idx.append("predictor%s_train%s" % (predictor_name,train_name))
        # End Training Variable loop >>>
    # End Predictor Variable loop >>>


modelaccs_all = modelaccs_all.reshape(4,nlead,nmodels,nclasses)
totalacc_all = totalacc_all.reshape(4,nlead,nmodels)

ablation_names = ("HTR-Test, HTR-Train",
                  "HTR-Test, PIC-Train",
                  "PIC-Test, HTR-Train",
                  "PIC-Test, PIC-Train")

#%% Make the predictor ablation maps

bbox_plot = [-80,0,0,64]
fsz_axlbl = 16
fsz_title = 18
debug=False

# Plotting Parameters
normalize = 2
cmax      = 1
cints     = np.arange(-.5,.525,.025)

train_labels = ["Historical","PiControl","Historical","PiControl"]
test_labels  = ["Historical","Historical","PiControl","PiControl"]

train_labels = [expinfo[0]["expname_long"],expinfo[1]["expname_long"],expinfo[0]["expname_long"],expinfo[1]["expname_long"]]
test_labels = [expinfo[0]["expname_long"],expinfo[0]["expname_long"],expinfo[1]["expname_long"],expinfo[1]["expname_long"]]


iexp_loop   = [0,0,1,1]

ens_all_loop = [ens_all[0],ens_all[0],ens_all[1],ens_all[1]]
ntimes_loop       = [indicts_all[0]['target'].shape[1],indicts_all[0]['target'].shape[1],
                indicts_all[1]['target'].shape[1],indicts_all[1]['target'].shape[1]
                ]

iclass    = 0
ilead     = -1


for iclass in range(3):
    for ilead in range(nlead):
        fig,axs   = pviz.init_ablation_maps(bbox_plot,figsize=(12,11.0),fill_color="k")
        
        # More Plot Setup
        for aa in range(4):
            
            ax = axs.flatten()[aa]
            if debug:
                ax.set_title("%s (Test), %s (Train)" % (test_labels[aa],train_labels[aa]),fontsize=fsz_axlbl)
            
            if aa<2:
                ax.set_title("%s (Train)" % (train_labels[aa]),fontsize=fsz_axlbl)
            if aa%2 == 0:
                    ax.text(-0.15, 0.55, "%s (Test)" % (test_labels[aa]), va='bottom', ha='center',rotation='vertical',
                            rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_axlbl)
                    
            ax = viz.label_sp("%.2f" % (modelaccs_all[aa,ilead,:,iclass].mean()*100) + "%",ax=ax,fig=fig,
                              usenumber=True,labelstyle="%s",fontsize=fsz_axlbl)
        
        title = "Relevance and Predictor Composites for Correct %s Predictions,\n%i-year Leadtime" % (classes[iclass],leads[ilead])
        plt.suptitle(title,fontsize=fsz_title)
        # 
        
        # Make composites and plot them
        for aa in range(4):
            
            ax = axs.flatten()[aa]
            
            # Determine Indices of Correct Predictions # [Model x Sample x 3]
            predicted   = np.array(predictions_all[aa][ilead]).argmax(2) # [Model x Sample]
            actual      = np.array(labels_all[aa][ilead]) # [Model x Sample]
            idtest      = np.array(idtest_all[aa][ilead])
            
            correct_id  = (predicted == actual)
            acc_correct = correct_id.sum()/actual.flatten().shape[0]
            print(acc_correct)
            
            # Get linear indices w.r.t. original predictor
            idtest_correct = idtest[correct_id]
            
            # Get the info for the predictor/target dataset
            predictor    = indicts_all[iexp_loop[aa]]['data'][[0],...] # Just take first variable!! [channel x ens x time x lat x lon]
            target_class = indicts_all[iexp_loop[aa]]['target_class']
            ens          = ens_all[iexp_loop[aa]]
            nchannels,nens,ntime,nlat,nlon=predictor.shape
            X,y_class = am.apply_lead(predictor,target_class,leads[ilead],reshape=True,ens=ens,tstep=ntime)
            predictor_sel = X[idtest_correct,...]
            plotvar = predictor_sel.mean(0).squeeze()
            
        
            # First, get the relevances
            relevances_sel = np.array(relevances_all[aa][ilead]) # [Model x Sample x Lat x Lon]
            nm,ns,_,_      = relevances_sel.shape
            relevances_sel = relevances_sel.reshape(nm*ns,nlat,nlon)[correct_id.flatten(),...]
            plotrel        = relevances_sel.mean(0) * mask
            if normalize == 2:
                plotrel = plotrel/np.nanmax(np.abs(plotrel))
                
            # Plot the relevances
            pcm = ax.pcolormesh(lon,lat,plotrel,vmin=-cmax,vmax=cmax,cmap="RdBu_r")
            #cl  = ax.contour(lon,lat,plotvar,levels=cints,linewidths=0.75,colors="k")
            #ax.clabel(cl,levels=cints)
        
        cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.02,orientation="horizontal")
        cb.set_label("Normalized Relevance",fontsize=fsz_axlbl)
        
        savename = "%sAblation_LRPComposites_%s_%s_lead%02i.png" % (figpath,compare_name,classes[iclass],leads[ilead])
        plt.savefig(savename,dpi=150,bbox_inches="tight")
            
    


#%% Visualize the accuracy Differences

acolors = ("red","blue",
           "orange","magenta")

# First check total accuracy
ii = 0
fig,ax = plt.subplots(1,1,constrained_layout=True)
for aa in range(4):
    
    plotacc_model = totalacc_all[aa,:,:]
    
    for r in range(nmodels):
        ax.plot(leads,plotacc_model[:,r],color=acolors[ii],alpha=0.05)
    mu    = plotacc_model.mean(1)
    sigma = plotacc_model.std(1)
    ax.plot(leads,mu,color=acolors[ii],label=ablation_names[ii])
    ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.15,color=acolors[ii])
    ii+=1

ax.legend()
ax.set_xlabel("Prediction Lead (Years)")
ax.set_ylabel("Accuracy")
ax.set_xticks(leads)
ax.grid(True,ls="dotted")
ax.set_xlim([0,24])
ax.set_title("Total Accuracy for Predicting AMV (Predictor Uncertainty Test)")
savename = "%sPredictor_Ablation_Test_TotalAcc_%s.png" % (figpath,expdir)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Visualize accuracy differences by class

fig,axs = plt.subplots(1,3,figsize=(18,4),constrained_layout=True)
for c in range(3):
    # Initialize plot
    ax = axs[c]
    ax.set_title("%s" %(classes[c]),fontsize=16,)
    ax.set_xlim([0,24])
    ax.set_xticks(leads)
    ax.set_ylim([0,1])
    ax.set_yticks(np.arange(0,1.25,.25))
    ax.grid(True,ls='dotted')
    
    # Do the plotting
    ii = 0
    for aa in range(4):
        plotacc_model = modelaccs_all[aa,:,:,c]
        
        for r in range(nmodels):
            ax.plot(leads,plotacc_model[:,r],color=acolors[ii],alpha=0.02)
        mu    = plotacc_model.mean(1)
        sigma = plotacc_model.std(1)
        ax.plot(leads,mu,color=acolors[ii],label=ablation_names[ii])
        ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.05,color=acolors[ii],zorder=-9)
        ii+=1
    if c == 1:
        ax.legend()
        ax.set_xlabel("Prediction Lead (Years)")
    if c == 0:
        ax.set_ylabel("Accuracy")
        

plt.suptitle("Total Accuracy for Predicting AMV (Predictor Uncertainty Test)")
savename = "%sPredictor_Ablation_Test_ClassAcc_%s.png" % (figpath,expdir)
plt.savefig(savename,dpi=150,bbox_inches="tight")
