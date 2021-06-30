#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:20:16 2021

@author: gliu
"""
import numpy as np
import matplotlib.pyplot as plt

#
path   = "/Users/gliu/Downloads/2020_Fall/6.862/Project/CESM_data/Metrics/10runtest/"
#outfigpath = "/Users/gliu/Downloads/2020_Fall/6.862/Project/CESM_data/Figures/10runtest/"
nruns  = 10 
nleads = 9
leads = np.arange(0,25,3)
#%%
# Read in the ResNet Results
trainlosses = np.zeros([2,nruns,20,9])
testlosses  = np.zeros([2,nruns,20,9])
testcorrs   = np.zeros([2,nruns,9])
ypreds = []
ylabs = []
# ypreds      = np.zeros([2,nruns,9,664])
# ylabs       = np.zeros([2,nruns,9,664])
for f,freeze in enumerate([True,False]):
    fpred = []
    flab = []
    for i in range(nruns):
        # ['train_loss', 'test_loss', 'test_corr', 'yvalpred', 'yvallabels']
        searchterm = "leadtime_testing_ALL_HPT_resnet50_nepoch20_nens40_maxlead24_detrend0_noise0_unfreeze_all%s_run%i_ALL.npz" % (freeze,i)
        ld = np.load(path+searchterm,allow_pickle=True)
        trainloss = ld['train_loss']
        testloss = ld['test_loss'] # [20,9]
        testcorr = ld['test_corr']  # [9x]
        ypred = ld['yvalpred'] # [9,]
        ylab = ld['yvallabels'] # [9][0] 688
        
        trainlosses[f,i,:,:] = trainloss
        testlosses[f,i,:,:] = testloss
        testcorrs[f,i,:] = testcorr
        
        fpred.append(ypred)
        flab.append(ylab)
        # for l in range(9):
        #     ypreds[f,i,l,:]=ypred[l]
        #     ylabs[f,i,l,:]=ylab[l]
    ypreds.append(fpred)
    ylabs.append(flab)


#%%
# Read in the simplecnn results
ctrainlosses = np.zeros([2,nruns,20,9])
ctestlosses  = np.zeros([2,nruns,20,9])
ctestcorrs   = np.zeros([2,nruns,9])
cypreds = []
cylabs = []
# ypreds      = np.zeros([2,nruns,9,664])
# ylabs       = np.zeros([2,nruns,9,664])
for f,freeze in enumerate([True,False]):
    fpred = []
    flab = []
    for i in range(nruns):
        # ['train_loss', 'test_loss', 'test_corr', 'yvalpred', 'yvallabels']
        searchterm = "leadtime_testing_ALL_HPT_simplecnn_nepoch20_nens40_maxlead24_detrend0_noise0_cnndropout%s_run%i_ALL.npz" % (freeze,i)
        ld = np.load(path+searchterm,allow_pickle=True)
        trainloss = ld['train_loss']
        testloss = ld['test_loss'] # [20,9]
        testcorr = ld['test_corr']  # [9x]
        ypred = ld['yvalpred'] # [9,]
        ylab = ld['yvallabels'] # [9][0] 688
        
        ctrainlosses[f,i,:,:] = trainloss
        ctestlosses[f,i,:,:] = testloss
        ctestcorrs[f,i,:] = testcorr
        
        fpred.append(ypred)
        flab.append(ylab)
        # for l in range(9):
        #     ypreds[f,i,l,:]=ypred[l]
        #     ylabs[f,i,l,:]=ylab[l]
    cypreds.append(fpred)
    cylabs.append(flab)


#%% Read in the AutoML Results

# Load data from automl results
autopath = "/Users/gliu/Downloads/2020_Fall/6.862/Project/predict_amv/automl/"
fnr  = 'automl_accuracy_t3600_regression.npy'
autor = np.load(autopath+fnr)

# Load in newer automl results


#%% Calculate autocorrelation
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
def calculate_CESM_autocorrelation(detrend,nmembers=40,resolution='2deg'):
    # Calculate AMV Index Autocorrleation
    
    # Load in data [ens x yr x lat x lon]
    sst_normed = np.load('/Users/gliu/Downloads/2020_Fall/6.862/Project/CESM_data/CESM_sst_normalized_lat_weighted_%s_NAT_Ann.npy' % (resolution)).astype(np.float32)
    lon = np.load("../../CESM_data/lon_%s_NAT.npy"%(resolution))
    lat = np.load("../../CESM_data/lat_%s_NAT.npy"%(resolution))
    
    # Detrend if set
    if detrend: # Remove ensemble average
        sst_normed = sst_normed - np.mean(sst_normed,axis=0)[None,:,:,:]
    
    # Calculate Autocorrelation
    tstep = 86
    lags  = 25
    sst_ensemble = calc_AMV_index('NAT',sst_normed[:,:,:,:],lat,lon)
    sst_lagged_corr = np.zeros((nmembers,lags))
    for lead in range(lags):
        sst_lead = sst_ensemble[:,lead:]
        sst_lag = sst_ensemble[:,0:tstep-lead]
        #sss_lag = sss_ensemble[:,0:tstep-lead]

        for ien in range(nmembers):
            sst_lagged_corr[ien,lead] = np.corrcoef( sst_lead[ien,:],sst_lag[ien,:] )[0,1]    
    sst_auto = sst_lagged_corr.copy()
    return sst_auto

sst_auto = calculate_CESM_autocorrelation(0)

#%%
# Make some correlation plots
fig,ax = plt.subplots(1,1)


ax.plot(leads,sst_auto.mean(0)[::3],color='k',label='Persistence')
ax.fill_between(leads,sst_auto.min(0)[::3],sst_auto.max(0)[::3],color='gray',alpha=0.2,label="Persistence Range")


ax.plot(leads,testcorrs[0,:,:].T, label="",color='m',alpha=0.15)
ax.plot(leads,testcorrs[0,:,:].mean(0), label="ResNet50 (All Weights Unfrozen)",color='m',alpha=1)

ax.plot(leads,testcorrs[1,:,:].T, label="",color='r',alpha=0.15)
ax.plot(leads,testcorrs[1,:,:].mean(0), label="ResNet50 (Only Last Layer Unfrozen)",color='r',alpha=1)

ax.plot(leads,ctestcorrs[1,:,:].T, label="",color='b',alpha=0.15)
ax.plot(leads,np.nanmean(ctestcorrs[1,:,:],0), label="Simple CNN",color='b',alpha=1)


ax.plot(leads,autor[::3],label="AutoML",ls='dotted',color='k')
# ax.plot(leads,ctestcorrs[0,:,:].T, label="",color='cornflowerblue',alpha=0.15)
# ax.plot(leads,np.nanmean(ctestcorrs[0,:,:],0), label="Simple CNN (without Dropout)",color='cornflowerblue',alpha=1)

# ax.plot(leads,ctestcorrs[0,:,:].T, label="",color='m',alpha=0.15)
# ax.plot(leads,np.nanmean(ctestcorrs[0,:,:],0), label="CNN (With Dropout)",color='yellow',alpha=1)

ax.legend()
ax.set_yticks(np.arange(-0.1,1.1,.1))
ax.set_xticks(np.arange(0,25,3))
ax.set_ylim([-.1,1])
ax.grid(True,ls='dotted')
ax.set_ylabel("Correlation")
ax.set_xlabel("Prediction Lead Time (Years)")
ax.set_title("Correlation vs. Prediction Lead Time")
plt.savefig(outfigpath+"ResNet_WeightFreeze_Comparison",dpi=200)

    