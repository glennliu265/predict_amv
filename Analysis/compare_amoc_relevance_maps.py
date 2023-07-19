#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare the AMOC and Relevance Composites

Created on Fri Jul 14 11:30:49 2023

@author: gliu
"""

# ++++++++++++++++++++++++++++++++++++++++++
#%% Import Packages
# ++++++++++++++++++++++++++++++++++++++++++

import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import sys
import os

from tqdm import tqdm
import cartopy.crs as ccrs

# ++++++++++++++++++++++++++++++++++++++++++
#%% Import Custom Packages
# ++++++++++++++++++++++++++++++++++++++++++

machine = "Astraeus"

# LRP Methods
sys.path.append("/Users/gliu/Downloads/02_Research/03_Code/github/Pytorch-LRP-master/")
from innvestigator import InnvestigateModel

# Load modules (LRPutils by Peidong)
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/scrap/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/predict_amv/")
import LRPutils as utils

# Load visualization module
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz,proc

# Load parameter files
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams
import amvmod as am
import amv_dataloader as dl
import train_cesm_params as train_cesm_params
import pamv_visualizer as pviz

# Load relevant variables from parameter files
bboxes  = pparams.bboxes
regions = pparams.regions
rcolors = pparams.rcolors

classes = pparams.classes
proj    = pparams.proj
bbox    = pparams.bbox

datpath = pparams.datpath
figpath = pparams.figpath
proc.makedir(figpath)

# Load model_dict
nn_param_dict = pparams.nn_param_dict

# ==========================================
#%% User Edits
# ==========================================

# Relevance Map Loading Information
expdir      = "FNN4_128_SingleVar_PaperRun"
#varnames    = ['SSH', 'SST', 'SSS', 'SLP']  ("SST","SSH","SSS","SLP")
varcolors   = ['dodgerblue','r','violet','gold']

# AMOC Loading Information
amocpath    = "../../CESM_data/Regression_Maps/"
detrend     = 0
startyr     = 1920
endyr       = 2005
coordinate  = "depth"
amoc_lead   = False

# Get experiment Information and other parameters
eparams     = train_cesm_params.train_params_all[expdir]

# Other Toggles
debug       = True


# ><><><><><><><><><><><><><><><><><><><><><
#%% Main Body Start
# ><><><><><><><><><><><><><><><><><><><><><


# Load AMOC regression maps
amoc_name = "%sAMOC_Regression_Maps_%ito%i_amooclead%i_detrend%i_%s.npz" % (amocpath,startyr,endyr,amoc_lead,detrend,coordinate)
ld        = np.load(amoc_name,allow_pickle=True)

# Get constituent variables
amoc_maps = ld['regression_maps']
leads     = ld['leads']
varnames  = ld['varnames']
ens_nums  = ld['ens']
lon       = ld['lon']
lat       = ld['lat']

#%% Get information from the experiment






#%% Load the data and target (copied from [viz_LRP_predictor.py] on 2023.07.14)

# Load predictor and labels, lat/lon, cut region
target                          = dl.load_target_cesm(detrend=eparams['detrend'],region=eparams['region'],newpath=True)
data_all,lat,lon                = dl.load_data_cesm(varnames,eparams['bbox'],detrend=eparams['detrend'],return_latlon=True,newpath=True)

# Apply Preprocessing
#target_all                      = target[:eparams['ens'],:]
#data_all                        = data_all[:,:eparams['ens'],:,:,:]
nchannels,nens,ntime,nlat,nlon  = data_all.shape

# Make land mask
data_mask = np.sum(data_all,(0,1,2))
data_mask[~np.isnan(data_mask)] = 1
if debug:
    plt.pcolormesh(data_mask),plt.colorbar()

# Remove all NaN points
data_all[np.isnan(data_all)]    = 0

# Get Sizes
nchannels                       = 1 # Change to 1, since we are just processing 1 variable at a time
inputsize                       = nchannels*nlat*nlon    # Compute inputsize to remake FNN
nclasses                        = len(eparams['thresholds']) + 1
nlead                           = len(leads)

# Create Classes
std1         = target.std(1).mean() * eparams['thresholds'][1] # Multiple stdev by threshold value 
if eparams['quantile'] is False:
    thresholds_in = [-std1,std1]
else:
    thresholds_in = eparams['thresholds']

# Classify AMV Events
target_class = am.make_classes(target.flatten()[:,None],thresholds_in,exact_value=True,reverse=True,quantiles=eparams['quantile'])
target_class = target_class.reshape(target.shape)

#%% Load the relevance composites [from viz_LRP_predictor.py] on 2023.07.14
nvars       = len(varnames)
nleads      = len(leads)
metrics_dir = "%s%s/Metrics/Test_Metrics/" % (datpath,expdir)
pcomps   = []
rcomps   = []
ds_all   = []
acc_dict = []
for v in range(nvars):
    # Load the composites
    varname = varnames[v]
    ncname = "%sTest_Metrics_CESM1_%s_evensample0_relevance_maps.nc" % (metrics_dir,varname)
    ds     = xr.open_dataset(ncname)
    #ds_all.append(ds)
    rcomps.append(ds['relevance_composites'].values)
    pcomps.append(ds['predictor_composites'].values)
    
    # Load the accuracies
    ldname  = "%sTest_Metrics_CESM1_%s_evensample0_accuracy_predictions.npz" % (metrics_dir,varname)
    npz     = np.load(ldname,allow_pickle=True)
    expdict = proc.npz_to_dict(npz)
    acc_dict.append(expdict)

nleads,nruns,nclasses,nlat,nlon=rcomps[v].shape


rcomps = np.array(rcomps) # [variable x lead x run x class x lat x lon]
pcomps = np.array(pcomps) # [variable x lead       x class x lat x lon]
class_accs  = np.array([acc_dict[v]['class_acc'] for v in range(nvars)]) # (4, 100, 26, 3)

#%%


#%% First, lets visualize key leadtimes of the AMOC Map


# Set darkmode
darkmode = False
if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
    transparent      = True
else:
    plt.style.use('default')
    dfcol = "k"
    transparent      = False

#Same as above but reduce the number of leadtimes
plot_bbox        = [-80,0,0,60]
leadsplot        = [25,20,10,5,0]

normalize_sample = 0 # 0=None, 1=samplewise, 2=after composite
absval           = False
cmax             = 1
cmin             = 1
clvl             = np.arange(-2.1,2.1,0.3)
no_sp_label      = True
fsz_title        = 20
fsz_axlbl        = 18
fsz_ticks        = 16
cmap='cmo.balance'


amoc_cints = { # Currently calibrated towards ensemble average values 
    "SSH" : np.arange(-2,2.1,0.1),
    "SST" : np.arange(-.55,.55,0.05),
    "SSS" : np.arange(-0.04,0.044,0.004),
    "SLP" : np.arange(-20,22,2)
    }

for c in range(3): # Loop for class
    ia = 0
    fig,axs = plt.subplots(4,5,figsize=(24,16),
                           subplot_kw={'projection':proj},constrained_layout=True)
    # Loop for variable
    for v,varname in enumerate(varnames):
        # Loop for leadtime
        for l,lead in enumerate(leadsplot):
            
            # Get lead index
            id_lead    = list(leads).index(lead)
            
            if debug:
                print("Lead %02i, idx=%i" % (lead,id_lead))
            
            # Axis Formatting
            ax = axs[v,l]
            blabel = [0,0,0,0]
            
            #ax.set_extent(plot_bbox)
            #ax.coastlines()
            
            if v == 0:
                ax.set_title("Lead %02i Years" % (leads[id_lead]),fontsize=fsz_title)
            if l == 0:
                blabel[0] = 1
                ax.text(-0.15, 0.55, varnames[v], va='bottom', ha='center',rotation='vertical',
                        rotation_mode='anchor',transform=ax.transAxes,fontsize=fsz_axlbl)
            if v == (len(varnames)-1):
                blabel[-1]=1
            
            ax = viz.add_coast_grid(ax,bbox=plot_bbox,blabels=blabel,fill_color="k")
            if no_sp_label is False:
                ax = viz.label_sp(ia,ax=ax,fig=fig,alpha=0.8,fontsize=fsz_axlbl)
                
            
            # -----------------------------
            
            # --------- Composite the Relevances and variables
            plotrel = amoc_map_ensavg[id_lead,v,:,:] #rcomps_topN[v,id_lead,c,:,:]
            if normalize_sample == 2:
                plotrel = plotrel/np.nanmax(np.abs(plotrel))
                cint = np.arange(-1,1.1,0.1)
            else:
                cint = amoc_cints[varname]
            # plotvar = pcomps[v][id_lead,c,:,:]
            #plotvar = plotvar/np.max(np.abs(plotvar))
            
            
            # Set Land Points to Zero
            plotrel[plotrel==0] = np.nan
            #plotvar[plotrel==0] = np.nan
            
            # Do the plotting
            #pcm=ax.pcolormesh(lon,lat,plotrel*data_mask,cmap=cmap,vmin=cint[0],vmax=cint[-1])
            
            pcm=ax.contourf(lon,lat,plotrel*data_mask,cmap=cmap,levels=cint,extend='both')
            #cl = ax.contour(lon,lat,plotvar*data_mask,levels=clvl,colors="k",linewidths=0.75)
            #ax.clabel(cl,clvl[::2])
            
            #fig.colorbar(pcm,ax=ax)
            
            ia += 1
            # Finish Leadtime Loop (Column)
        
        cb = fig.colorbar(pcm,ax=axs[v,:].flatten(),orientation='vertical',fraction=0.025,pad=0.01)
        # Finish Variable Loop (Row)
        
    #cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.025,pad=0.01)
    #cb.set_label("Normalized Relevance",fontsize=fsz_axlbl)
    #cb.ax.tick_params(labelsize=fsz_ticks)
    
    #plt.suptitle("Mean LRP Maps for Predicting %s using %s, \n Composite of Top %02i FNNs per leadtime" % (classes[c],varname,topN,))
    savename = "%sAMOC_Regression_Maps_Test.png" % (figpath,)
    if darkmode:
        savename = proc.addstrtoext(savename,"_darkmode")
    plt.savefig(savename,dpi=150,bbox_inches="tight",transparent=transparent)



#%% Examine the Pattern Correlation

"""
amoc_maps = (26, 4, 42, 69, 65)
rcomps    = (4, 26, 100, 3, 69, 65)
"""
nmodels         = rcomps.shape[2]
amoc_map_ensavg = amoc_maps.mean(2) # (26, 4, 69, 65)


flag_pts = np.zeros(rcomps.shape)
R_calc   = np.zeros((nvars,nleads,nclasses,nmodels)) * np.nan

for v in range(nvars):
    for l in range(nleads):
        #l_ref = l
        l_ref = l
        ref_map   = amoc_map_ensavg[l_ref,v,:,:]
        targ_maps = rcomps[v,l,:,:,:,:].reshape(nclasses*nmodels,nlat,nlon) # [3*100 x lat x lon] 
        targ_maps[targ_maps == 0] = np.nan
        
        #test2 = []
        for i in range(300):
            if np.any(np.all(np.isnan(targ_maps[i,...]))):
                idclass,idmod = np.unravel_index(i,(nclasses,nmodels))
                print(i)
                targ_maps[i,...] = 0
                
            #test2.append(proc.patterncorr(ref_map,targ_maps[i,:])) # test to make sure vectorized version is the same, it is 
            # becase np.nanmax(np.abs(test1.flatten()-test2)) --> 2.55351295663786e-15
        R,N_space_ok    = proc.patterncorr_nd(ref_map,targ_maps,axis=0,return_N=True)
        R_calc[v,l,:,:] = R.reshape(nclasses,nmodels)


#%% Compute the variance of each predictor as a baseline


var_maps   = np.nanvar(data_all,2).mean(1) # [varname, lat, lon]

R_baseline = np.zeros((nvars,nleads,))
for v in range(nvars):
    for l in range(nleads):
        
        R_baseline[v,l] = proc.patterncorr(amoc_map_ensavg[l,v,:,:],var_maps[v,:,:])

#%% Now, examine the distribution of pattern correlation for leadtime and variable


# Compute Rho Critical
rhocrit = proc.ttest_rho(0.01,2,N_space_ok)

# First, a bulk view of the pattern correlation by leadtime
ytks = np.arange(-1,1.2,.2)
xtks = np.arange(0,26,5)
fig,axs = pviz.init_classacc_fig(leads)

for c in range(3):
    ax = axs[c]
    
    # Re-tick and label
    ax.set_ylim([-1,1])
    ax.set_yticks(ytks)
    ax.set_xticks(xtks)
    
    for v in range(nvars):
        
        
        mu    = np.nanmean(R_calc[v,:,c,:],1)
        sigma = np.nanstd(R_calc[v,:,c,:],1)
        ax.plot(leads,mu,label=varnames[v],color=varcolors[v])
        ax.fill_between(leads,mu-sigma,mu+sigma,color=varcolors[v],alpha=0.1)
        ax.plot(R_baseline[v,:],ls="dashed",color=varcolors[v],lw=0.5)
        
    if c == 0:
        ax.legend()
        ax.set_ylabel("Pattern Correlation")
        
    ax.axhline([0],ls="dashed",color="k")
    
    #ax.axhline([rhocrit],ls="dotted",color="k")
plt.suptitle("Pattern Correlation (Relevance Composite,Ens. Avg. AMOC Regression Map)",fontsize=fsz_title)


figname = "%sAMOC_Relevance_PatternCorr_%s_amocdetrend%i_bylead.png" % (figpath,expdir,detrend,)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Lets look at the distributions of leadtime

lead_groups = [np.arange(0,10),np.arange(10,20),np.arange(20,26)]
lead_colors = ["yellow","darkturquoise","darkviolet"]
lead_labels = ["Interannual","Decadal","Multidecadal"]
bins = np.arange(-1,1.05,0.05)
fig,axs = plt.subplots(4,3,constrained_layout=True,figsize=(12,12))

for v in range(4):
    
    for c in range(3):
        
        ax = axs[v,c]
        
        for lg in range(3):
            plotval = R_calc[v,lead_groups[lg],c,:].flatten()
            mu      = np.nanmean(plotval)
            ax.hist(plotval,bins=bins,color=lead_colors[lg],
                    alpha=0.5,edgecolor="w",label="%s" % (lead_labels[lg]))
            #        alpha=0.5,edgecolor="w",label="%s ($\mu$=%.2f)" % (lead_labels[lg],mu*100)+"%")
            ax.axvline([mu],color=lead_colors[lg])
        if v == 0:
            ax.set_title(pparams.classes[c])
            if c == 0:
                ax.legend()
        if c == 0:
            ax.set_ylabel(varnames[v])
    

plt.suptitle("Histogram of Relevance Composite-AMOC-Regression Pattern Correlations")
figname = "%sAMOC_Relevance_PatternCorr_%s_amocdetrend%i_Histograms_bypredictor.png" % (figpath,expdir,detrend,)
plt.savefig(figname,dpi=150,bbox_inches='tight')
#plt.savefig("")


#fig,ax = plt.subplots(1)

#%% What is the correlation between pattern correlation and test accuracy
# Is there a strong relationship between AMOC pattern correlation and accuracy?


fig,axs = plt.subplots(4,3,constrained_layout=True,figsize=(12,12),sharex=True,sharey=True)

for v in range(4):
    for c in range(3):
        ax = axs[v,c]
        
        for lg in range(3):
            ploty = R_calc[v,lead_groups[lg],c,:].flatten()
            plotx = class_accs[v,:,lead_groups[lg],c].flatten()
            ax.scatter(plotx,ploty,c=lead_colors[lg],label=lead_labels[lg],alpha=0.2,marker="x")
            
        if v == 0:
            ax.set_title(pparams.classes[c])
            if c == 0:
                ax.legend()
        if c == 0:
            ax.set_ylabel(varnames[v])
        
        if (c ==1) and (v == 3):
            ax.set_xlabel("Test Accuracy")
            
plt.suptitle("Test Accuracy vs. AMOC Pattern Correlation")
figname = "%sTestAcc_v_AMOC_Relevance_PatternCorr_%s_amocdetrend%i_Scatter.png" % (figpath,expdir,detrend,)
plt.savefig(figname,dpi=150,bbox_inches='tight')
            
            
        
        
