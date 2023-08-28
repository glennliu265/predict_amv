#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Classification Accuracy by Predictor

Copied sectinos from viz_results.ipynb on 2022.11.16
Major changes on 2023.07.10 to work with output from compute_test_metrics

Created on Wed Nov 16 10:45:21 2022

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

#%%
# Load my own custom modules
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz,proc

sys.path.append("../")
import amv_dataloader as dl
import pamv_visualizer as pviz

#%% User Edits

# varnames     = ("SST","SSS","PSL","BSF","SSH","HMXL")
# varnamesplot = ("SST","SSS","SLP","BSF","SSH","MLD")
# varcolors    = ("r","violet","yellow","darkblue","dodgerblue","cyan")
# varmarker    = ("o","d","x","v","^","*")
"""
Notes on experiments 

expdirs
    - For comparing undetrended and detrended: 
        expdirs         = ("FNN4_128_Singlevar_PaperRun","FNN4_128_Singlevar_PaperRun_detrended")
        expdirs_long    = ("Forced","Unforced",)
    - For comparing CNN and FNN:
        expdirs         = ("FNN4_128_Singlevar_PaperRun","CNN2_PaperRun")
        expdirs_long    = ("FNN","CNN")

"""
detrend      = True
expdirs      = ("FNN4_128_Singlevar_PaperRun","CNN2_PaperRun")
expdirs_long = ("FNN","CNN")# "CNN (Undetrended)")
skipvars     = ("UOHC","UOSC","HMXL","BSF","TAUX","TAUY","TAUCURL")#
#threscolors = ("r","gray","cornflowerblue")
expnames     = ("FNN","CNN")
expcolors    = ("gold","dodgerblue")
quantile     = False
nsamples     = None
no_vals      = (True,) * len(expdirs)

load_test_metrics = True # Set to true to use output computed by [compute_test_metrics.py]

if quantile:
    chance_baseline = [0.33,]*3
else:
    #chance_baseline = [0.16,0.68,0.16] # Commented out for now b/c code :0
    chance_baseline  = [0.33,0.33,0.33]
#datpath   = "../../CESM_data/"
#figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20221231/"

# Old figpath: datpath + expdir + "/Figures/"

classes   = ["NASST+","Neutral","NASST-"] # [Class1 = AMV+, Class2 = Neutral, Class3 = AMV-]
leads     = np.arange(0,26,1)

# Plotting Parameters
lwall     = 2.5

darkmode  = False
if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
else:
    plt.style.use('default')
    dfcol = "k"

# Other Toggles

debug = True
#%% Load variables from main parameter file

# Note; Need to set script into current working directory (need to think of a better way)
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import predict_amv_params as pparams

# Import paths
figpath         = pparams.figpath
proc.makedir(figpath)
datpath         = pparams.datpath

# Import class information
classes         = pparams.classes
threscolors     = pparams.class_colors

# Import variable name information
varnames        = pparams.varnames
varnamesplot    = pparams.varnamesplot
varcolors       = pparams.varcolors
varmarker       = pparams.varmarker
varcolors_dark  = pparams.varcolors_dark


#%% Load the data (with functions)
import amvmod as am

alloutputs = []

if load_test_metrics: # Load new output
    for expdir in expdirs:
        output_byvar = []
        for v in varnames:
            if v=="PSL":
                v = "SLP"
            if v in skipvars:
                
                print("Skipping %s" % v)
                continue
            fn              = "%s%s/Metrics/Test_Metrics/Test_Metrics_CESM1_%s_evensample0_accuracy_predictions.npz" % (datpath,expdir,v)
            npz             = np.load(fn,allow_pickle=True)
            expdict         = proc.npz_to_dict(npz)
            output_byvar.append(expdict)
        alloutputs.append(output_byvar)
    
else: # Old Loading Script
    e = 0
    for expdir in expdirs:
        # Get list of files for each variable
        flists = []
        for v in varnames:
            varnames.pop(v)
            varcolors.pop(v)
            varnamesplot.pop(v)
            if v in skipvars:
                print("Skipping %s" % v)
                continue
            search = "%s%s/Metrics/*%s*" % (datpath,expdir,v)
            flist  = glob.glob(search)
            flist  = [f for f in flist if "of" not in f]
            flist.sort()
            print("Found %i files for %s" % (len(flist),v,))
            flists.append(flist)
        
        # Make the experiment dictionary
        expdict = am.make_expdict(flists,leads,no_val=no_vals[e])
        
        expdict['classacc'] = np.array(expdict['classacc'])
        _,nruns,nleads,nclasses      = expdict['classacc'].shape
        #nruns = len(expdict['classacc'][0])
        #nleads,nclasses = expdict['classacc'][0][0].shape
        
        # Add to outputs
        alloutputs.append(expdict)
        e += 1

#%% Updated load of persistence baseline
# ======================================

persaccclass = []
persacctotal = []
persleads   = []
for detrend in [False,True]:
    pers_leads,pers_class_acc,pers_total_acc = dl.load_persistence_baseline("CESM1",
                                                                            return_npfile=False,region=None,quantile=quantile,
                                                                            detrend=detrend,limit_samples=False,nsamples=nsamples,repeat_calc=1)
    
    persaccclass.append(pers_class_acc)
    persacctotal.append(pers_total_acc)
    persleads.append(pers_leads)
    
    
# Plot baselines
fig,axs = plt.subplots(1,3,constrained_layout=True,figsize=(12,4))
for c in range(3):
    print(c)
    ax = axs.flatten()[c]
    
    for i in range(2):
        ax.plot(persleads[i],persaccclass[i][:,c],label="Detrend %i" % i)
    ax.legend()

#%% Make the plot (2023.07.10)

expnum     = 0
plot_leads = np.arange(0,26,1)
nvar       = len(varnames)
fig,axs    = pviz.init_classacc_fig(leads)

for c in range(3):
    ax = axs[c]
    
    # Plot Neural Network Averages
    for v in range(nvar):
        varname = varnames[v]
        class_acc = alloutputs[expnum][v]['class_acc'] # [Models, Leads, Class]
        
        mu  = class_acc[:,plot_leads,c].mean(0)
        std = class_acc[:,plot_leads,c].std(0)
        
        
        ax.plot(leads[plot_leads],mu,label=varname,color=varcolors[v],marker=varmarker[v],lw=2.5)
        ax.fill_between(leads[plot_leads],(mu-std),(mu+std),color=varcolors[v],alpha=0.10,zorder=1)
    
    # Plot persistence baselines
    ax.plot(persleads[i],persaccclass[i][:,c],color=dfcol,label="Persistence",lw=lwall)
    ax.hlines([chance_baseline[c]],xmin=-1,xmax=25,ls="dashed",color=dfcol,label="Random Chance")
    
    # Label Legend
    if c == 1:
        ax.legend(ncol=4)

savename = "%sAccuracy_by_Predictor_%s.png" % (figpath,expdir)
plt.savefig(savename,dpi=150,bbox_inches='tight')
#%% Load the case for all predictors
# ======================================

# expdir   = "FNN4_128_ALL"
# varname  = "ALL"
# flist = glob.glob("%s%s/Metrics/leadtime_testing_%s*ALL.npz"%(datpath,expdir,varname))
# flist.sort()
# nruns = len(flist)
# #print('Found %i files for %s'%(nruns,varnames[v]))

# # Load Result for each model
# totalm    = []
# classm    = []
# ypredm    = []
# ylabsm    = []
# shuffidsm = []
# for i in range(nruns): # Load for 10 files

#     output,vnames = load_result(flist[i],debug=False)
    
    
#     if len(output[4]) > len(leads):
#         print("Selecting Specific Leads!")
#         output = [out[leads] for out in output]
        

#     totalm.append(output[4])
#     classm.append(output[5])
#     ypredm.append(output[6])
#     ylabsm.append(output[7])
#     shuffidsm.append(output[8])
#     print("Loaded %s, %s, %s, and %s for run %i, predictor %s" % (vnames[4],vnames[5],vnames[6],vnames[7],i,varnames[v]))




# %% The Section below does visualizations for a single experiment
# Set the experiment number here
nvars      = len(alloutputs[expnum])
expnum     = 0
if load_test_metrics: # New unpacking method (should I put this in a func... :()
    
    totalacc = np.array([alloutputs[expnum][v]['total_acc'] for v in range(nvars)])
    classacc = np.array([alloutputs[expnum][v]['class_acc'] for v in range(nvars)])
    ypred    = np.array([alloutputs[expnum][v]['predictions'] for v in range(nvars)])
    ylabs    = np.array([alloutputs[expnum][v]['targets'] for v in range(nvars)])
    
else:
    
    # old format, need to get it in the shape: # [variable x run? x lead x class]
    totalacc,classacc,ypred,ylabs,shuffids=am.unpack_expdict(alloutputs[expnum])
    
#%% Check to see how many models are just predicting one or two class (currently 24/10400)
# --------------------------- ------------------------------------------------------------
ignore_one_class = np.zeros((4,26,100))
cnt_pred1 = 0
cnt_pred2 = 0
for v in range(nvars):
    for l in range(nleads):
        for r in range(nruns):
            ypred_network = ypred[v,l,r]
            
            nclass_predicted = len(np.unique(ypred_network))
            ignore_one_class[v,l,r] = nclass_predicted
            if nclass_predicted == 1:
                cnt_pred1 += 1
            if nclass_predicted == 2:
                cnt_pred2 += 1
print("Models predicting only 1 class  : %i" % cnt_pred1)
print("Models predicting only 2 classes: %i" % cnt_pred1)
print("Model indices %s" % (str(np.where(ignore_one_class <3))))
                
#%% Visualize Accuracy by Class, compare between predictors (this one doesnt seem to work...)

nvar       = len(varnames)
nruns      = 100#totalacc.shape[1]
nleads     = len(leads)
plotmodels = np.arange(0,4)
plotmax    = False # Set to True to plot maximum
fig,axs = plt.subplots(1,3,figsize=(18,4))

for c in range(3):
    
    # Initialize plot
    ax = axs[c]
    ax.set_title("%s" %(classes[c]),fontsize=12)
    ax.set_xlim([0,24])
    ax.set_xticks(leads)
    ax.set_ylim([0,1])
    ax.set_yticks(np.arange(0,1.25,.25))
    ax.grid(True,ls='dotted')
    
    for i in plotmodels:
        if plotmax:
            plotacc = classacc[i,:,:,c].max(0) # [variable x run? x lead x class]
        else:
            plotacc = classacc[i,:,:,c].mean(0)
        
        ax.plot(leads,plotacc,color=varcolors[i],alpha=1,lw=lwall,label=varnames[i],
                marker=varmarker[i],markersize=8)
        
        
        # Add max/min predictability dots (removed this b/c it looks messy)
        # ax.scatter(leads,classacc[i,:,:,c].max(0),color=varcolors[i])
        # ax.scatter(leads,classacc[i,:,:,c].min(0),color=varcolors[i])
        
    #ax.plot(leads,autodat[::3,c],color='k',ls='dotted',label="AutoML",lw=lwall)
    ax.plot(leads,persaccclass[:,c],color=dfcol,label="Persistence",lw=lwall)
    
    ax.hlines([chance_baseline[c]],xmin=-1,xmax=25,ls="dashed",color=dfcol)
        
    if c == 0:
        ax.legend(ncol=2,fontsize=10)
        ax.set_ylabel("Accuracy")
    if c == 1:
        ax.set_xlabel("Prediction Lead (Years)")
        
plt.savefig("%sPredictor_Intercomparison_byclass_plotmax%i_%s.png"% (figpath,plotmax,expdirs[expnum]),dpi=200)


#%% Make the same plot as above, but output in increments (for AC presentation)

plotmodels = [0,2,1,3]
ex         = expnum
add_conf   = True
plotconf   = False #0.95
fill_alpha = 0.20
plotmax    = False # Set to True to plot maximum
maxmod     = 100 # Maximum networks to plot
mks        = 5 # Marker Size

def init_accplot(c,leads,figsize=(6,4),labelx=True,labely=True):
    # Initialize plot
    
    fig,ax = plt.subplots(1,1,figsize=(6,4),constrained_layout=True)
    
    pviz.format_acc_plot(leads,ax)
    
    ax.set_title("%s" %(classes[c]),fontsize=16,)
    ax.set_xlim([0,25])
    ax.set_xticks(np.arange(0,26,5))
    ax.set_ylim([0,1])
    ax.set_yticks(np.arange(0,1.25,.25))
    
    ax.plot(leads,persaccclass[detrend][:,c],color='w',label="Persistence",lw=lwall,ls="solid")
    ax.hlines([0.33],xmin=-1,xmax=25,ls="dashed",color=dfcol,label="Random Chance")
    
    return fig,ax

# Just Plot AMV+, introduce each predictor
c = 0

# Plot just the baseline
pcounter = 0
fig,ax = init_accplot(c,leads)
ax.legend(ncol=3,fontsize=10)
savename = "%sPredictor_Intercomparison_byclass_plotmax%i_%s_AGUver_%02i.png"% (figpath,plotmax,expdirs[expnum],pcounter)

# Initial Save
plt.savefig(savename,
            dpi=200,bbox_inches="tight",transparent=True)
pcounter += 1
for p in np.arange(1,len(plotmodels)+1):
    
    print(plotmodels[:p])
    plotmodels_loop= plotmodels[:p]
    fig,ax = init_accplot(c,leads)
    
    for i in plotmodels_loop:
        
        
        
        # Plot Predictor
        if plotmax:
            plotacc = classacc[i,:,:,c].max(0)
        else:
            plotacc = classacc[i,:,:,c].mean(0)
        mu        = classacc[i,:maxmod,:,c].mean(0)
        sigma     = classacc[i,:maxmod,:,c].std(0)
        sortacc  = np.sort(classacc[i,:,:,c],0)
        idpct    = sortacc.shape[0] * plotconf
        lobnd   = np.floor(idpct).astype(int)
        hibnd   = np.ceil(sortacc.shape[0]-idpct).astype(int)
        
        ax.plot(leads,mu,color=varcolors[i],marker=varmarker[i],markersize=mks,
                alpha=1.0,lw=2.5,label=varnamesplot[i],zorder=9)
        if add_conf:
            if plotconf:
                ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=fill_alpha,color=varcolors[i],zorder=1,label="")
            else:
                ax.fill_between(leads,mu-sigma,mu+sigma,alpha=fill_alpha,color=varcolors[i],zorder=1)
        ax.legend(ncol=3,fontsize=10)
    
    savename = "%sPredictor_Intercomparison_byclass_plotmax%i_%s_AGUver_%02i.png"% (figpath,plotmax,expdirs[expnum],pcounter)
    # Initial Save
    plt.savefig(savename,
                dpi=200,bbox_inches="tight",transparent=True)
    pcounter += 1


for c in [1,2]:
    fig,ax = init_accplot(c,leads,labelx=False,labely=False)
    
    for i in plotmodels:
        
        # Plot Predictor
        if plotmax:
            plotacc = classacc[i,:,:,c].max(0)
        else:
            plotacc = classacc[i,:,:,c].mean(0)
        mu        = classacc[i,:maxmod,:,c].mean(0)
        sigma     = classacc[i,:maxmod,:,c].std(0)
        sortacc  = np.sort(classacc[i,:,:,c],0)
        idpct    = sortacc.shape[0] * plotconf
        lobnd   = np.floor(idpct).astype(int)
        hibnd   = np.ceil(sortacc.shape[0]-idpct).astype(int)
        
        ax.plot(leads,mu,color=varcolors[i],marker=varmarker[i],markersize=mks,
                alpha=1.0,lw=2.5,label=varnamesplot[i],zorder=9)
        if add_conf:
            if plotconf:
                ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=.3,color=varcolors[i],zorder=1,label="")
            else:
                ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.4,color=varcolors[i],zorder=1)
        
        #ax.legend(ncol=3,fontsize=10)
        
    
    savename = "%sPredictor_Intercomparison_byclass_plotmax%i_%s_AGUver_%02i.png"% (figpath,plotmax,expdirs[expnum],pcounter)
    # Initial Save
    plt.savefig(savename,
                dpi=200,bbox_inches="tight",transparent=True)
    pcounter += 1

# ----------------------------------------------------
#%% Remake the plots, but for the GRL Paper Outline... + Draft 3
# ----------------------------------------------------

# Set Color Mode
darkmode = False
plt.style.use('default')#('seaborn_v0-8')
if darkmode == True:
    dfcol = "w"
else:
    dfcol = "k"

# Toggles and ticks
plotmodels   = [0,1,2,3] # Which predictors to plot
plotclasses  = [0,2]     # Just plot positive/negative
classes_new  = pparams.classes
expnums      = [0,1]     # Which Experiments to Plot
detrends     = [0,1]     # Whether or not it was detrended
leadticks    = np.arange(0,26,5)
legend_sp    = 2         # Subplot where legend is included
ytks         = np.arange(0,1.2,.2)

# Error Bars
plotstderr   = True  # If True, plot standard error (95%)
add_conf     = True  # If True, add empirical error bars. If false, compute stdev
plotconf     = False # If a value (ex. 0.66), include number of models within that range 
plotmax      = False # Set to True to plot maximum rather than the mean
alpha        = 0.25  # Alpha of error bars

# Initialize figures
fig,axs =  plt.subplots(2,2,constrained_layout=True,figsize=(12.5,6))
it = 0
for iplot,ex in enumerate(expnums):
    
    # Get the axes row
    axs_row = axs[iplot,:]
    
    # Load the data
    if load_test_metrics:
        totalacc = np.array([alloutputs[ex][v]['total_acc'] for v in range(nvars)])
        classacc = np.array([alloutputs[ex][v]['class_acc'] for v in range(nvars)])
        ypred    = np.array([alloutputs[ex][v]['predictions'] for v in range(nvars)])
        ylabs    = np.array([alloutputs[ex][v]['targets'] for v in range(nvars)])
    else:
        totalacc,classacc,ypred,ylabs,shuffids=am.unpack_expdict(alloutputs[ex])
    
    # Indicate detrending
    exp_dt = detrends[ex]
    
    for rowid,c in enumerate(plotclasses):
        
        ax = axs_row[rowid]
        
        # Initialize plot
        viz.label_sp(it,ax=ax,fig=fig,fontsize=pparams.fsz_splbl,
                     alpha=0.2,x=0.02)
        
        # Set Ticks/limits
        ax.set_xlim([0,24])
        ax.set_xticks(leadticks,fontsize=pparams.fsz_ticks)
        ax.set_ylim([0,1])
        ax.set_yticks(ytks,fontsize=pparams.fsz_ticks)
        ax.set_yticklabels((ytks*100).astype(int),)
        ax = viz.add_ticks(ax,facecolor="#eaeaf2",grid_lw=1.5,grid_col="w",grid_ls="solid",
                            spinecolor="darkgray",tickcolor="dimgray",
                            ticklabelcolor="k",fontsize=pparams.fsz_ticks)
        
        # Add Class Labels
        if iplot == 0:
            ax.set_title("%s" %(classes_new[c]),fontsize=pparams.fsz_title,)
        
        for i in plotmodels:
            if plotmax:
                plotacc = classacc[i,:,:,c].max(0)
            else:
                plotacc = classacc[i,:,:,c].mean(0)
            
            mu        = classacc[i,:,:,c].mean(0)
            if plotstderr:
                sigma = 2*classacc[i,:,:,c].std(0) / np.sqrt(classacc.shape[1])
            else:
                sigma = np.array(plotacc).std(0)
            
            sortacc  = np.sort(classacc[i,:,:,c],0)
            idpct    = sortacc.shape[0] * plotconf
            lobnd    = np.floor(idpct).astype(int)
            hibnd    = np.ceil(sortacc.shape[0]-idpct).astype(int)
            
            
            ax.plot(leads,mu,color=varcolors_dark[i],marker=varmarker[i],alpha=1.0,lw=2.5,label=varnamesplot[i],zorder=3)
            if add_conf:
                if plotconf:
                    ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=alpha,color=varcolors_dark[i],zorder=1,label="")
                else:
                    ax.fill_between(leads,mu-sigma,mu+sigma,alpha=alpha,color=varcolors_dark[i],zorder=1)
        
        ax.plot(leads,persaccclass[exp_dt][:,c],color=dfcol,label="Persistence",ls="dashed")
        ax.axhline(chance_baseline[c],color=dfcol,label="Random Chance",ls="dotted")
        
            # Add max/min predictability dots (removed this b/c it looks messy)
            # ax.scatter(leads,classacc[i,:,:,c].max(0),color=varcolors[i])
            # ax.scatter(leads,classacc[i,:,:,c].min(0),color=varcolors[i])
        
        #ax.plot(leads,autodat[::3,c],color='k',ls='dotted',label="AutoML",lw=lwall)
        #ax.plot(leads,persaccclass[:,c],color='k',label="Persistence",lw=lwall)
        #ax.hlines([0.33],xmin=-1,xmax=25,ls="dashed",color=dfcol)
        
        if c == 0:
            ax.set_ylabel("Prediction Accuracy (%)",fontsize=pparams.fsz_axlbl,) # Label Y-axis for first column
            ax.text(-0.14, 0.55,expdirs_long[ex], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=pparams.fsz_title)
            
        if (ex == 1):
            ax.set_xlabel("Prediction Lead (Years)",fontsize=pparams.fsz_axlbl,) # Label Y-axis for first column
        if it == legend_sp:
            ax.legend(ncol=2,fontsize=pparams.fsz_legend)
        it += 1

plt.savefig("%sPredictor_Intercomparison_byclass_detredn%i_plotmax%i_%s_OutlineVer_stderr%i.png"% (figpath,detrend,plotmax,expdirs[expnum],plotstderr),
            dpi=200,bbox_inches="tight",transparent=False)

#%% Same as above, but just for the Neutral class (Draft 02 Appendix Figure)


darkmode = False
plt.style.use('default')#('seaborn_v0-8')
if darkmode == True:
    dfcol = "w"
else:
    dfcol = "k"

# Toggles
plotmodels = [0,1,2,3]
plotclasses = [1,]
classes_new= ["NASST+","Neutral","NASST-"]

leadticks  = np.arange(0,26,5)
#plot_classes = [1,]
expnums    = [0,1]
detrends   = [0,1]
add_conf   = True
plotconf   = 0.68
plotmax    = False # Set to True to plot maximum
alpha      = 0.15
legend_sp  = 2

# Initialize figures
fig,axs =  plt.subplots(2,1,constrained_layout=True,figsize=(7,6))
it = 0
for iplot,ex in enumerate(expnums):
    
    # Get the axes row
    axs_row = axs[iplot]
    
    # Load the data
    if load_test_metrics:
        totalacc = np.array([alloutputs[ex][v]['total_acc'] for v in range(nvars)])
        classacc = np.array([alloutputs[ex][v]['class_acc'] for v in range(nvars)])
        ypred    = np.array([alloutputs[ex][v]['predictions'] for v in range(nvars)])
        ylabs    = np.array([alloutputs[ex][v]['targets'] for v in range(nvars)])
    else:
        totalacc,classacc,ypred,ylabs,shuffids=am.unpack_expdict(alloutputs[ex])
    
    # Indicate detrending
    exp_dt = detrends[ex]
    
    for rowid,c in enumerate(plotclasses):
        
        ax = axs_row
        
        # Initialize plot
        viz.label_sp(it,ax=ax,fig=fig,fontsize=16,alpha=0.2,x=0.02)
        
        ax.set_xlim([0,24])
        ax.set_xticks(leadticks)
        ax.set_ylim([0,1])
        ax.set_yticks(np.arange(0,1.2,.2))
        #ax.grid(True,ls='dotted')
        #ax.minorticks_on()
        
        ax = viz.add_ticks(ax,facecolor="#eaeaf2",grid_lw=1.5,grid_col="w",grid_ls="solid",
                            spinecolor="darkgray",tickcolor="dimgray",
                            ticklabelcolor="k")
        
        # Add Class Labels
        if iplot == 0:
            ax.set_title("%s" %(classes_new[c]),fontsize=16,)
        
        for i in plotmodels:
            if plotmax:
                plotacc = classacc[i,:,:,c].max(0)
            else:
                plotacc = classacc[i,:,:,c].mean(0)
           # ax.plot(leads,plotacc,color=varcolors[i],alpha=1,lw=lwall,label=varnames[i])
            
            mu        = classacc[i,:,:,c].mean(0)
            sigma     = classacc[i,:,:,c].std(0)
            
            sortacc  = np.sort(classacc[i,:,:,c],0)
            idpct    = sortacc.shape[0] * plotconf
            lobnd    = np.floor(idpct).astype(int)
            hibnd    = np.ceil(sortacc.shape[0]-idpct).astype(int)
            
            ax.plot(leads,mu,color=varcolors_dark[i],marker=varmarker[i],alpha=1.0,lw=2.5,label=varnamesplot[i],zorder=3)
            if add_conf:
                if plotconf:
                    ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=alpha,color=varcolors_dark[i],zorder=1,label="")
                else:
                    ax.fill_between(leads,mu-sigma,mu+sigma,alpha=alpha,color=varcolors_dark[i],zorder=1)
            
        ax.plot(leads,persaccclass[exp_dt][:,c],color=dfcol,label="Persistence",ls="dashed")
        ax.axhline(chance_baseline[c],color=dfcol,label="Random Chance",ls="dotted")
        
        if c == 1:
            
            if iplot == 0:
                ax.set_ylabel("Test Accuracy")
            #     ax.text(-0.09, -0.05,"Test Accuracy", va='bottom', ha='center',rotation='vertical',
            #             rotation_mode='anchor',transform=ax.transAxes)


            ax.text(-0.10, 0.55,expdirs_long[ex], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=14)
        if (c == 1):
            if iplot == 1:
                ax.set_xlabel("Prediction Lead (Years)")
        if it == legend_sp:
            ax.legend(ncol=2,fontsize=10)
        it += 1

plt.savefig("%sPredictor_Intercomparison_byclass_detredn%i_plotmax%i_%s_OutlineVer_NEUTRAL.png"% (figpath,detrend,plotmax,expdirs[expnum]),
            dpi=200,bbox_inches="tight",transparent=False)



#%% Same as above, but for all three classes

darkmode = False

if darkmode == True:
    dfcol = "w"
    plt.style.use('dark_background')#('seaborn_v0-8')
else:
    dfcol = "k"
    plt.style.use('default')
    
# Toggles
plotmodels = [0,1,2,3]
plotclasses = [0,1,2]
classes_new= ["NASST+","Neutral","NASST-"]

expnums    = [0,1]
detrends   = [0,1]
add_conf   = True
plotconf   = 0.68
plotstderr = True
plotmax    = False # Set to True to plot maximum
alpha      = 0.15
legend_sp  = 2
mks        = 5

# Initialize figures

fig,axs =  plt.subplots(2,3,constrained_layout=True,figsize=(14,6))
it = 0

for iplot,ex in enumerate(expnums):
    
    # Get the axes row
    axs_row = axs[iplot,:]
    
    # Load the data
    if load_test_metrics:
        totalacc = np.array([alloutputs[ex][v]['total_acc'] for v in range(nvars)])
        classacc = np.array([alloutputs[ex][v]['class_acc'] for v in range(nvars)])
        ypred    = np.array([alloutputs[ex][v]['predictions'] for v in range(nvars)])
        ylabs    = np.array([alloutputs[ex][v]['targets'] for v in range(nvars)])
    else:
        totalacc,classacc,ypred,ylabs,shuffids=am.unpack_expdict(alloutputs[ex])
    
    # Indicate detrending
    exp_dt = detrends[ex]
    
    for rowid,c in enumerate(plotclasses):
        
        ax = axs_row[rowid]
        
        # Initialize plot
        viz.label_sp(it,ax=ax,fig=fig,fontsize=16,alpha=0.7,x=0.02)
        
        ax = pviz.format_acc_plot(leads,ax)
        ax.grid(False)
        ax.set_xlim([0,25])
        ax.set_xticks(np.arange(0,26,5))
        ax.set_ylim([0,1])
        ax.set_yticks(np.arange(0,1.2,.2))
        #ax.grid(True,ls='dotted')
        #ax.minorticks_on()
        
        # ax = viz.add_ticks(ax,facecolor=None,grid_lw=1.5,grid_col="w",grid_ls="solid",
        #                    spinecolor="darkgray",tickcolor="dimgray",
        #                    ticklabelcolor="k")
        
        # Add Class Labels
        if iplot == 0:
            ax.set_title("%s" %(classes_new[c]),fontsize=16,)
        
        for i in plotmodels:
            if plotmax:
                plotacc = classacc[i,:,:,c].max(0)
            else:
                plotacc = classacc[i,:,:,c].mean(0)
           # ax.plot(leads,plotacc,color=varcolors[i],alpha=1,lw=lwall,label=varnames[i])
            
            mu        = classacc[i,:,:,c].mean(0)
            if plotstderr:
                sigma = 2*classacc[i,:,:,c].std(0) / np.sqrt(classacc.shape[1])
            else:
                sigma = classacc[i,:,:,c].std(0)
            
            sortacc  = np.sort(classacc[i,:,:,c],0)
            idpct    = sortacc.shape[0] * plotconf
            lobnd    = np.floor(idpct).astype(int)
            hibnd    = np.ceil(sortacc.shape[0]-idpct).astype(int)
            
            
            ax.plot(leads,mu,color=varcolors_dark[i],marker=varmarker[i],markersize=mks,alpha=1.0,lw=2.5,label=varnamesplot[i],zorder=3)
            if add_conf:
                if plotconf:
                    ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=alpha,color=varcolors_dark[i],zorder=1,label="")
                else:
                    ax.fill_between(leads,mu-sigma,mu+sigma,alpha=alpha,color=varcolors_dark[i],zorder=1)
            
        ax.plot(leads,persaccclass[exp_dt][:,c],color=dfcol,label="Persistence",ls="solid")
        ax.axhline(chance_baseline[c],color=dfcol,label="Random Chance",ls="dashed")
        
        if c == 0:
            
            if iplot == 0:
                ax.set_ylabel("Test Accuracy")
            #     ax.text(-0.09, -0.05,"Test Accuracy", va='bottom', ha='center',rotation='vertical',
            #             rotation_mode='anchor',transform=ax.transAxes)


            ax.text(-0.15, 0.55,expdirs_long[ex], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=14)
        if (c == 1):
            if iplot == 1:
                ax.set_xlabel("Prediction Lead (Years)")
        if it == legend_sp:
            ax.legend(ncol=2,fontsize=10)
        it += 1

plt.savefig("%sPredictor_Intercomparison_byclass_detredn%i_plotmax%i_%s_PresVer.png"% (figpath,detrend,plotmax,expdirs[expnum]),
            dpi=200,bbox_inches="tight",transparent=True)

#%% Same as above plot, but specificall for AGU 2022 (outdated now!)
plotmodels = [0,1,2,3]
ex         = expnum
add_conf   = True
plotconf   = 0.68
plotmax    = False # Set to True to plot maximum
alpha      = 0.25
fig,axs = plt.subplots(1,3,figsize=(18,4))

for c in range(3):
    
    # Initialize plot
    ax = axs[c]
    ax.set_title("%s" %(classes[c]),fontsize=16,)
    ax.set_xlim([0,24])
    ax.set_xticks(leads)
    ax.set_ylim([0,1])
    ax.set_yticks(np.arange(0,1.25,.25))
    ax.grid(True,ls='dotted')
    
    for i in plotmodels:
        if plotmax:
            plotacc = classacc[i,:,:,c].max(0)
        else:
            plotacc = classacc[i,:,:,c].mean(0)
       # ax.plot(leads,plotacc,color=varcolors[i],alpha=1,lw=lwall,label=varnames[i])
        
        mu        = classacc[i,:,:,c].mean(0)
        sigma     = classacc[i,:,:,c].std(0)
        
        sortacc  = np.sort(classacc[i,:,:,c],0)
        idpct    = sortacc.shape[0] * plotconf
        lobnd    = np.floor(idpct).astype(int)
        hibnd    = np.ceil(sortacc.shape[0]-idpct).astype(int)
        
        ax.plot(leads,mu,color=varcolors[i],marker=varmarker[i],alpha=1.0,lw=2.5,label=varnames[i],zorder=9)
        if add_conf:
            if plotconf:
                ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=alpha,color=varcolors[i],zorder=1,label="")
            else:
                ax.fill_between(leads,mu-sigma,mu+sigma,alpha=alpha,color=varcolors[i],zorder=1)
        
    ax.plot(leads,persaccclass[detrend][:,c],color=dfcol,label="Persistence",ls="dashed")
    ax.axhline(chance_baseline[c],color=dfcol,label="Random Chance",ls="dotted")
    
        # Add max/min predictability dots (removed this b/c it looks messy)
        # ax.scatter(leads,classacc[i,:,:,c].max(0),color=varcolors[i])
        # ax.scatter(leads,classacc[i,:,:,c].min(0),color=varcolors[i])
    
    #ax.plot(leads,autodat[::3,c],color='k',ls='dotted',label="AutoML",lw=lwall)
    #ax.plot(leads,persaccclass[:,c],color='k',label="Persistence",lw=lwall)

    #ax.hlines([0.33],xmin=-1,xmax=25,ls="dashed",color=dfcol)
        
    if c == 0:
        ax.legend(ncol=2,fontsize=10)
        ax.set_ylabel("Accuracy")
    if c == 1:
        ax.set_xlabel("Prediction Lead (Years)")
        
plt.savefig("%sPredictor_Intercomparison_byclass_detredn%i_plotmax%i_%s_AGUver.png"% (figpath,detrend,plotmax,expdirs[expnum]),
            dpi=200,bbox_inches="tight",transparent=True)

# --------------------------------------------------------------------------------
# %% Do comparison plot of CNN vs FNN (Outline Ver, copied from AGU version below)
# --------------------------------------------------------------------------------
# Note that this plots for all variables. The Outline Version (Final) is the next code block)
nvar = 4

v            = 0 # Choose the first variable
justbaseline = False
plotconf     = 0.05
detrend_plot = False
plot_exs     = [1,2]
fsz          = 14
fszt         = 12
fszb         = 16

for v in range(nvar):
    fig,ax = plt.subplots(1,1,figsize=(8,5.5),sharex=True,constrained_layout=True)
    
    # Plotting for each experiment
    if justbaseline is False:
        for ex,expid in enumerate(plot_exs):
            
            # Load the data
            if load_test_metrics:
                totalacc = np.array([alloutputs[ex][v]['total_acc'] for v in range(nvars)])
                classacc = np.array([alloutputs[ex][v]['class_acc'] for v in range(nvars)])
                ypred    = np.array([alloutputs[ex][v]['predictions'] for v in range(nvars)])
                ylabs    = np.array([alloutputs[ex][v]['targets'] for v in range(nvars)])
            else:
                totalacc,classacc,ypred,ylabs,shuffids=am.unpack_expdict(alloutputs[ex])
            
            # Select what to plot
            plotacc   = np.array(totalacc)[v,:,:]
            mu        = np.array(plotacc).mean(0)
            sigma     = np.array(plotacc).std(0)
            
            # Plot a percentile
            sortacc  = np.sort(plotacc,0)
            idpct    = sortacc.shape[0] * plotconf
            lobnd   = np.floor(idpct).astype(int)
            hibnd   = np.ceil(sortacc.shape[0]-idpct).astype(int)
            
            
            ax.plot(leads,mu,color=expcolors[ex],marker="o",alpha=1.0,lw=2.5,label=expnames[ex] + " (mean)",zorder=9)
            if plotconf:
                ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=.3,color=expcolors[ex],zorder=1,label=expnames[ex]+" (95% conf.)")
            else:
                ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.4,color=expcolors[ex],zorder=1)
    

    
    ax.plot(leads,persacctotal[detrend_plot],color=dfcol,label="Persistence",ls="dashed")
    ax.axhline(.33,color=dfcol,label="Random Chance",ls="dotted")
    ax.set_xlim([0,24])
    ax.set_xticks(leads,fontsize=fszt)
    ax.set_ylim([.25,1])
    ax.set_yticks(np.arange(.30,1.1,.1))
    ax.set_yticklabels((np.arange(.30,1.1,.1)*100).astype(int),fontsize=fszt)
    ax.set_ylabel("Accuracy (%)",fontsize=fsz)
    ax.set_xlabel("Prediction Lead Time (Years)",fontsize=fsz)
    ax.grid(True,ls='dotted')
    ax.legend(fontsize=fsz)
    ax.minorticks_on()
    
    
    ax.set_title("Total Accuracy, Predictor: %s" % (varnames[v]),fontsize=fszb)
    
    if justbaseline:
        savename = "%sTotalAcc_CNNvFNN_conf%03i_baselineonly.png" % (figpath,plotconf*100)
    else:
        savename = "%sTotalAcc_CNNvFNN_%s_conf%03i.png" % (figpath,varnames[v],plotconf*100)
    
    print(savename)
    plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=False)
    #ax.set_title("")
    
    
#%% Remake above, but for the GRL Outline (Draft 3, Supplemental Material)

v            = 0 # Choose the first variable
justbaseline = False
plotconf     = 0.05
detrend_plot = False
plot_exs     = [0,1]
fsz          = 16
fszt         = 18
fszb         = 18

# Error Bars
plotstderr   = True  # If True, plot standard error (95%)
add_conf     = True  # If True, add empirical error bars. If false, compute stdev
plotconf     = False # If a value (ex. 0.66), include number of models within that range 
plotmax      = False # Set to True to plot maximum rather than the mean
alpha        = 0.25  # Alpha of error bars


fig,ax = plt.subplots(1,1,figsize=(9,3),sharex=True,constrained_layout=True)

# Plotting for each experiment
if justbaseline is False:
    for ex,expid in enumerate(plot_exs):
        
        # Load the data (note this is the updated blok copied from above)
        if load_test_metrics:
            totalacc = np.array([alloutputs[ex][v]['total_acc'] for v in range(nvars)])
            classacc = np.array([alloutputs[ex][v]['class_acc'] for v in range(nvars)])
            ypred    = np.array([alloutputs[ex][v]['predictions'] for v in range(nvars)])
            ylabs    = np.array([alloutputs[ex][v]['targets'] for v in range(nvars)])
        else:
            totalacc,classacc,ypred,ylabs,shuffids=am.unpack_expdict(alloutputs[ex])
        # ==========
        
        # Compute Mean
        plotacc   = np.array(classacc)[v,:,:,:] # Select the variable
        plotacc   = plotacc[:,:,[0,2]].mean(2)  # Take mean along + and - classes
        mu        = np.array(plotacc).mean(0)   # Take mean along models
        
        # Compute Error bars
        if plotstderr:
            nmodels = plotacc.shape[0]
            sigma = 2*plotacc.std(0) / np.sqrt(nmodels)
            sigma_label = "2$\sigma_E$"
        else:
            sigma = plotacc.std(0)
            sigma_label = "1$\sigma$"
        
        sortacc  = np.sort(plotacc,0)
        idpct    = sortacc.shape[0] * plotconf
        lobnd   = np.floor(idpct).astype(int)
        hibnd   = np.ceil(sortacc.shape[0]-idpct).astype(int)
        
        ax.plot(leads,mu,color=expcolors[ex],marker="o",alpha=1.0,lw=2.5,label=expnames[ex] + " (mean)",zorder=9)
        if plotconf:
            ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=.3,color=expcolors[ex],zorder=1,label=expnames[ex]+" (95% conf.)")
        else:
            ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.4,color=expcolors[ex],zorder=1,label=expnames[ex]+" "+sigma_label)


ax.plot(leads,persacctotal[detrend_plot],color=dfcol,label="Persistence",ls="dashed")
ax.axhline(.33,color=dfcol,label="Random Chance",ls="dotted")
ax.set_xlim([0,24])
ax.set_xticks(leads,labelsize=fszt)
ax.set_ylim([.25,1])
ax.set_yticks(np.arange(.30,1.1,.1))
ax.set_yticklabels((np.arange(.30,1.1,.1)*100).astype(int),fontsize=fszt)
ax.set_ylabel("Accuracy (%)",fontsize=fsz)
ax.set_xlabel("Prediction Lead Time (Years)",fontsize=fsz)
#ax.grid(True,ls='dotted')
ax.legend(fontsize=fsz,ncol=3)
#ax.minorticks_on()

ax = viz.add_ticks(ax,facecolor="#eaeaf2",grid_lw=1.5,grid_col="w",grid_ls="solid",
                   spinecolor="darkgray",tickcolor="dimgray",
                   ticklabelcolor="k")

#ax.set_title("Total Accuracy, Predictor: %s" % (varnames[v]),fontsize=fszb)

if justbaseline:
    savename = "%sTotalAcc_CNNvFNN_conf%03i_baselineonly.png" % (figpath,plotconf*100)
else:
    savename = "%sTotalAcc_CNNvFNN_%s_conf%03i_PaperVer.svg" % (figpath,varnames[v],plotconf*100)

print(savename)
plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=False)
#ax.set_title("")

    


#%% ---------------------------------------------------------------------------
# Scrap below


#%% Save as above, but this time, for a single predictor

for v in range(nvar):
    fig,axs = plt.subplots(1,3,figsize=(18,4))
    
    for c in range(3):
        # Initialize plot
        ax = axs[c]
        ax.set_title("%s" %(classes[c]),fontsize=12)
        ax.set_xlim([0,24])
        ax.set_xticks(leads)
        ax.set_ylim([0,1])
        ax.set_yticks(np.arange(0,1.25,.25))
        ax.grid(True,ls='dotted')
        
        for r in range(nruns):
            plotacc = classacc[v,r,:,c]
            ax.plot(leads,plotacc,color=varcolors[v],alpha=0.25,lw=lwall,label="")
        plotacc = classacc[v,:,:,c].mean(0)
        ax.plot(leads,plotacc,color=dfcol,alpha=1,lw=lwall,label=varnames[v])
        
        ax.hlines([0.33],xmin=-1,xmax=25,ls="dashed",color=dfcol)
            
        if c == 0:
            ax.legend(ncol=2,fontsize=10)
            ax.set_ylabel("Accuracy")
        if c == 1:
            ax.set_xlabel("Prediction Lead (Years)")
        
    plt.savefig("%s%s_Acc_byclass.png"% (figpath,varnames[v]),dpi=200)
    
#%% For a given predictor, visualize the distribution in accuracies

# Leadtime x Variable Plots
binedges = np.arange(0,1.05,0.05)
v        = 0

for c in range(3):
    fig,axs  = plt.subplots(nvar,nleads,figsize=(24,10),sharex=True)
    
    for v in range(nvar):
        for l in range(nleads):
            
            ax = axs[v,l]
            if v == 0:
                ax.set_title("Lead %02i Years" % (leads[l]))
            
            if l == 0:
                ax.text(-0.2, 0.55, varnames[v], va='bottom', ha='center',rotation='vertical',
                                 rotation_mode='anchor',transform=ax.transAxes)
                
            plotvar = classacc[v,:,l,c]
            #h = ax.hist(plotvar)
            
            h = ax.hist(plotvar,binedges,color=varcolors[v],alpha=0.6,
                        label="")
            ax.axvline(plotvar.mean(),color="k",ls='dashed',lw=0.9,
                       label="%.2f" %(plotvar.mean()*100)+"%")
            ax.legend()
            
            ax.set_xticks(binedges[::4])
            ax.set_ylim([0,20])
            ax.grid(True,ls='dotted')
    
    savename = "%sHistograms_By_Leadtime_Variable_Class%s.png" % (figpath,classes[c])
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
#%% Visualize the total accuracy relative to a few baselines

fig,ax = plt.subplots(1,1,figsize=(6,4),sharex=True)
for v in range(nvar):
    
    plotacc   = totalacc[v,:,:]
    mu        = plotacc.mean(0)
    sigma     = plotacc.std(0)
    
    ax.plot(leads,mu,color=varcolors[v],marker="o",alpha=1.0,lw=2,label=varnames[v])
    #ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.05,color=varcolors[v])

ax.plot(leads,persacctotal,color='k',label="Persistence",ls="dashed")
ax.axhline(.33,color='k',label="Random Chance",ls="dotted")

ax.set_xlim([0,24])
ax.set_xticks(leads)
ax.set_ylim([.25,1])
ax.set_yticks(np.arange(.30,1.1,.1))
ax.set_yticklabels((np.arange(.30,1.1,.1)*100).astype(int))
ax.set_ylabel("Total Accuracy (%)")
ax.set_xlabel("Prediction Lead Time (years)")
ax.grid(True,ls='dotted')
ax.legend()

savename = "%sTotal_Accuracy_%s.png" % (figpath,expdirs[expnum])
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot CNN vs NN for a selected variable (AGU2022)

v = 0

justbaseline = False

plotconf = 0.05

fsz   = 14
fszt  = 12
fszb  = 16


for v in range(nvar):
    fig,ax = plt.subplots(1,1,figsize=(8,5.5),sharex=True,constrained_layout=True)
    
    # Plotting for each experiment
    if justbaseline is False:
        for ex in range(2):
            totalacc,classacc,ypred,ylabs,shuffids=am.unpack_expdict(alloutputs[ex])
            
            plotacc   = np.array(totalacc)[v,:,:]
            mu        = np.array(plotacc).mean(0)
            sigma     = np.array(plotacc).std(0)
            
            
            sortacc  = np.sort(plotacc,0)
            idpct    = sortacc.shape[0] * plotconf
            lobnd   = np.floor(idpct).astype(int)
            hibnd   = np.ceil(sortacc.shape[0]-idpct).astype(int)
            
            
            ax.plot(leads,mu,color=expcolors[ex],marker="o",alpha=1.0,lw=2.5,label=expnames[ex] + " (mean)",zorder=9)
            if plotconf:
                ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=.3,color=expcolors[ex],zorder=1,label=expnames[ex]+" (95% conf.)")
            else:
                ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.4,color=expcolors[ex],zorder=1)
        
    ax.plot(leads,persacctotal,color=dfcol,label="Persistence",ls="solid")
    ax.axhline(.33,color=dfcol,label="Random Chance",ls="dashed")
    
    
    ax.set_xlim([0,24])
    ax.set_xticks(leads,fontsize=fszt)
    ax.set_ylim([.25,1])
    ax.set_yticks(np.arange(.30,1.1,.1))
    ax.set_yticklabels((np.arange(.30,1.1,.1)*100).astype(int),fontsize=fszt)
    ax.set_ylabel("Accuracy (%)",fontsize=fsz)
    ax.set_xlabel("Prediction Lead Time (Years)",fontsize=fsz)
    ax.grid(True,ls='dotted')
    ax.legend(fontsize=fsz)
    ax.set_title("Total Accuracy, Predictor: %s" % (varnames[v]),fontsize=fszb)
    
    if justbaseline:
        savename = "%sTotalAcc_CNNvFNN_conf%03i_baselineonly.png" % (figpath,plotconf*100)
    else:
        savename = "%sTotalAcc_CNNvFNN_%s_conf%03i.png" % (figpath,varnames[v],plotconf*100)
    
    print(savename)
    plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=True)
    #ax.set_title("")
    
    
#%% Same as above for for each class


fsz   = 10
fszt  = 8
fszb  = 14

ylower = 0.2
add_conf = False
incl_title = False

for v in range(nvar):
    fig,axs = plt.subplots(3,1,figsize=(3,10),sharex=True,)
    
    # Plotting for each experiment
    for c in range(3):
        ax = axs[c]
        viz.label_sp(classes[c],labelstyle="%s",usenumber=True,ax=ax,alpha=0,fontcolor=threscolors[c])
        for ex in range(2):
            totalacc,classacc,ypred,ylabs,shuffids=unpack_expdict(alloutputs[ex])
            
            plotacc   = classacc[v,:,:,c]
            # ------
            mu        = plotacc.mean(0)
            sigma     = plotacc.std(0)
            
            sortacc  = np.sort(plotacc,0)
            idpct    = sortacc.shape[0] * plotconf
            lobnd   = np.floor(idpct).astype(int)
            hibnd   = np.ceil(sortacc.shape[0]-idpct).astype(int)
            
            ax.plot(leads,mu,color=expcolors[ex],marker="o",alpha=1.0,lw=2.5,label=expnames[ex] + " (mean)",zorder=9)
            if add_conf:
                if plotconf:
                    ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=.3,color=expcolors[ex],zorder=1,label=expnames[ex]+" (95% conf.)")
                else:
                    ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.4,color=expcolors[ex],zorder=1)
            
            
        ax.plot(leads,persacctotal,color=dfcol,label="Persistence",ls="dashed")
        ax.axhline(.33,color=dfcol,label="Random Chance",ls="dotted")
        
        
        ax.set_xlim([0,24])
        ax.set_xticks(leads,fontsize=fszt)
        ax.set_ylim([ylower,1])
        ax.set_yticks(np.arange(ylower,1.1,.1))
        ax.set_yticklabels((np.arange(ylower,1.1,.1)*100).astype(int),fontsize=fszt)
        
        ax.grid(True,ls='dotted')
        
        if c == 0:
            if incl_title:
                ax.set_title("Total Accuracy, Predictor: %s" % (varnames[v]),fontsize=fszb)
        if c == 1:
            #ax.legend(fontsize=fszt,ncol=3)
            ax.set_ylabel("Accuracy (%)",fontsize=fsz)
        if c == 2:
            ax.set_xlabel("Prediction Lead Time (Years)",fontsize=fsz)
        
    savename = "%sClassAcc_CNNvFNN_%s_conf%03i.png" % (figpath,varnames[v],plotconf*100)
    print(savename)
    plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=True)
    #ax.set_title("")

#%% In this section below, we make some comparisons of skill by experiment...

#%% Sort by accuracies

#%% Get (in order) 

plotvar = classacc[v,:,l,c]
fig,ax = plt.subplots(1,1)


