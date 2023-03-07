#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Classification Accuracy by Predictor

Copied sectinos from viz_results.ipynb on 2022.11.16

Created on Wed Nov 16 10:45:21 2022

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
# Load my own custom modules
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz,proc

#%% User Edits

# varnames     = ("SST","SSS","PSL","BSF","SSH","HMXL")
# varnamesplot = ("SST","SSS","SLP","BSF","SSH","MLD")
# varcolors    = ("r","violet","yellow","darkblue","dodgerblue","cyan")
# varmarker    = ("o","d","x","v","^","*")
detrend      = True
expdirs      = ("FNN4_128_detrend","CNN2_singlevar",)#"FNN4_128_Singlevar")
skipvars     = ("UOHC","UOSC")
#threscolors = ("r","gray","cornflowerblue")
expnames   = ("FNN","CNN")
expcolors  = ("gold","dodgerblue")
quantile   = False


if quantile:
    chance_baseline = [0.33,]*3
else:
    chance_baseline = [0.16,0.68,0.16]
#datpath   = "../../CESM_data/"
#figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20221231/"

# Old figpath: datpath + expdir + "/Figures/"

classes   = ["AMV+","Neutral","AMV-"] # [Class1 = AMV+, Class2 = Neutral, Class3 = AMV-]
leads     = np.arange(0,25,3)

# Plotting Parameters
lwall     = 2.5

darkmode  = True
if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
else:
    plt.style.use('default')
    dfcol = "k"


# Other Toggles
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
threscolors    = pparams.class_colors

# Import variable name information
varnames        = pparams.varnames
varnamesplot    = pparams.varnamesplot
varcolors       = pparams.varcolors
varmarker       = pparams.varmarker

#%% Functions (Delete eventually if function form works...)

# def load_result(fn,debug=False):
#     """
#     Load results for each of the variable names
    
#     input: fn (str), Name of the file
#     """
    
#     ld = np.load(fn,allow_pickle=True)
#     vnames = ld.files
#     if debug:
#         print(vnames)
#     output = []
#     for v in vnames:
#         output.append(ld[v])
#     return output,vnames



# def retrieve_lead(shuffidx,lead,nens,tstep):
    
#     orishape = [nens,tstep-lead]
#     outidx   = np.unravel_index(shuffidx,orishape)
#     return outidx

# def unpack_expdict(expdict,dictkeys=None):
#     if dictkeys is None:
#         dictkeys = ("totalacc","classacc","ypred","ylabs","shuffids")
#     unpacked = [expdict[key] for key in expdict]
#     return unpacked

# def pack_expdict(outputs):
#     
#     expdict = {dictkeys[o]: outputs[o] for (dictkeys[o],outputs[o]) in range(len(outputs))}
#     return expdict

#%% Load the data (with functions)
import amvmod as am

alloutputs = []
for expdir in expdirs:
    
    # Get list of files for each variable
    flists = []
    for v in varnames:
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
    expdict = am.make_expdict(flists,leads)
    
    expdict['classacc'] = np.array(expdict['classacc'])
    _,nruns,nleads,nclasses      = expdict['classacc'].shape
    #nruns = len(expdict['classacc'][0])
    #nleads,nclasses = expdict['classacc'][0][0].shape
    
    
    
    # Add to outputs
    alloutputs.append(expdict)
    


#%% Load the data (Delete eventually if function form works...)
# Read in results

# Preallocate. Some quick definitions:
# model    : Network type [simplecnn,resnet50,resnet50(retrained)]
# run      : Run Number (1 to 10)
# leadtime : Leadtime in years, 0,24 in 3-year steps


# alloutputs = []
# for expdir in expdirs:
#     totalacc = [] # Accuracy for all classes combined [model x run x leadtime]
#     classacc = [] # Accuracy by class [model x run x leadtime x class]
#     ypred    = [] # Predictions [model x run x leadtime x sample]
#     ylabs    = [] # Labels [model x run x leadtime x sample]
#     shuffids = [] # Indices [model x run x leadtime x sample]
#     for v in range(len(varnames)):
        
#         flist = glob.glob("%s%s/Metrics/leadtime_testing_%s*ALL.npz"%(datpath,expdir,varnames[v]))
#         flist.sort()
#         nruns = len(flist)
#         print('Found %i files for %s'%(nruns,varnames[v]))
        
#         # Load Result for each model
#         totalm    = []
#         classm    = []
#         ypredm    = []
#         ylabsm    = []
#         shuffidsm = []
#         for i in range(nruns): # Load for 10 files
            
#             output,vnames = load_result(flist[i],debug=False)
            
            
#             if len(output[4]) > len(leads):
#                 print("Selecting Specific Leads!")
#                 output = [out[leads] for out in output]
                
    
#             totalm.append(output[4])
#             classm.append(output[5])
#             ypredm.append(output[6])
#             ylabsm.append(output[7])
#             shuffidsm.append(output[8])
#             print("Loaded %s, %s, %s, and %s for run %i, predictor %s" % (vnames[4],vnames[5],vnames[6],vnames[7],i,varnames[v]))
        
#         #print(totalm)
#         # Append to array
#         totalacc.append(totalm)
#         classacc.append(classm)
#         ypred.append(ypredm)
#         ylabs.append(ylabsm)
#         shuffids.append(shuffidsm)
    
#     # Turn results into arrays
#     totalacc = np.array(totalacc) # [predictor x run x lead]
#     classacc = np.array(classacc) # [predictor x run x lead x class]
#     ypred    = np.array(ypred)    # [predictor x run x lead x sample] # Last array (tercile based) is not an even sample size...
#     ylabs    = np.array(ylabs)    # [predictor x run x lead x sample]
#     shuffids = np.array(shuffids) # [predictor x run x lead x sample]
    
#     # Add to dictionary
#     outputs = (totalacc,classacc,ypred,ylabs,shuffids)
#     expdict = {}
#     dictkeys = ("totalacc","classacc","ypred","ylabs","shuffids")
#     for k,key in enumerate(dictkeys):
#         expdict[key] = outputs[k]
#     alloutputs.append(expdict)
#     print(varnames)

#%% Load persistence baseline

fpath = "../../CESM_data/"
fnp   = "AMVClassification_Persistence_Baseline_ens40_RegionNone_maxlead24_step3_nsamplesNone_detrend%i_020pctdata.npz" % detrend

ldp = np.load(fpath+fnp,allow_pickle=True)#.f#.arr_0

persaccclass = np.array(ldp['arr_0'][None][0]['acc_by_class']) # [Lead x Class]}
persacctotal = np.array(ldp['arr_0'][None][0]['total_acc'])


#%% Load the case for all predictors

expdir   = "FNN4_128_ALL"
varname  = "ALL"
flist = glob.glob("%s%s/Metrics/leadtime_testing_%s*ALL.npz"%(datpath,expdir,varname))
flist.sort()
nruns = len(flist)
#print('Found %i files for %s'%(nruns,varnames[v]))

# Load Result for each model
totalm    = []
classm    = []
ypredm    = []
ylabsm    = []
shuffidsm = []
for i in range(nruns): # Load for 10 files

    output,vnames = load_result(flist[i],debug=False)
    
    
    if len(output[4]) > len(leads):
        print("Selecting Specific Leads!")
        output = [out[leads] for out in output]
        

    totalm.append(output[4])
    classm.append(output[5])
    ypredm.append(output[6])
    ylabsm.append(output[7])
    shuffidsm.append(output[8])
    print("Loaded %s, %s, %s, and %s for run %i, predictor %s" % (vnames[4],vnames[5],vnames[6],vnames[7],i,varnames[v]))




# %% The Section below does visualizations for a single experiment
# Set the experiment number here

expnum     = 0
totalacc,classacc,ypred,ylabs,shuffids=am.unpack_expdict(alloutputs[expnum])

#%% Visualize Accuracy by Class, compare between predictors

nvar       = len(varnames)
nruns      = 50#totalacc.shape[1]
nleads     = len(leads)
plotmodels = np.arange(0,5)
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
            plotacc = classacc[i,:,:,c].max(0)
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
#%% Same as above plot, but specificall for AGU
plotmodels = [0,1,2,4]
ex         = expnum
add_conf   = True
plotconf   = 0.95
plotmax    = False # Set to True to plot maximum

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
        
        mu        = classacc[i,:50,:,c].mean(0)
        sigma     = classacc[i,:50,:,c].std(0)
        
        sortacc  = np.sort(classacc[i,:,:,c],0)
        idpct    = sortacc.shape[0] * plotconf
        lobnd   = np.floor(idpct).astype(int)
        hibnd   = np.ceil(sortacc.shape[0]-idpct).astype(int)
        
        
        ax.plot(leads,mu,color=varcolors[i],marker=varmarker[i],alpha=1.0,lw=2.5,label=varnames[i],zorder=9)
        if add_conf:
            if plotconf:
                ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=.3,color=varcolors[i],zorder=1,label="")
            else:
                ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.4,color=varcolors[i],zorder=1)
        
    ax.plot(leads,persaccclass[:,c],color=dfcol,label="Persistence",ls="dashed")
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
        
plt.savefig("%sPredictor_Intercomparison_byclass_plotmax%i_%s_AGUver.png"% (figpath,plotmax,expdirs[expnum]),
            dpi=200,bbox_inches="tight",transparent=True)

#%% Make the same plot as above, but output in increments

plotmodels = [0,2,1,4]
ex         = expnum
add_conf   = True
plotconf   = 0.95
fill_alpha = 0.20
plotmax    = False # Set to True to plot maximum


def init_accplot(c,figsize=(6,4),labelx=True,labely=True):
    # Initialize plot
    fig,ax = plt.subplots(1,1,figsize=(6,4),constrained_layout=True)

    ax.set_title("%s" %(classes[c]),fontsize=16,)
    ax.set_xlim([0,24])
    ax.set_xticks(leads)
    ax.set_ylim([0,1])
    ax.set_yticks(np.arange(0,1.25,.25))
    ax.grid(True,ls='dotted')
    
    ax.plot(leads,persaccclass[:,c],color='w',label="Persistence",lw=lwall,ls="solid")
    ax.hlines([0.33],xmin=-1,xmax=25,ls="dashed",color=dfcol,label="Random Chance")
    
    # if labely:
    #     ax.set_ylabel("Accuracy")
    # if labelx:
    #     ax.set_xlabel("Prediction Lead (Years)")
    
    return fig,ax


# Just Plot AMV+, introduce each predictor
c = 0

# Plot just the baseline
pcounter = 0
fig,ax = init_accplot(c)
ax.legend(ncol=3,fontsize=10)
savename = "%sPredictor_Intercomparison_byclass_plotmax%i_%s_AGUver_%02i.png"% (figpath,plotmax,expdirs[expnum],pcounter)
# Initial Save
plt.savefig(savename,
            dpi=200,bbox_inches="tight",transparent=True)
pcounter += 1
for p in np.arange(1,len(plotmodels)+1):
    
    print(plotmodels[:p])
    plotmodels_loop= plotmodels[:p]
    fig,ax = init_accplot(c)
    
    for i in plotmodels_loop:
        
        
        
        # Plot Predictor
        if plotmax:
            plotacc = classacc[i,:,:,c].max(0)
        else:
            plotacc = classacc[i,:,:,c].mean(0)
        mu        = classacc[i,:50,:,c].mean(0)
        sigma     = classacc[i,:50,:,c].std(0)
        sortacc  = np.sort(classacc[i,:,:,c],0)
        idpct    = sortacc.shape[0] * plotconf
        lobnd   = np.floor(idpct).astype(int)
        hibnd   = np.ceil(sortacc.shape[0]-idpct).astype(int)
        
        ax.plot(leads,mu,color=varcolors[i],marker=varmarker[i],markersize=8,
                alpha=1.0,lw=2.5,label=varnamesplot[i],zorder=9)
        if add_conf:
            if plotconf:
                ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=fill_alpha,color=varcolors[i],zorder=1,label="")
            else:
                ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.4,color=varcolors[i],zorder=1)
        
        ax.legend(ncol=3,fontsize=10)
        
    
    savename = "%sPredictor_Intercomparison_byclass_plotmax%i_%s_AGUver_%02i.png"% (figpath,plotmax,expdirs[expnum],pcounter)
    # Initial Save
    plt.savefig(savename,
                dpi=200,bbox_inches="tight",transparent=True)
    pcounter += 1


for c in [1,2]:
    fig,ax = init_accplot(c,labelx=False,labely=False)
    
    for i in plotmodels:
        
        # Plot Predictor
        if plotmax:
            plotacc = classacc[i,:,:,c].max(0)
        else:
            plotacc = classacc[i,:,:,c].mean(0)
        mu        = classacc[i,:50,:,c].mean(0)
        sigma     = classacc[i,:50,:,c].std(0)
        sortacc  = np.sort(classacc[i,:,:,c],0)
        idpct    = sortacc.shape[0] * plotconf
        lobnd   = np.floor(idpct).astype(int)
        hibnd   = np.ceil(sortacc.shape[0]-idpct).astype(int)
        
        ax.plot(leads,mu,color=varcolors[i],marker=varmarker[i],markersize=8,
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



# fig,axs = plt.subplots(1,3,figsize=(18,4))

# for c in range(3):
    
#     # Initialize plot
#     ax = axs[c]
#     ax.set_title("%s" %(classes[c]),fontsize=16,)
#     ax.set_xlim([0,24])
#     ax.set_xticks(leads)
#     ax.set_ylim([0,1])
#     ax.set_yticks(np.arange(0,1.25,.25))
#     ax.grid(True,ls='dotted')
    
#     for i in plotmodels:
#         if plotmax:
#             plotacc = classacc[i,:,:,c].max(0)
#         else:
#             plotacc = classacc[i,:,:,c].mean(0)
#        # ax.plot(leads,plotacc,color=varcolors[i],alpha=1,lw=lwall,label=varnames[i])
        
#         mu        = classacc[i,:50,:,c].mean(0)
#         sigma     = classacc[i,:50,:,c].std(0)
        
#         sortacc  = np.sort(classacc[i,:,:,c],0)
#         idpct    = sortacc.shape[0] * plotconf
#         lobnd   = np.floor(idpct).astype(int)
#         hibnd   = np.ceil(sortacc.shape[0]-idpct).astype(int)
        
        
#         ax.plot(leads,mu,color=varcolors[i],marker="o",alpha=1.0,lw=2.5,label=varnames[i],zorder=9)
#         if add_conf:
#             if plotconf:
#                 ax.fill_between(leads,sortacc[lobnd,:],sortacc[hibnd],alpha=.3,color=varcolors[i],zorder=1,label="")
#             else:
#                 ax.fill_between(leads,mu-sigma,mu+sigma,alpha=.4,color=varcolors[i],zorder=1)
        
#     ax.plot(leads,persacctotal,color=dfcol,label="Persistence",ls="dashed")
#     ax.axhline(.33,color=dfcol,label="Random Chance",ls="dotted")
    
        
#         # Add max/min predictability dots (removed this b/c it looks messy)
#         # ax.scatter(leads,classacc[i,:,:,c].max(0),color=varcolors[i])
#         # ax.scatter(leads,classacc[i,:,:,c].min(0),color=varcolors[i])
        
#     #ax.plot(leads,autodat[::3,c],color='k',ls='dotted',label="AutoML",lw=lwall)

        

        
# plt.savefig("%sPredictor_Intercomparison_byclass_plotmax%i_%s_AGUver.png"% (figpath,plotmax,expdirs[expnum]),
#             dpi=200,bbox_inches="tight",transparent=True)



#%%
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


