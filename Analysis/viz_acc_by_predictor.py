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



#%% User Edits

varnames  = ("SST","SSS","PSL","BSF","SSH","HMXL")
varcolors = ("r","limegreen","pink","darkblue","purple","cyan")
expdirs    = ("FNN4_128_SingleVar","CNN2_singlevar")

datpath   = "../../CESM_data/"
figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20221209/"

# Old figpath: datpath + expdir + "/Figures/"

classes   = ["AMV+","Neutral","AMV-"] # [Class1 = AMV+, Class2 = Neutral, Class3 = AMV-]
leads     = np.arange(0,25,3)

# Plotting Parameters
lwall     = 2.5


#%% Functions

def load_result(fn,debug=False):
    """
    Load results for each of the variable names
    
    input: fn (str), Name of the file
    """
    
    ld = np.load(fn,allow_pickle=True)
    vnames = ld.files
    if debug:
        print(vnames)
    output = []
    for v in vnames:
        output.append(ld[v])
    return output,vnames



def retrieve_lead(shuffidx,lead,nens,tstep):
    
    orishape = [nens,tstep-lead]
    outidx   = np.unravel_index(shuffidx,orishape)
    return outidx

def unpack_expdict(expdict,dictkeys=None):
    if dictkeys is None:
        dictkeys = ("totalacc","classacc","ypred","ylabs","shuffids")
    unpacked = [expdict[key] for key in expdict]
    return unpacked

# def pack_expdict(outputs):
#     
#     expdict = {dictkeys[o]: outputs[o] for (dictkeys[o],outputs[o]) in range(len(outputs))}
#     return expdict



#%% Load the data
# Read in results

# Preallocate. Some quick definitions:
# model    : Network type [simplecnn,resnet50,resnet50(retrained)]
# run      : Run Number (1 to 10)
# leadtime : Leadtime in years, 0,24 in 3-year steps


alloutputs = []
for expdir in expdirs:
    totalacc = [] # Accuracy for all classes combined [model x run x leadtime]
    classacc = [] # Accuracy by class [model x run x leadtime x class]
    ypred    = [] # Predictions [model x run x leadtime x sample]
    ylabs    = [] # Labels [model x run x leadtime x sample]
    shuffids = [] # Indices [model x run x leadtime x sample]
    for v in range(len(varnames)):
        
        flist = glob.glob("%s%s/Metrics/leadtime_testing_%s*ALL.npz"%(datpath,expdir,varnames[v]))
        flist.sort()
        nruns = len(flist)
        print('Found %i files for %s'%(nruns,varnames[v]))
        
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
        
        #print(totalm)
        # Append to array
        totalacc.append(totalm)
        classacc.append(classm)
        ypred.append(ypredm)
        ylabs.append(ylabsm)
        shuffids.append(shuffidsm)
    
    # Turn results into arrays
    totalacc = np.array(totalacc) # [predictor x run x lead]
    classacc = np.array(classacc) # [predictor x run x lead x class]
    ypred    = np.array(ypred)    # [predictor x run x lead x sample] # Last array (tercile based) is not an even sample size...
    ylabs    = np.array(ylabs)    # [predictor x run x lead x sample]
    shuffids = np.array(shuffids) # [predictor x run x lead x sample]
    
    # Add to dictionary
    outputs = (totalacc,classacc,ypred,ylabs,shuffids)
    expdict = {}
    dictkeys = ("totalacc","classacc","ypred","ylabs","shuffids")
    for k,key in enumerate(dictkeys):
        expdict[key] = outputs[k]
    alloutputs.append(expdict)
    print(varnames)


# %% The Section below does visualizations for a single experiment
# Set the experiment number here

expnum     = 0
totalacc,classacc,ypred,ylabs,shuffids=unpack_expdict(alloutputs[expnum])

#%% Visualize Accuracy by Class, compare between predictors

nvar       = len(varnames)
nruns      = totalacc.shape[1]
nleads     = len(leads)
plotmodels = np.arange(0,nvar)
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
        ax.plot(leads,plotacc,color=varcolors[i],alpha=1,lw=lwall,label=varnames[i])
        
        
        # Add max/min predictability dots (removed this b/c it looks messy)
        # ax.scatter(leads,classacc[i,:,:,c].max(0),color=varcolors[i])
        # ax.scatter(leads,classacc[i,:,:,c].min(0),color=varcolors[i])
        
    #ax.plot(leads,autodat[::3,c],color='k',ls='dotted',label="AutoML",lw=lwall)
    #ax.plot(leads,persaccclass[:,c],color='k',label="Persistence",lw=lwall)

    ax.hlines([0.33],xmin=-1,xmax=25,ls="dashed",color='k')
        
    if c == 0:
        ax.legend(ncol=2,fontsize=10)
        ax.set_ylabel("Accuracy")
    if c == 1:
        ax.set_xlabel("Prediction Lead (Years)")
        
plt.savefig("%sPredictor_Intercomparison_byclass_plotmax%i_%s.png"% (figpath,plotmax,expdirs[expnum]),dpi=200)

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
        ax.plot(leads,plotacc,color="k",alpha=1,lw=lwall,label=varnames[v])
        
        ax.hlines([0.33],xmin=-1,xmax=25,ls="dashed",color='k')
            
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


#%% In this section below, we make some comparisons of skill by experiment...

#%% Sort by accuracies

#%% Get (in order) 

plotvar = classacc[v,:,l,c]
fig,ax = plt.subplots(1,1)


