#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check sensitivity of results to number of models


1) Load in the test accuracies for all 100 models
2) Plot # of models vs. mean accuracy to see how it converges/stabilizes


Copied upper section of viz_acc_bypredictor
Created on Fri Oct 20 10:55:07 2023

@author: gliu
"""



import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
from tqdm import tqdm

#%% Load my own custom modules

sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz,proc

sys.path.append("../")
sys.path.append("..")
import amv_dataloader as dl
import pamv_visualizer as pviz
import amvmod as am
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
expdirs      = ("FNN4_128_Singlevar_PaperRun",)
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
cwd             = os.getcwd()
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

#%% Load Test Metrics

alloutputs      = []
meanaccs_all    = []
for expdir in expdirs:
    output_byvar       = []
    output_acc_extreme = []
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
        
        # Take mean of extreme AMV accuracies
        output_acc_extreme.append(expdict['class_acc'][:,:,[0,2]].mean(2)) # [nmodel x leadtime]
    meanaccs_all.append(output_acc_extreme)
    alloutputs.append(output_byvar)

meanaccs_byvar = np.array(meanaccs_all) # [1 x var x nmodel x leadtime]

#%% For a given leadtime + variable, randomly select n samples and compute the variance in mean

ilead   = 25
ivar    = 2
in_acc  = meanaccs_byvar[0,ivar,:,ilead] # [nmodels]
nmodels = len(in_acc)
nvars   = meanaccs_byvar.shape[1]
# Get mean accuracy
mu_acc = np.nanmean(in_acc)

# Indicate iterations, preallocate, and loop
'''
1. Looping for a given number of networks N : [1,100]
    A. For a given number of simulations
        i . Select random number of networks
        ii. Compute mean accuracy
        
'''
mciter      = 10000
num_models  = np.arange(1,nmodels+1,1) # Number of models for plotting
samplemeans = np.zeros((nmodels,mciter))
for N in tqdm(range(nmodels)):
    for mc in range(mciter):
        selection = np.random.choice(in_acc,size=N+1) # Add one so you select at least 1
        samplemeans[N,mc] = np.nanmean(selection)

#%% Plot convergence

tol   = 1e-4 # Set Tolerance around the mean
plotN = 50 # Vertical line at given number


fig,ax = plt.subplots(1,1)

mu_iter = samplemeans.mean(1)
lb,hb   = np.percentile(samplemeans,axis=1,q=[2.5,97.5])


ax.plot(num_models,mu_iter,label="Monte Carlo Mean")
#ax.fill_between(num_models,lb,hb,color="cornflowerblue",alpha=0.2)
ax.axhline([mu_acc],color="k",label="True Mean (100 Networks)")
ax.fill_between(num_models,mu_acc-tol,mu_acc+tol,color="cornflowerblue",alpha=0.2,label="%e"%tol)
ax.axvline([plotN],color='r')
ax.set_xlabel("Number of Networks Averaged (N)")
ax.set_ylabel("Mean Acc. of N Networks \n (Mean of %s Iterations)" % (mciter))
ax.grid(True,ls='dotted')
ax.legend()
ax.set_title("Accuracy Convergence for Extreme AMV Events\n Predictor: %s, Lead: %02i" % (varnames[ivar],leads[ilead]))

figname = "%sAccuracy_Convergence_%s_lead%02i_maxiter%i.png" % (figpath,varnames[ivar],leads[ilead],mciter)
plt.savefig(figname,dpi=150)
#ax.set_ylim([0.6,0.7])


#%% now repeat loop for leadtimes and variables (Copy section for above)


mciter        = 10000
nleads        = len(leads)
meanacc_iters = np.zeros((nvars,nleads,nmodels))
nmodels       = len(in_acc)
nvars         = meanaccs_byvar.shape[1]
for ivar in range(nvars):
    for ilead in range(nleads):
        
        # Select Variables
        in_acc  = meanaccs_byvar[0,ivar,:,ilead] # [nmodels]
        
        # Get mean accuracy
        mu_acc = np.nanmean(in_acc)
        
        # Indicate iterations, preallocate, and loop
        '''
        1. Looping for a given number of networks N : [1,100]
            A. For a given number of simulations
                i . Select random number of networks
                ii. Compute mean accuracy
                
        '''
        num_models  = np.arange(1,nmodels+1,1) # Number of models for plotting
        samplemeans = np.zeros((nmodels,mciter))
        for N in tqdm(range(nmodels)):
            for mc in range(mciter):
                selection = np.random.choice(in_acc,size=N+1) # Add one so you select at least 1
                samplemeans[N,mc] = np.nanmean(selection)
        
        meanacc_iters[ivar,ilead,:] = np.nanmean(samplemeans,1)

