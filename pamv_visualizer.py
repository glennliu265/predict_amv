#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict AMV Visualizer

Visualization/Plots for Predict AMV Project

Created on Thu Apr  6 22:46:26 2023

@author: gliu
"""

import matplotlib.pyplot as plt
import numpy as np

def format_acc_plot(leads,ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_xlim([leads[0],leads[-1]])
    if len(leads) == 9:
        ax.set_xticks(leads)
    else:
        ax.set_xticks(leads[::3])
    ax.set_ylim([0,1])
    ax.set_yticks(np.arange(0,1.25,.25))
    ax.grid(True,ls='dotted')
    ax.minorticks_on()
    ax.set_title(sp_titles[a],fontsize=20)
    if a == 0:
        ax.set_ylabel("Accuracy")
    if a == 1:
        ax.set_xlabel("Prediction Leadtime (Years)")
    return ax
    

def init_classacc_fig(leads,sp_titles=None):
    fig,axs=plt.subplots(1,3,constrained_layout=True,figsize=(18,4),sharey=True)
    if sp_titles is None:
        sp_titles=["AMV+","Neutral","AMV-"]
    for a,ax in enumerate(axs):
        ax.set_xlim([leads[0],leads[-1]])
        if len(leads) == 9:
            ax.set_xticks(leads)
        else:
            ax.set_xticks(leads[::3])
        ax.set_ylim([0,1])
        ax.set_yticks(np.arange(0,1.25,.25))
        ax.grid(True,ls='dotted')
        ax.minorticks_on()
        ax.set_title(sp_titles[a],fontsize=20)
        if a == 0:
            ax.set_ylabel("Accuracy")
        if a == 1:
            ax.set_xlabel("Prediction Leadtime (Years)")
    return fig,axs
    
    
    




