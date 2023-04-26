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


import cartopy.crs as ccrs

import sys

# Import my own custom module....
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz

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
    

def init_ablation_maps(bbox_plot,figsize=(10,8),fill_color="k"):

    fig,axs = plt.subplots(2,2,constrained_layout=True,
                           subplot_kw={'projection':ccrs.PlateCarree()},figsize=figsize)
    
    for a in range(4):
        ax = axs.flatten()[a]
        blabel=[0,0,0,0]
        
        if a%2 == 0:
            blabel[0] = 1
        if a>1:
            blabel[-1] = 1
        ax = viz.add_coast_grid(ax,bbox=bbox_plot,fill_color=fill_color,blabels=blabel)
    return fig,axs
    
    




