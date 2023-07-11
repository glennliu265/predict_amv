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
    # if sp_title is not None:
    #     ax.set_title(sp_titles[a],fontsize=20)
    # if a == 0:
    #     ax.set_ylabel("Accuracy")
    # if a == 1:
    #     ax.set_xlabel("Prediction Leadtime (Years)")
    return ax
    

def init_classacc_fig(leads,sp_titles=None):
    fig,axs=plt.subplots(1,3,constrained_layout=True,figsize=(18,4),sharey=True)
    if sp_titles is None:
        sp_titles=["NASST+","Neutral","AMV-"]
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
    
    


def make_count_barplot(count_by_year,lead,target,thresholds_in,leadmax=24,classes=['AMV+', 'Neutral', 'AMV-'],
                       class_colors=('salmon', 'gray', 'cornflowerblue'),startyr=1870
                       ):
    # Target = [ens x year]
    # Thresholds_in [th1,th2,...,thN]
    
    timeaxis      = np.arange(0,len(target.squeeze()))
    timeaxis_in   = np.arange(leadmax,target.shape[1])
    
    fig,ax       = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))
    for c in range(3):
        label = classes[c]
        ax.bar(timeaxis_in+startyr,count_by_year[:,c],bottom=count_by_year[:,:c].sum(1),
               label=label,color=class_colors[c],alpha=0.75,edgecolor="white")
        
    ax.set_ylabel("Frequency of Predicted Class")
    ax.set_xlabel("Year")
    ax.legend()
    ax.minorticks_on()
    ax.grid(True,ls="dotted")
    #ax.set_xlim([1880,2025])
    #ax.set_ylim([0,450])

    ax2 = ax.twinx()
    # ax2.plot(timeaxis,target.squeeze(),color="k",label="HadISST NASST Index")
    # ax2.set_ylabel("NASST Index ($\degree C$)")
    # ax2.set_ylim([-1.3,1.3])
    # for th in thresholds_in:
    #     ax2.axhline([th],color="k",ls="dashed")
    # ax2.axhline([0],color="k",ls="solid",lw=0.5)
    axs = [ax,ax2]
    return fig,axs

