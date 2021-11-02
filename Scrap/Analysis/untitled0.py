#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Script to combine results ()

Created on Thu Jul  1 11:37:19 2021

@author: gliu
"""

import numpy as np
#/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/Metrics/batch1_metrics
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/Metrics/"


# Process first batch

fnstart = "leadtime_testiang_ALL_AMVClass3_fractaldb_nepoch20_nens40_maxlead24_detrend0_noise0_unfreeze_allFalse_"

for run in range(3):
    for lead in range(25):
        expname = "run%i_unfreezeall_ALL.npz" % (run)
        #print(expname)
        
        fn =  datpath + "batch1_metrics/" + fnstart+expname
        
        ld = np.load(fn,allow_pickle=True)
        
        trainloss = ld['train_loss'] # [1, ]
        testloss  = ld['test_loss']
        trainacc  = ld['test_acc']
        totalacc = ld['total_acc']
        accbyclass = ld['acc_by_class']
        yvpred   = ld['yvalpred']
        yvlab = ld['yvallabels']
        sampid = ld['sampled_idx']
        



run0_unfreezeall_lead00of24.npz