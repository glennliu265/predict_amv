#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 00:04:12 2023

@author: gliu
"""

# Get indices
nsamples        = y.shape[0]
percent_splits  = [percent_train,1-percent_train-percent_val,percent_val]
segments        = ("Train","Test","Validation")
cumulative_pct  = 0
segment_indices = []
for p,pct in enumerate(percent_splits):
    # Add modulo accounting for offset
    pct_rng = np.array([cumulative_pct+offset,cumulative_pct+pct+offset])%1
    if pct_rng[0] == pct_rng[1]:
        print("Exceeded max percentage on segment [%s], Skipping..."%segments[p])
        continue
    # Add ranges to account for endpoints
    shift_flag=False # True: Offset shifts the chunk beyond 100%, 4 points required
    if pct_rng[0] > pct_rng[1]: # Shift to beginning of dataset
        pct_rng = np.array([pct_rng[0],1,0,pct_rng[1]]) # [larger idx, end, beginning, smaller idx]
        shift_flag=True
    
    # Get range of indices
    idx_rng = np.floor(nsamples*pct_rng).astype(int)
    
    if shift_flag:
        seg_idx = np.concatenate([np.arange(idx_rng[0],idx_rng[1]),np.arange(idx_rng[2],idx_rng[3])])
        segment_indices.append(seg_idx)
        if debug:
            print("Range of percent for %s segment is [%.2f to 1] and [0 to %.2f], idx %i:%i" % (segments[p],
                                                                                  pct_rng[1],pct_rng[0],
                                                                                  idx_rng[0],idx_rng[1],
                                                                                  idx_rng[2],idx-rng[3]
                                                                                               ))
    else:
        idx_rng = np.floor(nsamples*pct_rng).astype(int)
        segment_indices.append(np.arange(idx_rng[0],idx_rng[1]))
        if debug:
            print("Range of percent for %s segment is %.2f to %.2f, idx %i:%i" % (segments[p],
                                                                                  pct_rng[0],pct_rng[1],
                                                                                  segment_indices[p][0],segment_indices[p][-1]
                                                                                               ))
    cumulative_pct += pct
    # End pct Loop
    
    
    

