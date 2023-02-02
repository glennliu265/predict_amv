#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Recombine Lag Metrics

Script to fix a silly mistake where I have two lag file styles

One is 0,3, ... 24 in intervals of 3, other is missed lags.
Just change the datpath to the metrics folder, and this will automatically combine both files
and move it into a folder /proc/ 


Created on Wed Feb  1 15:26:47 2023

@author: gliu
"""


import numpy as np
import glob
import sys



sys.path.append("../")
import amvmod as am

sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz,proc

ll1 = np.arange(0,25,3)
ll2 = [a for a in np.arange(0,26,1) if a not in ll1]


nleads = len(ll1) + len(ll2)

llidx = [ll1,ll2]




#%%


datpath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/LENS_30_1950/FNN4_128_ALL_canesm2_lens_Train/Metrics/"
datpath_new = datpath + "proc/"
debug   = False


proc.makedir(datpath_new)
#%%


flist = glob.glob(datpath+"*.npz")
flist.sort()
print(len(flist))



for run in range(int(len(flist)/2)):
    print("Processing run%02i ..." % run)
    
    runlist = list(filter(lambda fn: "run%02i" % run in fn,flist))
    runlist.sort()
    
    outputs1,vnames = am.load_result(runlist[0],debug=False)
    outputs2,vnames = am.load_result(runlist[1],debug=False)
    
    outputs_new = []
    for v in range(len(vnames)):
        if debug:
            print("Variable %i, %s" % (v,vnames[v]))
            print(outputs1[v].shape)
            print(outputs2[v].shape)
        
        # Preallocate for lead x otherdims
        if len(outputs1[v].shape) > 1:
            newshape = (nleads,) + outputs1[v].shape[1:] # [26 x otherdims]
            concat_v = np.empty(newshape,dtype='object')
        else:
            concat_v = np.empty(nleads,dtype='object')
            
        # Read in the variables, placing them in the correct leadtimes
        
        concat_v[ll1,...] = outputs1[v]
        concat_v[ll2,...] = outputs2[v]
        
        # Set to the same dtype
        dtype = outputs1[v].dtype
        concat_v = concat_v.astype(dtype)
        
        # Append to output
        outputs_new.append(concat_v)
        
    # Save the new output
    
    # Manually Coded
    # _,pos  = proc.get_stringnum(runlist[0],"maxlead",nchars=1,verbose=True,return_pos=True)
    # newstr = runlist[0]
    # newstr = runlist[0][:pos] + str(nleads) + runlist[0][pos+2:]
    
    # Python String
    newstr  = runlist[0].replace('maxlead24',"maxlead%i"%nleads)
    
    newstr  = newstr.replace(datpath,datpath_new)
    dictnew = dict(zip(vnames,outputs_new)) 
    np.savez(newstr,**dictnew,allow_pickle=True)
    
    
    

    
    
        
        
        
        
            
        
    
    