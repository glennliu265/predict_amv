#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:37:05 2023

@author: gliu
"""

# 42 x 86

yrs = np.tile(np.arange(1,87)[:,None],42).T + 1919
ens = np.tile(np.arange(1,43)[:,None],86)


# 
l = 24
yr24  = yrs[:,l:].flatten()
ens24 = ens[:,l:].flatten()



l = 12
yr12  = yrs[:,l:].flatten()
ens12 = ens[:,l:].flatten()



def check(yr,ens,idx):
    print("Idx is %i " % idx)
    print("\tYear is %i " % yr[idx])
    print("\tEns is %i " % ens[idx])
    return yr[idx],ens[idx]



check(yr24,ens24,3000)
# Check if lead 12 is equivalent to just going 12 years back..
check(yr24,ens24,3000-12)
#check(yr12,ens12,)

check(yr12,ens12,3000-(12*41))

#%%
def make_ensyr(ens=42,yr=86,meshgrid=True):
    """Make either meshgrid or index array for [nens] x [nyr]"""
    if meshgrid:
        yrs = np.tile(np.arange(0,yr)[:,None],ens).T #+ startyr # [ens x yr]
        ens = np.tile(np.arange(0,ens)[:,None],yr) #[ens x yr]
        return yrs,ens
    else:
        id_ensyr = np.zeros((ens,yr),dtype='object')
        for e in range(ens):
            for y in range(yr):
                id_ensyr[e,y] = (e,y)
        return id_ensyr # [ens x yr]


def get_ensyr_linear(lead,linearids,
              reflead=0,nens=42,nyr=86,
              apply_lead=True,ref_lead=True,
              return_labels=False):
    """
    Given linear indices for a ens x year array where the lead/lag has been applied...
    Retrieve the corresponding linear indices for a reference lead/lag application
    Also optionally recover the lead and ensemble member labels
    
    Parameters
    ----------
    lead (INT)          : Lead applied to data
    linearids (LIST)    : Linear indices to find
    reflead (INT)       : Lead applied to reference (default = 0)
    nens     (INT)      : Number of ensemble members, default is 42
    nyr      (INT)      : Number of years, default is 86
    apply_lead (BOOL)   : True to apply lead, false to apply lag to data
    ref_lead  (BOOL)    : Same but for reference set
    return_labels   (BOOL) : Set to true to return ens,yr labels for a dataset

    Returns
    -------


    """

    # Get the arrays (absolute)
    yrs,ens  = make_ensyr(ens=nens,yr=nyr,meshgrid=True)
    id_ensyr = make_ensyr(ens=nens,yr=nyr,meshgrid=False)
    in_arrs  = [yrs,ens,id_ensyr]

    # Apply lead/lag
    if apply_lead: # Lead the data
        apply_arr = [arr[:,lead:].flatten() for arr in in_arrs]
    else: # Lag the data
        apply_arr = [arr[:,:lead].flatten() for arr in in_arrs]

    # Get the corresponding indices where lead/lag is applied
    apply_ids = [arr[linearids] for arr in apply_arr]
    yrslead,enslead,idlead = apply_ids

    # Find the corresponding indices where it is not flattened, at a specified lead
    if ref_lead: # Lead the data
        ref_arr = [arr[:,reflead:].flatten() for arr in in_arrs]
    else: # Lag the data
        ref_arr = [arr[:,:reflead].flatten() for arr in in_arrs]
    refyrs,refens,refids = ref_arr

    ref_linearids = [] # Contains linear ids in the lead
    for ii,ens_yr_set in enumerate(idlead):
        sel_ens,sel_yr = ens_yr_set
        foundid = np.where((refyrs ==sel_yr) * (refens == sel_ens))[0]
        assert len(foundid) == 1
        ref_linearids.append(foundid[0])
        
        if debug:
            print("For linear id %i..." % (linearids[ii]))
            print("\tApplied Lead id          : %s" % (str(ens_yr_set)))
            print("\tReference Lead (l=%i) id : %s" % (reflead,refids[foundid[0]]))
            print("\tReference linear id      : %i" % (foundid[0]))
        assert refids[foundid[0]] == ens_yr_set
    if return_labels:
        return ref_linearids,refids[ref_linearids]
    else:
        return ref_linearids

nens      = 42
nyr       = 86
lead      = 24
reflead   = 0
apply_lead = True # Lead is applied to current
ref_lead   = True # Lead is applied to reference
linearids = np.array([0,1,2,3])

ref_linearids,refids = get_ensyr(lead,linearids,
              reflead=reflead,nens=nens,nyr=nyr,
              apply_lead=apply_lead,ref_lead=ref_lead,
              return_labels=True)

#%% Function Drafting
nens      = 42
nyr       = 86
lead      = 24
reflead   = 12
apply_lead = True # Lead is applied to current
ref_lead   = True # Lead is applied to reference
linearids = np.array([2555,1414,444,22,892])


# Get the arrays (absolute)
yrs,ens  = make_ensyr(ens=nens,yr=nyr,meshgrid=True)
id_ensyr = make_ensyr(ens=nens,yr=nyr,meshgrid=False)
in_arrs  = [yrs,ens,id_ensyr]

# Apply lead/lag
if apply_lead: # Lead the data
    apply_arr = [arr[:,lead:].flatten() for arr in in_arrs]
else: # Lag the data
    apply_arr = [arr[:,:lead].flatten() for arr in in_arrs]

# Get the corresponding indices where lead/lag is applied
apply_ids = [arr[linearids] for arr in apply_arr]
yrslead,enslead,idlead = apply_ids
# yrslead  = yrsflat[linearids]
# enslead  = ensflat[linearids]
# idlead   = id_flat[linearids]

# Find the corresponding indices where it is not flattened, at a specified lead
if ref_lead: # Lead the data
    ref_arr = [arr[:,reflead:].flatten() for arr in in_arrs]
else: # Lag the data
    ref_arr = [arr[:,:reflead].flatten() for arr in in_arrs]
refyrs,refens,refids = ref_arr
# refyrs = yrs[:,reflead:].flatten()
# refens = ens[:,reflead:].flatten()
# refids = id_ensyr[:,reflead:].flatten()

ref_linearids = [] # Contains linear ids in the lead
for ii,ens_yr_set in enumerate(idlead):
    sel_ens,sel_yr = ens_yr_set
    foundid = np.where((refyrs ==sel_yr) * (refens == sel_ens))[0]
    assert len(foundid) == 1
    ref_linearids.append(foundid[0])
    
    if debug:
        print("For linear id %i:" % (linearids[ii]))
        print("\tApplied Lead id was %s" % (str(ens_yr_set)))
        print("\tReference Lead (%i) found was %s" % (reflead,refids[foundid[0]]))
        print("\tWhere linear index is %i" % (foundid[0]))
    assert refids[foundid[0]] == ens_yr_set

    
    
#%% Debug session

lead      = leadmax
linearids = shuffidx_max
reflead   = lead
nens      = nens
nyr       = ntime
apply_lead = True
ref_lead    = False
return_labels = True
debug         = True

    