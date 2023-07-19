#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:58:32 2023

@author: gliu
"""



eparams = {'netname':'FNN4_128'}

inputsize = 4485
nclasses  = 3
nlon = 65
nlat = 69










niter = 1000
for it in range(niter):
    pmodel = am.recreate_model(eparams['netname'],nn_param_dict,inputsize,nclasses,nlon=nlon,nlat=nlat)
    nnweights = pmodel[0].weight.detach().numpy()
    nunits,ninput = nnweights.shape
    if it == 0:
        weights_all = np.zeros((niter,nunits,ninput))
    
    weights_all[it,:,:] = nnweights.copy()
    
    
# Ok, it seems like the weights are different, distributed between e-4 to e-2    
plt.pcolormesh(weights_all.var(0)),plt.colorbar()