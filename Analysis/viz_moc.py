#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize the MOC variable from CESM1-LENS


Created on Mon Mar  6 12:43:07 2023

@author: gliu
"""


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

datpath = "/Users/gliu/"

#"/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/ocn/proc/tseries/monthly/MOC/"
fn      = 'b.e11.B20TRC5CNBDRD.f09_g16.002.pop.h.MOC.192001-200512.nc'

#
region_names=np.array([b'Global Ocean - Marginal Seas',
       b'Atlantic Ocean + Mediterranean Sea + Labrador Sea + GIN Sea + Arctic Ocean + Hudson Bay'],
      dtype='|S256')
moc_comp = np.array([b'Eulerian Mean', b'Eddy-Induced (bolus)', b'Submeso'], dtype='|S256')

# Load the data
ds      = xr.open_dataset(datpath+fn)
moc     = ds.MOC.values # [time x transport_reg x moc_comp x moc_z x lat_aux]
z       = ds.moc_z.values/100 # Convert cm --> meters
lat     = ds.lat_aux_grid.values
times   = ds.time.values

times   = [str(t) for t in times]
yrs = [t[:4] for t in times]


def maxid_2d(invar,debug=False):
    x1,x2     = invar.shape
    idmax     = np.argmax(invar.flatten())
    idx1,idx2 = np.unravel_index(idmax,invar.shape)
    return idx1,idx2

    
#%% Plot AMOC Streamfunction Maximum Location
icomp    = 0
iregion  = 1

fig,ax      = plt.subplots(1,1)
plotmoc     = moc[:,iregion,icomp,:,:].mean(0)
idz,idlat   = maxid_2d(plotmoc) 
cf          = ax.contourf(lat,z,plotmoc)
cl          = ax.contour(lat,z,plotmoc,colors="k",linewidths=0.7)
ax.clabel(cl,)
ax.plot(lat[idlat],z[idz],marker="x",color='k',ls="",markersize=10,label="$\psi_{max}$")       
ax.legend()
ax.invert_yaxis()
plt.colorbar(cf)
ax.set_title("Mean AMOC Streamfunction\n$z_{max}$: %.2fm, Latitude$_{max}$: %.2f$\degree$" % (z[idz],lat[idlat]))
ax.set_xlabel("Latitude")
ax.set_ylabel("Depth (m)")

#%% Plot AMOC Timeseries

fig,ax      = plt.subplots(1,1)
plot_ts     = moc[:,iregion,icomp,idz,idlat]

xtks        = np.arange(0,len(times)+1,120)
xtk_labels  = np.array(yrs)[xtks]




ax.plot(plot_ts)
ax.set_title("AMOC Strength\n$z_{max}$: %.2fm, Latitude$_{max}$: %.2f$\degree$" % (z[idz],lat[idlat]))
ax.set_xlabel("Time (months)")
ax.set_ylabel("AMOC Strength at Maximum Streamfunction (Sv)")
ax.grid(True,ls='dotted')

ax.set_xticks(xtks)
ax.set_xticklabels(xtk_labels)












