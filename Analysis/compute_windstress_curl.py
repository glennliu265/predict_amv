#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Wind Stress Curl from CESM1

Uses gradient functions from Theo Carr (https://github.com/ktcarr/)

Created on Mon Jul 10 11:24:17 2023

@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs

#%% Functions from Theo

def ddzeta(data, zeta, lonname='longitude',latname='latitude'):
    """Differentiate data WRT zeta using centered differences.
    - zeta argument is a string indicating the coordinate name to diff.,
    and is assumed to increase monotonically"""
    
    ## Empty array to hold results
    ddata_dzeta = xr.ones_like(data.isel({zeta : slice(1,-1)}))
    
    ## Compute differences
    data_plus = data.isel({zeta : slice(2, None)})
    data_minus = data.isel({zeta : slice(None, -2)})
    ddata = data_plus - data_minus.values
    
    ## reassign coordinate as center differences
    ddata = ddata.assign_coords({zeta : ddata_dzeta[zeta]})

    ## Get dzeta

    
    gridsize = get_gridsize(lat=ddata_dzeta[latname].values, lon=ddata_dzeta[lonname].values,
                            lonname=lonname,latname=latname)
    if zeta == lonname:
        dzeta = gridsize["dx"]
    elif zeta == latname:
        dzeta = gridsize["dy"]
    else:
        print("Not a valid coordinate")
    
    ## compute derivative
    ddata_dzeta.values = ddata / (2*dzeta)

    return ddata_dzeta


def curl(u, v, lonname, latname):
    """compute curl of vector field"""
    

    ## Compute derivatives
    dudy = ddzeta(u, zeta=latname,lonname=lonname,latname=latname)#.isel(latitude=slice(1, -1))
    dvdx = ddzeta(v, zeta=lonname,lonname=lonname,latname=latname)#.isel(longitude=slice(1, -1))

    return dvdx - dudy

def div(u, v, lonname, latname):
    """compute divergence of vector field"""

    ## Compute derivatives
    dudx = ddzeta(u, zeta=lonname,lonname=lonname,latname=latname)#.isel(latitude=slice(1, -1))
    dvdy = ddzeta(v, zeta=latname,lonname=lonname,latname=latname)#.isel(longitude=slice(1, -1))

    return dudx + dvdy

def get_gridsize(lat, lon, dlat=None, dlon=None,lonname='longitude',latname='latitude'):
    """Grid the size of each gridcell"""
    
    if dlat is None:
        dlat = np.nanmean(lat[1:] - lat[:-1]) # Get mean latitude difference
    if dlon is None:
        dlon = np.nanmean(lon[1:] - lon[:-1])
    
    ## Constants
    dlat_rad = dlat / 180.0 * np.pi
    dlon_rad = dlon / 180.0 * np.pi
    R = 6.378e6  # earth radius (meters)

    ## height of gridcell doesn't depend on longitude
    dy = R * dlat_rad  # unit: meters
    dy *= np.ones([len(lat), len(lon)])

    ## Compute width of gridcell
    lat_rad = lat / 180 * np.pi  # latitude in radians
    dx = R * np.cos(lat_rad) * dlon_rad
    dx = dx[:, None] * np.ones([len(lat), len(lon)])

    ## Compute area
    A = dx * dy
    
    ## Put in dataset
    coords    = {latname: lat, lonname: lon}
    dims      = (latname, lonname)
    grid_dims = xr.Dataset(
        {"A": (dims, A), "dx": (dims, dx), "dy": (dims, dy)}, coords=coords
    )
    return grid_dims

# xxxxxxxxxxxxxxxxxxxx
#%% User Edits
# xxxxxxxxxxxxxxxxxxxx

datpath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/Predictors/"

#%% Open wind stress curl files

ds_taux   = xr.open_dataset(datpath+"CESM1LE_TAUX_NAtl_19200101_20051201_bilinear_detrend0_regridNone.nc").load()
ds_tauy   = xr.open_dataset(datpath+"CESM1LE_TAUY_NAtl_19200101_20051201_bilinear_detrend0_regridNone.nc").load()
ds_slp    =xr.open_dataset(datpath+"CESM1LE_SLP_NAtl_19200101_20051201_bilinear_detrend0_regridNone.nc").load()

lon       = ds_taux.lon
lat       = ds_taux.lat
grid_dims = get_gridsize(lat.values,lon.values,latname='lat',lonname='lon')
curl_tau  = curl(-ds_taux.TAUX,-ds_tauy.TAUY,'lon','lat')

#%% Save the variable

# Replace into array with full box, but pad with zeros
nens,nyr,_,_  = curl_tau.shape
curl_tau_fill = np.zeros((nens,nyr,lat.shape[0],lon.shape[0]))
curl_tau_fill[:,:,1:-1,1:-1] = curl_tau.values.copy()

coords_dict = ds_taux.coords
dims        = ('ensemble', 'year','lat','lon')
ds_curl_tau = xr.Dataset(
    {"TAUCURL": (dims,curl_tau_fill)}, coords=coords_dict)
savename    = datpath+"CESM1LE_TAUCURL_NAtl_19200101_20051201_bilinear_detrend0_regridNone.nc"
ds_curl_tau.to_netcdf(savename,encoding={"TAUCURL": {'zlib': True}})

#%% Visualize the quivers
iens     = 5
iyear    = 10
plotvars = [-ds_taux.TAUX.isel(ensemble=iens,year=iyear),
            -ds_tauy.TAUY.isel(ensemble=iens,year=iyear),
            curl_tau.isel(ensemble=iens,year=iyear)
            ]


qint = 3
cint = np.arange(-300,320,20)
cint_curl = np.arange(-2.,2.,0.1) * 1e-7
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},
                      figsize=(12,6),constrained_layout=True)

#ax.set_extent([-50,-10,30,55])
ax.coastlines(color='k')

# Plot Curl
plotvar = plotvars[2]
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                        vmin=cint_curl[0],vmax=cint_curl[-1],cmap="RdBu_r")#plotvars[a].plot(ax=ax)
fig.colorbar(pcm,ax=ax)

# Plot Quivers
plotvar = plotvars[1]
ax.quiver(plotvar.lon.values[::qint],plotvar.lat.values[::qint],
          plotvars[0].values[::qint,::qint],plotvars[1].values[::qint,::qint],
          color="gray")

# Plot SLP
cl = ax.contour(ds_slp.lon,ds_slp.lat,ds_slp.SLP.isel(ensemble=iens,year=iyear),levels=cint,
                linewidths=0.75,colors="magenta",alpha=0.95)
ax.clabel(cl,)
ax.set_title("Wind Stress Curl (color), SLP (contours) and Wind Stress (quivers)\n Ens %i Year %i" % (ds_taux.ensemble.values[iens],
                                                     ds_taux.year.values[iyear]))                                                 
                                                                                                      
                                                 
#%% CHeck distnace


klon = 22
klat = 2

print("Lon is %f, lat is %f, new lon is %f" % (lon[klon],lat[klat],lon[klon+1]))
print("Computed distance is %f meters" % (grid_dims.dx.isel(lon=klon,lat=klat).values))



#%% Check computed wind stress curl

itime = 25
iens  = 2

print("Tauy / dx - Taux / dy +  = curlTau")
print("%f / %f - %f / %f = %e" % (-ds_tauy.TAUY.isel(lon=klon,lat=klat,year=itime,ensemble=iens).values,
                                  grid_dims.dx.isel(lon=klon,lat=klat).values,
                                  -ds_taux.TAUX.isel(lon=klon,lat=klat,year=itime,ensemble=iens).values,
                                  grid_dims.dy.isel(lon=klon,lat=klat).values,
                                  curl_tau.sel(lon=lon[klon],lat=lat[klat],method='nearest').isel(year=itime,ensemble=iens).values
                                  ))

#%% Visualize wind stress curl

iens     = 2
iyear    = 4
plotvars = [ds_taux.TAUX.isel(ensemble=iens,year=iyear),
            ds_tauy.TAUY.isel(ensemble=iens,year=iyear),
            curl_tau.isel(ensemble=iens,year=iyear)
            ]
varnames = ["TAUX","TAUY","Wind Stress Curl"]
fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()})

for a in range(3):
    ax      = axs[a]
    ax.coastlines()
    plotvar = plotvars[a]
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar)#plotvars[a].plot(ax=ax)
    fig.colorbar(pcm,ax=ax,fraction=0.026,orientation='horizontal')
    ax.set_title(varnames[a])
    

