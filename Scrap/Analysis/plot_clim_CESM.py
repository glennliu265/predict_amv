#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Summary Plots


Created on Thu Mar 25 19:49:10 2021

@author: gliu
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean

from scipy.signal import butter,filtfilt,detrend

from tqdm import tqdm


#%% User Edits
datpath = "../../CESM_data/"
outpath = "../../CESM_data/Figures/"
# Read in the data

# Set File Names
vnames = ['sst','sss','psl']
vnamelong = ["Sea Surface Temperature (degC)","Sea Surface Salinity (psu)","Sea Level Pressure (hPa)"]
fn1 = "CESM1LE_sst_NAtl_19200101_20051201_Regridded2deg.nc"
fn2 = "CESM1LE_sss_NAtl_19200101_20051201_Regridded2deg.nc"
fn3 = "CESM1LE_psl_NAtl_19200101_20051201_Regridded2deg.nc"
fns = [fn1,fn2,fn3]

# Plotting Box
bbox = [-80,0,0,65] # North Atlantic [lonW, lonE, latS, latN]



#%% Functions

def add_coast_grid(ax,bbox=[-180,180,-90,90],proj=None):
    """
    Add Coastlines, grid, and set extent for geoaxes
    
    Parameters
    ----------
    ax : matplotlib geoaxes
        Axes to plot on 
    bbox : [LonW,LonE,LatS,LatN], optional
        Bounding box for plotting. The default is [-180,180,-90,90].
    proj : cartopy.crs, optional
        Projection. The default is None.

    Returns
    -------
    ax : matplotlib geoaxes
        Axes with setup
    """
    if proj is None:
        proj = ccrs.PlateCarree()
    ax.add_feature(cfeature.COASTLINE,color='black',lw=0.75)
    ax.set_extent(bbox)
    gl = ax.gridlines(crs=proj, draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle="dotted",lw=0.75)
    gl.top_labels = False
    gl.right_labels = False
    return ax#%%


def calc_AMV_index(region,invar,lat,lon,lp=False,order=5,cutofftime=120,dtr=False):
    """
    Select bounding box for a given AMV region for an input variable
        "SPG" - Subpolar Gyre
        "STG" - Subtropical Gyre
        "TRO" - Tropics
        "NAT" - North Atlantic
    
    Parameters
    ----------
    region : STR
        One of following the 3-letter combinations indicating selected region
        ("SPG","STG","TRO","NAT")
        
    var : ARRAY [Ensemble x time x lat x lon]
        Input Array to select from
    lat : ARRAY
        Latitude values
    lon : ARRAY
        Longitude values    

    Returns
    -------
    amv_index [ensemble x time]
        AMV Index for a given region/variable

    """
    
    # Select AMV Index region
    bbox_SP = [-60,-15,40,65]
    bbox_ST = [-80,-10,20,40]
    bbox_TR = [-75,-15,0,20]
    bbox_NA = [-80,0 ,0,65]
    regions = ("SPG","STG","TRO","NAT")        # Region Names
    bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA) # Bounding Boxes
    
    # Get bounding box
    bbox = bboxes[regions.index(region)]
    
    # Select Region
    selvar = invar.copy()
    klon = np.where((lon>=bbox[0]) & (lon<=bbox[1]))[0]
    klat = np.where((lat>=bbox[2]) & (lat<=bbox[3]))[0]
    selvar = selvar[:,:,klat[:,None],klon[None,:]]
    
    # Take mean ove region
    amv_index = np.nanmean(selvar,(2,3))
    
    # If detrend
    if dtr:
        for i in range(amv_index.shape[0]):
            amv_index[i,:] = detrend(amv_index[i,:])
    
    if lp:
        
        for i in range(amv_index.shape[0]):
            amv_index[i,:]=lp_butter(amv_index[i,:],cutofftime,order)
        
    
    return amv_index

def regress_2d(A,B,nanwarn=1):
    """
    Regresses A (independent variable) onto B (dependent variable), where
    either A or B can be a timeseries [N-dimensions] or a space x time matrix 
    [N x M]. Script automatically detects this and permutes to allow for matrix
    multiplication.
    
    Returns the slope (beta) for each point, array of size [M]
    
    
    """
    # Determine if A or B is 2D and find anomalies
    
    # Compute using nan functions (slower)
    if np.any(np.isnan(A)) or np.any(np.isnan(B)):
        if nanwarn == 1:
            print("NaN Values Detected...")
    
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
            
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.nanmean(A,axis=a_axis)[:,None]
            Banom = B - np.nanmean(B,axis=b_axis)
            
        
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.nanmean(A,axis=a_axis)
            Banom = B - np.nanmean(B,axis=b_axis)[None,:]
        
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom = np.nansum(Aanom2,axis=a_axis)    
        
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        
        b = (np.nansum(B,axis=b_axis) - beta * np.nansum(A,axis=a_axis))/A.shape[a_axis]
    else:
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
            
            
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.mean(A,axis=a_axis)[:,None]
            Banom = B - np.mean(B,axis=b_axis)
            
        
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.mean(A,axis=a_axis)
            Banom = B - np.mean(B,axis=b_axis)[None,:]
        
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom = np.sum(Aanom2,axis=a_axis)    
        
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        
        b = (np.sum(B,axis=b_axis) - beta * np.sum(A,axis=a_axis))/A.shape[a_axis]
    
    
    return beta,b

def plot_box(bbox,ax=None,return_line=False,leglab="Bounding Box",color='k',linestyle='solid',linewidth=1):
    
    """
    Plot bounding box
    Inputs:
        1) bbox [1D-ARRAY] [lonW,lonE,latS,latN]
        Optional Arguments...
        2) ax           [axis] axis to plot onto
        3) return_line  [Bool] return line object for legend labeling
        4) leglabel     [str]  Label for legend
        5) color        [str]  Line Color, default = black
        6) linestyle    [str]  Line style, default = solid
        7) linewidth    [#]    Line width, default = 1  
    
    
    """
    if ax is None:
        ax = plt.gca()
    # Plot North Boundary
    ax.plot([bbox[0],bbox[1]],[bbox[3],bbox[3]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_')
    # Plot East Boundary
    ax.plot([bbox[1],bbox[1]],[bbox[3],bbox[2]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_')
    # Plot South Boundary
    ax.plot([bbox[1],bbox[0]],[bbox[2],bbox[2]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_')
    # Plot West Boundary
    ax.plot([bbox[0],bbox[0]],[bbox[2],bbox[3]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_')
    
    if return_line == True:
        linesample =  ax.plot([bbox[0],bbox[0]],[bbox[2],bbox[3]],color=color,ls=linestyle,lw=linewidth,label=leglab)
        return ax,linesample
    return ax


def lp_butter(varmon,cutofftime,order):
    # Input variable is assumed to be monthy with the following dimensions:
    flag1d=False
    if len(varmon.shape) > 2:
        nmon,nlat,nlon = varmon.shape
    else:
        
        flag1d = True
        nmon = varmon.shape[0]
    
    # Design Butterworth Lowpass Filter
    filtfreq = nmon/cutofftime
    nyquist  = nmon/2
    cutoff = filtfreq/nyquist
    b,a    = butter(order,cutoff,btype="lowpass")
    
    # Reshape input
    if flag1d is False: # For 3d inputs, loop thru each point
        varmon = varmon.reshape(nmon,nlat*nlon)
        # Loop
        varfilt = np.zeros((nmon,nlat*nlon)) * np.nan
        for i in tqdm(range(nlon*nlat)):
            varfilt[:,i] = filtfilt(b,a,varmon[:,i])
        
        varfilt=varfilt.reshape(nmon,nlat,nlon)
    else: # 1d input
        varfilt = filtfilt(b,a,varmon)
    return varfilt
# %%Read in the variables
dsopen = []
values = [] # [lon x lat x time x ensemble]
for i,f in enumerate(tqdm(fns)):
    ds = xr.open_dataset(datpath+f)
    val = ds[vnames[i]].values
    
    
    
    
    dsopen.append(ds)
    values.append(val)


# Convert SST To deg C
values[0] = values[0]-273.15

# Apply Mask to PSL
msk = values[0].sum((2,3))
msk[~np.isnan(msk)] = 1
values[2] *= msk[:,:,None,None]

# Read in lat/lon
lat = ds.lat.values
lon = ds.lon.values


#%% Calculate Climatology, Standard Deviation, and plot
climmn  = [] # Mean
climstd = [] # Stdev
climvar = []
for v in values:
    climmn.append(np.nanmean(v,(2,3)))
    climstd.append(np.nanstd(v,(2,3)))
    climvar.append(np.nanvar(v,(2,3)))


#%% Plot the Results (climatological mean)

fig,axs=plt.subplots(1,3,figsize=(14,4.5),subplot_kw={'projection':ccrs.PlateCarree()})
varplot = climmn

v = 0
ax = axs[v]
ax = add_coast_grid(ax,bbox)
pcm1 = ax.pcolormesh(lon,lat,varplot[v],cmap=cmocean.cm.thermal)
ax.set_title(vnamelong[v])
fig.colorbar(pcm1,ax=ax,fraction=0.05,orientation='horizontal',pad=0.08)
ax.add_feature(cfeature.LAND,facecolor='black')


v = 1
ax = axs[v]
ax = add_coast_grid(ax,bbox)
pcm2 = ax.pcolormesh(lon,lat,varplot[v],vmin=30,vmax=37,cmap=cmocean.cm.dense)
ax.set_title(vnamelong[v])
fig.colorbar(pcm2,ax=ax,fraction=0.05,orientation='horizontal',pad=0.08)
ax.add_feature(cfeature.LAND,facecolor='black')

v = 2
ax = axs[v]
ax = add_coast_grid(ax,bbox)
pcm3 = ax.pcolormesh(lon,lat,varplot[v]/100,cmap=cmocean.cm.balance)
ax.set_title(vnamelong[v])
fig.colorbar(pcm3,ax=ax,fraction=0.05,orientation='horizontal',pad=0.08)
ax.add_feature(cfeature.LAND,facecolor='black')

plt.suptitle("Climatological Mean of Predictors in CESM1-LE (1920-2005)")
plt.savefig(outpath+"Fig01_Top_CESM_ClimPlots.png",dpi=200)


#%% Plot variance of each variable


fig,axs=plt.subplots(1,3,figsize=(14,4.5),subplot_kw={'projection':ccrs.PlateCarree()})
varplot = climstd

v = 0
ax = axs[v]
ax = add_coast_grid(ax,bbox)
pcm1 = ax.pcolormesh(lon,lat,varplot[v],vmin=0,vmax=8,cmap=cmocean.cm.thermal)
ax.set_title(vnamelong[v])
fig.colorbar(pcm1,ax=ax,fraction=0.05,orientation='horizontal',pad=0.08)
ax.add_feature(cfeature.LAND,facecolor='black')

v = 1
ax = axs[v]
ax = add_coast_grid(ax,bbox)
pcm2 = ax.pcolormesh(lon,lat,varplot[v],vmin=0,vmax=1,cmap=cmocean.cm.haline)
ax.set_title(vnamelong[v])
fig.colorbar(pcm2,ax=ax,fraction=0.05,orientation='horizontal',pad=0.08)
ax.add_feature(cfeature.LAND,facecolor='black')

v = 2
ax = axs[v]
ax = add_coast_grid(ax,bbox)
pcm3 = ax.pcolormesh(lon,lat,varplot[v]/100,vmin=0,vmax=10,cmap=cmocean.cm.ice)
ax.set_title(vnamelong[v])
fig.colorbar(pcm3,ax=ax,fraction=0.05,orientation='horizontal',pad=0.08)
ax.add_feature(cfeature.LAND,facecolor='black')

plt.suptitle(r"1 Standard Deviation of Input Predictors in CESM1-LE (1920-2005)")
plt.savefig(outpath+"Fig01_Top_CESM_StdPlots.png",dpi=200)


#%% Calculate AMV Indices and Spatial Pattern

# Calculate Monthly Anomalies
ssts = values[0].copy()
nlon,nlat,nmon,nens = ssts.shape
ssts = ssts.reshape(nlon,nlat,int(nmon/12),12,nens)
ssta = ssts - ssts.mean(2)[:,:,None,:,:]
ssta = ssta.reshape(nlon,nlat,nmon,nens)

# Transpose to [ens time lat lon]
ssta   = ssta.transpose(3,2,1,0)
amvid = calc_AMV_index('NAT',ssta,lat,lon)
amvidstd = amvid/amvid.std(1)[:,None] # Standardize

# Regress back to sstanomalies to obtain AMV pattern
#ssta   = ssta.transpose(1,0,2,3) # [time x ens x lon x lat]
sstar  = ssta.reshape(nens,nmon,nlat*nlon) 
amvpat = np.zeros((nens,nlat*nlon))*np.nan

for e in tqdm(range(nens)):
    
    sste = sstar[e,:,:]
    ide  = amvidstd[e,:]
    beta,_=regress_2d(ide,sste)
    amvpat[e,:] = beta
    
amvpat = amvpat.reshape(nens,nlat,nlon)

#%% Visualize AMV Pattern

bbox2 = [lon[0],5,-5,65]
# Plot Ense
cints=np.arange(-0.5,0.55,0.05)

fig,ax = plt.subplots(1,1,figsize=(5,5),subplot_kw={'projection':ccrs.PlateCarree()})
ax = add_coast_grid(ax,bbox=bbox2)
ax = plot_box(bbox,ax=ax,linestyle='dashed',linewidth=2)
pcm = ax.contourf(lon,lat,amvpat.mean(0).T,levels=cints,cmap=cmocean.cm.balance)
cl = ax.contour(lon,lat,amvpat.mean(0).T,colors='k',linewidths=0.75)
ax.clabel(cl,fontsize=8,fmt="%.2f")
ax.set_title("CESM1-LE AMV Spatial Pattern ($^{\circ}C / 1\sigma_{AMV}$) \n40-member Ensemble Average")
fig.colorbar(pcm,ax=ax,fraction=0.05,orientation='horizontal',pad=0.07)
ax.add_feature(cfeature.LAND,facecolor='gray')
#plt.tight_layout()
plt.savefig("%sCESMLE_AMVPAttern_EnsaVg.png"%outpath,dpi=200)

#%% Load preprocessed labels, visualize AMV Timeseries and dsitribution

# Load Post-processed Inputs (After Normalizing)
target  = np.load('/Users/gliu/Downloads/2020_Fall/6.862/Project/CESM_data/CESM_label_amv_index_detrend0.npy')
ens = 40
y_std = np.std(target)

# View in a timeseries sense
yrs = np.arange(1920,2006,1)

fig,ax = plt.subplots(1,1,figsize=(8,3))

ax.grid(True,ls='dotted')
for i in range(ens):
    plotdata = target[i,:]
    col = np.where(plotdata<=-y_std,'cornflowerblue',np.where(plotdata>y_std,'salmon','gray'))
    ax.scatter(yrs,plotdata,c=col,alpha=0.2)
ax.plot(yrs,target[3,:],label="1-member",color='k',lw=.75,ls='dashdot')
ax.plot(yrs,np.convolve(target[3,:],np.ones(10)/10,mode='same'),label="1-member (10-yr running mean)",color='k',lw=0.9)
ax.hlines([-y_std,y_std],xmin=1920,xmax=2005,color='k',ls='dashed',lw=0.9)
ax.hlines([0],xmin=1920,xmax=2005,color='w',ls='dotted',lw=0.9)
ax.set_ylabel("AMV Index ($^{\circ}C$)")
ax.set_ylim([-1.75,1.75])
ax.set_xlim([1920,2005])
ax.set_xlabel("Years")
ax.set_xticks(np.arange(1920,2005,10))
ax.set_title("AMV Index, Distribution by Year, \n $\sigma=%.4f^{\circ}C$"% (y_std))
ax.legend(fontsize=10,ncol=3)
plt.tight_layout()
plt.savefig("%sAMV_Index_intime_ECML.png"%outpath,dpi=200)


#%%% Do the same but for HadISST

dtr = True

# 
fn  = "hadisst.1870-01-01_2018-12-01.nc"
dsh = xr.open_dataset(datpath+fn)
ssth = dsh.sst.values # [time x lat x lon]
lath = dsh.lat.values
lonh = dsh.lon.values
times = dsh.time.values
timesmon = np.datetime_as_string(times,unit="M")
#timesmon = timesmon.astype('str')
timesyr  = np.datetime_as_string(times,unit="Y")[:]

# Calculate Monthly Anomalies
ssts = ssth.transpose(2,1,0)
nlon,nlat,nmon = ssts.shape
ssts = ssts.reshape(nlon,nlat,int(nmon/12),12)
ssta = ssts - ssts.mean(2)[:,:,None,:]
ssta = ssta.reshape(nlon,nlat,nmon)


# Transpose to [time lat lon]
ssta   = ssta.transpose(2,1,0)
amvid = calc_AMV_index('NAT',ssta[None,:,:,:],lath,lonh,lp=True,dtr=dtr)
amvidstd = amvid/amvid.std(1)[:,None] # Standardize
amvid = amvid.squeeze()
amvidraw= calc_AMV_index('NAT',ssta[None,:,:,:],lath,lonh,lp=False,dtr=dtr)
amvidraw = amvidraw.squeeze()

# Regress back to sstanomalies to obtain AMV pattern
#ssta   = ssta.transpose(1,0,2,3) # [time x ens x lon x lat]
sstar  = ssta.reshape(nmon,nlat*nlon) 
beta,_=regress_2d(amvidstd.squeeze(),sstar)
amvpath = beta
amvpath = amvpath.reshape(nlat,nlon)

#%%

plotdark=True

pdark = True
if pdark:
    plt.style.use('dark_background')
    basecol = "w"
else:
    plt.style.use('default')
    basecol = "k"

# Plot the AMV Index
maskneg = amvidraw<0
maskpos = amvidraw>=0
timeplot = np.arange(0,len(amvid),1)
fig,ax = plt.subplots(1,1,figsize=(8,3))
ax.grid(True,ls='dotted')
ax.set_xticks(timeplot[::120])
ax.set_xticklabels(timesyr[::120])
#ax.plot(timeplot,amvid,label="AMV Index",color='gray',lw=.75,ls='dashdot')
ax.bar(timeplot[maskneg],amvidraw[maskneg],label="AMV-",color='cornflowerblue',width=1,alpha=1)
ax.bar(timeplot[maskpos],amvidraw[maskpos],label="AMV+",color='tomato',width=1,alpha=1)
ax.plot(timeplot,np.convolve(amvid,np.ones(20)/20,mode='same'),label="10-yr Low-Pass Filter",color=basecol,lw=1.2)
ax.axhline([0],color=basecol,ls='dashed',lw=0.9)
ax.set_ylabel("AMV Index ($^{\circ}C$)")
ax.set_ylim([-1,1])
ax.set_xlim([0,len(amvid)])
ax.set_xlabel("Years")
ax.set_title("AMV Index, Distribution by Year (HadISST)")
ax.legend(fontsize=10,ncol=3)
plt.tight_layout()
if plotdark:
    plt.savefig("%sHadISST_AMV_Index_intime_ECML_detrend%i_dark.png"% (outpath,dtr),dpi=200,transparent=True)
else:
    plt.savefig("%sHadISST_AMV_Index_intime_ECML_detrend%i.png"% (outpath,dtr),dpi=200)


# Plot Spatial Pattern
bbox2 = [lon[0],5,-5,65]
# Plot Ense
cints=np.arange(-0.60,0.65,0.05)
cintsl = np.arange(-0.6,0.7,0.1)
fig,ax = plt.subplots(1,1,figsize=(5,5),subplot_kw={'projection':ccrs.PlateCarree()})
ax = add_coast_grid(ax,bbox=bbox2)
ax = plot_box(bbox,ax=ax,linestyle='dashed',linewidth=2,color=basecol)
pcm = ax.contourf(lonh,lath,amvpath,levels=cints,cmap=cmocean.cm.balance)
cl = ax.contour(lonh,lath,amvpath,levels=cintsl,colors="k",linewidths=0.75)
ax.clabel(cl,fontsize=8,fmt="%.2f")
ax.set_title("HadISST AMV Spatial Pattern ($^{\circ}C / 1\sigma_{AMV}$) \n1870-2018")
fig.colorbar(pcm,ax=ax,fraction=0.05,orientation='horizontal',pad=0.07)
ax.add_feature(cfeature.LAND,facecolor='gray')
#plt.tight_layout()
if plotdark:
    plt.savefig("%sHadISST_AMVPAttern_EnsaVg_detrend%i_dark.png"% (outpath,dtr),dpi=200,transparent=True)
else:
    plt.savefig("%sHadISST_AMVPAttern_EnsaVg_detrend%i.png"% (outpath,dtr),dpi=200)



