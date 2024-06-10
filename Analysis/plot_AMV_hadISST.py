#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Copied from plot_clim_CESM.py
Make AMV Pattern and Index Plots for Presentation

Created on Thu May 30 16:37:10 2024

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
import sys

import matplotlib.ticker as mticker


#%% Import Custom Modules

amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd




#%% User Edits
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/"#"../../CESM_data/"
outpath = "../../../CESM_data/Figures/"
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

#%%% Load HadISST

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

# Calculate AMV and AMV Pattern
#amvid,amvpattern=proc.calc_AMVquick(ssta,lonh,lath,bbox,runmean=False,)




amvid = calc_AMV_index('NAT',ssta[None,:,:,:],lath,lonh,lp=True,dtr=dtr)
amvidstd = amvid/amvid.std(1)[:,None] # Standardize
amvid = amvid.squeeze()
amvidraw= calc_AMV_index('NAT',ssta[None,:,:,:],lath,lonh,lp=False,dtr=dtr)
amvidraw = amvidraw.squeeze()

# Calculate undetrended version
amvid_undtr     = calc_AMV_index('NAT',ssta[None,:,:,:],lath,lonh,lp=True,dtr=False)
amvidraw_undtr  = calc_AMV_index('NAT',ssta[None,:,:,:],lath,lonh,lp=False,dtr=False)

# Regress back to sstanomalies to obtain AMV pattern
#ssta   = ssta.transpose(1,0,2,3) # [time x ens x lon x lat]
sstar  = ssta.reshape(nmon,nlat*nlon) 
beta,_=regress_2d(amvidstd.squeeze(),sstar)
beta_undtr,_ = regress_2d((amvid_undtr/amvid_undtr.std(1)[:,None]).squeeze(),sstar)
amvpath = beta
amvpath = amvpath.reshape(nlat,nlon)



#%%


# amvidraw = amvid


plotdark   = True
plot_undtr = False
pdark      = True

if pdark:
    plt.style.use('dark_background')
    basecol = "w"
else:
    plt.style.use('default')
    basecol = "k"

# Plot the AMV Index

timeplot = np.arange(0,len(amvid),1)
fig,ax = plt.subplots(1,1,figsize=(8,3))
ax.grid(True,ls='dotted')
ax.set_xticks(timeplot[::120])
ax.set_xticklabels(timesyr[::120])
#ax.plot(timeplot,amvid,label="AMV Index",color='gray',lw=.75,ls='dashdot')

if plot_undtr:
    ax.set_title("Undetrended AMV Index (HadISST)")
    maskneg = amvidraw_undtr.squeeze()<0
    maskpos = amvidraw_undtr.squeeze()>=0
    ax.bar(timeplot[maskneg],amvidraw_undtr.squeeze()[maskneg],label="AMV-",color='cornflowerblue',width=1,alpha=1)
    ax.bar(timeplot[maskpos],amvidraw_undtr.squeeze()[maskpos],label="AMV+",color='tomato',width=1,alpha=1)
    # ax.plot(timeplot,amvidraw_undtr.squeeze(),label="Undetrended NASST Index",
    #         color="lightgray",lw=.75,ls='solid')
    ax.plot(timeplot,amvid_undtr.squeeze(),label="10-yr Low-Pass Filter",
            color=basecol,lw=1.2,ls='solid')
else:
    ax.set_title("AMV Index (HadISST)")
    maskneg = amvidraw<0
    maskpos = amvidraw>=0
    ax.bar(timeplot[maskneg],amvidraw[maskneg],label="AMV-",color='cornflowerblue',width=1,alpha=1)
    ax.bar(timeplot[maskpos],amvidraw[maskpos],label="AMV+",color='tomato',width=1,alpha=1)
    ax.plot(timeplot,np.convolve(amvid,np.ones(20)/20,mode='same'),label="10-yr Low-Pass Filter",color=basecol,lw=1.2)
ax.axhline([0],color=basecol,ls='dashed',lw=0.9)
ax.set_ylabel("AMV Index ($^{\circ}C$)")
ax.set_ylim([-1,1])
ax.set_xlim([0,len(amvid)])
ax.set_xlabel("Years")

ax.legend(fontsize=10,ncol=4)
plt.tight_layout()
if plotdark:
    plt.savefig("%sHadISST_AMV_Index_intime_ECML_detrend%i_dark.png"% (outpath,not plot_undtr),dpi=200,transparent=True)
else:
    plt.savefig("%sHadISST_AMV_Index_intime_ECML_detrend%i.png"% (outpath,not plot_undtr),dpi=200)

#%% Plot the Timeseries (updated version)

fsz_title = 22
fsz_axis  = 18
fsz_ticks = 16


plot_undtr = False
#add_

for ii in range(2):
        
    fig,ax = plt.subplots(1,1,figsize=(12,4))
    ax.set_xticks(timeplot[::120])
    ax.set_xticklabels(timesyr[::120],fontsize=fsz_ticks)
    
    
    
    # Plotting
    if plot_undtr:
        #ax.set_title("Undetrended AMV Index (HadISST)",fontsize=fsz_title)
        maskneg = amvidraw_undtr.squeeze()<0
        maskpos = amvidraw_undtr.squeeze()>=0
        ax.bar(timeplot[maskneg],amvidraw_undtr.squeeze()[maskneg],label="NASST-",color='cornflowerblue',width=1,alpha=1)
        ax.bar(timeplot[maskpos],amvidraw_undtr.squeeze()[maskpos],label="NASST+",color='tomato',width=1,alpha=1)
        # ax.plot(timeplot,amvidraw_undtr.squeeze(),label="Undetrended NASST Index",
        #         color="lightgray",lw=.75,ls='solid')
        if ii == 1:
            ax.plot(timeplot,amvid_undtr.squeeze(),label="10-yr Low-Pass Filter",
                    color=basecol,lw=2,ls='solid')
    else:
        #ax.set_title("AMV and NASST Index [$\degree C$]",fontsize=fsz_title)
        maskneg = amvidraw<0
        maskpos = amvidraw>=0
        ax.bar(timeplot[maskneg],amvidraw[maskneg],label="NASST-",color='cornflowerblue',width=1,alpha=1)
        ax.bar(timeplot[maskpos],amvidraw[maskpos],label="NASST-",color='tomato',width=1,alpha=1)
        if ii == 1:
            #ax.plot(timeplot,amvid_undtr.squeeze(),label="10-yr Low-Pass Filter",
            #        color=basecol,lw=1.2,ls='solid')
            ax.plot(timeplot,np.convolve(amvid,np.ones(20)/20,mode='same'),label="AMV Index",color=basecol,lw=2)
    ax.axhline([0],color=basecol,ls='dashed',lw=0.8)
    #ax.set_ylabel("AMV Index ($^{\circ}C$)")
    ax.set_ylim([-1,1])
    ax.set_xlim([0,len(amvid)])
    ax.set_xlabel("Year",fontsize=fsz_axis)
    ax.spines[['right', 'top']].set_visible(False)
    
    ax.legend(ncols=3,fontsize=fsz_ticks-4,loc="upper center")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(labelsize=fsz_ticks)
    
    
    plt.savefig("%sHadISST_AMV_Index_intime_SLS_detrend%i_frame%0i.png"% (outpath,plot_undtr,ii+1),dpi=200,bbox_inches='tight',transparent=True)


#%% Plot Spatial Pattern

# amvpath = amvpattern.T
# lon = lonh
# lat = lath


bbox2 = [-90,5,-5,65]

# Plot Ense
cints=np.arange(-0.60,0.65,0.05)
cintsl = np.arange(-0.6,0.7,0.1)
fig,ax = plt.subplots(1,1,figsize=(5,5),subplot_kw={'projection':ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=bbox2)
ax = viz.plot_box(bbox,ax=ax,linestyle='dashed',linewidth=2,color=basecol)
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


#%% Do same plot as above, but using orthomap

# Set the font sizes
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Avenir'
fsz_ticks = 18

proj     = ccrs.PlateCarree(0)
bbox3    = [-90,10,-5,65]

fig,ax,_ = viz.init_orthomap(1,1,bbox3,figsize=(16,8))

ax       = viz.add_coast_grid(ax,bbox=bbox3,fill_color='gray',line_color='k',fontsize=fsz_ticks)
pcm      = ax.contourf(lonh,lath,amvpath,levels=cints,cmap=cmocean.cm.balance,transform=proj)
cl       = ax.contour(lonh,lath,amvpath,levels=cints,colors="dimgray",linewidths=0.55,transform=proj)


viz.plot_box([-80,0,0,60],ax=ax,color='w',linestyle="dashed",proj=proj,linewidth=2)
#ax.clabel(cl,fontsize=8,fmt="%.2f")

cb       = fig.colorbar(pcm,ax=ax,fraction=0.055,orientation='horizontal',pad=0.01)
cb.ax.tick_params(labelsize=fsz_ticks)
cb.set_label("AMV Pattern [$\degree C \, \sigma_{AMV}^{-1}$]",fontsize=fsz_axis)

savename = "%sHadISST_AMVPAttern_EnsaVg_detrend%i_orthomap_dark.png"% (outpath,dtr)
plt.savefig(savename,dpi=200,transparent=True,bbox_inches='tight')

#%% Load and make the same plot for CESM1
nppath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/Targets/"
npname = "CESM1LE_label_NAT_NASST_index_detrend0_regridNone_norm0.npy"#"CESM1LE_label_NAT_NASST_index_detrend1_regridNone_norm1.npy"
amvids = np.load(nppath+npname)

cesmyr = np.arange(86) + 1920
xtks   = np.arange(1920,2010,10)
fig,ax = plt.subplots(1,1,figsize=(12,4))

for e in range(42):
    if e == 0:
        label="NASST (CESM1 Ensemble Member)"
        alpha=0.5
    else:
        label=""
        alpha=0.25
        
    ax.plot(cesmyr,amvids[e,:],alpha=alpha,c='gray',label=label)
ax.plot(cesmyr,np.convolve(amvids[36,:],np.ones(10)/10,mode='same')
        ,c='lightgray',alpha=0.9,label="Ens. Member 37")


# bwidth = .2
# amvidraw = amvids[e,:]
# maskneg = amvidraw<0
# maskpos = amvidraw>=0
# ax.bar(cesmyr[maskneg],amvidraw[maskneg],label="NASST-",color='cornflowerblue',width=bwidth,alpha=1)
# ax.bar(cesmyr[maskpos],amvidraw[maskpos],label="NASST-",color='tomato',width=bwidth,alpha=1)
#ax.plot(cesmyr,np.convolve(amvidraw,np.ones(20)/20,mode='same'),label="AMV Index",color=basecol,lw=2)

#ax.plot(amvids.mean(0))

ax.set_ylim([-.5,.5])
ax.set_xlim([1920,2005])

ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel("Year",fontsize=fsz_axis)
ax.set_xticks(xtks)
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.tick_params(labelsize=fsz_ticks)
ax.axhline([0],color=basecol,ls='dashed',lw=0.8)
ax.legend(ncols=3,fontsize=fsz_ticks-4,loc="upper center")

plt.savefig("%sCESM1_AMVPAttern_EnsaVg.png"% (outpath),dpi=200,transparent=True,bbox_inches='tight')
# ax.set_xticks(timeplot[::120])
# ax.set_xticklabels(timesyr[::120],fontsize=fsz_ticks)