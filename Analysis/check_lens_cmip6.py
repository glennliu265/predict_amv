#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check CMIP6 LENS

- copied from check_lens_data.py, but for cmip6 rather than cmip5 mmle
- copied preprocessing steps from prep_data_lens.py (Part 1, prior to deseason & detrend)

Created on Tue Feb  7 13:52:22 2023

@author: gliu
"""

import time
import numpy as np
import xarray as xr
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import cartopy.crs as ccrs

#%% User Edits

# I/O, dataset, paths
varname        = "zos" # (tos, sos, zos)
cesm_varname   = "SSH"
varunits       = "$\degree C$"
bbox           = [-90,10,0,65]
outpath        = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/CMIP6_LENS/analysis/"
# Preprocessing and Cropping Options

# Paths
machine = "Astraeus"
if machine == "stormtrack":
    lenspath       = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/ts/"
    datpath        = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/LENS_other/processed/"
elif machine == "gliu_mbp":
    datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/LENS_other/processed/"
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
    import viz,proc
elif machine == "Astraeus":
    datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/CMIP6_LENS/regridded/"
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
    cesm2path     = "/Users/gliu/Globus_File_Transfer/CESM2_LE/1x1/"
    
    import viz,proc


## Load some global information (lat.lon) <Note, I need to customize this better..
#%% Import packages and universal variables

# Note; Need to set script into current working directory (need to think of a better way)
import os
cwd = os.getcwd()

sys.path.append(cwd+"/../")
import predict_amv_params as pparams

# Import paths
figpath         = pparams.figpath
proc.makedir(figpath)

# Import class information
classes         = pparams.classes
class_colors    = pparams.class_colors

# Import dataset inforation
dataset_names   = pparams.cmip6_names
dataset_long    = pparams.cmip6_names
dataset_colors  = pparams.cmip6_colors
dataset_starts  = (1850,) * len(dataset_names)

#
proj            = pparams.proj

bbox_plot       = pparams.amvbbox
#

#%% Some functions
def drop_time_bnds(ds):
    return ds.drop('time_bnds')
#%% Get dataset lists

ndata  = len(dataset_names)
nclists = []
for d in range(ndata):
    if dataset_names[d] == "CESM2":
        ncsearch = "%s/%s/%s_%s*.nc" % (cesm2path,cesm_varname,cesm_varname,"LE2")
    else:
        ncsearch = "%s%s_%s*.nc" % (datpath,varname,dataset_names[d])
    nclist   = glob.glob(ncsearch)
    nclist.sort()
    print("Found %02i files for %s!" % (len(nclist),dataset_names[d]))
    nclists.append(nclist)
nens = [len(lst) for lst in nclists] # Get # of ensemble members based on list length


#%% Load out the data (copying prep_data_lens preprocessing step 1)
st_ld = time.time()
ds_all = []
for d in tqdm(range(ndata)):
    
    # <1> Concatenate Ensemble Members
    if dataset_names[d] == "CESM2":
        varname_in = cesm_varname
    else:
        varname_in = varname
    
    # Read in data [ens x time x lat x lon]
    dsall   = xr.open_mfdataset(nclists[d],concat_dim="ensemble",combine="nested",preprocess=drop_time_bnds)
    
    if not np.any((dsall.lon.values)<0): # Longitude not flipped
        print("Correcting longitude values for %s because no negative one was found..." % (dataset_names[d]))
        dsall.coords['lon'] = (dsall.coords['lon'] + 180) % 360 - 180
        dsall = dsall.sortby(dsall['lon'])
    dsall = dsall.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
    
    ds_all.append(dsall.load())
    
    
lon = ds_all[-1].lon.values
lat = ds_all[-1].lat.values
print("Loaded all data in %.2fs" % (time.time()-st_ld))

#%% Take mean over ensemble and save

close_ds_all = False

bbfn,bbtitle = proc.make_locstring_bbox(bbox,)
ds_ensmean_all = []
for d in tqdm(range(ndata)):
    
    # Take ensemble mean
    ds_ensmean  = ds_all[d].mean("ensemble")
    
    # Edit variable name
    if dataset_names[d] == "CESM2":
        varname_in = cesm_varname
    else:
        varname_in = varname
    varname_out = cesm_varname.upper()
    rename_dict = {varname_in:varname_out}
    ds_ensmean = ds_ensmean.rename(rename_dict)
    
    # Change the name
    encoding_dict = {varname_out : {'zlib': True}}
    savename    = "%s%s_%s_%s_EnsAvg.nc" % (outpath,dataset_names[d],varname_out,bbfn)
    ds_ensmean.to_netcdf(savename,encoding=encoding_dict)
    ds_ensmean_all.append(ds_ensmean)
    
    # Close the larger netCDF (might need to undo this part if I want to do more calculations)
    if close_ds_all:
        ds_all[d].close()
    

#%% Make a plot of ensemble and time mean variables over the North Atlantic


fig,axs = plt.subplots(2,3,constrained_layout=True,
                       subplot_kw={"projection":proj},
                       figsize=(16,8.5))
for d in tqdm(range(ndata)):
    
    ax = axs.flatten()[d]
    ax = viz.add_coast_grid(ax,bbox=bbox,fill_color="k")
    
    plotvar = ds_ensmean_all[d][varname_out].mean('time')
    
    if varname_out == "SSH": # SSH Plot Options
        if dataset_names[d] == "CESM2": # Conver to m
            plotvar  = plotvar / 100
        if dataset_names[d] =="ACCESS-ESM1-5":
            cints = np.arange(-4.4,-2.8,0.2)
        else:
            cints = np.arange(-1.8,1.6,0.2)
    if varname_out == "SST": # SST Plot Options
        plotvar = plotvar.squeeze()
        cints = np.arange(0,34,2)
    
    cf = ax.contourf(lon,lat,plotvar,levels=cints,cmap="jet")
    cl = ax.contour(lon,lat,plotvar,levels=cints,colors="k")
    ax.clabel(cl)
    
    ax.set_title("%s (n=%i)" % (dataset_names[d],nens[d]))
    fig.colorbar(cf,ax=ax,orientation='horizontal',fraction=0.055)

plt.suptitle("Ensemble and Time Mean %s (%s)" % (varname_out,varunits))

plt.savefig("%sCMIP6_LENS_%s_Ensmean_TimeMean.png" % (figpath,varname_out,),dpi=150,bbox_inches="tight")



#%%

# maybe investigate what is going on with ACCESS-1
d             = 0
all_colorbars = False # Set to True to have colorbars for all subplots
set_cints     = True  # Set to True to set cints manually


if d == 0:
    # Figure size for 30 ens ACCESS-ESM5-1
    nrows   = 5
    ncols   = 6
    figsizes = ((25,20),(24,18)) # [all_colorbar,shared_colorbar]
    if varname_out == "SSH":
        cints = np.arange(-4.4,-2.8,0.2)
    

# Adjust figure size depending on the colorbar setting
if all_colorbars:
    figsize = figsizes[0]
else:
    figsize = figsizes[1]
        
        
fig,axs = plt.subplots(nrows,ncols,constrained_layout=True,
                       subplot_kw={"projection":proj},
                       figsize=figsize)

for e in range(nens[d]):
    ax = axs.flatten()[e]
    ax.set_extent(bbox_plot)
    ax.coastlines()
    
    plotvar = ds_all[d].zos.isel(ensemble=e).mean('time')
    
    # Do plotting
    if set_cints:
        cf = ax.contourf(lon,lat,plotvar,levels=cints,cmap="jet")
        cl = ax.contour(lon,lat,plotvar,levels=cints,colors="k",linewidths=0.75)
    else:
        cf = ax.contourf(lon,lat,plotvar,cmap="jet")
        cl = ax.contour(lon,lat,plotvar,colors="k",linewidths=0.75)
    
    ax.clabel(cl)
    viz.label_sp("%02i" % (e+1),ax=ax,usenumber=True,fontsize=16,alpha=0.7,labelstyle="Ens%s")
    
    if all_colorbars:
        fig.colorbar(cf,ax=axs.flatten(),orientation='horizontal',fraction=0.055)
if all_colorbars is False:
    fig.colorbar(cf,ax=axs.flatten(),orientation='horizontal',fraction=0.02,pad=0.01)
plt.suptitle("Time Mean %s (%s)" % (varname_out,varunits),fontsize=20)
plt.savefig("%sCMIP6_LENS_%s_AllEns_TimeMean.png" % (figpath,varname_out,),dpi=150,bbox_inches="tight")



#%% Check what is going on...




#%% Below is just copied pasted.... nothing actuall yworks


#%% Load datasets in to dataarrays

# Load datasets for each

ds_all = []
ds_all_nomask = []
ds_landmask = []
ds_icemask  = []

for d in range(ndata):
    
    
    # Load masked data
    savename       = "%s%s_%s_NAtl_%sto2005_detrend%i_regrid%sdeg.nc" % (datpath,
                                                                         dataset_names[d],
                                                                         varname,
                                                                         dataset_starts[d],
                                                                         detrend,regrid)
    ds = xr.open_dataset(savename).load()
    ds_all.append(ds)
    
    # Load unmasked data
    savename       = "%s/../processed_nomask/%s_%s_NAtl_%sto2005_detrend%i_regrid%sdeg.nc" % (datpath,
                                                                         dataset_names[d],
                                                                         varname,
                                                                         dataset_starts[d],
                                                                         detrend,regrid)
    ds = xr.open_dataset(savename).load()
    ds_all_nomask.append(ds)
    
    # Load masks
    mmnames = ("land", "ice")
    mmds    = (ds_landmask, ds_icemask)
    for mm in range(2):
        savename       = "%s/../processed_nomask/%s_mask_%s_byens_regrid%sdeg.npy" % (datpath,
                                                                                   mmnames[mm],
                                                                                   dataset_names[d],
                                                                                   regrid
                                                                                   )
        
        msk = np.load(savename)
        # Load global lat/lon for selection
        if "CESM1" in dataset_names[d]:
            ds = xr.open_dataset("%s../ensAVG/%s_htr_ts_regrid%ideg_ensAVG_nomask.nc" % (datpath,dataset_names[d],regrid))
        else:
            ds = xr.open_dataset("%s../ensAVG/%s_ts_regrid%ideg_ensAVG_nomask.nc" % (datpath,dataset_names[d],regrid))
        longlob = ds.lon.values
        latglob = ds.lat.values
        # Quickly select the target region
        mskreg,lonr,latr = proc.sel_region(msk.transpose(2,1,0),longlob,latglob,bbox)
        mmds[mm].append(mskreg.transpose(2,1,0))
        
        
dataset_enssize = [len(ds.ensemble) for ds in ds_all] # Get Ensemble Sizes
samplesize      = dataset_enssize * (2005-np.array(dataset_starts)) # Rough Calculation of sample size

print([len(ds.lon) for ds in ds_all])
print([len(ds.lat) for ds in ds_all])

print([ds.lon for ds in ds_all])
# Get lat/lon
lon = ds_all[0].lon.values
lat = ds_all[0].lat.values

#%% Visualize land/ice masks

fig,axs = plt.subplots(2,ndata,figsize=(14,5.5),subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True)
for d in range(ndata):
    
    for ii in range(2):
        ax = axs[ii,d]
        ax.coastlines()
        ax.set_extent(amvbbox)
        
        
        plotpat = mmds[ii][d].mean(0)
        if ii == 0: # Plot the Land Mask
            ax.set_title("%s" % (dataset_long[d]))
        
            
        pcm = ax.pcolormesh(lon,lat,plotpat,cmap="RdBu_r")
        # ax.clabel(cl,cints[::2],fontsize=8)
        
        # if d == 0:
        #     ax.text(-0.05, 0.55, ylabelnames[ii], va='bottom', ha='center',rotation='vertical',
        #             rotation_mode='anchor',transform=ax.transAxes)
fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=.045)
#%% Decide whether or not to use land ice masks

ds_masked = ds_all.copy()
if apply_limasks:
    ds_in = ds_all.copy()
else:
    ds_in = ds_all_nomask.copy()
    
ds_all = []
for d in range(ndata): # Apply landmask
    # Apply landmask
    ds_all.append(ds_in[d] * ds_landmask[d][:,None,:,:])

#%% Compute NASST from each dataset

amvids    = []
amvids_lp = []
for d in range(ndata):
    ds        = ds_masked[d].sel(lon=slice(amvbbox[0],amvbbox[1]),lat=slice(amvbbox[2],amvbbox[3]))
    dsidx     = (np.cos(np.pi*ds.lat/180) * ds).mean(dim=('lat','lon'))
    amvids.append(dsidx)
    
    amvid_lp = proc.lp_butter(dsidx.sst.values.T[...,None],10,order=6).squeeze()
    amvids_lp.append(amvid_lp)
    
    # Save the labels
    savename       = "%s%s_nasst_label_%sto2005_detrend%i_regrid%sdeg.npy" % (datpath,
                                                                         dataset_names[d],
                                                                         dataset_starts[d],
                                                                         detrend,regrid)
    np.save(savename,dsidx.sst.values)
    print("Saved Target to %s"%savename)
    
#%% Visualize the ens-avg timeseries for each large ensemble

fig,ax = plt.subplots(1,1,figsize=(12,4),constrained_layout=True)

for d in range(ndata):
    t = np.arange(dataset_starts[d],2005+1,1)
    label = "%s (N=%i)" % (dataset_long[d],dataset_enssize[d])
    ax.plot(t,amvids[d].sst.mean('ensemble'),label="",color=dataset_colors[d],lw=2.5,alpha=0.5)
    ax.plot(t,amvids_lp[d].mean(1),label=label,color=dataset_colors[d],lw=1.5)
    

ax.axhline([0],ls="dashed",lw=0.75,color="k")
ax.set_ylim([-1.25,1.25])
ax.set_xlim([1920,2005])
ax.grid(True,ls="dotted")
ax.set_title("Ensemble Average AMV and NASST Index")
ax.set_xlabel("Years")
ax.set_ylabel("Index Value ($\degree$C)")
ax.legend()

savename = "%sNASST_%s_EnsAvg_Lens.png" % (figpath,bbox_fn)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Check out what is happening for particular datasets (map)

# ------------------------------------------------------------
# Plot Ann Mean Ens Avg SST Anomalies for a selected <<YEAR>>
# ------------------------------------------------------------
y=1950
fig,axs = plt.subplots(1,ndata,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True)
for d in range(ndata):
    ax = axs.flatten()[d]
    ax.coastlines()
    ax.set_extent(amvbbox)
    ax.set_title("%s" % (dataset_long[d]))
    
    plotvar = ds_all[d].sst.sel(year=y).mean("ensemble")
    print(plotvar)
    
    pcm     = ax.pcolormesh(lon,lat,plotvar,cmap="RdBu_r",vmin=-2,vmax=2)
    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.025)
plt.suptitle("Ensemble Mean SST ($\degree C$) for y=%i"%y,y=0.75)
    
savename = "%sAnnMeanSST_EnsAvg_LENS_y%i.png" % (figpath,y)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Plot first 5 ensemble members for a given model

t = 0
for d in range(ndata):
    
    fig,axs = plt.subplots(1,ndata,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()},
                           constrained_layout=True)
    
    for e in range(5):
        ax = axs.flatten()[e]
        
        ax.coastlines()
        ax.set_extent(amvbbox)
        ax.set_title("%s ens %i" % (dataset_long[d],e+1))

        plotvar = ds_all[d].sst.isel(year=t,ensemble=e)
        pcm     = ax.pcolormesh(lon,lat,plotvar)
        fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.025)

#%% Calculate and visualize the AMV pattern

nasstpats_all = []
amvpats_all   = []

for d in range(ndata):

    indata   = ds_all_nomask[d].sst.values
    nens,ntime,nlat,nlon = indata.shape
    indata   = indata.transpose(0,3,2,1) # [ens x lon x lat x time]
    inidx    = amvids[d].sst.values
    inidx_lp = amvids_lp[d]
    
    amvpats = np.zeros((nlon,nlat,nens))
    nasstpats = amvpats.copy()
    for e in range(nens):
        
        nasstpats[:,:,e] = proc.regress2ts(indata[e,...],inidx[e,:]/inidx[e,:].std())
        amvpats[:,:,e]   = proc.regress2ts(indata[e,...],inidx_lp[:,e]/inidx_lp[:,e].std())
        
    amvpats_all.append(amvpats.transpose(2,1,0)) # [ens x lat x lon]
    nasstpats_all.append(nasstpats.transpose(2,1,0))

#%% Visualize the AMV Patterns

ylabelnames = ("NASST","AMV")

cints   = np.arange(-1,1.1,0.1)
fig,axs = plt.subplots(2,ndata,figsize=(14,5.5),subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True)
for d in range(ndata):
    
    for ii in range(2):
        ax = axs[ii,d]
        ax.coastlines()
        ax.set_extent(amvbbox)
        
        if ii == 0:   # Plot the NASST Pattern
            plotpat = nasstpats_all[d].mean(0)
            ax.set_title("%s" % (dataset_long[d]))
        elif ii == 1: # Plot the AMV Pattern
            plotpat = amvpats_all[d].mean(0)
            
        cf = ax.contourf(lon,lat,plotpat,levels=cints,cmap="RdBu_r",extend="both")
        cl = ax.contour(lon,lat,plotpat,levels=cints,colors="k",linewidths=0.45)
        ax.clabel(cl,cints[::2],fontsize=8)
        
        if d == 0:
            ax.text(-0.05, 0.55, ylabelnames[ii], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
fig.colorbar(cf,ax=axs.flatten(),orientation='horizontal',fraction=.045)
            
            
savename = "%sAMV_NASST_Patterns_EnsAvg_LENS.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Visualize the distribution of + and - AMV events

# Some visualization toggles
makepie = False # If false, include a barplot instead
addtxt  = True # If true, include count in each class

if makepie:
    fig,axs =  plt.subplots(ndata,2,figsize=(8,12),constrained_layout=True,
                            sharex=True,sharey=False)
else:
    
    fig,axs =  plt.subplots(ndata,1,figsize=(8,12),constrained_layout=True,
                            sharex=True,sharey=False)
    
binedges = np.arange(-1.5,1.6,.1)
for d in range(ndata):
    
    # Make the Bar Plot
    if makepie:
        ax = axs[d,0]
    else:
        ax = axs[d]
    plotdata = amvids[d].sst.values.flatten()
    mu    = np.mean(plotdata)
    stdev = np.std(plotdata)
    
    ax.hist(plotdata,bins=binedges,edgecolor="k",alpha=0.60,color=dataset_colors[d])
    
    ax.axvline([mu]      ,ls="solid",lw=0.7,color="k")
    ax.axvline([mu+stdev],ls="dashed",lw=0.7,color="k")
    ax.axvline([mu-stdev],ls="dashed",lw=0.7,color="k")
    

    cntpos       = np.sum(plotdata > mu+stdev)
    cntneg       = np.sum(plotdata < mu-stdev)
    cntneu       = np.sum( (plotdata < mu+stdev) * (plotdata > mu-stdev) )
    class_counts = [cntpos,cntneu,cntneg]
    
    
    title = "%s (N=%i) \n $\mu=%.2e$, $\sigma=%.2f$" % (dataset_long[d],
                                                        dataset_enssize[d],
                                                        mu,
                                                        stdev)
    
    # Text Labels (too messy, but works for single panel..)
    if addtxt:
        ax.text(0.05,.7,"AMV-\n%i" % (cntneg),transform=ax.transAxes,
                bbox=dict(facecolor='w', alpha=0.2))
        ax.text(0.45,.7,"Neutral\n%i" % (cntneu),transform=ax.transAxes,
                bbox=dict(facecolor='w', alpha=0.2))
        ax.text(0.75,.7,"AMV+\n%i" % (cntpos),transform=ax.transAxes,
                bbox=dict(facecolor='w', alpha=0.2))
        
    # pcts   = np.array([cntneg,cntneu,cntpos])/len(plotdata)
    # ax.text(0.05,.7,"AMV-\n %i \n%.2f" % (cntneg,pcts[0]),transform=ax.transAxes,
    #         bbox=dict(facecolor='w', alpha=0.2))
    # ax.text(0.40,.7,"Neutral\n %i \n%.2f" % (cntneu,pcts[1]),transform=ax.transAxes,
    #         bbox=dict(facecolor='w', alpha=0.2))
    # ax.text(0.75,.7,"AMV+\n %i \n%.2f" % (cntpos,pcts[2]),transform=ax.transAxes,
    #         bbox=dict(facecolor='w', alpha=0.2))
    ax.set_title(title)
    ax.grid(True,ls="dotted")
    
    
    # Make pie plot
    if makepie:
        ax =axs[d,1]
        labels = ["%s\n %.2f" % (classes[i],class_counts[i]/len(plotdata)*100)+"%" for i in range(3)]
        ax.pie(class_counts,colors=class_colors,
                labels=labels,labeldistance=1)
savename = "%sNASST_%s_Histogram_Lens_makepie%i.png" % (figpath,bbox_fn,makepie)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%%

fig,ax = plt.subplots(1,1)
labels = ["%s\n(%.2f)" % (classes[i],class_counts[i]/len(plotdata)*100)+"%" for i in range(3)]
ax.pie(class_counts,colors=class_colors,
       labels=labels,labeldistance=0.4)


#%% Compute Power Spectra, AMV, Etc
