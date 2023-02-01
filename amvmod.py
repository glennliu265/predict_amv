#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amvmod

Module containing functions for predict_amv
Working on updating documentation...

------------------------
  Metrics and Analysis  
------------------------
    Convenience functions for working with the metrics output
    
    --- Loading the metrics file ---
    load_result        : Given a metrics file, load out the results 
    load_metrics_byrun : Load all training runs for a given experiment
    
    --- Organization into an experiment dictionary ---
    make_expdict       : Load experiment metrics/runs into array and make into a dict
    unpack_expdict     : Unpack variables from expdict of a metrics file
    
    --- 
    retrieve_lead      : Get prediction leadtime/index from shuffled indices
    
@author: gliu
"""

from scipy.signal import butter,filtfilt
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from torch import nn
#%% Metrics and Analysis ----

def load_result(fn,debug=False):
    """
    Load results for each of the variable names (testacc, etc)
    input: fn (str), Name of the file
    Copied from viz_acc_by_predictor.py on 2023.01.25
    """
    
    ld = np.load(fn,allow_pickle=True)
    vnames = ld.files
    if debug:
        print(vnames)
    output = []
    for v in vnames:
        output.append(ld[v])
    return output,vnames

def load_metrics_byrun(flist,leads,debug=False,runmax=None):
    """
    Given a list of metric files [flist] and leadtimes for each training run,
    Load the output and append.
    Dependencies: load_result()
    """
    flist.sort()
    if runmax is None:
        nruns = len(flist)
    else:
        nruns = runmax
    # Load Result for each model training run
    totalm    = [] # Total Test Accuracy
    classm    = [] # Test Accuracy by Class
    ypredm    = [] # Predicted Class
    ylabsm    = [] # Actual Class
    shuffidsm = [] # Shuffled Indices
    for i in range(nruns): # Load for [nruns] files
        output,vnames = load_result(flist[i],debug=debug)
        # if len(output[4]) > len(leads):
        #     print("Selecting Specific Leads!")
        #     output = [out[leads] for out in output]
        totalm.append(output[4])
        classm.append(output[5])
        ypredm.append(output[6])
        ylabsm.append(output[7])
        shuffidsm.append(output[8])
        print("\tLoaded %s, %s, %s, and %s for run %02i" % (vnames[4],vnames[5],vnames[6],vnames[7],i))
    return totalm,classm,ypredm,ylabsm,shuffidsm,vnames
    
def make_expdict(flists,leads):
    """
    Given a nested list of metric files for the 
    training runs for each experiment, ([experiment][run]),
    Load out the data into arrays and create and experiment dictionary for analysis
    This data can later be unpacked by unpack_expdict
    
    Contents of expdict: 
        totalacc = [] # Accuracy for all classes combined [exp x run x leadtime]
        classacc = [] # Accuracy by class                 [exp x run x leadtime x class]
        ypred    = [] # Predictions                       [exp x run x leadtime x sample]
        ylabs    = [] # Labels                            [exp x run x leadtime x sample]
        shuffids = [] # Indices                           [exp x run x leadtime x sample]
    
    Dependencies: 
        - load_metrics_byrun
        - load_result
    """
    # Check the # of runs
    nruns = [len(f) for f in flists]
    if len(np.unique(nruns)) > 1:
        print("Warning, limiting experiments to %i runs" % np.min(nruns))
    runmax = np.min(nruns)
    
    # Preallocate
    totalacc = [] # Accuracy for all classes combined [exp x run x leadtime]
    classacc = [] # Accuracy by class                 [exp x run x leadtime x class]
    ypred    = [] # Predictions                       [exp x run x leadtime x sample] # Last array (tercile based) is not an even sample size...
    ylabs    = [] # Labels                            [exp x run x leadtime x sample]
    shuffids = [] # Indices                           [exp x run x leadtime x sample]
    for exp in range(len(flists)):
        # Load metrics for a given experiment
        exp_metrics = load_metrics_byrun(flists[exp],leads,runmax=runmax)
        
        # Load out and append variables
        totalm,classm,ypredm,ylabsm,shuffidsm,vnames = exp_metrics
        totalacc.append(totalm)
        classacc.append(classm)
        ypred.append(ypredm)
        ylabs.append(ylabsm)
        shuffids.append(shuffidsm)
        print("Loaded data for experiment %02i!" % (exp+1))
    
    # Add to dictionary
    outputs = [totalacc,classacc,ypred,ylabs,shuffids]
    expdict = {}
    dictkeys = ("totalacc","classacc","ypred","ylabs","shuffids")
    for k,key in enumerate(dictkeys):
        expdict[key] = np.array(outputs[k])
    return expdict

def retrieve_lead(shuffidx,lead,nens,tstep):
    """
    Get prediction leadtime/index from shuffled indices (?)
    Copied from viz_acc_by_predictor.py on 2023.01.25
    """
    orishape = [nens,tstep-lead]
    outidx   = np.unravel_index(shuffidx,orishape)
    return outidx

def unpack_expdict(expdict,dictkeys=None):
    """
    Unpack expdict generated by load_result from the metrics file
    
    Copied from viz_acc_by_predictor.py on 2023.01.25
    """
    if dictkeys is None:
        dictkeys = ("totalacc","classacc","ypred","ylabs","shuffids")
    unpacked = [expdict[key] for key in expdict]
    return unpacked



#%% Unorganized section below

## Processing/Analysis ----
def find_nan(data,dim):
    """
    For a 2D array, remove any point if there is a nan in dimension [dim]
    
    Inputs:
        1) data: 2d array, which will be summed along last dimension
        2) dim: dimension to sum along. 0 or 1
    Outputs:
        1) okdata: data with nan points removed
        2) knan: boolean array with indices of nan points
        
    """
    
    # Sum along select dimension
    if len(data.shape) > 1:
        datasum = np.sum(data,axis=dim)
    else:
        datasum = data.copy()
    
    
    # Find non nan pts
    knan  = np.isnan(datasum)
    okpts = np.invert(knan)
    
    if len(data.shape) > 1:
        if dim == 0:
            okdata = data[:,okpts]
        elif dim == 1:    
            okdata = data[okpts,:]
    else:
        okdata = data[okpts]
        
    return okdata,knan,okpts


def eof_simple(pattern,N_mode,remove_timemean):
    """
    Simple EOF function based on script by Yu-Chiao
    
    
    Inputs:
        1) pattern: Array of Space x Time [MxN], no NaNs
        2) N_mode:  Number of Modes to output
        3) remove_timemean: Set 1 to remove mean along N
    
    Outputs:
        1) eof: EOF patterns   [M x N_mode]
        2) pcs: PC time series [N x N_mode]
        3) varexp: % Variance explained [N_mode]
    
    Dependencies:
        import numpy as np
    
    """
    pattern1 = pattern.copy()
    nt = pattern1.shape[1] # Get time dimension size
    ns = pattern1.shape[0] # Get space dimension size
    
    # Preallocate
    eofs = np.zeros((ns,N_mode))
    pcs  = np.zeros((nt,N_mode))
    varexp = np.zeros((N_mode))
    
    # Remove time mean if option is set
    if remove_timemean == 1:
        pattern1 = pattern1 - pattern1.mean(axis=1)[:,None] # Note, the None adds another dimension and helps with broadcasting
    
    # Compute SVD
    [U, sigma, V] = np.linalg.svd(pattern1, full_matrices=False)
    
    # Compute variance (total?)
    norm_sq_S = (sigma**2).sum()
    
    for II in range(N_mode):
        
        # Calculate explained variance
        varexp[II] = sigma[II]**2/norm_sq_S
        
        # Calculate PCs
        pcs[:,II] = np.squeeze(V[II,:]*np.sqrt(nt-1))
        
        # Calculate EOFs and normalize
        eofs[:,II] = np.squeeze(U[:,II]*sigma[II]/np.sqrt(nt-1))
    return eofs, pcs, varexp

def coarsen_byavg(invar,lat,lon,deg,tol,bboxnew=False,latweight=True,verbose=True):
    """
    Coarsen an input variable to specified resolution [deg]
    by averaging values within a search tolerance for each new grid box.
    To take the area-weighted average, set latweight=True
    
    Dependencies: numpy as np

    Parameters
    ----------
    invar : ARRAY [TIME x LAT x LON]
        Input variable to regrid
    lat : ARRAY [LAT]
        Latitude values of input
    lon : ARRAY [LON]
        Longitude values of input
    deg : INT
        Resolution of the new grid (in degrees)
    tol : TYPE
        Search tolerance (pulls all lat/lon +/- tol)
    
    OPTIONAL ---
    bboxnew : ARRAY or False
        New bounds to regrid in order - [lonW, lonE, latS, latN]
        Set to False to pull directly from first and last coordinates
    latweight : BOOL
        Set to true to apply latitude weighted-average
    verbose : BOOL
        Set to true to print status
    

    Returns
    -------
    outvar : ARRAY [TIME x LAT x LON]
        Regridded variable       
    lat5 : ARRAY [LAT]
        New Latitude values of input
    lon5 : ARRAY [LON]
        New Longitude values of input

    """

    # Make new Arrays
    if not bboxnew:
        lon5 = np.arange(lon[0],lon[-1]+deg,deg)
        lat5 = np.arange(lat[0],lat[-1]+deg,deg)
    else:
        lon5 = np.arange(bboxnew[0],bboxnew[1]+deg,deg)
        lat5 = np.arange(bboxnew[2],bboxnew[3]+deg,deg)
    
    # Check to see if any longitude values are degrees Easy
    if any(lon>180):
        lonflag = True
    
    # Set up latitude weights
    if latweight:
        _,Y = np.meshgrid(lon,lat)
        wgt = np.cos(np.radians(Y)) # [lat x lon]
        invar *= wgt[None,:,:] # Multiply by latitude weight
    
    # Get time dimension and preallocate
    nt = invar.shape[0]
    outvar = np.zeros((nt,len(lat5),len(lon5)))
    
    # Loop and regrid
    i=0
    for o in range(len(lon5)):
        for a in range(len(lat5)):
            lonf = lon5[o]
            latf = lat5[a]
            
            # check longitude
            if lonflag:
                if lonf < 0:
                    lonf+=360
            
            lons = np.where((lon >= lonf-tol) & (lon <= lonf+tol))[0]
            lats = np.where((lat >= latf-tol) & (lat <= latf+tol))[0]
            
            varf = invar[:,lats[:,None],lons[None,:]]
            
            if latweight:
                wgtbox = wgt[lats[:,None],lons[None,:]]
                varf = np.sum(varf/np.sum(wgtbox,(0,1)),(1,2)) # Divide by the total weight for the box
            else:
                varf = varf.mean((1,2))
            outvar[:,a,o] = varf.copy()
            i+= 1
            msg="\rCompleted %i of %i"% (i,len(lon5)*len(lat5))
            print(msg,end="\r",flush=True)
    return outvar,lat5,lon5

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
            
            
            # Set axis for summing/averaging
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

def sel_region(var,lon,lat,bbox,reg_avg=0,reg_sum=0,warn=1):
    """
    
    Select Region
    
    Inputs
        1) var: ARRAY, variable with dimensions [lon x lat x otherdims]
        2) lon: ARRAY, Longitude values
        3) lat: ARRAY, Latitude values
        4) bbox: ARRAY, bounding coordinates [lonW lonE latS latN]
        5) reg_avg: BOOL, set to 1 to return regional average
        6) reg_sum: BOOL, set to 1 to return regional sum
        7) warn: BOOL, set to 1 to print warning text for region selection
    Outputs:
        1) varr: ARRAY: Output variable, cut to region
        2+3), lonr, latr: ARRAYs, new cut lat/lon
    
    
    """    
        
    # Find indices
    klat = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]
    if bbox[0] < bbox[1]:
        klon = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]
    elif bbox[0] > bbox[1]:
        if warn == 1:
            print("Warning, crossing the prime meridian!")
        klon = np.where((lon <= bbox[1]) | (lon >= bbox[0]))[0]
    
    
    lonr = lon[klon]
    latr = lat[klat]
    
    #print("Bounds from %.2f to %.2f Latitude and %.2f to %.2f Longitude" % (latr[0],latr[-1],lonr[0],lonr[-1]))
        
    
    # Index variable
    varr = var[klon[:,None],klat[None,:],...]
    
    if reg_avg==1:
        varr = np.nanmean(varr,(0,1))
        return varr
    elif reg_sum == 1:
        varr = np.nansum(varr,(0,1))
        return varr
    return varr,lonr,latr

def calc_AMV(lon,lat,sst,bbox,order,cutofftime,awgt,lpf=1):
    """
    Calculate AMV Index for detrended/anomalized SST data [LON x LAT x Time]
    given bounding box [bbox]. Applies area weight based on awgt
    
    Parameters
    ----------
    lon : ARRAY [LON]
        Longitude values
    lat : ARRAY [LAT]
        Latitude Values
    sst : ARRAY [LON x LAT x TIME]
        Sea Surface Temperature
    bbox : ARRAY [LonW,LonE,LonS,LonN]
        Bounding Box for Area Average
    order : INT
        Butterworth Filter Order
    cutofftime : INT
        Filter Cutoff, expressed in same timesteps as input data
    awgt : INT
        0 = No weight, 1 = cos(lat), 2 = sqrt(cos(lat))
        
    Returns
    -------
    amv: ARRAY [TIME]
        AMV Index (Not Standardized)
    
    aa_sst: ARRAY [TIME]
        Area Averaged SST

    # Dependencies
    functions
        area_avg
    modules
        numpy as np
        from scipy.signal import butter,filtfilt
    """
    
    # Take the weighted area average
    aa_sst = area_avg(sst,bbox,lon,lat,awgt)

    # Design Butterworth Lowpass Filter
    filtfreq = len(aa_sst)/cutofftime
    nyquist  = len(aa_sst)/2
    cutoff = filtfreq/nyquist
    b,a    = butter(order,cutoff,btype="lowpass")
    
    # Compute AMV Index
    amv = filtfilt(b,a,aa_sst)

    return amv,aa_sst


def detrend_poly(x,y,deg):
    """
    Matrix for of polynomial detrend
    # Based on :https://stackoverflow.com/questions/27746297/detrend-flux-time-series-with-non-linear-trend
    
    Inputs:
        1) x --> independent variable
        2) y --> 2D Array of dependent variables
        3) deg --> degree of polynomial to fit
    
    """
    # Transpose to align dimensions for polyfit
    if len(y) != len(x):
        y = y.T
    
    # Get the fit
    fit = np.polyfit(x,y,deg=deg)
    
    # Prepare matrix (x^n, x^n-1 , ... , x^0)
    #inputs = np.array([np.power(x,d) for d in range(len(fit))])
    inputs = np.array([np.power(x,d) for d in reversed(range(len(fit)))])
    # Calculate model
    model = fit.T.dot(inputs)
    # Remove trend
    ydetrend = y - model.T
    return ydetrend,model

def lon360to180(lon360,var):
    """
    Convert Longitude from Degrees East to Degrees West 
    Inputs:
        1. lon360 - array with longitude in degrees east
        2. var    - corresponding variable [lon x lat x time]
    """
    kw = np.where(lon360 >= 180)[0]
    ke = np.where(lon360 < 180)[0]
    lon180 = np.concatenate((lon360[kw]-360,lon360[ke]),0)
    var = np.concatenate((var[kw,...],var[ke,...]),0)
    
    return lon180,var


def area_avg(data,bbox,lon,lat,wgt):
    
    """
    Function to find the area average of [data] within bounding box [bbox], 
    based on wgt type (see inputs)
    
    Inputs:
        1) data: target array [lat x lon x otherdims]
        2) bbox: bounding box [lonW, lonE, latS, latN]
        3) lon:  longitude coordinate
        4) lat:  latitude coodinate
        5) wgt:  number to indicate weight type
                    0 = no weighting
                    1 = cos(lat)
                    2 = sqrt(cos(lat))
    
    Output:
        1) data_aa: Area-weighted array of size [otherdims]
        
    Dependencies:
        numpy as np
    

    """
        
    # Find lat/lon indices 
    kw = np.abs(lon - bbox[0]).argmin()
    ke = np.abs(lon - bbox[1]).argmin()
    ks = np.abs(lat - bbox[2]).argmin()
    kn = np.abs(lat - bbox[3]).argmin()
    
        
    # Select the region
    sel_data = data[kw:ke+1,ks:kn+1,:]
    
    # If wgt == 1, apply area-weighting 
    if wgt != 0:
        
        # Make Meshgrid
        _,yy = np.meshgrid(lon[kw:ke+1],lat[ks:kn+1])
        
        
        # Calculate Area Weights (cosine of latitude)
        if wgt == 1:
            wgta = np.cos(np.radians(yy)).T
        elif wgt == 2:
            wgta = np.sqrt(np.cos(np.radians(yy))).T
        
        # Remove nanpts from weight, ignoring any pt with nan in otherdims
        nansearch = np.sum(sel_data,2) # Sum along otherdims
        wgta[np.isnan(nansearch)] = 0
        
        # Apply area weights
        #data = data * wgtm[None,:,None]
        sel_data  = sel_data * wgta[:,:,None]

    
    # Take average over lon and lat
    if wgt != 0:

        # Sum weights to get total area
        sel_lat  = np.sum(wgta,(0,1))
        
        # Sum weighted values
        data_aa = np.nansum(sel_data/sel_lat,axis=(0,1))
    else:
        # Take explicit average
        data_aa = np.nanmean(sel_data,(0,1))
    
    return data_aa

def regress2ts(var,ts,normalizeall=0,method=1,nanwarn=1):
    """
    regress variable var [lon x lat x time] to timeseries ts [time]
    
    Parameters
    ----------
    var : TYPE
        DESCRIPTION.
    ts : TYPE
        DESCRIPTION.
    normalizeall : TYPE, optional
        DESCRIPTION. The default is 0.
    method : TYPE, optional
        DESCRIPTION. The default is 1.
    nanwarn : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    var_reg : TYPE
        DESCRIPTION.
    
    """

    
    # Anomalize and normalize the data (time series is assumed to have been normalized)
    if normalizeall == 1:
        varmean = np.nanmean(var,2)
        varstd  = np.nanstd(var,2)
        var = (var - varmean[:,:,None]) /varstd[:,:,None]
        
    # Get variable shapes
    londim = var.shape[0]
    latdim = var.shape[1]
    
    # 1st method is matrix multiplication
    if method == 1:
        
        # Combine the spatial dimensions 

        var = np.reshape(var,(londim*latdim,var.shape[2]))
        
        
        # Find Nan Points
        # sumvar = np.sum(var,1)
        
        # # Find indices of nan pts and non-nan (ok) pts
        # nanpts = np.isnan(sumvar)
        # okpts  = np.invert(nanpts)
    
        # # Drop nan pts and reshape again to separate space and time dimensions
        # var_ok = var[okpts,:]
        #var[np.isnan(var)] = 0
        
        
        # Perform regression
        #var_reg = np.matmul(np.ma.anomalies(var,axis=1),np.ma.anomalies(ts,axis=0))/len(ts)
        var_reg,_ = regress_2d(ts,var,nanwarn=nanwarn)
        
        
        # Reshape to match lon x lat dim
        var_reg = np.reshape(var_reg,(londim,latdim))
    
    
    
    
    # 2nd method is looping point by point
    elif method == 2:
        
        
        # Preallocate       
        var_reg = np.zeros((londim,latdim))
        
        # Loop lat and long
        for o in range(londim):
            for a in range(latdim):
                
                # Get time series for that period
                vartime = np.squeeze(var[o,a,:])
                
                # Skip nan points
                if any(np.isnan(vartime)):
                    var_reg[o,a]=np.nan
                    continue
                
                # Perform regression 
                r = np.polyfit(ts,vartime,1)
                #r=stats.linregress(vartime,ts)
                var_reg[o,a] = r[0]
                #var_reg[o,a]=stats.pearsonr(vartime,ts)[0]
    
    return var_reg

## Plotting ----
def plot_AMV(amv,ax=None):
    
    """
    Plot amv time series
    
    Dependencies:
        
    matplotlib.pyplot as plt
    numpy as np
    """
    if ax is None:
        ax = plt.gca()
    
    
    htimefull = np.arange(len(amv))
    
    ax.plot(htimefull,amv,color='k')
    ax.fill_between(htimefull,0,amv,where=amv>0,facecolor='red',interpolate=True,alpha=0.5)
    ax.fill_between(htimefull,0,amv,where=amv<0,facecolor='blue',interpolate=True,alpha=0.5)

    return ax

def plot_AMV_spatial(var,lon,lat,bbox,cmap,cint=[0,],clab=[0,],ax=None,pcolor=0,labels=True,fmt="%.1f",clabelBG=False,fontsize=10):
    fig = plt.gcf()
    
    if ax is None:
        ax = plt.gca()
        ax = plt.axes(projection=ccrs.PlateCarree())
        
    # Add cyclic point to avoid the gap
    var,lon1 = add_cyclic_point(var,coord=lon)
    

    
    # Set  extent
    ax.set_extent(bbox)
    
    # Add filled coastline
    ax.add_feature(cfeature.LAND,color=[0.4,0.4,0.4])
    
    
    if len(cint) == 1:
        # Automaticall set contours to max values
        cmax = np.nanmax(np.abs(var))
        cmax = np.round(cmax,decimals=2)
        cint = np.linspace(cmax*-1,cmax,9)
    
    
    
    if pcolor == 0:

        # Draw contours
        cs = ax.contourf(lon1,lat,var,cint,cmap=cmap)
    
    
    
        # Negative contours
        cln = ax.contour(lon1,lat,var,
                    cint[cint<0],
                    linestyles='dashed',
                    colors='k',
                    linewidths=0.5,
                    transform=ccrs.PlateCarree())
    
        # Positive Contours
        clp = ax.contour(lon1,lat,var,
                    cint[cint>=0],
                    colors='k',
                    linewidths=0.5,
                    transform=ccrs.PlateCarree())    
                          
        if labels is True:
            clabelsn= ax.clabel(cln,colors=None,fmt=fmt,fontsize=fontsize)
            clabelsp= ax.clabel(clp,colors=None,fmt=fmt,fontsize=fontsize)
            
            # if clabelBG is True:
            #     [txt.set_backgroundcolor('white') for txt in clabelsn]
            #     [txt.set_backgroundcolor('white') for txt in clabelsp]
    else:
        
        cs = ax.pcolormesh(lon1,lat,var,vmin = cint[0],vmax=cint[-1],cmap=cmap)
        
                                
                
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.75,color='gray',linestyle=':')

    gl.top_labels = gl.right_labels = False
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')
    gl.xlabel_style={'size':8}
    gl.ylabel_style={'size':8}
    if len(clab) == 1:
        cbar= fig.colorbar(cs,ax=ax,fraction=0.046, pad=0.04,format=fmt)
        cbar.ax.tick_params(labelsize=8)
    else:
        cbar = fig.colorbar(cs,ax=ax,ticks=clab,fraction=0.046, pad=0.04,format=fmt)
        cbar.ax.tick_params(labelsize=8)
    #cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in cint], fontsize=10, weight='bold')
    
    return ax

def deseason_lazy(ds,return_scycle=False):
    """
    Deseason function without reading out the values. Remove the seasonal cycle by subtracting the monthly anomalies
    Input:
        ds : DataArray
            Data to be deseasoned
        return_scycle : BOOL (Optional)
            Set to true to return the seasonal cycle that was removed
    Output:
        data_deseason : DataArray
            Deseasoned data
    """
    data_deseason = ds.groupby('time.month') - ds.groupby('time.month').mean('time')
    
    if return_scycle:
        return data_deseason,ds.groupby('time.month').mean('time')
    return data_deseason

def init_map(bbox,crs=ccrs.PlateCarree(),ax=None):
    """
    Quickly initialize a map for plotting
    """
    # Create Figure/axes
    #fig = plt.gcf() 
    
    #ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    if ax is None:
        ax = plt.gca()
    #ax = plt.axes(projection=ccrs.PlateCarree())
        
    
    ax.set_extent(bbox,crs)
    
    # Add Filled Coastline
    ax.add_feature(cfeature.COASTLINE)
    #ax.add_feature(cfeature.LAND,facecolor='k',zorder=-1)
    
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='gray',linestyle=':')
    gl.top_labels = gl.right_labels = False
    

    
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')
    
    return ax


def make_classes(y,thresholds,exact_value=False,reverse=False,
                 quantiles=False):
    """
    Makes classes based on given thresholds. 

    Parameters
    ----------
    y : ARRAY
        Labels to classify
    thresholds : ARRAY
        1D Array of thresholds to partition the data
    exact_value: BOOL, optional
        Set to True to use the exact value in thresholds (rather than scaling by
                                                          standard deviation)

    Returns
    -------
    y_class : ARRAY [samples,class]
        Classified samples, where the second dimension contains an integer
        representing each threshold

    """
    
    if quantiles is False:
        if ~exact_value: # Scale thresholds by standard deviation
            y_std = np.std(y) # Get standard deviation
            thresholds = np.array(thresholds) * y_std
    else: # Determine Thresholds from quantiles
        thresholds = np.quantile(y,thresholds,axis=0) # Replace Thresholds with quantiles
    
    nthres  = len(thresholds)
    y_class = np.zeros((y.shape[0],1))
    
    if nthres == 1: # For single threshold cases
        thres = thresholds[0]
        y_class[y<=thres] = 0
        y_class[y>thres] = 1
        
        print("Class 0 Threshold is y <= %.2f " % (thres))
        print("Class 0 Threshold is y > %.2f " % (thres))
        return y_class
    
    for t in range(nthres+1):
        if t < nthres:
            thres = thresholds[t]
        else:
            thres = thresholds[-1]
        
        if reverse: # Assign class 0 to largest values
            tassign = nthres-t
        else:
            tassign = t
        
        if t == 0: # First threshold
            y_class[y<=thres] = tassign
            print("Class %i Threshold is y <= %.2f " % (tassign,thres))
        elif t == nthres: # Last threshold
            y_class[y>thres] = tassign
            print("Class %i Threshold is y > %.2f " % (tassign,thres))
        else: # Intermediate values
            thres0 = thresholds[t-1]
            y_class[(y>thres0) * (y<=thres)] = tassign
            print("Class %i Threshold is %.2f < y <= %.2f " % (tassign,thres0,thres))
    if quantiles is True:
        return y_class,thresholds
    return y_class

def build_FNN_simple(inputsize,outsize,nlayers,nunits,activations,dropout=0.5,
                     use_softmax=False):
    """
    Build a Feed-foward neural network with N layers, each with corresponding
    number of units indicated in nunits and activations. 
    
    A dropbout layer is included at the end
    
    inputs:
        inputsize:  INT - size of the input layer
        outputsize: INT  - size of output layer
        nlayers:    INT - number of hidden layers to include 
        nunits:     Tuple of units in each layer
        activations: Tuple of pytorch.nn activations
        --optional--
        dropout: percentage of units to dropout before last layer
        use_softmax : BOOL, True to end with softmax layer
        
    outputs:
        Tuple containing FNN layers
        
    dependencies:
        from pytorch import nn
        
    """
    layers = []
    for n in range(nlayers+1):
        #print(n)
        if n == 0:
            #print("First Layer")
            layers.append(nn.Linear(inputsize,nunits[n]))
            layers.append(activations[n])
            
        elif n == (nlayers):
            #print("Last Layer")
            if use_softmax:
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.Linear(nunits[n-1],outsize))
                layers.append(nn.Softmax(dim=0))
            else:
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.Linear(nunits[n-1],outsize))
            
        else:
            #print("Intermediate")
            layers.append(nn.Linear(nunits[n-1],nunits[n]))
            layers.append(activations[n])
    return layers


def get_barcount(y_in,axis=0):

    # Get Existing Classes
    classes  = np.unique(y_in.flatten())
    nclasses = len(classes)

    # Get count of each class along an axis
    counts = []
    for c in range(nclasses):
        count_tot = (y_in==c).sum(axis) # [time,]
        counts.append(count_tot)
    return counts


def calc_confmat(ypred,ylabel,c,getcounts=True,debug=True):
    """
    Calculate Confusion Matrices
      TP  FP
            
      FN  TN
    
    ypred     : [N x 1]
    ylabel    : [N x 1]
    c         : the class number or label, as found in ypred/ylabel
    getcounts : Set True to return indices,counts,total_counts,accuracy
    debug     : Set True to print debugging messages
    """
    
    nsamples = ypred.shape[0]
    TP       = ((ypred==c) * (ylabel==c))
    FP       = ((ypred==c) * (ylabel!=c))
    TN       = ((ypred!=c) * (ylabel!=c))
    FN       = ((ypred!=c) * (ylabel==c))
    cm       = np.array([TP,FP,FN,TN],dtype='object') # [4,#samples]
    
    if debug:
        TP,FP,FN,TN = cm
        print("Predict 0: %i, Actual 0: %i, TN Count: %i " % ((ypred!=c).sum(),(ylabel!=c).sum(),TN.sum())) # Check True Negative 
        print("Predict 1: %i, Actual 1: %i, TP Count: %i " % ((ypred==c).sum(),(ylabel==c).sum(),TP.sum())) # Check True Positive
        print("Predict 1: %i, Actual 0: %i, FP Count: %i (+ %i = %i total)" % ((ypred==c).sum(),(ylabel!=c).sum(),FP.sum(),TP.sum(),FP.sum()+TP.sum())) # Check False Positive
        print("Predict 0: %i, Actual 1: %i, FN Count: %i (+ %i = %i total)" % ((ypred!=c).sum(),(ylabel==c).sum(),FN.sum(),TN.sum(),FN.sum()+TN.sum())) # Check False Negative
    if getcounts: # Get counts and accuracy
        cm_counts               = np.array([c.sum() for c in cm])#.reshape(2,2)
        #cm                      = cm.reshape(2,2,nsamples) #
        count_pred_total        = np.ones(4) * nsamples #np.ones((2,2)) * nsamples  #np.vstack([np.ones(2)*(ypred==c).sum(),np.ones(2)*(ypred!=c).sum()]) # [totalpos,totalpos,totalneg,totalneg]
        cm_acc                  = cm_counts / nsamples     #count_pred_total
        return cm,cm_counts,count_pred_total,cm_acc
    else:
        return cm.reshape(4,nsamples)#.reshape(2,2,nsamples)
    
def calc_confmat_loop(y_pred,y_class):
    """
    Given predictions and labels, retrieves confusion matrix indices in the
    following order for each class: 
        [True Positive, False Positive, False Negative, True Positive]
    
    Parameters
    ----------
    y_pred : ARRAY [nsamples x 1]
        Predicted Class.
    y_class : ARRAY[nsamples x 1]
        Actual Class.
        
    Returns
    -------
    cm_ids :    ARRAY [Class,Confmat_qudrant,Indices]
        Confusion matrix Boolean indices
    cm_counts : ARRAY [Class,Confmat_qudrant]
        Counts of predicted values for each quadrant
    cm_totals : ARRAY [Class,Confmat_qudrant]
        Total count (for division)
    cm_acc :    ARRAY [Class,Confmat_qudrant]
        Accuracy values
    cm_names :  ARRAY [Class,Confmat_qudrant]
        Names of each confmat quadrant
    """
    nsamples = y_pred.shape[0]
    
    # Preallocate for confusion matrices
    cm_ids     = np.empty((3,4,nsamples),dtype='object')# Confusion matrix Boolean indices [Class,Confmat_quadrant,Indices]
    cm_counts  = np.empty((3,4),dtype='object')         # Counts of predicted values for each [Class,Actual_class,Pred_class]
    cm_totals  = cm_counts.copy() # Total count (for division)
    cm_acc     = cm_counts.copy() # Accuracy values
    cm_names   = ["TP","FP","FN","TN"] # Names of each
    
    for th in range(3):
        # New Script
        confmat,ccounts,tcounts,acc = calc_confmat(y_pred,y_class,th)
        cm_ids[th,:]    = confmat.copy().squeeze()
        cm_counts[th,:] = ccounts.copy()
        cm_totals[th,:] = tcounts.copy()
        cm_acc[th,:]    = acc.copy()
    return cm_ids,cm_counts,cm_totals,cm_acc,cm_names
        

def get_topN(arr,N,bot=False,sort=False,absval=False):
    """
    Copied from proc on 2022.11.01
    Get the indices for the top N values of an array.
    Searches along the last dimension. Option to sort output.
    Set [bot]=True for the bottom 5 values
    
    Parameters
    ----------
    arr : TYPE
        Input array with partition/search dimension as the last axis
    N : INT
        Top or bottom N values to find
    bot : BOOL, optional
        Set to True to find bottom N values. The default is False.
    sort : BOOL, optional
        Set to True to sort output. The default is False.
    absval : BOOL, optional
        Set to True to apply abs. value before sorting. The default is False.
        
    Returns
    -------
    ids : ARRAY
        Indices of found values
    """
    
    if absval:
        arr = np.abs(arr)
    if bot is True:
        ids = np.argpartition(arr,N,axis=-1)[...,:N]
    else:
        ids = np.argpartition(arr,-N,axis=-1)[...,-N:]
         # Parition up to k, and take first k elements
    if sort:
        if bot:
            return ids[np.argsort(arr[ids])] # Least to greatest
        else:
            return ids[np.argsort(-arr[ids])] # Greatest to least
    return ids


# %% ImageNet Scripts copied from NN_test_lead_classification_singlevar.py on 2022.12.05

# def transfer_model(modelname,num_classes,cnndropout=False,unfreeze_all=False
#                    ,nlat=224,nlon=224,nchannels=3):
#     """
#     Load pretrained weights and architectures based on [modelname]
    
#     Parameters
#     ----------
#     modelname : STR
#         Name of model (currently supports 'simplecnn',or any resnet/efficientnet from timms)
#     num_classes : INT
#         Dimensions of output (ex. number of classes)
#     cnndropout : BOOL, optional
#         Include dropout layer in simplecnn. The default is False.
#     unfreeze_all : BOOL, optional
#         Set to True to unfreeze all weights in the model. Otherwise, just
#         the last layer is unfrozen. The default is False.
    
#     Returns
#     -------
#     model : PyTorch Model
#         Returns loaded Pytorch model
#     """
#     if 'resnet' in modelname: # Load ResNet
#         model = timm.create_model(modelname,pretrained=True)
#         if unfreeze_all is False: # Freeze all layers except the last
#             for param in model.parameters():
#                 param.requires_grad = False
#         model.fc = nn.Linear(model.fc.in_features, num_classes) # Set last layer size
        
#     elif modelname == 'simplecnn': # Use Simple CNN from previous testing framework
#         # 2 layer CNN settings
#         nchannels     = [32,64]
#         filtersizes   = [[2,3],[3,3]]
#         filterstrides = [[1,1],[1,1]]
#         poolsizes     = [[2,3],[2,3]]
#         poolstrides   = [[2,3],[2,3]]
#         firstlineardim = calc_layerdims(nlat,nlon,filtersizes,filterstrides,poolsizes,poolstrides,nchannels)
#         if cnndropout: # Include Dropout
#             layers = [
#                     nn.Conv2d(in_channels=channels, out_channels=nchannels[0], kernel_size=filtersizes[0]),
#                     nn.Tanh(),
#                     #nn.ReLU(),
#                     #nn.Sigmoid(),
#                     nn.MaxPool2d(kernel_size=poolsizes[0]),
    
#                     nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1]),
#                     nn.Tanh(),
#                     #nn.ReLU(),
#                     #nn.Sigmoid(),
#                     nn.MaxPool2d(kernel_size=poolsizes[1]),
    
#                     nn.Flatten(),
#                     nn.Linear(in_features=firstlineardim,out_features=64),
#                     nn.Tanh(),
#                     #nn.ReLU(),
#                     #nn.Sigmoid(),
    
#                     nn.Dropout(p=0.5),
#                     nn.Linear(in_features=64,out_features=num_classes)
#                     ]
#         else: # Do not include dropout
#             layers = [
#                     nn.Conv2d(in_channels=channels, out_channels=nchannels[0], kernel_size=filtersizes[0]),
#                     nn.Tanh(),
#                     #nn.ReLU(),
#                     #nn.Sigmoid(),
#                     nn.MaxPool2d(kernel_size=poolsizes[0]),
    
#                     nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1]),
#                     nn.Tanh(),
#                     #nn.ReLU(),
#                     #nn.Sigmoid(),
#                     nn.MaxPool2d(kernel_size=poolsizes[1]),
    
#                     nn.Flatten(),
#                     nn.Linear(in_features=firstlineardim,out_features=64),
#                     nn.Tanh(),
#                     #nn.ReLU(),
#                     #nn.Sigmoid(),

#                     nn.Linear(in_features=64,out_features=num_classes)
#                     ]
#         model = nn.Sequential(*layers) # Set up model
#     else: # Load Efficientnet from Timmm
#         model = timm.create_model(modelname,pretrained=True)
#         if unfreeze_all is False: # Freeze all layers except the last
#             for param in model.parameters():
#                 param.requires_grad = False
#         model.classifier=nn.Linear(model.classifier.in_features,num_classes)
#     return model
    

def build_simplecnn(num_classes,cnndropout=False,unfreeze_all=False
                    ,nlat=224,nlon=224,num_inchannels=3):
    
    # 2 layer CNN settings
    nchannels     = [32,64]
    filtersizes   = [[2,3],[3,3]]
    filterstrides = [[1,1],[1,1]]
    poolsizes     = [[2,3],[2,3]]
    poolstrides   = [[2,3],[2,3]]
    firstlineardim = calc_layerdims(nlat,nlon,filtersizes,filterstrides,poolsizes,poolstrides,nchannels)
    if cnndropout: # Include Dropout
        layers = [
                nn.Conv2d(in_channels=num_inchannels, out_channels=nchannels[0], kernel_size=filtersizes[0]),
                #nn.Tanh(),
                nn.ReLU(),
                #nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=poolsizes[0]),

                nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1]),
                #nn.Tanh(),
                nn.ReLU(),
                #nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=poolsizes[1]),

                nn.Flatten(),
                nn.Linear(in_features=firstlineardim,out_features=64),
                #nn.Tanh(),
                nn.ReLU(),
                #nn.Sigmoid(),

                nn.Dropout(p=0.5),
                nn.Linear(in_features=64,out_features=num_classes)
                ]
    else: # Do not include dropout
        layers = [
                nn.Conv2d(in_channels=num_inchannels, out_channels=nchannels[0], kernel_size=filtersizes[0]),
                #nn.Tanh(),
                nn.ReLU(),
                #nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=poolsizes[0]),

                nn.Conv2d(in_channels=nchannels[0], out_channels=nchannels[1], kernel_size=filtersizes[1]),
                #nn.Tanh(),
                nn.ReLU(),
                #nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=poolsizes[1]),

                nn.Flatten(),
                nn.Linear(in_features=firstlineardim,out_features=64),
                #nn.Tanh(),
                nn.ReLU(),
                #nn.Sigmoid(),

                nn.Linear(in_features=64,out_features=num_classes)
                ]
    model = nn.Sequential(*layers) # Set up model
    return model

def calc_layerdims(nx,ny,filtersizes,filterstrides,poolsizes,poolstrides,nchannels):
    """
    For a series of N convolutional layers, calculate the size of the first fully-connected
    layer

    Inputs:
        nx:           x dimensions of input
        ny:           y dimensions of input
        filtersize:   [ARRAY,length N] sizes of the filter in each layer [(x1,y1),[x2,y2]]
        poolsize:     [ARRAY,length N] sizes of the maxpooling kernel in each layer
        nchannels:    [ARRAY,] number of out_channels in each layer
    output:
        flattensize:  flattened dimensions of layer for input into FC layer

    """
    N = len(filtersizes)
    xsizes = [nx]
    ysizes = [ny]
    fcsizes  = []
    for i in range(N):
        xsizes.append(np.floor((xsizes[i]-filtersizes[i][0])/filterstrides[i][0])+1)
        ysizes.append(np.floor((ysizes[i]-filtersizes[i][1])/filterstrides[i][1])+1)

        xsizes[i+1] = np.floor((xsizes[i+1] - poolsizes[i][0])/poolstrides[i][0]+1)
        ysizes[i+1] = np.floor((ysizes[i+1] - poolsizes[i][1])/poolstrides[i][1]+1)

        fcsizes.append(np.floor(xsizes[i+1]*ysizes[i+1]*nchannels[i]))
    return int(fcsizes[-1])

#%% Convenience Functions

def prep_traintest_classification(data,target,lead,thresholds,percent_train,
                                  ens=None,tstep=None,
                                  quantile=False,return_ic=False):
    """
    

    Parameters
    ----------
    data : ARRAY [variable x ens x yr x lat x lon ]
        Network Inputs
    target : ARRAY [ens x yr]
        Network Outputs
    lead : INT
        Leadtime (in years)
    thresholds : List
        List of stdev thresholds. See make_classes()
    percent_train : FLOAT
        Percentage of data to use for training. Rest for validation.
    ens : INT, optional
        # of Ens to include. The default is None (all of them).
    tstep : INT, optional
        # of Years to include. The default is None (all of them).
    quantile : BOOL, optional
        Use quantiles rather than stdev based thresholds. Default is False.
    return_ic : BOOL, optional
        Return the starting class. Quantile thresholds not supported

    Returns
    -------
    None.

    """
    # Get dimensions
    if ens is None:
        ens = data.shape[1]
    if tstep is None:
        tstep = data.shape[2]
    nchannels,_,_,nlat,nlon = data.shape
    
    # Apply Lead
    y                            = target[:ens,lead:].reshape(ens*(tstep-lead),1)
    X                            = (data[:,:ens,:tstep-lead,:,:]).reshape(nchannels,ens*(tstep-lead),nlat,nlon).transpose(1,0,2,3)
    nsamples,_,_,_ = X.shape
    
    # Make the labels
    y_class = make_classes(y,thresholds,reverse=True,quantiles=quantile)
    if quantile == True:
        thresholds = y_class[1].T[0]
        y_class    = y_class[0]
    if (nsamples is None) or (quantile is True):
        nthres = len(thresholds) + 1
        threscount = np.zeros(nthres)
        for t in range(nthres):
            threscount[t] = len(np.where(y_class==t)[0])
        nsamples = int(np.min(threscount))
    y_val  = y.copy()
    
    # Compute class of initial state if option is set
    if return_ic:
        y_start    = target[:ens,:tstep-lead].reshape(ens*(tstep-lead),1)
        y_class_ic = make_classes(y_start,thresholds,reverse=True,quantiles=quantile)
        
        y_train_ic = y_class_ic[0:int(np.floor(percent_train*nsamples)),:]
        y_val_ic   = y_class_ic[int(np.floor(percent_train*nsamples)):,:]
        
    
    # Test/Train Split
    X_train = X[0:int(np.floor(percent_train*nsamples)),...]
    X_val   = X[int(np.floor(percent_train*nsamples)):,...]
    y_train = y_class[0:int(np.floor(percent_train*nsamples)),:]
    y_val   = y_class[int(np.floor(percent_train*nsamples)):,:]
        
    if return_ic:
        return X_train,X_val,y_train,y_val,y_train_ic,y_val_ic
    return X_train,X_val,y_train,y_val


def get_ensyr(id_val,lead,ens=40,tstep=86,percent_train=0.8,get_train=False):
    # Get ensemble and year of reshaped valdation indices (or training if get_train=True)
    # Assume target is of the order [ens  x time] (default is (40,86))
    # Assumes default 80% used for training
    id_ensyr = np.zeros((ens,tstep),dtype='object')
    for e in range(ens):
        for y in range(tstep):
            id_ensyr[e,y] = (e,y)
    reshape_id = id_ensyr[:ens,lead:].reshape(ens*(tstep-lead),1)
    nsamples = reshape_id.shape[0]
    if get_train:
        val_id = reshape_id[0:int(np.floor(percent_train*nsamples)),:]
    else:
        val_id = reshape_id[int(np.floor(percent_train*nsamples)):,:]
    return val_id[id_val]

#def data_loader(varname=None,datpath=None):
## Added LRP Functions

        
