#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualization/Experiment Options for viz_acc_by_exp.

Copy the given section and paste on the upper section of viz_acc_byexp (or load this
                                                                        file)

Possible Experiments (keys):
    CESM1_res_1v3       : CESM1 Resolution Comparison  for SST (1deg vs. 3deg Regridding)
    CMIP5_Lens_30_1950  : TS comparison between select CMIP5 MMLE, 30 ensemble members, years 1950-2005
    
    
Created on Wed Feb 15 12:25:37 2023

@author: gliu
"""
import numpy as np

expdict_master = {}

# -------------
#%% [TEMPLATE]
# -------------


"""
# Copy this template and enter the experiment information

exp1 = {"expdir"        :  , # Directory of the experiment
        "searchstr"     :  , # Search/Glob string used for pulling files
        "expname"       :  , # Name of the experiment (Short)
        "expname_long"  :  , # Long name of the experiment (for labeling on plots)
        "c"             :  , # Color for plotting
        "marker"        :  , # Marker for plotting
        "ls"            :  , # Linestyle for plotting
        
        }
"""

# --------------------------------------------------------
#%% CESM1 Resolution Comparison (1deg vs. 3deg Regridding)
# --------------------------------------------------------
exp0 = {"expdir"        : "FNN4_128_SingleVar"   , # Directory of the experiment
        "searchstr"     :  "*SST*"               , # Search/Glob string used for pulling files
        "expname"       : "CESM1_Original"       , # Name of the experiment (Short)
        "expname_long"  : "CESM1 (1-deg. SST)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "b"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }

exp1 = {"expdir"        : "FNN4_128_ALL_CESM1_Train" , # Directory of the experiment
        "searchstr"     :  "*", # Search/Glob string used for pulling files
        "expname"       : "CESM1_3deg"           , # Name of the experiment (Short)
        "expname_long"  : "CESM1 (3-deg. SST)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "r"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "dashed"               , # Linestyle for plotting
        }

inexps                          = [exp0,exp1] # Put in experiments here...
compname                        = "CESM1_res_1v3"# CHANGE THIS for each new comparison
expdict_master["CESM1_res_1v3"] = inexps

# --------------------------
#%% CMIP5 TS Intercomparison
# --------------------------

exp0 = {"expdir"        : "LENS_30_1950/FNN4_128_ALL_canesm2_lens_Train/"   , # Directory of the experiment
        "searchstr"     :  "*"               , # Search/Glob string used for pulling files
        "expname"       : "canesm2_lens"       , # Name of the experiment (Short)
        "expname_long"  : "CCCma-CanESM2"   , # Long name of the experiment (for labeling on plots)
        "c"             : "r"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }

exp1 = {"expdir"        : "LENS_30_1950/FNN4_128_ALL_csiro_mk36_lens_Train/"   , # Directory of the experiment
        "searchstr"     :  "*"               , # Search/Glob string used for pulling files
        "expname"       : "csiro_mk36_lens"       , # Name of the experiment (Short)
        "expname_long"  : "CSIRO-MK3.6"   , # Long name of the experiment (for labeling on plots)
        "c"             : "b"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }

exp2 = {"expdir"        : "LENS_30_1950/FNN4_128_ALL_gfdl_esm2m_lens_Train/"   , # Directory of the experiment
        "searchstr"     :  "*"               , # Search/Glob string used for pulling files
        "expname"       : "gfdl_esm2m_lens"       , # Name of the experiment (Short)
        "expname_long"  : "GFDL-ESM2M"   , # Long name of the experiment (for labeling on plots)
        "c"             : "magenta"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }

exp3 = {"expdir"        : "LENS_30_1950/FNN4_128_ALL_mpi_lens_Train/"   , # Directory of the experiment
        "searchstr"     :  "*"               , # Search/Glob string used for pulling files
        "expname"       : "mpi_lens"       , # Name of the experiment (Short)
        "expname_long"  : "MPI-ESM-LR"   , # Long name of the experiment (for labeling on plots)
        "c"             : "gold"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }

exp4 = {"expdir"        : "LENS_30_1950/FNN4_128_ALL_CESM1_Train/"   , # Directory of the experiment
        "searchstr"     :  "*"               , # Search/Glob string used for pulling files
        "expname"       : "CESM1"       , # Name of the experiment (Short)
        "expname_long"  : "NCAR-CESM1"   , # Long name of the experiment (for labeling on plots)
        "c"             : "limegreen"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }

inexps   = [exp0,exp1,exp2,exp3,exp4] # Put in experiments here...
compname = "CMIP5_Lens_30_1950"# CHANGE THIS for each new comparison
quartile = True
expdict_master["CMIP5_Lens_30_1950"] = inexps

# --------------------------
#%% CESM1 v CESM2 (Tercile Thresholding vs. Stdev. Thresholding, with Limit Time Period to 1920-2005)
# --------------------------

varname = "ssh"

# Limited CESM2 Traininng Period to 1920-2005
exp0 = {"expdir"        : "CMIP6_LENS/models/FNN4_128_SingleVar_CESM2_Train_Quartile", # Directory of the experiment
        "searchstr"     :  "*%s*"  % varname        , # Search/Glob string used for pulling files
        "expname"       : "CESM2_Tercile"          , # Name of the experiment (Short)
        "expname_long"  : "CESM2 (Tercile)"          , # Long name of the experiment (for labeling on plots)
        "c"             : "r"                       , # Color for plotting
        "marker"        : "o"                       , # Marker for plotting
        "ls"            : "dashed"                    , # Linestyle for plotting
        }

exp1 = {"expdir"        : "CMIP6_LENS/models/Limit_1920to2005/FNN4_128_SingleVar_CESM2_Train", # Directory of the experiment
        "searchstr"     :  "*%s*"  % varname        , # Search/Glob string used for pulling files
        "expname"       : "CESM2_Exact"          , # Name of the experiment (Short)
        "expname_long"  : "CESM2 (Stdev)"          , # Long name of the experiment (for labeling on plots)
        "c"             : "darkred"                       , # Color for plotting
        "marker"        : "o"                       , # Marker for plotting
        "ls"            : "solid"                   , # Linestyle for plotting
        }


exp2 = {"expdir"        : "FNN4_128_SingleVar_Quartile", # Directory of the experiment
        "searchstr"     :  "*%s*"  % varname.upper()        , # Search/Glob string used for pulling files
        "expname"       : "CESM1_Tercile"          , # Name of the experiment (Short)
        "expname_long"  : "CESM1 (Tercile)"          , # Long name of the experiment (for labeling on plots)
        "c"             : "cornflowerblue"               , # Color for plotting
        "marker"        : "x"                       , # Marker for plotting
        "ls"            : "dashed"                    , # Linestyle for plotting
        }

exp3 = {"expdir"        : "FNN4_128_SingleVar_Exact_Thres", # Directory of the experiment
        "searchstr"     :  "*%s*"  % varname.upper()        , # Search/Glob string used for pulling files
        "expname"       : "CESM1_Exact_Thres"          , # Name of the experiment (Short)
        "expname_long"  : "CESM1 (Stdev)"          , # Long name of the experiment (for labeling on plots)
        "c"             : "darkblue"                   , # Color for plotting
        "marker"        : "x"                          , # Marker for plotting
        "ls"            : "solid"                     , # Linestyle for plotting
        }

inexps    = [exp0,exp1,exp2,exp3]                        # Put in experiments here...
compname  = "CESM1_2_Comparison_%s" % varname # CHANGE THIS for each new comparison
quartile  = [True,True,False]
leads     = np.arange(0,25,3)

# ------------------------------------------
#%% SingleVar Comparisons (Version Agnostic)
# ------------------------------------------
# Train for all predictors of a given experiment

"""
# CESM2, Quartile Thresholds
expname     = "CESM2_Quant"
expdir      = "CMIP6_LENS/models/FNN4_128_SingleVar_CESM2_Train_Quartile"
cmipver     = 6
quartile    = True
leads       = np.arange(0,25,3)
var_include = ["ssh","sst","sss"]
"""

"""
# CESM1, Quartile Thresholds
expname     = "CESM1_Quant"
expdir      = "FNN4_128_SingleVar_Quartile"
cmipver     = 5
quartile    = True
leads       = np.arange(0,25,3)
var_include  = ["SSH","SST","SSS"]
"""

"""
# CESM1, Exact Thresholds
expname     = "CESM1"
expdir      = "FNN4_128_SingleVar"
cmipver     = 5
quartile    = False
leads       = np.arange(0,25,3)
var_include  = ["SSH","SST","SSS"]
"""


"""
# MIROC6, Exact Thresholds, Limit 1920 - 2005
expname     = "MIROC6_1920to2005"
expdir      = "CMIP6_LENS/models/Limit_1920to2005/FNN4_128_SingleVar_MIROC6_Train"
cmipver     = 6
quartile    = True
leads       = np.arange(0,26,3)
var_include = ["ssh","sst","sss"]
"""

"""
# MIROC6, Exact Thresholds, Full Range
expname     = "MIROC6"
expdir      = "CMIP6_LENS/models/FNN4_128_SingleVar_MIROC6_Train"
cmipver     = 6
quartile    = True
leads       = np.arange(0,26,1)
var_include = ["ssh","sst","sss"]
"""

# -----------------------
#%% CMIP6 Intercomparison
# -----------------------

varname ="ssh"
skipmodels = ["MPI-ESM1-2-LR",]

inexps = []
for d,dataset_name in enumerate(cmip6_names):
    if dataset_name in skipmodels : # Skip this model
        continue
    exp = {"expdir"        : "CMIP6_LENS/models/FNN4_128_SingleVar_%s_Train" % dataset_name , # Directory of the experiment
            "searchstr"     :  "*%s*" % varname             , # Search/Glob string used for pulling files
            "expname"       : dataset_name                  , # Name of the experiment (Short)
            "expname_long"  : dataset_name                  , # Long name of the experiment (for labeling on plots)
            "c"             : cmip6_colors[d]               , # Color for plotting
            "marker"        : cmip6_markers[d]              , # Marker for plotting
            "ls"            : "solid"                       , # Linestyle for plotting
        }
    inexps.append(exp)

compname = "%s_CMIP6_Intercomparison" % varname# CHANGE THIS for each new comparison
quartile = True
leads    = np.arange(0,26,1)

# ----------------------------
#%% CMIP6 SingleVar Comparison
# ----------------------------

dataset_name = "ACCESS-ESM1-5"

"""
"CESM2"
"IPSL-CM6A-LR"
"CanESM5"
"MIROC6"
"ACCESS-ESM1-5"
"""

exp0 = {"expdir"        : "CMIP6_LENS/models/FNN4_128_SingleVar_%s_Train" % dataset_name , # Directory of the experiment
        "searchstr"     :  "*sst*"               , # Search/Glob string used for pulling files
        "expname"       : "SST"       , # Name of the experiment (Short)
        "expname_long"  : "SST"   , # Long name of the experiment (for labeling on plots)
        "c"             : "r"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }

exp1 = {"expdir"        : "CMIP6_LENS/models/FNN4_128_SingleVar_%s_Train" % dataset_name , # Directory of the experiment
        "searchstr"     :  "*sss*"               , # Search/Glob string used for pulling files
        "expname"       : "SSS"       , # Name of the experiment (Short)
        "expname_long"  : "SSS"   , # Long name of the experiment (for labeling on plots)
        "c"             : "limegreen"                    , # Color for plotting
        "marker"        : "x"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }

exp2 = {"expdir"        : "CMIP6_LENS/models/FNN4_128_SingleVar_%s_Train" % dataset_name , # Directory of the experiment
        "searchstr"     :  "*ssh*"               , # Search/Glob string used for pulling files
        "expname"       : "SSH"       , # Name of the experiment (Short)
        "expname_long"  : "SSH"   , # Long name of the experiment (for labeling on plots)
        "c"             : "darkblue"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }

inexps   = [exp0,exp1,exp2,] # Put in experiments here...
compname = "%s_SingleVar" % dataset_name# CHANGE THIS for each new comparison
quartile = True


# --------------------------------------------------------
#%% NN Script Rewrite Comparison
# --------------------------------------------------------
exp0 = {"expdir"        : "FNN4_128_SingleVar"   , # Directory of the experiment
        "searchstr"     :  "*SST*"               , # Search/Glob string used for pulling files
        "expname"       : "SST_Original"       , # Name of the experiment (Short)
        "expname_long"  : "SST (Original Script)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "r"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }

exp1 = {"expdir"        : "FNN4_128_Singlevar_Rewrite" , # Directory of the experiment
        "searchstr"     :  "*SST*", # Search/Glob string used for pulling files
        "expname"       : "SST_Rewrite"           , # Name of the experiment (Short)
        "expname_long"  : "SST (Rewrite)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "r"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "dashed"               , # Linestyle for plotting
        }

exp2 = {"expdir"        : "FNN4_128_SingleVar"   , # Directory of the experiment
        "searchstr"     :  "*SSH*"               , # Search/Glob string used for pulling files
        "expname"       : "SSH_Original"       , # Name of the experiment (Short)
        "expname_long"  : "SSH (Original Script)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "b"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }

exp3 = {"expdir"        : "FNN4_128_Singlevar_Rewrite" , # Directory of the experiment
        "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
        "expname"       : "SSH_Rewrite"           , # Name of the experiment (Short)
        "expname_long"  : "SSH (Rewrite)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "b"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "dashed"               , # Linestyle for plotting
        }


inexps                          = [exp0,exp1,exp2,exp3] # Put in experiments here...
compname                        = "NN_Script_Rewrite"# CHANGE THIS for each new comparison
#expdict_master["NN_Script_Rewrite"] = inexps

# --------------------------------------------------------
#%% Cross Validation Analysis
# --------------------------------------------------------

nfolds = 4
inexps = []

vcolors = ["r","b"]
markers = ["d","o","x","+"]
lss     = ["dashed","solid","dotted","dashdot"]

for v,vname in enumerate(['SST','SSH']):
    for k in range(nfolds):
        exp = {"expdir"         : "FNN4_128_Singlevar_CV"      , # Directory of the experiment
                "searchstr"     :  "*%s*kfold%02i*" % (vname,k), # Search/Glob string used for pulling files
                "expname"       : "%s_fold%02i" % (vname,k)    , # Name of the experiment (Short)
                "expname_long"  : "%s (fold=%02i)" % (vname,k)   , # Long name of the experiment (for labeling on plots)
                "c"             : vcolors[v]                    , # Color for plotting
                "marker"        : markers[k]                   , # Marker for plotting
                "ls"            : lss[k]               , # Linestyle for plotting
                }
        inexps.append(exp)
compname                        = "FNN4_128_CV_SSH_SST"# CHANGE THIS for each new comparison

leads = np.arange(0,26,3)

# ---------------------------------------------------
# %% Updated Cross Validation with consistent samples
# ---------------------------------------------------
nfolds = 4
inexps = []

vcolors = ["r","b"]
markers = ["d","o","x","+"]
lss     = ["dashed","solid","dotted","dashdot"]

for v,vname in enumerate(['SST','SSH']):
    for k in range(nfolds):
        exp = {"expdir"         : "FNN4_128_Singlevar_CV_consistent"      , # Directory of the experiment
                "searchstr"     :  "*%s*kfold%02i*" % (vname,k), # Search/Glob string used for pulling files
                "expname"       : "%s_fold%02i" % (vname,k)    , # Name of the experiment (Short)
                "expname_long"  : "%s (fold=%02i)" % (vname,k)   , # Long name of the experiment (for labeling on plots)
                "c"             : vcolors[v]                    , # Color for plotting
                "marker"        : markers[k]                   , # Marker for plotting
                "ls"            : lss[k]               , # Linestyle for plotting
                }
        inexps.append(exp)
compname                        = "FNN4_128_CV_consistent_SSH_SST"# CHANGE THIS for each new comparison

leads = np.arange(0,26,3)

# ----------------------------
#%% CESM1 SingleVar Comparison
# ----------------------------

expdir   = "FNN4_128_SingleVar_Rerun100"
allpred       = ("SST","SSS","PSL","SSH")
apcolors      = ("r","limegreen","pink","darkblue")

"""
"CESM2"
"IPSL-CM6A-LR"
"CanESM5"
"MIROC6"
"ACCESS-ESM1-5"
"""
inexps = []
for p,pred in enumerate(allpred):
    
    exp0 = {"expdir"        : expdir, # Directory of the experiment
            "searchstr"     :  "*%s*"  % pred              , # Search/Glob string used for pulling files
            "expname"       : pred       , # Name of the experiment (Short)
            "expname_long"  : pred   , # Long name of the experiment (for labeling on plots)
            "c"             : apcolors[p]                    , # Color for plotting
            "marker"        : "o"                    , # Marker for plotting
            "ls"            : "solid"               , # Linestyle for plotting
            }
    
    inexps.append(exp0)

quartile = False
leads    = np.arange(0,26,1)
detrend  = False


# --------------------------------------------------
# %% Compare particular predictor across experiments for wrtiten version
# --------------------------------------------------


exp2 = {"expdir"        : "FNN4_128_SingleVar"   , # Directory of the experiment
        "searchstr"     :  "*SSH*"               , # Search/Glob string used for pulling files
        "expname"       : "SSH_Original"       , # Name of the experiment (Short)
        "expname_long"  : "SSH (Original Script)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "b"                    , # Color for plotting
        "marker"        : "o"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        "no_val"        : True  # Whether or not there is a validation dataset
        }

exp3 = {"expdir"        : "FNN4_128_SingleVar_Rewrite" , # Directory of the experiment
        "searchstr"     :  "*SSH*",                      # Search/Glob string used for pulling files
        "expname"       : "SSH_Rewrite"           ,      # Name of the experiment (Short)
        "expname_long"  : "SSH (Rewrite)"   ,            # Long name of the experiment (for labeling on plots)
        "c"             : "orange"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "dashed"               , # Linestyle for plotting
        "no_val"        : True  # Whether or not there is a validation dataset
        }

exp4 = {"expdir"        : "FNN4_128_SingleVar_debug1_shuffle_all" , # Directory of the experiment
        "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
        "expname"       : "SSH_Rewrite_newest"           , # Name of the experiment (Short)
        "expname_long"  : "SSH (Rewrite Newest)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "r"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "dashed"               , # Linestyle for plotting
        "no_val"        : False  # Whether or not there is a validation dataset
        }

exp5 = {"expdir"        : "FNN4_128_SingleVar_debug1_shuffle_all_20ep_3ES_32bs" , # Directory of the experiment
        "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
        "expname"       : "SSH_Rewrite_newest_redEp"           , # Name of the experiment (Short)
        "expname_long"  : "SSH (Rewrite Newest, Reduce Epochs)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "magenta"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "dashed"               , # Linestyle for plotting
        "no_val"        : False  # Whether or not there is a validation dataset
        }


exp6 = {"expdir"        : "FNN4_128_SingleVar_debug1_shuffle_all_20ep_3ES_16bs" , # Directory of the experiment
        "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
        "expname"       : "SSH_Rewrite_newest_redEp_redBS"           , # Name of the experiment (Short)
        "expname_long"  : "SSH (Rewrite Newest, Reduce Epochs and Batch Size)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "limegreen"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "dashed"               , # Linestyle for plotting
        "no_val"        : False  # Whether or not there is a validation dataset
        }


exp7 = {"expdir"        : "FNN4_128_SingleVar_debug1_shuffle_all_no_val" , # Directory of the experiment
        "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
        "expname"       : "SSH_Rewrite_newest_no_val"           , # Name of the experiment (Short)
        "expname_long"  : "SSH (Rewrite Newest, No Validation)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "cyan"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        "no_val"        : False  # Whether or not there is a validation dataset
        }

exp8 = {"expdir"        : "FNN4_128_SingleVar_debug1_shuffle_all_no_val_8020" , # Directory of the experiment
        "searchstr"     :  "*SSH*", # Search/Glob string used for pulling files
        "expname"       : "SSH_Rewrite_newest_no_val_8020"           , # Name of the experiment (Short)
        "expname_long"  : "SSH (Rewrite Newest, No Validation 80-20)"   , # Long name of the experiment (for labeling on plots)
        "c"             : "yellow"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        "no_val"        : False  # Whether or not there is a validation dataset
        }

inexps   = (exp2,exp3,exp4,exp5,exp6,exp7,exp8)
compname = "Rewrite"
quartile = False
leads    = np.arange(0,26,3)
detrend  = False
no_vals  = [d['no_val'] for d in inexps]

# --------------------------------------------------------
#%% CESM PIC vs HTR
# --------------------------------------------------------


inexps   = []
vcolors  = ["r","b"]
markers  = ["d","o","d","o"]
lss      = ["dashed","solid","dashed","solid"]
exps     = ["FNN4_128_detrend","FNN4_128_SingleVar_PIC"]
expnames = ["Historical Detrended","PiControl"]


for v,vname in enumerate(['SST','SSH']):
    for exp in range(2):
        
        
        exp = {"expdir"         : exps[exp]     , # Directory of the experiment
                "searchstr"     :  "*%s*" % (vname), # Search/Glob string used for pulling files
                "expname"       : "%s_%s" % (exps,vname), # Name of the experiment (Short)
                "expname_long"  : "%s (%s)" % (expnames[exp],vname)   , # Long name of the experiment (for labeling on plots)
                "c"             : vcolors[v]                    , # Color for plotting
                "marker"        : markers[exp]                   , # Marker for plotting
                "ls"            : lss[exp]               , # Linestyle for plotting
                "no_val"        : False,  # Whether or not there is a validation dataset
                }
        
        inexps.append(exp)
        
compname                        = "FNN4_128_HTR_v_PiC"# CHANGE THIS for each new comparison
leads                           = np.arange(0,26,3)
quartile                        = False
detrend                         = True
no_vals                         = [False,True,False,True]

