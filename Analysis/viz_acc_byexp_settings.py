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
#%% CESM1 v CESM2 (Limit Time Period to 1920-2005) Effect + Quartile
# --------------------------

varname = "ssh"

# Limited CESM2 Traininng Period to 1920-2005
exp0 = {"expdir"        : "CMIP6_LENS/models/FNN4_128_SingleVar_CESM2_Train_Quartile", # Directory of the experiment
        "searchstr"     :  "*%s*"  % varname        , # Search/Glob string used for pulling files
        "expname"       : "CESM2_Quartile"          , # Name of the experiment (Short)
        "expname_long"  : "CESM2 Quartile"          , # Long name of the experiment (for labeling on plots)
        "c"             : "r"                       , # Color for plotting
        "marker"        : "o"                       , # Marker for plotting
        "ls"            : "solid"                   , # Linestyle for plotting
        }

exp1 = {"expdir"        : "CMIP6_LENS/models/Limit_1920to2005/FNN4_128_SingleVar_CESM2_Train", # Directory of the experiment
        "searchstr"     :  "*%s*"  % varname        , # Search/Glob string used for pulling files
        "expname"       : "CESM2_Exact"          , # Name of the experiment (Short)
        "expname_long"  : "CESM2_Exact"          , # Long name of the experiment (for labeling on plots)
        "c"             : "darkred"                       , # Color for plotting
        "marker"        : "o"                       , # Marker for plotting
        "ls"            : "solid"                   , # Linestyle for plotting
        }


exp2 = {"expdir"        : "FNN4_128_SingleVar_Quartile", # Directory of the experiment
        "searchstr"     :  "*%s*"  % varname.upper()        , # Search/Glob string used for pulling files
        "expname"       : "CESM1_Quartile"          , # Name of the experiment (Short)
        "expname_long"  : "CESM1 Quartile"          , # Long name of the experiment (for labeling on plots)
        "c"             : "cornflowerblue"               , # Color for plotting
        "marker"        : "x"                       , # Marker for plotting
        "ls"            : "solid"                   , # Linestyle for plotting
        }

exp3 = {"expdir"        : "FNN4_128_SingleVar_Exact_Thres", # Directory of the experiment
        "searchstr"     :  "*%s*"  % varname.upper()        , # Search/Glob string used for pulling files
        "expname"       : "CESM1_Exact_Thres"          , # Name of the experiment (Short)
        "expname_long"  : "CESM1_Exact_Thres"          , # Long name of the experiment (for labeling on plots)
        "c"             : "darkblue"                   , # Color for plotting
        "marker"        : "x"                          , # Marker for plotting
        "ls"            : "dashed"                     , # Linestyle for plotting
        }

exp4 = {"expdir"        : "FNN4_128_SingleVar", # Directory of the experiment
        "searchstr"     :  "*%s*"  % varname.upper()        , # Search/Glob string used for pulling files
        "expname"       : "CESM1"          , # Name of the experiment (Short)
        "expname_long"  : "CESM1"          , # Long name of the experiment (for labeling on plots)
        "c"             : "darkblue"                    , # Color for plotting
        "marker"        : "d"                    , # Marker for plotting
        "ls"            : "solid"               , # Linestyle for plotting
        }


# exp2 = {"expdir"        : "CMIP6_LENS/models/FNN4_128_SingleVar_%s_Train" % dataset_name , # Directory of the experiment
#         "searchstr"     :  "*ssh*"               , # Search/Glob string used for pulling files
#         "expname"       : "SSH"       , # Name of the experiment (Short)
#         "expname_long"  : "SSH"   , # Long name of the experiment (for labeling on plots)
#         "c"             : "darkblue"                    , # Color for plotting
#         "marker"        : "d"                    , # Marker for plotting
#         "ls"            : "solid"               , # Linestyle for plotting
#         }


inexps    = [exp0,exp1,exp2,exp3,exp4]                        # Put in experiments here...
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

