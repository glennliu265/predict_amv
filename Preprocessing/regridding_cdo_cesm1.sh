#! /bin/zsh

# Regrid CESM1 data from POP (ocean) to CAM (atmospheric) grid.


# Set the scenario and variable names
datasets=("PiControl")
vars=("SSH" )
var ="SSH"

# Data Directory (note backslash included below for both paths)
dd=/Users/gliu/Globus_File_Transfer/CESM1_LE/PiControl/SSH/

# Output Directory
dic=/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LE/proc/SSH

# Remapping Options
# this is for 1x1 grid 
# for 192x188 grid, '-remapbil,~/b.e11.B20TRC5CNBDRD.f09_g16.002.cam.h0.TS.192001-200512.nc'
mappp='-remapbil,~/regrid_re1x1.nc'

cd $dd

        
for f in *.nc; do # Loop for all found files
    if [ -e "$f" ] # Check file existence
    then
        echo "$f exists! Regridding..."
        newname="${f%.nc}_regrid.nc"
        printf '\t Renamed to %s\n' $newname
        cdo -O -f nc1 setmissval,1e20 -setmissval,nan ${mappp} -selname,${var} $f ${dic}/$newname
    else
        echo "Did not find $FILE"
    fi
    done
