#! /bin/zsh

# Set the experiments, models, realizations, and variables
expes=("historical") 
models=("ACCESS-ESM1-5") #"MIROC6" "IPSL-CM6A-LR" "CanESM5" "MPI-ESM1-2-LR") #
vars=("tos" "sos" "zos") # zos sos

# Data Directory (note backslash included below for both paths)
dd=/Users/gliu/Globus_File_Transfer/CMIP6
# Output Directory
dic=/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/CMIP6_LENS/regridded/

# Remapping Options
# this is for 1x1 grid 
# for 192x188 grid, '-remapbil,~/b.e11.B20TRC5CNBDRD.f09_g16.002.cam.h0.TS.192001-200512.nc'
mappp='-remapbil,~/regrid_re1x1.nc'

for expe in $expes; do # This loop format is for zsh. Use ${expes[@]} if you are using bash.
    for var in $vars; do
        for model in $models; do

            # Get list of realizations (all the below didn't work, leaving it here for reference...)
            # dir="${dd}/${var}/${model}//"
            # dir="${dir%"${dir##*[!/]}"}"
            # dir="${dir##*/}" 
            #dir="${dir%/}"
            #subdir="${dir##*/}"  
            #reals=$(basename ${dd}/${var}/${model}/*/) # Not sure how to split this

            cd ${dd}/${var}/${model}/ # cd To Directory..
            for real in $( ls -d -- *(/) ); do

                # Get File Path
                FILE=$( ls ${dd}/${var}/${model}/${real}/${var}_Omon_${model}_${expe}_${real}_*.nc )

                if [ -e "$FILE" ] # Check for Existence of File
                then
                    echo "$FILE exists! Regridding..."
                    cdo -O -f nc1 setmissval,1e20 -setmissval,nan ${mappp} -selname,${var} $FILE ${dic}/${var}_${model}_${expe}_${real}_185001-201412.nc
                else
                    echo "Did not find $FILE..."
                fi
            done
        done
    done
done
									
