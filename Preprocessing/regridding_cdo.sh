expes=(historical)
models=()  # model lists
reals=()  # realization lists
vars=(tos)
dd=    # input file path
dic=   # output file path

mappp='-remapbil,~/regrid_re1x1.nc' # this is for 1x1 grid
									# for 192x188 grid, '-remapbil,~/b.e11.B20TRC5CNBDRD.f09_g16.002.cam.h0.TS.192001-200512.nc'

for expe in $expes # This loop format is for zsh. Use ${expes[@]} if you are using bash. 
do 
	for var in $vars
	do
		for model in $models; do
			for real in $reals; do

				if ls ${dd}/${var}/${model}/${real}/${var}_?mon_${model}_${expe}_*.nc 1> /dev/null 2>&1; then
					m=$( ls ${dd}/${var}/${model}/${real}/${var}_?mon_${model}_${expe}_${real}_*.nc )
					echo $m # Source File full path
					
					cdo -O -f nc1 setmissval,1e20 -setmissval,nan ${mappp} -selname,${var} $m ${dic}${var}_${model}_${expe}_${real}_185001-201412.nc 
				else
				fi

			done
		done
	done
done