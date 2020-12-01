This repository contains the data and code used in 6.826 (Applied Machine Learning) class final project. 

./Analysis/ directory:
./Analysis/plot_climatology.ipynb: code to plot the climatology and input variables and AMV timeseries for introduction illustration purpose in our paper.
./Analysis/plot_correlation_ResNet50_results.ipynb: code to plot the predictive skill obtained from using ResNet50 (trained and tested on google cloud server).
./Analysis/plot_leadtime_network_comparisons.ipynb: code to plot the annual NAT region AMV predictive skill from FNN and CNN.
./Analysis/plot_leadtime_regional_comparisons.ipynb: code to plot the annual sub-region AMV predictive skill from FNN and CNN.
./Analysis/plot_leadtime_seasonal_comparisons.ipynb: code to plot the seasonal NAT region AMV predictive skill from FNN and CNN.


./Data/ directory:
./Data/CESM*: data that are used as training and validation set.
./Data/leadtime*: stored output including correlation and loss for training and validation set for each model, trained at each leadtime, for each input variable combination.


./Figures/ directory:
./*: all the figures appeared in the final report


./Linear_regression/ directory:
./Linear_regression/build_plot_linear_regression_at_lags.ipynb: build the linear regression model (using NAT averaged SST, SSS, PSL all together) from CESM and calculate the correlation for predicted AMV using linear regression model.


./NNs/ directory:
./NNs/NN_test_lead_ann.py: code to train and validate FNN and CNN in CESM. Detailed function descriptions in the code. The results (e.g. loss, correlation) are stored as output in the ./Data/ directory with prefix leadtime*.


./Preprocessing/ directory: 
./Preprocessing/coarsen_data.py: code to re-grid the CESM and reanalysis data to 2 degrees horizontal resolution.
./Preprocessing/output_normalized_data.ipynb: code to de-seasonlize the data, and normalize the data before training and validation. 


./Reanalysis_validation/ directory:
./Reanalysis_validation/corrs_for_best_models.csv: results (correlation for each combinaion of NNs, sub-regions and seasons) calculated from ./Reanalysis_validation/validate_reanalysis.ipynb.
./Reanalysis_validation/plot_correlation_for_reanalysis_data.ipynb: code to plot the correlation when testing the NNs on reanalysis data.
./Reanalysis_validation/validate_reanalysis.ipynb: code to use the weights learned by NNs with CESM data and directly apply the weights and model to reanalysis data. Correlation is output in the corrs_for_best_models.csv file.


./Scrap/ directory:
./*: code used for testing purposes in milestone and other prior tests, and are not used in producing the results for the final report.


