<h1>This repository contains the data and code used in Predicting Atlantic Multidecadal Variability. </h1>

<b> Analysis :</b>
This directory contains the code to load and visualize the results of training/testing for each Machine Learning Model.

<b> Data/Metrics :</b>
This directory contains the outputs (total accuracy, accuracy by class for each lead time) generated by scripts within ./Models/, and are visualized in ./Analysis

<b> Models :</b>
The scripts to train and test each model are contained here. This includes...
*NN_test_lead_ann_ImageNet_classification.py : Train/Test a Convolutional Neural Network or ResNet50 (pretrained with either FractalDB or ImageNet)
*autosklearn_calc_baseline_metrics.py : Train/Test using AutoML
*Persistence_Classification_Baseline.py : Calculate the accuracy expected using the persistence baseline

<b> Preprocessing :</b>
Reads in data located in ../CESM_data/. Regrids the data and also calculates the AMV Index (objective)

<b> Scrap :</b>
Catchall folder for old scripts and figures. Code in here may not be completely documented.

Note: In the same directory which contains predict_amv, place the CESM_data folder. The data for this project can be downloaded at this Dropbox link: https://www.dropbox.com/s/d2wokw4f15mejlk/CESM_data.zip?dl=0
