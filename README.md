# Spectral-Data-Soft-Sensor

## Introduction
The project aims to predict traits and investigate anomalies in A2 spectral dataset. We create 
individual prediction models by using multivariate regression for all the traits. Since there are 
20 traits in our data, we will have 20 different multivariate regression models. Each of these 
models will take the wavelength data as input variables. We evaluate the model predictions and 
rank the models by their performance. For five of the models that have the worst performance, 
we investigate the input matrix using control charts (SPEx and T2). We impose control limits 
for better visualization for the control charts. Further, if there are values exceeding the imposed 
limits, we investigate the wavelength contributions to the control charts. We check, are the 
models performing better if the exceeding samples are excluded from calibration. 
The dataset utilized for this study is the result of a thorough multi-sensor analysis that included 
spectral data and vegetation features from more than 40 different datasets gathered from 
different geographical locations, climatic zones, and plant kinds. For modelling we will use 
partial least squares (PLS) regression. All the coding will be done in Matlab-software. 

## Description of the dataset 
Hyperspectral data that serve as input variables make up the dataset under review. There are 
more than 1700 different input characteristics based on this spectral data, which cover 
wavelengths from 450 to 2500 nm and are taken at 1 nm intervals. However, several 
wavelengths are absent, notably those between 1351 and 1430 nm and 1801 and 2050 nm. 
The response variables, on the other hand, are related to leaf and canopy qualities and include 
20 target variables in total, which can be seen in figure 1. These variables include a wide range 
of characteristics, such as leaf pigments, leaf area index, equivalent water thickness, and more. 
Though there is a linear association between the features, it is critical to remember that this 
dataset lacks time-series characteristics. This dataset differs from standard time-series datasets 
in that all characteristics are concurrently retrieved without regard to time, as opposed to timeseries data, which records information at different time intervals. 
