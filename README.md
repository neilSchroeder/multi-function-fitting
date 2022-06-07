# multi-function-fitting

A demonstration of how to fit multiple functions to data using Scipy

## Features

Fits 2 gaussians to some input data

## usage

`python multi_fun_fit.py -d [DATASET] --name [COLUMN NAME] --p0 [INITIAL GUESS] --low-bounds [LOWER BOUNDS] --up-bounds [UPPER BOUNDS]`

## example

`python multi_fun_fit.py -d data/050522_file01_R1_Processed.csv --name "current_0.01s_average" --p0 6000 12000 -0.0015 0.0015 0.00044 0.00044 --low-bounds 0 0 -1 -1 0 0 --up-bounds 999999 999999 1 1 1 1`


![Double gaussian fit to suppressed data](https://github.com/neilSchroeder/multi-function-fitting/blob/main/plots/two_gauss_fit_050522_file01_R1_Processed.png)