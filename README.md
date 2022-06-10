# multi-function-fitting

A demonstration of how to fit multiple functions to data using Scipy

## Features

Fits up to 6 gaussians to some input data

## usage

`python multi_fun_fit.py -d [DATASET] --name [COLUMN NAME] --p0 [INITIAL GUESS] --low-bounds [LOWER BOUNDS] --up-bounds [UPPER BOUNDS]`

guess and bounds need to be entered in the following format:
`[scale] [mean] [width] ...`

## example

The code will fit an unlimited number of gaussians

`python multi_fin_fit.py -d data/050522_file01_R1_Processed.csv --name current_0.01s_average`


![Double gaussian fit to suppressed data](https://github.com/neilSchroeder/multi-function-fitting/blob/main/plots/two_gauss_fit_050522_file01_R1_Processed.png)

## To Do:

handle situations where no initial guess or bounds are given