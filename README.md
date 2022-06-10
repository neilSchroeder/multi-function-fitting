# multi-function-fitting

A demonstration of how to fit multiple functions to data using Scipy

## Features

Fits up to 6 gaussians to some input data

## usage

`python multi_fun_fit.py -d [DATASET] --name [COLUMN NAME] --p0 [INITIAL GUESS] --low-bounds [LOWER BOUNDS] --up-bounds [UPPER BOUNDS]`

guess and bounds need to be entered in the following format:
`[scale] [mean] [width] ...`

## example

The code will fit an unlimited number of gaussians, but will fit 2 by default:

`python multi_fin_fit.py -d data/050522_file01_R1_Processed.csv --name current_0.01s_average`

which produces the following plot:

![Double gaussian fit to suppressed data](https://github.com/neilSchroeder/multi-function-fitting/blob/main/plots/gauss_fit_2_050522_file01_R1_Processed.png)

It is suggested you first fit 2, check the agreement of the fit, then if necessary you can add more using a command with the following form:

`python multi_fun_fit.py -d data/050522_file01_R1_Processed.csv --name current_0.01s_average --p0 8188 -0.00128 0.00068 14113 0.0011 0.00068 3000 0.002 0.002`

which produces a fit using 3 gaussians, shown in the following plot: 

![Three gaussian fit to suppressed data](https://github.com/neilSchroeder/multi-function-fitting/blob/main/plots/gauss_fit_3_050522_file01_R1_Processed.png)

## To Do:

Proper error handling, but for now this is probably as far as the code will go