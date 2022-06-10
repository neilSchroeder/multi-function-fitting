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

`python multi_fin_fit.py -d data/050522_file01_R1_Processed.csv --name current_0.01s_average --no-fit`

this returns the following suggested parameters:

```
suggested gaussian 0: A 14113, $\mu$ 0.0011, $\sigma$ 0.00068  
suggested gaussian 1: A 8188, $\mu$ -0.00128, $\sigma$ 0.00068
```

so we run with these suggestions as follows:

`python multi_fun_fit.py -d data/050522_file01_R1_Processed.csv --name current_0.01s_average --p0 14113 0.0011 0.00068 8188 -0.00128 0.00068 --low-bounds 0 -1 0 0 -1 0 --up-bounds 999999 1 1 999999 1 1`


![Double gaussian fit to suppressed data](https://github.com/neilSchroeder/multi-function-fitting/blob/main/plots/two_gauss_fit_050522_file01_R1_Processed.png)

## To Do:

handle situations where no initial guess or bounds are given