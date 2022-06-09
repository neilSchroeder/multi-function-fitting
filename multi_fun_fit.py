"""
Desc: A script which will fit multiple functions to some data
Author: Neil Schroeder
"""

import argparse as ap
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from os.path import basename

def gauss(x, scale, mean, width):
    # returns a gaussian
    return scale * np.exp(-0.5 * np.multiply(x-mean, x-mean)/(width**2))

def n_gauss(x, *gaussian_params):
    # returns the sum of n gaussians
    n = int(len(gaussian_params)/3)
    ret = np.zeros(len(x))
    for i in range(n):
        param_index = 3*i
        a, b, c = gaussian_params[param_index:param_index+3] if i != n-1 else gaussian_params[param_index::]
        ret = np.add(ret, gauss(x,a,b,c))
    return ret


def main():

    # manage command line options
    parser = ap.ArgumentParser(description="options for this script")

    parser.add_argument("-d","--data", type=str, default=None, 
                        help="path to dataset (csv) which we will open")
    parser.add_argument("--name", type=str, default=None, 
                        help="name of column to fit")
    parser.add_argument("--p0", type=float, nargs="*", default=[], 
                        help="initial guess")
    parser.add_argument("--low-bounds", dest="lowbounds", type=float, nargs="*", default=[], 
                        help="lower bound of each parameter in the fit")
    parser.add_argument("--up-bounds", dest="upbounds", type=float, nargs="*", default=[], 
                        help="upper bound of each parameter in the fit")
    parser.add_argument("--suppress", type=int, default=50,
                        help="suppress the histogram by x counts to get a clean peak")

    args = parser.parse_args()

    # read in the data and drop rows without valid data
    df = pd.read_csv(args.data)
    df.dropna(inplace=True)

    # create a histogram of the data
    hist, bins = np.histogram(df[args.name].values, bins="auto")

    # use the bins to make x values for the center of each bin
    mids = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]

    # suppress the histogram, and the mids, by some amount
    suppressed_hist = [x for x in hist if x > args.suppress]
    suppressed_mids = [mids[i] for i in range(len(mids)) if hist[i] > args.suppress]

    # use the suppressed data and fit with n gaussians
    popt, pcov = curve_fit(n_gauss, suppressed_mids, suppressed_hist, p0=args.p0, bounds=(args.lowbounds,args.upbounds))
    # note: we can use pcov to evaluate the error on the fit parameters if necessary

    # clean things up
    n = int(len(args.p0)/3)
    g = [popt[3*i:3*i+3] if i != n-1 else popt[3*i::] for i in range(n)]
    for i in range(len(g)):
        g[i] = [round(x,5) for x in g[i]]

    # plot the result
    fig, ax = plt.subplots(1,1)
    ax.plot(suppressed_mids, suppressed_hist, "k.", label="Data")
    ax.plot(suppressed_mids, n_gauss(suppressed_mids, *popt), "r--", label=f"{n} Gaussian Fit")
    for i in range(n):
        print(f'gaussian {i}: scale = {g[i][0]}, mean = {g[i][1]}, width = {g[i][2]}')
        ax.plot(suppressed_mids, gauss(suppressed_mids, g[i][0], g[i][1], g[i][2]), '-.', label=f"g{i}: $\mu$ {g[i][1]}")

    ax.legend()

    # create a name for the plot and save
    tag = basename(args.data).split(".csv")[0]
    plt.savefig(f"plots/two_gauss_fit_{tag}.png")
    print(f"[INFO] the plot was written to plots/two_gauss_fit_{tag}.png")

    # write all params to a file for later inspection
    with open(f"plots/fit_results_{tag}.dat","w") as f:
        for i in range(n):
            f.write(f"gaussian {i}: scale = {g[i][0]}, mean = {g[i][1]}, width = {g[i][2]}\n")

    print(f"[INFO] the fit results were written to plots/fit_results_{tag}.dat")


if __name__ == '__main__':
    main()