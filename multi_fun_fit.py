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

def find_widths(x, y, peaks, means):
    # returns the estimated full width at half max of each peak
    ascend = 1 if means[0] > means[1] else 0
    descend = 0 if ascend == 1 else 1
    widths = []
    for i in range(len(y)-1):
        if y[i] < peaks[ascend]*2/3 and y[i+1] > peaks[ascend]*2/3:
            widths.append(2*abs(means[ascend]-x[i]))
            break

    for i in range(len(y)-1,1,-1):
        if y[i] < peaks[descend]*2/3 and y[i-1] > peaks[descend]*2/3:
            widths.append(2*abs(means[descend]-x[i]))
            break

    return widths


def find_peaks(x,y):
    peaks = []
    means = []
    for i in range(1,len(y)-1):
        if y[i-1] < y[i] > y[i+1]:
            peaks.append(y[i])
            means.append(x[i])

    ret_peaks = []
    ret_means = []
    for i in range(2):
        ret_peaks.append(max(peaks))
        ret_means.append(means[peaks.index(ret_peaks[i])])
        peaks.pop(peaks.index(ret_peaks[i]))
        means.pop(means.index(ret_means[i]))
    return ret_means, ret_peaks 


def get_suggested_params(x,y):
    # returns the suggested parameters for 2 gaussians
    means, scales = find_peaks(x,y)
    widths = find_widths(x, y, scales, means)

    for i in range(2):
        print(f'suggested gaussian {i}: scale = {round(scales[i],5)}, mean = {round(means[i],5)}, width = {round(widths[i],5)}')

    return [scales[0], means[0], widths[0], scales[1], means[1], widths[1]]


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

    parser.add_argument("-d","--data", type=str, default=None, required=True,
                        help="path to dataset (csv) which we will open")
    parser.add_argument("--name", type=str, default=None, required=True,
                        help="name of column to fit")
    parser.add_argument("--p0", type=float, nargs="*", default=[],
                        help="initial guess")
    parser.add_argument("--low-bounds", dest="lowbounds", type=float, nargs="*", default=[],
                        help="lower bound of each parameter in the fit")
    parser.add_argument("--up-bounds", dest="upbounds", type=float, nargs="*", default=[],
                        help="upper bound of each parameter in the fit")
    parser.add_argument("--suppress", type=int, default=50,
                        help="suppress the histogram by x counts to get a clean peak")
    parser.add_argument("--no-fit", default=False, action="store_true", dest="_kNoFit",
                        help="don't fit, just plot the data")

    args = parser.parse_args()

    # read in the data and drop rows without valid data
    df = pd.read_csv(args.data)
    df.dropna(inplace=True)

      # create a histogram of the data
    hist, bins = np.histogram(df[args.name].values, bins="auto")
    hist, bins = np.histogram(df[args.name].values, bins=int(len(bins)*0.8))

    # use the bins to make x values for the center of each bin
    mids = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]

    # suppress the histogram, and the mids, by some amount
    suppressed_hist = [x for x in hist if x > args.suppress]
    suppressed_mids = [mids[i] for i in range(len(mids)) if hist[i] > args.suppress]

    if len(args.p0)==0:
        args.p0 = get_suggested_params(suppressed_mids, suppressed_hist)
    
    if len(args.lowbounds) == 0:
        lb = [0, -1, 0]
        for i in range(int(len(args.p0)/3)):
            args.lowbounds = args.lowbounds + lb

    if len(args.upbounds)==0:
        ub = [999999, 1, 1]
        for i in range(int(len(args.p0)/3)):
            args.upbounds = args.upbounds + ub

    n = 0
    
    if args._kNoFit:
        print("the suggested parameters for 2 gaussians are:")
        get_suggested_params(suppressed_mids, suppressed_hist)
    else:
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
    if not args._kNoFit:
        ax.plot(suppressed_mids, n_gauss(suppressed_mids, *popt), "r--", label=f"{n} Gaussian Fit")
        for i in range(n):
            print(f'gaussian {i}: scale = {g[i][0]}, mean = {g[i][1]}, width = {g[i][2]}')
            ax.plot(suppressed_mids, gauss(suppressed_mids, g[i][0], g[i][1], g[i][2]), '-.', label=f"g{i}: $\mu$ {g[i][1]}")

    

    ax.legend()

    # create a name for the plot and save
    tag = basename(args.data).split(".csv")[0]
    plt.savefig(f"plots/gauss_fit_{n}_{tag}.png")
    print(f"[INFO] the plot was written to plots/gauss_fit_{n}_{tag}.png")

    # write all params to a file for later inspection
    with open(f"plots/fit_results_{n}_gauss_{tag}.dat","w") as f:
        for i in range(n):
            f.write(f"gaussian {i}: scale = {g[i][0]}, mean = {g[i][1]}, width = {g[i][2]}\n")

    print(f"[INFO] the fit results were written to plots/fit_results_{n}_gauss_{tag}.dat")


if __name__ == '__main__':
    main()