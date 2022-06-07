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

def two_gauss(x, a1, a2, m1, m2, w1, w2):
    # returns the sum of 2 gaussians
    return np.add(gauss(x, a1, m1, w1), gauss(x, a2, m2, w2))


def main():

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

    df = pd.read_csv(args.data)
    df.dropna(inplace=True)

    hist, bins = np.histogram(df[args.name].values, bins="auto")
    mids = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    suppressed_hist = [x for x in hist if x > args.suppress]
    suppressed_mids = [mids[i] for i in range(len(mids)) if hist[i] > args.suppress]

    popt, pcov = curve_fit(two_gauss, suppressed_mids, suppressed_hist, p0=args.p0, bounds=(args.lowbounds,args.upbounds))
    g1 = (round(popt[0], 5), round(popt[2], 5), round(popt[4],5))
    g2 = (round(popt[1], 5), round(popt[3], 5), round(popt[5],5))

    print(f'gaussian 1: scale = {g1[0]}, mean = {g1[1]}, width = {g1[2]}')
    print(f'gaussian 2: scale = {g2[0]}, mean = {g2[1]}, width = {g2[2]}')


    fig, ax = plt.subplots(1,1)
    ax.plot(suppressed_mids, suppressed_hist, "k.", label="Data")
    ax.plot(suppressed_mids, two_gauss(suppressed_mids, *popt), "r--", label="Two Gaussian Fit")
    ax.plot([], [], label=f"g1: A {g1[0]}, $\mu$ {g1[1]}, $\sigma$ {g1[2]}")
    ax.plot([], [], label=f"g2: A {g2[0]}, $\mu$ {g2[1]}, $\sigma$ {g2[2]}")
    ax.legend()

    tag = basename(args.data).split(".csv")[0]
    plt.savefig(f"plots/two_gauss_fit_{tag}.png")


if __name__ == '__main__':
    main()