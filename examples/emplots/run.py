"""
Plot some stuff related to EM.

Check the readme.txt for more info.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize
from scipy.special import xlogy

from npmctree import dynamic_fset_lhood

from npctmctree import hkymodel, expect

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as pyplot




def main(args):
    args.iterations
    args.samples

    # Define the model and the 'true' parameter values.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=10,
            help='number of iterations of EM')
    parser.add_argument('--samples', type=int, default=10000,
            help='number of sampled trajectories per EM iteration')
    args = parser.parse_args()
    main(args)

