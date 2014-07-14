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

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as pyplot




def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)

