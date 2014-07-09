"""
This is an simple parameterized rate matrix for tests and examples.

"""
from __future__ import division, print_function, absolute_import

import numpy as np


def get_pre_Q(kappa, nt_distn1d):
    """
    State order is ACGT.

    Parameters
    ----------
    kappa : float
        The rate scaling ratio of transitions to transversions.
        Nucleotide substitutions A <--> G and C <--> T are called transitions,
        while all other nucleotide substitutions are called transversions.
    nt_distn1d : 1d ndarray of floats
        Mutational nucleotide distribution.

    Returns
    -------
    pre_Q : float 2d ndarray with shape (4, 4).
        A pre-rate-matrix with zeros on the diagonal.

    References
    ----------
    .. [1] Masami Hasegawa and HIrohisa Kishino and Taka-aki Yano,
       Dating of the human-ape splitting by a molecular clock
       of mitochondrial DNA. Journal of Molecular Evolution,
       October 1985, Volume 22, Issue 2, pages 160--174,
       doi:10.1007/BF02101694, PMID 3934395.

    """
    k = kappa
    a, c, g, t = nt_distn1d
    return np.array([
        [  0,   c, k*g,   t],
        [  a,   0,   g, k*t],
        [k*a,   c,   0,   t],
        [  a, k*c,   g,   0]])


def get_normalized_Q(kappa, nt_distn1d):
    """
    HKY rate matrix normalized to have expected rate 1.

    See get_pre_Q for more details.

    """
    pre_Q = get_pre_Q(kappa, nt_distn1d)
    rates_out = pre_Q.sum(axis=1)
    expected_rate = nt_distn1d.dot(rates_out)
    Q = (pre_Q - np.diag(rates_out)) / expected_rate
    return Q

