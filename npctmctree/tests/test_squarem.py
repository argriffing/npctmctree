"""
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_equal


def test_table_2():
    # init the data
    n = 10
    freqs = np.array([162, 267, 271, 185, 111, 61, 27, 8, 3, 1])
    deaths = np.arange(n)

    # define the em update
    def em_update(t):
        # unpack the parameters
        p = t[0]
        mu = t[1:]
        # compute a summary
        pi_weights = np.empty((n, 2), dtype=float)
        pi_weights[:, 0] = p * np.power(mu[0], deaths) * np.exp(-mu[0])
        pi_weights[:, 1] = (1-p) * np.power(mu[1], deaths) * np.exp(-mu[1])
        pi = pi_weights / pi_weights.sum(axis=1)[:, None]
        # compute updated parameter values
        p_star = freqs.dot(pi[:, 0]) / freqs.sum()
        mu_star_numer = (deaths[:, None] * freqs[:, None] * pi).sum(axis=0)
        mu_star_denom = (freqs[:, None] * pi).sum(axis=0)
        mu_star = mu_star_numer / mu_star_denom
        t_star = np.array([p_star, mu_star[0], mu_star[1]])
        return t_star

    t0 = np.array([0.6, 10, 20])
    t = t0
    for i in range(1000):
        t = em_update(t)
        print(t)

