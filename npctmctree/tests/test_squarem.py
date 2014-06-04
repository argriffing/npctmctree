"""
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_equal
import scipy.stats

import npctmctree
from npctmctree.squarem import squarem


def _xdivy(x, y):
    if not x:
        return 0
    else:
        return x / y
xdivy = np.vectorize(_xdivy)


def test_table_2():
    # poisson mixture estimation
    # init the data
    n = 10
    freqs = np.array([162, 267, 271, 185, 111, 61, 27, 8, 3, 1], dtype=float)
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
        print('em step p_star:', p_star)
        mu_star_numer = (deaths[:, None] * freqs[:, None] * pi).sum(axis=0)
        mu_star_denom = (freqs[:, None] * pi).sum(axis=0)
        try:
            mu_star = xdivy(mu_star_numer, mu_star_denom)
        except RuntimeWarning:
            print(mu_star_numer)
            print(mu_star_denom)
            raise
        t_star = np.array([p_star, mu_star[0], mu_star[1]])
        return t_star

    def likelihood(t):
        p = t[0]
        mu = t[1:]
        print('before filter', p, mu, deaths)
        if not (0 <= p <= 1):
            return -np.inf
        if np.any(mu < 0):
            return -np.inf
        print('after filter', p, mu, deaths)
        n = 10
        rv0 = scipy.stats.poisson(mu[0])
        rv1 = scipy.stats.poisson(mu[1])
        a = p * rv0.pmf(deaths)
        b = (1 - p) * rv1.pmf(deaths)
        return np.prod(np.power((a + b), freqs))
    
    def log_likelihood(t):
        p = t[0]
        mu = t[1:]
        print('before filter', p, mu, deaths)
        if not (0 <= p <= 1):
            return -np.inf
        if np.any(mu < 0):
            return -np.inf
        print('after filter', p, mu, deaths)
        n = 10
        rv0 = scipy.stats.poisson(mu[0])
        rv1 = scipy.stats.poisson(mu[1])
        a = p * rv0.pmf(deaths)
        b = (1 - p) * rv1.pmf(deaths)
        try:
            return freqs.dot(np.log(a+b))
        except RuntimeWarning as e:
            print(p, mu)
            print(a+b)
            raise

    t0 = np.array([0.6, 10, 20])
    t = t0
    for i in range(1000):
        t = em_update(t)
        print(t)

    """
    #a, b = squarem(t0, em_update, L=None, atol=1e-7, em_maxcalls=10000)
    for i in range(100):
        t0 = np.array([
            np.random.uniform(0.05, 0.95),
            np.random.uniform(0, 100),
            np.random.uniform(0, 100),
            ])
        print(t0)
        #a, b = squarem(t0, em_update, log_likelihood)
        a, b = squarem(t0, em_update, likelihood)
        print(a)
        print(b)
    """

