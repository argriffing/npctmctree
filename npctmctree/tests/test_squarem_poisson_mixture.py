"""
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_equal
import scipy.stats
from scipy.misc import logsumexp
from scipy.special import log1p
from numpy.linalg import norm

import npctmctree
from npctmctree.squarem import squarem, counted_calls


class DegenerateMixtureError(Exception):
    pass


def _xdivy(x, y):
    if not x:
        return 0
    else:
        return x / y
xdivy = np.vectorize(_xdivy)


"""
def poisson_mix_log_likelihood(counts, weights, mix, mu):
    # counts : a data vector of observed counts
    # weights : the number of times each count was observed
    # mix : finite distribution over mixture components
    # mu : means of poisson mixture components
    assert_equal(counts.shape, weights.shape)
    assert_equal(mix.shape, mu.shape)
"""


def log_weights_to_distn(log_weights):
    # Try to be a bit clever about scaling.
    #print('log weights:', log_weights)
    m = min(x for x in log_weights if x > -np.inf)
    reduced_weights = [
            np.exp(x - m) if x > -np.inf else 0 for x in log_weights]
    distn = np.array(reduced_weights) / np.sum(reduced_weights)
    return distn
    #print(log_weights)
    #min_log_weight = np.min(log_weights)
    #reduced_weights = np.exp(log_weights - min_log_weight)
    #distn = reduced_weights / reduced_weights.sum()
    ##print('distribution update:', distn)
    #return distn


def test_table_2():
    # poisson mixture estimation
    # init the data
    # mle should be p0=0.3599, mu0=1.256, mu1=2.663
    np.random.seed(1234)
    n = 10
    freqs = np.array([162, 267, 271, 185, 111, 61, 27, 8, 3, 1], dtype=float)
    deaths = np.arange(n)

    def inbounds(t):
        p0 = t[0]
        mu = t[1:]
        if p0 < 0:
            return False
        if p0 > 1:
            return False
        if np.any(mu <= 0):
            return False
        return True

    # define the em update
    def em_update(t):
        #print('attempting em input:', t)
        if not inbounds(t):
            raise Exception('input to the em update is out of bounds (%s)' % t)
        ll_before_update = log_likelihood(t)
        p0 = t[0]
        mu = t[1:]
        # define the poisson components
        rs = [scipy.stats.poisson(m) for m in mu]
        # compute the per-count posterior distribution over
        pi_log_weights = np.empty((n, 2), dtype=float)
        # vectorize this later
        for i in range(n):
            if p0 > 0:
                alpha = np.log(p0)
                beta = rs[0].logpmf(deaths[i])
                loga = alpha + beta
                if np.isnan(loga):
                    print('nan', mu[0], deaths[i], alpha, beta, loga)
            else:
                loga = -np.inf
            if p0 < 1:
                alpha = log1p(-p0)
                beta = rs[1].logpmf(deaths[i])
                logb = alpha + beta
                if np.isnan(logb):
                    print('nan', mu[1], deaths[i], alpha, beta, logb)
            else:
                logb = -np.inf
            pi_log_weights[i, 0] = loga
            pi_log_weights[i, 1] = logb
        # convert log weights to a distribution, being careful about scaling
        pi = np.empty((n, 2), dtype=float)
        for i in range(n):
            pi[i] = log_weights_to_distn(pi_log_weights[i])
        #pi_weights[:, 0] = p0 * np.power(mu[0], deaths) * np.exp(-mu[0])
        #pi_weights[:, 1] = (1-p0) * np.power(mu[1], deaths) * np.exp(-mu[1])
        #pi = pi_weights / pi_weights.sum(axis=1)[:, None]
        # compute updated parameter values
        p_star = freqs.dot(pi[:, 0]) / freqs.sum()
        #print('em step p_star:', p_star)
        mu_star = np.zeros(2, dtype=float)
        for j in range(2):
            numer = sum(deaths[i] * freqs[i] * pi[i, j] for i in range(n))
            denom = sum(freqs[i] * pi[i, j] for i in range(n))
            if numer:
                mu_star[j] = numer / denom
            else:
                raise DegenerateMixtureError('a poisson mean is zero')
        #mu_star_numer = (deaths[:, None] * freqs[:, None] * pi).sum(axis=0)
        #mu_star_denom = (freqs[:, None] * pi).sum(axis=0)
        #try:
            #mu_star = xdivy(mu_star_numer, mu_star_denom)
        #except RuntimeWarning:
            #print(mu_star_numer)
            #print(mu_star_denom)
            #raise
        t_star = np.array([p_star, mu_star[0], mu_star[1]])
        ll_after_update = log_likelihood(t_star)
        #if ll_after_update < ll_before_update:
            #print('log likelihoods:', ll_before_update, ll_after_update)
            #raise Exception('em step reduced observed data log likelihood')
        if not inbounds(t_star):
            raise Exception('em update output is out of bounds (%s)' % t_star)
        return t_star

    def likelihood(t):
        if not inbounds(t):
            return 0
        p = t[0]
        mu = t[1:]
        n = 10
        rv0 = scipy.stats.poisson(mu[0])
        rv1 = scipy.stats.poisson(mu[1])
        a = p * rv0.pmf(deaths)
        b = (1 - p) * rv1.pmf(deaths)
        return np.prod(np.power((a + b), freqs))
    
    def log_likelihood(t):
        if not inbounds(t):
            return -np.inf
        p = t[0]
        mu = t[1:]
        n = 10
        rv0 = scipy.stats.poisson(mu[0])
        rv1 = scipy.stats.poisson(mu[1])
        #a = p * rv0.pmf(deaths)
        #b = (1 - p) * rv1.pmf(deaths)
        #try:
            #return freqs.dot(np.log(a+b))
        #except RuntimeWarning as e:
            #print(p, mu)
            #print(a+b)
            #raise
        ll = 0
        for i in range(n):
            loga = np.log(p) + rv0.logpmf(deaths[i])
            logb = log1p(-p) + rv1.logpmf(deaths[i])
            ll += freqs[i] * logsumexp([loga, logb])
        return ll

    def neg_log_likelihood(t):
        return -log_likelihood(t)


    # Initialization used in the R tutorial.
    # This works for the vanilla fixed-point EM iteration,
    # requiring 2909 EM evaluations to reach 1e-8 absolute error
    # between iterations.
    # With no step length adjustment
    # this gives 75 fixed point evaluations for accelerated EM,
    # whereas the tutorial says it should take only 72.
    # With step length adjustment, too many evaluations are used.
    # The tutorial claims that for this configuration
    # no step length adjustment is necessary.
    """
    > setRNG(list(kind="Wichmann-Hill", normal.kind="Box-Muller", seed=123)
            + )
    > c(runif(1), runif(2, 0, 6))
    [1] 0.4462944 5.3433981 0.8713513
    """
    #atol = 1e-8
    #t0 = np.array([0.4462944, 5.3433981, 0.8713513])

    # TurboEM tutorial by Jennifer Bobb, page 4.
    #
    # squarem
    # uses the objective function
    # 23 iterations
    # 45 em calls
    # 24 objective function calls
    # converged
    #
    # plain em
    # strangely, reports objective function evaluations
    # 1500 iterations
    # 1500 em calls
    # 1501 objective function calls
    # did not converge
    t0 = np.array([0.5, 1, 3])
    # For squarem without globalization I see 54 em calls
    # For squarem with globalization I see 117 em calls
    # These observed numbers of EM calls are too many...

    # from table in slides:
    # mle should be p0=0.3599, mu0=1.256, mu1=2.663
    # the following starting point was used in the paper and slides:
    #t0 = np.array([0.3, 1.0, 2.5])
    # in the table in the slides, this starting guess iss associated
    # with the following number of EM function evals (not objective function)
    # for each of the various schemes:
    # method  reported  observed
    # EM      2055       255 (atol=1e-7)
    # S1       396
    # S2       477
    # S3       576
    # SqS1      66
    # SqS2      84
    # SqS3      66        66 (without step length adjustment, atol=1e-7)

    #"""
    # the following starting point was causing nans:
    #t0 = np.array([0.68539781, 14.9833716, 74.60634091])
    #t0 = np.array([0.28, 1.06, 2.59])
    objective = counted_calls(log_likelihood)
    counted_em_update = counted_calls(em_update)

    """
    # run the acclerated expectation maximization
    atol = 1e-7
    t = t0
    print('accelerated EM without merit function:')
    result = squarem(t, counted_em_update, atol=atol)
    print(result)
    print('number of EM update calls:', counted_em_update.ncalls)
    print()

    # run the acclerated expectation maximization
    atol = 1e-7
    t = t0
    print('accelerated EM with merit function:')
    result = squarem(t, counted_em_update, objective, atol=atol)
    print(result)
    print('number of EM update calls:', counted_em_update.ncalls)
    print('number of objective function calls:', objective.ncalls)
    print()

    # run an iterated expectation maximization
    t = t0
    print('plain EM:')
    counted_em_update = counted_calls(em_update)
    while True:
        tnext = counted_em_update(t)
        if norm(tnext - t) < atol:
            break
        t = tnext
    print('number of EM update calls:', counted_em_update.ncalls)
    print()
    """

    """
    t0 = np.array([0.6, 10, 20])
    t = t0
    for i in range(1000):
        t, b = squarem(t0, em_update, log_likelihood)
        print(t)
    """

    #"""
    #a, b = squarem(t0, em_update, L=None, atol=1e-7, em_maxcalls=10000)
    ndegenerate = 0
    for i in range(100):
        print('iteration', i)
        try:
            t0 = np.array([
                np.random.uniform(0.05, 0.95),
                np.random.uniform(0, 100),
                np.random.uniform(0, 100),
                ])
            print(t0)
            a, b = squarem(t0, em_update, log_likelihood)
            #a, b = squarem(t0, em_update, likelihood)
            print(a)
            print(b)
        except DegenerateMixtureError as e:
            ndegenerate += 1
            print('found', ndegenerate, 'degenerate solutions so far')
    #"""

