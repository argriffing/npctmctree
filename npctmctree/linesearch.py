"""
Attempt to implement an EM acceleration.

The main Quasi-Newton algorithm is from the following source:
Acceleration of the EM algorithm by using quasi-Newton methods.
Jamshidian and Jennrich, 1997

The line search is from the following source:
Conjugate gradient acceleration of the EM algorithm.
Jamshidian and Jennrich, 1993
Appendix A.5.

"""
from __future__ import division, print_function, absolute_import

import itertools

import numpy as np


class LineSearchError(Exception):
    pass


def get_dS(dg, dt, dtp):
    """
    helper function for jj97 qn2.
    Equation 4.1.
    """
    d = np.dot(dg, dt)
    a = (1 + np.dot(dg, dtp) / d)
    b = np.outer(dt, dt)
    c = np.outer(dtp, dt) + np.outer(dt, dtp)
    return (a*b - c) / d


def jj97_qn2(t0, grad, em, bounds=None):
    """

    Parameters
    ----------
    t0 : ndarray
        initial guess
    grad : function
        gradient of log likelihood
    em : function
        em displacement function
    bounds : sequence, optional
        inclusive box constraints on the search

    """
    n = len(t0)
    t = t0
    for jj97_start_count in itertools.count(1):
        print('jj97 start:', jj97_start_count)
        S = np.zeros((n, n), dtype=float)
        g = grad(t)
        h = em(t)
        for jj97_iteration in itertools.count(1):
            print('jj97 iteration:', jj97_iteration)
            d = -h + S.dot(g) # not sure why the -1 coefficient on h...
            try:
                a1, t1, g1 = jj93_linesearch(grad, t, d, 1.0, g, bounds)
            except LineSearchError as e:
                break
            h1 = em(t1)
            dt = t1 - t
            dg = g1 - g
            dh = h1 - h
            dtp = -dh + S.dot(dg) # not sure why the -1 coefficient on dh...
            dS = get_dS(dg, dt, dtp)
            S1 = S + dS
            t = t + dt
            t, g, h, S = t1, g1, h1, S1
            print(t)


def box_ok(bounds, x):
    if bounds is None:
        return True
    if len(bounds) != len(x):
        raise ValueError
    for x, (low, high) in zip(x, bounds):
        if low is not None and low > x:
            return False
        if high is not None and x < high:
            return False
    return True


def jj93_linesearch(g, theta, d, a1, g0=None, bounds=None):
    """
    Find the maximum of F(a) = f(theta + a1*d).

    helper function for jj97 qn2.

    Assumptions are as follows.
    alpha >= 0.
    The slope of F(alpha) is nonnegative at alpha=0.
    This linesearch is based on the secant method.

    Parameters
    ----------
    g : function
        gradient of function to maximize
    theta : point
        initial point
    d : vector
        direction of the search
    a1 : float
        initial distance in the direction of the search
    g0 : vector, optional
        gradient at the initial point
    bounds : sequence, optional
        inclusive box constraints on the search

    Returns
    -------
    a1 : float
        alpha value for estimated maximum
    x1 : float
        estimated maximum value
    g1 : float
        gradient at x1

    """
    split_limit = 10
    # Step 0
    n = 0
    a0 = 0
    x0 = theta + a0*d
    if g0 is None:
        g0 = g(x0)
    G0 = np.dot(d, g0)
    Ga0 = G0
    while True:
        # Step 1
        while n < split_limit:
            print('n =', n)
            x1 = theta + a1*d
            if box_ok(bounds, x1):
                ga1 = g(x1)
                Ga1 = np.dot(d, ga1)
                n += 1
                break
            else:
                a1 /= 2
                n += 1
        # Step 2
        if n == split_limit:
            raise LineSearchError('please restart search')
        Ga0m = np.abs(Ga0)
        Ga1m = np.abs(Ga1)
        if n != 1 and Ga1m < 0.1 * G0:
            return a1, x1, ga1
        criterion = np.sign(a1 - a0) * (Ga0 - Ga1) / (Ga0m + Ga1m)
        #print('criterion calculation...')
        #print(a1)
        #print(a0)
        #print(Ga0)
        #print(Ga1)
        #print(Ga0m)
        #print(Ga1m)
        #print(criterion)
        if criterion < 1e-5:
            raise LineSearchError('please restart search')
        else:
            a0, Ga0, a1 = a1, Ga1, (a1*Ga0 - a0*Ga1) / (Ga0 - Ga1)

