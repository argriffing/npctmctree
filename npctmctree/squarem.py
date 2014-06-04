"""
This will be coded according to the published literature.

The R code by the original SQUAREM authors is not BSD-compatible
so to minimize licensing issues this Python code will be implemented
according to descriptions in the literature.

"""
#TODO move this into scipy eventually

from functools import partial

import numpy as np
from numpy.linalg import norm


class counted_calls(object):
    def __init__(self, f):
        self._f = f
        self.ncalls = 0
    def __call__(self, *args, **kwargs):
        self.ncalls += 1
        return self._f(*args, **kwargs)


def _compute_step_length(r, v):
    """
    Implementation of equation (9) in the paper.

    This is the third of three schemes, labeled S3 in the paper,
    and according to the paper it has the nice properties of being 
    always negative and bounded, and it is a lower bound for the
    step length defined by scheme S1 when dot(r, v) < 0.

    """
    return -norm(r) / norm(v)


def _modify_step_length(a, L, step):
    """
    The point is to preserve monotonicity.

    Parameters
    ----------
    a : float
        Proposed step length.
    L : function
        Lyapunov function expensively mapping parameter
        values to the merit function value.
    step : function
        Cheaply maps step length to parameter values.

    """
    if a >= 0:
        raise Exception('non-negative step length ' + str(a))
    if a > -1:
        return a
    L0 = L(step(0))
    Ln = L(step(a))
    while Ln < L0:
        a = (a - 1) / 2
        Ln = L(step(a))
    return a


def _step(t, r, v, a):
    return t - 2*a*r + a*a*v


def _check_for_convergence(ta, tb, atol):
    """
    Equation (23) in section 7 of the paper.

    """
    return norm(tb - ta) < atol


def squarem(t0, em_update, L=None, atol=1e-7, em_maxcalls=10000):
    """
    Implementation of pseudocode from Table 1 in the paper.

    Parameters
    ----------
    t0 : ndarray
        Initial guess of parameter values.
    em_update : function
        Updates parameter vector according to EM.
    L : function, optional
        The function to maximize.  If available, this function is required to
        have nice 'Lyapunov function' properties.  An observed data log
        likelihood function will have these properties.

    """
    em_update = counted_calls(em_update)
    converged = False
    while not converged:
        t1 = em_update(t0)
        t2 = em_update(t1)
        r = t1 - t0
        v = (t2 - t1) - r
        step = partial(_step, t1, r, v)
        a = _compute_step_length(r, v)
        if L is not None:
            a = _modify_step_length(a, L, step)
        t0 = em_update(step(a))
        if em_update.ncalls > em_maxcalls:
            raise Exception('too many em calls: ' + str(em_update.ncalls))
        converged = _check_for_convergence(step(a), t0, atol)
    return converged, t0

