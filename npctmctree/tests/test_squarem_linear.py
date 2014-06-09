"""
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.linalg import norm

import npctmctree
from npctmctree.squarem import squarem


def test_linear():
    """
    Solve the linear problem in the squarem paper.

    Note that this paper probably has a typo.
    F(x) = (I + Q)x + b
    should probably be
    F(x) = (I - Q)x + b.

    """
    n = 3
    lam = np.array([0.1, 1, 100])
    b = np.array([-10, -100, 0.1])
    x = np.array([-1, -100, 10], dtype=float)  # exact solution
    t0 = np.zeros(n, dtype=float)  # initial point in the text
    #t0 = np.array([-1.1, -100.2, 10.3], dtype=float)
    Q = np.diag(np.reciprocal(lam))
    ident = np.identity(3)

    def update_em(t):
        #tn = t - (Q.dot(t) - b)
        #tn = (ident + Q).dot(t) + b
        #tn = (ident - Q).dot(t) + b
        tn = (ident - Q).dot(t) + b
        #print(t, '->', tn, ';', norm(Q.dot(t) - b))
        #print(t, '->', tn)
        print(norm(Q.dot(t) - b), norm(t - x))
        return tn

    def merit(t):
        return -norm(Q.dot(t) - b)

    """
    t = t0
    for i in range(100):
        t = update_em(t)
        print(t)
    """

    #result = squarem(t0, update_em, merit)
    result = squarem(t0, update_em, method='SqS1')
    print(result)
