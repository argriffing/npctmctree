"""
"""
from __future__ import division, print_function, absolute_import

#TODO convert this to a unit test

import numpy as np
from numpy.testing import assert_allclose

from scipy.linalg import expm, expm_frechet





def hamming_distance(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def get_hky_pre_Q(kappa, nt_probs):
    """
    This is just hky.

    State order is ACGT.

    """
    n = 4
    transitions = ((0, 2), (2, 0), (1, 3), (3, 1))
    pre_Q = np.zeros((n, n), dtype=float)
    for sa, pa in enumerate(nt_probs):
        for sb, pb in enumerate(nt_probs):
            if sa == sb:
                continue
            rate = 1.0
            rate *= pb
            if (sa, sb) in transitions:
                rate *= kappa
            pre_Q[sa, sb] = rate
    return pre_Q


def check_hky_expectations(t):
    n = 4
    kappa = 3.3
    nt_probs = np.array([0.1, 0.2, 0.3, 0.4])

    # Get an HKY rate matrix with arbitrary expected rate.
    pre_Q = get_hky_pre_Q(kappa, nt_probs)

    # Rescale the rate matrix to have expected rate of 1.0.
    rates = pre_Q.sum(axis=1)
    expected_rate = rates.dot(nt_probs)
    pre_Q /= expected_rate

    # Convert the pre-rate matrix to an actual rate matrix
    # by subtracting row sums from the diagonal.
    rates = pre_Q.sum(axis=1)
    Q = pre_Q - np.diag(rates)

    # Create the transition probability matrix over time t.
    P = expm(Q*t)
    assert_allclose(P.sum(axis=1), 1)

    # Create a joint state distribution matrix.
    J = np.diag(nt_probs).dot(P)
    assert_allclose(J.sum(), 1)

    # Get the expm frechet matrix.
    C = pre_Q * t
    S = expm_frechet(Q*t, C, compute_expm=False)

    # Get the weighted sum of entries of S.
    expectation_a = ((S / P) * J).sum()

    # Try an equivalent calculation which does not use P or J.
    expectation_b = np.diag(nt_probs).dot(S).sum()

    print('t:', t)
    print('Q:')
    print(Q)
    print('P:')
    print(P)
    print('C:')
    print(C)
    print('J:')
    print(J)
    print('S:')
    print(S)
    print('expectation_a:', expectation_a)
    print('expectation_b:', expectation_b)
    print()


def main():
    for t in (0.2, 1.0, 3.0):
        check_hky_expectations(t)


if __name__ == '__main__':
    main()

