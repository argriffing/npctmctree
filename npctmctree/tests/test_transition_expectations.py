"""
"""
from __future__ import division, print_function, absolute_import

from itertools import product

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose

from scipy.linalg import expm, expm_frechet

from npctmctree.expect import get_edge_to_expectation
from npctmctree import hkymodel


def _check_hky_transition_expectations(t):
    n = 4
    kappa = 3.3
    nt_probs = np.array([0.1, 0.2, 0.3, 0.4])

    # Get an HKY rate matrix with arbitrary expected rate.
    pre_Q = hkymodel.get_pre_Q(kappa, nt_probs)

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
    assert_allclose(expectation_a, t)

    # Try an equivalent calculation which does not use P or J.
    expectation_b = np.diag(nt_probs).dot(S).sum()
    assert_allclose(expectation_b, t)

    # Use the library function.
    T = nx.DiGraph()
    root = 'N0'
    edge = ('N0', 'N1')
    T.add_edge(*edge)
    edge_to_Q = {edge : Q * t}
    edge_to_combination = {edge : pre_Q * t}
    root_distn = nt_probs
    data_weight_pairs = []
    for sa, sb in product(range(n), repeat=2):
        vec_a = np.zeros(n)
        vec_a[sa] = 1
        vec_b = np.zeros(n)
        vec_b[sb] = 1
        data = {'N0' : vec_a, 'N1' : vec_b}
        weight = J[sa, sb]
        data_weight_pairs.append((data, weight))
    edge_to_expectation = get_edge_to_expectation(
            T, root, edge_to_Q, edge_to_combination,
            root_distn, data_weight_pairs)
    expectation_c = edge_to_expectation[edge]
    assert_allclose(expectation_c, t)


def test_hky_transition_expectations():
    for t in 0.2, 1.0, 3.0:
        yield _check_hky_transition_expectations, t

