"""
This script checks inference of branch lengths and HKY parameter values.

The observed data consists of the exact joint leaf state distribution.
The inference uses EM, for which the expectation step is computed exactly
and the maximization step is computed numerically.
The purpose of the EM inference is for comparison to Monte Carlo EM
for which the observed data is not exact and the conditional expectations
are computed using Monte Carlo.

"""
from __future__ import division, print_function, absolute_import

from itertools import product

import numpy as np
import networkx as nx
from numpy.testing import assert_allclose
from scipy.linalg import expm

from npmctree import dynamic_fset_lhood

from npctmctree import hkymodel


def run_inference(T, root, bfs_edges, leaves,
        data_prob_pairs,
        init_kappa, init_nt_probs, init_edge_rates,
        ):
    """
    """
    return 0, 0, 0


def get_hky_edge_to_P(T, root, bfs_edges, kappa, nt_probs, edge_rates):
    """
    Get the per-edge probability transition matrices under the HKY model.

    Compute the conditional transition probability matrices on edges,
    under the given parameter values.
    Use a careful interpretation of rate scaling.

    Parameters
    ----------
    x : x
        x

    Returns
    -------
    x : x
        x

    """
    pre_Q = hkymodel.get_pre_Q(kappa, nt_probs)
    rates_out = pre_Q.sum(axis=1)
    expected_rate = nt_probs.dot(rates_out)
    Q = (pre_Q - np.diag(rates_out)) / expected_rate
    edge_to_P = {}
    for edge, edge_rate in zip(bfs_edges, edge_rates):
        edge_to_P[edge] = expm(edge_rate * Q)
    return edge_to_P


def main():

    # Define the rooted tree shape.
    root = 'N0'
    leaves = ('N2', 'N3', 'N4', 'N5')
    bfs_edges = [
            ('N0', 'N1'),
            ('N0', 'N2'),
            ('N0', 'N3'),
            ('N1', 'N4'),
            ('N1', 'N5'),
            ]
    T = nx.DiGraph()
    T.add_edges_from(bfs_edges)

    # Define some arbitrary 'true' parameter values.
    true_kappa = 2.4
    true_nt_probs = np.array([0.1, 0.2, 0.3, 0.4])
    true_edge_rates = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # Define parameter value guesses to be used for initializing the search.
    init_kappa = 3.0
    init_nt_probs = np.array([0.25, 0.25, 0.25, 0.25])
    init_edge_rates = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # Compute the map from edge to transition probability matrix,
    # under the true parameter values.
    edge_to_P = get_hky_edge_to_P(T, root, bfs_edges,
            true_kappa, true_nt_probs, true_edge_rates)

    # Compute the state distribution at the leaves,
    # under the arbitrary 'true' parameter values.
    root_prior_distn1d = true_nt_probs
    data_prob_pairs = dynamic_fset_lhood.get_unconditional_joint_distn(
            T, edge_to_P, root, root_prior_distn1d, leaves)

    # Check that the computed joint distribution over leaf states
    # is actually a distribution.
    sites, probs = zip(*data_prob_pairs)
    assert_allclose(sum(probs), 1)

    # Check that the 'true' parameters can be inferred given
    # the 'true' state distribution at the leaves and arbitrary
    # initial parameter guesses.
    mle_kappa, mle_nt_probs, mle_edge_rates = run_inference(
            T, root, bfs_edges, leaves,
            data_prob_pairs,
            init_kappa, init_nt_probs, init_edge_rates)


if __name__ == '__main__':
    main()

