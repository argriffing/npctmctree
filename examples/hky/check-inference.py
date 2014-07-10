"""
This script checks inference of branch lengths and HKY parameter values.

The observed data consists of the exact joint leaf state distribution.
The inference uses EM, for which the expectation step is computed exactly
and the maximization step is computed numerically.
The purpose of the EM inference is for comparison to Monte Carlo EM
for which the observed data is not exact and the conditional expectations
are computed using Monte Carlo.

Expectation maximization can be modeled after the code in nxctmctree
which uses Monte Carlo EM with trajectory samples.

"""
from __future__ import division, print_function, absolute_import

import argparse
import itertools
from functools import partial

import numpy as np
import networkx as nx
from numpy.testing import assert_allclose
from scipy.special import xlogy
from scipy.linalg import expm
from scipy.optimize import minimize

from npmctree import dynamic_fset_lhood

from npctmctree import hkymodel, expect



#NOTE from nxctmctree
def pack_params(edges, edge_rates, nt_probs, kappa):
    """
    Pack parameters into a 1d ndarray.

    """
    params = np.concatenate([edge_rates, nt_probs, [kappa]])
    log_params = np.log(params)
    return log_params


#NOTE from nxctmctree
def unpack_params(edges, log_params):
    """
    Unpack parameters from a 1d ndarray.

    """
    params = np.exp(log_params)
    nedges = len(edges)
    edge_rates = params[:nedges]
    nt_distn1d = params[nedges:nedges+4]
    penalty = np.square(np.log(nt_distn1d.sum()))
    nt_distn1d = nt_distn1d / nt_distn1d.sum()
    kappa = params[-1]

    # Get the transition rate matrix, carefully scaled.
    Q = hkymodel.get_normalized_Q(kappa, nt_distn1d)

    # Return the unpacked parameters.
    return edge_rates, Q, nt_distn1d, kappa, penalty


#TODO replace some code somewhere with a call to the new
# npmctree get_unconditional_joint_distribution function.
# the code to be replaced should be somewhere in npctmctree.


def get_expected_log_likelihood(T, root, edges,
        edge_to_Q, edge_to_rate, root_prior_distn1d,
        root_state_counts, edge_to_dwell_times, edge_to_transition_counts):
    """
    Expected log likelihood of trajectories.

    """
    # Log likelihood contribution of root state expectation.
    init_ll = xlogy(root_state_counts, root_prior_distn1d).sum()

    # Log likelihood contribution of dwell times and transitions.
    dwell_ll = 0
    trans_ll = 0
    for edge in edges:
        dwell_times = edge_to_dwell_times[edge]
        transition_counts = edge_to_transition_counts[edge]
        edge_rate = edge_to_rate[edge]
        Q = edge_to_Q[edge]
        dwell_ll += edge_rate * np.diag(Q).dot(dwell_times)
        trans_ll += xlogy(transition_counts, edge_rate * Q).sum()
    log_likelihood = init_ll + dwell_ll + trans_ll
    return log_likelihood


#NOTE from nxctmctree
def objective(T, root, edges,
        root_state_counts, edge_to_dwell_times, edge_to_transition_counts,
        log_params):
    """
    Negative expected log likelihood.

    It is penalized if the nucleotide probabilities do not add up to 1.
    The nucleotide penalties are already forced to be positive
    using a transformation of variables.

    """
    unpacked = unpack_params(edges, log_params)
    edge_rates, Q, nt_distn1d, kappa, penalty = unpacked
    edge_to_rate = dict(zip(edges, edge_rates))
    edge_to_Q = dict((e, Q) for e in edges)
    root_prior_distn1d = nt_distn1d
    log_likelihood = get_expected_log_likelihood(T, root, edges,
            edge_to_Q, edge_to_rate, root_prior_distn1d,
            root_state_counts, edge_to_dwell_times, edge_to_transition_counts)
    penalized_neg_ll = -log_likelihood + penalty
    return penalized_neg_ll


def run_inference(T, root, bfs_edges, leaves,
        data_weight_pairs,
        kappa, nt_distn1d, edge_rates,
        max_iterations,
        ):
    """
    Run the inference.

    Parameters
    ----------
    T : networkx DiGraph
        Rooted tree.
    root : hashable node
        Root of the tree as a hashable networkx node in T.
    bfs_edges : sequence
        Ordered edges in a preorder from the root.
        Each edge is a directed pair of nodes.
    leaves : sequence
        Leaf nodes.
    data_weight_pairs : sequence of pairs
        Weighted data or simulated data or an exact distribution.
    kappa : float
        Initial guess for kappa parameter.
        Kappa is the rate scaling ratio of transitions to transversions.
        Nucleotide substitutions A <--> G and C <--> T are called transitions,
        while all other nucleotide substitutions are called transversions.
    nt_distn1d : 1d ndarray of floats
        Initial guess for mutational nucleotide distribution.
    edge_rates : 1d ndarray of floats
        Initial guess for the edge rate scaling factors.
    max_iterations : integer or None
        Optionally limit the number of iterations.

    Returns
    -------
    mle : (mle_kappa, mle_nt_distn1d, mle_edge_rates)
        Maximum likelihood estimates.

    """
    # Look at nxctmctree for a template for the full MLE.
    nstates = nt_distn1d.shape[0]

    for iteration_idx in itertools.count():

        # Check early stop condition.
        if max_iterations is not None:
            if iteration_idx >= max_iterations:
                break

        # Report the EM iteration underway.
        print('iteration', iteration_idx+1, '...')

        # Use the unpacked parameters to create the carefullly scaled
        # transition rate matrix.
        Q = hkymodel.get_normalized_Q(kappa, nt_distn1d)

        # Create the edge specific rate matrices,
        # carefully scaled by the edge-specific rate scaling factors.
        edge_to_Q = {}
        for edge, edge_rate in zip(bfs_edges, edge_rates):
            edge_to_Q[edge] = edge_rate * Q

        # Get posterior expected root distribution.
        root_prior_distn1d = nt_distn1d
        edge_to_P = dict((e, expm(Q)) for e, Q in edge_to_Q.items())
        root_state_counts = np.zeros(nstates)
        for data, weight in data_weight_pairs:
            node_to_distn1d = dynamic_fset_lhood.get_node_to_distn1d(
                    T, edge_to_P, root, root_prior_distn1d, data)
            root_post_distn1d = node_to_distn1d[root]
            root_state_counts += weight * root_post_distn1d

        # Get posterior expected dwell times and transition counts.
        edge_to_dwell_times = expect.get_edge_to_dwell(
                T, root, edge_to_Q, root_prior_distn1d, data_weight_pairs)
        edge_to_transition_counts = expect.get_edge_to_trans(
                T, root, edge_to_Q, root_prior_distn1d, data_weight_pairs)

        # Maximization step of EM.
        f = partial(objective, T, root, bfs_edges,
                root_state_counts,
                edge_to_dwell_times,
                edge_to_transition_counts)
        x0 = pack_params(bfs_edges, edge_rates, nt_distn1d, kappa)
        result = minimize(f, x0, method='L-BFGS-B')

        # Unpack optimization output.
        log_params = result.x
        unpacked = unpack_params(bfs_edges, log_params)
        edge_rates, Q, nt_distn1d, kappa, penalty = unpacked

        # Summarize the EM step.
        edge_to_rate = dict(zip(bfs_edges, edge_rates))
        print('EM step summary:')
        print('objective function value:', result.fun)
        for edge, rate in zip(bfs_edges, edge_rates):
            print('edge:', edge, 'rate:', rate)
        print('nucleotide distribution:', nt_distn1d)
        print('kappa:', kappa)
        print('penalty:', penalty)
        print()

    # Return the maximum likelihood estimates computed with EM.
    return kappa, nt_distn1d, edge_rates


def main(args):

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
    Q = hkymodel.get_normalized_Q(true_kappa, true_nt_probs)
    edge_to_P = {}
    for edge, edge_rate in zip(bfs_edges, true_edge_rates):
        edge_to_P[edge] = expm(edge_rate * Q)

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
            init_kappa, init_nt_probs, init_edge_rates,
            args.iterations,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int,
            help='restrict the EM to this many iterations')
    args = parser.parse_args()
    main(args)

