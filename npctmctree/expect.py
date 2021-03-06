"""
Expectation maximization helper functions.

Specifically, estimate edge specific rate scaling factors for
rate matrices on a rooted tree with known prior state distribution at the root
and with partial observations at nodes of the tree.

"""
from __future__ import division, print_function, absolute_import

from itertools import permutations

import networkx as nx
import numpy as np

from numpy.testing import assert_equal
from scipy.linalg import expm_frechet
from scipy.special import xlogy

from .cyem import conditional_expectation


def get_expected_log_likelihood(T, root, edges,
        edge_to_Q, edge_to_rate, root_prior_distn1d,
        root_state_counts, edge_to_dwell_times, edge_to_transition_counts):
    """
    Expected log likelihood of trajectories.

    Returns
    -------
    log_likelihood : float
        Expected log likelihood.

    Notes
    -----
    This function is for generic EM for parameterized rate matrices,
    and could be further optimized for speed in a way that is similar
    to the EM functions that are specialized for estimation
    of edge rate scaling factors.
    Each Q in the edge_to_Q dict is normalized and does not include the rate
    in the edge_to_rate dict.

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


def get_edge_to_dwell(T, root, edge_to_Q, root_distn, data_weight_pairs):
    """
    Compute dwell times in each state on each edge, summed over sites.

    Notes
    -----
    This function is for generic EM for parameterized rate matrices,
    and could be further optimized for speed in a way that is similar
    to the EM functions that are specialized for estimation
    of edge rate scaling factors.

    """
    # Define the state space.
    nstates = root_distn.shape[0]
    states = range(nstates)

    # Get edges in no particular order.
    edges = list(edge_to_Q)

    # Compute the map from edge to dwell times.
    edge_to_dwell = dict((e, np.zeros(nstates)) for e in edges)
    for s in states:
        combination = np.zeros((nstates, nstates), dtype=float)
        combination[s, s] = 1
        edge_to_combination = {}
        for edge in edges:
            edge_to_combination[edge] = combination
        edge_to_expectation = get_edge_to_expectation(
                T, root, edge_to_Q, edge_to_combination,
                root_distn, data_weight_pairs)
        for edge in edges:
            edge_to_dwell[edge][s] = edge_to_expectation[edge]
    return edge_to_dwell


def get_edge_to_trans(T, root, edge_to_Q, root_distn, data_weight_pairs):
    """
    Compute counts for each transition type on each edge, summed over sites.

    Notes
    -----
    This function is for generic EM for parameterized rate matrices,
    and could be further optimized for speed in a way that is similar
    to the EM functions that are specialized for estimation
    of edge rate scaling factors.

    """
    # Define the state space.
    nstates = root_distn.shape[0]
    states = range(nstates)

    # Get edges in no particular order.
    edges = list(edge_to_Q)

    # Compute the map from edge to expected transition counts.
    edge_to_trans = dict((e, np.zeros((nstates, nstates))) for e in edges)
    for sa, sb in permutations(states, 2):
        edge_to_combination = {}
        for edge in edges:
            Q = edge_to_Q[edge]
            combination = np.zeros((nstates, nstates), dtype=float)
            combination[sa, sb] = Q[sa, sb]
            edge_to_combination[edge] = combination
        edge_to_expectation = get_edge_to_expectation(
                T, root, edge_to_Q, edge_to_combination,
                root_distn, data_weight_pairs)
        for edge in edges:
            edge_to_trans[edge][sa, sb] = edge_to_expectation[edge]
    return edge_to_trans



def get_edge_to_expectation(T, root, edge_to_Q, edge_to_combination,
        root_distn, data_weight_pairs):
    """
    This function does not require highly processed inputs.

    Parameters
    ----------
    T : networkx DiGraph
        The directed tree graph.
    root : hashable
        Root of the tree.
    edge_to_Q : dict
        Map from edge to unit scaled numpy transition rate matrix.
    edge_to_combination : dict
        Map from edge to an ndarray specifying the expectation request.
    root_distn : 1d ndarray
        Prior state distribution at the root, as a 1d ndarray.
    data_weight_pairs : sequence of (data, weight) pairs
        Observed data from which to compute conditional expectations.
        For each pair in the sequence, the data consists of a map from
        nodes in the tree to an ndarray defining observation likelihoods
        conditional on each potential state at the node.

    Returns
    -------
    edge_to_expectation : dict
        Map from each edge to the weighted sum of expectations on the edge.
        Note that because of linearity of expectation, a weighted sum
        of expectations is the expectation of the weighted sum.

    """
    # Define a toposort node ordering and a corresponding csr matrix.
    nodes = nx.topological_sort(T, [root])
    node_to_idx = dict((na, i) for i, na in enumerate(nodes))
    m = nx.to_scipy_sparse_matrix(T, nodes)

    # Summarize the process.
    nnodes = len(nodes)
    nstates = root_distn.shape[0]

    # Stack the transition rate matrices into a single array.
    transq = np.empty((nnodes-1, nstates, nstates), dtype=float)
    for (na, nb), Q in edge_to_Q.items():
        eidx = node_to_idx[nb] - 1
        transq[eidx] = Q

    # Stack the combination matrices into a single array.
    combinations = np.empty((nnodes-1, nstates, nstates), dtype=float)
    for (na, nb), combination in edge_to_combination.items():
        eidx = node_to_idx[nb] - 1
        combinations[eidx] = combination

    # Stack the data into a single array,
    # and construct an array of site weights.
    nsites = len(data_weight_pairs)
    datas, weights = zip(*data_weight_pairs)
    site_weights = np.array(weights, dtype=float)
    data = np.empty((nsites, nnodes, nstates), dtype=float)
    for site_index, site_data in enumerate(datas):
        for i, na in enumerate(nodes):
            data[site_index, i] = site_data[na]

    # Call a function that wraps a Cython function.
    edge_expectations = expectation_function(T, node_to_idx, site_weights,
            m, transq, data, root_distn, combinations)

    # Return a map from edge to expectation.
    edge_to_expectation = dict()
    for edge in T.edges():
        na, nb = edge
        eidx = node_to_idx[nb] - 1
        edge_to_expectation[edge] = edge_expectations[eidx]
    return edge_to_expectation


#TODO add unit tests
def expectation_function(
        T, node_to_idx, site_weights,
        m,
        transq,
        data,
        root_distn,
        combination,
        ):
    """
    Endpoint conditioned expectation linear combination.

    This function requires highly processed inputs and wraps a cython function.

    Parameters
    ----------
    T : x
        x
    node_to_idx : x
        x
    site_weights : x
        x
    m : x
        x
    transq : x
        x
    data : x
        x
    root_distn : x
        x
    combination : x
        x

    Returns
    -------
    x : x
        x

    """
    # Unpack some stuff.
    nsites, nnodes, nstates = data.shape

    # Check some shapes.
    assert_equal(transq.shape, (nnodes-1, nstates, nstates))
    assert_equal(combination.shape, (nnodes-1, nstates, nstates))

    # Allocate some empty arrays.
    # These could go into a separate memory management object.
    def _alloc(*args):
        return np.empty(args, dtype=float)
    transp = _alloc(nnodes-1, nstates, nstates)
    interact = _alloc(nnodes-1, nstates, nstates)
    expect_out = _alloc(nsites, nnodes-1)

    # Compute the probability transition matrix arrays
    # and the interaction matrix arrays.
    for edge in T.edges():
        na, nb = edge
        eidx = node_to_idx[nb] - 1
        transp[eidx], interact[eidx] = expm_frechet(
                transq[eidx], combination[eidx])

    # Compute the expectations using Cython.
    validation = 0
    conditional_expectation(
            m.indices, m.indptr,
            transp,
            transq,
            interact,
            data,
            root_distn,
            expect_out,
            validation,
            )

    # At this point we have the linear combination of conditional expectations
    # for each independent observation and each edge of the tree.
    
    # Return the per-edge expectations.
    edge_expectations = (expect_out * site_weights[:, None]).sum(axis=0)
    return edge_expectations

