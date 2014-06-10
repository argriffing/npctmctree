"""
"""
from __future__ import division, print_function, absolute_import

from functools import partial
from itertools import product
from collections import defaultdict

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose, assert_
from scipy.linalg import inv, eigvalsh, expm, expm_frechet
import scipy.optimize

import npmctree
from npmctree.puzzles import sample_distn1d
from npmctree.dynamic_fset_lhood import get_lhood, get_edge_to_distn2d
#from npmctree.cy_dynamic_lmap_lhood import get_lhood, get_edge_to_distn2d

import npctmctree
from npctmctree.cyem import expectation_step
from npctmctree.derivatives import (
        LikelihoodShapeStorage, get_log_likelihood_info)


def get_tree_info():
    """
    Define an arbitrary hardcoded tree structure.

    Also define arbitrary edge-specific scaling factors.

    """
    T = nx.DiGraph()
    edge_to_rate = {}
    root = 'N1'
    leaves = ('N0', 'N5', 'N3', 'N4')
    internal_nodes = ('N1', 'N2')
    triples = (
            ('N1', 'N0', 0.1),
            ('N1', 'N2', 0.2),
            ('N1', 'N5', 0.3),
            ('N2', 'N3', 0.4),
            ('N2', 'N4', 0.5),
            )
    for va, vb, rate in triples:
        edge = (va, vb)
        T.add_edge(*edge)
        edge_to_rate[edge] = rate
    return T, root, edge_to_rate, leaves, internal_nodes


def help_get_lhood_diff_xx(T, root,
        root_distn1d, node_to_data_fvec1d, edge_to_Q,
        edge_to_rate, special_edge):
    edge_to_P = {}
    for edge in T.edges():
        edge_rate = edge_to_rate[edge]
        edge_Q = edge_to_Q[edge]
        Q = edge_rate * edge_Q
        if edge == special_edge:
            P = np.dot(np.dot(edge_Q, edge_Q), expm(Q))
        else:
            P = expm(Q)
        edge_to_P[edge] = P
    lhood = get_lhood(T, edge_to_P, root, root_distn1d,
            node_to_data_fvec1d)
    return lhood


def help_get_lhood_diff_xy(T, root,
        root_distn1d, node_to_data_fvec1d, edge_to_Q,
        edge_to_rate, edge_x, edge_y):
    if edge_x == edge_y:
        return help_get_ll_diff_xx(T, root,
                root_distn1d, node_to_data_fvec1d, edge_to_Q,
                edge_to_rate, edge_x)
    edge_to_P = {}
    for edge in T.edges():
        edge_rate = edge_to_rate[edge]
        edge_Q = edge_to_Q[edge]
        Q = edge_rate * edge_Q
        if edge in (edge_x, edge_y):
            P = np.dot(edge_Q, expm(Q))
        else:
            P = expm(Q)
        edge_to_P[edge] = P
    lhood = get_lhood(T, edge_to_P, root, root_distn1d,
            node_to_data_fvec1d)
    return lhood


def help_get_lhood_diff(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q,
        edge_to_rate, special_edge):
    edge_to_P = {}
    for edge in T.edges():
        edge_rate = edge_to_rate[edge]
        edge_Q = edge_to_Q[edge]
        Q = edge_rate * edge_Q
        if edge == special_edge:
            P = np.dot(edge_Q, expm(Q))
        else:
            P = expm(Q)
        edge_to_P[edge] = P
    lhood = get_lhood(T, edge_to_P, root, root_distn1d,
            node_to_data_fvec1d)
    return lhood


def help_get_lhood(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q,
        edge_to_rate):
    edge_to_P = {}
    for edge in T.edges():
        edge_rate = edge_to_rate[edge]
        edge_Q = edge_to_Q[edge]
        P = expm(edge_rate * edge_Q)
        edge_to_P[edge] = P
    lhood = get_lhood(T, edge_to_P, root, root_distn1d,
            node_to_data_fvec1d)
    return lhood


def help_get_ll(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q,
        edge_to_rate):
    lhood = help_get_lhood(T, root,
            root_distn1d, node_to_data_fvec1d, edge_to_Q, edge_to_rate)
    return np.log(lhood)


def new_edges(edge_to_rate, edge_incr_pairs):
    d = edge_to_rate.copy()
    for edge, incr in edge_incr_pairs:
        d[edge] = d[edge] + incr
    return d


def help_get_iid_info(T, root, root_distn1d, edge_to_Q,
        nstates, leaves, internal_nodes,
        data_weight_pairs, edge_to_scale,
        degree=0, use_log_scale=False):
    """
    """
    # unpack a bit
    nleaves = len(leaves)
    n = nstates

    # Define a toposort node ordering and a corresponding csr matrix.
    nodes = nx.topological_sort(T, [root])
    node_to_idx = dict((na, i) for i, na in enumerate(nodes))
    m = nx.to_scipy_sparse_matrix(T, nodes)

    # Stack the transition rate matrices into a single array.
    nnodes = len(nodes)
    nstates = root_distn1d.shape[0]
    n = nstates
    transq = np.empty((nnodes-1, nstates, nstates), dtype=float)
    for (na, nb), Q in edge_to_Q.items():
        edge_idx = node_to_idx[nb] - 1
        transq[edge_idx] = Q

    # Allocate a transition probability matrix array.
    transp_ws = np.empty_like(transq)
    transp_mod_ws = np.empty_like(transq)

    # Stack the data into a single array,
    # and construct an array of site weights.
    nsites = len(data_weight_pairs)
    datas, weights = zip(*data_weight_pairs)
    site_weights = np.array(weights, dtype=float)
    data = np.empty((nsites, nnodes, nstates), dtype=float)
    for site_index, site_data in enumerate(datas):
        for i, na in enumerate(nodes):
            data[site_index, i] = site_data[na]

    # Initialize the per-edge rate matrix scaling factor guesses.
    scaling_guesses = np.empty(nnodes-1, dtype=float)
    for (na, nb), rate in edge_to_scale.items():
        eidx = node_to_idx[nb] - 1
        scaling_guesses[eidx] = rate

    # preallocate some memory
    # compute iid log likelihood and derivatives
    mem = LikelihoodShapeStorage(nsites, nnodes, nstates, degree)
    return get_log_likelihood_info(
            T, node_to_idx, site_weights, m,
            transq, data, root_distn1d, mem, scaling_guesses,
            degree=degree, use_log_scale=use_log_scale)


def check_iid_info(T, root, root_distn1d, edge_to_Q,
        nstates, leaves, internal_nodes):
    """
    Log likelihood and its gradient and hessian, for iid observations.

    """
    # unpack a bit
    nleaves = len(leaves)
    n = nstates

    data_weight_pairs = []
    nsites = 2
    for i in range(nsites):

        # Sample a random assignment.
        assignment = np.random.randint(0, nstates, size=nleaves)

        # Get the map from leaf to state.
        leaf_to_state = dict(zip(leaves, assignment))

        # Define the data associated with this assignment.
        # All leaf states are fully observed.
        # All internal states are completely unobserved.
        node_to_data_fvec1d = {}
        for node in leaves:
            state = leaf_to_state[node]
            fvec1d = np.zeros(n, dtype=bool)
            fvec1d[state] = True
            node_to_data_fvec1d[node] = fvec1d
        for node in internal_nodes:
            fvec1d = np.ones(n, dtype=bool)
            node_to_data_fvec1d[node] = fvec1d

        # Add the data object into the array.
        pair = (node_to_data_fvec1d, 1)
        data_weight_pairs.append(pair)

    # Try to guess the edge-specific scaling factors using EM,
    # starting with an initial guess that is wrong.
    guess_edge_to_rate = {}
    for edge in T.edges():
        guess_edge_to_rate[edge] = np.random.random()

    degree = 2
    for use_log_scale in (False, True):
        print('use_log_scale:', use_log_scale)
        print()

        if use_log_scale:
            edge_to_scale = dict(
                    (e, np.log(r)) for e, r in guess_edge_to_rate.items())
        else:
            edge_to_scale = guess_edge_to_rate.copy()

        # Compute some likelihood surface info for the iid samples
        f, g, h = help_get_iid_info(T, root, root_distn1d, edge_to_Q,
                nstates, leaves, internal_nodes,
                data_weight_pairs, edge_to_scale,
                degree=degree, use_log_scale=use_log_scale)


        print('iid info:')
        print(f)
        print(g)
        print(h)
        print('eigvalsh(h):', eigvalsh(h))
        print('inv(h):')
        print(inv(h))
        print()

        # check finite differences results
        eps = 1e-5
        fn = partial(help_get_iid_info,
                T, root, root_distn1d, edge_to_Q,
                nstates, leaves, internal_nodes,
                data_weight_pairs)
        edges = list(T.edges())
        edge_x = edges[0]
        edge_y = edges[1]

        print('iid finite central differences first derivative:')
        d = new_edges(edge_to_scale, [(edge_x, -eps)])
        lla = fn(d, use_log_scale=use_log_scale)
        d = new_edges(edge_to_scale, [(edge_x, +eps)])
        llb = fn(d, use_log_scale=use_log_scale)
        print(lla, llb)
        print((llb - lla) / (2 * eps))
        print()

        print('finite central differences second derivative single edge:')
        d = edge_to_scale.copy()
        llb = fn(d, use_log_scale=use_log_scale)
        d = new_edges(edge_to_scale, [(edge_x, -eps)])
        lla = fn(d, use_log_scale=use_log_scale)
        d = new_edges(edge_to_scale, [(edge_x, +eps)])
        llc = fn(d, use_log_scale=use_log_scale)
        print(lla, llb, llc)
        print((llc - 2*llb + lla) / (eps * eps))
        print()

        # Approximation of mixed derivatives.
        print('finite central differences second derivative two edges:')
        d = new_edges(edge_to_scale, [(edge_x, -eps), (edge_y, -eps)])
        ll00 = fn(d, use_log_scale=use_log_scale)
        d = new_edges(edge_to_scale, [(edge_x, +eps), (edge_y, -eps)])
        ll10 = fn(d, use_log_scale=use_log_scale)
        d = new_edges(edge_to_scale, [(edge_x, -eps), (edge_y, +eps)])
        ll01 = fn(d, use_log_scale=use_log_scale)
        d = new_edges(edge_to_scale, [(edge_x, +eps), (edge_y, +eps)])
        ll11 = fn(d, use_log_scale=use_log_scale)
        print(ll00, ll01, ll10, ll11)
        print((ll11 - ll10 - ll01 + ll00) / (4 * eps * eps))
        print()


def main():
    np.random.seed(12345)

    # Define the size of the state space
    # which will be constant across the whole tree.
    nstates = 4
    n = nstates

    # Sample a random root distribution as a 1d numpy array.
    pzero = 0
    root_distn1d = sample_distn1d(n, pzero)

    # Hardcode a tree with four leaves
    # and some arbitrary hardcoded rate scaling factors per edge.
    T, root, edge_to_rate, leaves, internal_nodes = get_tree_info()
    nleaves = len(leaves)

    # Sample a random rate matrix for each edge.
    # These rate matrices have no redeeming qualities --
    # they are not scaled in any particular way,
    # they are not symmetric, they are not time-reversible,
    # and their equilibrium distributions are unrelated
    # to the root distribution.
    edge_to_Q = {}
    for edge in T.edges():
        M = np.exp(np.random.randn(n, n))
        Q = M - np.diag(M.sum(axis=1))
        edge_to_Q[edge] = Q

    # Sample a random assignment.
    assignment = np.random.randint(0, nstates, size=nleaves)

    # Get an edge of interest to check the effects of perturbing its scale.
    edge_of_interest = list(T.edges())[1]

    # Get the map from leaf to state.
    leaf_to_state = dict(zip(leaves, assignment))

    # Define the data associated with this assignment.
    # All leaf states are fully observed.
    # All internal states are completely unobserved.
    node_to_data_fvec1d = {}
    for node in leaves:
        state = leaf_to_state[node]
        fvec1d = np.zeros(n, dtype=bool)
        fvec1d[state] = True
        node_to_data_fvec1d[node] = fvec1d
    for node in internal_nodes:
        fvec1d = np.ones(n, dtype=bool)
        node_to_data_fvec1d[node] = fvec1d

    eps = 1e-5
    #eps = 1e-8
    #eps = 1e-10

    print('finite central differences first derivative:')
    d = edge_to_rate.copy()
    d[edge_of_interest] -= eps
    lla = help_get_ll(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q, d)
    d = edge_to_rate.copy()
    d[edge_of_interest] += eps
    llb = help_get_ll(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q, d)
    print(lla, llb)
    print((llb - lla) / (2 * eps))
    print()

    print('analytical first derivative:')
    ldiff = help_get_lhood_diff(T, root,
            root_distn1d, node_to_data_fvec1d, edge_to_Q,
            edge_to_rate, edge_of_interest)
    print(ldiff)
    print(ldiff / np.exp(lla))
    print()

    print('finite central differences second derivative single edge:')
    d = edge_to_rate.copy()
    llb = help_get_ll(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q, d)
    d = edge_to_rate.copy()
    d[edge_of_interest] -= eps
    lla = help_get_ll(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q, d)
    d = edge_to_rate.copy()
    d[edge_of_interest] += eps
    llc = help_get_ll(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q, d)
    print(lla, llb, llc)
    print((llc - 2*llb + lla) / (eps * eps))
    print()

    print('analytical second derivative single edge:')
    l = help_get_lhood(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q,
            edge_to_rate)
    l_x = help_get_lhood_diff(T, root,
            root_distn1d, node_to_data_fvec1d, edge_to_Q,
            edge_to_rate, edge_of_interest)
    l_xx = help_get_lhood_diff_xx(T, root,
            root_distn1d, node_to_data_fvec1d, edge_to_Q,
            edge_to_rate, edge_of_interest)
    print(l, l_x, l_xx)
    print(l_xx / l - (l_x*l_x) / (l*l))
    print()

    # specify two edges of interest
    edges = list(T.edges())
    edge_x = edges[1]
    edge_y = edges[2]

    # Approximation of mixed derivatives.
    print('finite central differences second derivative two edges:')
    d = edge_to_rate.copy()
    d[edge_x] -= eps
    d[edge_y] -= eps
    ll00 = help_get_ll(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q, d)
    d = edge_to_rate.copy()
    d[edge_x] += eps
    d[edge_y] -= eps
    ll10 = help_get_ll(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q, d)
    d = edge_to_rate.copy()
    d[edge_x] -= eps
    d[edge_y] += eps
    ll01 = help_get_ll(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q, d)
    d = edge_to_rate.copy()
    d[edge_x] += eps
    d[edge_y] += eps
    ll11 = help_get_ll(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q, d)
    print(ll00, ll01, ll10, ll11)
    print((ll11 - ll10 - ll01 + ll00) / (4 * eps * eps))
    print()

    print('analytical second derivative two edges:')
    l = help_get_lhood(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q,
            edge_to_rate)
    l_x = help_get_lhood_diff(T, root,
            root_distn1d, node_to_data_fvec1d, edge_to_Q,
            edge_to_rate, edge_x)
    l_y = help_get_lhood_diff(T, root,
            root_distn1d, node_to_data_fvec1d, edge_to_Q,
            edge_to_rate, edge_y)
    l_xy = help_get_lhood_diff_xy(T, root,
            root_distn1d, node_to_data_fvec1d, edge_to_Q,
            edge_to_rate, edge_x, edge_y)
    print(l, l_x, l_y, l_xy)
    print(l_xy / l - (l_x * l_y) / (l * l))
    print()

    # Check log likelihood info for i.i.d. samples.
    check_iid_info(T, root, root_distn1d, edge_to_Q,
            nstates, leaves, internal_nodes)


if __name__ == '__main__':
    main()

