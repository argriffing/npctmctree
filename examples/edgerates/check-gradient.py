"""
"""
from __future__ import division, print_function, absolute_import

from functools import partial
from itertools import product
from collections import defaultdict

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose, assert_
from scipy.linalg import expm, expm_frechet
import scipy.optimize

import npmctree
from npmctree.puzzles import sample_distn1d
from npmctree.dynamic_fset_lhood import get_lhood, get_edge_to_distn2d
#from npmctree.cy_dynamic_lmap_lhood import get_lhood, get_edge_to_distn2d

import npctmctree
from npctmctree.cyem import expectation_step


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


def get_ll_diff(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q,
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
    #return np.log(lhood)

def get_ll(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q,
        edge_to_rate):
    edge_to_P = {}
    for edge in T.edges():
        edge_rate = edge_to_rate[edge]
        edge_Q = edge_to_Q[edge]
        P = expm(edge_rate * edge_Q)
        edge_to_P[edge] = P
    lhood = get_lhood(T, edge_to_P, root, root_distn1d,
            node_to_data_fvec1d)
    return np.log(lhood)


def main():
    np.random.seed(1234)

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

    eps = 1e-8
    d = edge_to_rate.copy()
    d[edge_of_interest] -= eps
    lla = get_ll(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q, d)
    d = edge_to_rate.copy()
    d[edge_of_interest] += eps
    llb = get_ll(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q, d)
    print(lla, llb)
    print((llb - lla) / (2 * eps))

    ldiff = get_ll_diff(T, root, root_distn1d, node_to_data_fvec1d, edge_to_Q,
            edge_to_rate, edge_of_interest)
    print(ldiff)
    print(ldiff / np.exp(lla))


if __name__ == '__main__':
    main()

