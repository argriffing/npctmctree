"""
"""
from __future__ import division, print_function, absolute_import

from itertools import product
from collections import defaultdict

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose, assert_
from scipy.linalg import expm, expm_frechet

import npmctree
from npmctree.puzzles import sample_distn1d
from npmctree.dynamic_fset_lhood import get_lhood, get_edge_to_distn2d


def get_tree_info():
    """
    Define an arbitrary hardcoded tree structure.

    Also define arbitrary edge-specific scaling factors.

    """
    T = nx.DiGraph()
    edge_to_rate = {}
    root = 'N1'
    triples = (
            ('N1', 'N0', 0.1),
            ('N1', 'N2', 0.2),
            ('N1', 'N5', 0.3),
            ('N2', 'N3', 0.4),
            ('N2', 'N4', 0.5))
    for va, vb, rate in triples:
        edge = (va, vb)
        T.add_edge(*edge)
        edge_to_rate[edge] = rate
    return T, root, edge_to_rate


def main():
    np.random.seed(1234)

    # Define the size of the state space
    # which will be constant across the whole tree.
    n = 4

    # Sample a random root distribution as a 1d numpy array.
    pzero = 0
    root_distn1d = sample_distn1d(n, pzero)

    # Hardcode a tree with four leaves
    # and some arbitrary hardcoded rate scaling factors per edge.
    T, root, edge_to_rate = get_tree_info()

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

    # Convert edge-specific transition rate matrices
    # to edge-specific transition probability matrices.
    edge_to_P = {}
    for edge in T.edges():
        edge_rate = edge_to_rate[edge]
        edge_Q = edge_to_Q[edge]
        P = expm(edge_rate * edge_Q)
        edge_to_P[edge] = P

    # Instead of sampling states at the leaves,
    # find the exact joint distribution of leaf states.
    leaves = ('N0', 'N5', 'N3', 'N4')
    internal_nodes = ('N1', 'N2')
    states = range(n)
    # Initialize the distribution over leaf data (yes this is confusing).
    data_prob_pairs = []
    for assignment in product(states, repeat=len(leaves)):

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

        # Compute the likelihood for this data.
        lhood = get_lhood(T, edge_to_P, root, root_distn1d, node_to_data_fvec1d)
        data_prob_pairs.append((node_to_data_fvec1d, lhood))

    # Check that the computed joint distribution over leaf states
    # is actually a distribution.
    datas, probs = zip(*data_prob_pairs)
    assert_allclose(sum(probs), 1)

    # Try to guess the edge-specific scaling factors using EM,
    # starting with an initial guess that is wrong.
    guess_edge_to_rate = {}
    for edge in T.edges():
        guess_edge_to_rate[edge] = 0.2

    # Do the EM iterations.
    while True:
        
        # Compute the scaled edge-specific transition rate matrices.
        edge_to_scaled_Q = {}
        for edge in T.edges():
            rate = guess_edge_to_rate[edge]
            Q = edge_to_Q[edge]
            edge_to_scaled_Q[edge] = rate * Q

        # Compute the edge-specific transition probability matrices.
        edge_to_P = {}
        for edge in T.edges():
            scaled_Q = edge_to_scaled_Q[edge]
            P = expm(scaled_Q)
            edge_to_P[edge] = P

        # For each edge, compute the interaction matrices
        # corresponding to all transition counts and dwell times.
        trans_indicator = np.ones((n, n)) - np.identity(n)
        dwell_indicator = np.identity(n)
        edge_to_interact_trans = {}
        edge_to_interact_dwell = {}
        for edge in T.edges():

            # extract edge-specific transition matrices
            Q = edge_to_scaled_Q[edge]
            P = edge_to_P[edge]

            # compute the transition interaction matrix
            interact = expm_frechet(Q, Q * trans_indicator, compute_expm=False)
            edge_to_interact_trans[edge] = interact

            # compute the dwell interaction matrix
            interact = expm_frechet(Q, Q * dwell_indicator, compute_expm=False)
            edge_to_interact_dwell[edge] = interact

        # Initialize edge-specific summaries.
        edge_to_trans_expectation = defaultdict(float)
        edge_to_dwell_expectation = defaultdict(float)

        # Compute the edge-specific summaries
        # conditional on the current edge-specific rate guesses
        # and on the leaf state distribution computed
        # from the true edge-specific rates.
        for node_to_data_fvec1d, lhood in data_prob_pairs:

            # Compute the joint endpoint state distribution for each edge.
            edge_to_J = get_edge_to_distn2d(
                    T, edge_to_P, root, root_distn1d, node_to_data_fvec1d)
            
            # For each edge, compute the transition expectation contribution
            # and compute the dwell expectation contribution.
            # These will be scaled by the data likelihood.
            for edge in T.edges():

                # extract some edge-specific matrices
                P = edge_to_P[edge]
                J = edge_to_J[edge]

                # transition contribution
                interact = edge_to_interact_trans[edge]
                total = 0
                for i in range(n):
                    for j in range(n):
                        if J[i, j]:
                            total += J[i, j] * interact[i, j] / P[i, j]
                edge_to_trans_expectation[edge] += lhood * total

                # dwell contribution
                interact = edge_to_interact_dwell[edge]
                total = 0
                for i in range(n):
                    for j in range(n):
                        if J[i, j]:
                            total += J[i, j] * interact[i, j] / P[i, j]
                edge_to_dwell_expectation[edge] += lhood * total

        # According to EM, update each edge-specific rate guess
        # using a ratio of transition and dwell expectations.
        for edge in T.edges():
            trans = edge_to_trans_expectation[edge]
            dwell = edge_to_dwell_expectation[edge]
            ratio = trans / -dwell
            old_rate = guess_edge_to_rate[edge] 
            new_rate = old_rate * ratio
            guess_edge_to_rate[edge] = new_rate
            print(edge, trans, dwell, ratio, old_rate, new_rate)


if __name__ == '__main__':
    main()

