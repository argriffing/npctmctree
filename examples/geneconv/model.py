"""
A small example based on the biological phenomenon of gene conversion.

This module has the model definition.

"""
from __future__ import division, print_function, absolute_import

from functools import partial
from itertools import product
import argparse

import numpy as np
import networkx as nx
import scipy.linalg
from numpy.testing import assert_allclose

import npmctree
from npmctree.sampling import sample_histories


__all__ = ['get_state_space', 'get_tree_info', 'get_pre_Q', 'get_Q_and_distn']


def get_state_space():
    nt_pairs = []
    pair_to_state = {}
    for i, pair in enumerate(product('ACGT', repeat=2)):
        nt_pairs.append(pair)
        pair_to_state[pair] = i
    return nt_pairs, pair_to_state


def get_tree_info(common_blen=0.1):
    T = nx.DiGraph()
    edge_to_blen = {}
    root = 'N1'
    triples = (
            ('N1', 'N0', common_blen),
            ('N1', 'N2', common_blen),
            ('N1', 'N5', common_blen),
            ('N2', 'N3', common_blen),
            ('N2', 'N4', common_blen))
    for va, vb, blen in triples:
        edge = (va, vb)
        T.add_edge(*edge)
        edge_to_blen[edge] = blen
    return T, root, edge_to_blen


def get_pre_Q(nt_pairs, phi):
    """
    Parameters
    ----------
    nt_pairs : sequence
        Ordered pairs of nucleotide states.
        The order of this sequence defines the order of rows and columns of Q.
    phi : float
        Under this model nucleotide substitutions are more likely to
        consist of changes that make paralogous sequences more similar
        to each other.  This parameter is the ratio capturing this effect.

    Returns
    -------
    pre_Q : numpy ndarray
        The rate matrix without normalization and without the diagonal.

    """
    slow = 1
    fast = phi
    n = len(nt_pairs)
    pre_Q = np.zeros((n, n), dtype=float)
    for i, (s0a, s1a) in enumerate(nt_pairs):
        for j, (s0b, s1b) in enumerate(nt_pairs):
            # Diagonal entries will be set later.
            if i == j:
                continue
            # Only one change is allowed at a time.
            if s0a != s0b and s1a != s1b:
                continue
            # Determine which paralog changes.
            if s0a != s0b:
                sa = s0a
                sb = s0b
                context = s1a
            if s1a != s1b:
                sa = s1a
                sb = s1b
                context = s0a
            # Set the rate according to the kind of change.
            if context == sb:
                rate = fast
            else:
                rate = slow
            pre_Q[i, j] = rate
    return pre_Q


def get_distn_clever(phi, nt_pairs):
    """
    Use a clever guess which can be checked later.

    """
    root_distn = []
    for nt0, nt1 in nt_pairs:
        root_distn.append(phi if nt0 == nt1 else 1)
    root_distn = np.array(root_distn) / sum(root_distn)
    return root_distn

def get_distn_brute(Q):
    """
    This method is slow for huge matrices;
    for huge matrices use something like the ARPACK-based methods
    like scipy.sparse.linalg with scipy.sparse rate matrices.

    """
    w, v = scipy.sparse.linalg.eigs(Q.T, k=1, which='SM')
    weights = v[:, 0].real
    distn = weights / weights.sum()
    return distn

def get_Q_and_distn(nt_pairs, phi):

    # Define the unnormalized rate matrix with negative diagonal.
    pre_Q = get_pre_Q(nt_pairs, phi)
    unnormalized_Q = pre_Q - np.diag(pre_Q.sum(axis=1))

    # Compute the stationary distribution in a couple of ways.
    root_distn = get_distn_clever(phi, nt_pairs)
    root_distn_brute = get_distn_brute(unnormalized_Q)
    assert_allclose(root_distn, root_distn_brute)

    # Check that the stationary distribution is correct.
    equilibrium_rates = np.dot(root_distn, unnormalized_Q)
    assert_allclose(equilibrium_rates, 0, atol=1e-12)

    # Normalize the rate matrix so that branch lengths
    # have the usual interpretation.
    expected_rate = np.dot(root_distn, -np.diag(unnormalized_Q))
    Q = unnormalized_Q / expected_rate

    return Q, root_distn
