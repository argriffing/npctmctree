"""
Generic modeling functions related to the gene conversion model.

Find the equilibrium distribution,
and build a gene conversion rate matrix on top of an existing rate matrix.

"""
from __future__ import division, print_function, absolute_import

import itertools

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.sparse.linalg import eigs


def hamming_distance(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def get_tree_info_with_outgroup():
    T = nx.DiGraph()
    root = 'N0'
    T.add_edges_from([
            ('N0', 'Tamarin'),
            ('N0', 'N1'),
            ('N1', 'Macaque'),
            ('N1', 'N2'),
            ('N2', 'Orangutan'),
            ('N2', 'N3'),
            ('N3', 'Chimpanzee'),
            ('N3', 'Gorilla'),
            ])
    return T, root


def get_distn_brute(Q):
    """
    This method is slow for huge matrices;
    for huge matrices use something like the ARPACK-based methods
    like scipy.sparse.linalg with scipy.sparse rate matrices.

    """
    w, v = eigs(Q.T, k=1, which='SM')
    assert_allclose(np.linalg.norm(v), 1)
    assert_allclose(w, 0, atol=1e-10)
    weights = v[:, 0].real
    distn = weights / weights.sum()
    return distn


def get_nt_geneconv_state_space():
    nt_pairs = list(itertools.product('ACGT', repeat=2))
    pair_to_state = dict((pair, i) for i, pair in enumerate(nt_pairs))
    return nt_pairs, pair_to_state


def get_lockstep_pre_Q(pre_Q):
    """
    For making an artificially large state space.

    """
    n = pre_Q.shape[0]
    assert_equal(pre_Q.shape, (n, n))
    pre_R = np.zeros((n*n, n*n), dtype=float)
    nt_pairs = list(itertools.product(range(n), repeat=2))
    for i, sa in enumerate(nt_pairs):
        for j, sb in enumerate(nt_pairs):
            if sa == sb:
                continue
            sa0, sa1 = sa
            sb0, sb1 = sb
            if sa0 == sa1 and sb0 == sb1:
                # meaningful transitions
                pre_R[i, j] = pre_Q[sa0, sb0]
            elif sa0 != sa1 and sb0 == sb1:
                # dummy transitions that will never happen
                pre_R[i, j] = 1
    return pre_R


def get_combined_pre_Q(pre_Q, tau):
    """
    This is for gene conversion.

    States of pre_Q input are like {0, 1, 2, ..., n}.
    States of pre_R output are like {0, 1, 2, ..., n*n}.

    Parameters
    ----------
    pre_Q : 2d ndarray
        Unnormalized pre-rate-matrix.
    tau : float
        Non-negative additive gene conversion rate parameter.
        It has units like the expected number of gene conversion events
        per point mutation event.

    Returns
    -------
    pre_R : 2d ndarray
        Unnormalized pre-rate-matrix.
        If the pre_Q input matrix has shape (n, n) then the pre_R output matrix
        will have shape (n*n, n*n).

    """
    n = pre_Q.shape[0]
    assert_equal(pre_Q.shape, (n, n))
    pre_R = np.zeros((n*n, n*n), dtype=float)
    nt_pairs = list(itertools.product(range(n), repeat=2))
    for i, sa in enumerate(nt_pairs):
        for j, sb in enumerate(nt_pairs):
            if hamming_distance(sa, sb) != 1:
                continue
            sa0, sa1 = sa
            sb0, sb1 = sb
            rate = 0
            if sa0 != sb0:
                # rate contribution of point mutation from sa0
                rate += pre_Q[sa0, sb0]
                if sa1 == sb0:
                    # rate contribution of gene conversion from sa1
                    rate += tau
            if sa1 != sb1:
                # rate contribution of point mutation from sa1
                rate += pre_Q[sa1, sb1]
                if sa0 == sb1:
                    # rate contribution of gene conversion from sa0
                    rate += tau
            pre_R[i, j] = rate
    return pre_R


def get_pure_geneconv_pre_Q(n, tau):
    pre_R = np.zeros((n*n, n*n), dtype=float)
    nt_pairs = list(itertools.product(range(n), repeat=2))
    for i, sa in enumerate(nt_pairs):
        for j, sb in enumerate(nt_pairs):
            if hamming_distance(sa, sb) != 1:
                continue
            sa0, sa1 = sa
            sb0, sb1 = sb
            rate = 0
            if sa0 != sb0:
                # rate contribution of point mutation from sa0
                #rate += pre_Q[sa0, sb0]
                if sa1 == sb0:
                    # rate contribution of gene conversion from sa1
                    rate += tau
            if sa1 != sb1:
                # rate contribution of point mutation from sa1
                #rate += pre_Q[sa1, sb1]
                if sa0 == sb1:
                    # rate contribution of gene conversion from sa0
                    rate += tau
            pre_R[i, j] = rate
    return pre_R


def get_pure_mutation_pre_Q(pre_Q):
    n = pre_Q.shape[0]
    assert_equal(pre_Q.shape, (n, n))
    pre_R = np.zeros((n*n, n*n), dtype=float)
    nt_pairs = list(itertools.product(range(n), repeat=2))
    for i, sa in enumerate(nt_pairs):
        for j, sb in enumerate(nt_pairs):
            if hamming_distance(sa, sb) != 1:
                continue
            sa0, sa1 = sa
            sb0, sb1 = sb
            rate = 0
            if sa0 != sb0:
                # rate contribution of point mutation from sa0
                rate += pre_Q[sa0, sb0]
                #if sa1 == sb0:
                    # rate contribution of gene conversion from sa1
                    #rate += tau
            if sa1 != sb1:
                # rate contribution of point mutation from sa1
                rate += pre_Q[sa1, sb1]
                #if sa0 == sb1:
                    # rate contribution of gene conversion from sa0
                    #rate += tau
            pre_R[i, j] = rate
    return pre_R
