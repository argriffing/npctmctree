"""
This is an simple parameterized rate matrix for tests and examples.

In the future, packing and unpacking should be handled elsewhere.

"""
from __future__ import division, print_function, absolute_import

from itertools import permutations

import networkx as nx
import numpy as np


def get_nx_Q(kappa, nt_distn):
    Q = nx.DiGraph()
    transitions = ({'A', 'G'}, {'C', 'T'})
    states = set(nt_distn)
    for sa, sb in permutations(states, 2):
        rate = nt_distn[sb]
        if {sa, sb} in transitions:
            rate *= kappa
        Q.add_edge(sa, sb, weight=rate)
    state_to_rate = Q.out_degree(weight='weight')
    expected_rate = sum(nt_distn[s] * state_to_rate[s] for s in states)
    for sa in Q:
        for sb in Q[sa]:
            Q[sa][sb]['weight'] /= expected_rate
    return Q


def get_pre_Q(kappa, nt_distn1d):
    """
    State order is ACGT.

    Parameters
    ----------
    kappa : float
        The rate scaling ratio of transitions to transversions.
        Nucleotide substitutions A <--> G and C <--> T are called transitions,
        while all other nucleotide substitutions are called transversions.
    nt_distn1d : 1d ndarray of floats
        Mutational nucleotide distribution.

    Returns
    -------
    pre_Q : float 2d ndarray with shape (4, 4).
        A pre-rate-matrix with zeros on the diagonal.

    References
    ----------
    .. [1] Masami Hasegawa and HIrohisa Kishino and Taka-aki Yano,
       Dating of the human-ape splitting by a molecular clock
       of mitochondrial DNA. Journal of Molecular Evolution,
       October 1985, Volume 22, Issue 2, pages 160--174,
       doi:10.1007/BF02101694, PMID 3934395.

    """
    k = kappa
    a, c, g, t = nt_distn1d
    return np.array([
        [  0,   c, k*g,   t],
        [  a,   0,   g, k*t],
        [k*a,   c,   0,   t],
        [  a, k*c,   g,   0]])


def get_normalized_Q(kappa, nt_distn1d):
    """
    HKY rate matrix normalized to have expected rate 1.

    See get_pre_Q for more details.

    """
    pre_Q = get_pre_Q(kappa, nt_distn1d)
    rates_out = pre_Q.sum(axis=1)
    expected_rate = nt_distn1d.dot(rates_out)
    Q = (pre_Q - np.diag(rates_out)) / expected_rate
    return Q


#NOTE from nxctmctree
def pack_params(edge_rates, nt_distn1d, kappa):
    """
    Pack parameters into a 1d ndarray.

    Parameters
    ----------
    edge_rates : 1d ndarray dtype float
        Edge-specific rates, using a fixed edge order.
        This function does not care about the identity of the edges;
        it packs the edge rates into the output array
        in the same order as the input order.
    nt_distn1d : 1d ndarray dtype float
        Mutational nucleotide probabilities, which should sum to 1.
        It is not technically required that the probabilities sum to 1
        because a penalty is applied if the constraint is violated.
    kappa : float
        A parameter controlling transition vs. transversion rate.
        This rate is interpreted mutationally (vs. selectionally).

    Returns
    -------
    log_params : 1d ndarray dtype float
        Parameter vector for the purposes of numerical optimization.

    """
    params = np.concatenate([edge_rates, nt_distn1d, [kappa]])
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
    Q = get_normalized_Q(kappa, nt_distn1d)

    # Return the unpacked parameters.
    return edge_rates, Q, nt_distn1d, kappa, penalty

