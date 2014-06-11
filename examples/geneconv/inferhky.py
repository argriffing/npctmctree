"""
"""
from __future__ import division, print_function, absolute_import

import argparse
import functools
import itertools

import numpy as np
import networkx as nx
from numpy.testing import assert_equal
from scipy.linalg import expm
import scipy.optimize

import npmctree
from npmctree.dynamic_lmap_lhood import get_iid_lhoods

import npctmctree
from npctmctree.optimize import estimate_edge_rates
from npctmctree.expect import get_edge_to_expectation

from util import ad_hoc_fasta_reader
from model import (
        get_distn_brute,
        get_tree_info_with_outgroup,
        get_hky_pre_Q,
        get_combined_pre_Q,
        get_lockstep_pre_Q,
        get_pure_geneconv_pre_Q,
        )


# TODO out of date
def get_tree_info():
    T = nx.DiGraph()
    common_blen = 1.0
    edge_to_blen = {}
    root = 'N0'
    triples = (
            ('N0', 'Macaque', common_blen),
            ('N0', 'N1', common_blen),
            ('N1', 'Orangutan', common_blen),
            ('N1', 'N2', common_blen),
            ('N2', 'Chimpanzee', common_blen),
            ('N2', 'Gorilla', common_blen))
    for va, vb, blen in triples:
        edge = (va, vb)
        T.add_edge(*edge)
        edge_to_blen[edge] = blen
    return T, root, edge_to_blen


def get_edge_to_pre_R(T, root, kappa, nt_probs, tau):
    """

    Returns
    -------
    edge_to_pre_R : x
        x

    """
    # Compute the unscaled nucleotide pre-rate-matrix.
    pre_Q = get_hky_pre_Q(kappa, nt_probs)
    rates = pre_Q.sum(axis=1)
    scaled_pre_Q = pre_Q / np.dot(rates, nt_probs)

    # Compute the gene conversion pre-rate-matrix.
    # Define the diagonal entries of the gene conversion rate matrix.
    pre_R = get_combined_pre_Q(scaled_pre_Q, tau)

    # Do a similar thing for the pre-rate matrix.
    # This does not use the tau parameter.
    pre_S = get_lockstep_pre_Q(scaled_pre_Q)

    # Get the rate matrices on edges.
    # The terminal edge leading to the Tamarin outgroup will use S.
    edge_to_pre_R = {}
    for edge in T.edges():
        na, nb = edge
        if nb == 'Tamarin':
            edge_to_pre_R[edge] = pre_S
        else:
            edge_to_pre_R[edge] = pre_R

    return edge_to_pre_R


def get_edge_to_R(T, root, kappa, nt_probs, tau):
    """

    Returns
    -------
    edge_to_R : x
        x
    root_distn : x
        x

    """
    # Compute the unscaled nucleotide pre-rate-matrix.
    pre_Q = get_hky_pre_Q(kappa, nt_probs)
    rates = pre_Q.sum(axis=1)
    scaled_pre_Q = pre_Q / np.dot(rates, nt_probs)

    # Compute the gene conversion pre-rate-matrix.
    # Define the diagonal entries of the gene conversion rate matrix.
    pre_R = get_combined_pre_Q(scaled_pre_Q, tau)
    R = pre_R - np.diag(pre_R.sum(axis=1))

    # Do a similar thing for the pre-rate matrix.
    # This does not use the tau parameter.
    pre_S = get_lockstep_pre_Q(scaled_pre_Q)
    S = pre_S - np.diag(pre_S.sum(axis=1))
    #print('rate matrix S:')
    #print(S)

    # The distribution at the root will be the distribution of S.
    # It should be like nt_probs with some zeros.
    # Note that this distribution is invariant to the scale of the rate matrix.
    root_distn = get_distn_brute(S)
    #print('root distribution:')
    #print(root_distn)

    # Get the rate matrices on edges.
    # The terminal edge leading to the Tamarin outgroup will use S.
    edge_to_R = {}
    for edge in T.edges():
        na, nb = edge
        if nb == 'Tamarin':
            edge_to_R[edge] = S
        else:
            edge_to_R[edge] = R

    return edge_to_R, root_distn


def get_log_likelihood(T, root, data, edges, kappa, nt_probs, tau, edge_rates):
    """

    """
    #print('getting log likelihood:')
    #print('kappa:', kappa)
    #print('nt probs:', nt_probs)
    #print('tau:', tau)

    edge_to_R, root_distn = get_edge_to_R(T, root, kappa, nt_probs, tau)

    # Compute the transition probability matrix for each edge.
    edge_to_P = {}
    for edge_index, edge_rate in enumerate(edge_rates):
        edge = edges[edge_index]
        P = expm(edge_rate * edge_to_R[edge])
        edge_to_P[edge] = P

    # Get the likelihood at each site.
    lhoods = get_iid_lhoods(T, edge_to_P, root, root_distn, data)

    # Return the log likelihood.
    """
    print('search info parameters...')
    print('kappa:', kappa)
    print('nt probs:', nt_probs)
    print('tau:', tau)
    print('edge rates:', edge_rates)
    print('search info likelihoods...')
    print('lhoods:', lhoods)
    print()
    """
    ll = np.log(lhoods).sum()
    print('log likelihood:', ll)
    return ll


def objective(T, root, data, edges, log_params):
    """
    The objective is a penalized negative log likelihood.

    The penalty is related to violation of the simplex constraint
    on the mutational process nucleotide probabilities.

    """
    # transform the parameters
    params = np.exp(log_params)

    # unpack the parameters which are now forced to be positive
    kappa = params[0]
    nt_weights = params[1:5]
    tau = params[5]
    edge_rates = params[6:]

    #NOTE setting tau to zero
    tau = 0

    # normalize the nt probs and get the constraint violation penalty
    nt_sum = nt_weights.sum()
    nt_probs = nt_weights / nt_sum
    nt_penalty = np.square(np.log(nt_sum))

    # compute the log likelihood
    ll = get_log_likelihood(T, root, data, edges,
            kappa, nt_probs, tau, edge_rates)

    # return the penalized negative log likelihood
    #print(ll, nt_penalty)
    return -ll + nt_penalty


def main(args):
    # Read the hardcoded tree information.
    #T, root, edge_to_blen = get_tree_info()
    T, root = get_tree_info_with_outgroup()
    leaves = set(v for v, d in T.degree().items() if d == 1)
    outgroup = 'Tamarin'

    # Read the data as name sequence pairs.
    with open(args.fasta) as fin:
        name_seq_pairs = ad_hoc_fasta_reader(fin)
    name_to_seq = dict(name_seq_pairs)

    # Define a state space.
    nt_pairs = list(itertools.product('ACGT', repeat=2))
    pair_to_state = dict((p, i) for i, p in enumerate(nt_pairs))

    # Convert the (name, sequence) pairs to observed data
    # for the gene conversion stochastic process.
    suffixes = ('EDN', 'ECP')
    nsites = len(name_seq_pairs[0][1])
    print('number of sites:', nsites)
    constraints = []
    for site in range(nsites):
        node_to_lmap = {}
        for node in T:
            if node in leaves:
                lmap = np.zeros(len(nt_pairs), dtype=float)
                if node == outgroup:
                    # tamarin has only EDN
                    nt_pair = (
                            name_to_seq[node + suffixes[0]][site],
                            name_to_seq[node + suffixes[0]][site])
                else:
                    # non-outgroup leaves have both EDN and ECP
                    nt_pair = (
                            name_to_seq[node + suffixes[0]][site],
                            name_to_seq[node + suffixes[1]][site])
                state = pair_to_state[nt_pair]
                lmap[state] = 1.0
            else:
                lmap = np.ones(len(nt_pairs), dtype=float)
            node_to_lmap[node] = lmap
        constraints.append(node_to_lmap)

    # Make some initial parameter value guesses.
    edges = list(T.edges())
    kappa = 2.0
    nt_probs = [0.30, 0.25, 0.20, 0.25]
    tau = 2.0
    edge_rates = [0.1] * len(edges)

    # Pack the initial parameter guesses.
    x0 = [kappa] + nt_probs + [tau] + edge_rates
    logx0 = np.log(x0)

    # Define the objective function to minimize.
    f = functools.partial(objective, T, root, constraints, edges)

    # Use a black box search.
    res = scipy.optimize.minimize(f, logx0, method='L-BFGS-B')

    # Report the raw search output.
    print('raw search output:')
    print(res)
    print()

    # Transform the results of the search.
    logxopt = res.x
    xopt = np.exp(logxopt)

    # Unpack.
    kappa = xopt[0]
    nt_weights = xopt[1:5]
    tau = xopt[5]

    #NOTE setting tau to zero
    tau = 0

    edge_rates = xopt[6:]
    nt_probs = nt_weights / nt_weights.sum()
    print('max likelihood parameter estimates...')
    print('kappa:', kappa)
    print('nt probs:', nt_probs)
    print('tau:', tau)
    print('edge rates:')
    for i, edge in enumerate(edges):
        print('edge:', edge, 'rate:', edge_rates[i])
    print()

    # Let's do some post-processing to re-estimate
    # branch-specific rates using accelerated EM.
    edge_to_R, root_distn = get_edge_to_R(T, root, kappa, nt_probs, tau)

    # Use the relatively sophisticated optimizer.
    print('updating edge rates with the sophisticated search...')
    data_weight_pairs = [(x, 1) for x in constraints]
    edge_to_rate, neg_ll = estimate_edge_rates(
            T, root, edge_to_R, root_distn, data_weight_pairs)
    print('estimated edge rates:', edge_to_rate)
    print('corresponding neg log likelihood:', neg_ll)
    print()

    # Compute posterior expected gene conversion event counts.
    edge_to_combination = {}
    for edge in T.edges():
        na, nb = edge
        if nb == 'Tamarin':
            edge_to_combination[edge] = np.zeros((4*4, 4*4), dtype=float)
        else:
            geneconv_rate = edge_to_rate[edge] * tau
            pre_G = get_pure_geneconv_pre_Q(4, geneconv_rate)
            edge_to_combination[edge] = pre_G
    print('computing gene conversion event expectations on edges...')
    edge_to_expectation = get_edge_to_expectation(
            T, root, edge_to_R, edge_to_combination,
            root_distn, data_weight_pairs)
    for edge in edges:
        x = edge_to_expectation[edge]
        print('edge:', edge, 'geneconv event expectation:', x)
    print()

    # Compute posterior expected transition event counts.
    edge_to_combination = get_edge_to_pre_R(T, root, kappa, nt_probs, tau)
    for edge in edge_to_combination:
        edge_to_combination[edge] = edge_to_rate[edge] * (
                edge_to_combination[edge])
    print('computing total transition event expectations on edges...')
    edge_to_expectation = get_edge_to_expectation(
            T, root, edge_to_R, edge_to_combination,
            root_distn, data_weight_pairs)
    for edge in edges:
        x = edge_to_expectation[edge]
        print('edge:', edge, 'total event expectation:', x)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True,
            help='fasta file with paralog alignment of EDN and ECP')
    main(parser.parse_args())

