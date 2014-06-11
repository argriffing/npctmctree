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

from model import get_distn_brute, get_combined_pre_Q


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


def get_hky_pre_Q(kappa, nt_probs):
    """
    This is just hky.

    """
    n = 4
    transitions = ((0, 3), (3, 0), (1, 2), (2, 1))
    pre_Q = np.zeros((n, n), dtype=float)
    for sa, pa in enumerate(nt_probs):
        for sb, pb in enumerate(nt_probs):
            if sa == sb:
                continue
            rate = 1.0
            rate *= pb
            if (sa, sb) in transitions:
                rate *= kappa
            pre_Q[sa, sb] = rate
    return pre_Q


def ad_hoc_fasta_reader(fin):
    name_seq_pairs = []
    while True:

        # read the name
        line = fin.readline().strip()
        if not line:
            return name_seq_pairs
        assert_equal(line[0], '>')
        name = line[1:].strip()

        # read the single line sequence
        line = fin.readline().strip()
        seq = line
        unrecognized = set(line) - set('ACGT')
        if unrecognized:
            raise Exception('unrecognized nucleotides: ' + str(unrecognized))

        name_seq_pairs.append((name, seq))


def get_log_likelihood(T, root, data, kappa, nt_probs, tau):
    """

    """
    print('getting log likelihood:')
    print('kappa:', kappa)
    print('nt probs:', nt_probs)
    print('tau:', tau)
    print()

    # Compute the unscaled nucleotide pre-rate-matrix.
    pre_Q = get_hky_pre_Q(kappa, nt_probs)
    rates = pre_Q.sum(axis=1)
    scaled_pre_Q = pre_Q / np.dot(rates, nt_probs)

    # Compute the gene conversion pre-rate-matrix.
    pre_R = get_combined_pre_Q(scaled_pre_Q, tau)

    # Define the diagonal entries of the gene conversion rate matrix.
    R = pre_R - np.diag(pre_R.sum(axis=1))

    # Compute the equilibrium distribution.
    # Note that this distribution is invariant to the scale of the rate matrix.
    root_distn = get_distn_brute(R)
    print('root distribution:')
    print(root_distn)

    # Compute the transition probability matrix for each edge.
    #TODO remove this debugging thing
    edge_to_R = dict([
        (('N0', 'Macaque'), 0.1 * R),
        (('N0', 'N1'), 0.2 * R),
        (('N1', 'Orangutan'), 0.3 * R),
        (('N1', 'N2'), 0.4 * R),
        (('N2', 'Chimpanzee'), 0.5 * R),
        (('N2', 'Gorilla'), 0.6 * R),
        ])
    #edge_to_R = dict((e, R) for e in T.edges())

    # Get the likelihood at each site.
    #lhoods = get_iid_lhoods(T, edge_to_P, root, root_distn, data)

    # Let's do some post-processing to re-estimate
    # branch-specific rates using accelerated EM.
    #pre_Q = get_hky_pre_Q(kappa, nt_probs)
    #pre_R = get_combined_pre_Q(pre_Q, tau)
    #R = pre_R - np.diag(pre_R.sum(axis=1))
    #root_distn = get_distn_brute(R)
    #guess_edge_to_rate = dict((edge, 0.1) for edge in edges)
    data_prob_pairs = [(x, 1) for x in data]
    #edge_to_Q = dict((edge, R) for edge in edges)

    # Use the relatively sophisticated optimizer.
    #print('updating edge rates with the sophisticated search...')
    print('estimating edge rates...')
    edge_to_rate, neg_ll = estimate_edge_rates(
            T, root, edge_to_R, root_distn, data_prob_pairs)
    print('estimated edge rates:', edge_to_rate)
    print('corresponding neg log likelihood:', neg_ll)
    print()

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
    #return np.log(lhoods).sum()
    return neg_ll


def objective(T, root, data, log_params):
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

    # normalize the nt probs and get the constraint violation penalty
    nt_sum = nt_weights.sum()
    nt_probs = nt_weights / nt_sum
    nt_penalty = np.square(np.log(nt_sum))

    # compute the log likelihood
    neg_ll = get_log_likelihood(T, root, data, kappa, nt_probs, tau)

    # return the penalized negative log likelihood
    #print(ll, nt_penalty)
    return neg_ll + nt_penalty


def main(args):
    #print(get_hky_pre_Q(1e-5, [0.1, 0.2, 0.3, 0.4]))
    #return

    # Read the hardcoded tree information.
    T, root, edge_to_blen = get_tree_info()

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
    taxa = ('Gorilla', 'Macaque', 'Chimpanzee', 'Orangutan')
    nsites = len(name_seq_pairs[0][0])
    constraints = []
    for site in range(nsites):
        node_to_lmap = {}
        for node in T:
            if node in taxa:
                lmap = np.zeros(len(nt_pairs), dtype=float)
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
    #kappa = 2.0
    #nt_probs = [0.25] * 4
    #tau = 0.1
    kappa = 1.2
    tau = 0.5
    nt_probs = np.array([0.1, 0.2, 0.3, 0.4])

    # Pack the initial parameter guesses.
    x0 = np.concatenate([[kappa], nt_probs, [tau]])
    logx0 = np.log(x0)

    # Define the objective function to minimize.
    f = functools.partial(objective, T, root, constraints)

    # Report something about the initial guess.
    print('initial guess:')
    print(logx0)
    print('initial exp log guess:')
    print(np.exp(logx0))
    print('objective value of initial guess:')
    print(f(logx0))
    print()

    # Use a black box search.
    res = scipy.optimize.minimize(f, logx0,
            #method='L-BFGS-B')
            method='Nelder-Mead')

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
    nt_probs = nt_weights / nt_weights.sum()
    print('max likelihood parameter estimates...')
    print('kappa:', kappa)
    print('nt probs:', nt_probs)
    print('tau:', tau)
    #print('edge rates:')
    #for i, edge in enumerate(edges):
        #print('edge:', edge, 'rate:', edge_rates[i])
    print()

    # Let's do some post-processing to re-estimate
    # branch-specific rates using accelerated EM.
    pre_Q = get_hky_pre_Q(kappa, nt_probs)
    pre_R = get_combined_pre_Q(pre_Q, tau)
    R = pre_R - np.diag(pre_R.sum(axis=1))
    root_distn = get_distn_brute(R)
    guess_edge_to_rate = dict((edge, 0.1) for edge in edges)
    data_prob_pairs = [(x, 1) for x in constraints]
    edge_to_Q = dict((edge, R) for edge in edges)

    # Use the relatively sophisticated optimizer.
    print('updating edge rates with the sophisticated search...')
    edge_to_rate, neg_ll = estimate_edge_rates(
            T, root, edge_to_Q, root_distn, data_prob_pairs)
    print('estimated edge rates:', edge_to_rate)
    print('corresponding neg log likelihood:', neg_ll)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True,
            help='fasta file with paralog alignment of EDN and ECP')
    main(parser.parse_args())
