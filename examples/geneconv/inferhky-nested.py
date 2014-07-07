"""
Estimate parameters for the gene conversion project.

This is the result of the Nelder-Mead on top of L-BFGS-B search
for the sample that is 1000 nucleotides long.

raw search output:
status: 0
nfev: 341
success: True
fun: 9375.3454959203027
x: array([ 0.00906516, -2.46924945, -1.72693729, -1.29795235, -0.97236314,
-0.78168797])
message: 'Optimization terminated successfully.'
nit: 210

This is the result polishing the above results
using a Nelder-Mead on top of trust-ncg search:

raw search output:
status: 0
nfev: 281
success: True
fun: 9375.337236070578
x: array([ 0.00882672, -2.37894293, -1.63666853, -1.20775805, -0.88213141,
-0.78150991])
message: 'Optimization terminated successfully.'
nit: 173

max likelihood parameter estimates...
kappa: 1.00886579031
nt probs: [ 0.09264455  0.19461915  0.29885397  0.41388233]
tau: 0.457714379511


"""
from __future__ import division, print_function, absolute_import

import argparse
import functools
import itertools

import numpy as np
import networkx as nx
from numpy.testing import assert_equal, assert_allclose
from scipy.linalg import expm
import scipy.optimize

import npmctree
from npmctree.dynamic_fset_lhood import get_lhood, get_edge_to_distn2d
from npmctree.dynamic_lmap_lhood import get_iid_lhoods

import npctmctree
from npctmctree.optimize import estimate_edge_rates
from npctmctree import hkymodel

from util import ad_hoc_fasta_reader
from model import (
        get_distn_brute,
        get_tree_info_with_outgroup,
        get_combined_pre_Q,
        get_lockstep_pre_Q,
        )


#TODO move this somewhere else
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


#TODO move this somewhere else
def get_exact_data():
    # Define a tree structure with edge-specific rates.
    T = nx.DiGraph()
    root = 'N0'
    edge_to_rate = {}
    triples = (
        ('N0', 'Macaque', 0.1),
        ('N0', 'N1', 0.2),
        ('N1', 'Orangutan', 0.3),
        ('N1', 'N2', 0.4),
        ('N2', 'Chimpanzee', 0.5),
        ('N2', 'Gorilla', 0.6),
        )
    for na, nb, rate in triples:
        T.add_edge(na, nb)
        edge_to_rate[na, nb] = rate

    # Define some hky parameter values.
    kappa = 1.2
    nt_probs = np.array([0.1, 0.2, 0.3, 0.4])
    assert_allclose(nt_probs.sum(), 1)

    # Define a gene conversion rate parameter.
    # Because of the scaling of the hky rate matrix,
    # this parameter will have units of something like
    # "expected number of gene conversion events per point mutation event."
    tau = 0.5

    # Construct the hky rate matrix.
    # Scale it to have expected rate 1.0.
    pre_Q = hkymodel.get_pre_Q(kappa, nt_probs)
    rates = pre_Q.sum(axis=1)
    expected_rate = np.dot(rates, nt_probs)
    Q = (pre_Q - np.diag(rates)) / expected_rate
    
    # Construct the gene conversion rate matrix.
    scaled_pre_Q = pre_Q / expected_rate
    pre_R = get_combined_pre_Q(scaled_pre_Q, tau)
    R = pre_R - np.diag(pre_R.sum(axis=1))

    # Map each edge to a transition probability matrix.
    edge_to_P = {}
    for edge, rate in edge_to_rate.items():
        edge_to_P[edge] = expm(rate * R)

    # Define the root distribution.
    root_distn = get_distn_brute(R)

    # NOTE This is where we compute a distribution instead of sampling
    leaves = set(v for v, d in T.degree().items() if d == 1)
    internal_nodes = set(T) - leaves

    # Instead of sampling states at the leaves,
    # find the exact joint distribution of leaf states.
    nstates = root_distn.shape[0]
    states = range(nstates)
    # Initialize the distribution over leaf data (yes this is confusing).
    data_prob_pairs = []
    assignments = list(itertools.product(states, repeat=len(leaves)))
    for i, assignment in enumerate(assignments):
        print(i+1, 'of', len(assignments))

        # Get the map from leaf to state.
        leaf_to_state = dict(zip(leaves, assignment))

        # Define the data associated with this assignment.
        # All leaf states are fully observed.
        # All internal states are completely unobserved.
        node_to_data_fvec1d = {}
        for node in leaves:
            state = leaf_to_state[node]
            fvec1d = np.zeros(nstates, dtype=bool)
            fvec1d[state] = True
            node_to_data_fvec1d[node] = fvec1d
        for node in internal_nodes:
            fvec1d = np.ones(nstates, dtype=bool)
            node_to_data_fvec1d[node] = fvec1d

        # Compute the likelihood for this data.
        lhood = get_lhood(T, edge_to_P, root, root_distn, node_to_data_fvec1d)
        data_prob_pairs.append((node_to_data_fvec1d, lhood))

    # Check that the computed joint distribution over leaf states
    # is actually a distribution.
    datas, probs = zip(*data_prob_pairs)
    assert_allclose(sum(probs), 1)

    return data_prob_pairs


def get_log_likelihood(T, root, data_weight_pairs, kappa, nt_probs, tau,
        rate_hint_object, log_params):
    """

    """
    print('getting log likelihood:')
    print('kappa:', kappa)
    print('nt probs:', nt_probs)
    print('tau:', tau)

    # Compute the unscaled nucleotide pre-rate-matrix.
    pre_Q = hkymodel.get_pre_Q(kappa, nt_probs)
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
    print('root distribution:')
    print(root_distn)

    # Get the rate matrices on edges.
    # The terminal edge leading to the Tamarin outgroup will use S.
    edge_to_R = {}
    for edge in T.edges():
        na, nb = edge
        if nb == 'Tamarin':
            edge_to_R[edge] = S
        else:
            edge_to_R[edge] = R

    # Compute the transition probability matrix for each edge.
    #edge_to_R = dict([
        #(('N0', 'Macaque'), 0.1 * R),
        #(('N0', 'N1'), 0.2 * R),
        #(('N1', 'Orangutan'), 0.3 * R),
        #(('N1', 'N2'), 0.4 * R),
        #(('N2', 'Chimpanzee'), 0.5 * R),
        #(('N2', 'Gorilla'), 0.6 * R),
        #])
    # Get a hint if possible.
    """
    hint = rate_hint_object.get_hint(log_params)
    if hint is None:
        print('no edge length scaling factor hint')
        edge_to_R = dict((e, R) for e in T.edges())
    else:
        print('using edge length scaling factor hint:')
        print(hint)
        edge_to_R = {}
        for edge in T.edges():
            edge_to_R[edge] = hint[edge] * R
    """

    # Get the likelihood at each site.
    #lhoods = get_iid_lhoods(T, edge_to_P, root, root_distn, data)

    # Let's do some post-processing to re-estimate
    # branch-specific rates using accelerated EM.
    #pre_Q = hkymodel.get_pre_Q(kappa, nt_probs)
    #pre_R = get_combined_pre_Q(pre_Q, tau)
    #R = pre_R - np.diag(pre_R.sum(axis=1))
    #root_distn = get_distn_brute(R)
    #guess_edge_to_rate = dict((edge, 0.1) for edge in edges)
    #data_prob_pairs = [(x, 1) for x in data]
    #edge_to_Q = dict((edge, R) for edge in edges)

    # Use the relatively sophisticated optimizer.
    #print('updating edge rates with the sophisticated search...')
    print('estimating edge rates...')
    edge_to_rate, neg_ll = estimate_edge_rates(
            T, root, edge_to_R, root_distn, data_weight_pairs,
            #method='trust-ncg')
            method='L-BFGS-B')
    print('estimated edge rates:', edge_to_rate)
    print('corresponding neg log likelihood:', neg_ll)
    print()

    """
    # Set the hint.
    if hint is None:
        next_hint = edge_to_rate
    else:
        next_hint = hint.copy()
        for edge, hint_rate in hint.items():
            next_hint[edge] = hint_rate * edge_to_rate[edge]
    rate_hint_object.set_hint(log_params, next_hint)
    """

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


class RateHint(object):
    def __init__(self):
        # The first element of the pair is the log_params.
        # The second element of the pair is a map from edge to rate.
        self.cached_pairs = []

    def get_hint(self, log_params):
        if not self.cached_pairs:
            return None
        best_dist = None
        best_hint = None
        for x, hint in self.cached_pairs:
            dist = np.linalg.norm(log_params - x)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_hint = hint
        return best_hint.copy()

    def set_hint(self, log_params, edge_to_rate):
        pair = (log_params.copy(), edge_to_rate.copy())
        self.cached_pairs.append(pair)


def objective(T, root, data_weight_pairs, rate_hint_object, log_params):
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
    neg_ll = get_log_likelihood(T, root,
            data_weight_pairs, kappa, nt_probs, tau,
            rate_hint_object, log_params)

    # return the penalized negative log likelihood
    #print(ll, nt_penalty)
    return neg_ll + nt_penalty


def main(args):
    #print(hkymodel.get_pre_Q(1e-5, [0.1, 0.2, 0.3, 0.4]))
    #return

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

    # NOTE use exact data vs. use the data on the command line
    #data_weight_pairs = get_exact_data()
    data_weight_pairs = [(c, 1.0) for c in constraints]

    # Make some initial parameter value guesses.
    kappa = 2.0
    nt_probs = [0.25] * 4
    tau = 2.0
    #kappa = 1.2
    #tau = 0.5
    #nt_probs = np.array([0.1, 0.2, 0.3, 0.4])

    # Pack the initial parameter guesses.
    x0 = np.concatenate([[kappa], nt_probs, [tau]])
    logx0 = np.log(x0)

    # NOTE start with a close guess computed using L-BFGS-B
    #logx0 = np.array([
        #0.00906516,
        #-2.46924945, -1.72693729, -1.29795235, -0.97236314,
        #-0.78168797])


    # Initialize the rate hint object.
    rate_hint_object = RateHint()

    # Define the objective function to minimize.
    f = functools.partial(objective,
            T, root, data_weight_pairs, rate_hint_object)

    # Report something about the initial guess.
    print('initial guess:')
    print(logx0)
    print('initial exp log guess:')
    print(np.exp(logx0))
    print('objective value of initial guess:')
    print(f(logx0))
    print()

    # Use a black box search.
    res = scipy.optimize.minimize(f, logx0, method='Nelder-Mead')
    #res = scipy.optimize.basinhopping(f, logx0, T=30, stepsize=1e-3)
    #nparams = 6
    #bounds = [np.log([0.1, 3]) for i in range(6)]
    #res = scipy.optimize.differential_evolution(f, bounds)

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
    pre_Q = hkymodel.get_pre_Q(kappa, nt_probs)
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

