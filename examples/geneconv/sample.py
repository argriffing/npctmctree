"""
Sample alignments from the hky+geneconv model.

See also the sample.py script in npmctree.

"""
from __future__ import division, print_function, absolute_import

import argparse
import itertools

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm

import npmctree
from npmctree.sampling import sample_histories

from model import get_distn_brute, get_combined_pre_Q


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


def main(args):

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
    #kappa = 1
    #nt_probs = np.array([0.25] * 4)
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
    pre_Q = get_hky_pre_Q(kappa, nt_probs)
    rates = pre_Q.sum(axis=1)
    expected_rate = np.dot(rates, nt_probs)
    Q = (pre_Q - np.diag(rates)) / expected_rate

    # Check that Q is time-reversible.
    DQ = np.dot(np.diag(nt_probs), Q)
    assert_allclose(DQ, DQ.T)
    
    # Construct the gene conversion rate matrix.
    scaled_pre_Q = pre_Q / expected_rate
    pre_R = get_combined_pre_Q(scaled_pre_Q, tau)
    R = pre_R - np.diag(pre_R.sum(axis=1))

    # Define the root distribution.
    root_distn = get_distn_brute(R)

    # Check that the gene conversion rate matrix is not time-reversible.
    DR = np.dot(np.diag(root_distn), R)
    if np.allclose(DR, DR.T):
        raise Exception('the gene conversion rate matrix '
                'is unexpectedly time-reversible')

    # Map each edge to a transition probability matrix.
    edge_to_P = {}
    for edge, rate in edge_to_rate.items():
        edge_to_P[edge] = expm(rate * R)

    # Do not impose any data constraints.
    node_to_data_lmap = {}
    for node in T:
        node_to_data_lmap[node] = np.ones_like(root_distn)

    # Sample histories.
    leaves = set(v for v, d in T.degree().items() if d == 1)
    nhistories = args.nsites
    leaf_to_state_seq = dict((v, []) for v in leaves)
    for node_to_state in sample_histories(T, edge_to_P, root,
            root_distn, node_to_data_lmap, nhistories):
        for leaf in leaves:
            state = node_to_state[leaf]
            leaf_to_state_seq[leaf].append(state)

    # Write the fasta file with paralog pairs.
    expansion = list(itertools.product(range(4), repeat=2))
    for leaf, state_seq in leaf_to_state_seq.items():
        pairs = [expansion[s] for s in state_seq]
        EDN, ECP = zip(*pairs)
        for name, seq in ('EDN', EDN), ('ECP', ECP):
            ntseq = ''.join('ACGT'[i] for i in seq)
            print('>' + leaf + name)
            print(ntseq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsites', type=int, default=10,
            help='sample an alignment consisting of this many nucleotide sites')
    main(parser.parse_args())

