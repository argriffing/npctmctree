"""
"""
from __future__ import division, print_function, absolute_import

import functools
import itertools

import numpy as np
import networkx as nx
from numpy.testing import assert_equal
from scipy.linalg import expm, expm_frechet
import scipy.optimize

import npmctree
from npmctree.dynamic_lmap_lhood import get_iid_lhoods, get_lhood

import npctmctree
from npctmctree.cyem import expectation_step

from model import get_distn_brute


def do_cythonized_em(T, root,
        edge_to_Q, root_distn1d,
        data_prob_pairs, guess_edge_to_rate):
    """
    Try the Cython implementation.
    def expectation_step(
            idx_t[:] csr_indices, # (nnodes-1,)
            idx_t[:] csr_indptr, # (nnodes+1,)
            cnp.float_t[:, :, :] transp, # (nnodes-1, nstates, nstates)
            cnp.float_t[:, :, :] transq, # (nnodes-1, nstates, nstates)
            cnp.float_t[:, :, :] interact_trans, # (nnodes-1, nstates, nstates)
            cnp.float_t[:, :, :] interact_dwell, # (nnodes-1, nstates, nstates)
            cnp.float_t[:, :, :] data, # (nsites, nnodes, nstates)
            cnp.float_t[:] root_distn, # (nstates,)
            cnp.float_t[:, :] trans_out, # (nsites, nnodes-1)
            cnp.float_t[:, :] dwell_out, # (nsites, nnodes-1)
            int validation=1,
            ):

    """
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

    # Allocate a transition probability matrix array
    # and some interaction matrix arrays.
    transp = np.empty_like(transq)
    interact_trans = np.empty_like(transq)
    interact_dwell = np.empty_like(transq)

    # Stack the data into a single array,
    # and construct an array of site weights.
    nsites = len(data_prob_pairs)
    datas, probs = zip(*data_prob_pairs)
    site_weights = np.array(probs, dtype=float)
    data = np.empty((nsites, nnodes, nstates), dtype=float)
    for site_index, site_data in enumerate(datas):
        for i, na in enumerate(nodes):
            data[site_index, i] = site_data[na]

    # Initialize expectation arrays.
    trans_out = np.empty((nsites, nnodes-1), dtype=float)
    dwell_out = np.empty((nsites, nnodes-1), dtype=float)

    # Initialize the per-edge rate matrix scaling factor guesses.
    scaling_guesses = np.empty(nnodes-1, dtype=float)
    scaling_ratios = np.ones(nnodes-1, dtype=float)
    for (na, nb), rate in guess_edge_to_rate.items():
        eidx = node_to_idx[nb] - 1
        scaling_guesses[eidx] = rate

    # Pre-scale the rate matrix.
    transq *= scaling_guesses[:, None, None]

    # Do the EM iterations.
    nsteps = 1000
    for em_iteration_index in range(nsteps):

        # Scale the rate matrices according to the edge ratios.
        transq *= scaling_ratios[:, None, None]

        # Compute the probability transition matrix arrays
        # and the interaction matrix arrays.
        trans_indicator = np.ones((n, n)) - np.identity(n)
        dwell_indicator = np.identity(n)
        for edge in T.edges():
            na, nb = edge
            eidx = node_to_idx[nb] - 1
            Q = transq[eidx]
            #print(edge, 'Q:')
            #print(Q)
            transp[eidx] = expm(Q)
            interact_trans[eidx] = expm_frechet(
                    Q, Q * trans_indicator, compute_expm=False)
            interact_dwell[eidx] = expm_frechet(
                    Q, Q * dwell_indicator, compute_expm=False)

        # Compute the expectations.
        validation = 1
        expectation_step(
                m.indices, m.indptr,
                transp, transq,
                interact_trans, interact_dwell,
                data,
                root_distn1d,
                trans_out, dwell_out,
                validation,
                )

        # Compute the per-edge ratios.
        trans_sum = (trans_out * site_weights[:, None]).sum(axis=0)
        dwell_sum = (dwell_out * site_weights[:, None]).sum(axis=0)
        scaling_ratios = trans_sum / -dwell_sum
        scaling_guesses *= scaling_ratios

        # Report the guesses.
        if not (em_iteration_index+1) % 100:
            print(em_iteration_index+1)
            for edge in T.edges():
                na, nb = edge
                eidx = node_to_idx[nb] - 1
                print(edge, scaling_guesses[eidx])
            print()


def em_objective_for_broyden(*args):
    """
    Recast EM as a root finding problem.
    
    This approach is inspired by method Q1 of the following paper.
    Acceleration of the EM Algorithm by Using Quasi-Newton Methods
    Mortaza Jamshidian and Robert I. Jennrich
    1997

    """
    scale = args[-1]
    return em_objective_for_aitken(*args) - scale


def em_objective_for_aitken(
        T, node_to_idx, site_weights,
        m,
        transq_unscaled, transp,
        interact_trans, interact_dwell,
        data,
        root_distn1d,
        trans_out, dwell_out,
        scale,
        ):
    """
    Recast EM as a fixed-point problem.
    
    This approach is inspired by the introduction of the following paper.
    A QUASI-NEWTON ACCELERATION OF THE EM ALGORITHM
    Kenneth Lange
    1995

    """
    # Unpack some stuff.
    nsites, nnodes, nstates = data.shape
    n = nstates

    # Scale the rate matrices according to the edge ratios.
    transq = transq_unscaled * scale[:, None, None]

    # Compute the probability transition matrix arrays
    # and the interaction matrix arrays.
    trans_indicator = np.ones((n, n)) - np.identity(n)
    dwell_indicator = np.identity(n)
    for edge in T.edges():
        na, nb = edge
        eidx = node_to_idx[nb] - 1
        Q = transq[eidx]
        transp[eidx] = expm(Q)
        interact_trans[eidx] = expm_frechet(
                Q, Q * trans_indicator, compute_expm=False)
        interact_dwell[eidx] = expm_frechet(
                Q, Q * dwell_indicator, compute_expm=False)

    # Compute the expectations.
    validation = 0
    expectation_step(
            m.indices, m.indptr,
            transp, transq,
            interact_trans, interact_dwell,
            data,
            root_distn1d,
            trans_out, dwell_out,
            validation,
            )

    # Compute the per-edge ratios.
    trans_sum = (trans_out * site_weights[:, None]).sum(axis=0)
    dwell_sum = (dwell_out * site_weights[:, None]).sum(axis=0)
    scaling_ratios = trans_sum / -dwell_sum

    # Return the new scaling factors.
    return scale * scaling_ratios


def do_cythonized_accelerated_em(T, root,
        edge_to_Q, root_distn1d,
        data_prob_pairs, guess_edge_to_rate):
    """
    """
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

    # Allocate a transition probability matrix array
    # and some interaction matrix arrays.
    transp = np.empty_like(transq)
    interact_trans = np.empty_like(transq)
    interact_dwell = np.empty_like(transq)

    # Stack the data into a single array,
    # and construct an array of site weights.
    nsites = len(data_prob_pairs)
    datas, probs = zip(*data_prob_pairs)
    site_weights = np.array(probs, dtype=float)
    data = np.empty((nsites, nnodes, nstates), dtype=float)
    for site_index, site_data in enumerate(datas):
        for i, na in enumerate(nodes):
            data[site_index, i] = site_data[na]

    # Initialize expectation arrays.
    trans_out = np.empty((nsites, nnodes-1), dtype=float)
    dwell_out = np.empty((nsites, nnodes-1), dtype=float)

    # Initialize the per-edge rate matrix scaling factor guesses.
    scaling_guesses = np.empty(nnodes-1, dtype=float)
    for (na, nb), rate in guess_edge_to_rate.items():
        eidx = node_to_idx[nb] - 1
        scaling_guesses[eidx] = rate

    # Define the fixed-point function and the initial guess.
    f_aitken = functools.partial(em_objective_for_aitken,
            T, node_to_idx, site_weights,
            m,
            transq, transp,
            interact_trans, interact_dwell,
            data,
            root_distn1d,
            trans_out, dwell_out)
    f_broyden = functools.partial(em_objective_for_broyden,
            T, node_to_idx, site_weights,
            m,
            transq, transp,
            interact_trans, interact_dwell,
            data,
            root_distn1d,
            trans_out, dwell_out)
    x0 = scaling_guesses
    
    # Do a few unaccelerated EM iterations.
    for i in range(5):
        x0 = f_aitken(x0)

    # Use the fixed point optimization to accelerate the EM.
    #result = scipy.optimize.fixed_point(f_aitken, x0, maxiter=10000)

    # Use a root search to accelerate the EM.
    result = scipy.optimize.root(f_broyden, x0)

    # Look at the results of the accelerated EM search.
    print(result)


def hamming_distance(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def get_state_space():
    nt_pairs = []
    pair_to_state = {}
    for i, pair in enumerate(list(product('ACGT', repeat=2))):
        nt_pairs.append(pair)
        pair_to_state[pair] = i
    return nt_pairs, pair_to_state


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


def get_combined_pre_Q(pre_Q, tau):
    """
    This is for gene conversion.

    Parameters
    ----------
    pre_Q : 2d ndarray
        unnormalized pre-rate-matrix
    tau : float
        non-negative additive gene conversion rate parameter

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


def get_log_likelihood(T, root, data, edges,
        kappa, nt_probs, tau, edge_rates):
    """

    """
    # Compute the unscaled nucleotide pre-rate-matrix.
    pre_Q = get_hky_pre_Q(kappa, nt_probs)

    # Compute the gene conversion pre-rate-matrix.
    pre_R = get_combined_pre_Q(pre_Q, tau)

    # Define the diagonal entries of the gene conversion rate matrix.
    R = pre_R - np.diag(pre_R.sum(axis=1))

    # Compute the equilibrium distribution.
    # Note that this distribution is invariant to the scale of the rate matrix.
    root_distn = get_distn_brute(R)
    #print('root distn:', root_distn)

    # Compute the transition probability matrix for each edge.
    edge_to_P = {}
    for edge_index, edge_rate in enumerate(edge_rates):
        edge = edges[edge_index]
        P = expm(edge_rate * R)
        #print('row sums of P:')
        #print(P).sum(axis=1)
        edge_to_P[edge] = P

    # Get the likelihood at each site.
    lhoods = get_iid_lhoods(T, edge_to_P, root, root_distn, data)
    #lhoods = []
    #for d in data:
        #lhood = get_lhood(T, edge_to_P, root, root_distn, d)
        #lhoods.append(lhood)

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
    return np.log(lhoods).sum()


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


def main():

    # Read the hardcoded tree information.
    T, root, edge_to_blen = get_tree_info()

    # Read the data as name sequence pairs.
    with open('simdata.fasta') as fin:
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
    edges = list(T.edges())
    kappa = 2.0
    nt_probs = [0.25] * 4
    tau = 0.1
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
    pre_Q = get_hky_pre_Q(kappa, nt_probs)
    pre_R = get_combined_pre_Q(pre_Q, tau)
    R = pre_R - np.diag(pre_R.sum(axis=1))
    root_distn = get_distn_brute(R)
    guess_edge_to_rate = dict((edge, 0.1) for edge in edges)
    data_prob_pairs = [(x, 1) for x in constraints]
    edge_to_Q = dict((edge, R) for edge in edges)
    do_cythonized_em(T, root,
            edge_to_Q, root_distn,
            data_prob_pairs, guess_edge_to_rate)


if __name__ == '__main__':
    main()

