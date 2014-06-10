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
from npmctree.dynamic_lmap_lhood import get_iid_lhoods
from npmctree.cyfels import iid_likelihoods

import npctmctree
from npctmctree.cyem import expectation_step

import npctmctree
from npctmctree.linesearch import jj97_qn2
from npctmctree.derivatives import (
        LikelihoodShapeStorage, get_log_likelihood_info)
from npctmctree.em import EMStorage, em_function
from npctmctree.squarem import fixed_point_squarem
from npctmctree.optimize import estimate_edge_rates


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


def get_ll_gradient(*args):
    # this can be used for grad in jj97 search
    neg_ll_f, neg_ll_g = objective_and_gradient_for_search(*args)
    return -neg_ll_g
    #return neg_ll_g


def get_em_displacement(*args):
    # this can be used for em in jj97 search (???)
    displacement = em_objective_for_broyden(*args)
    return displacement


def get_logscale_fgh(
        T, node_to_idx, site_weights, m,
        transq_unscaled,
        data,
        root_distn1d,
        mem,
        degree,
        log_scale,
        ):
    # return includes hessian and can deal with log scale rates
    return get_log_likelihood_info(
            T, node_to_idx, site_weights, m,
            transq_unscaled,
            data, root_distn1d, mem, log_scale,
            degree=degree, use_log_scale=True)


def objective_and_gradient_for_search(
        T, node_to_idx, site_weights, m,
        transq_unscaled, transp, transp_grad,
        data,
        root_distn1d,
        scale,
        ):
    """

    """
    # Unpack some stuff.
    nsites, nnodes, nstates = data.shape
    n = nstates

    # Scale the rate matrices according to the edge ratios.
    transq = transq_unscaled * scale[:, None, None]

    # Compute the probability transition matrix arrays.
    for edge in T.edges():
        na, nb = edge
        eidx = node_to_idx[nb] - 1
        Q = transq[eidx]
        transp[eidx] = expm(Q)

    # Compute the site likelihoods.
    validation = 1
    likelihoods = np.empty(nsites, dtype=float)
    iid_likelihoods(
            m.indices, m.indptr,
            transp,
            data,
            root_distn1d,
            likelihoods,
            validation,
            )

    # Compute the sum of log likelihoods.
    # This is the negative of the objective function.
    ll_total = np.log(likelihoods).dot(site_weights)

    # For each edge,
    # Adjust the transition probability matrix to compute the gradient.
    lhood_gradients = np.empty((nnodes-1, nsites), dtype=float)
    for edge in T.edges():
        na, nb = edge
        eidx = node_to_idx[nb] - 1
        transp_grad[...] = transp
        transp_grad[eidx] = np.dot(transq_unscaled[eidx], transp[eidx])
        iid_likelihoods(
                m.indices, m.indptr,
                transp_grad,
                data,
                root_distn1d,
                lhood_gradients[eidx],
                validation,
                )

    # Compute the log likelihood gradient.
    ll_gradient = np.dot(lhood_gradients / likelihoods[None, :], site_weights)

    # Return the objective function and its gradient.
    return -ll_total, -ll_gradient


def do_jj97_qn2_search(T, root,
        edge_to_rate, edge_to_Q, root_distn1d,
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

    # Allocate a transition probability matrix array.
    transp = np.empty_like(transq)
    transp_grad = np.empty_like(transp)

    # Stack the data into a single array,
    # and construct an array of site weights.
    nsites = len(data_prob_pairs)
    datas, probs = zip(*data_prob_pairs)
    site_weights = np.array(probs, dtype=float)
    data = np.empty((nsites, nnodes, nstates), dtype=float)
    for site_index, site_data in enumerate(datas):
        for i, na in enumerate(nodes):
            data[site_index, i] = site_data[na]

    # Initialize the per-edge rate matrix scaling factor guesses.
    scaling_guesses = np.empty(nnodes-1, dtype=float)
    for (na, nb), rate in guess_edge_to_rate.items():
        eidx = node_to_idx[nb] - 1
        scaling_guesses[eidx] = rate

    # Initialize guesses and bounds.
    x0 = scaling_guesses
    bounds = [(0, None)]*(nnodes-1)

    # Initialize gradient function.
    grad = partial(get_ll_gradient,
            T, node_to_idx, site_weights, m,
            transq, transp, transp_grad,
            data,
            root_distn1d,
            )

    # Initialize temporary arrays for EM.
    interact_trans = np.empty_like(transq)
    interact_dwell = np.empty_like(transq)
    trans_out = np.empty((nsites, nnodes-1), dtype=float)
    dwell_out = np.empty((nsites, nnodes-1), dtype=float)

    # Initialize EM function.
    em = partial(get_em_displacement,
            T, node_to_idx, site_weights,
            m,
            transq, transp,
            interact_trans, interact_dwell,
            data,
            root_distn1d,
            trans_out, dwell_out)

    # Do a few EM rounds.
    for i in range(5):
        x0 += em(x0)

    out = jj97_qn2(x0, grad, em, bounds)
    print(out)


def do_a_few_EM_iterations(f, x0, niter, verbose=True):
    # Do a few EM rounds.
    if verbose:
        print('doing a few EM rounds...')
        print(x0)
    x = x0
    for i in range(niter):
        x = f(x)
        print(x)
    return x


def do_hessian_search(T, root,
        edge_to_rate, edge_to_Q, root_distn1d,
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

    # Allocate a transition probability matrix array.
    transp = np.empty_like(transq)
    transp_grad = np.empty_like(transp)

    # Stack the data into a single array,
    # and construct an array of site weights.
    nsites = len(data_prob_pairs)
    datas, probs = zip(*data_prob_pairs)
    site_weights = np.array(probs, dtype=float)
    data = np.empty((nsites, nnodes, nstates), dtype=float)
    for site_index, site_data in enumerate(datas):
        for i, na in enumerate(nodes):
            data[site_index, i] = site_data[na]

    # Initialize the per-edge rate matrix scaling factor guesses.
    scaling_guesses = np.empty(nnodes-1, dtype=float)
    for (na, nb), rate in guess_edge_to_rate.items():
        eidx = node_to_idx[nb] - 1
        scaling_guesses[eidx] = np.log(rate)

    # Initialize temporary arrays for EM.
    # Initialize the em step function for our tree structure, data,
    # and unscaled rate matrices.
    em_mem = EMStorage(nsites, nnodes, nstates)
    use_log_scale = True

    # This partial function will take log scaling factors per edge
    # and return better log scaling factors per edge.
    em = partial(em_function,
            T, node_to_idx, site_weights, m,
            transq,
            data,
            root_distn1d,
            em_mem,
            use_log_scale,
            )

    # Initialize memory for log likelihood shape calculation.
    degree = 2
    ll_shape_mem = LikelihoodShapeStorage(nsites, nnodes, nstates, degree)

    # Initialize the objective function and its shape information.
    # This is also used as the Lyapunov function for the stable EM.
    fgh = partial(get_logscale_fgh,
            T, node_to_idx, site_weights, m,
            transq,
            data,
            root_distn1d,
            ll_shape_mem,
            )
    def f(X):
        degree = 0
        fx = fgh(degree, X)
        return -fx
    def g(X):
        degree = 1
        fx, gx = fgh(degree, X)
        return -gx
    def h(X):
        degree = 2
        fx, gx, hx = fgh(degree, X)
        return -hx

    # Define the initial point in the search.
    x0 = scaling_guesses
    print('initial guesses:')
    print(x0)
    print(np.exp(x0))

    # Do a few accelerated EM rounds, guided by the log likelihood function.
    # When acceleration fails, as indicated by not reducing the log likelihood,
    # a pure EM step is taken instead of an accelerated EM step.
    # This behavior can be controlled by the backtrack_rate.
    result = fixed_point_squarem(em, x0, L=f,
            backtrack_rate=1, atol=1e-8, maxiter=20, maxfun=20)

    print('acclerated EM results:')
    print(result)
    x0 = result.x

    # Do a few EM rounds.
    # This is a separate function instead of a loop so that
    # the time can be seen for in the profiler.
    #niter = 50
    #x0 = do_a_few_EM_iterations(em, x0, niter)

    print('guesses after EM:')
    print(x0)
    print(np.exp(x0))

    print('guesses before hessian-guided search:')
    print(x0)
    print(np.exp(x0))

    result = scipy.optimize.minimize(f, x0, jac=g, hess=h, tol=1e-8,
            method='trust-ncg')
            #method='dogleg')

    print('result of hessian-guided search:')
    print(result)
    print(np.exp(result.x))



def do_gradient_search(T, root,
        edge_to_rate, edge_to_Q, root_distn1d,
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

    # Allocate a transition probability matrix array.
    transp = np.empty_like(transq)
    transp_grad = np.empty_like(transp)

    # Stack the data into a single array,
    # and construct an array of site weights.
    nsites = len(data_prob_pairs)
    datas, probs = zip(*data_prob_pairs)
    site_weights = np.array(probs, dtype=float)
    data = np.empty((nsites, nnodes, nstates), dtype=float)
    for site_index, site_data in enumerate(datas):
        for i, na in enumerate(nodes):
            data[site_index, i] = site_data[na]

    # Initialize the per-edge rate matrix scaling factor guesses.
    scaling_guesses = np.empty(nnodes-1, dtype=float)
    for (na, nb), rate in guess_edge_to_rate.items():
        eidx = node_to_idx[nb] - 1
        scaling_guesses[eidx] = rate

    f = partial(objective_and_gradient_for_search,
            T, node_to_idx, site_weights, m,
            transq, transp, transp_grad,
            data,
            root_distn1d,
            )
    x0 = scaling_guesses
    bounds = [(0, None)]*(nnodes-1)
    result = scipy.optimize.minimize(f, x0,
            method='L-BFGS-B', jac=True, bounds=bounds)
    print(result)


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
        edge_to_rate, edge_to_Q, root_distn1d,
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
    f_aitken = partial(em_objective_for_aitken,
            T, node_to_idx, site_weights,
            m,
            transq, transp,
            interact_trans, interact_dwell,
            data,
            root_distn1d,
            trans_out, dwell_out)
    f_broyden = partial(em_objective_for_broyden,
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


def do_cythonized_em(T, root,
        edge_to_rate, edge_to_Q, root_distn1d,
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


def do_em(T, root, edge_to_rate, edge_to_Q, root_distn1d,
        data_prob_pairs, guess_edge_to_rate):
    # Extract the number of states.
    n = root_distn1d.shape[0]

    # Do the EM iterations.
    nsteps = 3
    for em_iteration_index in range(nsteps):
        
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
            #print(edge, 'Q:')
            #print(scaled_Q)
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
                #print(edge)
                #print(P)
                #print(J)
                #print()

                # transition contribution
                interact = edge_to_interact_trans[edge]
                total = 0
                for i in range(n):
                    for j in range(n):
                        if J[i, j]:
                            coeff = J[i, j] / P[i, j]
                            total += coeff * interact[i, j]
                edge_to_trans_expectation[edge] += lhood * total

                # dwell contribution
                interact = edge_to_interact_dwell[edge]
                total = 0
                for i in range(n):
                    for j in range(n):
                        if J[i, j]:
                            coeff = J[i, j] / P[i, j]
                            total += coeff * interact[i, j]
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
            #print(edge, trans, dwell, ratio, old_rate, new_rate)
            print(edge, trans, dwell, new_rate)
        print()


def main():
    np.random.seed(12354)

    # Define the size of the state space
    # which will be constant across the whole tree.
    n = 6

    # Sample a random root distribution as a 1d numpy array.
    pzero = 0
    root_distn1d = sample_distn1d(n, pzero)

    # Hardcode a tree with four leaves
    # and some arbitrary hardcoded rate scaling factors per edge.
    T, root, edge_to_rate, leaves, internal_nodes = get_tree_info()

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
        #guess_edge_to_rate[edge] = edge_to_rate[edge] + 0.05

    #f = do_em
    #f = do_cythonized_em
    #f = do_cythonized_accelerated_em
    #f = do_gradient_search
    #f = do_jj97_qn2_search
    #f = do_hessian_search

    print('searching...')
    edge_to_rate, neg_ll = estimate_edge_rates(
            T, root, edge_to_Q, root_distn1d, data_prob_pairs)
    print('edge to rate:')
    print(edge_to_rate)
    print('negative log likelihood of estimate:', neg_ll)


    #def get_ll_gradient(*args):
    #def get_em_displacement(*args):
    #def jj97_qn2(t0, grad, em, bounds=None):


if __name__ == '__main__':
    main()

