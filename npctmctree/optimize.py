"""
Maximum likelihood estimation of edge-specific rate scaling factors.

"""
from __future__ import division, print_function, absolute_import

import functools

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from .derivatives import LikelihoodShapeStorage, get_log_likelihood_info
from .em import EMStorage, em_function
from .squarem import fixed_point_squarem


def estimate_edge_rates(T, root, edge_to_Q, root_distn1d, data_weight_pairs,
        method='trust-ncg'):
    """
    Estimate edge-specific rate scaling factors.

    Returns
    -------
    edge_to_rate : dict
        Map from edge to estimated rate matrix scaling factor.
    neg_ll : float
        Negative log likelihood computed using the estimated edge rates.

    """
    #TODO docstring and unit tests

    # Define a toposort node ordering and a corresponding csr matrix.
    nodes = nx.topological_sort(T, [root])
    node_to_idx = dict((na, i) for i, na in enumerate(nodes))
    m = nx.to_scipy_sparse_matrix(T, nodes)

    # Stack the transition rate matrices into a single array.
    nnodes = len(nodes)
    nstates = root_distn1d.shape[0]
    transq = np.empty((nnodes-1, nstates, nstates), dtype=float)
    for (na, nb), Q in edge_to_Q.items():
        eidx = node_to_idx[nb] - 1
        transq[eidx] = Q

    # Stack the data into a single array,
    # and construct an array of site weights.
    nsites = len(data_weight_pairs)
    datas, weights = zip(*data_weight_pairs)
    site_weights = np.array(weights, dtype=float)
    data = np.empty((nsites, nnodes, nstates), dtype=float)
    for site_index, site_data in enumerate(datas):
        for i, na in enumerate(nodes):
            data[site_index, i] = site_data[na]

    # Initialize the per-edge rate matrix log scaling factor guesses.
    x0 = np.zeros(nnodes-1, dtype=float)

    # Initialize temporary arrays for EM.
    # Initialize the em step function for our tree structure, data,
    # and unscaled rate matrices.
    em_mem = EMStorage(nsites, nnodes, nstates)
    use_log_scale = True

    # This partial function will take log scaling factors per edge
    # and return better log scaling factors per edge.
    em = functools.partial(em_function,
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
    use_log_scale = True
    fgh = functools.partial(get_log_likelihood_info,
            T, node_to_idx, site_weights, m,
            transq, data, root_distn1d, ll_shape_mem, use_log_scale)
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

    # Do a few accelerated EM rounds, guided by the log likelihood function.
    # When acceleration fails, as indicated by not reducing the log likelihood,
    # a pure EM step is taken instead of an accelerated EM step.
    # This behavior can be controlled by the backtrack_rate.
    print('neg log likelihood using unchanged rates:', f(x0))
    result = fixed_point_squarem(em, x0, L=f,
            backtrack_rate=1, atol=1e-8, maxiter=20, maxfun=20)

    #print('acclerated EM results:')
    #print(result)
    x0 = result.x

    # Do a few EM rounds.
    # This is a separate function instead of a loop so that
    # the time can be seen for in the profiler.
    #niter = 50
    #x0 = do_a_few_EM_iterations(em, x0, niter)

    #print('guesses after EM:')
    #print(x0)
    #print(np.exp(x0))

    #print('guesses before hessian-guided search:')
    #print(x0)
    #print(np.exp(x0))

    if method == 'trust-ncg':
        result = minimize(f, x0, jac=g, hess=h, tol=1e-8, method=method)
    elif method == 'L-BFGS-B':
        result = minimize(f, x0, jac=g, method=method)
    else:
        result = minimize(f, x0, jac=g, hess=h, method=method)

    #print('result of hessian-guided search:')
    #print(result)
    #print(np.exp(result.x))

    # Return a map from edge to rate matrix scaling factor.
    # Also return negative log likelihood.
    edge_rates = np.exp(result.x)
    edge_to_rate = dict()
    for edge in T.edges():
        na, nb = edge
        eidx = node_to_idx[nb] - 1
        edge_to_rate[edge] = edge_rates[eidx]
    return edge_to_rate, result.fun

