"""
Compute the shape of log likelihood for i.i.d observations.

For example compute gradient and hessian, with respect to parameters
defining edge-specific scaling factors or their logarithms.

"""
from __future__ import division, print_function, absolute_import

import numpy as np

from scipy.linalg import expm

from npmctree.cyfels import iid_likelihoods


class LikelihoodShapeStorage(object):
    """
    Preallocate a bunch of memory here.

    Parameters
    ----------
    nsites : int
        The number of iid observations.
    nnodes : int
        The number of nodes in the rooted tree.
        This is one more than the number of edges.
    nstates : int
        The number of states in the stochastic process.
    degree : integer in (0, 1, 2)
        The maximum degree of derivatives to compute.

    Notes
    -----
    transp_ws : ndarray with shape (nnodes-1, nstates, nstates)
        Space for a transition probability matrix for each edge of the tree.
    transp_mod_ws : ndarray with shape (nnodes-1, nstates, nstates)
        Space for gradient and hessian calculations.

    """
    def __init__(self, nsites, nnodes, nstates, degree):
        def _alloc(*args):
            return np.empty(args, dtype=float)
        self.likelihoods = _alloc(nsites)
        self.transp_ws = _alloc(nnodes-1, nstates, nstates)
        self.transq = _alloc(nnodes-1, nstates, nstates)
        if degree > 0:
            self.lhood_gradients = _alloc(nnodes-1, nsites)
            self.transp_mod_ws = _alloc(nnodes-1, nstates, nstates)
        if degree > 1:
            self.lhood_diff_xy = _alloc(nnodes-1, nnodes-1, nsites)


def get_log_likelihood_info(
        T, node_to_idx, site_weights, m,
        transq_unscaled, data, root_distn1d, mem, use_log_scale,
        degree, scale):
    """
    Evaluate log likelihood and derivatives for iid observations.

    Note that the scale is deliberately the final argument
    so that it can be easily used with functools.partial.

    Parameters
    ----------
    T : networkx DiGraph
        rooted tree
    node_to_idx : dict
        map from networkx node to toposort order index
    site_weights : 1d ndarray
        The weight for each observation.
        For example, this could be the number of repeats of each observation,
        or it could be a site probability for the application of computing
        Kullback-Leibler divergence.
    m : sparse matrix in csr format
        A csr matrix representation of the rooted tree.
    transq_unscaled : ndarray with shape (nnodes-1, nstates, nstates)
        The transition rate matrix for each edge of the tree.
    data : ndarray with shape (nsites, nnodes, nstates)
        Observation data.
    root_distn1d : 1d ndarray
        Prior distribution of states at the root of the tree.
    mem : LikelihoodShapeStorage object
        An object with preallocated arrays.
    use_log_scale : bool
        If True then the scale input parameters will be interpreted
        as logs of scaling factors, and the reported first and second
        derivatives will be with respect to logs of scaling factors.
    degree : integer in {0, 1, 2}
        Max degree of computed derivatives.
        If degree is 0 then report only the log likelihood.
        If degree is 1 then also report the 1d ndarray of first derivatives.
        If degree is 2 then also report the 2d ndarray of second derivatives.
    scale : 1d ndarray
        Scaling factors or logs of scaling factors
        to be applied to the transition rate matrices.
        
    """
    #TODO add unit tests

    # If we are using log scale then the scale parameter
    # is a vector of logarithms of edge-specific rate scaling factors.
    # In that case use exp(scale) for the rates,
    # and adjust the gradient and hessian by multiplying by
    # exp(log_rate) and exp(log_rate_x + log_rate_y) respectively.
    # For example using math notation,
    # log(f(exp(x)))' == exp(x) f'(exp(x)) / f(exp(x)).
    if use_log_scale:
        log_rates = scale
        scale = np.exp(log_rates)

    # Optionally request input validation for Cythonized functions.
    validation = 0

    # Check the requested degree.
    allowed_degrees = (0, 1, 2)
    if degree not in allowed_degrees:
        raise ValueError('expected degree in ' + str(allowed_degrees))

    # Unpack some stuff.
    nsites, nnodes, nstates = data.shape
    n = nstates

    # Scale the rate matrices according to the edge ratios.
    # Use in-place operations to avoid unnecessary memory copies.
    #transq = transq_unscaled * scale[:, None, None]
    mem.transq[...] = transq_unscaled
    mem.transq *= scale[:, None, None]

    # Compute the probability transition matrix arrays.
    for edge in T.edges():
        na, nb = edge
        eidx = node_to_idx[nb] - 1
        Q = mem.transq[eidx]
        mem.transp_ws[eidx] = expm(Q)

    # Compute the site likelihoods.
    iid_likelihoods(
            m.indices, m.indptr,
            mem.transp_ws,
            data,
            root_distn1d,
            mem.likelihoods,
            validation,
            )

    # Compute the sum of log likelihoods.
    # This is the negative of the objective function.
    ll_total = np.log(mem.likelihoods).dot(site_weights)

    # If the degree is limited to zero then we are done.
    if degree == 0:
        return ll_total

    # To compute the gradient,
    # for each edge adjust the transition probability matrix.
    for edge in T.edges():
        na, nb = edge
        eidx = node_to_idx[nb] - 1
        mem.transp_mod_ws[...] = mem.transp_ws
        mem.transp_mod_ws[eidx] = np.dot(
                transq_unscaled[eidx], mem.transp_ws[eidx])
        iid_likelihoods(
                m.indices, m.indptr,
                mem.transp_mod_ws,
                data,
                root_distn1d,
                mem.lhood_gradients[eidx],
                validation,
                )

    # Compute the log likelihood gradient.
    # Adjust for log scale if necessary.
    ll_gradient = np.dot(
            mem.lhood_gradients / mem.likelihoods[None, :], site_weights)
    if use_log_scale:
        ll_gradient = ll_gradient * scale

    # If the degree is limited to one then we are done.
    if degree == 1:
        return ll_total, ll_gradient

    # To compute the hessian,
    # for each edge pair adjust the transition probability matrix.
    mem.lhood_diff_xy[...] = 1
    edges = list(T.edges())
    for edge0 in edges:
        na0, nb0 = edge0
        eidx0 = node_to_idx[nb0] - 1
        Q0 = transq_unscaled[eidx0]
        for edge1 in edges:
            na1, nb1 = edge1
            eidx1 = node_to_idx[nb1] - 1
            Q1 = transq_unscaled[eidx1]

            # Compute the hessian.
            mem.transp_mod_ws[...] = mem.transp_ws
            mem.transp_mod_ws[eidx0, :, :] = np.dot(
                    Q0, mem.transp_mod_ws[eidx0])
            mem.transp_mod_ws[eidx1, :, :] = np.dot(
                    Q1, mem.transp_mod_ws[eidx1])
            iid_likelihoods(
                    m.indices, m.indptr,
                    mem.transp_mod_ws,
                    data,
                    root_distn1d,
                    mem.lhood_diff_xy[eidx0, eidx1, :],
                    validation,
                    )

    # Compute the ingredients of the hessian.
    # TODO Should this be computed within cython?

    # l_xy / l
    a = mem.lhood_diff_xy / mem.likelihoods[None, None, :]

    # l_x * l_y (vectorized outer product)
    b = np.einsum('i...,j...->ij...', mem.lhood_gradients, mem.lhood_gradients)

    # l * l
    c = (mem.likelihoods * mem.likelihoods)[None, None, :]

    # Compute the hessian.
    # Adjust for log scale if necessary.
    ll_hessian = np.dot(a - b/c, site_weights)
    if use_log_scale:
        ll_hessian = ll_hessian * np.outer(scale, scale) + np.diag(ll_gradient)

    return ll_total, ll_gradient, ll_hessian

