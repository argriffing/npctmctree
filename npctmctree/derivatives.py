"""
Compute the shape of log likelihood for i.i.d observations.

For example compute gradient and hessian, with respect to parameters
defining edge-specific scaling factors or their logarithms.

"""
from __future__ import division, print_function, absolute_import

import numpy as np

from scipy.linalg import expm

from npmctree.cyfels import iid_likelihoods


def get_log_likelihood_info(
        T, node_to_idx, site_weights, m,
        transq_unscaled, transp_ws, transp_mod_ws,
        data, root_distn1d, scale, degree=0):
    """
    """
    # Request input validation for Cythonized functions.
    validation = 1

    # Check the requested degree.
    allowed_degrees = (0, 1, 2)
    if degree not in allowed_degrees:
        raise ValueError('expected degree in ' + str(allowed_degrees))

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
        transp_ws[eidx] = expm(Q)

    # Compute the site likelihoods.
    likelihoods = np.empty(nsites, dtype=float)
    iid_likelihoods(
            m.indices, m.indptr,
            transp_ws,
            data,
            root_distn1d,
            likelihoods,
            validation,
            )

    # Compute the sum of log likelihoods.
    # This is the negative of the objective function.
    ll_total = np.log(likelihoods).dot(site_weights)

    # If the degree is limited to zero then we are done.
    if degree == 0:
        return ll_total

    # To compute the gradient,
    # for each edge adjust the transition probability matrix.
    lhood_gradients = np.empty((nnodes-1, nsites), dtype=float)
    for edge in T.edges():
        na, nb = edge
        eidx = node_to_idx[nb] - 1
        transp_mod_ws[...] = transp_ws
        transp_mod_ws[eidx] = np.dot(transq_unscaled[eidx], transp_ws[eidx])
        iid_likelihoods(
                m.indices, m.indptr,
                transp_mod_ws,
                data,
                root_distn1d,
                lhood_gradients[eidx],
                validation,
                )

    # Compute the log likelihood gradient.
    ll_gradient = np.dot(lhood_gradients / likelihoods[None, :], site_weights)

    # If the degree is limited to one then we are done.
    if degree == 1:
        return ll_total, ll_gradient

    # To compute the hessian,
    # for each edge pair adjust the transition probability matrix.
    lhood_diff_xy = np.ones((nnodes-1, nnodes-1, nsites), dtype=float) * 666
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
            transp_mod_ws[...] = transp_ws
            transp_mod_ws[eidx0, :, :] = np.dot(Q0, transp_mod_ws[eidx0])
            transp_mod_ws[eidx1, :, :] = np.dot(Q1, transp_mod_ws[eidx1])
            iid_likelihoods(
                    m.indices, m.indptr,
                    transp_mod_ws,
                    data,
                    root_distn1d,
                    lhood_diff_xy[eidx0, eidx1, :],
                    validation,
                    )

    print('lhood diff xy:')
    print(lhood_diff_xy)

    # Compute the hessian.

    # l_xy / l
    a = lhood_diff_xy / likelihoods[None, None, :]
    print('a shape:', a.shape)

    # l_x * l_y (vectorized outer product)
    b = np.einsum('i...,j...->ij...', lhood_gradients, lhood_gradients)
    print('b shape:', b.shape)

    # l * l
    c = (likelihoods * likelihoods)[None, None, :]
    print('c shape:', c.shape)

    print('hessian parts:')
    print(a)
    print(b)
    print(c)
    print()

    ll_hessian = np.dot(a - b/c, site_weights)

    return ll_total, ll_gradient, ll_hessian

