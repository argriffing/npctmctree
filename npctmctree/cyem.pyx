"""
Speed up expectation maximization calculations.

This is a newer and more focused version of the pyfelscore package
without the application-specific functions.
This Cython module should be built automatically
by the setup.py infrastructure, so the end user does not need to invoke
any special command or know anything about Cython.
But because the intermediate .c file will not be included in the repo,
the end user will need to have Cython.

To check for full vs. partial cythonization try the following.
$ cython -a mycythonfile.pyx

"""

#TODO check if this line can be removed
from cython.view cimport array as cvarray

import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_array_less
cimport numpy as cnp
cimport cython
from libc.math cimport log, exp, sqrt

#TODO check if this line can be removed
cnp.import_array()


# Use fused types to support both 32 bit and 64 bit sparse matrix indices.
# The following mailing list question has some doubt about how well this works.
# https://mail.python.org/pipermail/cython-devel/2014-March/004002.html
ctypedef fused idx_t:
    cnp.int32_t
    cnp.int64_t


#__all__ = ['assert_csr_tree', 'esd_site_first_pass']


def assert_shape_equal(arr, desired_shape):
    # Work around Cython problems.
    # http://trac.cython.org/cython_trac/ticket/780
    n = arr.ndim
    assert_equal(n, len(desired_shape))
    for i in range(n):
        assert_equal(arr.shape[i], desired_shape[i])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def assert_csr_tree(
        idx_t[:] csr_indices,
        idx_t[:] csr_indptr,
        int nnodes,
        ):
    """
    Assume the node indices are 0..(nnodes-1) and are in toposort preorder.

    """
    # Require at least one node.
    # For example networkx raises an exception if you try to build
    # a csr matrix from a graph without nodes.
    assert_array_less(0, nnodes)

    # Check the conformability of the inputs.
    # Note that the global interpreter lock (gil) should be in effect
    # for this section.
    assert_shape_equal(csr_indices, (nnodes-1,))
    assert_shape_equal(csr_indptr, (nnodes+1,))

    # Check that each indptr element is either a valid index
    # into the indices array or is equal to the length of the indices array.
    assert_array_less(-1, csr_indptr)
    assert_array_less(csr_indptr, nnodes+1)
    assert_array_less(-1, csr_indices)
    assert_array_less(csr_indices, nnodes)

    # Check preorder.
    cdef int j
    cdef idx_t indstart, indstop
    cdef idx_t na, nb
    cdef cnp.int_t[:] visited = np.zeros(nnodes, dtype=int)
    cdef cnp.int_t[:] head = np.zeros(nnodes, dtype=int)
    cdef cnp.int_t[:] tail = np.zeros(nnodes, dtype=int)
    with nogil:
        visited[0] = 1
        for na in range(nnodes):
            head[na] = visited[na]
            indstart = csr_indptr[na]
            indstop = csr_indptr[na+1]
            for j in range(indstart, indstop):
                nb = csr_indices[j]
                tail[nb] = visited[nb]
                visited[nb] += 1

    # Check that each node had been visited exactly once.
    assert_array_equal(visited, 1)

    # Check that each head node had been visited exactly once.
    assert_array_equal(head, 1)

    # Check that each tail node had not been visited.
    assert_array_equal(tail, 0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def expectation_step(
        idx_t[:] csr_indices, # (nnodes-1,)
        idx_t[:] csr_indptr, # (nnodes+1,)
        cnp.float64_t[:, :, :] transp, # (nnodes-1, nstates, nstates)
        cnp.float64_t[:, :, :] transq, # (nnodes-1, nstates, nstates)
        cnp.float64_t[:, :, :] interact_trans, # (nnodes-1, nstates, nstates)
        cnp.float64_t[:, :, :] interact_dwell, # (nnodes-1, nstates, nstates)
        cnp.float64_t[:, :, :] data, # (nsites, nnodes, nstates)
        cnp.float64_t[:] root_distn, # (nstates,)
        cnp.float64_t[:, :] trans_out, # (nsites, nnodes-1)
        cnp.float64_t[:, :] dwell_out, # (nsites, nnodes-1)
        int validation=1,
        ):
    """
    Compute scaled transition count and scaled dwell time expectations for EM.

    The esd abbreviation refers to 'edge-specific dense' transition matrices.
    Nodes of the tree are indexed according to a topological sort,
    starting at the root.
    Edges of the tree are indexed such that if x and y are node indices,
    then the directed edge (x, y) has index y-1.
    Note that the number of edges in a tree is one fewer
    than the number of nodes in the tree.

    Note that csr_indices and csr_indptr can be computed by using
    networkx to construct a csr sparse matrix by calling
    nx.to_scipy_sparse_matrix function and passing a node ordering
    constructed using nx.topological_sort.
    Because we want to accept csr_indices and csr_indptr arrays
    from scipy.sparse.csr_matrix objects, we must allow both 32 bit and 64 bit
    integer types.

    Parameters
    ----------
    csr_indices : ndarray view
        Part of the Compressed Sparse Row format of the tree structure.
    csr_indptr : ndarray view
        Part of the Compressed Sparse Row format of the tree structure.
    transp : ndarray view
        For each edge, a dense transition probability matrix.
    transq : ndarray view
        For each edge, a dense transition rate matrix.
    data : ndarray view
        For each site and node, the emission likelihood for each state.
    validation : int
        Indicates whether to check the input formats.
        This would include checking the csr representation of the tree and
        the dimensions of the inputs.

    """
    # Get the number of nodes and the number of states.
    cdef int nsites = data.shape[0]
    cdef int nnodes = data.shape[1]
    cdef int nstates = data.shape[2]

    # Check the conformability of the inputs.
    # Note that the global interpreter lock (gil) should be in effect
    # for this section.
    if validation:
        assert_shape_equal(transp, (nnodes-1, nstates, nstates))
        assert_shape_equal(transq, (nnodes-1, nstates, nstates))
        assert_csr_tree(csr_indices, csr_indptr, nnodes)

    # Declare some variables for iterating over edges.
    cdef int i, j
    cdef idx_t indstart, indstop
    cdef idx_t na, nb, eidx

    # Declare variables for iterating over sites and states.
    cdef int c
    cdef int sa, sb

    # Allocate workspace for partial likelihoods and posterior distributions.
    cdef cnp.float64_t[:, :] lhood = np.empty((nnodes, nstates), dtype=float)
    cdef cnp.float64_t[:, :] post = np.empty((nnodes, nstates), dtype=float)
    cdef cnp.float64_t[:] cond = np.empty(nstates, dtype=float)

    # multiplicative and additive accumulators
    cdef double multiplicative_prob, additive_prob
    cdef double accum, joint, coeff

    with nogil:

        # Clear the expectation accumulators.
        for c in range(nsites):
            for eidx in range(nnodes-1):
                trans_out[c, eidx] = 0
                dwell_out[c, eidx] = 0

        # Iterate over sites.
        for c in range(nsites):
        
            # Clear the likelihood and posterior distribution arrays.
            for i in range(nnodes):
                for j in range(nstates):
                    lhood[i, j] = 0
                    post[i, j] = 0

            # Fill the partial likelihood array for each subtree.
            for i in range(nnodes):
                na = (nnodes - 1) - i
                indstart = csr_indptr[na]
                indstop = csr_indptr[na+1]

                # Compute the subtree likelihood for each possible state.
                for sa in range(nstates):
                    multiplicative_prob = data[c, na, sa]
                    for j in range(indstart, indstop):

                        # Compute the tail node index and the edge index.
                        nb = csr_indices[j]
                        eidx = nb - 1

                        # Compute the additive probability for the edge.
                        additive_prob = 0 
                        for sb in range(nstates):
                            additive_prob += (
                                    transp[eidx, sa, sb] * lhood[nb, sb])

                        # Contribute the probability associated with the edge.
                        multiplicative_prob *= additive_prob

                    # The subtree probability is a product.
                    lhood[na, sa] = multiplicative_prob

            # Apply the prior distribution at the root.
            for sa in range(nstates):
                lhood[0, sa] *= root_distn[sa]

            # Initialize the posterior distribution at the root.
            accum = 0
            for sa in range(nstates):
                accum += lhood[0, sa]
            for sa in range(nstates):
                post[0, sa] = lhood[0, sa] / accum

            # Iterate over edges.
            # At each edge, accumulate the trans and dwell expectations,
            # and compute the posterior distribution over states
            # at the tail node.
            for na in range(nnodes):
                indstart = csr_indptr[na]
                indstop = csr_indptr[na+1]
                for j in range(indstart, indstop):
                    nb = csr_indices[j]
                    eidx = nb - 1

                    # Iterate over head node states.
                    for sa in range(nstates):

                        # Compute the conditional posterior distribution
                        # over tail node states.
                        accum = 0
                        for sb in range(nstates):
                            cond[sb] = transp[eidx, sa, sb] * lhood[nb, sb]
                            accum += cond[sb]
                        for sb in range(nstates):
                            cond[sb] /= accum

                        # Iterate over joint states on the edge.
                        for sb in range(nstates):

                            # Compute the posterior joint probability.
                            joint = post[na, sa] * cond[sb]
                            if not joint:
                                continue

                            # Contribute to the expectations.
                            coeff = joint / transp[eidx, sa, sb]
                            trans_out[c, eidx] += (
                                coeff * interact_trans[eidx, sa, sb])
                            dwell_out[c, eidx] += (
                                coeff * interact_dwell[eidx, sa, sb])

                            # Contribute to the marginal posterior distribution
                            # over tail node states.
                            post[nb, sb] += joint

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def conditional_expectation(
        idx_t[:] csr_indices, # (nnodes-1,)
        idx_t[:] csr_indptr, # (nnodes+1,)
        cnp.float64_t[:, :, :] transp, # (nnodes-1, nstates, nstates)
        cnp.float64_t[:, :, :] transq, # (nnodes-1, nstates, nstates)
        cnp.float64_t[:, :, :] interact, # (nnodes-1, nstates, nstates)
        cnp.float64_t[:, :, :] data, # (nsites, nnodes, nstates)
        cnp.float64_t[:] root_distn, # (nstates,)
        cnp.float64_t[:, :] expect_out, # (nsites, nnodes-1)
        int validation=1,
        ):
    """
    Generic conditional expectation.

    This function is copied from the edge-specific rate EM step code.
    The caller must use something like the Frechet derivative of the
    matrix exponential to provide the interaction matrix for each edge.

    Parameters
    ----------
    csr_indices : ndarray view
        Part of the Compressed Sparse Row format of the tree structure.
    csr_indptr : ndarray view
        Part of the Compressed Sparse Row format of the tree structure.
    transp : ndarray view
        For each edge, a dense transition probability matrix.
    transq : ndarray view
        For each edge, a dense transition rate matrix.
    data : ndarray view
        For each site and node, the emission likelihood for each state.
    validation : int
        Indicates whether to check the input formats.
        This would include checking the csr representation of the tree and
        the dimensions of the inputs.

    """
    # Get the number of nodes and the number of states.
    cdef int nsites = data.shape[0]
    cdef int nnodes = data.shape[1]
    cdef int nstates = data.shape[2]

    # Check the conformability of the inputs.
    # Note that the global interpreter lock (gil) should be in effect
    # for this section.
    if validation:
        assert_shape_equal(transp, (nnodes-1, nstates, nstates))
        assert_shape_equal(transq, (nnodes-1, nstates, nstates))
        assert_shape_equal(interact, (nnodes-1, nstates, nstates))
        assert_csr_tree(csr_indices, csr_indptr, nnodes)

    # Declare some variables for iterating over edges.
    cdef int i, j
    cdef idx_t indstart, indstop
    cdef idx_t na, nb, eidx

    # Declare variables for iterating over sites and states.
    cdef int c
    cdef int sa, sb

    # Allocate workspace for partial likelihoods and posterior distributions.
    cdef cnp.float64_t[:, :] lhood = np.empty((nnodes, nstates), dtype=float)
    cdef cnp.float64_t[:, :] post = np.empty((nnodes, nstates), dtype=float)
    cdef cnp.float64_t[:] cond = np.empty(nstates, dtype=float)

    # multiplicative and additive accumulators
    cdef double multiplicative_prob, additive_prob
    cdef double accum, joint, coeff

    with nogil:

        # Clear the expectation accumulators.
        for c in range(nsites):
            for eidx in range(nnodes-1):
                expect_out[c, eidx] = 0

        # Iterate over sites.
        for c in range(nsites):
        
            # Clear the likelihood and posterior distribution arrays.
            for i in range(nnodes):
                for j in range(nstates):
                    lhood[i, j] = 0
                    post[i, j] = 0

            # Fill the partial likelihood array for each subtree.
            for i in range(nnodes):
                na = (nnodes - 1) - i
                indstart = csr_indptr[na]
                indstop = csr_indptr[na+1]

                # Compute the subtree likelihood for each possible state.
                for sa in range(nstates):
                    multiplicative_prob = data[c, na, sa]
                    for j in range(indstart, indstop):

                        # Compute the tail node index and the edge index.
                        nb = csr_indices[j]
                        eidx = nb - 1

                        # Compute the additive probability for the edge.
                        additive_prob = 0 
                        for sb in range(nstates):
                            additive_prob += (
                                    transp[eidx, sa, sb] * lhood[nb, sb])

                        # Contribute the probability associated with the edge.
                        multiplicative_prob *= additive_prob

                    # The subtree probability is a product.
                    lhood[na, sa] = multiplicative_prob

            # Apply the prior distribution at the root.
            for sa in range(nstates):
                lhood[0, sa] *= root_distn[sa]

            # Initialize the posterior distribution at the root.
            accum = 0
            for sa in range(nstates):
                accum += lhood[0, sa]
            for sa in range(nstates):
                post[0, sa] = lhood[0, sa] / accum

            # Iterate over edges.
            # At each edge, accumulate the trans and dwell expectations,
            # and compute the posterior distribution over states
            # at the tail node.
            for na in range(nnodes):
                indstart = csr_indptr[na]
                indstop = csr_indptr[na+1]
                for j in range(indstart, indstop):
                    nb = csr_indices[j]
                    eidx = nb - 1

                    # Iterate over head node states.
                    for sa in range(nstates):

                        # Compute the conditional posterior distribution
                        # over tail node states.
                        accum = 0
                        for sb in range(nstates):
                            cond[sb] = transp[eidx, sa, sb] * lhood[nb, sb]
                            accum += cond[sb]
                        for sb in range(nstates):
                            cond[sb] /= accum

                        # Iterate over joint states on the edge.
                        for sb in range(nstates):

                            # Compute the posterior joint probability.
                            joint = post[na, sa] * cond[sb]
                            if not joint:
                                continue

                            # Contribute to the expectations.
                            coeff = joint / transp[eidx, sa, sb]
                            expect_out[c, eidx] += (
                                coeff * interact[eidx, sa, sb])

                            # Contribute to the marginal posterior distribution
                            # over tail node states.
                            post[nb, sb] += joint

    return 0
