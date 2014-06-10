"""
Expectation maximization helper functions.

Specifically, estimate edge specific rate scaling factors for
rate matrices on a rooted tree with known prior state distribution at the root
and with partial observations at nodes of the tree.

"""
from __future__ import division, print_function, absolute_import

import numpy as np

from scipy.linalg import expm, expm_frechet

from .cyem import expectation_step


class EMStorage(object):
    """
    """
    def __init__(self, nsites, nnodes, nstates):
        # allocate some empty arrays
        def _alloc(*args):
            return np.empty(args, dtype=float)
        self.transp = _alloc(nnodes-1, nstates, nstates)
        self.transq = _alloc(nnodes-1, nstates, nstates)
        self.interact_trans = _alloc(nnodes-1, nstates, nstates)
        self.interact_dwell = _alloc(nnodes-1, nstates, nstates)
        self.trans_out = _alloc(nsites, nnodes-1)
        self.dwell_out = _alloc(nsites, nnodes-1)

        # Define the trans indicator and dwell indicator.
        # These are constant across all EM iterations.
        ident = np.identity(nstates)
        self.trans_indicator = np.ones((nstates, nstates)) - ident
        self.dwell_indicator = ident


def em_function(
        T, node_to_idx, site_weights,
        m,
        transq_unscaled,
        data,
        root_distn1d,
        mem,
        use_log_scale,
        scale,
        ):
    """
    Recast EM as a fixed-point problem.
    
    This approach is inspired by the introduction of the following paper.
    A QUASI-NEWTON ACCELERATION OF THE EM ALGORITHM
    Kenneth Lange
    1995

    """
    #TODO improve docstring and add unit tests

    # Transform the scaling factors if necessary.
    if use_log_scale:
        log_scale = scale
        scale = np.exp(scale)

    # Unpack some stuff.
    nsites, nnodes, nstates = data.shape

    # Scale the rate matrices according to the edge ratios.
    # Use in-place operations to avoid unnecessary memory copies.
    #transq = transq_unscaled * scale[:, None, None]
    mem.transq[...] = transq_unscaled
    mem.transq *= scale[:, None, None]

    # Compute the probability transition matrix arrays
    # and the interaction matrix arrays.
    for edge in T.edges():
        na, nb = edge
        eidx = node_to_idx[nb] - 1
        Q = mem.transq[eidx]
        mem.transp[eidx] = expm(Q)
        mem.interact_trans[eidx] = expm_frechet(
                Q, Q * mem.trans_indicator, compute_expm=False)
        mem.interact_dwell[eidx] = expm_frechet(
                Q, Q * mem.dwell_indicator, compute_expm=False)

    # Compute the expectations using Cython.
    validation = 0
    expectation_step(
            m.indices, m.indptr,
            mem.transp, mem.transq,
            mem.interact_trans, mem.interact_dwell,
            data,
            root_distn1d,
            mem.trans_out, mem.dwell_out,
            validation,
            )

    # Compute the per-edge ratios.
    trans_sum = (mem.trans_out * site_weights[:, None]).sum(axis=0)
    dwell_sum = (mem.dwell_out * site_weights[:, None]).sum(axis=0)
    if use_log_scale:
        log_scaling_ratios = np.log(trans_sum) - np.log(-dwell_sum)
        return log_scale + log_scaling_ratios
    else:
        scaling_ratios = trans_sum / -dwell_sum
        return scale * scaling_ratios

