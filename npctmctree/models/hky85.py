"""
Hasegawa-Kishino-Yano 1985 continuous-time Markov model of nucleotide evolution.

"""
from StringIO import StringIO

import numpy as np
from numpy.testing import assert_equal
from scipy.sparse import csr_matrix, coo_matrix

from .base import AbstractModel, ConcreteModel

__all__ = ['HKY85_Abstract', 'HKY85_Concrete']


# Instances are not associated with actual parameter values.
class HKY85_Abstract(AbstractModel):
    def get_state_space_size(self):
        return 4

    def instantiate(self, x=None):
        return HKY85_Concrete(x)


# instances are associated with actual parameter values
class HKY85_Concrete(ConcreteModel):
    def __init__(self, x=None):
        """
        It is important that x can be an unconstrained vector
        that the caller does not need to know or care about.

        """
        # Unpack the parameters or use default values.
        if x is None:
            self.nt_probs = np.ones(4) / 4
            self.kappa = 2.0
            self.penalty = 0
        else:
            info = self._unpack_params(x)
            self.nt_probs, self.kappa, self.penalty = info

        # Mark some downstream attributes as not initialized.
        self._invalidate()

    def set_kappa(self, kappa):
        self.kappa = kappa
        self._invalidate()

    def set_nt_probs(self, nt_probs):
        self.nt_probs = nt_probs
        self._invalidate()

    def get_x(self):
        if self.x is None:
            self.x = self._pack_params(self.nt_probs, self.kappa)
        return self.x

    def _pack_params(self, nt_distn1d, kappa):
        # helper function
        # This differs from the module-level function by not caring
        # about edge specific parameters.
        params = np.concatenate([nt_distn1d, [kappa]])
        log_params = np.log(params)
        return log_params

    def _unpack_params(self, log_params):
        # helper function
        # This differs from the module-level function by not caring
        # about edge specific parameters, and it does not create
        # the rate matrix.
        assert_equal(len(log_params.shape), 1)
        params = np.exp(log_params)
        nt_distn1d = params[:4]
        penalty = np.square(np.log(nt_distn1d.sum()))
        nt_distn1d = nt_distn1d / nt_distn1d.sum()
        kappa = params[-1:]
        return nt_distn1d, kappa, penalty

    def _process_sparse(self):
        self.distn, self.triples = _get_distn_and_triples(
                self.kappa, self.nt_probs)
        row, col, data = zip(*self.triples)
        self.Q_sparse = coo_matrix((data, (row, col)))
        self.exit_rates = Q_sparse.sum(axis=1)


def _get_distn_and_triples(kappa, nt_probs):
    """
    Parameters
    ----------
    kappa : float
        transition/transversion rate ratio
    nt_probs : sequence of floats
        sequence of four probabilities in acgt order

    Returns
    -------
    distribution : sequence of floats
        stationary distribution
    triples : sequence of triples
        each triple is a (row, col, rate) triple

    """
    nts = 'acgt'
    nt_to_idx = dict((nt, i) for i, nt in enumerate(nts))

    distribution = nt_probs

    triples = []
    nt_transitions = {'ag', 'ga', 'ct', 'tc'}
    for i, ni in enumerate(nts):
        for j, nj in enumerate(nts):
            if i != j:
                rate = np.prod([
                    nt_probs[j],
                    kappa if ni+nj in nt_transitions else 1,
                    ])
                triples.append((i, j, rate))

    # scale the distribution and the rates to be friendly
    distribution = np.array(distribution) / sum(distribution)
    expected_rate = sum(distribution[i]*rate for i, j, rate in triples)
    triples = [(i, j, rate/expected_rate) for i, j, rate in triples]

    # return the distribution and triples
    return distribution, triples
