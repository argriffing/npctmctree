"""
Muse-Gaut 1994 continuous-time Markov model of codon evolution.

"""
from StringIO import StringIO

import numpy as np
from numpy.testing import assert_equal
from scipy.sparse import csr_matrix, coo_matrix

import genetic

__all__ = ['MG94_Abstract', 'MG94_Concrete']


def _gen_site_changes(sa, sb):
    for a, b in zip(sa, sb):
        if a != b:
            yield a, b


# Instances are not associated with actual parameter values.
class MG94_Abstract(object):
    def __init__(self):
        nts = 'acgt'
        nt_to_idx = dict((nt, i) for i, nt in enumerate(nts))
        resids = []
        codons = []
        for line in StringIO(genetic.code).readlines():
            si, resid, codon = line.strip().split()
            if resid != 'stop':
                resids.append(resid)
                codons.append(codon)
        ncodons = len(codons)
        assert_equal(ncodons, 61)

    #TODO under construction
    def get_structural_transitions(self):
        """
        Yield (row, col) pairs corresponding to allowed transitions.

        These transitions may have zero rate for specific parameter values,
        but they are not forbidden by the structure of the model itself.

        """
        pass

    def get_state_space_size(self):
        return 61

    def instantiate(self, x=None):
        return MG94_Concrete(x)


# instances are associated with actual parameter values
# TODO After a few models have been defined,
# TODO make a base class with the common methods.
class MG94_Concrete(object):
    def __init__(self, x=None):
        """
        It is important that x can be an unconstrained vector
        that the caller does not need to know or care about.

        """
        # Unpack the parameters or use default values.
        if x is None:
            self.nt_probs = np.ones(4) / 4
            self.kappa = 2.0
            self.omega = 0.2
            self.penalty = 0
        else:
            info = self._unpack_params(x)
            self.nt_probs, self.kappa, self.omega, self.penalty = info

        # Mark some downstream attributes as not initialized.
        self._invalidate()

    def _invalidate(self):
        # this is called internally when parameter values change
        self.x = None
        self.distn = None
        self.triples = None
        self.exit_rates = None
        self.Q_sparse = None
        self.Q_dense = None

    def set_kappa(self, kappa):
        self.kappa = kappa
        self._invalidate()

    def set_omega(self, omega):
        self.omega = omega
        self._invalidate()

    def set_nt_probs(self, nt_probs):
        self.nt_probs = nt_probs
        self._invalidate()

    def get_x(self):
        if self.x is None:
            self.x = self._pack_params(self.nt_probs, self.kappa, self.omega)
        return self.x

    def _pack_params(self, nt_distn1d, kappa, omega):
        # helper function
        # This differs from the module-level function by not caring
        # about edge specific parameters.
        params = np.concatenate([nt_distn1d, [kappa, omega]])
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
        kappa, omega = params[-2:]
        return nt_distn1d, kappa, omega, penalty

    def get_canonicalization_penalty(self):
        return self.penalty

    def _process_sparse(self):
        self.distn, self.triples = _get_distn_and_triples(
                self.kappa, self.omega, self.nt_probs)
        row, col, data = zip(*self.triples)
        self.Q_sparse = coo_matrix((data, (row, col)))
        self.exit_rates = Q_sparse.sum(axis=1)

    def _process_dense(self):
        if self.Q_sparse is None:
            self._process_sparse()
        self.Q_dense = self.Q_sparse.A - np.diag(self.exit_rates)

    def get_distribution(self):
        if self.distn is None:
            self._process_sparse()
        return self.distn
    
    def get_exit_rates(self):
        if self.exit_rates is None:
            self._process_sparse()
        return self.exit_rates

    def get_sparse_rates(self):
        # return a scipy.sparse.coo_matrix with zeros on diagonals
        if self.R_sparse:
            self._process_sparse()
        return self.R_sparse

    def get_dense_rates(self):
        # return a 2d rate matrix with zeros on diagonals
        if self.R_dense is None:
            self._process_dense()
        return self.R_dense




def _get_distn_and_triples(kappa, omega, nt_probs):
    """
    Parameters
    ----------
    kappa : float
        transition/transversion rate ratio
    omega : float
        synonymous/nonsynonymous rate ratio
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
    resids = []
    codons = []
    for line in StringIO(genetic.code).readlines():
        si, resid, codon = line.strip().split()
        if resid != 'stop':
            resids.append(resid)
            codons.append(codon)
    ncodons = len(codons)
    assert_equal(ncodons, 61)

    distribution = []
    for codon in codons:
        probs = [nt_probs[nt_to_idx[nt]] for nt in codon]
        distribution.append(np.prod(probs))

    triples = []
    nt_transitions = {'ag', 'ga', 'ct', 'tc'}
    for i in range(ncodons):
        ri, ci = resids[i], codons[i]
        for j in range(ncodons):
            rj, cj = resids[j], codons[j]
            pairs = list(_gen_site_changes(ci, cj))
            if len(pairs) == 1:
                ni, nj = pairs[0]
                rate = np.prod([
                    nt_probs[nt_to_idx[nj]],
                    kappa if ni+nj in nt_transitions else 1,
                    omega if ri != rj else 1,
                    ])
                triples.append((i, j, rate))

    # scale the distribution and the rates to be friendly
    distribution = np.array(distribution) / sum(distribution)
    expected_rate = sum(distribution[i]*rate for i, j, rate in triples)
    triples = [(i, j, rate/expected_rate) for i, j, rate in triples]

    # return the distribution and triples
    return distribution, triples
