"""
This is a helper module for code related to the evolutionary model.

"""
from __future__ import division, print_function, absolute_import

import math


def _penorm(weights):
    total = sum(weights)
    sqrt_penalty = math.log(total)
    penalty = sqrt_penalty * sqrt_penalty
    probs = [w / total for w in weights]
    return probs, penalty


class ParamManager(object):
    def __init__(self, edge_labels, nucleotide_labels):
        self.edge_labels = edge_labels
        self.nucleotide_labels = nucleotide_labels
        self.edge_rates = None
        self.nt_probs = None
        self.kappa = None
        self.penalty = None

    def set_explicit(self, edge_to_rate, nt_distn, kappa):
        self.edge_rates = [edge_to_rate[e] for e in self.edge_labels]
        nt_weights = [nt_distn[s] for s in self.nucleotide_labels]
        self.nt_probs, self.penalty = _penorm(nt_weights)
        self.kappa = kappa

    def set_implicit(self, edge_rates, nt_probs, kappa):
        self.edge_rates = edge_rates
        self.nt_probs, self.penalty = _penorm(nt_probs)
        self.kappa = kappa

    def set_packed(self, packed):
        params = [math.exp(p) for x in packed]
        k, n = 0, len(self.edge_labels)
        self.edge_rates = params[k:k+n]
        k, n = k+n, len(self.nucleotide_labels)
        self.nt_probs, self.penalty = _penorm(params[k:k+n])
        k, n = k+n, 1
        self.kappa, = params[k:k+n]

    def get_explicit(self):
        edge_to_rate = zip(self.edge_labels, self.edge_rates)
        nt_distn = zip(self.nucleotide_labels, self.nt_probs)
        return edge_to_rate, nt_distn, self.kappa, self.penalty

    def get_implicit(self):
        return self.edge_rates, self.nt_probs, self.kappa, self.penalty

    def get_packed(self):
        params = list(self.edge_rates) + list(self.nt_probs) + [self.kappa]
        packed = [math.log(p) for p in params]
        return packed, self.penalty