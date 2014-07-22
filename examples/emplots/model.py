"""
This is a helper module for code related to the evolutionary model.

"""
from __future__ import division, print_function, absolute_import

from StringIO import StringIO
import math

import networkx


def _penorm(weights):
    total = sum(weights)
    sqrt_penalty = math.log(total)
    penalty = sqrt_penalty * sqrt_penalty
    probs = [w / total for w in weights]
    return probs, penalty


class ParamManager(object):
    def __init__(self, edge_labels, nucleotide_labels='ACGT'):
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
        return self

    def set_implicit(self, edge_rates, nt_probs, kappa):
        self.edge_rates = edge_rates
        self.nt_probs, self.penalty = _penorm(nt_probs)
        self.kappa = kappa
        return self

    def set_packed(self, packed):
        params = [math.exp(p) for p in packed]
        k, n = 0, len(self.edge_labels)
        self.edge_rates = params[k:k+n]
        k, n = k+n, len(self.nucleotide_labels)
        self.nt_probs, self.penalty = _penorm(params[k:k+n])
        k, n = k+n, 1
        self.kappa, = params[k:k+n]
        return self

    def get_explicit(self):
        edge_to_rate = dict(zip(self.edge_labels, self.edge_rates))
        nt_distn = dict(zip(self.nucleotide_labels, self.nt_probs))
        return edge_to_rate, nt_distn, self.kappa, self.penalty

    def get_implicit(self):
        return self.edge_rates, self.nt_probs, self.kappa, self.penalty

    def get_packed(self):
        params = list(self.edge_rates) + list(self.nt_probs) + [self.kappa]
        packed = [math.log(p) for p in params]
        return packed, self.penalty

    def copy(self):
        # make a shallow copy of each member
        other = ParamManager(list(self.edge_labels))
        other.nucleotide_labels = list(self.nucleotide_labels))
        other.edge_rates = self.edge_rates
        #TODO unfinished

    def __str__(self):
        out = StringIO()
        print('edge rates:', file=out)
        for edge_label, edge_rate in zip(self.edge_labels, self.edge_rates):
            print(edge_label, ':', edge_rate, file=out)
        print('nucleotide distribution parameters:', file=out)
        for nt, p in zip(self.nucleotide_labels, self.nt_probs):
            print(nt, ':', p, file=out)
        print('kappa:', self.kappa, file=out)
        print('penalty:', self.penalty, file=out)
        return out.getvalue().strip()

