"""
Abstract base classes.

"""

__all__ = ['AbstractModel', 'ConcreteModel']


# Instances are not associated with actual parameter values.
# OK so this doesn't do so much right now...
class AbstractModel(object):
    pass


# Instances are associated with actual parameter values.
class ConcreteModel(object):
    def _invalidate(self):
        # this is called internally after parameter values change
        self.x = None
        self.distn = None
        self.triples = None
        self.exit_rates = None
        self.Q_sparse = None
        self.Q_dense = None

    def get_canonicalization_penalty(self):
        return self.penalty

    def _process_dense(self):
        if self.Q_sparse is None:
            self._process_sparse()
        self.Q_dense = self.Q_sparse.A

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
        if self.Q_sparse:
            self._process_sparse()
        return self.Q_sparse

    def get_dense_rates(self):
        # return a 2d rate matrix with zeros on diagonals
        if self.Q_dense is None:
            self._process_dense()
        return self.Q_dense
