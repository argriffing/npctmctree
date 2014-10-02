from __future__ import print_function, division

import numpy as np
import networkx as nx
from numpy.testing import assert_array_equal

from npctmctree import expect

def main():

    # define the rate matrix
    Q = np.ones((4, 4))
    Q = Q - np.diag(np.diag(Q))
    exit_rates = Q.sum(axis=1)
    Q = Q - np.diag(exit_rates)
    assert_array_equal(Q, [
        [-3, 1, 1, 1],
        [1, -3, 1, 1],
        [1, 1, -3, 1],
        [1, 1, 1, -3]])

    # define the tree
    T = nx.DiGraph()
    T.add_edge(0, 1)
    root = 0
    distn = np.ones(4) / 4

    # define the single column of data
    # node 0 has 100% observation that the state is 0
    # node 1 has 100% observation that the state is 1
    data = [
        [1, 0, 0, 0],
        [0, 1, 0, 0]]
    weighted_data = [(data, 1)]

    # compute posterior expectations
    edge_to_dwell = expect.get_edge_to_dwell(
        T, root, {(0, 1) : Q}, distn, weighted_data)
    edge_to_trans = expect.get_edge_to_trans(
        T, root, {(0, 1) : Q}, distn, weighted_data)

    print('dwell time expectations:')
    print(edge_to_dwell[(0, 1)])
    print()
    print('transition count expectations:')
    print(edge_to_trans[(0, 1)])
    print()


if __name__ == '__main__':
    main()
