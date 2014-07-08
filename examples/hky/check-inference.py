"""
This script checks inference of branch lengths and HKY parameter values.

The observed data consists of the exact joint leaf state distribution.
The inference uses EM, for which the expectation step is computed exactly
and the maximization step is computed numerically.
The purpose of the EM inference is for comparison to Monte Carlo EM
for which the observed data is not exact and the conditional expectations
are computed using Monte Carlo.

"""
from __future__ import division, print_function, absolute_import


def run_inference(T, root, bfs_edges, leaves,
        data_prob_pairs,
        init_kappa, init_nt_probs, init_edge_rates,
        ):
    """
    """
    pass


def get_data_prob_pairs(T, root, bfs_edges, leaves,
        kappa, nt_probs, edge_rates):
    """
    """
    # Compute the conditional transition probability matrices on edges,
    # under the given parameter values.
    pre_Q = hkymodel.get_pre_Q(kappa, nt_probs)
    rates_out = pre_Q.sum(axis=1)
    expected_rate = nt_probs.dot(rates_out)
    Q = (pre_Q - np.diag(rates_out)) / expected_rate
    edge_to_P = {}
    for edge, edge_rate in zip(bfs_edges, edge_rates):
        edge_to_P[edge] = expm(edge_rate * Q)

    # Compute the state distribution at the leaves,
    # under the given parameter values.
    states = range(n)
    data_prob_pairs = []
    for assignment in product(states, repeat=len(leaves)):

        # Get the map from leaf to state.
        leaf_to_state = dict(zip(leaves, assignment))

        # Define the data associated with this assignment.
        # All leaf states are fully observed.
        # All internal states are completely unobserved.
        node_to_data_fvec1d = {}
        for node in leaves:
            state = leaf_to_state[node]
            fvec1d = np.zeros(n, dtype=bool)
            fvec1d[state] = True
            node_to_data_fvec1d[node] = fvec1d
        for node in internal_nodes:
            fvec1d = np.ones(n, dtype=bool)
            node_to_data_fvec1d[node] = fvec1d

        # Compute the likelihood for this data.
        lhood = get_lhood(T, edge_to_P, root, root_distn1d, node_to_data_fvec1d)
        data_prob_pairs.append((node_to_data_fvec1d, lhood))

    # Check that the computed joint distribution over leaf states
    # is actually a distribution.
    datas, probs = zip(*data_prob_pairs)
    assert_allclose(sum(probs), 1)


def main():

    # Define the rooted tree shape.
    root = 'N0'
    leaves = ('N2', 'N3', 'N4', 'N5')
    bfs_edges = [
            ('N0', 'N1'),
            ('N0', 'N2'),
            ('N0', 'N3'),
            ('N1', 'N4'),
            ('N1', 'N5'),
            ]
    T = nx.DiGraph()
    T.add_edges_from(bfs_edges)

    # Define some arbitrary 'true' parameter values.
    true_kappa = 2.4
    true_nt_probs = np.array([0.1, 0.2, 0.3, 0.4])
    true_edge_rates = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # Define parameter value guesses to be used for initializing the search.
    init_kappa = 3.0
    init_nt_probs = np.array([0.25, 0.25, 0.25, 0.25])
    init_edge_rates = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # Compute the state distribution at the leaves,
    # under the arbitrary 'true' parameter values.
    data_prob_pairs = get_data_prob_pairs(T, root, bfs_edges, leaves,
            kappa, nt_probs, edge_rates)

    # Check that the 'true' parameters can be inferred given
    # the 'true' state distribution at the leaves and arbitrary
    # initial parameter guesses.
    mle_kappa, mle_nt_probs, mle_edge_rates = run_inference(
            T, root, bfs_edges, leaves,
            data_prob_pairs,
            init_kappa, init_nt_probs, init_edge_rates)


if __name__ == '__main__':
    main()

