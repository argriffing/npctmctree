"""
Plot some stuff related to EM.

The Monte Carlo EM steps that use Rao-Teh use code from
the nxctmctree/examples/check-inference.py example.

The EM steps that use exact expectations use code from
the npctmctree/examples/hky/check-inference.py example.

Check the readme.txt for more info.

ingredients:
 * Run k gillespie simulations and get summary statistics.
  - The summary statistics depend on the downstream application.
    One downstream application is EM estimation using all sufficient stats.
    Another downstream application is EM estimation using
    only leaf state patterns.
    Possibly add another object that is called on each sampled trajectory
    and which accumulates the latter summary.

"""
from __future__ import division, print_function, absolute_import

import argparse
import random

from functools import partial
from collections import defaultdict

import numpy as np
import networkx as nx

from numpy.testing import assert_allclose
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize
from scipy.special import xlogy

import npmctree
from npmctree.dynamic_lmap_lhood import get_iid_lhoods

import npctmctree
import npctmctree.hkymodel

import nxctmctree
import nxctmctree.hkymodel
from nxctmctree import gillespie
from nxctmctree.trajectory import get_node_to_tm, FullTrackSummary
from nxctmctree.likelihood import get_trajectory_log_likelihood

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as pyplot


class PlotInfo(object):
    """
    Track the information required to make the plot.

    The following notation is used here:
    fd : full data
    od : observed data

    """
    def __init__(self, iterations):
        self.iterations = iterations
        self.full_data_estimates = []
        self.observed_data_estimates = []

        # Set these manually without getters/setters.
        self.true_value = None
        self.fd_sample_mle = None
        self.od_sample_mle = None

    def add_full_data_estimate(self, value):
        self.full_data_estimates.append(value)

    def add_observed_data_estimate(self, value):
        self.observed_data_estimates.append(value)

    def _validated_iterations(self, arr):
        n = len(arr)
        if n != self.iterations:
            raise Exception('expected %s values but observed %s' % (
                self.iterations, n))
        return arr

    def get_full_data_estimates(self):
        # Return an array for plotting.
        return self._validated_iterations(self.full_data_estimates)

    def get_observed_data_estimates(self):
        # Return an array for plotting.
        return self._validated_iterations(self.observed_data_estimates)


def nx_objective(T, root, edges, full_track_summary, log_params):
    unpacked = nxctmctree.hkymodel.unpack_params(edges, log_params)
    edge_to_rate, Q, nt_distn, kappa, penalty = unpacked
    edge_to_Q = dict((e, Q) for e in edges)
    root_prior_distn = nt_distn
    log_likelihood = get_trajectory_log_likelihood(T, root,
            edge_to_Q, edge_to_rate, root_prior_distn, full_track_summary)
    return -log_likelihood + penalty


def observed_objective(T, root, edges, data_count_pairs, log_params):
    unpacked = npctmctree.hkymodel.unpack_params(edges, log_params)
    edge_rates, Q, nt_distn1d, kappa, penalty = unpacked
    edge_to_P = {}
    for edge, edge_rate in zip(edges, edge_rates):
        edge_to_P[edge] = expm(edge_rate * Q)
    node_to_data_lmaps, counts = zip(*data_count_pairs)
    lhoods = get_iid_lhoods(T, edge_to_P, root, nt_distn1d, node_to_data_lmaps)
    log_likelihood = xlogy(counts, lhoods).sum()
    return -log_likelihood + penalty


def get_value_of_interest(edge_to_rate, nt_distn, kappa):
    edge_of_interest = ('N1', 'N5')
    return edge_to_rate[edge_of_interest]


def main(args):
    random.seed(1234)

    # Define the model and the 'true' parameter values.

    # Define an edge ordering.
    edges = (
            ('N0', 'N1'),
            ('N0', 'N2'),
            ('N0', 'N3'),
            ('N1', 'N4'),
            ('N1', 'N5'),
            )

    # Define a rooted tree shape.
    T = nx.DiGraph()
    T.add_edges_from(edges)
    root = 'N0'
    leaves = ('N2', 'N3', 'N4', 'N5')

    # Define edge-specific rate scaling factors.
    # Define HKY parameter values.
    true_edge_rates = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    true_nt_probs = np.array([0.1, 0.2, 0.3, 0.4])
    true_kappa = 2.4


    # Report parameter values used for sampling.
    #print('parameter values used for sampling:')
    #print('edge to rate:', edge_to_rate)
    #print('nt distn:', nt_distn)
    #print('kappa:', kappa)
    #print()

    #print('state_to_rate:')
    #print(state_to_rate)
    #print('state_to_distn:')
    #print(state_to_distn)
    #print()

    # Do some iterations.
    plot_info = PlotInfo(args.iterations)
    for iid_iteration_idx in range(args.iterations):

        print('iteration', iid_iteration_idx+1, '...')

        # Set the current values to the true values.
        edge_rates = np.array(true_edge_rates)
        nt_probs = np.array(true_nt_probs)
        kappa = true_kappa

        # Initialize some more stuff before getting the gillespie samples.
        edge_to_rate = dict(zip(edges, edge_rates))
        edge_to_blen = dict((e, 1) for e in edges)
        Q, nt_distn = nxctmctree.hkymodel.create_rate_matrix(nt_probs, kappa)
        root_prior_distn = nt_distn
        state_to_rate, state_to_distn = gillespie.expand_Q(Q)
        edge_to_Q = dict((e, Q) for e in edges)
        edge_to_state_to_rate = dict((e, state_to_rate) for e in edges)
        edge_to_state_to_distn = dict((e, state_to_distn) for e in edges)
        node_to_tm = get_node_to_tm(T, root, edge_to_blen)
        bfs_edges = list(nx.bfs_edges(T, root))

        # Initialize the true value for plotting the horizontal line.
        if plot_info.true_value is None:
            true_value = get_value_of_interest(edge_to_rate, nt_distn, kappa)
            plot_info.true_value = true_value

        # At each iteration, sample a bunch of trajectories.
        # Accumulate a summary of each bunch of trajectories,
        # and also accumulate the leaf pattern.
        # For each bunch of trajectories, we will plot the mle
        # of the parameter of interest for each of the two summaries.
        # The mle computed from the full trajectories should be
        # more accurate than the mle computed from only the leaf
        # state patterns.
        full_track_summary = FullTrackSummary(T, root, edge_to_blen)
        pattern_to_count = defaultdict(int)
        ngillespie = args.sites * args.samples
        for track in gillespie.gen_trajectories(T, root, root_prior_distn,
                edge_to_rate, edge_to_blen, edge_to_Q, ngillespie):
            full_track_summary.on_track(track)
            pattern = tuple(track.history[v] for v in leaves)
            pattern_to_count[pattern] += 1

        # Define some initial guesses for the parameters.
        x0_edge_rates = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        x0_nt_probs = np.array([0.25, 0.25, 0.25, 0.25])
        x0_kappa = 3.0
        x0 = nxctmctree.hkymodel.pack_params(
                x0_edge_rates, x0_nt_probs, x0_kappa)
        x0 = np.array(x0)

        x_sim = nxctmctree.hkymodel.pack_params(edge_rates, nt_probs, kappa)
        x_sim = np.array(x_sim)
        print('objective function value using the sampling parameters:')
        print(nx_objective(T, root, edges, full_track_summary, x_sim))
        print()

        f = partial(nx_objective, T, root, edges, full_track_summary)
        result = minimize(f, x0, method='L-BFGS-B')

        print(result)
        log_params = result.x
        unpacked = nxctmctree.hkymodel.unpack_params(edges, log_params)
        edge_to_rate, Q, nt_distn, kappa, penalty = unpacked
        print('max likelihood estimates from sampled trajectories:')
        print('edge to rate:', edge_to_rate)
        print('nt distn:', nt_distn)
        print('kappa:', kappa)
        print('penalty:', penalty)
        print()

        # Add the maximum likelihood estimate into the plot info.
        value = get_value_of_interest(edge_to_rate, nt_distn, kappa)
        plot_info.add_full_data_estimate(value)
        if plot_info.fd_sample_mle is None:
            plot_info.fd_sample_mle = value

        # mle using only observations at leaves

        # make the node_to_data_lmaps
        #TODO add a utility function for this
        data_count_pairs = []
        nt_to_state = dict((s, i) for i, s in enumerate('ACGT'))
        nstates = len(nt_to_state)
        for pattern, count in pattern_to_count.items():
            node_to_data_lmap = {}
            for node, nt in zip(leaves, pattern):
                state = nt_to_state[nt]
                lmap = np.zeros(nstates, dtype=float)
                lmap[state] = 1
                node_to_data_lmap[node] = lmap
            for node in set(T) - set(leaves):
                lmap = np.ones(nstates, dtype=float)
                node_to_data_lmap[node] = lmap
            data_count_pairs.append((node_to_data_lmap, count))

        # Define some initial guesses for the parameters.
        x0_edge_rates = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        x0_nt_probs = np.array([0.25, 0.25, 0.25, 0.25])
        x0_kappa = 3.0
        x0 = nxctmctree.hkymodel.pack_params(
                x0_edge_rates, x0_nt_probs, x0_kappa)
        x0 = np.array(x0)

        f = partial(observed_objective, T, root, edges, data_count_pairs)
        result = minimize(f, x0, method='L-BFGS-B')

        print(result)
        log_params = result.x
        unpacked = nxctmctree.hkymodel.unpack_params(edges, log_params)
        edge_to_rate, Q, nt_distn, kappa, penalty = unpacked
        print('max likelihood estimates from sampled trajectories:')
        print('edge to rate:', edge_to_rate)
        print('nt distn:', nt_distn)
        print('kappa:', kappa)
        print('penalty:', penalty)
        print()

        # Add the maximum likelihood estimate into the plot info.
        value = get_value_of_interest(edge_to_rate, nt_distn, kappa)
        plot_info.add_observed_data_estimate(value)
        if plot_info.od_sample_mle is None:
            plot_info.od_sample_mle = value

    # Draw the plot.
    # Patterned on ctmczoo/two-state.py

    # define some color styles corresponding to reduced information
    exact_color = 'black'
    fd_color = 'slateblue'
    od_color = 'skyblue'

    # draw the plot
    fix, ax = pyplot.subplots()
    ts = range(1, args.iterations+1)
    ax.set_ylim([0.3, 0.7])
    ax.set_xlim([min(ts), max(ts)])
    ax.axhline(plot_info.true_value, color=exact_color, linestyle='-',
            label='parameter value used for simulation')
    ax.axhline(plot_info.fd_sample_mle, color=fd_color, linestyle='-',
            label='initial full data MLE')
    ax.axhline(plot_info.od_sample_mle, color=od_color, linestyle='-',
            label='initial observed data MLE')
    ax.plot(ts, plot_info.get_full_data_estimates(),
            color=fd_color, linestyle=':',
            label='iid full data MLEs')
    ax.plot(ts, plot_info.get_observed_data_estimates(),
            color=od_color, linestyle=':',
            label='iid observed data MLEs')
    legend = ax.legend(loc='upper center')
    pyplot.savefig('monte-carlo-estimates-c.png')


def unused():
    #TODO put this back into the loop.

    # Define initial parameter values for the expectation maximization.
    x0_edge_rates = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    x0_nt_probs = np.array([0.25, 0.25, 0.25, 0.25])
    x0_kappa = 3.0
    x0 = nxctmctree.hkymodel.pack_params(x0_edge_rates, x0_nt_probs, x0_kappa)
    x0 = np.array(x0)
    packed = x0
    unpacked = nxctmctree.hkymodel.unpack_params(edges, x0)
    edge_to_rate, Q, nt_distn, kappa, penalty = unpacked
    edge_to_Q = dict((e, Q) for e in edges)
    root_prior_distn = nt_distn

    # Do some burn-in samples for each pattern,
    # using the initial parameter values.
    # Do not store summaries of these sampled trajectories.
    nburn = 10
    pattern_to_track = {}
    pattern_to_data = {}
    set_of_all_states = set('ACGT')
    for idx, pattern in enumerate(pattern_to_count):
        #print('burning in the trajectory',
                #'for pattern', idx+1, 'of', npatterns, '...')
        
        # Create the data representation.
        leaf_to_state = dict(zip(leaves, pattern))
        node_to_data_fset = {}
        for node in T:
            if node in leaves:
                fset = {leaf_to_state[node]}
            else:
                fset = set_of_all_states
            node_to_data_fset[node] = fset

        # Save the data representation constructed for each pattern.
        pattern_to_data[pattern] = node_to_data_fset

        # Create and burn in the track.
        track = None
        for updated_track in raoteh.gen_raoteh_trajectories(
                T, edge_to_Q, root, root_prior_distn, node_to_data_fset,
                edge_to_blen, edge_to_rate,
                set_of_all_states, initial_track=track, ntrajectories=nburn):
            track = updated_track

        # Add the track.
        pattern_to_track[pattern] = track

    # Do some EM iterations.
    for em_iteration_index in itertools.count():
        print('starting EM iteration', em_iteration_index+1, '...')

        # Each EM iteration gets its own summary object.
        full_track_summary = FullTrackSummary(T, root, edge_to_blen)

        # Do a few Rao-Teh samples for each pattern within each EM iteration.
        for idx, (pattern, track) in enumerate(pattern_to_track.items()):
            #print('sampling Rao-Teh trajectories for pattern', idx+1, '...')
            count = pattern_to_count[pattern]
            node_to_data_fset = pattern_to_data[pattern]

            # Note that the track is actually updated in-place
            # even though the track object is yielded at each iteration.
            for updated_track in raoteh.gen_raoteh_trajectories(
                    T, edge_to_Q, root, root_prior_distn, node_to_data_fset,
                    edge_to_blen, edge_to_rate, set_of_all_states,
                    initial_track=track, ntrajectories=count):
                full_track_summary.on_track(updated_track)

        # This is the M step of EM.
        f = partial(nx_objective, T, root, edges, full_track_summary)
        result = minimize(f, packed, method='L-BFGS-B')
        #print(result)
        packed = result.x
        unpacked = nxctmcttree.hkymodel.unpack_params(edges, packed)
        edge_to_rate, Q, nt_distn, kappa, penalty = unpacked
        print('max likelihood estimates from sampled trajectories:')
        print('penalized negative log likelihood:', result.fun)
        print('edge to rate:', edge_to_rate)
        print('nt distn:', nt_distn)
        print('kappa:', kappa)
        print('penalty:', penalty)
        print()


#NOTE from npctmctree/examples/hky/check-inference
def hky_check_inference_objective(T, root, edges,
        root_state_counts, edge_to_dwell_times, edge_to_transition_counts,
        log_params):
    """
    Negative expected log likelihood.

    It is penalized if the nucleotide probabilities do not add up to 1.
    The nucleotide penalties are already forced to be positive
    using a transformation of variables.

    """
    unpacked = hkymodel.unpack_params(edges, log_params)
    edge_rates, Q, nt_distn1d, kappa, penalty = unpacked
    edge_to_rate = dict(zip(edges, edge_rates))
    edge_to_Q = dict((e, Q) for e in edges)
    root_prior_distn1d = nt_distn1d
    log_likelihood = expect.get_expected_log_likelihood(
            T, root, edges,
            edge_to_Q, edge_to_rate, root_prior_distn1d,
            root_state_counts, edge_to_dwell_times, edge_to_transition_counts)
    penalized_neg_ll = -log_likelihood + penalty
    return penalized_neg_ll


#NOTE from npctmctree/examples/hky/check-inference
def hky_check_inference_run_inference(T, root, bfs_edges, leaves,
        data_weight_pairs,
        kappa, nt_distn1d, edge_rates,
        max_iterations,
        ):
    """
    Run the inference.

    Parameters
    ----------
    T : networkx DiGraph
        Rooted tree.
    root : hashable node
        Root of the tree as a hashable networkx node in T.
    bfs_edges : sequence
        Ordered edges in a preorder from the root.
        Each edge is a directed pair of nodes.
    leaves : sequence
        Leaf nodes.
    data_weight_pairs : sequence of pairs
        Weighted data or simulated data or an exact distribution.
    kappa : float
        Initial guess for kappa parameter.
        Kappa is the rate scaling ratio of transitions to transversions.
        Nucleotide substitutions A <--> G and C <--> T are called transitions,
        while all other nucleotide substitutions are called transversions.
    nt_distn1d : 1d ndarray of floats
        Initial guess for mutational nucleotide distribution.
    edge_rates : 1d ndarray of floats
        Initial guess for the edge rate scaling factors.
    max_iterations : integer or None
        Optionally limit the number of iterations.

    Returns
    -------
    mle : (mle_kappa, mle_nt_distn1d, mle_edge_rates)
        Maximum likelihood estimates.

    """
    # Look at nxctmctree for a template for the full MLE.
    nstates = nt_distn1d.shape[0]

    for iteration_idx in itertools.count():

        # Check early stop condition.
        if max_iterations is not None:
            if iteration_idx >= max_iterations:
                break

        # Report the EM iteration underway.
        print('iteration', iteration_idx+1, '...')

        # Use the unpacked parameters to create the carefullly scaled
        # transition rate matrix.
        Q = hkymodel.get_normalized_Q(kappa, nt_distn1d)

        # Create the edge specific rate matrices,
        # carefully scaled by the edge-specific rate scaling factors.
        edge_to_Q = {}
        for edge, edge_rate in zip(bfs_edges, edge_rates):
            edge_to_Q[edge] = edge_rate * Q

        # Get posterior expected root distribution.
        root_prior_distn1d = nt_distn1d
        edge_to_P = dict((e, expm(Q)) for e, Q in edge_to_Q.items())
        root_state_counts = np.zeros(nstates)
        for data, weight in data_weight_pairs:
            node_to_distn1d = dynamic_fset_lhood.get_node_to_distn1d(
                    T, edge_to_P, root, root_prior_distn1d, data)
            root_post_distn1d = node_to_distn1d[root]
            root_state_counts += weight * root_post_distn1d

        # Get posterior expected dwell times and transition counts.
        edge_to_dwell_times = expect.get_edge_to_dwell(
                T, root, edge_to_Q, root_prior_distn1d, data_weight_pairs)
        edge_to_transition_counts = expect.get_edge_to_trans(
                T, root, edge_to_Q, root_prior_distn1d, data_weight_pairs)

        # Maximization step of EM.
        f = partial(objective, T, root, bfs_edges,
                root_state_counts,
                edge_to_dwell_times,
                edge_to_transition_counts)
        x0 = hkymodel.pack_params(edge_rates, nt_distn1d, kappa)
        result = minimize(f, x0, method='L-BFGS-B')

        # Unpack optimization output.
        log_params = result.x
        unpacked = hkymodel.unpack_params(bfs_edges, log_params)
        edge_rates, Q, nt_distn1d, kappa, penalty = unpacked

        # Summarize the EM step.
        edge_to_rate = dict(zip(bfs_edges, edge_rates))
        print('EM step summary:')
        print('objective function value:', result.fun)
        for edge, rate in zip(bfs_edges, edge_rates):
            print('edge:', edge, 'rate:', rate)
        print('nucleotide distribution:', nt_distn1d)
        print('kappa:', kappa)
        print('penalty:', penalty)
        print()

    # Return the maximum likelihood estimates computed with EM.
    return kappa, nt_distn1d, edge_rates


def hky_check_inference_main(args):

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

    # Compute the map from edge to transition probability matrix,
    # under the true parameter values.
    Q = hkymodel.get_normalized_Q(true_kappa, true_nt_probs)
    edge_to_P = {}
    for edge, edge_rate in zip(bfs_edges, true_edge_rates):
        edge_to_P[edge] = expm(edge_rate * Q)

    # Compute the state distribution at the leaves,
    # under the arbitrary 'true' parameter values.
    root_prior_distn1d = true_nt_probs
    data_prob_pairs = dynamic_fset_lhood.get_unconditional_joint_distn(
            T, edge_to_P, root, root_prior_distn1d, leaves)

    # Check that the computed joint distribution over leaf states
    # is actually a distribution.
    sites, probs = zip(*data_prob_pairs)
    assert_allclose(sum(probs), 1)

    # Check that the 'true' parameters can be inferred given
    # the 'true' state distribution at the leaves and arbitrary
    # initial parameter guesses.
    mle_kappa, mle_nt_probs, mle_edge_rates = run_inference(
            T, root, bfs_edges, leaves,
            data_prob_pairs,
            init_kappa, init_nt_probs, init_edge_rates,
            args.iterations,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100,
            help='number of iterations of EM')
    parser.add_argument('--sites', type=int, default=1000,
            help='number of iid observations')
    parser.add_argument('--burnin', type=int, default=10,
            help=('number of samples of burn-in '
                'per site per EM iteration, for Rao-Teh sampling'))
    parser.add_argument('--samples', type=int, default=1,
            help=('number of sampled trajectories '
                'per site per EM iteration, for Rao-Teh sampling'))
    args = parser.parse_args()
    main(args)

