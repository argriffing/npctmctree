"""
Plot some stuff related to EM.

This is very much like the nxctmctree/examples/check-inference.py example.
Check the readme.txt for more info.

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

    """
    def __init__(self, iterations, true_value):
        self.iterations = iterations
        self.true_value = true_value
        self.full_data_estimates = []
        self.observed_data_estimates = []

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

    def get_true_value_array(self):
        # Return an array for plotting.
        return np.ones(self.iterations) * self.true_value

    def get_full_data_estimates(self):
        # Return an array for plotting.
        return self._validated_iterations(self.full_data_estimates)

    def get_observed_data_estimates(self):
        # Return an array for plotting.
        return self._validated_iterations(self.observed_data_estimates)


def objective(T, root, edges, full_track_summary, log_params):
    unpacked = nxctmctree.hkymodel.unpack_params(edges, log_params)
    edge_to_rate, Q, nt_distn, kappa, penalty = unpacked
    edge_to_Q = dict((e, Q) for e in edges)
    root_prior_distn = nt_distn
    log_likelihood = get_trajectory_log_likelihood(T, root,
            edge_to_Q, edge_to_rate, root_prior_distn, full_track_summary)
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
    plot_info = None
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

        # Initialize the info for plotting.
        if plot_info is None:
            true_value = get_value_of_interest(edge_to_rate, nt_distn, kappa)
            plot_info = PlotInfo(args.iterations, true_value)

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
        x0 = nxctmctree.hkymodel.pack_params(x0_edge_rates, x0_nt_probs, x0_kappa)
        x0 = np.array(x0)

        x_sim = nxctmctree.hkymodel.pack_params(edge_rates, nt_probs, kappa)
        x_sim = np.array(x_sim)
        print('objective function value using the sampling parameters:')
        print(objective(T, root, edges, full_track_summary, x_sim))
        print()

        f = partial(objective, T, root, edges, full_track_summary)
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

    # Draw the plot.
    # Patterned on ctmczoo/two-state.py
    fix, ax = pyplot.subplots()
    ts = range(1, args.iterations+1)
    ax.set_ylim([0.3, 0.7])
    ax.plot(ts, plot_info.get_true_value_array(),
            'k--', label='value used for sampling')
    ax.plot(ts, plot_info.get_full_data_estimates(),
            'k:', label='full data estimate')
    legend = ax.legend(loc='upper center')
    pyplot.savefig('monte-carlo-estimates.png')


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
        f = partial(objective, T, root, edges, full_track_summary)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=10,
            help='number of iterations of EM')
    parser.add_argument('--sites', type=int, default=10000,
            help='number of iid observations')
    parser.add_argument('--burnin', type=int, default=10,
            help=('number of samples of burn-in '
                'per site per EM iteration, for Rao-Teh sampling'))
    parser.add_argument('--samples', type=int, default=10,
            help=('number of sampled trajectories '
                'per site per EM iteration, for Rao-Teh sampling'))
    args = parser.parse_args()
    main(args)

