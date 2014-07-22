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

For parameter management, the notation pman is used for the managed parameters.
This includes implicit, explicit, and packed representations of
the parameter list consisting of edge rates, nucleotide distribution,
and kappa, and it manages parameter transformation and penalization
to deal with constraints (e.g. parameters constrained to be positive
and sets of parameters constrained to sum to 1).

"""
from __future__ import division, print_function, absolute_import

from StringIO import StringIO
import argparse
import random

from functools import partial

import numpy as np
import networkx as nx

from numpy.testing import assert_allclose, assert_array_less, assert_equal
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize
from scipy.special import xlogy

import npmctree
from npmctree import dynamic_fset_lhood, dynamic_xmap_lhood
from npmctree.util import xmap_to_lmap

import npctmctree
import npctmctree.hkymodel
from npctmctree import expect

import nxctmctree
import nxctmctree.hkymodel
from nxctmctree import gillespie, raoteh
from nxctmctree.trajectory import get_node_to_tm
from nxctmctree.trajectory import FullTrackSummary, NodeStateSummary
from nxctmctree.likelihood import get_trajectory_log_likelihood

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as pyplot

from model import ParamManager


#TODO put this somewhere more useful
def xmap_to_fmap(all_nodes, all_states, xmap, validate=True):
    """
    Convert from a more restricted to a less restricted observation format.

    """
    all_states = set(all_states)
    all_nodes = set(all_nodes)
    observed_nodes = set(xmap)
    hidden_nodes = all_nodes - observed_nodes
    if validate:
        extra_nodes = observed_nodes - all_nodes
        if extra_nodes:
            raise ValueError('extra nodes: %s' % extra_nodes)
        observed_states = set(xmap.values())
        extra_states = observed_states - all_states
        if extra_states:
            raise ValueError('extra states: %s' % extra_states)
    node_to_fset = {}
    for node, state in xmap.items():
        node_to_fset[node] = {state}
    for node in hidden_nodes:
        node_to_fset[node] = all_states
    return node_to_fset


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
        self.exact_em_estimates = []
        self.stochastic_em_estimates = []

        # Set these manually without getters/setters.
        self.true_value = None
        self.fd_sample_mle = None
        self.od_sample_mle = None

    def add_full_data_estimate(self, value):
        self.full_data_estimates.append(value)

    def add_observed_data_estimate(self, value):
        self.observed_data_estimates.append(value)

    def add_exact_em_estimate(self, value):
        self.exact_em_estimates.append(value)

    def add_stochastic_em_estimate(self, value):
        self.stochastic_em_estimates.append(value)

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

    def get_exact_em_estimates(self):
        # Return an array for plotting.
        return self._validated_iterations(self.exact_em_estimates)

    def get_stochastic_em_estimates(self):
        # Return an array for plotting.
        return self._validated_iterations(self.stochastic_em_estimates)


class OptimizationRunner(object):
    def __init__(self, f, true_pman, guess_pman):
        self.f = f
        self.true_pman = true_pman
        self.guess_pman = guess_pman
        self.raw_search_result = None
        self.opt_pman = None
        self.guess_obj = None
        self.true_obj = None
        self.opt_obj = None

    def run(self):
        true_packed, true_penalty = self.true_pman.get_packed()
        guess_packed, guess_penalty = self.guess_pman.get_packed()
        edges = self.true_pman.edge_labels

        # Get the objective function value for some parameter values.
        self.true_obj = self.f(true_packed) + true_penalty
        self.guess_obj = self.f(guess_packed) + guess_penalty

        # Get the max likelihood parameter values.
        result = minimize(self.f, guess_packed, method='L-BFGS-B')
        self.raw_search_result = result
        self.opt_pman = ParamManager(edges).set_packed(result.x)
        self.opt_obj = result.fun

        # Sanity checks.
        opt_packed, opt_penalty = self.opt_pman.get_packed()
        assert_allclose(self.f(opt_packed) + opt_penalty, self.opt_obj)
        assert_array_less(self.opt_obj, self.true_obj)
        assert_array_less(self.opt_obj, self.guess_obj)

        return self

    def __str__(self):
        out = StringIO()

        # Report the objective value using the sampling parameters.
        print('objective values:', file=out)
        print('using initial guess parameters:', self.guess_obj, file=out)
        print('using actual sampling parameters:', self.true_obj, file=out)
        print('using mle parameters:', self.opt_obj, file=out)
        print(file=out)

        # Report raw optimization output.
        print('raw optimization search result:', file=out)
        print(self.raw_search_result, file=out)
        print(file=out)

        # Report max likelihood parameter estimates.
        print('max likelihood estimates:', file=out)
        print(self.opt_pman, file=out)
        print(file=out)

        return out.getvalue().strip()


def full_objective(T, root, edges, full_track_summary, log_params):
    pman = ParamManager(edge_labels=edges).set_packed(log_params)
    edge_to_rate, nt_distn, kappa, penalty = pman.get_explicit()
    Q = npctmctree.hkymodel.get_nx_Q(kappa, nt_distn)
    edge_to_Q = dict((e, Q) for e in edges)
    root_prior_distn = nt_distn
    log_likelihood = get_trajectory_log_likelihood(T, root,
            edge_to_Q, edge_to_rate, root_prior_distn, full_track_summary)
    return -log_likelihood + penalty


def observed_objective(T, root, edges, data_count_pairs, log_params):
    pman = ParamManager(edge_labels=edges).set_packed(log_params)
    edge_rates, nt_probs, kappa, penalty = pman.get_implicit()
    nt_distn1d = np.array(nt_probs)
    Q = npctmctree.hkymodel.get_normalized_Q(kappa, nt_distn1d)
    edge_to_P = {}
    for edge, edge_rate in zip(edges, edge_rates):
        edge_to_P[edge] = expm(edge_rate * Q)
    xmaps, counts = zip(*data_count_pairs)
    lhoods = dynamic_xmap_lhood.get_iid_lhoods(
            T, edge_to_P, root, nt_distn1d, xmaps)
    log_likelihood = xlogy(counts, lhoods).sum()
    return -log_likelihood + penalty


def exact_em_objective(T, root, edges,
        root_state_counts, edge_to_dwell_times, edge_to_transition_counts,
        log_params):
    """
    Penalized negative expected log likelihood.

    It is penalized if the nucleotide probabilities do not add up to 1.
    The nucleotide penalties are already forced to be positive
    using a transformation of variables.

    """
    pman = ParamManager(edge_labels=edges).set_packed(log_params)
    edge_rates, nt_probs, kappa, penalty = pman.get_implicit()
    nt_distn1d = np.array(nt_probs)
    Q = npctmctree.hkymodel.get_normalized_Q(kappa, nt_distn1d)
    edge_to_Q = dict((e, Q) for e in edges)
    edge_to_rate = dict(zip(edges, edge_rates))
    root_prior_distn1d = nt_distn1d
    log_likelihood = expect.get_expected_log_likelihood(
            T, root, edges,
            edge_to_Q, edge_to_rate, root_prior_distn1d,
            root_state_counts, edge_to_dwell_times, edge_to_transition_counts)
    penalized_neg_ll = -log_likelihood + penalty
    return penalized_neg_ll


def get_value_of_interest(pman):
    edge_to_rate, nt_distn, kappa, penalty = pman.get_explicit()
    edge_of_interest = ('N1', 'N5')
    return edge_to_rate[edge_of_interest]


def gen_unconditional_tracks(T, root, pman, ntracks):
    """
    Sample and summarize a few trajectories.

    Parameters
    ----------
    T : networkx DiGraph
        The shape of the tree.
    root : hashable
        Root node of the tree.
    pman : ParamManager object
        Has information about the edge rates, nt distn, and kappa parameter.
    ntracks : integer
        Number of requested iid track samples.

    """
    edge_to_rate, nt_distn, kappa, penalty = pman.get_explicit()
    Q = npctmctree.hkymodel.get_nx_Q(kappa, nt_distn)
    edges = list(T.edges())
    edge_to_Q = dict((e, Q) for e in edges)
    edge_to_blen = dict((e, 1) for e in edges)
    for track in gillespie.gen_trajectories(T, root, nt_distn,
            edge_to_rate, edge_to_blen, edge_to_Q, ntracks):
        yield track


def main(args):
    random.seed(1234)

    # Define the shape of the tree.
    # This shape remains constant across the entire analysis.
    edges = (
            ('N0', 'N1'),
            ('N0', 'N2'),
            ('N0', 'N3'),
            ('N1', 'N4'),
            ('N1', 'N5'),
            )
    T = nx.DiGraph()
    T.add_edges_from(edges)
    root = 'N0'
    leaves = ('N2', 'N3', 'N4', 'N5')
    edge_to_blen = dict((e, 1) for e in edges)
    nt_to_state = dict((i, s) for s, i in enumerate('ACGT'))
    nstates = 4

    # Define 'managed' parameter values used for simulation.
    true_pman = ParamManager(edges).set_implicit(
            [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4], 2.4)

    # Define an arbitrary bad guess as 'managed' parameter values.
    guess_pman = ParamManager(edges).set_implicit(
            [0.2, 0.2, 0.2, 0.2, 0.2], [0.25, 0.25, 0.25, 0.25], 3.0)

    # Initialize the plot info.
    plot_info = PlotInfo(args.iterations)

    # Set the true value of interest in the plot.
    plot_info.true_value = get_value_of_interest(true_pman)

    # For each iteration, independently sample an alignment of iid sites.
    for iid_iteration_idx in range(args.iterations):

        print('iteration', iid_iteration_idx+1, '...')

        # Accumulate a summary of each bunch of trajectories,
        # and also accumulate the leaf pattern.
        # For each bunch of trajectories, we will plot the mle
        # of the parameter of interest for each of the two summaries.
        # The mle computed from the full trajectories should be
        # more accurate than the mle computed from only the leaf
        # state patterns.
        trajectory_summary = FullTrackSummary(T, root, edge_to_blen)
        leaf_state_summary = NodeStateSummary(leaves)
        for track in gen_unconditional_tracks(T, root, true_pman, args.sites):
            for summary in trajectory_summary, leaf_state_summary:
                summary.on_track(track)

        # Compute MLE for full track data.
        f = partial(full_objective, T, root, edges, trajectory_summary)
        runner = OptimizationRunner(f, true_pman, guess_pman).run()
        print('MLE for fully observed sampled trajectories:')
        print(runner)
        print()

        # Add MLE for full track data to the plot info.
        value = get_value_of_interest(runner.opt_pman)
        plot_info.add_full_data_estimate(value)

        # Compute MLE for leaf observations.
        data_count_pairs = []
        for xmap, count in leaf_state_summary.gen_xmap_count_pairs():
            xmap = dict((node, nt_to_state[nt]) for node, nt in xmap.items())
            data_count_pairs.append((xmap, count))
        f = partial(observed_objective, T, root, edges, data_count_pairs)
        runner = OptimizationRunner(f, true_pman, guess_pman).run()
        print('MLE for leaf-restricted observations in sampled trajectories:')
        print(runner)
        print()

        # Add the leaf data maximum likelihood estimate into the plot info.
        value = get_value_of_interest(runner.opt_pman)
        plot_info.add_observed_data_estimate(value)

    # Sample a single alignment, for the purposes of examining Monte Carlo EM.
    print('Sampling a single alignment for testing Monte Carlo EM...')
    trajectory_summary = FullTrackSummary(T, root, edge_to_blen)
    leaf_state_summary = NodeStateSummary(leaves)
    for track in gen_unconditional_tracks(T, root, true_pman, args.sites):
        for summary in trajectory_summary, leaf_state_summary:
            summary.on_track(track)

    # Compute max likelihood estimates
    # using the full data along the entire trajectory.
    f = partial(full_objective, T, root, edges, trajectory_summary)
    runner = OptimizationRunner(f, true_pman, guess_pman).run()
    print('MLE for fully observed sampled trajectories:')
    print(runner)
    print()
    plot_info.fd_sample_mle = get_value_of_interest(runner.opt_pman)

    # Compute max likelihood estimates
    # using only the observed data at the leaves.
    data_count_pairs = []
    for xmap, count in leaf_state_summary.gen_xmap_count_pairs():
        xmap = dict((node, nt_to_state[nt]) for node, nt in xmap.items())
        data_count_pairs.append((xmap, count))
    f = partial(observed_objective, T, root, edges, data_count_pairs)
    runner = OptimizationRunner(f, true_pman, guess_pman).run()
    print('MLE for leaf-restricted observations in sampled trajectories:')
    print(runner)
    print()
    plot_info.od_sample_mle = get_value_of_interest(runner.opt_pman)

    # Convert xmap count pairs to node_to_lmap count pairs.
    # The more general observation model is used for exact expectations.
    data_weight_pairs = []
    all_nodes = set(T)
    for xmap, count in data_count_pairs:
        print(xmap)
        lmap = xmap_to_lmap(all_nodes, nstates, xmap)
        data_weight_pairs.append((lmap, count))

    # Add the initial parameter value guess to the list of EM estimates.
    value = get_value_of_interest(guess_pman)
    plot_info.add_exact_em_estimate(value)

    # Initialize the current parameters for the EM iteration.
    curr_pman = guess_pman.copy()

    # Do some iterations of EM using exact expectations.
    for em_iteration_idx in range(args.iterations-1):
        print('exact em iteration', em_iteration_idx+1, '...')

        # Unpack the current parameter values.
        edge_rates, nt_probs, kappa, penalty = curr_pman.get_implicit()
        nt_distn1d = np.array(nt_probs)
        Q = npctmctree.hkymodel.get_normalized_Q(kappa, nt_distn1d)
        edge_to_rate = dict(zip(edges, edge_rates))

        # Create the edge specific rate matrices,
        # carefully scaled by the edge-specific rate scaling factors.
        edge_to_Q = {}
        for edge, edge_rate in zip(edges, edge_rates):
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
        f = partial(exact_em_objective, T, root, edges,
                root_state_counts,
                edge_to_dwell_times,
                edge_to_transition_counts)
        runner = OptimizationRunner(f, true_pman, curr_pman).run()
        print('EM maximization step optimization info:')
        print(runner)
        print()

        # Update the plot.
        value = get_value_of_interest(runner.opt_pman)
        plot_info.add_exact_em_estimate(value)

        # Update the current parameter values.
        curr_pman = runner.opt_pman

    # Use data-conditioned Rao-Teh track samples for Monte Carlo EM.
    # For each Monte Carlo EM iteration, for each site,
    # begin with the corresponding track from the previous iteration,
    # and run a few burn-in Rao-Teh iterations.
    # Use only the last Rao-Teh track for the summary.

    # Initialize tracks.
    track_list = [None] * args.sites

    # Initialize node to fset data for each site using leaf summary.
    data_list = []
    all_nodes = set(T)
    all_states = set('ACGT')
    for xmap in leaf_state_summary.gen_xmaps_with_repetition():
        node_to_fset = xmap_to_fmap(all_nodes, all_states, xmap)
        data_list.append(node_to_fset)

    # Add the initial parameter value guess to the list of EM estimates.
    value = get_value_of_interest(guess_pman)
    plot_info.add_stochastic_em_estimate(value)

    # Initialize the current parameters for the EM iteration.
    curr_pman = guess_pman.copy()

    # Sanity checking.
    assert_equal(len(track_list), len(data_list))

    for em_iteration_idx in range(args.iterations-1):
        print('stochastic em iteration', em_iteration_idx+1, '...')

        # Unpack the parameter values to be used in this EM iteration.
        edge_to_rate, nt_distn, kappa, penalty = curr_pman.get_explicit()
        Q = npctmctree.hkymodel.get_nx_Q(kappa, nt_distn)
        edges = list(T.edges())
        edge_to_Q = dict((e, Q) for e in edges)
        edge_to_blen = dict((e, 1) for e in edges)
        root_prior_distn = nt_distn

        # Initialize the summary object for the EM iteration.
        trajectory_summary = FullTrackSummary(T, root, edge_to_blen)

        # For each track, burn in a few Rao-Teh iterations
        # and take a single track sample.
        next_track_list = []
        for track, node_to_data_fset in zip(track_list, data_list):
            
            # Burn in a few iterations.
            for updated_track in raoteh.gen_raoteh_trajectories(
                    T, edge_to_Q, root, root_prior_distn, node_to_data_fset,
                    edge_to_blen, edge_to_rate, all_states,
                    initial_track=track, ntrajectories=args.burnin):
                track = updated_track

            # Sample a single extra track.
            for updated_track in raoteh.gen_raoteh_trajectories(
                    T, edge_to_Q, root, root_prior_distn, node_to_data_fset,
                    edge_to_blen, edge_to_rate, all_states,
                    initial_track=track, ntrajectories=1):
                track = updated_track

            # Summarize the track and add it to the list.
            trajectory_summary.on_track(track)
            next_track_list.append(track)

        # Compute MLE for full track data.
        f = partial(full_objective, T, root, edges, trajectory_summary)
        runner = OptimizationRunner(f, true_pman, curr_pman).run()
        print('MLE for Monte Carlo EM trajectories:')
        print(runner)
        print()

        # Add MLE for full track data to the plot info.
        value = get_value_of_interest(runner.opt_pman)
        plot_info.add_stochastic_em_estimate(value)

        # Update the parameter values for the next EM iteration.
        curr_pman = runner.opt_pman

    # Draw the plot.
    # Patterned on ctmczoo/two-state.py

    # define some color styles corresponding to reduced information
    exact_color = 'black'
    fd_color = 'slateblue'
    od_color = 'skyblue'
    exact_em_color = 'green'
    stoch_em_color = 'limegreen'

    # draw the plot
    fix, ax = pyplot.subplots()
    ts = range(1, args.iterations+1)
    ax.set_ylim([0.1, 0.7])
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
    ax.plot(ts, plot_info.get_exact_em_estimates(),
            color=exact_em_color, linestyle='--',
            label='exact EM')
    ax.plot(ts, plot_info.get_stochastic_em_estimates(),
            color=stoch_em_color, linestyle='--',
            label='Monte Carlo EM')
    #legend = ax.legend(loc='upper center')
    legend = ax.legend(loc='lower right')
    pyplot.savefig('monte-carlo-estimates-j.png')


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

