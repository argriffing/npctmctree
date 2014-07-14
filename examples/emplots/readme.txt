Make some plots related to Monte Carlo EM for phylogenetics.

Let n=10000 be a constant sample size.
Let t* be the 'true' parameter values.
Let t0 be the initial parameter values for EM.
Let x be the sample of n trajectories under the 'true' parameter values.
Let y be the subset (submeasure?) of x consisting of extant taxon observations.

Pick a single parameter to plot, say p, to keep the plots uncluttered.

Draw the plots as follows.

*** estimates of p from unconditional iid monte carlo samples
1) At each point on the x axis, sample n unconditional trajectories
   given only t*.
1a) Compute max likelihood parameter value estimates given these trajectories
    (expm is not required).
    Plot the estimated value of p.
1b) Compute max likelihood parameter value estimates given only the
    extant taxon observations from each sampled trajectory (expm is required).
    Plot the estimated value of p.

*** estimates of p from conditional correlated monte carlo Rao-Teh samples
2) Sample n unconditional trajectories given only t*.
   Keep only the extant taxon observations from each sampled trajectory.
   These n observations of extant taxon states will be used as
   simulated data for the plots in the remainder of this section.
2a) At each point on the x axis, sample one conditional trajectory,
    using Rao-Teh sampling, for each of the n extant state observations in (2),
    using the true parameter value t*.
2a.i) Compute max likelihood parameter value
      estimates given the sampled trajectories from (2a) (expm is not required).
      This is analogous to (1a).
      Plot the estimated value of p.
2a.ii) Compute max likelihood parameter value
       estimates given the extant taxon observations
       from each sampled trajectory from (2a) (expm is required).
       This is analogous to (1b).
       Plot the estimated value of p.

*** sequence of Monte Carlo EM estimates of p using Rao-Teh

*** sequence of EM estimates of p using exact conditional expectations

