"""
This script checks inference of branch lengths and HKY parameter values.

The observed data consists of the exact joint leaf state distribution.
The inference uses EM, for which the expectation step is computed exactly
and the maximization step is computed numerically.
The purpose of the EM inference is for comparison to Monte Carlo EM
for which the observed data is not exact and the conditional expectations
are computed using Monte Carlo.

"""
