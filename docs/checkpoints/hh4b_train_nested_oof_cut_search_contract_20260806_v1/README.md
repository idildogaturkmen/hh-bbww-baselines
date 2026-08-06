# Nested OOF cut-search contract

This checkpoint freezes the train-only cut-optimization protocol before any
candidate threshold is evaluated.

## Important accounting distinction

The complete dataset contains 5.2 million generated events across train,
validation, and test. Cut optimization is not allowed to use all 5.2 million.

The optimization scope is:

- 3,799,873 generated training events for accounting;
- 1,171,072 resolved training rows across primary and auxiliary populations;
- 1,042,397 authorized primary physical resolved rows used for the cut study;
- 128,675 auxiliary-QCD resolved rows excluded from physical optimization;
- validation and test remain sealed.

## Exact scan

The cut is `r_hh_125_125 < threshold`.

A so-called continuous threshold scan is implemented exactly on the finite
sample by scanning every empirical event boundary in the four development
folds. A one-dimensional strict-cut selection can change only when a threshold
crosses an observed value, so this is exhaustive without an arbitrary coarse
step size.

## Primary operating point

For each outer fold, the threshold is selected using only the other four folds
and the frozen nonnegative `comparison_weight_outer_fold_k` weights.

The nominal objective is to minimize comparison-weighted background efficiency
subject to comparison-weighted signal efficiency of at least 0.585957. This is
the already frozen primary development metric. Statistical-only significance is
reported as a diagnostic, not used to choose the nominal threshold before the
systematics model is frozen.

Each selected threshold is applied once to its untouched outer fold. The pooled
five-fold OOF result is the primary train-only performance estimate.

## Deployment candidate

The median of the five selected outer-fold thresholds is the predeclared
train-only deployment candidate. It is not substituted for the OOF performance
estimate and cannot be applied to validation or test until a later freeze.

## State at freeze

- candidate thresholds evaluated: 0
- threshold scan performed: false
- new threshold selected: false
- validation opened: 0
- test opened: 0
- models trained: 0
