# Repository-native multivariate cut optimizer

This checkpoint freezes the first reusable, repository-native implementation of
the resolved HH→4b multivariate cut scan. It does not contain a physics result
and does not select a production cut.

## Frozen implementation

- deterministic category/family optimizer;
- immutable fold/category table preparation;
- one structure job per outer fold, b-tag category, and continuous structure;
- exact 270-job manifest generation;
- train-only merge, family selection, physical OOF aggregation, and deployment-rule audit;
- committed-code canary orchestrator;
- HTCondor wrapper and submit description;
- synthetic deterministic self-test;
- frozen production configuration.

The production decomposition is

- 5 outer folds;
- 2 disjoint categories (`exact3tag`, `ge4tag`);
- 27 continuous structures per category;
- 270 independent structure jobs.

Each structure job performs four genuine inner held-out fits, pools only those
cross-fitted predictions for family eligibility, refits on all four outer
development folds, and applies once to the untouched outer fold. The merge code
selects a structure using only pooled inner-OOF metrics and then reads the
corresponding outer physical result.

## Search algorithm

The full scan uses 63 quantile positions, at most 128 deterministic product-grid
beam seeds, the best 16 refinement starts, and up to 8 exact empirical
coordinate-refinement passes. The fully open boundary is always included.
This is a bounded deterministic search and is not described as an exhaustive
Cartesian scan.

## Weight roles

- nonnegative `comparison_weight_outer_fold_k` columns are used for threshold
  and family selection;
- signed `resolved_selection_contribution_weight` is used only for physical
  yields and sumw2;
- the held-out outer fold's own comparison-weight column must remain null;
- auxiliary `qcd_bbbb` sources are excluded.

## State at freeze

- production thresholds scanned: 0;
- production families ranked: 0;
- production cut selected: false;
- validation payloads opened: 0;
- test payloads opened: 0.

The next gate is a committed-code canary using deterministic 64-row-per-source-
category preparation. Only after that passes may an HTCondor pilot be submitted.
