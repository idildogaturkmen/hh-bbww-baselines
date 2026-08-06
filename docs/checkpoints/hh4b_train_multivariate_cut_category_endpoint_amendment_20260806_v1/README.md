# Multivariate cut category-endpoint amendment

The full-population feasibility audit proved that the former absolute signal
efficiency target of 0.585957 was unreachable for every frozen cut family.
Before continuous cuts, the inclusive >=3-tag branch retained only about
40.7--40.9% of resolved signal and the >=4-tag branch retained about 10.3%.
No threshold scan or family ranking had occurred, so the contract can be
repaired without performance-driven tuning.

## Amended category design

The same 27 continuous cut structures are retained in two mutually exclusive
b-tag categories:

- `exact3tag`: exactly three tagged candidate jets;
- `ge4tag`: at least four tagged candidate jets.

This produces 27 x 2 = 54 category-family configurations, preserving the
original bounded complexity while removing the overlap of the former >=3 and
>=4 branches.

A structure is selected separately inside each category by true nested
source-group cross-validation. The target 0.585957 is now a category-conditional
signal retention measured relative to the signal already in that category
before continuous cuts. Both selected categories are retained and combined by
summing their disjoint physical yields and sumw2 values.

Category-conditional efficiencies, absolute resolved efficiencies, and
generated-event acceptance must all be reported with distinct labels.

## Why this is rigorous

The amendment is based only on an exact feasibility ceiling, not on a threshold
scan or observed family performance. It preserves the frozen variable set,
nested folds, deterministic search, sealed validation/test roles, resampling
plan, and paper-figure policy.

The prior v1 and v2 canaries remain failed implementation evidence and cannot be
used as physics results.
