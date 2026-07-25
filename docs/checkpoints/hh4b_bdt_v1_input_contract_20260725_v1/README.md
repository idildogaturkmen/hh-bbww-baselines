# Canonical HH4b BDT-v1 input contract

## Result

- Status: `hh4b_bdt_v1_input_contract_pass`
- Canonical development members: 581
- Train members assigned to five folds: 458
- Train candidate rows audited in memory: 57326
- Mass-aware features: 34
- Explicit dijet-mass-plane-blind features: 30
- Missing, nonfinite, or constant selected features: 0, 0, 0
- Models trained, predictions written: 0, 0

## Population and label contract

The binary target is HH→bbbb signal (`training_target=1`) versus every
configured canonical background family (`training_target=0`) in the frozen
at-least-four-tag candidate population. Labels and signal modes come from the
manifest fields `sample_class`, `training_target`, and `process_or_mode`.
Because the manifest has no `process_family` field, the committed YAML freezes
one explicit `process_or_mode` → broad physics-family mapping.

All 458 train candidate Parquets were
opened. Validation counts are manifest metadata only: validation candidate
files opened = 0. Test members
and candidate files considered = 0 and
0.

The audit encountered 162 train
Parquets with no physical columns; every one is a manifest-declared zero-row
member and was opened without a feature projection to confirm its zero row
count. Every candidate-bearing file was required to supply the complete
selected-feature source contract.

No $R_{HH}^{125,125}<34$ requirement is applied to the BDT input. That
requirement remains the final decision rule of the optimized cut baseline.
No 40 GeV jet threshold, $|\eta|<2.4$ requirement, or additional signal-region
selection was introduced.

## Features

The mass-aware model has the frozen 30 common detector-level scalar features
plus `mbb1`, `mbb2`, `delta_mbb`, and `r_hh_125_125`. The 30-feature ablation
is accurately described as an **explicit dijet-mass-plane-blind ablation**,
not as fully mass-decorrelated. Generator truth, identity, labels, paths,
flavors, raw indices, absolute azimuths, b-tag values, redundant mass aliases,
and exactly-3b promoted-jet provenance are forbidden.

No feature was imputed, clipped, or used to drop a row. The row-level feature
matrix existed only in memory and is not written or committed.

## Grouped folds

The installed environment has no scikit-learn, so the recorded fallback
`deterministic_stratum_greedy_largest_candidate_count_first_v1` assigns immutable source members
deterministically. No member appears in more than one fold. All signal modes
and all configured background families have at least five train members and
occur in every fold.

## Development-balancing weights

These are development-balancing weights, not cross-section, luminosity,
generator, importance, or physical event weights. Signal and background each
receive half of the total; ggF and VBF receive equal signal shares; and the
8 background families receive equal background
shares. Candidate-bearing source members receive equal totals within their
mode or family, then each member total is distributed uniformly over its
candidate rows. Members with zero candidates remain assigned to folds and
appear in the audit, but no row weight can exist for a member with no rows.
The positive row weights are rescaled to mean one.

## Legacy BDT inventory

The inventory retains older BDT-related scripts, configurations, models,
predictions, checkpoints, and result directories unchanged. Each is marked
legacy/noncanonical unless the current 581-member population and this exact
feature/fold/weight contract can be proven; none is reused as canonical-v1.

## Plotting

All three figures use `scripts/plotting/hh4b_cms_style.py`, a truthful
CMS-publication-inspired but nonofficial style. mplhep was
unavailable, so the deterministic fallback was used.
Every figure is provided as a 300 dpi PNG and vector PDF without system LaTeX.

## Authorization boundary

- `canonical_bdt_training_authorized: false`
- `validation_evaluation_authorized: false`
- `test_access_authorized: false`
- `physical_significance_authorized: false`

## Reproducibility

- Source commit: `3d26a8ba6e018cbf80d445985013a664a2ec4586`
- Manifest SHA-256: `b3e42a0af56594445e297e2347ccfd3bd799fd107a578af844c9733bfd2ebe6a`
- Configuration SHA-256: `e85d37ea0d24202ef601112284c9ee8a1541fde651f6324efead493f48b98354`
- Common helper SHA-256: `d68f1477b7c1a3d91dd24e904b7e2384e3642b4b0fb546801101bb243494efe0`
- Plot style helper SHA-256: `7184a04275fa3550cfbe11ad94cb3fe771835d9fb03fe6eea281fa311df829f6`
- Preparation runner SHA-256: `200a5428dca5726bd5fa7cb55edd6ce635f3f24120841c2355bacf41ca76c3e5`

## Next gate

`train_hh4b_bdt_v1_grouped_cross_validation`
