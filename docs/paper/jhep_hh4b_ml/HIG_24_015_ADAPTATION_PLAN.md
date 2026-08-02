# HIG-24-015-inspired categorization plan for the HH→4b train study

Status: frozen implementation plan, written before categorized-BDT training.

This plan adapts only the methodological structure of the public CMS Physics
Analysis Summary HIG-24-015: a broad first discriminator, a specialized second
discriminator, and exclusive categories in their two-dimensional score plane.
The supplied internal draft was consulted only to understand technical details;
it is not a public citation, is not copied here, and must not be redistributed.

## Scope and non-equivalences

This is a Delphes-based, train-only HH→4b simulation study. It contains no
observed data and opens neither validation nor test payloads. HIG-24-015 uses a
diphoton mass signal window and data sidebands; neither exists in this workflow.
Its score thresholds, five-category result, and ten-sideband-event constraint are
therefore not imported. The public PAS will be cited as a methodological
reference, not as evidence for this study's numerical results.

The HH→4b primary multijet estimate is the frozen lower-b-tag three-b-to-four-b
transfer. Direct four-b QCD is secondary closure only. This role assignment and
its uncertainty are invariant inputs to the categorized study.

## Taxonomy justified by the frozen train population

The authoritative c7s development table contains 9,337 signal rows and 21,368
background rows over 441 source members. Its eight background strata are the
only grouping authority. No group is inferred from a filename or manually
relabelled event-by-event.

The first-stage classifier uses signal versus all background families. The
second-stage specialized background is the union of the frozen `single_higgs`,
`single_top`, `top_associated`, and `ttbar` strata. This is supported by 15,615
rows from 132 source members (130 candidate-bearing members). It captures the
resonant-Higgs and top-rich ordinary backgrounds that motivate a specialized
discriminator.

The complementary nonresonant-like group is the union of `qcd_multijet`,
`zbbbb`, `diboson`, and `triboson`: 5,753 direct-four-b rows from 222 source
members (58 candidate-bearing members). The direct-QCD contribution is only 56
rows; consequently it must never be used as the primary physical QCD yield.

## Frozen classifiers and variants

For each variant:

- stage 1 is signal versus every background row;
- stage 2 is signal versus the specialized single-Higgs/top union;
- stage 2 is trained only above a stage-1 threshold selected inside the outer
  training partition;
- the mass-aware variant uses the frozen ordered 34-feature contract;
- the mass-plane-blind variant uses the frozen ordered 30-feature contract;
- estimator parameters and random seeds are fixed before outer-fold evaluation.

The existing deterministic five-fold source-group assignment from c7s is reused
exactly. The source group, never an event row, is the unit of exclusion.

## Nested source-group procedure

For outer fold *k*, all boundary choices are made using only the other four
source folds:

1. generate inner source-group OOF stage-1 scores;
2. choose the high-stage-1 working point from the predefined weighted signal
   efficiencies 0.65, 0.75, 0.85, and 0.90;
3. generate inner source-group OOF stage-2 scores using only specialized rows
   above that inner stage-1 threshold;
4. optimize exclusive staircase boundaries for each predefined category count
   from two through six;
5. select the category count and boundaries using the systematic-aware objective
   and stability penalty below;
6. refit both classifiers on the complete outer-training partition using the
   selected contract and evaluate the untouched outer fold once.

No outer-held row or source contributes to stage-2 threshold selection, category
boundaries, category-count selection, feature selection, or hyperparameters.

## Category geometry and support contract

Categories are exclusive and exhaustive over the primary projection. A low
stage-1 catch-all category is always retained. The high-stage-1 region is divided
by ordered stage-2 boundaries, producing a reproducible staircase in the score
plane. Two through six total categories are evaluated; five is reported as the
HIG-24-015-inspired benchmark even if it fails support or is not selected.

Every category must contain, in the inner OOF primary projection:

- at least 50 raw background rows;
- at least 10 unique background source groups;
- background effective statistics of at least 2.0;
- at least 25 raw signal rows and nonzero physical signal yield;
- at least 5 transferred-QCD rows from at least 3 source groups.

Candidates that fail any constraint are recorded and rejected, not repaired by
moving sources, changing the transfer, or opening another split.

## Optimization and likelihood contract

The nominal independent-category statistic is the joint Poisson Asimov
likelihood, equivalently `sqrt(sum(ZA_i^2))` under the documented assumptions of
exclusive categories and independent Poisson counts.

The systematic-aware objective profiles one shared Gaussian multijet nuisance
across all categories. Its category responses use the same frozen transfer-factor
statistical term, alternative-factor envelope, and direct-QCD nonclosure
definition as c7t. The nuisance is correlated because the frozen transfer contract
does not authorize category-independent multijet normalizations.

Candidate ranking first maximizes systematic-aware sensitivity relative to the
inclusive systematic-aware result, then uses nominal sensitivity as a bounded
secondary term, and finally penalizes additional categories and inner-fold
instability. Nominal sensitivity alone cannot select the result.

## Mass-sculpting and stability tests

For each outer fold, variant, stage, and final category, diagnostics are frozen
for `mbb1`, `mbb2`, `r_hh_125_120`, and `mhh`. They include weighted correlations,
pre/post-selection mean and width changes, and distribution-distance measures.
Mass-aware results are explicitly labelled as such; the mass-plane-blind variant
is the primary sculpting control, not an assertion of complete decorrelation.

Category boundaries, support margins, yields, S/B, effective statistics,
nominal sensitivity, systematic-aware sensitivity, and source-bootstrap
stability are reported per outer fold and in the combined train-only OOF result.

## Publication status

The categorized result will remain a train-only nested-OOF diagnostic. It cannot
be called a Run-2 expected sensitivity or used to choose a publication model.
Validation/test opening, transfer-systematic changes, and final model selection
all require a separate scientific authorization.
