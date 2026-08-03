# Categorized-BDT implementation amendment

Status: frozen before categorized-BDT training on 2026-08-03.

This amendment fills only implementation details that were not numerically specified in `HIG_24_015_ADAPTATION_PLAN.md`. It does not change the train-only scope, taxonomy, feature contracts, fold assignment, multijet model, support contract, or prohibition on validation/test access.

## Verified frozen taxonomy

The c7s development table verifies the planned labels. The specialized `single_higgs`, `single_top`, `top_associated`, and `ttbar` union has 15,615 rows from 130 candidate-bearing source members. The complementary `qcd_multijet`, `zbbbb`, `diboson`, and `triboson` union has 5,753 rows from 58 candidate-bearing source members. No event-level relabelling is permitted.

## Frozen estimator and nested folds

- The exact c7s five-fold `registry_oof_fold` assignment is the outer split.
- Within each outer-training partition, its four remaining fold labels are the inner source-group folds.
- Both BDT stages use the c7t XGBoost parameters and seed contract from `hh4b_expanded_cut_bdt_protocol_v2.json`: 400 trees, learning rate 0.03, depth 5, histogram tree method, and the recorded regularization/subsampling parameters.
- The mass-aware and mass-plane-blind variants use the frozen ordered 34- and 30-feature c7s contracts without additions or reordering.
- Stage-1 targets are evaluated only at 0.65, 0.75, 0.85, and 0.90 weighted signal efficiency.

## Frozen staircase candidates and ranking

The low-stage-1 catch-all is category 0. For a total of two through six categories, the high-stage-1 region is split by ordered stage-2 score boundaries. Candidate boundary values are weighted-signal quantiles at the predefined fractions 0.10 through 0.90 in steps of 0.10; all ordered combinations required for the requested category count are tested. Unsupported candidates remain rejected.

Candidate ranking is lexicographic and fixed before outer evaluation:

1. maximize the shared-nuisance systematics-aware joint Asimov sensitivity;
2. maximize the nominal independent-Poisson joint sensitivity, with its ratio to the inclusive result capped at two only for reporting;
3. prefer fewer categories;
4. prefer the smaller median absolute inner-fold boundary displacement.

The selected result is the best supported candidate across two through six categories. The five-category benchmark is independently the best supported five-category candidate and is reported even when it is not selected.

## Correlated multijet implementation

The nominal exactly-$3b$ to $\geq4b$ factor, its statistical uncertainty, and the alternative-factor envelope are frozen c7r inputs. Direct-$\geq4b$ QCD is too sparse to define a stable independent nonclosure in every category. Therefore the direct-QCD nonclosure is recomputed for the union of the exhaustive categories and applied as one shared relative response to all category QCD yields. A single Gaussian nuisance is profiled jointly across categories. No category-independent QCD nuisance is introduced. Category-level direct-QCD rows, sources, yields, and effective statistics are still reported explicitly.

In each common source-member bootstrap replica the category yields, shared union nonclosure, common nuisance, support, and joint statistics are recomputed. Undefined replicas remain undefined and retain their failure reason.

## Predefined support-sensitivity contracts

In addition to the frozen nominal minima (background rows/sources/$N_{\rm eff}$: 50/10/2; signal rows: 25; transferred-QCD rows/sources: 5/3), two non-selecting studies are frozen:

- strict A: 75/12/3 background support, 35 signal rows, and 7/4 transferred-QCD rows/sources;
- strict B: 100/15/4 background support, 50 signal rows, and 10/5 transferred-QCD rows/sources.

Neither stricter contract may be used to choose the primary result after outer-fold performance is known.
