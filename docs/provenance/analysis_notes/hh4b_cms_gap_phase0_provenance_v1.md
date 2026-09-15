# HH4b CMS-gap Phase-0 provenance audit v1

Status: train-only read-only audit complete for the first expected-limit study. `CMS_GAP_VALIDATION_PAYLOADS_OPENED=0`; `CMS_GAP_TEST_PAYLOADS_OPENED=0`.

## Repository and checkpoint

- Worktree: `/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-cms-gap`
- Branch: `cms-resolved-sensitivity-gap-v1`
- HEAD: `c8f5ba4b5ca1ff3238d14690619109793b9d7f90`
- Subject: `analysis: freeze blocked cut baseline final status`
- Frozen train result: `HH4B_CUT_BASELINE_FINAL_STATUS.{md,json}`. It records zero validation/test payload access.

## Existing authorized train products

- Ten physical fold tables (five folds × `exact3tag`/`ge4tag`) listed with SHA-256 in `artifacts/hh4b_cut_baseline/train_performance/manifests/fold_table_inventory.tsv`.
- Frozen physical source/yield table: `fixed_selection_source_metrics.tsv`, with source UID, class, fold, row counts, signed yields, and sumw2 before/after the fixed selection.
- Broad train registry: 464 primary train sources (87 signal, 354 physical-background source entries represented by access categories in the summary, plus 23 auxiliary QCD sources excluded from physical evaluation); 3,799,873 generated train events and 1,171,072 resolved rows in the frozen common-table accounting contract.
- Authoritative event weight in the fold tables: `resolved_selection_contribution_weight`. It is signed; the frozen fixed selection contains one selected negative-weight background row.
- Process/mode and category information: `process_or_mode`, `sample_class`, `candidate_tagged_jet_count`; exact-three and at-least-four tables are separate.
- Reconstruction variables include `mbb1`, `mbb2`, `r_hh_125_125`, `mhh`, candidate kinematics, and pairing-derived quantities.

## What is and is not possible

The fold tables support one-bin, category, fixed-shape, process/source, signed-yield, sumw2, and finite-MC diagnostics. They do not contain truth particles or truth-jet matching, and only contain the exact3/≥4-tag selected populations. Therefore an unbiased truth-pairing study and the requested 0/1/2-tag multiplicity distributions cannot be derived from these products. No validation/test payload will be opened to fill those gaps. A broader train-source study would require separately reading authorized train ROOT sources; no batch campaign is authorized or submitted here.

`pyhf` and `pytest` are unavailable in the active environment. The inference implementation is dependency-free and tested with Python `unittest`; `pyarrow` is available for the frozen Parquet inputs. No package was installed.
