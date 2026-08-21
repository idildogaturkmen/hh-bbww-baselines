# Track-B project status — 2026-08-21

> **DEVELOPMENT ONLY. Not final, governing, independently evaluated, or
> physically normalized. No final model ranking should be claimed.**

## Fixed development cohort and headline results

All headline classifier numbers use the same 400,000-event development
cohort: **193,358 signal, 174,485 QCD, and 32,157 ttbar** (206,642 total
background). The frozen cohort identity has SHA256
`9588d0fe79e32d455467c719eaaa57541fcbead9835b8c32bebb9d68b96a2d30`.

| Model | AUC all background | AUC QCD | AUC ttbar |
|---|---:|---:|---:|
| BDT-K | 0.8366047 | 0.8270892 | 0.8882361 |
| BDT-KF | 0.9681764 | 0.9661312 | 0.9792733 |
| SPA-Net 2M | 0.9692812 | 0.9681550 | 0.9753918 |
| SPA-Net 10M | 0.9692723 | 0.9677557 | 0.9775018 |

These values originate in
`/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_development_snapshot_20260821_v1/master_results/MASTER_DEVELOPMENT_RESULTS.json`;
its SHA256 is recorded in the adjacent archived `SHA256SUMS` and field-level
sources are in `master_results/PROVENANCE_MAP.tsv`.

## BDT convergence and final-fit state

The 200/1000/4000/6000-round study shows steep early improvement followed by
a small but measurable 4000-to-6000 gain. BDT-K validation logloss is
0.528945, 0.505537, 0.496148, and 0.494990; BDT-KF is 0.288252, 0.238045,
0.226978, and 0.225868. The last 2000 rounds improve logloss by only 0.233%
(K) and 0.489% (KF) while adding 50% more rounds. Neither run triggered the
configured 50-round validation-logloss early stopping because small new best
values arrived frequently enough to reset patience.

**4000 rounds is currently the leading candidate, not a final decision.**
The full-144M BDT final fits have **not been submitted**. The checkpoint/resume
canary established that chunked resume is non-equivalent to uninterrupted
XGBoost training and must not be treated as an exact continuation.

Source:
`/uscms_data/d3/iturkmen/hh4b_delphes/track_b_harvey_bdt_working_points_20260818_v1/step2m_harvey_round_budget_diagnostics_20260821_v1`.
The governing K/KF history hashes are respectively
`28243cb51345f6c545e3ff685799d88970fb1214a9bf11298d625cf2bc91a67d`
and `ccace063c516eb34d90a828132b9f28c133d0b7a38c749f647f802ce86c1b6cf`.

## SPA-Net scaling and remaining studies

SPA-Net 10M seed 0 completed all 50 epochs successfully on the fixed 400k
validation cohort. Its final adjudication is at
`/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AI_spanet_10M_scaling_seed0_20260821_v1/full_run_attempt2_eaf_local_staging/ADJUDICATION_REPORT_10M_SEED0_50EPOCH.md`.
The primary epoch-47 checkpoint SHA256 is
`fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5`;
the checkpoint binary itself is deliberately omitted.

Event classification is essentially flat from 2M to 10M: all-background AUC
changes by -0.000009, QCD by -0.000399, and ttbar by +0.002110. The modest
assignment-accuracy improvement is not evidence of improved event
classification. A final SPA-Net 144M baseline is planned but **has not been
launched**. A ParT frozen-embedding study is planned. JP-JEPA artifacts are
still pending.

## Residual backgrounds and evaluation boundary

The residual-background prototype uses raw, unnormalized cohort counts. At
epsilon_S=0.50, surviving QCD/ttbar counts are 16,254/2,923 (K), 824/93
(KF), 665/86 (SPA-Net 2M), and 711/83 (SPA-Net 10M summary only). Survivors
become more Higgs-pair-like in the descriptive four-jet topology variables;
KF and SPA-Net 2M mostly agree, with a finite disagreement population. No
causal interpretation is claimed, and small ttbar tails are support-limited.

Source:
`/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_development_residual_backgrounds_20260821_v1`;
the matched-cohort NPZ SHA256 is
`9588d0fe79e32d455467c719eaaa57541fcbead9835b8c32bebb9d68b96a2d30`.

Independent Stage-C has **not** been opened for final evaluation. Physical
cross-section/luminosity/generator-weight normalization and significance are
still pending final inference. Consequently, neither the development AUCs nor
the raw residual mixture constitute a final physics result.
