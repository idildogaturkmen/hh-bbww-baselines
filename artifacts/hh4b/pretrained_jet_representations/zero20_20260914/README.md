# ZERO20 width-control ablation — final result (2026-09-14)

This bundle freezes the final, completed ZERO20 experiment: the same
27-input SPA-Net architecture and training contract as the frozen-ParT
"active20" result, with the 20 auxiliary input channels held at exactly
`0.0` from the first training step, instead of holding the 20 frozen ParT
embedding dimensions. It isolates whether **widening** the input layer, by
itself, is sufficient to explain the active20 harm result, or whether the
harm is tied to the specific ParT content.

All numbers below are read directly from the final, `n_boot=10000`
comparison files in `metrics/` — none are copied from a prompt, a draft,
or an intermediate/recovery smoke test. See `provenance/` for exact
source paths and hashes.

## Result

| Model | AUC (all-bg) | AUC (QCD) | AUC (ttbar) | Exact-event reco. | Per-Higgs reco. |
|---|---:|---:|---:|---:|---:|
| Native SPA-Net 2M | 0.969281 | 0.968155 | 0.975392 | 0.866588 | 0.895299 |
| Native SPA-Net 10M | 0.969272 | 0.967756 | 0.977502 | 0.872229* | 0.898838* |
| SPA-Net + ParT active20 (2M) | 0.944222 | 0.941220 | 0.960516 | 0.496015 | 0.513756 |
| **SPA-Net + ZERO20 (2M)** | **0.969116** | 0.967868 | 0.975886 | **0.866201** | 0.895051 |

\*Native 10M reconstruction is a **point estimate only** — computed directly
from the frozen `native10m_eval_400k.npz` export using the exact same
`match_higgs_pairs()`/`reconstruction_metrics()` code as every other
reconstruction number in this project (verbatim copy, not reimplemented;
see `code/compute_native10m_reconstruction.py`). Before trusting it, that
script recomputed the identical code on `control_2m_native_eval.npz` and
required **exact** (bit-for-bit) reproduction of the already-published
Native 2M numbers (0.8665880160209383 / 0.8952988063607884) — confirmed,
see `metrics/native10m_reconstruction_crosscheck.json`. No paired-bootstrap
CI was computed for the Native10M number (would require a new bootstrap,
out of scope for this addition), so unlike every other number in this
table it cannot be called statistically significant or not — it is
reported as a raw point estimate, flagged as such everywhere it appears.

Paired differences (test minus control), final 10,000-replicate bootstrap:

| Comparison | ΔAUC (all-bg) | 95% CI | Δexact-event reco. | 95% CI | McNemar p |
|---|---:|---:|---:|---:|---:|
| ParT20 − Native2M | −0.025059 | [−0.025504, −0.024619] | −0.370573 | [−0.373936, −0.367229] | 0 |
| ZERO20 − Native2M | −0.000165 | [−0.000287, −0.000044] | −0.000387 | [−0.001527, 0.000750] | 0.515 |
| ZERO20 − ParT20 | +0.024893 | [0.024455, 0.025326] | +0.370187 | [0.366901, 0.373524] | 0 |

## Statistical vs. practical significance (read this before the classification below)

The ZERO20-vs-native AUC difference (−0.000165) has a 95% CI that **excludes
zero** — with 400,000 paired events, even a difference this small is
statistically resolvable. But it is **~30× smaller** than the pre-registered
practical-effect floor of **0.005 AUC** (the project's own
`additional8M`-authorization floor, itself set at ~2.5× the previously
measured native 2M→10M scaling noise ceiling of ~0.002 AUC — not invented
for this result). The reconstruction difference is not even statistically
resolvable (McNemar p = 0.515, CI includes zero). **Statistical
significance and practical relevance are answering different questions
here, and this result answers "no" to the practical one.**

## Causal-branch classification — mechanically derived, not assumed

`code/assemble_zero20_final.py` applies the pre-registered decision rule
(paired-bootstrap 95% CI + the 0.005 AUC floor for classification;
McNemar exact p<0.05 for reconstruction, no separate floor — none was ever
pre-registered for reconstruction) to the three final comparison files.
Both metrics independently agree:

# **Branch B**

ZERO20 is statistically/practically indistinguishable from native SPA-Net
on both classification and reconstruction, while the original ParT
active20 result remains significantly and substantially degraded on both.
**This falsifies the pure-width explanation**: widening the input
embedding from 7 to 27 channels is not, by itself, sufficient to reproduce
the harm. The harm is associated with the actual selected ParT values
and/or how they interact with preprocessing and optimization — not with
input width alone.

This is a claim about *this specific* frozen-ParT integration (the 20
TRAIN-only, variance-selected dimensions, z-scored and appended to
SPA-Net's input, at this training contract) — it is **not** a claim that
pretrained ParT representations are inherently unable to help jet
reconstruction. See `next_experiments.md` (in
`docs/studies/pretrained_jet_representations/`) for what would need to
change to test that more general question.

The training-history record (`metrics/training_history_three_way.json`,
`plots/epoch_assignment_loss.svg`, `plots/epoch_jet_accuracy.svg`) shows
the same pattern from the very first training epoch: ZERO20's epoch-0
assignment loss (h1+h2 = 0.998) and validation jet accuracy (0.460) sit
close to native's (1.111, 0.466), while ParT active20's sit far off
(2.572, 0.164) — the divergence is not something that develops later in
training, it is present immediately, and ZERO20 does not exhibit it.

## Fixed-efficiency background rejection

`tables/fixed_efficiency_rejection.{md,csv}` reproduces every predeclared
working point (εS ∈ {10, 7.5, 5, 4, 3}%) for all three 2M models, all-
background, **with the raw survivor counts and sparse-Monte-Carlo warnings
preserved verbatim** — several working points have single-digit background
survivors out of 206,642 events (e.g. native at εS=5%: 3 survivors; at
εS=3%: 0 survivors, rejection formally undefined). Do not read these
tail-rejection numbers as precise without checking the survivor count in
the same row.

## ROC and background-rejection figures (all four models, common cohort)

`plots/roc_all_background_four_models.svg` and
`plots/background_rejection_four_models.svg` are built directly from the
frozen per-event `score`/`process` arrays of all four models
(`code/compute_roc_curves.py`), not from the coarse 5-point
fixed-efficiency table. Before computing either curve, the script verifies
that all four models' `event_id` and `process` arrays are **identical**
after sorting by `event_id` — the same matched-cohort identity check used
throughout this project — and refuses to proceed otherwise. The AUC shown
in the ROC legend is computed with the exact same dependency-free,
tie-averaged rank statistic (`bootstrap_utils.roc_auc`) used for every
other AUC in this project, and reproduces the already-published values
exactly (0.969281 / 0.969272 / 0.944222 / 0.969116).

Because Native 2M, Native 10M, and ZERO20 are statistically/practically
indistinguishable (see above), their ROC curves nearly overlap — the
figure includes a zoomed inset panel (εB∈[0,0.08], εS∈[0.75,1]) showing
the same underlying curve data at a scale where the separation from ParT
active20 is visible; no curve is refit or altered for the inset.

The rejection figure plots signal efficiency vs. 1/εB on a log axis,
**stopping each curve at its own last real background survivor** — no
fitted or extrapolated tail is drawn past that point. The segment where
fewer than 10 raw background events survive (this project's established
"Poisson-limited" threshold) is drawn dashed with hollow markers, and each
curve's sparsest endpoint is annotated with its exact survivor count
(e.g. ParT active20 reaches εS as low as 0.0021 with 1 survivor; Native 2M
cannot go below εS≈0.039 without running out of background entirely).

## What this bundle does not contain

Per this repository's provenance policy, the training HDF5s, the ZERO20
model checkpoint, and the event-level `.npz` evaluation exports are **not
committed** here — they live in the external working area recorded in
`provenance/SOURCE_FILES.json`/`SHA256SUMS`. Everything needed to check or
regenerate the tables/plots above (the three comparison JSONs and the
training-history JSON) **is** committed, in `metrics/`.

## Directory layout

- `metrics/` — the three final comparison JSONs (unmodified copies), the
  training-history JSON, the Native10M reconstruction cross-check result,
  the ROC/rejection curve data, and this bundle's own assembled
  `model_summary.json` / `paired_statistics.json` /
  `causal_branch_classification.json`.
- `tables/` — the five required Markdown+CSV tables (model summary,
  paired statistics, reconstruction, fixed-efficiency rejection, training
  resources).
- `plots/` — eight SVG figures: AUC comparison, reconstruction comparison
  (now including Native 10M), paired-AUC forest plot, epoch-by-epoch
  assignment loss, epoch-by-epoch validation jet accuracy, all-background
  ROC (four models, with zoomed inset), and background rejection (four
  models, sparse-tail-aware).
- `provenance/` — `SHA256SUMS` and `SOURCE_FILES.json` for the exact
  external files this bundle was assembled from (now including all four
  models' per-event `.npz` exports, used for the reconstruction
  cross-check and the ROC/rejection figures).
- `code/` — the deterministic assembler, the Native10M reconstruction
  cross-check, the ROC/rejection curve computation, and all seven
  plotting scripts — every one re-runnable against already-frozen files
  alone (no bootstrap, no training, no GPU, no inference).

## Regenerating this bundle

```bash
python3 code/assemble_zero20_final.py \
  --native-vs-part20-json metrics/comparison_native2M_vs_part2m_active20.json \
  --zero20-vs-native-json metrics/comparison_zero20_vs_native2M.json \
  --zero20-vs-part20-json metrics/comparison_zero20_vs_part2m_active20.json \
  --native10m-reconstruction-json metrics/native10m_reconstruction_crosscheck.json \
  --out-dir .

# Native10M reconstruction (hard-gated on exact reproduction of the frozen Native2M numbers):
python3 code/compute_native10m_reconstruction.py --out-json metrics/native10m_reconstruction_crosscheck.json

# ROC / background-rejection curve data (verifies all 4 event_id/process arrays match first):
python3 code/compute_roc_curves.py --out-json metrics/roc_curves_data.json

python3 code/plot_auc_comparison.py --model-summary-json metrics/model_summary.json --out-svg plots/auc_comparison.svg
python3 code/plot_reconstruction_comparison.py --model-summary-json metrics/model_summary.json --out-svg plots/reconstruction_comparison.svg
python3 code/plot_forest_auc_deltas.py --paired-statistics-json metrics/paired_statistics.json --out-svg plots/forest_auc_deltas.svg
python3 code/plot_epoch_assignment_loss.py --training-history-json metrics/training_history_three_way.json --out-svg plots/epoch_assignment_loss.svg
python3 code/plot_epoch_jet_accuracy.py --training-history-json metrics/training_history_three_way.json --out-svg plots/epoch_jet_accuracy.svg
python3 code/plot_roc_all_background.py --roc-curves-json metrics/roc_curves_data.json --out-svg plots/roc_all_background_four_models.svg
python3 code/plot_background_rejection.py --roc-curves-json metrics/roc_curves_data.json --out-svg plots/background_rejection_four_models.svg
```

All scripts are deterministic reads/recomputations over already-frozen
per-event files — none launches training, embedding extraction, inference,
or a new bootstrap.
