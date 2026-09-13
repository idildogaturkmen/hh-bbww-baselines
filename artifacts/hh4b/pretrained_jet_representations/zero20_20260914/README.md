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
| Native SPA-Net 10M | 0.969272 | 0.967756 | 0.977502 | — | — |
| SPA-Net + ParT active20 (2M) | 0.944222 | 0.941220 | 0.960516 | 0.496015 | 0.513756 |
| **SPA-Net + ZERO20 (2M)** | **0.969116** | 0.967868 | 0.975886 | **0.866201** | 0.895051 |

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

## What this bundle does not contain

Per this repository's provenance policy, the training HDF5s, the ZERO20
model checkpoint, and the event-level `.npz` evaluation exports are **not
committed** here — they live in the external working area recorded in
`provenance/SOURCE_FILES.json`/`SHA256SUMS`. Everything needed to check or
regenerate the tables/plots above (the three comparison JSONs and the
training-history JSON) **is** committed, in `metrics/`.

## Directory layout

- `metrics/` — the three final comparison JSONs (unmodified copies),
  the training-history JSON, and this bundle's own assembled
  `model_summary.json` / `paired_statistics.json` /
  `causal_branch_classification.json`.
- `tables/` — the five required Markdown+CSV tables (model summary,
  paired statistics, reconstruction, fixed-efficiency rejection, training
  resources).
- `plots/` — six SVG figures: AUC comparison, reconstruction comparison,
  paired-AUC forest plot, epoch-by-epoch assignment loss, epoch-by-epoch
  validation jet accuracy.
- `provenance/` — `SHA256SUMS` and `SOURCE_FILES.json` for the exact
  external files this bundle was assembled from.
- `code/` — the deterministic assembler and all five plotting scripts,
  each re-runnable against the files in `metrics/` alone (no bootstrap,
  no training, no GPU).

## Regenerating this bundle

```bash
python3 code/assemble_zero20_final.py \
  --native-vs-part20-json metrics/comparison_native2M_vs_part2m_active20.json \
  --zero20-vs-native-json metrics/comparison_zero20_vs_native2M.json \
  --zero20-vs-part20-json metrics/comparison_zero20_vs_part2m_active20.json \
  --out-dir .

python3 code/plot_auc_comparison.py --model-summary-json metrics/model_summary.json --out-svg plots/auc_comparison.svg
python3 code/plot_reconstruction_comparison.py --model-summary-json metrics/model_summary.json --out-svg plots/reconstruction_comparison.svg
python3 code/plot_forest_auc_deltas.py --paired-statistics-json metrics/paired_statistics.json --out-svg plots/forest_auc_deltas.svg
python3 code/plot_epoch_assignment_loss.py --training-history-json metrics/training_history_three_way.json --out-svg plots/epoch_assignment_loss.svg
python3 code/plot_epoch_jet_accuracy.py --training-history-json metrics/training_history_three_way.json --out-svg plots/epoch_jet_accuracy.svg
```

All five scripts are deterministic reads of already-computed files — none
launches training, embedding extraction, or a new bootstrap.
