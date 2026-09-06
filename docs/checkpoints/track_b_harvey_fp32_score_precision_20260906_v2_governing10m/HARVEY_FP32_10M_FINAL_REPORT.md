# Harvey follow-up v2: governing SPA-Net 10M, full 400k-event rerun with raw logits

Status: **COMPLETE.** Real, authorized, CPU-only inference of the
governing checkpoint over the full frozen development cohort. This
supersedes v1's 2M-sibling stand-in with the actual governing-model
answer. `GOVERNING_10M_FULL_COHORT_RERUN = YES`, `NEW_TRAINING_LAUNCHED = NO`.

See `STATUS.md` for the pre-inference identity/environment verification
log (written before any inference command ran, per instruction).

---

## What was run

**Environment.** The pinned SPA-Net runtime already used to produce
every frozen SPA-Net result in this project:

```
/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AE_spanet_10jet_literature_aligned_training_preflight_20260818_v1/gpu_canary_10jet_trainonlyweighted_v23exact_r2/.pixi/envs/default/bin/python
```
identified because the governing checkpoint's own training wrapper
(`run_10M_seed0_full_50epoch_attempt2.sh`) hard-codes this exact
interpreter path, and `evaluate_classification.py` (the code the
governing evaluation itself imports) pins `spanet_repo_v23exact` at
this same location. Verified directly (no assumption):

| Check | Result |
|---|---|
| Python | 3.9.23 (conda-forge) |
| torch | 2.8.0+cu128, `cuda.is_available()==False` on this CPU-only host -- exactly what "CPU-only inference" requires |
| pytorch_lightning | 2.6.0 (matches the version cited throughout this project's frozen training reports) |
| h5py | 3.14.0 |
| `spanet` import path | `.../gpu_canary_10jet_trainonlyweighted_v23exact_r2/spanet_repo_v23exact/spanet/__init__.py` |
| `spanet_repo_v23exact` commit | `debbdc999bfb785eb110a36c5fd3eff211ebf234` -- matches `SPANET_PINNED_COMMIT_EXPECTED` in `evaluate_classification.py` exactly |
| `git status --porcelain` on that repo | empty (clean) |

**No `pip install`/`uninstall`/`upgrade` was run.** `pandas` is broken
in this pinned env (`ModuleNotFoundError`-chain on `pytz`/`dateutil`);
rather than touch the environment, the per-event archive was written
with `pyarrow` directly (already present, v21.0.0), which needs no
`pandas`.

**Identity checks (Steps 1-4, detailed in `STATUS.md`):** governing
checkpoint SHA256 `fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5`
-- **PASS**. Shared validation HDF5
(`production_2M_val.h5`) SHA256 `3c94bf8300d1dc3324e2cf25f9b6c618dffb94819cb3a297a43ac06320d5c2a2`
-- **PASS**, and proven (by code inspection, not assumption) to be the
exact same file/row-order the frozen `evaluate_classification_10M.py`
scored, since it imports `evaluate_classification.py` and never
reassigns `HDF5_VAL`.

**Inference command actually run** (`scripts/run_governing_10m_tail_inference.py`,
via the pinned interpreter above), CPU-only, `model.eval()` +
`torch.no_grad()`, `options.num_gpu=0`, `batch_size=2048`,
`shuffle=False`, `drop_last=False` -- the **full** 400,000-event
cohort, not the 1,256-event v1 tail subset:

```
governing checkpoint sha256: fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5
HDF5_VAL sha256: 3c94bf8300d1dc3324e2cf25f9b6c618dffb94819cb3a297a43ac06320d5c2a2
==== Preflight OK ====
process reconstruction VERIFIED: {'n_mismatch': 0, 'n_signal': 193358, 'n_qcd': 174485, 'n_ttbar': 32157}
checkpoint load: missing=[] unexpected=[]
forward pass wall time: 127.93s for 400000 events (3126.7 events/s)
```

Exit code 0. No command failed; nothing here required a stop-and-report.

**Correctness cross-check (independent of anything claimed):** the
production float32 score column from this rerun reproduces
`roc_auc.signal_vs_all_background = 0.9692723276616946` **to all 16
reported significant digits** against the frozen
`classification_evaluation_10M_result.json` from the original 2026-08-21
evaluation. Since SPA-Net inference here is deterministic
(`pl.seed_everything(0)`, no dropout at `model.eval()`, no data
augmentation), this exact digit-for-digit match confirms this rerun is
scoring the identical model on the identical data in the identical
order -- not an approximation, not a different checkpoint.

---

## Task 1 (redone against the real governing archive)

Per-event archive now exists for the governing checkpoint:
[`SPA10M_EVENT_LOGITS_400K.parquet`](SPA10M_EVENT_LOGITS_400K.parquet)
(400,000 rows x 11 columns: `row_index`, `process_label`,
`z_background`, `z_signal`, `delta`, `score_float32_production`,
`score_float32_hex_be`, `k_lattice`, `u_prob32`, `u_logit`,
`sigmoid_delta_float64`). `score_float32_production` is byte-identical
to what `torch.softmax(logits, dim=-1)[:,1]` produces in the frozen
production code path (same call, same dtype, same code) -- stored here
as a float64 parquet column purely for I/O convenience, but every value
in it is an exact float32 value with no additional rounding (verified:
recomputed AUC matches the frozen result to 16 digits).

---

## Task 2 -- unchanged from v1 (pure arithmetic, re-cited for convenience)

`u_2 = 6.9236899003`, `u_4 = 6.6226599046` (k=2 and k=4 on the `2^-24`
float32 lattice below 1.0). Not recomputed here; see v1's
`FP32_LATTICE_TABLE.csv`.

---

## Task 3 -- exact governing-10M census (real, full 400,000-event cohort)

Full unique-value census (27 rows, u>5.5, all populations):
[`SPA10M_FP32_UNIQUE_SCORE_CENSUS.csv`](SPA10M_FP32_UNIQUE_SCORE_CENSUS.csv).
Explicit k=1..64 lattice table (all 5 populations x 65 rows each,
including zero-count bins, plus the exact-1.0 row):
[`SPA10M_LATTICE_BINS_K1_TO_K64.csv`](SPA10M_LATTICE_BINS_K1_TO_K64.csv).

### Exact counts, u > 5.5

| Population | n_total | n(u>5.5) | n unique float32 values | n(score==1.0) |
|---|---:|---:|---:|---:|
| signal | 193,358 | **2,239** | 27 | **67** |
| all_background | 206,642 | **0** | 0 | 0 |
| qcd | 174,485 | **0** | 0 | 0 |
| ttbar | 32,157 | **0** | 0 | 0 |
| other_background | 0 | 0 | 0 | 0 |

### The two spikes, governing 10M checkpoint, exact counts

| Lattice point | u_k | signal | all-background | QCD | ttbar |
|---|---:|---:|---:|---:|---:|
| k=2 ("u~6.9") | 6.9236899003 | **99** | **0** | 0 | 0 |
| k=4 ("u~6.6") | 6.6226599046 | **100** | **0** | 0 | 0 |

### Answering Harvey exactly

1. **How many SPA10M events are in the u~6.9 spike?** **99**, all signal.
2. **How many are in the u~6.6 spike?** **100**, all signal.
3. **How many are exact score=1?** **67**, all signal.
4. **Are any of these background?** **No. Zero.** Across the entire
   u>5.5 tail (2,239 events, 27 distinct float32 values, spanning
   exact-1.0 down through k=52), not one QCD or ttbar event appears.
   This holds for the actual governing checkpoint, not just the v1 2M
   stand-in -- the same qualitative finding, now on the real model.

Same still-tens-not-hundreds conclusion as v1, with slightly larger
(but still real, exact) counts for the governing model: 99, 100, and 67
respectively, none of them background.

**Observed even-k-only pattern, again confirmed:** every nonzero-count
`k` in the governing-10M tail is even (2,4,6,...,52) -- identical to
the pattern found in the unrelated 2M-checkpoint weights in v1. Since
this now holds across two independently trained sets of weights
(different seeds' worth of training, same architecture/code/library
stack), this strengthens the v1 hypothesis that the even-only pattern
is a property of the CPU `torch.softmax` kernel's rounding behavior for
a 2-class input on this platform, not a property of any specific
trained model. Still not independently proven bit-for-bit at the
kernel level; reported as a reinforced observation.

---

## Task 4 -- raw logits (now real, not "unavailable")

`RAW_LOGITS_AVAILABLE = YES` (this rerun). `z_background`, `z_signal`,
and `delta = z_signal - z_background` are stored per-event in
`SPA10M_EVENT_LOGITS_400K.parquet`, float64 (cast up from the
production float32 logits with no additional rounding -- float32 to
float64 is always exact).

- `figures/u_prob32_tail_10M.png` -- histogram of `u_prob32`, tail.
- `figures/u_logit_tail_10M.png` -- histogram of `u_logit`, tail: **no
  comb structure** -- smoothly and continuously populated all the way
  out past u=8.5, in sharp visual contrast to the `u_prob32` histogram.
- `figures/u_prob32_vs_u_logit.png` -- scatter, all finite-score signal
  events plus the 67 exact-1.0 events (plotted as triangles at a fixed
  `u_prob32` marker since their true value is `+inf`). The finite
  points trace flat horizontal steps (the FP32 lattice) crossing a
  continuously-varying `u_logit` x-axis; the 67 exact-1.0 triangles
  alone span `u_logit` from ~7.22 to beyond 8.5 -- i.e. dozens of
  genuinely different confidence levels, at the raw-logit level, all
  get flattened onto the identical stored probability of exactly 1.0.
- `figures/delta_by_fp32_bin.png` -- boxplot of `Delta` grouped by the
  12 largest FP32 tail bins (k=0,2,4,...,22): medians decrease
  monotonically and smoothly as k increases (k=0 median Delta ~17.2
  down to k=22 median ~13.6), with narrow but non-degenerate spread and
  a few high-Delta outliers in the k=0 (exact-1.0) bin -- exactly the
  expected picture of one continuous underlying quantity (Delta) being
  binned by a coarser downstream quantization (the float32 probability).

**Distinct-value counts, u>5.5:** `n_distinct_prob32 = 27`,
`n_distinct_delta = 2,239`, `n_distinct_u_logit = 2,239` -- **every
single one of the 2,239 tail events has its own distinct Delta and its
own distinct u_logit**, despite collapsing onto only 27 distinct stored
probabilities. Events tied by float32 probability remain **fully
separated and orderable** in logit space at this threshold.

`sigmoid(Delta)` computed independently in float64
(`sigmoid_delta_float64` column) is, as expected, numerically
consistent with the production float32 score wherever the latter is
below 1.0 (both derive from the same logits; the float64 path merely
avoids the `1-score` cancellation and the coarser float32 grid), and
remains finite and strictly less than 1 even for the 67 nominally
"score==1.0" events -- confirming those events are not infinitely
confident, only confident enough to saturate the float32
representation.

---

## Task 5 -- resolved (no rerun plan needed anymore; this **is** the rerun)

The v1-prepared tail-only rerun script
(`../track_b_harvey_fp32_score_precision_20260906_v1/scripts/tail_only_rerun_logits_PREPARED_NOT_LAUNCHED.py`)
is now superseded: this task authorized and executed a **full**
400,000-event governing-checkpoint rerun instead of the smaller
1,256-event tail-only subset, so there was no need to fall back to the
narrower prepared plan. That v1 script is left untouched and unlaunched
(it was never a prerequisite for this task; the full-cohort rerun makes
it redundant, not obsolete-and-deleted).

---

## Task 6 -- output quantization vs. model precision vs. model pathology (settled)

This time the claim that could not be checked in v1 -- "does the logit
margin stay smooth through the FP32 probability spikes" -- **can** be
checked, because real logits now exist.

- **Post-softmax FP32 quantization: confirmed**, exactly as in v1,
  now on the real governing model (Task 3).
- **Logit ordering is NOT lost before softmax.** At u>5.5, every one of
  2,239 tail events has a distinct `Delta`; the `Delta`-by-bin boxplot
  shows smooth, monotonic, non-overlapping medians across 12 probability
  bins. Even at looser cuts the picture barely changes: at u>3.5,
  9,258/9,258 events split into 9,252 distinct `Delta` values (6 exact
  coincidental ties, 0.065%); at u>4.5, 6,454 events split into 6,449
  distinct values (5 ties, 0.077%). The underlying ranking the network
  actually computes is essentially fully preserved; what collapses is
  only the coarse, 24-bit *display/storage* representation of that
  ranking near 1.
- **Not evidence the model learned something pathological.** A
  well-separated classifier (AUC ~0.969-0.978) is expected to produce
  many very-high-confidence events; some of them will differ in true
  confidence by less than 1 part in 2^24 relative to 1, which is
  exactly what float32 cannot resolve.
- **Not evidence FP32 training is invalid.** Training used
  `precision="32-true"` (v1 finding); this quantization is a property
  of the 32-bit *storage* of a probability near 1, orthogonal to
  whether 32-bit arithmetic was adequate during training.
- **The model is not called calibrated anywhere in this report.**
  Nothing here measures calibration.

`LOGIT_MARGIN_SMOOTH_THROUGH_SPIKES = YES.`

---

## FP32 relevance at u>3.5 and u>4.5 (Harvey's physics regions)

| Threshold | FP32 lattice steps remaining to 1 (`k` at threshold) | events above threshold (signal / background) | distinct float32 values used | mean events per distinct value |
|---|---:|---:|---:|---:|
| u>3.5 | ~5,305 | 9,258 / **0** | 1,705 | 5.43 |
| u>4.5 | ~531 | 6,454 / **0** | 266 | 24.26 |
| u>5.5 (for reference) | ~53 | 2,239 / **0** | 27 | 82.9 |

**Interpretation.** As the threshold tightens, the number of available
float32 slots shrinks geometrically (~5,305 -> ~531 -> ~53 for a
1-decade change in u each time) while the population of very-confident
signal events shrinks more slowly, so the average occupancy per
available float32 value rises sharply (5.4 -> 24.3 -> 82.9 events/value)
-- FP32 collisions among individual signal events become common well
before u=5.5, and are already measurable at u=3.5.

**Does this materially affect interpretation at u>3.5/u>4.5? No.**
Three independent reasons, each checked directly rather than assumed:

1. **Zero background events** populate either region in this 400,000-event
   cohort. Any background-yield or background-rejection estimate at
   these thresholds is unaffected by FP32 quantization because there is
   nothing there to quantize (it is exactly zero either way, at full
   float64 precision or float32).
2. **Logit ordering survives.** Even at these looser thresholds,
   99.9%+ of events retain a distinct `Delta` (9,252/9,258 at u>3.5;
   6,449/6,454 at u>4.5) -- any analysis that reasons in logit/Delta
   space, or that only needs raw pass/fail counts against a threshold
   (not a full re-ranking of tied events), is not distorted.
3. **Threshold-based physics quantities are unaffected by ties.** The
   working-point machinery this project already uses
   (`roc_auc_and_working_points` in `evaluate_classification.py`) only
   ever counts `probs >= threshold`; float32 ties among events that all
   clear (or all miss) a threshold change nothing about that count.
   Ties would only matter for exercises that need a strict total order
   among the tied events themselves (e.g. picking "the single most
   signal-like event") -- not for efficiency/rejection/yield reporting.

`FP32_QUANTIZATION_AFFECTS_U35_U45_INTERPRETATION = NO`, with the
above caveat stated precisely rather than glossed over: FP32 *does*
create real, measurable score-level ties among individual high-confidence
signal events in these regions; it does not corrupt any of the
aggregate physics quantities this project actually reports there, and
it is moot for background because no background event reaches these
regions at all in this cohort.

---

## Files in this package

- `STATUS.md` -- Steps 1-5 identity/environment verification, written first
- `HARVEY_FP32_10M_FINAL_REPORT.md` -- this file
- `HARVEY_FP32_EMAIL_PARAGRAPH.md` -- one-paragraph summary for Harvey
- `SPA10M_EVENT_LOGITS_400K.parquet` -- full 400,000-event per-event archive (real, this task's inference)
- `SPA10M_FP32_UNIQUE_SCORE_CENSUS.csv` -- Task 3, u>5.5, 27 rows
- `SPA10M_LATTICE_BINS_K1_TO_K64.csv` -- explicit k=1..64 + exact-1.0 census, all 5 populations
- `SPA10M_FP32_VS_LOGIT_BIN_SUMMARY.csv` -- per-bin Delta/u_logit distinctness summary
- `figures/u_prob32_tail_10M.png`, `figures/u_logit_tail_10M.png`,
  `figures/u_prob32_vs_u_logit.png`, `figures/delta_by_fp32_bin.png`
- `scripts/run_governing_10m_tail_inference.py` -- the actual inference script that was run
- `scripts/build_census_and_figures.py` -- post-processing (executed)
- `work/run_governing_10m_meta.json`, `work/census_summary_v2.json`, `work/work_inference.log` -- raw run records
- `receipt.json`, `SHA256SUMS`

## Explicitly not done

No training, no gradient update anywhere (`model.eval()` +
`torch.no_grad()` throughout the one inference script that was run). No
Stage-C/holdout_B file opened -- only `production_2M_val.h5`, the same
frozen 400,000-event development cohort every prior evaluation in this
project used. No EAF GPU allocation used or requested -- this ran
CPU-only on the local host, `torch.cuda.is_available()==False`,
`options.num_gpu=0`. No ParT code, checkpoint, or embedding referenced
anywhere in this package. No package installed, uninstalled, or
upgraded -- the pinned `.pixi/envs/default` environment was used exactly
as found; the one broken import found there (`pandas`) was worked
around by using `pyarrow` directly, not fixed. No inference command
failed silently -- the one inference command run exited 0 and its full
stdout/stderr is preserved verbatim in `work/work_inference.log`.
