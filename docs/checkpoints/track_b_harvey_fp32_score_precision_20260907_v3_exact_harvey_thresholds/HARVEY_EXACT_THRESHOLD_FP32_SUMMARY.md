# v3 -- exact-Harvey-threshold FP32 vs. raw-logit audit

Status: **COMPLETE.** Purely additive, post-processing-only package.
Sole input: `../track_b_harvey_fp32_score_precision_20260906_v2_governing10m/SPA10M_EVENT_LOGITS_400K.parquet`
(SHA256 `bb282dac7717d8e5145e5c2f842beb9e1b6a6ac2d64c95ab04aa2a40ad0d5412`,
re-verified unchanged at the end of this task -- see `receipt.json`).
No model, checkpoint, HDF5, GPU, or training touched anywhere in this
package (AST-verified, not just grepped -- see Task 6).
`NEW_INFERENCE_LAUNCHED = NO`, `NEW_TRAINING_LAUNCHED = NO`.

This answers Harvey's own literal thresholds -- **score > 0.9997** and
**score > 0.99997** -- rather than the nearby rounded u=3.5/4.5 cuts
used in v2.

---

## Task 1 -- exact threshold definitions

Computed independently two ways: (a) float64 `math.log10`/`math.log`,
(b) 50-digit `decimal` arithmetic (`scripts/adversarial_self_check.py`,
Task 6). Both agree to better than 1e-10.

| Harvey threshold | u = -log10(1-score) | Delta = ln(score/(1-score)) |
|---|---:|---:|
| score > 0.9997 | **3.5228787452803854** | **8.11142803829918** |
| score > 0.99997 | **4.522878745280707** | **10.41428317585296** |

(Matches the approximate values Harvey stated, ~3.522878745... and
~4.522878745..., to all quoted digits.) A third, independent
round-trip check (`u` recomputed from `Delta` via the same stable
`softplus(Delta)/ln(10)` formula the v2 parquet uses) reproduces both
`u` values to better than 1e-12 -- the two thresholds are exactly the
same real number expressed in score/`u`/`Delta` form, not three
different cuts.

---

## Task 2 -- event-by-event FP32 vs. raw-logit threshold crosscheck

Exact `>` semantics both sides (never `>=`) -- explicitly checked in
Task 6 to confirm this choice does not matter here (no event lands
exactly on either threshold).

**PRODUCTION DECISION:** `score_float32_production > threshold`
**RAW-LOGIT DECISION:** `delta > exact_float64_logit_threshold`

| Threshold | Population | N pop | N pass FP32 | N pass logit | FP32-only | logit-only | disagreements | disagreement fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| >0.9997 | signal | 193,358 | 9,170 | 9,170 | 0 | 0 | **0** | 0% |
| >0.9997 | qcd | 174,485 | 0 | 0 | 0 | 0 | 0 | -- |
| >0.9997 | ttbar | 32,157 | 0 | 0 | 0 | 0 | 0 | -- |
| >0.9997 | all_background | 206,642 | 0 | 0 | 0 | 0 | 0 | -- |
| >0.9997 | full_cohort | 400,000 | 9,170 | 9,170 | 0 | 0 | **0** | 0% |
| >0.99997 | signal | 193,358 | 6,368 | 6,370 | 0 | **2** | **2** | 0.00104% (of signal) |
| >0.99997 | qcd | 174,485 | 0 | 0 | 0 | 0 | 0 | -- |
| >0.99997 | ttbar | 32,157 | 0 | 0 | 0 | 0 | 0 | -- |
| >0.99997 | all_background | 206,642 | 0 | 0 | 0 | 0 | 0 | -- |
| >0.99997 | full_cohort | 400,000 | 6,368 | 6,370 | 0 | **2** | **2** | 0.0005% (of cohort) |

Every disagreement, every background population, at both of Harvey's
literal thresholds, in this 400,000-event development cohort, comes
from **QCD/ttbar never reaching either threshold at all** (0 pass
either way) -- not from a background-specific FP32/logit discrepancy.

### The 2 exact disagreements at score > 0.99997

Both are `only_logit_pass`: the raw logit margin says these events'
true confidence exceeds 0.99997, but the float32-rounded production
score does not. Full row detail in
[`SPA10M_EXACT_THRESHOLD_DISAGREEMENTS.csv`](SPA10M_EXACT_THRESHOLD_DISAGREEMENTS.csv):

| row_index | process_label | score_float32 | u_prob32 | delta | u_logit |
|---:|---|---:|---:|---:|---:|
| 245484 | signal | 0.9999699592590332 | 4.522289359 | 10.414567471 | 4.523002209 |
| 352452 | signal | 0.9999699592590332 | 4.522289359 | 10.414351463 | 4.522908401 |

Both share the identical stored float32 score
(`0.9999699592590332`), which sits 4.07e-8 below the literal 0.99997
threshold in real-number terms -- close enough that float32's ~6e-8
resolution at this magnitude pulls the displayed score to the wrong
side of the line, while the two events' distinct raw logit margins
(10.414567 vs. 10.414351) both genuinely exceed the threshold's exact
`Delta`. **This is a real, directly-measured effect, not a tie among
"identical" events** -- the two events have distinct `Delta`/`u_logit`
values and are not model outputs that are "the same"; they merely
happen to round to the same stored float32 probability.

---

## Task 3 -- exact tail counts at Harvey's thresholds

**Scope note:** every count below is within this specific 400,000-event
development-cohort numerical audit. It is not a measurement of, and
does not stand in for, the separate, much larger full physical-
background production study.

| Threshold | signal | QCD | ttbar | all-background | distinct float32 probs | distinct Delta | distinct u_logit | exact score==1.0 | fraction of tail events FP32-tied | max multiplicity, one value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| >0.9997 | 9,170 | 0 | 0 | 0 | 1,637 | 9,164 | 9,164 | 67 | **92.37%** | **113** |
| >0.99997 | 6,368 | 0 | 0 | 0 | 252 | 6,363 | 6,363 | 67 | **100.00%** | 113 |

**"Fraction of tail events FP32-tied" is the headline number for
Harvey**, and it is much larger than the handful of threshold-crossing
disagreements above: even at score > 0.9997, **92.4% of the 9,170
surviving signal events already share their stored float32 probability
with at least one other event**; at score > 0.99997, **every single
surviving event (100%)** does. This is compression of the *displayed/
stored* value, not evidence the underlying events are indistinguishable
-- 9,164 of 9,170 events (99.93%) and 6,363 of 6,368 (99.92%) keep a
fully distinct raw `Delta`/`u_logit`.

### Supplementary table (0.9999 / 0.99999 -- no new inference, same parquet)

| Threshold | signal | QCD | ttbar | all-background | distinct float32 probs | distinct Delta | exact score==1.0 | fraction FP32-tied | max multiplicity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| >0.9999 | 7,930 | 0 | 0 | 0 | 766 | 7,925 | 67 | 98.49% | 113 |
| >0.99999 | 4,400 | 0 | 0 | 0 | 84 | 4,398 | 67 | 100.00% | 113 |

Full CSVs:
[`SPA10M_EXACT_THRESHOLD_TAIL_CENSUS.csv`](SPA10M_EXACT_THRESHOLD_TAIL_CENSUS.csv),
[`SPA10M_SUPPLEMENTARY_0P9999_0P99999.csv`](SPA10M_SUPPLEMENTARY_0P9999_0P99999.csv).

---

## Task 4 -- what the compression means

Figure: [`figures/harvey_exact_threshold_compression.png`](figures/harvey_exact_threshold_compression.png)
-- `u_logit` (continuous, x-axis) vs. `u_prob32` (stored float32,
y-axis), restricted to signal events near and above the plot floor
u=2.92. Vertical/horizontal reference lines mark the exact `u`
equivalents of both 0.9997 and 0.99997. Score==1.0 events (67 of them)
are drawn as triangles at a fixed, clearly-labeled finite y-coordinate
purely so they can appear on the plot -- **their true `u_prob32` is
+infinity, not that coordinate**; the legend and caption both state
this explicitly, and no line or annotation implies the plotted
coordinate is the true stored value.

Visible in the figure: near the top (u_logit gtrsim 6.7), the stored
`u_prob32` collapses into a handful of flat horizontal bands even
though `u_logit` continues to spread out -- the visual definition of
FP32 compression. In the lower-left, the scatter looks like a smooth
diagonal; **this is a resolution effect of the plot, not the absence of
compression** -- the Task 3 census already showed 92.4% of events are
FP32-tied even at score > 0.9997 (u approx 3.52), because at that
magnitude there are enough distinct float32 slots (1,637) for
consecutive steps to look continuous at this zoom level, while still
being heavily populated by more than one event apiece on average.

### Tie-examples table: one stored probability, many distinct logit margins

The single largest FP32 tie in this cohort above score 0.9997 is at
stored value `0.9999994039535522` (this is exactly the `k=10` point on
the `2^-24` float32 lattice below 1, `u_prob32=6.224720`): **113
signal events** share this one stored probability. Their raw `Delta`
spans `14.238361` to `14.435105` -- a real, non-degenerate spread -- and
**all 113 have distinct `Delta` values** (113 distinct out of 113).
Ten representative examples (5 highest-Delta, 5 lowest-Delta members of
this one bin; full 113-row detail not needed to see the pattern), from
[`SPA10M_TIE_EXAMPLES_TABLE.csv`](SPA10M_TIE_EXAMPLES_TABLE.csv):

| row_index | shared stored score (float32) | Delta | u_logit |
|---:|---:|---:|---:|
| 282824 | 0.9999994039535522 | 14.435105 | 6.269087 |
| 295471 | 0.9999994039535522 | 14.434672 | 6.268899 |
| 196032 | 0.9999994039535522 | 14.431667 | 6.267594 |
| 180828 | 0.9999994039535522 | 14.430520 | 6.267095 |
| 269610 | 0.9999994039535522 | 14.428962 | 6.266419 |
| ... (103 more, all distinct) | | | |
| 362775 | 0.9999994039535522 | 14.242622 | 6.185492 |
| 342011 | 0.9999994039535522 | 14.238361 | 6.183642 |

These 113 events share an identical *stored probability* -- they do
**not** share identical model outputs; their logit margins differ by up
to 0.197 (a genuine, orderable spread of about 8.5% of `Delta`'s own
range in this bin), and the model ranks them differently even though
the 32-bit probability representation cannot show that ranking.

---

## Task 5 -- Harvey-facing documents

See [`HARVEY_EXACT_THRESHOLD_EMAIL_PARAGRAPH.md`](HARVEY_EXACT_THRESHOLD_EMAIL_PARAGRAPH.md).

---

## Task 6 -- adversarial self-check (independently re-verified, not merely re-asserted)

All performed in `scripts/adversarial_self_check.py`, results in
`work/adversarial_self_check.json`:

1. **Threshold conversion, second method.** 50-digit `decimal`
   recomputation of both `u` and `Delta` at both thresholds agrees with
   the float64 `math.log10`/`math.log` path used in Task 1 to better
   than 1e-10 in both cases. **PASS.**
2. **Pass counts, second method.** A plain Python `for`-loop over the
   signal population's score/Delta lists (not numpy boolean masking)
   reproduces `n_pass_fp32` and `n_pass_logit` exactly at both
   thresholds (9,170/9,170 and 6,368/6,370). **PASS.**
3. **`>` vs. `>=` semantics.** Explicitly checked: `n(score > thr)`
   equals `n(score >= thr)` at both thresholds (no event in this cohort
   lands exactly on either literal threshold value), so the strict `>`
   Harvey specified was actually applied and, separately, would not
   have differed from `>=` here even if that mattered. **PASS,
   confirmed not merely assumed.**
4. **score==1.0 / `+inf` handling.** `n(score_float32==1.0) == 67`
   exactly equals `n(u_prob32 non-finite) == 67`. **PASS.**
5. **v2 package untouched.** `SPA10M_EVENT_LOGITS_400K.parquet`
   SHA256 re-hashed at the end of this task and matches the value
   recorded in v2's own `SHA256SUMS` exactly. **PASS.**
6. **No new inference.** An AST-based import check (not a naive
   substring grep -- a first attempt at a substring grep produced a
   self-referential false positive, since this very script must name
   the forbidden module strings to check for them; replaced with
   parsing actual `import`/`from...import` statements) confirms none of
   this package's three scripts import `torch`, `spanet`, `h5py`, or
   `pytorch_lightning`. **PASS.**
7. **No background-scope overclaim.** Every count above is explicitly
   labeled as being within this 400,000-event development cohort; nowhere
   in this package is "zero background" stated without that
   qualification, and nowhere does this package claim to represent the
   full physical-background production study.
8. **File-writing bug, caught by this same adversarial pass, disclosed
   here.** `task3_tail_census()` originally wrote its CSV/JSON outputs
   to fixed filenames regardless of which thresholds were passed to it;
   calling it a second time for the 0.9999/0.99999 supplementary table
   silently overwrote `SPA10M_EXACT_THRESHOLD_TAIL_CENSUS.csv` and
   `work/task3_tail_census.json` with the supplementary numbers. **This
   was a file-writing/filename bug only** -- the in-memory census dict
   used everywhere else in this report, `receipt.json`, and
   `work/master_results.json` was already correct throughout, since
   Python's `task3 = task3_tail_census(...)` bound the correct
   dictionary to a distinct local variable before the second call ran.
   Fixed by parameterizing the output filenames
   (`json_name`/`csv_name` arguments); the script was rerun end-to-end
   and every number in this report and in `receipt.json` was
   re-confirmed unchanged (9,170/9,170/0 at >0.9997; 6,368/6,370/2 at
   >0.99997; 7,930 and 4,400 signal at the two supplementary
   thresholds). `SPA10M_EXACT_THRESHOLD_TAIL_CENSUS.csv` and
   `SPA10M_SUPPLEMENTARY_0P9999_0P99999.csv` are now confirmed distinct
   files holding their own intended thresholds.

---

## Files in this package

- `STATUS.md`, `HARVEY_EXACT_THRESHOLD_FP32_SUMMARY.md` (this file),
  `HARVEY_EXACT_THRESHOLD_EMAIL_PARAGRAPH.md`
- `SPA10M_EXACT_THRESHOLD_DISAGREEMENTS.csv` -- Task 2, 2 rows
- `SPA10M_EXACT_THRESHOLD_TAIL_CENSUS.csv` -- Task 3, Harvey's 2 literal thresholds
- `SPA10M_SUPPLEMENTARY_0P9999_0P99999.csv` -- Task 3 supplementary
- `SPA10M_TIE_EXAMPLES_TABLE.csv` -- Task 4, the 113-event tie example
- `figures/harvey_exact_threshold_compression.png` -- Task 4
- `scripts/exact_threshold_analysis.py`, `scripts/build_harvey_figure.py`,
  `scripts/adversarial_self_check.py` -- all executed, all post-processing only
- `work/task1_exact_thresholds.json`, `work/task2_crosscheck_by_population.json`,
  `work/task3_tail_census.json`, `work/master_results.json`,
  `work/adversarial_self_check.json` -- raw computation records
- `receipt.json`, `SHA256SUMS`

## Explicitly not done

No model inference, no checkpoint load, no HDF5 access, no training, no
gradient computation, no EAF/GPU use, no Stage-C/holdout_B reference, no
ParT code/checkpoint/embedding, no package installed/uninstalled/
upgraded (AST-verified for imports; no new environment used at all --
this package needs only `numpy`/`pyarrow`/`matplotlib`, already used by
v1/v2). The existing v2 package was not modified or deleted (SHA256
re-verified unchanged). This is a 400,000-event development-cohort
numerical audit; it does not claim to represent, and should not be read
as, the separate full physical-background production study, and it does
not call the score calibrated or claim FP32 training is proven optimal.
