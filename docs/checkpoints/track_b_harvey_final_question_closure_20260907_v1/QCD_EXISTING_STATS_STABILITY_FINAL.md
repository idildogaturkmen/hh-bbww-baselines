# Empirical QCD tail-yield stability, using only the existing frozen production

**This tests stability WITHIN the two currently existing QCD production
lanes (`inference_qcd_1`/`forInfer`, `inference_qcd_2`/`forInfer2`). It does
NOT replace validation with a future, newly-generated independent sample —
that is a different, unexecuted question (see
`HARVEY_MC_DOUBLING_ESTIMATE.md`/`HARVEY_GENERATION_RESOURCE_ESTIMATE.md`,
unchanged by this task).**

Read-only. No Condor/EAF job launched. No new MC generation. Inputs:
`FULL_QCD_SURVIVOR_SIDECAR.h5` (557 files, 87,623,306-event union, every QCD
event above the SPA10M 7.5%-efficiency WP `score>=0.998958945274353` — looser
than both `u>3.5` and `u>4.5`, so both tails are fully contained) and
`all_file_tree_entries.tsv` (independently measured, exact per-file selected
entry counts, summing to 27,683,689 / 59,939,617 — bit-for-bit matching the
already-audited lane totals).

## 1. Per-file generated-event verification (the blocker check, per instruction)

| lane | jobs (author-stated) | merged files | jobs/file | uniform 5,000,000-event-job count per file provable? |
|---|---:|---:|---:|---|
| `inference_qcd_1` (forInfer) | 17,600 | 176 | **100.000** (exact integer) | **YES** — 176 × (100×5,000,000) = 88,000,000,000, exact match to the author-confirmed lane total |
| `inference_qcd_2` (forInfer2) | 38,072 | 381 | 99.9265... (not an integer) | **NO** — no per-mergeid job-count manifest exists anywhere in this project's frozen record |

**Blocker disclosed, not hidden:** lane 2's per-file generated-event count is
therefore reported as an **approximation** (each file's generated exposure
scaled from its own exact, measured selected-event count by the lane's
overall generated/selected efficiency, 190,360,000,000/59,939,617 = 3.1765×),
supported by two pieces of corroborating evidence but not independently
proven: (a) per-file selected-event counts are tightly uniform within lane 2
(mean ≈157,321, low file-to-file spread), and (b) the two lanes' overall
generated/selected efficiencies agree to 0.1% (3.1459e-4 vs 3.1488e-4).
Files are ordered by their `mergeid` (0, 1, 2, ...) within each lane — the
only deterministic ordering available; this is disclosed as a convention,
not proven to equal true chronological generation/completion order.

## 2. Cumulative checkpoints (8 geometric steps + lane boundary), combined lane1→lane2 order

Full table: `QCD_CUMULATIVE_STABILITY.csv`. Selected rows:

| frac. of final selected | cum. generated events (≈, lane2 approx.) | raw survivors u>3.5 | raw survivors u>4.5 | running B(u>3.5), 450 fb⁻¹ | running B(u>4.5), 450 fb⁻¹ |
|---:|---:|---:|---:|---:|---:|
| 12.5% | 3.50e10 | 12 | 3 | 697.8 | 174.4 |
| 25% | 7.00e10 | 18 | 4 | 523.3 | 116.3 |
| 50% | 1.394e11 | 36 | 6 | 525.4 | 87.6 |
| 100% (full sample) | 2.7836e11 | 68 | 12 | 497.2 | 87.7 |

The running B estimate (raw survivors so far × weight computed from the
generated exposure so far, i.e. "what would this analysis have reported had
it stopped at this point") visibly fluctuates at low cumulative exposure
(697.8 → 523.3 → 525.4) and settles toward the full-sample value (497.2) as
exposure grows — exactly the qualitative behavior expected of a converging,
statistically well-behaved MC estimate, not a red flag by itself. The u>4.5
running estimate (174.4 → 116.3 → 87.6 → 87.7) is far noisier at low
cumulative exposure, consistent with its much smaller final raw count (12).

Figures: `figures/qcd_tail_rate_vs_generated_events.png` (raw cumulative
survivor counts vs. exposure), `figures/qcd_u35_yield_stability.png`,
`figures/qcd_u45_yield_stability.png` (running rate with Poisson error bars
and the final rate overlaid).

## 3. Quantitative stability verdicts

A Pearson chi-square dispersion statistic compares each checkpoint's
*incremental* survivor count (new survivors in that exposure slice) against
the count expected under one constant underlying rate for the whole
combined sample (8 checkpoints, 7 degrees of freedom).

| threshold | total raw N | χ²/dof | verdict |
|---|---:|---:|---|
| u>3.5 | 68 | 0.96 | **STABLE_WITHIN_CURRENT_MC** |
| u>4.5 | 12 | — (below the 20-event minimum this task requires before attempting a dispersion judgment) | **TOO_FEW_EVENTS_TO_JUDGE** |

χ²/dof ≈ 1 at u>3.5 is exactly the signature of a constant-rate Poisson
process — no evidence of a trend, jump, or lane-transition artifact in the
data as currently accumulated. This is not declared "stable" merely because
an uncertainty band shrinks (that would be true even under a real trend);
it is declared stable because the *increments themselves* are statistically
consistent with one constant rate.

At u>4.5, 12 raw events cannot support this test at all — reported
descriptively (the table/figure exist) but explicitly not judged.

## 4. QCD1 vs QCD2 independent comparison

| | u>3.5 | u>4.5 |
|---|---:|---:|
| n(QCD1) | 22 | 5 |
| n(QCD2) | 46 | 7 |
| expected QCD1 fraction (generated-exposure ratio) | 0.3161 | 0.3161 |
| observed QCD1 fraction | 0.3235 | 0.4167 |
| exact binomial p-value (rate compatibility) | **0.897** | **0.536** |

Both lanes are statistically compatible with sharing one true tail rate,
proportional to generated exposure — no tension between the two independent
productions at either threshold, though u>4.5's compatibility test itself
carries very little power at n=12.

**Kinematic composition** (medians, u>3.5 survivors): QCD1 HT=590 GeV,
mHH=811 GeV, R_HH=143; QCD2 HT=515 GeV, mHH=725 GeV, R_HH=151 — broadly
similar, no qualitative shape difference between lanes at this raw-count
level. Figure: `figures/qcd1_vs_qcd2_tail_composition.png`.

## 5. Bottom line

- **u>3.5: STABLE_WITHIN_CURRENT_MC.** The 68 raw QCD survivors accumulated
  consistently with one constant tail rate as the existing two production
  lanes were processed, and the two lanes agree with each other within
  Poisson statistics.
- **u>4.5: TOO_FEW_EVENTS_TO_JUDGE.** 12 raw events is not enough to
  distinguish a genuine trend from statistical noise; this is reported
  honestly rather than forcing a stability verdict the data cannot support.
- This is an empirical read of the *existing* sample only. It says nothing
  about whether a *future*, independently generated QCD sample would confirm
  the same rate — that requires new generation, explicitly out of scope here.
