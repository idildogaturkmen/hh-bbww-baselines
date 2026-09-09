# How much QCD MC is needed to double the SPA-Net compressed-tail statistics?

**Scope.** Read-only analysis of already-frozen artifacts. No Condor/EAF job launched,
no scoring run, no frozen physics result modified. `NEW_JOB_LAUNCHED = NO`.

**u-transform, frozen convention (unchanged from `track_b_harvey_spanet_compressed_tail_20260903_v2`):**
`u = -log10(1 - score)`, where `score` is the SPA-Net 10M class-1 softmax output
(`fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5`). Harvey's
"u > 3/5" is interpreted as **u > 3.5** because the previously-used cut
`score >= 0.9997` maps to `u = 3.5229`, i.e. "3.5" in this notation. A second
region, **u > 4.5** (`score >= 0.99996838`, close to the previously-used
`score >= 0.99997` -> `u = 4.5229`), is studied in parallel per the task instructions.

All current-statistics numbers below are reused unchanged from
`HARVEY_TAIL_STATS_CURRENT.csv` in this package (exact per-event counts from the
frozen `FULL_QCD_SURVIVOR_SIDECAR.h5` and the non-QCD `job_summary_*.json`
survivor lists, both belonging to the already-existing
`track_b_harvey_tail_characterization_20260825_v1` /
`track_b_harvey_complete_3to5pct_study_20260825_v1` packages).

## 1. What "doubling the tail" means for *ordinary* generation

For ordinary (non-importance-sampled) Pythia8+Delphes generation at fixed
generator-level cuts and fixed detector/selection/scoring chain, the
acceptance into **any** fixed score region (including both u>3.5 and u>4.5)
is a fixed efficiency of the total generated sample. Doubling the raw
survivor count in a given tail region therefore requires **doubling the
entire generated QCD sample** — there is no way, under *ordinary* generation,
to double one tail region without also (approximately) doubling every other
region, including the bulk. This is a direct, model-free consequence of
fixed-acceptance sampling, not an estimate.

**Consequence: the additional generated-event requirement is the same for
u>3.5 and u>4.5** — both need a full second production of the current QCD
sample's size. (A *targeted*, importance-sampled campaign, e.g. native Pythia
`bias2Selection` biasing on `pTHat`, could reach a *given* tail-count target
far more cheaply and *could* in principle be tuned differently for u>3.5 vs
u>4.5 — see `HARVEY_PRESELECTION_CANDIDATES.md` Part 5 and the existing
`track_b_qcd_tail_generation_strategy_20260902_v1` package. That is
explicitly **not** what this Task asks for here: Task 2 asks for the
*ordinary*-generation, fixed-normalization case only.)

## 2. Current QCD generation/exposure statistics (exact, author-confirmed)

| quantity | value | source |
|---|---:|---|
| QCD1 (`forInfer`) generated events | 88,000,000,000 (17,600 x 5,000,000) | author formula, `AUTHOR_QCD_NORMALIZATION_AND_DATASET_USAGE_CLARIFICATION.md` |
| QCD2 (`forInfer2`) generated events | 190,360,000,000 (38,072 x 5,000,000) | same |
| **QCD generated total** | **278,360,000,000** | same; also independently stated as the "current author generation denominator" in `TENFOLD_BACKGROUND_FEASIBILITY_MEMO.md` Sec.1 |
| QCD1 selected/stored (post-Delphes, post "4j3b or 4j2b") | 27,683,689 | `exact_dataset_counts.tsv` |
| QCD2 selected/stored | 59,939,617 | same |
| **QCD selected/inference total** | **87,623,306** | `NORMALIZATION_CONSTANTS.json` |
| implied overall selection efficiency | 3.148e-4 (0.03148%) | 87,623,306 / 278,360,000,000 |
| QCD flat weight at 450 fb-1 | 7.3112875413134075 | `NORMALIZATION_CONTRACT.md`; formula `lumi_scale x 4522600.0 x 100000.0 / ((17600+38072) x 5,000,000)`, `lumi_scale=4.5` |
| generator chain | direct Pythia8 HardQCD-style multijet, single inclusive `pTHat > 75 GeV` slice (NOT MG5/LHE) | `CURRENT_QCD_GENERATION_AUDIT.md` |
| `gen_weight` | exactly 1.0 for every checked QCD event (unweighted generation) | `track_b_harvey_score_tail_diagnostics_20260829_v5_background_stats_audit` Sec.6 |

**Not resolved / not locally available (stated, not guessed):** the process/run
cards for the governing QCD sample itself (it is IHEP-side, received as
finished ntuples, `CURRENT_QCD_GENERATION_AUDIT.md` Sec.4); the exact physical
units/derivation of the `4522600.0` and `100000.0` constants in the weight
formula (status `STRONG_CANDIDATE`, not `PROVEN`); a per-event generator
`pTHat` branch (does not exist in the frozen ntuples).

## 3. Additional generated-QCD-event count required to approximately double each tail

Doubling requires generating a second, independent QCD production of
**the same size as the current one**:

| target | additional generated events (author-denominator convention) | additional selected/inference events | resulting raw tail count (approx., ordinary generation) |
|---|---:|---:|---|
| double u>3.5 | **+278,360,000,000** | **+87,623,306** | all-bkg 110 -> ~220 (QCD 68->~136) |
| double u>4.5 | **+278,360,000,000** | **+87,623,306** | all-bkg 15 -> ~30 (QCD 12->~24) |

(The "resulting raw tail count" column is the expected value under Poisson
statistics at 2x exposure; see Sec.4 for the associated uncertainty, which is
non-negligible at these small counts, especially at u>4.5.)

## 4. Statistical uncertainty after 2x / 4x / 10x QCD MC statistics (physical yield held fixed)

Reusing the corrected v5 methodology (`track_b_harvey_score_tail_diagnostics_
20260829_v5_background_stats_audit/ERRATA.md`, Correction 1): generating N x
more QCD MC at the **same** cross-section/luminosity does **not** change the
physical prediction `B_QCD` — it holds `B_QCD` fixed while the per-event
weight scales down by 1/N and the raw survivor count scales up by N, so only
the **precision** (`N_eff`, `sum_w^2`) improves:

```
sum_w2_QCD(Nx) = sum_w2_QCD(1x) / N        (weight -> weight/N, count -> count*N)
sum_w2_total(Nx) = sum_w2_QCD(Nx) + sum_w2_ttbar(fixed) + sum_w2_other(fixed)
N_eff(Nx) = B_total(fixed)^2 / sum_w2_total(Nx)
rel_unc(Nx) = 1 / sqrt(N_eff(Nx))
```

### u > 3.5 (current: n_QCD=68, B_total=588.94, N_eff=89.31, rel.unc.=10.58%)

| QCD-stat multiplier | n_QCD (raw) | QCD-only N_eff / rel.unc. | combined all-bkg N_eff | combined rel.unc. |
|---:|---:|---|---:|---:|
| 1x (current) | 68 | 68.0 / 12.13% | 89.31 | 10.58% |
| 2x | 136 | 136.0 / 8.57% | 167.86 | 7.72% |
| 4x | 272 | 272.0 / 6.06% | 299.63 | 5.78% |
| 10x | 680 | 680.0 / 3.83% | 566.42 | 4.20% |

### u > 4.5 (current: n_QCD=12, B_total=97.26, N_eff=13.99, rel.unc.=26.73%)

| QCD-stat multiplier | n_QCD (raw) | QCD-only N_eff / rel.unc. | combined all-bkg N_eff | combined rel.unc. |
|---:|---:|---|---:|---:|
| 1x (current) | 12 | 12.0 / 28.87% | 13.99 | 26.73% |
| 2x | 24 | 24.0 / 20.41% | 26.62 | 19.38% |
| 4x | 48 | 48.0 / 14.43% | 48.51 | 14.36% |
| 10x | 120 | 120.0 / 9.13% | 95.77 | 10.22% |

(The u>4.5, 1x row reproduces, to 2 decimal places, the already-frozen
`N_eff=13.99`, `rel.unc.=26.73%` at the eps_S=3.5% fine-scan working point
in `track_b_harvey_score_tail_diagnostics_20260829_v5_background_stats_audit`
and `track_b_harvey_followup_finescan_likelihood_20260829_v1` — an
independent cross-check, not a coincidence, since that working point's score
threshold, u=4.5229, is only 0.023 away from the u=4.5 studied here and
happens to contain the identical 15-event population.)

**Reading the table:** because the minor backgrounds (ttbar + 14 "other"
processes) are *not* regenerated in this QCD-only scenario, their fixed
`sum_w2` becomes an increasing floor as QCD's own variance shrinks — this is
why the combined N_eff improves by *less* than the multiplier at large N
(e.g. at u>3.5, 10x QCD stats gives only a 6.3x improvement in combined
N_eff, not 10x), exactly the effect already documented in the v5 audit.

## 5. Generated vs. selected vs. weighted — kept explicitly distinct

- **Generated event count**: raw Pythia8+Delphes output before any selection
  (278,360,000,000 currently; +278,360,000,000 to double).
- **Tail survivor count**: raw events landing above u=3.5 or u=4.5 after the
  full selection+scoring chain (68 / 12 for QCD currently; ~136 / ~24 after
  doubling, in expectation).
- **Weighted physical background yield (B)**: the fixed, luminosity-scaled
  physical prediction (588.94 / 97.26 at u>3.5 / u>4.5, respectively) — this
  number is a property of the true cross-section and luminosity and does
  **not** change when more MC is generated; only its statistical precision
  (N_eff, rel. MC uncertainty) does.

See `HARVEY_GENERATION_RESOURCE_ESTIMATE.md` for the CPU-hour/wall-time/
storage/scoring cost of producing the additional generated events in Sec.3.
