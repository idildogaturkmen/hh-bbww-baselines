# Surviving ttbar characterization (Task 5)

**v2 Correction 2 applied: the "13 of 27 removed" wording in v1 was a prose bug -- 13 is the number of ttbar events SURVIVING the cut, not the number removed. The correct count is 14/27 = 51.85% removed, 13/27 remaining. The underlying computed fraction (0.518519) was always arithmetically correct in v1; only the prose was wrong. Corrected throughout below.**

**EXPLORATORY, NOT VALIDATED: the `pT_H1 < 406.115 GeV` cut below is the median of the same 27-event ttbar sample it is then evaluated on. It is reported as a candidate observation, not a held-out or cross-validated result. No independent test sample exists to validate it against.**

**Raw support used for every conclusion below: 27 raw ttbar events at u>3.5 (search/selection population); 2 raw ttbar events at u>4.5 (descriptive check only, never used for selection).**

## What distinguishes surviving ttbar from signal

1D signal-vs-ttbar AUC ranking (u>3.5, n_sig=27,339, n_ttbar=27):

| feature | AUC (signal vs ttbar) | median, signal | median, ttbar |
|---|---:|---:|---:|
| pT_H1 | 0.126 | 173.0 GeV | 406.1 GeV |
| jet1_pt | 0.144 | 165.2 GeV | 454.8 GeV |
| HT | 0.145 | 456.5 GeV | 910.3 GeV |
| pT_H2 | 0.172 | 146.8 GeV | 392.0 GeV |
| jet2_pt | 0.190 | 114.0 GeV | 309.1 GeV |
| met_pt | 0.261 | 49.2 GeV | 106.9 GeV |

**Surviving ttbar events are systematically much harder/boosted than signal** — roughly 2–2.5× higher leading-jet pT, Higgs-candidate pT, and HT. This is a coherent physical picture: the ttbar events that survive to this extreme classifier tail are boosted, high-multiplicity events whose jet kinematics happen to mimic the hard scale of genuine HH→4b, not simply combinatoric ttbar background.

## Simple cut candidate

Searched 1- and 2-variable cuts, thresholds set from ttbar's own quartiles, on the u>3.5 population **only** (never on the 2 surviving u>4.5 events).

**Recommended: `pT_H1 < 406 GeV`** (single cut, simplest option, evaluated at ttbar's own median):
- Signal efficiency: **98.6%** (26,951/27,339 retained)
- ttbar rejection: **51.9%** (14 of 27 removed, 13 remaining)
- All-background rejection: **24.5%** (83 of the full 110-event all-background u>3.5 population removed)

A 2-cut variant (`pT_H1<406 AND pT_H2<392`) improves ttbar rejection only marginally (55.6% vs 51.9%, i.e. 1 additional event out of 27) at a comparable signal-efficiency cost — **not judged worth the added complexity given only 27 raw ttbar events support the search**; the single-cut candidate is preferred.

**Descriptive-only check at u>4.5** (2 raw ttbar events, cut fixed from u>3.5, not re-optimized): both surviving u>4.5 ttbar events are **not** independently characterized further here — with only 2 events, any pass/fail statement would be a single-event anecdote, not evidence; not reported as a rate.

## Process-by-process table for the frozen cut (v2 Correction 2)

Applying the exact, un-re-optimized threshold `pT_H1 < 406.1153676240428 GeV` to every process individually (raw = number of surviving MC events in the u>3.5 tail sample; weighted = sum of frozen 450 fb⁻¹ per-event analysis weights). Produced by `work/analysis_correction2_ttbar_table.py`.

| process | n_raw before | n_raw after | weighted before (450 fb⁻¹) | weighted after (450 fb⁻¹) | raw survival | raw rejection |
|---|---:|---:|---:|---:|---:|---:|
| signal | 27,339 | 26,951 | 12.939 | 12.755 | 98.58% | 1.42% |
| QCD | 68 | 60 | 497.168 | 438.677 | 88.24% | 11.76% |
| ttbar (inference_ttbar_1+2) | 27 | 13 | 62.678 | 30.178 | 48.15% | 51.85% |
| SingleTop | 2 | 0 | 9.769 | 0.000 | 0% | 100% |
| TTbarW | 2 | 1 | 0.671 | 0.335 | 50% | 50% |
| TTbarZ | 1 | 1 | 0.387 | 0.387 | 100% | 0% |
| TW | 1 | 0 | 6.227 | 0.000 | 0% | 100% |
| ZJetsToQQ | 8 | 7 | 11.282 | 9.872 | 87.5% | 12.5% |
| ttH | 1 | 1 | 0.761 | 0.761 | 100% | 0% |
| **TOTAL_BACKGROUND** (QCD+ttbar+other, n=110) | **110** | **83** | **588.942** | **480.210** | **75.45%** | **24.55%** |

**The raw-count all-background rejection (24.5%) and the weighted all-background rejection (18.5% = 1 − 480.210/588.942) differ**, because QCD dominates the weighted sum (497/589 ≈ 84% of total weighted background) but is rejected less efficiently (11.8%) than ttbar (51.9%) by this ttbar-tuned cut. The v1/v2 headline "24.5%" figure is the raw-count version; it should not be read as the weighted physical background reduction.

Several `other_background` sub-processes have raw support of 1–2 events (TTbarZ, TW, TTbarW, ttH, SingleTop) — their individual survival/rejection numbers are single- or double-event statements, not rates with statistical meaning.

### Exploratory downstream significance (Z_A, B95)

Using the project's already-established, verbatim-reused formulas (`asimov_z`, `b95_upper_mean` — Poisson↔Gamma duality, stdlib-only bisection — from `track_b_harvey_score_tail_diagnostics_20260829_v3/work/common.py`, not re-derived here), applied to the weighted yields **after** the frozen exploratory cut:

- S (signal, weighted, 450 fb⁻¹, after cut) = 12.755
- B (all background, weighted, 450 fb⁻¹, after cut, nominal) = 480.210
- **Z_A (nominal) = 0.580**
- B (95%-upper-mean, summing each background process's own Poisson-Gamma 95% upper mean given its own small raw survivor count and frozen per-event weight) = 617.44
- **Z_A (using B95-upper) = 0.512**

**This is EXPLORATORY, not a validated result**: the cut was chosen by looking at the same finite ttbar sample it is evaluated on (no independent/held-out test set exists at this population size), so both Z_A values above should be read as an order-of-magnitude illustration of what this single candidate cut would do to significance, not a claim that this specific cut is optimal or that this Z_A generalizes.

## Explicit caveat

All numbers above rest on 27 raw MC events (ttbar) — a real but small sample. The 2-cut search space was deliberately restricted to combinations of the top-4 single features to avoid multiple-comparison overfitting on this small population; even so, the reported ttbar-rejection numbers should be read as indicative, not precise, until validated on additional MC statistics.
