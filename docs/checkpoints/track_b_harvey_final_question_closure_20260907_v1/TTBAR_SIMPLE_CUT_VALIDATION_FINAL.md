# Validating the exploratory ttbar candidate cut on two independent lanes

The Track B exploratory candidate (`pT_H1 < 406.1153676240428 GeV`, 14/27
ttbar rejected, 98.6% signal retained) was chosen and evaluated on the same
pooled 27-event ttbar sample. This task validates it using the two distinct
frozen ttbar production lanes recorded independently in
`HARVEY_MASTER_EVENT_TABLE.parquet` (`process` = `inference_ttbar_1` /
`inference_ttbar_2`), confirmed disjoint by `event_uid` (n=11 and n=16
raw events respectively, 27 total, matching the original pooled count
exactly).

Read-only. No new inference. No cut re-optimization beyond the single
predeclared leave-one-lane-out rule below.

## A. Frozen cut, applied independently to each lane

| lane | n_ttbar before | n_ttbar after | ttbar rejection | 68% CI | signal efficiency (frozen 27,339-event population) |
|---|---:|---:|---:|---:|---:|
| `inference_ttbar_1` | 11 | 7 | 36.4% | [23.6%, 51.4%] | 98.58% |
| `inference_ttbar_2` | 16 | 6 | 62.5% | [50.0%, 73.5%] | 98.58% |
| pooled (both lanes) | 27 | 13 | 51.9% | [42.3%, 61.2%] | 98.58% |

The two lanes' 68% confidence intervals overlap (36–51% vs 50–74%), so this
alone does not prove incompatibility, but the point estimates differ by
nearly a factor of 1.7 — a real dispersion the small pooled sample masked.

## B. Leave-one-lane-out: derive on one lane, evaluate on the other

Predeclared rule: threshold = median of the derivation lane's own `pT_H1`
distribution (the same rule used to pick the original 406.115 GeV value on
the pooled sample), never re-tuned after seeing the evaluation lane.

| derived on | derived threshold | evaluated on | n before | n after | rejection | signal efficiency |
|---|---:|---|---:|---:|---:|---:|
| `inference_ttbar_1` | 337.0 GeV | `inference_ttbar_2` | 16 | 5 | 68.75% | 95.88% |
| `inference_ttbar_2` | 549.5 GeV | `inference_ttbar_1` | 11 | 7 | 36.4% | 99.78% |

**The learned thresholds are unstable**: 337.0 GeV vs. 549.5 GeV, a ratio of
**1.63×** — far apart, and each carries a very different signal-efficiency
cost (95.9% vs. 99.8%) despite targeting the same nominal ttbar-rejection
goal. This is disclosed plainly, as instructed, rather than downplayed.

## C. Is the qualitative "hard/boosted ttbar" picture reproduced in both lanes?

| feature | signal median | lane1 median | lane1/signal | lane2 median | lane2/signal | both lanes >1.5x boosted? |
|---|---:|---:|---:|---:|---:|---|
| pT_H1 | 173.0 GeV | 337.0 GeV | 1.95x | 549.5 GeV | 3.18x | **Yes** |
| jet1_pt | 165.2 GeV | 254.4 GeV | 1.54x | 549.9 GeV | 3.33x | **Yes** |
| HT | 456.5 GeV | 575.7 GeV | 1.26x | 1138.6 GeV | 2.49x | No (lane1 only 1.26x) |
| pT_H2 | 146.8 GeV | 179.0 GeV | 1.22x | 433.0 GeV | 2.95x | No (lane1 only 1.22x) |
| jet2_pt | 114.0 GeV | 130.5 GeV | 1.14x | 342.7 GeV | 3.01x | No (lane1 only 1.14x) |

2 of 5 features (`pT_H1`, `jet1_pt`) show a consistent ≥1.5× boost in both
lanes; the other 3 show the boost clearly in lane 2 but only mildly in lane
1. The *direction* (ttbar survivors are harder than signal) holds in every
feature in both lanes — but the *magnitude* is lane-dependent, consistent
with the small per-lane sample sizes (11 and 16 raw events).

## D. Verdict

**`TTBAR_SUPPRESSION = NOT_REPRODUCED`**

Rationale: the leave-one-lane-out derived thresholds differ by 1.63× (this
task's predeclared instability criterion is a ratio <1.15 for
"reproducible"), and only 2 of 5 kinematic features meet the strict ≥1.5×
both-lanes-boosted bar. The underlying qualitative direction (surviving
ttbar is boosted relative to signal) is real and visible in both lanes, but
the specific numeric cut (`pT_H1<406.115 GeV`) and its exact rejection power
should **not** be treated as validated or generalizable from 27 raw events
split 11/16 across two lanes. This is a downgrade in confidence relative to
the original v2 Track B language ("EXPLORATORY, not validated"), not a
reversal of the qualitative finding.

**Do not call this cut final or validated for publication.** It remains a
legitimate, disclosed candidate observation; a larger, held-out ttbar
sample (not existing anywhere in this project) would be needed before it
could be treated as a validated selection.

Figure: `figures/ttbar_lane_validation_pTH1.png` (pT_H1 distributions,
signal vs. both lanes, frozen cut overlaid). Full numeric detail:
`TTBAR_LANE_VALIDATION.csv`, `work/ttbar_leave_one_lane_out.csv`,
`work/ttbar_qualitative_feature_check.csv`.
