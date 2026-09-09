# Harvey follow-up: SPA-Net compressed-tail statistics, MC-doubling cost, and preselection variables

**Scope and safety.** Read-only analysis of already-frozen artifacts across
this repository and the broader Track-B analysis workspace. No Condor/EAF
job launched, no new SPA-Net/BDT scoring run, no ParT/EAF interaction, no
frozen physics result (canonical four-model normalization, fine-scan tables,
75-epoch adjudication, etc.) modified in any way.

## u-transform and interpretation

`u = -log10(1 - score)`, `score` = SPA-Net 10M class-1 softmax output
(checkpoint `fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5`,
per `track_b_harvey_spanet_compressed_tail_20260903_v2/HARVEY_COMPRESSED_TAIL_REPORT.md`).
**Harvey's "u > 3/5" is interpreted as u > 3.5**, because the previously-used
cut `score >= 0.9997` maps to `u = 3.5229`. A second region, **u > 4.5**
(near the previously-used `score >= 0.99997`, `u = 4.5229`), is studied in
parallel per the task instructions.

## Files in this package

- `HARVEY_TAIL_STATS_CURRENT.csv` — Task 1: exact current raw count,
  weighted yield, sum(w^2), N_eff, relative MC uncertainty for QCD / ttbar /
  other (14 minor processes, itemized) / all-background, at u>3.5 and u>4.5.
- `HARVEY_MC_DOUBLING_ESTIMATE.md` — Task 2: exact current QCD
  generation/exposure statistics, additional generated-event requirement to
  double each tail under ordinary (fixed-acceptance) generation, and the
  2x/4x/10x QCD-statistics precision projection (physical yield held fixed).
- `HARVEY_GENERATION_RESOURCE_ESTIMATE.md` — Task 3: CPU-hours, wall time at
  representative Condor parallelism, storage, scoring cost, and bottleneck
  stage, using only actually-measured throughput receipts.
- `HARVEY_PRESELECTION_FEATURE_RANKING.csv` — Task 4: univariate
  AUC/Spearman ranking of candidate preselection variables against a
  SPA-Net-2M tail proxy, tagged by generation-level availability.
- `HARVEY_PRESELECTION_CANDIDATES.md` — Task 4 (cut scans, pairwise
  combinations, compact cross-validated models) and Task 5 (generation-level
  availability matrix, A/B/C).
- `HARVEY_STABILITY_PLAN.md` — Task 6: the cumulative-block stability study
  design, convergence-plot templates, and a quantitative stability
  criterion, built on top of the existing (unmodified) validation/closure
  contract and canary plan.
- `figures/` — `tail_count_vs_u.png`, `neff_vs_qcd_multiplier.png`,
  `relunc_vs_qcd_multiplier.png`, `feature_auc_ranking.png`.
- `RECEIPT.json` — sources, methodology notes, and an explicit
  not-found/not-resolved list.
- `SHA256SUMS` — checksums of every file in this package.

## Key finding this package adds on top of existing frozen material

Prior packages (`track_b_harvey_spanet_compressed_tail_20260903_v2`,
`track_b_harvey_score_tail_diagnostics_20260829_v5_background_stats_audit`,
`track_b_harvey_followup_finescan_likelihood_20260829_v1`) already
established exact tail statistics at the nearby fixed cuts
`score>=0.9997`/`score>=0.99997`. This package computes the **exact** counts
at Harvey's literal `u>3.5`/`u>4.5` thresholds — which do not fall exactly
on any previously-tabulated grid point — by directly re-thresholding the
real, already-frozen per-event score arrays (`FULL_QCD_SURVIVOR_SIDECAR.h5`
for QCD; the per-event `survivors` lists inside the frozen non-QCD
`job_summary_*.json` files for ttbar and the 14 minor backgrounds), rather
than interpolating. No event was rescored.

## Final status

```
U35_INTERPRETATION = u>3.5  (score > 0.9996837722339832; Harvey's "u>3/5" read as u>3.5 per score>=0.9997 -> u=3.5229 correspondence)
CURRENT_QCD_RAW_U35 = 68            (B_450=497.17, N_eff=68.00, rel.MC.unc.=12.13%; all-background: n=110, B=588.94, N_eff=89.31, rel.unc.=10.58%)
CURRENT_QCD_RAW_U45 = 12            (B_450=87.74, N_eff=12.00, rel.MC.unc.=28.87%; all-background: n=15, B=97.26, N_eff=13.99, rel.unc.=26.73% -- matches frozen eps_S=3.5% fine-scan point)
ADDITIONAL_QCD_EVENTS_TO_DOUBLE_U35 = +87,623,306 selected / +278,360,000,000 generated (a full second production, same size as the current sample -- identical requirement for U45; ordinary fixed-acceptance generation cannot differentially target one tail)
ADDITIONAL_QCD_EVENTS_TO_DOUBLE_U45 = +87,623,306 selected / +278,360,000,000 generated (same as above)
ESTIMATED_WALLTIME_DOUBLE_U35 = ~6.9-7.5 million CPU-slot-hours total (measured Pythia8+Delphes pilot rate); ~3-5 days at a hypothetical 64,000-100,000-concurrent-slot allocation, multi-year at the largest concurrency (150 slots) actually used for a QCD campaign in this project to date
ESTIMATED_WALLTIME_DOUBLE_U45 = identical to U35 (same additional generation required)
BEST_PRESELECTION_FOR_U35 = analysis-level: min-probB(leading 4)>0.9 AND reconstructed HT>500 GeV (417x tail enrichment, 99.98% bulk-QCD rejection); generation-level (the only lever that reduces generation cost itself): native Pythia PhaseSpace:bias2Selection biased on pTHat, Ref~500 GeV (Design B, already specified, not yet run)
BEST_PRESELECTION_FOR_U45 = same candidate variables; population too small (12 QCD events) for any independently cross-validated preselection at this exact extremity -- treat as directionally consistent with U35, not separately validated
GENERATOR_LEVEL_PROXY_AVAILABLE = YES  (pTHat, native to the governing direct-Pythia HardQCD chain and structurally compatible with PhaseSpace:bias2Selection; reconstructed HT is the best available post-Delphes proxy that motivates where to center the bias)
NEW_JOB_LAUNCHED = NO
```
