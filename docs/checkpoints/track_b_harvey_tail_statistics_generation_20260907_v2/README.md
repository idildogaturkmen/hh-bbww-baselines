# Harvey Track C: physical-background tail statistics, MC scaling, next step

**Scope.** Read-only reconciliation and post-processing of already-frozen
Track-B physical-background artifacts. No SPA-Net training, no ParT, no
EAF/GPU, no `holdout_B`/Stage-C access, no new MC generation, no Condor
launch, no package installation. Every existing package is unmodified,
including this repository's own `track_b_harvey_tail_statistics_
generation_20260906_v1` from the previous day, which this package
supersedes in scope (physical statistics reconciled at Harvey's literal
thresholds plus the existing governing WP) but does not delete or edit.

This is the Track C companion to the separate FP32/logit precision audit
(Track A): Track A explains *why* the score comb appears near 1.0 (a
float32 softmax representation effect); this package answers a different,
physical question — how much real Monte Carlo statistical support actually
underlies the QCD/ttbar/minor-background tail, and what it would cost to
improve it. **The two tracks use different event populations** (Track A:
the 400k development cohort; this package: the full physical-background
production, `inference_qcd_1`+`inference_qcd_2` and the 16-process non-QCD
inference population) and their numbers must not be mixed.

## Files

- `HARVEY_PHYSICAL_TAIL_STATS_FINAL.md` — the full technical answer
  (Sections 0-7: threshold semantics, reconciled tail counts, finite-MC
  uncertainty, QCD scaling, resource estimate, brute-force-vs-targeted
  comparison, stability assessment, cross-checks).
- `HARVEY_PHYSICAL_TAIL_STATS_EMAIL_PARAGRAPH.md` — the concise, two-
  paragraph Harvey-facing summary.
- `figures/neff_vs_qcd_multiplier.png`, `figures/relunc_vs_qcd_multiplier.png`
  — 2x/4x/10x QCD-statistics-scaling projections for all four populations
  studied (the existing eps_S=4.0% WP, u>3.5, score>=0.9997, u>4.5/0.99997).
- `work/cross_check_recompute.py` / `work/cross_check_output.txt` — the
  independent, from-source recomputation of every headline number, with
  hard assertions against already-published independent values.
- `receipt.json`, `SHA256SUMS`, `STATUS.md` — provenance and package status.

## Headline result

At Harvey's literal `score>=0.9997` cut: 100 raw background events, B=532.4
at 450 fb-1, N_eff=81.0, 11.1% relative MC uncertainty (84% of B from QCD).
At `score>=0.99997` (identical population to `u>4.5`): 15 raw events,
B=97.3, N_eff=14.0, 26.7% relative uncertainty. Doubling QCD MC statistics
at fixed physical normalization would reduce these uncertainties by
~27%; 4x by ~46%; 10x by ~60%. A literal 4x QCD regeneration costs
~22.6 million CPU-slot-hours using the only real measured Pythia8+Delphes
rate in this project — a targeted, `pTHat`-biased extension is very likely
more efficient, but is not yet a validated method (no compensating-weight
closure test has been run). Full detail and every caveat in
`HARVEY_PHYSICAL_TAIL_STATS_FINAL.md`.

## Final status

```
PHYSICAL_TAIL_RECONCILED = YES
CURRENT_4PCT_BKG_RAW = 23            (eps_S=4.0% WP: 17 QCD + 4 ttbar + 1 ZJetsToQQ + 1 SingleTop)
CURRENT_4PCT_BKG_NEFF = 20.4618
CURRENT_4PCT_REL_MC_UNC = 22.1069%   (naive 1/sqrt(23)=20.85% understates this by ~6% relative)
QCD_2X_REL_MC_UNC = 16.0148%         (eps_S=4.0% WP; u>3.5: 7.72%; score>=0.9997: 8.12%; u>4.5/0.99997: 19.38%)
QCD_4X_REL_MC_UNC = 11.8470%         (eps_S=4.0% WP; u>3.5: 5.78%; score>=0.9997: 6.10%; u>4.5/0.99997: 14.36%)
QCD_10X_REL_MC_UNC = 8.4070%         (eps_S=4.0% WP; u>3.5: 4.20%; score>=0.9997: 4.46%; u>4.5/0.99997: 10.22%)
QCD_4X_ADDITIONAL_CPU_SLOT_HOURS = 22,575,553   (allocated-wall basis; 20,811,864 on a CPU-busy basis; cross-checks the project's own published ~22.5M figure)
BRUTE_FORCE_RECOMMENDED = NO         (scientifically clean but costs tens of millions of CPU-slot-hours for a sub-2x precision gain at 4x statistics)
TARGETED_EXTENSION_RECOMMENDED = YES  (conditional: only after the ~20,000-event bias2Selection mechanism-validation canary passes its closure tests -- not yet run)
NEW_MC_GENERATION_LAUNCHED = NO
NEW_TRAINING_LAUNCHED = NO
READY_FOR_HARVEY = YES
```
