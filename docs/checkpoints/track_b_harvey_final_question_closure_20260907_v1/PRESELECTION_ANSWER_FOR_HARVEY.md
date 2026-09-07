# Preselection: three different questions, disentangled

Harvey's "which preselection variables isolate the tail" question conflates
three genuinely different things. This note separates them and answers each
using only already-existing, frozen analysis
(`track_b_harvey_tail_statistics_generation_20260906_v1/HARVEY_PRESELECTION_CANDIDATES.md`,
reorganized here into the requested table, not recomputed). No new MC, no new
scoring, no frozen result modified.

## The three ideas

1. **Analysis-level / post-Delphes variables that distinguish the tail** —
   computed after full detector simulation and jet-flavor tagging; can only
   reduce *downstream analysis/storage* cost (e.g., which events get scored
   by the expensive SPA-Net forward pass), never generation cost.
2. **Variables available before SPA-Net scoring** — a subset of (1): every
   reconstructed quantity except the score itself.
3. **Generator-level variables capable of reducing generation cost** — only
   quantities knowable at or before the Pythia8 hard-process step, before any
   detector simulation runs at all.

## Variable table

| variable | availability_stage | evidence_for_tail_enrichment | usable_for_analysis_presel? | usable_to_reduce_Delphes_cost? | usable_to_reduce_generation_cost? | bias/reweighting_needed? | evidence_level |
|---|---|---|---|---|---|---|---|
| `min-probB(leading 4)` (b-tag mistag rate) | post-Delphes + tagging (stage 1) | AUC 0.875 vs top-0.1% QCD-score proxy tail (n_pos=175); 219–395x tail enrichment for simple thresholds | **Yes — the strongest single lever found** | No (tagging already ran) | No | None (already-observed quantity, no resampling) | Real, direct measurement (proxy-tail cross-check against the real >=0.9997 SPA10M tail agrees qualitatively — Part C of `HARVEY_COMPRESSED_TAIL_REPORT.md`) |
| `probB_leading4_mean` | post-Delphes + tagging (stage 1) | AUC 0.988 (top-0.1% proxy) | Yes | No | No | None | Real, direct measurement |
| reconstructed HT | post-Delphes, pre-pairing (stage 1/2) | AUC 0.63–0.66 alone; 1.35–2.8x enrichment; conditional tail efficiency rises 4–14x from 500–700 GeV to ≥1000 GeV (`STRATUM_TAIL_TABLE.csv`), conditional on already passing the 7.5% WP | Yes (moderate, best combined with b-tag cut: 417x) | No | **No by itself — but motivates where to bias `pTHat`** | None for analysis use; a *compensating weight* would be needed only if HT itself were used to bias generation (it cannot be — see below) | Real, direct measurement |
| `n_selected_jets` | post-Delphes (stage 1) | AUC 0.526 (weakest of all variables tested); conditional gradient 2.2%→15.9% (4→6 jets) in the tail-survivor stratum table | Weak alone | No | No directly, but a generator-level final-state-multiplicity analogue exists (not yet built into a bias hook) | N/A | Real, direct measurement |
| leading/2nd/3rd/4th/5th-jet pT | post-Delphes (stage 1) | AUC 0.58–0.70 individually; noisier proxies for `pTHat` than HT | Weak–moderate | No | No directly (correlated with, but not identical to, `pTHat`) | N/A | Real, direct measurement |
| `mHH`, `R_HH`, `pT(H1)/pT(H2)`, `ΔR(bb)`, mass asymmetry | post-HH-pairing (stage 2/3, needs jet-pairing/assignment) | AUC 0.63–0.83 for the best of these (HT/mHH) | Yes (moderate) | No | No | N/A | Real, direct measurement |
| SPA-Net score / `u` | the quantity being predicted | — | Excluded by definition — not a preselection candidate | — | — | — | — |
| **`pTHat`** (true generator hard-process scale) | **generator level, before Delphes** | Not stored per-event in the frozen ntuples; inferred indirectly via the reconstructed-HT conditional-efficiency gradient above, since `pTHat` is the generator quantity that most directly drives reconstructed HT | N/A (not an analysis-level cut) | N/A | **Yes — the only variable in this table that is genuinely available early enough to reduce generation cost itself** | **Yes, mandatory** — native Pythia8 `PhaseSpace:bias2Selection` on `pTHat` requires an exact compensating per-event weight (`Info::weight()`), carried through every downstream `N_eff`/significance calculation | **PROMISING BUT CANARY NOT YET RUN** — status unchanged from `track_b_qcd_tail_generation_strategy_20260902_v1/CANARY_PRODUCTION_PLAN.md`; not executed in this task (no Condor/MC launch permitted) |
| inclusive generator-level HT (parton-level `iht`, MG5 slicing) | generator level | Used successfully in the *separate* MG5 `pp→bbbb` chain only; demonstrably shifts reconstructed R_HH/⟨mbb⟩ away from the signal region when used alone (`qcd_iht_closure_metrics_2026_07_08.csv`: R_HH<30 under-populated by −16.8%) | N/A | N/A | Not directly transferable — the governing QCD sample is confirmed direct-Pythia HardQCD, not this MG5 chain | Yes, and previously shown to be non-trivial to get right (the −16.8% shape distortion) | Real, direct measurement, but on a different production chain |
| Reconstructed HT / jet multiplicity as a *generation-time* bias | — | — | — | — | **Explicitly excluded, not physically implementable** — the hard-process event has no reconstructed HT yet when Pythia's bias hook fires | N/A | N/A |

## Do not conflate (explicit, per instruction)

- `n_btag_loose` / `probB` are **stage-1 detector/tagging quantities** — they
  are not generator-level and cannot be used to bias or reduce generation
  cost, only downstream analysis/storage cost.
- Reconstructed HT is **not** `pTHat`. HT is a stage-1/2 reconstructed
  quantity; `pTHat` is the generator-level hard-process scale. HT is used
  here only as *motivation* for where to center a `pTHat` bias, never as a
  cut applied before generation.
- An **observed correlation** (e.g., the b-tag/HT tail-enrichment numbers
  above) is not the same as a **validated importance-sampling method**. No
  targeted/biased QCD generation has been run in this project; the only
  generator-level lever identified (`pTHat` bias) remains a *design*, gated
  on a small mechanism-validation canary that has not been launched.

## Bottom line

The strongest *statistical* preselection (b-tag mistag rate, AUC up to 0.99)
cannot reduce generation cost — it only exists after the full detector and
tagging chain has already run. The only lever identified anywhere in this
project that could reduce *generation* cost itself is a native Pythia8
`bias2Selection` bias on `pTHat`, which is structurally compatible with the
confirmed direct-Pythia governing QCD chain, requires no new code, and is
**PROMISING BUT CANARY NOT YET RUN** — unchanged from the existing plan, not
re-litigated or re-decided in this task.
