# Residual Background Diagnostic — Summary

**DEVELOPMENT ONLY — matched 400k training-production cohort; not independent Stage-C; not physically normalized.**

Prototype of Prof. Harvey Newman's proposed residual-background-composition study, on the exact frozen matched-400k cohort (SHA256 `9588d0fe79e32d455467c719eaaa57541fcbead9835b8c32bebb9d68b96a2d30`, verified before any computation). Purpose: establish a repeatable analysis contract to be re-run unchanged on final independent Stage-C data once available — **not itself a physical result**.

n_qcd=174,485, n_ttbar=32,157, n_background_total=206,642. Of these, 392,410/400,000 events have ≥4 real candidate jets (7,590 excluded from four-jet topology observables for having fewer than 4 real jet slots — included normally in composition/kinematic counts).

## Task A — residual composition (raw, not normalized)

Full table: `RESIDUAL_BACKGROUND_COMPOSITION.csv`. At the two PRIMARY working points:

| Model | εS | QCD survive | ttbar survive | raw QCD fraction |
|---|---|---|---|---|
| BDT-K | 0.50 | 16,254 | 2,923 | 0.848 |
| BDT-K | 0.40 | 10,192 | 2,100 | 0.829 |
| BDT-KF | 0.50 | 824 | 93 | 0.899 |
| BDT-KF | 0.40 | 398 | 43 | 0.902 |
| SPA-Net-2M | 0.50 | 665 | 86 | 0.885 |
| SPA-Net-2M | 0.40 | 318 | 42 | 0.883 |
| SPA-Net-10M (**summary only**) | 0.50 | 711 | 83 | 0.895 |
| SPA-Net-10M (**summary only**) | 0.40 | 366 | 39 | 0.904 |

Pre-selection raw QCD fraction (reference): 174,485/206,642 = 0.845.

**These are RAW DEVELOPMENT-COHORT FRACTIONS, NOT PHYSICALLY NORMALIZED BACKGROUND COMPOSITION.** No cross-section, luminosity, or generator weighting was applied anywhere in this phase. The raw QCD fraction stays close to the pre-selection value (~0.83–0.90) across all models and both primary working points — i.e. classifier selection does not strongly reshape the raw QCD:ttbar mixture in this uncalibrated cohort; that is a distinct question from the *physically normalized* composition, which this phase does not address.

**SPA-Net-10M note**: only frozen summary survivor counts from `phase4AI_spanet_10M_scaling_seed0_20260821_v1/classification_evaluation_10M` (checkpoint epoch 47, hash-verified) are used. Event-level topology for SPA-Net-10M is marked **UNAVAILABLE** in this phase and excluded from every Task C/D topology table — a separately governed frozen-checkpoint rescoring pass would be required to add it, and none was performed here.

Figure: `figures/TaskA_composition_vs_epsS.*`

## Task B — feature semantics

Verified directly from `feature_extraction.py` (not memory) and frozen in `FEATURE_SEMANTICS_FREEZE.md`. 82-column layout: pt/eta/phi/mass (10 slots each) → mask (10) → n_selected_jets → HT → probB/C/L (10 slots each). Sort order (descending pT, applied identically to every per-jet field) and padding convention (`mask=0` ⇒ zero-filled) explicitly verified before any observable was built on top.

## Task C — four-jet topology

Full table: `RESIDUAL_BACKGROUND_TOPOLOGY_SUMMARY.csv` (21 populations × ~17 observables each, mean/median/std/min/max/N). Pairing rule: of the 3 leading-4-jet pairings, the one minimizing `R_HH = sqrt((m_bb1-125)² + (m_bb2-125)²)` is used as the fixed descriptive pairing (pair1 := higher-pT dijet, deterministic); `m_HH` (sum of all 4 jets) is reported separately as pairing-independent. All three pairings' `(m_a, m_b, R_HH)` are also retained per-event in `results/topology_full.npz` for pairing-independent summaries.

**Qualitative pattern, all three classifiers, both εS**: relative to pre-selection background, K/KF/SPA-Net-2M survivors show a visibly narrower `R_HH` distribution (more concentrated near the (125,125) point) and a shift toward smaller Δ`R_bb` — i.e. surviving background events look kinematically "more Higgs-pair-like" than the pre-selection population, which is the expected residual-background signature of any signal-enriching selection. See `figures/TaskC_topology_*` and `figures/TaskC_kinematics_*`.

## Task D — model disagreement (KF vs SPA-Net-2M)

Full table: `MODEL_DISAGREEMENT_COUNTS.csv`.

| εS | process | both pass | KF-only | SPA-only | neither |
|---|---|---|---|---|---|
| 0.50 | QCD | 516 | 308 | 149 | 173,512 |
| 0.50 | ttbar | 64 | 29 (LIMITED) | 22 (LIMITED) | 32,042 |
| 0.40 | QCD | 248 | 150 | 70 (LIMITED) | 174,017 |
| 0.40 | ttbar | 27 | 16 (EXTREMELY LIMITED) | 15 (EXTREMELY LIMITED) | 32,099 |

At both primary working points and both processes, KF and SPA-Net-2M **mostly agree** (the "neither" cell dwarfs all others, as expected since both are strong, correlated classifiers), but a non-trivial disagreement population exists — KF passes noticeably more QCD events that SPA-Net-2M rejects than the reverse (308 vs 149 at εS=0.50), while the ttbar disagreement is small and statistically limited in both directions.

**KF-only vs SPA-only topology** (combined QCD+ttbar background, `figures/TaskD_disagreement_epsS*`): compared at the population level, no dramatic topology separation is evident between the two disagreement populations in this diagnostic — both show similarly shifted (relative to pre-selection) `R_HH`/ΔR_bb/flavor distributions. This is **observational only**; no causal claim about *why* the two classifiers disagree is made here.

## Task E — statistical support

Every population's N is printed in every table and every figure legend. Flags applied: N<100 → LIMITED, N<20 → EXTREMELY LIMITED (both explicitly marked in `MODEL_DISAGREEMENT_COUNTS.csv` and topology CSVs). **εS=0.25/0.20/0.10 populations are tail diagnostics only** and are excluded from every Task C/D quantitative topology comparison in this phase (only used in the Task A composition table, where they are explicitly flagged). No shape claim in this report is drawn from an N<100 population; the ttbar disagreement cells (both LIMITED/EXTREMELY LIMITED) are reported as counts only, not interpreted distributionally.

## Outputs

`RESIDUAL_BACKGROUND_COMPOSITION.csv`, `RESIDUAL_BACKGROUND_TOPOLOGY_SUMMARY.csv`, `MODEL_DISAGREEMENT_COUNTS.csv`, `FEATURE_SEMANTICS_FREEZE.md`, `SOURCE_PROVENANCE.tsv`, `SHA256SUMS`, `figures/*.{pdf,svg,png}` (21 figures). Every figure carries the DEVELOPMENT ONLY footer.

## Explicitly did not do

Did not access independent/Stage-C/inference data. Did not perform physical normalization anywhere. Did not start or resume any training. Did not modify Step2j/Step2k/Step2l/Step2m or the existing `paper_exports/track_b_development_snapshot_20260821_v1` frozen paper snapshot (verified untouched). Did not recompute or reoptimize any threshold — all six εS thresholds per classifier were read directly from the already-frozen `step2j/matched_400k/results/K_KF_SPANET_MATCHED_400K_COMPARISON.json`. Did not generate a SPA-Net-10M per-event score.
