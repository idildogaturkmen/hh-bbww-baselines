# Whole-project model landscape

Every model this project has trained, in one table, **with its dataset and
evaluation contract stated explicitly** — because the numbers are not on a
common scale and must not be read as a single ranking. In particular, the
bbWW/HH→4b cut-baseline/BDT rows report a **physical-yield-projected
significance** (Run 2, 138 fb⁻¹-equivalent, train-only, no systematics),
while the SPA-Net-era rows (native scaling, ParT active20, ZERO20) report
**raw AUC/reconstruction accuracy on a fixed, matched 2,000,000/400,000-event
development cohort** — a different quantity, on a different (though
overlapping-generation) simulated sample, under a contract with no
Run-2-luminosity projection at all. **No single bar chart correctly
represents both families at once** — this project deliberately does not
produce one.

Only **completed** results appear below. Full128/full144 (a proposed
sensitivity run and a scale-feasibility audit — see
[`docs/studies/07_pretrained_jet_representations/`](../studies/07_pretrained_jet_representations/README.md))
are explicitly excluded because neither is a completed result as of this
reorganization.

| Study | Model | Channel | Dataset / evaluation contract | Training size | Main metric | Result | Directly comparable to | Frozen artifact |
|---|---|---|---|---|---|---|---|---|
| [01](../studies/01_hh_bbww/README.md) | Tabular multiclass DNN | HH→bbWW | bbWW one-lepton selection, test-region Z-significance, no systematics | COLLIDE-1M subset | Z-significance | Z≈0.0055 | other study-01 rows only | `Results Summaries/RESULTS_HHBBWW_BASELINE.md` |
| [01](../studies/01_hh_bbww/README.md) | LBN v0 (obj+aux, no-pair) | HH→bbWW | same as above | COLLIDE-1M subset | Z-significance | Z≈0.00664, N_eff≈60.1 | other study-01 rows only | `docs/methods/ml_baselines/multiclass_dnn_comparison.md` |
| [01](../studies/01_hh_bbww/README.md) | Corrected-recoMET BDT (reference) | HH→bbWW | same as above | COLLIDE-1M subset | Z-significance | Z≈0.00676, N_eff≈58.7 | other study-01 rows only | same |
| [04](../studies/04_hh4b_classical_ml/README.md) | Cut baseline (nested outer-OOF) | HH→4b | train-only, 5-fold pooled OOF, Run-2 138 fb⁻¹-eq. projected significance | own HH4b Delphes production | stat-only Z_A | Z_A=0.02698; **validation blocked, not measured** | other study-04 physical-yield rows | `HH4B_CUT_BASELINE_FINAL_STATUS.md` (root) |
| [04](../studies/04_hh4b_classical_ml/README.md) | Cut baseline (historical R_HH<34) | HH→4b | same contract as above | own HH4b Delphes production | stat-only Z_A | Z_A=0.02648 | other study-04 physical-yield rows | same |
| [04](../studies/04_hh4b_classical_ml/README.md) | BDT `global_v1_mass_aware` (primary) | HH→4b | train-only weighted OOF AUC, same physical-yield convention | own HH4b Delphes production | AUC | 0.7737 | other study-04 physical-yield rows | `docs/checkpoints/hh4b_bdt_model_choice_20260727_v1/` |
| [04](../studies/04_hh4b_classical_ml/README.md) | BDT categorized (secondary, not adopted) | HH→4b | same contract; bootstrap-unstable improvement | own HH4b Delphes production | AUC | 0.7537 / 0.8053 by category | other study-04 physical-yield rows | same |
| [05](../studies/05_spanet_reconstruction/README.md) | Native SPA-Net (2M) | HH→4b | matched 2M-train/400k-val cohort, raw AUC + exact-event reconstruction, no luminosity projection | 2,000,000 events | AUC / exact-event reco. | AUC=0.969281, reco=0.866588 | other matched-cohort rows (05/06/07) | `artifacts/hh4b_spanet_part_20260911/` |
| [06](../studies/06_scaling_and_tail_reliability/README.md) | Native SPA-Net (10M) | HH→4b | same contract, 5× training data | 10,000,000 events | AUC | 0.969272 (saturated vs. 2M) | other matched-cohort rows (05/06/07) | same |
| [07](../studies/07_pretrained_jet_representations/README.md) | SPA-Net + ParT active20 (2M) | HH→4b | same contract as native SPA-Net 2M | 2,000,000 events | AUC / exact-event reco. | AUC=0.944222, reco=0.496015 — **harmed** | other matched-cohort rows (05/06/07) | same as native 2M |
| [07](../studies/07_pretrained_jet_representations/README.md) | SPA-Net + ZERO20 (2M) | HH→4b | same contract as native SPA-Net 2M | 2,000,000 events | AUC / exact-event reco. | AUC=0.969116, reco=0.866201 — **native-like** | other matched-cohort rows (05/06/07) | `artifacts/hh4b/pretrained_jet_representations/zero20_20260914/` |

A separate, external, non-project reference significance point (retained in
the paper-preparation provenance as `EXTERNAL_REFERENCE_NOT_OUR_MODEL`,
never one of this project's own models) exists for cross-checking the
physical-yield background-budget methodology the cut/BDT rows above use; it
is not listed as a project result here and should not be cited as one.

Full statistics, confidence intervals, and citations for every row above are
in [`RESULTS_OVERVIEW.md`](RESULTS_OVERVIEW.md).
