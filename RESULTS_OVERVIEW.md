# Results overview

At-a-glance headline results for every scientific study in this project. Each row
links to the corresponding curated study page in [`docs/studies/`](docs/studies/README.md),
which in turn links to the exact frozen artifact/provenance directory behind the
number. Numbers below are development/validation results unless stated otherwise —
see each study page for caveats.

| # | Study | Headline result |
|---|---|---|
| 1 | [bbWW baseline](docs/studies/bbww_baseline/README.md) (superseded) | H→bb jet-pairing purity 0.558; H→WW-side target quarks matched in only ~2.7% (fully hadronic) to ~22% (semileptonic) of events — motivated the pivot to HH→4b |
| 2 | [HH→4b simulation](docs/studies/hh4b_simulation/README.md) | 144,291,951-event resolved HH→4b training population (69.75M signal / 62.94M QCD / 11.6M ttbar, 1,100 source files), fully provenance-traced |
| 3 | [Classical/deep baselines](docs/studies/classical_deep_baselines/README.md) | BDT-KF (kinematics + flavor tags) beats BDT-K (kinematics only) by roughly an order of magnitude in background-rejection significance |
| 4 | [Physical normalization](docs/studies/physical_normalization/README.md) | Run-2 138 fb⁻¹ normalization contract authorized (441 primary sources, full closure); four-model comparison at 450 fb⁻¹ ranks native SPA-Net 10M > SPA-Net 2M > BDT-KF > BDT-K |
| 5 | [SPA-Net reconstruction/classification](docs/studies/spanet_reconstruction_classification/README.md) | Native SPA-Net (2M): all-background AUC **0.969281**, exact-event HH-reconstruction rate **0.866588** |
| 6 | [Training-scale study](docs/studies/training_scale_study/README.md) | 2M → 10M training events: AUC 0.969281 → 0.969272 — **saturated**, no meaningful gain from 5x more data |
| 7 | [Tail/numerical reliability](docs/studies/tail_numerical_reliability/README.md) | Extreme-tail classifier scores confirmed numerically stable under FP32-vs-higher-precision cross-check on the governing 10M model |
| 8 | [ParT representation study](docs/studies/part_representation_study/README.md) | Frozen ParT active20 augmentation: AUC 0.969281 → **0.944222** (Δ −0.025), reconstruction rate 0.866588 → **0.496015** (Δ −0.371) — a diagnosed **harm** result, not an improvement |
| 9 | [ZERO20 causal ablation](docs/studies/zero20_causal_ablation/README.md) | **Planned, not yet run** — isolates "wider input embedding" from "ParT content" as the cause of row 8's harm |
| 10 | [Future representation learning](docs/studies/future_representation_learning/README.md) | External released model reproduced/validated on real inference; JP-JEPA compatibility preparation complete with one open caveat (bounded, not-yet-fully-explained CPU/GPU numerical discrepancy in the frozen embedding); no JP-JEPA-augmented model trained yet |

For the narrative behind these numbers, see [`PROJECT_STORY.md`](PROJECT_STORY.md).
For the repository layout, see [`README.md`](README.md).
