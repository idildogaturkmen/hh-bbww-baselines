# Study 05 — SPA-Net reconstruction and classification

## Question

Does a symmetry-preserving, permutation-aware jet-assignment network
(SPA-Net) both **classify** HH→4b events and **correctly pair** jets into
two Higgs candidates better than the classical-ML ladder
([study 04](../04_hh4b_classical_ml/README.md)) plateaued at? And separately
— does an **externally released** representation model (Sophon /
jetfree-hh4b) offer a useful comparison point?

## Dataset / samples

The matched, fixed 2,000,000-event training / 400,000-event validation HH→4b
cohort used throughout the SPA-Net line of work (native 7-feature input:
pT, η, φ, mass, probB, probC, probL per jet, up to 10 jet slots). This is a
**development/validation cohort, not a final blind test** — stated
explicitly in every governing source document.

## Method

1. **First qcdplus SPA-Net (2026-07-13/14)** — a single classification head
   + pair-assignment head on a 42,322-event leading-8-jet dataset, compared
   directly against the classical-ML ladder.
2. **Released-model side-study ("Track B" Phase 2, 2026-08-01 → 08-06)** —
   reproduced the externally-released PKU-HEP Sophon/jetfree-hh4b
   variable-mass, 138-class ONNX ensemble bit-for-bit; validated its ZH/ZZ
   mass localization against physical resonances via DCB fits; ran real
   inference on QCD/ttbar control samples. A Phase 3 adapter-integration
   attempt was explicitly **paused** on 2026-08-06 to refocus on the
   project's own baseline (see
   [study 06](../06_scaling_and_tail_reliability/README.md) for its
   continuation as a tail-reliability question).
3. **Four-model development snapshot (2026-08-21)** — BDT-K, BDT-KF, SPA-Net
   2M, SPA-Net 10M compared on the fixed 400k cohort.
4. **Four-model physical-normalization archive (2026-08-24, paper-facing)**
   — translated all four models' operating points into physical SM
   signal-efficiency/QCD-yield numbers at 450 fb⁻¹ across 17 working points,
   using a finite-QCD-MC-aware reporting policy (one-sided 95% Poisson
   upper bound below 20 observed QCD events), cross-checked by three
   independent implementations to ~1e-8 agreement.
5. **Governing native SPA-Net result (2026-09-11, refined 2026-09-14)** — the
   frozen 2M/10M comparison this project cites as authoritative; see
   [study 06](../06_scaling_and_tail_reliability/README.md) for the scaling
   question itself.

## Main findings

- The first SPA-Net attempt (S/√B=0.163) sat between the DNN (0.154) and BDT
  (0.199) baselines but with weak top-background rejection (weighted
  AUC=0.460) — directly motivating a two-head (qcd_score + top_score +
  assignment) diagnostic redesign.
- Continuous flavor information (moving from a hard b-tag to a continuous
  K→KF score) was the **dominant** gain in the four-model development
  snapshot; SPA-Net gave a further, statistically robust gain over KF in the
  QCD-tail-supported region.
- The Sophon/jetfree-hh4b released model reproduced correctly and localized
  physical resonance masses as expected, but the adapter-integration
  extension was paused rather than pursued to completion — an early
  representation-learning direction that did not continue past a
  feasibility canary. A later draft manuscript section on this line of work
  exists **unmerged** on `track-b-sophon-transfer` — see
  [`docs/history/BRANCH_GUIDE.md`](../../history/BRANCH_GUIDE.md).
- **Governing native SPA-Net (2M)**: all-background AUC 0.969281, exact-event
  HH reconstruction 0.866588 — the baseline every later scaling
  ([study 06](../06_scaling_and_tail_reliability/README.md)) and pretrained-
  representation ([study 07](../07_pretrained_jet_representations/README.md))
  result is compared against.

## Figures

- [`figures/four_model_za_b95_significance_vs_sm_efficiency.png`](figures/four_model_za_b95_significance_vs_sm_efficiency.png) — the headline four-model (BDT-K/KF, SPA-Net 2M/10M) physically-normalized significance comparison.

Also see [study 07](../07_pretrained_jet_representations/README.md)'s
figures for the governing SPA-Net ROC/reconstruction curves used as the
native baseline throughout.

## Reproducible code

- `docs/paper/jhep_hh4b_ml/single_head_spanet/` — the single-head SPA-Net
  paper-asset package.
- `docs/checkpoints/track_b_phase2_*` — Sophon released-model validation
  scripts/contracts (11 directories).
- `docs/paper/jhep_hh4b_ml/track_b_physical_normalization/` — the four-model
  physical-normalization archive's frozen tables/figures.
- `artifacts/hh4b_spanet_part_20260911/` — the frozen native-SPA-Net
  evaluation bundle (see [study 07](../07_pretrained_jet_representations/README.md)
  for its ParT-comparison half).

## Relationship to the final HH→4b study

This study's 2M governing checkpoint **is** the final HH→4b reconstruction
baseline — every later result (10M scaling, tail-reliability checks, ParT
active20, ZERO20) is measured relative to it on the identical matched
cohort. The Sophon/jetfree-hh4b released-model side-study is a parallel,
not-pursued-to-completion representation-learning thread, kept for
provenance and because its Phase 3 adapter work is a direct conceptual
precursor to the later ParT integration.

## Provenance

`docs/checkpoints/track_b_phase2_*` (11 dirs, 2026-08-01 → 08-06),
`docs/checkpoints/track_b_phase3_*` (6 dirs, paused 2026-08-06),
`docs/track_b/development_snapshot_20260821/` (2026-08-21),
`docs/paper/jhep_hh4b_ml/track_b_physical_normalization/` (2026-08-24),
`outputs/plots/spanet_qcdplus_*`, `outputs/summaries/hh4b_spanet_embedding_extension_roadmap_2026_07_14.md`,
`outputs/summaries/hhh_paper_spanet_strategy_for_hh4b_2026_07_14.md`.
`docs/plans/track_b_track_a_integration_plan_20260801_v1.md` defines the
Track A (this project's own analysis) / Track B (Sophon study) relationship.
A draft representation-learning manuscript section exists unmerged on
`track-b-sophon-transfer` — see
[`docs/history/BRANCH_GUIDE.md`](../../history/BRANCH_GUIDE.md).
