# Future representation learning (external transfer, then JP-JEPA)

## Question

Is there a learned-representation approach that captures genuinely useful
jet/event information beyond SPA-Net's native features, without the
optimization/selection pathology found in the ParT active20 study (see
[`../part_representation_study/`](../part_representation_study/README.md))? This
project has explored this question twice: first via transfer from an external
released model, and currently via preparation for a joint-embedding predictive
architecture (JP-JEPA).

## Method

- **External-model transfer (earlier).** Reproduce and validate an external,
  publicly released transfer-learning model on real inference workloads (QCD,
  ttbar, and diboson/Z+4b events) as a faithfulness check, before attempting to
  bridge its representations into this project's own evaluation framework.
- **JP-JEPA compatibility preparation (current).** A preparation-and-diagnostics
  pass for using a frozen JP-JEPA "Mini" jet embedding as a SPA-Net input feature,
  by direct analogy with the (now-diagnosed-harmful) ParT contract: pinning the
  upstream source, reproducing its exact preprocessing pipeline, building a
  feature-compatibility matrix against this project's real jet ntuples, and running
  a bounded CPU-vs-GPU numerical audit of the frozen embedding — all ahead of any
  full-scale production run.

## Dataset

- External-model transfer: real QCD/ttbar/ZH/ZZ inference workloads evaluated
  against the released model.
- JP-JEPA: a 1,050-jet compatibility canary and a separate 12,089-jet CPU/GPU
  numerics-diagnostic subset — both far below the 2M-event scale used elsewhere in
  this project.

## Main result

- The external released model was reproduced and validated on real inference
  workloads, establishing a faithful baseline ahead of any transfer attempt into
  this project's framework.
- JP-JEPA compatibility preparation is **complete, with one documented caveat**:
  every required JP-JEPA input was verified derivable from this project's real jet
  ntuples, and the frozen jet-embedding extraction point was verified line-by-line
  against the upstream source. However, a real (if bounded) CPU-vs-GPU numerical
  discrepancy was found in the frozen embedding — max abs diff 0.060, mean abs diff
  0.00157, minimum cosine similarity 0.999989, on 12,089 real jets — that exceeds
  the tolerance derived from ParT's own CPU/GPU behavior and is **not yet fully
  mechanistically explained**. No JP-JEPA-augmented SPA-Net has been trained.

## Key figures/tables

None published yet; the JP-JEPA numerics audit is machine-readable only (see
Provenance).

## Code/config pointers

None in `scripts/` yet. All current code for both efforts lives inside the frozen
preparation checkpoints listed under Provenance.

## Frozen artifact

None yet at the `artifacts/` level for either effort; the compatibility-preparation
checkpoint itself (see Provenance) is the frozen unit so far.

## Provenance

- External-model transfer: `docs/checkpoints/track_b_phase2_released_model_validation_20260801_v1/`,
  `track_b_phase2_qcd_real_inference_20260804_v1/`,
  `track_b_phase2_ttbar_real_inference_20260804_v1/`,
  `track_b_phase2_zh_real_inference_20260801_v1/`,
  `track_b_phase2_zz_real_inference_20260804_v1/`,
  `track_b_phase2_zh_zz_released_model_validation_20260805_v1/`;
  `track_b_phase3_bounded_multifile_sophon_adapter_canary_20260806_v1/`,
  `track_b_phase3_evaluation_representation_bridge_contract_20260806_v1/` — the
  representation-bridge groundwork toward this project's own evaluation framework.
- JP-JEPA: `docs/checkpoints/track_g_jpjepa_spanet_compatibility_prep_20260908_v1/`
  — see in particular `STATUS.md` (overall gate:
  "compatibility-preparation complete, with documented numerical tolerance — not a
  production authorization"), `CPU_GPU_NUMERICS_AUDIT.md`,
  `FEATURE_COMPATIBILITY_MATRIX.md`, and `UPSTREAM_PROVENANCE.md`.
- `docs/plans/track_b_track_a_integration_plan_20260801_v1.md` — the original plan
  motivating the external-model transfer effort.

(Internal codenames attached to these checkpoint directory names are not used in
this study's narrative; they are retained above only as exact,
reproducibility-relevant paths.)
