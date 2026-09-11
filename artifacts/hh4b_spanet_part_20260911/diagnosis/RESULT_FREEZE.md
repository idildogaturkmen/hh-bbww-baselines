# RESULT_FREEZE — track_i_part_harm_diagnosis_20260911_v1

**Purpose.** Before any diagnostic touches a single byte of the governing Track F
result, this document cryptographically freezes it. `SHA256SUMS` (this directory)
lists every file hashed below, one `sha256  path` line each, produced by
`work/freeze_manifest.json`'s generator (script text reproduced in
`code/freeze_manifest.py`, additive copy). All hashing was read-only
(`open(path, "rb")`, chunked, never `w`/`a` mode) against files under
`track_f_evaluation_readiness_protocol_20260909_v1/`,
`track_f_postproduction_pipeline_20260908_v1/`, and the frozen native-SPA2M
training package under `track_b_phase4_preflight_20260812/...phase4AF.../`.
**Nothing under any of those trees was modified, moved, or renamed.**

Run: 2026-09-11, this session. 61 files hashed, 0 missing, 0 read failures.

## Governing numbers this freeze locks in place

From `comparison_result_seed0.json` / `go_no_go_decision_seed0_only.json`
(sha256 `61e26b03...` / `10665677...` below), re-read directly, not taken on
faith from the task prompt:

| metric | native SPA2M (CONTROL) | SPA2M+ParT active20 (TEST) | paired Δ | 95% CI excludes 0 |
|---|---|---|---|---|
| all-background AUC | 0.969281 | 0.944222 | −0.025059 | yes |
| exact-event reconstruction | 0.866588 | 0.496015 | −0.370573 | yes |
| **decision** | | | | **NO_GO_INVESTIGATE_HARM** |
| `harm_class` | | | | True |
| `harm_reco` | | | | True |
| `underpowered` | | | | False |

## What is frozen, by category

| category | n files | representative sha256 | notes |
|---|---|---|---|
| final comparison JSON | 1 | `61e26b03fe36e298...` | `comparison_result_seed0.json` |
| GO/NO-GO JSON | 1 | `10665677942862b3...` | `go_no_go_decision_seed0_only.json` |
| ParT2M checkpoint | 1 | `a8d1ef0ff754fe93...` | `primary-epoch=43-...0.2859.ckpt`, 5,727,937 B — **matches** the sha256 the TEST exporter's own preflight logged in `export_test.log` line 2 (independently reconfirmed, not just re-quoted) |
| TEST export npz | 1 | `3bc703656ba2073d...` | `spa2m_part_active_seed0_val_eval.npz`, 400,000/400,000 events |
| final training-result JSON (ParT2M) | 1 | `abf8c038f5ab6f31...` | `training_spanet_2M_part_active_seed0_20260910_result.json` |
| tables | 9 | — | all `.tsv` under `results/.../tables/` |
| plots | 4 | — | all `.svg` under `results/.../plots/` |
| overnight/paper summary docs | 3 | — | `OVERNIGHT_FINAL_RESULT_SUMMARY.{json,md}`, `PAPER_RESULTS_WITH_PART2M_SEED0_20260911.md` |
| run/eval/export logs | 4 | — | `run.log`, `evaluate_multi_model.log`, `decide_go_no_go.log`, `export_test.log` |
| **CONTROL / native reference** | 5 | | |
| — native SPA2M checkpoint | 1 | `dc39cf76f0d8e40d...` | **exact match** to `FROZEN_FACTS.json`'s `native_spa2m_control.checkpoint_sha256` — independently re-hashed from disk, not copied from the JSON |
| — native training-result JSON | 1 | `18d54114e24694c6...` | `training_2M_seed0_result.json` |
| — CONTROL export npz | 1 | `bebfb680bf8e76a9...` | matches STATUS.md's quoted hash |
| — native10M export npz | 1 | `dd6d86ab2eb72982...` | |
| — `FROZEN_FACTS.json` | 1 | `a5b5d1d46d717b13...` | single source of truth for paths/hashes this freeze cross-checks against |
| **preprocessing / active27 build chain** | 14 | | |
| — `PREPROCESSING_DECISION.md` | 1 | `379030834b0d3373...` | |
| — `PART_TRAIN_STATS_FROZEN.json` | 1 | `a00da99dc8e5e9d3...` | **matches** `training_spanet_2M_part_active_seed0_20260910_result.json`'s own `build_receipt_used.train_stats_source_sha256` |
| — `JOIN_RECEIPT_{train,val,COMBINED}.json` | 3 | | |
| — `BUILD_RECEIPT_active_{train,val}.json` | 2 | | |
| — `VALIDATION_active_{train,val}.json` | 2 | | |
| — `ACTIVE27_BUILD_VALIDATION_COMBINED.json` | 1 | | |
| — `HANDOFF_UPLOAD_RECEIPT.json` | 1 | | |
| — `part_augmented_active_hh4b{,_val}.yaml` | 2 | `bb9c848e10d2c50f...` (both) | train and val event-YAML copies are **byte-identical** (same hash) — expected, both describe the same 27-feature schema |
| — native `trackb_hh4b.yaml` | 1 | `278885c34c53aee7...` | |
| **architecture / protocol contract** | 9 | | |
| — `SPANET_PART_COMPARISON_CONTRACT_v2.md` | 1 | | |
| — `PREREGISTRATION.md` | 1 | `425f390bf7937ce8...` | |
| — `track_f_evaluation_readiness_protocol_20260909_v1/receipt.json` | 1 | | |
| — prior `SHA256SUMS` (Track F's own, pre-existing) | 1 | `33fe06e193c4531c...` | recorded for cross-reference; this freeze's own `SHA256SUMS` is a separate, additive file |
| — schemas (4) | 4 | | |
| — `export_model_eval_events.py` | 1 | `40a8c09e0b2d3364...` | **exact match** to STATUS.md's quoted hash |
| — `evaluate_multi_model.py`, `decide_go_no_go.py` | 2 | | |
| **native 2M input HDF5 (val, small enough to hash directly)** | 1 | `3c94bf8300d1dc33...` | `production_2M_val.h5`, 135,213,912 B — **exact match** to `FROZEN_FACTS.json` |

## Deliberately NOT re-hashed here (multi-GB inputs, handled by the audit phase instead)

`spa2m_part_active_{train,val}.h5` (2.28 GB / 457 MB), `joined_{train,val}.h5`
(10.8 GB / 2.1 GB), and `production_2M_train.h5` (676 MB) are not re-hashed by
this freeze step — hashing them is folded into the H1/H2/H3 audit below, which
reads them anyway and independently re-derives (not just re-hashes) their
content. Their previously-recorded hashes (from `BUILD_RECEIPT_active_*.json`,
`JOIN_RECEIPT_*.json`, `FROZEN_FACTS.json`) are quoted in this document's table
above only where a *small* file's hash could be cross-checked directly.

## Explicitly not done by this freeze step

- No file under any Track F/B/postproduction directory was modified.
- No checkpoint was loaded into a model or run forward (that starts in the
  audit phase, read-only, in-memory only).
- No training or GPU production was launched.
