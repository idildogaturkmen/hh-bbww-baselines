# code/ — provenance

| file | provenance | notes |
|---|---|---|
| `bootstrap_utils.py` | **copied verbatim, byte-identical** from `track_f_postproduction_pipeline_20260908_v1/scripts/eval/bootstrap_utils.py` | not edited; see `SHA256SUMS` in this package's root — hash matches the source file exactly |
| `bootstrap_utils_ext.py` | **new, additive** | imports (does not duplicate) from `bootstrap_utils.py`; adds exactly one new function, `exact_binomial_interval()` (Clopper-Pearson), the one genuinely new statistical formula this package introduces |
| `evaluate_multi_model.py` | **new**, extends the logic of `track_f_postproduction_pipeline_20260908_v1/scripts/eval/evaluate_matched_2m.py` (not edited; that file is untouched) | adds the native-SPA10M third arm, extra_jet_activity/le4 strata, single-seed story flags, and (2026-09-10) `match_higgs_pairs()` — see PREREGISTRATION.md Amendment 2026-09-10 for the H<->H permutation-symmetry correction and a second bug fix (a degenerate/always-zero paired-bootstrap delta) found while rewriting this file |
| `decide_go_no_go.py` | **new** | implements `../PREREGISTRATION.md` section 7 mechanically; never hand-edit a decision it produces |
| `validate_schema.py` | **new** | minimal dependency-free JSON Schema subset validator — this environment has no `jsonschema` package installed (confirmed 2026-09-09, not assumed) |
| `build_tables.py` | **new** | TSV table generator, follows this project's `results/<name>/tables/*.tsv` convention |
| `build_figures.py` | **new** | SVG figure generator, follows this project's `results/<name>/plots/*.svg` convention (style applied by hand, not imported cross-repo — see file docstring) |
| `export_model_eval_events.py` | **new (2026-09-10)** | closes G5 in `track_f_postproduction_pipeline_20260908_v1/FINAL_EXECUTION_PACKET.md` Step 9. Adapts (parameterizes, does not hardcode) `evaluate_classification.py`'s architecture/options block and `_patched_load_assignments` weight-loading patch; reuses `model.predict(sources).assignments` (same call as `part2_spanet_assignment.py`) and `four_vec()`/`pair_mass()` (copied verbatim from `part3_event_listing_and_truth.py`/`part6_pairing_accuracy_and_mass.py`). Run for real against CONTROL — see "Real CONTROL export" below. |

## Real CONTROL export (2026-09-10 — run for real, not synthetic)

`export_model_eval_events.py` was run against the frozen native SPA2M checkpoint (read-only inference on an
already-public, non-ParT2M checkpoint) and produced:

```
/uscms_data/d3/iturkmen/hh4b_delphes/track_f_postproduction_pipeline_20260908_v1/exports/control_2m_native_eval.npz
sha256: bebfb680bf8e76a90bd0e4d626997a92b00a300c21444b1c2889986dff7ee26d
400,000/400,000 events, process=signal:193358/qcd:174485/ttbar:32157 (exact match to FROZEN_FACTS.json)
```

Cross-checked against existing frozen artifacts (full numbers, methodology, and the environment/threading
explanation for the one non-exact cross-check are in `../STATUS.md`):
- truth (`truth_targets_400k.npz`): **exact match**, all 400,000 events, b1/b2/b3/b4.
- classification score/AUC (`classification_evaluation_result.json`): matches to **~1e-10** (floating-point
  noise level).
- predicted assignments (`spanet2m_signal_pairing_sample.npz`, 10,000-event sample): **99.11% exact match**
  (9,911/10,000 rows); the 89-row (0.89%) residual is attributable to CPU floating-point non-determinism on
  near-tied assignment decisions — confirmed **identical** torch 2.8.0+cu128/numpy 2.0.2 in both environments
  (ruling out a version mismatch) — and is symmetric/non-systematic (net aggregate shift of only 3-5 events
  out of 10,000 on every pairing-accuracy metric in `signal_pairing_accuracy_and_mass_10k.json`).

Environment used (torch/pytorch_lightning are NOT installed in this LPC shell's default `/usr/bin/python3`,
confirmed 2026-09-10):
```
/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4U_eaf_official_spanet_gpu_canary_20260815_v1/.pixi/envs/default/bin/python3
```
This environment's own editable-installed `spanet` package resolves to a **different** git commit
(`46c68051...`) than the project's pinned commit (`debbdc999...`) — `export_model_eval_events.py`'s
`--spanet-repo` argument is inserted at `sys.path[0]` explicitly to force the pinned-commit tree to be used
regardless (verified: `spanet.__file__` resolves under the pinned path after the override, and the script
hard-asserts this itself before proceeding).

## Running the real pipeline (once TEST/ParT2M results exist — NOT run against real data by this package)

```bash
python3 evaluate_multi_model.py \
    --control-npz <native_spa2m_val_eval.npz> \
    --test-npz <spa2m_part_val_eval.npz> \
    --control-training-result <training_2M_seed0_result.json> \
    --test-training-result <training_spa2m_part_seed0_result.json> \
    --build-receipt <BUILD_RECEIPT_active_val.json> \
    --native10m-aggregate-json <classification_evaluation_10M_result.json> \
    --seed 0 --n-boot 10000 \
    --out-json comparison_result_seed0.json

# repeat for additional seeds if PREREGISTRATION.md section 7.1 calls for them, then:
python3 decide_go_no_go.py \
    --seed-result seed0=comparison_result_seed0.json \
    --seed-result seed1=comparison_result_seed1.json \
    --seed-result seed2=comparison_result_seed2.json \
    --out-json go_no_go_decision.json

python3 build_tables.py --comparison-result comparison_result_seed0.json \
    --go-no-go-decision go_no_go_decision.json --out-dir tables/
python3 build_figures.py --comparison-result comparison_result_seed0.json --out-dir figures/

python3 validate_schema.py --schema ../schemas/comparison_result.schema.json --instance comparison_result_seed0.json
```

## Testing (synthetic fixtures only — see `../tests/`)

```bash
python3 ../tests/run_pipeline_smoke_test.py   # runs the 3 unit-test modules below first, then the full integration suite
python3 ../tests/test_symmetry_invariance.py             # standalone: H<->H / inner-pair symmetry proofs (20 checks)
python3 ../tests/test_bootstrap_threshold_recompute.py   # standalone: secondary-audit verification (3 checks)
python3 ../tests/test_exporter_contract.py               # standalone: against the REAL CONTROL export above (19 checks)
```
