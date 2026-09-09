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
python3 ../tests/run_pipeline_smoke_test.py   # runs the unit tests below first, then the full integration suite
python3 ../tests/test_symmetry_invariance.py             # standalone: H<->H / inner-pair symmetry proofs (20 checks)
python3 ../tests/test_bootstrap_threshold_recompute.py   # standalone: secondary-audit verification (3 checks)
```
