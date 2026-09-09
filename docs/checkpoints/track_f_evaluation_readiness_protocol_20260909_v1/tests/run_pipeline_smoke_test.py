#!/usr/bin/env python3
"""End-to-end smoke test for this package's code/ pipeline, run against
SYNTHETIC_TEST_FIXTURE_* data only (tests/fixtures/make_synthetic_fixtures.py)
-- never real model output. Proves the scripts named in the task (result
JSON schemas, table generators, figure scripts, GO/NO-GO decision logic)
actually run end-to-end, produce schema-conformant output, and that the
decision logic reaches the mechanically-correct conclusion on data
engineered to demonstrate each of Story A / Story C / a harm scenario.

This is a correctness/plumbing test of the TOOLING, not a scientific
result -- every artifact it produces is confined to a throwaway temp
directory and is not part of this package's frozen deliverables.

Usage: python3 run_pipeline_smoke_test.py
Exit 0 = all checks passed. Exit 1 = at least one failed (details printed).
"""
import json
import os
import subprocess
import sys
import tempfile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(THIS_DIR)
CODE_DIR = os.path.join(PKG_DIR, "code")
SCHEMA_DIR = os.path.join(PKG_DIR, "schemas")
FIXTURES_DIR = os.path.join(THIS_DIR, "fixtures")

PASS = []
FAIL = []


def check(desc, cond):
    if cond:
        PASS.append(desc)
        print(f"  PASS: {desc}")
    else:
        FAIL.append(desc)
        print(f"  FAIL: {desc}")


def run(cmd, **kwargs):
    r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    return r


def main():
    # ---- unit tests first: symmetry invariance (task-required, PREREGISTRATION.md
    # Amendment 2026-09-10) and the bootstrap-threshold secondary-audit verification.
    # Both are pure unit tests (no npz/tempdir needed); run as subprocesses so a
    # single invocation of this script is the one command that runs everything. ----
    for unit_test in ("test_symmetry_invariance.py", "test_bootstrap_threshold_recompute.py"):
        r = run([sys.executable, os.path.join(THIS_DIR, unit_test)])
        print(r.stdout)
        if r.stderr:
            print(r.stderr, file=sys.stderr)
        check(f"{unit_test} exits 0 (all its own internal checks passed)", r.returncode == 0)

    with tempfile.TemporaryDirectory(prefix="spa2m_part_eval_smoke_") as tmp:
        print(f"=== working dir: {tmp} ===")

        for seed in (0, 1, 2):
            r = run([sys.executable, os.path.join(FIXTURES_DIR, "make_synthetic_fixtures.py"),
                     "--out-dir", tmp, "--seed", str(seed)])
            check(f"fixture generation seed={seed} exits 0", r.returncode == 0)

        control_npz = os.path.join(tmp, "SYNTHETIC_TEST_FIXTURE_control_seed0.npz")
        native10m_json = os.path.join(tmp, "SYNTHETIC_TEST_FIXTURE_native10m_aggregate.json")

        # ---- storyA: 3 seeds, expect a clean, replicated improvement signal ----
        storyA_results = []
        for seed in (0, 1, 2):
            out_json = os.path.join(tmp, f"comparison_result_storyA_seed{seed}.json")
            r = run([sys.executable, os.path.join(CODE_DIR, "evaluate_multi_model.py"),
                     "--control-npz", control_npz,
                     "--test-npz", os.path.join(tmp, f"SYNTHETIC_TEST_FIXTURE_test_storyA_seed{seed}.npz"),
                     "--native10m-aggregate-json", native10m_json,
                     "--seed", str(seed), "--n-boot", "500",
                     "--synthetic-test-fixture", "--out-json", out_json])
            check(f"evaluate_multi_model.py storyA seed={seed} exits 0", r.returncode == 0)
            if r.returncode == 0:
                with open(out_json) as f:
                    storyA_results.append(json.load(f))
                storyA_results[-1]["_path"] = out_json

        if len(storyA_results) == 3:
            check("storyA seed0: sig_class True (engineered AUC improvement detected)",
                  storyA_results[0]["story_evaluation"]["single_seed_flags"]["sig_class"] is True)
            check("storyA seed0: no harm flags", not (storyA_results[0]["story_evaluation"]["single_seed_flags"]["harm_class"] or
                                                        storyA_results[0]["story_evaluation"]["single_seed_flags"]["harm_reco"]))
            check("storyA: all 3 seeds show positive observed delta-AUC",
                  all(r["paired_bootstrap_delta_auc"]["all_background"]["observed_delta_test_minus_control"] > 0 for r in storyA_results))
            check("storyA: native10m Story B block present with a path", "path" in storyA_results[0]["story_evaluation"]["story_b"])

        # ---- storyC: expect no significant signal ----
        out_json_c = os.path.join(tmp, "comparison_result_storyC_seed0.json")
        r = run([sys.executable, os.path.join(CODE_DIR, "evaluate_multi_model.py"),
                 "--control-npz", control_npz,
                 "--test-npz", os.path.join(tmp, "SYNTHETIC_TEST_FIXTURE_test_storyC_seed0.npz"),
                 "--seed", "0", "--n-boot", "500", "--synthetic-test-fixture", "--out-json", out_json_c])
        check("evaluate_multi_model.py storyC exits 0", r.returncode == 0)
        storyC_flags = None
        if r.returncode == 0:
            with open(out_json_c) as f:
                storyC_flags = json.load(f)["story_evaluation"]["single_seed_flags"]
            check("storyC: no significant classification signal (sig_class False)", storyC_flags["sig_class"] is False)
            check("storyC: no harm signal (harm_class False)", storyC_flags["harm_class"] is False)

        # ---- harm scenario: expect a detected regression ----
        out_json_h = os.path.join(tmp, "comparison_result_harm_seed0.json")
        r = run([sys.executable, os.path.join(CODE_DIR, "evaluate_multi_model.py"),
                 "--control-npz", control_npz,
                 "--test-npz", os.path.join(tmp, "SYNTHETIC_TEST_FIXTURE_test_harm_seed0.npz"),
                 "--seed", "0", "--n-boot", "500", "--synthetic-test-fixture", "--out-json", out_json_h])
        check("evaluate_multi_model.py harm exits 0", r.returncode == 0)
        if r.returncode == 0:
            with open(out_json_h) as f:
                harm_flags = json.load(f)["story_evaluation"]["single_seed_flags"]
            check("harm scenario: harm_class True (engineered regression detected)", harm_flags["harm_class"] is True)

        # ---- decide_go_no_go.py: storyA multi-seed should GO then AUTHORIZE ----
        if len(storyA_results) == 3:
            decision_json = os.path.join(tmp, "go_no_go_decision_storyA.json")
            r = run([sys.executable, os.path.join(CODE_DIR, "decide_go_no_go.py"),
                     "--seed-result", f"seed0={storyA_results[0]['_path']}",
                     "--seed-result", f"seed1={storyA_results[1]['_path']}",
                     "--seed-result", f"seed2={storyA_results[2]['_path']}",
                     "--out-json", decision_json])
            check("decide_go_no_go.py storyA (3 seeds) exits 0", r.returncode == 0)
            if r.returncode == 0:
                with open(decision_json) as f:
                    decision = json.load(f)
                check("storyA more_seeds_decision == GO_RUN_2_ADDITIONAL_SEEDS",
                      decision["more_seeds_decision"]["decision"] == "GO_RUN_2_ADDITIONAL_SEEDS")
                check("storyA additional8m_decision reachable (3 seeds evaluated)",
                      decision["additional8m_decision"]["reachable"] is True)
                check("storyA additional8m_decision == AUTHORIZE (large, consistent engineered effect)",
                      decision["additional8m_decision"]["decision"] == "AUTHORIZE")

        # ---- decide_go_no_go.py: harm-as-original-seed should NO_GO_INVESTIGATE_HARM ----
        if os.path.exists(out_json_h):
            decision_json_h = os.path.join(tmp, "go_no_go_decision_harm.json")
            r = run([sys.executable, os.path.join(CODE_DIR, "decide_go_no_go.py"),
                     "--seed-result", f"seed0={out_json_h}", "--out-json", decision_json_h])
            check("decide_go_no_go.py harm-as-original exits 0", r.returncode == 0)
            if r.returncode == 0:
                with open(decision_json_h) as f:
                    decision_h = json.load(f)
                check("harm-as-original more_seeds_decision == NO_GO_INVESTIGATE_HARM",
                      decision_h["more_seeds_decision"]["decision"] == "NO_GO_INVESTIGATE_HARM")

        # ---- build_tables.py / build_figures.py on storyA seed0 ----
        if storyA_results:
            tables_dir = os.path.join(tmp, "tables")
            r = run([sys.executable, os.path.join(CODE_DIR, "build_tables.py"),
                     "--comparison-result", storyA_results[0]["_path"],
                     "--go-no-go-decision", os.path.join(tmp, "go_no_go_decision_storyA.json"),
                     "--out-dir", tables_dir])
            check("build_tables.py exits 0", r.returncode == 0)
            expected_tables = ["auc_summary.tsv", "paired_bootstrap_deltas.tsv", "rejection_at_fixed_efficiency.tsv",
                                "reconstruction_summary.tsv", "mcnemar_reconstruction.tsv",
                                "jet_multiplicity_stratified_auc.tsv", "resource_summary.tsv",
                                "go_no_go_per_seed.tsv", "go_no_go_summary.tsv"]
            for t in expected_tables:
                p = os.path.join(tables_dir, t)
                check(f"table written and non-empty: {t}", os.path.isfile(p) and os.path.getsize(p) > 0)

            figs_dir = os.path.join(tmp, "figures")
            r = run([sys.executable, os.path.join(CODE_DIR, "build_figures.py"),
                     "--comparison-result", storyA_results[0]["_path"], "--out-dir", figs_dir])
            check("build_figures.py exits 0", r.returncode == 0)
            expected_figs = ["delta_auc_forest_plot.svg", "auc_comparison_bar.svg",
                              "jet_multiplicity_stratified_auc.svg", "rejection_vs_epsS.svg"]
            for fig in expected_figs:
                p = os.path.join(figs_dir, fig)
                ok = os.path.isfile(p) and os.path.getsize(p) > 0
                if ok:
                    with open(p) as fh:
                        head = fh.read(200)
                    ok = "<svg" in head or "<?xml" in head
                check(f"figure written and looks like SVG: {fig}", ok)

        # ---- schema validation ----
        if storyA_results:
            r = run([sys.executable, os.path.join(CODE_DIR, "validate_schema.py"),
                     "--schema", os.path.join(SCHEMA_DIR, "comparison_result.schema.json"),
                     "--instance", storyA_results[0]["_path"]])
            check("comparison_result.json validates against its schema", r.returncode == 0)

            r = run([sys.executable, os.path.join(CODE_DIR, "validate_schema.py"),
                     "--schema", os.path.join(SCHEMA_DIR, "go_no_go_decision.schema.json"),
                     "--instance", os.path.join(tmp, "go_no_go_decision_storyA.json")])
            check("go_no_go_decision.json validates against its schema", r.returncode == 0)

        r = run([sys.executable, os.path.join(CODE_DIR, "validate_schema.py"),
                 "--schema", os.path.join(SCHEMA_DIR, "model_eval_aggregate.schema.json"),
                 "--instance", native10m_json])
        check("native10m aggregate fixture validates against model_eval_aggregate.schema.json", r.returncode == 0)

        # ---- negative control: validator must actually catch a broken instance ----
        broken = os.path.join(tmp, "broken_aggregate.json")
        with open(native10m_json) as f:
            broken_data = json.load(f)
        del broken_data["roc_auc"]
        with open(broken, "w") as f:
            json.dump(broken_data, f)
        r = run([sys.executable, os.path.join(CODE_DIR, "validate_schema.py"),
                 "--schema", os.path.join(SCHEMA_DIR, "model_eval_aggregate.schema.json"),
                 "--instance", broken])
        check("validator correctly REJECTS an instance missing a required field", r.returncode == 1)

    print(f"\n=== SMOKE TEST SUMMARY: {len(PASS)} passed, {len(FAIL)} failed (of {len(PASS) + len(FAIL)}) ===")
    if FAIL:
        print("FAILED CHECKS:")
        for f in FAIL:
            print(f"  - {f}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
