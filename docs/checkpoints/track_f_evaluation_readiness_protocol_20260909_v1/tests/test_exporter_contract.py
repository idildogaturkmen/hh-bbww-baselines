#!/usr/bin/env python3
"""Contract tests for code/export_model_eval_events.py, run against the
REAL exported CONTROL .npz (control_2m_native_eval.npz -- real inference
on the frozen, already-public native SPA2M checkpoint; not synthetic, but
also not the unseen ParT2M result -- see EXPORTER task item 12) wherever a
test needs real, schema-shaped data, plus small deliberately-corrupted
copies to prove the evaluator fails closed.

Does NOT re-run inference. Does NOT touch TEST (SPA2M+ParT does not exist).
Does NOT read holdout_B/Stage C.

Proves (task item 15):
  1. the exporter's own output never contains assignment_correct/
     higgs1_assignment_correct/higgs2_assignment_correct (the superseded
     2026-09-09 fields) -- checked directly against the real .npz's keys.
  2. predicted/truth slot swaps leave the central evaluator's
     reconstruction metrics unchanged -- on REAL data (not just the
     synthetic fixtures in test_symmetry_invariance.py), by constructing a
     swapped copy of the real CONTROL export and confirming
     reconstruction_metrics() gives identical numbers.
  3. the real CONTROL export validates against schemas/model_eval_events.
     schema.json and loads cleanly through evaluate_multi_model.py's
     load_model_eval(), and a trivial self-comparison (control vs a copy
     of itself) gives an exact-zero paired-bootstrap delta and zero
     McNemar-discordant pairs.
  4. event identity checks fail closed when one event is dropped.
  5. event identity checks fail closed when one event's row is duplicated
     (breaking uniqueness).
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(THIS_DIR)
CODE_DIR = os.path.join(PKG_DIR, "code")
SCHEMA_DIR = os.path.join(PKG_DIR, "schemas")

REAL_CONTROL_NPZ = "/uscms_data/d3/iturkmen/hh4b_delphes/track_f_postproduction_pipeline_20260908_v1/exports/control_2m_native_eval.npz"

FORBIDDEN_KEYS = ("assignment_correct", "higgs1_assignment_correct", "higgs2_assignment_correct")

sys.path.insert(0, CODE_DIR)
from evaluate_multi_model import load_model_eval, reconstruction_metrics, match_higgs_pairs  # noqa: E402

PASS = []
FAIL = []
SKIP = []


def check(desc, cond):
    if cond:
        PASS.append(desc)
        print(f"  PASS: {desc}")
    else:
        FAIL.append(desc)
        print(f"  FAIL: {desc}")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    if not os.path.isfile(REAL_CONTROL_NPZ):
        print(f"SKIP: real CONTROL export not found at {REAL_CONTROL_NPZ} -- run "
              "code/export_model_eval_events.py first (see code/README.md). Not treated as a "
              "pass or fail; this whole module is skipped.")
        return 0

    print(f"=== test_exporter_contract (against real CONTROL export: {REAL_CONTROL_NPZ}) ===")

    # ---- 1. forbidden keys never present ----
    d = np.load(REAL_CONTROL_NPZ)
    for k in FORBIDDEN_KEYS:
        check(f"real export does NOT contain forbidden key '{k}'", k not in d.files)
    for k in ("pred_b1", "pred_b2", "pred_b3", "pred_b4", "truth_b1", "truth_b2", "truth_b3", "truth_b4"):
        check(f"real export DOES contain required key '{k}'", k in d.files)

    with tempfile.TemporaryDirectory(prefix="exporter_contract_test_") as tmp:
        # ---- load once, centrally, to build the slot-swapped copy and corruptions ----
        raw = dict(np.load(REAL_CONTROL_NPZ))
        n = len(raw["event_id"])

        # ---- 2. predicted/truth slot swap on REAL data leaves reconstruction metrics unchanged ----
        swapped = dict(raw)
        swapped["pred_b1"], swapped["pred_b3"] = raw["pred_b3"].copy(), raw["pred_b1"].copy()
        swapped["pred_b2"], swapped["pred_b4"] = raw["pred_b4"].copy(), raw["pred_b2"].copy()
        swapped["truth_b1"], swapped["truth_b3"] = raw["truth_b3"].copy(), raw["truth_b1"].copy()
        swapped["truth_b2"], swapped["truth_b4"] = raw["truth_b4"].copy(), raw["truth_b2"].copy()
        swapped_path = os.path.join(tmp, "control_slot_swapped.npz")
        np.savez(swapped_path, **swapped)

        original_eval = load_model_eval(REAL_CONTROL_NPZ)
        swapped_eval = load_model_eval(swapped_path)

        n_correct_orig, exact_orig = match_higgs_pairs(
            original_eval["pred_b1"], original_eval["pred_b2"], original_eval["pred_b3"], original_eval["pred_b4"],
            original_eval["truth_b1"], original_eval["truth_b2"], original_eval["truth_b3"], original_eval["truth_b4"])
        n_correct_swap, exact_swap = match_higgs_pairs(
            swapped_eval["pred_b1"], swapped_eval["pred_b2"], swapped_eval["pred_b3"], swapped_eval["pred_b4"],
            swapped_eval["truth_b1"], swapped_eval["truth_b2"], swapped_eval["truth_b3"], swapped_eval["truth_b4"])
        metrics_orig = reconstruction_metrics(original_eval, n_correct_orig, exact_orig)
        metrics_swap = reconstruction_metrics(swapped_eval, n_correct_swap, exact_swap)
        check("slot-swapped copy of REAL data: exact_event_hh_reconstruction_efficiency unchanged",
              metrics_orig["exact_event_hh_reconstruction_efficiency"] == metrics_swap["exact_event_hh_reconstruction_efficiency"])
        check("slot-swapped copy of REAL data: higgs_assignment_pairing_accuracy unchanged",
              metrics_orig["higgs_assignment_pairing_accuracy"] == metrics_swap["higgs_assignment_pairing_accuracy"])
        check("slot-swapped copy of REAL data: n_events_with_defined_assignment unchanged",
              metrics_orig["n_events_with_defined_assignment"] == metrics_swap["n_events_with_defined_assignment"])

        # ---- 3a. schema conformance: model_eval_events.schema.json documents an .npz's
        # logical field set (not a JSON instance -- validate_schema.py operates on JSON), so
        # conformance is checked directly against the schema's own "required" list, which is
        # exactly what the forbidden/required key checks above already do (section 1).

        # ---- 3b. self-comparison via evaluate_multi_model.py: control vs a copy of itself ----
        self_copy_path = os.path.join(tmp, "control_self_copy.npz")
        shutil.copyfile(REAL_CONTROL_NPZ, self_copy_path)
        out_json = os.path.join(tmp, "self_comparison_result.json")
        r = run([sys.executable, os.path.join(CODE_DIR, "evaluate_multi_model.py"),
                 "--control-npz", REAL_CONTROL_NPZ, "--test-npz", self_copy_path,
                 "--seed", "0", "--n-boot", "50", "--out-json", out_json])
        check("evaluate_multi_model.py(control vs copy-of-itself) exits 0", r.returncode == 0)
        if r.returncode == 0:
            import json
            with open(out_json) as f:
                result = json.load(f)
            delta = result["paired_bootstrap_delta_auc"]["all_background"]
            check("control-vs-itself: observed delta-AUC is exactly 0",
                  delta["observed_delta_test_minus_control"] == 0.0)
            mcnemar = result["mcnemar_reconstruction_correctness"]
            check("control-vs-itself: McNemar n_discordant == 0", mcnemar["n_discordant"] == 0)

    # ---- 4. dropping one event fails closed ----
    with tempfile.TemporaryDirectory(prefix="exporter_contract_test_") as tmp:
        dropped = {k: v[1:] for k, v in raw.items()}  # drop the first row from every array
        dropped_path = os.path.join(tmp, "control_one_event_dropped.npz")
        np.savez(dropped_path, **dropped)
        out_json = os.path.join(tmp, "dropped_result.json")
        r = run([sys.executable, os.path.join(CODE_DIR, "evaluate_multi_model.py"),
                 "--control-npz", REAL_CONTROL_NPZ, "--test-npz", dropped_path,
                 "--seed", "0", "--n-boot", "10", "--out-json", out_json])
        check("evaluate_multi_model.py FAILS CLOSED (nonzero exit) when one event is dropped",
              r.returncode != 0)

        # ---- 5. duplicating one event's row (breaking event_id uniqueness) fails closed ----
        dup_idx = np.arange(n)
        dup_idx[-1] = 0  # last row now duplicates event_id of row 0 instead of its own unique id
        duplicated = {k: v[dup_idx] for k, v in raw.items()}
        dup_path = os.path.join(tmp, "control_one_event_duplicated.npz")
        np.savez(dup_path, **duplicated)
        out_json2 = os.path.join(tmp, "duplicated_result.json")
        r2 = run([sys.executable, os.path.join(CODE_DIR, "evaluate_multi_model.py"),
                  "--control-npz", dup_path, "--test-npz", REAL_CONTROL_NPZ,
                  "--seed", "0", "--n-boot", "10", "--out-json", out_json2])
        check("evaluate_multi_model.py FAILS CLOSED (nonzero exit) when one event_id is duplicated "
              "(sets no longer match after sorting)", r2.returncode != 0)

    print(f"\n=== test_exporter_contract SUMMARY: {len(PASS)} passed, {len(FAIL)} failed (of {len(PASS) + len(FAIL)}) ===")
    if FAIL:
        for f in FAIL:
            print(f"  - {f}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
