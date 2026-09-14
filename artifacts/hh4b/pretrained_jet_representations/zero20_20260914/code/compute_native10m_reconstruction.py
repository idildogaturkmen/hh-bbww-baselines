#!/usr/bin/env python3
"""Computes Native SPA-Net 10M's exact-event and per-Higgs reconstruction
metrics directly from the frozen per-event export
(native10m_eval_400k.npz), using the EXACT same match_higgs_pairs() /
reconstruction_metrics() convention as the frozen Track-F evaluator
(evaluate_multi_model.py) -- copied verbatim below, not reimplemented, so
there is no risk of a subtly different symmetry/masking convention.

Hard gate: before trusting any Native10M number, this script recomputes
the SAME functions on control_2m_native_eval.npz (Native 2M) and asserts
EXACT (not approximate) reproduction of the already-published, frozen
Native 2M numbers:
    exact_event_hh_reconstruction_efficiency = 0.8665880160209383
    higgs_assignment_pairing_accuracy        = 0.8952988063607884
If that cross-check fails, this script exits non-zero and writes nothing
-- no Native10M number is ever recorded without first proving the
identical code reproduces the known-correct Native 2M answer exactly.

Read-only: opens both .npz files for reading only. No training, no
inference, no bootstrap -- this is pure arithmetic over already-frozen
per-event predictions and truth.
"""
import argparse
import json
import sys

import numpy as np

CONTROL_2M_NPZ = "/uscms_data/d3/iturkmen/hh4b_delphes/track_f_postproduction_pipeline_20260908_v1/exports/control_2m_native_eval.npz"
NATIVE_10M_NPZ = "/uscms_data/d3/iturkmen/hh4b_delphes/track_f_postproduction_pipeline_20260908_v1/exports/native10m_eval_400k.npz"

EXPECTED_NATIVE2M_EXACT_EVENT = 0.8665880160209383
EXPECTED_NATIVE2M_PER_HIGGS = 0.8952988063607884


# ============================================================================
# Verbatim copies from track_f_evaluation_readiness_protocol_20260909_v1/
# code/evaluate_multi_model.py (event_id-sort load convention, _pair_key,
# match_higgs_pairs, reconstruction_metrics) -- not reimplemented.
# ============================================================================

def load_model_eval(path):
    d = np.load(path)
    required = ["event_id", "process", "score", "assignment_defined",
                "pred_b1", "pred_b2", "pred_b3", "pred_b4",
                "truth_b1", "truth_b2", "truth_b3", "truth_b4",
                "higgs_mass_1", "higgs_mass_2", "n_jets"]
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"{path}: missing required array(s) {missing}")
    order = np.argsort(d["event_id"])
    out = {k: d[k][order] for k in d.files}

    defined = out["assignment_defined"]
    truth_b1 = out["truth_b1"]
    if not np.array_equal(defined, truth_b1 != -1):
        raise AssertionError(f"{path}: assignment_defined != (truth_b1 != -1) for at least one event")
    if defined.any():
        truth_pairs = np.stack([out["truth_b1"][defined], out["truth_b2"][defined],
                                 out["truth_b3"][defined], out["truth_b4"][defined]], axis=1)
        if (truth_pairs < 0).any():
            raise AssertionError(f"{path}: a -1 truth jet index appears in an assignment_defined event")
        n_unique = np.apply_along_axis(lambda row: len(set(row.tolist())), 1, truth_pairs)
        if (n_unique != 4).any():
            raise AssertionError(f"{path}: truth_b1..truth_b4 do not name 4 distinct jets for some event")
        n_jets_defined = out["n_jets"][defined]
        if n_jets_defined.min() < 4:
            raise AssertionError(f"{path}: n_jets < 4 for a matchable (assignment_defined) event")
    return out


def _pair_key(jet_a, jet_b):
    return np.minimum(jet_a, jet_b) * 100 + np.maximum(jet_a, jet_b)


def match_higgs_pairs(pred_b1, pred_b2, pred_b3, pred_b4, truth_b1, truth_b2, truth_b3, truth_b4):
    pred1 = _pair_key(pred_b1, pred_b2)
    pred2 = _pair_key(pred_b3, pred_b4)
    truth1 = _pair_key(truth_b1, truth_b2)
    truth2 = _pair_key(truth_b3, truth_b4)

    direct = (pred1 == truth1).astype(np.int8) + (pred2 == truth2).astype(np.int8)
    swapped = (pred1 == truth2).astype(np.int8) + (pred2 == truth1).astype(np.int8)
    n_correct_higgs = np.maximum(direct, swapped)
    exact_event_correct = n_correct_higgs == 2
    return n_correct_higgs, exact_event_correct


def reconstruction_metrics(model_eval, n_correct_higgs, exact_event_correct):
    defined = model_eval["assignment_defined"]
    n_defined = int(defined.sum())
    if n_defined == 0:
        return {"n_events_with_defined_assignment": 0,
                "exact_event_hh_reconstruction_efficiency": None,
                "higgs_assignment_pairing_accuracy": None}
    return {
        "n_events_with_defined_assignment": n_defined,
        "exact_event_hh_reconstruction_efficiency": float(exact_event_correct[defined].mean()),
        "higgs_assignment_pairing_accuracy": float(n_correct_higgs[defined].mean() / 2.0),
    }


# ============================================================================


def compute(path):
    m = load_model_eval(path)
    n_correct_higgs, exact_event_correct = match_higgs_pairs(
        m["pred_b1"], m["pred_b2"], m["pred_b3"], m["pred_b4"],
        m["truth_b1"], m["truth_b2"], m["truth_b3"], m["truth_b4"])
    return reconstruction_metrics(m, n_correct_higgs, exact_event_correct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    print(f"==== Hard cross-check: recomputing Native 2M reconstruction from {CONTROL_2M_NPZ} ====")
    native2m = compute(CONTROL_2M_NPZ)
    print(json.dumps(native2m, indent=2))

    exact_match = (native2m["exact_event_hh_reconstruction_efficiency"] == EXPECTED_NATIVE2M_EXACT_EVENT)
    higgs_match = (native2m["higgs_assignment_pairing_accuracy"] == EXPECTED_NATIVE2M_PER_HIGGS)

    print(f"exact_event_hh_reconstruction_efficiency: got={native2m['exact_event_hh_reconstruction_efficiency']!r} "
          f"expected={EXPECTED_NATIVE2M_EXACT_EVENT!r} EXACT_MATCH={exact_match}")
    print(f"higgs_assignment_pairing_accuracy: got={native2m['higgs_assignment_pairing_accuracy']!r} "
          f"expected={EXPECTED_NATIVE2M_PER_HIGGS!r} EXACT_MATCH={higgs_match}")

    if not (exact_match and higgs_match):
        print("FATAL: cross-check against the frozen Native 2M numbers did NOT reproduce exactly. "
              "Refusing to compute or record any Native 10M number.", file=sys.stderr)
        sys.exit(3)

    print(f"\n==== Cross-check PASSED. Computing Native 10M reconstruction from {NATIVE_10M_NPZ} ====")
    native10m = compute(NATIVE_10M_NPZ)
    print(json.dumps(native10m, indent=2))

    out = dict(
        cross_check=dict(
            source_npz=CONTROL_2M_NPZ,
            computed=native2m,
            expected_exact_event_hh_reconstruction_efficiency=EXPECTED_NATIVE2M_EXACT_EVENT,
            expected_higgs_assignment_pairing_accuracy=EXPECTED_NATIVE2M_PER_HIGGS,
            exact_event_exact_match=exact_match,
            higgs_pairing_exact_match=higgs_match,
            PASS=True,
        ),
        native10m=dict(source_npz=NATIVE_10M_NPZ, **native10m),
    )
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
