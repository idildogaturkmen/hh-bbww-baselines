#!/usr/bin/env python3
"""Matched evaluation package: native SPA2M (CONTROL) vs SPA2M+ParT (TEST).

Labels every number produced here as a MATCHED DEVELOPMENT/VALIDATION
result -- never "final test performance" (no untouched final test cohort
is currently authorized for this comparison; holdout_B/Stage C are never
read by this script or anything it calls).

INPUT SCHEMA (one .npz per model, CONTROL and TEST, over the IDENTICAL
val cohort = production_2M_val.h5's 400,000 events = the project's
"holdout_A"; both files must carry the SAME event_id set, and this script
re-sorts both by event_id and hard-asserts exact equality before computing
anything paired):
    event_id                 int64   (native_hdf5_row_index)
    process                  int8    0=signal, 1=qcd, 2=ttbar
    score                    float32 signal-vs-all-background discriminant, higher=more signal-like
    assignment_correct       bool    predicted (h1,h2) jet-pair assignment matches truth
    assignment_defined       bool    whether truth assignment exists for this event (typically signal-only)
    higgs_mass_1, higgs_mass_2  float32   reconstructed di-jet masses under the predicted assignment
    n_jets                   int8    real jet multiplicity (native MASK.sum() for this event)
    part_embedding_norm_mean float32 OPTIONAL, TEST-only: mean L2 norm of the (retained) ParT
                                      embedding across this event's real jets; NaN/absent for CONTROL
    jet_pt_leading, jet_eta_leading  float32  OPTIONAL, for the pT/eta stratification diagnostic

Everything here is prepared to RUN once both .npz files exist (i.e. once
TEST has actually been trained and scored) -- this script does not launch
inference and is not itself run against real model outputs in this
session; see EVALUATION_2M_PART_RUNBOOK.md.

Usage:
    python3 evaluate_matched_2m.py \
        --control-npz control_2m_native_eval.npz \
        --test-npz test_2m_part_eval.npz \
        --control-training-result training_2M_seed0_result.json \
        --test-training-result training_spanet_2M_part_active_seed0_result.json \
        --build-receipt BUILD_RECEIPT_active_val.json \
        --out-json EVALUATION_RESULT.json
"""
import argparse
import json
import sys

import numpy as np

from bootstrap_utils import roc_auc, paired_bootstrap_delta, mcnemar_test, exact_poisson_count_interval

TARGET_EFFICIENCIES = [0.10, 0.075, 0.05, 0.04, 0.03]
PROCESS_SIGNAL, PROCESS_QCD, PROCESS_TTBAR = 0, 1, 2


def load_model_eval(path):
    d = np.load(path)
    required = ["event_id", "process", "score", "assignment_correct", "assignment_defined",
                "higgs_mass_1", "higgs_mass_2", "n_jets"]
    for k in required:
        if k not in d:
            raise KeyError(f"{path}: missing required array '{k}'")
    order = np.argsort(d["event_id"])
    return {k: d[k][order] for k in d.files}


def rejection_at_fixed_efficiency(scores, labels_binary, target_eff, bg_process=None, process=None):
    """Threshold derived via exact order statistics on the SIGNAL score
    distribution of THIS model's own dev cohort (per-model threshold
    derivation, consistent target efficiency -- matches the project's
    established exact-rank-threshold convention). bg_process: None=all
    background, or PROCESS_QCD/PROCESS_TTBAR to restrict the background
    side of rejection to one process (process array required in that case).
    """
    sig_scores = np.sort(scores[labels_binary == 1])[::-1]
    n_signal_total = len(sig_scores)
    rank = int(round(target_eff * n_signal_total)) - 1
    rank = max(0, min(rank, n_signal_total - 1))
    threshold = sig_scores[rank]
    n_signal_pass = int((scores[labels_binary == 1] >= threshold).sum())
    achieved_eff = n_signal_pass / n_signal_total

    if bg_process is None:
        bg_mask = labels_binary == 0
    else:
        bg_mask = process == bg_process
    n_bg_total = int(bg_mask.sum())
    n_bg_pass = int((scores[bg_mask] >= threshold).sum())
    eps_b = n_bg_pass / n_bg_total if n_bg_total else None
    rejection = (n_bg_total / n_bg_pass) if n_bg_pass else None

    result = {
        "target_epsS": target_eff, "threshold": float(threshold), "achieved_epsS": achieved_eff,
        "n_signal_pass": n_signal_pass, "n_signal_total": n_signal_total,
        "n_bg_pass": n_bg_pass, "n_bg_total": n_bg_total,
        "epsB": eps_b, "rejection": rejection,
        "finite_support_warnings": [],
    }
    if n_bg_pass == 0:
        result["finite_support_warnings"].append(
            f"zero survivors at epsS={target_eff} (threshold={threshold:.6g}) -- efficiency=0 exactly, "
            f"rejection undefined (only a lower bound R > {n_bg_total} is supported)")
    elif n_bg_pass < 10:
        result["finite_support_warnings"].append(
            f"only {n_bg_pass} survivors at epsS={target_eff} -- efficiency/rejection is low-statistics "
            f"(Poisson-limited), not a precise estimate")
        result["exact_poisson_count_interval"] = exact_poisson_count_interval(n_bg_pass)
    return result


def stratify_by_jet_multiplicity(control, test):
    strata = {"eq4": lambda n: n == 4, "eq5": lambda n: n == 5, "ge5": lambda n: n >= 5, "ge6": lambda n: n >= 6}
    out = {}
    for name, pred in strata.items():
        m_c = pred(control["n_jets"])
        m_t = pred(test["n_jets"])
        if m_c.sum() < 50 or m_t.sum() < 50:
            out[name] = {"n_events_control": int(m_c.sum()), "n_events_test": int(m_t.sum()),
                         "note": "fewer than 50 events -- AUC not computed, too few for a stable estimate"}
            continue
        labels_c = (control["process"][m_c] == PROCESS_SIGNAL).astype(int)
        labels_t = (test["process"][m_t] == PROCESS_SIGNAL).astype(int)
        out[name] = {
            "n_events_control": int(m_c.sum()), "n_events_test": int(m_t.sum()),
            "auc_control": roc_auc(control["score"][m_c], labels_c),
            "auc_test": roc_auc(test["score"][m_t], labels_t),
        }
    return out


def score_mass_correlation(model_eval):
    sig = model_eval["process"] == PROCESS_SIGNAL
    m1, m2, s = model_eval["higgs_mass_1"][sig], model_eval["higgs_mass_2"][sig], model_eval["score"][sig]
    finite = np.isfinite(m1) & np.isfinite(m2) & np.isfinite(s)
    m1, m2, s = m1[finite], m2[finite], s[finite]

    def pearson(x, y):
        if len(x) < 3 or x.std() == 0 or y.std() == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    def spearman(x, y):
        if len(x) < 3:
            return None
        rx, ry = x.argsort().argsort().astype(float), y.argsort().argsort().astype(float)
        return pearson(rx, ry)

    return {
        "n_signal_events_used": int(len(s)),
        "pearson_score_vs_mass1": pearson(s, m1), "pearson_score_vs_mass2": pearson(s, m2),
        "spearman_score_vs_mass1": spearman(s, m1), "spearman_score_vs_mass2": spearman(s, m2),
        "mass1_resolution_std": float(m1.std()) if len(m1) else None,
        "mass2_resolution_std": float(m2.std()) if len(m2) else None,
    }


def embedding_norm_diagnostics(test):
    if "part_embedding_norm_mean" not in test:
        return {"note": "part_embedding_norm_mean not present in TEST eval arrays -- skipped"}
    norm = test["part_embedding_norm_mean"]
    finite = np.isfinite(norm)
    out = {"n_events": int(finite.sum()), "norm_mean": float(norm[finite].mean()), "norm_std": float(norm[finite].std())}
    for var in ("jet_pt_leading", "jet_eta_leading", "n_jets"):
        if var in test:
            x = test[var][finite]
            xf = np.isfinite(x) if x.dtype.kind == "f" else np.ones_like(x, dtype=bool)
            if xf.sum() > 2 and x[xf].std() > 0 and norm[finite][xf].std() > 0:
                out[f"pearson_norm_vs_{var}"] = float(np.corrcoef(x[xf], norm[finite][xf])[0, 1])
    return out


def reconstruction_metrics(model_eval):
    defined = model_eval["assignment_defined"]
    n_defined = int(defined.sum())
    acc = float(model_eval["assignment_correct"][defined].mean()) if n_defined else None
    exact_event_reco = acc  # single pairing definition here == exact-event reconstruction under this schema
    return {
        "n_events_with_defined_assignment": n_defined,
        "higgs_pairing_accuracy": acc,
        "exact_event_reconstruction_efficiency": exact_event_reco,
    }


def resource_metrics(training_result_path):
    if training_result_path is None:
        return {"note": "no training result JSON provided"}
    with open(training_result_path) as f:
        r = json.load(f)
    return {
        "total_fit_wall_s": r.get("total_fit_wall_s"),
        "gpu_peak_reserved_bytes": r.get("gpu_peak_reserved_bytes"),
        "gpu_peak_allocated_bytes": r.get("gpu_peak_allocated_bytes"),
        "host_peak_rss_kib_self": r.get("host_peak_rss_kib_self"),
        "gpu_device_name": r.get("gpu_device_name"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--control-training-result", default=None)
    ap.add_argument("--test-training-result", default=None)
    ap.add_argument("--build-receipt", default=None, help="BUILD_RECEIPT_<variant>_val.json, for parameter-count delta")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    control = load_model_eval(args.control_npz)
    test = load_model_eval(args.test_npz)

    if not np.array_equal(control["event_id"], test["event_id"]):
        print("FATAL: control and test event_id arrays are not identical after sorting -- these are not "
              "the same matched dev cohort in the same identity. Refusing to compute any paired statistic.",
              file=sys.stderr)
        sys.exit(2)
    if not np.array_equal(control["process"], test["process"]):
        print("FATAL: control and test 'process' labels disagree for at least one shared event_id -- "
              "identity/label mismatch. Refusing.", file=sys.stderr)
        sys.exit(3)

    labels_signal = (control["process"] == PROCESS_SIGNAL).astype(int)
    n_events = len(labels_signal)

    result = {
        "_label": "MATCHED_DEVELOPMENT_VALIDATION_RESULT_NOT_FINAL_TEST_PERFORMANCE",
        "n_events_shared_cohort": n_events,
        "process_composition": {
            "signal": int((control["process"] == PROCESS_SIGNAL).sum()),
            "qcd": int((control["process"] == PROCESS_QCD).sum()),
            "ttbar": int((control["process"] == PROCESS_TTBAR).sum()),
        },
    }

    # ---- Classification: AUC (all-bg, QCD-only, ttbar-only) ----
    def auc_subset(model_eval, bg_process=None):
        if bg_process is None:
            mask = np.ones(n_events, dtype=bool)
        else:
            mask = (model_eval["process"] == PROCESS_SIGNAL) | (model_eval["process"] == bg_process)
        return roc_auc(model_eval["score"][mask], (model_eval["process"][mask] == PROCESS_SIGNAL).astype(int))

    result["auc"] = {
        "control": {"all_background": auc_subset(control), "qcd": auc_subset(control, PROCESS_QCD), "ttbar": auc_subset(control, PROCESS_TTBAR)},
        "test": {"all_background": auc_subset(test), "qcd": auc_subset(test, PROCESS_QCD), "ttbar": auc_subset(test, PROCESS_TTBAR)},
    }

    def auc_stat_fn_all(scores, labels, idx):
        return roc_auc(scores[idx], labels[idx])

    result["paired_bootstrap_delta_auc"] = {
        "all_background": paired_bootstrap_delta(test["score"], control["score"], labels_signal, auc_stat_fn_all, n_boot=args.n_boot, seed=args.seed),
    }
    for name, proc in [("qcd", PROCESS_QCD), ("ttbar", PROCESS_TTBAR)]:
        mask = (control["process"] == PROCESS_SIGNAL) | (control["process"] == proc)
        sub_labels = (control["process"][mask] == PROCESS_SIGNAL).astype(int)
        result["paired_bootstrap_delta_auc"][name] = paired_bootstrap_delta(
            test["score"][mask], control["score"][mask], sub_labels, auc_stat_fn_all, n_boot=args.n_boot, seed=args.seed)

    # ---- Classification: rejection at fixed signal efficiencies ----
    result["rejection_at_fixed_efficiency"] = {"control": {}, "test": {}}
    for model_name, model_eval in [("control", control), ("test", test)]:
        for eff in TARGET_EFFICIENCIES:
            key = f"epsS_{eff}"
            result["rejection_at_fixed_efficiency"][model_name][key] = {
                "all_background": rejection_at_fixed_efficiency(model_eval["score"], labels_signal, eff),
                "qcd": rejection_at_fixed_efficiency(model_eval["score"], labels_signal, eff, bg_process=PROCESS_QCD, process=model_eval["process"]),
                "ttbar": rejection_at_fixed_efficiency(model_eval["score"], labels_signal, eff, bg_process=PROCESS_TTBAR, process=model_eval["process"]),
            }

    def rejection_stat_fn(scores, labels, idx, eff=None, bg_process=None, process_full=None):
        proc = process_full[idx] if process_full is not None else None
        r = rejection_at_fixed_efficiency(scores[idx], labels[idx], eff, bg_process=bg_process, process=proc)
        return r["rejection"] if r["rejection"] is not None else 0.0

    result["paired_bootstrap_delta_rejection"] = {}
    for eff in TARGET_EFFICIENCIES:
        result["paired_bootstrap_delta_rejection"][f"epsS_{eff}"] = paired_bootstrap_delta(
            test["score"], control["score"], labels_signal, rejection_stat_fn,
            n_boot=args.n_boot, seed=args.seed, eff=eff, bg_process=None, process_full=control["process"])

    # ---- Reconstruction / assignment ----
    result["reconstruction"] = {"control": reconstruction_metrics(control), "test": reconstruction_metrics(test)}

    # ---- McNemar on paired event-level reconstruction correctness ----
    both_defined = control["assignment_defined"] & test["assignment_defined"]
    result["mcnemar_reconstruction_correctness"] = mcnemar_test(
        control["assignment_correct"][both_defined], test["assignment_correct"][both_defined])
    result["mcnemar_n_events_both_defined"] = int(both_defined.sum())

    # ---- Robustness / physics ----
    result["score_mass_correlation"] = {"control": score_mass_correlation(control), "test": score_mass_correlation(test)}
    result["jet_multiplicity_stratified_auc"] = stratify_by_jet_multiplicity(control, test)
    result["embedding_norm_diagnostics_test_only"] = embedding_norm_diagnostics(test)

    # ---- Resource ----
    result["resource"] = {
        "control": resource_metrics(args.control_training_result),
        "test": resource_metrics(args.test_training_result),
    }
    if args.build_receipt:
        with open(args.build_receipt) as f:
            br = json.load(f)
        result["resource"]["parameter_count_increase"] = {
            "n_retained_part_dims": br["n_retained_part_dims"],
            "delta_w_first_layer_scalars": 16 * br["n_retained_part_dims"],
            "note": "first-layer Source embedding parameter delta = 16 x n_retained_part_dims (initial_embedding_dim=16, unchanged), per project convention",
        }

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ("jet_multiplicity_stratified_auc",)}, indent=2)[:4000])
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
