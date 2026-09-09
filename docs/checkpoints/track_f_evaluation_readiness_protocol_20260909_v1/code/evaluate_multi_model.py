#!/usr/bin/env python3
"""Three-arm evaluation: native SPA2M (CONTROL) vs SPA2M+ParT (TEST) vs
native SPA10M, implementing PREREGISTRATION.md sections 2-6 and producing
a comparison_result.schema.json-conformant JSON. One invocation = one TEST
seed; run decide_go_no_go.py across multiple such outputs for the
multi-seed GO/NO-GO decision (PREREGISTRATION.md section 7).

Extends (does not edit) track_f_postproduction_pipeline_20260908_v1/
scripts/eval/evaluate_matched_2m.py: same CONTROL/TEST cohort-identity
discipline and paired-bootstrap/McNemar machinery, reused verbatim via
bootstrap_utils.py (copied forward in this directory, hash-identical --
see README.md). What is NEW here:
  - a third arm, native SPA10M, compared via the paired path (if a
    matching per-event array is supplied) or the unpaired fallback
    (point-estimate + single-arm bootstrap CI vs the arm's own aggregate
    JSON), per PREREGISTRATION.md section 8 Story B -- the path is chosen
    by data availability alone, recorded in the output, never chosen by
    which one looks better.
  - exact_binomial_interval() attached to every working point's achieved
    epsB (bootstrap_utils_ext.py), alongside the existing Poisson count
    interval for n_bg_pass < 10.
  - le4 stratum (asserted == eq4 in this topology) and extra_jet_activity
    bins added to the jet-multiplicity stratification.
  - single-seed story_a/story_b flags attached (story_c and the multi-seed
    GO/NO-GO rule live in decide_go_no_go.py, which needs >=1 seed of this
    script's output as input).

SYMMETRY CORRECTION (2026-09-10, PREREGISTRATION.md Amendment 2026-09-10,
written before any ParT2M result existed): the two Higgs candidates in
HH->4b are an explicit PERMUTATIONS.EVENT: [h1, h2] symmetry in this
project's own SPA-Net event topology (event_config/trackb_hh4b.yaml), and
each candidate's two jets are a further [b1,b2]/[b3,b4] inner symmetry.
Neither a model's output-slot order nor a truth-storage convention is
guaranteed to track a fixed physical H1-vs-H2 identity across events --
this is precisely what "symmetry preserving attention" means, and is
confirmed by the official SPANet library's own validation_average_jet_
accuracy code (spanet/network/jet_reconstruction/jet_reconstruction_
validation.py), which computes accuracy under EVERY event-level truth
permutation and takes the argmax before scoring. This module reads RAW
predicted/truth jet indices (pred_b1..pred_b4, truth_b1..truth_b4 --
the same naming convention already proven in paper_exports/track_b_harvey_
tail_characterization_20260825_v1/work/{part2_spanet_assignment.py,
part6_pairing_accuracy_and_mass.py}) and computes exact-event/per-Higgs
correctness CENTRALLY via match_higgs_pairs() below, which reproduces that
same argmax-over-permutations principle (direct vs swapped, take max) --
never by trusting a precomputed, potentially slot-order-biased correctness
boolean. The 2026-09-09 version of this module read precomputed
higgs1_assignment_correct/higgs2_assignment_correct booleans directly;
that version is superseded, not silently amended -- see PREREGISTRATION.md
Amendment 2026-09-10 for the full explanation, including a correction to
that version's provenance claim.

Does not launch inference. Does not read holdout_B/Stage C. Refuses (hard
exit, not a warning) to proceed if CONTROL/TEST event_id or process labels
disagree after sorting.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_utils import roc_auc, paired_bootstrap_delta, mcnemar_test, exact_poisson_count_interval
from bootstrap_utils_ext import exact_binomial_interval

TARGET_EFFICIENCIES = [0.10, 0.075, 0.05, 0.04, 0.03]
PROCESS_SIGNAL, PROCESS_QCD, PROCESS_TTBAR = 0, 1, 2
PRIMARY_DECISION_EPSS = 0.05
GO_NO_GO_N_BOOT = 10000


def load_model_eval(path):
    d = np.load(path)
    required = ["event_id", "process", "score", "assignment_defined",
                "pred_b1", "pred_b2", "pred_b3", "pred_b4",
                "truth_b1", "truth_b2", "truth_b3", "truth_b4",
                "higgs_mass_1", "higgs_mass_2", "n_jets"]
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"{path}: missing required array(s) {missing} -- see schemas/model_eval_events.schema.json "
                        "(2026-09-10: raw pred_b*/truth_b* jet indices required; precomputed higgs1/higgs2 "
                        "correctness booleans are no longer read -- see PREREGISTRATION.md Amendment 2026-09-10)")
    order = np.argsort(d["event_id"])
    out = {k: d[k][order] for k in d.files}

    defined = out["assignment_defined"]
    truth_b1 = out["truth_b1"]
    if not np.array_equal(defined, truth_b1 != -1):
        raise AssertionError(
            f"{path}: assignment_defined != (truth_b1 != -1) for at least one event -- the explicit "
            "matchability flag and the truth sentinel convention disagree; refusing to guess which is right")
    if defined.any():
        truth_pairs = np.stack([out["truth_b1"][defined], out["truth_b2"][defined],
                                 out["truth_b3"][defined], out["truth_b4"][defined]], axis=1)
        if (truth_pairs < 0).any():
            raise AssertionError(f"{path}: a -1 (undefined) truth jet index appears in an assignment_defined "
                                  "event -- truth_b1..truth_b4 must all be real indices wherever assignment_defined")
        n_unique = np.apply_along_axis(lambda row: len(set(row.tolist())), 1, truth_pairs)
        if (n_unique != 4).any():
            raise AssertionError(f"{path}: truth_b1..truth_b4 do not name 4 distinct jets for at least one "
                                  "matchable event -- truth pairs must use 4 distinct jets by physical construction")
        n_jets_defined = out["n_jets"][defined]
        if n_jets_defined.min() < 4:
            raise AssertionError(
                f"{path}: n_jets < 4 for a matchable (assignment_defined) event -- the le4==eq4 "
                "equivalence PREREGISTRATION.md section 4.3 asserts does not hold for this data; "
                "stratification logic below must be revisited before trusting le4/eq4 output")

    pred_pairs = np.stack([out["pred_b1"], out["pred_b2"], out["pred_b3"], out["pred_b4"]], axis=1)
    n_unique_pred = np.apply_along_axis(lambda row: len(set(row.tolist())), 1, pred_pairs)
    n_degenerate_pred = int((n_unique_pred < 4).sum())
    if n_degenerate_pred:
        print(f"WARNING {path}: {n_degenerate_pred}/{len(pred_pairs)} events have a predicted jet reused "
              "across pred_b1..pred_b4 (degenerate pair) -- counted, not fatal, matching the established "
              "diagnostic convention in part6_pairing_accuracy_and_mass.py's n_degenerate_predicted_pairs.",
              file=sys.stderr)
    out["_n_degenerate_pred"] = n_degenerate_pred
    return out


def _pair_key(jet_a, jet_b):
    """Order-invariant integer key for an unordered pair of jet indices
    (jet indices are 0..9, well under 100 -- see MAXLEN/N_JETS in the
    frozen extract_shard.py; *100 gives ample, unambiguous headroom)."""
    return np.minimum(jet_a, jet_b) * 100 + np.maximum(jet_a, jet_b)


def match_higgs_pairs(pred_b1, pred_b2, pred_b3, pred_b4, truth_b1, truth_b2, truth_b3, truth_b4):
    """Symmetry-invariant matching of predicted Higgs-candidate pairs
    against truth (PREREGISTRATION.md Amendment 2026-09-10). Reproduces
    the official SPANet library's own argmax-over-event-permutations
    methodology (jet_reconstruction_validation.py) and matches the
    already-established, independently-written implementation in
    part6_pairing_accuracy_and_mass.py (full_correct / n_pairs_correct),
    just vectorized via integer pair-keys instead of a per-event Python
    frozenset loop. Handles BOTH declared symmetries in trackb_hh4b.yaml:
    the outer EVENT: [h1,h2] interchange (via direct-vs-swapped, take max)
    and each pair's own inner [b1,b2]/[b3,b4] interchange (via the
    order-invariant _pair_key -- pair identity never depends on which of
    its two jets is stored first).

    All eight arguments are equal-length 1-D integer arrays (predicted
    values always defined; truth values may be -1 where undefined -- the
    caller is responsible for masking by assignment_defined before
    interpreting output for those rows, exactly as every other per-event
    array in this module is masked by its caller, not by the array itself).

    Returns (n_correct_higgs, exact_event_correct): n_correct_higgs is an
    int8 array in {0,1,2} (task's "n_correct_higgs = max(direct, swapped)");
    exact_event_correct is n_correct_higgs == 2.
    """
    pred1 = _pair_key(pred_b1, pred_b2)
    pred2 = _pair_key(pred_b3, pred_b4)
    truth1 = _pair_key(truth_b1, truth_b2)
    truth2 = _pair_key(truth_b3, truth_b4)

    direct = (pred1 == truth1).astype(np.int8) + (pred2 == truth2).astype(np.int8)
    swapped = (pred1 == truth2).astype(np.int8) + (pred2 == truth1).astype(np.int8)
    n_correct_higgs = np.maximum(direct, swapped)
    exact_event_correct = n_correct_higgs == 2
    return n_correct_higgs, exact_event_correct


def rejection_at_fixed_efficiency(scores, labels_binary, target_eff, bg_process=None, process=None):
    """Per-model exact order-statistic threshold (matches evaluate_matched_2m.py's
    established convention -- each model supplies its own threshold at the
    target efficiency, never a control-derived shared threshold)."""
    sig_scores = np.sort(scores[labels_binary == 1])[::-1]
    n_signal_total = len(sig_scores)
    rank = int(round(target_eff * n_signal_total)) - 1
    rank = max(0, min(rank, n_signal_total - 1))
    threshold = sig_scores[rank]
    n_signal_pass = int((scores[labels_binary == 1] >= threshold).sum())
    achieved_eff = n_signal_pass / n_signal_total

    bg_mask = (labels_binary == 0) if bg_process is None else (process == bg_process)
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
    if n_bg_total:
        result["epsB_exact_binomial_interval"] = exact_binomial_interval(n_bg_pass, n_bg_total)
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


def reconstruction_metrics(model_eval, n_correct_higgs, exact_event_correct):
    """n_correct_higgs/exact_event_correct: full-length arrays from
    match_higgs_pairs(), masked here by assignment_defined -- the one
    place this masking happens, so exact_event_hh_reconstruction_efficiency
    (primary metric #1) and higgs_assignment_pairing_accuracy (primary
    metric #2) are guaranteed to be computed over the identical event set."""
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


def stratify_by_jet_multiplicity(control, test):
    strata = {
        "le4": lambda n: n <= 4, "eq4": lambda n: n == 4, "eq5": lambda n: n == 5,
        "ge5": lambda n: n >= 5, "ge6": lambda n: n >= 6,
    }
    extra_edges = [0, 1, 2, 3, np.inf]
    out = {}
    for name, pred in strata.items():
        m_c, m_t = pred(control["n_jets"]), pred(test["n_jets"])
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
    if not np.array_equal(strata["le4"](control["n_jets"]), strata["eq4"](control["n_jets"])):
        out["le4_eq4_equivalence_warning"] = "le4 != eq4 for CONTROL -- topology assumption in PREREGISTRATION.md 4.3 does not hold"

    extra_c = np.clip(control["n_jets"].astype(float) - 4, 0, None)
    extra_t = np.clip(test["n_jets"].astype(float) - 4, 0, None)
    extra_activity = {}
    for lo, hi in zip(extra_edges[:-1], extra_edges[1:]):
        label = f"extra_{int(lo)}" if np.isfinite(hi) else f"extra_ge_{int(lo)}"
        m_c = (extra_c >= lo) & (extra_c < hi)
        m_t = (extra_t >= lo) & (extra_t < hi)
        extra_activity[label] = {"n_events_control": int(m_c.sum()), "n_events_test": int(m_t.sum())}
    out["extra_jet_activity"] = extra_activity
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


def resource_metrics(training_result_path):
    if training_result_path is None:
        return {"note": "no training result JSON provided"}
    with open(training_result_path) as f:
        r = json.load(f)
    return {k: r.get(k) for k in (
        "total_fit_wall_s", "gpu_peak_reserved_bytes", "gpu_peak_allocated_bytes",
        "host_peak_rss_kib_self", "gpu_device_name")}


def single_arm_bootstrap_auc_ci(scores, labels, n_boot=2000, seed=0, ci=0.95):
    rng = np.random.default_rng(seed)
    n = len(scores)
    observed = roc_auc(scores, labels)
    reps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        reps[i] = roc_auc(scores[idx], labels[idx])
    alpha = 1 - ci
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return {"point_estimate": float(observed), "ci_level": ci, "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


def evaluate_native10m(test, native10m_aggregate_path, native10m_npz_path, n_boot, seed):
    """PREREGISTRATION.md section 8, Story B. Returns (auc_dict_or_None, story_b_dict)."""
    with open(native10m_aggregate_path) as f:
        agg = json.load(f)
    auc10m = agg["roc_auc"]["signal_vs_all_background"]

    if native10m_npz_path:
        n10 = load_model_eval(native10m_npz_path)
        if not np.array_equal(n10["event_id"], test["event_id"]):
            raise AssertionError("native10m --native10m-npz provided but event_id does not match TEST's "
                                  "cohort -- refusing the paired path; use the unpaired fallback instead "
                                  "(omit --native10m-npz)")
        labels = (test["process"] == PROCESS_SIGNAL).astype(int)

        def auc_stat(scores, lab, idx):
            return roc_auc(scores[idx], lab[idx])

        delta = paired_bootstrap_delta(test["score"], n10["score"], labels, auc_stat, n_boot=n_boot, seed=seed)
        story_b = {
            "path": "paired",
            "delta_test_minus_native10m": delta,
            "approaches": bool(delta["ci_low"] <= 0.0 <= delta["ci_high"]),
            "beats": bool(delta["excludes_zero"] and delta["observed_delta_test_minus_control"] > 0),
        }
        return {"all_background": auc10m, "qcd": agg["roc_auc"]["signal_vs_qcd"], "ttbar": agg["roc_auc"]["signal_vs_ttbar"]}, story_b

    labels = (test["process"] == PROCESS_SIGNAL).astype(int)
    test_ci = single_arm_bootstrap_auc_ci(test["score"], labels, n_boot=n_boot, seed=seed)
    half_width = (test_ci["ci_high"] - test_ci["ci_low"]) / 2.0
    story_b = {
        "path": "unpaired_fallback",
        "_flag": "UNPAIRED_COMPARISON_WIDER_UNCERTAINTY",
        "test_single_arm_auc_ci": test_ci,
        "native10m_point_estimate": auc10m,
        "approaches": bool(abs(test_ci["point_estimate"] - auc10m) <= half_width),
        "beats": bool(test_ci["ci_low"] > auc10m),
    }
    return {"all_background": auc10m, "qcd": agg["roc_auc"]["signal_vs_qcd"], "ttbar": agg["roc_auc"]["signal_vs_ttbar"]}, story_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--control-training-result", default=None)
    ap.add_argument("--test-training-result", default=None)
    ap.add_argument("--build-receipt", default=None)
    ap.add_argument("--native10m-aggregate-json", default=None, help="classification_evaluation_10M_result.json-shaped file")
    ap.add_argument("--native10m-npz", default=None, help="OPTIONAL per-event array, event_id-matched to TEST, for the paired Story B path")
    ap.add_argument("--seed", type=int, default=0, help="TEST model seed -- recorded in output, used as bootstrap RNG seed")
    ap.add_argument("--n-boot", type=int, default=GO_NO_GO_N_BOOT)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--synthetic-test-fixture", action="store_true",
                     help="MUST be passed when running against tests/fixtures/ synthetic data; sets _synthetic_test_fixture=true in the output")
    args = ap.parse_args()

    control = load_model_eval(args.control_npz)
    test = load_model_eval(args.test_npz)

    if not np.array_equal(control["event_id"], test["event_id"]):
        print("FATAL: control and test event_id arrays are not identical after sorting.", file=sys.stderr)
        sys.exit(2)
    if not np.array_equal(control["process"], test["process"]):
        print("FATAL: control and test 'process' labels disagree for at least one shared event_id.", file=sys.stderr)
        sys.exit(3)

    labels_signal = (control["process"] == PROCESS_SIGNAL).astype(int)
    n_events = len(labels_signal)

    result = {
        "_label": "MATCHED_DEVELOPMENT_VALIDATION_RESULT_NOT_FINAL_TEST_PERFORMANCE",
        "_seed": args.seed,
        "n_events_shared_cohort": n_events,
        "process_composition": {
            "signal": int((control["process"] == PROCESS_SIGNAL).sum()),
            "qcd": int((control["process"] == PROCESS_QCD).sum()),
            "ttbar": int((control["process"] == PROCESS_TTBAR).sum()),
        },
    }
    if args.synthetic_test_fixture:
        result["_synthetic_test_fixture"] = True

    def auc_subset(model_eval, bg_process=None):
        mask = np.ones(n_events, dtype=bool) if bg_process is None else \
            (model_eval["process"] == PROCESS_SIGNAL) | (model_eval["process"] == bg_process)
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

    # ---- Reconstruction correctness: computed CENTRALLY via the symmetry-invariant
    # match_higgs_pairs(), never from a precomputed per-model boolean.
    # PREREGISTRATION.md Amendment 2026-09-10.
    control_n_correct_higgs, control_exact_correct = match_higgs_pairs(
        control["pred_b1"], control["pred_b2"], control["pred_b3"], control["pred_b4"],
        control["truth_b1"], control["truth_b2"], control["truth_b3"], control["truth_b4"])
    test_n_correct_higgs, test_exact_correct = match_higgs_pairs(
        test["pred_b1"], test["pred_b2"], test["pred_b3"], test["pred_b4"],
        test["truth_b1"], test["truth_b2"], test["truth_b3"], test["truth_b4"])

    result["reconstruction"] = {
        "control": reconstruction_metrics(control, control_n_correct_higgs, control_exact_correct),
        "test": reconstruction_metrics(test, test_n_correct_higgs, test_exact_correct),
    }

    both_defined = control["assignment_defined"] & test["assignment_defined"]
    result["mcnemar_reconstruction_correctness"] = mcnemar_test(
        control_exact_correct[both_defined], test_exact_correct[both_defined])
    result["mcnemar_n_events_both_defined"] = int(both_defined.sum())

    # NOTE (2026-09-10 audit): the pre-amendment version of this stat_fn took a
    # `correct_arr` kwarg that was fixed to TEST's correctness array regardless of
    # whether paired_bootstrap_delta was internally evaluating the test or the
    # control statistic (the function ignored its own `scores` argument) -- so the
    # "delta" was always test_mean - test_mean = 0 with a degenerate zero-width CI,
    # never actually comparing control to test. Fixed here by using `scores`
    # (which paired_bootstrap_delta already passes as test_exact_correct /
    # control_exact_correct respectively) directly, with no separate correct_arr.
    def exact_event_stat_fn(scores, labels, idx, defined_arr=None):
        sub_defined = defined_arr[idx]
        if sub_defined.sum() == 0:
            return 0.0
        return float(scores[idx][sub_defined].mean())

    result["paired_bootstrap_delta_exact_event_efficiency"] = paired_bootstrap_delta(
        test_exact_correct.astype(float), control_exact_correct.astype(float),
        np.zeros(n_events),  # labels unused by this stat_fn; kept for paired_bootstrap_delta's shared signature
        exact_event_stat_fn, n_boot=args.n_boot, seed=args.seed, defined_arr=both_defined)

    result["score_mass_correlation"] = {"control": score_mass_correlation(control), "test": score_mass_correlation(test)}
    result["jet_multiplicity_stratified_auc"] = stratify_by_jet_multiplicity(control, test)

    result["resource"] = {"control": resource_metrics(args.control_training_result), "test": resource_metrics(args.test_training_result)}
    if args.build_receipt:
        with open(args.build_receipt) as f:
            br = json.load(f)
        result["resource"]["parameter_count_increase"] = {
            "n_retained_part_dims": br["n_retained_part_dims"],
            "delta_w_first_layer_scalars": 16 * br["n_retained_part_dims"],
        }

    if args.native10m_aggregate_json:
        auc10m, story_b = evaluate_native10m(test, args.native10m_aggregate_json, args.native10m_npz, args.n_boot, args.seed)
        result["auc"]["native10m"] = auc10m
        story_b_out = story_b
    else:
        story_b_out = {"path": "not_evaluated", "note": "--native10m-aggregate-json not provided"}

    delta_auc = result["paired_bootstrap_delta_auc"]["all_background"]
    delta_reco = result["paired_bootstrap_delta_exact_event_efficiency"]
    mcnemar = result["mcnemar_reconstruction_correctness"]
    sig_class = bool(delta_auc["excludes_zero"] and delta_auc["observed_delta_test_minus_control"] > 0)
    harm_class = bool(delta_auc["excludes_zero"] and delta_auc["observed_delta_test_minus_control"] < 0)
    sig_reco = bool((mcnemar["exact_binomial_p_value"] < 0.05 and mcnemar["test_better_than_control"]) or
                     (delta_reco["excludes_zero"] and delta_reco["observed_delta_test_minus_control"] > 0))
    harm_reco = bool(mcnemar["exact_binomial_p_value"] < 0.05 and not mcnemar["test_better_than_control"])
    ci_half_width = (delta_auc["ci_high"] - delta_auc["ci_low"]) / 2.0
    obs = abs(delta_auc["observed_delta_test_minus_control"])
    underpowered = bool((not sig_class and not harm_class) and (not sig_reco and not harm_reco) and
                         (obs == 0.0 or ci_half_width > 2 * obs))
    story_a_supported = sig_class or sig_reco
    story_c_supported = (not sig_class) and (not sig_reco) and (not underpowered)

    result["story_evaluation"] = {
        "single_seed_flags": {"sig_class": sig_class, "sig_reco": sig_reco, "harm_class": harm_class,
                               "harm_reco": harm_reco, "underpowered": underpowered},
        "story_a": {"supported": story_a_supported, "grade": "suggestive" if story_a_supported else "not_supported"},
        "story_b": story_b_out,
        "story_c": {"supported": story_c_supported},
    }

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"seed": args.seed, "n_events": n_events,
                       "delta_auc_all_background": delta_auc["observed_delta_test_minus_control"],
                       "delta_auc_ci": [delta_auc["ci_low"], delta_auc["ci_high"]],
                       "story_evaluation": result["story_evaluation"]}, indent=2))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
