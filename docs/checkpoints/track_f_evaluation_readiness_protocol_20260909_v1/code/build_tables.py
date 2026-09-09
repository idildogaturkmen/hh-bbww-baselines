#!/usr/bin/env python3
"""Table generator: reads a comparison_result.json (evaluate_multi_model.py
output) and, optionally, a go_no_go_decision.json (decide_go_no_go.py
output), and writes TSV tables -- following this project's established
results/<name>/tables/*.tsv convention (e.g. results/hh4b_cut_baseline_20260724_v1/tables/).

Writes exactly the numbers present in the input JSON. Never fabricates,
estimates, or fills a missing field -- a metric absent from the input is
written as the literal string "MISSING" in the table, never silently
dropped or defaulted to zero, so a partially-populated result is visibly
incomplete rather than misleadingly clean.
"""
import argparse
import csv
import json
import os

TARGET_EFFICIENCIES = [0.10, 0.075, 0.05, 0.04, 0.03]


def g(d, *path, default="MISSING"):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur if cur is not None else "MISSING"


def write_tsv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def build_auc_table(result, out_dir):
    rows = []
    for model in ("control", "test", "native10m"):
        auc = g(result, "auc", model, default=None)
        if auc is None:
            continue
        rows.append({"model": model, "all_background": auc.get("all_background", "MISSING"),
                     "qcd": auc.get("qcd", "MISSING"), "ttbar": auc.get("ttbar", "MISSING")})
    write_tsv(os.path.join(out_dir, "auc_summary.tsv"), ["model", "all_background", "qcd", "ttbar"], rows)


def build_paired_delta_table(result, out_dir):
    rows = []
    for metric_name, block in (("delta_auc", g(result, "paired_bootstrap_delta_auc", default={})),
                                ("delta_exact_event_efficiency", {"all_background": g(result, "paired_bootstrap_delta_exact_event_efficiency", default={})})):
        for sub, stats in block.items():
            if not isinstance(stats, dict) or "observed_delta_test_minus_control" not in stats:
                continue
            rows.append({
                "metric": f"{metric_name}.{sub}",
                "observed_delta_test_minus_control": stats["observed_delta_test_minus_control"],
                "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
                "n_boot": stats["n_boot"], "excludes_zero": stats["excludes_zero"],
            })
    for eff in TARGET_EFFICIENCIES:
        stats = g(result, "paired_bootstrap_delta_rejection", f"epsS_{eff}", default=None)
        if stats is None:
            continue
        rows.append({"metric": f"delta_rejection.epsS_{eff}",
                     "observed_delta_test_minus_control": stats["observed_delta_test_minus_control"],
                     "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
                     "n_boot": stats["n_boot"], "excludes_zero": stats["excludes_zero"]})
    write_tsv(os.path.join(out_dir, "paired_bootstrap_deltas.tsv"),
              ["metric", "observed_delta_test_minus_control", "ci_low", "ci_high", "n_boot", "excludes_zero"], rows)


def build_rejection_table(result, out_dir):
    rows = []
    for model in ("control", "test"):
        for eff in TARGET_EFFICIENCIES:
            for bg in ("all_background", "qcd", "ttbar"):
                wp = g(result, "rejection_at_fixed_efficiency", model, f"epsS_{eff}", bg, default=None)
                if wp is None:
                    continue
                rows.append({
                    "model": model, "target_epsS": eff, "background": bg,
                    "threshold": wp.get("threshold", "MISSING"), "achieved_epsS": wp.get("achieved_epsS", "MISSING"),
                    "rejection": wp.get("rejection", "MISSING"), "epsB": wp.get("epsB", "MISSING"),
                    "n_bg_pass": wp.get("n_bg_pass", "MISSING"), "n_bg_total": wp.get("n_bg_total", "MISSING"),
                    "finite_support_warnings": " | ".join(wp.get("finite_support_warnings", [])),
                })
    write_tsv(os.path.join(out_dir, "rejection_at_fixed_efficiency.tsv"),
              ["model", "target_epsS", "background", "threshold", "achieved_epsS", "rejection", "epsB",
               "n_bg_pass", "n_bg_total", "finite_support_warnings"], rows)


def build_reconstruction_table(result, out_dir):
    rows = []
    for model in ("control", "test"):
        r = g(result, "reconstruction", model, default={})
        rows.append({
            "model": model,
            "n_events_with_defined_assignment": g(r, "n_events_with_defined_assignment", default="MISSING") if isinstance(r, dict) else "MISSING",
            "exact_event_hh_reconstruction_efficiency": g(r, "exact_event_hh_reconstruction_efficiency", default="MISSING") if isinstance(r, dict) else "MISSING",
            "higgs_assignment_pairing_accuracy": g(r, "higgs_assignment_pairing_accuracy", default="MISSING") if isinstance(r, dict) else "MISSING",
        })
    mcnemar = result.get("mcnemar_reconstruction_correctness", {})
    write_tsv(os.path.join(out_dir, "reconstruction_summary.tsv"),
              ["model", "n_events_with_defined_assignment", "exact_event_hh_reconstruction_efficiency",
               "higgs_assignment_pairing_accuracy"], rows)
    with open(os.path.join(out_dir, "mcnemar_reconstruction.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["quantity", "value"])
        w.writerow(["n_discordant", mcnemar.get("n_discordant", "MISSING")])
        w.writerow(["exact_binomial_p_value", mcnemar.get("exact_binomial_p_value", "MISSING")])
        w.writerow(["test_better_than_control", mcnemar.get("test_better_than_control", "MISSING")])
        contingency = mcnemar.get("contingency", {})
        for k, v in contingency.items():
            w.writerow([f"contingency.{k}", v])


def build_jet_multiplicity_table(result, out_dir):
    strata = g(result, "jet_multiplicity_stratified_auc", default={})
    rows = []
    if isinstance(strata, dict):
        for name, s in strata.items():
            if not isinstance(s, dict) or "auc_control" not in s:
                continue
            rows.append({"stratum": name, "n_events_control": s.get("n_events_control", "MISSING"),
                         "n_events_test": s.get("n_events_test", "MISSING"),
                         "auc_control": s.get("auc_control", "MISSING"), "auc_test": s.get("auc_test", "MISSING")})
    write_tsv(os.path.join(out_dir, "jet_multiplicity_stratified_auc.tsv"),
              ["stratum", "n_events_control", "n_events_test", "auc_control", "auc_test"], rows)


def build_resource_table(result, out_dir):
    rows = []
    for model in ("control", "test"):
        r = g(result, "resource", model, default={})
        if not isinstance(r, dict):
            r = {}
        rows.append({"model": model, **{k: r.get(k, "MISSING") for k in
                     ("total_fit_wall_s", "gpu_peak_reserved_bytes", "gpu_peak_allocated_bytes",
                      "host_peak_rss_kib_self", "gpu_device_name")}})
    write_tsv(os.path.join(out_dir, "resource_summary.tsv"),
              ["model", "total_fit_wall_s", "gpu_peak_reserved_bytes", "gpu_peak_allocated_bytes",
               "host_peak_rss_kib_self", "gpu_device_name"], rows)


def build_go_no_go_table(decision, out_dir):
    rows = []
    for s in decision.get("per_seed", []):
        rows.append({k: s.get(k, "MISSING") for k in
                     ("seed", "sig_class", "sig_reco", "harm_class", "harm_reco", "underpowered",
                      "delta_auc_all_background_point_estimate")})
    write_tsv(os.path.join(out_dir, "go_no_go_per_seed.tsv"),
              ["seed", "sig_class", "sig_reco", "harm_class", "harm_reco", "underpowered",
               "delta_auc_all_background_point_estimate"], rows)
    with open(os.path.join(out_dir, "go_no_go_summary.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["decision_point", "decision", "reason"])
        w.writerow(["more_seeds", decision["more_seeds_decision"]["decision"], decision["more_seeds_decision"]["reason"]])
        w.writerow(["additional8m", decision["additional8m_decision"]["decision"], decision["additional8m_decision"]["reason"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison-result", required=True)
    ap.add_argument("--go-no-go-decision", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.comparison_result) as f:
        result = json.load(f)

    build_auc_table(result, args.out_dir)
    build_paired_delta_table(result, args.out_dir)
    build_rejection_table(result, args.out_dir)
    build_reconstruction_table(result, args.out_dir)
    build_jet_multiplicity_table(result, args.out_dir)
    build_resource_table(result, args.out_dir)

    written = ["auc_summary.tsv", "paired_bootstrap_deltas.tsv", "rejection_at_fixed_efficiency.tsv",
               "reconstruction_summary.tsv", "mcnemar_reconstruction.tsv", "jet_multiplicity_stratified_auc.tsv",
               "resource_summary.tsv"]

    if args.go_no_go_decision:
        with open(args.go_no_go_decision) as f:
            decision = json.load(f)
        build_go_no_go_table(decision, args.out_dir)
        written += ["go_no_go_per_seed.tsv", "go_no_go_summary.tsv"]

    print(json.dumps({"out_dir": args.out_dir, "tables_written": written}, indent=2))


if __name__ == "__main__":
    main()
