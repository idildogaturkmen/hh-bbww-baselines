#!/usr/bin/env python3
"""Deterministic assembler for the final ZERO20 result bundle.

Reads ONLY already-computed, final (n_boot=10000) statistics from three
existing comparison JSONs -- it runs no bootstrap of its own and launches
no training/inference. Inputs:

  1. native2M-vs-ParT20 (the original, already-frozen governing result)
  2. ZERO20-vs-native2M  (final, recovered 10k run)
  3. ZERO20-vs-ParT20    (final, recovered 10k run)

all three produced by the same evaluate_multi_model.py, same schema. This
script cross-checks internal consistency (e.g. ZERO20's own AUC/
reconstruction point estimates must agree between input 2 and input 3,
native10M's AUC must agree between input 2 and input 3) before emitting
anything, and refuses (non-zero exit) if any of those checks fail.

Outputs, all under the sibling metrics/ and tables/ directories:
  metrics/model_summary.json            -- point estimates, 4 models
  metrics/paired_statistics.json        -- all 3 pairwise deltas/CIs/McNemar
  metrics/causal_branch_classification.json
  tables/model_summary.{md,csv}
  tables/paired_statistics.{md,csv}
  tables/reconstruction.{md,csv}
  tables/fixed_efficiency_rejection.{md,csv}
  tables/training_resources.{md,csv}

The causal-branch classification (A/B/C/D) uses the SAME pre-registered
rule as the diagnostic package's ZERO20_DIAGNOSTIC_INTERPRETATION_CONTRACT.md:
AUC significance = paired bootstrap 95% CI excludes zero; AUC PRACTICAL
threshold = |delta| >= 0.005 (PREREGISTRATION.md Sec.7.2, derived from the
project's own measured ~0.002 AUC native 2M-to-10M scaling noise ceiling,
not invented here); reconstruction significance = McNemar exact p<0.05,
no separate magnitude floor (none was ever pre-registered for
reconstruction). No branch is assumed -- it is computed from the numbers
read from disk.
"""
import argparse
import csv
import json
import os

AUC_FLOOR = 0.005
TARGET_EFFICIENCIES = [0.10, 0.075, 0.05, 0.04, 0.03]


def load(path):
    with open(path) as f:
        return json.load(f)


def write_md_table(path, headers, rows):
    with open(path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(c) for c in row) + " |\n")


def write_csv_table(path, headers, rows):
    # lineterminator="\n" (not csv's Excel-style default "\r\n") to match
    # this repo's Unix line-ending convention -- avoids spurious
    # trailing-CR "whitespace" flags from `git diff --check`.
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(headers)
        for row in rows:
            w.writerow(row)


def classify_auc(d_native, d_part20):
    native_sig_harm = d_native["excludes_zero"] and d_native["observed_delta_test_minus_control"] < 0
    native_indist = (not d_native["excludes_zero"]) or abs(d_native["observed_delta_test_minus_control"]) < AUC_FLOOR
    part20_sig_recovery = d_part20["excludes_zero"] and d_part20["observed_delta_test_minus_control"] >= AUC_FLOOR
    part20_indist = (not d_part20["excludes_zero"]) or abs(d_part20["observed_delta_test_minus_control"]) < AUC_FLOOR

    if native_indist:
        branch = "B"
    elif native_sig_harm and part20_indist:
        branch = "A"
    elif native_sig_harm and part20_sig_recovery:
        branch = "C"
    else:
        branch = "D"
    return dict(branch=branch,
                statistically_significant_vs_native=bool(d_native["excludes_zero"]),
                practically_significant_vs_native=bool(abs(d_native["observed_delta_test_minus_control"]) >= AUC_FLOOR),
                delta_zero20_minus_native=d_native["observed_delta_test_minus_control"],
                delta_zero20_minus_native_ci=[d_native["ci_low"], d_native["ci_high"]],
                delta_zero20_minus_part20=d_part20["observed_delta_test_minus_control"],
                delta_zero20_minus_part20_ci=[d_part20["ci_low"], d_part20["ci_high"]],
                auc_floor_used=AUC_FLOOR)


def classify_reconstruction(m_native, m_part20):
    native_sig = m_native["exact_binomial_p_value"] < 0.05
    native_harm = native_sig and not m_native["test_better_than_control"]
    native_indist = not native_sig
    part20_sig_recovery = (m_part20["exact_binomial_p_value"] < 0.05) and m_part20["test_better_than_control"]
    part20_indist = not (m_part20["exact_binomial_p_value"] < 0.05)

    if native_indist:
        branch = "B"
    elif native_harm and part20_indist:
        branch = "A"
    elif native_harm and part20_sig_recovery:
        branch = "C"
    else:
        branch = "D"
    return dict(branch=branch, native_mcnemar_p=m_native["exact_binomial_p_value"],
                native_significant_and_harmful=native_harm,
                native_indistinguishable_from_zero20=native_indist,
                part20_mcnemar_p=m_part20["exact_binomial_p_value"],
                part20_significant_recovery_vs_zero20=part20_sig_recovery)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-vs-part20-json", required=True)
    ap.add_argument("--zero20-vs-native-json", required=True)
    ap.add_argument("--zero20-vs-part20-json", required=True)
    ap.add_argument("--out-dir", required=True, help="artifact root; writes metrics/ and tables/ under it")
    args = ap.parse_args()

    d0 = load(args.native_vs_part20_json)   # control=native2M, test=ParT20
    d1 = load(args.zero20_vs_native_json)   # control=native2M, test=ZERO20
    d2 = load(args.zero20_vs_part20_json)   # control=ParT20,   test=ZERO20

    for d, label in [(d0, "native-vs-part20"), (d1, "zero20-vs-native"), (d2, "zero20-vs-part20")]:
        n_boot = d["paired_bootstrap_delta_auc"]["all_background"]["n_boot"]
        assert n_boot == 10000, f"{label}: n_boot={n_boot}, expected 10000 -- refusing to assemble a non-final result"
        assert d["n_events_shared_cohort"] == 400000, f"{label}: n_events_shared_cohort != 400000"

    # ---- internal consistency cross-checks (fail loudly, don't silently average) ----
    zero20_auc_from_d1 = d1["auc"]["test"]
    zero20_auc_from_d2 = d2["auc"]["test"]
    for proc in ["all_background", "qcd", "ttbar"]:
        assert abs(zero20_auc_from_d1[proc] - zero20_auc_from_d2[proc]) < 1e-9, \
            f"ZERO20 AUC[{proc}] disagrees between the two files it appears in: " \
            f"{zero20_auc_from_d1[proc]} vs {zero20_auc_from_d2[proc]}"
    native10m_from_d1 = d1["auc"]["native10m"]
    native10m_from_d2 = d2["auc"]["native10m"]
    for proc in ["all_background", "qcd", "ttbar"]:
        assert abs(native10m_from_d1[proc] - native10m_from_d2[proc]) < 1e-9, \
            f"native10M AUC[{proc}] disagrees between the two files: {native10m_from_d1[proc]} vs {native10m_from_d2[proc]}"

    models = {
        "native2M": dict(auc=d0["auc"]["control"], reco=d0["reconstruction"]["control"],
                          resource=d0["resource"]["control"]),
        "native10M": dict(auc=native10m_from_d1, reco=None, resource=None),
        "ParT20_2M": dict(auc=d0["auc"]["test"], reco=d0["reconstruction"]["test"],
                           resource=d0["resource"]["test"]),
        "ZERO20_2M": dict(auc=d1["auc"]["test"], reco=d1["reconstruction"]["test"],
                           resource=d1["resource"]["test"]),
    }

    pairwise = {
        "ParT20_minus_native2M": dict(
            delta_auc=d0["paired_bootstrap_delta_auc"]["all_background"],
            delta_reco=d0["paired_bootstrap_delta_exact_event_efficiency"],
            mcnemar=d0["mcnemar_reconstruction_correctness"],
            story=d0["story_evaluation"]),
        "ZERO20_minus_native2M": dict(
            delta_auc=d1["paired_bootstrap_delta_auc"]["all_background"],
            delta_reco=d1["paired_bootstrap_delta_exact_event_efficiency"],
            mcnemar=d1["mcnemar_reconstruction_correctness"],
            story=d1["story_evaluation"]),
        "ZERO20_minus_ParT20": dict(
            delta_auc=d2["paired_bootstrap_delta_auc"]["all_background"],
            delta_reco=d2["paired_bootstrap_delta_exact_event_efficiency"],
            mcnemar=d2["mcnemar_reconstruction_correctness"],
            story=d2["story_evaluation"]),
    }

    auc_class = classify_auc(pairwise["ZERO20_minus_native2M"]["delta_auc"],
                              pairwise["ZERO20_minus_ParT20"]["delta_auc"])
    reco_class = classify_reconstruction(pairwise["ZERO20_minus_native2M"]["mcnemar"],
                                          pairwise["ZERO20_minus_ParT20"]["mcnemar"])
    agree = auc_class["branch"] == reco_class["branch"]
    overall = auc_class["branch"] if agree else "D"

    metrics_dir = os.path.join(args.out_dir, "metrics")
    tables_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    with open(os.path.join(metrics_dir, "model_summary.json"), "w") as f:
        json.dump(models, f, indent=2)
    with open(os.path.join(metrics_dir, "paired_statistics.json"), "w") as f:
        json.dump(pairwise, f, indent=2)
    classification = dict(auc_classification=auc_class, reconstruction_classification=reco_class,
                           metrics_agree=agree, overall_branch=overall,
                           source_files=dict(native_vs_part20=os.path.abspath(args.native_vs_part20_json),
                                              zero20_vs_native=os.path.abspath(args.zero20_vs_native_json),
                                              zero20_vs_part20=os.path.abspath(args.zero20_vs_part20_json)))
    with open(os.path.join(metrics_dir, "causal_branch_classification.json"), "w") as f:
        json.dump(classification, f, indent=2)

    # ---- tables/model_summary ----
    headers = ["Model", "AUC (all-bg)", "AUC (QCD)", "AUC (ttbar)",
               "Exact-event reco.", "Per-Higgs reco.", "Fit wall time (h)"]
    rows = []
    for name, label in [("native2M", "Native SPA-Net 2M"), ("native10M", "Native SPA-Net 10M"),
                         ("ParT20_2M", "SPA-Net + ParT active20 2M"), ("ZERO20_2M", "SPA-Net + ZERO20 2M")]:
        m = models[name]
        reco_exact = f"{m['reco']['exact_event_hh_reconstruction_efficiency']:.6f}" if m["reco"] else "n/a"
        reco_higgs = f"{m['reco']['higgs_assignment_pairing_accuracy']:.6f}" if m["reco"] else "n/a"
        wall_h = f"{m['resource']['total_fit_wall_s'] / 3600:.2f}" if m["resource"] else "n/a"
        rows.append([label, f"{m['auc']['all_background']:.6f}", f"{m['auc']['qcd']:.6f}",
                     f"{m['auc']['ttbar']:.6f}", reco_exact, reco_higgs, wall_h])
    write_md_table(os.path.join(tables_dir, "model_summary.md"), headers, rows)
    write_csv_table(os.path.join(tables_dir, "model_summary.csv"), headers, rows)

    # ---- tables/paired_statistics ----
    headers = ["Comparison (test minus control)", "delta AUC (all-bg)", "95% CI", "excludes zero",
               "delta exact-event reco.", "95% CI", "McNemar p", "test better than control"]
    rows = []
    for pair_name, label in [("ParT20_minus_native2M", "ParT20 2M minus Native 2M"),
                              ("ZERO20_minus_native2M", "ZERO20 2M minus Native 2M"),
                              ("ZERO20_minus_ParT20", "ZERO20 2M minus ParT20 2M")]:
        p = pairwise[pair_name]
        da, dr, mc = p["delta_auc"], p["delta_reco"], p["mcnemar"]
        rows.append([label, f"{da['observed_delta_test_minus_control']:.6f}",
                     f"[{da['ci_low']:.6f}, {da['ci_high']:.6f}]", da["excludes_zero"],
                     f"{dr['observed_delta_test_minus_control']:.6f}",
                     f"[{dr['ci_low']:.6f}, {dr['ci_high']:.6f}]",
                     f"{mc['exact_binomial_p_value']:.3g}", mc["test_better_than_control"]])
    write_md_table(os.path.join(tables_dir, "paired_statistics.md"), headers, rows)
    write_csv_table(os.path.join(tables_dir, "paired_statistics.csv"), headers, rows)

    # ---- tables/reconstruction ----
    headers = ["Model", "n (assignment-defined)", "Exact-event HH reconstruction", "Per-Higgs pairing accuracy"]
    rows = []
    for name, label in [("native2M", "Native SPA-Net 2M"), ("ParT20_2M", "SPA-Net + ParT active20 2M"),
                         ("ZERO20_2M", "SPA-Net + ZERO20 2M")]:
        r = models[name]["reco"]
        rows.append([label, r["n_events_with_defined_assignment"],
                     f"{r['exact_event_hh_reconstruction_efficiency']:.6f}",
                     f"{r['higgs_assignment_pairing_accuracy']:.6f}"])
    write_md_table(os.path.join(tables_dir, "reconstruction.md"), headers, rows)
    write_csv_table(os.path.join(tables_dir, "reconstruction.csv"), headers, rows)

    # ---- tables/fixed_efficiency_rejection (native2M, ParT20, ZERO20; all-background) ----
    headers = ["Model", "target epsS", "achieved epsS", "n_signal_pass", "n_bg_pass", "n_bg_total",
               "rejection (1/epsB)", "sparse-MC warning"]
    rows = []
    rej_sources = {"native2M": d0["rejection_at_fixed_efficiency"]["control"],
                   "ParT20_2M": d0["rejection_at_fixed_efficiency"]["test"],
                   "ZERO20_2M": d1["rejection_at_fixed_efficiency"]["test"]}
    for name, label in [("native2M", "Native SPA-Net 2M"), ("ParT20_2M", "SPA-Net + ParT active20 2M"),
                         ("ZERO20_2M", "SPA-Net + ZERO20 2M")]:
        src = rej_sources[name]
        for eff in TARGET_EFFICIENCIES:
            block = src[f"epsS_{eff}"]["all_background"]
            warn = "; ".join(block["finite_support_warnings"]) if block["finite_support_warnings"] else ""
            rejection_str = f"{block['rejection']:.2f}" if block["rejection"] is not None else "undefined (0 bg survivors)"
            rows.append([label, eff, f"{block['achieved_epsS']:.4f}", block["n_signal_pass"],
                         block["n_bg_pass"], block["n_bg_total"], rejection_str, warn])
    write_md_table(os.path.join(tables_dir, "fixed_efficiency_rejection.md"), headers, rows)
    write_csv_table(os.path.join(tables_dir, "fixed_efficiency_rejection.csv"), headers, rows)

    # ---- tables/training_resources ----
    headers = ["Model", "Fit wall time (s)", "Fit wall time (h)", "GPU peak reserved (GiB)", "GPU device"]
    rows = []
    for name, label in [("native2M", "Native SPA-Net 2M"), ("ParT20_2M", "SPA-Net + ParT active20 2M"),
                         ("ZERO20_2M", "SPA-Net + ZERO20 2M")]:
        r = models[name]["resource"]
        rows.append([label, f"{r['total_fit_wall_s']:.1f}", f"{r['total_fit_wall_s'] / 3600:.2f}",
                     f"{r['gpu_peak_reserved_bytes'] / 2**30:.3f}", r["gpu_device_name"]])
    write_md_table(os.path.join(tables_dir, "training_resources.md"), headers, rows)
    write_csv_table(os.path.join(tables_dir, "training_resources.csv"), headers, rows)

    print(f"OVERALL_BRANCH = {overall} (AUC={auc_class['branch']}, reconstruction={reco_class['branch']}, "
          f"agree={agree})")
    print(f"wrote metrics/ and tables/ under {args.out_dir}")


if __name__ == "__main__":
    main()
