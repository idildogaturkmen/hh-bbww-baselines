#!/usr/bin/env python3
"""Task 6: Harvey's fifth-jet question. For events with >=5 selected
jets, study (pT5, minDeltaR5) vs: SPA-Net assignment (QCD only, genuine),
truth-matching (signal only, this analysis's own DeltaR<0.4 convention),
u-distribution, and fraction crossing u>3.5/4.5.

Distinguishes OBSERVATIONAL ASSOCIATION (what this script computes) from
a genuine COUNTERFACTUAL PERTURBATION test (masking j5 and re-running
SPA-Net on the reduced 4-jet input) -- the latter is NOT performed here,
only specified, per instruction."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1/work")

PKG = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1"


def main():
    master = pd.read_parquet(f"{PKG}/HARVEY_MASTER_EVENT_TABLE.parquet")
    has5 = master[master["has_5th_jet"] == True].copy()  # noqa: E712
    sig5 = has5[has5["label"] == 1]
    qcd5 = has5[has5["background_class"] == "QCD"]
    ttbar5 = has5[has5["background_class"] == "ttbar"]

    summary = {}

    # ---- signal: truth-matching, u distribution, u>3.5/4.5 fractions ----
    summary["signal"] = dict(
        n_with_5th_jet_u_gt_3p5=int(len(sig5)),
        frac_j5_truth_matched_to_HH_b=float(sig5["j5_is_truth_matched_to_HH_b"].mean()),
        frac_truth_all4_in_leading4_given_has5=float(sig5["truth_all4_in_leading4"].mean()),
        median_pt5=float(sig5["pt5"].median()), median_min_dR5=float(sig5["min_dR_j5_leading4"].median()),
        median_u=float(sig5["u"].median()),
        n_u_gt_4p5=int((sig5["u"] > 4.5).sum()), frac_u_gt_4p5_given_has5_and_u_gt_3p5=float((sig5["u"] > 4.5).mean()),
    )
    # compare to signal WITHOUT a 5th jet (n_selected_jets==4), same u>3.5 population
    sig4 = master[(master["label"] == 1) & (master["has_5th_jet"] == False)]  # noqa: E712
    summary["signal"]["n_without_5th_jet"] = int(len(sig4))
    summary["signal"]["median_u_without_5th_jet"] = float(sig4["u"].median())
    summary["signal"]["frac_u_gt_4p5_without_5th_jet"] = float((sig4["u"] > 4.5).mean())

    # ---- QCD: genuine SPA-Net assignment probability, correctness proxy ----
    if len(qcd5):
        summary["QCD"] = dict(
            n_with_5th_jet=int(len(qcd5)),
            frac_j5_in_genuine_assignment=float(qcd5["j5_in_assignment"].mean()),
            frac_assignment_eq_leading4=float(qcd5["assignment_eq_leading4"].mean()),
            median_pt5=float(qcd5["pt5"].median()), median_min_dR5=float(qcd5["min_dR_j5_leading4"].median()),
            median_u=float(qcd5["u"].median()),
            note="assignment correctness relative to TRUTH is not answerable for QCD (no truth Higgs to match) -- only genuine-model-assignment vs geometric-leading-4 agreement is reported",
        )
    if len(ttbar5):
        summary["ttbar"] = dict(
            n_with_5th_jet=int(len(ttbar5)), median_pt5=float(ttbar5["pt5"].median()),
            median_min_dR5=float(ttbar5["min_dR_j5_leading4"].median()),
            note="no genuine SPA-Net assignment or truth-HH-matching available for ttbar -- descriptive kinematics only",
        )

    # ---- 2D bin table (pT5, minDeltaR5) for signal: mean u, frac truth-matched, frac u>3.5/4.5 per bin ----
    pt5_bins = [0, 40, 60, 80, 120, 200, np.inf]
    dr_bins = [0, 0.6, 0.8, 1.0, 1.3, 1.6, np.inf]
    sig5 = sig5.copy()
    sig5["pt5_bin"] = pd.cut(sig5["pt5"], pt5_bins)
    sig5["dr_bin"] = pd.cut(sig5["min_dR_j5_leading4"], dr_bins)
    heat_truth = sig5.groupby(["pt5_bin", "dr_bin"], observed=False)["j5_is_truth_matched_to_HH_b"].agg(["mean", "count"])
    heat_u45 = sig5.groupby(["pt5_bin", "dr_bin"], observed=False).apply(lambda d: (d["u"] > 4.5).mean() if len(d) else np.nan)

    heat_truth.to_csv(f"{PKG}/work/fifth_jet_signal_heatmap_truth.csv")
    heat_u45.to_csv(f"{PKG}/work/fifth_jet_signal_heatmap_u4p5frac.csv")

    if len(qcd5):
        qcd5 = qcd5.copy()
        qcd5["pt5_bin"] = pd.cut(qcd5["pt5"], pt5_bins)
        qcd5["dr_bin"] = pd.cut(qcd5["min_dR_j5_leading4"], dr_bins)
        heat_qcd_assign = qcd5.groupby(["pt5_bin", "dr_bin"], observed=False)["j5_in_assignment"].agg(["mean", "count"])
        heat_qcd_assign.to_csv(f"{PKG}/work/fifth_jet_qcd_heatmap_assignment.csv")

    import json
    json.dump(summary, open(f"{PKG}/work/fifth_jet_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
