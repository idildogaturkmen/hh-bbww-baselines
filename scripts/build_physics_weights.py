#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def as_float_or_nan(x):
    try:
        if pd.isna(x) or x == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skim",
        default="outputs/all_background_reco_skim/all_processes_reco_skim.parquet",
    )
    parser.add_argument(
        "--classifier-input",
        default="outputs/classifier_input_lepemu/onelep_pre_mbb_classifier_input.parquet",
    )
    parser.add_argument(
        "--metadata",
        default="config/collide_sample_metadata.csv",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/physics_weights_lepemu",
    )
    parser.add_argument("--lumi-fb", type=float, default=138.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    skim = pd.read_parquet(args.skim, columns=["sample", "process_group"])
    clf = pd.read_parquet(args.classifier_input)
    meta = pd.read_csv(args.metadata)

    # Local generated/processed proxy counts from the full skim, not the selected classifier table.
    local_counts = (
        skim.groupby(["sample", "process_group"])
        .size()
        .reset_index(name="n_local_skim_events")
        .sort_values(["process_group", "sample"])
    )

    meta["xsec_fb"] = meta["xsec_fb"].apply(as_float_or_nan)
    meta["filter_eff"] = meta["filter_eff"].apply(as_float_or_nan).fillna(1.0)
    meta["k_factor"] = meta["k_factor"].apply(as_float_or_nan).fillna(1.0)

    report = local_counts.merge(
        meta,
        on=["sample", "process_group"],
        how="left",
    )

    report["has_xsec"] = np.isfinite(report["xsec_fb"])
    report["expected_events_before_selection"] = (
        report["xsec_fb"]
        * args.lumi_fb
        * report["filter_eff"]
        * report["k_factor"]
    )

    report["physics_weight_nominal"] = np.where(
        report["has_xsec"],
        report["expected_events_before_selection"] / report["n_local_skim_events"],
        np.nan,
    )

    report["status"] = report["status"].fillna("missing_xsec")
    report["source"] = report["source"].fillna("")
    report["note"] = report["note"].fillna("")

    report.to_csv(outdir / "sample_normalization_report.csv", index=False)

    missing = report[~report["has_xsec"]].copy()
    missing.to_csv(outdir / "missing_xsec_report.csv", index=False)

    # Add available weights to classifier table.
    weight_cols = [
        "sample",
        "xsec_fb",
        "filter_eff",
        "k_factor",
        "status",
        "source",
        "n_local_skim_events",
        "expected_events_before_selection",
        "physics_weight_nominal",
    ]
    clf_w = clf.merge(report[weight_cols], on="sample", how="left")

    clf_w["has_physics_weight"] = np.isfinite(clf_w["physics_weight_nominal"])
    clf_w.to_parquet(outdir / "onelep_pre_mbb_classifier_input_with_weights.parquet", index=False)
    clf_w.head(500).to_csv(outdir / "weighted_classifier_input_preview.csv", index=False)

    # Raw and available-weight yields by process.
    raw_yields = (
        clf_w.groupby("process_group")
        .size()
        .reset_index(name="raw_events_in_classifier_preselection")
    )

    weighted_available = (
        clf_w[clf_w["has_physics_weight"]]
        .groupby("process_group")["physics_weight_nominal"]
        .sum()
        .reset_index(name="weighted_events_available_xsec")
    )

    yield_report = raw_yields.merge(weighted_available, on="process_group", how="left")
    yield_report["weighted_events_available_xsec"] = yield_report["weighted_events_available_xsec"].fillna(0.0)
    yield_report.to_csv(outdir / "preselection_yields_raw_and_available_weighted.csv", index=False)

    # Signal-only weighted cut baseline, since backgrounds are not filled yet.
    cut = (
        (clf_w["ht"] >= 100)
        & (clf_w["mbb_pt_binned_scaled"] >= 80)
        & (clf_w["mbb_pt_binned_scaled"] <= 150)
    )

    rows = []
    for label, mask in [
        ("onelep_premass", np.ones(len(clf_w), dtype=bool)),
        ("onelep_ht100_mbbcorr_80_150", cut),
    ]:
        sub = clf_w[mask]
        sig = sub[sub["process_group"] == "signal"]
        bkg = sub[sub["process_group"] != "signal"]

        rows.append(
            {
                "selection": label,
                "raw_signal": int(len(sig)),
                "raw_background": int(len(bkg)),
                "weighted_signal_available_xsec": float(sig["physics_weight_nominal"].sum()),
                "weighted_background_available_xsec": float(
                    bkg.loc[bkg["has_physics_weight"], "physics_weight_nominal"].sum()
                ),
                "raw_background_with_missing_xsec": int((~bkg["has_physics_weight"]).sum()),
                "is_full_physics_normalization_complete": bool(
                    bkg["has_physics_weight"].all() and sig["has_physics_weight"].all()
                ),
            }
        )

    cut_report = pd.DataFrame(rows)
    cut_report.to_csv(outdir / "cut_baseline_signal_weight_report.csv", index=False)

    print("Wrote outputs to:", outdir)
    print("\n=== Sample normalization report ===")
    print(report[[
        "sample",
        "process_group",
        "n_local_skim_events",
        "xsec_fb",
        "expected_events_before_selection",
        "physics_weight_nominal",
        "status",
    ]].to_string(index=False))

    print("\n=== Missing xsec samples ===")
    print(missing[["sample", "process_group", "n_local_skim_events"]].to_string(index=False))

    print("\n=== Preselection yields ===")
    print(yield_report.to_string(index=False))

    print("\n=== Signal weighted cut baseline report ===")
    print(cut_report.to_string(index=False))


if __name__ == "__main__":
    main()
