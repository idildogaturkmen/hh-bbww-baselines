'''
Diagnose the m_bb shapes in the baseline region, and produce a CSV summary of metrics.
Baseline: m_bb between 90 and 140 GeV, HT >= 100 GeV, at least 3 jets, at most 5 jets, second-highest b-tag flag is 1 (requiring at least two b-tagged selected jets), and passing the 1-lepton selection.

'''
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def width68(x: np.ndarray) -> tuple[float, float, float]:
    q16, q84 = np.percentile(x, [16, 84])
    return q16, q84, 0.5 * (q84 - q16)


def summarize(group_name: str, x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "group": group_name,
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "q16": np.nan,
            "q84": np.nan,
            "width68": np.nan,
            "width68_over_median": np.nan,
            "frac_90_140": np.nan,
            "frac_100_150": np.nan,
            "frac_gt_160": np.nan,
        }

    q16, q84, w68 = width68(x)
    med = np.median(x)

    return {
        "group": group_name,
        "n": len(x),
        "mean": np.mean(x),
        "median": med,
        "q16": q16,
        "q84": q84,
        "width68": w68,
        "width68_over_median": w68 / med if med > 0 else np.nan,
        "frac_90_140": np.mean((x >= 90) & (x <= 140)),
        "frac_100_150": np.mean((x >= 100) & (x <= 150)),
        "frac_gt_160": np.mean(x > 160),
    }


def summarize_scaled(group_name: str, x: np.ndarray, scale_mode: str) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        row = summarize(group_name, x)
        row["scale_mode"] = scale_mode
        row["scale_factor"] = np.nan
        return row

    if scale_mode == "median_to_125":
        center = np.median(x)
    elif scale_mode == "mean_to_125":
        center = np.mean(x)
    else:
        raise ValueError(scale_mode)

    scale = 125.0 / center if center > 0 else np.nan
    row = summarize(group_name, x * scale)
    row["scale_mode"] = scale_mode
    row["scale_factor"] = scale
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skim",
        default="outputs/all_background_reco_skim/all_processes_reco_skim.parquet",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/mbb_shape_baseline_diagnostics",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.skim)

    # Pre-mass candidate region: good for studying the full m_bb shape.
    pre = df[
        df["pass_1lep_ge3jets"].astype(bool)
        & (df["n_jets"] >= 3)
        & (df["n_jets"] <= 5)
        & (df["b2_btag"] >= 1)
    ].copy()

    # Baseline region: after applying Higgs-like mass window.
    base = pre[
        (pre["mbb_top2_btag"] >= 90)
        & (pre["mbb_top2_btag"] <= 140)
        & (pre["ht"] >= 100)
    ].copy()

    rows = []
    scaled_rows = []

    for region_name, sub in [
        ("onelep_3to5_b2eq1_pre_mbb", pre),
        ("onelep_baseline_mbb90_140_ht100", base),
    ]:
        for group, g in sub.groupby("process_group"):
            x = g["mbb_top2_btag"].to_numpy(dtype=float)
            row = summarize(group, x)
            row["region"] = region_name
            rows.append(row)

            for mode in ["median_to_125", "mean_to_125"]:
                srow = summarize_scaled(group, x, mode)
                srow["region"] = region_name
                scaled_rows.append(srow)

        # Combined background row.
        bkg = sub[sub["process_group"] != "signal"]
        x_bkg = bkg["mbb_top2_btag"].to_numpy(dtype=float)
        row = summarize("all_background", x_bkg)
        row["region"] = region_name
        rows.append(row)

        for mode in ["median_to_125", "mean_to_125"]:
            srow = summarize_scaled("all_background", x_bkg, mode)
            srow["region"] = region_name
            scaled_rows.append(srow)

    pd.DataFrame(rows).to_csv(outdir / "mbb_shape_metrics_by_process.csv", index=False)
    pd.DataFrame(scaled_rows).to_csv(outdir / "mbb_shape_metrics_scaled_to_125.csv", index=False)

    print("Wrote:")
    print(outdir / "mbb_shape_metrics_by_process.csv")
    print(outdir / "mbb_shape_metrics_scaled_to_125.csv")

    print("\nKey unscaled rows:")
    key = pd.DataFrame(rows)
    print(
        key[
            key["group"].isin(["signal", "all_background", "ttbar", "ttV_ttH_tttt", "diboson", "single_higgs", "HH_other"])
        ].sort_values(["region", "group"]).to_string(index=False)
    )


if __name__ == "__main__":
    main()