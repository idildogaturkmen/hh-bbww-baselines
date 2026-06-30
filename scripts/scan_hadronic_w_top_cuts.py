"""
Scan hadronic-W and top-veto variables after rough normalization.

This script is an evaluation/diagnostic script, not a feature-building script.
It filters to samples with assigned rough physics weights before computing
weighted yields, S/B, Z proxies, and effective background statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def asimov_z(s: float, b: float) -> float:
    if s <= 0 or b <= 0:
        return 0.0
    return float(np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s)))


def neff(w) -> float:
    w = np.asarray(w, dtype=float)
    if len(w) == 0 or np.sum(w * w) <= 0:
        return 0.0
    return float(np.sum(w) ** 2 / np.sum(w * w))


def weighted_yield(df: pd.DataFrame, mask) -> float:
    return float(df.loc[mask, "physics_weight_nominal"].sum())


def summarize_region(df: pd.DataFrame, mask, label: dict) -> dict:
    sub = df[mask].copy()
    sig = sub[sub["target"] == 1]
    bkg = sub[sub["target"] == 0]

    s = float(sig["physics_weight_nominal"].sum())
    b = float(bkg["physics_weight_nominal"].sum())

    row = dict(label)
    row.update(
        {
            "S_raw": int(len(sig)),
            "B_raw": int(len(bkg)),
            "S_weighted": s,
            "B_weighted": b,
            "S_over_B_weighted": s / b if b > 0 else np.nan,
            "S_over_SplusB_weighted": s / (s + b) if (s + b) > 0 else np.nan,
            "asimovZ_weighted": asimov_z(s, b),
            "background_neff": neff(bkg["physics_weight_nominal"]),
            "ttbar_weighted": weighted_yield(sub, sub["process_group"] == "ttbar"),
            "WW_weighted": weighted_yield(
                sub, sub["sample"].astype(str).str.contains("WW_", na=False)
            ),
            "ZJetsTobb_weighted": weighted_yield(
                sub, sub["sample"].astype(str).str.contains("ZJetsTobb", na=False)
            ),
            "WJets_weighted": weighted_yield(
                sub, sub["process_group"].astype(str).str.lower().eq("wjets")
            ),
            "DY_Zjets_weighted": weighted_yield(
                sub, sub["process_group"].astype(str).str.lower().eq("dy_zjets")
            ),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/hadronic_w_top_features_lepemu_rough/onelep_pre_mbb_classifier_input_with_hadronic_w_top_features.parquet",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/hadronic_w_top_features_lepemu_rough",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)

    n_before = len(df)

    # Important: only use rows with assigned rough physics weights
    # for yield/significance/neff scans.
    df = df[
        df["has_physics_weight"]
        & np.isfinite(df["physics_weight_nominal"])
    ].copy()

    n_after = len(df)

    print(f"Input rows: {n_before}")
    print(f"Rows with rough physics weights: {n_after}")
    print(f"Excluded rows without safe rough weights: {n_before - n_after}")

    base = (
        (df["ht"] >= 100)
        & (df["mbb_pt_binned_scaled"] >= 80)
        & (df["mbb_pt_binned_scaled"] <= 150)
        & (df["mt_lep_met"] >= 30)
    )

    rows = []

    # Scan loose W/top requirements.
    n_nonb_options = [0, 1, 2]
    w_window_options = [None, 80, 60, 50, 40, 30, 20]
    top_veto_options = [None, 20, 30, 40, 50, 60, 80]

    for n_nonb_min in n_nonb_options:
        for w_window in w_window_options:
            for top_veto in top_veto_options:
                mask = base.copy()

                mask = mask & (df["n_nonb_jets"] >= n_nonb_min)

                if w_window is not None:
                    mask = mask & (df["abs_mjj_minus_W"] <= float(w_window))

                if top_veto is not None:
                    # Keep events whose best bjj candidate is NOT close to top mass.
                    # NaN means no top candidate could be formed, so keep it.
                    mask = mask & (
                        df["min_abs_m_bjj_minus_top"].isna()
                        | (df["min_abs_m_bjj_minus_top"] >= float(top_veto))
                    )

                label = {
                    "scan": "w_window_top_veto",
                    "btag_region": "inclusive_btags",
                    "n_nonb_min": n_nonb_min,
                    "w_window_abs_mjj_minus_W": "none" if w_window is None else w_window,
                    "top_veto_min_abs_m_bjj_minus_top": "none" if top_veto is None else top_veto,
                }
                rows.append(summarize_region(df, mask, label))

    # Also scan b-tag regions, because ttbar often has extra b-like activity.
    btag_regions = {
        "ge2_btags": df["n_btagged_jets"] >= 2,
        "exactly2_btags": df["n_btagged_jets"] == 2,
        "ge3_btags": df["n_btagged_jets"] >= 3,
    }

    for btag_name, btag_mask in btag_regions.items():
        for n_nonb_min in n_nonb_options:
            for top_veto in top_veto_options:
                mask = base & btag_mask & (df["n_nonb_jets"] >= n_nonb_min)

                if top_veto is not None:
                    mask = mask & (
                        df["min_abs_m_bjj_minus_top"].isna()
                        | (df["min_abs_m_bjj_minus_top"] >= float(top_veto))
                    )

                label = {
                    "scan": "btag_top_veto",
                    "btag_region": btag_name,
                    "n_nonb_min": n_nonb_min,
                    "w_window_abs_mjj_minus_W": "none",
                    "top_veto_min_abs_m_bjj_minus_top": "none" if top_veto is None else top_veto,
                }
                rows.append(summarize_region(df, mask, label))

    out = pd.DataFrame(rows)

    out = out.sort_values(
        ["asimovZ_weighted", "S_over_B_weighted", "background_neff"],
        ascending=[False, False, False],
    )

    out_path = outdir / "hadronic_w_top_cut_scan_weighted_filtered.csv"
    out.to_csv(out_path, index=False)

    print("\n=== Top rows by rough-weighted Asimov-Z ===")
    print(out.head(40).to_string(index=False))

    print("\n=== Top rows with background_neff >= 50 ===")
    stable = out[out["background_neff"] >= 50].copy()
    if len(stable) == 0:
        print("No rows pass background_neff >= 50.")
    else:
        print(stable.head(40).to_string(index=False))

    print("\n=== Top rows by S/B with background_neff >= 50 ===")
    if len(stable) == 0:
        print("No rows pass background_neff >= 50.")
    else:
        print(
            stable.sort_values(
                ["S_over_B_weighted", "asimovZ_weighted"],
                ascending=[False, False],
            )
            .head(40)
            .to_string(index=False)
        )

    print("\nWrote:", out_path)


if __name__ == "__main__":
    main()