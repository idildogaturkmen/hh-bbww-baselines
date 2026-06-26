'''
Scan signal and background processes for various cut combinations, and rank the cuts by Asimov Z ( Poisson statistical distributions), S/B (Signal-to-background ratio, purity), and S/sqrt(S+B) (naive significance estimator).
'''
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = {
    "inclusive_ge2jets": "pass_ge2jets",
    "onelep_ge3jets": "pass_1lep_ge3jets",
    "twolep_ge2jets": "pass_2lep_ge2jets",
}

MBB_WINDOWS = [
    (70, 160),
    (80, 150),
    (90, 140),
    (95, 145),
    (100, 150),
    (100, 140),
    (105, 145),
    (110, 140),
]

N_JETS_REGIONS = [
    ("any", None, None),
    ("3to5", 3, 5),
    ("3to6", 3, 6),
    ("4to6", 4, 6),
    ("ge3", 3, None),
    ("ge4", 4, None),
    ("ge5", 5, None),
]

B2_BTAG_CUTS = [
    ("any", None),
    ("b2eq1", 1.0),
]

MET_CUTS = [0, 20, 40, 60, 80, 100, 150]
HT_CUTS = [0, 100, 200, 300, 400, 500, 600]


def asimov_z(s: float, b: float) -> float:
    if s <= 0:
        return 0.0
    if b <= 0:
        return math.sqrt(2.0 * s)
    return math.sqrt(2.0 * ((s + b) * math.log(1.0 + s / b) - s))


def get_weight(df: pd.DataFrame) -> pd.Series:
    if "weight" in df.columns:
        return df["weight"].astype(float)
    return pd.Series(np.ones(len(df)), index=df.index)


def apply_njets_region(df: pd.DataFrame, name: str, lo, hi):
    if name == "any":
        return pd.Series(True, index=df.index)
    if hi is None:
        return df["n_jets"] >= lo
    return (df["n_jets"] >= lo) & (df["n_jets"] <= hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skim",
        default="outputs/all_background_reco_skim/all_processes_reco_skim.parquet",
    )
    parser.add_argument("--outdir", default="outputs/signal_vs_background_scan")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.skim)
    df = df.copy()
    df["weight_for_scan"] = get_weight(df)

    rows = []

    for channel, channel_col in CHANNELS.items():
        channel_mask = df[channel_col].astype(bool)

        for mbb_low, mbb_high in MBB_WINDOWS:
            mbb_mask = df["mbb_top2_btag"].between(mbb_low, mbb_high)

            for nj_name, nj_lo, nj_hi in N_JETS_REGIONS:
                nj_mask = apply_njets_region(df, nj_name, nj_lo, nj_hi)

                for b2_name, b2_value in B2_BTAG_CUTS:
                    if b2_value is None:
                        b2_mask = pd.Series(True, index=df.index)
                    else:
                        b2_mask = df["b2_btag"] >= b2_value

                    for met_min in MET_CUTS:
                        met_mask = df["met"].fillna(-999.0) >= met_min

                        for ht_min in HT_CUTS:
                            ht_mask = df["ht"].fillna(-999.0) >= ht_min

                            mask = (
                                channel_mask
                                & mbb_mask
                                & nj_mask
                                & b2_mask
                                & met_mask
                                & ht_mask
                            )

                            sub = df[mask]
                            sig = sub[sub["process_group"] == "signal"]
                            bkg = sub[sub["process_group"] != "signal"]

                            s = float(sig["weight_for_scan"].sum())
                            b = float(bkg["weight_for_scan"].sum())

                            rows.append(
                                {
                                    "channel": channel,
                                    "mbb_low": mbb_low,
                                    "mbb_high": mbb_high,
                                    "n_jets_region": nj_name,
                                    "b2_btag_cut": b2_name,
                                    "met_min": met_min,
                                    "ht_min": ht_min,
                                    "S_raw": s,
                                    "B_raw": b,
                                    "S_over_B_raw": s / b if b > 0 else np.nan,
                                    "S_over_sqrt_SplusB_raw": s / math.sqrt(s + b)
                                    if (s + b) > 0
                                    else 0.0,
                                    "asimov_Z_raw": asimov_z(s, b),
                                }
                            )

    out = pd.DataFrame(rows)

    # Useful rankings.
    out_sorted_z = out.sort_values("asimov_Z_raw", ascending=False)
    out_sorted_sb = out.sort_values("S_over_B_raw", ascending=False)
    out_sorted_sroot = out.sort_values("S_over_sqrt_SplusB_raw", ascending=False)

    out.to_csv(outdir / "cut_scan_all.csv", index=False)
    out_sorted_z.head(100).to_csv(outdir / "cut_scan_top100_by_asimovZ.csv", index=False)
    out_sorted_sb.head(100).to_csv(outdir / "cut_scan_top100_by_SoverB.csv", index=False)
    out_sorted_sroot.head(100).to_csv(outdir / "cut_scan_top100_by_SoverSqrtSB.csv", index=False)

    print("Read:", args.skim)
    print("Rows:", len(df))
    print("\nWrote:")
    print(outdir / "cut_scan_all.csv")
    print(outdir / "cut_scan_top100_by_asimovZ.csv")
    print(outdir / "cut_scan_top100_by_SoverB.csv")
    print(outdir / "cut_scan_top100_by_SoverSqrtSB.csv")

    print("\nTop 20 by raw Asimov Z:")
    cols = [
        "channel",
        "mbb_low",
        "mbb_high",
        "n_jets_region",
        "b2_btag_cut",
        "met_min",
        "ht_min",
        "S_raw",
        "B_raw",
        "S_over_B_raw",
        "S_over_sqrt_SplusB_raw",
        "asimov_Z_raw",
    ]
    print(out_sorted_z[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()