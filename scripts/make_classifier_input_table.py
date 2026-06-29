'''
Makes a table of features for training a classifier to separate signal from background events.

'''
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["sample", "file_index", "event_in_file"]

EXTRA_COLUMNS_FROM_SKIM = [
    "sample",
    "file_index",
    "event_in_file",
    "file",
    "weight",
    "process_group",
    "n_jets_raw",
    "n_electrons",
    "n_muons",
    "leading_lepton_pt",
    "lead_jet_pt",
    "sublead_jet_pt",
    "b1_eta",
    "b2_eta",
    "b1_phi",
    "b2_phi",
    "b1_mass",
    "b2_mass",
    "b1_btag",
    "b2_btag",
    "b1_btag_phys",
    "b2_btag_phys",
    "dr_bb_top2_btag",
]


BASE_FEATURES = [
    "mbb_uncorrected",
    "mbb_global_median_scaled",
    "mbb_pt_binned_scaled",
    "n_jets",
    "n_leptons",
    "n_electrons",
    "n_muons",
    "ht",
    "met",
    "leading_lepton_pt",
    "lead_jet_pt",
    "sublead_jet_pt",
    "b1_pt",
    "b2_pt",
    "soft_b_pt",
    "b1_eta",
    "b2_eta",
    "abs_b1_eta",
    "abs_b2_eta",
    "deta_bb",
    "dphi_bb",
    "dr_bb_top2_btag",
    "b1_mass",
    "b2_mass",
    "b1_btag",
    "b2_btag",
    "b1_btag_phys",
    "b2_btag_phys",
    "b2_over_b1_pt",
    "met_over_ht",
    "mbb_pt_binned_over_ht",
]


def deterministic_split(df: pd.DataFrame) -> pd.Series:
    key = (
        df["sample"].astype(str)
        + "_"
        + df["file_index"].astype(str)
        + "_"
        + df["event_in_file"].astype(str)
    )
    hashed = pd.util.hash_pandas_object(key, index=False).to_numpy()
    bucket = hashed % 100

    split = np.full(len(df), "test", dtype=object)
    split[bucket < 60] = "train"
    split[(bucket >= 60) & (bucket < 80)] = "val"
    return pd.Series(split, index=df.index)


def wrapped_delta_phi(phi1: pd.Series, phi2: pd.Series) -> pd.Series:
    dphi = phi1.to_numpy(dtype=float) - phi2.to_numpy(dtype=float)
    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
    return pd.Series(np.abs(dphi), index=phi1.index)


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    num = num.to_numpy(dtype=float)
    den = den.to_numpy(dtype=float)
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den != 0)
    return pd.Series(out)


def asimov_z(s: float, b: float) -> float:
    if s <= 0 or b <= 0:
        return 0.0
    return float(np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s)))


def baseline_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    windows = [
        ("uncorrected_80_150", "mbb_uncorrected", 80, 150),
        ("uncorrected_90_140", "mbb_uncorrected", 90, 140),
        ("pt_binned_80_150", "mbb_pt_binned_scaled", 80, 150),
        ("pt_binned_90_140", "mbb_pt_binned_scaled", 90, 140),
    ]

    for split_name, sub in [("all", df), *list(df.groupby("split"))]:
        for label, mass_col, lo, hi in windows:
            mask = (
                (sub["ht"] >= 100)
                & (sub[mass_col] >= lo)
                & (sub[mass_col] <= hi)
            )
            s = int((mask & (sub["target"] == 1)).sum())
            b = int((mask & (sub["target"] == 0)).sum())

            rows.append(
                {
                    "split": split_name,
                    "selection": label,
                    "mass_col": mass_col,
                    "window_low": lo,
                    "window_high": hi,
                    "ht_min": 100,
                    "S_raw": s,
                    "B_raw": b,
                    "S_over_B": s / b if b > 0 else np.nan,
                    "S_over_sqrt_SB": s / np.sqrt(s + b) if (s + b) > 0 else np.nan,
                    "asimovZ_raw": asimov_z(s, b),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skim",
        default="outputs/all_background_reco_skim/all_processes_reco_skim.parquet",
    )
    parser.add_argument(
        "--mass-table",
        default="outputs/final_mbb_correction_summary/onelep_pre_mbb_final_mass_variants.parquet",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/classifier_input",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    skim = pd.read_parquet(args.skim, columns=[c for c in EXTRA_COLUMNS_FROM_SKIM])
    mass = pd.read_parquet(args.mass_table)

    # Avoid duplicate process_group from skim; keep mass-table process_group.
    skim_extra = skim.drop(columns=["process_group"])

    df = mass.merge(
        skim_extra,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    df["target"] = (df["process_group"] == "signal").astype(int)
    df["unit_weight"] = 1.0
    df["split"] = deterministic_split(df)

    # Derived features.
    df["abs_b1_eta"] = df["b1_eta"].abs()
    df["abs_b2_eta"] = df["b2_eta"].abs()
    df["deta_bb"] = (df["b1_eta"] - df["b2_eta"]).abs()
    df["dphi_bb"] = wrapped_delta_phi(df["b1_phi"], df["b2_phi"])
    df["b2_over_b1_pt"] = safe_div(df["b2_pt"], df["b1_pt"])
    df["met_over_ht"] = safe_div(df["met"], df["ht"])
    df["mbb_pt_binned_over_ht"] = safe_div(df["mbb_pt_binned_scaled"], df["ht"])

    # Keep only features that exist.
    feature_columns = [c for c in BASE_FEATURES if c in df.columns]

    # Check finite coverage.
    finite_summary = []
    for col in feature_columns:
        x = df[col].to_numpy(dtype=float)
        finite_summary.append(
            {
                "feature": col,
                "n_total": len(x),
                "n_finite": int(np.isfinite(x).sum()),
                "finite_fraction": float(np.isfinite(x).mean()),
                "mean": float(np.nanmean(x)),
                "std": float(np.nanstd(x)),
            }
        )

    finite_summary = pd.DataFrame(finite_summary)

    # Save outputs.
    df.to_parquet(outdir / "onelep_pre_mbb_classifier_input.parquet", index=False)
    df.head(200).to_csv(outdir / "onelep_pre_mbb_classifier_input_preview.csv", index=False)

    with open(outdir / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    finite_summary.to_csv(outdir / "feature_finite_summary.csv", index=False)

    split_counts = (
        df.groupby(["split", "target"])
        .size()
        .reset_index(name="n_events")
        .sort_values(["split", "target"])
    )
    split_counts.to_csv(outdir / "split_target_counts.csv", index=False)

    process_counts = (
        df.groupby(["split", "process_group"])
        .size()
        .reset_index(name="n_events")
        .sort_values(["split", "n_events"], ascending=[True, False])
    )
    process_counts.to_csv(outdir / "split_process_counts.csv", index=False)

    metrics = baseline_metrics(df)
    metrics.to_csv(outdir / "cut_based_baseline_metrics_by_split.csv", index=False)

    print("Wrote classifier input table to:", outdir)
    print("Shape:", df.shape)
    print("\nTarget counts:")
    print(df["target"].value_counts().rename(index={0: "background", 1: "signal"}).to_string())

    print("\nSplit/target counts:")
    print(split_counts.to_string(index=False))

    print("\nFeature columns:")
    for c in feature_columns:
        print(" ", c)

    print("\nCut-based baseline metrics:")
    print(metrics.sort_values(["split", "asimovZ_raw"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
