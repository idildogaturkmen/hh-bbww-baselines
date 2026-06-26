'''
Apply a pT-binned m_bb correction derived from the signal calibration events to all events in the one-lepton pre-mass region.
The correction is derived from the median m_bb of the signal calibration events in each pT bin, and then applied to all events in the pre-mass region based on the pT of the softer of the two selected b-tagged jets.
The correction is clipped to a minimum and maximum scale factor to avoid extreme corrections in sparse bins.
The output is a parquet file with the original and corrected m_bb values, as well as a CSV file with the correction table and metrics for the corrected m_bb shapes.
'''
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROCESS_ORDER = [
    "minbias",
    "gamma",
    "upsilon",
    "qcd",
    "wjets",
    "dy_zjets",
    "diboson",
    "triboson",
    "single_higgs",
    "HH_other",
    "ttbar",
    "ttV_ttH_tttt",
]


LABELS = {
    "signal": "HH→bbWW signal",
    "all_background": "All backgrounds",
    "HH_other": "Other HH",
    "single_higgs": "Single Higgs",
    "diboson": "Diboson",
    "ttbar": "ttbar",
    "ttV_ttH_tttt": "ttV/ttH/tttt",
    "dy_zjets": "DY/Z+jets",
    "wjets": "W+jets",
    "qcd": "QCD",
    "gamma": "γ+jets",
    "minbias": "Minbias",
    "upsilon": "Upsilon",
    "triboson": "Triboson",
}


def width68(x: np.ndarray) -> tuple[float, float, float]:
    q16, q84 = np.percentile(x, [16, 84])
    return q16, q84, 0.5 * (q84 - q16)


def summarize_array(name: str, x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "group": name,
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
        "group": name,
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


def asimov_z(s: float, b: float) -> float:
    if s <= 0 or b <= 0:
        return 0.0
    return np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s))


def build_preselection(df: pd.DataFrame) -> pd.Series:
    return (
        df["pass_1lep_ge3jets"].astype(bool)
        & (df["n_jets"] >= 3)
        & (df["n_jets"] <= 5)
        & (df["b2_btag"] >= 1)
        & np.isfinite(df["mbb_top2_btag"])
        & np.isfinite(df["b1_pt"])
        & np.isfinite(df["b2_pt"])
    )


def add_deterministic_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic half split for signal calibration/evaluation.
    This avoids calibrating and evaluating the signal correction on exactly
    the same signal events.
    """
    df = df.copy()
    key = (
        df["sample"].astype(str)
        + "_"
        + df["file_index"].astype(str)
        + "_"
        + df["event_in_file"].astype(str)
    )
    hashed = pd.util.hash_pandas_object(key, index=False).to_numpy()
    df["is_calib"] = (hashed % 2) == 0
    return df


def make_quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)

    # Protect against duplicate quantile edges.
    if len(edges) < 3:
        lo, hi = np.nanmin(x), np.nanmax(x)
        edges = np.linspace(lo, hi, min(n_bins, 4) + 1)

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def calibrate_pt_binned_scale(
    df_pre: pd.DataFrame,
    n_bins: int,
    target_mass: float,
    min_signal_per_bin: int,
    clip_min: float,
    clip_max: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    sig_calib = df_pre[(df_pre["process_group"] == "signal") & (df_pre["is_calib"])].copy()

    if len(sig_calib) < n_bins * min_signal_per_bin:
        print(
            f"[warning] Only {len(sig_calib)} calibration signal events. "
            f"Using fewer/effective bins may be safer."
        )

    edges = make_quantile_bins(sig_calib["soft_b_pt"].to_numpy(dtype=float), n_bins)

    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (sig_calib["soft_b_pt"] >= lo) & (sig_calib["soft_b_pt"] < hi)
        sub = sig_calib[in_bin]
        x = sub["mbb_uncorrected"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]

        if len(x) < min_signal_per_bin:
            median_mbb = np.nan
            scale = np.nan
        else:
            median_mbb = np.median(x)
            scale = target_mass / median_mbb if median_mbb > 0 else np.nan

        rows.append(
            {
                "bin_index": i,
                "soft_b_pt_low": lo,
                "soft_b_pt_high": hi,
                "n_signal_calib": len(x),
                "median_mbb_signal_calib": median_mbb,
                "raw_scale_factor": scale,
            }
        )

    table = pd.DataFrame(rows)

    # Fill any sparse bins with global calibration scale.
    global_median = np.median(sig_calib["mbb_uncorrected"].to_numpy(dtype=float))
    global_scale = target_mass / global_median

    table["scale_factor"] = table["raw_scale_factor"].fillna(global_scale)
    table["scale_factor"] = table["scale_factor"].clip(lower=clip_min, upper=clip_max)
    table["global_signal_median_calib"] = global_median
    table["global_scale_factor"] = global_scale

    return table, edges


def apply_binned_scale(df: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
    corrected = np.full(len(df), np.nan, dtype=float)
    soft = df["soft_b_pt"].to_numpy(dtype=float)
    mbb = df["mbb_uncorrected"].to_numpy(dtype=float)

    for _, row in table.iterrows():
        lo = row["soft_b_pt_low"]
        hi = row["soft_b_pt_high"]
        scale = row["scale_factor"]
        mask = (soft >= lo) & (soft < hi)
        corrected[mask] = mbb[mask] * scale

    return corrected


def make_shape_metrics(df_pre: pd.DataFrame, mass_columns: list[str], outdir: Path):
    rows = []

    for mass_col in mass_columns:
        for group, g in df_pre.groupby("process_group"):
            row = summarize_array(group, g[mass_col].to_numpy(dtype=float))
            row["mass_variant"] = mass_col
            rows.append(row)

        bkg = df_pre[df_pre["process_group"] != "signal"]
        row = summarize_array("all_background", bkg[mass_col].to_numpy(dtype=float))
        row["mass_variant"] = mass_col
        rows.append(row)

    pd.DataFrame(rows).to_csv(outdir / "shape_metrics_by_process_and_variant.csv", index=False)


def make_window_scan(df_pre: pd.DataFrame, mass_columns: list[str], outdir: Path):
    windows = [
        (80, 150),
        (90, 140),
        (95, 145),
        (100, 140),
        (100, 150),
        (105, 145),
    ]

    rows = []
    for mass_col in mass_columns:
        for lo, hi in windows:
            mask = (
                (df_pre[mass_col] >= lo)
                & (df_pre[mass_col] <= hi)
                & (df_pre["ht"] >= 100)
            )

            s = int((mask & (df_pre["process_group"] == "signal")).sum())
            b = int((mask & (df_pre["process_group"] != "signal")).sum())

            rows.append(
                {
                    "mass_variant": mass_col,
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

    pd.DataFrame(rows).to_csv(outdir / "mass_window_scan_raw_counts.csv", index=False)


def plot_signal_comparison(df_pre: pd.DataFrame, mass_columns: list[str], outdir: Path):
    sig = df_pre[df_pre["process_group"] == "signal"]
    bins = np.linspace(0, 280, 70)

    plt.figure(figsize=(9, 6))
    for col in mass_columns:
        x = sig[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        plt.hist(x, bins=bins, histtype="step", density=True, linewidth=2, label=col)

    plt.axvline(90, linestyle="--", linewidth=1)
    plt.axvline(125, linestyle="-", linewidth=1)
    plt.axvline(140, linestyle="--", linewidth=1)
    plt.xlabel(r"$m_{bb}$ [GeV]")
    plt.ylabel("Normalized signal events")
    plt.title("Signal m_bb correction comparison in one-lepton pre-mass region")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "signal_mbb_variant_overlay.png", dpi=200)
    plt.close()


def plot_signal_vs_background_overlay(df_pre: pd.DataFrame, mass_col: str, outdir: Path):
    bins = np.linspace(0, 350, 71)

    plt.figure(figsize=(9, 6))

    for group in ["signal", "all_background", "single_higgs", "HH_other", "diboson", "ttbar", "ttV_ttH_tttt"]:
        if group == "all_background":
            x = df_pre.loc[df_pre["process_group"] != "signal", mass_col].to_numpy(dtype=float)
        else:
            x = df_pre.loc[df_pre["process_group"] == group, mass_col].to_numpy(dtype=float)

        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue

        plt.hist(
            x,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=2,
            label=f"{LABELS.get(group, group)} (n={len(x)})",
        )

    plt.axvline(90, linestyle="--", linewidth=1)
    plt.axvline(125, linestyle="-", linewidth=1)
    plt.axvline(140, linestyle="--", linewidth=1)
    plt.xlabel(r"$m_{bb}$ [GeV]")
    plt.ylabel("Normalized events")
    plt.title(mass_col)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / f"{mass_col}_normalized_overlay.png", dpi=200)
    plt.close()


def plot_scale_vs_pt(table: pd.DataFrame, outdir: Path):
    centers = []
    labels = []

    for _, row in table.iterrows():
        lo = row["soft_b_pt_low"]
        hi = row["soft_b_pt_high"]

        if np.isneginf(lo):
            center = hi
            label = f"<{hi:.0f}"
        elif np.isposinf(hi):
            center = lo
            label = f">{lo:.0f}"
        else:
            center = 0.5 * (lo + hi)
            label = f"{lo:.0f}-{hi:.0f}"

        centers.append(center)
        labels.append(label)

    plt.figure(figsize=(8, 5))
    plt.plot(centers, table["scale_factor"], marker="o")
    for x, y, label in zip(centers, table["scale_factor"], labels):
        plt.text(x, y, label, fontsize=8, ha="center", va="bottom")

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel(r"Softer selected b-jet $p_T$ [GeV]")
    plt.ylabel("m_bb scale factor")
    plt.title("Signal-derived pT-binned mass correction")
    plt.tight_layout()
    plt.savefig(outdir / "pt_binned_scale_factors.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skim", default="outputs/all_background_reco_skim/all_processes_reco_skim.parquet")
    parser.add_argument("--outdir", default="outputs/pt_binned_mbb_correction")
    parser.add_argument("--target-mass", type=float, default=125.0)
    parser.add_argument("--n-bins", type=int, default=6)
    parser.add_argument("--min-signal-per-bin", type=int, default=25)
    parser.add_argument("--clip-min", type=float, default=0.75)
    parser.add_argument("--clip-max", type=float, default=1.50)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    plotdir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.skim)
    df = add_deterministic_split(df)

    pre_mask = build_preselection(df)
    df_pre = df[pre_mask].copy()

    df_pre["soft_b_pt"] = np.minimum(df_pre["b1_pt"], df_pre["b2_pt"])
    df_pre["mbb_uncorrected"] = df_pre["mbb_top2_btag"]

    sig_calib = df_pre[(df_pre["process_group"] == "signal") & (df_pre["is_calib"])]
    global_median = np.median(sig_calib["mbb_uncorrected"].to_numpy(dtype=float))
    global_scale = args.target_mass / global_median

    df_pre["mbb_global_median_scaled"] = df_pre["mbb_uncorrected"] * global_scale

    table, edges = calibrate_pt_binned_scale(
        df_pre=df_pre,
        n_bins=args.n_bins,
        target_mass=args.target_mass,
        min_signal_per_bin=args.min_signal_per_bin,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )

    df_pre["mbb_pt_binned_scaled"] = apply_binned_scale(df_pre, table)

    mass_columns = [
        "mbb_uncorrected",
        "mbb_global_median_scaled",
        "mbb_pt_binned_scaled",
    ]

    table.to_csv(outdir / "pt_binned_correction_table.csv", index=False)

    keep_cols = [
        "sample",
        "process_group",
        "file_index",
        "event_in_file",
        "is_calib",
        "n_jets",
        "n_leptons",
        "ht",
        "met",
        "b1_pt",
        "b2_pt",
        "soft_b_pt",
        "mbb_uncorrected",
        "mbb_global_median_scaled",
        "mbb_pt_binned_scaled",
    ]
    df_pre[keep_cols].to_parquet(outdir / "onelep_pre_mbb_with_corrected_masses.parquet", index=False)

    make_shape_metrics(df_pre, mass_columns, outdir)
    make_window_scan(df_pre, mass_columns, outdir)

    plot_scale_vs_pt(table, plotdir)
    plot_signal_comparison(df_pre, mass_columns, plotdir)

    for col in mass_columns:
        plot_signal_vs_background_overlay(df_pre, col, plotdir)

    print("Pre-mass selected events:", len(df_pre))
    print("Signal:", int((df_pre["process_group"] == "signal").sum()))
    print("Background:", int((df_pre["process_group"] != "signal").sum()))
    print("\nCorrection table:")
    print(table.to_string(index=False))
    print("\nWrote outputs to:", outdir)


if __name__ == "__main__":
    main()
