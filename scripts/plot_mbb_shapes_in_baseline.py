'''
Plot the m_bb shapes in the baseline region, and produce a CSV summary of metrics.
Baseline: m_bb between 90 and 140 GeV, HT >= 100 GeV, at least 3 jets, at most 5 jets, second-highest b-tag flag is 1 (requiring at least two b-tagged selected jets), and passing the 1-lepton selection.
Plots signal and background m_bb distributions, both normalized and raw, and also plots the signal fraction S/(S+B) per bin.
'''
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SKIM = "outputs/all_background_reco_skim/all_processes_reco_skim.parquet"
OUTDIR = Path("outputs/mbb_shape_baseline_diagnostics/plots")


PROCESS_ORDER = [
    "minbias",
    "gamma",
    "upsilon",
    "QCD",
    "Wjets",
    "DY_Zjets",
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
    "DY_Zjets": "DY/Z+jets",
    "Wjets": "W+jets",
    "QCD": "QCD",
    "gamma": "γ+jets",
    "minbias": "Minbias",
    "upsilon": "Upsilon",
    "triboson": "Triboson",
}


def region_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    pre = (
        df["pass_1lep_ge3jets"].astype(bool)
        & (df["n_jets"] >= 3)
        & (df["n_jets"] <= 5)
        & (df["b2_btag"] >= 1)
        & np.isfinite(df["mbb_top2_btag"])
    )

    baseline = (
        pre
        & (df["mbb_top2_btag"] >= 90)
        & (df["mbb_top2_btag"] <= 140)
        & (df["ht"] >= 100)
    )

    return {
        "onelep_3to5_b2eq1_pre_mbb": pre,
        "onelep_baseline_mbb90_140_ht100": baseline,
    }


def add_reference_lines():
    plt.axvline(90, linestyle="--", linewidth=1)
    plt.axvline(125, linestyle="-", linewidth=1)
    plt.axvline(140, linestyle="--", linewidth=1)


def plot_normalized_overlay(df: pd.DataFrame, mask: pd.Series, region: str):
    sub = df[mask].copy()

    bins = np.linspace(0, 350, 71)

    plt.figure(figsize=(9, 6))

    groups = [
        "signal",
        "all_background",
        "single_higgs",
        "HH_other",
        "diboson",
        "ttbar",
        "ttV_ttH_tttt",
    ]

    for group in groups:
        if group == "all_background":
            x = sub.loc[sub["process_group"] != "signal", "mbb_top2_btag"].to_numpy()
        else:
            x = sub.loc[sub["process_group"] == group, "mbb_top2_btag"].to_numpy()

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

    add_reference_lines()
    plt.xlabel(r"$m_{bb}$ from two highest b-tag jets [GeV]")
    plt.ylabel("Normalized events")
    plt.title(region)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTDIR / region / "mbb_normalized_overlay.png", dpi=200)
    plt.close()


def plot_raw_stack(df: pd.DataFrame, mask: pd.Series, region: str, signal_scale: float = 20.0):
    sub = df[mask].copy()

    bins = np.linspace(0, 350, 71)

    bkg_arrays = []
    bkg_labels = []

    for group in PROCESS_ORDER:
        x = sub.loc[sub["process_group"] == group, "mbb_top2_btag"].to_numpy()
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        bkg_arrays.append(x)
        bkg_labels.append(LABELS.get(group, group))

    sig = sub.loc[sub["process_group"] == "signal", "mbb_top2_btag"].to_numpy()
    sig = sig[np.isfinite(sig)]

    plt.figure(figsize=(10, 6))

    if bkg_arrays:
        plt.hist(
            bkg_arrays,
            bins=bins,
            stacked=True,
            label=bkg_labels,
        )

    if len(sig):
        sig_counts, edges = np.histogram(sig, bins=bins)
        centers = 0.5 * (edges[1:] + edges[:-1])
        plt.step(
            centers,
            sig_counts * signal_scale,
            where="mid",
            linewidth=2,
            label=f"Signal ×{signal_scale:g}",
        )

    add_reference_lines()
    plt.xlabel(r"$m_{bb}$ from two highest b-tag jets [GeV]")
    plt.ylabel("Raw events")
    plt.title(region)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTDIR / region / "mbb_raw_stack_signal_scaled.png", dpi=200)
    plt.close()


def plot_signal_fraction(df: pd.DataFrame, mask: pd.Series, region: str):
    sub = df[mask].copy()

    bins = np.linspace(0, 350, 36)

    sig = sub.loc[sub["process_group"] == "signal", "mbb_top2_btag"].to_numpy()
    bkg = sub.loc[sub["process_group"] != "signal", "mbb_top2_btag"].to_numpy()

    sig = sig[np.isfinite(sig)]
    bkg = bkg[np.isfinite(bkg)]

    s_counts, edges = np.histogram(sig, bins=bins)
    b_counts, _ = np.histogram(bkg, bins=bins)

    denom = s_counts + b_counts
    frac = np.divide(
        s_counts,
        denom,
        out=np.zeros_like(s_counts, dtype=float),
        where=denom > 0,
    )

    centers = 0.5 * (edges[1:] + edges[:-1])

    plt.figure(figsize=(9, 5))
    plt.step(centers, frac, where="mid", linewidth=2)
    add_reference_lines()
    plt.xlabel(r"$m_{bb}$ from two highest b-tag jets [GeV]")
    plt.ylabel(r"Raw $S/(S+B)$ per bin")
    plt.title(region)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(OUTDIR / region / "mbb_signal_fraction_by_bin.png", dpi=200)
    plt.close()


def plot_signal_median_scaled(df: pd.DataFrame, mask: pd.Series, region: str):
    sub = df[mask].copy()
    sig = sub.loc[sub["process_group"] == "signal", "mbb_top2_btag"].to_numpy()
    sig = sig[np.isfinite(sig)]

    if len(sig) == 0:
        return

    median = np.median(sig)
    scale = 125.0 / median

    bins = np.linspace(0, 250, 60)

    plt.figure(figsize=(9, 5))
    plt.hist(sig, bins=bins, histtype="step", density=True, linewidth=2, label=f"Unscaled, median={median:.1f}")
    plt.hist(sig * scale, bins=bins, histtype="step", density=True, linewidth=2, label=f"Median scaled to 125, scale={scale:.3f}")
    plt.axvline(125, linestyle="-", linewidth=1)
    plt.xlabel(r"$m_{bb}$ [GeV]")
    plt.ylabel("Normalized signal events")
    plt.title(region + " signal-only scaling diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / region / "signal_mbb_median_scaled_comparison.png", dpi=200)
    plt.close()


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SKIM)
    masks = region_masks(df)

    for region, mask in masks.items():
        (OUTDIR / region).mkdir(parents=True, exist_ok=True)

        print(region)
        print("  total events:", int(mask.sum()))
        print("  signal:", int((mask & (df["process_group"] == "signal")).sum()))
        print("  background:", int((mask & (df["process_group"] != "signal")).sum()))

        plot_normalized_overlay(df, mask, region)
        plot_raw_stack(df, mask, region)
        plot_signal_fraction(df, mask, region)
        plot_signal_median_scaled(df, mask, region)

    print("Wrote plots to:", OUTDIR)


if __name__ == "__main__":
    main()
