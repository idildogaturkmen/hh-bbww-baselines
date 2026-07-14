#!/usr/bin/env python3

from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(os.environ["HH4B_REPO"])

IN_TABLE = REPO / "outputs/tables/spanet_qcdplus_diagnostics_2026_07_14/spanet_test_predictions.csv"
OUT_TABLE = REPO / "outputs/tables/spanet_qcdplus_lumi_projection_2026_07_14"
OUT_PLOT = REPO / "outputs/plots/spanet_qcdplus_lumi_projection_2026_07_14"

OUT_TABLE.mkdir(parents=True, exist_ok=True)
OUT_PLOT.mkdir(parents=True, exist_ok=True)

LUMIS_FB = [138, 350, 450, 4000]


def analysis_label(ax, extra="Private work"):
    ax.text(
        0.02, 0.96, "Delphes simulation",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.02, 0.90, extra,
        transform=ax.transAxes,
        fontsize=11,
        style="italic",
        va="top",
    )
    ax.text(
        0.98, 0.96,
        r"13 TeV",
        transform=ax.transAxes,
        fontsize=11,
        ha="right",
        va="top",
    )


def main():
    if not IN_TABLE.exists():
        raise FileNotFoundError(
            f"Missing {IN_TABLE}. Run plot_spanet_qcdplus_diagnostics.py first."
        )

    df = pd.read_csv(IN_TABLE)
    thresholds = np.linspace(0.0, 0.99, 100)

    rows = []
    for lumi_fb in LUMIS_FB:
        lumi_pb = lumi_fb * 1000.0

        for thr in thresholds:
            sel = df["score"] >= thr
            sig = sel & (df["y"] == 1)
            bkg = sel & (df["y"] == 0)

            S = float((df.loc[sig, "weight_pb_scaled"] * lumi_pb).sum())
            B = float((df.loc[bkg, "weight_pb_scaled"] * lumi_pb).sum())

            rows.append({
                "lumi_fb": lumi_fb,
                "threshold": thr,
                "S": S,
                "B": B,
                "S_over_B": S / B if B > 0 else np.nan,
                "S_over_sqrtB_stat": S / np.sqrt(B) if B > 0 else np.nan,
                "S_over_sqrtB_10pct_syst": S / np.sqrt(B + (0.10 * B) ** 2) if B > 0 else np.nan,
                "signal_test_rows": int(sig.sum()),
                "background_test_rows": int(bkg.sum()),
            })

    proj = pd.DataFrame(rows)
    proj.to_csv(OUT_TABLE / "spanet_lumi_threshold_projection.csv", index=False)

    best = (
        proj.sort_values(["lumi_fb", "S_over_sqrtB_stat"], ascending=[True, False])
        .groupby("lumi_fb")
        .head(1)
        .reset_index(drop=True)
    )

    best.to_csv(OUT_TABLE / "spanet_lumi_best_thresholds.csv", index=False)
    (OUT_TABLE / "spanet_lumi_best_thresholds.md").write_text(
        best.to_markdown(index=False) + "\n"
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for lumi_fb in LUMIS_FB:
        d = proj[proj["lumi_fb"] == lumi_fb]
        ax.plot(
            d["threshold"],
            d["S_over_sqrtB_stat"],
            linewidth=2,
            label=rf"{lumi_fb:g} fb$^{{-1}}$",
        )

    ax.set_xlabel("SPA-Net score threshold")
    ax.set_ylabel(r"$S/\sqrt{B}$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    analysis_label(ax, extra="Luminosity projection, statistical only")
    fig.tight_layout()
    fig.savefig(OUT_PLOT / "spanet_threshold_scan_by_lumi_stat.png", dpi=220)
    fig.savefig(OUT_PLOT / "spanet_threshold_scan_by_lumi_stat.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(
        best["lumi_fb"],
        best["S_over_sqrtB_stat"],
        marker="o",
        linewidth=2,
        label="stat. only",
    )
    ax.plot(
        best["lumi_fb"],
        best["S_over_sqrtB_10pct_syst"],
        marker="o",
        linewidth=2,
        label="10% background syst.",
    )

    ax.set_xlabel(r"Integrated luminosity [fb$^{-1}$]")
    ax.set_ylabel("Best projected sensitivity")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    analysis_label(ax, extra="Same trained SPA-Net score")
    fig.tight_layout()
    fig.savefig(OUT_PLOT / "spanet_best_sensitivity_vs_lumi.png", dpi=220)
    fig.savefig(OUT_PLOT / "spanet_best_sensitivity_vs_lumi.pdf")
    plt.close(fig)

    print("Wrote:", OUT_TABLE)
    print("Wrote:", OUT_PLOT)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
