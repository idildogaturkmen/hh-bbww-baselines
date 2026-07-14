#!/usr/bin/env python3

from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

REPO = Path(os.environ["HH4B_REPO"])

IN_DIR = REPO / "outputs/tables/hh4b_deepspanet_twohead_qcdplus_qcdplus_btag4_v1_2026_07_14"
OUT_TABLE = REPO / "outputs/tables/deepspanet_twohead_qcdplus_diagnostics_2026_07_14"
OUT_PLOT = REPO / "outputs/plots/deepspanet_twohead_qcdplus_diagnostics_2026_07_14"

OUT_TABLE.mkdir(parents=True, exist_ok=True)
OUT_PLOT.mkdir(parents=True, exist_ok=True)

LUMI_PB = 450_000.0


def analysis_label(ax, lumi_fb=450):
    fig = ax.figure
    fig.text(0.105, 0.988, "Delphes simulation", fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.90, 0.988, rf"13 TeV, {lumi_fb:g} fb$^{{-1}}$", fontsize=10.5, ha="right", va="top")


def safe_auc(y, score, weight=None):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, score, sample_weight=weight)


def make_roc(y, score, weight, label, outname):
    fpr, tpr, _ = roc_curve(y, score, sample_weight=weight)
    auc_w = safe_auc(y, score, weight)

    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(OUT_TABLE / f"{outname}.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot(fpr, tpr, linewidth=2, label=f"{label} AUC={auc_w:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="0.5")
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    analysis_label(ax)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(OUT_PLOT / f"{outname}.png", dpi=220)
    fig.savefig(OUT_PLOT / f"{outname}.pdf")
    plt.close(fig)

    eps = np.clip(fpr, 1e-5, 1.0)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot(tpr, 1.0 / eps, linewidth=2, label=f"{label} AUC={auc_w:.3f}")
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection")
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    analysis_label(ax)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(OUT_PLOT / f"{outname}_background_rejection.png", dpi=220)
    fig.savefig(OUT_PLOT / f"{outname}_background_rejection.pdf")
    plt.close(fig)

    return auc_w


def main():
    pred_path = IN_DIR / "test_predictions.csv"
    scan_path = IN_DIR / "threshold_scan.csv"
    summary_path = IN_DIR / "summary.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}; wait for the two-head job to finish.")

    pred = pd.read_csv(pred_path)
    y = pred["y"].to_numpy(dtype=int)
    w = pred["weight_pb_scaled"].to_numpy(dtype=float)

    qcd_like = pred["sample_id"].isin([3, 4, 5, 6, 7]).to_numpy()
    top_like = pred["sample_id"].isin([2]).to_numpy()
    sig = y == 1

    rows = []
    rows.append({
        "roc": "combined_signal_vs_all",
        "auc_unweighted": safe_auc(y, pred["combined_score"]),
        "auc_weighted": make_roc(y, pred["combined_score"], w, "Combined score: signal vs all", "roc_combined_signal_vs_all"),
    })

    mask = sig | qcd_like
    rows.append({
        "roc": "qcd_head_signal_vs_qcdlike",
        "auc_unweighted": safe_auc(y[mask], pred.loc[mask, "qcd_score"]),
        "auc_weighted": make_roc(y[mask], pred.loc[mask, "qcd_score"], w[mask], "QCD head: signal vs QCD-like", "roc_qcd_head_signal_vs_qcdlike"),
    })

    mask = sig | top_like
    rows.append({
        "roc": "top_head_signal_vs_ttbar",
        "auc_unweighted": safe_auc(y[mask], pred.loc[mask, "top_score"]),
        "auc_weighted": make_roc(y[mask], pred.loc[mask, "top_score"], w[mask], "Top head: signal vs ttbar", "roc_top_head_signal_vs_ttbar"),
    })

    roc_summary = pd.DataFrame(rows)
    roc_summary.to_csv(OUT_TABLE / "deepspanet_twohead_roc_summary.csv", index=False)
    (OUT_TABLE / "deepspanet_twohead_roc_summary.md").write_text(roc_summary.to_markdown(index=False) + "\n")

    # Score plane.
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.scatter(pred.loc[y == 0, "qcd_score"], pred.loc[y == 0, "top_score"], s=8, alpha=0.25, label="Background")
    ax.scatter(pred.loc[y == 1, "qcd_score"], pred.loc[y == 1, "top_score"], s=16, alpha=0.7, label="Signal")
    ax.set_xlabel("QCD-head signal score")
    ax.set_ylabel("Top-head signal score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    analysis_label(ax)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(OUT_PLOT / "deepspanet_twohead_score_plane.png", dpi=220)
    fig.savefig(OUT_PLOT / "deepspanet_twohead_score_plane.pdf")
    plt.close(fig)

    # Rectangle scan heatmap.
    if scan_path.exists():
        scan = pd.read_csv(scan_path)
        scan.to_csv(OUT_TABLE / "threshold_scan_copy.csv", index=False)

        pivot = scan.pivot(index="top_threshold", columns="qcd_threshold", values="S_over_sqrtB")
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        im = ax.imshow(
            pivot.values,
            origin="lower",
            aspect="auto",
            extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$S/\sqrt{B}$")
        ax.set_xlabel("QCD-head threshold")
        ax.set_ylabel("Top-head threshold")
        analysis_label(ax)
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
        fig.savefig(OUT_PLOT / "deepspanet_twohead_threshold_scan_heatmap.png", dpi=220)
        fig.savefig(OUT_PLOT / "deepspanet_twohead_threshold_scan_heatmap.pdf")
        plt.close(fig)

    # Baseline comparison.
    if summary_path.exists():
        s = pd.read_csv(summary_path).iloc[0]
        comp = pd.DataFrame([
            {"model": "BDT mass-aware", "best_S_over_sqrtB": 0.198708},
            {"model": "Deep two-head SPA-Net", "best_S_over_sqrtB": float(s["S_over_sqrtB"])},
            {"model": "SPA-Net qcdplus", "best_S_over_sqrtB": 0.163051},
            {"model": "DNN mass-aware", "best_S_over_sqrtB": 0.154042},
            {"model": "BDT topology-only", "best_S_over_sqrtB": 0.142005},
            {"model": "LBN p4 + topology", "best_S_over_sqrtB": 0.122621},
        ])
        comp.to_csv(OUT_TABLE / "deepspanet_vs_baselines.csv", index=False)
        (OUT_TABLE / "deepspanet_vs_baselines.md").write_text(comp.to_markdown(index=False) + "\n")

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.bar(comp["model"], comp["best_S_over_sqrtB"])
        ax.set_ylabel(r"Best $S/\sqrt{B}$ at 450 fb$^{-1}$")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.25)
        analysis_label(ax)
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
        fig.savefig(OUT_PLOT / "deepspanet_vs_baselines.png", dpi=220)
        fig.savefig(OUT_PLOT / "deepspanet_vs_baselines.pdf")
        plt.close(fig)

    print("Wrote:", OUT_TABLE)
    print("Wrote:", OUT_PLOT)
    print(roc_summary.to_string(index=False))


if __name__ == "__main__":
    main()
