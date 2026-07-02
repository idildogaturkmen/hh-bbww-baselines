#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUTDIR = Path("outputs/hh4b_baseline/model_comparison")
PLOTDIR = OUTDIR / "plots"
NOTEDIR = Path("notes")

OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)
NOTEDIR.mkdir(parents=True, exist_ok=True)


SOURCES = {
    "cut": {
        "label": "Cut baseline",
        "stability_csv": Path("outputs/hh4b_baseline/bdt_v2_stability/bdt_v2_stability_ranked_categories.csv"),
        "selection": "cut_all_resolved_4b",
        "metrics_json": None,
        "score_label": "Resolved 4b preselection",
        "notes": "Inclusive resolved HH4b cut baseline with >=4 b-tagged AK4 jets.",
    },
    "bdt": {
        "label": "BDT v2",
        "stability_csv": Path("outputs/hh4b_baseline/bdt_v2_stability/bdt_v2_stability_ranked_categories.csv"),
        "selection": "bdt_score_ge_0.55",
        "metrics_json": Path("outputs/hh4b_baseline/bdt_v2/hh4b_resolved_bdt_v2_metrics.json"),
        "score_label": "BDT score >= 0.55",
        "notes": "Best stable BDT working point from Poisson-bootstrap stability study.",
    },
    "dense_dnn": {
        "label": "Dense DNN v1",
        "stability_csv": Path("outputs/hh4b_baseline/dnn_v1_stability/bdt_v2_stability_ranked_categories.csv"),
        "selection": "bdt_score_ge_0.35",
        "metrics_json": Path("outputs/hh4b_baseline/dnn_v1/hh4b_resolved_dnn_v1_metrics.json"),
        "score_label": "DNN score >= 0.35",
        "notes": "Dense neural network trained on the same continuous resolved HH4b features as BDT v2.",
    },
    "lbn_dnn": {
        "label": "LBN-DNN v2",
        "stability_csv": Path("outputs/hh4b_baseline/lbn_dnn_v2_stability/bdt_v2_stability_ranked_categories.csv"),
        "selection": "bdt_score_ge_0.65",
        "metrics_json": Path("outputs/hh4b_baseline/lbn_dnn_v2/hh4b_resolved_lbn_dnn_v2_metrics.json"),
        "score_label": "LBN-DNN score >= 0.65",
        "notes": "LBN-inspired four-vector neural network. The selected working point has the best stable bootstrap-median Z_A but is close to the single-event dominance threshold.",
    },
}

# Optional alternative working points to include in the note but not headline table.
ALTERNATIVES = {
    "lbn_dnn_robust_alt": {
        "label": "LBN-DNN v2 alternative",
        "stability_csv": Path("outputs/hh4b_baseline/lbn_dnn_v2_stability/bdt_v2_stability_ranked_categories.csv"),
        "selection": "bdt_score_ge_0.60",
        "metrics_json": Path("outputs/hh4b_baseline/lbn_dnn_v2/hh4b_resolved_lbn_dnn_v2_metrics.json"),
        "score_label": "LBN-DNN score >= 0.60",
        "notes": "Slightly less pure than score >= 0.65, but less close to the max-single-event-fraction boundary.",
    },
}


def read_metrics(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        print(f"[WARN] Missing metrics json: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def get_row(config: dict) -> dict:
    csv_path = config["stability_csv"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing stability CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    match = df[df["selection"] == config["selection"]].copy()

    if len(match) != 1:
        raise RuntimeError(
            f"Expected exactly one row for selection {config['selection']} in {csv_path}, found {len(match)}"
        )

    row = match.iloc[0].to_dict()
    metrics = read_metrics(config["metrics_json"])

    out = {
        "model": config["label"],
        "working_point": config["score_label"],
        "selection_key": config["selection"],
        "passes_stability": bool(row["passes_stability"]),
        "raw_auc_test": metrics.get("raw_auc_test", np.nan),
        "weighted_auc_test": metrics.get("weighted_auc_test", np.nan),
        "raw_signal": row["raw_signal"],
        "raw_background": row["raw_background"],
        "S_w": row["S_w"],
        "B_w": row["B_w"],
        "S_over_B": row["S_over_B"],
        "asimov_Z_A": row["asimov_Z_A"],
        "Z_A_boot_p16": row["Z_A_boot_p16"],
        "Z_A_boot_p50": row["Z_A_boot_p50"],
        "Z_A_boot_p84": row["Z_A_boot_p84"],
        "Z_A_boot_rel_half_width": row["Z_A_boot_rel_half_width"],
        "N_eff_bkg": row["N_eff_bkg"],
        "max_single_bkg_frac": row["max_single_bkg_frac"],
        "top_sample": row["top_sample"],
        "top_sample_frac_B_w": row["top_sample_frac_B_w"],
        "notes": config["notes"],
    }

    return out


def fmt(x, nd=4):
    if pd.isna(x):
        return ""
    x = float(x)
    if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.3e}"
    return f"{x:.{nd}f}"


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


rows = [get_row(cfg) for cfg in SOURCES.values()]
comparison = pd.DataFrame(rows)

alt_rows = [get_row(cfg) for cfg in ALTERNATIVES.values()]
alternatives = pd.DataFrame(alt_rows)

comparison.to_csv(OUTDIR / "hh4b_model_comparison_summary.csv", index=False)
alternatives.to_csv(OUTDIR / "hh4b_model_comparison_alternative_working_points.csv", index=False)

paper_cols = [
    "model",
    "working_point",
    "passes_stability",
    "raw_auc_test",
    "weighted_auc_test",
    "raw_signal",
    "raw_background",
    "S_w",
    "B_w",
    "S_over_B",
    "asimov_Z_A",
    "Z_A_boot_p50",
    "Z_A_boot_p16",
    "Z_A_boot_p84",
    "N_eff_bkg",
    "max_single_bkg_frac",
    "top_sample_frac_B_w",
]

paper = comparison[paper_cols].copy()
paper_display = paper.copy()

paper_display["passes_stability"] = paper_display["passes_stability"].map({True: "yes", False: "no"})
for c in [
    "raw_auc_test",
    "weighted_auc_test",
    "S_w",
    "B_w",
    "S_over_B",
    "asimov_Z_A",
    "Z_A_boot_p50",
    "Z_A_boot_p16",
    "Z_A_boot_p84",
    "N_eff_bkg",
    "max_single_bkg_frac",
    "top_sample_frac_B_w",
]:
    paper_display[c] = paper_display[c].map(lambda x: fmt(x, 4))

paper_display = paper_display.rename(
    columns={
        "model": "Model",
        "working_point": "Working point",
        "passes_stability": "Stable?",
        "raw_auc_test": "Raw AUC",
        "weighted_auc_test": "Weighted AUC",
        "raw_signal": "Raw S",
        "raw_background": "Raw B",
        "S_w": "S_w",
        "B_w": "B_w",
        "S_over_B": "S/B",
        "asimov_Z_A": "Nominal Z_A",
        "Z_A_boot_p50": "Boot median Z_A",
        "Z_A_boot_p16": "Boot p16 Z_A",
        "Z_A_boot_p84": "Boot p84 Z_A",
        "N_eff_bkg": "N_eff bkg",
        "max_single_bkg_frac": "Max single bkg frac",
        "top_sample_frac_B_w": "Top sample frac",
    }
)

paper_md = markdown_table(paper_display)
(OUTDIR / "hh4b_model_comparison_summary.md").write_text(paper_md + "\n", encoding="utf-8")

# Plot 1: bootstrap Z_A comparison
x = np.arange(len(comparison))
labels = comparison["model"].tolist()
y = comparison["Z_A_boot_p50"].to_numpy(dtype=float)
yerr_low = y - comparison["Z_A_boot_p16"].to_numpy(dtype=float)
yerr_high = comparison["Z_A_boot_p84"].to_numpy(dtype=float) - y

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="o", capsize=5, label="Bootstrap median with 16-84% range")
ax.scatter(x, comparison["asimov_Z_A"].to_numpy(dtype=float), marker="x", label="Nominal Z_A")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Asimov Z_A")
ax.set_title("Resolved HH4b model comparison: stable working points")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(PLOTDIR / "hh4b_model_comparison_ZA_bootstrap.png", dpi=200)
plt.close(fig)

# Plot 2: S/B
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.barh(comparison["model"], comparison["S_over_B"].to_numpy(dtype=float) * 1e4)
ax.set_xlabel("S/B multiplied by 1e4")
ax.set_title("Resolved HH4b model comparison: signal purity")
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTDIR / "hh4b_model_comparison_signal_purity.png", dpi=200)
plt.close(fig)

# Plot 3: AUC comparison for ML models
auc_df = comparison.dropna(subset=["raw_auc_test", "weighted_auc_test"]).copy()
x = np.arange(len(auc_df))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.bar(x - width / 2, auc_df["raw_auc_test"], width, label="Raw test AUC")
ax.bar(x + width / 2, auc_df["weighted_auc_test"], width, label="Weighted test AUC")
ax.set_xticks(x)
ax.set_xticklabels(auc_df["model"], rotation=20, ha="right")
ax.set_ylabel("AUC")
ax.set_ylim(0.55, 0.82)
ax.set_title("Resolved HH4b model comparison: ranking performance")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(PLOTDIR / "hh4b_model_comparison_auc.png", dpi=200)
plt.close(fig)

# Plot 4: background dominance
fig, ax = plt.subplots(figsize=(9, 5))
ypos = np.arange(len(comparison))
height = 0.35
ax.barh(ypos - height / 2, comparison["max_single_bkg_frac"], height, label="Largest single background event")
ax.barh(ypos + height / 2, comparison["top_sample_frac_B_w"], height, label="Largest background sample")
ax.axvline(0.30, linestyle="--", label="Single-event criterion")
ax.axvline(0.70, linestyle=":", label="Top-sample criterion")
ax.set_yticks(ypos)
ax.set_yticklabels(comparison["model"])
ax.set_xlabel("Fraction of total weighted background")
ax.set_title("Resolved HH4b model comparison: background dominance")
ax.grid(True, axis="x", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(PLOTDIR / "hh4b_model_comparison_background_dominance.png", dpi=200)
plt.close(fig)

# Find the best model by bootstrap median Z_A among stable rows.
stable = comparison[comparison["passes_stability"]].copy()
best = stable.sort_values(["Z_A_boot_p50", "asimov_Z_A", "S_over_B"], ascending=False).iloc[0]

bdt = comparison[comparison["model"] == "BDT v2"].iloc[0]
dense = comparison[comparison["model"] == "Dense DNN v1"].iloc[0]
lbn = comparison[comparison["model"] == "LBN-DNN v2"].iloc[0]
cut = comparison[comparison["model"] == "Cut baseline"].iloc[0]

future_work = """
## Future-work idea kept for later

If time remains after the rigorous baseline comparison and SPA-Net setup, a more novel end-of-project direction is:

A hybrid boosted/resolved HH4b reconstruction network that combines SPA-Net assignment with GN2X-style boosted Hbb tagging, bJR-style mass regression, and RINO/DINO-style pretraining or regularization for QCD robustness.

Possible inputs:

- AK4 jets
- AK8 jets
- optional jet constituents/subjets/tracks if available
- global event features

Possible outputs:

- resolved H1 assignment: two AK4 jets
- resolved H2 assignment: two AK4 jets
- boosted H candidates: one AK8 jet each
- event topology class: resolved-resolved, resolved-boosted, boosted-boosted
- H candidate mass/pT regression
- event-level HH vs QCD/ttbar/Z+bb discriminant

Possible combined loss:

- assignment loss
- H mass regression loss
- topology classification loss
- event classification loss
- mass-decorrelation or anti-sculpting penalty
- optional DINO/RINO-style consistency loss across clustered views

This should remain future work until the resolved BDT/DNN/LBN-DNN/SPA-Net baselines are complete and the specific failure modes are documented.
"""

note = f"""# Resolved HH4b model comparison summary

## Purpose

This note compares the current resolved HH4b baselines under the same held-out test split and the same Poisson-bootstrap stability criteria.

The compared approaches are:

1. Cut baseline
2. BDT v2
3. Dense DNN v1
4. LBN-inspired DNN v2

The main metric is not only AUC. The main decision criterion is whether the model produces a stable analysis category with competitive S/B, Asimov Z_A, bootstrap-median Z_A, background effective statistics, and limited single-event/background-sample dominance.

## Headline result

The best stable baseline remains:

- {best['model']}
- Working point: {best['working_point']}
- Nominal Z_A: {best['asimov_Z_A']:.4f}
- Bootstrap median Z_A: {best['Z_A_boot_p50']:.4f}
- Bootstrap 16-84% Z_A range: {best['Z_A_boot_p16']:.4f}-{best['Z_A_boot_p84']:.4f}
- S/B: {best['S_over_B']:.3e}
- N_eff_bkg: {best['N_eff_bkg']:.2f}

## Paper-style comparison table

{paper_md}

## Interpretation

The cut baseline gives the lowest sensitivity, with nominal Z_A = {cut['asimov_Z_A']:.4f} and bootstrap median Z_A = {cut['Z_A_boot_p50']:.4f}.

The BDT v2 is currently the preferred baseline. Its stable working point, {bdt['working_point']}, gives nominal Z_A = {bdt['asimov_Z_A']:.4f}, bootstrap median Z_A = {bdt['Z_A_boot_p50']:.4f}, and S/B = {bdt['S_over_B']:.3e}.

The dense DNN v1 improves over the cut baseline and passes stability, but it does not beat BDT v2. Its stable working point, {dense['working_point']}, gives nominal Z_A = {dense['asimov_Z_A']:.4f}, bootstrap median Z_A = {dense['Z_A_boot_p50']:.4f}, and S/B = {dense['S_over_B']:.3e}.

The LBN-DNN v2 is a useful physics-aware model. It gives better weighted ranking behavior than the dense DNN and reaches BDT-like S/B. However, under the same stability criteria, its best stable working point, {lbn['working_point']}, gives nominal Z_A = {lbn['asimov_Z_A']:.4f} and bootstrap median Z_A = {lbn['Z_A_boot_p50']:.4f}, slightly below the BDT baseline.

The LBN-DNN has higher-score regions with larger apparent S/B and nominal Z_A, but those regions fail stability because they have low N_eff_bkg and are too sensitive to high-weight Z+bb events. Therefore, they should be treated as promising diagnostics rather than final categories.

## Recommended plots

Main plots for paper or presentation:

- outputs/hh4b_baseline/model_comparison/plots/hh4b_model_comparison_ZA_bootstrap.png
- outputs/hh4b_baseline/model_comparison/plots/hh4b_model_comparison_signal_purity.png
- outputs/hh4b_baseline/model_comparison/plots/hh4b_model_comparison_auc.png

Useful backup plot:

- outputs/hh4b_baseline/model_comparison/plots/hh4b_model_comparison_background_dominance.png

## Suggested wording

A concise result statement is:

> In the resolved HH4b baseline study, a BDT, dense DNN, and LBN-inspired DNN were compared under the same held-out test split and Poisson-bootstrap stability criteria. The dense DNN and LBN-DNN both improve over the cut baseline, but neither clearly outperforms the stable BDT working point. The LBN-DNN achieves BDT-like purity and improved weighted ranking, but its most aggressive high-score categories fail stability due to high-weight Z+bb background dominance. The BDT v2 therefore remains the preferred resolved HH4b classifier baseline before moving to assignment-aware SPA-Net reconstruction.

{future_work}
"""

note_path = NOTEDIR / "hh4b_model_comparison_summary.md"
note_path.write_text(note, encoding="utf-8")

print("Wrote:")
print(f"  {OUTDIR / 'hh4b_model_comparison_summary.csv'}")
print(f"  {OUTDIR / 'hh4b_model_comparison_summary.md'}")
print(f"  {OUTDIR / 'hh4b_model_comparison_alternative_working_points.csv'}")
print(f"  {PLOTDIR / 'hh4b_model_comparison_ZA_bootstrap.png'}")
print(f"  {PLOTDIR / 'hh4b_model_comparison_signal_purity.png'}")
print(f"  {PLOTDIR / 'hh4b_model_comparison_auc.png'}")
print(f"  {PLOTDIR / 'hh4b_model_comparison_background_dominance.png'}")
print(f"  {note_path}")

print("\n=== Comparison table ===")
print(comparison.to_string(index=False))

print("\n=== Alternative working points ===")
print(alternatives.to_string(index=False))
