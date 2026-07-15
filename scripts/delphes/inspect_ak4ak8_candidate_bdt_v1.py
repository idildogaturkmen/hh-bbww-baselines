#!/usr/bin/env python3

from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(os.environ["HH4B_REPO"])
TABLE_DIR = REPO / "outputs/tables/ak4ak8_candidate_bdt_v1_2026_07_14"
PLOT_DIR = REPO / "outputs/plots/ak4ak8_candidate_bdt_v1_2026_07_14"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv(TABLE_DIR / "ak4ak8_candidate_bdt_seed0_predictions.csv")
summary = pd.read_csv(TABLE_DIR / "ak4ak8_candidate_bdt_multiseed_summary.csv")

seed0 = summary.loc[summary["seed"] == 0].iloc[0]
thr = float(seed0["threshold"])

pred["selected_at_seed0_best"] = pred["score"] >= thr

rows = []

for key in ["analysis_group", "analysis_sample"]:
    g = (
        pred.groupby(key)
        .agg(
            n_rows=("score", "count"),
            n_selected=("selected_at_seed0_best", "sum"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            p90_score=("score", lambda x: x.quantile(0.90)),
            p99_score=("score", lambda x: x.quantile(0.99)),
        )
        .reset_index()
    )
    g["selection_fraction"] = g["n_selected"] / g["n_rows"]
    g.insert(0, "grouping", key)
    g = g.rename(columns={key: "category"})
    rows.append(g)

out = pd.concat(rows, ignore_index=True)
out.to_csv(TABLE_DIR / "ak4ak8_candidate_bdt_seed0_score_inspection.csv", index=False)
(TABLE_DIR / "ak4ak8_candidate_bdt_seed0_score_inspection.md").write_text(out.to_markdown(index=False) + "\n")

selected = pred[pred["selected_at_seed0_best"]].copy()
sel_group = selected.groupby(["analysis_group", "analysis_sample"]).size().reset_index(name="n_selected")
sel_group.to_csv(TABLE_DIR / "ak4ak8_candidate_bdt_seed0_selected_composition.csv", index=False)
(TABLE_DIR / "ak4ak8_candidate_bdt_seed0_selected_composition.md").write_text(sel_group.to_markdown(index=False) + "\n")

# Score histogram by broad group.
plt.figure(figsize=(7, 5))
for group, df in pred.groupby("analysis_group"):
    plt.hist(df["score"], bins=40, histtype="step", density=True, label=group)
plt.axvline(thr, linestyle="--", label=f"seed0 best threshold = {thr:.3f}")
plt.xlabel("BDT score")
plt.ylabel("Normalized density")
plt.title("AK4/AK8 candidate BDT pilot: score distributions")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "ak4ak8_candidate_bdt_seed0_score_by_group.png", dpi=200)
plt.savefig(PLOT_DIR / "ak4ak8_candidate_bdt_seed0_score_by_group.pdf")
plt.close()

# Score histogram by sample.
plt.figure(figsize=(8, 5))
for sample, df in pred.groupby("analysis_sample"):
    plt.hist(df["score"], bins=40, histtype="step", density=True, label=sample)
plt.axvline(thr, linestyle="--", label=f"seed0 best threshold = {thr:.3f}")
plt.xlabel("BDT score")
plt.ylabel("Normalized density")
plt.title("AK4/AK8 candidate BDT pilot: score distributions by sample")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR / "ak4ak8_candidate_bdt_seed0_score_by_sample.png", dpi=200)
plt.savefig(PLOT_DIR / "ak4ak8_candidate_bdt_seed0_score_by_sample.pdf")
plt.close()

print("Seed 0 threshold:", thr)
print("\n=== Score inspection ===")
print(out.to_string(index=False))
print("\n=== Selected composition ===")
print(sel_group.to_string(index=False))
print("\nWrote:", TABLE_DIR)
print("Wrote:", PLOT_DIR)
