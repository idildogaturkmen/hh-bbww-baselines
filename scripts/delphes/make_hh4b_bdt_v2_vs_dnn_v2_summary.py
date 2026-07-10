#!/usr/bin/env python3

from pathlib import Path
import os
import pandas as pd
import numpy as np

REPO = Path(os.environ["HH4B_REPO"])
OUTDIR = REPO / "outputs/tables/hh4b_bdt_v2_vs_dnn_v2_summary_2026_07_10"
OUTDIR.mkdir(parents=True, exist_ok=True)

BDT = REPO / "outputs/tables/sm_normalized_hh4b_two_bdt_event_features_safe_resampling_v2_optimized_2026_07_10"
DNN = REPO / "outputs/tables/sm_normalized_hh4b_two_dnn_event_features_safe_v2_2026_07_10"

rows = []

# BDT-v2 optimized
auc = pd.read_csv(BDT / "resampling_auc_by_seed.csv")
boot = pd.read_csv(BDT / "resampling_poisson_bootstrap_summary.csv")
best_bdt = boot.sort_values("S_over_sqrtB_mean", ascending=False).iloc[0]

rows.append({
    "model": "BDT-v2 optimized",
    "qcd_weighted_auc_mean": auc["qcd_weighted_auc"].mean(),
    "qcd_weighted_auc_std": auc["qcd_weighted_auc"].std(),
    "top_weighted_auc_mean": auc["top_weighted_auc"].mean(),
    "top_weighted_auc_std": auc["top_weighted_auc"].std(),
    "best_working_point": best_bdt["working_point"],
    "qcd_threshold": best_bdt["qcd_threshold"],
    "top_threshold": best_bdt["top_threshold"],
    "S_over_sqrtB_mean": best_bdt["S_over_sqrtB_mean"],
    "S_over_sqrtB_std_or_boot": best_bdt["S_over_sqrtB_std_bootstrap"],
    "S_over_sqrtB_p16": best_bdt["S_over_sqrtB_p16"],
    "S_over_sqrtB_p50": best_bdt["S_over_sqrtB_p50"],
    "S_over_sqrtB_p84": best_bdt["S_over_sqrtB_p84"],
    "S_over_B_mean": best_bdt["S_over_B_mean"],
})

# DNN-v2
dnn_summary = pd.read_csv(DNN / "dnn_v2_summary.csv").set_index("metric")["value"]
dnn_auc = pd.read_csv(DNN / "dnn_v2_auc_by_seed.csv")
dnn_best = pd.read_csv(DNN / "dnn_v2_best_stable_per_seed.csv")

rows.append({
    "model": "DNN-v2 simple MLP",
    "qcd_weighted_auc_mean": dnn_summary["qcd_weighted_auc_mean"],
    "qcd_weighted_auc_std": dnn_summary["qcd_weighted_auc_std"],
    "top_weighted_auc_mean": dnn_summary["top_weighted_auc_mean"],
    "top_weighted_auc_std": dnn_summary["top_weighted_auc_std"],
    "best_working_point": "best_per_seed_mean",
    "qcd_threshold": dnn_best["qcd_threshold"].median(),
    "top_threshold": dnn_best["top_threshold"].median(),
    "S_over_sqrtB_mean": dnn_summary["best_per_seed_S_over_sqrtB_mean"],
    "S_over_sqrtB_std_or_boot": dnn_summary["best_per_seed_S_over_sqrtB_std"],
    "S_over_sqrtB_p16": np.nan,
    "S_over_sqrtB_p50": dnn_best["S_over_sqrtB"].median(),
    "S_over_sqrtB_p84": np.nan,
    "S_over_B_mean": dnn_summary["best_per_seed_S_over_B_mean"],
})

df = pd.DataFrame(rows)

df["relative_to_BDT_S_over_sqrtB"] = df["S_over_sqrtB_mean"] / df.loc[df["model"].eq("BDT-v2 optimized"), "S_over_sqrtB_mean"].iloc[0]

df.to_csv(OUTDIR / "bdt_v2_vs_dnn_v2_summary.csv", index=False)
(OUTDIR / "bdt_v2_vs_dnn_v2_summary.md").write_text(df.to_markdown(index=False) + "\n")

print(df.to_string(index=False))
print("\nWrote:", OUTDIR)
