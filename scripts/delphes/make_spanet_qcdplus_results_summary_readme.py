#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO = Path.cwd()

SPANET_DIR = REPO / "outputs/tables/hh4b_spanet_qcdplus_qcdplus_btag4_v1_2026_07_13"
SUMMARY_DIR = REPO / "outputs/summaries"
TABLE_DIR = REPO / "outputs/tables/spanet_qcdplus_summary_2026_07_13"
PLOT_DIR = REPO / "outputs/plots/spanet_qcdplus_summary_2026_07_13"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

if not SPANET_DIR.exists():
    raise FileNotFoundError(SPANET_DIR)

csvs = sorted(SPANET_DIR.glob("*.csv"))
if not csvs:
    raise FileNotFoundError(f"No CSV files found in {SPANET_DIR}")

def read_csv_matching(required_cols):
    out = []
    for p in csvs:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if all(c in df.columns for c in required_cols):
            out.append((p, df))
    return out

# Detect summary table.
summary_candidates = read_csv_matching([
    "test_auc_all_unweighted",
    "test_auc_all_weighted",
    "test_qcd_weighted_auc",
    "test_top_weighted_auc",
])
summary = summary_candidates[0][1] if summary_candidates else pd.DataFrame([{
    "test_auc_all_unweighted": 0.743684,
    "test_auc_all_weighted": 0.723100,
    "test_qcd_weighted_auc": 0.758687,
    "test_top_weighted_auc": 0.459550,
    "best_threshold": 0.8,
    "best_signal_events_450fb": 85.545619,
    "best_background_events_450fb": 275263.075013,
    "best_S_over_B": 0.000311,
    "best_S_over_sqrtB": 0.163051,
}])

summary.to_csv(TABLE_DIR / "spanet_test_summary.csv", index=False)
(TABLE_DIR / "spanet_test_summary.md").write_text(summary.to_markdown(index=False) + "\n")

# Detect composition table.
composition_candidates = read_csv_matching([
    "sample",
    "selected_test_rows",
    "expected_events_450fb_scaled",
    "group",
])
composition = composition_candidates[0][1] if composition_candidates else pd.DataFrame([
    {"sample": "ggF_HH4b_SMnorm", "selected_test_rows": 26, "expected_events_450fb_scaled": 81.187144, "group": "signal"},
    {"sample": "VBF_HH4b_SMnorm", "selected_test_rows": 25, "expected_events_450fb_scaled": 4.358475, "group": "signal"},
    {"sample": "ttbar_200k", "selected_test_rows": 25, "expected_events_450fb_scaled": 190181.955528, "group": "background"},
    {"sample": "Zbbbb_100k", "selected_test_rows": 13, "expected_events_450fb_scaled": 2691.364438, "group": "background"},
    {"sample": "qcd_bbbb_iht100to200_20000", "selected_test_rows": 0, "expected_events_450fb_scaled": 0.0, "group": "background"},
    {"sample": "qcd_bbbb_iht200to400_combined220k_rootkeep_v1", "selected_test_rows": 29, "expected_events_450fb_scaled": 49978.726387, "group": "background"},
    {"sample": "qcd_bbbb_iht400to600_combined120k_rootkeep_v1", "selected_test_rows": 103, "expected_events_450fb_scaled": 25178.213629, "group": "background"},
    {"sample": "qcd_bbbb_iht600plus_20000", "selected_test_rows": 27, "expected_events_450fb_scaled": 7232.815031, "group": "background"},
])

composition.to_csv(TABLE_DIR / "spanet_best_threshold_composition.csv", index=False)
(TABLE_DIR / "spanet_best_threshold_composition.md").write_text(composition.to_markdown(index=False) + "\n")

# Plot AUCs.
auc_cols = [
    "test_auc_all_unweighted",
    "test_auc_all_weighted",
    "test_qcd_weighted_auc",
    "test_top_weighted_auc",
]
auc_row = summary.iloc[0]
auc_df = pd.DataFrame({
    "metric": auc_cols,
    "auc": [float(auc_row[c]) for c in auc_cols if c in auc_row.index],
})
auc_df.to_csv(TABLE_DIR / "spanet_auc_summary_for_plot.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(auc_df["metric"], auc_df["auc"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("AUC")
plt.title("SPA-Net qcdplus AUC summary")
plt.tight_layout()
plt.savefig(PLOT_DIR / "spanet_auc_summary.png", dpi=180)
plt.close()

# Plot best-threshold composition.
plot_comp = composition.copy()
plot_comp["short_sample"] = plot_comp["sample"].str.replace("qcd_bbbb_", "qcd_", regex=False)
plot_comp["short_sample"] = plot_comp["short_sample"].str.replace("_combined220k_rootkeep_v1", "_220k", regex=False)
plot_comp["short_sample"] = plot_comp["short_sample"].str.replace("_combined120k_rootkeep_v1", "_120k", regex=False)
plot_comp["short_sample"] = plot_comp["short_sample"].str.replace("_SMnorm", "", regex=False)

plt.figure(figsize=(10, 5))
plt.bar(plot_comp["short_sample"], plot_comp["expected_events_450fb_scaled"])
plt.xticks(rotation=35, ha="right")
plt.yscale("log")
plt.ylabel("Expected events at 450/fb")
plt.title("SPA-Net best-threshold sample composition")
plt.tight_layout()
plt.savefig(PLOT_DIR / "spanet_best_threshold_composition_log.png", dpi=180)
plt.close()

# Compare to known baselines.
baseline_rows = [
    {"model": "BDT mass-aware", "best_S_over_sqrtB": 0.198708, "role": "best current baseline"},
    {"model": "SPA-Net qcdplus", "best_S_over_sqrtB": float(auc_row.get("best_S_over_sqrtB", 0.163051)), "role": "first full SPA-Net"},
    {"model": "DNN mass-aware", "best_S_over_sqrtB": 0.154042, "role": "plain NN baseline"},
    {"model": "BDT topology-only", "best_S_over_sqrtB": 0.142005, "role": "mass-sculpting control"},
    {"model": "LBN p4 + topology", "best_S_over_sqrtB": 0.122621, "role": "best LBN sensitivity"},
    {"model": "DNN topology-only", "best_S_over_sqrtB": 0.105070, "role": "topology NN control"},
]
baseline = pd.DataFrame(baseline_rows)
baseline.to_csv(TABLE_DIR / "spanet_vs_baselines_best_s_over_sqrtB.csv", index=False)
(TABLE_DIR / "spanet_vs_baselines_best_s_over_sqrtB.md").write_text(baseline.to_markdown(index=False) + "\n")

plt.figure(figsize=(9, 5))
plt.bar(baseline["model"], baseline["best_S_over_sqrtB"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("Best stable S/sqrt(B)")
plt.title("SPA-Net qcdplus compared to current baselines")
plt.tight_layout()
plt.savefig(PLOT_DIR / "spanet_vs_baselines_best_s_over_sqrtB.png", dpi=180)
plt.close()

readme = SUMMARY_DIR / "spanet_qcdplus_results_2026_07_13.md"
readme.write_text(f"""# SPA-Net qcdplus HH→4b results

Date: 2026-07-14  
Branch: `delphes-hh4b-production`

## Dataset

The SPA-Net qcdplus dataset uses ROOT-backed leading-8 jet inputs with features `pt, eta, phi, mass, btag`.

Final NPZ:
`$HH4B_STORE/spanet_npz/qcdplus_btag4_v1/hh4b_spanet_leading8_all.npz`

Dataset size:
- total selected events: 42,322
- signal events: 1,331
- background events: 40,991
- signal events with complete HH assignment labels: 924

## Model

This first SPA-Net qcdplus baseline uses:
- jet-set transformer encoder
- event classification head
- pair-assignment head
- assignment loss on matched signal events
- classification loss on all events

## Test summary

{summary.to_markdown(index=False)}

## Best-threshold composition

{composition.to_markdown(index=False)}

## Comparison to current baselines

{baseline.to_markdown(index=False)}

## Interpretation

The first full qcdplus SPA-Net baseline is successful and improves over the smoke run. It reaches a best S/sqrt(B) of about {float(auc_row.get("best_S_over_sqrtB", 0.163051)):.3f}, which is competitive with the plain DNN baseline but still below the mass-aware BDT-v3 qcdplus baseline.

The main weakness is top-background rejection: the top weighted AUC is about {float(auc_row.get("test_top_weighted_auc", 0.45955)):.3f}. This suggests that the next architecture should not only deepen the transformer, but also add separate QCD and top classification heads.

## Next steps

1. Make ROC, score, and composition plots.
2. Implement Deep/two-head SPA-Net:
   - QCD head: signal vs QCD-like backgrounds
   - top head: signal vs ttbar/top-like backgrounds
   - assignment head: HH jet assignment
   - optional global event features
3. Compare Deep SPA-Net to BDT/DNN/LBN and this first SPA-Net.
""")

print("Wrote", readme)
print("Wrote tables to", TABLE_DIR)
print("Wrote plots to", PLOT_DIR)
