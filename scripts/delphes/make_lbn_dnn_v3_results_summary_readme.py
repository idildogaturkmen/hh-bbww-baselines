#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO = Path.cwd()

LBN_DIR = REPO / "outputs/tables/lbn_dnn_v3_qcdplus_2026_07_13"
DNN_SUMMARY_DIR = REPO / "outputs/tables/dnn_v3_qcdplus_summary_2026_07_13"
BDT_COMPARE_DIR = REPO / "outputs/tables/bdt_v3_mass_aware_vs_topology_only_2026_07_13"

SUMMARY_DIR = REPO / "outputs/summaries"
TABLE_DIR = REPO / "outputs/tables/lbn_dnn_v3_qcdplus_summary_2026_07_13"
PLOT_DIR = REPO / "outputs/plots/lbn_dnn_v3_qcdplus_summary_2026_07_13"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

required = [
    LBN_DIR / "lbn_auc_summary.csv",
    LBN_DIR / "lbn_best_stable_rectangles.csv",
    LBN_DIR / "lbn_category_yields.csv",
    BDT_COMPARE_DIR / "mass_aware_vs_topology_only_inclusive_cat012.csv",
    BDT_COMPARE_DIR / "mass_aware_vs_topology_only_category_summary.csv",
    DNN_SUMMARY_DIR / "bdt_vs_dnn_best_region_comparison.csv",
    DNN_SUMMARY_DIR / "bdt_vs_dnn_auc_comparison.csv",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

lbn_auc = pd.read_csv(LBN_DIR / "lbn_auc_summary.csv")
lbn_best = pd.read_csv(LBN_DIR / "lbn_best_stable_rectangles.csv")
lbn_yields = pd.read_csv(LBN_DIR / "lbn_category_yields.csv")

bdt_dnn_auc = pd.read_csv(DNN_SUMMARY_DIR / "bdt_vs_dnn_auc_comparison.csv")
bdt_dnn_best = pd.read_csv(DNN_SUMMARY_DIR / "bdt_vs_dnn_best_region_comparison.csv")

# LBN compact AUC table
lbn_auc_compact = lbn_auc.copy()
for c in ["unweighted_auc", "physics_weighted_auc"]:
    lbn_auc_compact[c] = lbn_auc_compact[c].astype(float).round(4)

lbn_auc_compact.to_csv(TABLE_DIR / "lbn_auc_summary_compact.csv", index=False)
(TABLE_DIR / "lbn_auc_summary_compact.md").write_text(
    lbn_auc_compact.to_markdown(index=False) + "\n"
)

# LBN best rows by mode
best_rows = []
for mode in ["lbn_p4_only", "lbn_p4_plus_topology"]:
    row = lbn_best[lbn_best["mode"] == mode].sort_values("S_over_sqrtB", ascending=False).iloc[0]
    best_rows.append(row)

lbn_best_compact = pd.DataFrame(best_rows)
for c in ["signal_events_450fb", "background_events_450fb"]:
    lbn_best_compact[c] = lbn_best_compact[c].astype(float).round(3)
for c in ["S_over_B", "S_over_sqrtB", "S_over_10pctB", "neff_signal", "neff_background"]:
    lbn_best_compact[c] = lbn_best_compact[c].astype(float).round(6)

lbn_best_compact.to_csv(TABLE_DIR / "lbn_best_region_compact.csv", index=False)
(TABLE_DIR / "lbn_best_region_compact.md").write_text(
    lbn_best_compact.to_markdown(index=False) + "\n"
)

# LBN category yields compact
lbn_yields_compact = lbn_yields.copy()
for c in ["signal_events_450fb", "background_events_450fb"]:
    lbn_yields_compact[c] = lbn_yields_compact[c].astype(float).round(3)
for c in ["S_over_B", "S_over_sqrtB", "S_over_10pctB", "neff_signal", "neff_background"]:
    lbn_yields_compact[c] = lbn_yields_compact[c].astype(float).round(6)

lbn_yields_compact.to_csv(TABLE_DIR / "lbn_category_yields_compact.csv", index=False)
(TABLE_DIR / "lbn_category_yields_compact.md").write_text(
    lbn_yields_compact.to_markdown(index=False) + "\n"
)

# All-model best-region comparison
rows = []
for _, r in bdt_dnn_best.iterrows():
    rows.append({
        "model": r["model"],
        "region": r["region"],
        "signal_events_450fb": r["signal_events_450fb"],
        "background_events_450fb": r["background_events_450fb"],
        "S_over_B": r["S_over_B"],
        "S_over_sqrtB": r["S_over_sqrtB"],
        "S_over_10pctB": r["S_over_10pctB"],
    })

for _, r in lbn_best_compact.iterrows():
    rows.append({
        "model": r["mode"],
        "region": f"best stable rectangle: qcd>={r['qcd_threshold']}, top>={r['top_threshold']}",
        "signal_events_450fb": r["signal_events_450fb"],
        "background_events_450fb": r["background_events_450fb"],
        "S_over_B": r["S_over_B"],
        "S_over_sqrtB": r["S_over_sqrtB"],
        "S_over_10pctB": r["S_over_10pctB"],
    })

all_best = pd.DataFrame(rows)
all_best = all_best.sort_values("S_over_sqrtB", ascending=False)
all_best.to_csv(TABLE_DIR / "all_model_best_region_comparison.csv", index=False)
(TABLE_DIR / "all_model_best_region_comparison.md").write_text(
    all_best.to_markdown(index=False) + "\n"
)

# Plots
plt.figure(figsize=(9, 5))
plt.bar(all_best["model"], all_best["S_over_sqrtB"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("Best stable S/sqrt(B)")
plt.title("Best stable sensitivity across BDT, DNN, and LBN baselines")
plt.tight_layout()
plt.savefig(PLOT_DIR / "all_model_best_s_over_sqrtB.png", dpi=180)
plt.close()

# LBN AUC plot
auc_pivot = lbn_auc_compact.pivot(index="mode", columns="classifier", values="physics_weighted_auc")
plt.figure(figsize=(7, 5))
x = np.arange(len(auc_pivot.index))
plt.bar(x - 0.18, auc_pivot["LBN_QCD"], width=0.36, label="QCD weighted AUC")
plt.bar(x + 0.18, auc_pivot["LBN_top"], width=0.36, label="top weighted AUC")
plt.xticks(x, auc_pivot.index, rotation=20, ha="right")
plt.ylabel("Physics-weighted AUC")
plt.title("LBN-DNN weighted AUC")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "lbn_weighted_auc.png", dpi=180)
plt.close()

summary_path = SUMMARY_DIR / "lbn_dnn_v3_qcdplus_results_2026_07_13.md"

text = f"""# LBN-DNN v3 qcdplus results summary

Date: 2026-07-13  
Branch: `delphes-hh4b-production`

## Purpose

This note documents the lightweight LBN-style neural-network baseline for the HH→4b Delphes analysis.

The LBN-DNN uses the same BDT-v3 qcdplus candidate dataset as the BDT and ordinary DNN studies. The input dataset contains four selected candidate jet four-vectors in `[E, px, py, pz]` format.

Two modes are evaluated:

1. **lbn_p4_only**
   - Uses only the four candidate jet four-vectors.
2. **lbn_p4_plus_topology**
   - Uses the four candidate jet four-vectors plus topology-only scalar auxiliary features.

## Main conclusion

The LBN-DNN does not beat the BDT baseline. The four-vector-only model is weak, especially for the top-background classifier. Adding topology features improves performance substantially, but the result remains below the BDT and ordinary mass-aware DNN baselines.

The LBN-DNN is still useful because it provides a physics-structured neural-network baseline between the plain dense DNN and SPA-Net.

## LBN AUC summary

{lbn_auc_compact.to_markdown(index=False)}

## LBN best stable regions

{lbn_best_compact.to_markdown(index=False)}

## LBN fixed-category yields

{lbn_yields_compact.to_markdown(index=False)}

## All-model best-region comparison

{all_best.to_markdown(index=False)}

## Interpretation

### lbn_p4_only

The four-vector-only model performs poorly. This suggests that the current LBN implementation and four selected candidate jet four-vectors alone do not capture enough information to compete with the tabular baselines.

### lbn_p4_plus_topology

Adding topology-only auxiliary variables improves the LBN significantly. This confirms that global event topology and candidate-level scalar features are important. However, the model still underperforms the BDT.

### Current model ordering

Using best stable S/sqrt(B), the current ordering is:

1. Mass-aware BDT-v3 qcdplus
2. Mass-aware DNN-v3 qcdplus
3. Topology-only BDT-v3 qcdplus
4. LBN-DNN p4 plus topology
5. Topology-only DNN-v3 qcdplus
6. LBN-DNN p4 only

## Next steps

1. Try one improved LBN variant with explicit b-tag auxiliary inputs.
2. If the improved LBN still does not approach BDT performance, stop optimizing LBN.
3. Move to SPA-Net as the next architecture baseline.
4. In parallel, begin ZZ→4b and ZH→4b Delphes validation samples.
"""

summary_path.write_text(text)

print("Wrote", summary_path)
print("Wrote tables to", TABLE_DIR)
print("Wrote plots to", PLOT_DIR)
