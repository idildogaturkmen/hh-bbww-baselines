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
    DNN_SUMMARY_DIR / "bdt_vs_dnn_best_region_comparison.csv",
    DNN_SUMMARY_DIR / "bdt_vs_dnn_auc_comparison.csv",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

lbn_auc = pd.read_csv(LBN_DIR / "lbn_auc_summary.csv")
lbn_best = pd.read_csv(LBN_DIR / "lbn_best_stable_rectangles.csv")
lbn_yields = pd.read_csv(LBN_DIR / "lbn_category_yields.csv")

bdt_dnn_best = pd.read_csv(DNN_SUMMARY_DIR / "bdt_vs_dnn_best_region_comparison.csv")
bdt_dnn_auc = pd.read_csv(DNN_SUMMARY_DIR / "bdt_vs_dnn_auc_comparison.csv")

# Compact LBN AUC table
lbn_auc_compact = lbn_auc.copy()
for c in ["unweighted_auc", "physics_weighted_auc"]:
    lbn_auc_compact[c] = lbn_auc_compact[c].astype(float).round(4)

lbn_auc_compact.to_csv(TABLE_DIR / "lbn_auc_summary_compact.csv", index=False)
(TABLE_DIR / "lbn_auc_summary_compact.md").write_text(
    lbn_auc_compact.to_markdown(index=False) + "\n"
)

# Best stable row per LBN mode
best_rows = []
for mode in sorted(lbn_best["mode"].unique()):
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

# Category yields
lbn_yields_compact = lbn_yields.copy()
for c in ["signal_events_450fb", "background_events_450fb"]:
    lbn_yields_compact[c] = lbn_yields_compact[c].astype(float).round(3)
for c in ["S_over_B", "S_over_sqrtB", "S_over_10pctB", "neff_signal", "neff_background"]:
    lbn_yields_compact[c] = lbn_yields_compact[c].astype(float).round(6)

lbn_yields_compact.to_csv(TABLE_DIR / "lbn_category_yields_compact.csv", index=False)
(TABLE_DIR / "lbn_category_yields_compact.md").write_text(
    lbn_yields_compact.to_markdown(index=False) + "\n"
)

# All-model comparison
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

# Plot: all model S/sqrtB
plt.figure(figsize=(10, 5))
plt.bar(all_best["model"], all_best["S_over_sqrtB"])
plt.xticks(rotation=35, ha="right")
plt.ylabel("Best stable S/sqrt(B)")
plt.title("Best stable sensitivity across BDT, DNN, and LBN baselines")
plt.tight_layout()
plt.savefig(PLOT_DIR / "all_model_best_s_over_sqrtB.png", dpi=180)
plt.close()

# Plot: LBN weighted AUC
auc_pivot = lbn_auc_compact.pivot(index="mode", columns="classifier", values="physics_weighted_auc")
plt.figure(figsize=(9, 5))
x = np.arange(len(auc_pivot.index))
plt.bar(x - 0.18, auc_pivot["LBN_QCD"], width=0.36, label="QCD weighted AUC")
plt.bar(x + 0.18, auc_pivot["LBN_top"], width=0.36, label="top weighted AUC")
plt.xticks(x, auc_pivot.index, rotation=30, ha="right")
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

This note documents the lightweight LBN-style neural-network baselines for the HH→4b Delphes analysis.

The LBN-DNN uses the same BDT-v3 qcdplus candidate dataset as the BDT and ordinary DNN studies. The input dataset contains four selected candidate jet four-vectors in `[E, px, py, pz]` format.

Modes evaluated:

1. `lbn_p4_only`: four candidate jet four-vectors only.
2. `lbn_p4_plus_topology`: four-vectors plus topology-only scalar features.
3. `lbn_p4_plus_topology_btag`: four-vectors plus topology-only features plus candidate b-tag scores.
4. `lbn_p4_plus_massaware_btag`: four-vectors plus mass-aware scalar features plus candidate b-tag scores. This is an upper-bound, mass-aware LBN mode.

## Main conclusion

The LBN-DNN does not beat the BDT baseline. The best LBN mode by AUC is `lbn_p4_plus_massaware_btag`, but the best LBN mode by expected sensitivity is `lbn_p4_plus_topology`.

Adding b-tag and mass-aware auxiliary information improves the global AUC but does not improve the best stable S/sqrt(B). Therefore, further LBN optimization is not the highest-priority next step.

## LBN AUC summary

{lbn_auc_compact.to_markdown(index=False)}

## LBN best stable regions

{lbn_best_compact.to_markdown(index=False)}

## LBN fixed-category yields

{lbn_yields_compact.to_markdown(index=False)}

## All-model best-region comparison

{all_best.to_markdown(index=False)}

## Interpretation

### Four-vector-only LBN

The four-vector-only model is too weak, especially for top-background rejection. Fixed candidate jet four-vectors alone are not enough.

### LBN plus topology

Adding topology features gives the strongest LBN sensitivity. This shows that global event topology is important.

### LBN plus topology and b-tags

Adding b-tags improves the weighted AUC slightly but worsens the best stable S/sqrt(B). It does not improve the analysis-level performance.

### LBN plus mass-aware features and b-tags

This upper-bound mode gives the best LBN AUC, but not the best LBN sensitivity. It remains below the mass-aware BDT and mass-aware DNN baselines.

## Current conclusion

The current fixed-candidate LBN-DNN models do not outperform the BDT baseline. The next architecture worth trying is SPA-Net, because SPA-Net can address the jet-assignment and permutation problem directly instead of only classifying already chosen candidate jets.

## Next steps

1. Commit this final LBN summary.
2. Stop optimizing LBN for now.
3. Begin the SPA-Net dataset audit or SPA-Net training workflow.
4. In parallel, begin ZZ→4b and ZH→4b Delphes validation samples.
"""

summary_path.write_text(text)

print("Wrote", summary_path)
print("Wrote tables to", TABLE_DIR)
print("Wrote plots to", PLOT_DIR)
