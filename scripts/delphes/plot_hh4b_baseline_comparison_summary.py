#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

table_dir = Path("outputs/tables/hh4b_baseline_comparison_2026_07_08")
plot_dir = Path("outputs/plots/hh4b_baseline_comparison_2026_07_08")
plot_dir.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(table_dir / "cut_vs_bdt_summary.csv")
yields = pd.read_csv(table_dir / "cut_vs_bdt_process_yields.csv")
auc = pd.read_csv(table_dir / "cut_vs_bdt_auc.csv")

nice_names = {
    "cut_rhh_lt_30": "R_HH < 30",
    "cut_rhh_lt_55": "R_HH < 55",
    "cut_rhh_lt_30_mhh_gt_400": "R_HH < 30,\nmHH > 400",
    "cut_rhh_lt_55_mhh_gt_400": "R_HH < 55,\nmHH > 400",
    "bdt_all_features_best10pct_thr_0.96": "BDT all features\nthr 0.96",
    "bdt_topology_only_best10pct_thr_0.90": "BDT topology only\nthr 0.90",
    "bdt_all_features": "BDT all features",
    "bdt_topology_only": "BDT topology only",
}

summary["selection_pretty"] = summary["selection"].map(nice_names).fillna(summary["selection"])
yields["selection_pretty"] = yields["selection"].map(nice_names).fillna(yields["selection"])
auc["model_pretty"] = auc["model"].map(nice_names).fillna(auc["model"])

# 1. AUC comparison
plt.figure(figsize=(6, 4))
plt.bar(auc["model_pretty"], auc["weighted_auc"])
plt.ylabel("Weighted ROC AUC")
plt.title("HH4b BDT discrimination")
plt.ylim(0.5, 1.0)
plt.xticks(rotation=15, ha="right")
for i, v in enumerate(auc["weighted_auc"]):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig(plot_dir / "presentation_auc_comparison.png", dpi=200)
plt.close()

# 2. S/B comparison
plt.figure(figsize=(9, 4.8))
plt.bar(summary["selection_pretty"], summary["s_over_b"])
plt.yscale("log")
plt.ylabel("S/B")
plt.title("Signal-to-background ratio by selection")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(plot_dir / "presentation_s_over_b_by_selection.png", dpi=200)
plt.close()

# 3. S/sqrt(B) and systematic-aware comparison
plt.figure(figsize=(9, 4.8))
x = np.arange(len(summary))
width = 0.28
plt.bar(x - width, summary["s_over_sqrt_b"], width, label="stat only")
plt.bar(x, summary["s_over_sqrt_b_10pct_bkg_syst"], width, label="10% bkg syst")
plt.bar(x + width, summary["s_over_sqrt_b_20pct_bkg_syst"], width, label="20% bkg syst")
plt.yscale("log")
plt.ylabel("Approx. significance metric")
plt.title("Cut baseline vs BDT baseline")
plt.xticks(x, summary["selection_pretty"], rotation=25, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "presentation_significance_metrics_by_selection.png", dpi=200)
plt.close()

# 4. Expected S and B event counts
plt.figure(figsize=(9, 4.8))
x = np.arange(len(summary))
plt.bar(x - 0.18, summary["s_events_450fb"], 0.36, label="Signal")
plt.bar(x + 0.18, summary["b_events_450fb"], 0.36, label="Background")
plt.yscale("log")
plt.ylabel("Expected events at 450 fb$^{-1}$")
plt.title("Selected signal and background yields")
plt.xticks(x, summary["selection_pretty"], rotation=25, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "presentation_signal_background_event_counts.png", dpi=200)
plt.close()

# 5. Process-by-process selected background/signal yield
pivot = yields.pivot_table(
    index="selection_pretty",
    columns="label",
    values="selected_events_450fb",
    aggfunc="sum",
    fill_value=0.0,
)

# Keep the same order as summary
pivot = pivot.reindex(summary["selection_pretty"])

plt.figure(figsize=(10, 5.2))
bottom = np.zeros(len(pivot))
for col in pivot.columns:
    vals = pivot[col].values
    plt.bar(pivot.index, vals, bottom=bottom, label=col)
    bottom += vals
plt.yscale("log")
plt.ylabel("Expected events at 450 fb$^{-1}$")
plt.title("Process composition after each selection")
plt.xticks(rotation=25, ha="right")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(plot_dir / "presentation_process_composition_by_selection.png", dpi=200)
plt.close()

# 6. Background-only process composition
bkg_cols = [c for c in pivot.columns if "HH4b" not in c]
plt.figure(figsize=(10, 5.2))
bottom = np.zeros(len(pivot))
for col in bkg_cols:
    vals = pivot[col].values
    plt.bar(pivot.index, vals, bottom=bottom, label=col)
    bottom += vals
plt.yscale("log")
plt.ylabel("Expected background events at 450 fb$^{-1}$")
plt.title("Background composition after each selection")
plt.xticks(rotation=25, ha="right")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(plot_dir / "presentation_background_composition_by_selection.png", dpi=200)
plt.close()

# 7. Compact table for slides/Harvey
compact = summary[[
    "selection",
    "s_events_450fb",
    "b_events_450fb",
    "s_over_b",
    "s_over_sqrt_b",
    "s_over_sqrt_b_10pct_bkg_syst",
    "s_over_sqrt_b_20pct_bkg_syst",
]].copy()

compact["selection"] = compact["selection"].map(nice_names).fillna(compact["selection"])

compact.to_csv(table_dir / "presentation_cut_vs_bdt_compact_summary.csv", index=False)
(table_dir / "presentation_cut_vs_bdt_compact_summary.md").write_text(
    compact.to_markdown(index=False) + "\n"
)

print("Wrote presentation plots to:", plot_dir)
for p in sorted(plot_dir.glob("presentation_*.png")):
    print(" ", p)

print("\nCompact summary:")
print(compact.to_string(index=False))
