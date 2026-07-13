#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REPO = Path.cwd()
INDIR = REPO / "outputs/tables/bdt_v3_qcdplus_topology_only_categories_2026_07_13"
OUTDIR = REPO / "outputs/plots/bdt_v3_qcdplus_topology_only_categories_2026_07_13"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INDIR / "scored_test_events_bdt_v3_qcdplus_topology_only.parquet")

cats = [
    "CAT0_high_purity_diagnostic",
    "CAT1_tight",
    "CAT2_medium_tight",
    "CAT3_medium",
    "CAT4_loose",
]

# 1. Category yield bar table as CSV-friendly plot source
yields = pd.read_csv(INDIR / "category_yields.csv")
yields["S_over_sqrtB"] = yields["S_over_sqrtB"].astype(float)

plt.figure(figsize=(9, 5))
plt.bar(yields["category"], yields["S_over_sqrtB"])
plt.xticks(rotation=35, ha="right")
plt.ylabel("S/sqrt(B) at 450/fb")
plt.title("BDT-v3 qcdplus topology-only exclusive category sensitivity")
plt.tight_layout()
plt.savefig(OUTDIR / "category_s_over_sqrtB.png", dpi=180)
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(yields["category"], yields["S_over_B"])
plt.xticks(rotation=35, ha="right")
plt.ylabel("S/B")
plt.title("BDT-v3 qcdplus topology-only exclusive category purity")
plt.tight_layout()
plt.savefig(OUTDIR / "category_s_over_b.png", dpi=180)
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(yields["category"], yields["neff_background"])
plt.xticks(rotation=35, ha="right")
plt.ylabel("Effective background count")
plt.title("BDT-v3 qcdplus topology-only category MC support")
plt.tight_layout()
plt.savefig(OUTDIR / "category_neff_background.png", dpi=180)
plt.close()

# 2. Background composition stacked-style source plot using grouped bars
comp = pd.read_csv(INDIR / "category_background_composition.csv")
pivot = comp.pivot_table(
    index="category",
    columns="analysis_sample",
    values="background_events_450fb",
    aggfunc="sum",
    fill_value=0,
).reindex(cats)

ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 6))
ax.set_ylabel("Background events at 450/fb")
ax.set_title("BDT-v3 qcdplus topology-only category background composition")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUTDIR / "category_background_composition_stacked.png", dpi=180)
plt.close()

# 3. BDT score plane
plot_df = df[df["category"].isin(cats)].copy()
plt.figure(figsize=(7, 6))
for cat in cats:
    sub = plot_df[plot_df["category"] == cat]
    plt.scatter(sub["bdt_qcd_score"], sub["bdt_top_score"], s=5, alpha=0.4, label=cat)
plt.xlabel("BDT QCD score")
plt.ylabel("BDT top score")
plt.title("BDT-v3 qcdplus topology-only selected events in score plane")
plt.legend(fontsize=7, loc="lower left")
plt.tight_layout()
plt.savefig(OUTDIR / "bdt_score_plane_categories.png", dpi=180)
plt.close()

# 4. Mass distributions by category: signal vs background
mass_vars = ["mbb1", "mbb2", "avg_mbb", "mhh", "r_hh"]

for cat in cats:
    sub = df[df["category"] == cat].copy()
    if len(sub) == 0:
        continue

    for var in mass_vars:
        plt.figure(figsize=(7, 5))

        sig = sub[sub["is_signal"]]
        bkg = sub[~sub["is_signal"]]

        if var == "mhh":
            bins = np.linspace(200, 1000, 41)
        elif var == "r_hh":
            bins = np.linspace(0, 80, 41)
        else:
            bins = np.linspace(40, 220, 37)

        plt.hist(
            bkg[var],
            bins=bins,
            weights=bkg["weight_events_450fb"],
            histtype="step",
            label="background",
        )
        plt.hist(
            sig[var],
            bins=bins,
            weights=sig["weight_events_450fb"],
            histtype="step",
            label="signal",
        )

        plt.xlabel(var)
        plt.ylabel("Events at 450/fb")
        plt.title(f"{cat}: {var}")
        plt.legend()
        plt.tight_layout()
        safe_cat = cat.replace("/", "_")
        plt.savefig(OUTDIR / f"{safe_cat}_{var}.png", dpi=180)
        plt.close()

print("Wrote plots to:", OUTDIR)
