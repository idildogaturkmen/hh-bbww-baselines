#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO = Path.cwd()

DNN_DIR = REPO / "outputs/tables/dnn_v3_qcdplus_mass_aware_and_topology_only_2026_07_13"
BDT_COMPARE_DIR = REPO / "outputs/tables/bdt_v3_mass_aware_vs_topology_only_2026_07_13"

SUMMARY_DIR = REPO / "outputs/summaries"
TABLE_DIR = REPO / "outputs/tables/dnn_v3_qcdplus_summary_2026_07_13"
PLOT_DIR = REPO / "outputs/plots/dnn_v3_qcdplus_summary_2026_07_13"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

summary_path = SUMMARY_DIR / "dnn_v3_qcdplus_results_2026_07_13.md"

required = [
    DNN_DIR / "dnn_auc_summary.csv",
    DNN_DIR / "dnn_best_stable_rectangles.csv",
    DNN_DIR / "dnn_category_yields.csv",
    DNN_DIR / "dnn_category_background_composition.csv",
    DNN_DIR / "scored_test_events_dnn_v3_qcdplus_mass_aware.parquet",
    DNN_DIR / "scored_test_events_dnn_v3_qcdplus_topology_only.parquet",
    BDT_COMPARE_DIR / "mass_aware_vs_topology_only_inclusive_cat012.csv",
    BDT_COMPARE_DIR / "mass_aware_vs_topology_only_category_summary.csv",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def neff(weights):
    w = np.asarray(weights, dtype=float)
    if len(w) == 0 or np.sum(w * w) <= 0:
        return 0.0
    return float((np.sum(w) ** 2) / np.sum(w * w))


# -----------------------------
# Load DNN tables
# -----------------------------
dnn_auc = pd.read_csv(DNN_DIR / "dnn_auc_summary.csv")
dnn_best = pd.read_csv(DNN_DIR / "dnn_best_stable_rectangles.csv")
dnn_yields = pd.read_csv(DNN_DIR / "dnn_category_yields.csv")
dnn_comp = pd.read_csv(DNN_DIR / "dnn_category_background_composition.csv")

# -----------------------------
# Build BDT vs DNN AUC table
# -----------------------------
bdt_cat = pd.read_csv(BDT_COMPARE_DIR / "mass_aware_vs_topology_only_category_summary.csv")

bdt_auc_rows = []
for run in ["mass_aware", "topology_only"]:
    sub = bdt_cat[bdt_cat["run"] == run].iloc[0]
    bdt_auc_rows.append({
        "model": f"BDT {run}",
        "QCD weighted AUC": sub["qcd_physics_weighted_auc"],
        "top weighted AUC": sub["top_physics_weighted_auc"],
    })

dnn_auc_rows = []
for run in ["mass_aware", "topology_only"]:
    qcd = dnn_auc[(dnn_auc["run"] == run) & (dnn_auc["classifier"] == "DNN_QCD")].iloc[0]
    top = dnn_auc[(dnn_auc["run"] == run) & (dnn_auc["classifier"] == "DNN_top")].iloc[0]
    dnn_auc_rows.append({
        "model": f"DNN {run}",
        "QCD weighted AUC": qcd["physics_weighted_auc"],
        "top weighted AUC": top["physics_weighted_auc"],
    })

auc_compare = pd.DataFrame(bdt_auc_rows + dnn_auc_rows)
auc_compare["QCD weighted AUC"] = auc_compare["QCD weighted AUC"].astype(float).round(4)
auc_compare["top weighted AUC"] = auc_compare["top weighted AUC"].astype(float).round(4)

auc_compare.to_csv(TABLE_DIR / "bdt_vs_dnn_auc_comparison.csv", index=False)
(TABLE_DIR / "bdt_vs_dnn_auc_comparison.md").write_text(
    auc_compare.to_markdown(index=False) + "\n"
)

# -----------------------------
# Best-region comparison
# -----------------------------
bdt_inclusive = pd.read_csv(BDT_COMPARE_DIR / "mass_aware_vs_topology_only_inclusive_cat012.csv")

best_rows = []
for _, r in bdt_inclusive.iterrows():
    best_rows.append({
        "model": f"BDT {r['run']}",
        "region": r["region"],
        "signal_events_450fb": r["signal_events_450fb"],
        "background_events_450fb": r["background_events_450fb"],
        "S_over_B": r["S_over_B"],
        "S_over_sqrtB": r["S_over_sqrtB"],
        "S_over_10pctB": r["S_over_10pctB"],
        "n_signal_test_rows": r["n_signal_test_rows"],
        "n_background_test_rows": r["n_background_test_rows"],
    })

for run in ["mass_aware", "topology_only"]:
    sub = dnn_best[dnn_best["run"] == run].sort_values("S_over_sqrtB", ascending=False).iloc[0]
    best_rows.append({
        "model": f"DNN {run}",
        "region": f"best stable rectangle: qcd>={sub['qcd_threshold']}, top>={sub['top_threshold']}",
        "signal_events_450fb": sub["signal_events_450fb"],
        "background_events_450fb": sub["background_events_450fb"],
        "S_over_B": sub["S_over_B"],
        "S_over_sqrtB": sub["S_over_sqrtB"],
        "S_over_10pctB": sub["S_over_10pctB"],
        "n_signal_test_rows": sub["n_signal_test_rows"],
        "n_background_test_rows": sub["n_background_test_rows"],
    })

best_compare = pd.DataFrame(best_rows)
for c in ["signal_events_450fb", "background_events_450fb"]:
    best_compare[c] = best_compare[c].astype(float).round(3)
for c in ["S_over_B", "S_over_sqrtB", "S_over_10pctB"]:
    best_compare[c] = best_compare[c].astype(float).round(6)

best_compare.to_csv(TABLE_DIR / "bdt_vs_dnn_best_region_comparison.csv", index=False)
(TABLE_DIR / "bdt_vs_dnn_best_region_comparison.md").write_text(
    best_compare.to_markdown(index=False) + "\n"
)

# -----------------------------
# DNN fixed category table
# -----------------------------
dnn_yields_out = dnn_yields.copy()
for c in ["signal_events_450fb", "background_events_450fb"]:
    dnn_yields_out[c] = dnn_yields_out[c].astype(float).round(3)
for c in ["S_over_B", "S_over_sqrtB", "S_over_10pctB", "neff_signal", "neff_background"]:
    dnn_yields_out[c] = dnn_yields_out[c].astype(float).round(6)

dnn_yields_out.to_csv(TABLE_DIR / "dnn_fixed_category_yields.csv", index=False)
(TABLE_DIR / "dnn_fixed_category_yields.md").write_text(
    dnn_yields_out.to_markdown(index=False) + "\n"
)

# -----------------------------
# DNN mass-sculpting table from scored events
# -----------------------------
cats = [
    "CAT0_high_purity_diagnostic",
    "CAT1_tight",
    "CAT2_medium_tight",
    "CAT3_medium",
    "CAT4_loose",
]
mass_vars = ["mbb1", "mbb2", "avg_mbb", "mhh", "r_hh"]

mass_rows = []
for run in ["mass_aware", "topology_only"]:
    df = pd.read_parquet(DNN_DIR / f"scored_test_events_dnn_v3_qcdplus_{run}.parquet")
    bkg = df[~df["is_signal"]].copy()

    for cat in cats:
        sub = bkg[bkg["category"] == cat].copy()
        if len(sub) == 0:
            continue

        row = {
            "run": run,
            "category": cat,
            "background_yield_450fb": sub["weight_events_450fb"].sum(),
            "neff_background": neff(sub["weight_events_450fb"]),
        }
        for var in mass_vars:
            row[f"{var}_median"] = sub[var].median()

        row["avg_mbb_distance_from_125"] = abs(row["avg_mbb_median"] - 125.0)
        mass_rows.append(row)

dnn_mass = pd.DataFrame(mass_rows)
for c in dnn_mass.columns:
    if c not in ["run", "category"]:
        dnn_mass[c] = dnn_mass[c].astype(float).round(3)

dnn_mass.to_csv(TABLE_DIR / "dnn_background_mass_sculpting.csv", index=False)
(TABLE_DIR / "dnn_background_mass_sculpting.md").write_text(
    dnn_mass.to_markdown(index=False) + "\n"
)

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(8, 5))
x = np.arange(len(auc_compare))
plt.bar(x - 0.18, auc_compare["QCD weighted AUC"], width=0.36, label="QCD weighted AUC")
plt.bar(x + 0.18, auc_compare["top weighted AUC"], width=0.36, label="top weighted AUC")
plt.xticks(x, auc_compare["model"], rotation=25, ha="right")
plt.ylabel("Physics-weighted AUC")
plt.title("BDT vs DNN weighted AUC comparison")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "bdt_vs_dnn_weighted_auc.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(best_compare["model"], best_compare["S_over_sqrtB"])
plt.xticks(rotation=25, ha="right")
plt.ylabel("Best stable S/sqrt(B)")
plt.title("BDT vs DNN best stable sensitivity")
plt.tight_layout()
plt.savefig(PLOT_DIR / "bdt_vs_dnn_best_s_over_sqrtB.png", dpi=180)
plt.close()

plt.figure(figsize=(10, 5))
labels = dnn_yields_out["run"] + " " + dnn_yields_out["category"].str.replace("CAT", "C")
plt.bar(labels, dnn_yields_out["S_over_sqrtB"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("S/sqrt(B)")
plt.title("DNN fixed-category sensitivity")
plt.tight_layout()
plt.savefig(PLOT_DIR / "dnn_fixed_category_s_over_sqrtB.png", dpi=180)
plt.close()

# Background composition plot for DNN categories
comp = dnn_comp.copy()
comp["run_category"] = comp["run"] + " " + comp["category"]
pivot = comp.pivot_table(
    index="run_category",
    columns="analysis_sample",
    values="background_events_450fb",
    aggfunc="sum",
    fill_value=0,
)

plt.figure(figsize=(12, 6))
ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 6))
ax.set_ylabel("Background events at 450/fb")
ax.set_title("DNN category background composition")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOT_DIR / "dnn_category_background_composition.png", dpi=180)
plt.close()

# -----------------------------
# README
# -----------------------------
text = f"""# DNN-v3 qcdplus results summary

Date: 2026-07-13  
Branch: `delphes-hh4b-production`

## Purpose

This note documents the ordinary dense neural-network baseline trained after the BDT-v3 qcdplus study.

Two DNNs were trained:

1. **DNN mass-aware**
   - Uses the same 46 reconstructed features as the mass-aware BDT-v3 qcdplus baseline.
   - Includes direct candidate mass information.

2. **DNN topology-only**
   - Uses the same 33 topology-only features as the topology-only BDT-v3 qcdplus baseline.
   - Removes direct mass variables.

Both DNNs use the same enlarged-QCD samples and the same train/test split strategy as the BDT-v3 qcdplus studies.

## Main conclusion

The ordinary dense DNN does **not** outperform the BDT baseline. The BDT remains the best current tabular baseline. The DNN is still useful as a controlled neural-network baseline before moving to LBN-DNN and SPA-Net.

## BDT vs DNN AUC comparison

{auc_compare.to_markdown(index=False)}

## BDT vs DNN best-region comparison

For the BDT, the main region is CAT0+CAT1+CAT2. For the DNN, the best stable rectangle from the DNN score scan is used because DNN scores are not calibrated the same way as BDT scores.

{best_compare.to_markdown(index=False)}

## DNN fixed-category yields

These use the same numerical category definitions as the BDT category audit. They are useful for consistency checks, but the DNN best-rectangle scan is more important for performance comparison.

{dnn_yields_out.to_markdown(index=False)}

## DNN background mass-sculpting summary

{dnn_mass.to_markdown(index=False)}

## Interpretation

### DNN mass-aware

The mass-aware DNN performs better than the topology-only DNN, but worse than the mass-aware BDT. Its best stable rectangle has S/sqrt(B) around 0.154, compared with about 0.199 for the mass-aware BDT CAT0+CAT1+CAT2 region.

### DNN topology-only

The topology-only DNN is the weakest of the tabular models studied so far. This confirms that both mass information and the BDT model class are important for the current performance.

### Model hierarchy so far

Current performance ordering:

1. Mass-aware BDT-v3 qcdplus
2. Topology-only BDT-v3 qcdplus
3. Mass-aware DNN-v3 qcdplus
4. Topology-only DNN-v3 qcdplus

The DNN result supports moving next to a more physics-structured neural network, such as LBN-DNN, rather than spending too much time on a plain dense DNN.

## Plots

Generated plots are stored in:

`outputs/plots/dnn_v3_qcdplus_summary_2026_07_13`

Important plots:
- `bdt_vs_dnn_weighted_auc.png`
- `bdt_vs_dnn_best_s_over_sqrtB.png`
- `dnn_fixed_category_s_over_sqrtB.png`
- `dnn_category_background_composition.png`

## Next steps

1. Commit this DNN baseline and summary.
2. Train an LBN-DNN using the four selected jet four-vectors.
3. Compare LBN-DNN against BDT and ordinary DNN using the same metrics.
4. Move to SPA-Net only after the tabular and structured-DNN baselines are documented.
"""

summary_path.write_text(text)

print("Wrote", summary_path)
print("Wrote tables to", TABLE_DIR)
print("Wrote plots to", PLOT_DIR)
