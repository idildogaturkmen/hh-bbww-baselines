#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

REPO = Path.cwd()

RUNS = {
    "mass_aware": REPO / "outputs/tables/bdt_v3_qcdplus_categories_2026_07_13",
    "topology_only": REPO / "outputs/tables/bdt_v3_qcdplus_topology_only_categories_2026_07_13",
}

OUTDIR = REPO / "outputs/tables/bdt_v3_mass_aware_vs_topology_only_2026_07_13"
OUTDIR.mkdir(parents=True, exist_ok=True)

rows = []

for run, d in RUNS.items():
    auc = pd.read_csv(d / "category_audit_auc_summary.csv")
    yields = pd.read_csv(d / "category_yields.csv")

    qcd_auc = auc[auc["classifier"].str.contains("QCD", case=False, na=False)].iloc[0]
    top_auc = auc[auc["classifier"].str.contains("top", case=False, na=False)].iloc[0]

    selected = yields.copy()
    selected["run"] = run
    selected["qcd_physics_weighted_auc"] = qcd_auc["physics_weighted_auc"]
    selected["top_physics_weighted_auc"] = top_auc["physics_weighted_auc"]

    rows.append(selected)

summary = pd.concat(rows, ignore_index=True)

summary = summary[
    [
        "run",
        "category",
        "qcd_physics_weighted_auc",
        "top_physics_weighted_auc",
        "n_signal_test_rows",
        "n_background_test_rows",
        "signal_events_450fb",
        "background_events_450fb",
        "S_over_B",
        "S_over_sqrtB",
        "S_over_10pctB",
        "neff_signal",
        "neff_background",
    ]
]

summary.to_csv(OUTDIR / "mass_aware_vs_topology_only_category_summary.csv", index=False)
(OUTDIR / "mass_aware_vs_topology_only_category_summary.md").write_text(
    summary.to_markdown(index=False) + "\n"
)

# Also compare inclusive CAT0+CAT1+CAT2, which corresponds to qcd>=0.8, top>=0.5.
inclusive_rows = []
for run, d in RUNS.items():
    y = pd.read_csv(d / "category_yields.csv")
    keep = y[y["category"].isin([
        "CAT0_high_purity_diagnostic",
        "CAT1_tight",
        "CAT2_medium_tight",
    ])].copy()

    s = keep["signal_events_450fb"].sum()
    b = keep["background_events_450fb"].sum()
    n_sig = keep["n_signal_test_rows"].sum()
    n_bkg = keep["n_background_test_rows"].sum()

    # Conservative: do not combine neff exactly from category-level numbers.
    # Instead report category-level neff separately in the main table.
    inclusive_rows.append({
        "run": run,
        "region": "CAT0+CAT1+CAT2",
        "n_signal_test_rows": n_sig,
        "n_background_test_rows": n_bkg,
        "signal_events_450fb": s,
        "background_events_450fb": b,
        "S_over_B": s / b if b > 0 else float("nan"),
        "S_over_sqrtB": s / (b ** 0.5) if b > 0 else float("nan"),
        "S_over_10pctB": s / (0.10 * b) if b > 0 else float("nan"),
    })

inclusive = pd.DataFrame(inclusive_rows)
inclusive.to_csv(OUTDIR / "mass_aware_vs_topology_only_inclusive_cat012.csv", index=False)
(OUTDIR / "mass_aware_vs_topology_only_inclusive_cat012.md").write_text(
    inclusive.to_markdown(index=False) + "\n"
)

print("\n=== Category comparison ===")
print(summary.to_string(index=False))

print("\n=== Inclusive CAT0+CAT1+CAT2 comparison ===")
print(inclusive.to_string(index=False))

print("\nWrote:", OUTDIR)
