#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path.cwd()

RUNS = {
    "mass_aware": REPO / "outputs/tables/bdt_v3_qcdplus_categories_2026_07_13/category_mass_summary.csv",
    "topology_only": REPO / "outputs/tables/bdt_v3_qcdplus_topology_only_categories_2026_07_13/category_mass_summary.csv",
}

OUTDIR = REPO / "outputs/tables/bdt_v3_mass_sculpting_comparison_2026_07_13"
OUTDIR.mkdir(parents=True, exist_ok=True)

rows = []

for run, path in RUNS.items():
    df = pd.read_csv(path)

    # Focus on background because sculpting is mostly a background-modeling issue.
    bkg = df[df["group"] == "background"].copy()

    for _, r in bkg.iterrows():
        category = r["category"]

        rows.append({
            "run": run,
            "category": category,
            "yield_450fb": r["yield_450fb"],
            "neff": r["neff"],

            "mbb1_median": r["mbb1_median"],
            "mbb2_median": r["mbb2_median"],
            "avg_mbb_median": r["avg_mbb_median"],
            "mhh_median": r["mhh_median"],
            "r_hh_median": r["r_hh_median"],

            "avg_mbb_distance_from_125": abs(r["avg_mbb_median"] - 125.0),
            "mbb1_distance_from_125": abs(r["mbb1_median"] - 125.0),
            "mbb2_distance_from_125": abs(r["mbb2_median"] - 125.0),
        })

out = pd.DataFrame(rows)

# Put the most analysis-relevant categories first.
cat_order = [
    "CAT0_high_purity_diagnostic",
    "CAT1_tight",
    "CAT2_medium_tight",
    "CAT3_medium",
    "CAT4_loose",
]
run_order = ["mass_aware", "topology_only"]

out["category"] = pd.Categorical(out["category"], categories=cat_order, ordered=True)
out["run"] = pd.Categorical(out["run"], categories=run_order, ordered=True)
out = out.sort_values(["category", "run"])

out.to_csv(OUTDIR / "background_mass_sculpting_comparison.csv", index=False)
(OUTDIR / "background_mass_sculpting_comparison.md").write_text(
    out.to_markdown(index=False) + "\n"
)

# Compact CAT0+CAT1+CAT2 comparison
compact = out[out["category"].isin(cat_order[:3])].copy()
compact.to_csv(OUTDIR / "background_mass_sculpting_CAT012.csv", index=False)
(OUTDIR / "background_mass_sculpting_CAT012.md").write_text(
    compact.to_markdown(index=False) + "\n"
)

print("\n=== Background mass sculpting comparison ===")
print(out.to_string(index=False))

print("\nWrote:", OUTDIR)
