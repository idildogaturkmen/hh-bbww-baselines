#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np
import pandas as pd

store = Path(os.environ["HH4B_STORE"])

v1_path = store / "parquet/ttbb_50k_shard000_hh4b_candidates.parquet"
v2_path = store / "parquet/test_v2_ttbb50k_shard000_hh4b_candidates.parquet"

v1 = pd.read_parquet(v1_path)
v2 = pd.read_parquet(v2_path)

merged = v1.merge(
    v2,
    on="event",
    suffixes=("_v1", "_v2"),
    how="inner",
)

rows = []

for col in ["mbb1", "mbb2", "avg_mbb", "delta_mbb", "mhh", "drbb1", "drbb2"]:
    diff = merged[f"{col}_v2"] - merged[f"{col}_v1"]
    rows.append({
        "variable": col,
        "n_common_events": len(merged),
        "median_v1": merged[f"{col}_v1"].median(),
        "median_v2": merged[f"{col}_v2"].median(),
        "median_delta_v2_minus_v1": diff.median(),
        "mean_abs_delta": diff.abs().mean(),
        "frac_changed_gt_1e_minus_6": float((diff.abs() > 1e-6).mean()),
    })

df = pd.DataFrame(rows)

outdir = Path("outputs/tables/hh4b_reco_v2_validation_2026_07_08")
outdir.mkdir(parents=True, exist_ok=True)

df.to_csv(outdir / "ttbb50k_shard000_v1_v2_comparison.csv", index=False)
(outdir / "ttbb50k_shard000_v1_v2_comparison.md").write_text(df.to_markdown(index=False) + "\n")

print(df.to_string(index=False))

print("\nRow count check:")
print("v1 rows:", len(v1))
print("v2 rows:", len(v2))
print("common events:", len(merged))
print("events only in v1:", len(set(v1['event']) - set(v2['event'])))
print("events only in v2:", len(set(v2['event']) - set(v1['event'])))

print("\nExtra-object diagnostics in v2:")
print(v2[[
    "n_selected_jets",
    "n_selected_bjets",
    "n_extra_selected_jets",
    "n_extra_selected_bjets",
    "ht_selected_jets",
    "ht_selected_bjets",
    "h_pt_balance",
    "h_delta_r",
]].describe())
