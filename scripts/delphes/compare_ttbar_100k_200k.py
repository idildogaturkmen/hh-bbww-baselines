#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import numpy as np

STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])

OUTDIR = REPO / "outputs/tables/ttbar_200k_validation_2026_07_09"
OUTDIR.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "ttbar_100k": {
        "path": STORE / "parquet/ttbar_100k_merged_hh4b_candidates.parquet",
        "n_generated": 100000,
        "xsec_pb": 512.217926,
    },
    "ttbar_200k": {
        "path": STORE / "parquet/ttbar_200k_merged_hh4b_candidates.parquet",
        "n_generated": 200000,
        "xsec_pb": 512.2746276855469,
    },
}

def rhh(df):
    return np.sqrt((df["mbb1"] - 125.0)**2 + (df["mbb2"] - 125.0)**2)

REGIONS = {
    "all_candidates": lambda df: np.ones(len(df), dtype=bool),
    "rhh_lt_30": lambda df: rhh(df) < 30,
    "rhh_lt_55": lambda df: rhh(df) < 55,
    "mhh_gt_600": lambda df: df["mhh"] > 600,
    "mhh_gt_800": lambda df: df["mhh"] > 800,
    "rhh_lt_30_and_mhh_gt_400": lambda df: (rhh(df) < 30) & (df["mhh"] > 400),
    "rhh_lt_55_and_mhh_gt_400": lambda df: (rhh(df) < 55) & (df["mhh"] > 400),
}

rows = []

for sample, cfg in SAMPLES.items():
    if not cfg["path"].exists():
        print(f"Missing {sample}: {cfg['path']}")
        continue

    df = pd.read_parquet(cfg["path"])
    weight_pb = cfg["xsec_pb"] / cfg["n_generated"]

    for region, selector in REGIONS.items():
        mask = selector(df)
        n = int(mask.sum())
        eff = n / cfg["n_generated"]
        region_xsec_pb = n * weight_pb
        rel_mc_unc = 1 / np.sqrt(n) if n > 0 else np.nan

        rows.append({
            "sample": sample,
            "region": region,
            "n_generated": cfg["n_generated"],
            "candidate_rows_total": len(df),
            "rows_in_region": n,
            "eff_vs_generated": eff,
            "region_xsec_pb": region_xsec_pb,
            "expected_events_450fb": region_xsec_pb * 450000.0,
            "rel_mc_stat_unc": rel_mc_unc,
        })

out = pd.DataFrame(rows)

wide = out.pivot(index="region", columns="sample", values=["rows_in_region", "region_xsec_pb", "rel_mc_stat_unc"])
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
wide = wide.reset_index()

if "region_xsec_pb_ttbar_100k" in wide.columns and "region_xsec_pb_ttbar_200k" in wide.columns:
    wide["xsec_ratio_200k_over_100k"] = wide["region_xsec_pb_ttbar_200k"] / wide["region_xsec_pb_ttbar_100k"]

out.to_csv(OUTDIR / "ttbar_100k_200k_region_yields_long.csv", index=False)
wide.to_csv(OUTDIR / "ttbar_100k_200k_region_yields_comparison.csv", index=False)

(OUTDIR / "ttbar_100k_200k_region_yields_long.md").write_text(out.to_markdown(index=False) + "\n")
(OUTDIR / "ttbar_100k_200k_region_yields_comparison.md").write_text(wide.to_markdown(index=False) + "\n")

print("\n=== ttbar 100k vs 200k comparison ===")
print(wide.to_string(index=False))
print(f"\nWrote outputs to: {OUTDIR}")
