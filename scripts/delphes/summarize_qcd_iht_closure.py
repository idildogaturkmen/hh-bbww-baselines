#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np
import pandas as pd

store = Path(os.environ["HH4B_STORE"])

summary_path = store / "metadata" / "qcd_iht_importance_sampling_summary_2026_07_08.csv"
outdir = store / "metadata"
out = outdir / "qcd_iht_closure_metrics_2026_07_08.csv"

inclusive_event = store / "parquet" / "qcd_bbbb_presel_100k_merged_event_summary.parquet"
inclusive_cand = store / "parquet" / "qcd_bbbb_presel_100k_merged_hh4b_candidates.parquet"

slice_tags = [
    "qcd_bbbb_iht100to200_2000",
    "qcd_bbbb_iht200to400_2000",
    "qcd_bbbb_iht400to600_2000",
    "qcd_bbbb_iht600plus_2000",
]

def add_rhh(df):
    df = df.copy()
    df["r_hh"] = np.sqrt((df["mbb1"] - 125.0)**2 + (df["mbb2"] - 120.0)**2)
    return df

inc_ev = pd.read_parquet(inclusive_event)
inc = add_rhh(pd.read_parquet(inclusive_cand))
inc_xsec = float(inc_ev["event_cross_section_pb"].median())
inc_w = inc_xsec / len(inc_ev)
inc["event_weight_pb"] = inc_w

sliced_frames = []
slice_xsec_sum = 0.0
slice_gen_sum = 0

for tag in slice_tags:
    ev = pd.read_parquet(store / "parquet" / f"{tag}_event_summary.parquet")
    cand = add_rhh(pd.read_parquet(store / "parquet" / f"{tag}_hh4b_candidates.parquet"))
    xsec = float(ev["event_cross_section_pb"].median())
    w = xsec / len(ev)
    cand["sample"] = tag
    cand["event_weight_pb"] = w
    sliced_frames.append(cand)
    slice_xsec_sum += xsec
    slice_gen_sum += len(ev)

sliced = pd.concat(sliced_frames, ignore_index=True)

def weighted_yield(df, mask):
    return float(df.loc[mask, "event_weight_pb"].sum())

rows = []

regions = {
    "all_candidates": lambda d: np.ones(len(d), dtype=bool),
    "mhh_gt_600": lambda d: d["mhh"] > 600,
    "mhh_gt_800": lambda d: d["mhh"] > 800,
    "rhh_lt_30": lambda d: d["r_hh"] < 30,
    "rhh_lt_55": lambda d: d["r_hh"] < 55,
    "avg_mbb_100_150": lambda d: (d["avg_mbb"] > 100) & (d["avg_mbb"] < 150),
}

for region, fn in regions.items():
    y_inc = weighted_yield(inc, fn(inc))
    y_slice = weighted_yield(sliced, fn(sliced))
    rows.append({
        "region": region,
        "inclusive_qcd100k_yield_pb": y_inc,
        "iht_sliced_yield_pb": y_slice,
        "relative_difference": (y_slice - y_inc) / y_inc if y_inc else np.nan,
    })

# Shape closure using normalized histograms
variables = {
    "mhh": np.linspace(0, 1600, 65),
    "avg_mbb": np.linspace(0, 300, 61),
    "r_hh": np.linspace(0, 300, 61),
}

for var, bins in variables.items():
    h_inc, _ = np.histogram(inc[var], bins=bins, weights=inc["event_weight_pb"])
    h_slice, _ = np.histogram(sliced[var], bins=bins, weights=sliced["event_weight_pb"])

    h_inc_norm = h_inc / h_inc.sum() if h_inc.sum() else h_inc
    h_slice_norm = h_slice / h_slice.sum() if h_slice.sum() else h_slice

    l1_shape_distance = 0.5 * np.sum(np.abs(h_inc_norm - h_slice_norm))

    rows.append({
        "region": f"{var}_normalized_shape_l1_distance",
        "inclusive_qcd100k_yield_pb": np.nan,
        "iht_sliced_yield_pb": np.nan,
        "relative_difference": l1_shape_distance,
    })

df = pd.DataFrame(rows)
df.to_csv(out, index=False)

print(df.to_string(index=False))
print("\nWrote:", out)
