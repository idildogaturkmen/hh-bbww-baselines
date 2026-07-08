#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np
import pandas as pd

store = Path(os.environ["HH4B_STORE"])
outdir = Path("outputs/tables/hh4b_background_campaigns_2026_07_08")
outdir.mkdir(parents=True, exist_ok=True)

def add_rhh(df):
    df = df.copy()
    df["r_hh"] = np.sqrt((df["mbb1"] - 125.0)**2 + (df["mbb2"] - 120.0)**2)
    return df

def load_one_sample(tag, label):
    ev_path = store / "parquet" / f"{tag}_merged_event_summary.parquet"
    cand_path = store / "parquet" / f"{tag}_merged_hh4b_candidates.parquet"

    if not ev_path.exists() or not cand_path.exists():
        print(f"Skipping missing sample: {tag}")
        return None

    ev = pd.read_parquet(ev_path)
    cand = add_rhh(pd.read_parquet(cand_path))

    xsec = float(ev["event_cross_section_pb"].median())
    n_gen = len(ev)
    weight = xsec / n_gen

    cand["sample"] = label
    cand["event_weight_pb"] = weight
    return cand, n_gen, xsec

def load_qcd_iht_10k():
    meta_path = store / "metadata" / "qcd_bbbb_iht_slice_scan_10000.csv"
    if not meta_path.exists():
        print("Skipping QCD HT 10k: metadata not found")
        return None

    meta = pd.read_csv(meta_path)
    frames = []
    total_gen = 0
    total_xsec = 0.0

    for _, row in meta.iterrows():
        tag = row["tag"]
        cand_path = store / "parquet" / f"{tag}_hh4b_candidates.parquet"
        if not cand_path.exists():
            print(f"Missing HT slice candidate file: {cand_path}")
            continue

        cand = add_rhh(pd.read_parquet(cand_path))
        n_gen = int(row["n_generated"])
        xsec = float(row["xsec_pb"])
        weight = xsec / n_gen

        cand["sample"] = "QCD bbbb HT-sliced 10k/slice"
        cand["event_weight_pb"] = weight
        cand["slice_label"] = row["slice_label"]

        frames.append(cand)
        total_gen += n_gen
        total_xsec += xsec

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True), total_gen, total_xsec

samples = []

for tag, label in [
    ("qcd_bbbb_presel_100k", "QCD bbbb inclusive 100k"),
    ("zbbbb_presel_100k", "Zbbbb 100k"),
    ("ttbar_100k", "inclusive ttbar 100k"),
    ("ttbb_50k", "ttbb 50k diagnostic"),
]:
    out = load_one_sample(tag, label)
    if out is not None:
        samples.append((label, *out))

qcd_iht = load_qcd_iht_10k()
if qcd_iht is not None:
    samples.append(("QCD bbbb HT-sliced 10k/slice", *qcd_iht))

regions = {
    "all_candidates": lambda d: np.ones(len(d), dtype=bool),
    "rhh_lt_30": lambda d: d["r_hh"] < 30,
    "rhh_lt_55": lambda d: d["r_hh"] < 55,
    "mhh_gt_600": lambda d: d["mhh"] > 600,
    "mhh_gt_800": lambda d: d["mhh"] > 800,
    "rhh_lt_30_and_mhh_gt_400": lambda d: (d["r_hh"] < 30) & (d["mhh"] > 400),
    "rhh_lt_55_and_mhh_gt_400": lambda d: (d["r_hh"] < 55) & (d["mhh"] > 400),
    "avg_mbb_100_150": lambda d: (d["avg_mbb"] > 100) & (d["avg_mbb"] < 150),
}

rows = []

for label, cand, n_gen, xsec in samples:
    for region_name, region_fn in regions.items():
        mask = region_fn(cand)
        n_region = int(mask.sum())
        weighted_xsec = float(cand.loc[mask, "event_weight_pb"].sum())

        rows.append({
            "sample": label,
            "n_generated": n_gen,
            "xsec_pb": xsec,
            "region": region_name,
            "candidate_rows_in_region": n_region,
            "weighted_region_xsec_pb": weighted_xsec,
            "fraction_of_selected_candidates": n_region / len(cand) if len(cand) else np.nan,
            "fraction_of_total_xsec": weighted_xsec / xsec if xsec else np.nan,
        })

df = pd.DataFrame(rows)

csv_path = outdir / "background_region_yields_with_ttbb50k.csv"
md_path = outdir / "background_region_yields_with_ttbb50k.md"

df.to_csv(csv_path, index=False)
md_path.write_text(df.to_markdown(index=False) + "\n")

print(df.to_string(index=False))
print("\nWrote:", csv_path)
print("Wrote:", md_path)
