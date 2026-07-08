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

def summarize_weighted(df, weight_col, mask):
    w = df.loc[mask, weight_col].to_numpy()
    y = float(w.sum())
    err = float(np.sqrt((w*w).sum()))
    rel = err / y if y > 0 else np.nan
    neff = (y*y / (w*w).sum()) if len(w) and (w*w).sum() > 0 else 0.0
    return y, err, rel, neff, len(w)

def load_sample(tag, label):
    ev = pd.read_parquet(store / "parquet" / f"{tag}_merged_event_summary.parquet")
    cand = add_rhh(pd.read_parquet(store / "parquet" / f"{tag}_merged_hh4b_candidates.parquet"))
    xsec = float(ev["event_cross_section_pb"].median())
    n_gen = len(ev)
    cand["weight_pb"] = xsec / n_gen
    return label, cand, xsec, n_gen

def load_qcd_iht():
    meta = pd.read_csv(store / "metadata" / "qcd_bbbb_iht_slice_scan_10000.csv")
    frames = []
    total_xsec = 0.0
    total_gen = 0
    for _, row in meta.iterrows():
        tag = row["tag"]
        cand = add_rhh(pd.read_parquet(store / "parquet" / f"{tag}_hh4b_candidates.parquet"))
        xsec = float(row["xsec_pb"])
        n_gen = int(row["n_generated"])
        cand["weight_pb"] = xsec / n_gen
        frames.append(cand)
        total_xsec += xsec
        total_gen += n_gen
    return "QCD bbbb HT-sliced 10k/slice", pd.concat(frames, ignore_index=True), total_xsec, total_gen

samples = [
    load_qcd_iht(),
    load_sample("ttbar_100k", "inclusive ttbar 100k"),
    load_sample("zbbbb_presel_100k", "Zbbbb 100k"),
    load_sample("ttbb_50k", "ttbb 50k diagnostic"),
]

regions = {
    "all_candidates": lambda d: np.ones(len(d), dtype=bool),
    "rhh_lt_30": lambda d: d["r_hh"] < 30,
    "rhh_lt_55": lambda d: d["r_hh"] < 55,
    "rhh_lt_30_mhh_gt_400": lambda d: (d["r_hh"] < 30) & (d["mhh"] > 400),
    "mhh_gt_600": lambda d: d["mhh"] > 600,
    "mhh_gt_800": lambda d: d["mhh"] > 800,
}

rows = []
for label, cand, xsec, n_gen in samples:
    for region, fn in regions.items():
        mask = fn(cand)
        y, err, rel, neff, nrows = summarize_weighted(cand, "weight_pb", mask)
        rows.append({
            "sample": label,
            "region": region,
            "selected_rows": nrows,
            "weighted_xsec_pb": y,
            "mc_stat_err_pb": err,
            "relative_mc_stat_unc": rel,
            "effective_n_weighted": neff,
        })

df = pd.DataFrame(rows)
df.to_csv(outdir / "region_mc_stat_uncertainties.csv", index=False)
(outdir / "region_mc_stat_uncertainties.md").write_text(df.to_markdown(index=False) + "\n")

print(df.to_string(index=False))
