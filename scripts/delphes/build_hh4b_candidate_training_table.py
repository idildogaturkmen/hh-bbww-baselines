#!/usr/bin/env python3
from pathlib import Path
import os
import pandas as pd

store = Path(os.environ["HH4B_STORE"])

samples = {
    "vbf_hh4b": {
        "path": store / "parquet" / "HH4b_vbf_hh4b_10k_transfer_cluster_3473183_merged_10k_hh4b_candidates.parquet",
        "label": 1,
        "process_group": "signal",
        "xsec_pb": 0.0009477308194618672,
        "n_generated": 10000,
    },
    "ggf_heft_hh4b": {
        "path": store / "parquet" / "HH4b_ggf_hh4b_10k_transfer_cluster_84730223_merged_10k_hh4b_candidates.parquet",
        "label": 1,
        "process_group": "signal",
        "xsec_pb": 0.000993955007288605,
        "n_generated": 10000,
    },
    "qcd_bbbb_presel": {
        "path": store / "parquet" / "qcd_bbbb_presel_10k_hh4b_candidates.parquet",
        "label": 0,
        "process_group": "background",
        "xsec_pb": 345.1562805175781,
        "n_generated": 10000,
    },
    "zbbbb_presel": {
        "path": store / "parquet" / "zbbbb_presel_10k_hh4b_candidates.parquet",
        "label": 0,
        "process_group": "background",
        "xsec_pb": 6.9116411209106445,
        "n_generated": 10000,
    },
    "ttbar": {
        "path": store / "parquet" / "ttbar_10k_hh4b_candidates.parquet",
        "label": 0,
        "process_group": "background",
        "xsec_pb": 512.3977661132812,
        "n_generated": 10000,
    },
}

frames = []
for name, meta in samples.items():
    p = meta["path"]
    if not p.exists():
        print("MISSING:", p)
        continue

    df = pd.read_parquet(p)
    df["process"] = name
    df["label"] = meta["label"]
    df["process_group"] = meta["process_group"]
    df["xsec_pb"] = meta["xsec_pb"]
    df["n_generated"] = meta["n_generated"]
    df["event_weight_norm"] = meta["xsec_pb"] / meta["n_generated"]
    frames.append(df)

outdir = store / "ml"
outdir.mkdir(exist_ok=True)

all_df = pd.concat(frames, ignore_index=True)
out = outdir / "hh4b_candidate_training_10k_v0.parquet"
csv_out = store / "metadata" / "hh4b_candidate_training_10k_v0_summary.csv"

all_df.to_parquet(out, index=False)

summary = (
    all_df.groupby(["process", "label", "process_group"])
    .agg(
        rows=("event", "count"),
        median_mbb1=("mbb1", "median"),
        median_mbb2=("mbb2", "median"),
        median_avg_mbb=("avg_mbb", "median"),
        median_delta_mbb=("delta_mbb", "median"),
        median_mhh=("mhh", "median"),
        xsec_pb=("xsec_pb", "median"),
    )
    .reset_index()
)
summary.to_csv(csv_out, index=False)

print("Wrote:", out)
print("Wrote:", csv_out)
print()
print(summary.to_string(index=False))
print()
print("Total rows:", len(all_df))
print("Signal rows:", int((all_df["label"] == 1).sum()))
print("Background rows:", int((all_df["label"] == 0).sum()))
