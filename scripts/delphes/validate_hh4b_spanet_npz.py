#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_file(path):
    d = np.load(path, allow_pickle=True)
    y = d["y"]
    mask = d["jet_mask"]
    assignment_mask = d["assignment_mask"]
    weight = d["weight_pb"]
    sample_id = d["sample_id"]
    sample_names = d["sample_names"]

    rows = []
    for sid in np.unique(sample_id):
        idx = sample_id == sid
        rows.append({
            "file": path.name,
            "sample": str(sample_names[sid]),
            "events": int(idx.sum()),
            "signal_events": int((y[idx] == 1).sum()),
            "background_events": int((y[idx] == 0).sum()),
            "mean_n_jets": float(mask[idx].sum(axis=1).mean()) if idx.sum() else 0,
            "assignment_mask_events": int(assignment_mask[idx].sum()),
            "assignment_mask_fraction": float(assignment_mask[idx].mean()) if idx.sum() else 0,
            "sum_weight_pb": float(weight[idx].sum()),
            "expected_events_450fb": float(weight[idx].sum() * 450000.0),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    repo = Path(os.environ["HH4B_REPO"])
    store = Path(os.environ["HH4B_STORE"])
    data_dir = store / "spanet_npz" / args.tag
    outdir = repo / "outputs/tables" / f"hh4b_spanet_dataset_{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in ["train", "val", "test", "all"]:
        path = data_dir / f"hh4b_spanet_leading8_{split}.npz"
        if path.exists():
            rows.extend(summarize_file(path))

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "spanet_dataset_validation_summary.csv", index=False)
    (outdir / "spanet_dataset_validation_summary.md").write_text(df.to_markdown(index=False) + "\n")

    print(df.to_string(index=False))
    print("\nWrote:", outdir)


if __name__ == "__main__":
    main()
