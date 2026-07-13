#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import pandas as pd

STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])

NPZ_DIR = STORE / "spanet_npz/nominal_btag4_v0"
OUTDIR = REPO / "outputs/tables/spanet_npz_audit_2026_07_13"
OUTDIR.mkdir(parents=True, exist_ok=True)

files = {
    "all": NPZ_DIR / "hh4b_spanet_leading8_all.npz",
    "train": NPZ_DIR / "hh4b_spanet_leading8_train.npz",
    "val": NPZ_DIR / "hh4b_spanet_leading8_val.npz",
    "test": NPZ_DIR / "hh4b_spanet_leading8_test.npz",
}

rows = []
sample_rows = []
key_rows = []

for split, path in files.items():
    if not path.exists():
        raise FileNotFoundError(path)

    z = np.load(path, allow_pickle=True)

    for k in sorted(z.files):
        arr = z[k]
        key_rows.append({
            "split": split,
            "key": k,
            "shape": str(arr.shape) if hasattr(arr, "shape") else "",
            "dtype": str(arr.dtype) if hasattr(arr, "dtype") else type(arr).__name__,
        })

    y = z["y"]
    assignment_mask = z["assignment_mask"]
    weight_pb = z["weight_pb"]
    sample_id = z["sample_id"]
    sample_names = z["sample_names"].astype(str)
    sample_groups = z["sample_groups"].astype(str)

    rows.append({
        "split": split,
        "path": str(path),
        "n_events": len(y),
        "n_signal_y1": int((y == 1).sum()),
        "n_background_y0": int((y == 0).sum()),
        "n_assignment_valid": int((assignment_mask == 1).sum()),
        "assignment_valid_fraction": float((assignment_mask == 1).mean()),
        "sum_weight_pb": float(weight_pb.sum()),
        "X_jets_shape": str(z["X_jets"].shape),
        "feature_names": ", ".join(z["feature_names"].astype(str)),
    })

    for i, name in enumerate(sample_names):
        m = sample_id == i
        sample_rows.append({
            "split": split,
            "sample_id": i,
            "sample_name": name,
            "sample_group": sample_groups[i] if i < len(sample_groups) else "",
            "rows": int(m.sum()),
            "signal_rows": int(((y == 1) & m).sum()),
            "background_rows": int(((y == 0) & m).sum()),
            "assignment_valid_rows": int(((assignment_mask == 1) & m).sum()),
            "sum_weight_pb": float(weight_pb[m].sum()),
        })

summary = pd.DataFrame(rows)
samples = pd.DataFrame(sample_rows)
keys = pd.DataFrame(key_rows)

summary.to_csv(OUTDIR / "spanet_npz_summary.csv", index=False)
samples.to_csv(OUTDIR / "spanet_npz_by_sample.csv", index=False)
keys.to_csv(OUTDIR / "spanet_npz_keys.csv", index=False)

(OUTDIR / "spanet_npz_summary.md").write_text(summary.to_markdown(index=False) + "\n")
(OUTDIR / "spanet_npz_by_sample.md").write_text(samples.to_markdown(index=False) + "\n")
(OUTDIR / "spanet_npz_keys.md").write_text(keys.to_markdown(index=False) + "\n")

readme = OUTDIR / "README.md"
readme.write_text(
    "# SPA-Net NPZ audit\n\n"
    "This audit summarizes the existing `nominal_btag4_v0` SPA-Net NPZ files.\n\n"
    "The goal is to decide whether to train on the existing nominal SPA-Net dataset first, "
    "or rebuild a qcdplus SPA-Net dataset aligned with the BDT/DNN/LBN qcdplus baselines.\n"
)

print("Wrote", OUTDIR)
print("\n=== summary ===")
print(summary.to_string(index=False))
print("\n=== by sample ===")
print(samples.to_string(index=False))
