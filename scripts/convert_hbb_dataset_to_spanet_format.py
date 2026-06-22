#!/usr/bin/env python3
'''
Converts the HBB dataset to the SPANet format. The output files are written to `data/spanet_hbb/` by default.
The output consists of two HDF5 files, `spanet_hbb_trainval.h5` and `spanet_hbb_test.h5`, which contain the training+validation and test splits, respectively. Each HDF5 file contains the following groups and datasets:
- `INPUTS/Jets/MASK`: A boolean mask indicating which jets are valid (shape: [n_events, n_jets]).
- `INPUTS/Jets/<feature_name>`: Datasets for each jet feature (shape: [n_events, n_jets]).
- `TARGETS/h/b1`: The index of the first b-jet in the target assignment (shape: [n_events]).
- `TARGETS/h/b2`: The index of the second b-jet in the target assignment (shape: [n_events]).  
'''
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def write_spanet_file(out_path, jets, mask, target, event_id, split, ww_mode, n_jets, reco_idx, feature_names):
    '''
    Writes the given data to an HDF5 file in the SPANet format. The metadata is written to a separate CSV file with the same name but a `.metadata.csv` extension, which contains the following columns:
- `row`: The row index in the HDF5 file (0-based).
- `event_id`: The original event ID from the input dataset.
- `split`: The split index (0 for training, 1 for validation, 2 for test).
- `ww_mode`: The WW mode of the event (0 for non-WW, 1 for WW).
- `n_jets`: The number of jets in the event.
- `target_pos_0`: The index of the first b-jet in the target assignment.
- `target_pos_1`: The index of the second b-jet in the target assignment.
    '''
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        inputs = f.create_group("INPUTS")
        jets_group = inputs.create_group("Jets")

        jets_group.create_dataset("MASK", data=mask.astype(bool), compression="gzip")

        for k, name in enumerate(feature_names):
            jets_group.create_dataset(name, data=jets[:, :, k].astype("float32"), compression="gzip")

        targets = f.create_group("TARGETS")
        h = targets.create_group("h")
        h.create_dataset("b1", data=target[:, 0].astype("int64"), compression="gzip")
        h.create_dataset("b2", data=target[:, 1].astype("int64"), compression="gzip")

    # Write metadata separately, because SPANet does not need it for training.
    meta_path = out_path.with_suffix(".metadata.csv")
    pd.DataFrame(
        {
            "row": np.arange(len(event_id)),
            "event_id": event_id,
            "split": split,
            "ww_mode": ww_mode,
            "n_jets": n_jets,
            "target_pos_0": target[:, 0],
            "target_pos_1": target[:, 1],
        }
    ).to_csv(meta_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Wrote {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/spanet_hbb_assignment/spanet_hbb_assignment.h5")
    parser.add_argument("--outdir", default="data/spanet_hbb")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as f:
        jets = f["inputs/Jets/data"][:]
        mask = f["inputs/Jets/mask"][:]
        target = f["targets/hbb"][:]
        event_id = f["metadata/event_id"][:]
        split = f["metadata/split"][:]
        ww_mode = f["metadata/ww_mode"][:]
        n_jets = f["metadata/n_jets"][:]
        reco_idx = f["metadata/reco_idx"][:]
        feature_names = json.loads(f["inputs/Jets"].attrs["features"])

    print("Loaded:")
    print("  jets:", jets.shape)
    print("  mask:", mask.shape)
    print("  target:", target.shape)
    print("  features:", feature_names)

    trainval_idx = np.where((split == 0) | (split == 1))[0]
    test_idx = np.where(split == 2)[0]

    write_spanet_file(
        outdir / "spanet_hbb_trainval.h5",
        jets[trainval_idx],
        mask[trainval_idx],
        target[trainval_idx],
        event_id[trainval_idx],
        split[trainval_idx],
        ww_mode[trainval_idx],
        n_jets[trainval_idx],
        reco_idx[trainval_idx],
        feature_names,
    )

    write_spanet_file(
        outdir / "spanet_hbb_test.h5",
        jets[test_idx],
        mask[test_idx],
        target[test_idx],
        event_id[test_idx],
        split[test_idx],
        ww_mode[test_idx],
        n_jets[test_idx],
        reco_idx[test_idx],
        feature_names,
    )

    print("\nDone.")
    print(f"train+val events: {len(trainval_idx)}")
    print(f"test events:      {len(test_idx)}")


if __name__ == "__main__":
    main()
