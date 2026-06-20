#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from make_hbb_bjet_regression_dataset import (
    COLUMNS,
    find_hh_files,
    event_from_batch,
    build_reco_jets,
)


FEATURE_NAMES = [
    "log_pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "log_mass",
    "btag",
    "btag_phys",
    "response_corr",
    "log_corrected_pt",
    "log_corrected_mass",
]

SPLIT_NAME_TO_CODE = {"train": 0, "val": 1, "test": 2}
SPLIT_CODE_TO_NAME = {0: "train", 1: "val", 2: "test"}

WW_MODE_TO_CODE = {
    "unknown": 0,
    "had_had": 1,
    "lep_had_e": 2,
    "lep_had_mu": 3,
    "lep_had_tau": 4,
    "lep_lep": 5,
}
WW_CODE_TO_MODE = {v: k for k, v in WW_MODE_TO_CODE.items()}


def split_from_event_id(event_id):
    mod = int(event_id) % 5
    if mod == 0:
        return "test"
    if mod == 1:
        return "val"
    return "train"


def load_targets_and_corrections(pairs_csv):
    print(f"Reading pair table from {pairs_csv}")
    pairs = pd.read_csv(pairs_csv)

    required = ["event_id", "jet_a_idx", "jet_b_idx", "is_truth_pair", "corr_a", "corr_b"]
    missing = [c for c in required if c not in pairs.columns]
    if missing:
        raise RuntimeError(f"Missing required pair-table columns: {missing}")

    truth = pairs[pairs["is_truth_pair"].astype(bool)].copy()
    if truth.empty:
        raise RuntimeError("No truth-pair rows found in pair table.")

    targets = {}
    for _, row in truth.iterrows():
        event_id = int(row["event_id"])
        targets[event_id] = (int(row["jet_a_idx"]), int(row["jet_b_idx"]))

    corr_map = {}
    for _, row in pairs.iterrows():
        event_id = int(row["event_id"])
        ja = int(row["jet_a_idx"])
        jb = int(row["jet_b_idx"])
        ca = float(row["corr_a"])
        cb = float(row["corr_b"])

        if np.isfinite(ca) and ca > 0:
            corr_map[(event_id, ja)] = ca
        if np.isfinite(cb) and cb > 0:
            corr_map[(event_id, jb)] = cb

    print(f"Pair table candidate rows: {len(pairs)}")
    print(f"Events with truth Hbb pair: {len(targets)}")
    print(f"Jet correction map entries: {len(corr_map)}")

    return targets, corr_map


def load_ww_metadata(ww_csv):
    ww_csv = Path(ww_csv)
    if not ww_csv.exists():
        print(f"WARNING: WW metadata file not found: {ww_csv}")
        return {}

    df = pd.read_csv(ww_csv, usecols=["event_id", "ww_mode"])
    out = {int(r["event_id"]): str(r["ww_mode"]) for _, r in df.iterrows()}
    print(f"WW metadata events: {len(out)}")
    return out


def jet_get(jet, names, default=0.0):
    for name in names:
        if name in jet and jet[name] is not None:
            return jet[name]
    return default


def make_feature_vector(jet, corr):
    pt = float(jet_get(jet, ["pt", "PT"], 0.0))
    eta = float(jet_get(jet, ["eta", "Eta"], 0.0))
    phi = float(jet_get(jet, ["phi", "Phi"], 0.0))
    mass = float(jet_get(jet, ["mass", "Mass"], 0.0))
    btag = float(jet_get(jet, ["btag", "BTag"], 0.0))
    btag_phys = float(jet_get(jet, ["btag_phys", "btagPhys", "BTagPhys"], 0.0))

    pt = max(pt, 0.0)
    mass = max(mass, 0.0)
    corr = float(corr)
    if not np.isfinite(corr) or corr <= 0:
        corr = 1.0

    corrected_pt = pt * corr
    corrected_mass = mass * corr

    return np.array(
        [
            np.log1p(pt),
            eta,
            np.sin(phi),
            np.cos(phi),
            np.log1p(mass),
            btag,
            btag_phys,
            corr,
            np.log1p(max(corrected_pt, 0.0)),
            np.log1p(max(corrected_mass, 0.0)),
        ],
        dtype=np.float32,
    )


def build_dataset_from_parquets(
    targets,
    corr_map,
    ww_by_event,
    n_files,
    max_jets,
    max_events,
    batch_size,
    min_reco_pt,
    max_abs_eta,
    max_reco_jets,
):
    files = find_hh_files(n_files)

    event_ids = []
    split_codes = []
    ww_codes = []
    n_jets_arr = []
    target_positions = []
    target_reco_indices = []
    reco_indices_padded = []
    jet_features = []
    jet_masks = []

    counters = {
        "raw_events_scanned": 0,
        "events_with_truth_target_in_pair_table": 0,
        "usable_events": 0,
        "skipped_not_in_targets": 0,
        "skipped_too_few_reco_jets": 0,
        "skipped_target_not_in_reco_jets": 0,
        "skipped_target_truncated_by_max_jets": 0,
        "max_jets": int(max_jets),
        "n_features": int(len(FEATURE_NAMES)),
    }

    stop = False
    global_event_id = 0

    for file_path in files:
        print(f"Reading parquet: {file_path}")
        pf = pq.ParquetFile(file_path)

        for batch in pf.iter_batches(columns=COLUMNS, batch_size=batch_size):
            cols = batch.to_pydict()
            n_rows = len(next(iter(cols.values())))

            for row_idx in range(n_rows):
                if max_events > 0 and counters["raw_events_scanned"] >= max_events:
                    stop = True
                    break

                event_id = global_event_id
                global_event_id += 1
                counters["raw_events_scanned"] += 1

                if event_id not in targets:
                    counters["skipped_not_in_targets"] += 1
                    continue

                counters["events_with_truth_target_in_pair_table"] += 1

                event = event_from_batch(cols, row_idx)
                reco_jets, n_raw_reco = build_reco_jets(
                    event,
                    max_reco_jets=max_reco_jets,
                    min_reco_pt=min_reco_pt,
                    max_abs_eta=max_abs_eta,
                )

                if len(reco_jets) < 2:
                    counters["skipped_too_few_reco_jets"] += 1
                    continue

                target_reco = tuple(int(x) for x in targets[event_id])
                reco_indices_available = {int(jet["idx"]) for jet in reco_jets}

                if target_reco[0] not in reco_indices_available or target_reco[1] not in reco_indices_available:
                    counters["skipped_target_not_in_reco_jets"] += 1
                    continue

                # Sort jets by corrected pT, using correction factors from pair table.
                def corrected_pt_for_sort(jet):
                    ridx = int(jet["idx"])
                    corr = corr_map.get((event_id, ridx), 1.0)
                    pt = float(jet_get(jet, ["pt", "PT"], 0.0))
                    return pt * corr

                reco_jets = sorted(reco_jets, key=corrected_pt_for_sort, reverse=True)

                if max_jets > 0:
                    jets_kept = reco_jets[:max_jets]
                else:
                    raise RuntimeError("For padded SPA-Net arrays, --max-jets must be positive.")

                reco_to_pos = {int(jet["idx"]): i for i, jet in enumerate(jets_kept)}

                if target_reco[0] not in reco_to_pos or target_reco[1] not in reco_to_pos:
                    counters["skipped_target_truncated_by_max_jets"] += 1
                    continue

                X = np.zeros((max_jets, len(FEATURE_NAMES)), dtype=np.float32)
                mask = np.zeros(max_jets, dtype=np.bool_)
                reco_idx_pad = -1 * np.ones(max_jets, dtype=np.int64)

                for i, jet in enumerate(jets_kept):
                    ridx = int(jet["idx"])
                    corr = corr_map.get((event_id, ridx), 1.0)
                    X[i] = make_feature_vector(jet, corr)
                    mask[i] = True
                    reco_idx_pad[i] = ridx

                target_pos = np.array(
                    [reco_to_pos[target_reco[0]], reco_to_pos[target_reco[1]]],
                    dtype=np.int64,
                )

                mode = ww_by_event.get(event_id, "unknown")
                ww_code = WW_MODE_TO_CODE.get(mode, 0)

                event_ids.append(event_id)
                split_codes.append(SPLIT_NAME_TO_CODE[split_from_event_id(event_id)])
                ww_codes.append(ww_code)
                n_jets_arr.append(len(jets_kept))
                target_positions.append(target_pos)
                target_reco_indices.append(np.array(target_reco, dtype=np.int64))
                reco_indices_padded.append(reco_idx_pad)
                jet_features.append(X)
                jet_masks.append(mask)

                counters["usable_events"] += 1

            if stop:
                break
        if stop:
            break

    arrays = {
        "event_id": np.asarray(event_ids, dtype=np.int64),
        "split": np.asarray(split_codes, dtype=np.int64),
        "ww_mode": np.asarray(ww_codes, dtype=np.int64),
        "n_jets": np.asarray(n_jets_arr, dtype=np.int64),
        "target": np.asarray(target_positions, dtype=np.int64),
        "target_reco_idx": np.asarray(target_reco_indices, dtype=np.int64),
        "reco_idx": np.asarray(reco_indices_padded, dtype=np.int64),
        "jets": np.asarray(jet_features, dtype=np.float32),
        "mask": np.asarray(jet_masks, dtype=np.bool_),
    }

    return arrays, counters


def write_h5(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        inputs = f.create_group("inputs")
        jets = inputs.create_group("Jets")
        jets.create_dataset("data", data=arrays["jets"], compression="gzip")
        jets.create_dataset("mask", data=arrays["mask"], compression="gzip")
        jets.attrs["features"] = json.dumps(FEATURE_NAMES)

        targets = f.create_group("targets")
        targets.create_dataset("hbb", data=arrays["target"], compression="gzip")
        targets.create_dataset("hbb_reco_idx", data=arrays["target_reco_idx"], compression="gzip")

        meta = f.create_group("metadata")
        meta.create_dataset("event_id", data=arrays["event_id"], compression="gzip")
        meta.create_dataset("split", data=arrays["split"], compression="gzip")
        meta.create_dataset("ww_mode", data=arrays["ww_mode"], compression="gzip")
        meta.create_dataset("n_jets", data=arrays["n_jets"], compression="gzip")
        meta.create_dataset("reco_idx", data=arrays["reco_idx"], compression="gzip")
        meta.attrs["split_code_to_name"] = json.dumps(SPLIT_CODE_TO_NAME)
        meta.attrs["ww_code_to_mode"] = json.dumps(WW_CODE_TO_MODE)


def write_summary(path, arrays, counters):
    n = len(arrays["event_id"])
    split_names = pd.Series(arrays["split"]).map(SPLIT_CODE_TO_NAME)
    ww_names = pd.Series(arrays["ww_mode"]).map(WW_CODE_TO_MODE)

    split_counts = split_names.value_counts().to_dict()
    ww_counts = ww_names.value_counts().to_dict()

    lines = []
    lines.append("# SPA-Net H→bb Assignment Dataset")
    lines.append("")
    lines.append("This dataset converts HH→bbWW events into a padded event-level jet dataset for SPA-Net-style H→bb assignment.")
    lines.append("")
    lines.append("The target is the pair of reconstructed AK4 jets matched to the two status-23 H→bb b quarks. WW decay mode is stored as metadata only.")
    lines.append("")
    lines.append("## Counters")
    lines.append("")
    for k, v in counters.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Dataset shape")
    lines.append("")
    lines.append(f"- events: {n}")
    lines.append(f"- jets: `{tuple(arrays['jets'].shape)}`")
    lines.append(f"- mask: `{tuple(arrays['mask'].shape)}`")
    lines.append(f"- target: `{tuple(arrays['target'].shape)}`")
    lines.append(f"- features: `{FEATURE_NAMES}`")
    lines.append("")
    lines.append("## Split counts")
    lines.append("")
    lines.append("| Split | Events |")
    lines.append("|---|---:|")
    for split in ["train", "val", "test"]:
        lines.append(f"| {split} | {int(split_counts.get(split, 0))} |")
    lines.append("")
    lines.append("## WW-mode metadata counts")
    lines.append("")
    lines.append("| WW mode | Events |")
    lines.append("|---|---:|")
    for mode, count in sorted(ww_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {mode} | {int(count)} |")
    lines.append("")
    if n > 0:
        lines.append("## Jet multiplicity")
        lines.append("")
        lines.append(f"- min n_jets: {int(np.min(arrays['n_jets']))}")
        lines.append(f"- median n_jets: {float(np.median(arrays['n_jets'])):.1f}")
        lines.append(f"- max n_jets: {int(np.max(arrays['n_jets']))}")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This is the first event-level dataset for comparing SPA-Net against the existing physics, BDT, and Pair-DNN H→bb assignment baselines.")
    lines.append("")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def write_event_index(path, arrays):
    rows = []
    for i in range(len(arrays["event_id"])):
        rows.append(
            {
                "row": i,
                "event_id": int(arrays["event_id"][i]),
                "split": SPLIT_CODE_TO_NAME[int(arrays["split"][i])],
                "ww_mode": WW_CODE_TO_MODE[int(arrays["ww_mode"][i])],
                "n_jets": int(arrays["n_jets"][i]),
                "target_pos_0": int(arrays["target"][i, 0]),
                "target_pos_1": int(arrays["target"][i, 1]),
                "target_reco_idx_0": int(arrays["target_reco_idx"][i, 0]),
                "target_reco_idx_1": int(arrays["target_reco_idx"][i, 1]),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def make_plots(plot_dir, arrays):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    bins = np.arange(np.min(arrays["n_jets"]), np.max(arrays["n_jets"]) + 2) - 0.5
    plt.hist(arrays["n_jets"], bins=bins)
    plt.xlabel("Number of retained jets")
    plt.ylabel("Events")
    plt.title("SPA-Net H→bb dataset jet multiplicity")
    plt.tight_layout()
    plt.savefig(plot_dir / "spanet_hbb_n_jets.png", dpi=180)
    plt.close()

    split_counts = pd.Series(arrays["split"]).map(SPLIT_CODE_TO_NAME).value_counts().reindex(["train", "val", "test"]).fillna(0)
    plt.figure(figsize=(6, 4))
    plt.bar(split_counts.index, split_counts.values)
    plt.xlabel("Split")
    plt.ylabel("Events")
    plt.title("SPA-Net H→bb split counts")
    plt.tight_layout()
    plt.savefig(plot_dir / "spanet_hbb_split_counts.png", dpi=180)
    plt.close()

    ww_counts = pd.Series(arrays["ww_mode"]).map(WW_CODE_TO_MODE).value_counts().sort_values()
    plt.figure(figsize=(7, 4))
    plt.barh(ww_counts.index, ww_counts.values)
    plt.xlabel("Events")
    plt.title("SPA-Net H→bb WW-mode metadata")
    plt.tight_layout()
    plt.savefig(plot_dir / "spanet_hbb_ww_mode_counts.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-csv", default="outputs/pairing_with_bjet_correction/all_candidate_pairs.csv")
    parser.add_argument("--ww-csv", default="outputs/hhbbww_status23_ww_targets/event_status23_ww_targets.csv")
    parser.add_argument("--outdir", default="outputs/spanet_hbb_assignment")
    parser.add_argument("--plot-dir", default="outputs/plots/spanet_hbb_assignment")
    parser.add_argument("--n-files", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--max-jets", type=int, default=20)
    parser.add_argument("--min-reco-pt", type=float, default=20.0)
    parser.add_argument("--max-abs-eta", type=float, default=2.5)
    parser.add_argument("--max-reco-jets", type=int, default=-1)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    plot_dir = Path(args.plot_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    targets, corr_map = load_targets_and_corrections(args.pairs_csv)
    ww_by_event = load_ww_metadata(args.ww_csv)

    arrays, counters = build_dataset_from_parquets(
        targets=targets,
        corr_map=corr_map,
        ww_by_event=ww_by_event,
        n_files=args.n_files,
        max_jets=args.max_jets,
        max_events=args.max_events,
        batch_size=args.batch_size,
        min_reco_pt=args.min_reco_pt,
        max_abs_eta=args.max_abs_eta,
        max_reco_jets=args.max_reco_jets,
    )

    print("\n=== Counters ===")
    for k, v in counters.items():
        print(f"{k}: {v}")

    if counters["usable_events"] == 0:
        raise RuntimeError("No usable events produced.")

    h5_path = outdir / "spanet_hbb_assignment.h5"
    npz_path = outdir / "spanet_hbb_assignment.npz"
    event_index_path = outdir / "event_index.csv"
    feature_path = outdir / "feature_names.json"
    summary_path = outdir / "RESULTS_SPANET_HBB_ASSIGNMENT_DATASET.md"

    write_h5(h5_path, arrays)
    np.savez_compressed(npz_path, **arrays)
    write_event_index(event_index_path, arrays)
    feature_path.write_text(json.dumps(FEATURE_NAMES, indent=2) + "\n")
    write_summary(summary_path, arrays, counters)

    summary_dir = Path("Results Summaries")
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_summary(summary_dir / "RESULTS_SPANET_HBB_ASSIGNMENT_DATASET.md", arrays, counters)

    make_plots(plot_dir, arrays)

    print(f"\nSaved HDF5: {h5_path}")
    print(f"Saved NPZ:  {npz_path}")
    print(f"Saved event index: {event_index_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved plots in: {plot_dir}")


if __name__ == "__main__":
    main()
