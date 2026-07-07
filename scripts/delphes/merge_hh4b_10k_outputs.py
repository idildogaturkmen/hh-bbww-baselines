#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import pandas as pd


def default_store() -> Path:
    return Path(os.environ.get("HH4B_STORE", f"/uscms_data/d3/{os.environ['USER']}/hh4b_delphes"))


def shard_id(path: Path) -> int:
    match = re.search(r"_shard_(\d+)_\d+_", path.name)
    if not match:
        raise ValueError(f"Cannot parse shard id from {path.name}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge and summarize HH4b Condor transfer outputs."
    )
    parser.add_argument("--store", default=str(default_store()), help="Submit-side HH4B store")
    parser.add_argument("--campaign", default="vbf_hh4b_10k_transfer")
    parser.add_argument("--cluster", required=True, help="Condor cluster id")
    parser.add_argument("--expected-shards", type=int, default=10)
    parser.add_argument("--events-per-shard", type=int, default=1000)
    parser.add_argument("--merged-label", default="merged_10k")
    args = parser.parse_args()

    store = Path(args.store)
    root_dir = store / "root"
    parquet_dir = store / "parquet"
    metadata_dir = store / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    base = f"HH4b_{args.campaign}_cluster_{args.cluster}_shard_*_{args.events_per_shard}"
    root_files = sorted(root_dir.glob(f"{base}_pythia8_delphes.root"), key=shard_id)
    event_files = sorted(parquet_dir.glob(f"{base}_event_summary.parquet"), key=shard_id)
    candidate_files = sorted(parquet_dir.glob(f"{base}_hh4b_candidates.parquet"), key=shard_id)

    print(f"ROOT files: {len(root_files)}")
    for path in root_files:
        print(f"  {path}")
    print(f"event_summary parquet files: {len(event_files)}")
    for path in event_files:
        print(f"  {path}")
    print(f"candidate parquet files: {len(candidate_files)}")
    for path in candidate_files:
        print(f"  {path}")

    if len(root_files) != args.expected_shards:
        raise SystemExit(f"ERROR: expected {args.expected_shards} ROOT files, found {len(root_files)}")
    if len(event_files) != args.expected_shards:
        raise SystemExit(
            f"ERROR: expected {args.expected_shards} event_summary parquet files, found {len(event_files)}"
        )
    if len(candidate_files) != args.expected_shards:
        raise SystemExit(
            f"ERROR: expected {args.expected_shards} candidate parquet files, found {len(candidate_files)}"
        )

    event_summary = pd.concat((pd.read_parquet(path) for path in event_files), ignore_index=True)
    candidates = pd.concat((pd.read_parquet(path) for path in candidate_files), ignore_index=True)

    out_prefix = f"HH4b_{args.campaign}_cluster_{args.cluster}_{args.merged_label}"
    event_out = parquet_dir / f"{out_prefix}_event_summary.parquet"
    candidate_out = parquet_dir / f"{out_prefix}_hh4b_candidates.parquet"
    text_out = metadata_dir / f"{out_prefix}_summary.txt"

    event_summary.to_parquet(event_out, index=False)
    candidates.to_parquet(candidate_out, index=False)

    total_events = len(event_summary)
    events_ge4_jets = int((event_summary["n_jet_pt30_eta25"] >= 4).sum())
    events_ge4_bjets = int((event_summary["n_bjet_pt30_eta25"] >= 4).sum())
    candidate_events = len(candidates)

    describe_columns = ["mbb1", "mbb2", "avg_mbb", "delta_mbb", "mhh", "n_selected_bjets"]
    describe = candidates[describe_columns].describe() if candidate_events else pd.DataFrame()
    optional_medians = []
    for column in ("event_weight", "event_cross_section_pb"):
        if column in event_summary.columns:
            optional_medians.append(f"median {column}: {event_summary[column].median()}")

    lines = [
        f"campaign: {args.campaign}",
        f"cluster: {args.cluster}",
        f"number of ROOT files: {len(root_files)}",
        f"number of event_summary parquet files: {len(event_files)}",
        f"number of candidate parquet files: {len(candidate_files)}",
        f"total events: {total_events}",
        f"events with >=4 selected jets: {events_ge4_jets}",
        f"events with >=4 selected b-tagged jets: {events_ge4_bjets}",
        f"candidate events: {candidate_events}",
        "",
        "describe() for mbb1, mbb2, avg_mbb, delta_mbb, mhh, n_selected_bjets:",
        describe.to_string(),
        "",
        *optional_medians,
        "",
        f"merged event summary: {event_out}",
        f"merged candidates: {candidate_out}",
    ]

    text = "\n".join(lines) + "\n"
    text_out.write_text(text)
    print()
    print(text)
    print(f"summary text: {text_out}")


if __name__ == "__main__":
    main()
