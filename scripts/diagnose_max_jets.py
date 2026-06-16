import argparse
import json
import math
from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download


REPO_ID = "fastmachinelearning/collide-1m"

DATA_FILES = {
    "train": [
        "HH_bbWW/HH_bbWW-NEVENT10000-RS23000001.parquet",
        "HH_bbWW/HH_bbWW-NEVENT10000-RS23000002.parquet",
    ]
}

DEFAULT_CAPS = [6, 8, 10, 12, 14, 16, 20, 24]
OUTDIR = Path("outputs/max_jets_diagnostics")
CACHE_DIR = Path("outputs/dataset_cache/collide_1m")

NEEDED_COLUMNS = [
    "FullReco_GenPart_PID",
    "FullReco_GenPart_Status",
    "FullReco_GenPart_PT",
    "FullReco_GenPart_Eta",
    "FullReco_GenPart_Phi",
    "FullReco_JetAK4_PT",
    "FullReco_JetAK4_Eta",
    "FullReco_JetAK4_Phi",
]


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi <= -math.pi:
        dphi += 2 * math.pi
    return dphi


def delta_r(eta1, phi1, eta2, phi2):
    return math.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2)


def find_hbb_bquarks(event):
    candidates = []
    for i, pid in enumerate(event["FullReco_GenPart_PID"]):
        if abs(pid) == 5 and event["FullReco_GenPart_Status"][i] == 23:
            candidates.append(i)

    candidates = sorted(
        candidates,
        key=lambda i: event["FullReco_GenPart_PT"][i],
        reverse=True,
    )
    return candidates[:2]


def match_bquarks_to_jets(event, b_indices, max_dr=0.4):
    matches = []

    for bidx in b_indices:
        b_eta = event["FullReco_GenPart_Eta"][bidx]
        b_phi = event["FullReco_GenPart_Phi"][bidx]

        best_j = None
        best_dr = 999.0

        for j in range(len(event["FullReco_JetAK4_PT"])):
            dr = delta_r(
                b_eta,
                b_phi,
                event["FullReco_JetAK4_Eta"][j],
                event["FullReco_JetAK4_Phi"][j],
            )
            if dr < best_dr:
                best_dr = dr
                best_j = j

        if best_dr < max_dr:
            matches.append(best_j)
        else:
            matches.append(-1)

    return matches


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {}

    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    return {f"p{p}": float(np.percentile(values, p)) for p in percentiles}


def iter_streaming_events(n_events):
    ds = load_dataset(
        REPO_ID,
        data_files=DATA_FILES,
        split="train",
        streaming=True,
    )
    yield from (islice(ds, n_events) if n_events is not None else ds)


def iter_downloaded_events(n_events, cache_dir):
    yielded = 0
    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename in DATA_FILES["train"]:
        print(f"Downloading/caching {filename}...", flush=True)
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            cache_dir=str(cache_dir),
        )

        parquet_file = pq.ParquetFile(local_path)
        for batch in parquet_file.iter_batches(batch_size=256, columns=NEEDED_COLUMNS):
            columns = batch.to_pydict()
            for row in range(batch.num_rows):
                yield {name: columns[name][row] for name in NEEDED_COLUMNS}
                yielded += 1
                if n_events is not None and yielded >= n_events:
                    return


def scan_events(caps, n_events, source, cache_dir):
    if source == "download":
        events = iter_downloaded_events(n_events, cache_dir)
    else:
        events = iter_streaming_events(n_events)

    raw_n_jets_all = []
    raw_n_jets_matchable = []
    max_match_indices = []
    kept_by_cap = {cap: [] for cap in caps}

    n_scanned = 0
    n_with_two_bquarks = 0
    n_matchable = 0
    n_unmatched_or_merged = 0

    for event in events:
        n_scanned += 1
        if n_scanned % 1000 == 0:
            print(f"Scanned {n_scanned}; matchable H->bb events: {n_matchable}", flush=True)

        raw_n_jets = len(event["FullReco_JetAK4_PT"])
        raw_n_jets_all.append(raw_n_jets)

        b_indices = find_hbb_bquarks(event)
        if len(b_indices) != 2:
            continue

        n_with_two_bquarks += 1
        matches = match_bquarks_to_jets(event, b_indices)
        if len(matches) != 2 or matches[0] < 0 or matches[1] < 0 or matches[0] == matches[1]:
            n_unmatched_or_merged += 1
            continue

        n_matchable += 1
        raw_n_jets_matchable.append(raw_n_jets)
        max_match = max(matches)
        max_match_indices.append(max_match)

        for cap in caps:
            if max_match < cap:
                kept_by_cap[cap].append(raw_n_jets)

    rows = []
    for cap in caps:
        kept_raw_n_jets = np.asarray(kept_by_cap[cap], dtype=np.float64)
        kept_events = int(len(kept_raw_n_jets))
        pair_candidates = cap * (cap - 1) // 2

        rows.append(
            {
                "max_jets": cap,
                "pair_candidates": pair_candidates,
                "kept_events": kept_events,
                "dropped_matchable_events": int(n_matchable - kept_events),
                "kept_fraction_of_scanned": kept_events / n_scanned if n_scanned else float("nan"),
                "kept_fraction_of_matchable": kept_events / n_matchable if n_matchable else float("nan"),
                "mean_raw_n_jets_kept": float(np.mean(kept_raw_n_jets)) if kept_events else float("nan"),
                "median_raw_n_jets_kept": float(np.median(kept_raw_n_jets)) if kept_events else float("nan"),
            }
        )

    summary = {
        "scanned_events": n_scanned,
        "events_with_two_hbb_bquarks": n_with_two_bquarks,
        "unmatched_or_merged_hbb_events": n_unmatched_or_merged,
        "matchable_hbb_events_before_jet_cap": n_matchable,
        "raw_n_jets_all_events": percentile_summary(raw_n_jets_all),
        "raw_n_jets_matchable_hbb_events": percentile_summary(raw_n_jets_matchable),
        "max_matched_jet_index": percentile_summary(max_match_indices),
        "candidate_caps": caps,
    }

    return pd.DataFrame(rows), summary, raw_n_jets_all, raw_n_jets_matchable


def save_plots(df, raw_n_jets_all, raw_n_jets_matchable, outdir):
    bins_max = max(
        max(raw_n_jets_all, default=0),
        max(raw_n_jets_matchable, default=0),
        int(df["max_jets"].max()),
    )
    bins = np.arange(-0.5, bins_max + 1.5, 1)

    plt.figure()
    plt.hist(raw_n_jets_all, bins=bins, histtype="step", linewidth=1.4, label="all scanned HH events")
    plt.hist(
        raw_n_jets_matchable,
        bins=bins,
        histtype="step",
        linewidth=1.4,
        label="truth-matchable H->bb events",
    )
    for cap in df["max_jets"]:
        plt.axvline(cap, color="gray", alpha=0.25, linewidth=0.8)
    plt.xlabel("Raw AK4 jet multiplicity")
    plt.ylabel("Events")
    plt.title("Raw HH->bbWW AK4 jet multiplicity before MAX_JETS cap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "raw_jet_multiplicity.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(df["max_jets"], df["kept_fraction_of_matchable"], marker="o")
    plt.xlabel("MAX_JETS")
    plt.ylabel("Fraction of truth-matchable H->bb events retained")
    plt.title("H->bb truth-pair retention vs jet cap")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(outdir / "retention_vs_max_jets.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(df["max_jets"], df["pair_candidates"], marker="o")
    plt.xlabel("MAX_JETS")
    plt.ylabel("Unordered jet pairs per event")
    plt.title("Pair-assignment combinatorics vs jet cap")
    plt.tight_layout()
    plt.savefig(outdir / "pair_candidates_vs_max_jets.png", dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan raw COLLIDE-1M HH->bbWW events to choose a data-informed "
            "MAX_JETS cap for fixed-size H->bb reconstruction inputs."
        )
    )
    parser.add_argument(
        "--caps",
        nargs="+",
        type=int,
        default=DEFAULT_CAPS,
        help="Candidate MAX_JETS values to evaluate.",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=None,
        help="Optional number of raw events to scan. Default scans all configured files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=OUTDIR,
        help="Directory for CSV, JSON, and plots.",
    )
    parser.add_argument(
        "--source",
        choices=["streaming", "download"],
        default="streaming",
        help=(
            "Use Hugging Face streaming range reads, or download/cache the parquet "
            "first and scan it from local disk. Download mode is slower initially "
            "but more robust to streaming timeout errors."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help="Cache directory used with --source download.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    caps = sorted(set(args.caps))
    df, summary, raw_n_jets_all, raw_n_jets_matchable = scan_events(
        caps,
        args.n_events,
        args.source,
        args.cache_dir,
    )

    df.to_csv(outdir / "max_jets_scan.csv", index=False)
    with open(outdir / "max_jets_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    save_plots(df, raw_n_jets_all, raw_n_jets_matchable, outdir)

    print("\nMAX_JETS scan:")
    print(df.to_string(index=False))
    print(f"\nSaved diagnostics to {outdir}")


if __name__ == "__main__":
    main()
