"""
Build a local COLLIDE-1M inventory.

This script scans the locally cached fastmachinelearning/collide-1m snapshot,
summarizes all sample directories and parquet files, and makes process-group
guesses without claiming that missing processes are absent.

Outputs:
  outputs/collide_inventory/collide_file_inventory.csv
  outputs/collide_inventory/collide_sample_summary.csv
  outputs/collide_inventory/collide_process_group_summary.csv
  outputs/collide_inventory/collide_inventory.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


GROUP_PATTERNS = OrderedDict(
    [
        ("signal_HH_bbWW", r"^HH_bbWW$"),
        ("signal_HH_other", r"^HH_"),
        ("ttbar", r"^(tt0123j|TT)"),
        ("ttV_ttH_tttt", r"^(ttW|ttZ|ttH|tttt)"),
        ("single_top_candidate", r"(^ST_|single.?top|tW|t-channel|s-channel|tZq|tHq)"),
        ("Wjets", r"^WJets"),
        ("DY_Zjets", r"^(DYJets|ZJets)"),
        ("diboson", r"^(WW|WZ|ZZ)_"),
        ("triboson", r"^VVV"),
        ("single_higgs", r"^(ggH|VBFH|VH_)"),
        ("QCD", r"^QCD"),
        ("gamma", r"^gamma"),
        ("minbias", r"^minbias"),
        ("upsilon", r"^upsilon"),
        ("other", r".*"),
    ]
)


def guess_group(sample: str) -> str:
    s = sample.lower()

    # Signal and other HH modes
    if sample == "HH_bbWW":
        return "signal_HH_bbWW"
    if sample.startswith("HH_"):
        return "HH_other"

    # Top backgrounds
    if sample.startswith("tt0123j_"):
        return "ttbar"
    if sample in {"ttH_incl", "ttW_incl", "ttZ_incl", "tttt_incl"}:
        return "ttV_ttH_tttt"

    # Single Higgs
    if (
        sample.startswith("ggH")
        or sample.startswith("VBFH")
        or sample.startswith("VH")
    ):
        return "single_higgs"

    # Vector boson + jets
    if sample.startswith("WJets"):
        return "Wjets"
    if sample.startswith("DYJets") or sample.startswith("ZJets"):
        return "DY_Zjets"

    # Diboson / triboson
    if sample.startswith(("WW_", "WZ_", "ZZ_")):
        return "diboson"
    if sample.startswith("VVV"):
        return "triboson"

    # Other backgrounds
    if sample.startswith("QCD"):
        return "QCD"
    if sample.startswith("gamma"):
        return "gamma"
    if sample.startswith("minbias"):
        return "minbias"
    if sample.startswith("upsilon"):
        return "upsilon"

    return "other"


def looks_like_sample_root(path: Path) -> bool:
    """Return True if path looks like a directory containing COLLIDE sample folders."""
    expected_samples = [
        "HH_bbWW",
        "DYJetsToLL_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_13TeV-madgraphMLM-pythia8",
        "tt0123j_5f_ckm_LO_MLM_semiLeptonic",
    ]
    return any((path / name).is_dir() for name in expected_samples)


def find_snapshot_root(user_root: str | None) -> Path:
    if user_root:
        root = Path(user_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        return root

    candidates = [
        Path("outputs/collide_selected_backgrounds"),
        Path("/workspace/hh-spanet-surf/repos/hh-bbww-baselines/outputs/collide_selected_backgrounds"),
        Path("outputs/dataset_cache/collide_1m/datasets--fastmachinelearning--collide-1m/snapshots"),
        Path("/workspace/hh-spanet-surf/repos/hh-bbww-baselines/outputs/dataset_cache/collide_1m/datasets--fastmachinelearning--collide-1m/snapshots"),
        Path("/workspace/hh-spanet-surf/repos/hh-bbww-baselines/outputs/dataset_cache/collide_1m"),
        Path("/workspace"),
    ]

    for c in candidates:
        c = c.expanduser()
        if not c.exists():
            continue

        # First check whether this directory itself contains sample folders.
        # This must happen BEFORE the snapshot-hash logic, because many sample
        # names are longer than 20 characters.
        if looks_like_sample_root(c):
            return c.resolve()

        # If this is the HF snapshots directory, descend into the snapshot hash.
        snapshot_dirs = [
            p for p in c.iterdir()
            if p.is_dir() and looks_like_sample_root(p)
        ]
        if snapshot_dirs:
            return sorted(
                snapshot_dirs,
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[0].resolve()

    raise FileNotFoundError(
        "Could not auto-locate COLLIDE-1M cache. Pass --root /path/to/snapshot."
    )

def parquet_info(path: Path) -> dict:
    try:
        pf = pq.ParquetFile(path)
        meta = pf.metadata
        schema_names = pf.schema.names
        return {
            "n_rows": int(meta.num_rows) if meta is not None else None,
            "n_row_groups": int(meta.num_row_groups) if meta is not None else None,
            "n_columns": len(schema_names),
            "columns": ",".join(schema_names),
            "read_error": "",
        }
    except Exception as e:
        return {
            "n_rows": None,
            "n_row_groups": None,
            "n_columns": None,
            "columns": "",
            "read_error": repr(e),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=None,
        help="Path to local COLLIDE-1M snapshot root. If omitted, try common cache paths.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/collide_inventory",
    )
    parser.add_argument(
        "--max-files-per-sample",
        type=int,
        default=-1,
        help="Use -1 for all parquet files; otherwise only inspect first N per sample.",
    )
    args = parser.parse_args()

    root = find_snapshot_root(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[inventory] Using COLLIDE-1M root: {root}")

    sample_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    )
    file_rows = []

    for sample_dir in sample_dirs:
        parquet_files = sorted(sample_dir.rglob("*.parquet"))
        if args.max_files_per_sample > 0:
            parquet_files = parquet_files[: args.max_files_per_sample]

        if not parquet_files:
            file_rows.append(
                {
                    "sample": sample_dir.name,
                    "process_group_guess": guess_group(sample_dir.name),
                    "file": "",
                    "file_size_gb": 0.0,
                    "n_rows": 0,
                    "n_row_groups": 0,
                    "n_columns": 0,
                    "columns": "",
                    "read_error": "no parquet files found",
                }
            )
            continue

        for f in parquet_files:
            info = parquet_info(f)
            file_rows.append(
                {
                    "sample": sample_dir.name,
                    "process_group_guess": guess_group(sample_dir.name),
                    "file": str(f),
                    "file_size_gb": f.stat().st_size / 1e9,
                    **info,
                }
            )

    files = pd.DataFrame(file_rows)
    files.to_csv(outdir / "collide_file_inventory.csv", index=False)

    samples = (
        files.groupby(["sample", "process_group_guess"], dropna=False)
        .agg(
            n_files=("file", lambda x: int((x != "").sum())),
            total_size_gb=("file_size_gb", "sum"),
            total_rows=("n_rows", "sum"),
            n_columns_first=("n_columns", "first"),
            read_errors=("read_error", lambda x: "; ".join(sorted(set(v for v in x if v)))),
        )
        .reset_index()
        .sort_values(["process_group_guess", "sample"])
    )
    samples.to_csv(outdir / "collide_sample_summary.csv", index=False)

    groups = (
        samples.groupby("process_group_guess", dropna=False)
        .agg(
            n_samples=("sample", "count"),
            n_files=("n_files", "sum"),
            total_size_gb=("total_size_gb", "sum"),
            total_rows=("total_rows", "sum"),
            samples=("sample", lambda x: ", ".join(x)),
        )
        .reset_index()
        .sort_values("process_group_guess")
    )
    groups.to_csv(outdir / "collide_process_group_summary.csv", index=False)

    single_top_candidates = samples[
        samples["sample"].str.contains(
            r"^(ST_|single.?top|tW_|tW-|t-channel|s-channel|tZq|tHq)",
            case=False,
            regex=True,
            na=False,
        )
    ].copy()

    payload = {
        "root": str(root),
        "n_samples": int(samples.shape[0]),
        "n_files": int((files["file"] != "").sum()),
        "outputs": {
            "file_inventory": str(outdir / "collide_file_inventory.csv"),
            "sample_summary": str(outdir / "collide_sample_summary.csv"),
            "process_group_summary": str(outdir / "collide_process_group_summary.csv"),
        },
        "single_top_candidate_samples": single_top_candidates["sample"].tolist(),
    }

    with open(outdir / "collide_inventory.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Process group summary ===")
    print(groups[["process_group_guess", "n_samples", "n_files", "total_rows"]].to_string(index=False))

    print("\n=== Note ===")
    print("No dedicated single-top samples were found locally.")
    print("ttW_incl is classified as ttV_ttH_tttt, not single-top.")
    if len(single_top_candidates):
        print(single_top_candidates[["sample", "process_group_guess", "n_files", "total_rows"]].to_string(index=False))
    else:
        print("No local sample name matched ST_/single/tW/t-channel/s-channel/tZq/tHq patterns.")

    print("\nWrote:")
    for p in payload["outputs"].values():
        print(f"  {p}")


if __name__ == "__main__":
    main()