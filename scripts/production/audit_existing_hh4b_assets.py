#!/usr/bin/env python3

from pathlib import Path
import json
import os
import re

import pandas as pd
import pyarrow.parquet as pq
import uproot

REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])

OUTDIR = REPO / "outputs/audits/existing_hh4b_assets_2026_07_16"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEARCH_DIRS = [
    STORE / "root",
    STORE / "root_run2_frozen_v2_smoke",
    STORE / "hepmc",
    STORE / "parquet",
    STORE / "parquet_run2_frozen_v2_smoke",
    STORE / "metadata",
]


def classify_process(name: str) -> str:
    low = name.lower()

    rules = [
        ("ggF_HH_4b", ["ggf", "hh4b"]),
        ("VBF_HH_4b", ["vbf", "hh4b"]),
        ("QCD_bbbb", ["qcd", "bbbb"]),
        ("Zbbbb", ["zbbbb"]),
        ("ttbb", ["ttbb"]),
        ("ttbar", ["ttbar"]),
        ("ZZ_4b", ["zz4b"]),
        ("ZH_4b", ["zh4b"]),
    ]

    for process, tokens in rules:
        if all(token in low for token in tokens):
            return process

    return "unclassified"


def analysis_use(process: str) -> str:
    mapping = {
        "ggF_HH_4b": "signal_pilot_physics_and_ml",
        "VBF_HH_4b": "signal_pilot_physics_and_ml",
        "ttbar": "physics_pilot_and_ml_pending_generator_review",
        "QCD_bbbb": "ml_enrichment_not_inclusive_physics_yield",
        "Zbbbb": "ml_enrichment_not_inclusive_physics_yield",
        "ttbb": "ml_enrichment_not_physics_yield_without_overlap_removal",
        "ZZ_4b": "validation",
        "ZH_4b": "validation",
    }
    return mapping.get(process, "manual_review")


def file_format(path: Path) -> str:
    name = path.name.lower()

    if name.endswith(".root"):
        return "root"
    if ".hepmc" in name:
        return "hepmc"
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith(".json"):
        return "json"
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".txt") or name.endswith(".log"):
        return "text"

    return path.suffix.lstrip(".") or "other"


def detector_version(path: Path, fmt: str) -> str:
    low = str(path).lower()

    if fmt == "hepmc":
        return "generator_level_reprocessable"

    if "run2_frozen_v2" in low:
        return "run2_frozen_v2"

    if fmt == "root":
        return "legacy_card_v1"

    return "derived_or_unknown"


def root_events(path: Path):
    try:
        with uproot.open(path) as root_file:
            if "Delphes" not in root_file:
                return None
            return int(root_file["Delphes"].num_entries)
    except Exception:
        return None


def parquet_rows(path: Path):
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


rows = []

for directory in SEARCH_DIRS:
    if not directory.exists():
        continue

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        fmt = file_format(path)

        if fmt not in {
            "root",
            "hepmc",
            "parquet",
            "json",
            "csv",
            "text",
        }:
            continue

        process = classify_process(path.name)

        n_entries = None
        if fmt == "root":
            n_entries = root_events(path)
        elif fmt == "parquet":
            n_entries = parquet_rows(path)

        rows.append({
            "process": process,
            "analysis_use": analysis_use(process),
            "format": fmt,
            "detector_version": detector_version(path, fmt),
            "path": str(path),
            "filename": path.name,
            "size_GB": path.stat().st_size / 1.0e9,
            "n_entries_or_rows": n_entries,
        })

inventory = pd.DataFrame(rows)

if inventory.empty:
    raise SystemExit("No matching local assets were found.")

inventory.to_csv(OUTDIR / "existing_hh4b_file_inventory.csv", index=False)
(
    OUTDIR / "existing_hh4b_file_inventory.md"
).write_text(inventory.to_markdown(index=False) + "\n")

summary = (
    inventory.groupby(
        [
            "process",
            "analysis_use",
            "format",
            "detector_version",
        ],
        dropna=False,
        as_index=False,
    )
    .agg(
        n_files=("path", "count"),
        total_size_GB=("size_GB", "sum"),
        total_entries_or_rows=("n_entries_or_rows", "sum"),
    )
    .sort_values(
        ["process", "format", "detector_version"]
    )
)

summary.to_csv(OUTDIR / "existing_hh4b_asset_summary.csv", index=False)
(
    OUTDIR / "existing_hh4b_asset_summary.md"
).write_text(summary.to_markdown(index=False) + "\n")

hepmc = inventory[inventory["format"] == "hepmc"].copy()

reuse = (
    hepmc.groupby(
        ["process", "analysis_use"],
        as_index=False,
    )
    .agg(
        n_hepmc_files=("path", "count"),
        total_hepmc_GB=("size_GB", "sum"),
        hepmc_paths=("path", lambda x: ";".join(x)),
    )
)

reuse["recommended_action"] = reuse["process"].map({
    "ggF_HH_4b": "rerun_frozen_v2_and_count_as_signal_pilot",
    "VBF_HH_4b": "rerun_frozen_v2_and_count_as_signal_pilot",
    "ttbar": "rerun_frozen_v2_as_efficiency_pilot",
    "QCD_bbbb": "rerun_frozen_v2_as_ml_enrichment",
    "Zbbbb": "rerun_frozen_v2_as_ml_enrichment",
    "ttbb": "rerun_frozen_v2_as_ml_enrichment",
    "ZZ_4b": "rerun_frozen_v2_as_validation",
    "ZH_4b": "rerun_frozen_v2_as_validation",
}).fillna("manual_review")

reuse.to_csv(OUTDIR / "existing_hh4b_reuse_plan.csv", index=False)
(
    OUTDIR / "existing_hh4b_reuse_plan.md"
).write_text(reuse.to_markdown(index=False) + "\n")

payload = {
    "store": str(STORE),
    "n_files": int(len(inventory)),
    "total_size_GB": float(inventory["size_GB"].sum()),
    "n_root_files": int((inventory["format"] == "root").sum()),
    "n_hepmc_files": int((inventory["format"] == "hepmc").sum()),
    "n_parquet_files": int((inventory["format"] == "parquet").sum()),
}

(OUTDIR / "summary.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)

print("\n=== Existing asset summary ===")
print(summary.to_string(index=False))

print("\n=== HepMC reuse plan ===")
print(reuse.to_string(index=False))

print("\n=== Overall ===")
print(json.dumps(payload, indent=2))

print("\nWrote:", OUTDIR)
