#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq


TEXT_SUFFIXES = {
    ".json",
    ".tsv",
    ".csv",
    ".txt",
    ".yaml",
    ".yml",
    ".md",
    ".py",
    ".sh",
    ".sub",
}

MAX_TEXT_BYTES = 5_000_000


def classify_modes(
    raw: str,
) -> set[str]:
    text = raw.lower()

    modes: set[str] = set()

    # Do not confuse these single-Higgs backgrounds with ggF/VBF HH.
    single_higgs_patterns = {
        "ggh_hbb",
        "ggh-hbb",
        "gghhbb",
        "ggh_to_bb",
        "vbfh_hbb",
        "vbfh-hbb",
        "vbf_hbb",
        "vbf-hbb",
    }

    contains_single_higgs = any(
        pattern in text
        for pattern in single_higgs_patterns
    )

    ggf_markers = {
        "gghh",
        "gg_hh",
        "gg-hh",
        "ggfhh",
        "ggf_hh",
        "ggf-hh",
        "ggtohh",
        "gg_to_hh",
        "gg2hh",
        "gg_2_hh",
        "hh4b_ggf",
        "hh_ggf",
    }

    if (
        not contains_single_higgs
        and any(
            marker in text
            for marker in ggf_markers
        )
    ):
        modes.add("ggf_hh4b")

    if (
        not contains_single_higgs
        and "vbf" in text
        and "hh" in text
    ):
        modes.add("vbf_hh4b")

    return modes


def iter_files(
    roots: list[Path],
) -> Iterator[Path]:
    seen: set[Path] = set()

    for root in roots:
        if not root.exists():
            continue

        if root.is_file():
            candidates = [root]
        else:
            candidates = (
                Path(directory) / filename
                for directory, _, filenames
                in os.walk(root)
                for filename in filenames
            )

        for path in candidates:
            try:
                resolved = path.resolve()
            except OSError:
                continue

            if resolved in seen:
                continue

            seen.add(resolved)
            yield resolved


def inspect_parquet(
    path: Path,
) -> tuple[str, str]:
    try:
        parquet = pq.ParquetFile(path)

        rows = str(
            parquet.metadata.num_rows
        )

        columns = str(
            len(
                parquet.schema_arrow.names
            )
        )

        return rows, columns
    except Exception:
        return "", ""


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--store",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
    )

    arguments = parser.parse_args()

    repo = arguments.repo.resolve()
    store = arguments.store.resolve()
    outdir = arguments.outdir.resolve()

    if outdir.exists():
        raise RuntimeError(
            f"refusing to overwrite {outdir}"
        )

    roots = [
        repo / "outputs" / "agent_runs",
        repo / "configs",
        repo / "scripts",
        repo / "docs",
        repo / "metadata",
        store / "metadata",
        store / "parquet",
        store / "root",
        store / "lhe",
        store / "hepmc",
        store / "condor_submit",
        store / "condor_return",
    ]

    path_rows: list[dict[str, object]] = []
    text_rows: list[dict[str, object]] = []

    seen_path_keys: set[
        tuple[str, str]
    ] = set()

    for path in iter_files(roots):
        path_modes = classify_modes(
            str(path)
        )

        for mode in sorted(path_modes):
            key = (
                mode,
                str(path),
            )

            if key in seen_path_keys:
                continue

            seen_path_keys.add(key)

            try:
                size_bytes = path.stat().st_size
            except OSError:
                size_bytes = -1

            parquet_rows = ""
            parquet_columns = ""

            if path.suffix.lower() == ".parquet":
                (
                    parquet_rows,
                    parquet_columns,
                ) = inspect_parquet(path)

            path_rows.append({
                "signal_mode":
                    mode,
                "path":
                    str(path),
                "suffix":
                    path.suffix.lower(),
                "size_bytes":
                    size_bytes,
                "parquet_rows":
                    parquet_rows,
                "parquet_columns":
                    parquet_columns,
            })

        if (
            path.suffix.lower()
            not in TEXT_SUFFIXES
        ):
            continue

        try:
            if path.stat().st_size > MAX_TEXT_BYTES:
                continue

            lines = path.read_text(
                errors="replace"
            ).splitlines()
        except OSError:
            continue

        for line_number, line in enumerate(
            lines,
            start=1,
        ):
            modes = classify_modes(line)

            for mode in sorted(modes):
                text_rows.append({
                    "signal_mode":
                        mode,
                    "path":
                        str(path),
                    "line":
                        line_number,
                    "text":
                        line[:4000],
                })

    path_rows.sort(
        key=lambda row: (
            str(row["signal_mode"]),
            str(row["path"]),
        )
    )

    text_rows.sort(
        key=lambda row: (
            str(row["signal_mode"]),
            str(row["path"]),
            int(row["line"]),
        )
    )

    outdir.mkdir(
        parents=True
    )

    paths_tsv = (
        outdir
        / "signal_path_inventory.tsv"
    )

    with paths_tsv.open(
        "w",
        newline="",
    ) as handle:
        columns = [
            "signal_mode",
            "path",
            "suffix",
            "size_bytes",
            "parquet_rows",
            "parquet_columns",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(path_rows)

    hits_tsv = (
        outdir
        / "signal_text_hits.tsv"
    )

    with hits_tsv.open(
        "w",
        newline="",
    ) as handle:
        columns = [
            "signal_mode",
            "path",
            "line",
            "text",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(text_rows)

    path_counts = Counter(
        str(row["signal_mode"])
        for row in path_rows
    )

    text_counts = Counter(
        str(row["signal_mode"])
        for row in text_rows
    )

    local_parquet_counts = Counter()
    local_parquet_rows = Counter()

    for row in path_rows:
        if row["suffix"] != ".parquet":
            continue

        if row["parquet_rows"] == "":
            continue

        mode = str(
            row["signal_mode"]
        )

        local_parquet_counts[mode] += 1
        local_parquet_rows[mode] += int(
            row["parquet_rows"]
        )

    summary = {
        "schema_version": 1,
        "status": "inventory_complete",
        "path_hits_by_mode":
            dict(path_counts),
        "text_hits_by_mode":
            dict(text_counts),
        "local_parquet_files_by_mode":
            dict(local_parquet_counts),
        "local_parquet_rows_by_mode":
            dict(local_parquet_rows),
        "ggf_signal_registry_frozen":
            False,
        "vbf_signal_registry_already_known":
            True,
        "combined_signal_registry_frozen":
            False,
        "single_higgs_ggh_hbb_excluded":
            True,
        "path_inventory":
            str(paths_tsv),
        "text_hits":
            str(hits_tsv),
    }

    summary_path = (
        outdir
        / "signal_source_inventory.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "HH4B_SIGNAL_SOURCE_INVENTORY_COMPLETE"
    )

    print(
        "PATH_HITS_BY_MODE="
        + json.dumps(
            dict(path_counts),
            sort_keys=True,
        )
    )

    print(
        "TEXT_HITS_BY_MODE="
        + json.dumps(
            dict(text_counts),
            sort_keys=True,
        )
    )

    print(
        "LOCAL_PARQUET_FILES_BY_MODE="
        + json.dumps(
            dict(local_parquet_counts),
            sort_keys=True,
        )
    )

    print(
        "LOCAL_PARQUET_ROWS_BY_MODE="
        + json.dumps(
            dict(local_parquet_rows),
            sort_keys=True,
        )
    )

    if (
        path_counts["ggf_hh4b"] > 0
        or text_counts["ggf_hh4b"] > 0
    ):
        print(
            "GGF_HH4B_SIGNAL_EVIDENCE_FOUND"
        )
    else:
        print(
            "GGF_HH4B_SIGNAL_EVIDENCE_NOT_FOUND"
        )

    print(
        "GGF_HH4B_CANONICAL_RECONCILIATION_REQUIRED"
    )
    print(
        "COMBINED_GGF_PLUS_VBF_SIGNAL_REGISTRY_NOT_YET_FROZEN"
    )
    print(
        f"summary_json={summary_path}"
    )
    print(
        f"path_inventory={paths_tsv}"
    )
    print(
        f"text_hits={hits_tsv}"
    )


if __name__ == "__main__":
    main()
