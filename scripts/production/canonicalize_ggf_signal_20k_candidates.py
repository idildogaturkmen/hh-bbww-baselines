#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pandas as pd
import pyarrow.parquet as pq


ANALYSIS_SAMPLE = "ggF_HH4b_SMnorm"

EXPECTED_ROWS_BY_SPLIT = {
    "train": 918,
    "validation": 197,
    "test": 174,
}

EXPECTED_EVENTS_BY_SPLIT = {
    "train": 14000,
    "validation": 3000,
    "test": 3000,
}

EXPECTED_TOTAL_CANDIDATES = 1289
EXPECTED_TOTAL_EVENTS = 20000
EXPECTED_SHARDS = 20

# The ten legacy ggF ROOT files already use source_root_index 0-9.
SOURCE_ROOT_INDEX_OFFSET = 10


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def safe_extract(
    archive: Path,
    destination: Path,
) -> None:
    destination = destination.resolve()

    with tarfile.open(
        archive,
        "r:gz",
    ) as handle:
        for member in handle.getmembers():
            target = (
                destination
                / member.name
            ).resolve()

            if (
                target != destination
                and destination
                not in target.parents
            ):
                raise RuntimeError(
                    "unsafe archive member: "
                    f"{member.name}"
                )

        handle.extractall(
            destination
        )


def exactly_one(
    paths: list[Path],
    description: str,
) -> Path:
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one {description}, "
            f"found {len(paths)}"
        )

    return paths[0]


def schema_without_metadata(
    path: Path,
):
    return (
        pq.ParquetFile(path)
        .schema_arrow
        .remove_metadata()
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--reference-parquet",
        required=True,
    )

    parser.add_argument(
        "--eos-host",
        required=True,
    )

    parser.add_argument(
        "--temp-root",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--registry",
        required=True,
    )

    parser.add_argument(
        "--summary",
        required=True,
    )

    args = parser.parse_args()

    manifest_path = Path(
        args.manifest
    ).resolve()

    reference_path = Path(
        args.reference_parquet
    ).resolve()

    temp_root = Path(
        args.temp_root
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    registry_path = Path(
        args.registry
    ).resolve()

    summary_path = Path(
        args.summary
    ).resolve()

    if not manifest_path.is_file():
        raise SystemExit(
            f"ERROR: missing manifest: {manifest_path}"
        )

    if not reference_path.is_file():
        raise SystemExit(
            "ERROR: missing canonical reference: "
            f"{reference_path}"
        )

    with manifest_path.open(
        newline=""
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    if len(rows) != EXPECTED_SHARDS:
        raise SystemExit(
            f"ERROR: expected {EXPECTED_SHARDS} manifest rows, "
            f"found {len(rows)}"
        )

    reference_schema = schema_without_metadata(
        reference_path
    )

    reference_columns = list(
        reference_schema.names
    )

    expected_provenance_columns = [
        "analysis_sample",
        "source_root",
        "source_root_index",
    ]

    if (
        reference_columns[-3:]
        != expected_provenance_columns
    ):
        raise SystemExit(
            "ERROR: reference does not end with "
            "the three canonical provenance columns"
        )

    if len(reference_columns) != 75:
        raise SystemExit(
            "ERROR: canonical reference does not "
            f"contain 75 columns: {len(reference_columns)}"
        )

    source_columns = (
        reference_columns[:-3]
    )

    if len(source_columns) != 72:
        raise SystemExit(
            "ERROR: expected 72 pre-provenance columns"
        )

    shutil.rmtree(
        temp_root,
        ignore_errors=True,
    )

    temp_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_dir.exists():
        existing_parquets = list(
            output_dir.glob(
                "*.parquet"
            )
        )

        if existing_parquets:
            raise SystemExit(
                "ERROR: output directory already contains "
                "Parquet files: "
                f"{output_dir}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []
    errors = []
    split_frames: dict[str, list[pd.DataFrame]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    observed_identity_pairs = set()

    for position, row in enumerate(
        rows,
        start=1,
    ):
        local_shard = int(
            row["local_shard"]
        )

        inner_shard = int(
            row["inner_shard"]
        )

        seed = int(
            row["seed"]
        )

        generated_events = int(
            row["n_events"]
        )

        split = row[
            "immutable_split"
        ]

        remote_bundle = row[
            "remote_bundle"
        ]

        if split not in split_frames:
            errors.append(
                f"local_shard={local_shard}: "
                f"invalid split {split!r}"
            )
            continue

        source_root_index = (
            SOURCE_ROOT_INDEX_OFFSET
            + local_shard
        )

        work_dir = (
            temp_root
            / f"shard_{local_shard:03d}"
        )

        bundle_path = (
            work_dir
            / "bundle.tar.gz"
        )

        outer_dir = (
            work_dir
            / "outer"
        )

        reconstruction_dir = (
            work_dir
            / "reconstruction"
        )

        work_dir.mkdir(
            parents=True
        )

        outer_dir.mkdir()
        reconstruction_dir.mkdir()

        print(
            f"[{position:02d}/20] "
            f"local_shard={local_shard} "
            f"split={split}"
        )

        try:
            source_url = (
                f"{args.eos_host}/"
                f"{remote_bundle}"
            )

            copy_result = subprocess.run(
                [
                    "xrdcp",
                    "--force",
                    source_url,
                    str(bundle_path),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
            )

            if copy_result.returncode != 0:
                raise RuntimeError(
                    "xrdcp failed: "
                    + copy_result.stderr.strip()
                )

            safe_extract(
                bundle_path,
                outer_dir,
            )

            nested_archive = exactly_one(
                sorted(
                    outer_dir.rglob(
                        "*_reconstruction.tar.gz"
                    )
                ),
                "nested reconstruction archive",
            )

            safe_extract(
                nested_archive,
                reconstruction_dir,
            )

            source_parquet = exactly_one(
                sorted(
                    reconstruction_dir.rglob(
                        "*_hh4b_candidates_v2.parquet"
                    )
                ),
                "72-column v2 candidate Parquet",
            )

            source_root = exactly_one(
                sorted(
                    reconstruction_dir.rglob(
                        "*.root"
                    )
                ),
                "ROOT file",
            )

            source_schema = schema_without_metadata(
                source_parquet
            )

            if list(
                source_schema.names
            ) != source_columns:
                raise RuntimeError(
                    "72-column source schema names "
                    "do not match canonical reference"
                )

            frame = pd.read_parquet(
                source_parquet
            )

            frame = frame.copy()

            frame[
                "analysis_sample"
            ] = ANALYSIS_SAMPLE

            frame[
                "source_root"
            ] = source_root.name

            frame[
                "source_root_index"
            ] = source_root_index

            missing_columns = [
                column
                for column in reference_columns
                if column not in frame.columns
            ]

            extra_columns = [
                column
                for column in frame.columns
                if column not in reference_columns
            ]

            if missing_columns:
                raise RuntimeError(
                    "missing canonical columns: "
                    + ", ".join(
                        missing_columns
                    )
                )

            if extra_columns:
                raise RuntimeError(
                    "unexpected columns: "
                    + ", ".join(
                        extra_columns
                    )
                )

            frame = frame[
                reference_columns
            ]

            identity_pairs = list(
                zip(
                    frame[
                        "source_root"
                    ].astype(str),
                    frame[
                        "event"
                    ].astype(int),
                )
            )

            duplicates = [
                pair
                for pair in identity_pairs
                if pair in observed_identity_pairs
            ]

            if duplicates:
                raise RuntimeError(
                    "duplicate source_root/event "
                    "identity encountered"
                )

            observed_identity_pairs.update(
                identity_pairs
            )

            output_path = (
                output_dir
                / (
                    f"ggF_HH4b_SMnorm_ext20k_"
                    f"shard{local_shard:03d}_"
                    f"{split}_v2_hh4b_candidates.parquet"
                )
            )

            frame.to_parquet(
                output_path,
                index=False,
            )

            output_schema = schema_without_metadata(
                output_path
            )

            if not output_schema.equals(
                reference_schema
            ):
                raise RuntimeError(
                    "canonical output schema differs "
                    "from legacy 75-column reference"
                )

            output_rows = (
                pq.ParquetFile(
                    output_path
                )
                .metadata
                .num_rows
            )

            if output_rows != len(frame):
                raise RuntimeError(
                    "written Parquet row count mismatch"
                )

            split_frames[
                split
            ].append(
                frame
            )

            records.append({
                "campaign": row[
                    "campaign"
                ],
                "local_shard": (
                    local_shard
                ),
                "inner_shard": (
                    inner_shard
                ),
                "seed": seed,
                "immutable_split": (
                    split
                ),
                "generated_events": (
                    generated_events
                ),
                "candidate_rows": (
                    len(frame)
                ),
                "analysis_sample": (
                    ANALYSIS_SAMPLE
                ),
                "source_root": (
                    source_root.name
                ),
                "source_root_index": (
                    source_root_index
                ),
                "candidate_parquet": str(
                    output_path
                ),
                "candidate_parquet_sha256": (
                    sha256_file(
                        output_path
                    )
                ),
                "remote_bundle": (
                    remote_bundle
                ),
                "training_authorized": (
                    split == "train"
                ),
                "validation_authorized": (
                    split
                    == "validation"
                ),
                "test_sealed": (
                    split == "test"
                ),
                "physics_yield_authorized": (
                    False
                ),
                "status": "pass",
            })

            print(
                "  PASS "
                f"candidates={len(frame)} "
                f"source_root_index="
                f"{source_root_index}"
            )

        except Exception as exc:
            error = (
                f"local_shard={local_shard}: "
                f"{exc}"
            )

            errors.append(
                error
            )

            print(
                f"  ERROR {error}"
            )

        finally:
            shutil.rmtree(
                work_dir,
                ignore_errors=True,
            )

    split_candidate_rows = {}

    merged_paths = {}

    for split in (
        "train",
        "validation",
        "test",
    ):
        frames = split_frames[
            split
        ]

        if frames:
            merged = pd.concat(
                frames,
                ignore_index=True,
            )

            merged = merged.sort_values(
                [
                    "source_root_index",
                    "event",
                ],
                kind="stable",
            ).reset_index(
                drop=True
            )
        else:
            merged = pd.DataFrame(
                columns=reference_columns
            )

        merged_path = (
            output_dir
            / (
                f"ggF_HH4b_SMnorm_ext20k_"
                f"{split}_v2_hh4b_candidates.parquet"
            )
        )

        merged.to_parquet(
            merged_path,
            index=False,
        )

        split_candidate_rows[
            split
        ] = len(
            merged
        )

        merged_paths[
            split
        ] = str(
            merged_path
        )

        if len(merged):
            merged_schema = (
                schema_without_metadata(
                    merged_path
                )
            )

            if not merged_schema.equals(
                reference_schema
            ):
                errors.append(
                    f"{split}: merged schema mismatch"
                )

    generated_events = sum(
        record[
            "generated_events"
        ]
        for record in records
    )

    candidate_rows = sum(
        record[
            "candidate_rows"
        ]
        for record in records
    )

    split_generated_events = {
        split: sum(
            record[
                "generated_events"
            ]
            for record in records
            if record[
                "immutable_split"
            ]
            == split
        )
        for split in (
            "train",
            "validation",
            "test",
        )
    }

    if len(records) != EXPECTED_SHARDS:
        errors.append(
            f"expected {EXPECTED_SHARDS} canonical shards, "
            f"found {len(records)}"
        )

    if generated_events != EXPECTED_TOTAL_EVENTS:
        errors.append(
            f"expected {EXPECTED_TOTAL_EVENTS} generated events, "
            f"found {generated_events}"
        )

    if candidate_rows != EXPECTED_TOTAL_CANDIDATES:
        errors.append(
            f"expected {EXPECTED_TOTAL_CANDIDATES} candidates, "
            f"found {candidate_rows}"
        )

    if (
        split_generated_events
        != EXPECTED_EVENTS_BY_SPLIT
    ):
        errors.append(
            "unexpected split event totals: "
            f"{split_generated_events}"
        )

    if (
        split_candidate_rows
        != EXPECTED_ROWS_BY_SPLIT
    ):
        errors.append(
            "unexpected split candidate totals: "
            f"{split_candidate_rows}"
        )

    source_root_indices = [
        record[
            "source_root_index"
        ]
        for record in records
    ]

    if sorted(
        source_root_indices
    ) != list(
        range(
            10,
            30,
        )
    ):
        errors.append(
            "source_root_index values are not 10-29"
        )

    registry_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if records:
        with registry_path.open(
            "w",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    records[0]
                ),
                delimiter="\t",
                lineterminator="\n",
            )

            writer.writeheader()
            writer.writerows(
                records
            )
    else:
        registry_path.write_text(
            "status\n"
        )

    summary = {
        "schema_version": 1,
        "status": (
            "pass"
            if not errors
            else "fail"
        ),
        "campaign": (
            rows[0]["campaign"]
            if rows
            else ""
        ),
        "canonical_shards": len(
            records
        ),
        "validated_generated_events": (
            generated_events
        ),
        "candidate_rows": (
            candidate_rows
        ),
        "split_generated_events": (
            split_generated_events
        ),
        "split_candidate_rows": (
            split_candidate_rows
        ),
        "training_candidate_rows": (
            split_candidate_rows[
                "train"
            ]
        ),
        "validation_candidate_rows": (
            split_candidate_rows[
                "validation"
            ]
        ),
        "sealed_test_candidate_rows": (
            split_candidate_rows[
                "test"
            ]
        ),
        "test_used_for_training": (
            False
        ),
        "analysis_sample": (
            ANALYSIS_SAMPLE
        ),
        "canonical_columns": len(
            reference_columns
        ),
        "source_root_index_min": (
            min(
                source_root_indices
            )
            if source_root_indices
            else None
        ),
        "source_root_index_max": (
            max(
                source_root_indices
            )
            if source_root_indices
            else None
        ),
        "merged_parquets": (
            merged_paths
        ),
        "registry": str(
            registry_path
        ),
        "physics_yield_authorized": (
            False
        ),
        "errors": errors,
    }

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        f"canonical_shards={len(records)}"
    )

    print(
        f"validated_generated_events={generated_events}"
    )

    print(
        f"candidate_rows={candidate_rows}"
    )

    print(
        "split_candidate_rows="
        f"{split_candidate_rows}"
    )

    print(
        "source_root_index_range="
        f"{summary['source_root_index_min']}-"
        f"{summary['source_root_index_max']}"
    )

    print(
        "canonical_columns="
        f"{summary['canonical_columns']}"
    )

    print(
        "test_used_for_training=False"
    )

    print(
        "physics_yield_authorized=False"
    )

    print(
        f"registry={registry_path}"
    )

    print(
        f"summary={summary_path}"
    )

    if errors:
        for error in errors:
            print(
                f"ERROR: {error}"
            )

        raise SystemExit(2)

    print(
        "GGF_SIGNAL_20K_CANONICALIZATION_VALID"
    )


if __name__ == "__main__":
    main()
