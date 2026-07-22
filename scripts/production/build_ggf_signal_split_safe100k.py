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

EXPECTED_NEW_EVENTS = {
    "train": 66000,
    "validation": 14000,
}

EXPECTED_NEW_CANDIDATES = {
    "train": 3975,
    "validation": 848,
}

EXPECTED_FINAL_EVENTS = {
    "train": 80000,
    "validation": 17000,
    "test": 3000,
}

EXPECTED_FINAL_CANDIDATES = {
    "train": 4893,
    "validation": 1045,
    "test": 174,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
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

        handle.extractall(destination)


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


def schema_without_metadata(path: Path):
    return (
        pq.ParquetFile(path)
        .schema_arrow
        .remove_metadata()
    )


def write_parquet(
    frame: pd.DataFrame,
    path: Path,
    reference_schema,
) -> None:
    frame.to_parquet(
        path,
        index=False,
    )

    observed_schema = schema_without_metadata(
        path
    )

    if not observed_schema.equals(
        reference_schema
    ):
        raise RuntimeError(
            f"schema mismatch after writing {path}"
        )


def identity_set(
    frame: pd.DataFrame,
) -> set[tuple[str, int]]:
    identities = set(
        zip(
            frame[
                "source_root"
            ].astype(str),
            frame[
                "event"
            ].astype(int),
        )
    )

    if len(identities) != len(frame):
        raise RuntimeError(
            "duplicate source_root/event identities"
        )

    return identities


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cross-layer-tsv",
        required=True,
    )

    parser.add_argument(
        "--reference-parquet",
        required=True,
    )

    parser.add_argument(
        "--old20-dir",
        required=True,
    )

    parser.add_argument(
        "--old20-registry",
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
        "--new80-dir",
        required=True,
    )

    parser.add_argument(
        "--new80-registry",
        required=True,
    )

    parser.add_argument(
        "--new80-summary",
        required=True,
    )

    parser.add_argument(
        "--final100-dir",
        required=True,
    )

    parser.add_argument(
        "--final100-registry",
        required=True,
    )

    parser.add_argument(
        "--final100-summary",
        required=True,
    )

    args = parser.parse_args()

    cross_layer_path = Path(
        args.cross_layer_tsv
    ).resolve()

    reference_path = Path(
        args.reference_parquet
    ).resolve()

    old20_dir = Path(
        args.old20_dir
    ).resolve()

    old20_registry_path = Path(
        args.old20_registry
    ).resolve()

    temp_root = Path(
        args.temp_root
    ).resolve()

    new80_dir = Path(
        args.new80_dir
    ).resolve()

    new80_registry_path = Path(
        args.new80_registry
    ).resolve()

    new80_summary_path = Path(
        args.new80_summary
    ).resolve()

    final100_dir = Path(
        args.final100_dir
    ).resolve()

    final100_registry_path = Path(
        args.final100_registry
    ).resolve()

    final100_summary_path = Path(
        args.final100_summary
    ).resolve()

    required_files = [
        cross_layer_path,
        reference_path,
        old20_registry_path,
    ]

    for path in required_files:
        if not path.is_file():
            raise SystemExit(
                f"ERROR: missing required file: {path}"
            )

    reference_schema = schema_without_metadata(
        reference_path
    )

    reference_columns = list(
        reference_schema.names
    )

    if len(reference_columns) != 75:
        raise SystemExit(
            "ERROR: reference does not have 75 columns"
        )

    if reference_columns[-3:] != [
        "analysis_sample",
        "source_root",
        "source_root_index",
    ]:
        raise SystemExit(
            "ERROR: canonical provenance columns "
            "are not the final three columns"
        )

    source_columns = reference_columns[:-3]

    with cross_layer_path.open(
        newline=""
    ) as handle:
        cross_layer_rows = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    if len(cross_layer_rows) != 80:
        raise SystemExit(
            "ERROR: expected 80 cross-layer rows, "
            f"found {len(cross_layer_rows)}"
        )

    with old20_registry_path.open(
        newline=""
    ) as handle:
        old_reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        registry_fields = (
            old_reader.fieldnames
            or []
        )

        old_registry_rows = list(
            old_reader
        )

    if len(old_registry_rows) != 20:
        raise SystemExit(
            "ERROR: expected 20 rows in old registry, "
            f"found {len(old_registry_rows)}"
        )

    required_registry_fields = {
        "campaign",
        "local_shard",
        "inner_shard",
        "seed",
        "immutable_split",
        "generated_events",
        "candidate_rows",
        "analysis_sample",
        "source_root",
        "source_root_index",
        "candidate_parquet",
        "candidate_parquet_sha256",
        "remote_bundle",
        "training_authorized",
        "validation_authorized",
        "test_sealed",
        "physics_yield_authorized",
        "status",
    }

    if set(registry_fields) != required_registry_fields:
        raise SystemExit(
            "ERROR: unexpected old registry fields"
        )

    shutil.rmtree(
        temp_root,
        ignore_errors=True,
    )

    temp_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    new80_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final100_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_records = []

    split_frames: dict[
        str,
        list[pd.DataFrame],
    ] = {
        "train": [],
        "validation": [],
    }

    seen_identities: set[
        tuple[str, int]
    ] = set()

    for position, row in enumerate(
        sorted(
            cross_layer_rows,
            key=lambda item: int(
                item["local_shard"]
            ),
        ),
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

        split = row[
            "dataset_split"
        ]

        generated_events = int(
            row["generated_events"]
        )

        remote_bundle = row[
            "remote_bundle"
        ]

        if split not in split_frames:
            raise RuntimeError(
                f"invalid split for shard "
                f"{local_shard}: {split}"
            )

        source_root_index = (
            10
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
            f"[{position:02d}/80] "
            f"local_shard={local_shard} "
            f"split={split}"
        )

        try:
            source_url = (
                f"{args.eos_host}/"
                f"{remote_bundle}"
            )

            result = subprocess.run(
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

            if result.returncode != 0:
                raise RuntimeError(
                    "xrdcp failed: "
                    + result.stderr.strip()
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
                "72-column candidate Parquet",
            )

            source_root = exactly_one(
                sorted(
                    reconstruction_dir.rglob(
                        "*.root"
                    )
                ),
                "ROOT file",
            )

            source_schema = (
                schema_without_metadata(
                    source_parquet
                )
            )

            if list(
                source_schema.names
            ) != source_columns:
                raise RuntimeError(
                    "72-column source schema mismatch"
                )

            frame = pd.read_parquet(
                source_parquet
            ).copy()

            frame[
                "analysis_sample"
            ] = ANALYSIS_SAMPLE

            frame[
                "source_root"
            ] = source_root.name

            frame[
                "source_root_index"
            ] = source_root_index

            frame = frame[
                reference_columns
            ]

            identities = identity_set(
                frame
            )

            overlap = (
                seen_identities
                & identities
            )

            if overlap:
                raise RuntimeError(
                    "duplicate identity across new shards"
                )

            seen_identities.update(
                identities
            )

            output_path = (
                new80_dir
                / (
                    "ggF_HH4b_SMnorm_ext80k_"
                    f"shard{local_shard:03d}_"
                    f"{split}_v2_"
                    "hh4b_candidates.parquet"
                )
            )

            write_parquet(
                frame,
                output_path,
                reference_schema,
            )

            split_frames[
                split
            ].append(
                frame
            )

            record = {
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
                "test_sealed": False,
                "physics_yield_authorized": (
                    False
                ),
                "status": "pass",
            }

            new_records.append(
                record
            )

            print(
                "  PASS "
                f"candidates={len(frame)} "
                "source_root_index="
                f"{source_root_index}"
            )

        finally:
            shutil.rmtree(
                work_dir,
                ignore_errors=True,
            )

    new_merged_frames = {}

    new_split_candidates = {}

    for split in (
        "train",
        "validation",
    ):
        merged = pd.concat(
            split_frames[
                split
            ],
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

        merged_path = (
            new80_dir
            / (
                "ggF_HH4b_SMnorm_ext80k_"
                f"{split}_v2_"
                "hh4b_candidates.parquet"
            )
        )

        write_parquet(
            merged,
            merged_path,
            reference_schema,
        )

        new_merged_frames[
            split
        ] = merged

        new_split_candidates[
            split
        ] = len(
            merged
        )

    new_split_events = {
        split: sum(
            int(
                record[
                    "generated_events"
                ]
            )
            for record in new_records
            if record[
                "immutable_split"
            ] == split
        )
        for split in (
            "train",
            "validation",
        )
    }

    if len(new_records) != 80:
        raise RuntimeError(
            "new registry does not contain 80 rows"
        )

    if (
        new_split_events
        != EXPECTED_NEW_EVENTS
    ):
        raise RuntimeError(
            "unexpected new event totals: "
            f"{new_split_events}"
        )

    if (
        new_split_candidates
        != EXPECTED_NEW_CANDIDATES
    ):
        raise RuntimeError(
            "unexpected new candidate totals: "
            f"{new_split_candidates}"
        )

    source_indices = sorted(
        int(
            record[
                "source_root_index"
            ]
        )
        for record in new_records
    )

    if source_indices != list(
        range(
            30,
            110,
        )
    ):
        raise RuntimeError(
            "new source_root_index values "
            "are not exactly 30-109"
        )

    new80_registry_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with new80_registry_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=registry_fields,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            new_records
        )

    old_paths = {
        "train": (
            old20_dir
            / (
                "ggF_HH4b_SMnorm_ext20k_"
                "train_v2_hh4b_candidates.parquet"
            )
        ),
        "validation": (
            old20_dir
            / (
                "ggF_HH4b_SMnorm_ext20k_"
                "validation_v2_hh4b_candidates.parquet"
            )
        ),
        "test": (
            old20_dir
            / (
                "ggF_HH4b_SMnorm_ext20k_"
                "test_v2_hh4b_candidates.parquet"
            )
        ),
    }

    for path in old_paths.values():
        if not path.is_file():
            raise RuntimeError(
                f"missing old split Parquet: {path}"
            )

        if not schema_without_metadata(
            path
        ).equals(
            reference_schema
        ):
            raise RuntimeError(
                f"old split schema mismatch: {path}"
            )

    final_frames = {
        "train": pd.concat(
            [
                pd.read_parquet(
                    old_paths[
                        "train"
                    ]
                ),
                new_merged_frames[
                    "train"
                ],
            ],
            ignore_index=True,
        ),
        "validation": pd.concat(
            [
                pd.read_parquet(
                    old_paths[
                        "validation"
                    ]
                ),
                new_merged_frames[
                    "validation"
                ],
            ],
            ignore_index=True,
        ),
        "test": pd.read_parquet(
            old_paths[
                "test"
            ]
        ),
    }

    final_identities = {}

    final_candidate_counts = {}

    final_paths = {}

    for split in (
        "train",
        "validation",
        "test",
    ):
        frame = final_frames[
            split
        ]

        frame = frame.sort_values(
            [
                "source_root_index",
                "event",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        )

        identities = identity_set(
            frame
        )

        final_identities[
            split
        ] = identities

        output_path = (
            final100_dir
            / (
                "ggF_HH4b_SMnorm_"
                "split_safe100k_"
                f"{split}_v2_"
                "hh4b_candidates.parquet"
            )
        )

        write_parquet(
            frame,
            output_path,
            reference_schema,
        )

        final_candidate_counts[
            split
        ] = len(
            frame
        )

        final_paths[
            split
        ] = str(
            output_path
        )

    if (
        final_identities["train"]
        & final_identities[
            "validation"
        ]
    ):
        raise RuntimeError(
            "train/validation identity overlap"
        )

    if (
        final_identities["train"]
        & final_identities[
            "test"
        ]
    ):
        raise RuntimeError(
            "train/test identity overlap"
        )

    if (
        final_identities[
            "validation"
        ]
        & final_identities[
            "test"
        ]
    ):
        raise RuntimeError(
            "validation/test identity overlap"
        )

    if (
        final_candidate_counts
        != EXPECTED_FINAL_CANDIDATES
    ):
        raise RuntimeError(
            "unexpected final candidate counts: "
            f"{final_candidate_counts}"
        )

    final_registry_rows = (
        old_registry_rows
        + [
            {
                field: record[
                    field
                ]
                for field in registry_fields
            }
            for record in new_records
        ]
    )

    if len(
        final_registry_rows
    ) != 100:
        raise RuntimeError(
            "final registry does not contain 100 rows"
        )

    final_split_events = {
        split: sum(
            int(
                row[
                    "generated_events"
                ]
            )
            for row in final_registry_rows
            if row[
                "immutable_split"
            ] == split
        )
        for split in (
            "train",
            "validation",
            "test",
        )
    }

    if (
        final_split_events
        != EXPECTED_FINAL_EVENTS
    ):
        raise RuntimeError(
            "unexpected final event counts: "
            f"{final_split_events}"
        )

    final100_registry_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with final100_registry_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=registry_fields,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            final_registry_rows
        )

    new80_summary = {
        "schema_version": 1,
        "status": "pass",
        "canonical_shards": 80,
        "validated_generated_events": 80000,
        "candidate_rows": sum(
            new_split_candidates.values()
        ),
        "split_generated_events": (
            new_split_events
        ),
        "split_candidate_rows": (
            new_split_candidates
        ),
        "canonical_columns": 75,
        "source_root_index_min": 30,
        "source_root_index_max": 109,
        "test_generated": False,
        "physics_yield_authorized": False,
        "registry": str(
            new80_registry_path
        ),
    }

    final100_summary = {
        "schema_version": 1,
        "status": "pass",
        "registry_rows": 100,
        "validated_generated_events": 100000,
        "candidate_rows": sum(
            final_candidate_counts.values()
        ),
        "split_generated_events": (
            final_split_events
        ),
        "split_candidate_rows": (
            final_candidate_counts
        ),
        "canonical_columns": 75,
        "train_validation_overlap": 0,
        "train_test_overlap": 0,
        "validation_test_overlap": 0,
        "sealed_test_events": 3000,
        "sealed_test_candidate_rows": 174,
        "legacy10k_included": False,
        "physics_yield_authorized": False,
        "parquets": final_paths,
        "registry": str(
            final100_registry_path
        ),
    }

    new80_summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new80_summary_path.write_text(
        json.dumps(
            new80_summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    final100_summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final100_summary_path.write_text(
        json.dumps(
            final100_summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "new80_canonical_shards=80"
    )

    print(
        "new80_split_events="
        f"{new_split_events}"
    )

    print(
        "new80_split_candidates="
        f"{new_split_candidates}"
    )

    print(
        "final100_registry_rows=100"
    )

    print(
        "final100_split_events="
        f"{final_split_events}"
    )

    print(
        "final100_split_candidates="
        f"{final_candidate_counts}"
    )

    print(
        "sealed_test_events=3000"
    )

    print(
        "sealed_test_candidates=174"
    )

    print(
        "legacy10k_included=False"
    )

    print(
        "physics_yield_authorized=False"
    )

    print(
        f"new80_registry={new80_registry_path}"
    )

    print(
        f"final100_registry={final100_registry_path}"
    )

    print(
        f"final100_summary={final100_summary_path}"
    )

    print(
        "GGF_SIGNAL_SPLIT_SAFE100K_VALID"
    )


if __name__ == "__main__":
    main()
