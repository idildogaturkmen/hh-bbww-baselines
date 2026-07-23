#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


LEGACY_V1_COLUMNS = {
    "sample",
    "event",
    "n_selected_bjets",
    "mbb1",
    "mbb2",
    "avg_mbb",
    "delta_mbb",
    "mhh",
    "drbb1",
    "drbb2",
    "pairing",
    "j1_pt",
    "j2_pt",
    "j3_pt",
    "j4_pt",
}

RICH_V2_REQUIRED = {
    "sample",
    "event",
    "n_selected_jets",
    "n_selected_bjets",
    "pairing_combo_bjet_ranks",
    "higgs_ordering",
    "mbb1",
    "mbb2",
    "r_hh_125_120",
    "mhh",
    "h1_pt",
    "h2_pt",
    "j1_pt",
    "j1_eta",
    "j1_btag",
    "j4_pt",
    "j4_eta",
    "j4_btag",
}


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def run(
    arguments: list[str],
) -> str:
    result = subprocess.run(
        arguments,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return result.stdout


def schema_signature(
    path: Path,
) -> tuple[str, list[dict[str, object]]]:
    schema = pq.ParquetFile(
        path
    ).schema_arrow

    fields = sorted(
        [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in schema
        ],
        key=lambda item: str(item["name"]),
    )

    encoded = json.dumps(
        fields,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    return (
        hashlib.sha256(encoded).hexdigest(),
        fields,
    )


def classify_schema(
    columns: set[str],
) -> str:
    if columns == LEGACY_V1_COLUMNS:
        return "legacy_v1"

    if RICH_V2_REQUIRED.issubset(
        columns
    ):
        return "rich_v2"

    return "other"


def file_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--eos-host",
        required=True,
    )

    parser.add_argument(
        "--eos-canary",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
    )

    arguments = parser.parse_args()

    outdir = arguments.outdir.resolve()

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    with arguments.manifest.open(
        newline="",
        errors="replace",
    ) as handle:
        manifest = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    local_paths = sorted({
        Path(
            row["local_candidate_parquet"]
        ).resolve()
        for row in manifest
        if (
            row["availability"]
            == "local_candidate_parquet"
            and row[
                "local_candidate_parquet"
            ]
        )
    })

    require(
        len(local_paths) == 27,
        (
            "expected 27 local reference Parquets, "
            f"found {len(local_paths)}"
        ),
    )

    local_schema_hashes = Counter()

    for path in local_paths:
        require(
            path.is_file(),
            f"missing local reference {path}",
        )

        schema_hash, _ = schema_signature(
            path
        )

        local_schema_hashes[
            schema_hash
        ] += 1

    require(
        len(local_schema_hashes) == 1,
        (
            "local references do not have one "
            "uniform schema"
        ),
    )

    local_schema_hash = next(
        iter(local_schema_hashes)
    )

    receipt_lfns = [
        value.strip()
        for value in run([
            "xrdfs",
            arguments.eos_host,
            "ls",
            f"{arguments.eos_canary}/receipts",
        ]).splitlines()
        if value.strip()
    ]

    parquet_lfns = [
        value.strip()
        for value in run([
            "xrdfs",
            arguments.eos_host,
            "ls",
            f"{arguments.eos_canary}/parquets",
        ]).splitlines()
        if value.strip()
    ]

    require(
        len(receipt_lfns) == 8,
        (
            "expected 8 EOS receipts, "
            f"found {len(receipt_lfns)}"
        ),
    )

    require(
        len(parquet_lfns) == 8,
        (
            "expected 8 EOS Parquets, "
            f"found {len(parquet_lfns)}"
        ),
    )

    parquet_by_index: dict[
        int,
        str,
    ] = {}

    pattern = re.compile(
        r"member_(\d+)_"
    )

    for lfn in parquet_lfns:
        match = pattern.search(
            Path(lfn).name
        )

        require(
            match is not None,
            f"cannot parse registry index from {lfn}",
        )

        index = int(
            match.group(1)
        )

        require(
            index not in parquet_by_index,
            f"duplicate Parquet index {index}",
        )

        parquet_by_index[index] = lfn

    rows = []
    schema_classes = Counter()
    schema_hashes = Counter()
    total_rows = 0
    exact_local_matches = 0

    with tempfile.TemporaryDirectory(
        prefix="hh4b_schema_canary_",
        dir="/tmp",
    ) as temporary_raw:
        temporary = Path(
            temporary_raw
        )

        receipts = []

        for lfn in receipt_lfns:
            local = (
                temporary
                / Path(lfn).name
            )

            run([
                "xrdcp",
                "-f",
                f"{arguments.eos_host}/{lfn}",
                str(local),
            ])

            receipts.append(
                json.loads(
                    local.read_text()
                )
            )

        require(
            len(receipts) == 8,
            "did not load eight receipts",
        )

        for receipt in sorted(
            receipts,
            key=lambda item: int(
                item["registry_index"]
            ),
        ):
            index = int(
                receipt["registry_index"]
            )

            require(
                index in parquet_by_index,
                (
                    "missing EOS Parquet for "
                    f"registry index {index}"
                ),
            )

            lfn = parquet_by_index[
                index
            ]

            local = (
                temporary
                / Path(lfn).name
            )

            run([
                "xrdcp",
                "-f",
                f"{arguments.eos_host}/{lfn}",
                str(local),
            ])

            actual_sha256 = file_sha256(
                local
            )

            require(
                actual_sha256
                == receipt[
                    "candidate_sha256"
                ],
                (
                    f"registry {index}: "
                    "candidate SHA-256 mismatch"
                ),
            )

            parquet_file = pq.ParquetFile(
                local
            )

            actual_rows = int(
                parquet_file.metadata.num_rows
            )

            expected_rows = int(
                receipt[
                    "expected_candidate_rows"
                ]
            )

            require(
                actual_rows == expected_rows,
                (
                    f"registry {index}: expected "
                    f"{expected_rows} rows, "
                    f"found {actual_rows}"
                ),
            )

            schema_hash, fields = (
                schema_signature(
                    local
                )
            )

            columns = {
                str(field["name"])
                for field in fields
            }

            schema_class = (
                classify_schema(
                    columns
                )
            )

            matches_local = (
                schema_hash
                == local_schema_hash
            )

            if matches_local:
                exact_local_matches += 1

            schema_classes[
                schema_class
            ] += 1

            schema_hashes[
                schema_hash
            ] += 1

            total_rows += actual_rows

            rows.append({
                "registry_index":
                    index,
                "component":
                    receipt["component"],
                "family":
                    receipt["family"],
                "dataset_split":
                    receipt[
                        "dataset_split"
                    ],
                "expected_rows":
                    expected_rows,
                "actual_rows":
                    actual_rows,
                "column_count":
                    len(columns),
                "schema_class":
                    schema_class,
                "schema_sha256":
                    schema_hash,
                "exact_match_local":
                    matches_local,
                "eos_parquet":
                    lfn,
            })

    outdir.mkdir(
        parents=True
    )

    table_path = (
        outdir
        / "schema_canary_audit.tsv"
    )

    with table_path.open(
        "w",
        newline="",
    ) as handle:
        columns = [
            "registry_index",
            "component",
            "family",
            "dataset_split",
            "expected_rows",
            "actual_rows",
            "column_count",
            "schema_class",
            "schema_sha256",
            "exact_match_local",
            "eos_parquet",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    uniform_exact_legacy = (
        exact_local_matches == 8
        and schema_classes
        == Counter({"legacy_v1": 8})
    )

    summary = {
        "schema_version": 1,
        "status": "pass",
        "remote_canaries": 8,
        "local_reference_files": 27,
        "total_remote_candidate_rows":
            total_rows,
        "local_reference_schema_sha256":
            local_schema_hash,
        "remote_schema_classes":
            dict(schema_classes),
        "remote_schema_hashes":
            dict(schema_hashes),
        "remote_exact_matches_local":
            exact_local_matches,
        "column_order_ignored":
            True,
        "uniform_exact_legacy_v1":
            uniform_exact_legacy,
        "full_materialization_authorized":
            uniform_exact_legacy,
        "rich_v2_local_rebuild_required":
            (
                schema_classes
                == Counter({"rich_v2": 8})
            ),
        "audit_table":
            str(table_path),
    }

    summary_path = (
        outdir
        / "schema_canary_audit.json"
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
        "BACKGROUND_5M_SCHEMA_CANARY_AUDIT_VALID"
    )
    print("REMOTE_CANARIES=8")
    print(
        "TOTAL_REMOTE_CANDIDATE_ROWS="
        f"{total_rows}"
    )
    print(
        "REMOTE_SCHEMA_CLASSES="
        + json.dumps(
            dict(schema_classes),
            sort_keys=True,
        )
    )
    print(
        "REMOTE_DISTINCT_SCHEMAS="
        f"{len(schema_hashes)}"
    )
    print(
        "REMOTE_EXACT_MATCHES_LOCAL="
        f"{exact_local_matches}"
    )

    if uniform_exact_legacy:
        print(
            "REMOTE_CANDIDATES_UNIFORM_LEGACY_V1"
        )
        print(
            "FULL_BACKGROUND_5M_CANDIDATE_MATERIALIZATION_AUTHORIZED"
        )
        print(
            "REDUCED_CANDIDATE_CUTFLOW_PREPARATION_AUTHORIZED"
        )
    elif (
        schema_classes
        == Counter({"rich_v2": 8})
    ):
        print(
            "REMOTE_CANDIDATES_UNIFORM_RICH_V2"
        )
        print(
            "LOCAL_LEGACY_TTBAR_REBUILD_REQUIRED"
        )
        print(
            "FULL_MATERIALIZATION_NOT_YET_AUTHORIZED"
        )
    else:
        print(
            "REMOTE_CANDIDATE_SCHEMAS_MIXED_OR_UNKNOWN"
        )
        print(
            "FULL_MATERIALIZATION_BLOCKED"
        )

    print(
        f"summary_json={summary_path}"
    )
    print(
        f"audit_table={table_path}"
    )


if __name__ == "__main__":
    main()
