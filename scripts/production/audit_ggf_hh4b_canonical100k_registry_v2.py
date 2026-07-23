#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


EXPECTED_CAMPAIGN20 = (
    "ggf_hh4b_ml_ext20k_frozen_v2_20260716"
)

EXPECTED_CAMPAIGN80 = (
    "ggf_hh4b_ml_ext80k_frozen_v2_20260721_v1"
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def load_table(
    path: Path,
    delimiter: str,
) -> list[dict[str, str]]:
    with path.open(
        newline="",
        errors="replace",
    ) as handle:
        return list(
            csv.DictReader(
                handle,
                delimiter=delimiter,
            )
        )


def load_json(
    path: Path,
) -> dict[str, Any]:
    document = json.loads(
        path.read_text(
            errors="replace",
        )
    )

    require(
        isinstance(document, dict),
        f"{path}: JSON is not an object",
    )

    return document


def parse_bool(
    value: object,
) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(
        value
    ).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise RuntimeError(
        f"cannot parse Boolean value {value!r}"
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def schema_signature(
    path: Path,
) -> tuple[str, list[str]]:
    schema = pq.ParquetFile(
        path
    ).schema_arrow

    fields = sorted(
        [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": bool(
                    field.nullable
                ),
            }
            for field in schema
        ],
        key=lambda item: str(
            item["name"]
        ),
    )

    payload = json.dumps(
        fields,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    signature = hashlib.sha256(
        payload
    ).hexdigest()

    columns = [
        str(field["name"])
        for field in fields
    ]

    return signature, columns


def eos_adler32(
    eos_host: str,
    lfn: str,
) -> str:
    completed = subprocess.run(
        [
            "xrdfs",
            eos_host,
            "query",
            "checksum",
            lfn,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    matches = re.findall(
        r"\b[0-9a-fA-F]{8}\b",
        completed.stdout,
    )

    require(
        matches,
        (
            "could not parse EOS checksum for "
            f"{lfn}: {completed.stdout!r}"
        ),
    )

    return matches[-1].lower()


def receipt_required(
    receipt: dict[str, Any],
    field: str,
    path: Path,
) -> Any:
    require(
        field in receipt,
        f"{path}: missing required field {field}",
    )

    return receipt[field]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--registry20",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--registry80",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--manifest20",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--manifest80",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--receipts20",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--receipts80",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--schema-preflight-summary",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--schema-preflight-members",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--eos-host",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    outdir = args.outdir.resolve()

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    for path in (
        args.registry20,
        args.registry80,
        args.manifest20,
        args.manifest80,
        args.schema_preflight_summary,
        args.schema_preflight_members,
    ):
        require(
            path.is_file(),
            f"missing required input {path}",
        )

    require(
        args.receipts20.is_dir(),
        f"missing receipt directory {args.receipts20}",
    )

    require(
        args.receipts80.is_dir(),
        f"missing receipt directory {args.receipts80}",
    )

    preflight_summary = load_json(
        args.schema_preflight_summary
    )

    require(
        preflight_summary["status"]
        == "preflight_complete",
        "schema preflight status is not complete",
    )

    require(
        int(
            preflight_summary[
                "canonical_members"
            ]
        )
        == 100,
        "schema preflight does not contain 100 members",
    )

    require(
        preflight_summary[
            "all_registered_files_rows_and_checksums_valid"
        ]
        is True,
        "schema preflight file checks did not pass",
    )

    require(
        preflight_summary[
            "all_common72_ml_compatible"
        ]
        is True,
        "ggF candidates are not common-72 compatible",
    )

    require(
        preflight_summary[
            "uniform_signal_schema"
        ]
        is True,
        "ggF candidates do not share one signal schema",
    )

    require(
        int(
            preflight_summary[
                "distinct_signal_schemas"
            ]
        )
        == 1,
        "ggF candidates have more than one schema",
    )

    common72_schema_sha256 = str(
        preflight_summary[
            "reference_schema_sha256"
        ]
    )

    preflight_members = load_table(
        args.schema_preflight_members,
        "\t",
    )

    require(
        len(preflight_members) == 100,
        (
            "expected 100 preflight member rows, "
            f"found {len(preflight_members)}"
        ),
    )

    preflight_by_shard = {
        int(row["local_shard"]): row
        for row in preflight_members
    }

    require(
        set(preflight_by_shard)
        == set(range(100)),
        "preflight membership is not shards 0–99",
    )

    signal_schema_hashes = {
        row["schema_sha256"]
        for row in preflight_members
    }

    signal_column_counts = {
        int(row["column_count"])
        for row in preflight_members
    }

    extra_column_sets = {
        row["extra_columns"]
        for row in preflight_members
    }

    require(
        len(signal_schema_hashes) == 1,
        "preflight contains multiple ggF schemas",
    )

    require(
        signal_column_counts == {75},
        (
            "expected one 75-column ggF schema, "
            f"found {sorted(signal_column_counts)}"
        ),
    )

    require(
        len(extra_column_sets) == 1,
        "ggF members do not share the same extra columns",
    )

    signal_schema_sha256 = next(
        iter(signal_schema_hashes)
    )

    signal_extra_columns = [
        column
        for column in next(
            iter(extra_column_sets)
        ).split(";")
        if column
    ]

    for shard, row in preflight_by_shard.items():
        require(
            row["schema_class"]
            == "compatible_superset_of_background_rich_v2",
            (
                f"shard {shard}: unexpected preflight "
                f"classification {row['schema_class']!r}"
            ),
        )

        require(
            parse_bool(
                row["file_exists"]
            ),
            f"shard {shard}: candidate file is absent",
        )

        require(
            parse_bool(
                row["row_count_match"]
            ),
            f"shard {shard}: preflight row count failed",
        )

        require(
            parse_bool(
                row["sha256_match"]
            ),
            f"shard {shard}: preflight SHA-256 failed",
        )

        require(
            row["missing_reference_columns"] == "",
            (
                f"shard {shard}: missing common-72 columns"
            ),
        )

        require(
            row["type_mismatches"] == "",
            (
                f"shard {shard}: common-column type mismatch"
            ),
        )

        require(
            row["missing_baseline_columns"] == "",
            (
                f"shard {shard}: missing baseline columns"
            ),
        )

        require(
            row["error"] == "",
            f"shard {shard}: preflight error is nonempty",
        )

    registry20 = load_table(
        args.registry20,
        "\t",
    )

    registry80 = load_table(
        args.registry80,
        "\t",
    )

    manifest20 = load_table(
        args.manifest20,
        ",",
    )

    manifest80 = load_table(
        args.manifest80,
        ",",
    )

    require(
        len(registry20) == 20,
        (
            "expected 20 rows in 20k registry, "
            f"found {len(registry20)}"
        ),
    )

    require(
        len(registry80) == 80,
        (
            "expected 80 rows in 80k registry, "
            f"found {len(registry80)}"
        ),
    )

    require(
        len(manifest20) == 20,
        (
            "expected 20 rows in 20k manifest, "
            f"found {len(manifest20)}"
        ),
    )

    require(
        len(manifest80) == 80,
        (
            "expected 80 rows in 80k manifest, "
            f"found {len(manifest80)}"
        ),
    )

    registry20_by_shard = {
        int(row["local_shard"]): row
        for row in registry20
    }

    registry80_by_shard = {
        int(row["local_shard"]): row
        for row in registry80
    }

    manifest20_by_shard = {
        int(row["local_shard"]): row
        for row in manifest20
    }

    manifest80_by_shard = {
        int(row["local_shard"]): row
        for row in manifest80
    }

    require(
        set(registry20_by_shard)
        == set(range(0, 20)),
        "20k registry is not shards 0–19",
    )

    require(
        set(registry80_by_shard)
        == set(range(20, 100)),
        "80k registry is not shards 20–99",
    )

    require(
        set(manifest20_by_shard)
        == set(range(0, 20)),
        "20k manifest is not shards 0–19",
    )

    require(
        set(manifest80_by_shard)
        == set(range(20, 100)),
        "80k manifest is not shards 20–99",
    )

    receipt20_paths = sorted(
        args.receipts20.glob(
            "*receipt.json"
        )
    )

    receipt80_paths = sorted(
        args.receipts80.glob(
            "*receipt.json"
        )
    )

    require(
        len(receipt20_paths) == 21,
        (
            "expected 21 total historical 20k receipts, "
            f"found {len(receipt20_paths)}"
        ),
    )

    require(
        len(receipt80_paths) == 80,
        (
            "expected 80 canonical 80k receipts, "
            f"found {len(receipt80_paths)}"
        ),
    )

    receipts20_by_name = {
        path.name: (
            path,
            load_json(path),
        )
        for path in receipt20_paths
    }

    receipts80_by_shard: dict[
        int,
        tuple[Path, dict[str, Any]],
    ] = {}

    for path in receipt80_paths:
        receipt = load_json(path)

        local_shard = int(
            receipt_required(
                receipt,
                "local_shard",
                path,
            )
        )

        require(
            local_shard
            not in receipts80_by_shard,
            (
                "duplicate 80k receipt for shard "
                f"{local_shard}"
            ),
        )

        receipts80_by_shard[
            local_shard
        ] = (
            path,
            receipt,
        )

    require(
        set(receipts80_by_shard)
        == set(range(20, 100)),
        "80k receipts are not shards 20–99",
    )

    selected20_receipts: set[str] = set()

    generated_events_total = 0
    candidate_rows_total = 0

    generated_events_by_split: Counter[str] = Counter()
    candidate_rows_by_split: Counter[str] = Counter()

    live_adler32_values: set[
        tuple[str, str]
    ] = set()

    output_rows: list[
        dict[str, object]
    ] = []

    for local_shard in range(100):
        if local_shard < 20:
            registry_row = registry20_by_shard[
                local_shard
            ]

            manifest_row = manifest20_by_shard[
                local_shard
            ]

            expected_campaign = EXPECTED_CAMPAIGN20

            split = manifest_row[
                "immutable_split"
            ]

            receipt_name = manifest_row[
                "receipt_file"
            ]

            require(
                receipt_name
                in receipts20_by_name,
                (
                    f"shard {local_shard}: missing "
                    f"canonical receipt {receipt_name}"
                ),
            )

            receipt_path, receipt = (
                receipts20_by_name[
                    receipt_name
                ]
            )

            selected20_receipts.add(
                receipt_name
            )

            recorded_adler32 = manifest_row[
                "bundle_adler32"
            ].lower()

            adler_reference_source = (
                "immutable_20k_manifest"
            )

        else:
            registry_row = registry80_by_shard[
                local_shard
            ]

            manifest_row = manifest80_by_shard[
                local_shard
            ]

            expected_campaign = EXPECTED_CAMPAIGN80

            split = manifest_row[
                "dataset_split"
            ]

            receipt_path, receipt = (
                receipts80_by_shard[
                    local_shard
                ]
            )

            receipt_name = receipt_path.name

            recorded_adler32 = ""

            adler_reference_source = (
                "live_eos_frozen_by_this_audit"
            )

        expected_inner_shard = (
            16000 + local_shard
        )

        expected_seed = (
            86000 + local_shard
        )

        require(
            registry_row["campaign"]
            == expected_campaign,
            (
                f"shard {local_shard}: "
                "registry campaign mismatch"
            ),
        )

        require(
            manifest_row["campaign"]
            == expected_campaign,
            (
                f"shard {local_shard}: "
                "manifest campaign mismatch"
            ),
        )

        require(
            str(
                receipt_required(
                    receipt,
                    "campaign",
                    receipt_path,
                )
            )
            == expected_campaign,
            (
                f"shard {local_shard}: "
                "receipt campaign mismatch"
            ),
        )

        require(
            int(
                registry_row[
                    "inner_shard"
                ]
            )
            == expected_inner_shard,
            (
                f"shard {local_shard}: "
                "registry inner-shard mismatch"
            ),
        )

        require(
            int(
                manifest_row[
                    "inner_shard"
                ]
            )
            == expected_inner_shard,
            (
                f"shard {local_shard}: "
                "manifest inner-shard mismatch"
            ),
        )

        require(
            int(
                receipt_required(
                    receipt,
                    "inner_shard",
                    receipt_path,
                )
            )
            == expected_inner_shard,
            (
                f"shard {local_shard}: "
                "receipt inner-shard mismatch"
            ),
        )

        require(
            int(
                registry_row["seed"]
            )
            == expected_seed,
            (
                f"shard {local_shard}: "
                "registry seed mismatch"
            ),
        )

        require(
            int(
                manifest_row["seed"]
            )
            == expected_seed,
            (
                f"shard {local_shard}: "
                "manifest seed mismatch"
            ),
        )

        require(
            int(
                receipt_required(
                    receipt,
                    "seed",
                    receipt_path,
                )
            )
            == expected_seed,
            (
                f"shard {local_shard}: "
                "receipt seed mismatch"
            ),
        )

        require(
            int(
                registry_row[
                    "generated_events"
                ]
            )
            == 1000,
            (
                f"shard {local_shard}: "
                "registry event count is not 1000"
            ),
        )

        require(
            int(
                manifest_row["n_events"]
            )
            == 1000,
            (
                f"shard {local_shard}: "
                "manifest event count is not 1000"
            ),
        )

        require(
            int(
                receipt_required(
                    receipt,
                    "n_events",
                    receipt_path,
                )
            )
            == 1000,
            (
                f"shard {local_shard}: "
                "receipt event count is not 1000"
            ),
        )

        require(
            registry_row[
                "immutable_split"
            ]
            == split,
            (
                f"shard {local_shard}: "
                "registry/manifest split mismatch"
            ),
        )

        remote_bundle = manifest_row[
            "remote_bundle"
        ]

        require(
            registry_row[
                "remote_bundle"
            ]
            == remote_bundle,
            (
                f"shard {local_shard}: "
                "registry/manifest bundle mismatch"
            ),
        )

        require(
            str(
                receipt_required(
                    receipt,
                    "remote_bundle",
                    receipt_path,
                )
            )
            == remote_bundle,
            (
                f"shard {local_shard}: "
                "receipt bundle mismatch"
            ),
        )

        require(
            str(
                receipt_required(
                    receipt,
                    "stage",
                    receipt_path,
                )
            )
            == "complete_copied_and_verified",
            (
                f"shard {local_shard}: "
                "receipt stage is not complete"
            ),
        )

        require(
            int(
                receipt_required(
                    receipt,
                    "exit_status",
                    receipt_path,
                )
            )
            == 0,
            (
                f"shard {local_shard}: "
                "receipt exit status is nonzero"
            ),
        )

        candidate_path = Path(
            registry_row[
                "candidate_parquet"
            ]
        )

        require(
            candidate_path.is_file(),
            (
                f"shard {local_shard}: missing "
                f"candidate Parquet {candidate_path}"
            ),
        )

        expected_candidate_rows = int(
            registry_row[
                "candidate_rows"
            ]
        )

        expected_candidate_sha256 = (
            registry_row[
                "candidate_parquet_sha256"
            ]
        )

        preflight_row = preflight_by_shard[
            local_shard
        ]

        require(
            preflight_row[
                "candidate_parquet"
            ]
            == str(candidate_path),
            (
                f"shard {local_shard}: "
                "registry/preflight path mismatch"
            ),
        )

        require(
            int(
                preflight_row[
                    "actual_rows"
                ]
            )
            == expected_candidate_rows,
            (
                f"shard {local_shard}: "
                "registry/preflight row mismatch"
            ),
        )

        require(
            preflight_row[
                "actual_sha256"
            ]
            == expected_candidate_sha256,
            (
                f"shard {local_shard}: "
                "registry/preflight SHA mismatch"
            ),
        )

        parquet = pq.ParquetFile(
            candidate_path
        )

        actual_candidate_rows = int(
            parquet.metadata.num_rows
        )

        actual_candidate_sha256 = (
            sha256_file(
                candidate_path
            )
        )

        observed_schema_sha256, columns = (
            schema_signature(
                candidate_path
            )
        )

        require(
            actual_candidate_rows
            == expected_candidate_rows,
            (
                f"shard {local_shard}: "
                "candidate row-count mismatch"
            ),
        )

        require(
            actual_candidate_sha256
            == expected_candidate_sha256,
            (
                f"shard {local_shard}: "
                "candidate SHA-256 mismatch"
            ),
        )

        require(
            len(columns) == 75,
            (
                f"shard {local_shard}: "
                f"expected 75 columns, found {len(columns)}"
            ),
        )

        require(
            observed_schema_sha256
            == signal_schema_sha256,
            (
                f"shard {local_shard}: "
                "signal schema mismatch"
            ),
        )

        require(
            registry_row[
                "analysis_sample"
            ]
            == "ggF_HH4b_SMnorm",
            (
                f"shard {local_shard}: "
                "analysis-sample mismatch"
            ),
        )

        require(
            registry_row["status"]
            == "pass",
            (
                f"shard {local_shard}: "
                "registry status is not pass"
            ),
        )

        training_authorized = parse_bool(
            registry_row[
                "training_authorized"
            ]
        )

        validation_authorized = parse_bool(
            registry_row[
                "validation_authorized"
            ]
        )

        test_sealed = parse_bool(
            registry_row[
                "test_sealed"
            ]
        )

        physics_yield_authorized = parse_bool(
            registry_row[
                "physics_yield_authorized"
            ]
        )

        require(
            physics_yield_authorized
            is False,
            (
                f"shard {local_shard}: "
                "physics yield unexpectedly authorized"
            ),
        )

        if split == "train":
            require(
                training_authorized
                and not validation_authorized
                and not test_sealed,
                (
                    f"shard {local_shard}: "
                    "invalid train flags"
                ),
            )

        elif split == "validation":
            require(
                validation_authorized
                and not training_authorized
                and not test_sealed,
                (
                    f"shard {local_shard}: "
                    "invalid validation flags"
                ),
            )

        elif split == "test":
            require(
                test_sealed
                and not training_authorized
                and not validation_authorized,
                (
                    f"shard {local_shard}: "
                    "invalid sealed-test flags"
                ),
            )

        else:
            raise RuntimeError(
                (
                    f"shard {local_shard}: "
                    f"unknown split {split!r}"
                )
            )

        live_adler32 = eos_adler32(
            args.eos_host,
            remote_bundle,
        )

        if local_shard < 20:
            require(
                live_adler32
                == recorded_adler32,
                (
                    f"shard {local_shard}: "
                    "live EOS Adler-32 differs "
                    "from immutable 20k manifest"
                ),
            )

        live_adler32_values.add((
            remote_bundle,
            live_adler32,
        ))

        generated_events_total += 1000
        candidate_rows_total += (
            actual_candidate_rows
        )

        generated_events_by_split[
            split
        ] += 1000

        candidate_rows_by_split[
            split
        ] += actual_candidate_rows

        output_rows.append({
            "signal_mode":
                "ggf_hh4b",
            "member_index":
                local_shard,
            "campaign":
                expected_campaign,
            "local_shard":
                local_shard,
            "inner_shard":
                expected_inner_shard,
            "seed":
                expected_seed,
            "generated_events":
                1000,
            "immutable_split":
                split,
            "candidate_rows":
                actual_candidate_rows,
            "candidate_parquet":
                str(candidate_path),
            "candidate_parquet_sha256":
                actual_candidate_sha256,
            "candidate_schema_columns":
                75,
            "candidate_schema_sha256":
                signal_schema_sha256,
            "common72_schema_sha256":
                common72_schema_sha256,
            "remote_bundle":
                remote_bundle,
            "bundle_adler32":
                live_adler32,
            "bundle_adler32_reference_source":
                adler_reference_source,
            "receipt_file":
                receipt_name,
            "receipt_path":
                str(receipt_path),
            "training_authorized":
                training_authorized,
            "validation_authorized":
                validation_authorized,
            "test_sealed":
                test_sealed,
            "physics_yield_authorized":
                physics_yield_authorized,
            "status":
                "pass",
        })

        if (
            (local_shard + 1) % 20
            == 0
        ):
            print(
                "GGF_EOS_CHECKSUM_PROGRESS="
                f"{local_shard + 1}/100"
            )

    require(
        len(output_rows) == 100,
        "final registry does not contain 100 members",
    )

    require(
        generated_events_total == 100_000,
        (
            "generated-event total is "
            f"{generated_events_total}, not 100000"
        ),
    )

    require(
        len({
            row["seed"]
            for row in output_rows
        })
        == 100,
        "duplicate seed found",
    )

    require(
        len({
            row["inner_shard"]
            for row in output_rows
        })
        == 100,
        "duplicate inner shard found",
    )

    require(
        len({
            row["remote_bundle"]
            for row in output_rows
        })
        == 100,
        "duplicate remote bundle found",
    )

    require(
        len(live_adler32_values) == 100,
        "live EOS checksum membership is not unique",
    )

    extra20_receipts = sorted(
        set(receipts20_by_name)
        - selected20_receipts
    )

    require(
        len(extra20_receipts) == 1,
        (
            "expected one excluded historical "
            f"20k receipt, found {len(extra20_receipts)}"
        ),
    )

    outdir.mkdir(
        parents=True,
        exist_ok=False,
    )

    registry_path = (
        outdir
        / "ggf_hh4b_canonical100k_registry.tsv"
    )

    with registry_path.open(
        "w",
        newline="",
    ) as handle:
        columns = list(
            output_rows[0]
        )

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    summary = {
        "schema_version": 2,
        "status": "pass",
        "signal_mode": "ggf_hh4b",
        "canonical_members": 100,
        "generated_events":
            generated_events_total,
        "candidate_rows":
            candidate_rows_total,
        "generated_events_by_split":
            dict(
                generated_events_by_split
            ),
        "candidate_rows_by_split":
            dict(
                candidate_rows_by_split
            ),
        "signal_schema_columns": 75,
        "signal_schema_sha256":
            signal_schema_sha256,
        "common_ml_schema_columns": 72,
        "common_ml_schema_sha256":
            common72_schema_sha256,
        "signal_extra_columns":
            signal_extra_columns,
        "signal_schema_policy":
            "uniform_75_column_superset_of_common_72",
        "all_common72_columns_present":
            True,
        "common72_types_match":
            True,
        "live_eos_adler32_checks":
            100,
        "canonical_20k_receipts":
            len(selected20_receipts),
        "excluded_historical_20k_receipts":
            extra20_receipts,
        "canonical_80k_receipts":
            len(receipt80_paths),
        "membership_frozen":
            True,
        "training_members_authorized":
            True,
        "validation_members_authorized":
            True,
        "test_members_sealed":
            generated_events_by_split[
                "test"
            ] > 0,
        "physics_yield_authorized":
            False,
        "registry":
            str(registry_path),
    }

    summary_path = (
        outdir
        / "ggf_hh4b_canonical100k_registry.json"
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
        "GGF_HH4B_CANONICAL100K_REGISTRY_VALID"
    )
    print(
        "GGF_HH4B_CANONICAL100K_MEMBERSHIP_FROZEN"
    )
    print(
        "GGF_HH4B_UNIFORM75_SCHEMA_VALID"
    )
    print(
        "GGF_HH4B_COMMON72_ML_SCHEMA_COMPATIBLE"
    )
    print(
        "GGF_HH4B_GENERATED_EVENTS=100000"
    )
    print(
        "GGF_HH4B_CANDIDATE_ROWS="
        f"{candidate_rows_total}"
    )
    print(
        "GGF_HH4B_GENERATED_EVENTS_BY_SPLIT="
        + json.dumps(
            dict(
                generated_events_by_split
            ),
            sort_keys=True,
        )
    )
    print(
        "GGF_HH4B_CANDIDATE_ROWS_BY_SPLIT="
        + json.dumps(
            dict(
                candidate_rows_by_split
            ),
            sort_keys=True,
        )
    )
    print(
        "GGF_HH4B_LIVE_EOS_CHECKSUMS_VALID=100"
    )
    print(
        "GGF_HH4B_RICH_V2_RECONSTRUCTION_NOT_REQUIRED"
    )
    print(
        "GGF_HH4B_PHYSICS_YIELD_NOT_YET_AUTHORIZED"
    )
    print(
        f"summary_json={summary_path}"
    )
    print(
        f"registry_tsv={registry_path}"
    )


if __name__ == "__main__":
    main()
