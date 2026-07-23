#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import tempfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


EXPECTED_MEMBERS = 10
EXPECTED_EVENTS = 100_000
EXPECTED_CANDIDATE_ROWS = 6_283
EXPECTED_EVENTS_BY_SPLIT = {
    "train": 80_000,
    "validation": 20_000,
}
EXPECTED_CANDIDATE_ROWS_BY_SPLIT = {
    "train": 5_082,
    "validation": 1_201,
}


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def load_tsv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        newline="",
        errors="replace",
    ) as handle:
        return list(
            csv.DictReader(
                handle,
                delimiter="\t",
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

    normalized = str(value).strip().lower()

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


def adler32_file(
    path: Path,
) -> str:
    checksum = 1

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            checksum = zlib.adler32(
                block,
                checksum,
            )

    return f"{checksum & 0xffffffff:08x}"


def schema_signature(
    path: Path,
) -> tuple[str, int]:
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

    encoded = json.dumps(
        fields,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    signature = hashlib.sha256(
        encoded
    ).hexdigest()

    return signature, len(fields)


def remote_url(
    eos_host: str,
    lfn: str,
) -> str:
    # XRootD absolute LFNs require a double slash after
    # the host: root://host//store/user/...
    return (
        eos_host.rstrip("/")
        + "//"
        + lfn.lstrip("/")
    )


def run_command(
    command: list[str],
) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return completed.stdout


def eos_list(
    eos_host: str,
    lfn: str,
) -> list[str]:
    output = run_command([
        "xrdfs",
        eos_host,
        "ls",
        lfn,
    ])

    return [
        line.strip()
        for line in output.splitlines()
        if line.strip()
    ]


def eos_size(
    eos_host: str,
    lfn: str,
) -> int:
    output = run_command([
        "xrdfs",
        eos_host,
        "stat",
        lfn,
    ])

    match = re.search(
        r"(?m)^Size:\s*(\d+)\s*$",
        output,
    )

    require(
        match is not None,
        (
            f"could not parse EOS size for {lfn}: "
            f"{output!r}"
        ),
    )

    return int(match.group(1))


def eos_adler32(
    eos_host: str,
    lfn: str,
) -> str:
    output = run_command([
        "xrdfs",
        eos_host,
        "query",
        "checksum",
        lfn,
    ])

    matches = re.findall(
        r"\b[0-9a-fA-F]{8}\b",
        output,
    )

    require(
        matches,
        (
            f"could not parse EOS checksum for {lfn}: "
            f"{output!r}"
        ),
    )

    return matches[-1].lower()


def xrdcp_to_local(
    eos_host: str,
    lfn: str,
    destination: Path,
) -> None:
    subprocess.run(
        [
            "xrdcp",
            "-f",
            "--nopbar",
            remote_url(
                eos_host,
                lfn,
            ),
            str(destination),
        ],
        check=True,
    )

    require(
        destination.is_file(),
        f"xrdcp did not create {destination}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-registry",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--materialization-plan",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--ggf-summary",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--eos-host",
        required=True,
    )

    parser.add_argument(
        "--eos-base",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    for path in (
        args.source_registry,
        args.materialization_plan,
        args.ggf_summary,
    ):
        require(
            path.is_file(),
            f"missing required input {path}",
        )

    require(
        not args.outdir.exists(),
        f"refusing to overwrite {args.outdir}",
    )

    source_rows = load_tsv(
        args.source_registry
    )

    plan_rows = load_tsv(
        args.materialization_plan
    )

    ggf_summary = load_json(
        args.ggf_summary
    )

    require(
        len(source_rows) == EXPECTED_MEMBERS,
        (
            "source registry has "
            f"{len(source_rows)} rows, not 10"
        ),
    )

    require(
        len(plan_rows) == EXPECTED_MEMBERS,
        (
            "materialization plan has "
            f"{len(plan_rows)} rows, not 10"
        ),
    )

    common72_schema_sha256 = str(
        ggf_summary[
            "common_ml_schema_sha256"
        ]
    )

    require(
        re.fullmatch(
            r"[0-9a-f]{64}",
            common72_schema_sha256,
        )
        is not None,
        "invalid common-72 schema SHA-256",
    )

    source_by_tag = {
        row["target_tag"]: row
        for row in source_rows
    }

    plan_by_tag = {
        row["target_tag"]: row
        for row in plan_rows
    }

    require(
        len(source_by_tag) == EXPECTED_MEMBERS,
        "source target tags are not unique",
    )

    require(
        len(plan_by_tag) == EXPECTED_MEMBERS,
        "plan target tags are not unique",
    )

    require(
        set(source_by_tag)
        == set(plan_by_tag),
        (
            "source registry and materialization "
            "plan membership differ"
        ),
    )

    expected_candidate_lfns = {
        (
            f"{args.eos_base}/parquets/"
            f"{target_tag}_hh4b_candidates_v2.parquet"
        )
        for target_tag in plan_by_tag
    }

    expected_receipt_lfns = {
        (
            f"{args.eos_base}/receipts/"
            f"{target_tag}_candidate_materialization_receipt.json"
        )
        for target_tag in plan_by_tag
    }

    observed_candidate_lfns = {
        path
        for path in eos_list(
            args.eos_host,
            f"{args.eos_base}/parquets",
        )
        if path.endswith(".parquet")
    }

    observed_receipt_lfns = {
        path
        for path in eos_list(
            args.eos_host,
            f"{args.eos_base}/receipts",
        )
        if path.endswith(".json")
    }

    require(
        observed_candidate_lfns
        == expected_candidate_lfns,
        (
            "EOS candidate membership differs from plan.\n"
            f"missing={sorted(expected_candidate_lfns - observed_candidate_lfns)}\n"
            f"extra={sorted(observed_candidate_lfns - expected_candidate_lfns)}"
        ),
    )

    require(
        observed_receipt_lfns
        == expected_receipt_lfns,
        (
            "EOS receipt membership differs from plan.\n"
            f"missing={sorted(expected_receipt_lfns - observed_receipt_lfns)}\n"
            f"extra={sorted(observed_receipt_lfns - expected_receipt_lfns)}"
        ),
    )

    output_rows: list[
        dict[str, object]
    ] = []

    generated_events = 0
    candidate_rows_total = 0

    generated_by_split: Counter[str] = Counter()
    candidates_by_split: Counter[str] = Counter()

    candidate_paths: set[str] = set()
    candidate_sha256_values: set[str] = set()
    source_bundles: set[str] = set()

    with tempfile.TemporaryDirectory(
        prefix="vbf_hh4b_materialization_audit.",
        dir="/tmp",
    ) as temporary_directory:
        temporary = Path(
            temporary_directory
        )

        for registry_index in range(
            EXPECTED_MEMBERS
        ):
            matching_plan_rows = [
                row
                for row in plan_rows
                if int(
                    row["registry_index"]
                )
                == registry_index
            ]

            require(
                len(matching_plan_rows) == 1,
                (
                    "expected exactly one plan member "
                    f"for registry index {registry_index}"
                ),
            )

            plan = matching_plan_rows[0]

            target_tag = plan[
                "target_tag"
            ]

            source = source_by_tag[
                target_tag
            ]

            split = plan[
                "dataset_split"
            ]

            events = int(
                plan["events"]
            )

            expected_rows = int(
                plan["candidate_rows"]
            )

            expected_candidate_sha256 = (
                plan["candidate_sha256"]
            )

            expected_bundle_sha256 = (
                plan["bundle_sha256"]
            )

            expected_bundle_adler32 = (
                plan["bundle_adler32"].lower()
            )

            source_bundle = plan[
                "remote_bundle"
            ]

            source_receipt = plan[
                "source_receipt"
            ]

            require(
                source["family"]
                == "vbf_hh4b_sm",
                (
                    f"{target_tag}: unexpected "
                    "source family"
                ),
            )

            require(
                source["target_tag"]
                == target_tag,
                f"{target_tag}: source target mismatch",
            )

            require(
                int(source["shard_id"])
                == int(plan["shard_id"]),
                f"{target_tag}: shard mismatch",
            )

            require(
                source["dataset_split"]
                == split,
                f"{target_tag}: split mismatch",
            )

            require(
                int(source["events"])
                == events,
                f"{target_tag}: event-count mismatch",
            )

            require(
                int(source["candidate_rows"])
                == expected_rows,
                (
                    f"{target_tag}: source candidate "
                    "row-count mismatch"
                ),
            )

            require(
                source["remote_bundle"]
                == source_bundle,
                f"{target_tag}: source bundle mismatch",
            )

            require(
                source["bundle_sha256"]
                == expected_bundle_sha256,
                (
                    f"{target_tag}: source bundle "
                    "SHA-256 mismatch"
                ),
            )

            require(
                source["bundle_adler32"].lower()
                == expected_bundle_adler32,
                (
                    f"{target_tag}: source bundle "
                    "Adler-32 mismatch"
                ),
            )

            require(
                source["source_receipt"]
                == source_receipt,
                (
                    f"{target_tag}: source receipt "
                    "path mismatch"
                ),
            )

            require(
                parse_bool(
                    source[
                        "count_toward_signal_target"
                    ]
                ),
                (
                    f"{target_tag}: source does not "
                    "count toward target"
                ),
            )

            require(
                parse_bool(
                    source[
                        "final_audit_valid"
                    ]
                ),
                (
                    f"{target_tag}: source final "
                    "audit is invalid"
                ),
            )

            require(
                not parse_bool(
                    source[
                        "physics_yield_authorized"
                    ]
                ),
                (
                    f"{target_tag}: physics yield "
                    "unexpectedly authorized"
                ),
            )

            candidate_lfn = (
                f"{args.eos_base}/parquets/"
                f"{target_tag}_hh4b_candidates_v2.parquet"
            )

            receipt_lfn = (
                f"{args.eos_base}/receipts/"
                f"{target_tag}_candidate_materialization_receipt.json"
            )

            local_candidate = (
                temporary
                / f"{registry_index}_candidate.parquet"
            )

            local_receipt = (
                temporary
                / f"{registry_index}_receipt.json"
            )

            xrdcp_to_local(
                args.eos_host,
                receipt_lfn,
                local_receipt,
            )

            receipt = load_json(
                local_receipt
            )

            require(
                receipt["status"] == "pass",
                f"{target_tag}: receipt status is not pass",
            )

            require(
                receipt["stage"]
                == "candidate_materialized_and_verified",
                f"{target_tag}: incomplete receipt stage",
            )

            require(
                int(receipt["exit_status"]) == 0,
                f"{target_tag}: nonzero receipt exit",
            )

            require(
                receipt["signal_mode"]
                == "vbf_hh4b",
                f"{target_tag}: receipt mode mismatch",
            )

            require(
                int(receipt["registry_index"])
                == registry_index,
                f"{target_tag}: receipt index mismatch",
            )

            require(
                receipt["target_tag"]
                == target_tag,
                f"{target_tag}: receipt target mismatch",
            )

            require(
                int(receipt["shard_id"])
                == int(plan["shard_id"]),
                f"{target_tag}: receipt shard mismatch",
            )

            require(
                receipt["dataset_split"]
                == split,
                f"{target_tag}: receipt split mismatch",
            )

            require(
                int(receipt["generated_events"])
                == events,
                (
                    f"{target_tag}: receipt generated "
                    "event mismatch"
                ),
            )

            require(
                int(
                    receipt[
                        "candidate_rows_expected"
                    ]
                )
                == expected_rows,
                (
                    f"{target_tag}: receipt candidate "
                    "row mismatch"
                ),
            )

            require(
                receipt["candidate_sha256"]
                == expected_candidate_sha256,
                (
                    f"{target_tag}: receipt candidate "
                    "SHA-256 mismatch"
                ),
            )

            require(
                receipt["source_remote_bundle"]
                == source_bundle,
                (
                    f"{target_tag}: receipt source "
                    "bundle mismatch"
                ),
            )

            require(
                receipt["source_bundle_sha256"]
                == expected_bundle_sha256,
                (
                    f"{target_tag}: receipt bundle "
                    "SHA-256 mismatch"
                ),
            )

            require(
                receipt[
                    "source_bundle_adler32"
                ].lower()
                == expected_bundle_adler32,
                (
                    f"{target_tag}: receipt bundle "
                    "Adler-32 mismatch"
                ),
            )

            require(
                receipt["source_receipt"]
                == source_receipt,
                (
                    f"{target_tag}: receipt provenance "
                    "path mismatch"
                ),
            )

            require(
                receipt["remote_candidate"]
                == candidate_lfn,
                (
                    f"{target_tag}: remote candidate "
                    "path mismatch"
                ),
            )

            require(
                not parse_bool(
                    receipt[
                        "physics_yield_authorized"
                    ]
                ),
                (
                    f"{target_tag}: materialization "
                    "receipt authorizes physics yield"
                ),
            )

            xrdcp_to_local(
                args.eos_host,
                candidate_lfn,
                local_candidate,
            )

            local_candidate_size = (
                local_candidate.stat().st_size
            )

            local_candidate_sha256 = (
                sha256_file(
                    local_candidate
                )
            )

            local_candidate_adler32 = (
                adler32_file(
                    local_candidate
                )
            )

            require(
                local_candidate_size
                == int(
                    receipt[
                        "candidate_size_bytes"
                    ]
                ),
                (
                    f"{target_tag}: candidate size "
                    "differs from receipt"
                ),
            )

            require(
                local_candidate_sha256
                == expected_candidate_sha256,
                (
                    f"{target_tag}: downloaded candidate "
                    "SHA-256 mismatch"
                ),
            )

            require(
                local_candidate_adler32
                == receipt[
                    "candidate_adler32"
                ].lower(),
                (
                    f"{target_tag}: downloaded candidate "
                    "Adler-32 mismatch"
                ),
            )

            live_size = eos_size(
                args.eos_host,
                candidate_lfn,
            )

            live_adler32 = eos_adler32(
                args.eos_host,
                candidate_lfn,
            )

            require(
                live_size
                == local_candidate_size,
                (
                    f"{target_tag}: live EOS candidate "
                    "size mismatch"
                ),
            )

            require(
                live_adler32
                == local_candidate_adler32,
                (
                    f"{target_tag}: live EOS candidate "
                    "Adler-32 mismatch"
                ),
            )

            parquet = pq.ParquetFile(
                local_candidate
            )

            actual_rows = int(
                parquet.metadata.num_rows
            )

            observed_schema_sha256, column_count = (
                schema_signature(
                    local_candidate
                )
            )

            require(
                actual_rows == expected_rows,
                (
                    f"{target_tag}: actual candidate "
                    f"rows {actual_rows}, expected "
                    f"{expected_rows}"
                ),
            )

            require(
                column_count == 72,
                (
                    f"{target_tag}: expected 72 columns, "
                    f"found {column_count}"
                ),
            )

            require(
                observed_schema_sha256
                == common72_schema_sha256,
                (
                    f"{target_tag}: schema differs from "
                    "frozen common-72 schema"
                ),
            )

            generated_events += events
            candidate_rows_total += actual_rows

            generated_by_split[
                split
            ] += events

            candidates_by_split[
                split
            ] += actual_rows

            candidate_paths.add(
                candidate_lfn
            )

            candidate_sha256_values.add(
                local_candidate_sha256
            )

            source_bundles.add(
                source_bundle
            )

            output_rows.append({
                "signal_mode":
                    "vbf_hh4b",
                "member_index":
                    registry_index,
                "target_tag":
                    target_tag,
                "source_campaign":
                    source[
                        "source_campaign"
                    ],
                "shard_id":
                    int(plan["shard_id"]),
                "seed":
                    int(source["seed"]),
                "pythia_seed":
                    int(source["pythia_seed"]),
                "generated_events":
                    events,
                "immutable_split":
                    split,
                "dataset_role":
                    source["dataset_role"],
                "candidate_rows":
                    actual_rows,
                "candidate_parquet":
                    candidate_lfn,
                "candidate_parquet_sha256":
                    local_candidate_sha256,
                "candidate_adler32":
                    local_candidate_adler32,
                "candidate_size_bytes":
                    local_candidate_size,
                "candidate_schema_columns":
                    72,
                "candidate_schema_sha256":
                    observed_schema_sha256,
                "source_remote_bundle":
                    source_bundle,
                "source_bundle_sha256":
                    expected_bundle_sha256,
                "source_bundle_adler32":
                    expected_bundle_adler32,
                "source_receipt":
                    source_receipt,
                "materialization_receipt":
                    receipt_lfn,
                "training_authorized":
                    split == "train",
                "validation_authorized":
                    split == "validation",
                "test_sealed":
                    False,
                "physics_yield_authorized":
                    False,
                "status":
                    "pass",
            })

            print(
                "VBF_MATERIALIZATION_AUDIT_PROGRESS="
                f"{registry_index + 1}/10"
            )

    require(
        len(output_rows) == EXPECTED_MEMBERS,
        (
            "output registry does not contain "
            "10 members"
        ),
    )

    require(
        generated_events == EXPECTED_EVENTS,
        (
            f"generated events are {generated_events}, "
            "not 100000"
        ),
    )

    require(
        candidate_rows_total
        == EXPECTED_CANDIDATE_ROWS,
        (
            "candidate rows are "
            f"{candidate_rows_total}, not 6283"
        ),
    )

    require(
        dict(generated_by_split)
        == EXPECTED_EVENTS_BY_SPLIT,
        (
            "unexpected generated split totals: "
            f"{dict(generated_by_split)}"
        ),
    )

    require(
        dict(candidates_by_split)
        == EXPECTED_CANDIDATE_ROWS_BY_SPLIT,
        (
            "unexpected candidate split totals: "
            f"{dict(candidates_by_split)}"
        ),
    )

    require(
        len(candidate_paths) == EXPECTED_MEMBERS,
        "candidate paths are not unique",
    )

    require(
        len(candidate_sha256_values)
        == EXPECTED_MEMBERS,
        "candidate SHA-256 values are not unique",
    )

    require(
        len(source_bundles) == EXPECTED_MEMBERS,
        "source bundles are not unique",
    )

    args.outdir.mkdir(
        parents=True,
        exist_ok=False,
    )

    registry_output = (
        args.outdir
        / "vbf_hh4b_canonical100k_materialized_registry.tsv"
    )

    with registry_output.open(
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
        writer.writerows(output_rows)

    summary = {
        "schema_version": 1,
        "status": "pass",
        "signal_mode": "vbf_hh4b",
        "canonical_members":
            EXPECTED_MEMBERS,
        "generated_events":
            generated_events,
        "generated_events_by_split":
            dict(generated_by_split),
        "candidate_rows":
            candidate_rows_total,
        "candidate_rows_by_split":
            dict(candidates_by_split),
        "candidate_schema_columns": 72,
        "candidate_schema_sha256":
            common72_schema_sha256,
        "live_eos_candidate_sizes_valid":
            EXPECTED_MEMBERS,
        "live_eos_candidate_adler32_valid":
            EXPECTED_MEMBERS,
        "candidate_sha256_valid":
            EXPECTED_MEMBERS,
        "materialization_receipts_valid":
            EXPECTED_MEMBERS,
        "membership_frozen": True,
        "training_members_authorized": True,
        "validation_members_authorized": True,
        "test_members_present": False,
        "test_members_sealed": False,
        "physics_yield_authorized": False,
        "candidate_reconstruction_required": False,
        "new_event_generation_required": False,
        "registry":
            str(registry_output),
    }

    summary_output = (
        args.outdir
        / "vbf_hh4b_canonical100k_materialization.json"
    )

    summary_output.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "VBF_HH4B_CANONICAL100K_MATERIALIZATION_VALID"
    )
    print(
        "VBF_HH4B_CANONICAL100K_MEMBERSHIP_FROZEN"
    )
    print(
        "VBF_HH4B_GENERATED_EVENTS=100000"
    )
    print(
        "VBF_HH4B_CANDIDATE_ROWS=6283"
    )
    print(
        "VBF_HH4B_COMMON72_SCHEMA_VALID"
    )
    print(
        "VBF_HH4B_MATERIALIZATION_RECEIPTS_VALID=10"
    )
    print(
        "VBF_HH4B_LIVE_EOS_CANDIDATES_VALID=10"
    )
    print(
        "VBF_HH4B_NO_TEST_SPLIT"
    )
    print(
        "VBF_HH4B_RECONSTRUCTION_NOT_REQUIRED"
    )
    print(
        "VBF_HH4B_PHYSICS_YIELD_NOT_YET_AUTHORIZED"
    )
    print(
        f"summary_json={summary_output}"
    )
    print(
        f"registry_tsv={registry_output}"
    )


if __name__ == "__main__":
    main()
