#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
import subprocess


EXPECTED_SPLIT_SHARDS = {
    "train": 14,
    "validation": 3,
    "test": 3,
}

EXPECTED_SPLIT_EVENTS = {
    "train": 14000,
    "validation": 3000,
    "test": 3000,
}

EXPECTED_EVENTS = 20000
EXPECTED_ROWS = 20


def integer(
    value: object,
    default: int = 0,
) -> int:
    text = str(
        value
        if value is not None
        else ""
    ).strip()

    if not text:
        return default

    return int(text)


def normalize_adler(
    value: object,
) -> str:
    text = str(
        value or ""
    ).strip().lower()

    if text.startswith("adler32:"):
        text = text.split(
            ":",
            1,
        )[1]

    if text.startswith("0x"):
        text = text[2:]

    return text.zfill(8)


def run_command(
    command: list[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--receipt-dir",
        required=True,
    )

    parser.add_argument(
        "--eos-host",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    manifest_path = Path(
        args.manifest
    ).resolve()

    receipt_dir = Path(
        args.receipt_dir
    ).resolve()

    output_path = Path(
        args.output
    ).resolve()

    errors: list[str] = []

    if not manifest_path.is_file():
        raise SystemExit(
            f"ERROR: missing manifest: {manifest_path}"
        )

    if not receipt_dir.is_dir():
        raise SystemExit(
            f"ERROR: missing receipt directory: {receipt_dir}"
        )

    with manifest_path.open(
        newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    required_fields = {
        "campaign",
        "local_shard",
        "inner_shard",
        "seed",
        "n_events",
        "immutable_split",
        "remote_bundle",
        "bundle_adler32",
        "finished_utc",
        "receipt_file",
    }

    missing_fields = sorted(
        required_fields
        - set(fieldnames)
    )

    if missing_fields:
        errors.append(
            "manifest missing fields: "
            + ", ".join(missing_fields)
        )

    if len(rows) != EXPECTED_ROWS:
        errors.append(
            f"expected {EXPECTED_ROWS} manifest rows, "
            f"found {len(rows)}"
        )

    campaigns = {
        row.get(
            "campaign",
            "",
        )
        for row in rows
    }

    if len(campaigns) != 1:
        errors.append(
            "manifest does not contain exactly one campaign"
        )

    for field in (
        "receipt_file",
        "local_shard",
        "inner_shard",
        "seed",
        "remote_bundle",
    ):
        values = [
            row.get(
                field,
                "",
            )
            for row in rows
        ]

        if len(set(values)) != len(values):
            errors.append(
                f"duplicate manifest value: {field}"
            )

    valid_records = []

    split_shards = Counter()
    split_events = Counter()

    seen_bundles = set()

    manifest_receipt_names = {
        row["receipt_file"]
        for row in rows
    }

    for row in rows:
        receipt_path = (
            receipt_dir
            / row["receipt_file"]
        )

        if not receipt_path.is_file():
            errors.append(
                f"missing receipt: {receipt_path.name}"
            )
            continue

        try:
            receipt = json.loads(
                receipt_path.read_text()
            )
        except Exception as exc:
            errors.append(
                f"{receipt_path.name}: "
                f"unreadable receipt: {exc}"
            )
            continue

        expected_events = integer(
            row["n_events"]
        )

        expected_adler = normalize_adler(
            row["bundle_adler32"]
        )

        remote_bundle = row[
            "remote_bundle"
        ]

        immutable_split = row[
            "immutable_split"
        ]

        checks = {
            "campaign": (
                receipt.get("campaign")
                == row["campaign"]
            ),
            "local_shard": (
                integer(
                    receipt.get(
                        "local_shard",
                    ),
                    -1,
                )
                == integer(
                    row["local_shard"],
                    -1,
                )
            ),
            "inner_shard": (
                integer(
                    receipt.get(
                        "inner_shard",
                    ),
                    -1,
                )
                == integer(
                    row["inner_shard"],
                    -1,
                )
            ),
            "seed": (
                integer(
                    receipt.get(
                        "seed",
                    ),
                    -1,
                )
                == integer(
                    row["seed"],
                    -1,
                )
            ),
            "n_events": (
                integer(
                    receipt.get(
                        "n_events",
                    ),
                    -1,
                )
                == expected_events
            ),
            "stage": (
                receipt.get("stage")
                == "complete_copied_and_verified"
            ),
            "exit_status": (
                integer(
                    receipt.get(
                        "exit_status",
                    ),
                    -1,
                )
                == 0
            ),
            "remote_bundle": (
                receipt.get(
                    "remote_bundle"
                )
                == remote_bundle
            ),
            "receipt_adler32": (
                normalize_adler(
                    receipt.get(
                        "adler32"
                    )
                )
                == expected_adler
            ),
            "immutable_split": (
                immutable_split
                in {
                    "train",
                    "validation",
                    "test",
                }
            ),
        }

        failed_checks = [
            name
            for name, passed
            in checks.items()
            if not passed
        ]

        if failed_checks:
            errors.append(
                f"{receipt_path.name}: "
                + ", ".join(
                    failed_checks
                )
            )
            continue

        stat_result = run_command([
            "xrdfs",
            args.eos_host,
            "stat",
            remote_bundle,
        ])

        if stat_result.returncode != 0:
            errors.append(
                f"{receipt_path.name}: EOS stat failed: "
                + stat_result.stderr.strip()
            )
            continue

        checksum_result = run_command([
            "xrdfs",
            args.eos_host,
            "query",
            "checksum",
            remote_bundle,
        ])

        if checksum_result.returncode != 0:
            errors.append(
                f"{receipt_path.name}: "
                "EOS checksum query failed: "
                + checksum_result.stderr.strip()
            )
            continue

        checksum_parts = (
            checksum_result.stdout
            .strip()
            .split()
        )

        observed_adler = normalize_adler(
            checksum_parts[-1]
            if checksum_parts
            else ""
        )

        if observed_adler != expected_adler:
            errors.append(
                f"{receipt_path.name}: "
                f"EOS Adler-32 mismatch: "
                f"{observed_adler} != {expected_adler}"
            )
            continue

        if remote_bundle in seen_bundles:
            errors.append(
                f"{receipt_path.name}: duplicate EOS bundle"
            )
            continue

        seen_bundles.add(
            remote_bundle
        )

        split_shards[
            immutable_split
        ] += 1

        split_events[
            immutable_split
        ] += expected_events

        valid_records.append({
            "receipt": str(
                receipt_path
            ),
            "local_shard": integer(
                row["local_shard"]
            ),
            "inner_shard": integer(
                row["inner_shard"]
            ),
            "seed": integer(
                row["seed"]
            ),
            "events": expected_events,
            "immutable_split": (
                immutable_split
            ),
            "remote_bundle": (
                remote_bundle
            ),
            "adler32": (
                expected_adler
            ),
        })

    all_receipt_paths = sorted(
        receipt_dir.glob(
            "*_receipt.json"
        )
    )

    stale_paths = [
        path
        for path in all_receipt_paths
        if path.name
        not in manifest_receipt_names
    ]

    stale_records = []

    for path in stale_paths:
        try:
            stale = json.loads(
                path.read_text()
            )
        except Exception as exc:
            errors.append(
                f"unreadable stale receipt: "
                f"{path.name}: {exc}"
            )
            continue

        stale_record = {
            "file": path.name,
            "stage": stale.get(
                "stage"
            ),
            "exit_status": stale.get(
                "exit_status"
            ),
            "has_remote_bundle": bool(
                stale.get(
                    "remote_bundle"
                )
            ),
        }

        stale_records.append(
            stale_record
        )

    if len(stale_paths) != 1:
        errors.append(
            "expected exactly one excluded stale receipt, "
            f"found {len(stale_paths)}"
        )

    elif stale_records:
        stale = stale_records[0]

        if (
            stale["stage"]
            == "complete_copied_and_verified"
            or stale["exit_status"] == 0
            or stale["has_remote_bundle"]
        ):
            errors.append(
                "excluded stale receipt unexpectedly "
                "appears successful"
            )

    validated_events = sum(
        record["events"]
        for record in valid_records
    )

    if len(valid_records) != EXPECTED_ROWS:
        errors.append(
            f"expected {EXPECTED_ROWS} valid rows, "
            f"found {len(valid_records)}"
        )

    if validated_events != EXPECTED_EVENTS:
        errors.append(
            f"expected {EXPECTED_EVENTS} events, "
            f"found {validated_events}"
        )

    if dict(
        split_shards
    ) != EXPECTED_SPLIT_SHARDS:
        errors.append(
            "unexpected split-shard totals: "
            f"{dict(split_shards)}"
        )

    if dict(
        split_events
    ) != EXPECTED_SPLIT_EVENTS:
        errors.append(
            "unexpected split-event totals: "
            f"{dict(split_events)}"
        )

    summary = {
        "schema_version": 3,
        "status": (
            "pass"
            if not errors
            else "fail"
        ),
        "campaign": (
            next(iter(campaigns))
            if len(campaigns) == 1
            else ""
        ),
        "manifest_rows": len(rows),
        "valid_manifest_receipt_eos_rows": len(
            valid_records
        ),
        "validated_generated_events": (
            validated_events
        ),
        "split_shards": dict(
            sorted(
                split_shards.items()
            )
        ),
        "split_events": dict(
            sorted(
                split_events.items()
            )
        ),
        "training_events": (
            split_events["train"]
        ),
        "validation_events": (
            split_events["validation"]
        ),
        "sealed_test_events": (
            split_events["test"]
        ),
        "test_used_for_training": False,
        "unique_remote_bundles": len(
            seen_bundles
        ),
        "receipt_files_total": len(
            all_receipt_paths
        ),
        "stale_receipts_excluded": (
            stale_records
        ),
        "product_contents_validated": False,
        "candidate_rows_validated": False,
        "errors": errors,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        f"manifest_rows={len(rows)}"
    )

    print(
        "valid_manifest_receipt_eos_rows="
        f"{len(valid_records)}"
    )

    print(
        f"validated_generated_events={validated_events}"
    )

    print(
        "split_shards="
        f"{dict(sorted(split_shards.items()))}"
    )

    print(
        "split_events="
        f"{dict(sorted(split_events.items()))}"
    )

    print(
        f"training_events={split_events['train']}"
    )

    print(
        f"validation_events={split_events['validation']}"
    )

    print(
        f"sealed_test_events={split_events['test']}"
    )

    print(
        "test_used_for_training=False"
    )

    print(
        f"unique_remote_bundles={len(seen_bundles)}"
    )

    print(
        f"stale_receipts_excluded={len(stale_paths)}"
    )

    print(
        f"audit={output_path}"
    )

    if errors:
        for error in errors:
            print(
                f"ERROR: {error}"
            )

        raise SystemExit(2)

    print(
        "GGF_SIGNAL_20K_CROSS_LAYER_VALID"
    )


if __name__ == "__main__":
    main()
