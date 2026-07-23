#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq


EXPECTED_EVENTS = 5_000_000
EXPECTED_MEMBERS = 520

EXPECTED_SPLIT_EVENTS = {
    "train": 3_791_373,
    "validation": 1_024_084,
    "test": 184_543,
}


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def as_int(
    value: Any,
    *,
    default: int | None = None,
) -> int:
    if value in (
        None,
        "",
    ):
        if default is None:
            raise ValueError(
                "missing integer value"
            )

        return default

    if isinstance(value, bool):
        raise ValueError(
            "boolean is not an integer count"
        )

    return int(
        float(
            str(value).strip()
        )
    )


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def sha256_file(path: Path) -> str:
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


def schema_sha256(path: Path) -> str:
    parquet_file = pq.ParquetFile(path)

    serialized = str(
        parquet_file.schema_arrow
    ).encode("utf-8")

    return hashlib.sha256(
        serialized
    ).hexdigest()


def walk(
    value: Any,
    prefix: str = "$",
) -> Iterator[tuple[str, Any]]:
    yield prefix, value

    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(
                child,
                f"{prefix}.{key}",
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(
                child,
                f"{prefix}[{index}]",
            )


def find_candidate_paths(
    document: dict[str, Any],
) -> list[str]:
    values: set[str] = set()

    for key_path, value in walk(document):
        if not isinstance(value, str):
            continue

        lower_key = key_path.lower()
        lower_value = value.lower()

        if not lower_value.endswith(
            ".parquet"
        ):
            continue

        if "candidate" not in (
            lower_key
            + " "
            + lower_value
        ):
            continue

        values.add(value)

    return sorted(values)


def find_scalar_by_key(
    document: dict[str, Any],
    wanted_keys: set[str],
) -> list[Any]:
    values: list[Any] = []

    for key_path, value in walk(document):
        final_key = (
            key_path.rsplit(
                ".",
                maxsplit=1,
            )[-1]
            .split("[", maxsplit=1)[0]
        )

        if final_key in wanted_keys:
            values.append(value)

    return values


def read_tsv(
    path: Path,
) -> list[dict[str, str]]:
    require(
        path.is_file(),
        f"missing TSV {path}",
    )

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


def write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


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

    require(
        store.is_dir(),
        f"missing store {store}",
    )

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    audit_dir = (
        repo
        / "outputs/agent_runs"
        / "final_background_unified5m_registry_audit_20260723_v2"
    )

    summary_path = (
        audit_dir
        / "unified_background_5m_registry.json"
    )

    summary = json.loads(
        summary_path.read_text()
    )

    require(
        summary["status"] == "pass",
        "unified 5M audit did not pass",
    )

    require(
        summary[
            "unified_background_5m_registry_valid"
        ]
        is True,
        "unified 5M registry is not valid",
    )

    require(
        summary[
            "unified_background_5m_membership_frozen"
        ]
        is True,
        "unified membership is not frozen",
    )

    require(
        summary[
            "classifier_table_preparation_authorized"
        ]
        is True,
        (
            "classifier-table preparation "
            "is not authorized"
        ),
    )

    require(
        summary[
            "physics_yield_authorized"
        ]
        is False,
        (
            "unexpected background physics-yield "
            "authorization"
        ),
    )

    registry_path = Path(
        summary["registry_tsv"]
    )

    require(
        sha256_file(registry_path)
        == summary["registry_sha256"],
        "unified registry SHA mismatch",
    )

    registry = read_tsv(
        registry_path
    )

    require(
        len(registry)
        == EXPECTED_MEMBERS,
        (
            f"expected {EXPECTED_MEMBERS} members, "
            f"found {len(registry)}"
        ),
    )

    source_rows: list[
        dict[str, Any]
    ] = []

    split_events = Counter()
    component_events = Counter()
    component_candidate_rows = Counter()
    availability_counts = Counter()

    canonical_identities: set[str] = set()

    for index, row in enumerate(
        registry
    ):
        canonical_identity = row[
            "canonical_identity"
        ]

        require(
            canonical_identity
            not in canonical_identities,
            (
                "duplicate canonical identity "
                f"{canonical_identity}"
            ),
        )

        canonical_identities.add(
            canonical_identity
        )

        require(
            as_bool(
                row[
                    "canonical_membership"
                ]
            ),
            (
                "noncanonical member at "
                f"registry row {index}"
            ),
        )

        split = row[
            "dataset_split"
        ]

        require(
            split in {
                "train",
                "validation",
                "test",
            },
            f"invalid split {split!r}",
        )

        events = as_int(
            row["events"]
        )

        split_events[split] += events
        component_events[
            row["component"]
        ] += events

        receipt_path_value = (
            row.get(
                "receipt_path",
                "",
            ).strip()
        )

        receipt_path = (
            Path(receipt_path_value)
            if receipt_path_value
            else None
        )

        receipt: dict[str, Any] = {}

        if receipt_path is not None:
            require(
                receipt_path.is_file(),
                (
                    "missing receipt "
                    f"{receipt_path}"
                ),
            )

            receipt = json.loads(
                receipt_path.read_text()
            )

            require(
                isinstance(
                    receipt,
                    dict,
                ),
                (
                    f"{receipt_path}: "
                    "receipt is not an object"
                ),
            )

        registry_candidate_rows = as_int(
            row.get(
                "candidate_rows",
                "",
            ),
            default=-1,
        )

        receipt_candidate_values = (
            find_scalar_by_key(
                receipt,
                {
                    "candidate_rows",
                    "n_candidates",
                },
            )
            if receipt
            else []
        )

        receipt_candidate_rows = -1

        for value in receipt_candidate_values:
            try:
                receipt_candidate_rows = (
                    as_int(value)
                )
                break
            except Exception:
                continue

        local_candidates: list[Path] = []

        explicit_local = row.get(
            "local_candidate_parquet",
            "",
        ).strip()

        if explicit_local:
            path = Path(explicit_local)

            if path.is_file():
                local_candidates.append(
                    path
                )

        for value in find_candidate_paths(
            receipt
        ):
            path = Path(value)

            if path.is_file():
                local_candidates.append(
                    path
                )

        unique_local = {
            str(path.resolve()): path.resolve()
            for path in local_candidates
        }

        require(
            len(unique_local) <= 1,
            (
                "multiple existing local candidate "
                "Parquets for "
                f"{canonical_identity}: "
                f"{sorted(unique_local)}"
            ),
        )

        local_path = (
            next(
                iter(
                    unique_local.values()
                )
            )
            if unique_local
            else None
        )

        local_rows = -1
        local_schema_sha256 = ""
        local_candidate_sha256 = ""

        if local_path is not None:
            parquet_file = pq.ParquetFile(
                local_path
            )

            local_rows = (
                parquet_file.metadata.num_rows
            )

            local_schema_sha256 = (
                schema_sha256(
                    local_path
                )
            )

            local_candidate_sha256 = (
                sha256_file(
                    local_path
                )
            )

            availability = (
                "local_candidate_parquet"
            )

        else:
            remote_bundle = row.get(
                "remote_bundle",
                "",
            ).strip()

            require(
                remote_bundle.startswith(
                    "/store/"
                ),
                (
                    "member has neither a local "
                    "candidate Parquet nor a valid "
                    "EOS bundle: "
                    f"{canonical_identity}"
                ),
            )

            require(
                len(
                    row.get(
                        "bundle_sha256",
                        "",
                    ).strip()
                )
                == 64,
                (
                    "remote member has no valid "
                    "bundle SHA: "
                    f"{canonical_identity}"
                ),
            )

            availability = (
                "remote_bundle_extraction_required"
            )

        availability_counts[
            availability
        ] += 1

        candidate_rows = max(
            registry_candidate_rows,
            receipt_candidate_rows,
            local_rows,
        )

        require(
            candidate_rows >= 0,
            (
                "unable to resolve candidate row "
                f"count for {canonical_identity}"
            ),
        )

        if (
            registry_candidate_rows >= 0
            and local_rows >= 0
        ):
            require(
                registry_candidate_rows
                == local_rows,
                (
                    "registry/local candidate-row "
                    "mismatch for "
                    f"{canonical_identity}: "
                    f"{registry_candidate_rows} "
                    f"versus {local_rows}"
                ),
            )

        if (
            receipt_candidate_rows >= 0
            and local_rows >= 0
        ):
            require(
                receipt_candidate_rows
                == local_rows,
                (
                    "receipt/local candidate-row "
                    "mismatch for "
                    f"{canonical_identity}: "
                    f"{receipt_candidate_rows} "
                    f"versus {local_rows}"
                ),
            )

        component_candidate_rows[
            row["component"]
        ] += candidate_rows

        receipt_candidate_references = (
            find_candidate_paths(
                receipt
            )
            if receipt
            else []
        )

        expected_candidate_hashes = [
            str(value)
            for value in find_scalar_by_key(
                receipt,
                {
                    "candidate_sha256",
                    "candidate_parquet_sha256",
                },
            )
            if str(value)
        ]

        if (
            local_candidate_sha256
            and expected_candidate_hashes
        ):
            require(
                local_candidate_sha256
                in expected_candidate_hashes,
                (
                    "local candidate SHA does not "
                    "match receipt for "
                    f"{canonical_identity}"
                ),
            )

        source_rows.append({
            "registry_index":
                index,
            "component":
                row["component"],
            "family":
                row["family"],
            "dataset_split":
                split,
            "dataset_role":
                row["dataset_role"],
            "generated_events":
                events,
            "candidate_rows":
                candidate_rows,
            "availability":
                availability,
            "canonical_identity":
                canonical_identity,
            "source_campaign":
                row["source_campaign"],
            "target_tag":
                row["target_tag"],
            "shard_id":
                row["shard_id"],
            "seed":
                row["seed"],
            "receipt_path":
                receipt_path_value,
            "remote_bundle":
                row.get(
                    "remote_bundle",
                    "",
                ),
            "bundle_sha256":
                row.get(
                    "bundle_sha256",
                    "",
                ),
            "bundle_adler32":
                row.get(
                    "bundle_adler32",
                    "",
                ),
            "local_candidate_parquet":
                (
                    str(local_path)
                    if local_path
                    else ""
                ),
            "local_candidate_rows":
                (
                    local_rows
                    if local_rows >= 0
                    else ""
                ),
            "local_candidate_sha256":
                local_candidate_sha256,
            "local_schema_sha256":
                local_schema_sha256,
            "receipt_candidate_references":
                " | ".join(
                    receipt_candidate_references
                ),
            "expected_candidate_sha256":
                " | ".join(
                    sorted(
                        set(
                            expected_candidate_hashes
                        )
                    )
                ),
            "physics_weight_status":
                "not_frozen",
        })

    require(
        sum(
            split_events.values()
        )
        == EXPECTED_EVENTS,
        "classifier source event total is not 5M",
    )

    require(
        dict(split_events)
        == EXPECTED_SPLIT_EVENTS,
        (
            "classifier source split totals "
            f"mismatch: {dict(split_events)}"
        ),
    )

    require(
        len(canonical_identities)
        == EXPECTED_MEMBERS,
        "canonical identity count is not 520",
    )

    outdir.mkdir(
        parents=True
    )

    manifest_path = (
        outdir
        / "background_5m_classifier_source_manifest.tsv"
    )

    columns = [
        "registry_index",
        "component",
        "family",
        "dataset_split",
        "dataset_role",
        "generated_events",
        "candidate_rows",
        "availability",
        "canonical_identity",
        "source_campaign",
        "target_tag",
        "shard_id",
        "seed",
        "receipt_path",
        "remote_bundle",
        "bundle_sha256",
        "bundle_adler32",
        "local_candidate_parquet",
        "local_candidate_rows",
        "local_candidate_sha256",
        "local_schema_sha256",
        "receipt_candidate_references",
        "expected_candidate_sha256",
        "physics_weight_status",
    ]

    write_tsv(
        manifest_path,
        source_rows,
        columns,
    )

    component_rows = []

    for component in sorted(
        component_events
    ):
        component_rows.append({
            "component":
                component,
            "generated_events":
                component_events[
                    component
                ],
            "candidate_rows":
                component_candidate_rows[
                    component
                ],
            "members":
                sum(
                    1
                    for row in source_rows
                    if row["component"]
                    == component
                ),
            "local_members":
                sum(
                    1
                    for row in source_rows
                    if (
                        row["component"]
                        == component
                        and row[
                            "availability"
                        ]
                        == "local_candidate_parquet"
                    )
                ),
            "remote_members":
                sum(
                    1
                    for row in source_rows
                    if (
                        row["component"]
                        == component
                        and row[
                            "availability"
                        ]
                        == (
                            "remote_bundle_"
                            "extraction_required"
                        )
                    )
                ),
        })

    component_path = (
        outdir
        / "background_5m_classifier_source_summary.tsv"
    )

    write_tsv(
        component_path,
        component_rows,
        [
            "component",
            "generated_events",
            "candidate_rows",
            "members",
            "local_members",
            "remote_members",
        ],
    )

    summary_output = {
        "schema_version": 1,
        "status": "pass",
        "background_5m_classifier_source_manifest_valid":
            True,
        "generated_events":
            EXPECTED_EVENTS,
        "members":
            EXPECTED_MEMBERS,
        "events_by_split":
            dict(split_events),
        "members_by_availability":
            dict(
                availability_counts
            ),
        "candidate_rows_by_component":
            dict(
                component_candidate_rows
            ),
        "source_manifest":
            str(manifest_path),
        "source_manifest_sha256":
            sha256_file(
                manifest_path
            ),
        "component_summary":
            str(component_path),
        "component_summary_sha256":
            sha256_file(
                component_path
            ),
        "materialization_authorized":
            True,
        "physics_weights_frozen":
            False,
        "physics_yield_authorized":
            False,
    }

    summary_output_path = (
        outdir
        / "background_5m_classifier_source_audit.json"
    )

    summary_output_path.write_text(
        json.dumps(
            summary_output,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "BACKGROUND_5M_CLASSIFIER_SOURCE_MANIFEST_VALID"
    )
    print(
        "BACKGROUND_5M_CLASSIFIER_MEMBERS=520"
    )
    print(
        "BACKGROUND_5M_CLASSIFIER_GENERATED_EVENTS=5000000"
    )
    print(
        "BACKGROUND_5M_CLASSIFIER_LOCAL_CANDIDATE_SOURCES="
        f"{availability_counts['local_candidate_parquet']}"
    )
    print(
        "BACKGROUND_5M_CLASSIFIER_REMOTE_BUNDLE_SOURCES="
        f"{availability_counts['remote_bundle_extraction_required']}"
    )
    print(
        "BACKGROUND_5M_MATERIALIZATION_AUTHORIZED"
    )
    print(
        "PHYSICS_WEIGHTS_NOT_YET_FROZEN"
    )
    print(
        f"source_manifest={manifest_path}"
    )
    print(
        f"summary_json={summary_output_path}"
    )


if __name__ == "__main__":
    main()
