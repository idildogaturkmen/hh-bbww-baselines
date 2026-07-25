#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot


OUTPUT_TABLES = (
    "real_parity_canary_selection.tsv",
    "real_parity_file_summary.tsv",
    "real_parity_schema_comparison.tsv",
    "real_parity_column_comparison.tsv",
    "real_parity_mismatch_details.tsv",
    "real_parity_edge_coverage.tsv",
    "extraction_access_audit.tsv",
    "summary.json",
)

SELECTION_FIELDS = [
    "canary_index",
    "member_index",
    "sample_class",
    "process_family",
    "process_or_mode",
    "dataset_split",
    "candidate_rows_in_existing_registry",
    "canary_selection_reasons",
    "selection_policy",
    "resolution_method",
    "layout_rule",
    "remote_bundle_path",
    "root_container_member",
    "root_archive_member",
]

FILE_SUMMARY_FIELDS = [
    "member_index",
    "sample_class",
    "process_or_mode",
    "dataset_split",
    "candidate_rows_in_existing_registry",
    "canary_selection_reasons",
    "resolution_method",
    "layout_rule",
    "remote_bundle_path",
    "root_container_member",
    "root_archive_member",
    "root_sha256",
    "delphes_num_entries",
    "frozen_output_rows",
    "parity_output_rows",
    "ordered_schema_match",
    "event_order_match",
    "integer_columns_match",
    "string_columns_match",
    "floating_columns_match",
    "parquet_types_compatible",
    "mismatch_count",
    "parity_status",
]

SCHEMA_COMPARISON_FIELDS = [
    "member_index",
    "column_name",
    "column_order",
    "frozen_arrow_type",
    "parity_arrow_type",
    "frozen_nullable",
    "parity_nullable",
    "arrow_type_compatible",
    "nullability_match",
    "comparison_status",
]

COLUMN_COMPARISON_FIELDS = [
    "member_index",
    "column_name",
    "column_order",
    "logical_type_class",
    "frozen_arrow_type",
    "parity_arrow_type",
    "row_count",
    "exact_mismatch_count",
    "finite_comparison_count",
    "max_absolute_difference",
    "nonfinite_mask_match",
    "comparison_status",
]

MISMATCH_DETAIL_FIELDS = [
    "member_index",
    "scope",
    "column_name",
    "row_index",
    "frozen_value",
    "parity_value",
    "absolute_difference",
    "mismatch_type",
    "detail",
]

EDGE_COVERAGE_FIELDS = [
    "member_index",
    "delphes_num_entries",
    "frozen_selected_rows",
    "maximum_n_selected_jets",
    "maximum_n_selected_bjets",
    "selected_event_with_more_than_four_tagged_jets",
    "selected_event_with_more_than_eight_tagged_jets",
    "maximum_observed_pairing_bjet_rank",
    "zero_4b_candidates_selected",
]

EXTRACTION_AUDIT_FIELDS = [
    "member_index",
    "remote_bundle_path",
    "remote_bundle_uri",
    "expected_bundle_size_bytes",
    "xrdcp_attempted",
    "xrdcp_return_code",
    "xrdcp_status",
    "xrdcp_failure_text",
    "local_bundle_size_bytes",
    "bundle_size_match",
    "outer_archive_members_inspected",
    "outer_requested_member",
    "outer_exact_matches",
    "nested_archive_used",
    "nested_archive_members_inspected",
    "nested_requested_member",
    "nested_exact_matches",
    "root_file_extracted",
    "root_metadata_opened",
    "frozen_builder_invoked",
    "frozen_builder_return_code",
    "parity_builder_invoked",
    "parity_builder_return_code",
    "root_file_deleted",
    "nested_archive_deleted",
    "outer_bundle_deleted",
    "frozen_output_deleted",
    "parity_output_deleted",
    "temporary_files_remaining",
    "failure_reason",
    "extraction_status",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve(repo: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8", errors="strict") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    require(bool(fields), f"{path}: missing TSV header")
    return fields, rows


def write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_json_object(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    require(isinstance(document, dict), f"{path}: expected a JSON object")
    return document


def verify_checkpoint(checkpoint_dir: Path) -> None:
    result = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=checkpoint_dir,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(
        result.returncode == 0,
        (
            f"checkpoint checksum verification failed for {checkpoint_dir}: "
            f"{result.stderr.strip()}"
        ),
    )


def canonical_archive_name(value: str) -> str:
    raw = value.strip()
    require(bool(raw), "empty archive member path")
    path = PurePosixPath(raw)
    require(not path.is_absolute(), f"absolute archive path rejected: {raw}")
    require(".." not in path.parts, f"parent traversal rejected: {raw}")
    parts = [part for part in path.parts if part not in ("", ".")]
    if not parts:
        return "."
    return "/".join(parts)


def extract_exact_member(
    *,
    archive_path: Path,
    requested_member: str,
    destination_path: Path,
) -> tuple[int, int]:
    requested = canonical_archive_name(requested_member)
    matches: list[tarfile.TarInfo] = []
    member_count = 0
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive:
            member_count += 1
            canonical = canonical_archive_name(member.name)
            if canonical == requested:
                matches.append(member)
        require(
            len(matches) == 1,
            (
                f"{archive_path.name}: requested member {requested!r} "
                f"has {len(matches)} exact matches"
            ),
        )
        match = matches[0]
        require(match.isfile(), f"{match.name}: requested member is not a file")
        source = archive.extractfile(match)
        require(source is not None, f"{match.name}: cannot open archive member")
        with source, destination_path.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    return member_count, len(matches)


def condition_matches(row: dict[str, str], condition: dict[str, Any]) -> bool:
    field = str(condition["field"])
    require(field in row, f"selection condition references missing field {field}")
    operator = str(condition["operator"])
    actual = row[field]
    expected = condition["value"]
    if operator == "eq":
        return actual == str(expected)
    if operator == "contains":
        return str(expected) in actual
    if operator == "prefix":
        return actual.startswith(str(expected))
    if operator == "in":
        return actual in {str(value) for value in expected}
    if operator == "int_eq":
        return int(actual) == int(expected)
    if operator == "int_gt":
        return int(actual) > int(expected)
    raise RuntimeError(f"unsupported selection operator {operator!r}")


def select_canaries(
    source_rows: list[dict[str, str]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    allowed_split = str(config["allowed_dataset_split"])
    train_rows = [
        row for row in source_rows if row["dataset_split"] == allowed_split
    ]
    require(bool(train_rows), "no rows in the allowed train split")
    selected_reasons: dict[str, set[str]] = {}
    for rule in config["selection_rules"]:
        rule_id = str(rule["rule_id"])
        conditions = list(rule["all"])
        matches = [
            row
            for row in train_rows
            if all(condition_matches(row, condition) for condition in conditions)
        ]
        require(matches, f"selection rule {rule_id} has no train match")
        chosen = min(matches, key=lambda row: int(row["member_index"]))
        selected_reasons.setdefault(chosen["member_index"], set()).add(rule_id)

    source_by_member = {
        row["member_index"]: row for row in train_rows
    }
    selected: list[dict[str, Any]] = []
    for canary_index, member_index in enumerate(
        sorted(selected_reasons, key=int),
        start=1,
    ):
        source = dict(source_by_member[member_index])
        source.update(
            {
                "canary_index": canary_index,
                "candidate_rows_in_existing_registry": source[
                    "candidate_rows"
                ],
                "canary_selection_reasons": json.dumps(
                    sorted(selected_reasons[member_index]),
                    separators=(",", ":"),
                ),
                "selection_policy": config["selection_policy"],
            }
        )
        selected.append(source)
    require(
        len(selected) >= int(config["minimum_unique_canaries"]),
        "deduplicated canary set is smaller than required",
    )
    require(
        all(row["dataset_split"] == allowed_split for row in selected),
        "non-train canary selected",
    )
    return selected


def run_builder(
    *,
    repo: Path,
    builder_path: Path,
    root_path: Path,
    output_path: Path,
    sample: str,
    parameters: dict[str, Any],
    timeout_seconds: int,
    mode: str | None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(builder_path)]
    if mode is not None:
        command.extend(["--mode", mode])
    command.extend(
        [
            "--input",
            str(root_path),
            "--out",
            str(output_path),
            "--sample",
            sample,
            "--target-mass",
            str(parameters["target_mass"]),
            "--jet-pt-min",
            str(parameters["jet_pt_min"]),
            "--jet-eta-max",
            str(parameters["jet_eta_max"]),
            "--btag-min",
            str(parameters["btag_min"]),
            "--max-bjets-for-pairing",
            str(parameters["max_bjets_for_pairing"]),
            "--higgs-ordering",
            str(parameters["higgs_ordering"]),
        ]
    )
    return subprocess.run(
        command,
        cwd=repo,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
    )


def arrow_logical_class(data_type: pa.DataType) -> str:
    if pa.types.is_floating(data_type):
        return "floating"
    if pa.types.is_integer(data_type) or pa.types.is_boolean(data_type):
        return "integer"
    if (
        pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
        or pa.types.is_binary(data_type)
        or pa.types.is_large_binary(data_type)
    ):
        return "string"
    if pa.types.is_null(data_type):
        return "null"
    return "other"


def scalar_values_equal(left: Any, right: Any) -> bool:
    left_missing = bool(pd.isna(left))
    right_missing = bool(pd.isna(right))
    if left_missing or right_missing:
        return left_missing and right_missing
    return bool(left == right)


def compare_parquets(
    *,
    member_index: str,
    frozen_path: Path,
    parity_path: Path,
    expected_columns: int,
    threeb_only_columns: set[str],
    rtol: float,
    atol: float,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    pd.DataFrame,
]:
    frozen_schema = pq.read_schema(frozen_path)
    parity_schema = pq.read_schema(parity_path)
    frozen_frame = pd.read_parquet(frozen_path)
    parity_frame = pd.read_parquet(parity_path)
    frozen_names = list(frozen_frame.columns)
    parity_names = list(parity_frame.columns)
    ordered_schema_match = (
        len(frozen_names) == expected_columns
        and len(parity_names) == expected_columns
        and frozen_names == parity_names
        and not (set(parity_names) & threeb_only_columns)
    )
    row_count_match = len(frozen_frame) == len(parity_frame)

    schema_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    schema_mismatches = 0
    integer_mismatches = 0
    string_mismatches = 0
    floating_mismatches = 0

    comparison_names = frozen_names
    for order, column in enumerate(comparison_names, start=1):
        parity_has_column = column in parity_names
        frozen_field = frozen_schema.field(column)
        parity_field = (
            parity_schema.field(column) if parity_has_column else None
        )
        type_compatible = bool(
            parity_field is not None
            and frozen_field.type == parity_field.type
        )
        nullability_match = bool(
            parity_field is not None
            and frozen_field.nullable == parity_field.nullable
        )
        schema_status = (
            "pass"
            if parity_has_column
            and parity_names[order - 1] == column
            and type_compatible
            and nullability_match
            else "fail"
        )
        if schema_status != "pass":
            schema_mismatches += 1
            mismatch_rows.append(
                {
                    "member_index": member_index,
                    "scope": "schema",
                    "column_name": column,
                    "row_index": "",
                    "frozen_value": str(frozen_field),
                    "parity_value": (
                        str(parity_field) if parity_field is not None else ""
                    ),
                    "absolute_difference": "",
                    "mismatch_type": "schema_mismatch",
                    "detail": "column order, Arrow type, or nullability differs",
                }
            )
        schema_rows.append(
            {
                "member_index": member_index,
                "column_name": column,
                "column_order": order,
                "frozen_arrow_type": str(frozen_field.type),
                "parity_arrow_type": (
                    str(parity_field.type)
                    if parity_field is not None
                    else ""
                ),
                "frozen_nullable": frozen_field.nullable,
                "parity_nullable": (
                    parity_field.nullable
                    if parity_field is not None
                    else ""
                ),
                "arrow_type_compatible": type_compatible,
                "nullability_match": nullability_match,
                "comparison_status": schema_status,
            }
        )

        logical_class = arrow_logical_class(frozen_field.type)
        exact_mismatch_count = 0
        finite_comparison_count = 0
        max_absolute_difference: float | str = ""
        nonfinite_mask_match: bool | str = ""
        value_status = "fail"
        if parity_has_column and row_count_match:
            frozen_series = frozen_frame[column]
            parity_series = parity_frame[column]
            if logical_class == "floating":
                frozen_values = frozen_series.to_numpy(dtype=np.float64)
                parity_values = parity_series.to_numpy(dtype=np.float64)
                exact_equal = (frozen_values == parity_values) | (
                    np.isnan(frozen_values) & np.isnan(parity_values)
                )
                exact_mismatch_count = int(np.count_nonzero(~exact_equal))
                frozen_finite = np.isfinite(frozen_values)
                parity_finite = np.isfinite(parity_values)
                finite_mask = frozen_finite & parity_finite
                finite_comparison_count = int(np.count_nonzero(finite_mask))
                nonfinite_mask_match = bool(
                    np.array_equal(
                        np.isnan(frozen_values),
                        np.isnan(parity_values),
                    )
                    and np.array_equal(
                        np.isposinf(frozen_values),
                        np.isposinf(parity_values),
                    )
                    and np.array_equal(
                        np.isneginf(frozen_values),
                        np.isneginf(parity_values),
                    )
                )
                if finite_comparison_count:
                    differences = np.abs(
                        frozen_values[finite_mask]
                        - parity_values[finite_mask]
                    )
                    max_absolute_difference = float(np.max(differences))
                else:
                    max_absolute_difference = 0.0
                close = np.isclose(
                    frozen_values,
                    parity_values,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                )
                floating_mismatches += int(np.count_nonzero(~close))
                value_status = (
                    "pass"
                    if bool(np.all(close)) and nonfinite_mask_match
                    else "fail"
                )
            else:
                unequal_indices = [
                    index
                    for index, (left, right) in enumerate(
                        zip(frozen_series.tolist(), parity_series.tolist())
                    )
                    if not scalar_values_equal(left, right)
                ]
                exact_mismatch_count = len(unequal_indices)
                value_status = (
                    "pass" if exact_mismatch_count == 0 else "fail"
                )
                if logical_class == "integer":
                    integer_mismatches += exact_mismatch_count
                elif logical_class == "string":
                    string_mismatches += exact_mismatch_count
                elif logical_class == "null" and len(frozen_series) == 0:
                    value_status = "pass"
                elif exact_mismatch_count:
                    string_mismatches += exact_mismatch_count
        if value_status != "pass":
            mismatch_rows.append(
                {
                    "member_index": member_index,
                    "scope": "column_values",
                    "column_name": column,
                    "row_index": "",
                    "frozen_value": "",
                    "parity_value": "",
                    "absolute_difference": max_absolute_difference,
                    "mismatch_type": "value_mismatch",
                    "detail": (
                        f"{exact_mismatch_count} exact mismatches in "
                        f"logical class {logical_class}"
                    ),
                }
            )
        column_rows.append(
            {
                "member_index": member_index,
                "column_name": column,
                "column_order": order,
                "logical_type_class": logical_class,
                "frozen_arrow_type": str(frozen_field.type),
                "parity_arrow_type": (
                    str(parity_field.type)
                    if parity_field is not None
                    else ""
                ),
                "row_count": len(frozen_frame),
                "exact_mismatch_count": exact_mismatch_count,
                "finite_comparison_count": finite_comparison_count,
                "max_absolute_difference": max_absolute_difference,
                "nonfinite_mask_match": nonfinite_mask_match,
                "comparison_status": value_status,
            }
        )

    if len(parity_names) != len(frozen_names):
        schema_mismatches += abs(len(parity_names) - len(frozen_names))
    event_order_match = bool(
        row_count_match
        and "event" in frozen_frame
        and "event" in parity_frame
        and frozen_frame["event"].equals(parity_frame["event"])
    )
    if not event_order_match:
        mismatch_rows.append(
            {
                "member_index": member_index,
                "scope": "event_order",
                "column_name": "event",
                "row_index": "",
                "frozen_value": "",
                "parity_value": "",
                "absolute_difference": "",
                "mismatch_type": "event_order_mismatch",
                "detail": "event identifiers or row order differ",
            }
        )
    if not row_count_match:
        mismatch_rows.append(
            {
                "member_index": member_index,
                "scope": "row_count",
                "column_name": "",
                "row_index": "",
                "frozen_value": len(frozen_frame),
                "parity_value": len(parity_frame),
                "absolute_difference": "",
                "mismatch_type": "row_count_mismatch",
                "detail": "candidate row counts differ",
            }
        )

    types_compatible = bool(
        len(schema_rows) == expected_columns
        and all(
            row["arrow_type_compatible"]
            and row["nullability_match"]
            for row in schema_rows
        )
    )
    comparison = {
        "frozen_rows": len(frozen_frame),
        "parity_rows": len(parity_frame),
        "ordered_schema_match": ordered_schema_match,
        "row_count_match": row_count_match,
        "event_order_match": event_order_match,
        "integer_columns_match": integer_mismatches == 0,
        "string_columns_match": string_mismatches == 0,
        "floating_columns_match": floating_mismatches == 0,
        "parquet_types_compatible": types_compatible,
        "schema_mismatches": schema_mismatches,
        "integer_value_mismatches": integer_mismatches,
        "string_value_mismatches": string_mismatches,
        "floating_value_mismatches": floating_mismatches,
        "row_count_mismatch": int(not row_count_match),
        "event_order_mismatch": int(not event_order_match),
        "columns_compared": len(column_rows),
        "mismatch_count": (
            schema_mismatches
            + int(not row_count_match)
            + int(not event_order_match)
            + integer_mismatches
            + string_mismatches
            + floating_mismatches
        ),
    }
    comparison["status"] = (
        "pass"
        if comparison["mismatch_count"] == 0
        and ordered_schema_match
        and types_compatible
        else "fail"
    )
    return (
        comparison,
        schema_rows,
        column_rows,
        mismatch_rows,
        frozen_frame,
    )


def edge_coverage(
    *,
    member_index: str,
    delphes_num_entries: int,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "member_index": member_index,
            "delphes_num_entries": delphes_num_entries,
            "frozen_selected_rows": 0,
            "maximum_n_selected_jets": "",
            "maximum_n_selected_bjets": "",
            "zero_4b_candidates_selected": True,
            "selected_event_with_more_than_four_tagged_jets": False,
            "selected_event_with_more_than_eight_tagged_jets": False,
            "maximum_observed_pairing_bjet_rank": "",
        }
    rank_columns = [
        f"j{index}_bjet_rank" for index in range(1, 5)
    ]
    return {
        "member_index": member_index,
        "delphes_num_entries": delphes_num_entries,
        "frozen_selected_rows": len(frame),
        "maximum_n_selected_jets": int(frame["n_selected_jets"].max()),
        "maximum_n_selected_bjets": int(frame["n_selected_bjets"].max()),
        "zero_4b_candidates_selected": False,
        "selected_event_with_more_than_four_tagged_jets": bool(
            (frame["n_selected_bjets"] > 4).any()
        ),
        "selected_event_with_more_than_eight_tagged_jets": bool(
            (frame["n_selected_bjets"] > 8).any()
        ),
        "maximum_observed_pairing_bjet_rank": int(
            frame[rank_columns].to_numpy().max()
        ),
    }


def empty_file_summary(canary: dict[str, Any]) -> dict[str, Any]:
    return {
        "member_index": canary["member_index"],
        "sample_class": canary["sample_class"],
        "process_or_mode": canary["process_or_mode"],
        "dataset_split": canary["dataset_split"],
        "candidate_rows_in_existing_registry": canary[
            "candidate_rows_in_existing_registry"
        ],
        "canary_selection_reasons": canary["canary_selection_reasons"],
        "resolution_method": canary["resolution_method"],
        "layout_rule": canary["layout_rule"],
        "remote_bundle_path": canary["remote_bundle_path"],
        "root_container_member": canary["root_container_member"],
        "root_archive_member": canary["root_archive_member"],
        "root_sha256": "",
        "delphes_num_entries": "",
        "frozen_output_rows": "",
        "parity_output_rows": "",
        "ordered_schema_match": False,
        "event_order_match": False,
        "integer_columns_match": False,
        "string_columns_match": False,
        "floating_columns_match": False,
        "parquet_types_compatible": False,
        "mismatch_count": 1,
        "parity_status": "fail",
    }


def process_canary(
    *,
    repo: Path,
    canary: dict[str, Any],
    canary_number: int,
    canary_total: int,
    config: dict[str, Any],
    frozen_builder: Path,
    parity_builder: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any] | None,
    dict[str, Any],
]:
    member_index = canary["member_index"]
    print(
        (
            f"[{canary_number}/{canary_total}] member={member_index} "
            f"process={canary['process_or_mode']} "
            f"size={canary['bundle_size_bytes']}"
        ),
        flush=True,
    )
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"hh4b_4b_real_parity_{member_index}_",
            dir=config["temporary_root"],
        )
    )
    outer_path = temp_dir / "outer_bundle.tar.gz"
    nested_path = temp_dir / "nested_reconstruction.tar.gz"
    root_path = temp_dir / canary["root_basename"]
    frozen_output = temp_dir / "frozen.parquet"
    parity_output = temp_dir / "parity.parquet"
    summary_row = empty_file_summary(canary)
    schema_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    edge_row: dict[str, Any] | None = None
    access: dict[str, Any] = {
        "member_index": member_index,
        "remote_bundle_path": canary["remote_bundle_path"],
        "remote_bundle_uri": canary["remote_bundle_uri"],
        "expected_bundle_size_bytes": canary["bundle_size_bytes"],
        "xrdcp_attempted": True,
        "xrdcp_return_code": "",
        "xrdcp_status": "not_completed",
        "xrdcp_failure_text": "",
        "local_bundle_size_bytes": "",
        "bundle_size_match": False,
        "outer_archive_members_inspected": 0,
        "outer_requested_member": (
            canary["root_container_member"]
            or canary["root_archive_member"]
        ),
        "outer_exact_matches": 0,
        "nested_archive_used": bool(canary["root_container_member"]),
        "nested_archive_members_inspected": 0,
        "nested_requested_member": (
            canary["root_archive_member"]
            if canary["root_container_member"]
            else ""
        ),
        "nested_exact_matches": 0,
        "root_file_extracted": False,
        "root_metadata_opened": False,
        "frozen_builder_invoked": False,
        "frozen_builder_return_code": "",
        "parity_builder_invoked": False,
        "parity_builder_return_code": "",
        "root_file_deleted": False,
        "nested_archive_deleted": False,
        "outer_bundle_deleted": False,
        "frozen_output_deleted": False,
        "parity_output_deleted": False,
        "temporary_files_remaining": "",
        "extraction_status": "fail",
        "failure_reason": "",
    }
    try:
        download = subprocess.run(
            [
                "xrdcp",
                "--nopbar",
                "--force",
                canary["remote_bundle_uri"],
                str(outer_path),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(config["xrdcp_timeout_seconds"]),
        )
        access["xrdcp_return_code"] = download.returncode
        if download.returncode != 0:
            failure_text = (download.stderr or download.stdout).strip()
            access["xrdcp_failure_text"] = re.sub(
                r"\s+",
                " ",
                failure_text,
            )
            raise RuntimeError(
                f"xrdcp failed with return code {download.returncode}"
            )
        access["xrdcp_status"] = "pass"
        local_size = outer_path.stat().st_size
        access["local_bundle_size_bytes"] = local_size
        access["bundle_size_match"] = (
            local_size == int(canary["bundle_size_bytes"])
        )
        require(access["bundle_size_match"], "downloaded bundle size mismatch")

        if canary["root_container_member"]:
            count, matches = extract_exact_member(
                archive_path=outer_path,
                requested_member=canary["root_container_member"],
                destination_path=nested_path,
            )
            access["outer_archive_members_inspected"] = count
            access["outer_exact_matches"] = matches
            count, matches = extract_exact_member(
                archive_path=nested_path,
                requested_member=canary["root_archive_member"],
                destination_path=root_path,
            )
            access["nested_archive_members_inspected"] = count
            access["nested_exact_matches"] = matches
        else:
            count, matches = extract_exact_member(
                archive_path=outer_path,
                requested_member=canary["root_archive_member"],
                destination_path=root_path,
            )
            access["outer_archive_members_inspected"] = count
            access["outer_exact_matches"] = matches
        require(root_path.is_file(), "resolved ROOT file was not extracted")
        access["root_file_extracted"] = True
        access["extraction_status"] = "pass"
        summary_row["root_sha256"] = sha256_file(root_path)

        with uproot.open(root_path) as root_file:
            tree = root_file["Delphes"]
            delphes_num_entries = int(tree.num_entries)
        access["root_metadata_opened"] = True
        summary_row["delphes_num_entries"] = delphes_num_entries
        require(
            delphes_num_entries == int(canary["generated_events"]),
            "Delphes entry count differs from the source map",
        )

        access["frozen_builder_invoked"] = True
        frozen_run = run_builder(
            repo=repo,
            builder_path=frozen_builder,
            root_path=root_path,
            output_path=frozen_output,
            sample=canary["process_or_mode"],
            parameters=config["reconstruction_parameters"],
            timeout_seconds=int(config["builder_timeout_seconds"]),
            mode=None,
        )
        access["frozen_builder_return_code"] = frozen_run.returncode
        require(
            frozen_run.returncode == 0,
            (
                "frozen builder failed: "
                + re.sub(
                    r"\s+",
                    " ",
                    (frozen_run.stderr or frozen_run.stdout).strip(),
                )
            ),
        )
        require(frozen_output.is_file(), "frozen builder output is missing")

        access["parity_builder_invoked"] = True
        parity_run = run_builder(
            repo=repo,
            builder_path=parity_builder,
            root_path=root_path,
            output_path=parity_output,
            sample=canary["process_or_mode"],
            parameters=config["reconstruction_parameters"],
            timeout_seconds=int(config["builder_timeout_seconds"]),
            mode=config["parity_mode"],
        )
        access["parity_builder_return_code"] = parity_run.returncode
        require(
            parity_run.returncode == 0,
            (
                "parity builder failed: "
                + re.sub(
                    r"\s+",
                    " ",
                    (parity_run.stderr or parity_run.stdout).strip(),
                )
            ),
        )
        require(parity_output.is_file(), "parity builder output is missing")

        (
            comparison,
            schema_rows,
            column_rows,
            mismatch_rows,
            frozen_frame,
        ) = compare_parquets(
            member_index=member_index,
            frozen_path=frozen_output,
            parity_path=parity_output,
            expected_columns=int(
                config["comparison"]["expected_ordered_columns"]
            ),
            threeb_only_columns=set(config["threeb_only_columns"]),
            rtol=float(config["comparison"]["floating_rtol"]),
            atol=float(config["comparison"]["floating_atol"]),
        )
        summary_row.update(
            {
                "frozen_output_rows": comparison["frozen_rows"],
                "parity_output_rows": comparison["parity_rows"],
                "ordered_schema_match": comparison[
                    "ordered_schema_match"
                ],
                "event_order_match": comparison["event_order_match"],
                "integer_columns_match": comparison[
                    "integer_columns_match"
                ],
                "string_columns_match": comparison[
                    "string_columns_match"
                ],
                "floating_columns_match": comparison[
                    "floating_columns_match"
                ],
                "parquet_types_compatible": comparison[
                    "parquet_types_compatible"
                ],
                "mismatch_count": comparison["mismatch_count"],
                "parity_status": comparison["status"],
                "_comparison": comparison,
            }
        )
        edge_row = edge_coverage(
            member_index=member_index,
            delphes_num_entries=delphes_num_entries,
            frame=frozen_frame,
        )
    except subprocess.TimeoutExpired as error:
        access["failure_reason"] = (
            f"subprocess timed out after {error.timeout} seconds"
        )
    except Exception as error:
        access["failure_reason"] = str(error)
    finally:
        if access["failure_reason"]:
            mismatch_rows.append(
                {
                    "member_index": member_index,
                    "scope": "canary",
                    "column_name": "",
                    "row_index": "",
                    "frozen_value": "",
                    "parity_value": "",
                    "absolute_difference": "",
                    "mismatch_type": "canary_processing_failure",
                    "detail": access["failure_reason"],
                }
            )
        shutil.rmtree(temp_dir)
        require(not temp_dir.exists(), f"temporary directory remains: {temp_dir}")
        access["root_file_deleted"] = not root_path.exists()
        access["nested_archive_deleted"] = not nested_path.exists()
        access["outer_bundle_deleted"] = not outer_path.exists()
        access["frozen_output_deleted"] = not frozen_output.exists()
        access["parity_output_deleted"] = not parity_output.exists()
        access["temporary_files_remaining"] = 0

    print(
        (
            f"[{canary_number}/{canary_total}] member={member_index} "
            f"status={summary_row['parity_status']} "
            f"rows={summary_row['frozen_output_rows']} "
            f"failure={access['failure_reason']!r}"
        ),
        flush=True,
    )
    return (
        summary_row,
        schema_rows,
        column_rows,
        mismatch_rows,
        edge_row,
        access,
    )


def checkpoint_readme(
    summary: dict[str, Any],
    selection_rows: list[dict[str, Any]],
) -> str:
    selected_members = ", ".join(
        row["member_index"] for row in selection_rows
    )
    return f"""# HH4b real-Delphes four-b parity canary

## Purpose

This checkpoint compares the immutable frozen HH4b four-b candidate builder
with the new builder's explicit `fourb-parity` mode on a deterministic,
train-only set of real Delphes canaries.

## Result

- Status: `{summary['status']}`
- Selected and processed canaries: {summary['canaries_processed']}
- Passing canaries: {summary['canaries_passed']}
- Failing canaries: {summary['canaries_failed']}
- Candidate rows compared: {summary['parity_rows_compared']}
- Column comparisons: {summary['parity_columns_compared']}
- Total parity mismatches: {summary['total_parity_mismatches']}
- Zero-row canaries: {summary['zero_row_canaries']}
- Temporary files remaining: {summary['temporary_files_remaining']}

Selected member indices: {selected_members}.

The selection was generated from declarative rules using the lowest train
member index per rule, followed by deduplication. Both builders read the same
extracted ROOT file and used identical explicit reconstruction parameters.
Arrow types, nullability, row ordering, values, provenance indices, and
zero-row behavior were compared. No physics-shape plots were used.

## Safety record

- Validation canaries: {summary['validation_canaries']}
- Test canaries: {summary['test_canaries']}
- ROOT files opened in three-b mode: {summary['root_files_opened_in_threeb_mode']}
- Three-b candidates reconstructed: {summary['threeb_candidates_reconstructed']}
- Physics yields calculated: {summary['physics_yields_calculated']}
- Normalizations calculated: {summary['normalizations_calculated']}
- Transfer factors calculated: {summary['transfer_factors_calculated']}
- Significances calculated: {summary['significances_calculated']}

All downloaded bundles, nested archives, extracted ROOT files, and temporary
Parquet outputs were deleted after their individual comparison.

## Reproducibility

- Source commit: `{summary['source_commit']}`
- Source-map SHA-256: `{summary['source_map_sha256']}`
- Frozen builder SHA-256: `{summary['frozen_builder_sha256']}`
- Parity builder SHA-256: `{summary['parity_builder_sha256']}`
- Configuration SHA-256: `{summary['configuration_sha256']}`
- Validation script SHA-256: `{summary['validation_script_sha256']}`

## Next gate

`{summary['next_gate']}`
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate frozen-versus-new HH4b four-b reconstruction parity "
            "on deterministic train-only real Delphes canaries."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    script_path = Path(__file__).resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repo / args.config).resolve()
    )
    config = read_json_object(config_path)
    require(config.get("schema_version") == 1, "unsupported config schema")
    require(
        config.get("allowed_dataset_split") == "train",
        "only train canaries are permitted",
    )
    require(
        config.get("validation_access_allowed") is False,
        "validation access must be disabled",
    )
    require(
        config.get("test_access_allowed") is False,
        "test access must be disabled",
    )
    require(
        config.get("parity_mode") == "fourb-parity",
        "the only authorized builder mode is fourb-parity",
    )

    source_commit = str(config["source_commit"]).lower()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    require(head == source_commit, f"HEAD {head} does not match {source_commit}")

    for value in config["prerequisite_checkpoint_dirs"]:
        verify_checkpoint(resolve(repo, value))
    source_map_path = resolve(repo, config["source_map_path"])
    require(
        sha256_file(source_map_path) == config["source_map_sha256"],
        "source-map SHA-256 mismatch",
    )
    frozen_builder = resolve(repo, config["frozen_builder_path"])
    parity_builder = resolve(repo, config["parity_builder_path"])
    require(
        sha256_file(frozen_builder) == config["frozen_builder_sha256"],
        "frozen builder SHA-256 mismatch",
    )
    require(
        sha256_file(parity_builder) == config["parity_builder_sha256"],
        "parity builder SHA-256 mismatch",
    )
    builder_diff = subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "--",
            str(frozen_builder.relative_to(repo)),
            str(parity_builder.relative_to(repo)),
        ],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(builder_diff.returncode == 0, "a builder has uncommitted changes")

    output_dir = resolve(repo, config["output_dir"])
    checkpoint_dir = resolve(repo, config["checkpoint_dir"])
    output_tmp = output_dir.with_name(output_dir.name + "_incomplete")
    checkpoint_tmp = checkpoint_dir.with_name(
        checkpoint_dir.name + "_incomplete"
    )
    for path in (output_dir, output_tmp, checkpoint_dir, checkpoint_tmp):
        require(not path.exists(), f"refusing to overwrite {path}")
    temporary_root = resolve(repo, config["temporary_root"])
    require(temporary_root.is_dir(), "temporary root does not exist")

    source_fields, source_rows = read_tsv(source_map_path)
    required_source_fields = {
        "member_index",
        "sample_class",
        "process_family",
        "process_or_mode",
        "dataset_split",
        "generated_events",
        "candidate_rows",
        "resolution_method",
        "layout_rule",
        "remote_bundle_path",
        "remote_bundle_uri",
        "bundle_size_bytes",
        "root_container_member",
        "root_archive_member",
        "root_basename",
        "resolution_status",
        "ambiguity_count",
    }
    require(
        required_source_fields <= set(source_fields),
        "source map lacks a required field",
    )
    require(
        all(
            row["resolution_status"] == "resolved"
            and int(row["ambiguity_count"]) == 0
            for row in source_rows
        ),
        "source map contains unresolved or ambiguous members",
    )
    require(
        len({row["member_index"] for row in source_rows})
        == len(source_rows),
        "source map contains duplicate member indices",
    )
    canaries = select_canaries(source_rows, config)
    selection_rows = [
        {field: row.get(field, "") for field in SELECTION_FIELDS}
        for row in canaries
    ]

    output_tmp.mkdir(parents=True, exist_ok=False)
    write_tsv(
        output_tmp / "real_parity_canary_selection.tsv",
        selection_rows,
        SELECTION_FIELDS,
    )

    file_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    access_rows: list[dict[str, Any]] = []
    for canary_number, canary in enumerate(canaries, start=1):
        (
            file_row,
            canary_schema_rows,
            canary_column_rows,
            canary_mismatch_rows,
            edge_row,
            access_row,
        ) = process_canary(
            repo=repo,
            canary=canary,
            canary_number=canary_number,
            canary_total=len(canaries),
            config=config,
            frozen_builder=frozen_builder,
            parity_builder=parity_builder,
        )
        file_rows.append(file_row)
        schema_rows.extend(canary_schema_rows)
        column_rows.extend(canary_column_rows)
        mismatch_rows.extend(canary_mismatch_rows)
        if edge_row is not None:
            edge_rows.append(edge_row)
        access_rows.append(access_row)

    canaries_passed = sum(
        row["parity_status"] == "pass" for row in file_rows
    )
    comparison_rows = [
        row["_comparison"] for row in file_rows if "_comparison" in row
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "hh4b_4b_real_delphes_parity_fail",
        "source_commit": source_commit,
        "configuration": str(config_path),
        "configuration_sha256": sha256_file(config_path),
        "validation_script": str(script_path),
        "validation_script_sha256": sha256_file(script_path),
        "source_map_path": str(source_map_path),
        "source_map_sha256": sha256_file(source_map_path),
        "prerequisite_checkpoint_sha256s_verified": True,
        "frozen_builder_path": str(frozen_builder),
        "frozen_builder_sha256": sha256_file(frozen_builder),
        "parity_builder_path": str(parity_builder),
        "parity_builder_sha256": sha256_file(parity_builder),
        "canaries_selected": len(canaries),
        "canaries_processed": len(file_rows),
        "canaries_passed": canaries_passed,
        "canaries_failed": len(file_rows) - canaries_passed,
        "train_canaries": sum(
            row["dataset_split"] == "train" for row in canaries
        ),
        "validation_canaries": 0,
        "test_canaries": 0,
        "direct_layout_canaries": sum(
            not bool(row["root_container_member"]) for row in canaries
        ),
        "nested_layout_canaries": sum(
            bool(row["root_container_member"]) for row in canaries
        ),
        "zero_row_canaries": sum(
            row["frozen_selected_rows"] == 0 for row in edge_rows
        ),
        "nonzero_row_canaries": sum(
            row["frozen_selected_rows"] > 0 for row in edge_rows
        ),
        "signal_canaries": sum(
            row["sample_class"] == "signal" for row in canaries
        ),
        "background_canaries": sum(
            row["sample_class"] == "background" for row in canaries
        ),
        "frozen_builder_invocations": sum(
            bool(row["frozen_builder_invoked"]) for row in access_rows
        ),
        "parity_builder_invocations": sum(
            bool(row["parity_builder_invoked"]) for row in access_rows
        ),
        "real_root_files_opened_by_frozen_builder": sum(
            row["frozen_builder_return_code"] == 0 for row in access_rows
        ),
        "real_root_files_opened_by_parity_builder": sum(
            row["parity_builder_return_code"] == 0 for row in access_rows
        ),
        "real_root_files_opened_for_metadata": sum(
            bool(row["root_metadata_opened"]) for row in access_rows
        ),
        "parity_files_compared": len(comparison_rows),
        "parity_rows_compared": sum(
            row["frozen_rows"] for row in comparison_rows
        ),
        "parity_columns_compared": sum(
            row["columns_compared"] for row in comparison_rows
        ),
        "schema_mismatches": sum(
            row["schema_mismatches"] for row in comparison_rows
        ),
        "row_count_mismatches": sum(
            row["row_count_mismatch"] for row in comparison_rows
        ),
        "event_order_mismatches": sum(
            row["event_order_mismatch"] for row in comparison_rows
        ),
        "integer_value_mismatches": sum(
            row["integer_value_mismatches"] for row in comparison_rows
        ),
        "string_value_mismatches": sum(
            row["string_value_mismatches"] for row in comparison_rows
        ),
        "floating_value_mismatches": sum(
            row["floating_value_mismatches"] for row in comparison_rows
        ),
        "total_parity_mismatches": sum(
            row["mismatch_count"] for row in comparison_rows
        )
        + sum("_comparison" not in row for row in file_rows),
        "root_files_opened_in_threeb_mode": 0,
        "threeb_candidates_reconstructed": 0,
        "validation_members_considered": 0,
        "test_members_considered": 0,
        "physics_yields_calculated": 0,
        "normalizations_calculated": 0,
        "transfer_factors_calculated": 0,
        "significances_calculated": 0,
        "temporary_files_remaining": sum(
            int(row["temporary_files_remaining"]) for row in access_rows
        ),
        "outer_bundles_downloaded": sum(
            row["xrdcp_status"] == "pass" for row in access_rows
        ),
        "root_files_extracted": sum(
            bool(row["root_file_extracted"]) for row in access_rows
        ),
        "gate_errors": [],
        "next_gate": config["next_gate"],
    }

    gate_errors: list[str] = []

    def gate(condition: bool, message: str) -> None:
        if not condition:
            gate_errors.append(message)

    gate(summary["canaries_selected"] >= 8, "fewer than eight canaries")
    gate(
        summary["canaries_processed"] == summary["canaries_selected"],
        "not every selected canary was processed",
    )
    gate(
        summary["canaries_passed"] == summary["canaries_selected"],
        "not every selected canary passed",
    )
    gate(summary["canaries_failed"] == 0, "one or more canaries failed")
    gate(
        summary["train_canaries"] == summary["canaries_selected"],
        "non-train canary selected",
    )
    gate(summary["validation_canaries"] == 0, "validation canary selected")
    gate(summary["test_canaries"] == 0, "test canary selected")
    gate(summary["direct_layout_canaries"] >= 1, "direct layout uncovered")
    gate(summary["nested_layout_canaries"] >= 1, "nested layout uncovered")
    gate(summary["zero_row_canaries"] >= 1, "zero-row behavior uncovered")
    gate(summary["nonzero_row_canaries"] >= 1, "nonzero behavior uncovered")
    gate(summary["signal_canaries"] >= 1, "signal canary missing")
    gate(summary["background_canaries"] >= 1, "background canary missing")
    for field in (
        "frozen_builder_invocations",
        "parity_builder_invocations",
        "real_root_files_opened_by_frozen_builder",
        "real_root_files_opened_by_parity_builder",
        "parity_files_compared",
    ):
        gate(
            summary[field] == summary["canaries_selected"],
            f"{field} differs from selected canaries",
        )
    for field in (
        "schema_mismatches",
        "row_count_mismatches",
        "event_order_mismatches",
        "integer_value_mismatches",
        "string_value_mismatches",
        "floating_value_mismatches",
        "total_parity_mismatches",
        "root_files_opened_in_threeb_mode",
        "threeb_candidates_reconstructed",
        "validation_members_considered",
        "test_members_considered",
        "physics_yields_calculated",
        "normalizations_calculated",
        "transfer_factors_calculated",
        "significances_calculated",
        "temporary_files_remaining",
    ):
        gate(summary[field] == 0, f"{field} is nonzero")
    gate(not mismatch_rows, "mismatch detail table is not header-only")
    gate(
        len(schema_rows)
        == summary["canaries_selected"]
        * int(config["comparison"]["expected_ordered_columns"]),
        "schema comparison row count differs from contract",
    )
    gate(
        len(column_rows)
        == summary["canaries_selected"]
        * int(config["comparison"]["expected_ordered_columns"]),
        "column comparison row count differs from contract",
    )
    gate(
        all(row["extraction_status"] == "pass" for row in access_rows),
        "one or more extractions failed",
    )
    summary["gate_errors"] = gate_errors
    if not gate_errors:
        summary["status"] = "hh4b_4b_real_delphes_parity_pass"

    for row in file_rows:
        row.pop("_comparison", None)
    write_tsv(
        output_tmp / "real_parity_file_summary.tsv",
        file_rows,
        FILE_SUMMARY_FIELDS,
    )
    write_tsv(
        output_tmp / "real_parity_schema_comparison.tsv",
        schema_rows,
        SCHEMA_COMPARISON_FIELDS,
    )
    write_tsv(
        output_tmp / "real_parity_column_comparison.tsv",
        column_rows,
        COLUMN_COMPARISON_FIELDS,
    )
    write_tsv(
        output_tmp / "real_parity_mismatch_details.tsv",
        mismatch_rows,
        MISMATCH_DETAIL_FIELDS,
    )
    write_tsv(
        output_tmp / "real_parity_edge_coverage.tsv",
        edge_rows,
        EDGE_COVERAGE_FIELDS,
    )
    write_tsv(
        output_tmp / "extraction_access_audit.tsv",
        access_rows,
        EXTRACTION_AUDIT_FIELDS,
    )
    (output_tmp / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_tmp.rename(output_dir)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if gate_errors:
        raise RuntimeError(
            "real four-b parity gate failed: " + "; ".join(gate_errors)
        )

    checkpoint_tmp.mkdir(parents=True, exist_ok=False)
    for name in OUTPUT_TABLES:
        shutil.copy2(output_dir / name, checkpoint_tmp / name)
    (checkpoint_tmp / "README.md").write_text(
        checkpoint_readme(summary, selection_rows),
        encoding="utf-8",
    )
    checkpoint = {
        "schema_version": 1,
        "classification": (
            "paper_quality_real_delphes_fourb_parity_canary_checkpoint"
        ),
        "status": summary["status"],
        "source_commit": source_commit,
        "canaries_selected": summary["canaries_selected"],
        "canaries_passed": summary["canaries_passed"],
        "total_parity_mismatches": summary["total_parity_mismatches"],
        "parity_rows_compared": summary["parity_rows_compared"],
        "parity_columns_compared": summary["parity_columns_compared"],
        "validation_members_considered": 0,
        "test_members_considered": 0,
        "root_files_opened_in_threeb_mode": 0,
        "threeb_candidates_reconstructed": 0,
        "physics_yields_calculated": 0,
        "normalizations_calculated": 0,
        "transfer_factors_calculated": 0,
        "significances_calculated": 0,
        "temporary_files_remaining": 0,
        "next_gate": summary["next_gate"],
    }
    (checkpoint_tmp / "checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(checkpoint_tmp.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_lines.append(f"{sha256_file(path)}  {path.name}")
    (checkpoint_tmp / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
    checkpoint_tmp.rename(checkpoint_dir)


if __name__ == "__main__":
    main()
