#!/usr/bin/env python3
"""Audit full HH4b background and signal coverage from frozen metadata only.

This gate deliberately has no candidate-data reader.  It reads TSV, JSON, and
SHA256SUMS metadata; it never opens Parquet, ROOT, model, or prediction files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple


STATUS = "hh4b_full_5m_background_and_signal_coverage_audit_pass"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

BACKGROUND_REGISTRY_FIELDS = (
    "component",
    "family",
    "source_campaign",
    "target_tag",
    "shard_id",
    "seed",
    "dataset_split",
    "dataset_role",
    "events",
    "canonical_identity",
    "canonical_membership",
    "count_toward_background_target",
    "physics_yield_authorized",
)
BACKGROUND_SOURCE_FIELDS = (
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
    "local_candidate_parquet",
    "local_candidate_rows",
    "local_candidate_sha256",
    "local_schema_sha256",
    "remote_bundle",
)
STRICT_AUDIT_FIELDS = (
    "registry_index",
    "family",
    "dataset_split",
    "sealed_test",
    "target_tag",
    "generated_events",
    "expected_candidate_rows",
    "observed_candidate_rows",
    "candidate_columns",
    "candidate_schema_sha256",
    "candidate_schema_class",
    "candidate_sha256",
    "destination_path",
    "technical_payload_status",
)
SIGNAL_FIELDS = (
    "global_member_index",
    "signal_mode",
    "mode_member_index",
    "member_id",
    "source_campaign",
    "generated_events",
    "immutable_split",
    "candidate_rows",
    "candidate_access",
    "candidate_parquet",
    "candidate_parquet_sha256",
    "candidate_schema_columns",
    "candidate_schema_sha256",
    "common72_schema_sha256",
    "source_bundle",
    "source_registry",
    "source_registry_sha256",
    "source_summary",
    "source_summary_sha256",
    "training_authorized",
    "validation_authorized",
    "test_sealed",
    "physics_yield_authorized",
    "status",
)
DEVELOPMENT_FIELDS = (
    "member_index",
    "sample_class",
    "training_target",
    "process_or_mode",
    "dataset_split",
    "generated_events",
    "candidate_rows",
    "source_role",
    "registered_path",
    "local_path",
    "candidate_sha256",
    "canonical_schema_sha256",
    "canonical_projection_compatible",
    "test_member",
)

GENERATED_FILES = (
    "README.md",
    "background_component_coverage.tsv",
    "checkpoint.json",
    "development_disposition.tsv",
    "environment.json",
    "input_integrity.tsv",
    "member_coverage.tsv",
    "signal_mode_coverage.tsv",
    "summary.json",
    "validation_integrity.tsv",
)


class AuditError(RuntimeError):
    """Raised when a frozen source-level invariant does not hold."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    require(
        actual == expected,
        f"{label}: expected {expected!r}, observed {actual!r}",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_int(value: str, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label}: expected integer, observed {value!r}") from exc


def parse_bool(value: str, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise AuditError(f"{label}: expected boolean, observed {value!r}")


def require_sha256(value: str, label: str) -> None:
    require(bool(SHA256_RE.fullmatch(value)), f"{label}: invalid SHA-256 {value!r}")


def resolve_inside(repo: Path, relative: str) -> Path:
    require(not Path(relative).is_absolute(), f"configured path must be relative: {relative}")
    resolved = (repo / relative).resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError as exc:
        raise AuditError(f"configured path escapes repository: {relative}") from exc
    return resolved


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def load_tsv(path: Path, required_fields: Sequence[str]) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        missing = sorted(set(required_fields) - set(fields))
        require(not missing, f"{path}: missing required columns {missing}")
        rows = list(reader)
    require(rows, f"{path}: no data rows")
    return rows


def assert_unique(
    rows: Sequence[Mapping[str, str]],
    key: Callable[[Mapping[str, str]], Any],
    label: str,
) -> None:
    counts = Counter(key(row) for row in rows)
    duplicates = [item for item, count in counts.items() if count != 1]
    require(not duplicates, f"{label}: duplicate keys {duplicates[:5]}")


def group_counts(rows: Iterable[Mapping[str, str]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(row[field] for row in rows).items()))


def group_sums(
    rows: Iterable[Mapping[str, str]],
    group_field: str,
    value_field: str,
) -> Dict[str, int]:
    result: Counter[str] = Counter()
    for row in rows:
        result[row[group_field]] += parse_int(
            row[value_field],
            f"{group_field}={row[group_field]} {value_field}",
        )
    return dict(sorted(result.items()))


def total(rows: Iterable[Mapping[str, str]], field: str) -> int:
    return sum(parse_int(row[field], field) for row in rows)


def git_value(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def git_is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(
        completed.returncode in {0, 1},
        f"git ancestry check failed: {completed.stderr.strip()}",
    )
    return completed.returncode == 0


def verify_input_hashes(
    repo: Path,
    config: Mapping[str, Any],
) -> Tuple[Dict[str, Path], List[Dict[str, Any]]]:
    paths: Dict[str, Path] = {}
    audit_rows: List[Dict[str, Any]] = []
    for name, spec in config["inputs"].items():
        path = resolve_inside(repo, spec["path"])
        require(path.is_file(), f"{name}: input does not exist: {path}")
        expected = spec["sha256"]
        require_sha256(expected, f"{name} configured hash")
        observed = sha256_file(path)
        matched = observed == expected
        audit_rows.append(
            {
                "input_name": name,
                "path": spec["path"],
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matched": matched,
            }
        )
        require(matched, f"{name}: frozen input SHA-256 mismatch")
        paths[name] = path
    return paths, audit_rows


def verify_validation_checkpoint(
    repo: Path,
    config: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    spec = config["validation_checkpoint"]
    directory = resolve_inside(repo, spec["directory"])
    sums_path = directory / "SHA256SUMS"
    require(sums_path.is_file(), f"validation SHA256SUMS missing: {sums_path}")
    observed_sums_hash = sha256_file(sums_path)
    require_equal(
        observed_sums_hash,
        spec["sha256sums_sha256"],
        "canonical-v1 validation SHA256SUMS hash",
    )

    rows: List[Dict[str, Any]] = []
    with sums_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            require(line, f"{sums_path}:{line_number}: blank checksum line")
            parts = line.split(None, 1)
            require(len(parts) == 2, f"{sums_path}:{line_number}: malformed checksum")
            expected, relative = parts
            relative = relative.lstrip("*")
            require_sha256(expected, f"{sums_path}:{line_number}")
            target = (directory / relative).resolve()
            try:
                target.relative_to(directory.resolve())
            except ValueError as exc:
                raise AuditError(
                    f"{sums_path}:{line_number}: checksum path escapes checkpoint"
                ) from exc
            require(target.is_file(), f"validation artifact missing: {target}")
            observed = sha256_file(target)
            matched = observed == expected
            rows.append(
                {
                    "path": relative,
                    "expected_sha256": expected,
                    "observed_sha256": observed,
                    "matched": matched,
                }
            )
            require(matched, f"canonical-v1 validation artifact changed: {relative}")

    checkpoint = load_json(directory / "checkpoint.json")
    require_equal(
        checkpoint.get("status"),
        spec["required_status"],
        "canonical-v1 validation checkpoint status",
    )
    require_equal(
        checkpoint.get("next_gate"),
        config["audit_name"],
        "canonical-v1 validation next gate",
    )
    return rows, {
        "directory": spec["directory"],
        "sha256sums_sha256": observed_sums_hash,
        "artifacts_verified": len(rows),
        "status": checkpoint["status"],
        "unchanged": True,
    }


def audit_background(
    config: Mapping[str, Any],
    registry_rows: List[Dict[str, str]],
    source_rows: List[Dict[str, str]],
    strict_rows: List[Dict[str, str]],
    registry_summary: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    strict_summary: Mapping[str, Any],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[int, Dict[str, str]],
    Dict[str, Any],
]:
    expected = config["expected"]["background"]
    schema = config["schema_contract"]

    require_equal(registry_summary.get("status"), "pass", "background registry status")
    require_equal(
        registry_summary.get("unified_background_5m_registry_valid"),
        True,
        "background registry validity",
    )
    require_equal(source_summary.get("status"), "pass", "background source status")
    require_equal(
        source_summary.get("background_5m_classifier_source_manifest_valid"),
        True,
        "background source manifest validity",
    )
    require_equal(strict_summary.get("status"), "pass", "strict493 payload status")
    require_equal(
        strict_summary.get("test_row_content_accessed"),
        False,
        "prior strict493 sealed-test row-content access",
    )

    require_equal(len(registry_rows), expected["members"], "background registry members")
    require_equal(len(source_rows), expected["members"], "background source members")
    assert_unique(registry_rows, lambda row: row["canonical_identity"], "background registry")
    assert_unique(source_rows, lambda row: row["canonical_identity"], "background sources")
    assert_unique(source_rows, lambda row: row["registry_index"], "background registry indices")

    source_by_identity = {row["canonical_identity"]: row for row in source_rows}
    require_equal(
        set(source_by_identity),
        {row["canonical_identity"] for row in registry_rows},
        "background registry/source identity universe",
    )

    comparison_fields = (
        ("component", "component"),
        ("family", "family"),
        ("dataset_split", "dataset_split"),
        ("dataset_role", "dataset_role"),
        ("events", "generated_events"),
        ("source_campaign", "source_campaign"),
        ("target_tag", "target_tag"),
        ("shard_id", "shard_id"),
        ("seed", "seed"),
        ("receipt_path", "receipt_path"),
        ("remote_bundle", "remote_bundle"),
        ("bundle_sha256", "bundle_sha256"),
        ("bundle_adler32", "bundle_adler32"),
    )
    for registry_row in registry_rows:
        identity = registry_row["canonical_identity"]
        source_row = source_by_identity[identity]
        require(parse_bool(registry_row["canonical_membership"], identity), identity)
        require(
            parse_bool(registry_row["count_toward_background_target"], identity),
            f"{identity}: does not count toward background target",
        )
        require(
            not parse_bool(registry_row["physics_yield_authorized"], identity),
            f"{identity}: unexpected physics-yield authorization",
        )
        for registry_field, source_field in comparison_fields:
            require_equal(
                registry_row[registry_field],
                source_row[source_field],
                f"{identity} {registry_field}/{source_field}",
            )
        if registry_row["candidate_rows"]:
            require_equal(
                registry_row["candidate_rows"],
                source_row["candidate_rows"],
                f"{identity} candidate rows",
            )
        else:
            require_equal(
                source_row["component"],
                "ttbar",
                f"{identity} only legacy ttbar may omit registry candidate rows",
            )

    require_equal(total(registry_rows, "events"), expected["generated_events"], "background events")
    require_equal(
        total(source_rows, "candidate_rows"),
        expected["candidate_rows_metadata"],
        "background candidate-row metadata",
    )
    require_equal(
        group_counts(registry_rows, "component"),
        expected["component_members"],
        "background component members",
    )
    require_equal(
        group_sums(registry_rows, "component", "events"),
        expected["component_events"],
        "background component events",
    )
    require_equal(
        group_counts(registry_rows, "dataset_split"),
        expected["split_members"],
        "background split members",
    )
    require_equal(
        group_sums(registry_rows, "dataset_split", "events"),
        expected["split_events"],
        "background split events",
    )

    local_rows = [
        row for row in source_rows if row["availability"] == "local_candidate_parquet"
    ]
    remote_rows = [
        row
        for row in source_rows
        if row["availability"] == "remote_bundle_extraction_required"
    ]
    require_equal(
        len(local_rows),
        expected["legacy_ttbar_members"],
        "legacy ttbar local members",
    )
    require_equal(len(remote_rows), expected["strict493_members"], "strict493 members")
    require_equal(
        total(local_rows, "generated_events"),
        expected["legacy_ttbar_generated_events"],
        "legacy ttbar events",
    )
    require_equal(
        total(local_rows, "candidate_rows"),
        expected["legacy_ttbar_candidate_rows"],
        "legacy ttbar candidate rows",
    )
    require_equal(
        total(remote_rows, "generated_events"),
        expected["strict493_generated_events"],
        "strict493 events",
    )
    require_equal(
        total(remote_rows, "candidate_rows"),
        expected["strict493_candidate_rows"],
        "strict493 candidate rows",
    )
    require_equal(
        len(local_rows) + len(remote_rows),
        len(source_rows),
        "recognized background source availability",
    )

    for row in local_rows:
        label = row["canonical_identity"]
        require_equal(row["component"], "ttbar", f"{label} component")
        require(row["local_candidate_parquet"], f"{label}: missing local candidate reference")
        require_equal(row["remote_bundle"], "", f"{label} remote bundle")
        require_equal(
            row["local_candidate_rows"],
            row["candidate_rows"],
            f"{label} local candidate rows",
        )
        require_sha256(row["local_candidate_sha256"], f"{label} local candidate hash")
        require_equal(
            row["local_schema_sha256"],
            schema["legacy_ttbar_schema_sha256"],
            f"{label} legacy schema",
        )
        require_equal(row["dataset_split"] == "test", False, f"{label} sealed-test state")

    strict_by_index = {
        parse_int(row["registry_index"], "strict493 registry_index"): row
        for row in strict_rows
    }
    assert_unique(strict_rows, lambda row: row["registry_index"], "strict493 payload rows")
    remote_indices = {
        parse_int(row["registry_index"], "background source registry_index")
        for row in remote_rows
    }
    require_equal(set(strict_by_index), remote_indices, "strict493 source/audit indices")
    source_by_index = {
        parse_int(row["registry_index"], "background source registry_index"): row
        for row in source_rows
    }
    for index, strict_row in strict_by_index.items():
        source_row = source_by_index[index]
        label = source_row["canonical_identity"]
        for source_field, strict_field in (
            ("family", "family"),
            ("dataset_split", "dataset_split"),
            ("target_tag", "target_tag"),
            ("generated_events", "generated_events"),
            ("candidate_rows", "expected_candidate_rows"),
            ("candidate_rows", "observed_candidate_rows"),
        ):
            require_equal(
                source_row[source_field],
                strict_row[strict_field],
                f"{label} {source_field}/{strict_field}",
            )
        sealed = source_row["dataset_split"] == "test"
        require_equal(
            parse_bool(strict_row["sealed_test"], f"{label} sealed_test"),
            sealed,
            f"{label} sealed-test metadata",
        )
        require_equal(
            strict_row["technical_payload_status"],
            "pass",
            f"{label} technical payload status",
        )
        require(strict_row["destination_path"], f"{label}: missing audited destination")
        require_sha256(strict_row["candidate_sha256"], f"{label} candidate hash")
        candidate_rows = parse_int(source_row["candidate_rows"], f"{label} candidate rows")
        if candidate_rows:
            require_equal(
                strict_row["candidate_columns"],
                "72",
                f"{label} nonzero candidate columns",
            )
            require_equal(
                strict_row["candidate_schema_sha256"],
                schema["common72_sha256"],
                f"{label} nonzero candidate schema",
            )

    component_rows: List[Dict[str, Any]] = []
    for component in sorted(expected["component_members"]):
        component_sources = [row for row in source_rows if row["component"] == component]
        component_rows.append(
            {
                "component": component,
                "members": len(component_sources),
                "generated_events": total(component_sources, "generated_events"),
                "candidate_rows_metadata": total(component_sources, "candidate_rows"),
                "source_covered_members": len(component_sources),
                "development_members": sum(
                    row["dataset_split"] != "test"
                    and row["availability"] == "remote_bundle_extraction_required"
                    for row in component_sources
                ),
                "sealed_test_members": sum(
                    row["dataset_split"] == "test" for row in component_sources
                ),
                "legacy_schema_hold_members": sum(
                    row["availability"] == "local_candidate_parquet"
                    for row in component_sources
                ),
            }
        )

    coverage_rows: List[Dict[str, Any]] = []
    for source_row in sorted(source_rows, key=lambda row: int(row["registry_index"])):
        index = parse_int(source_row["registry_index"], "background registry_index")
        is_local = source_row["availability"] == "local_candidate_parquet"
        strict_row = strict_by_index.get(index)
        sealed = source_row["dataset_split"] == "test"
        if sealed:
            disposition = "sealed_background_test"
        elif is_local:
            disposition = "legacy_ttbar_schema_hold"
        else:
            disposition = "development_manifest_member"
        coverage_rows.append(
            {
                "sample_class": "background",
                "source_index": index,
                "member_id": source_row["canonical_identity"],
                "process_or_mode": source_row["family"],
                "component": source_row["component"],
                "dataset_split": source_row["dataset_split"],
                "generated_events": parse_int(
                    source_row["generated_events"], "background generated_events"
                ),
                "candidate_rows_metadata": parse_int(
                    source_row["candidate_rows"], "background candidate_rows"
                ),
                "source_kind": source_row["availability"],
                "source_locator": (
                    source_row["local_candidate_parquet"]
                    if is_local
                    else source_row["remote_bundle"]
                ),
                "candidate_reference": (
                    source_row["local_candidate_parquet"]
                    if is_local
                    else strict_row["destination_path"]
                ),
                "candidate_sha256_metadata": (
                    source_row["local_candidate_sha256"]
                    if is_local
                    else strict_row["candidate_sha256"]
                ),
                "source_covered": True,
                "development_disposition": disposition,
                "sealed_test": sealed,
                "candidate_content_opened_by_this_audit": False,
            }
        )

    info = {
        "members": len(source_rows),
        "generated_events": total(source_rows, "generated_events"),
        "candidate_rows_metadata": total(source_rows, "candidate_rows"),
        "source_covered_members": len(source_rows),
        "strict493_members": len(remote_rows),
        "legacy_ttbar_schema_hold_members": len(local_rows),
        "sealed_test_members": sum(row["dataset_split"] == "test" for row in source_rows),
    }
    return coverage_rows, component_rows, strict_by_index, info


def audit_signal(
    config: Mapping[str, Any],
    signal_rows: List[Dict[str, str]],
    signal_summary: Mapping[str, Any],
    background_registry_rows: Sequence[Mapping[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    expected = config["expected"]["signal"]
    schema = config["schema_contract"]
    require_equal(signal_summary.get("status"), "pass", "signal registry status")
    require_equal(
        signal_summary.get("mode_membership_preserved"),
        True,
        "signal mode membership",
    )
    require_equal(
        signal_summary.get("ggf_test_members_sealed"),
        True,
        "ggF sealed-test state",
    )
    require_equal(len(signal_rows), expected["members"], "signal members")

    assert_unique(signal_rows, lambda row: row["global_member_index"], "signal global indices")
    assert_unique(signal_rows, lambda row: row["member_id"], "signal member IDs")
    assert_unique(
        signal_rows,
        lambda row: (row["signal_mode"], row["mode_member_index"]),
        "signal mode member indices",
    )
    assert_unique(signal_rows, lambda row: row["candidate_parquet"], "signal candidate references")
    assert_unique(
        signal_rows,
        lambda row: row["candidate_parquet_sha256"],
        "signal candidate hashes",
    )
    assert_unique(signal_rows, lambda row: row["source_bundle"], "signal source bundles")

    require_equal(
        {parse_int(row["global_member_index"], "signal global index") for row in signal_rows},
        set(range(expected["members"])),
        "signal global index coverage",
    )
    require_equal(
        total(signal_rows, "generated_events"),
        expected["generated_events"],
        "signal events",
    )
    require_equal(
        total(signal_rows, "candidate_rows"),
        expected["candidate_rows_metadata"],
        "signal candidate-row metadata",
    )
    require_equal(
        group_counts(signal_rows, "signal_mode"),
        expected["mode_members"],
        "signal mode members",
    )
    require_equal(
        group_sums(signal_rows, "signal_mode", "generated_events"),
        expected["mode_events"],
        "signal mode events",
    )
    require_equal(
        group_sums(signal_rows, "signal_mode", "candidate_rows"),
        expected["mode_candidate_rows"],
        "signal mode candidate rows",
    )
    require_equal(
        group_counts(signal_rows, "immutable_split"),
        expected["split_members"],
        "signal split members",
    )
    require_equal(
        group_sums(signal_rows, "immutable_split", "generated_events"),
        expected["split_events"],
        "signal split events",
    )

    background_bundles = {
        row["remote_bundle"] for row in background_registry_rows if row["remote_bundle"]
    }
    signal_bundles = {row["source_bundle"] for row in signal_rows}
    require_equal(
        background_bundles & signal_bundles,
        set(),
        "background/signal source-bundle overlap",
    )

    coverage_rows: List[Dict[str, Any]] = []
    for row in sorted(signal_rows, key=lambda item: int(item["global_member_index"])):
        label = f"{row['signal_mode']}:{row['member_id']}"
        split = row["immutable_split"]
        require(split in {"train", "validation", "test"}, f"{label}: invalid split")
        require_equal(
            parse_bool(row["training_authorized"], f"{label} training_authorized"),
            split == "train",
            f"{label} training authorization",
        )
        require_equal(
            parse_bool(row["validation_authorized"], f"{label} validation_authorized"),
            split == "validation",
            f"{label} validation authorization",
        )
        require_equal(
            parse_bool(row["test_sealed"], f"{label} test_sealed"),
            split == "test",
            f"{label} test sealing",
        )
        require_equal(
            parse_bool(
                row["physics_yield_authorized"],
                f"{label} physics_yield_authorized",
            ),
            False,
            f"{label} physics-yield authorization",
        )
        require_equal(row["status"], "pass", f"{label} source status")
        require(row["candidate_parquet"], f"{label}: missing candidate reference")
        require(row["source_bundle"], f"{label}: missing source bundle")
        require_sha256(row["candidate_parquet_sha256"], f"{label} candidate hash")
        require_sha256(row["source_registry_sha256"], f"{label} source registry hash")
        require_sha256(row["source_summary_sha256"], f"{label} source summary hash")
        require_equal(
            row["common72_schema_sha256"],
            schema["common72_sha256"],
            f"{label} common72 schema",
        )
        if row["signal_mode"] == "ggf_hh4b":
            require_equal(row["candidate_access"], "local_path", f"{label} candidate access")
            require_equal(row["candidate_schema_columns"], "75", f"{label} schema columns")
            require_equal(
                row["candidate_schema_sha256"],
                schema["ggf_75_column_schema_sha256"],
                f"{label} signal schema",
            )
        elif row["signal_mode"] == "vbf_hh4b":
            require_equal(row["candidate_access"], "eos_lfn", f"{label} candidate access")
            require_equal(row["candidate_schema_columns"], "72", f"{label} schema columns")
            require_equal(
                row["candidate_schema_sha256"],
                schema["common72_sha256"],
                f"{label} signal schema",
            )
        else:
            raise AuditError(f"{label}: unrecognized signal mode")

        sealed = split == "test"
        coverage_rows.append(
            {
                "sample_class": "signal",
                "source_index": parse_int(
                    row["global_member_index"], "signal global_member_index"
                ),
                "member_id": row["member_id"],
                "process_or_mode": row["signal_mode"],
                "component": "signal",
                "dataset_split": split,
                "generated_events": parse_int(
                    row["generated_events"], "signal generated_events"
                ),
                "candidate_rows_metadata": parse_int(
                    row["candidate_rows"], "signal candidate_rows"
                ),
                "source_kind": row["candidate_access"],
                "source_locator": row["source_bundle"],
                "candidate_reference": row["candidate_parquet"],
                "candidate_sha256_metadata": row["candidate_parquet_sha256"],
                "source_covered": True,
                "development_disposition": (
                    "sealed_signal_test" if sealed else "development_manifest_member"
                ),
                "sealed_test": sealed,
                "candidate_content_opened_by_this_audit": False,
            }
        )

    mode_rows: List[Dict[str, Any]] = []
    for mode in sorted(expected["mode_members"]):
        rows = [row for row in signal_rows if row["signal_mode"] == mode]
        mode_rows.append(
            {
                "signal_mode": mode,
                "members": len(rows),
                "generated_events": total(rows, "generated_events"),
                "candidate_rows_metadata": total(rows, "candidate_rows"),
                "unique_member_ids": len({row["member_id"] for row in rows}),
                "unique_candidate_references": len(
                    {row["candidate_parquet"] for row in rows}
                ),
                "development_members": sum(
                    row["immutable_split"] != "test" for row in rows
                ),
                "sealed_test_members": sum(
                    row["immutable_split"] == "test" for row in rows
                ),
                "source_coverage_complete": True,
            }
        )
    info = {
        "members": len(signal_rows),
        "generated_events": total(signal_rows, "generated_events"),
        "candidate_rows_metadata": total(signal_rows, "candidate_rows"),
        "source_covered_members": len(signal_rows),
        "unique_member_ids": len({row["member_id"] for row in signal_rows}),
        "unique_candidate_references": len(
            {row["candidate_parquet"] for row in signal_rows}
        ),
        "sealed_test_members": sum(
            row["immutable_split"] == "test" for row in signal_rows
        ),
    }
    return coverage_rows, mode_rows, info


def counter_of_tuples(
    rows: Iterable[Mapping[str, str]],
    fields: Sequence[str],
) -> Counter[Tuple[str, ...]]:
    return Counter(tuple(row[field] for field in fields) for row in rows)


def audit_development(
    config: Mapping[str, Any],
    development_rows: List[Dict[str, str]],
    development_summary: Mapping[str, Any],
    background_sources: List[Dict[str, str]],
    strict_by_index: Mapping[int, Dict[str, str]],
    signal_rows: List[Dict[str, str]],
    member_coverage: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    expected = config["expected"]["development"]
    require_equal(
        development_summary.get("status"),
        "canonical72_development_manifest_frozen",
        "development manifest status",
    )
    require_equal(
        development_summary.get("candidate_rows_read"),
        0,
        "development-manifest construction candidate rows read",
    )
    require_equal(
        development_summary.get("test_candidate_files_opened"),
        0,
        "development-manifest construction test files opened",
    )
    require_equal(len(development_rows), expected["members"], "development members")
    assert_unique(
        development_rows,
        lambda row: row["member_index"],
        "development member indices",
    )
    require_equal(
        total(development_rows, "generated_events"),
        expected["generated_events"],
        "development generated events",
    )
    require_equal(
        total(development_rows, "candidate_rows"),
        expected["candidate_rows"],
        "development candidate rows",
    )
    require_equal(
        sum(row["sample_class"] == "background" for row in development_rows),
        expected["background_members"],
        "development background members",
    )
    require_equal(
        sum(row["sample_class"] == "signal" for row in development_rows),
        expected["signal_members"],
        "development signal members",
    )
    for row in development_rows:
        label = f"development member {row['member_index']}"
        require(row["dataset_split"] != "test", f"{label}: test split present")
        require(
            not parse_bool(row["test_member"], f"{label} test_member"),
            f"{label}: test member present",
        )
        require_equal(
            row["canonical_projection_compatible"],
            "True",
            f"{label} projection compatibility",
        )
        require_equal(
            row["canonical_schema_sha256"],
            config["schema_contract"]["common72_sha256"],
            f"{label} canonical schema",
        )
        require_sha256(row["candidate_sha256"], f"{label} candidate hash")

    development_signal = [
        row for row in development_rows if row["sample_class"] == "signal"
    ]
    non_test_signal = [
        row for row in signal_rows if row["immutable_split"] != "test"
    ]
    require_equal(
        counter_of_tuples(
            development_signal,
            (
                "candidate_sha256",
                "dataset_split",
                "process_or_mode",
                "candidate_rows",
                "generated_events",
                "registered_path",
            ),
        ),
        counter_of_tuples(
            non_test_signal,
            (
                "candidate_parquet_sha256",
                "immutable_split",
                "signal_mode",
                "candidate_rows",
                "generated_events",
                "candidate_parquet",
            ),
        ),
        "development/non-test signal member multiset",
    )

    background_by_index = {
        parse_int(row["registry_index"], "background registry_index"): row
        for row in background_sources
    }
    expected_background_rows: List[Dict[str, str]] = []
    for index, strict_row in strict_by_index.items():
        if strict_row["dataset_split"] == "test":
            continue
        source = background_by_index[index]
        expected_background_rows.append(
            {
                "candidate_sha256": strict_row["candidate_sha256"],
                "dataset_split": source["dataset_split"],
                "process_or_mode": source["family"],
                "candidate_rows": source["candidate_rows"],
                "generated_events": source["generated_events"],
                "registered_path": strict_row["destination_path"],
            }
        )
    development_background = [
        row for row in development_rows if row["sample_class"] == "background"
    ]
    fields = (
        "candidate_sha256",
        "dataset_split",
        "process_or_mode",
        "candidate_rows",
        "generated_events",
        "registered_path",
    )
    require_equal(
        counter_of_tuples(development_background, fields),
        counter_of_tuples(expected_background_rows, fields),
        "development/non-test strict493 background member multiset",
    )

    dispositions = (
        "development_manifest_member",
        "legacy_ttbar_schema_hold",
        "sealed_background_test",
        "sealed_signal_test",
    )
    disposition_rows: List[Dict[str, Any]] = []
    for disposition in dispositions:
        rows = [
            row
            for row in member_coverage
            if row["development_disposition"] == disposition
        ]
        disposition_rows.append(
            {
                "disposition": disposition,
                "members": len(rows),
                "generated_events": sum(int(row["generated_events"]) for row in rows),
                "candidate_rows_metadata": sum(
                    int(row["candidate_rows_metadata"]) for row in rows
                ),
                "candidate_content_opened_by_this_audit": False,
            }
        )

    observed_dispositions = {
        row["disposition"]: row["members"] for row in disposition_rows
    }
    require_equal(
        observed_dispositions["development_manifest_member"],
        expected["members"],
        "development disposition members",
    )
    require_equal(
        observed_dispositions["legacy_ttbar_schema_hold"],
        expected["excluded_legacy_ttbar_members"],
        "legacy ttbar disposition members",
    )
    require_equal(
        observed_dispositions["sealed_background_test"],
        expected["excluded_background_test_members"],
        "sealed background disposition members",
    )
    require_equal(
        observed_dispositions["sealed_signal_test"],
        expected["excluded_signal_test_members"],
        "sealed signal disposition members",
    )
    require_equal(
        sum(
            row["members"]
            for row in disposition_rows
            if row["disposition"] != "development_manifest_member"
        ),
        expected["excluded_source_members"],
        "excluded source members",
    )
    info = {
        "members": len(development_rows),
        "generated_events": total(development_rows, "generated_events"),
        "candidate_rows": total(development_rows, "candidate_rows"),
        "background_members": len(development_background),
        "signal_members": len(development_signal),
        "excluded_source_members": expected["excluded_source_members"],
        "coverage_exact": True,
    }
    return disposition_rows, info


def run_audit(repo: Path, config_path: Path) -> Dict[str, Any]:
    repo = repo.resolve()
    config_path = config_path.resolve()
    config = load_json(config_path)
    require_equal(config.get("schema_version"), 1, "configuration schema")
    require_equal(
        config.get("audit_name"),
        "audit_hh4b_full_5m_background_and_signal_coverage",
        "configuration audit name",
    )
    authorization = config["authorization"]
    required_false = (
        "candidate_content_access_authorized",
        "sealed_test_candidate_content_access_authorized",
        "model_training_authorized",
        "model_scoring_authorized",
        "model_evaluation_authorized",
        "physical_normalization_authorized",
        "physical_significance_authorized",
        "full_5m_development_table_evaluation_authorized",
    )
    require_equal(
        authorization.get("source_level_metadata_audit_only"),
        True,
        "source-level authorization",
    )
    for field in required_false:
        require_equal(authorization.get(field), False, f"authorization {field}")
    outcome = config["outcome"]
    require_equal(
        outcome.get("physical_normalization_ready"),
        False,
        "physical-normalization readiness",
    )
    require_equal(
        outcome.get("candidate_reconstruction_coverage_status"),
        "blocked_on_27_ttbar_schema_holds",
        "candidate-reconstruction coverage status",
    )
    require_equal(
        outcome.get("unresolved_coverage_count"),
        config["expected"]["background"]["legacy_ttbar_members"],
        "unresolved candidate-reconstruction coverage",
    )
    require(
        bool(outcome.get("next_gate")),
        "next gate must be explicit",
    )

    head = git_value(repo, "rev-parse", "HEAD")
    origin = git_value(repo, "rev-parse", "origin/delphes-hh4b-production")
    require(
        git_is_ancestor(repo, config["base_commit"], head),
        "required starting commit is not an ancestor of HEAD",
    )
    require(
        git_is_ancestor(repo, config["base_commit"], origin),
        "required starting commit is not an ancestor of origin",
    )

    input_paths, input_integrity = verify_input_hashes(repo, config)
    validation_rows, validation_info = verify_validation_checkpoint(repo, config)

    background_registry = load_tsv(
        input_paths["background_registry"], BACKGROUND_REGISTRY_FIELDS
    )
    background_sources = load_tsv(
        input_paths["background_source_manifest"], BACKGROUND_SOURCE_FIELDS
    )
    strict_rows = load_tsv(
        input_paths["strict493_payload_audit"], STRICT_AUDIT_FIELDS
    )
    signal_rows = load_tsv(input_paths["signal_registry"], SIGNAL_FIELDS)
    development_rows = load_tsv(
        input_paths["development_manifest"], DEVELOPMENT_FIELDS
    )

    background_coverage, component_rows, strict_by_index, background_info = (
        audit_background(
            config,
            background_registry,
            background_sources,
            strict_rows,
            load_json(input_paths["background_registry_summary"]),
            load_json(input_paths["background_source_summary"]),
            load_json(input_paths["strict493_payload_summary"]),
        )
    )
    signal_coverage, mode_rows, signal_info = audit_signal(
        config,
        signal_rows,
        load_json(input_paths["signal_registry_summary"]),
        background_registry,
    )
    member_coverage = background_coverage + signal_coverage
    combined_expected = config["expected"]["combined"]
    require_equal(len(member_coverage), combined_expected["members"], "combined members")
    require_equal(
        sum(row["generated_events"] for row in member_coverage),
        combined_expected["generated_events"],
        "combined generated events",
    )
    require_equal(
        sum(row["candidate_rows_metadata"] for row in member_coverage),
        combined_expected["candidate_rows_metadata"],
        "combined candidate-row metadata",
    )
    sealed_rows = [row for row in member_coverage if row["sealed_test"]]
    require_equal(
        len(sealed_rows),
        combined_expected["sealed_test_members"],
        "combined sealed-test members",
    )
    require_equal(
        sum(row["generated_events"] for row in sealed_rows),
        combined_expected["sealed_test_generated_events"],
        "combined sealed-test generated events",
    )
    require_equal(
        sum(row["candidate_rows_metadata"] for row in sealed_rows),
        combined_expected["sealed_test_candidate_rows_metadata"],
        "combined sealed-test candidate-row metadata",
    )

    disposition_rows, development_info = audit_development(
        config,
        development_rows,
        load_json(input_paths["development_manifest_summary"]),
        background_sources,
        strict_by_index,
        signal_rows,
        member_coverage,
    )

    config_relative = str(config_path.relative_to(repo))
    summary = {
        "schema_version": 1,
        "status": STATUS,
        "audit_name": config["audit_name"],
        "base_commit": config["base_commit"],
        "next_gate": outcome["next_gate"],
        "physical_normalization_ready": outcome["physical_normalization_ready"],
        "candidate_reconstruction_coverage_status": outcome[
            "candidate_reconstruction_coverage_status"
        ],
        "unresolved_coverage_count": outcome["unresolved_coverage_count"],
        "config_path": config_relative,
        "config_sha256": sha256_file(config_path),
        "background": background_info,
        "signal": signal_info,
        "combined": {
            "members": len(member_coverage),
            "generated_events": sum(
                row["generated_events"] for row in member_coverage
            ),
            "candidate_rows_metadata": sum(
                row["candidate_rows_metadata"] for row in member_coverage
            ),
            "source_covered_members": sum(
                bool(row["source_covered"]) for row in member_coverage
            ),
            "sealed_test_members": len(sealed_rows),
            "sealed_test_generated_events": sum(
                row["generated_events"] for row in sealed_rows
            ),
            "sealed_test_candidate_rows_metadata": sum(
                row["candidate_rows_metadata"] for row in sealed_rows
            ),
        },
        "development": development_info,
        "validation_checkpoint": validation_info,
        "controls": {
            "source_level_metadata_only": True,
            "candidate_content_opened": 0,
            "candidate_files_opened": 0,
            "candidate_rows_read": 0,
            "sealed_test_candidate_content_opened": 0,
            "sealed_test_candidate_files_opened": 0,
            "sealed_test_candidate_rows_read": 0,
            "root_files_opened": 0,
            "model_files_opened": 0,
            "models_trained": 0,
            "models_scored": 0,
            "models_evaluated": 0,
            "predictions_produced": 0,
            "scoring_performed": 0,
            "prediction_files_opened": 0,
            "physical_normalization_performed": False,
            "physical_significance_calculated": False,
        },
        "authorization": authorization,
        "pass_conditions": {
            "frozen_input_hashes_match": all(
                row["matched"] for row in input_integrity
            ),
            "background_5m_source_coverage_complete": (
                background_info["source_covered_members"]
                == config["expected"]["background"]["members"]
            ),
            "unique_signal_source_coverage_complete": (
                signal_info["unique_member_ids"]
                == config["expected"]["signal"]["members"]
            ),
            "development_manifest_exactly_reconciled": development_info[
                "coverage_exact"
            ],
            "sealed_test_content_closed": True,
            "canonical_v1_validation_unchanged": validation_info["unchanged"],
            "no_model_operation": True,
        },
    }
    require(all(summary["pass_conditions"].values()), "one or more pass conditions failed")
    return {
        "config": config,
        "summary": summary,
        "input_integrity": input_integrity,
        "validation_integrity": validation_rows,
        "member_coverage": member_coverage,
        "component_coverage": component_rows,
        "signal_mode_coverage": mode_rows,
        "development_disposition": disposition_rows,
        "git": {"head": head, "origin": origin},
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"cannot write empty table: {path}")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            require_equal(list(row.keys()), fields, f"{path.name} field order")
            writer.writerow(row)


def readme_text(summary: Mapping[str, Any]) -> str:
    background = summary["background"]
    signal = summary["signal"]
    combined = summary["combined"]
    development = summary["development"]
    validation = summary["validation_checkpoint"]
    return f"""# Full 5M background and unique-signal source coverage audit

Status: `{summary['status']}`.

This is a source-level metadata audit only. It opened no candidate Parquet,
ROOT, model, prediction, or sealed-test content.

## Coverage result

- Background: {background['source_covered_members']} / {background['members']} members,
  {background['generated_events']:,} generated events.
- Signal: {signal['source_covered_members']} / {signal['members']} unique members,
  {signal['generated_events']:,} generated events.
- Combined: {combined['source_covered_members']} / {combined['members']} members,
  {combined['generated_events']:,} generated events.
- Candidate-row metadata accounted: {combined['candidate_rows_metadata']:,}.

The 27 canonical legacy-ttbar members are fully represented at source level
but remain a separate schema hold. They are not silently mixed into the
frozen-v2 canonical72 development population.

## Development reconciliation

The frozen development manifest matches exactly {development['members']}
non-test, common72-compatible members ({development['background_members']}
background and {development['signal_members']} signal), with
{development['candidate_rows']:,} candidate rows in its already-frozen
metadata. The 49 non-development source members are exactly 27 legacy-ttbar
schema holds, 19 sealed background-test members, and 3 sealed ggF-test
members.

## Sealed-test boundary

All {combined['sealed_test_members']} test members are covered by frozen
source metadata only. Their {combined['sealed_test_candidate_rows_metadata']}
candidate rows are accounting metadata; candidate files opened and candidate
rows read by this audit are both zero.

## Canonical-v1 preservation

The completed validation checkpoint remains unchanged:
`{validation['sha256sums_sha256']}`. All
{validation['artifacts_verified']} entries in its `SHA256SUMS` manifest were
verified byte-for-byte.

## Authorization boundary

This audit does not authorize candidate-content access, full-5M table
evaluation, model training/scoring/evaluation, physical normalization, or
physical significance.

Candidate-reconstruction coverage status:
`{summary['candidate_reconstruction_coverage_status']}`. Physical
normalization ready: `{str(summary['physical_normalization_ready']).lower()}`.
Unresolved coverage count: {summary['unresolved_coverage_count']}. Next gate:
`{summary['next_gate']}`.
"""


def checkpoint_value(result: Mapping[str, Any]) -> Dict[str, Any]:
    summary = result["summary"]
    return {
        "schema_version": 1,
        "status": summary["status"],
        "audit_name": summary["audit_name"],
        "base_commit": summary["base_commit"],
        "next_gate": summary["next_gate"],
        "physical_normalization_ready": summary["physical_normalization_ready"],
        "candidate_reconstruction_coverage_status": summary[
            "candidate_reconstruction_coverage_status"
        ],
        "unresolved_coverage_count": summary["unresolved_coverage_count"],
        "background_5m_source_coverage_complete": True,
        "unique_signal_source_coverage_complete": True,
        "background_members": summary["background"]["members"],
        "signal_members": summary["signal"]["members"],
        "combined_members": summary["combined"]["members"],
        "legacy_ttbar_schema_hold_members": summary["background"][
            "legacy_ttbar_schema_hold_members"
        ],
        "sealed_test_members": summary["combined"]["sealed_test_members"],
        "sealed_test_candidate_content_opened": False,
        "canonical_v1_validation_unchanged": True,
        "model_operations_performed": 0,
        "full_5m_development_table_evaluation_authorized": False,
        "physical_normalization_authorized": False,
        "physical_significance_authorized": False,
    }


def environment_value(
    repo: Path,
    script_path: Path,
    config_path: Path,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "required_starting_head": result["summary"]["base_commit"],
        "required_starting_origin": result["summary"]["base_commit"],
        "script_path": str(script_path.resolve().relative_to(repo)),
        "script_sha256": sha256_file(script_path),
        "config_path": str(config_path.resolve().relative_to(repo)),
        "config_sha256": sha256_file(config_path),
        "candidate_data_libraries_imported": False,
        "model_libraries_imported": False,
    }


def prepare_directory(directory: Path, overwrite: bool) -> None:
    if directory.exists():
        existing = [path for path in directory.iterdir()]
        if existing and not overwrite:
            raise AuditError(
                f"output directory is not empty (use --overwrite): {directory}"
            )
        for path in existing:
            require(
                path.is_file() and path.name in set(GENERATED_FILES) | {"SHA256SUMS"},
                f"refusing to overwrite unexpected output: {path}",
            )
            path.unlink()
    else:
        directory.mkdir(parents=True)


def write_artifacts(
    directory: Path,
    result: Mapping[str, Any],
    environment: Mapping[str, Any],
    overwrite: bool,
) -> None:
    prepare_directory(directory, overwrite)
    write_json(directory / "summary.json", result["summary"])
    write_json(directory / "checkpoint.json", checkpoint_value(result))
    write_json(directory / "environment.json", environment)
    (directory / "README.md").write_text(
        readme_text(result["summary"]),
        encoding="utf-8",
    )
    write_tsv(directory / "input_integrity.tsv", result["input_integrity"])
    write_tsv(
        directory / "validation_integrity.tsv",
        result["validation_integrity"],
    )
    write_tsv(directory / "member_coverage.tsv", result["member_coverage"])
    write_tsv(
        directory / "background_component_coverage.tsv",
        result["component_coverage"],
    )
    write_tsv(
        directory / "signal_mode_coverage.tsv",
        result["signal_mode_coverage"],
    )
    write_tsv(
        directory / "development_disposition.tsv",
        result["development_disposition"],
    )
    missing = [name for name in GENERATED_FILES if not (directory / name).is_file()]
    require(not missing, f"generated artifacts missing: {missing}")
    checksum_lines = [
        f"{sha256_file(directory / name)}  {name}"
        for name in sorted(GENERATED_FILES)
    ]
    (directory / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/baselines/hh4b_full_5m_background_and_signal_coverage_v1.json"
        ),
        help="configuration path, relative to --repo unless absolute",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def path_from_arg(repo: Path, value: Path | None, configured: str) -> Path:
    if value is None:
        return resolve_inside(repo, configured)
    return value.resolve() if value.is_absolute() else (repo / value).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repo / args.config).resolve()
    )
    try:
        result = run_audit(repo, config_path)
        script_path = Path(__file__).resolve()
        environment = environment_value(
            repo,
            script_path,
            config_path,
            result,
        )
        output_dir = path_from_arg(
            repo,
            args.output_dir,
            result["config"]["outputs"]["runtime_directory"],
        )
        checkpoint_dir = path_from_arg(
            repo,
            args.checkpoint_dir,
            result["config"]["outputs"]["checkpoint_directory"],
        )
        require(output_dir != checkpoint_dir, "runtime and checkpoint directories coincide")
        write_artifacts(output_dir, result, environment, args.overwrite)
        write_artifacts(checkpoint_dir, result, environment, args.overwrite)
    except (AuditError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(STATUS)
    print(json.dumps(result["summary"]["combined"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
