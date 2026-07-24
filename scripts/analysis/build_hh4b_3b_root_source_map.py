#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq


OUTPUT_FILES = (
    "member_level_root_source_map.tsv",
    "resolution_method_summary.tsv",
    "source_access_audit.tsv",
    "unresolved_or_ambiguous_members.tsv",
    "duplicate_source_locators.tsv",
    "archive_layout_validation.tsv",
    "summary.json",
)

MAP_FIELDS = [
    "member_index",
    "sample_class",
    "training_target",
    "process_family",
    "process_or_mode",
    "dataset_split",
    "generated_events",
    "candidate_rows",
    "candidate_path",
    "candidate_sha256",
    "resolution_method",
    "registry_join_method",
    "analysis_sample",
    "source_root_index",
    "evidence_primary_path",
    "evidence_primary_sha256",
    "evidence_secondary_path",
    "remote_bundle_path",
    "remote_bundle_uri",
    "bundle_sha256_expected",
    "bundle_adler32_expected",
    "bundle_exists",
    "bundle_size_bytes",
    "bundle_stat_status",
    "root_container_member",
    "root_archive_member",
    "root_basename",
    "archive_depth",
    "layout_rule",
    "source_locator",
    "ambiguity_count",
    "resolution_status",
]

PROVENANCE_COLUMNS = (
    "source_root",
    "source_root_index",
    "analysis_sample",
)

RESOLUTION_METHODS = (
    "background_receipt_standard_bundle_layout",
    "vbf_registry_standard_bundle_layout",
    "candidate_source_root_nested_reconstruction_bundle",
)


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


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"cannot parse Boolean value {value!r}")


def normalize_location(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("root://"):
        return stripped
    return str(Path(stripped).expanduser().resolve())


def normalize_archive_member(value: str) -> str:
    normalized = value.strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def remote_uri(eos_host: str, remote_path: str) -> str:
    return eos_host.rstrip("/") + "//" + remote_path.lstrip("/")


def json_object(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    require(isinstance(document, dict), f"{path}: JSON is not an object")
    return document


def select_path_first(
    *,
    candidate_path: str,
    candidate_sha256: str,
    path_index: dict[str, list[dict[str, str]]],
    sha_index: dict[str, list[dict[str, str]]],
    label: str,
) -> tuple[dict[str, str], str, int]:
    exact = path_index.get(normalize_location(candidate_path), [])
    if exact:
        require(
            len(exact) == 1,
            f"{label}: exact destination-path match is ambiguous",
        )
        return exact[0], "exact_candidate_path", 0

    checksum = sha_index.get(candidate_sha256.lower(), [])
    require(bool(checksum), f"{label}: no path or checksum match")
    require(
        len(checksum) == 1,
        f"{label}: checksum fallback is ambiguous ({len(checksum)} rows)",
    )
    return checksum[0], "unique_candidate_sha256_fallback", 0


def index_rows(
    *,
    rows: list[dict[str, str]],
    path_field: str,
    sha_field: str,
    allowed_paths: set[str],
    allowed_hashes: set[str],
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
]:
    by_path: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_sha: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        raw_path = row.get(path_field, "").strip()
        raw_sha = row.get(sha_field, "").strip().lower()
        normalized_path = normalize_location(raw_path) if raw_path else ""
        if normalized_path not in allowed_paths and raw_sha not in allowed_hashes:
            continue
        if normalized_path:
            by_path[normalized_path].append(row)
        if raw_sha:
            by_sha[raw_sha].append(row)
    return dict(by_path), dict(by_sha)


def one_checksum_row(
    rows: list[dict[str, str]],
    field: str,
    checksum: str,
    label: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row.get(field, "").strip().lower() == checksum.lower()
    ]
    require(len(matches) == 1, f"{label}: expected one checksum row, got {len(matches)}")
    return matches[0]


def unique_nonempty_column_values(
    parquet_path: Path,
) -> dict[str, str]:
    table = pq.read_table(parquet_path, columns=list(PROVENANCE_COLUMNS))
    result: dict[str, str] = {}
    for column in PROVENANCE_COLUMNS:
        values = {
            str(value).strip()
            for value in table[column].to_pylist()
            if value is not None and str(value).strip()
        }
        require(
            len(values) == 1,
            f"{parquet_path}: {column} has {len(values)} nonempty values",
        )
        result[column] = next(iter(values))
    return result


def stat_remote_bundle(
    *,
    eos_host: str,
    remote_path: str,
    timeout_seconds: int,
    expected_size: str,
) -> dict[str, Any]:
    uri = remote_uri(eos_host, remote_path)
    try:
        completed = subprocess.run(
            ["xrdfs", eos_host, "stat", remote_path],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        return_code: int | str = completed.returncode
        size_matches = re.findall(r"(?m)^Size:\s+([0-9]+)\s*$", stdout)
        reported_size = size_matches[-1] if size_matches else ""
        exists = completed.returncode == 0 and bool(reported_size)
        status = "pass" if exists else "stat_failed"
        failure_text = ""
        if not exists:
            failure_text = (stderr or stdout or "stat returned no size").strip()
        elif expected_size and reported_size != expected_size:
            status = "size_mismatch"
            failure_text = (
                f"expected size {expected_size}, reported {reported_size}"
            )
        return {
            "remote_bundle_path": remote_path,
            "remote_bundle_uri": uri,
            "stat_return_code": return_code,
            "bundle_exists": exists,
            "reported_size_bytes": reported_size,
            "expected_size_bytes": expected_size,
            "size_matches_expected": (
                not expected_size or reported_size == expected_size
            ),
            "bundle_stat_status": status,
            "timeout_or_failure_text": re.sub(
                r"\s+", " ", failure_text
            ).strip(),
        }
    except subprocess.TimeoutExpired as error:
        text = " ".join(
            part
            for part in (
                str(error.stdout or ""),
                str(error.stderr or ""),
            )
            if part
        )
        return {
            "remote_bundle_path": remote_path,
            "remote_bundle_uri": uri,
            "stat_return_code": "timeout",
            "bundle_exists": False,
            "reported_size_bytes": "",
            "expected_size_bytes": expected_size,
            "size_matches_expected": False,
            "bundle_stat_status": "timeout",
            "timeout_or_failure_text": re.sub(r"\s+", " ", text).strip()
            or f"timed out after {timeout_seconds} seconds",
        }


def validate_archive_layouts(
    *,
    canary_path: Path,
    ggf_layout_path: Path,
    map_by_member: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    _, canary_rows = read_tsv(canary_path)
    require(
        Counter(row["canary_role"] for row in canary_rows)
        == Counter(
            {
                "background_nonzero_4b": 1,
                "background_zero_4b": 1,
                "ggf_signal": 1,
                "vbf_signal": 1,
            }
        ),
        "archive canary roles changed",
    )

    validation: list[dict[str, Any]] = []
    for row in canary_rows:
        member = map_by_member.get(row["member_index"])
        require(member is not None, "archive canary references a non-development member")
        require(
            member["remote_bundle_path"] == row["remote_bundle"],
            f"canary member {row['member_index']}: bundle mismatch",
        )
        roots = [
            normalize_archive_member(value)
            for value in json.loads(row["root_members_json"])
        ]
        if row["canary_role"] == "ggf_signal":
            require(
                int(row["root_member_count"]) == 0 and not roots,
                "ggF outer canary unexpectedly contains a direct ROOT member",
            )
            observed = member["root_container_member"]
            rule = "ggf_outer_bundle_nested_archive_presence"
        else:
            require(
                int(row["root_member_count"]) == 1 and len(roots) == 1,
                f"{row['canary_role']}: expected one direct ROOT member",
            )
            require(
                roots[0] == member["root_archive_member"],
                f"{row['canary_role']}: direct ROOT-member mismatch",
            )
            observed = roots[0]
            rule = "standard_bundle_direct_root_member"
        validation.append(
            {
                "evidence_source": str(canary_path),
                "evidence_kind": row["canary_role"],
                "member_index": row["member_index"],
                "remote_bundle_path": row["remote_bundle"],
                "root_container_member": member["root_container_member"],
                "root_archive_member": member["root_archive_member"],
                "observed_layout_value": observed,
                "layout_rule": rule,
                "validation_status": "pass",
            }
        )

    _, ggf_rows = read_tsv(ggf_layout_path)
    require(len(ggf_rows) == 2, "ggF layout evidence must contain two campaigns")
    require(
        {row["source_campaign"] for row in ggf_rows}
        == {
            "ggf_hh4b_ml_ext20k_frozen_v2_20260716",
            "ggf_hh4b_ml_ext80k_frozen_v2_20260721_v1",
        },
        "ggF layout evidence campaigns changed",
    )
    for row in ggf_rows:
        member = map_by_member.get(row["member_index"])
        require(member is not None, "ggF layout evidence references a non-development member")
        require(
            member["remote_bundle_path"] == row["remote_bundle"],
            f"ggF layout member {row['member_index']}: bundle mismatch",
        )
        nested_roots = [
            normalize_archive_member(value)
            for value in json.loads(row["nested_root_members_json"])
        ]
        require(
            int(row["discovered_archive_root_count"]) == 1
            and int(row["source_root_basename_match_count"]) == 1
            and nested_roots == [member["root_archive_member"]],
            f"ggF layout member {row['member_index']}: nested ROOT mismatch",
        )
        nested_archives = json.loads(row["nested_archives_json"])
        archive_members = [
            normalize_archive_member(value["archive_member"])
            for value in nested_archives
        ]
        require(
            archive_members == [member["root_container_member"]],
            f"ggF layout member {row['member_index']}: nested container mismatch",
        )
        validation.append(
            {
                "evidence_source": str(ggf_layout_path),
                "evidence_kind": row["source_campaign"],
                "member_index": row["member_index"],
                "remote_bundle_path": row["remote_bundle"],
                "root_container_member": member["root_container_member"],
                "root_archive_member": member["root_archive_member"],
                "observed_layout_value": (
                    member["root_container_member"]
                    + "::"
                    + member["root_archive_member"]
                ),
                "layout_rule": "nested_reconstruction_archive_root_member",
                "validation_status": "pass",
            }
        )

    return validation


def checkpoint_readme(summary: dict[str, Any]) -> str:
    return f"""# HH4b 3b ROOT source map

## Purpose

This checkpoint freezes one Delphes ROOT archive locator for every member of
the canonical HH4b train/validation development membership. It is a
provenance and source-access artifact; it does not reconstruct three-b-tag
candidates or inspect ROOT event data.

## Result

- Status: `{summary['status']}`
- Development members: {summary['development_members']}
- Resolved members: {summary['resolved_members']}
- Train / validation members: {summary['train_members']} / {summary['validation_members']}
- Signal / background members: {summary['signal_members']} / {summary['background_members']}
- Zero-row background members: {summary['zero_row_background_members']}
- Unique remote bundles checked: {summary['unique_development_bundles']}
- Failed remote metadata checks: {summary['remote_metadata_checks_failed']}
- Cross-split locator groups: {summary['cross_split_source_locator_groups']}

The background mapping uses exact destination paths before checksum fallback.
VBF sources use the canonical signal registry and lower materialized registry.
ggF sources use only the three candidate provenance columns and locate ROOT
files inside the nested reconstruction archive carried by each outer bundle.

## Safety record

- Candidate physics columns read: {summary['candidate_physics_columns_read']}
- Test members considered: {summary['test_members_considered']}
- ROOT files extracted: {summary['root_files_extracted']}
- ROOT files opened: {summary['root_files_opened']}
- ROOT event arrays read: {summary['root_event_arrays_read']}
- Event generation performed: {summary['event_generation_performed']}
- Physics yields calculated: {summary['physics_yields_calculated']}

## Reproducibility

- Source commit: `{summary['source_commit']}`
- Development manifest SHA-256: `{summary['development_manifest_sha256']}`
- Configuration SHA-256: `{summary['configuration_sha256']}`

## Next gate

`{summary['next_gate']}`
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a complete bundle/member Delphes ROOT source map for the "
            "frozen HH4b development membership without opening ROOT files."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repo / args.config).resolve()
    )
    require(config_path.is_file(), f"missing configuration: {config_path}")
    config = json_object(config_path)
    require(config.get("schema_version") == 1, "unsupported configuration schema")
    require(config.get("test_access_allowed") is False, "test access must be disabled")

    source_commit = str(config["source_commit"]).lower()
    require(
        re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None,
        "source_commit is not a full SHA",
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    require(head == source_commit, f"HEAD {head} does not match source commit {source_commit}")

    preflight_path = resolve(repo, config["preflight_config_path"])
    require(
        sha256_file(preflight_path) == config["preflight_config_sha256"],
        "preflight configuration SHA-256 mismatch",
    )
    preflight = json_object(preflight_path)
    require(preflight.get("test_access_allowed") is False, "preflight permits test access")

    manifest_path = resolve(repo, preflight["development_manifest_path"])
    signal_registry_path = resolve(repo, preflight["signal_registry_path"])
    background_registry_path = resolve(repo, preflight["background_registry_path"])
    require(
        sha256_file(manifest_path) == preflight["development_manifest_sha256"],
        "development manifest SHA-256 mismatch",
    )
    require(
        sha256_file(signal_registry_path) == config["signal_registry_sha256"],
        "canonical signal registry SHA-256 mismatch",
    )
    require(
        sha256_file(background_registry_path) == config["background_registry_sha256"],
        "background registry SHA-256 mismatch",
    )

    canary_path = resolve(repo, config["archive_canary_inventory_path"])
    ggf_layout_path = resolve(repo, config["ggf_layout_inventory_path"])
    require(
        sha256_file(canary_path) == config["archive_canary_inventory_sha256"],
        "archive canary inventory SHA-256 mismatch",
    )
    require(
        sha256_file(ggf_layout_path) == config["ggf_layout_inventory_sha256"],
        "ggF layout inventory SHA-256 mismatch",
    )

    output_dir = resolve(repo, config["output_dir"])
    checkpoint_dir = resolve(repo, config["checkpoint_dir"])
    output_tmp = output_dir.with_name(output_dir.name + "_incomplete")
    checkpoint_tmp = checkpoint_dir.with_name(checkpoint_dir.name + "_incomplete")
    for path in (output_dir, checkpoint_dir, output_tmp, checkpoint_tmp):
        require(not path.exists(), f"refusing to overwrite {path}")

    _, manifest_rows = read_tsv(manifest_path)
    expected = config["expected"]
    development_splits = set(config["development_splits"])
    require(
        len(manifest_rows) == int(expected["development_members"]),
        "development-member count mismatch",
    )
    require(
        all(
            row["dataset_split"] in development_splits
            and not parse_bool(row["test_member"])
            for row in manifest_rows
        ),
        "manifest contains a sealed or non-development member",
    )
    require(
        len({row["member_index"] for row in manifest_rows}) == len(manifest_rows),
        "duplicate development member index",
    )

    class_counts = Counter(row["sample_class"] for row in manifest_rows)
    split_counts = Counter(row["dataset_split"] for row in manifest_rows)
    mode_counts = Counter(row["process_or_mode"] for row in manifest_rows)
    zero_background = sum(
        row["sample_class"] == "background" and int(row["candidate_rows"]) == 0
        for row in manifest_rows
    )
    require(class_counts["signal"] == int(expected["signal_members"]), "signal count mismatch")
    require(
        class_counts["background"] == int(expected["background_members"]),
        "background count mismatch",
    )
    require(split_counts["train"] == int(expected["train_members"]), "train count mismatch")
    require(
        split_counts["validation"] == int(expected["validation_members"]),
        "validation count mismatch",
    )
    require(
        zero_background == int(expected["zero_row_background_members"]),
        "zero-row background count mismatch",
    )
    require(mode_counts["ggf_hh4b"] == int(expected["ggf_members"]), "ggF count mismatch")
    require(mode_counts["vbf_hh4b"] == int(expected["vbf_members"]), "VBF count mismatch")

    background_manifest = [
        row for row in manifest_rows if row["sample_class"] == "background"
    ]
    signal_manifest = [row for row in manifest_rows if row["sample_class"] == "signal"]
    background_paths = {
        normalize_location(row["local_path"]) for row in background_manifest
    }
    background_hashes = {row["candidate_sha256"].lower() for row in background_manifest}
    signal_paths = {normalize_location(row["local_path"]) for row in signal_manifest}
    signal_hashes = {row["candidate_sha256"].lower() for row in signal_manifest}

    _, background_rows = read_tsv(background_registry_path)
    background_by_path, background_by_sha = index_rows(
        rows=background_rows,
        path_field="destination_path",
        sha_field="candidate_sha256",
        allowed_paths=background_paths,
        allowed_hashes=background_hashes,
    )
    _, signal_rows = read_tsv(signal_registry_path)
    signal_by_path, signal_by_sha = index_rows(
        rows=signal_rows,
        path_field="candidate_parquet",
        sha_field="candidate_parquet_sha256",
        allowed_paths=signal_paths,
        allowed_hashes=signal_hashes,
    )

    lower_registry_cache: dict[Path, list[dict[str, str]]] = {}
    lower_registry_sha: dict[Path, str] = {}
    member_rows: list[dict[str, Any]] = []
    expected_bundle_sizes: dict[str, set[str]] = defaultdict(set)
    provenance_files_read = 0

    for manifest in sorted(manifest_rows, key=lambda row: int(row["member_index"])):
        member_index = manifest["member_index"]
        candidate_path = normalize_location(manifest["local_path"])
        candidate_sha256 = manifest["candidate_sha256"].lower()
        common = {
            "member_index": member_index,
            "sample_class": manifest["sample_class"],
            "training_target": manifest["training_target"],
            "process_family": manifest["process_or_mode"],
            "process_or_mode": manifest["process_or_mode"],
            "dataset_split": manifest["dataset_split"],
            "generated_events": manifest["generated_events"],
            "candidate_rows": manifest["candidate_rows"],
            "candidate_path": candidate_path,
            "candidate_sha256": candidate_sha256,
            "analysis_sample": "",
            "source_root_index": "",
            "bundle_sha256_expected": "",
            "bundle_adler32_expected": "",
            "root_container_member": "",
            "ambiguity_count": 0,
            "resolution_status": "resolved",
        }

        if manifest["sample_class"] == "background":
            registry, join_method, ambiguity = select_path_first(
                candidate_path=candidate_path,
                candidate_sha256=candidate_sha256,
                path_index=background_by_path,
                sha_index=background_by_sha,
                label=f"background member {member_index}",
            )
            require(not parse_bool(registry["sealed_test"]), "background join reached test data")
            receipt_path = Path(registry["materialization_receipt_path"]).resolve()
            receipt = json_object(receipt_path)
            require(receipt.get("status") == "pass", f"{receipt_path}: receipt did not pass")
            require(receipt.get("sealed_test") is False, f"{receipt_path}: receipt is test data")
            require(
                receipt["remote_bundle"].startswith("/store/user/"),
                f"{receipt_path}: invalid remote bundle",
            )
            require(
                str(receipt["candidate_sha256"]).lower() == candidate_sha256,
                f"{receipt_path}: candidate SHA-256 mismatch",
            )
            target_tag = str(receipt["target_tag"])
            root_basename = target_tag + "_delphes.root"
            remote_bundle = str(receipt["remote_bundle"])
            expected_size = str(receipt.get("bundle_bytes", ""))
            if expected_size:
                expected_bundle_sizes[remote_bundle].add(expected_size)
            common.update(
                {
                    "process_family": registry["family"],
                    "resolution_method": "background_receipt_standard_bundle_layout",
                    "registry_join_method": join_method,
                    "evidence_primary_path": registry["destination_path"],
                    "evidence_primary_sha256": registry["candidate_sha256"].lower(),
                    "evidence_secondary_path": str(receipt_path),
                    "bundle_sha256_expected": str(receipt["bundle_sha256"]).lower(),
                    "root_archive_member": f"root/{root_basename}",
                    "root_basename": root_basename,
                    "archive_depth": 1,
                    "layout_rule": "standard_bundle_root_target_tag",
                    "ambiguity_count": ambiguity,
                }
            )

        else:
            canonical, join_method, ambiguity = select_path_first(
                candidate_path=candidate_path,
                candidate_sha256=candidate_sha256,
                path_index=signal_by_path,
                sha_index=signal_by_sha,
                label=f"signal member {member_index}",
            )
            require(not parse_bool(canonical["test_sealed"]), "signal join reached test data")
            require(
                canonical["signal_mode"] == manifest["process_or_mode"],
                f"signal member {member_index}: mode mismatch",
            )
            lower_path = Path(canonical["source_registry"]).resolve()
            if lower_path not in lower_registry_cache:
                require(
                    sha256_file(lower_path) == canonical["source_registry_sha256"],
                    f"{lower_path}: lower registry SHA-256 mismatch",
                )
                lower_registry_sha[lower_path] = canonical["source_registry_sha256"]
                lower_registry_cache[lower_path] = read_tsv(lower_path)[1]
            lower = one_checksum_row(
                lower_registry_cache[lower_path],
                "candidate_parquet_sha256",
                candidate_sha256,
                f"signal member {member_index} lower registry",
            )
            require(not parse_bool(lower["test_sealed"]), "lower registry join reached test data")
            require(
                lower["immutable_split"] == manifest["dataset_split"],
                f"signal member {member_index}: lower-registry split mismatch",
            )

            if manifest["process_or_mode"] == "vbf_hh4b":
                source_receipt_path = Path(lower["source_receipt"]).resolve()
                source_receipt = json_object(source_receipt_path)
                target_tag = lower["target_tag"]
                remote_bundle = lower["source_remote_bundle"]
                require(
                    source_receipt["remote_bundle"] == remote_bundle
                    and source_receipt["target_tag"] == target_tag,
                    f"VBF member {member_index}: source receipt mismatch",
                )
                expected_size = str(source_receipt.get("bundle_size_bytes", ""))
                if expected_size:
                    expected_bundle_sizes[remote_bundle].add(expected_size)
                root_basename = target_tag + "_delphes.root"
                common.update(
                    {
                        "resolution_method": "vbf_registry_standard_bundle_layout",
                        "registry_join_method": (
                            join_method + "+lower_registry_unique_candidate_sha256"
                        ),
                        "evidence_primary_path": candidate_path,
                        "evidence_primary_sha256": candidate_sha256,
                        "evidence_secondary_path": str(source_receipt_path),
                        "bundle_sha256_expected": lower["source_bundle_sha256"].lower(),
                        "bundle_adler32_expected": lower["source_bundle_adler32"].lower(),
                        "root_archive_member": f"root/{root_basename}",
                        "root_basename": root_basename,
                        "archive_depth": 1,
                        "layout_rule": "standard_bundle_root_target_tag",
                        "ambiguity_count": ambiguity,
                    }
                )

            elif manifest["process_or_mode"] == "ggf_hh4b":
                provenance = unique_nonempty_column_values(Path(candidate_path))
                provenance_files_read += 1
                receipt_path = Path(lower["receipt_path"]).resolve()
                receipt = json_object(receipt_path)
                remote_bundle = lower["remote_bundle"]
                require(
                    receipt["remote_bundle"] == remote_bundle,
                    f"ggF member {member_index}: receipt bundle mismatch",
                )
                expected_root_basename = (
                    f"HH4b_{lower['campaign']}_cluster_{receipt['cluster_id']}"
                    f"_shard_{lower['inner_shard']}_{lower['generated_events']}"
                    "_pythia8_delphes.root"
                )
                source_root_basename = PurePosixPath(provenance["source_root"]).name
                require(
                    source_root_basename == expected_root_basename,
                    f"ggF member {member_index}: source_root mapping mismatch",
                )
                require(
                    provenance["analysis_sample"] == "ggF_HH4b_SMnorm",
                    f"ggF member {member_index}: analysis_sample changed",
                )
                bundle_basename = PurePosixPath(remote_bundle).name
                require(
                    bundle_basename.endswith("_bundle.tar.gz"),
                    f"ggF member {member_index}: invalid bundle suffix",
                )
                root_container = (
                    "products/"
                    + bundle_basename[: -len("_bundle.tar.gz")]
                    + "_reconstruction.tar.gz"
                )
                root_member = f"root/{source_root_basename}"
                common.update(
                    {
                        "resolution_method": (
                            "candidate_source_root_nested_reconstruction_bundle"
                        ),
                        "registry_join_method": (
                            join_method + "+lower_registry_unique_candidate_sha256"
                        ),
                        "analysis_sample": provenance["analysis_sample"],
                        "source_root_index": provenance["source_root_index"],
                        "evidence_primary_path": candidate_path,
                        "evidence_primary_sha256": candidate_sha256,
                        "evidence_secondary_path": str(receipt_path),
                        "bundle_adler32_expected": lower["bundle_adler32"].lower(),
                        "root_container_member": root_container,
                        "root_archive_member": root_member,
                        "root_basename": source_root_basename,
                        "archive_depth": 2,
                        "layout_rule": (
                            "nested_reconstruction_bundle_source_root_basename"
                        ),
                        "ambiguity_count": ambiguity,
                    }
                )
            else:
                raise RuntimeError(
                    f"signal member {member_index}: unsupported mode "
                    f"{manifest['process_or_mode']}"
                )

        common["remote_bundle_path"] = remote_bundle
        common["remote_bundle_uri"] = remote_uri(config["eos_host"], remote_bundle)
        if common["root_container_member"]:
            common["source_locator"] = (
                remote_bundle
                + "::"
                + common["root_container_member"]
                + "::"
                + common["root_archive_member"]
            )
        else:
            common["source_locator"] = remote_bundle + "::" + common["root_archive_member"]
        member_rows.append(common)

    require(
        provenance_files_read == int(expected["candidate_provenance_files_read"]),
        "candidate provenance file count mismatch",
    )
    require(
        len(lower_registry_sha) == 2,
        "expected exactly two lower signal registries",
    )
    require(
        all(len(values) == 1 for values in expected_bundle_sizes.values()),
        "conflicting expected bundle sizes",
    )

    bundles_to_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in member_rows:
        bundles_to_members[row["remote_bundle_path"]].append(row)
    audit_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(config["remote_stat_workers"])) as executor:
        futures = {
            executor.submit(
                stat_remote_bundle,
                eos_host=config["eos_host"],
                remote_path=bundle,
                timeout_seconds=int(config["remote_stat_timeout_seconds"]),
                expected_size=(
                    next(iter(expected_bundle_sizes[bundle]))
                    if expected_bundle_sizes.get(bundle)
                    else ""
                ),
            ): bundle
            for bundle in sorted(bundles_to_members)
        }
        for future in as_completed(futures):
            result = future.result()
            members = bundles_to_members[result["remote_bundle_path"]]
            hashes = sorted(
                {
                    row["bundle_sha256_expected"]
                    for row in members
                    if row["bundle_sha256_expected"]
                }
            )
            adlers = sorted(
                {
                    row["bundle_adler32_expected"]
                    for row in members
                    if row["bundle_adler32_expected"]
                }
            )
            require(len(hashes) <= 1, "conflicting bundle SHA-256 expectations")
            require(len(adlers) <= 1, "conflicting bundle Adler-32 expectations")
            result.update(
                {
                    "member_count": len(members),
                    "member_indices_json": json.dumps(
                        sorted(int(row["member_index"]) for row in members)
                    ),
                    "dataset_splits_json": json.dumps(
                        sorted({row["dataset_split"] for row in members})
                    ),
                    "bundle_sha256_expected": hashes[0] if hashes else "",
                    "bundle_adler32_expected": adlers[0] if adlers else "",
                }
            )
            audit_rows.append(result)
    audit_rows.sort(key=lambda row: row["remote_bundle_path"])
    audit_by_bundle = {row["remote_bundle_path"]: row for row in audit_rows}

    for row in member_rows:
        audit = audit_by_bundle[row["remote_bundle_path"]]
        row["bundle_exists"] = audit["bundle_exists"]
        row["bundle_size_bytes"] = audit["reported_size_bytes"]
        row["bundle_stat_status"] = audit["bundle_stat_status"]

    resolution_counts = Counter(row["resolution_method"] for row in member_rows)
    expected_method_counts = {
        method: int(expected[method]) for method in RESOLUTION_METHODS
    }
    require(
        resolution_counts == Counter(expected_method_counts),
        f"resolution-method counts changed: {dict(resolution_counts)}",
    )

    locator_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in member_rows:
        locator_groups[row["source_locator"]].append(row)
    duplicate_rows: list[dict[str, Any]] = []
    cross_split_groups = 0
    for group_index, (locator, rows) in enumerate(
        sorted(locator_groups.items()), start=1
    ):
        if len(rows) < 2:
            continue
        splits = sorted({row["dataset_split"] for row in rows})
        cross_split = len(splits) > 1
        cross_split_groups += int(cross_split)
        duplicate_rows.append(
            {
                "duplicate_group_index": group_index,
                "source_locator": locator,
                "member_count": len(rows),
                "member_indices_json": json.dumps(
                    sorted(int(row["member_index"]) for row in rows)
                ),
                "dataset_splits_json": json.dumps(splits),
                "cross_split": cross_split,
                "duplicate_status": (
                    "cross_split_failure" if cross_split else "within_split_reported"
                ),
            }
        )

    unresolved_rows = [
        {
            "member_index": row["member_index"],
            "sample_class": row["sample_class"],
            "process_or_mode": row["process_or_mode"],
            "dataset_split": row["dataset_split"],
            "candidate_path": row["candidate_path"],
            "candidate_sha256": row["candidate_sha256"],
            "ambiguity_count": row["ambiguity_count"],
            "resolution_status": row["resolution_status"],
            "failure_reason": "",
        }
        for row in member_rows
        if row["resolution_status"] != "resolved" or int(row["ambiguity_count"]) != 0
    ]
    archive_validation = validate_archive_layouts(
        canary_path=canary_path,
        ggf_layout_path=ggf_layout_path,
        map_by_member={row["member_index"]: row for row in member_rows},
    )
    metadata_failures = sum(
        row["bundle_stat_status"] != "pass" for row in audit_rows
    )
    resolved_members = sum(row["resolution_status"] == "resolved" for row in member_rows)
    ambiguous_members = sum(int(row["ambiguity_count"]) > 0 for row in member_rows)

    require(resolved_members == int(expected["development_members"]), "unresolved members")
    require(not unresolved_rows, "unresolved or ambiguous members remain")
    require(metadata_failures == 0, "one or more remote metadata checks failed")
    require(cross_split_groups == 0, "source locator crosses train/validation split")

    summary = {
        "schema_version": 1,
        "status": "hh4b_3b_root_source_map_pass",
        "source_commit": source_commit,
        "configuration": str(config_path),
        "configuration_sha256": sha256_file(config_path),
        "preflight_configuration": str(preflight_path),
        "preflight_configuration_sha256": sha256_file(preflight_path),
        "development_manifest": str(manifest_path),
        "development_manifest_sha256": sha256_file(manifest_path),
        "development_members": len(member_rows),
        "resolved_members": resolved_members,
        "unresolved_members": len(unresolved_rows),
        "ambiguous_members": ambiguous_members,
        "train_members": split_counts["train"],
        "validation_members": split_counts["validation"],
        "signal_members": class_counts["signal"],
        "background_members": class_counts["background"],
        "zero_row_background_members": zero_background,
        "resolution_method_counts": dict(sorted(resolution_counts.items())),
        "candidate_provenance_files_read": provenance_files_read,
        "candidate_provenance_columns_read": list(PROVENANCE_COLUMNS),
        "candidate_physics_columns_read": 0,
        "unique_development_bundles": len(audit_rows),
        "remote_metadata_checks_attempted": len(audit_rows),
        "remote_metadata_checks_failed": metadata_failures,
        "duplicate_source_locator_groups": len(duplicate_rows),
        "cross_split_source_locator_groups": cross_split_groups,
        "archive_layout_records_validated": len(archive_validation),
        "root_files_extracted": 0,
        "root_files_opened": 0,
        "root_event_arrays_read": 0,
        "test_members_considered": 0,
        "event_generation_performed": 0,
        "physics_yields_calculated": 0,
        "next_gate": "audit_required_delphes_branches_for_3b_reconstruction",
    }

    require(
        summary["test_members_considered"] == int(expected["test_members_considered"]),
        "test-member safety invariant changed",
    )
    require(
        summary["candidate_physics_columns_read"]
        == int(expected["candidate_physics_columns_read"]),
        "candidate physics-column safety invariant changed",
    )

    output_tmp.mkdir(parents=True, exist_ok=False)
    checkpoint_tmp.mkdir(parents=True, exist_ok=False)
    write_tsv(output_tmp / "member_level_root_source_map.tsv", member_rows, MAP_FIELDS)
    write_tsv(
        output_tmp / "resolution_method_summary.tsv",
        [
            {
                "resolution_method": method,
                "resolved_members": resolution_counts[method],
                "expected_members": expected_method_counts[method],
                "status": "pass",
            }
            for method in RESOLUTION_METHODS
        ],
        ["resolution_method", "resolved_members", "expected_members", "status"],
    )
    write_tsv(
        output_tmp / "source_access_audit.tsv",
        audit_rows,
        [
            "remote_bundle_path",
            "remote_bundle_uri",
            "member_count",
            "member_indices_json",
            "dataset_splits_json",
            "bundle_sha256_expected",
            "bundle_adler32_expected",
            "expected_size_bytes",
            "stat_return_code",
            "bundle_exists",
            "reported_size_bytes",
            "size_matches_expected",
            "bundle_stat_status",
            "timeout_or_failure_text",
        ],
    )
    write_tsv(
        output_tmp / "unresolved_or_ambiguous_members.tsv",
        unresolved_rows,
        [
            "member_index",
            "sample_class",
            "process_or_mode",
            "dataset_split",
            "candidate_path",
            "candidate_sha256",
            "ambiguity_count",
            "resolution_status",
            "failure_reason",
        ],
    )
    write_tsv(
        output_tmp / "duplicate_source_locators.tsv",
        duplicate_rows,
        [
            "duplicate_group_index",
            "source_locator",
            "member_count",
            "member_indices_json",
            "dataset_splits_json",
            "cross_split",
            "duplicate_status",
        ],
    )
    write_tsv(
        output_tmp / "archive_layout_validation.tsv",
        archive_validation,
        [
            "evidence_source",
            "evidence_kind",
            "member_index",
            "remote_bundle_path",
            "root_container_member",
            "root_archive_member",
            "observed_layout_value",
            "layout_rule",
            "validation_status",
        ],
    )
    (output_tmp / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for name in OUTPUT_FILES:
        shutil.copy2(output_tmp / name, checkpoint_tmp / name)
    (checkpoint_tmp / "README.md").write_text(
        checkpoint_readme(summary),
        encoding="utf-8",
    )
    checkpoint = {
        "schema_version": 1,
        "classification": "paper_quality_root_source_provenance_checkpoint",
        "status": summary["status"],
        "source_commit": source_commit,
        "development_members": len(member_rows),
        "resolved_members": resolved_members,
        "remote_metadata_checks_failed": metadata_failures,
        "cross_split_source_locator_groups": cross_split_groups,
        "root_files_opened": 0,
        "test_members_considered": 0,
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

    output_tmp.rename(output_dir)
    checkpoint_tmp.rename(checkpoint_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
