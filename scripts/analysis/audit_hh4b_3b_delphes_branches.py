#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import tokenize
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import uproot


OUTPUT_FILES = (
    "branch_audit_representatives.tsv",
    "member_to_branch_schema_stratum.tsv",
    "root_tree_branch_audit.tsv",
    "required_branch_summary.tsv",
    "observed_branch_type_summary.tsv",
    "incompatible_or_missing_branches.tsv",
    "extraction_access_audit.tsv",
    "summary.json",
)

PRIMARY_STRATUM_FIELDS = (
    "resolution_method",
    "layout_rule",
    "bundle_campaign",
    "process_or_mode",
)

REPRESENTATIVE_FIELDS = [
    "representative_index",
    "member_index",
    "sample_class",
    "process_or_mode",
    "dataset_split",
    "candidate_rows",
    "zero_row_4b_candidate_file",
    "resolution_method",
    "layout_rule",
    "bundle_campaign",
    "schema_stratum_key",
    "representative_selection_reasons",
    "remote_bundle_path",
    "root_container_member",
    "root_archive_member",
    "root_basename",
    "archive_depth",
]

BRANCH_AUDIT_FIELDS = [
    "member_index",
    "sample_class",
    "process_or_mode",
    "dataset_split",
    "candidate_rows",
    "zero_row_4b_candidate_file",
    "resolution_method",
    "layout_rule",
    "bundle_campaign",
    "schema_stratum_key",
    "representative_selection_reasons",
    "remote_bundle_path",
    "root_container_member",
    "root_archive_member",
    "root_basename",
    "archive_depth",
    "local_bundle_size_bytes",
    "local_root_size_bytes",
    "local_root_sha256",
    "root_open_status",
    "top_level_keys_json",
    "delphes_tree_present",
    "delphes_tree_cycle",
    "delphes_num_entries",
    "expected_generated_events",
    "entry_count_match",
    "branch_count",
    "required_branches_present_json",
    "missing_required_branches_json",
    "required_branch_typenames_json",
    "required_branch_interpretations_json",
    "incompatible_required_branches_json",
    "audit_status",
    "failure_reason",
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
    "root_file_deleted",
    "nested_archive_deleted",
    "outer_bundle_deleted",
    "temporary_files_remaining",
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


def json_object(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    require(isinstance(document, dict), f"{path}: JSON is not an object")
    return document


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"cannot parse Boolean value {value!r}")


def static_safety_review(script_path: Path) -> dict[str, Any]:
    source = script_path.read_text(encoding="utf-8", errors="strict")
    parsed = ast.parse(source)
    forbidden_attributes = {
        "array" + "s",
        "array",
        "iter" + "ate",
    }
    violations: list[str] = []
    calls_reviewed = 0
    for node in ast.walk(parsed):
        if not isinstance(node, ast.Call):
            continue
        calls_reviewed += 1
        if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_attributes:
            violations.append(
                f"line {node.lineno}: forbidden call attribute {node.func.attr}"
            )

    stripped_tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            stripped_tokens.append(
                tokenize.TokenInfo(
                    token.type,
                    "",
                    token.start,
                    token.end,
                    token.line,
                )
            )
        else:
            stripped_tokens.append(token)
    executable_text = tokenize.untokenize(stripped_tokens)
    textual_patterns = (
        "." + "arrays(",
        "." + "array(",
        "." + "iterate(",
        "uproot" + ".iterate(",
    )
    for pattern in textual_patterns:
        if pattern in executable_text:
            violations.append(f"forbidden executable pattern {pattern}")

    require(
        not violations,
        "static array-read safety review failed: " + "; ".join(violations),
    )
    return {
        "static_safety_review_status": "pass",
        "static_safety_calls_reviewed": calls_reviewed,
        "static_safety_forbidden_calls": 0,
    }


def bundle_campaign(remote_bundle_path: str) -> str:
    return PurePosixPath(remote_bundle_path).parent.name


def schema_stratum_tuple(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(row[field]) for field in PRIMARY_STRATUM_FIELDS)  # type: ignore[return-value]


def schema_stratum_key(row: dict[str, Any]) -> str:
    return json.dumps(
        list(schema_stratum_tuple(row)),
        separators=(",", ":"),
        ensure_ascii=True,
    )


def add_selection_reason(
    *,
    selected: dict[str, set[str]],
    candidates: list[dict[str, Any]],
    reason: str,
) -> None:
    require(bool(candidates), f"no candidate available for selection reason {reason}")
    already_selected = [
        row for row in candidates if row["member_index"] in selected
    ]
    chosen = min(
        already_selected or candidates,
        key=lambda row: int(row["member_index"]),
    )
    selected.setdefault(chosen["member_index"], set()).add(reason)


def select_representatives(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    strata: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[schema_stratum_tuple(row)].append(row)

    selected: dict[str, set[str]] = {}
    for key in sorted(strata):
        chosen = min(strata[key], key=lambda row: int(row["member_index"]))
        selected.setdefault(chosen["member_index"], set()).add(
            "primary_schema_stratum"
        )

    layouts = sorted({row["layout_rule"] for row in rows})
    for layout in layouts:
        for split in ("train", "validation"):
            candidates = [
                row
                for row in rows
                if row["layout_rule"] == layout and row["dataset_split"] == split
            ]
            add_selection_reason(
                selected=selected,
                candidates=candidates,
                reason=f"source_layout_{split}_coverage",
            )

    zero_campaigns = sorted(
        {
            row["bundle_campaign"]
            for row in rows
            if row["sample_class"] == "background"
            and int(row["candidate_rows"]) == 0
        }
    )
    for campaign in zero_campaigns:
        candidates = [
            row
            for row in rows
            if row["sample_class"] == "background"
            and int(row["candidate_rows"]) == 0
            and row["bundle_campaign"] == campaign
        ]
        add_selection_reason(
            selected=selected,
            candidates=candidates,
            reason="zero_row_background_campaign_coverage",
        )

    ggf_campaigns = sorted(
        {
            row["bundle_campaign"]
            for row in rows
            if row["process_or_mode"] == "ggf_hh4b"
        }
    )
    require(len(ggf_campaigns) == 2, "expected exactly two ggF source campaigns")
    for campaign in ggf_campaigns:
        add_selection_reason(
            selected=selected,
            candidates=[
                row
                for row in rows
                if row["process_or_mode"] == "ggf_hh4b"
                and row["bundle_campaign"] == campaign
            ],
            reason="ggf_source_campaign_coverage",
        )

    add_selection_reason(
        selected=selected,
        candidates=[row for row in rows if row["process_or_mode"] == "vbf_hh4b"],
        reason="vbf_coverage",
    )

    background_modes = sorted(
        {
            row["process_or_mode"]
            for row in rows
            if row["sample_class"] == "background"
        }
    )
    for mode in background_modes:
        add_selection_reason(
            selected=selected,
            candidates=[
                row
                for row in rows
                if row["sample_class"] == "background"
                and row["process_or_mode"] == mode
            ],
            reason="background_process_coverage",
        )

    representative_rows: list[dict[str, Any]] = []
    rows_by_member = {row["member_index"]: row for row in rows}
    for representative_index, member_index in enumerate(
        sorted(selected, key=int), start=1
    ):
        row = dict(rows_by_member[member_index])
        row.update(
            {
                "representative_index": representative_index,
                "zero_row_4b_candidate_file": (
                    row["sample_class"] == "background"
                    and int(row["candidate_rows"]) == 0
                ),
                "schema_stratum_key": schema_stratum_key(row),
                "representative_selection_reasons": json.dumps(
                    sorted(selected[member_index]),
                    separators=(",", ":"),
                ),
            }
        )
        representative_rows.append(row)

    representative_by_stratum: dict[str, dict[str, Any]] = {}
    for row in representative_rows:
        key = row["schema_stratum_key"]
        current = representative_by_stratum.get(key)
        if current is None or int(row["member_index"]) < int(current["member_index"]):
            representative_by_stratum[key] = row
    require(
        len(representative_by_stratum) == len(strata),
        "one or more primary strata lacks a representative",
    )
    return representative_rows, representative_by_stratum


def canonical_archive_name(value: str) -> str:
    raw = value.strip()
    require(bool(raw), "empty archive member path")
    path = PurePosixPath(raw)
    require(not path.is_absolute(), f"absolute archive path rejected: {raw}")
    require(".." not in path.parts, f"parent traversal archive path rejected: {raw}")
    parts = [part for part in path.parts if part not in ("", ".")]
    if not parts:
        return "."
    return "/".join(parts)


def extract_exact_member(
    *,
    archive_path: Path,
    requested_member: str,
    destination_path: Path,
) -> tuple[int, int, str]:
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
                f"has {len(matches)} exact canonical matches"
            ),
        )
        match = matches[0]
        require(match.isfile(), f"{match.name}: requested archive member is not a file")
        source = archive.extractfile(match)
        require(source is not None, f"{match.name}: archive member cannot be opened")
        with source, destination_path.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    return member_count, len(matches), matches[0].name


def branch_compatibility(
    *,
    typename: str,
    interpretation: str,
    requirement: str,
) -> tuple[bool, str]:
    lowered_type = typename.lower().replace(" ", "")
    lowered_interpretation = interpretation.lower().replace(" ", "")
    jagged = (
        "[]" in lowered_type
        or "asjagged" in lowered_interpretation
        or "jagged" in lowered_interpretation
    )
    floating = "float" in lowered_type or "double" in lowered_type
    boolean = "bool" in lowered_type
    integral = (
        not floating
        and not boolean
        and any(
            token in lowered_type
            for token in (
                "int",
                "char",
                "short",
                "long",
                "uint",
                "uchar",
                "ushort",
                "ulong",
            )
        )
    )
    numeric = floating or integral

    if not jagged:
        return False, "scalar_or_nonjagged"
    if requirement == "floating_point_jagged":
        return floating, "compatible" if floating else "not_floating_point"
    if requirement == "numeric_or_boolean_jagged":
        compatible = numeric or boolean
        return compatible, "compatible" if compatible else "not_numeric_or_boolean"
    if requirement == "integral_numeric_jagged":
        return integral, "compatible" if integral else "not_integral_numeric"
    raise RuntimeError(f"unknown branch compatibility requirement {requirement!r}")


def default_branch_audit_row(representative: dict[str, Any]) -> dict[str, Any]:
    return {
        **{field: representative.get(field, "") for field in REPRESENTATIVE_FIELDS},
        "local_bundle_size_bytes": "",
        "local_root_size_bytes": "",
        "local_root_sha256": "",
        "root_open_status": "not_attempted",
        "top_level_keys_json": "[]",
        "delphes_tree_present": False,
        "delphes_tree_cycle": "",
        "delphes_num_entries": "",
        "expected_generated_events": representative["generated_events"],
        "entry_count_match": False,
        "branch_count": "",
        "required_branches_present_json": "[]",
        "missing_required_branches_json": "[]",
        "required_branch_typenames_json": "{}",
        "required_branch_interpretations_json": "{}",
        "incompatible_required_branches_json": "[]",
        "audit_status": "fail",
        "failure_reason": "none",
    }


def audit_root_metadata(
    *,
    root_path: Path,
    representative: dict[str, Any],
    required_tree: str,
    required_branches: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = default_branch_audit_row(representative)
    row["local_root_size_bytes"] = root_path.stat().st_size
    row["local_root_sha256"] = sha256_file(root_path)
    branch_checks: list[dict[str, Any]] = []
    try:
        with uproot.open(root_path) as root_file:
            row["root_open_status"] = "pass"
            top_level_keys = [str(value) for value in root_file.keys()]
            row["top_level_keys_json"] = json.dumps(top_level_keys)
            cycles = []
            for key in top_level_keys:
                name, separator, cycle_text = key.rpartition(";")
                if name == required_tree and separator and cycle_text.isdigit():
                    cycles.append(int(cycle_text))
            require(bool(cycles), f"required tree {required_tree!r} is missing")
            cycle = max(cycles)
            tree = root_file[f"{required_tree};{cycle}"]
            row["delphes_tree_present"] = True
            row["delphes_tree_cycle"] = cycle
            row["delphes_num_entries"] = int(tree.num_entries)
            require(int(tree.num_entries) > 0, "Delphes tree has no entries")
            expected_entries = int(representative["generated_events"])
            row["entry_count_match"] = int(tree.num_entries) == expected_entries
            require(
                row["entry_count_match"],
                (
                    f"entry-count mismatch: expected {expected_entries}, "
                    f"observed {tree.num_entries}"
                ),
            )

            tree_keys = [str(value) for value in tree.keys()]
            typenames = {str(key): str(value) for key, value in tree.typenames().items()}
            row["branch_count"] = len(tree_keys)
            present: list[str] = []
            missing: list[str] = []
            observed_typenames: dict[str, str] = {}
            observed_interpretations: dict[str, str] = {}
            incompatible: list[str] = []
            for branch_name, requirement in required_branches.items():
                try:
                    branch = tree[branch_name]
                except KeyError:
                    missing.append(branch_name)
                    branch_checks.append(
                        {
                            "member_index": representative["member_index"],
                            "branch_name": branch_name,
                            "required_compatibility": requirement,
                            "branch_present": False,
                            "observed_typename": "",
                            "observed_interpretation": "",
                            "branch_compatible": False,
                            "compatibility_reason": "missing",
                        }
                    )
                    continue
                present.append(branch_name)
                typename = str(branch.typename)
                interpretation = str(branch.interpretation)
                typename_metadata = {
                    value
                    for key, value in typenames.items()
                    if key == branch_name or key.endswith("/" + branch_name)
                }
                if typename_metadata:
                    require(
                        typename in typename_metadata,
                        f"{branch_name}: typename metadata disagrees",
                    )
                compatible, reason = branch_compatibility(
                    typename=typename,
                    interpretation=interpretation,
                    requirement=requirement,
                )
                observed_typenames[branch_name] = typename
                observed_interpretations[branch_name] = interpretation
                if not compatible:
                    incompatible.append(branch_name)
                branch_checks.append(
                    {
                        "member_index": representative["member_index"],
                        "branch_name": branch_name,
                        "required_compatibility": requirement,
                        "branch_present": True,
                        "observed_typename": typename,
                        "observed_interpretation": interpretation,
                        "branch_compatible": compatible,
                        "compatibility_reason": reason,
                    }
                )

            row["required_branches_present_json"] = json.dumps(present)
            row["missing_required_branches_json"] = json.dumps(missing)
            row["required_branch_typenames_json"] = json.dumps(
                observed_typenames, sort_keys=True
            )
            row["required_branch_interpretations_json"] = json.dumps(
                observed_interpretations, sort_keys=True
            )
            row["incompatible_required_branches_json"] = json.dumps(incompatible)
            require(not missing, "missing required branches: " + ", ".join(missing))
            require(
                not incompatible,
                "incompatible required branches: " + ", ".join(incompatible),
            )
            row["audit_status"] = "pass"
    except Exception as error:
        row["failure_reason"] = str(error)
        if row["root_open_status"] == "not_attempted":
            row["root_open_status"] = "fail"
    return row, branch_checks


def audit_representative(
    *,
    representative: dict[str, Any],
    representative_number: int,
    representative_total: int,
    temporary_root: Path,
    xrdcp_timeout_seconds: int,
    required_tree: str,
    required_branches: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    print(
        (
            f"[{representative_number}/{representative_total}] "
            f"member={representative['member_index']} "
            f"campaign={representative['bundle_campaign']}"
        ),
        flush=True,
    )
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"hh4b_3b_branch_audit_{representative['member_index']}_",
            dir=temporary_root,
        )
    )
    outer_path = temp_dir / "outer_bundle.tar.gz"
    nested_path = temp_dir / "nested_reconstruction.tar.gz"
    root_path = temp_dir / representative["root_basename"]
    branch_row = default_branch_audit_row(representative)
    branch_checks: list[dict[str, Any]] = []
    access: dict[str, Any] = {
        "member_index": representative["member_index"],
        "remote_bundle_path": representative["remote_bundle_path"],
        "remote_bundle_uri": representative["remote_bundle_uri"],
        "expected_bundle_size_bytes": representative["bundle_size_bytes"],
        "xrdcp_attempted": True,
        "xrdcp_return_code": "",
        "xrdcp_status": "not_completed",
        "xrdcp_failure_text": "",
        "local_bundle_size_bytes": "",
        "bundle_size_match": False,
        "outer_archive_members_inspected": 0,
        "outer_requested_member": (
            representative["root_container_member"]
            or representative["root_archive_member"]
        ),
        "outer_exact_matches": 0,
        "nested_archive_used": bool(representative["root_container_member"]),
        "nested_archive_members_inspected": 0,
        "nested_requested_member": (
            representative["root_archive_member"]
            if representative["root_container_member"]
            else ""
        ),
        "nested_exact_matches": 0,
        "root_file_extracted": False,
        "root_file_deleted": False,
        "nested_archive_deleted": False,
        "outer_bundle_deleted": False,
        "temporary_files_remaining": "",
        "extraction_status": "fail",
    }
    failure = ""
    try:
        completed = subprocess.run(
            [
                "xrdcp",
                "--nopbar",
                "--force",
                representative["remote_bundle_uri"],
                str(outer_path),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=xrdcp_timeout_seconds,
        )
        access["xrdcp_return_code"] = completed.returncode
        if completed.returncode != 0:
            failure_text = (completed.stderr or completed.stdout).strip()
            access["xrdcp_failure_text"] = re.sub(r"\s+", " ", failure_text)
            raise RuntimeError(f"xrdcp failed with return code {completed.returncode}")
        access["xrdcp_status"] = "pass"
        local_bundle_size = outer_path.stat().st_size
        access["local_bundle_size_bytes"] = local_bundle_size
        branch_row["local_bundle_size_bytes"] = local_bundle_size
        access["bundle_size_match"] = (
            local_bundle_size == int(representative["bundle_size_bytes"])
        )
        require(access["bundle_size_match"], "downloaded bundle size mismatch")

        if representative["root_container_member"]:
            count, matches, _ = extract_exact_member(
                archive_path=outer_path,
                requested_member=representative["root_container_member"],
                destination_path=nested_path,
            )
            access["outer_archive_members_inspected"] = count
            access["outer_exact_matches"] = matches
            count, matches, _ = extract_exact_member(
                archive_path=nested_path,
                requested_member=representative["root_archive_member"],
                destination_path=root_path,
            )
            access["nested_archive_members_inspected"] = count
            access["nested_exact_matches"] = matches
        else:
            count, matches, _ = extract_exact_member(
                archive_path=outer_path,
                requested_member=representative["root_archive_member"],
                destination_path=root_path,
            )
            access["outer_archive_members_inspected"] = count
            access["outer_exact_matches"] = matches

        require(root_path.is_file(), "requested ROOT file was not extracted")
        access["root_file_extracted"] = True
        branch_row, branch_checks = audit_root_metadata(
            root_path=root_path,
            representative=representative,
            required_tree=required_tree,
            required_branches=required_branches,
        )
        branch_row["local_bundle_size_bytes"] = local_bundle_size
        require(branch_row["audit_status"] == "pass", branch_row["failure_reason"])
        access["extraction_status"] = "pass"
    except subprocess.TimeoutExpired:
        failure = f"xrdcp timed out after {xrdcp_timeout_seconds} seconds"
        access["xrdcp_return_code"] = "timeout"
        access["xrdcp_failure_text"] = failure
    except Exception as error:
        failure = str(error)
    finally:
        if failure and branch_row["failure_reason"] == "none":
            branch_row["failure_reason"] = failure
        for path, field in (
            (root_path, "root_file_deleted"),
            (nested_path, "nested_archive_deleted"),
            (outer_path, "outer_bundle_deleted"),
        ):
            if path.exists():
                path.unlink()
            access[field] = not path.exists()
        remaining = [path for path in temp_dir.rglob("*") if path.is_file()]
        access["temporary_files_remaining"] = len(remaining)
        shutil.rmtree(temp_dir)
        require(not temp_dir.exists(), f"temporary directory remains: {temp_dir}")

    print(
        (
            f"[{representative_number}/{representative_total}] "
            f"member={representative['member_index']} "
            f"status={branch_row['audit_status']} "
            f"cleanup_remaining={access['temporary_files_remaining']} "
            f"failure={branch_row['failure_reason']!r}"
        ),
        flush=True,
    )
    return branch_row, access, branch_checks


def checkpoint_readme(summary: dict[str, Any]) -> str:
    return f"""# HH4b 3b Delphes branch audit

## Purpose

This checkpoint establishes that the Delphes ROOT inputs needed for a future
three-b-tag control-candidate builder expose the six branches used by the
frozen four-b-tag builder. It is a representative ROOT metadata audit only.

## Result

- Status: `{summary['status']}`
- Development members assigned to strata: {summary['members_assigned_to_schema_strata']}
- Primary schema strata: {summary['schema_strata']}
- Representatives audited: {summary['representatives_audited']}
- Representatives failed: {summary['representatives_failed']}
- Required branch checks: {summary['required_branch_checks']}
- Missing checks: {summary['missing_required_branch_checks']}
- Incompatible checks: {summary['incompatible_required_branch_checks']}
- Entry-count mismatches: {summary['entry_count_mismatches']}
- Temporary files remaining: {summary['temporary_files_remaining']}

Representatives were selected deterministically from the complete committed
source map. Outer bundles were processed sequentially. Only the requested ROOT
member, or the requested nested archive and ROOT member for ggF, was extracted.
Temporary materializations were deleted after each metadata audit.

## Safety record

- ROOT event arrays read: {summary['root_event_arrays_read']}
- Candidate files read: {summary['candidate_files_read']}
- Test members considered: {summary['test_members_considered']}
- Reconstruction performed: {summary['reconstruction_performed']}
- Physics yields calculated: {summary['physics_yields_calculated']}
- Static array-read safety review: `{summary['static_safety_review_status']}`

## Reproducibility

- Source commit: `{summary['source_commit']}`
- Source-map SHA-256: `{summary['source_map_sha256']}`
- Configuration SHA-256: `{summary['configuration_sha256']}`

## Next gate

`{summary['next_gate']}`
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit required Delphes branch metadata for deterministic "
            "representatives of the HH4b 3b-control source strata."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    script_path = Path(__file__).resolve()
    safety = static_safety_review(script_path)
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
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    require(head == source_commit, f"HEAD {head} does not match {source_commit}")

    source_map_path = resolve(repo, config["source_map_path"])
    source_map_checkpoint_dir = resolve(repo, config["source_map_checkpoint_dir"])
    require(
        sha256_file(source_map_path) == config["source_map_sha256"],
        "source-map SHA-256 mismatch",
    )
    checkpoint_verification = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=source_map_checkpoint_dir,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(
        checkpoint_verification.returncode == 0,
        "source-map checkpoint SHA256SUMS verification failed",
    )

    output_dir = resolve(repo, config["output_dir"])
    checkpoint_dir = resolve(repo, config["checkpoint_dir"])
    output_tmp = output_dir.with_name(output_dir.name + "_incomplete")
    checkpoint_tmp = checkpoint_dir.with_name(checkpoint_dir.name + "_incomplete")
    for path in (output_dir, output_tmp, checkpoint_dir, checkpoint_tmp):
        require(not path.exists(), f"refusing to overwrite {path}")

    temporary_root = resolve(repo, config["temporary_root"])
    require(temporary_root.is_dir(), f"temporary root is missing: {temporary_root}")
    required_tree = str(config["required_tree"])
    required_branches = {
        str(key): str(value) for key, value in config["required_branches"].items()
    }
    expected = config["expected"]
    require(
        len(required_branches) == int(expected["required_branches"]),
        "required-branch count mismatch",
    )

    source_fields, source_rows = read_tsv(source_map_path)
    required_source_fields = {
        "member_index",
        "sample_class",
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
        "archive_depth",
        "resolution_status",
        "ambiguity_count",
    }
    require(
        required_source_fields <= set(source_fields),
        "source map is missing required fields",
    )
    require(
        len(source_rows) == int(expected["development_members"]),
        "development-member count mismatch",
    )
    require(
        all(
            row["dataset_split"] in set(config["development_splits"])
            and row["resolution_status"] == "resolved"
            and int(row["ambiguity_count"]) == 0
            for row in source_rows
        ),
        "source map contains a non-development or unresolved member",
    )
    require(
        len({row["member_index"] for row in source_rows}) == len(source_rows),
        "source map contains duplicate member indices",
    )

    rows: list[dict[str, Any]] = []
    for source in source_rows:
        row = dict(source)
        row["bundle_campaign"] = bundle_campaign(source["remote_bundle_path"])
        row["schema_stratum_key"] = schema_stratum_key(row)
        rows.append(row)

    class_counts = Counter(row["sample_class"] for row in rows)
    split_counts = Counter(row["dataset_split"] for row in rows)
    zero_background_members = sum(
        row["sample_class"] == "background" and int(row["candidate_rows"]) == 0
        for row in rows
    )
    require(split_counts["train"] == int(expected["train_members"]), "train mismatch")
    require(
        split_counts["validation"] == int(expected["validation_members"]),
        "validation mismatch",
    )
    require(class_counts["signal"] == int(expected["signal_members"]), "signal mismatch")
    require(
        class_counts["background"] == int(expected["background_members"]),
        "background mismatch",
    )
    require(
        zero_background_members == int(expected["zero_row_background_members"]),
        "zero-row background mismatch",
    )

    representatives, representative_by_stratum = select_representatives(rows)
    schema_strata = {row["schema_stratum_key"] for row in rows}
    strata_without_representative = len(schema_strata - set(representative_by_stratum))
    member_to_stratum_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["member_index"])):
        representative = representative_by_stratum.get(row["schema_stratum_key"])
        member_to_stratum_rows.append(
            {
                "member_index": row["member_index"],
                "sample_class": row["sample_class"],
                "process_or_mode": row["process_or_mode"],
                "dataset_split": row["dataset_split"],
                "candidate_rows": row["candidate_rows"],
                "resolution_method": row["resolution_method"],
                "layout_rule": row["layout_rule"],
                "bundle_campaign": row["bundle_campaign"],
                "schema_stratum_key": row["schema_stratum_key"],
                "representative_member_index": (
                    representative["member_index"] if representative else ""
                ),
                "schema_stratum_assigned": representative is not None,
                "representative_audit_status": "",
            }
        )

    branch_audit_rows: list[dict[str, Any]] = []
    extraction_audit_rows: list[dict[str, Any]] = []
    all_branch_checks: list[dict[str, Any]] = []
    for index, representative in enumerate(representatives, start=1):
        branch_row, extraction_row, branch_checks = audit_representative(
            representative=representative,
            representative_number=index,
            representative_total=len(representatives),
            temporary_root=temporary_root,
            xrdcp_timeout_seconds=int(config["xrdcp_timeout_seconds"]),
            required_tree=required_tree,
            required_branches=required_branches,
        )
        branch_audit_rows.append(branch_row)
        extraction_audit_rows.append(extraction_row)
        all_branch_checks.extend(branch_checks)

    audit_by_member = {
        row["member_index"]: row["audit_status"] for row in branch_audit_rows
    }
    for row in member_to_stratum_rows:
        row["representative_audit_status"] = audit_by_member.get(
            row["representative_member_index"], "missing"
        )

    issue_rows: list[dict[str, Any]] = []
    for check in all_branch_checks:
        if parse_bool(check["branch_present"]) and parse_bool(
            check["branch_compatible"]
        ):
            continue
        representative = next(
            row
            for row in representatives
            if row["member_index"] == check["member_index"]
        )
        issue_rows.append(
            {
                "member_index": check["member_index"],
                "process_or_mode": representative["process_or_mode"],
                "bundle_campaign": representative["bundle_campaign"],
                "schema_stratum_key": representative["schema_stratum_key"],
                "branch_name": check["branch_name"],
                "required_compatibility": check["required_compatibility"],
                "branch_present": check["branch_present"],
                "observed_typename": check["observed_typename"],
                "observed_interpretation": check["observed_interpretation"],
                "branch_compatible": check["branch_compatible"],
                "failure_reason": check["compatibility_reason"],
            }
        )

    branch_summary_rows: list[dict[str, Any]] = []
    for branch_name, requirement in required_branches.items():
        checks = [
            check for check in all_branch_checks if check["branch_name"] == branch_name
        ]
        present = sum(parse_bool(check["branch_present"]) for check in checks)
        compatible = sum(parse_bool(check["branch_compatible"]) for check in checks)
        branch_summary_rows.append(
            {
                "branch_name": branch_name,
                "required_compatibility": requirement,
                "representatives_checked": len(checks),
                "present_checks": present,
                "missing_checks": len(checks) - present,
                "compatible_checks": compatible,
                "incompatible_checks": present - compatible,
                "observed_typenames_json": json.dumps(
                    sorted(
                        {
                            check["observed_typename"]
                            for check in checks
                            if check["observed_typename"]
                        }
                    )
                ),
                "observed_interpretations_json": json.dumps(
                    sorted(
                        {
                            check["observed_interpretation"]
                            for check in checks
                            if check["observed_interpretation"]
                        }
                    )
                ),
                "status": (
                    "pass"
                    if len(checks) == len(representatives)
                    and present == compatible == len(checks)
                    else "fail"
                ),
            }
        )

    observed_groups: dict[
        tuple[str, str, str, bool], list[dict[str, Any]]
    ] = defaultdict(list)
    representatives_by_member = {
        row["member_index"]: row for row in representatives
    }
    for check in all_branch_checks:
        if not parse_bool(check["branch_present"]):
            continue
        key = (
            check["branch_name"],
            check["observed_typename"],
            check["observed_interpretation"],
            parse_bool(check["branch_compatible"]),
        )
        observed_groups[key].append(check)
    observed_type_rows: list[dict[str, Any]] = []
    for key, checks in sorted(observed_groups.items()):
        branch_name, typename, interpretation, compatible = key
        representatives_for_group = [
            representatives_by_member[check["member_index"]] for check in checks
        ]
        observed_type_rows.append(
            {
                "branch_name": branch_name,
                "observed_typename": typename,
                "observed_interpretation": interpretation,
                "branch_compatible": compatible,
                "representative_count": len(checks),
                "member_indices_json": json.dumps(
                    sorted(int(check["member_index"]) for check in checks)
                ),
                "processes_or_modes_json": json.dumps(
                    sorted(
                        {
                            row["process_or_mode"]
                            for row in representatives_for_group
                        }
                    )
                ),
                "bundle_campaigns_json": json.dumps(
                    sorted(
                        {
                            row["bundle_campaign"]
                            for row in representatives_for_group
                        }
                    )
                ),
                "status": "pass" if compatible else "fail",
            }
        )

    representatives_failed = sum(
        row["audit_status"] != "pass" for row in branch_audit_rows
    )
    members_without_stratum = sum(
        not parse_bool(row["schema_stratum_assigned"])
        for row in member_to_stratum_rows
    )
    root_open_failures = sum(
        row["root_open_status"] != "pass" for row in branch_audit_rows
    )
    trees_missing = sum(
        not parse_bool(row["delphes_tree_present"]) for row in branch_audit_rows
    )
    entry_mismatches = sum(
        not parse_bool(row["entry_count_match"]) for row in branch_audit_rows
    )
    missing_checks = sum(
        not parse_bool(check["branch_present"]) for check in all_branch_checks
    )
    incompatible_checks = sum(
        parse_bool(check["branch_present"])
        and not parse_bool(check["branch_compatible"])
        for check in all_branch_checks
    )
    temporary_files_remaining = sum(
        int(row["temporary_files_remaining"]) for row in extraction_audit_rows
    )
    direct_representatives = sum(
        int(row["archive_depth"]) == 1 for row in representatives
    )
    nested_representatives = sum(
        int(row["archive_depth"]) == 2 for row in representatives
    )

    gate_errors: list[str] = []

    def gate(condition: bool, message: str) -> None:
        if not condition:
            gate_errors.append(message)

    gate(len(branch_audit_rows) == len(representatives), "representative audit count")
    gate(representatives_failed == 0, "representatives failed")
    gate(len(member_to_stratum_rows) == int(expected["development_members"]), "member assignments")
    gate(members_without_stratum == 0, "members without schema stratum")
    gate(strata_without_representative == 0, "strata without representative")
    gate(trees_missing == 0, "Delphes trees missing")
    gate(root_open_failures == 0, "ROOT open failures")
    gate(entry_mismatches == 0, "entry-count mismatches")
    gate(
        len(all_branch_checks) == len(representatives) * len(required_branches),
        "required branch-check count",
    )
    gate(missing_checks == 0, "missing required branch checks")
    gate(incompatible_checks == 0, "incompatible required branch checks")
    gate(temporary_files_remaining == 0, "temporary files remain")
    gate(
        all(row["extraction_status"] == "pass" for row in extraction_audit_rows),
        "extraction failures",
    )

    summary = {
        "schema_version": 1,
        "status": (
            "hh4b_3b_delphes_branch_audit_pass"
            if not gate_errors
            else "hh4b_3b_delphes_branch_audit_fail"
        ),
        "source_commit": source_commit,
        "configuration": str(config_path),
        "configuration_sha256": sha256_file(config_path),
        "audit_script": str(script_path),
        "audit_script_sha256": sha256_file(script_path),
        "source_map_path": str(source_map_path),
        "source_map_sha256": sha256_file(source_map_path),
        "source_map_checkpoint_sha256s_verified": True,
        "development_members": len(rows),
        "train_members": split_counts["train"],
        "validation_members": split_counts["validation"],
        "signal_members": class_counts["signal"],
        "background_members": class_counts["background"],
        "zero_row_background_members": zero_background_members,
        "schema_strata": len(schema_strata),
        "representatives_selected": len(representatives),
        "representatives_audited": len(branch_audit_rows),
        "representatives_failed": representatives_failed,
        "members_assigned_to_schema_strata": len(member_to_stratum_rows)
        - members_without_stratum,
        "members_without_schema_stratum": members_without_stratum,
        "strata_without_representative": strata_without_representative,
        "direct_layout_representatives": direct_representatives,
        "nested_layout_representatives": nested_representatives,
        "train_representatives": sum(
            row["dataset_split"] == "train" for row in representatives
        ),
        "validation_representatives": sum(
            row["dataset_split"] == "validation" for row in representatives
        ),
        "zero_row_background_representatives": sum(
            parse_bool(row["zero_row_4b_candidate_file"]) for row in representatives
        ),
        "outer_bundles_downloaded": sum(
            row["xrdcp_status"] == "pass" for row in extraction_audit_rows
        ),
        "root_files_downloaded": sum(
            row["xrdcp_status"] == "pass" for row in extraction_audit_rows
        ),
        "root_files_extracted": sum(
            parse_bool(row["root_file_extracted"]) for row in extraction_audit_rows
        ),
        "root_files_opened_metadata_only": sum(
            row["root_open_status"] == "pass" for row in branch_audit_rows
        ),
        "root_file_open_failures": root_open_failures,
        "delphes_trees_found": len(branch_audit_rows) - trees_missing,
        "delphes_trees_missing": trees_missing,
        "entry_count_matches": len(branch_audit_rows) - entry_mismatches,
        "entry_count_mismatches": entry_mismatches,
        "required_branch_checks": len(all_branch_checks),
        "missing_required_branch_checks": missing_checks,
        "incompatible_required_branch_checks": incompatible_checks,
        "temporary_files_remaining": temporary_files_remaining,
        "root_event_arrays_read": 0,
        "candidate_files_read": 0,
        "candidate_physics_columns_read": 0,
        "test_members_considered": 0,
        "event_generation_performed": 0,
        "reconstruction_performed": 0,
        "physics_yields_calculated": 0,
        **safety,
        "gate_errors": gate_errors,
        "next_gate": "implement_hh4b_3b_control_candidate_builder_with_4b_parity_mode",
    }

    for field in (
        "root_event_arrays_read",
        "candidate_files_read",
        "candidate_physics_columns_read",
        "test_members_considered",
        "event_generation_performed",
        "reconstruction_performed",
        "physics_yields_calculated",
    ):
        gate(
            summary[field] == int(expected[field]),
            f"safety invariant changed: {field}",
        )
    summary["gate_errors"] = gate_errors
    if gate_errors:
        summary["status"] = "hh4b_3b_delphes_branch_audit_fail"

    output_tmp.mkdir(parents=True, exist_ok=False)
    write_tsv(
        output_tmp / "branch_audit_representatives.tsv",
        representatives,
        REPRESENTATIVE_FIELDS,
    )
    write_tsv(
        output_tmp / "member_to_branch_schema_stratum.tsv",
        member_to_stratum_rows,
        [
            "member_index",
            "sample_class",
            "process_or_mode",
            "dataset_split",
            "candidate_rows",
            "resolution_method",
            "layout_rule",
            "bundle_campaign",
            "schema_stratum_key",
            "representative_member_index",
            "schema_stratum_assigned",
            "representative_audit_status",
        ],
    )
    write_tsv(
        output_tmp / "root_tree_branch_audit.tsv",
        branch_audit_rows,
        BRANCH_AUDIT_FIELDS,
    )
    write_tsv(
        output_tmp / "required_branch_summary.tsv",
        branch_summary_rows,
        [
            "branch_name",
            "required_compatibility",
            "representatives_checked",
            "present_checks",
            "missing_checks",
            "compatible_checks",
            "incompatible_checks",
            "observed_typenames_json",
            "observed_interpretations_json",
            "status",
        ],
    )
    write_tsv(
        output_tmp / "observed_branch_type_summary.tsv",
        observed_type_rows,
        [
            "branch_name",
            "observed_typename",
            "observed_interpretation",
            "branch_compatible",
            "representative_count",
            "member_indices_json",
            "processes_or_modes_json",
            "bundle_campaigns_json",
            "status",
        ],
    )
    write_tsv(
        output_tmp / "incompatible_or_missing_branches.tsv",
        issue_rows,
        [
            "member_index",
            "process_or_mode",
            "bundle_campaign",
            "schema_stratum_key",
            "branch_name",
            "required_compatibility",
            "branch_present",
            "observed_typename",
            "observed_interpretation",
            "branch_compatible",
            "failure_reason",
        ],
    )
    write_tsv(
        output_tmp / "extraction_access_audit.tsv",
        extraction_audit_rows,
        EXTRACTION_AUDIT_FIELDS,
    )
    (output_tmp / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_tmp.rename(output_dir)

    if gate_errors:
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise RuntimeError("branch audit gate failed: " + "; ".join(gate_errors))

    checkpoint_tmp.mkdir(parents=True, exist_ok=False)
    for name in OUTPUT_FILES:
        shutil.copy2(output_dir / name, checkpoint_tmp / name)
    (checkpoint_tmp / "README.md").write_text(
        checkpoint_readme(summary),
        encoding="utf-8",
    )
    checkpoint = {
        "schema_version": 1,
        "classification": "paper_quality_delphes_branch_metadata_checkpoint",
        "status": summary["status"],
        "source_commit": source_commit,
        "source_map_sha256": summary["source_map_sha256"],
        "schema_strata": summary["schema_strata"],
        "representatives_audited": summary["representatives_audited"],
        "representatives_failed": summary["representatives_failed"],
        "required_branch_checks": summary["required_branch_checks"],
        "root_event_arrays_read": 0,
        "test_members_considered": 0,
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
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
