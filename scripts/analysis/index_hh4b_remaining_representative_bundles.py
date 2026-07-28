#!/usr/bin/env python3
"""Index all remaining HH4b production bundles without extracting members."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


NESTED_ARCHIVE_SUFFIXES = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
)

EXTRA_ARTIFACT_PATTERNS = {
    "generation_provenance_metadata": (
        "provenance.json",
        "_provenance.",
        "generator.json",
        "generation.json",
        "compile_arguments",
    ),
    "delphes_log_or_configuration": (
        "delphes.log",
        "delphes_card",
        "delphes.tcl",
        "delphes_version",
    ),
    "candidate_or_event_summary_log": (
        "event_summary.log",
        "candidates.log",
    ),
}

BUNDLE_FIELDS = [
    "index_source",
    "index_ordinal",
    "sample_class",
    "process_or_mode",
    "campaign",
    "representative_locator",
    "expected_size_bytes",
    "copied_size_bytes",
    "bundle_sha256",
    "archive_members",
    "regular_file_members",
    "event_payload_members",
    "provenance_named_members",
    "allowlist_candidate_members",
    "unsafe_member_names",
    "temporary_copy_removed",
    "event_payload_content_opened",
    "event_payload_extracted",
    "status",
]

MEMBER_FIELDS = [
    "index_source",
    "index_ordinal",
    "sample_class",
    "process_or_mode",
    "campaign",
    "member_index",
    "member_name",
    "member_type",
    "size_bytes",
    "safe_relative_path",
    "event_payload",
    "nested_archive",
    "artifact_types",
    "allowlist_candidate",
    "content_opened",
    "content_extracted",
    "status",
]

PROVENANCE_FIELDS = [
    "index_source",
    "index_ordinal",
    "sample_class",
    "process_or_mode",
    "campaign",
    "member_name",
    "member_type",
    "size_bytes",
    "artifact_types",
    "safe_relative_path",
    "nested_archive",
    "allowlist_candidate",
    "extraction_authorized_now",
    "status",
]

PAYLOAD_FIELDS = [
    "index_source",
    "index_ordinal",
    "sample_class",
    "process_or_mode",
    "campaign",
    "member_name",
    "member_type",
    "size_bytes",
    "content_opened",
    "content_extracted",
    "status",
]


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("bundle_canary_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import base index script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(
    path: Path,
    rows: Iterable[dict[str, object]],
    fields: list[str],
) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(materialized)


def classify(base, name: str) -> tuple[bool, bool, list[str]]:
    payload, kinds = base.classify(name)
    lowered = name.lower()
    nested_archive = any(
        lowered.endswith(suffix)
        for suffix in NESTED_ARCHIVE_SUFFIXES
    )
    expanded = set(kinds)
    for kind, patterns in EXTRA_ARTIFACT_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            expanded.add(kind)
    return payload, nested_archive, sorted(expanded)


def member_type(member: tarfile.TarInfo) -> str:
    if member.isfile():
        return "regular_file"
    if member.isdir():
        return "directory"
    if member.issym():
        return "symbolic_link"
    if member.islnk():
        return "hard_link"
    return "other"


def xrdcp(remote: str, local: Path) -> None:
    result = subprocess.run(
        ["xrdcp", "-f", remote, str(local)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "xrdcp failed\n"
            f"remote={remote}\n"
            f"local={local}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )


def progress_prefix(ordinal: int, process: str, campaign: str) -> str:
    safe = "_".join(
        part.replace("/", "_")
        for part in (process, campaign)
    )
    return f"{ordinal:03d}_{safe}"


def write_progress(
    progress: Path,
    prefix: str,
    bundle_row: dict[str, object],
    members: list[dict[str, object]],
    provenance: list[dict[str, object]],
    payloads: list[dict[str, object]],
) -> None:
    write_tsv(
        progress / f"{prefix}_members.tsv",
        members,
        MEMBER_FIELDS,
    )
    write_tsv(
        progress / f"{prefix}_provenance.tsv",
        provenance,
        PROVENANCE_FIELDS,
    )
    write_tsv(
        progress / f"{prefix}_payloads.tsv",
        payloads,
        PAYLOAD_FIELDS,
    )
    (progress / f"{prefix}_bundle.json").write_text(
        json.dumps(bundle_row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_progress(
    progress: Path,
    prefix: str,
) -> tuple[
    dict[str, object],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
] | None:
    paths = {
        "bundle": progress / f"{prefix}_bundle.json",
        "members": progress / f"{prefix}_members.tsv",
        "provenance": progress / f"{prefix}_provenance.tsv",
        "payloads": progress / f"{prefix}_payloads.tsv",
    }
    existing = {name: path.exists() for name, path in paths.items()}
    if not any(existing.values()):
        return None
    if not all(existing.values()):
        raise RuntimeError(
            f"incomplete progress fragment for {prefix}: {existing}"
        )
    bundle = json.loads(paths["bundle"].read_text(encoding="utf-8"))
    return (
        bundle,
        read_tsv(paths["members"]),
        read_tsv(paths["provenance"]),
        read_tsv(paths["payloads"]),
    )


def normalize_canary(
    base,
    canary_dir: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    bundle_input = read_tsv(canary_dir / "bundle_index_summary.tsv")
    member_input = read_tsv(canary_dir / "archive_member_inventory.tsv")

    bundles: list[dict[str, object]] = []
    for row in bundle_input:
        key = (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        )
        matching_members = [
            item
            for item in member_input
            if (
                item["sample_class"],
                item["process_or_mode"],
                item["campaign"],
            )
            == key
        ]
        allowlist_count = 0
        for item in matching_members:
            payload, nested, kinds = classify(base, item["member_name"])
            is_regular = item["member_type"] == "regular_file"
            safe = item["safe_relative_path"] == "True"
            allowlist = (
                is_regular
                and safe
                and not payload
                and not nested
                and bool(kinds)
                and int(item["size_bytes"]) <= base.ALLOWLIST_MAX_BYTES
            )
            allowlist_count += int(allowlist)

        bundles.append(
            {
                "index_source": "canary",
                "index_ordinal": int(row["canary_index"]),
                "sample_class": row["sample_class"],
                "process_or_mode": row["process_or_mode"],
                "campaign": row["campaign"],
                "representative_locator": row["representative_locator"],
                "expected_size_bytes": int(row["expected_size_bytes"]),
                "copied_size_bytes": int(row["copied_size_bytes"]),
                "bundle_sha256": row["bundle_sha256"],
                "archive_members": int(row["archive_members"]),
                "regular_file_members": int(row["regular_file_members"]),
                "event_payload_members": int(row["event_payload_members"]),
                "provenance_named_members": sum(
                    bool(classify(base, item["member_name"])[2])
                    for item in matching_members
                ),
                "allowlist_candidate_members": allowlist_count,
                "unsafe_member_names": int(row["unsafe_member_names"]),
                "temporary_copy_removed": True,
                "event_payload_content_opened": False,
                "event_payload_extracted": False,
                "status": "bundle_index_complete_from_canary",
            }
        )

    members: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    payloads: list[dict[str, object]] = []

    for row in member_input:
        payload, nested, kinds = classify(base, row["member_name"])
        is_regular = row["member_type"] == "regular_file"
        safe = row["safe_relative_path"] == "True"
        allowlist = (
            is_regular
            and safe
            and not payload
            and not nested
            and bool(kinds)
            and int(row["size_bytes"]) <= base.ALLOWLIST_MAX_BYTES
        )
        normalized = {
            "index_source": "canary",
            "index_ordinal": int(row["canary_index"]),
            "sample_class": row["sample_class"],
            "process_or_mode": row["process_or_mode"],
            "campaign": row["campaign"],
            "member_index": int(row["member_index"]),
            "member_name": row["member_name"],
            "member_type": row["member_type"],
            "size_bytes": int(row["size_bytes"]),
            "safe_relative_path": safe,
            "event_payload": payload,
            "nested_archive": nested,
            "artifact_types": ",".join(kinds) if kinds else "none",
            "allowlist_candidate": allowlist,
            "content_opened": False,
            "content_extracted": False,
            "status": "archive_header_indexed_only",
        }
        members.append(normalized)
        if kinds:
            provenance.append(
                {
                    "index_source": "canary",
                    "index_ordinal": int(row["canary_index"]),
                    "sample_class": row["sample_class"],
                    "process_or_mode": row["process_or_mode"],
                    "campaign": row["campaign"],
                    "member_name": row["member_name"],
                    "member_type": row["member_type"],
                    "size_bytes": int(row["size_bytes"]),
                    "artifact_types": ",".join(kinds),
                    "safe_relative_path": safe,
                    "nested_archive": nested,
                    "allowlist_candidate": allowlist,
                    "extraction_authorized_now": False,
                    "status": (
                        "candidate_for_pn_b3_allowlist_review"
                        if allowlist
                        else "not_allowlisted"
                    ),
                }
            )
        if payload:
            payloads.append(
                {
                    "index_source": "canary",
                    "index_ordinal": int(row["canary_index"]),
                    "sample_class": row["sample_class"],
                    "process_or_mode": row["process_or_mode"],
                    "campaign": row["campaign"],
                    "member_name": row["member_name"],
                    "member_type": row["member_type"],
                    "size_bytes": int(row["size_bytes"]),
                    "content_opened": False,
                    "content_extracted": False,
                    "status": "payload_identified_and_protected",
                }
            )
    return bundles, members, provenance, payloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--canary-dir", type=Path, required=True)
    parser.add_argument("--base-script", type=Path, required=True)
    parser.add_argument("--eos-host", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    base = load_base(args.base_script.resolve())
    plan = args.plan.resolve()
    canary_dir = args.canary_dir.resolve()
    scratch = args.scratch.resolve()
    output = args.output.resolve()

    if not scratch.is_dir() or any(scratch.iterdir()):
        raise RuntimeError("scratch must exist and be empty")

    plan_rows = read_tsv(plan)
    if len(plan_rows) != 54:
        raise RuntimeError(f"plan rows={len(plan_rows)}, expected 54")

    remote_rows = [
        row
        for row in plan_rows
        if row["source_type"] == "eos_bundle"
    ]
    legacy_rows = [
        row
        for row in plan_rows
        if row["source_type"] == "local_candidate_reference"
    ]
    if len(remote_rows) != 53 or len(legacy_rows) != 1:
        raise RuntimeError(
            "plan source-type counts mismatch: "
            f"remote={len(remote_rows)}, legacy={len(legacy_rows)}"
        )

    canary_keys = set(base.CANARIES)
    remaining = [
        row
        for row in remote_rows
        if (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        )
        not in canary_keys
    ]
    if len(remaining) != 47:
        raise RuntimeError(
            f"remaining EOS campaigns={len(remaining)}, expected 47"
        )

    if output.exists() and not output.is_dir():
        raise RuntimeError(f"output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress"
    progress.mkdir(exist_ok=True)

    state_path = output / "resume_state.json"
    plan_sha = sha256_file(plan)
    expected_state = {
        "schema_version": 1,
        "source_commit": args.source_commit,
        "plan_sha256": plan_sha,
        "remaining_campaigns": 47,
    }
    if state_path.exists():
        observed_state = json.loads(state_path.read_text(encoding="utf-8"))
        if observed_state != expected_state:
            raise RuntimeError(
                "resume-state mismatch\n"
                f"observed={observed_state}\n"
                f"expected={expected_state}"
            )
    else:
        state_path.write_text(
            json.dumps(expected_state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    scaleout_bundles: list[dict[str, object]] = []
    scaleout_members: list[dict[str, object]] = []
    scaleout_provenance: list[dict[str, object]] = []
    scaleout_payloads: list[dict[str, object]] = []

    for ordinal, row in enumerate(remaining, start=1):
        sample_class = row["sample_class"]
        process = row["process_or_mode"]
        campaign = row["campaign"]
        prefix = progress_prefix(ordinal, process, campaign)

        recovered = load_progress(progress, prefix)
        if recovered is not None:
            bundle, members, provenance, payloads = recovered
            scaleout_bundles.append(bundle)
            scaleout_members.extend(members)
            scaleout_provenance.extend(provenance)
            scaleout_payloads.extend(payloads)
            print(
                f"RESUME {ordinal}/47 {process} {campaign}",
                flush=True,
            )
            continue

        if row["event_payload_extraction_authorized"] != "False":
            raise RuntimeError(f"payload extraction authorized: {campaign}")
        if row["candidate_content_access_authorized"] != "False":
            raise RuntimeError(f"candidate access authorized: {campaign}")

        locator = row["representative_locator"]
        expected_size = int(row["representative_size_bytes"])
        remote = args.eos_host.rstrip("/") + "//" + locator.lstrip("/")
        local = scratch / f"{ordinal:03d}_{process}.tar.gz"

        free = shutil.disk_usage(scratch).free
        if free < expected_size + 2 * 1024 * 1024 * 1024:
            raise RuntimeError(
                f"insufficient scratch for {campaign}: "
                f"free={free}, expected_bundle={expected_size}"
            )

        print(
            f"INDEX {ordinal}/47 {process} {campaign} "
            f"bytes={expected_size}",
            flush=True,
        )

        try:
            xrdcp(remote, local)
            actual_size = local.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"size mismatch for {campaign}: "
                    f"{actual_size} != {expected_size}"
                )
            bundle_sha = sha256_file(local)

            members: list[dict[str, object]] = []
            provenance: list[dict[str, object]] = []
            payloads: list[dict[str, object]] = []
            regular_files = 0
            unsafe_names = 0
            allowlist_count = 0

            with tarfile.open(local, mode="r:*") as archive:
                for member_index, member in enumerate(archive, start=1):
                    kind = member_type(member)
                    is_regular = member.isfile()
                    regular_files += int(is_regular)
                    safe = base.safe_name(member.name)
                    unsafe_names += int(not safe)
                    payload, nested, artifact_types = classify(
                        base,
                        member.name,
                    )
                    allowlist = (
                        is_regular
                        and safe
                        and not payload
                        and not nested
                        and bool(artifact_types)
                        and member.size <= base.ALLOWLIST_MAX_BYTES
                    )
                    allowlist_count += int(allowlist)

                    member_row = {
                        "index_source": "scaleout",
                        "index_ordinal": ordinal,
                        "sample_class": sample_class,
                        "process_or_mode": process,
                        "campaign": campaign,
                        "member_index": member_index,
                        "member_name": member.name,
                        "member_type": kind,
                        "size_bytes": member.size,
                        "safe_relative_path": safe,
                        "event_payload": payload,
                        "nested_archive": nested,
                        "artifact_types": (
                            ",".join(artifact_types)
                            if artifact_types
                            else "none"
                        ),
                        "allowlist_candidate": allowlist,
                        "content_opened": False,
                        "content_extracted": False,
                        "status": "archive_header_indexed_only",
                    }
                    members.append(member_row)

                    if artifact_types:
                        provenance.append(
                            {
                                "index_source": "scaleout",
                                "index_ordinal": ordinal,
                                "sample_class": sample_class,
                                "process_or_mode": process,
                                "campaign": campaign,
                                "member_name": member.name,
                                "member_type": kind,
                                "size_bytes": member.size,
                                "artifact_types": ",".join(artifact_types),
                                "safe_relative_path": safe,
                                "nested_archive": nested,
                                "allowlist_candidate": allowlist,
                                "extraction_authorized_now": False,
                                "status": (
                                    "candidate_for_pn_b3_allowlist_review"
                                    if allowlist
                                    else "not_allowlisted"
                                ),
                            }
                        )

                    if payload:
                        payloads.append(
                            {
                                "index_source": "scaleout",
                                "index_ordinal": ordinal,
                                "sample_class": sample_class,
                                "process_or_mode": process,
                                "campaign": campaign,
                                "member_name": member.name,
                                "member_type": kind,
                                "size_bytes": member.size,
                                "content_opened": False,
                                "content_extracted": False,
                                "status": "payload_identified_and_protected",
                            }
                        )

            bundle = {
                "index_source": "scaleout",
                "index_ordinal": ordinal,
                "sample_class": sample_class,
                "process_or_mode": process,
                "campaign": campaign,
                "representative_locator": locator,
                "expected_size_bytes": expected_size,
                "copied_size_bytes": actual_size,
                "bundle_sha256": bundle_sha,
                "archive_members": len(members),
                "regular_file_members": regular_files,
                "event_payload_members": len(payloads),
                "provenance_named_members": len(provenance),
                "allowlist_candidate_members": allowlist_count,
                "unsafe_member_names": unsafe_names,
                "temporary_copy_removed": True,
                "event_payload_content_opened": False,
                "event_payload_extracted": False,
                "status": "bundle_index_scaleout_pass",
            }
        finally:
            if local.exists():
                local.unlink()

        if local.exists():
            raise RuntimeError(f"temporary bundle remains: {local}")

        write_progress(
            progress,
            prefix,
            bundle,
            members,
            provenance,
            payloads,
        )
        scaleout_bundles.append(bundle)
        scaleout_members.extend(members)
        scaleout_provenance.extend(provenance)
        scaleout_payloads.extend(payloads)

        print(
            f"COMPLETE {ordinal}/47 members={len(members)} "
            f"payloads={len(payloads)} "
            f"provenance={len(provenance)} "
            f"sha256={bundle_sha}",
            flush=True,
        )

    if len(scaleout_bundles) != 47:
        raise RuntimeError(
            f"completed scaleout bundles={len(scaleout_bundles)}, expected 47"
        )
    if any(scratch.iterdir()):
        raise RuntimeError("scratch is not empty after full indexing")

    canary_bundles, canary_members, canary_provenance, canary_payloads = (
        normalize_canary(base, canary_dir)
    )

    all_bundles = canary_bundles + scaleout_bundles
    all_members = canary_members + scaleout_members
    all_provenance = canary_provenance + scaleout_provenance
    all_payloads = canary_payloads + scaleout_payloads

    if len(all_bundles) != 53:
        raise RuntimeError(
            f"combined indexed EOS bundles={len(all_bundles)}, expected 53"
        )

    observed_keys = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        )
        for row in all_bundles
    }
    expected_keys = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        )
        for row in remote_rows
    }
    if observed_keys != expected_keys:
        raise RuntimeError("combined EOS campaign coverage mismatch")

    write_tsv(
        output / "all_eos_bundle_index_summary.tsv",
        all_bundles,
        BUNDLE_FIELDS,
    )
    write_tsv(
        output / "all_archive_member_inventory.tsv",
        all_members,
        MEMBER_FIELDS,
    )
    write_tsv(
        output / "all_provenance_member_candidates.tsv",
        all_provenance,
        PROVENANCE_FIELDS,
    )
    write_tsv(
        output / "all_event_payload_member_inventory.tsv",
        all_payloads,
        PAYLOAD_FIELDS,
    )

    allowlist_rows = [
        {
            "sample_class": row["sample_class"],
            "process_or_mode": row["process_or_mode"],
            "campaign": row["campaign"],
            "representative_locator": next(
                item["representative_locator"]
                for item in all_bundles
                if (
                    item["sample_class"],
                    item["process_or_mode"],
                    item["campaign"],
                )
                == (
                    row["sample_class"],
                    row["process_or_mode"],
                    row["campaign"],
                )
            ),
            "member_name": row["member_name"],
            "size_bytes": row["size_bytes"],
            "artifact_types": row["artifact_types"],
            "safe_relative_path": row["safe_relative_path"],
            "nested_archive": row["nested_archive"],
            "event_payload": False,
            "extraction_authorized_now": False,
            "content_review_authorized_now": False,
            "status": "draft_candidate_requires_pn_b3_policy_freeze",
        }
        for row in all_provenance
        if str(row["allowlist_candidate"]) == "True"
        or row["allowlist_candidate"] is True
    ]
    write_tsv(
        output / "provenance_extraction_allowlist_draft.tsv",
        allowlist_rows,
        [
            "sample_class",
            "process_or_mode",
            "campaign",
            "representative_locator",
            "member_name",
            "size_bytes",
            "artifact_types",
            "safe_relative_path",
            "nested_archive",
            "event_payload",
            "extraction_authorized_now",
            "content_review_authorized_now",
            "status",
        ],
    )

    artifact_coverage_rows: list[dict[str, object]] = []
    grouped_types: dict[
        tuple[str, str, str],
        Counter[str],
    ] = defaultdict(Counter)
    for row in all_provenance:
        key = (
            str(row["sample_class"]),
            str(row["process_or_mode"]),
            str(row["campaign"]),
        )
        for artifact_type in str(row["artifact_types"]).split(","):
            grouped_types[key][artifact_type] += 1

    for bundle in all_bundles:
        key = (
            str(bundle["sample_class"]),
            str(bundle["process_or_mode"]),
            str(bundle["campaign"]),
        )
        counts = grouped_types[key]
        artifact_coverage_rows.append(
            {
                "sample_class": key[0],
                "process_or_mode": key[1],
                "campaign": key[2],
                "artifact_types_located": (
                    ",".join(sorted(counts))
                    if counts
                    else "none"
                ),
                "artifact_type_counts": (
                    ",".join(
                        f"{name}:{counts[name]}"
                        for name in sorted(counts)
                    )
                    if counts
                    else "none"
                ),
                "allowlist_candidates": sum(
                    1
                    for row in allowlist_rows
                    if (
                        row["sample_class"],
                        row["process_or_mode"],
                        row["campaign"],
                    )
                    == key
                ),
                "provenance_content_reviewed": False,
                "sum_generator_weights_resolved": False,
                "cross_section_provenance_resolved": False,
                "physics_normalization_ready": False,
                "status": "archive_index_complete_content_review_pending",
            }
        )
    write_tsv(
        output / "artifact_type_coverage.tsv",
        artifact_coverage_rows,
        list(artifact_coverage_rows[0]),
    )

    coverage_rows = [
        {
            "sample_class": row["sample_class"],
            "process_or_mode": row["process_or_mode"],
            "campaign": row["campaign"],
            "source_type": row["source_type"],
            "representative_bundle_indexed": True,
            "bundle_sha256_recorded": True,
            "archive_members_extracted": 0,
            "event_payload_opened": False,
            "physics_normalization_ready": False,
            "status": "eos_representative_bundle_index_complete",
        }
        for row in remote_rows
    ]
    legacy = legacy_rows[0]
    coverage_rows.append(
        {
            "sample_class": legacy["sample_class"],
            "process_or_mode": legacy["process_or_mode"],
            "campaign": legacy["campaign"],
            "source_type": legacy["source_type"],
            "representative_bundle_indexed": False,
            "bundle_sha256_recorded": False,
            "archive_members_extracted": 0,
            "event_payload_opened": False,
            "physics_normalization_ready": False,
            "status": "blocking_legacy_generation_provenance_unresolved",
        }
    )
    write_tsv(
        output / "campaign_index_coverage.tsv",
        coverage_rows,
        list(coverage_rows[0]),
    )

    blocker_rows = [
        {
            "sample_class": legacy["sample_class"],
            "process_or_mode": legacy["process_or_mode"],
            "campaign": legacy["campaign"],
            "source_type": legacy["source_type"],
            "original_generation_artifacts_recovered": False,
            "candidate_file_opened": False,
            "blocking_reason": (
                "The legacy 270k-event ttbar candidate source lacks its "
                "original generator directory, cards, logs, and normalization "
                "denominator in the frozen registry."
            ),
            "status": "blocking_legacy_provenance_unresolved",
        }
    ]
    write_tsv(
        output / "legacy_ttbar_provenance_blocker.tsv",
        blocker_rows,
        list(blocker_rows[0]),
    )

    artifact_counts = Counter()
    for row in all_provenance:
        for artifact_type in str(row["artifact_types"]).split(","):
            artifact_counts[artifact_type] += 1

    summary = {
        "schema_version": 1,
        "status": "hh4b_physical_normalization_full_bundle_index_pass",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "plan": str(args.plan),
        "plan_sha256": plan_sha,
        "indexing": {
            "campaign_records": 54,
            "eos_campaigns": 53,
            "canary_eos_campaigns_reused": 6,
            "remaining_eos_campaigns_indexed": 47,
            "all_eos_campaigns_indexed": 53,
            "legacy_local_campaigns_unresolved": 1,
            "archive_members": len(all_members),
            "event_payload_members": len(all_payloads),
            "provenance_named_members": len(all_provenance),
            "provenance_extraction_allowlist_candidates": len(
                allowlist_rows
            ),
            "artifact_type_member_counts": dict(
                sorted(artifact_counts.items())
            ),
            "scaleout_transfer_bytes": sum(
                int(row["copied_size_bytes"])
                for row in scaleout_bundles
            ),
        },
        "readiness": {
            "all_eos_archive_headers_indexed": True,
            "provenance_extraction_allowlist_draft_complete": True,
            "provenance_file_extraction_authorized": False,
            "provenance_file_contents_reviewed": 0,
            "sum_generator_weights_resolved": 0,
            "cross_sections_assigned": 0,
            "physics_normalization_ready": False,
        },
        "controls": {
            "archive_members_extracted": 0,
            "candidate_parquet_files_opened": 0,
            "candidate_rows_read": 0,
            "validation_candidate_files_opened": 0,
            "evaluation_candidate_files_opened": 0,
            "event_payload_files_opened": 0,
            "event_payload_files_extracted": 0,
            "temporary_bundle_copies_remaining": 0,
            "provenance_files_extracted": 0,
            "cross_sections_assigned": 0,
            "physical_yields_calculated": 0,
            "models_trained": 0,
            "thresholds_selected": 0,
        },
        "next_gate": (
            "freeze_small_nonpayload_provenance_extraction_allowlist_"
            "and_recover_legacy_ttbar_generation_artifacts"
        ),
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# HH4b full representative production-bundle index\n\n"
        "All 53 EOS process-campaign representative bundles now have "
        "byte-size and SHA-256 provenance and complete archive-header "
        "inventories. Six validated canaries were reused and the remaining "
        "47 bundles were copied sequentially to temporary non-workspace "
        "scratch, indexed, and deleted.\n\n"
        "No archive member was extracted. ROOT, Parquet, HepMC, and LHE "
        "contents were not opened. The resulting draft allowlist contains "
        "only small, regular, safe-path, non-payload, non-archive members "
        "with provenance-like names. Extraction remains unauthorized.\n\n"
        "The legacy local ttbar campaign remains a blocking provenance "
        "exception. The next gate freezes the PN-b3 extraction allowlist "
        "and separately recovers the legacy ttbar generation artifacts.\n",
        encoding="utf-8",
    )

    shutil.rmtree(progress)
    state_path.unlink()

    checksum_path = output / "SHA256SUMS"
    products = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in products
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(json.dumps(summary, indent=2, sort_keys=True))
    print()
    print("REMAINING_47_EOS_BUNDLES_INDEXED_PASS")
    print("ALL_53_EOS_CAMPAIGNS_ARCHIVE_INDEXED_PASS")
    print("ALL_BUNDLE_SIZE_AND_SHA256_PROVENANCE_PASS")
    print("PROVENANCE_EXTRACTION_ALLOWLIST_DRAFT_PASS")
    print("LEGACY_TTBAR_PROVENANCE_BLOCKER_RECORDED_PASS")
    print("TEMPORARY_BUNDLE_COPIES_REMOVED_PASS")
    print("NO_ARCHIVE_MEMBER_EXTRACTED")
    print("NO_EVENT_PAYLOAD_CONTENT_OPENED")
    print("NO_CANDIDATE_PARQUET_OPENED")
    print("NO_VALIDATION_CANDIDATE_CONTENT_OPENED")
    print("NO_EVALUATION_CANDIDATE_CONTENT_OPENED")
    print("NO_CROSS_SECTIONS_ASSIGNED")
    print("NO_PHYSICAL_YIELDS_CALCULATED")
    print("HH4B_FULL_REPRESENTATIVE_BUNDLE_INDEX_PASS")


if __name__ == "__main__":
    main()
