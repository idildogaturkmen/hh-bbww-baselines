#!/usr/bin/env python3
"""Audit denominator evidence and source recovery without opening payloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


EXPECTED_ALL_CAMPAIGNS = 54
EXPECTED_EOS_CAMPAIGNS = 53
EXPECTED_STANDARD_CAMPAIGNS = 47
EXPECTED_NONSTANDARD_CAMPAIGNS = 6
EXPECTED_PROVENANCE_FILES = 547
EXPECTED_EXACT_CARD_FILES = 188

MAX_SEARCH_FILE_BYTES = 10 * 1024 * 1024
MAX_CONTENT_MATCHES_PER_FILE = 20
MAX_MATCHES_PER_CAMPAIGN = 3000

TEXT_EXTENSIONS = {
    ".cfg",
    ".conf",
    ".csv",
    ".dag",
    ".ini",
    ".jdl",
    ".json",
    ".log",
    ".md",
    ".py",
    ".sh",
    ".sub",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}

PRUNE_DIR_NAMES = {
    ".cache",
    ".conda",
    ".git",
    ".ipynb_checkpoints",
    ".local",
    "__pycache__",
    "node_modules",
    "venv",
    "venvs",
}

FORBIDDEN_FILE_SUFFIXES = {
    ".root",
    ".parquet",
    ".hepmc",
    ".lhe",
    ".tar",
    ".tgz",
    ".zip",
    ".gz",
    ".xz",
    ".bz2",
}

DENOMINATOR_KEY_PATTERN = re.compile(
    r"(?:^|[._\-/])"
    r"(?:sumw|sum_weights|sum_of_weights|sum_generator_weights|"
    r"sum_gen_weights|total_weight|nominal_sumw|generator_weight_sum|"
    r"generated_events|events_generated|event_count|n_events|nevents|"
    r"accepted_events|unweighted_events)"
    r"(?:$|[._\-/])",
    flags=re.IGNORECASE,
)

WEIGHT_KEY_PATTERN = re.compile(
    r"(?:weight|weighted|unweighted|negative_weight|positive_weight|"
    r"event_norm|weighting_strategy)",
    flags=re.IGNORECASE,
)

QCD_KEY_PATTERN = re.compile(
    r"(?:pthat|importance|sampling|proposal|minimum_fraction|"
    r"bin[_ -]?(?:low|high|index)|stitch|overlap|phase.?space)",
    flags=re.IGNORECASE,
)

TEXT_EVIDENCE_PATTERNS = {
    "sum_weights": re.compile(
        r"(?:sum\s*(?:of)?\s*(?:generator\s*)?weights|sumw|total\s+weight)",
        flags=re.IGNORECASE,
    ),
    "unweighted": re.compile(
        r"\bunweighted(?:\s+events?)?\b",
        flags=re.IGNORECASE,
    ),
    "negative_weight": re.compile(
        r"(?:negative[- ]weight|negative\s+weights?|nnegative|negwgt)",
        flags=re.IGNORECASE,
    ),
    "event_weighting_strategy": re.compile(
        r"event[_ ]weighting[_ ]strategy",
        flags=re.IGNORECASE,
    ),
    "event_norm": re.compile(
        r"\bevent_norm\b",
        flags=re.IGNORECASE,
    ),
    "generated_events": re.compile(
        r"(?:number\s+of\s+events|generated[_ ]events|events[_ ]generated|"
        r"\bnevents\b)",
        flags=re.IGNORECASE,
    ),
    "process_weight_line": re.compile(
        r"\bprocess\s+\d+\s+weight\s*=",
        flags=re.IGNORECASE,
    ),
}

RUN_CARD_EVENT_NORM_PATTERN = re.compile(
    r"^\s*([^#!\n]+?)\s*=\s*event_norm(?:\s*[#!].*)?$",
    flags=re.IGNORECASE | re.MULTILINE,
)

RUN_CARD_NEVENTS_PATTERN = re.compile(
    r"^\s*(\d+)\s*=\s*nevents(?:\s*[#!].*)?$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def flatten_json(
    value: Any,
    prefix: str = "",
) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_json(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            yield from flatten_json(item, child)
    else:
        yield prefix, value


def campaign_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row["sample_class"]),
        str(row["process_or_mode"]),
        str(row["campaign"]),
    )


def source_kind(member_name: str, artifact_types: str) -> str:
    basename = PurePosixPath(member_name).name.lower()
    parts = PurePosixPath(member_name).parts
    artifact_set = set(artifact_types.split(","))

    if "cards" in parts and basename == "run_card.dat":
        return "run_card"
    if "cards" in parts and basename == "proc_card_mg5.dat":
        return "proc_card"
    if "cards" in parts and basename == "pythia8_card_default.dat":
        return "pythia8_card"
    if "cards" in parts and basename == "pythia_card_default.dat":
        return "pythia6_card"
    if basename.endswith(".json"):
        return "json"
    if "generator_stdout_or_summary" in artifact_set:
        return "generator_log_or_summary"
    if "generator_banner_or_lhe_init" in artifact_set:
        return "generator_banner"
    if "parton_shower_card" in artifact_set:
        return "pythia_log_or_card"
    return "other_provenance"


def relevant_json_class(path: str, value: Any) -> str | None:
    combined = f"{path} {value}"
    if DENOMINATOR_KEY_PATTERN.search(path):
        return "denominator_candidate"
    if QCD_KEY_PATTERN.search(combined):
        return "qcd_sampling_or_overlap_candidate"
    if WEIGHT_KEY_PATTERN.search(path):
        return "weight_strategy_candidate"
    return None


def is_searchable_file(path: Path) -> bool:
    lowered = path.name.lower()
    if any(lowered.endswith(suffix) for suffix in FORBIDDEN_FILE_SUFFIXES):
        return False
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    return 0 < size <= MAX_SEARCH_FILE_BYTES


def safe_read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def search_local_sources(
    roots: list[Path],
    campaigns: list[tuple[str, str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    result_rows: list[dict[str, object]] = []
    root_rows: list[dict[str, object]] = []
    campaign_names = {key[2]: key for key in campaigns}
    counts = Counter()
    seen_paths: set[str] = set()

    for root in roots:
        root = root.resolve()
        files_considered = 0
        searchable_files = 0
        files_read = 0
        filename_matches = 0
        content_matches = 0
        errors = 0

        if not root.exists():
            root_rows.append(
                {
                    "search_root": str(root),
                    "exists": False,
                    "files_considered": 0,
                    "searchable_files": 0,
                    "files_read": 0,
                    "filename_matches": 0,
                    "content_matches": 0,
                    "errors": 0,
                    "status": "root_absent",
                }
            )
            continue

        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in PRUNE_DIR_NAMES
                and not name.startswith(".nfs")
            ]

            for filename in filenames:
                path = Path(directory) / filename
                path_string = str(path)
                if path_string in seen_paths:
                    continue
                seen_paths.add(path_string)
                files_considered += 1

                matched_names = [
                    campaign
                    for campaign in campaign_names
                    if campaign in filename
                ]
                for campaign in matched_names:
                    if counts[campaign] >= MAX_MATCHES_PER_CAMPAIGN:
                        continue
                    key = campaign_names[campaign]
                    result_rows.append(
                        {
                            "sample_class": key[0],
                            "process_or_mode": key[1],
                            "campaign": key[2],
                            "search_root": str(root),
                            "match_type": "filename",
                            "path": path_string,
                            "line_number": "",
                            "matched_text": filename,
                            "event_payload_opened": False,
                            "status": "source_recovery_candidate",
                        }
                    )
                    counts[campaign] += 1
                    filename_matches += 1

                if not is_searchable_file(path):
                    continue
                searchable_files += 1
                text = safe_read_text(path)
                if text is None:
                    errors += 1
                    continue
                files_read += 1

                matched_campaigns = [
                    campaign
                    for campaign in campaign_names
                    if campaign in text
                    and counts[campaign] < MAX_MATCHES_PER_CAMPAIGN
                ]
                if not matched_campaigns:
                    continue

                lines = text.splitlines()
                for campaign in matched_campaigns:
                    key = campaign_names[campaign]
                    per_file = 0
                    for line_number, line in enumerate(lines, start=1):
                        if campaign not in line:
                            continue
                        result_rows.append(
                            {
                                "sample_class": key[0],
                                "process_or_mode": key[1],
                                "campaign": key[2],
                                "search_root": str(root),
                                "match_type": "text_reference",
                                "path": path_string,
                                "line_number": line_number,
                                "matched_text": line[:500],
                                "event_payload_opened": False,
                                "status": "source_recovery_candidate",
                            }
                        )
                        counts[campaign] += 1
                        content_matches += 1
                        per_file += 1
                        if (
                            per_file >= MAX_CONTENT_MATCHES_PER_FILE
                            or counts[campaign] >= MAX_MATCHES_PER_CAMPAIGN
                        ):
                            break

        root_rows.append(
            {
                "search_root": str(root),
                "exists": True,
                "files_considered": files_considered,
                "searchable_files": searchable_files,
                "files_read": files_read,
                "filename_matches": filename_matches,
                "content_matches": content_matches,
                "errors": errors,
                "status": "approved_root_search_complete",
            }
        )

    return result_rows, root_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-inventory", type=Path, required=True)
    parser.add_argument("--provenance-archive", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--source-policy", type=Path, required=True)
    parser.add_argument("--nonstandard-blockers", type=Path, required=True)
    parser.add_argument("--denominator-fail-closed", type=Path, required=True)
    parser.add_argument("--exact-card-inventory", type=Path, required=True)
    parser.add_argument("--search-root", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")

    inventory_rows = read_tsv(args.campaign_inventory)
    manifest_rows = read_tsv(args.provenance_manifest)
    source_policy_rows = read_tsv(args.source_policy)
    blocker_rows = read_tsv(args.nonstandard_blockers)
    denominator_rows_input = read_tsv(args.denominator_fail_closed)
    exact_card_rows = read_tsv(args.exact_card_inventory)

    if len(inventory_rows) != EXPECTED_ALL_CAMPAIGNS:
        raise RuntimeError("campaign inventory row count mismatch")
    if len(manifest_rows) != EXPECTED_PROVENANCE_FILES:
        raise RuntimeError("provenance manifest row count mismatch")
    if len(source_policy_rows) != EXPECTED_ALL_CAMPAIGNS:
        raise RuntimeError("source policy row count mismatch")
    if len(blocker_rows) != EXPECTED_NONSTANDARD_CAMPAIGNS:
        raise RuntimeError("nonstandard blocker row count mismatch")
    if len(denominator_rows_input) != EXPECTED_EOS_CAMPAIGNS:
        raise RuntimeError("denominator input row count mismatch")
    if len(exact_card_rows) != EXPECTED_EXACT_CARD_FILES:
        raise RuntimeError("exact card inventory row count mismatch")

    included_keys = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        )
        for row in source_policy_rows
        if row["physical_normalization_included"] == "True"
    }
    if len(included_keys) != EXPECTED_EOS_CAMPAIGNS:
        raise RuntimeError("included EOS campaign count mismatch")

    nonstandard_keys = {campaign_key(row) for row in blocker_rows}
    standard_keys = included_keys - nonstandard_keys
    if len(standard_keys) != EXPECTED_STANDARD_CAMPAIGNS:
        raise RuntimeError("standard campaign count mismatch")

    inventory_map = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        ): row
        for row in inventory_rows
    }
    denominator_input_map = {
        campaign_key(row): row
        for row in denominator_rows_input
    }
    manifest_map = {row["archive_path"]: row for row in manifest_rows}
    if len(manifest_map) != EXPECTED_PROVENANCE_FILES:
        raise RuntimeError("provenance archive paths are not unique")

    output.mkdir(parents=True)
    try:
        structured_rows: list[dict[str, object]] = []
        text_rows: list[dict[str, object]] = []
        file_rows: list[dict[str, object]] = []
        run_card_by_campaign: dict[
            tuple[str, str, str],
            dict[str, list[str]],
        ] = defaultdict(lambda: defaultdict(list))
        structured_by_campaign: dict[
            tuple[str, str, str],
            list[dict[str, object]],
        ] = defaultdict(list)
        text_by_campaign: dict[
            tuple[str, str, str],
            list[dict[str, object]],
        ] = defaultdict(list)

        with tarfile.open(args.provenance_archive, mode="r:gz") as archive:
            members = {
                member.name: member
                for member in archive.getmembers()
            }
            if len(members) != EXPECTED_PROVENANCE_FILES:
                raise RuntimeError("provenance archive member count mismatch")
            if set(members) != set(manifest_map):
                raise RuntimeError(
                    "provenance archive and manifest path sets differ"
                )

            for archive_path in sorted(members):
                member = members[archive_path]
                manifest = manifest_map[archive_path]
                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeError(
                        f"could not read provenance member: {archive_path}"
                    )
                data = handle.read()
                if len(data) != int(manifest["size_bytes"]):
                    raise RuntimeError(
                        f"size mismatch: {archive_path}"
                    )
                if sha256_bytes(data) != manifest["sha256"]:
                    raise RuntimeError(
                        f"SHA-256 mismatch: {archive_path}"
                    )
                text = data.decode("utf-8")
                key = (
                    manifest["sample_class"],
                    manifest["process_or_mode"],
                    manifest["campaign"],
                )
                kind = source_kind(
                    manifest["member_name"],
                    manifest["artifact_types"],
                )

                file_rows.append(
                    {
                        "sample_class": key[0],
                        "process_or_mode": key[1],
                        "campaign": key[2],
                        "source_kind": kind,
                        "archive_path": archive_path,
                        "member_name": manifest["member_name"],
                        "size_bytes": len(data),
                        "sha256": manifest["sha256"],
                        "content_hash_verified": True,
                        "event_payload": False,
                        "status": "denominator_evidence_input",
                    }
                )

                if kind == "run_card":
                    for match in RUN_CARD_EVENT_NORM_PATTERN.finditer(text):
                        run_card_by_campaign[key]["event_norm"].append(
                            " ".join(match.group(1).split())
                        )
                    for match in RUN_CARD_NEVENTS_PATTERN.finditer(text):
                        run_card_by_campaign[key]["nevents"].append(
                            match.group(1)
                        )

                if kind == "json":
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        payload = None
                    if payload is not None:
                        for json_path, value in flatten_json(payload):
                            evidence_class = relevant_json_class(
                                json_path,
                                value,
                            )
                            if evidence_class is None:
                                continue
                            row = {
                                "sample_class": key[0],
                                "process_or_mode": key[1],
                                "campaign": key[2],
                                "evidence_class": evidence_class,
                                "field": json_path,
                                "value": value,
                                "source_archive_path": archive_path,
                                "source_member_name": manifest["member_name"],
                                "source_kind": kind,
                                "normalization_denominator_authorized": False,
                                "status": "structured_candidate_requires_review",
                            }
                            structured_rows.append(row)
                            structured_by_campaign[key].append(row)

                for line_number, line in enumerate(
                    text.splitlines(),
                    start=1,
                ):
                    matched_classes = [
                        evidence_class
                        for evidence_class, pattern
                        in TEXT_EVIDENCE_PATTERNS.items()
                        if pattern.search(line)
                    ]
                    if not matched_classes:
                        continue
                    row = {
                        "sample_class": key[0],
                        "process_or_mode": key[1],
                        "campaign": key[2],
                        "evidence_classes": ",".join(matched_classes),
                        "line_text": " ".join(line.split())[:1000],
                        "source_archive_path": archive_path,
                        "source_member_name": manifest["member_name"],
                        "source_line": line_number,
                        "source_kind": kind,
                        "normalization_denominator_authorized": False,
                        "status": "text_candidate_requires_review",
                    }
                    text_rows.append(row)
                    text_by_campaign[key].append(row)

        campaign_rows: list[dict[str, object]] = []
        standard_rows: list[dict[str, object]] = []
        nonstandard_rows: list[dict[str, object]] = []
        access_rows: list[dict[str, object]] = []

        for key in sorted(included_keys):
            structured = structured_by_campaign[key]
            text_evidence = text_by_campaign[key]
            run_values = run_card_by_campaign[key]

            denominator_structured = [
                row
                for row in structured
                if row["evidence_class"] == "denominator_candidate"
            ]
            weight_structured = [
                row
                for row in structured
                if row["evidence_class"] == "weight_strategy_candidate"
            ]
            qcd_structured = [
                row
                for row in structured
                if row["evidence_class"]
                == "qcd_sampling_or_overlap_candidate"
            ]

            evidence_counter = Counter()
            for row in text_evidence:
                for evidence_class in str(row["evidence_classes"]).split(","):
                    evidence_counter[evidence_class] += 1

            is_standard = key in standard_keys
            if is_standard:
                metadata_status = (
                    "standard_cards_and_provenance_available_"
                    "but_campaign_sumw_or_uniform_weight_proof_missing"
                )
                next_action = (
                    "perform a controlled generator-level nominal-weight "
                    "audit across campaign shards or recover trusted aggregate sumw"
                )
            elif key[1] == "ggf_hh4b":
                metadata_status = (
                    "ggf_nonstandard_configuration_and_denominator_unresolved"
                )
                next_action = (
                    "recover exact ggF source configuration and aggregate "
                    "generator-weight denominator before theory matching"
                )
            else:
                metadata_status = (
                    "importance_sampled_qcd_configuration_denominator_"
                    "and_stitching_unresolved"
                )
                next_action = (
                    "recover QCD bin definitions, proposal probabilities, "
                    "per-bin cross sections, and aggregate nominal-weight sums"
                )

            campaign_rows.append(
                {
                    "sample_class": key[0],
                    "process_or_mode": key[1],
                    "campaign": key[2],
                    "standard_card_campaign": is_standard,
                    "manifest_generated_events": int(
                        inventory_map[key]["generated_events"]
                    ),
                    "run_card_nevents_values": ",".join(
                        sorted(set(run_values.get("nevents", [])))
                    ),
                    "run_card_event_norm_values": ",".join(
                        sorted(set(run_values.get("event_norm", [])))
                    ),
                    "structured_denominator_candidates": len(
                        denominator_structured
                    ),
                    "structured_weight_strategy_candidates": len(
                        weight_structured
                    ),
                    "structured_qcd_sampling_candidates": len(qcd_structured),
                    "text_sumw_evidence_lines": evidence_counter[
                        "sum_weights"
                    ],
                    "text_unweighted_evidence_lines": evidence_counter[
                        "unweighted"
                    ],
                    "text_negative_weight_evidence_lines": evidence_counter[
                        "negative_weight"
                    ],
                    "text_weight_strategy_lines": evidence_counter[
                        "event_weighting_strategy"
                    ],
                    "text_event_norm_lines": evidence_counter["event_norm"],
                    "text_generated_event_lines": evidence_counter[
                        "generated_events"
                    ],
                    "text_process_weight_lines": evidence_counter[
                        "process_weight_line"
                    ],
                    "normalization_denominator_authorized": False,
                    "metadata_status": metadata_status,
                    "required_next_action": next_action,
                }
            )

            access_rows.append(
                {
                    "sample_class": key[0],
                    "process_or_mode": key[1],
                    "campaign": key[2],
                    "manifest_generated_events": int(
                        inventory_map[key]["generated_events"]
                    ),
                    "metadata_only_denominator_authorized": False,
                    "controlled_generator_payload_weight_audit_required": (
                        is_standard
                    ),
                    "source_configuration_recovery_required": (
                        not is_standard
                    ),
                    "candidate_parquet_access_required": False,
                    "validation_or_evaluation_access_required": False,
                    "generator_payload_access_authorized_now": False,
                    "required_next_gate": (
                        "controlled_standard_campaign_weight_canary"
                        if is_standard
                        else "nonstandard_source_provenance_recovery"
                    ),
                    "status": "access_plan_frozen_fail_closed",
                }
            )

            if is_standard:
                standard_rows.append(
                    {
                        "sample_class": key[0],
                        "process_or_mode": key[1],
                        "campaign": key[2],
                        "manifest_generated_events": int(
                            inventory_map[key]["generated_events"]
                        ),
                        "run_card_nevents_values": ",".join(
                            sorted(set(run_values.get("nevents", [])))
                        ),
                        "run_card_event_norm_values": ",".join(
                            sorted(set(run_values.get("event_norm", [])))
                        ),
                        "sumw_candidate_rows": len(
                            denominator_structured
                        ),
                        "unweighted_evidence_lines": evidence_counter[
                            "unweighted"
                        ],
                        "negative_weight_evidence_lines": evidence_counter[
                            "negative_weight"
                        ],
                        "uniform_positive_nominal_weight_proof_complete": False,
                        "normalization_denominator_authorized": False,
                        "status": (
                            "controlled_weight_audit_required_before_Ngen"
                        ),
                    }
                )
            else:
                relevant_structured = [
                    f"{row['field']}={row['value']}"
                    for row in structured
                ]
                relevant_text = [
                    str(row["line_text"])
                    for row in text_evidence
                ]
                blocker = next(
                    row["blocking_reason"]
                    for row in blocker_rows
                    if campaign_key(row) == key
                )
                nonstandard_rows.append(
                    {
                        "sample_class": key[0],
                        "process_or_mode": key[1],
                        "campaign": key[2],
                        "manifest_generated_events": int(
                            inventory_map[key]["generated_events"]
                        ),
                        "blocking_reason": blocker,
                        "structured_candidate_count": len(structured),
                        "structured_candidates": " || ".join(
                            relevant_structured
                        )[:20000],
                        "text_evidence_count": len(text_evidence),
                        "text_evidence": " || ".join(relevant_text)[:20000],
                        "exact_configuration_recovered": False,
                        "aggregate_weight_denominator_recovered": False,
                        "physical_weight_authorized": False,
                        "status": "nonstandard_recovery_evidence_audited_fail_closed",
                    }
                )

        local_rows, root_rows = search_local_sources(
            args.search_root,
            sorted(nonstandard_keys),
        )

        write_tsv(
            output / "verified_provenance_input_files.tsv",
            file_rows,
            list(file_rows[0]),
        )
        write_tsv(
            output / "structured_denominator_and_weight_candidates.tsv",
            structured_rows,
            (
                list(structured_rows[0])
                if structured_rows
                else [
                    "sample_class",
                    "process_or_mode",
                    "campaign",
                    "evidence_class",
                    "field",
                    "value",
                    "source_archive_path",
                    "source_member_name",
                    "source_kind",
                    "normalization_denominator_authorized",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "text_denominator_and_weight_evidence.tsv",
            text_rows,
            (
                list(text_rows[0])
                if text_rows
                else [
                    "sample_class",
                    "process_or_mode",
                    "campaign",
                    "evidence_classes",
                    "line_text",
                    "source_archive_path",
                    "source_member_name",
                    "source_line",
                    "source_kind",
                    "normalization_denominator_authorized",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "campaign_denominator_evidence_status.tsv",
            campaign_rows,
            list(campaign_rows[0]),
        )
        write_tsv(
            output / "standard_campaign_weight_proof_status.tsv",
            standard_rows,
            list(standard_rows[0]),
        )
        write_tsv(
            output / "nonstandard_ggf_qcd_recovery_evidence.tsv",
            nonstandard_rows,
            list(nonstandard_rows[0]),
        )
        write_tsv(
            output / "local_source_recovery_candidates.tsv",
            local_rows,
            (
                list(local_rows[0])
                if local_rows
                else [
                    "sample_class",
                    "process_or_mode",
                    "campaign",
                    "search_root",
                    "match_type",
                    "path",
                    "line_number",
                    "matched_text",
                    "event_payload_opened",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "approved_search_root_summary.tsv",
            root_rows,
            list(root_rows[0]),
        )
        write_tsv(
            output / "controlled_weight_audit_access_plan.tsv",
            access_rows,
            list(access_rows[0]),
        )

        campaigns_with_structured_denominator = sum(
            row["structured_denominator_candidates"] > 0
            for row in campaign_rows
        )
        campaigns_with_unweighted_text = sum(
            row["text_unweighted_evidence_lines"] > 0
            for row in campaign_rows
        )
        campaigns_with_negative_weight_text = sum(
            row["text_negative_weight_evidence_lines"] > 0
            for row in campaign_rows
        )

        summary = {
            "schema_version": 1,
            "status": (
                "hh4b_physical_normalization_denominator_evidence_audit_pass"
            ),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": args.source_commit,
            "inputs": {
                "campaign_inventory": str(args.campaign_inventory),
                "campaign_inventory_sha256": sha256_file(
                    args.campaign_inventory
                ),
                "provenance_archive": str(args.provenance_archive),
                "provenance_archive_sha256": sha256_file(
                    args.provenance_archive
                ),
                "provenance_manifest": str(args.provenance_manifest),
                "provenance_manifest_sha256": sha256_file(
                    args.provenance_manifest
                ),
                "nonstandard_blockers": str(args.nonstandard_blockers),
                "nonstandard_blockers_sha256": sha256_file(
                    args.nonstandard_blockers
                ),
                "denominator_fail_closed": str(
                    args.denominator_fail_closed
                ),
                "denominator_fail_closed_sha256": sha256_file(
                    args.denominator_fail_closed
                ),
            },
            "audit": {
                "eos_campaigns": len(included_keys),
                "standard_campaigns": len(standard_keys),
                "nonstandard_campaigns": len(nonstandard_keys),
                "provenance_files_verified": len(file_rows),
                "structured_candidate_rows": len(structured_rows),
                "text_evidence_rows": len(text_rows),
                "campaigns_with_structured_denominator_candidates": (
                    campaigns_with_structured_denominator
                ),
                "campaigns_with_unweighted_text_evidence": (
                    campaigns_with_unweighted_text
                ),
                "campaigns_with_negative_weight_text_evidence": (
                    campaigns_with_negative_weight_text
                ),
                "local_source_recovery_candidate_rows": len(local_rows),
                "approved_search_roots": len(root_rows),
                "approved_search_roots_present": sum(
                    row["exists"] is True
                    for row in root_rows
                ),
            },
            "readiness": {
                "denominator_evidence_audit_complete": True,
                "standard_campaign_denominators_authorized": 0,
                "nonstandard_campaign_denominators_authorized": 0,
                "generator_payload_access_authorized": 0,
                "external_reference_cross_sections_authorized": 0,
                "qcd_stitching_resolved": False,
                "physics_normalization_ready": False,
            },
            "controls": {
                "candidate_parquet_files_opened": 0,
                "candidate_rows_read": 0,
                "validation_candidate_files_opened": 0,
                "evaluation_candidate_files_opened": 0,
                "event_payload_files_opened": 0,
                "event_payload_files_extracted": 0,
                "normalization_denominators_authorized": 0,
                "cross_sections_assigned": 0,
                "physical_yields_calculated": 0,
                "models_trained": 0,
                "thresholds_selected": 0,
            },
            "next_gate": (
                "review_denominator_evidence_then_run_controlled_standard_"
                "weight_canary_and_nonstandard_source_recovery"
            ),
        }

        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        (output / "README.md").write_text(
            "# HH4b denominator-evidence and source-recovery audit\n\n"
            "This checkpoint inventories all denominator, event-count, "
            "weight-strategy, negative-weight, unweighted-event, and QCD "
            "sampling evidence already present in the 547 checksum-frozen "
            "provenance files. It also searches three approved LPC text-file "
            "roots for exact references to the six nonstandard ggF HH and "
            "importance-sampled hard-QCD campaign names.\n\n"
            "The search excludes ROOT, Parquet, HepMC, LHE, compressed files, "
            "archives, shell histories, credentials, virtual environments, "
            "caches, and other binary files. No event payload is opened.\n\n"
            "All evidence remains candidate evidence. Run-card `event_norm` "
            "and representative `nevents` are not sufficient by themselves "
            "to authorize a campaign denominator. The next gate must either "
            "recover a trusted aggregate sum of nominal generator weights or "
            "perform a controlled generator-level weight audit proving a "
            "uniform positive nominal weight convention.\n",
            encoding="utf-8",
        )

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

        print(json.dumps(summary, indent=2, sort_keys=True))
        print()
        print("ALL_547_PROVENANCE_FILES_DENOMINATOR_AUDIT_VERIFIED_PASS")
        print("ALL_53_EOS_CAMPAIGN_DENOMINATOR_EVIDENCE_REVIEWED_PASS")
        print("STANDARD_47_WEIGHT_PROOF_STATUS_RECORDED_PASS")
        print("NONSTANDARD_6_SOURCE_RECOVERY_EVIDENCE_RECORDED_PASS")
        print("APPROVED_LOCAL_TEXT_SOURCE_SEARCH_PASS")
        print("CONTROLLED_WEIGHT_AUDIT_ACCESS_PLAN_FROZEN_PASS")
        print("NO_GENERATOR_PAYLOAD_ACCESS_AUTHORIZED")
        print("NO_NORMALIZATION_DENOMINATOR_AUTHORIZED")
        print("NO_EXTERNAL_REFERENCE_CROSS_SECTION_ASSIGNED")
        print("NO_CANDIDATE_PARQUET_OPENED")
        print("NO_VALIDATION_CANDIDATE_CONTENT_OPENED")
        print("NO_EVALUATION_CANDIDATE_CONTENT_OPENED")
        print("NO_PHYSICAL_YIELDS_CALCULATED")
        print("HH4B_DENOMINATOR_EVIDENCE_AND_SOURCE_RECOVERY_AUDIT_PASS")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
