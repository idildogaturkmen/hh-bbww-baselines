#!/usr/bin/env python3
"""Search for legacy ttbar generation provenance without opening event payloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ANCHORS = (
    "ttbar_100k_shard000",
    "ttbar_100k",
    "ttbar_100k_shard000_hh4b_candidates.parquet",
)

REQUIRED_ARTIFACT_TYPES = (
    "generator_banner",
    "proc_card",
    "run_card",
    "parton_shower_card",
    "generator_log_or_summary",
    "normalization_denominator_evidence",
)

OPTIONAL_ARTIFACT_TYPES = (
    "generation_driver",
    "delphes_card_or_log",
    "provenance_metadata",
    "submission_or_receipt",
)

PAYLOAD_SUFFIXES = (
    ".root",
    ".root.gz",
    ".parquet",
    ".hepmc",
    ".hepmc.gz",
    ".hepmc3",
    ".hepmc3.gz",
    ".lhe",
    ".lhe.gz",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
)

TEXT_SUFFIXES = {
    "",
    ".bash",
    ".cfg",
    ".conf",
    ".csv",
    ".dat",
    ".err",
    ".ini",
    ".jdl",
    ".json",
    ".list",
    ".log",
    ".md",
    ".out",
    ".py",
    ".sh",
    ".sub",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}

EXCLUDED_DIRECTORY_NAMES = {
    ".cache",
    ".config",
    ".git",
    ".globus",
    ".local",
    ".ssh",
    "__pycache__",
    "node_modules",
    "venv",
    "venvs",
}

EXCLUDED_FILE_NAMES = {
    ".bash_history",
    ".python_history",
    ".lesshst",
    ".viminfo",
}

MAX_TEXT_BYTES = 5 * 1024 * 1024
MAX_FILES_SCANNED = 500_000


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


def is_payload(path: Path) -> bool:
    lowered = path.name.lower()
    return any(lowered.endswith(suffix) for suffix in PAYLOAD_SUFFIXES)


def is_text_candidate(path: Path, size: int) -> bool:
    return (
        size <= MAX_TEXT_BYTES
        and not is_payload(path)
        and path.name not in EXCLUDED_FILE_NAMES
        and path.suffix.lower() in TEXT_SUFFIXES
    )


def artifact_types(path: Path, text: str = "") -> list[str]:
    name = path.name.lower()
    combined = (name + "\n" + text[:200_000].lower())

    types: set[str] = set()

    if "banner" in name or "generation_banner" in combined:
        types.add("generator_banner")
    if "proc_card" in name or "process_card" in name:
        types.add("proc_card")
    if "run_card" in name or "runcard" in name:
        types.add("run_card")
    if (
        "pythia" in name
        or "shower_card" in name
        or "parton_shower" in name
    ):
        types.add("parton_shower_card")
    if (
        any(token in name for token in ("mg5.log", "madgraph.log", "generator.log"))
        or "cross section" in combined
        or "cross-section" in combined
    ):
        types.add("generator_log_or_summary")
    if (
        any(
            token in name
            for token in (
                "sumw",
                "sum_weights",
                "sumweights",
                "weight_summary",
                "event_summary",
                "generation_summary",
            )
        )
        or "sum of weights" in combined
        or "sum_weights" in combined
        or "nevents" in combined
        or "generated_events" in combined
    ):
        types.add("normalization_denominator_evidence")
    if path.suffix.lower() in {".sh", ".py"}:
        types.add("generation_driver")
    if "delphes" in name:
        types.add("delphes_card_or_log")
    if (
        "provenance" in name
        or "generator.json" in name
        or "generation.json" in name
    ):
        types.add("provenance_metadata")
    if any(
        token in name
        for token in (
            "manifest",
            "submission",
            "receipt",
            "cluster",
            "condor",
        )
    ):
        types.add("submission_or_receipt")

    return sorted(types)


def should_prune(path: Path) -> bool:
    return (
        path.name in EXCLUDED_DIRECTORY_NAMES
        or path.name.startswith(".nfs")
    )


def path_contains_anchor(path: Path) -> list[str]:
    lowered = str(path).lower()
    return [anchor for anchor in ANCHORS if anchor in lowered]


def read_text_safely(path: Path, size: int) -> tuple[str, bool]:
    if not is_text_candidate(path, size):
        return "", False
    data = path.read_bytes()
    if b"\x00" in data:
        return "", False
    return data.decode("utf-8", errors="replace"), True


def direct_legacy_directory(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    return any(
        any(anchor in part for anchor in ANCHORS[:2])
        for part in lowered_parts[:-1]
    )


def validate_plan(plan_rows: list[dict[str, str]]) -> dict[str, str]:
    if len(plan_rows) != 1:
        raise RuntimeError(
            f"legacy recovery-plan rows={len(plan_rows)}, expected 1"
        )
    row = plan_rows[0]
    if row["process_or_mode"] != "ttbar_inclusive":
        raise RuntimeError("legacy recovery plan process mismatch")
    if row["legacy_campaign"] != "parquet":
        raise RuntimeError("legacy recovery plan campaign mismatch")
    if row["candidate_content_access_authorized"] != "False":
        raise RuntimeError("candidate access unexpectedly authorized")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-plan", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")

    plan = validate_plan(read_tsv(args.recovery_plan))
    candidate_path = Path(plan["candidate_reference"]).resolve()

    roots = [
        Path(item).resolve()
        for item in plan["search_roots"].split(",")
        if item.strip()
    ]
    if args.repository.resolve() not in roots:
        roots.append(args.repository.resolve())

    if candidate_path.suffix.lower() != ".parquet":
        raise RuntimeError("legacy candidate reference is not parquet")

    output.mkdir(parents=True)
    try:
        root_rows: list[dict[str, object]] = []
        filename_rows: list[dict[str, object]] = []
        directory_rows: list[dict[str, object]] = []
        text_rows: list[dict[str, object]] = []
        associated_rows: list[dict[str, object]] = []

        total_files = 0
        total_dirs = 0
        text_files_opened = 0
        bytes_text_scanned = 0
        payload_files_stat_only = 0

        seen_files: set[Path] = set()

        for root in roots:
            root_exists = root.is_dir()
            root_files = 0
            root_dirs = 0
            root_text = 0
            root_payloads = 0

            if root_exists:
                for current, dirnames, filenames in os.walk(
                    root,
                    topdown=True,
                    followlinks=False,
                ):
                    current_path = Path(current)
                    dirnames[:] = [
                        name
                        for name in dirnames
                        if not should_prune(current_path / name)
                    ]
                    root_dirs += len(dirnames)
                    total_dirs += len(dirnames)

                    for dirname in dirnames:
                        directory = current_path / dirname
                        anchors = path_contains_anchor(directory)
                        if anchors:
                            directory_rows.append(
                                {
                                    "search_root": str(root),
                                    "directory_path": str(directory),
                                    "matched_anchors": ",".join(anchors),
                                    "association_tier": "exact_anchor_directory",
                                    "content_opened": False,
                                    "status": "legacy_named_directory_located",
                                }
                            )

                    for filename in filenames:
                        path = current_path / filename
                        if path in seen_files:
                            continue
                        seen_files.add(path)

                        if total_files >= MAX_FILES_SCANNED:
                            raise RuntimeError(
                                f"safe file cap exceeded: {MAX_FILES_SCANNED}"
                            )

                        try:
                            stat = path.stat()
                        except (FileNotFoundError, PermissionError, OSError):
                            continue
                        if not path.is_file():
                            continue

                        total_files += 1
                        root_files += 1
                        size = stat.st_size
                        payload = is_payload(path)
                        if payload:
                            payload_files_stat_only += 1
                            root_payloads += 1

                        anchors_in_path = path_contains_anchor(path)
                        if anchors_in_path:
                            filename_rows.append(
                                {
                                    "search_root": str(root),
                                    "path": str(path),
                                    "bytes": size,
                                    "matched_anchors": ",".join(anchors_in_path),
                                    "event_payload": payload,
                                    "content_opened": False,
                                    "sha256": (
                                        "not_computed_event_payload"
                                        if payload
                                        else sha256_file(path)
                                        if size <= MAX_TEXT_BYTES
                                        else "not_computed_size_limit"
                                    ),
                                    "status": "exact_legacy_filename_match",
                                }
                            )

                        text = ""
                        opened = False
                        if not payload:
                            try:
                                text, opened = read_text_safely(path, size)
                            except (PermissionError, OSError, UnicodeError):
                                text, opened = "", False

                        if opened:
                            text_files_opened += 1
                            root_text += 1
                            bytes_text_scanned += size
                            lowered = text.lower()
                            matches = [
                                anchor
                                for anchor in ANCHORS
                                if anchor in lowered
                            ]
                            if str(candidate_path).lower() in lowered:
                                matches.append("exact_candidate_absolute_path")
                            matches = sorted(set(matches))

                            if matches:
                                line_numbers: list[int] = []
                                for index, line in enumerate(
                                    text.splitlines(),
                                    start=1,
                                ):
                                    lowered_line = line.lower()
                                    if (
                                        any(
                                            anchor in lowered_line
                                            for anchor in ANCHORS
                                        )
                                        or str(candidate_path).lower()
                                        in lowered_line
                                    ):
                                        line_numbers.append(index)
                                text_rows.append(
                                    {
                                        "search_root": str(root),
                                        "path": str(path),
                                        "bytes": size,
                                        "matched_anchors": ",".join(matches),
                                        "matching_line_numbers": ",".join(
                                            str(item)
                                            for item in line_numbers[:100]
                                        ),
                                        "matching_line_count": len(line_numbers),
                                        "sha256": sha256_file(path),
                                        "event_payload": False,
                                        "content_opened": True,
                                        "status": "exact_legacy_text_reference",
                                    }
                                )

                        types = artifact_types(path, text)
                        association = ""
                        evidence = ""

                        if anchors_in_path and not payload:
                            association = "direct_anchor_filename"
                            evidence = ",".join(anchors_in_path)
                        elif direct_legacy_directory(path) and not payload:
                            association = "inside_exact_anchor_directory"
                            evidence = "ancestor_directory_contains_legacy_anchor"
                        elif opened and any(
                            anchor in text.lower()
                            for anchor in ANCHORS
                        ):
                            association = "direct_anchor_text_reference"
                            evidence = "file_content_contains_legacy_anchor"
                        elif opened and str(candidate_path).lower() in text.lower():
                            association = "direct_candidate_path_reference"
                            evidence = "file_content_contains_exact_candidate_path"

                        if association and types:
                            associated_rows.append(
                                {
                                    "search_root": str(root),
                                    "path": str(path),
                                    "bytes": size,
                                    "sha256": sha256_file(path),
                                    "association_tier": association,
                                    "association_evidence": evidence,
                                    "artifact_types": ",".join(types),
                                    "event_payload": False,
                                    "content_opened": opened,
                                    "recovery_authorized": False,
                                    "status": (
                                        "candidate_legacy_provenance_"
                                        "requires_manual_review"
                                    ),
                                }
                            )

            root_rows.append(
                {
                    "search_root": str(root),
                    "exists": root_exists,
                    "directories_visited": root_dirs,
                    "files_stat_checked": root_files,
                    "small_text_files_opened": root_text,
                    "event_payload_files_stat_only": root_payloads,
                    "status": (
                        "search_root_complete"
                        if root_exists
                        else "search_root_absent"
                    ),
                }
            )

        coverage = Counter()
        direct_coverage = Counter()
        for row in associated_rows:
            for artifact_type in str(row["artifact_types"]).split(","):
                coverage[artifact_type] += 1
                if row["association_tier"] in {
                    "direct_anchor_filename",
                    "inside_exact_anchor_directory",
                }:
                    direct_coverage[artifact_type] += 1

        coverage_rows = []
        for artifact_type in REQUIRED_ARTIFACT_TYPES + OPTIONAL_ARTIFACT_TYPES:
            coverage_rows.append(
                {
                    "artifact_type": artifact_type,
                    "required": artifact_type in REQUIRED_ARTIFACT_TYPES,
                    "associated_candidates": coverage.get(artifact_type, 0),
                    "strong_path_associated_candidates": direct_coverage.get(
                        artifact_type,
                        0,
                    ),
                    "resolved": False,
                    "resolution_reason": (
                        "Candidate paths require manual identity and content review; "
                        "no artifact is accepted automatically."
                    ),
                    "status": (
                        "candidate_evidence_located"
                        if coverage.get(artifact_type, 0) > 0
                        else "blocking_not_located"
                    ),
                }
            )

        missing_types = [
            artifact_type
            for artifact_type in REQUIRED_ARTIFACT_TYPES
            if coverage.get(artifact_type, 0) == 0
        ]
        candidate_types = [
            artifact_type
            for artifact_type in REQUIRED_ARTIFACT_TYPES
            if coverage.get(artifact_type, 0) > 0
        ]

        if not candidate_types:
            search_outcome = "legacy_provenance_not_located"
            next_action = (
                "exclude_or_regenerate_legacy_ttbar_before_physics_normalization"
            )
        elif missing_types:
            search_outcome = "legacy_provenance_partially_located"
            next_action = (
                "manually_review_candidates_and_recover_missing_artifacts"
            )
        else:
            search_outcome = "legacy_provenance_candidate_set_located"
            next_action = (
                "manually_review_and_freeze_exact_legacy_artifact_allowlist"
            )

        decision_rows = [
            {
                "process_or_mode": "ttbar_inclusive",
                "legacy_campaign": "parquet",
                "candidate_reference": str(candidate_path),
                "candidate_content_access_authorized": False,
                "search_outcome": search_outcome,
                "required_artifact_types": ",".join(REQUIRED_ARTIFACT_TYPES),
                "candidate_artifact_types_located": ",".join(candidate_types),
                "required_artifact_types_missing": (
                    ",".join(missing_types)
                    if missing_types
                    else "none_at_filename_or_reference_level"
                ),
                "legacy_sample_physics_weight_authorized": False,
                "legacy_sample_physical_yield_authorized": False,
                "next_action": next_action,
                "status": "fail_closed_manual_review_required",
            }
        ]

        write_tsv(
            output / "search_root_summary.tsv",
            root_rows,
            list(root_rows[0]),
        )
        write_tsv(
            output / "legacy_named_directory_matches.tsv",
            directory_rows,
            (
                list(directory_rows[0])
                if directory_rows
                else [
                    "search_root",
                    "directory_path",
                    "matched_anchors",
                    "association_tier",
                    "content_opened",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "legacy_filename_matches.tsv",
            filename_rows,
            (
                list(filename_rows[0])
                if filename_rows
                else [
                    "search_root",
                    "path",
                    "bytes",
                    "matched_anchors",
                    "event_payload",
                    "content_opened",
                    "sha256",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "legacy_text_reference_matches.tsv",
            text_rows,
            (
                list(text_rows[0])
                if text_rows
                else [
                    "search_root",
                    "path",
                    "bytes",
                    "matched_anchors",
                    "matching_line_numbers",
                    "matching_line_count",
                    "sha256",
                    "event_payload",
                    "content_opened",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "legacy_associated_artifact_candidates.tsv",
            associated_rows,
            (
                list(associated_rows[0])
                if associated_rows
                else [
                    "search_root",
                    "path",
                    "bytes",
                    "sha256",
                    "association_tier",
                    "association_evidence",
                    "artifact_types",
                    "event_payload",
                    "content_opened",
                    "recovery_authorized",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "legacy_artifact_coverage.tsv",
            coverage_rows,
            list(coverage_rows[0]),
        )
        write_tsv(
            output / "legacy_normalization_decision_gate.tsv",
            decision_rows,
            list(decision_rows[0]),
        )

        summary = {
            "schema_version": 1,
            "status": (
                "hh4b_physical_normalization_legacy_ttbar_search_pass"
            ),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": args.source_commit,
            "recovery_plan": str(args.recovery_plan),
            "recovery_plan_sha256": sha256_file(args.recovery_plan),
            "search": {
                "roots": len(roots),
                "roots_existing": sum(
                    row["exists"] is True
                    for row in root_rows
                ),
                "directories_visited": total_dirs,
                "files_stat_checked": total_files,
                "small_text_files_opened": text_files_opened,
                "text_bytes_scanned": bytes_text_scanned,
                "event_payload_files_stat_only": payload_files_stat_only,
                "legacy_named_directories": len(directory_rows),
                "legacy_filename_matches": len(filename_rows),
                "legacy_text_reference_matches": len(text_rows),
                "associated_artifact_candidates": len(associated_rows),
                "candidate_required_artifact_types": candidate_types,
                "missing_required_artifact_types": missing_types,
                "outcome": search_outcome,
            },
            "readiness": {
                "legacy_ttbar_candidate_content_opened": False,
                "legacy_ttbar_provenance_review_complete": False,
                "legacy_ttbar_physics_weight_authorized": False,
                "eos_provenance_ready_for_parsing": True,
                "cross_sections_assigned": 0,
                "sum_generator_weights_resolved": 0,
                "physics_normalization_ready": False,
            },
            "controls": {
                "candidate_parquet_files_opened": 0,
                "candidate_rows_read": 0,
                "validation_candidate_files_opened": 0,
                "evaluation_candidate_files_opened": 0,
                "event_payload_files_opened": 0,
                "event_payload_files_extracted": 0,
                "legacy_artifact_files_copied": 0,
                "cross_sections_assigned": 0,
                "physical_yields_calculated": 0,
                "models_trained": 0,
                "thresholds_selected": 0,
            },
            "next_gate": next_action,
        }

        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        (output / "README.md").write_text(
            "# Legacy ttbar generation-provenance search\n\n"
            "This checkpoint searches approved local roots for the exact "
            "`ttbar_100k_shard000` legacy source identity and associated "
            "small text metadata. Event payloads are stat-checked only and "
            "never opened.\n\n"
            "All located paths remain candidates pending manual identity "
            "review. No generic ttbar card is automatically attributed to "
            "the legacy sample. The legacy sample remains unauthorized for "
            "physical weighting unless an exact provenance set is frozen.\n\n"
            f"Search outcome: `{search_outcome}`.\n\n"
            "The EOS campaigns remain authorized for generator-provenance "
            "parsing in the next phase, but the final physical registry must "
            "either resolve, regenerate, replace, or exclude this legacy "
            "campaign.\n",
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
        print("LEGACY_TTBAR_APPROVED_ROOT_SEARCH_PASS")
        print("LEGACY_TTBAR_EXACT_ANCHOR_SEARCH_PASS")
        print("LEGACY_TTBAR_ARTIFACT_COVERAGE_GATE_PASS")
        print("LEGACY_TTBAR_FAIL_CLOSED_DECISION_RECORDED_PASS")
        print("NO_GENERIC_TTBAR_ARTIFACT_AUTOATTRIBUTED")
        print("NO_LEGACY_CANDIDATE_PARQUET_OPENED")
        print("NO_EVENT_PAYLOAD_CONTENT_OPENED")
        print("NO_CROSS_SECTIONS_ASSIGNED")
        print("NO_PHYSICAL_YIELDS_CALCULATED")
        print("HH4B_LEGACY_TTBAR_PROVENANCE_SEARCH_PASS")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
