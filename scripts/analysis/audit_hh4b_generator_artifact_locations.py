#!/usr/bin/env python3
"""Locate generator provenance artifacts without opening event payloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REQUIRED_ARTIFACT_TYPES = (
    "generator_banner_or_lhe_init",
    "proc_card",
    "run_card",
    "param_card",
    "generator_stdout_or_summary",
    "sum_generator_weights_record",
    "parton_shower_card",
    "delphes_card_and_version",
)

CONDITIONAL_ARTIFACT_TYPES = (
    "importance_sampling_record",
    "stitching_definition",
)

EVENT_PAYLOAD_SUFFIXES = (
    ".root",
    ".parquet",
    ".hepmc",
    ".hepmc.gz",
    ".lhe",
    ".lhe.gz",
)

ARTIFACT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "generator_banner_or_lhe_init",
        (
            "banner",
            "lhe_init",
            "initrwgt",
            "generator_info",
            "generation_metadata",
        ),
    ),
    (
        "proc_card",
        (
            "proc_card",
            "process_card",
            "madgraph_process",
            "mg5_process",
        ),
    ),
    (
        "run_card",
        (
            "run_card",
            "runcard",
            "generation_card",
        ),
    ),
    (
        "param_card",
        (
            "param_card",
            "paramcard",
        ),
    ),
    (
        "generator_stdout_or_summary",
        (
            "generator.log",
            "generation.log",
            "madgraph.log",
            "mg5.log",
            "pythia.log",
            "stdout",
            "job.log",
            "campaign_summary",
            "event_summary",
            "generation_summary",
            "cross_section",
            "xsection",
            "xsec",
        ),
    ),
    (
        "sum_generator_weights_record",
        (
            "sumw",
            "sum_weights",
            "sumweights",
            "generator_weights",
            "weight_summary",
            "runs_tree",
            "lhe_weights",
        ),
    ),
    (
        "parton_shower_card",
        (
            "pythia",
            "shower_card",
            "parton_shower",
        ),
    ),
    (
        "delphes_card_and_version",
        (
            "delphes_card",
            "delphes.tcl",
            "cms_phase",
            "delphes_version",
        ),
    ),
    (
        "importance_sampling_record",
        (
            "importance",
            "adaptive",
            "pthat",
            "sampling",
            "statistical_decision",
        ),
    ),
    (
        "stitching_definition",
        (
            "stitch",
            "overlap",
            "phase_space",
            "exclusive_bin",
        ),
    ),
    (
        "manifest_or_submission_record",
        (
            "manifest",
            "submission",
            "receipt",
            "clusters.tsv",
            "source_artifact",
            "sha256",
            "checksums",
            "review_decision",
            "authorization",
        ),
    ),
)


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
    fieldnames: list[str],
) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def classify_path(path: str) -> list[str]:
    lowered = Path(path).name.lower()
    matches = [
        artifact_type
        for artifact_type, patterns in ARTIFACT_PATTERNS
        if any(pattern in lowered for pattern in patterns)
    ]
    if lowered.endswith(".tar.gz") or lowered.endswith(".tgz"):
        matches.append("bundle_archive")
    if any(lowered.endswith(suffix) for suffix in EVENT_PAYLOAD_SUFFIXES):
        matches.append("event_payload")
    return sorted(set(matches))


def run_command(command: list[str], timeout: int = 120) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "command timed out",
            "status": "timeout",
        }

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "status": "pass" if result.returncode == 0 else "failed",
    }


def parse_stat_size(text: str) -> str:
    for pattern in (
        r"Size:\s*(\d+)",
        r"size\s+(\d+)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def parse_ls_path(line: str) -> str:
    fields = line.split()
    if not fields:
        return ""
    return fields[-1]


def candidate_local_directories(
    store: Path,
    repo: Path,
    campaign: str,
) -> list[Path]:
    directories = [
        store / "condor_submit" / campaign,
        store / "condor_return" / campaign,
        store / "production" / campaign,
        store / "campaigns" / campaign,
        store / "manifests" / campaign,
        store / "receipts" / campaign,
        repo / "outputs" / "agent_runs" / campaign,
        repo / "docs" / "checkpoints" / campaign,
    ]

    for parent in (
        repo / "outputs" / "agent_runs",
        repo / "docs" / "checkpoints",
    ):
        if parent.is_dir() and campaign != "parquet":
            directories.extend(
                path
                for path in parent.glob(f"*{campaign}*")
                if path.is_dir()
            )

    unique: dict[str, Path] = {}
    for directory in directories:
        unique[str(directory.resolve())] = directory
    return list(unique.values())


def list_local_files(directory: Path, maximum: int = 20000) -> list[Path]:
    output: list[Path] = []
    if not directory.is_dir():
        return output

    for root, _, files in os.walk(directory):
        for name in files:
            output.append(Path(root) / name)
            if len(output) > maximum:
                raise RuntimeError(
                    f"local artifact directory exceeds safe file cap: {directory}"
                )
    return output


def validate_campaign_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 54:
        raise RuntimeError(f"campaign inventory rows={len(rows)}, expected 54")

    required = {
        "sample_class",
        "process_or_mode",
        "campaign",
        "members",
        "generated_events",
        "source_locator_example",
        "status",
    }
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"campaign inventory missing fields: {sorted(missing)}")

    if sum(int(row["members"]) for row in rows) != 630:
        raise RuntimeError("campaign inventory does not account for 630 members")
    if sum(int(row["generated_events"]) for row in rows) != 5_200_000:
        raise RuntimeError("campaign inventory does not account for 5.2M events")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-inventory", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--eos-host", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"output already exists: {output_dir}")

    campaign_rows = read_tsv(args.campaign_inventory)
    validate_campaign_rows(campaign_rows)

    output_dir.mkdir(parents=True)
    try:
        representative_rows: list[dict[str, object]] = []
        remote_directory_rows: list[dict[str, object]] = []
        local_location_rows: list[dict[str, object]] = []
        coverage_sources: dict[
            tuple[str, str, str], dict[str, list[str]]
        ] = defaultdict(lambda: defaultdict(list))

        remote_directory_cache: dict[str, dict[str, object]] = {}
        remote_stat_passes = 0
        remote_rows = 0
        local_rows = 0

        for campaign_row in campaign_rows:
            sample_class = campaign_row["sample_class"]
            process = campaign_row["process_or_mode"]
            campaign = campaign_row["campaign"]
            locator = campaign_row["source_locator_example"]
            key = (sample_class, process, campaign)

            if locator.startswith("/store/"):
                source_type = "eos_bundle"
                remote_rows += 1
                stat_result = run_command(
                    ["xrdfs", args.eos_host, "stat", locator],
                    timeout=120,
                )
                stat_pass = stat_result["returncode"] == 0
                if stat_pass:
                    remote_stat_passes += 1
                size_bytes = parse_stat_size(
                    str(stat_result["stdout"])
                    + "\n"
                    + str(stat_result["stderr"])
                )
                parent = str(Path(locator).parent)

                if parent not in remote_directory_cache:
                    listing = run_command(
                        ["xrdfs", args.eos_host, "ls", "-l", parent],
                        timeout=180,
                    )
                    if listing["returncode"] != 0:
                        raise RuntimeError(
                            "EOS campaign directory listing failed:\n"
                            f"directory={parent}\n"
                            f"stderr={listing['stderr']}"
                        )
                    remote_directory_cache[parent] = listing

                    for line in str(listing["stdout"]).splitlines():
                        listed_path = parse_ls_path(line)
                        if not listed_path:
                            continue
                        artifact_types = classify_path(listed_path)
                        remote_directory_rows.append(
                            {
                                "eos_directory": parent,
                                "listed_path": listed_path,
                                "artifact_types": ",".join(artifact_types),
                                "event_payload": (
                                    "event_payload" in artifact_types
                                ),
                                "bundle_archive": (
                                    "bundle_archive" in artifact_types
                                ),
                                "raw_listing_line": line,
                            }
                        )

                for remote_entry in remote_directory_rows:
                    if remote_entry["eos_directory"] != parent:
                        continue
                    listed_path = str(remote_entry["listed_path"])
                    for artifact_type in classify_path(listed_path):
                        if artifact_type not in {
                            "bundle_archive",
                            "event_payload",
                        }:
                            coverage_sources[key][artifact_type].append(
                                "eos_sidecar:" + listed_path
                            )

                representative_rows.append(
                    {
                        "sample_class": sample_class,
                        "process_or_mode": process,
                        "campaign": campaign,
                        "source_type": source_type,
                        "representative_locator": locator,
                        "representative_exists": stat_pass,
                        "representative_size_bytes": size_bytes,
                        "metadata_access_method": "xrdfs_stat_only",
                        "event_payload_opened": False,
                        "bundle_downloaded": False,
                        "bundle_content_indexed": False,
                        "status": (
                            "representative_bundle_stat_pass"
                            if stat_pass
                            else "representative_bundle_stat_failed"
                        ),
                        "diagnostic": (
                            str(stat_result["stderr"]).strip()
                            or "none"
                        ),
                    }
                )
            else:
                source_type = "local_candidate_reference"
                local_rows += 1
                local_path = Path(locator)
                exists = local_path.is_file()
                representative_rows.append(
                    {
                        "sample_class": sample_class,
                        "process_or_mode": process,
                        "campaign": campaign,
                        "source_type": source_type,
                        "representative_locator": locator,
                        "representative_exists": exists,
                        "representative_size_bytes": (
                            local_path.stat().st_size if exists else ""
                        ),
                        "metadata_access_method": "local_stat_only",
                        "event_payload_opened": False,
                        "bundle_downloaded": False,
                        "bundle_content_indexed": False,
                        "status": (
                            "legacy_local_reference_stat_pass"
                            if exists
                            else "legacy_local_reference_stat_failed"
                        ),
                        "diagnostic": (
                            "Legacy ttbar source requires original production "
                            "artifact recovery; candidate file was not opened."
                        ),
                    }
                )

            local_directories = candidate_local_directories(
                args.store.resolve(),
                args.repository.resolve(),
                campaign,
            )

            for directory in local_directories:
                if not directory.is_dir():
                    continue
                for artifact in list_local_files(directory):
                    artifact_types = classify_path(str(artifact))
                    local_location_rows.append(
                        {
                            "sample_class": sample_class,
                            "process_or_mode": process,
                            "campaign": campaign,
                            "search_root": str(directory),
                            "artifact_path": str(artifact),
                            "bytes": artifact.stat().st_size,
                            "artifact_types": ",".join(artifact_types),
                            "event_payload": "event_payload" in artifact_types,
                            "content_opened": False,
                        }
                    )
                    for artifact_type in artifact_types:
                        if artifact_type not in {
                            "bundle_archive",
                            "event_payload",
                        }:
                            coverage_sources[key][artifact_type].append(
                                "local:" + str(artifact)
                            )

        failed_remote = [
            row
            for row in representative_rows
            if row["source_type"] == "eos_bundle"
            and row["representative_exists"] is not True
        ]
        if failed_remote:
            raise RuntimeError(
                "one or more representative EOS bundles failed stat:\n"
                + "\n".join(
                    str(row["representative_locator"])
                    for row in failed_remote
                )
            )

        coverage_rows: list[dict[str, object]] = []
        unresolved_rows: list[dict[str, object]] = []
        bundle_plan_rows: list[dict[str, object]] = []

        for campaign_row in campaign_rows:
            key = (
                campaign_row["sample_class"],
                campaign_row["process_or_mode"],
                campaign_row["campaign"],
            )
            source_map = coverage_sources[key]
            required = list(REQUIRED_ARTIFACT_TYPES)

            if (
                campaign_row["process_or_mode"].startswith("qcd_")
                or "importance" in campaign_row["campaign"].lower()
            ):
                required.extend(CONDITIONAL_ARTIFACT_TYPES)

            located_required = [
                artifact_type
                for artifact_type in required
                if source_map.get(artifact_type)
            ]
            unresolved = [
                artifact_type
                for artifact_type in required
                if not source_map.get(artifact_type)
            ]

            coverage_rows.append(
                {
                    "sample_class": campaign_row["sample_class"],
                    "process_or_mode": campaign_row["process_or_mode"],
                    "campaign": campaign_row["campaign"],
                    "required_artifact_types": ",".join(required),
                    "located_artifact_types": ",".join(located_required),
                    "unresolved_artifact_types": ",".join(unresolved),
                    "located_required_count": len(located_required),
                    "required_count": len(required),
                    "sidecar_provenance_complete": len(unresolved) == 0,
                    "physics_normalization_ready": False,
                    "status": (
                        "sidecar_inventory_complete_requires_content_review"
                        if not unresolved
                        else "representative_bundle_content_index_required"
                    ),
                }
            )

            for artifact_type in unresolved:
                unresolved_rows.append(
                    {
                        "sample_class": campaign_row["sample_class"],
                        "process_or_mode": campaign_row["process_or_mode"],
                        "campaign": campaign_row["campaign"],
                        "artifact_type": artifact_type,
                        "blocking_reason": (
                            "No matching sidecar path was located. A representative "
                            "bundle content index or legacy production recovery is required."
                        ),
                        "resolved": False,
                    }
                )

            representative = next(
                row
                for row in representative_rows
                if (
                    row["sample_class"],
                    row["process_or_mode"],
                    row["campaign"],
                )
                == key
            )
            bundle_plan_rows.append(
                {
                    "sample_class": campaign_row["sample_class"],
                    "process_or_mode": campaign_row["process_or_mode"],
                    "campaign": campaign_row["campaign"],
                    "representative_locator": representative[
                        "representative_locator"
                    ],
                    "representative_size_bytes": representative[
                        "representative_size_bytes"
                    ],
                    "source_type": representative["source_type"],
                    "content_index_required": bool(unresolved),
                    "event_payload_extraction_authorized": False,
                    "candidate_content_access_authorized": False,
                    "next_action": (
                        "stream_or_copy_one_representative_bundle_to_temporary_scratch_"
                        "then_list_archive_members_only"
                        if representative["source_type"] == "eos_bundle"
                        else
                        "recover_original_legacy_ttbar_generation_directory"
                    ),
                }
            )

        write_tsv(
            output_dir / "representative_locator_status.tsv",
            representative_rows,
            list(representative_rows[0]),
        )
        write_tsv(
            output_dir / "eos_campaign_directory_inventory.tsv",
            remote_directory_rows,
            list(remote_directory_rows[0]),
        )
        write_tsv(
            output_dir / "local_artifact_location_inventory.tsv",
            local_location_rows,
            (
                list(local_location_rows[0])
                if local_location_rows
                else [
                    "sample_class",
                    "process_or_mode",
                    "campaign",
                    "search_root",
                    "artifact_path",
                    "bytes",
                    "artifact_types",
                    "event_payload",
                    "content_opened",
                ]
            ),
        )
        write_tsv(
            output_dir / "campaign_artifact_coverage.tsv",
            coverage_rows,
            list(coverage_rows[0]),
        )
        write_tsv(
            output_dir / "unresolved_artifact_types.tsv",
            unresolved_rows,
            list(unresolved_rows[0]),
        )
        write_tsv(
            output_dir / "representative_bundle_index_plan.tsv",
            bundle_plan_rows,
            list(bundle_plan_rows[0]),
        )

        artifact_type_counts = Counter()
        for sources in coverage_sources.values():
            for artifact_type, paths in sources.items():
                if paths:
                    artifact_type_counts[artifact_type] += 1

        summary = {
            "schema_version": 1,
            "status": (
                "hh4b_physical_normalization_generator_artifact_location_audit_pass"
            ),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": args.source_commit,
            "campaign_inventory": str(args.campaign_inventory),
            "campaign_inventory_sha256": sha256_file(args.campaign_inventory),
            "source_registry": str(args.source_registry),
            "source_registry_sha256": sha256_file(args.source_registry),
            "campaigns": {
                "rows": len(campaign_rows),
                "remote_eos_rows": remote_rows,
                "local_legacy_rows": local_rows,
                "representative_remote_stat_passes": remote_stat_passes,
                "unique_eos_directories_listed": len(remote_directory_cache),
            },
            "located_metadata": {
                "eos_directory_entries": len(remote_directory_rows),
                "local_file_entries": len(local_location_rows),
                "campaigns_with_complete_sidecars": sum(
                    row["sidecar_provenance_complete"] is True
                    for row in coverage_rows
                ),
                "campaigns_requiring_bundle_content_index": sum(
                    row["content_index_required"] is True
                    for row in bundle_plan_rows
                ),
                "artifact_type_campaign_counts": dict(
                    sorted(artifact_type_counts.items())
                ),
            },
            "readiness": {
                "cross_sections_assigned": 0,
                "sum_generator_weights_resolved": 0,
                "generator_artifact_contents_reviewed": 0,
                "physics_normalization_ready": False,
            },
            "controls": {
                "candidate_parquet_files_opened": 0,
                "candidate_rows_read": 0,
                "event_payload_files_opened": 0,
                "bundles_downloaded": 0,
                "bundle_contents_indexed": 0,
                "validation_candidate_files_opened": 0,
                "evaluation_candidate_files_opened": 0,
                "cross_sections_assigned": 0,
                "physical_yields_calculated": 0,
                "models_trained": 0,
                "thresholds_selected": 0,
            },
            "next_gate": (
                "index_one_representative_bundle_per_campaign_without_"
                "extracting_event_payloads"
            ),
        }

        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        (output_dir / "README.md").write_text(
            """# HH4b generator-artifact location audit

This checkpoint verifies the availability of representative production sources and
locates provenance sidecars by filename and filesystem metadata only.

It does not open ROOT, Parquet, HepMC, or LHE event payloads. It does not download or
extract bundle archives. It does not assign cross sections or calculate physical yields.

EOS campaign directories are listed, representative bundles are checked with `xrdfs
stat`, and known local campaign submission/return directories are inventoried. Missing
artifact classes are converted into a deterministic representative-bundle indexing plan.

The next gate is to inspect the member names of one representative archive per
process-campaign, extracting only small provenance files after an explicit allowlist is
frozen.
""",
            encoding="utf-8",
        )

        checksum_path = output_dir / "SHA256SUMS"
        products = sorted(
            item
            for item in output_dir.iterdir()
            if item.is_file() and item != checksum_path
        )
        checksum_path.write_text(
            "\n".join(
                f"{sha256_file(item)}  {item.name}"
                for item in products
            )
            + "\n",
            encoding="utf-8",
        )

        print(json.dumps(summary, indent=2, sort_keys=True))
        print()
        print("CAMPAIGN_INVENTORY_54_ROWS_PASS")
        print("REPRESENTATIVE_EOS_BUNDLE_STAT_PASS")
        print("EOS_CAMPAIGN_DIRECTORY_METADATA_LISTING_PASS")
        print("LOCAL_CAMPAIGN_ARTIFACT_LOCATION_AUDIT_PASS")
        print("REPRESENTATIVE_BUNDLE_INDEX_PLAN_FROZEN_PASS")
        print("NO_BUNDLES_DOWNLOADED")
        print("NO_BUNDLE_CONTENTS_INDEXED")
        print("NO_EVENT_PAYLOAD_OPENED")
        print("NO_CANDIDATE_PARQUET_OPENED")
        print("NO_CROSS_SECTIONS_ASSIGNED")
        print("HH4B_GENERATOR_ARTIFACT_LOCATION_AUDIT_PASS")
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
