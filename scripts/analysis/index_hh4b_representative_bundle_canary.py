#!/usr/bin/env python3
"""Index six representative HH4b production bundles without extraction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


CANARIES = (
    ("signal", "ggf_hh4b", "ggf_hh4b_ml_ext20k_frozen_v2_20260716"),
    ("signal", "vbf_hh4b", "vbf_hh4b_sm_pilot10k_20260722_v1"),
    (
        "background",
        "ttbar_inclusive",
        "unified_background_5m_wavea_proven111_20260720_v1",
    ),
    (
        "background",
        "qcd_bbbb_general",
        "unified_background_5m_wavea_proven111_20260720_v1",
    ),
    (
        "background",
        "qcd_hardqcd",
        "qcd_hardqcd_importance_adaptive500k_phys_20260717_0022",
    ),
    (
        "background",
        "ttz_zbb",
        "unified_background_5m_waveb_priority_pilots10k_20260721_v1",
    ),
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
)

ARTIFACT_PATTERNS = {
    "generator_banner_or_lhe_init": (
        "banner",
        "lhe_init",
        "initrwgt",
        "generation_metadata",
        "generator_info",
    ),
    "proc_card": ("proc_card", "process_card", "mg5_process"),
    "run_card": ("run_card", "runcard", "generation_card"),
    "param_card": ("param_card", "paramcard"),
    "generator_stdout_or_summary": (
        "generator.log",
        "generation.log",
        "madgraph.log",
        "mg5.log",
        "pythia.log",
        "job.log",
        "stdout",
        "summary",
        "cross_section",
        "xsection",
        "xsec",
    ),
    "sum_generator_weights_record": (
        "sumw",
        "sum_weights",
        "sumweights",
        "generator_weights",
        "weight_summary",
        "lhe_weights",
    ),
    "parton_shower_card": ("pythia", "shower_card", "parton_shower"),
    "delphes_card_and_version": (
        "delphes_card",
        "delphes.tcl",
        "cms_phase",
        "delphes_version",
    ),
    "importance_sampling_record": (
        "importance",
        "adaptive",
        "pthat",
        "sampling",
        "statistical_decision",
    ),
    "stitching_definition": (
        "stitch",
        "overlap",
        "phase_space",
        "exclusive_bin",
    ),
    "manifest_or_receipt": (
        "manifest",
        "submission",
        "receipt",
        "sha256",
        "checksum",
        "authorization",
        "environment",
        "versions",
    ),
}

ALLOWLIST_MAX_BYTES = 5 * 1024 * 1024


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
    rows: list[dict[str, object]],
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


def classify(name: str) -> tuple[bool, list[str]]:
    lowered = name.lower()
    payload = any(lowered.endswith(suffix) for suffix in PAYLOAD_SUFFIXES)
    kinds = [
        kind
        for kind, patterns in ARTIFACT_PATTERNS.items()
        if any(pattern in lowered for pattern in patterns)
    ]
    return payload, sorted(kinds)


def safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and "\x00" not in name
    )


def xrdcp(remote: str, local: Path) -> None:
    result = subprocess.run(
        ["xrdcp", "-f", remote, str(local)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "xrdcp failed\n"
            f"remote={remote}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--eos-host", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError(f"output exists: {args.output}")
    if not args.scratch.is_dir() or any(args.scratch.iterdir()):
        raise RuntimeError("scratch must exist and be empty")

    plan_rows = read_tsv(args.plan)
    if len(plan_rows) != 54:
        raise RuntimeError(f"plan rows={len(plan_rows)}, expected 54")

    keyed = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        ): row
        for row in plan_rows
    }
    if len(keyed) != 54:
        raise RuntimeError("plan keys are not unique")

    missing = [key for key in CANARIES if key not in keyed]
    if missing:
        raise RuntimeError(f"missing canaries: {missing}")

    args.output.mkdir(parents=True)
    try:
        bundle_rows: list[dict[str, object]] = []
        member_rows: list[dict[str, object]] = []
        provenance_rows: list[dict[str, object]] = []
        payload_rows: list[dict[str, object]] = []
        total_bytes = 0

        for canary_index, key in enumerate(CANARIES, start=1):
            row = keyed[key]
            sample_class, process, campaign = key

            if row["source_type"] != "eos_bundle":
                raise RuntimeError(f"canary is not an EOS bundle: {key}")
            if row["event_payload_extraction_authorized"] != "False":
                raise RuntimeError(f"payload extraction authorized: {key}")
            if row["candidate_content_access_authorized"] != "False":
                raise RuntimeError(f"candidate access authorized: {key}")

            locator = row["representative_locator"]
            expected_size = int(row["representative_size_bytes"])
            remote = args.eos_host.rstrip("/") + "//" + locator.lstrip("/")
            local = args.scratch / f"{canary_index:02d}_{process}.tar.gz"

            free = shutil.disk_usage(args.scratch).free
            if free < expected_size + 2 * 1024 * 1024 * 1024:
                raise RuntimeError(
                    f"insufficient scratch for {process}: free={free}"
                )

            print(
                f"CANARY {canary_index}/6 {process} {campaign}",
                flush=True,
            )
            xrdcp(remote, local)

            actual_size = local.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"size mismatch for {process}: "
                    f"{actual_size} != {expected_size}"
                )

            bundle_sha = sha256_file(local)
            total_bytes += actual_size
            archive_members = 0
            regular_files = 0
            payload_members = 0
            provenance_members = 0
            unsafe_members = 0

            with tarfile.open(local, mode="r:*") as archive:
                for member_index, member in enumerate(archive, start=1):
                    archive_members += 1
                    is_regular = member.isfile()
                    if is_regular:
                        regular_files += 1

                    is_safe = safe_name(member.name)
                    if not is_safe:
                        unsafe_members += 1

                    payload, artifact_types = classify(member.name)
                    if payload:
                        payload_members += 1
                    if artifact_types:
                        provenance_members += 1

                    member_type = (
                        "regular_file"
                        if member.isfile()
                        else "directory"
                        if member.isdir()
                        else "symbolic_link"
                        if member.issym()
                        else "hard_link"
                        if member.islnk()
                        else "other"
                    )
                    allowlist_candidate = (
                        is_regular
                        and is_safe
                        and not payload
                        and bool(artifact_types)
                        and member.size <= ALLOWLIST_MAX_BYTES
                    )

                    inventory_row = {
                        "canary_index": canary_index,
                        "sample_class": sample_class,
                        "process_or_mode": process,
                        "campaign": campaign,
                        "member_index": member_index,
                        "member_name": member.name,
                        "member_type": member_type,
                        "size_bytes": member.size,
                        "safe_relative_path": is_safe,
                        "event_payload": payload,
                        "artifact_types": ",".join(artifact_types) or "none",
                        "allowlist_candidate": allowlist_candidate,
                        "content_opened": False,
                        "content_extracted": False,
                        "status": "archive_header_indexed_only",
                    }
                    member_rows.append(inventory_row)

                    if artifact_types:
                        provenance_rows.append(
                            {
                                "canary_index": canary_index,
                                "sample_class": sample_class,
                                "process_or_mode": process,
                                "campaign": campaign,
                                "member_name": member.name,
                                "member_type": member_type,
                                "size_bytes": member.size,
                                "artifact_types": ",".join(artifact_types),
                                "safe_relative_path": is_safe,
                                "allowlist_candidate": allowlist_candidate,
                                "extraction_authorized_now": False,
                                "status": (
                                    "candidate_for_pn_b3_allowlist_review"
                                    if allowlist_candidate
                                    else "not_allowlisted"
                                ),
                            }
                        )

                    if payload:
                        payload_rows.append(
                            {
                                "canary_index": canary_index,
                                "sample_class": sample_class,
                                "process_or_mode": process,
                                "campaign": campaign,
                                "member_name": member.name,
                                "member_type": member_type,
                                "size_bytes": member.size,
                                "content_opened": False,
                                "content_extracted": False,
                                "status": "payload_identified_and_protected",
                            }
                        )

            local.unlink()
            if local.exists():
                raise RuntimeError(f"temporary file remains: {local}")

            bundle_rows.append(
                {
                    "canary_index": canary_index,
                    "sample_class": sample_class,
                    "process_or_mode": process,
                    "campaign": campaign,
                    "representative_locator": locator,
                    "expected_size_bytes": expected_size,
                    "copied_size_bytes": actual_size,
                    "bundle_sha256": bundle_sha,
                    "archive_members": archive_members,
                    "regular_file_members": regular_files,
                    "event_payload_members": payload_members,
                    "provenance_named_members": provenance_members,
                    "unsafe_member_names": unsafe_members,
                    "temporary_copy_removed": True,
                    "event_payload_content_opened": False,
                    "event_payload_extracted": False,
                    "status": "bundle_index_canary_pass",
                }
            )

            print(
                f"INDEXED members={archive_members} "
                f"payloads={payload_members} "
                f"provenance={provenance_members}",
                flush=True,
            )

        if any(args.scratch.iterdir()):
            raise RuntimeError("scratch is not empty after indexing")

        write_tsv(
            args.output / "bundle_index_summary.tsv",
            bundle_rows,
            list(bundle_rows[0]),
        )
        write_tsv(
            args.output / "archive_member_inventory.tsv",
            member_rows,
            list(member_rows[0]),
        )
        write_tsv(
            args.output / "provenance_member_candidates.tsv",
            provenance_rows,
            list(provenance_rows[0]) if provenance_rows else [
                "canary_index",
                "sample_class",
                "process_or_mode",
                "campaign",
                "member_name",
                "member_type",
                "size_bytes",
                "artifact_types",
                "safe_relative_path",
                "allowlist_candidate",
                "extraction_authorized_now",
                "status",
            ],
        )
        write_tsv(
            args.output / "event_payload_member_inventory.tsv",
            payload_rows,
            list(payload_rows[0]) if payload_rows else [
                "canary_index",
                "sample_class",
                "process_or_mode",
                "campaign",
                "member_name",
                "member_type",
                "size_bytes",
                "content_opened",
                "content_extracted",
                "status",
            ],
        )

        artifact_counts = Counter()
        allowlist_candidates = 0
        for row in provenance_rows:
            for kind in str(row["artifact_types"]).split(","):
                artifact_counts[kind] += 1
            if row["allowlist_candidate"] is True:
                allowlist_candidates += 1

        summary = {
            "schema_version": 1,
            "status": "hh4b_physical_normalization_bundle_index_canary_pass",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": args.source_commit,
            "plan": str(args.plan),
            "plan_sha256": sha256_file(args.plan),
            "canary": {
                "bundles_selected": 6,
                "bundles_copied": len(bundle_rows),
                "bundles_indexed": len(bundle_rows),
                "temporary_bundle_copies_removed": len(bundle_rows),
                "transfer_bytes": total_bytes,
                "archive_members": len(member_rows),
                "event_payload_members": len(payload_rows),
                "provenance_named_members": len(provenance_rows),
                "pn_b3_allowlist_candidates": allowlist_candidates,
                "artifact_type_member_counts": dict(
                    sorted(artifact_counts.items())
                ),
            },
            "readiness": {
                "archive_schema_canary_complete": True,
                "full_campaign_bundle_index_authorized": True,
                "provenance_file_extraction_authorized": False,
                "cross_sections_assigned": 0,
                "sum_generator_weights_resolved": 0,
                "physics_normalization_ready": False,
            },
            "controls": {
                "bundles_downloaded_to_temporary_scratch": 6,
                "temporary_bundle_copies_remaining": 0,
                "archive_members_extracted": 0,
                "provenance_files_extracted": 0,
                "event_payload_files_opened": 0,
                "event_payload_files_extracted": 0,
                "candidate_parquet_files_opened": 0,
                "candidate_rows_read": 0,
                "validation_candidate_files_opened": 0,
                "evaluation_candidate_files_opened": 0,
                "cross_sections_assigned": 0,
                "physical_yields_calculated": 0,
                "models_trained": 0,
                "thresholds_selected": 0,
            },
            "next_gate": (
                "scale_out_representative_bundle_member_index_to_all_"
                "remaining_eos_campaigns"
            ),
        }

        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        (args.output / "README.md").write_text(
            "# HH4b representative-bundle index canary\n\n"
            "Six representative EOS production bundles were copied sequentially "
            "to temporary scratch, byte-size checked, SHA-256 hashed, and indexed "
            "through archive headers. The canaries cover ggF HH4b, VBF HH4b, "
            "inclusive ttbar, heavy-flavor QCD, importance-sampled hard QCD, and "
            "ttZ with a Z-to-bb label.\n\n"
            "No archive member was extracted. Event payload contents were not "
            "opened. Candidate, validation, and final-evaluation content remained "
            "closed. Cross sections and physical yields were not assigned.\n\n"
            "The resulting provenance-member candidates are inputs to a later "
            "explicit extraction allowlist. The legacy local ttbar production "
            "provenance remains a separate blocking recovery task.\n",
            encoding="utf-8",
        )

        legacy = [
            {
                "sample_class": "background",
                "process_or_mode": "ttbar_inclusive",
                "campaign": "parquet",
                "original_generation_artifacts_recovered": False,
                "candidate_file_opened": False,
                "blocking_reason": (
                    "Original cards, logs, and normalization denominator "
                    "for the legacy 270k-event ttbar source are unresolved."
                ),
                "status": "blocking_legacy_provenance_unresolved",
            }
        ]
        write_tsv(
            args.output / "legacy_ttbar_provenance_blocker.tsv",
            legacy,
            list(legacy[0]),
        )

        checksum_path = args.output / "SHA256SUMS"
        products = sorted(
            path
            for path in args.output.iterdir()
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
        print("SIX_REPRESENTATIVE_BUNDLE_CANARIES_INDEXED_PASS")
        print("BUNDLE_SIZE_AND_SHA256_PROVENANCE_PASS")
        print("ARCHIVE_MEMBER_HEADERS_ONLY_PASS")
        print("PN_B3_PROVENANCE_ALLOWLIST_CANDIDATES_FROZEN_PASS")
        print("TEMPORARY_BUNDLE_COPIES_REMOVED_PASS")
        print("NO_ARCHIVE_MEMBER_EXTRACTED")
        print("NO_EVENT_PAYLOAD_CONTENT_OPENED")
        print("NO_CANDIDATE_PARQUET_OPENED")
        print("NO_VALIDATION_CANDIDATE_CONTENT_OPENED")
        print("NO_EVALUATION_CANDIDATE_CONTENT_OPENED")
        print("NO_CROSS_SECTIONS_ASSIGNED")
        print("NO_PHYSICAL_YIELDS_CALCULATED")
        print("HH4B_REPRESENTATIVE_BUNDLE_INDEX_CANARY_PASS")
    except Exception:
        shutil.rmtree(args.output, ignore_errors=True)
        for item in list(args.scratch.iterdir()):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        raise


if __name__ == "__main__":
    main()
