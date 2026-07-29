#!/usr/bin/env python3
"""Extract only frozen small non-payload normalization provenance files."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable


EXPECTED_CAMPAIGNS = 53
EXPECTED_ALLOWLIST_ROWS = 547
MAX_MEMBER_BYTES = 5 * 1024 * 1024

ALLOWED_ARTIFACT_TYPES = {
    "candidate_or_event_summary_log",
    "delphes_log_or_configuration",
    "generation_provenance_metadata",
    "generator_banner_or_lhe_init",
    "generator_stdout_or_summary",
    "importance_sampling_record",
    "parton_shower_card",
    "proc_card",
    "run_card",
}

FORBIDDEN_SUFFIXES = (
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

ALLOWLIST_FIELDS = [
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
    "policy_version",
    "status",
]

MANIFEST_FIELDS = [
    "sample_class",
    "process_or_mode",
    "campaign",
    "representative_locator",
    "bundle_sha256",
    "member_name",
    "archive_path",
    "size_bytes",
    "sha256",
    "artifact_types",
    "utf8_decodable",
    "contains_nul_byte",
    "extracted_from_regular_file",
    "event_payload",
    "nested_archive",
    "status",
]

CAMPAIGN_FIELDS = [
    "sample_class",
    "process_or_mode",
    "campaign",
    "representative_locator",
    "bundle_sha256",
    "expected_files",
    "extracted_files",
    "extracted_bytes",
    "temporary_bundle_removed",
    "event_payload_files_opened",
    "status",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and "\x00" not in name
    )


def normalized_member_parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if not parts:
        raise ValueError(f"empty normalized member path: {name!r}")
    if ".." in parts:
        raise ValueError(f"unsafe normalized member path: {name!r}")
    return parts


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return cleaned.strip("._-") or "unnamed"


def run_xrdcp(remote: str, local: Path) -> None:
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


def validate_inputs(
    draft_rows: list[dict[str, str]],
    bundle_rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, object]],
    dict[tuple[str, str, str], dict[str, str]],
]:
    if len(draft_rows) != EXPECTED_ALLOWLIST_ROWS:
        raise RuntimeError(
            f"draft allowlist rows={len(draft_rows)}, "
            f"expected {EXPECTED_ALLOWLIST_ROWS}"
        )
    if len(bundle_rows) != EXPECTED_CAMPAIGNS:
        raise RuntimeError(
            f"bundle index rows={len(bundle_rows)}, "
            f"expected {EXPECTED_CAMPAIGNS}"
        )

    bundle_map = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        ): row
        for row in bundle_rows
    }
    if len(bundle_map) != EXPECTED_CAMPAIGNS:
        raise RuntimeError("bundle-index campaign keys are not unique")

    frozen: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for row in draft_rows:
        key3 = (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        )
        if key3 not in bundle_map:
            raise RuntimeError(f"allowlist campaign absent from bundle index: {key3}")

        key4 = key3 + (row["member_name"],)
        if key4 in seen:
            raise RuntimeError(f"duplicate allowlist member: {key4}")
        seen.add(key4)

        member_name = row["member_name"]
        size = int(row["size_bytes"])
        types = {
            item
            for item in row["artifact_types"].split(",")
            if item
        }

        if not safe_member_name(member_name):
            raise RuntimeError(f"unsafe allowlist member: {member_name}")
        if not parse_bool(row["safe_relative_path"]):
            raise RuntimeError(f"allowlist path is not marked safe: {member_name}")
        if parse_bool(row["nested_archive"]):
            raise RuntimeError(f"nested archive in draft allowlist: {member_name}")
        if parse_bool(row["event_payload"]):
            raise RuntimeError(f"event payload in draft allowlist: {member_name}")
        if parse_bool(row["extraction_authorized_now"]):
            raise RuntimeError("draft already authorizes extraction")
        if parse_bool(row["content_review_authorized_now"]):
            raise RuntimeError("draft already authorizes content review")
        if size < 0 or size > MAX_MEMBER_BYTES:
            raise RuntimeError(
                f"allowlist size outside policy: {member_name} size={size}"
            )
        if any(member_name.lower().endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            raise RuntimeError(f"forbidden suffix in allowlist: {member_name}")
        if not types or not types.issubset(ALLOWED_ARTIFACT_TYPES):
            raise RuntimeError(
                f"unsupported artifact types for {member_name}: {sorted(types)}"
            )

        bundle = bundle_map[key3]
        if row["representative_locator"] != bundle["representative_locator"]:
            raise RuntimeError(f"locator mismatch for {key4}")

        frozen.append(
            {
                "sample_class": row["sample_class"],
                "process_or_mode": row["process_or_mode"],
                "campaign": row["campaign"],
                "representative_locator": row["representative_locator"],
                "member_name": member_name,
                "size_bytes": size,
                "artifact_types": ",".join(sorted(types)),
                "safe_relative_path": True,
                "nested_archive": False,
                "event_payload": False,
                "extraction_authorized_now": True,
                "content_review_authorized_now": True,
                "policy_version": "pn_b3a_small_nonpayload_v1",
                "status": "frozen_authorized_normalization_provenance",
            }
        )

    campaigns_with_allowlist = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        )
        for row in frozen
    }
    if len(campaigns_with_allowlist) != EXPECTED_CAMPAIGNS:
        missing = sorted(set(bundle_map) - campaigns_with_allowlist)
        raise RuntimeError(
            "not every EOS campaign has an allowlisted provenance file; "
            f"missing={missing}"
        )

    return frozen, bundle_map


def campaign_progress_dir(
    output: Path,
    ordinal: int,
    key: tuple[str, str, str],
) -> Path:
    sample_class, process, campaign = key
    return (
        output
        / "progress"
        / f"{ordinal:03d}_{slug(sample_class)}_{slug(process)}_{slug(campaign)}"
    )


def write_campaign_progress(
    directory: Path,
    campaign_row: dict[str, object],
    manifest_rows: list[dict[str, object]],
    files: list[tuple[str, bytes]],
) -> None:
    if directory.exists():
        raise RuntimeError(f"campaign progress already exists: {directory}")

    files_dir = directory / "files"
    files_dir.mkdir(parents=True)

    for archive_path, data in files:
        destination = files_dir.joinpath(*PurePosixPath(archive_path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    write_tsv(
        directory / "manifest.tsv",
        manifest_rows,
        MANIFEST_FIELDS,
    )
    (directory / "campaign.json").write_text(
        json.dumps(campaign_row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    file_paths = sorted(path for path in directory.rglob("*") if path.is_file())
    checksums = directory / "SHA256SUMS"
    checksums.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}"
            for path in file_paths
            if path != checksums
        )
        + "\n",
        encoding="utf-8",
    )


def load_campaign_progress(
    directory: Path,
) -> tuple[
    dict[str, object],
    list[dict[str, str]],
    list[tuple[str, bytes]],
] | None:
    if not directory.exists():
        return None

    required = (
        directory / "campaign.json",
        directory / "manifest.tsv",
        directory / "SHA256SUMS",
        directory / "files",
    )
    if not all(path.exists() for path in required):
        raise RuntimeError(f"incomplete campaign progress: {directory}")

    result = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=directory,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"campaign progress checksum failure: {directory}\n"
            f"{result.stdout}\n{result.stderr}"
        )

    campaign = json.loads(
        (directory / "campaign.json").read_text(encoding="utf-8")
    )
    manifest = read_tsv(directory / "manifest.tsv")
    files: list[tuple[str, bytes]] = []
    for row in manifest:
        path = directory / "files" / Path(row["archive_path"])
        data = path.read_bytes()
        if len(data) != int(row["size_bytes"]):
            raise RuntimeError(f"progress size mismatch: {path}")
        if sha256_bytes(data) != row["sha256"]:
            raise RuntimeError(f"progress hash mismatch: {path}")
        files.append((row["archive_path"], data))
    return campaign, manifest, files


def deterministic_tar_gz(
    destination: Path,
    files: list[tuple[str, bytes]],
) -> None:
    ordered = sorted(files, key=lambda item: item[0])
    with destination.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.GNU_FORMAT,
            ) as archive:
                for archive_path, data in ordered:
                    info = tarfile.TarInfo(name=archive_path)
                    info.size = len(data)
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(data))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-allowlist", type=Path, required=True)
    parser.add_argument("--bundle-index", type=Path, required=True)
    parser.add_argument("--eos-host", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    draft_path = args.draft_allowlist.resolve()
    bundle_path = args.bundle_index.resolve()
    scratch = args.scratch.resolve()
    output = args.output.resolve()

    if not scratch.is_dir() or any(scratch.iterdir()):
        raise RuntimeError("scratch must exist and be empty")

    draft_rows = read_tsv(draft_path)
    bundle_rows = read_tsv(bundle_path)
    frozen_rows, bundle_map = validate_inputs(draft_rows, bundle_rows)

    if output.exists() and not output.is_dir():
        raise RuntimeError(f"output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "progress").mkdir(exist_ok=True)

    state = {
        "schema_version": 1,
        "source_commit": args.source_commit,
        "draft_allowlist_sha256": sha256_file(draft_path),
        "bundle_index_sha256": sha256_file(bundle_path),
        "campaigns": EXPECTED_CAMPAIGNS,
        "allowlist_rows": EXPECTED_ALLOWLIST_ROWS,
        "policy_version": "pn_b3a_small_nonpayload_v1",
    }
    state_path = output / "resume_state.json"
    if state_path.exists():
        observed = json.loads(state_path.read_text(encoding="utf-8"))
        if observed != state:
            raise RuntimeError(
                "resume-state mismatch\n"
                f"observed={observed}\nexpected={state}"
            )
    else:
        state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    grouped: dict[
        tuple[str, str, str],
        list[dict[str, object]],
    ] = defaultdict(list)
    for row in frozen_rows:
        grouped[
            (
                str(row["sample_class"]),
                str(row["process_or_mode"]),
                str(row["campaign"]),
            )
        ].append(row)

    all_campaign_rows: list[dict[str, object]] = []
    all_manifest_rows: list[dict[str, object]] = []
    all_files: list[tuple[str, bytes]] = []
    transfer_bytes = 0

    ordered_keys = sorted(bundle_map)

    for ordinal, key in enumerate(ordered_keys, start=1):
        bundle = bundle_map[key]
        allowlist = sorted(
            grouped[key],
            key=lambda row: str(row["member_name"]),
        )
        progress_dir = campaign_progress_dir(output, ordinal, key)
        recovered = load_campaign_progress(progress_dir)

        if recovered is not None:
            campaign_row, manifest_rows, files = recovered
            all_campaign_rows.append(campaign_row)
            all_manifest_rows.extend(manifest_rows)
            all_files.extend(files)
            print(
                f"RESUME {ordinal}/{EXPECTED_CAMPAIGNS} "
                f"{key[1]} {key[2]} files={len(files)}",
                flush=True,
            )
            continue

        locator = bundle["representative_locator"]
        expected_size = int(bundle["expected_size_bytes"])
        expected_bundle_sha = bundle["bundle_sha256"]
        remote = args.eos_host.rstrip("/") + "//" + locator.lstrip("/")
        local = scratch / f"{ordinal:03d}_{slug(key[1])}.tar.gz"

        free = shutil.disk_usage(scratch).free
        if free < expected_size + 2 * 1024 * 1024 * 1024:
            raise RuntimeError(
                f"insufficient scratch for {key}: "
                f"free={free}, expected_bundle={expected_size}"
            )

        print(
            f"EXTRACT {ordinal}/{EXPECTED_CAMPAIGNS} "
            f"{key[1]} {key[2]} files={len(allowlist)} "
            f"bundle_bytes={expected_size}",
            flush=True,
        )

        campaign_manifest: list[dict[str, object]] = []
        campaign_files: list[tuple[str, bytes]] = []

        try:
            run_xrdcp(remote, local)
            actual_size = local.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"bundle size mismatch for {key}: "
                    f"{actual_size} != {expected_size}"
                )
            actual_bundle_sha = sha256_file(local)
            if actual_bundle_sha != expected_bundle_sha:
                raise RuntimeError(
                    f"bundle SHA-256 mismatch for {key}: "
                    f"{actual_bundle_sha} != {expected_bundle_sha}"
                )
            transfer_bytes += actual_size

            with tarfile.open(local, mode="r:*") as archive:
                member_map = {
                    member.name: member
                    for member in archive.getmembers()
                }
                if len(member_map) != len(archive.getmembers()):
                    raise RuntimeError(f"duplicate archive member names in {key}")

                for policy_row in allowlist:
                    member_name = str(policy_row["member_name"])
                    if member_name not in member_map:
                        raise RuntimeError(
                            f"allowlisted member absent from bundle: "
                            f"{key} {member_name}"
                        )
                    member = member_map[member_name]
                    if not member.isfile():
                        raise RuntimeError(
                            f"allowlisted member is not regular: {key} {member_name}"
                        )
                    if member.size != int(policy_row["size_bytes"]):
                        raise RuntimeError(
                            f"allowlisted member size mismatch: "
                            f"{key} {member_name} "
                            f"{member.size} != {policy_row['size_bytes']}"
                        )
                    if member.size > MAX_MEMBER_BYTES:
                        raise RuntimeError(
                            f"allowlisted member exceeds policy: {member_name}"
                        )
                    if not safe_member_name(member.name):
                        raise RuntimeError(f"unsafe archive member: {member.name}")

                    handle = archive.extractfile(member)
                    if handle is None:
                        raise RuntimeError(
                            f"could not open allowlisted member: {member.name}"
                        )
                    data = handle.read(MAX_MEMBER_BYTES + 1)
                    if len(data) != member.size:
                        raise RuntimeError(
                            f"member read size mismatch: {member.name}"
                        )
                    if len(data) > MAX_MEMBER_BYTES:
                        raise RuntimeError(
                            f"member read exceeded policy: {member.name}"
                        )

                    parts = normalized_member_parts(member.name)
                    archive_path = PurePosixPath(
                        slug(key[0]),
                        slug(key[1]),
                        slug(key[2]),
                        *parts,
                    ).as_posix()

                    contains_nul = b"\x00" in data
                    try:
                        data.decode("utf-8")
                        utf8_decodable = True
                    except UnicodeDecodeError:
                        utf8_decodable = False

                    content_sha = sha256_bytes(data)
                    campaign_manifest.append(
                        {
                            "sample_class": key[0],
                            "process_or_mode": key[1],
                            "campaign": key[2],
                            "representative_locator": locator,
                            "bundle_sha256": actual_bundle_sha,
                            "member_name": member.name,
                            "archive_path": archive_path,
                            "size_bytes": len(data),
                            "sha256": content_sha,
                            "artifact_types": policy_row["artifact_types"],
                            "utf8_decodable": utf8_decodable,
                            "contains_nul_byte": contains_nul,
                            "extracted_from_regular_file": True,
                            "event_payload": False,
                            "nested_archive": False,
                            "status": "authorized_normalization_provenance_extracted",
                        }
                    )
                    campaign_files.append((archive_path, data))
        finally:
            if local.exists():
                local.unlink()

        if local.exists():
            raise RuntimeError(f"temporary bundle remains: {local}")

        if len(campaign_manifest) != len(allowlist):
            raise RuntimeError(
                f"extracted-file count mismatch for {key}: "
                f"{len(campaign_manifest)} != {len(allowlist)}"
            )

        campaign_row = {
            "sample_class": key[0],
            "process_or_mode": key[1],
            "campaign": key[2],
            "representative_locator": locator,
            "bundle_sha256": expected_bundle_sha,
            "expected_files": len(allowlist),
            "extracted_files": len(campaign_manifest),
            "extracted_bytes": sum(
                int(row["size_bytes"])
                for row in campaign_manifest
            ),
            "temporary_bundle_removed": True,
            "event_payload_files_opened": 0,
            "status": "campaign_normalization_provenance_extraction_pass",
        }

        write_campaign_progress(
            progress_dir,
            campaign_row,
            campaign_manifest,
            campaign_files,
        )

        all_campaign_rows.append(campaign_row)
        all_manifest_rows.extend(campaign_manifest)
        all_files.extend(campaign_files)

        print(
            f"COMPLETE {ordinal}/{EXPECTED_CAMPAIGNS} "
            f"files={len(campaign_files)} "
            f"bytes={campaign_row['extracted_bytes']}",
            flush=True,
        )

    if len(all_campaign_rows) != EXPECTED_CAMPAIGNS:
        raise RuntimeError(
            f"completed campaigns={len(all_campaign_rows)}, "
            f"expected {EXPECTED_CAMPAIGNS}"
        )
    if len(all_manifest_rows) != EXPECTED_ALLOWLIST_ROWS:
        raise RuntimeError(
            f"extracted provenance files={len(all_manifest_rows)}, "
            f"expected {EXPECTED_ALLOWLIST_ROWS}"
        )
    if len(all_files) != EXPECTED_ALLOWLIST_ROWS:
        raise RuntimeError(
            f"archive files={len(all_files)}, "
            f"expected {EXPECTED_ALLOWLIST_ROWS}"
        )
    if len({path for path, _ in all_files}) != len(all_files):
        raise RuntimeError("duplicate output archive paths")
    if any(scratch.iterdir()):
        raise RuntimeError("scratch is not empty after extraction")

    write_tsv(
        output / "provenance_extraction_allowlist_frozen.tsv",
        frozen_rows,
        ALLOWLIST_FIELDS,
    )
    write_tsv(
        output / "provenance_content_manifest.tsv",
        all_manifest_rows,
        MANIFEST_FIELDS,
    )
    write_tsv(
        output / "campaign_extraction_summary.tsv",
        all_campaign_rows,
        CAMPAIGN_FIELDS,
    )

    archive_path = output / "normalization_provenance_files.tar.gz"
    deterministic_tar_gz(archive_path, all_files)

    archive_inventory: list[dict[str, object]] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                raise RuntimeError(
                    f"nonregular final provenance archive member: {member.name}"
                )
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(
                    f"could not verify final provenance member: {member.name}"
                )
            data = handle.read()
            archive_inventory.append(
                {
                    "archive_path": member.name,
                    "size_bytes": member.size,
                    "sha256": sha256_bytes(data),
                    "regular_file": True,
                    "event_payload": False,
                    "status": "final_provenance_archive_member_verified",
                }
            )

    if len(archive_inventory) != EXPECTED_ALLOWLIST_ROWS:
        raise RuntimeError(
            f"final archive members={len(archive_inventory)}, "
            f"expected {EXPECTED_ALLOWLIST_ROWS}"
        )

    write_tsv(
        output / "normalization_provenance_archive_inventory.tsv",
        archive_inventory,
        [
            "archive_path",
            "size_bytes",
            "sha256",
            "regular_file",
            "event_payload",
            "status",
        ],
    )

    artifact_counts = Counter()
    process_counts = Counter()
    non_utf8 = 0
    nul_files = 0
    for row in all_manifest_rows:
        process_counts[str(row["process_or_mode"])] += 1
        for artifact_type in str(row["artifact_types"]).split(","):
            artifact_counts[artifact_type] += 1
        non_utf8 += int(row["utf8_decodable"] is False)
        nul_files += int(row["contains_nul_byte"] is True)

    legacy_search_plan = [
        {
            "process_or_mode": "ttbar_inclusive",
            "legacy_campaign": "parquet",
            "candidate_reference": (
                "/uscms_data/d3/iturkmen/hh4b_delphes/parquet/"
                "ttbar_100k_shard000_hh4b_candidates.parquet"
            ),
            "candidate_content_access_authorized": False,
            "search_roots": (
                "/uscms_data/d3/iturkmen/hh4b_delphes,"
                "/uscms_data/d3/iturkmen/repos/hh-bbww-baselines,"
                "/uscms/home/iturkmen"
            ),
            "required_artifacts": (
                "generator banner, process card, run card, shower card, "
                "generator log, generated-event denominator"
            ),
            "recovery_status": "blocking_separate_pn_b3b_search_required",
        }
    ]
    write_tsv(
        output / "legacy_ttbar_recovery_plan.tsv",
        legacy_search_plan,
        list(legacy_search_plan[0]),
    )

    summary = {
        "schema_version": 1,
        "status": (
            "hh4b_physical_normalization_provenance_extraction_pass"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "draft_allowlist": str(args.draft_allowlist),
        "draft_allowlist_sha256": sha256_file(draft_path),
        "bundle_index": str(args.bundle_index),
        "bundle_index_sha256": sha256_file(bundle_path),
        "extraction": {
            "eos_campaigns": EXPECTED_CAMPAIGNS,
            "frozen_allowlist_rows": EXPECTED_ALLOWLIST_ROWS,
            "provenance_files_extracted": len(all_manifest_rows),
            "provenance_bytes_extracted": sum(
                int(row["size_bytes"])
                for row in all_manifest_rows
            ),
            "final_archive_members": len(archive_inventory),
            "final_archive_bytes": archive_path.stat().st_size,
            "bundle_transfer_bytes_this_run": transfer_bytes,
            "non_utf8_files": non_utf8,
            "files_containing_nul_byte": nul_files,
            "artifact_type_counts": dict(sorted(artifact_counts.items())),
            "process_file_counts": dict(sorted(process_counts.items())),
        },
        "legacy_ttbar": {
            "campaigns_unresolved": 1,
            "candidate_file_opened": False,
            "original_generation_artifacts_recovered": False,
            "status": "blocking_separate_pn_b3b_search_required",
        },
        "readiness": {
            "eos_provenance_extraction_complete": True,
            "eos_provenance_content_parsing_authorized": True,
            "legacy_ttbar_provenance_complete": False,
            "sum_generator_weights_resolved": 0,
            "cross_sections_assigned": 0,
            "physics_normalization_ready": False,
        },
        "controls": {
            "archive_members_extracted": len(all_manifest_rows),
            "authorized_provenance_files_opened": len(all_manifest_rows),
            "candidate_parquet_files_opened": 0,
            "candidate_rows_read": 0,
            "validation_candidate_files_opened": 0,
            "evaluation_candidate_files_opened": 0,
            "event_payload_files_opened": 0,
            "event_payload_files_extracted": 0,
            "temporary_bundle_copies_remaining": 0,
            "cross_sections_assigned": 0,
            "physical_yields_calculated": 0,
            "models_trained": 0,
            "thresholds_selected": 0,
        },
        "next_gate": (
            "recover_legacy_ttbar_generation_artifacts_then_parse_"
            "all_campaign_generator_provenance"
        ),
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# HH4b normalization-provenance extraction\n\n"
        "A frozen policy authorized exactly 547 small, regular, safe-path, "
        "non-payload, non-archive provenance members from all 53 EOS "
        "process-campaign bundles. Each source bundle was byte-size and "
        "SHA-256 verified before the authorized files were read.\n\n"
        "The raw bytes are stored reproducibly in "
        "`normalization_provenance_files.tar.gz` with a complete per-file "
        "manifest. ROOT, Parquet, HepMC, LHE, and nested archives were not "
        "opened or extracted. Candidate, validation, and final-evaluation "
        "content remained closed.\n\n"
        "No cross sections, branching fractions, generator-weight "
        "denominators, physical yields, model thresholds, or model scores "
        "were assigned. The legacy local ttbar campaign remains a separate "
        "blocking recovery task.\n",
        encoding="utf-8",
    )

    shutil.rmtree(output / "progress")
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
    print("PN_B3A_547_FILE_ALLOWLIST_FROZEN_PASS")
    print("ALL_53_EOS_CAMPAIGN_PROVENANCE_EXTRACTION_PASS")
    print("NORMALIZATION_PROVENANCE_ARCHIVE_VERIFIED_PASS")
    print("LEGACY_TTBAR_RECOVERY_PLAN_RECORDED_PASS")
    print("TEMPORARY_BUNDLE_COPIES_REMOVED_PASS")
    print("NO_EVENT_PAYLOAD_CONTENT_OPENED")
    print("NO_CANDIDATE_PARQUET_OPENED")
    print("NO_VALIDATION_CANDIDATE_CONTENT_OPENED")
    print("NO_EVALUATION_CANDIDATE_CONTENT_OPENED")
    print("NO_CROSS_SECTIONS_ASSIGNED")
    print("NO_PHYSICAL_YIELDS_CALCULATED")
    print("HH4B_NORMALIZATION_PROVENANCE_EXTRACTION_PASS")


if __name__ == "__main__":
    main()
