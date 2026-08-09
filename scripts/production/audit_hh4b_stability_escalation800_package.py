#!/usr/bin/env python3
"""Independently audit the HH4b stability escalation package before submission."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import tarfile
from typing import Any


CATEGORIES = ("exact3tag", "ge4tag")
OUTER_FOLDS = tuple(range(5))
STRUCTURES_PER_JOB = 27
RUNTIME_SHA256 = "4a9c4061e1d975f276326022454e578c5f0de3c094b9b56345f192cccc654112"
CORE_MEMBER_HASHES = {
    "code/hh4b_multivariate_cut_optimizer.py": "7760ee7ae309bec6cd47eb1c73ed04ec12bb805f133527498fcb75aca8088a62",
    "code/hh4b_multivariate_cut_structure_worker.py": "739147c33d6597018f4dce8c483278c64ff8b41533de1f1b213749dfb39332ed",
    "config/multivariate_cut_category_endpoint_amendment.json": "02ae67f00f4180eb7e84563f37145a0f7f84579c75cae7b003ff098935fedb71",
    "config/runtime_config.template.json": "adcc9ee89147d0056be92ac9aa2e2eaafc4983f018882f6477f876ea81e99daa",
    "structure_ids.txt": "1d740db03fd2b94e1d9f19f8d40448340ae02402cd6b3f277933792d4c207faa",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stream_sha256(handle: Any) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".partial")
    require(not temporary.is_symlink(), f"refusing symlink temporary path: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def parse_sums(payload: bytes) -> dict[str, str]:
    result = {}
    for raw_line in payload.decode().splitlines():
        digest, relative_name = raw_line.split("  ", 1)
        require(len(digest) == 64 and relative_name not in result, "malformed internal SHA256SUMS")
        result[relative_name] = digest
    return result


def audit_archive(record: dict[str, Any], expected_head: str) -> dict[str, Any]:
    replica = int(record["replica"])
    category = str(record["category_id"])
    archive_path = Path(record["payload_archive"])
    require(200 <= replica <= 999, "record replica outside escalation authorization")
    require(category in CATEGORIES, "record category invalid")
    require(archive_path.is_file() and not archive_path.is_symlink(), f"archive missing or unsafe: {archive_path}")
    actual_archive_hash = sha256(archive_path)
    require(actual_archive_hash == record["payload_archive_sha256"], f"archive SHA mismatch: {archive_path}")
    require(archive_path.stat().st_size == int(record["payload_archive_size_bytes"]), "archive size mismatch")

    expected_tables = {
        f"fold_tables/fold_{fold}/{category}.parquet": entry["sha256"]
        for fold, entry in zip(OUTER_FOLDS, record["table_records"])
    }
    require(
        [entry["outer_source_fold"] for entry in record["table_records"]] == list(OUTER_FOLDS),
        "table record fold order mismatch",
    )
    expected_internal = {**CORE_MEMBER_HASHES, **expected_tables}
    provenance: dict[str, Any] | None = None
    observed_files: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        require(all(not (member.issym() or member.islnk()) for member in members), "archive contains a link")
        require(all(member.name == "payload" or member.name.startswith("payload/") for member in members), "archive path escapes payload root")
        for member in members:
            if not member.isfile():
                continue
            relative_name = member.name.removeprefix("payload/")
            observed_files.add(relative_name)
            extracted = archive.extractfile(member)
            require(extracted is not None, f"unable to read archive member: {member.name}")
            if relative_name == "SHA256SUMS":
                sums_payload = extracted.read()
                continue
            actual_hash = stream_sha256(extracted)
            if relative_name == "execution_provenance.json":
                extracted_again = archive.extractfile(member)
                require(extracted_again is not None, "unable to reread provenance")
                provenance = json.loads(extracted_again.read())
                expected_internal[relative_name] = actual_hash
            require(relative_name in expected_internal, f"unexpected archive file: {relative_name}")
            require(actual_hash == expected_internal[relative_name], f"internal member SHA mismatch: {relative_name}")
    expected_files = set(expected_internal) | {"SHA256SUMS"}
    require(observed_files == expected_files, f"archive file set mismatch: {archive_path}")
    sums = parse_sums(sums_payload)
    require(sums == expected_internal, f"internal SHA256SUMS mismatch: {archive_path}")
    require(provenance is not None, "provenance missing")
    require(provenance["status"] == "transfer_package_selection_stability_escalation800_category_payload", "provenance status mismatch")
    require(provenance["repository_head"] == expected_head, "provenance HEAD mismatch")
    require(int(provenance["bootstrap_replica"]) == replica, "provenance replica mismatch")
    require(provenance["transferred_category_id"] == category, "provenance category mismatch")
    require(provenance["escalation_200_999_production_authorized"] is True, "payload lacks production authorization")
    require(provenance["results_may_enter_all1000_stability_aggregation"] is True, "payload result role mismatch")
    require(provenance["pilot_results_may_enter_stability_aggregation"] is False, "pilot leakage role mismatch")
    require(provenance["nominal_selection_changed"] is False, "nominal selection changed")
    require(provenance["production_search_budget"] == "63_128_16_8", "search budget mismatch")
    require(float(provenance["target_signal_efficiency"]) == 0.585957, "target efficiency mismatch")
    require(int(provenance["validation_payloads_opened"]) == 0, "validation counter nonzero")
    require(int(provenance["test_payloads_opened"]) == 0, "test counter nonzero")
    require(record["repository_head"] == expected_head, "record HEAD mismatch")
    require(record["escalation_200_999_production_authorized"] is True, "record authorization mismatch")
    require(record["pilot_results_may_enter_stability_aggregation"] is False, "record pilot role mismatch")
    require(record["validation_payloads_opened"] == 0 and record["test_payloads_opened"] == 0, "record sealed counter mismatch")
    return {
        "payload_key": record["payload_key"],
        "replica": replica,
        "category_id": category,
        "transferred_rows": int(record["transferred_rows"]),
        "payload_archive_size_bytes": archive_path.stat().st_size,
        "payload_archive_sha256": actual_archive_hash,
        "archive_file_set_pass": True,
        "internal_sha256_pass": True,
        "provenance_pass": True,
    }


def audit_submission_tables(package_root: Path, expected_head: str, replica_start: int, replica_stop: int, records: list[dict[str, Any]]) -> tuple[int, str, str, str]:
    submission = package_root / "submission"
    matrix_path = submission / "escalation800_job_matrix.tsv"
    queue_path = submission / "escalation800_queue.items"
    submit_path = submission / "escalation800_category_fold.sub"
    runner_path = submission / "run_escalation800_category_fold_job.sh"
    for path in (matrix_path, queue_path, submit_path, runner_path):
        require(path.is_file() and not path.is_symlink(), f"submission file missing or unsafe: {path}")
    by_key = {record["payload_key"]: record for record in records}
    expected_job_count = (replica_stop - replica_start) * 10
    with matrix_path.open(newline="") as handle:
        matrix_rows = list(csv.DictReader(handle, delimiter="\t"))
    queue_rows = [line.split("\t") for line in queue_path.read_text().splitlines()]
    require(len(matrix_rows) == expected_job_count, "job matrix count mismatch")
    require(len(queue_rows) == expected_job_count, "queue item count mismatch")
    for job_index, (matrix, queue) in enumerate(zip(matrix_rows, queue_rows)):
        require(len(queue) == 10, "queue item field count mismatch")
        expected_replica = replica_start + job_index // 10
        offset = job_index % 10
        expected_category = CATEGORIES[offset // 5]
        expected_fold = offset % 5
        payload_key = f"replica_{expected_replica:04d}__{expected_category}"
        record = by_key[payload_key]
        require(int(matrix["job_index"]) == job_index and int(queue[0]) == job_index, "job index mismatch")
        require(int(matrix["replica"]) == expected_replica and int(queue[1]) == expected_replica, "job replica mismatch")
        require(int(matrix["outer_fold"]) == expected_fold and int(queue[2]) == expected_fold, "job outer-fold mismatch")
        require(matrix["category_id"] == expected_category and queue[3] == expected_category, "job category mismatch")
        require(matrix["payload_key"] == payload_key, "matrix payload key mismatch")
        require(matrix["role"] == "escalation800_selection_stability_production", "matrix role mismatch")
        require(queue[4] == expected_head, "queue HEAD mismatch")
        require(queue[5] == record["payload_archive"], "queue archive path mismatch")
        require(queue[6] == record["payload_archive_name"], "queue archive basename mismatch")
        require(queue[7] == record["payload_archive_sha256"], "queue archive SHA mismatch")
        require(queue[8] == RUNTIME_SHA256, "queue runtime SHA mismatch")
        expected_return = package_root / "returns" / f"job_{job_index:04d}"
        require(Path(queue[9]) == expected_return, "queue return path mismatch")
        require(expected_return.is_dir() and not expected_return.is_symlink(), "return directory missing or unsafe")
        require(not any(expected_return.iterdir()), "pre-submission return directory is not empty")
    submit_text = submit_path.read_text()
    require("condor_submit" not in submit_text and " -name " not in submit_text, "submit file contains forbidden submit invocation")
    require(str(queue_path) in submit_text and str(runner_path) in submit_text, "submit file path binding mismatch")
    require("request_cpus = 1" in submit_text and "request_memory = 2048MB" in submit_text, "resource model mismatch")
    return expected_job_count, sha256(matrix_path), sha256(queue_path), sha256(submit_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--expected-repository-head", required=True)
    parser.add_argument("--replica-start", type=int, default=200)
    parser.add_argument("--replica-stop", type=int, default=1000, help="exclusive")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    archive_root = (
        args.archive_root.resolve()
        if args.archive_root is not None
        else (package_root / "archives").resolve()
    )
    require(not package_root.is_symlink(), "package root may not be a symlink")
    require(not archive_root.is_symlink(), "archive root may not be a symlink")
    require(200 <= args.replica_start < args.replica_stop <= 1000, "replica range outside authorization")
    require(1 <= args.workers <= 8, "workers must be in [1, 8]")
    require(not (package_root / "production_submission_v1").exists(), "submission evidence exists before audit")
    evidence = package_root / "evidence"
    manifest_path = evidence / "payload_manifest.json"
    manifest_tsv_path = evidence / "payload_manifest.tsv"
    build_receipt_path = evidence / "escalation800_package_build_receipt.json"
    state_path = package_root / "build" / "build_state.json"
    for path in (manifest_path, manifest_tsv_path, build_receipt_path, state_path):
        require(path.is_file(), f"package evidence missing: {path}")
    manifest = json.loads(manifest_path.read_text())
    build_receipt = json.loads(build_receipt_path.read_text())
    state = json.loads(state_path.read_text())
    records = manifest["payloads"]
    replica_count = args.replica_stop - args.replica_start
    expected_payload_count = replica_count * 2
    require(manifest["status"] == "pass_escalation800_payload_manifest_complete", "manifest status mismatch")
    require(manifest["repository_head"] == args.expected_repository_head, "manifest HEAD mismatch")
    require(manifest["replica_range"] == [args.replica_start, args.replica_stop - 1], "manifest range mismatch")
    require(manifest["payload_count"] == expected_payload_count == len(records), "manifest payload count mismatch")
    require(manifest["production_submission_performed"] is False, "manifest reports prior submission")
    require(manifest["validation_payloads_opened"] == 0 and manifest["test_payloads_opened"] == 0, "manifest sealed counter mismatch")
    require(build_receipt["repository_head"] == args.expected_repository_head, "build receipt HEAD mismatch")
    require(state["repository_head"] == args.expected_repository_head, "build state HEAD mismatch")
    require(state["production_submission_performed"] is False, "build state reports prior submission")
    require(
        all(Path(record["payload_archive"]).parent == archive_root for record in records),
        "payload records do not bind the audited archive root",
    )
    archive_paths = sorted(archive_root.glob("*.tar.gz"))
    require(len(archive_paths) == expected_payload_count, "external archive count mismatch")
    expected_keys = [f"replica_{replica:04d}__{category}" for replica in range(args.replica_start, args.replica_stop) for category in CATEGORIES]
    require([record["payload_key"] for record in records] == expected_keys, "payload manifest ordering or coverage mismatch")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(audit_archive, record, args.expected_repository_head): record["payload_key"] for record in records}
        completed_count = 0
        for future in as_completed(futures):
            rows.append(future.result())
            completed_count += 1
            if completed_count % 20 == 0 or completed_count == len(records):
                print(f"PACKAGE_AUDIT_PROGRESS completed={completed_count}/{len(records)}", flush=True)
    rows.sort(key=lambda item: (item["replica"], CATEGORIES.index(item["category_id"])))
    job_count, matrix_hash, queue_hash, submit_hash = audit_submission_tables(
        package_root, args.expected_repository_head, args.replica_start, args.replica_stop, records
    )
    runner_path = package_root / "submission" / "run_escalation800_category_fold_job.sh"
    audit = {
        "aggregation_protocol_sha256": build_receipt["aggregation_protocol_sha256"],
        "authorized_replicas": [args.replica_start, args.replica_stop - 1],
        "categories": list(CATEGORIES),
        "condor_job_count": job_count,
        "condor_submission_performed": False,
        "escalation_200_999_production_authorized": True,
        "expected_structure_evaluations": job_count * STRUCTURES_PER_JOB,
        "job_matrix_sha256": matrix_hash,
        "materialized_total_rows_across_replicas": sum(int(row["transferred_rows"]) for row in rows),
        "next": "freeze_pre_submission_package_evidence_then_submit_exactly_8000_jobs_once",
        "nominal_selection_changed": False,
        "outer_folds": list(OUTER_FOLDS),
        "payload_archive_count": len(rows),
        "payload_archive_root": str(archive_root),
        "payload_archive_total_bytes": sum(int(row["payload_archive_size_bytes"]) for row in rows),
        "payload_internal_sha256_closure": True,
        "payload_manifest_json_sha256": sha256(manifest_path),
        "payload_manifest_tsv_sha256": sha256(manifest_tsv_path),
        "payload_reuse_factor_outer_folds": 5,
        "pilot_results_may_enter_stability_aggregation": False,
        "production_package_built": True,
        "production_submission_performed": False,
        "queue_items_sha256": queue_hash,
        "replica_count": replica_count,
        "repository_head": args.expected_repository_head,
        "runner_sha256": sha256(runner_path),
        "runtime_archive_sha256": RUNTIME_SHA256,
        "schema_version": 1,
        "status": "pass_escalation800_transfer_safe_production_package_pre_submission_audit",
        "structures_per_job": STRUCTURES_PER_JOB,
        "submit_file_sha256": submit_hash,
        "test_payloads_opened": 0,
        "validation_payloads_opened": 0,
    }
    audit_json_path = evidence / "escalation800_production_package_audit.json"
    audit_tsv_path = evidence / "escalation800_production_package_audit.tsv"
    atomic_write(audit_json_path, json_bytes(audit))
    header = list(rows[0])
    lines = ["\t".join(header) + "\n"]
    for row in rows:
        lines.append("\t".join(str(row[key]) for key in header) + "\n")
    atomic_write(audit_tsv_path, "".join(lines).encode())
    state.update({
        "package_audit_json": str(audit_json_path),
        "status": "pass_escalation800_production_package_built_and_audited_pre_submission",
    })
    atomic_write(state_path, json_bytes(state))
    print("ESCALATION800_PACKAGE_AUDIT=PASS")
    print(f"REPLICAS={replica_count}")
    print(f"PAYLOAD_ARCHIVES={len(rows)}")
    print(f"CONDOR_JOBS={job_count}")
    print(f"STRUCTURE_EVALUATIONS={job_count * STRUCTURES_PER_JOB}")
    print("ALL_ARCHIVE_SHA256=PASS")
    print("ALL_INTERNAL_SHA256=PASS")
    print("PRODUCTION_SUBMISSION_PERFORMED=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
