#!/usr/bin/env python3
"""Relocate completed escalation payload archives after an NFS quota failure.

Every source is rehashed, copied to a deterministic EOS target, rehashed there,
and only then removed from the quota-limited source. Payload records are updated
atomically and the operation is restart-safe if interrupted between those steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".relocation-partial")
    require(not temporary.is_symlink(), f"unsafe record temporary path: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    source_root = (package_root / "archives").resolve()
    archive_root = args.archive_root.resolve()
    receipt_root = args.receipt_root.resolve()
    require(package_root.is_dir() and not package_root.is_symlink(), "invalid package root")
    require(source_root.is_dir() and not source_root.is_symlink(), "invalid source archive root")
    require(not archive_root.is_symlink(), "target archive root may not be a symlink")
    require(str(archive_root).startswith("/eos/uscms/store/user/iturkmen/"), "target outside authorized EOS user area")
    require(str(receipt_root).startswith("/eos/uscms/store/user/iturkmen/"), "receipt outside authorized EOS user area")
    record_root = package_root / "build" / "payload_records"
    record_paths = sorted(record_root.glob("replica_*.json"))
    source_archives = sorted(source_root.glob("*.tar.gz"))
    require(record_paths, "no completed payload records")
    require(len(source_archives) <= len(record_paths), "more source archives than completed records")
    records = []
    expected_names = set()
    for record_path in record_paths:
        require(record_path.is_file() and not record_path.is_symlink(), f"unsafe record: {record_path}")
        record = json.loads(record_path.read_text())
        require(record["repository_head"] == "275a2aaebe83b2f1f5e7403a245b52daf30560b4", "execution HEAD mismatch")
        name = record["payload_archive_name"]
        require(name == f"escalation800_payload_replica_{int(record['replica']):04d}__{record['category_id']}.tar.gz", "archive name mismatch")
        require(name not in expected_names, "duplicate archive name")
        expected_names.add(name)
        records.append((record_path, record))
    require({path.name for path in source_archives}.issubset(expected_names), "unregistered source archive")
    if args.check_only:
        print("ESCALATION800_EOS_RELOCATION_CHECK=PASS")
        print(f"COMPLETED_RECORDS={len(records)}")
        print(f"SOURCE_ARCHIVES={len(source_archives)}")
        print(f"SOURCE_BYTES={sum(path.stat().st_size for path in source_archives)}")
        print(f"TARGET_ARCHIVE_ROOT={archive_root}")
        return

    archive_root.mkdir(parents=True, exist_ok=True)
    receipt_root.mkdir(parents=True, exist_ok=True)
    detail_path = receipt_root / "archive_relocation_detail.jsonl"
    migrated = 0
    already_target = 0
    total_bytes = 0
    with detail_path.open("a") as detail:
        for index, (record_path, record) in enumerate(records, start=1):
            name = record["payload_archive_name"]
            source = source_root / name
            target = archive_root / name
            expected_hash = record["payload_archive_sha256"]
            expected_size = int(record["payload_archive_size_bytes"])
            current = Path(record["payload_archive"])
            require(current == source or current == target, f"record path outside relocation endpoints: {current}")
            if target.exists():
                require(target.is_file() and not target.is_symlink(), f"unsafe target: {target}")
                require(target.stat().st_size == expected_size, f"target size mismatch: {target}")
                require(sha256(target) == expected_hash, f"target hash mismatch: {target}")
            else:
                require(source.is_file() and not source.is_symlink(), f"source missing before target publication: {source}")
                require(source.resolve().parent == source_root, f"source escapes archive root: {source}")
                require(source.stat().st_size == expected_size, f"source size mismatch: {source}")
                require(sha256(source) == expected_hash, f"source hash mismatch: {source}")
                partial = archive_root / (name + ".partial")
                if partial.exists():
                    require(partial.is_file() and not partial.is_symlink(), f"unsafe target partial: {partial}")
                    partial.unlink()
                shutil.copyfile(source, partial)
                require(partial.stat().st_size == expected_size, f"copied size mismatch: {partial}")
                require(sha256(partial) == expected_hash, f"copied hash mismatch: {partial}")
                os.replace(partial, target)
                require(sha256(target) == expected_hash, f"published target hash mismatch: {target}")
                migrated += 1
            if source.exists():
                require(source.is_file() and not source.is_symlink(), f"unsafe source before removal: {source}")
                require(source.resolve().parent == source_root, f"source escapes archive root: {source}")
                require(source.stat().st_size == expected_size, f"source changed before removal: {source}")
                require(sha256(source) == expected_hash, f"source hash changed before removal: {source}")
                source.unlink()
            else:
                already_target += int(current == target)
            record["payload_archive"] = str(target)
            record["payload_archive_storage_amendment"] = "eos_after_nfs_user_quota_exhaustion"
            atomic_json(record_path, record)
            total_bytes += expected_size
            detail.write(json.dumps({
                "archive_name": name,
                "payload_archive_sha256": expected_hash,
                "payload_archive_size_bytes": expected_size,
                "record_path": str(record_path),
                "source_path": str(source),
                "status": "relocated_and_source_removed_after_hash_closure",
                "target_path": str(target),
            }, sort_keys=True) + "\n")
            detail.flush()
            if index % 20 == 0 or index == len(records):
                print(f"EOS_RELOCATION_PROGRESS completed={index}/{len(records)}", flush=True)
    require(not list(source_root.glob("*.tar.gz")), "source archives remain after relocation")
    summary = {
        "archive_count": len(records),
        "archive_total_bytes": total_bytes,
        "execution_repository_head": "275a2aaebe83b2f1f5e7403a245b52daf30560b4",
        "nfs_quota_failure_preserved": True,
        "pilot_results_used": False,
        "schema_version": 1,
        "source_archive_root": str(source_root),
        "source_copies_removed_only_after_target_sha256_closure": True,
        "status": "pass_escalation800_completed_archive_eos_relocation",
        "target_archive_root": str(archive_root),
        "test_payloads_opened": 0,
        "validation_payloads_opened": 0,
    }
    atomic_json(receipt_root / "archive_relocation_summary.json", summary)
    print("ESCALATION800_EOS_RELOCATION=PASS")
    print(f"ARCHIVES={len(records)}")
    print(f"BYTES={total_bytes}")
    print(f"NEWLY_MIGRATED={migrated}")
    print(f"ALREADY_TARGET={already_target}")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
