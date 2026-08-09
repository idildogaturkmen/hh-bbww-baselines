#!/usr/bin/env python3
"""Build the immutable, transfer-safe HH4b stability replicas 200--999 package."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any


CATEGORIES = ("exact3tag", "ge4tag")
OUTER_FOLDS = tuple(range(5))
STRUCTURES_PER_JOB = 27
RUNTIME_SHA256 = "4a9c4061e1d975f276326022454e578c5f0de3c094b9b56345f192cccc654112"
TEMPLATE_ARCHIVE_SHA256 = "4ade0c4d4bc9d981ee1d9f11fe095b79b58ce5bd26c69b994c24c12f0d646586"
MATERIALIZER_SHA256 = "bea9bf35923c79e87aeb8097e524a2dee3712ffa1f8e44c78787f2683d19efcc"
DRAW_SHA256 = "091c2a401cb6ab74d9219449c117fc99832737cc2951e812dc7765540615a279"
AUTHORIZATION_SHA256 = "b881012318ba5264b9ac97788ff92bf53971faa0b00290d0c092daac10cff697"
AGGREGATION_PROTOCOL_SHA256 = "a856d9027cc1d2e11a9d60e72b4104b5b8c24ce61acf43def0d4d487ab867eea"
NOMINAL_SHA256SUMS_SHA256 = "02bb6e4722aaf130337696c707de4574b3a385bfdb145d52f5cbcc049e858691"
CORE_MEMBER_HASHES = {
    "code/hh4b_multivariate_cut_optimizer.py": "7760ee7ae309bec6cd47eb1c73ed04ec12bb805f133527498fcb75aca8088a62",
    "code/hh4b_multivariate_cut_structure_worker.py": "739147c33d6597018f4dce8c483278c64ff8b41533de1f1b213749dfb39332ed",
    "config/multivariate_cut_category_endpoint_amendment.json": "02ae67f00f4180eb7e84563f37145a0f7f84579c75cae7b003ff098935fedb71",
    "config/runtime_config.template.json": "adcc9ee89147d0056be92ac9aa2e2eaafc4983f018882f6477f876ea81e99daa",
    "structure_ids.txt": "1d740db03fd2b94e1d9f19f8d40448340ae02402cd6b3f277933792d4c207faa",
}
EXPECTED_NOMINAL_TABLE_HASHES = {
    "fold_0/exact3tag.parquet": "eb062c28af45a050fd81f6e3980afc430a6aca7cadc19b860634d4a91644597b",
    "fold_0/ge4tag.parquet": "db71164c992853ead333bc8dfc3670f316f77033157b0a6ce611f6bb569a5b28",
    "fold_1/exact3tag.parquet": "79acbfaa404e7f0538d668220c45093498a9dc48a81a1bae8a5b5bf81fe6d30e",
    "fold_1/ge4tag.parquet": "f151557a887864b13ab390b15a9cc59edb7c821a3e9c7ced2b4a60c0d3c30fa1",
    "fold_2/exact3tag.parquet": "ee9e9efd69b1f352e0b3c33289abaf89e81a88aa30709ae1a292fe9d84582c41",
    "fold_2/ge4tag.parquet": "ce3d4b035f438a42f2d76990f32bf2585e57f1735c5d54ee047561c0e02c59c6",
    "fold_3/exact3tag.parquet": "bfedfd7db4700caa8c5acf5162379d7872bd18ee3732ef8f78b1a0bb0b74af0b",
    "fold_3/ge4tag.parquet": "7ca7c784f10458db0064c6198e089de080f7cbb54a5cf35d7f509fe2de63e9d5",
    "fold_4/exact3tag.parquet": "e80b2acae8dbc6811e15e8ec95bfda7fec53fdd2fa206124d68e7c12f5c6d5e3",
    "fold_4/ge4tag.parquet": "b70d972f666a5d327b776fac56b86ce0d42b9e5e7a394a78439bea26e8e60a00",
}
TEMPLATE_MEMBERS = tuple(CORE_MEMBER_HASHES)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    require(not temporary.is_symlink(), f"refusing symlink temporary path: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def add_directory(archive: tarfile.TarFile, name: str) -> None:
    info = tarfile.TarInfo(name.rstrip("/") + "/")
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info)


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = mode
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(payload))


def add_file(archive: tarfile.TarFile, name: str, source: Path) -> None:
    require(source.is_file() and not source.is_symlink(), f"invalid payload source: {source}")
    info = archive.gettarinfo(str(source), arcname=name)
    info.mode = 0o644
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def make_deterministic_archive(
    destination: Path,
    core_members: dict[str, bytes],
    provenance: bytes,
    tables: list[tuple[str, Path]],
    sums: bytes,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), f"refusing to overwrite archive: {destination}")
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for directory in (
                    "payload",
                    "payload/code",
                    "payload/config",
                    "payload/fold_tables",
                    *[f"payload/fold_tables/fold_{fold}" for fold in OUTER_FOLDS],
                ):
                    add_directory(archive, directory)
                add_bytes(archive, "payload/SHA256SUMS", sums)
                for relative_name in sorted(core_members):
                    add_bytes(archive, f"payload/{relative_name}", core_members[relative_name])
                add_bytes(archive, "payload/execution_provenance.json", provenance)
                for relative_name, source in sorted(tables):
                    add_file(archive, f"payload/{relative_name}", source)


@dataclass(frozen=True)
class BuildContext:
    package_root: Path
    archive_root: Path
    repository_head: str
    tooling_repository_head: str
    nominal_root: Path
    draw_parquet: Path
    materializer: Path
    python: str
    core_members: dict[str, bytes]


def payload_record_path(context: BuildContext, replica: int, category: str) -> Path:
    return context.package_root / "build" / "payload_records" / f"replica_{replica:04d}__{category}.json"


def archive_path(context: BuildContext, replica: int, category: str) -> Path:
    return context.archive_root / f"escalation800_payload_replica_{replica:04d}__{category}.tar.gz"


def load_complete_record(context: BuildContext, replica: int, category: str) -> dict[str, Any] | None:
    record_path = payload_record_path(context, replica, category)
    archive = archive_path(context, replica, category)
    if not record_path.exists() and not archive.exists():
        return None
    require(record_path.is_file() and archive.is_file(), f"partial immutable payload state for replica {replica} {category}")
    record = json.loads(record_path.read_text())
    require(record["replica"] == replica, "existing record replica mismatch")
    require(record["category_id"] == category, "existing record category mismatch")
    require(record["repository_head"] == context.repository_head, "existing record HEAD mismatch")
    require(record["payload_archive_sha256"] == sha256(archive), "existing archive hash mismatch")
    require(record["payload_archive_size_bytes"] == archive.stat().st_size, "existing archive size mismatch")
    return record


def make_provenance(context: BuildContext, replica: int, category: str) -> dict[str, Any]:
    return {
        "aggregation_protocol_sha256": AGGREGATION_PROTOCOL_SHA256,
        "authorization_checkpoint_sha256": AUTHORIZATION_SHA256,
        "bootstrap_fold_reassignment": False,
        "bootstrap_replica": replica,
        "bootstrap_source_group_resampling": True,
        "escalation_200_999_production_authorized": True,
        "escalation_draw_registry_sha256": DRAW_SHA256,
        "fold_table_count": 10,
        "full_generated_event_accounting": 5200000,
        "nominal_selection_changed": False,
        "optimizer_sha256": CORE_MEMBER_HASHES["code/hh4b_multivariate_cut_optimizer.py"],
        "pilot_results_may_enter_stability_aggregation": False,
        "production_search_budget": "63_128_16_8",
        "repository_branch": "delphes-hh4b-production",
        "repository_head": context.repository_head,
        "results_may_enter_all1000_stability_aggregation": True,
        "schema_version": 1,
        "selection_stability_draw_registry_sha256": "723f11cf3f78cc0f8eb68013cc7ec1ff7dcb24279c4f317ef42927caf8b97cc2",
        "selection_stability_seed": 20260806,
        "status": "transfer_package_selection_stability_escalation800_category_payload",
        "structure_count": STRUCTURES_PER_JOB,
        "target_signal_efficiency": 0.585957,
        "test_payloads_opened": 0,
        "training_generated_events": 3799873,
        "transferred_category_fold_table_count": 5,
        "transferred_category_id": category,
        "validation_payloads_opened": 0,
        "worker_sha256": CORE_MEMBER_HASHES["code/hh4b_multivariate_cut_structure_worker.py"],
    }


def build_replica(context: BuildContext, replica: int) -> tuple[int, list[dict[str, Any]], int]:
    existing = {category: load_complete_record(context, replica, category) for category in CATEGORIES}
    if all(existing.values()):
        rows = sum(int(record["transferred_rows"]) for record in existing.values() if record)
        return replica, [existing[category] for category in CATEGORIES if existing[category]], rows

    work_parent = context.package_root / "build" / "work"
    work_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"replica_{replica:04d}_", dir=work_parent) as temporary_name:
        temporary = Path(temporary_name)
        materialized = temporary / "fold_tables"
        summary_temporary = temporary / "materialization_summary.json"
        command = [
            context.python,
            str(context.materializer),
            "--nominal-root", str(context.nominal_root),
            "--draw-parquet", str(context.draw_parquet),
            "--replica", str(replica),
            "--output-root", str(materialized),
            "--summary-json", str(summary_temporary),
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        require(completed.returncode == 0, f"materializer failed replica {replica}: {completed.stderr}")
        summary = json.loads(summary_temporary.read_text())
        require(summary["replica"] == replica and summary["table_count"] == 10, "materialization summary mismatch")
        require(summary["validation_payloads_opened"] == 0 and summary["test_payloads_opened"] == 0, "sealed payload counter changed")
        summary_path = context.package_root / "build" / "materialization_summaries" / f"replica_{replica:04d}.json"
        summary_payload = json_bytes(summary)
        if summary_path.exists():
            require(summary_path.read_bytes() == summary_payload, "materialization reproduction mismatch")
        else:
            atomic_write(summary_path, summary_payload)

        records: list[dict[str, Any]] = []
        for category in CATEGORIES:
            if existing[category] is not None:
                records.append(existing[category])
                continue
            table_records = []
            archive_tables = []
            internal_hashes = dict(CORE_MEMBER_HASHES)
            for fold in OUTER_FOLDS:
                source = materialized / f"fold_{fold}" / f"{category}.parquet"
                relative_name = f"fold_tables/fold_{fold}/{category}.parquet"
                expected = next(x for x in summary["tables"] if x["relative_path"] == f"fold_{fold}/{category}.parquet")
                actual_hash = sha256(source)
                require(actual_hash == expected["output_sha256"], "materialized table hash mismatch")
                archive_tables.append((relative_name, source))
                internal_hashes[relative_name] = actual_hash
                table_records.append({
                    "outer_source_fold": fold,
                    "relative_path": relative_name,
                    "rows": int(expected["materialized_rows"]),
                    "sha256": actual_hash,
                })

            provenance_payload = json_bytes(make_provenance(context, replica, category))
            provenance_hash = bytes_sha256(provenance_payload)
            internal_hashes["execution_provenance.json"] = provenance_hash
            sums_payload = "".join(
                f"{digest}  {relative_name}\n"
                for relative_name, digest in sorted(internal_hashes.items())
            ).encode()
            archive = archive_path(context, replica, category)
            partial = context.package_root / "build" / "partials" / (archive.name + ".partial")
            partial.parent.mkdir(parents=True, exist_ok=True)
            if partial.exists():
                require(partial.is_file() and not partial.is_symlink(), f"unsafe partial archive: {partial}")
                partial.unlink()
            make_deterministic_archive(
                partial,
                context.core_members,
                provenance_payload,
                archive_tables,
                sums_payload,
            )
            require(not archive.exists(), f"archive appeared during build: {archive}")
            os.replace(partial, archive)
            record = {
                "category_id": category,
                "escalation_200_999_production_authorized": True,
                "execution_provenance_sha256": provenance_hash,
                "outer_folds_served": list(OUTER_FOLDS),
                "payload_archive": str(archive),
                "payload_archive_name": archive.name,
                "payload_archive_sha256": sha256(archive),
                "payload_archive_size_bytes": archive.stat().st_size,
                "payload_key": f"replica_{replica:04d}__{category}",
                "pilot_results_may_enter_stability_aggregation": False,
                "replica": replica,
                "repository_head": context.repository_head,
                "results_may_enter_all1000_stability_aggregation": True,
                "schema_version": 1,
                "status": "pass_escalation800_replica_category_payload_build",
                "structure_count": STRUCTURES_PER_JOB,
                "table_records": table_records,
                "test_payloads_opened": 0,
                "transferred_fold_tables": 5,
                "transferred_rows": sum(item["rows"] for item in table_records),
                "validation_payloads_opened": 0,
            }
            atomic_write(payload_record_path(context, replica, category), json_bytes(record))
            records.append(record)
        return replica, sorted(records, key=lambda item: item["category_id"]), int(summary["total_materialized_rows"])


def load_template_members(template_archive: Path) -> dict[str, bytes]:
    require(sha256(template_archive) == TEMPLATE_ARCHIVE_SHA256, "template archive hash mismatch")
    members: dict[str, bytes] = {}
    with tarfile.open(template_archive, "r:gz") as archive:
        for relative_name in TEMPLATE_MEMBERS:
            member = archive.extractfile(f"payload/{relative_name}")
            require(member is not None, f"template member missing: {relative_name}")
            payload = member.read()
            require(bytes_sha256(payload) == CORE_MEMBER_HASHES[relative_name], f"template member hash mismatch: {relative_name}")
            members[relative_name] = payload
    return members


def validate_inputs(args: argparse.Namespace, repository_root: Path) -> dict[str, bytes]:
    require(args.replica_start >= 200 and args.replica_stop <= 1000, "replica range outside authorization")
    require(args.replica_start < args.replica_stop, "empty replica range")
    require(len(args.expected_repository_head) == 40, "expected HEAD must be full SHA")
    actual_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True, text=True, capture_output=True
    ).stdout.strip()
    if args.expected_tooling_head is None:
        require(actual_head == args.expected_repository_head, f"repository HEAD mismatch: {actual_head}")
    else:
        require(actual_head == args.expected_tooling_head, f"tooling repository HEAD mismatch: {actual_head}")
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", args.expected_repository_head, actual_head],
            cwd=repository_root,
            check=False,
        )
        require(ancestry.returncode == 0, "execution HEAD is not an ancestor of tooling HEAD")
    require(sha256(args.draw_parquet) == DRAW_SHA256, "escalation draw hash mismatch")
    require(sha256(args.materializer) == MATERIALIZER_SHA256, "materializer hash mismatch")
    require(sha256(args.runtime_archive) == RUNTIME_SHA256, "runtime archive hash mismatch")
    require(sha256(args.authorization_json) == AUTHORIZATION_SHA256, "authorization hash mismatch")
    require(sha256(args.aggregation_protocol) == AGGREGATION_PROTOCOL_SHA256, "aggregation protocol hash mismatch")
    require(sha256(args.nominal_root / "SHA256SUMS") == NOMINAL_SHA256SUMS_SHA256, "nominal manifest hash mismatch")
    for relative_name, expected_hash in EXPECTED_NOMINAL_TABLE_HASHES.items():
        require(sha256(args.nominal_root / relative_name) == expected_hash, f"nominal table hash mismatch: {relative_name}")
    require(not args.package_root.is_symlink(), "package root may not be a symlink")
    require(not args.archive_root.is_symlink(), "archive root may not be a symlink")
    return load_template_members(args.template_archive)


def write_final_package(context: BuildContext, records: list[dict[str, Any]], replica_start: int, replica_stop: int, runtime_archive: Path, runner_source: Path) -> None:
    package_root = context.package_root
    records = sorted(records, key=lambda item: (item["replica"], CATEGORIES.index(item["category_id"])))
    replica_count = replica_stop - replica_start
    require(len(records) == replica_count * 2, "payload record count mismatch")
    manifest = {
        "categories": list(CATEGORIES),
        "escalation_200_999_production_authorized": True,
        "outer_folds_served_per_payload": 5,
        "payload_count": len(records),
        "payloads": records,
        "production_submission_performed": False,
        "replica_count": replica_count,
        "replica_range": [replica_start, replica_stop - 1],
        "repository_head": context.repository_head,
        "schema_version": 1,
        "status": "pass_escalation800_payload_manifest_complete",
        "structure_count_per_job": STRUCTURES_PER_JOB,
        "test_payloads_opened": 0,
        "validation_payloads_opened": 0,
    }
    evidence = package_root / "evidence"
    submission = package_root / "submission"
    evidence.mkdir(parents=True, exist_ok=True)
    submission.mkdir(parents=True, exist_ok=True)
    manifest_json = evidence / "payload_manifest.json"
    atomic_write(manifest_json, json_bytes(manifest))
    manifest_tsv_lines = ["payload_key\treplica\tcategory_id\ttransferred_rows\tpayload_archive_size_bytes\tpayload_archive_sha256\n"]
    for record in records:
        manifest_tsv_lines.append(
            f"{record['payload_key']}\t{record['replica']}\t{record['category_id']}\t{record['transferred_rows']}\t"
            f"{record['payload_archive_size_bytes']}\t{record['payload_archive_sha256']}\n"
        )
    manifest_tsv = evidence / "payload_manifest.tsv"
    atomic_write(manifest_tsv, "".join(manifest_tsv_lines).encode())

    runner = submission / "run_escalation800_category_fold_job.sh"
    shutil.copyfile(runner_source, runner)
    runner.chmod(0o755)
    runner_hash = sha256(runner)
    matrix_lines = ["job_index\treplica\touter_fold\tcategory_id\tpayload_key\trole\n"]
    queue_lines = []
    job_index = 0
    by_key = {record["payload_key"]: record for record in records}
    for replica in range(replica_start, replica_stop):
        for category in CATEGORIES:
            record = by_key[f"replica_{replica:04d}__{category}"]
            for outer_fold in OUTER_FOLDS:
                return_dir = package_root / "returns" / f"job_{job_index:04d}"
                return_dir.mkdir(parents=True, exist_ok=True)
                matrix_lines.append(
                    f"{job_index}\t{replica}\t{outer_fold}\t{category}\t{record['payload_key']}\t"
                    "escalation800_selection_stability_production\n"
                )
                queue_lines.append(
                    f"{job_index}\t{replica}\t{outer_fold}\t{category}\t{context.repository_head}\t"
                    f"{record['payload_archive']}\t{record['payload_archive_name']}\t{record['payload_archive_sha256']}\t"
                    f"{RUNTIME_SHA256}\t{return_dir}\n"
                )
                job_index += 1
    require(job_index == replica_count * 10, "job count mismatch")
    matrix = submission / "escalation800_job_matrix.tsv"
    queue = submission / "escalation800_queue.items"
    atomic_write(matrix, "".join(matrix_lines).encode())
    atomic_write(queue, "".join(queue_lines).encode())
    logs = package_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    submit_text = f"""universe = vanilla
executable = {runner}
arguments = \"$(job_index) $(replica) $(outer_fold) $(category_id) $(expected_head) $(payload_basename) $(payload_sha) $(runtime_sha)\"

should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = $(payload_archive),{runtime_archive}
transfer_output_files = result_bundle.tar.gz,job_receipt.json,runner.log

getenv = False
environment = \"PYTHONNOUSERSITE=1\"

request_cpus = 1
request_memory = 2048MB
request_disk = 4000MB
+JobFlavour = \"workday\"

initialdir = $(return_dir)

log = {logs}/job_$(job_index).log
output = {logs}/job_$(job_index).out
error = {logs}/job_$(job_index).err

on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)

queue job_index, replica, outer_fold, category_id, expected_head, payload_archive, payload_basename, payload_sha, runtime_sha, return_dir from {queue}
"""
    submit_file = submission / "escalation800_category_fold.sub"
    atomic_write(submit_file, submit_text.encode())
    state = {
        "authorized_replicas": [replica_start, replica_stop - 1],
        "condor_submission_performed": False,
        "expected_condor_jobs": job_index,
        "expected_payload_archives": len(records),
        "expected_structure_evaluations": job_index * STRUCTURES_PER_JOB,
        "package_audit_json": str(evidence / "escalation800_production_package_audit.json"),
        "production_package_built": True,
        "production_submission_performed": False,
        "repository_head": context.repository_head,
        "schema_version": 1,
        "status": "pass_escalation800_production_package_built_pre_audit",
        "test_payloads_opened": 0,
        "validation_payloads_opened": 0,
    }
    atomic_write(package_root / "build" / "build_state.json", json_bytes(state))
    build_receipt = {
        "aggregation_protocol_sha256": AGGREGATION_PROTOCOL_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "draw_sha256": DRAW_SHA256,
        "job_count": job_index,
        "job_matrix_sha256": sha256(matrix),
        "materializer_sha256": MATERIALIZER_SHA256,
        "nominal_fold_table_manifest_sha256": NOMINAL_SHA256SUMS_SHA256,
        "payload_archive_count": len(records),
        "payload_archive_total_bytes": sum(int(record["payload_archive_size_bytes"]) for record in records),
        "payload_archive_root": str(context.archive_root),
        "payload_manifest_json_sha256": sha256(manifest_json),
        "payload_manifest_tsv_sha256": sha256(manifest_tsv),
        "queue_items_sha256": sha256(queue),
        "repository_head": context.repository_head,
        "tooling_repository_head": context.tooling_repository_head,
        "runner_sha256": runner_hash,
        "runtime_archive_sha256": RUNTIME_SHA256,
        "schema_version": 1,
        "status": "pass_escalation800_transfer_safe_package_build",
        "submit_file_sha256": sha256(submit_file),
        "template_archive_sha256": TEMPLATE_ARCHIVE_SHA256,
        "test_payloads_opened": 0,
        "validation_payloads_opened": 0,
    }
    atomic_write(evidence / "escalation800_package_build_receipt.json", json_bytes(build_receipt))


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    baselines = Path("/uscms_data/d3/iturkmen/hh4b_delphes/baselines")
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--expected-repository-head", required=True)
    parser.add_argument("--expected-tooling-head")
    parser.add_argument("--replica-start", type=int, default=200)
    parser.add_argument("--replica-stop", type=int, default=1000, help="exclusive")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--nominal-root", type=Path, default=baselines / "multivariate_cut_full_settings_production_v6_20260806T221023Z/fold_tables")
    parser.add_argument("--draw-parquet", type=Path, default=baselines / "multivariate_cut_selection_stability_draw_registry_v1_20260808/selection_stability_draw_counts_escalation_800.parquet")
    parser.add_argument("--materializer", type=Path, default=repository_root / "docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_reoptimization_canary_20260808_v1/evidence/selection_stability_materialize_replica.py")
    parser.add_argument("--template-archive", type=Path, default=baselines / "multivariate_cut_selection_stability_initial200_production_package_v1_20260808/archives/initial200_payload_replica_0000__exact3tag.tar.gz")
    parser.add_argument("--runtime-archive", type=Path, default=baselines / "multivariate_cut_bounded_six_job_transfer_pilot_v8_r1_20260807T004537Z/archives/hh4b_python_runtime_v8_r1.tar.gz")
    parser.add_argument("--authorization-json", type=Path, default=repository_root / "docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_escalation_200_999_authorization_20260809_v1/escalation_200_999_authorization.json")
    parser.add_argument("--aggregation-protocol", type=Path, default=repository_root / "docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_aggregation_protocol_20260808_v1/selection_stability_aggregation_protocol.json")
    parser.add_argument("--runner-source", type=Path, default=repository_root / "scripts/production/run_hh4b_stability_escalation800_category_fold_job.sh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    args.package_root = args.package_root.resolve()
    args.archive_root = (
        args.archive_root.resolve()
        if args.archive_root is not None
        else (args.package_root / "archives").resolve()
    )
    args.nominal_root = args.nominal_root.resolve()
    args.draw_parquet = args.draw_parquet.resolve()
    args.materializer = args.materializer.resolve()
    args.template_archive = args.template_archive.resolve()
    args.runtime_archive = args.runtime_archive.resolve()
    args.authorization_json = args.authorization_json.resolve()
    args.aggregation_protocol = args.aggregation_protocol.resolve()
    args.runner_source = args.runner_source.resolve()
    require(args.workers >= 1 and args.workers <= 8, "workers must be in [1, 8]")
    require(args.runner_source.is_file(), "runner source missing")
    require(not (args.package_root / "production_submission_v1").exists(), "submission evidence already exists")
    core_members = validate_inputs(args, repository_root)
    for directory in ("build/materialization_summaries", "build/payload_records", "build/partials", "build/work", "evidence", "logs", "returns", "submission"):
        (args.package_root / directory).mkdir(parents=True, exist_ok=True)
    args.archive_root.mkdir(parents=True, exist_ok=True)
    context = BuildContext(
        package_root=args.package_root,
        archive_root=args.archive_root,
        repository_head=args.expected_repository_head,
        tooling_repository_head=(
            args.expected_tooling_head
            if args.expected_tooling_head is not None
            else args.expected_repository_head
        ),
        nominal_root=args.nominal_root,
        draw_parquet=args.draw_parquet,
        materializer=args.materializer,
        python=args.python,
        core_members=core_members,
    )
    all_records: list[dict[str, Any]] = []
    total_rows = 0
    replicas = list(range(args.replica_start, args.replica_stop))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(build_replica, context, replica): replica for replica in replicas}
        completed_count = 0
        for future in as_completed(futures):
            replica, records, rows = future.result()
            all_records.extend(records)
            total_rows += rows
            completed_count += 1
            print(f"PACKAGE_BUILD_PROGRESS completed={completed_count}/{len(replicas)} replica={replica} payloads={len(all_records)}", flush=True)
    write_final_package(
        context,
        all_records,
        args.replica_start,
        args.replica_stop,
        args.runtime_archive,
        args.runner_source,
    )
    print("ESCALATION800_PACKAGE_BUILD=PASS")
    print(f"REPLICA_RANGE={args.replica_start}-{args.replica_stop - 1}")
    print(f"PAYLOAD_ARCHIVES={len(all_records)}")
    print(f"CONDOR_JOBS={(args.replica_stop - args.replica_start) * 10}")
    print(f"STRUCTURE_EVALUATIONS={(args.replica_stop - args.replica_start) * 270}")
    print(f"MATERIALIZED_ROWS={total_rows}")
    print("PRODUCTION_SUBMISSION_PERFORMED=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
