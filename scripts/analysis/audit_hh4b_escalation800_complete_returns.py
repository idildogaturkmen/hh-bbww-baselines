#!/usr/bin/env python3
"""Audit all escalation-800 selection-stability returns without extracting bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile


DEFAULT_PACKAGE = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/baselines/"
    "multivariate_cut_selection_stability_escalation800_production_package_v1_20260809"
)
CLUSTER_ID = 30020809
AUTHORITATIVE_SCHEDD = "lpcschedd5.fnal.gov"
SCIENTIFIC_HEAD = "275a2aaebe83b2f1f5e7403a245b52daf30560b4"
EXPECTED_JOBS = 8000
EXPECTED_STRUCTURES = 216000
REPLICA_MIN = 200
REPLICA_MAX = 999
RUNTIME_SHA256 = "4a9c4061e1d975f276326022454e578c5f0de3c094b9b56345f192cccc654112"
WORKER_SHA256 = "739147c33d6597018f4dce8c483278c64ff8b41533de1f1b213749dfb39332ed"
OPTIMIZER_SHA256 = "7760ee7ae309bec6cd47eb1c73ed04ec12bb805f133527498fcb75aca8088a62"
RECOVERY_CMD = Path(
    "/uscms_data/d3/iturkmen/repos/hh-bbww-baselines/scripts/production/"
    "run_hh4b_stability_escalation800_eos_recovery.sh"
)
RUNTIME = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/baselines/"
    "multivariate_cut_bounded_six_job_transfer_pilot_v8_r1_20260807T004537Z/"
    "archives/hh4b_python_runtime_v8_r1.tar.gz"
)

STRUCTURE_IDS = (
    "asymmetric_rectangular_mass__category__mass_only",
    "asymmetric_rectangular_mass__category__plus_abs_h_delta_eta",
    "asymmetric_rectangular_mass__category__plus_h2_pt",
    "asymmetric_rectangular_mass__category__plus_h2_pt_and_max_drbb",
    "asymmetric_rectangular_mass__category__plus_ht_candidate_jets",
    "asymmetric_rectangular_mass__category__plus_ht_candidate_jets_and_max_drbb",
    "asymmetric_rectangular_mass__category__plus_max_drbb",
    "asymmetric_rectangular_mass__category__plus_mhh",
    "asymmetric_rectangular_mass__category__plus_mhh_and_abs_h_delta_eta",
    "radial_mass__category__mass_only",
    "radial_mass__category__plus_abs_h_delta_eta",
    "radial_mass__category__plus_h2_pt",
    "radial_mass__category__plus_h2_pt_and_max_drbb",
    "radial_mass__category__plus_ht_candidate_jets",
    "radial_mass__category__plus_ht_candidate_jets_and_max_drbb",
    "radial_mass__category__plus_max_drbb",
    "radial_mass__category__plus_mhh",
    "radial_mass__category__plus_mhh_and_abs_h_delta_eta",
    "symmetric_rectangular_mass__category__mass_only",
    "symmetric_rectangular_mass__category__plus_abs_h_delta_eta",
    "symmetric_rectangular_mass__category__plus_h2_pt",
    "symmetric_rectangular_mass__category__plus_h2_pt_and_max_drbb",
    "symmetric_rectangular_mass__category__plus_ht_candidate_jets",
    "symmetric_rectangular_mass__category__plus_ht_candidate_jets_and_max_drbb",
    "symmetric_rectangular_mass__category__plus_max_drbb",
    "symmetric_rectangular_mass__category__plus_mhh",
    "symmetric_rectangular_mass__category__plus_mhh_and_abs_h_delta_eta",
)

STRUCTURE_HEADER = [
    "job_index", "replica", "outer_fold", "category_id", "payload_key",
    "structure_id", "category_family_id", "cut_count", "pooled_signal_efficiency",
    "pooled_background_efficiency", "pooled_inner_oof_feasible", "all_inner_support_pass",
    "pooled_support_pass", "pooled_support_failures_json", "refit_thresholds_json",
    "canonical_payload_sha256", "structure_result_sha256", "inner_crossfit_sha256",
    "execution_provenance_sha256", "runtime_seconds", "bundle_sha256", "bundle_member",
]

JOB_HEADER = [
    "job_index", "replica", "outer_fold", "category_id", "payload_key",
    "receipt_sha256", "runner_log_sha256", "bundle_sha256", "elapsed_seconds",
    "structures", "pooled_support_pass_results", "pooled_feasible_results",
    "receipt_bundle_sha_was_null", "validation_payloads_opened", "test_payloads_opened",
]

BASE_CUTS = {
    "asymmetric_rectangular_mass": {
        ("abs_mbb1_minus_125", "<"),
        ("abs_mbb2_minus_125", "<"),
    },
    "radial_mass": {("r_hh_125_125", "<")},
    "symmetric_rectangular_mass": {("max_abs_mbb_minus_125", "<")},
}
EXTRA_CUTS = {
    "mass_only": set(),
    "plus_abs_h_delta_eta": {("abs_h_delta_eta", "<")},
    "plus_h2_pt": {("h2_pt", ">")},
    "plus_h2_pt_and_max_drbb": {("h2_pt", ">"), ("max_drbb", "<")},
    "plus_ht_candidate_jets": {("ht_candidate_jets", ">")},
    "plus_ht_candidate_jets_and_max_drbb": {
        ("ht_candidate_jets", ">"),
        ("max_drbb", "<"),
    },
    "plus_max_drbb": {("max_drbb", "<")},
    "plus_mhh": {("mhh", ">")},
    "plus_mhh_and_abs_h_delta_eta": {("mhh", ">"), ("abs_h_delta_eta", "<")},
}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return sha256_bytes(encoded)


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def normalized_member_name(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    return name.rstrip("/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--final-queue-snapshot", type=Path)
    parser.add_argument("--final-queue-stderr", type=Path)
    parser.add_argument("--final-history-snapshot", type=Path)
    parser.add_argument("--final-history-stderr", type=Path)
    parser.add_argument("--audit-repository-head", required=True)
    parser.add_argument(
        "--check-job",
        type=int,
        help="Audit one complete return and print its normalized records; do not build final outputs.",
    )
    return parser.parse_args()


def load_queue_items(package: Path) -> dict[int, dict[str, object]]:
    path = package / "submission/escalation800_queue.items"
    rows: dict[int, dict[str, object]] = {}
    with path.open(newline="") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            require(len(fields) == 10, f"queue item field count mismatch: {fields!r}")
            job, replica, fold, category, head, archive, basename, payload_sha, runtime_sha, return_dir = fields
            job_index = int(job)
            require(job_index not in rows, f"duplicate queue item {job_index}")
            rows[job_index] = {
                "job_index": job_index,
                "replica": int(replica),
                "outer_fold": int(fold),
                "category_id": category,
                "repository_head": head,
                "payload_archive": archive,
                "payload_basename": basename,
                "payload_sha256": payload_sha,
                "runtime_sha256": runtime_sha,
                "return_dir": return_dir,
            }
    require(set(rows) == set(range(EXPECTED_JOBS)), "queue item job coverage is not 0..7999")
    return rows


def load_matrix(package: Path, queue_items: dict[int, dict[str, object]]) -> None:
    path = package / "submission/escalation800_job_matrix.tsv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(
            reader.fieldnames == ["job_index", "replica", "outer_fold", "category_id", "payload_key", "role"],
            "job matrix header mismatch",
        )
        count = 0
        for row in reader:
            job = int(row["job_index"])
            item = queue_items[job]
            require(job == count, f"job matrix is not in deterministic order at row {count}")
            require(int(row["replica"]) == item["replica"], f"matrix replica mismatch at job {job}")
            require(int(row["outer_fold"]) == item["outer_fold"], f"matrix fold mismatch at job {job}")
            require(row["category_id"] == item["category_id"], f"matrix category mismatch at job {job}")
            expected_key = f"replica_{int(item['replica']):04d}__{item['category_id']}"
            require(row["payload_key"] == expected_key, f"matrix payload key mismatch at job {job}")
            require(row["role"] == "escalation800_selection_stability_production", f"matrix role mismatch at job {job}")
            count += 1
    require(count == EXPECTED_JOBS, f"job matrix row count mismatch: {count}")


def load_payload_manifest(package: Path) -> tuple[dict[tuple[int, str], dict[str, object]], str]:
    path = package / "evidence/payload_manifest.json"
    manifest = json.loads(path.read_text())
    require(manifest.get("payload_count") == 1600, "payload manifest count mismatch")
    require(manifest.get("replica_range") == [REPLICA_MIN, REPLICA_MAX], "payload manifest replica range mismatch")
    require(manifest.get("replica_count") == 800, "payload manifest replica count mismatch")
    require(manifest.get("repository_head") == SCIENTIFIC_HEAD, "payload manifest scientific head mismatch")
    require(manifest.get("validation_payloads_opened") == 0, "payload manifest validation counter nonzero")
    require(manifest.get("test_payloads_opened") == 0, "payload manifest test counter nonzero")
    by_key: dict[tuple[int, str], dict[str, object]] = {}
    for row in manifest["payloads"]:
        key = (int(row["replica"]), row["category_id"])
        require(key not in by_key, f"duplicate payload manifest key: {key}")
        require(row["repository_head"] == SCIENTIFIC_HEAD, f"payload scientific head mismatch: {key}")
        require(row["structure_count"] == 27, f"payload structure count mismatch: {key}")
        require(row["results_may_enter_all1000_stability_aggregation"] is True, f"payload aggregation role mismatch: {key}")
        require(row["pilot_results_may_enter_stability_aggregation"] is False, f"payload pilot role mismatch: {key}")
        require(row["validation_payloads_opened"] == 0 and row["test_payloads_opened"] == 0, f"payload sealed counters changed: {key}")
        by_key[key] = row
    require(len(by_key) == 1600, "payload manifest unique-key count mismatch")
    return by_key, sha256_file(path)


def validate_job_mapping(job_index: int, item: dict[str, object]) -> None:
    expected_replica = REPLICA_MIN + job_index // 10
    within = job_index % 10
    expected_category = "exact3tag" if within < 5 else "ge4tag"
    expected_fold = within if within < 5 else within - 5
    require(
        (item["replica"], item["outer_fold"], item["category_id"]) ==
        (expected_replica, expected_fold, expected_category),
        f"job mapping mismatch at {job_index}",
    )
    require(item["repository_head"] == SCIENTIFIC_HEAD, f"scientific head mismatch at job {job_index}")
    require(item["runtime_sha256"] == RUNTIME_SHA256, f"runtime SHA mismatch at job {job_index}")
    require(item["return_dir"] == str(DEFAULT_PACKAGE / f"returns/job_{job_index:04d}"), f"return path mismatch at job {job_index}")


def read_bundle(path: Path) -> dict[str, bytes]:
    blobs: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            normalized = normalized_member_name(member.name)
            pure = PurePosixPath(normalized)
            require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe bundle member: {member.name}")
            require(not member.issym() and not member.islnk(), f"link in bundle: {member.name}")
            if member.isfile():
                require(normalized not in blobs, f"duplicate bundle member: {normalized}")
                handle = archive.extractfile(member)
                require(handle is not None, f"cannot read bundle member: {member.name}")
                blobs[normalized] = handle.read()
    return blobs


def audit_one_job(
    package: Path,
    job_index: int,
    item: dict[str, object],
    payload_manifest: dict[tuple[int, str], dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    validate_job_mapping(job_index, item)
    replica = int(item["replica"])
    outer_fold = int(item["outer_fold"])
    category = str(item["category_id"])
    payload_key = f"replica_{replica:04d}__{category}"
    payload = payload_manifest[(replica, category)]
    require(payload["payload_key"] == payload_key, f"payload key mismatch at job {job_index}")
    require(payload["payload_archive"] == item["payload_archive"], f"payload archive mismatch at job {job_index}")
    require(payload["payload_archive_name"] == item["payload_basename"], f"payload basename mismatch at job {job_index}")
    require(payload["payload_archive_sha256"] == item["payload_sha256"], f"payload SHA mismatch at job {job_index}")

    return_dir = package / f"returns/job_{job_index:04d}"
    expected_files = {"job_receipt.json", "result_bundle.tar.gz", "runner.log"}
    actual_files = {path.name for path in return_dir.iterdir()}
    require(actual_files == expected_files, f"return file set mismatch at job {job_index}: {actual_files}")
    receipt_path = return_dir / "job_receipt.json"
    bundle_path = return_dir / "result_bundle.tar.gz"
    runner_path = return_dir / "runner.log"
    for path in (receipt_path, bundle_path, runner_path):
        require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0, f"bad return file: {path}")
    receipt_sha = sha256_file(receipt_path)
    bundle_sha = sha256_file(bundle_path)
    runner_sha = sha256_file(runner_path)
    receipt = json.loads(receipt_path.read_text())
    required_receipt = {
        "schema_version": 1,
        "status": "selection_stability_escalation800_category_fold_job_complete",
        "job_index": job_index,
        "bootstrap_replica": replica,
        "outer_fold": outer_fold,
        "category_id": category,
        "expected_repository_head": SCIENTIFIC_HEAD,
        "runner_rc": "0",
        "structures_expected": 27,
        "structures_attempted": 27,
        "structures_completed_shell": 27,
        "valid_structure_results": 27,
        "expected_payload_archive_sha256": item["payload_sha256"],
        "expected_runtime_archive_sha256": RUNTIME_SHA256,
        "result_bundle_sha256": None,
        "runner_log_sha256": runner_sha,
        "escalation_200_999_production_authorized": True,
        "results_may_enter_all1000_stability_aggregation": True,
        "pilot_results_may_enter_stability_aggregation": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    for key, expected in required_receipt.items():
        require(receipt.get(key) == expected, f"receipt {key} mismatch at job {job_index}: {receipt.get(key)!r}")
    require(int(receipt.get("elapsed_seconds", 0)) > 0, f"non-positive receipt elapsed time at job {job_index}")

    runner_text = runner_path.read_text()
    require("SELECTION_STABILITY_ESCALATION800_CATEGORY_FOLD_START" in runner_text, f"runner start marker absent at job {job_index}")
    require(f"ACTUAL_PAYLOAD_SHA={item['payload_sha256']}" in runner_text, f"runner payload SHA marker mismatch at job {job_index}")
    require(f"ACTUAL_RUNTIME_SHA={RUNTIME_SHA256}" in runner_text, f"runner runtime SHA marker mismatch at job {job_index}")
    require(runner_text.count("RUN_STRUCTURE_INDEX=") == 27, f"runner structure marker count mismatch at job {job_index}")
    require("STRUCTURE_WORKER_FAILURE" not in runner_text, f"runner structure failure marker at job {job_index}")
    require("STRUCTURES_ATTEMPTED=27" in runner_text and "STRUCTURES_COMPLETED=27" in runner_text, f"runner completion counters mismatch at job {job_index}")
    require("RUNNER_RC=0" in runner_text, f"runner RC marker mismatch at job {job_index}")

    blobs = read_bundle(bundle_path)
    expected_bundle_files = {"runtime_config.json"}
    for structure in STRUCTURE_IDS:
        job_id = f"outer{outer_fold}__{category}__{structure}"
        expected_bundle_files.update({
            f"results/{job_id}/structure_result.json",
            f"results/{job_id}/inner_crossfit.tsv",
            f"results/{job_id}/SHA256SUMS",
        })
    require(set(blobs) == expected_bundle_files, f"bundle inventory mismatch at job {job_index}")

    structure_rows: list[dict[str, object]] = []
    feasible_count = 0
    support_count = 0
    for structure in STRUCTURE_IDS:
        job_id = f"outer{outer_fold}__{category}__{structure}"
        prefix = f"results/{job_id}/"
        checksum_rows = blobs[prefix + "SHA256SUMS"].decode().splitlines()
        checksum_names = set()
        for line in checksum_rows:
            digest, relative = line.split("  ./", 1)
            require(relative in {"inner_crossfit.tsv", "structure_result.json"}, f"unexpected internal target at job {job_index}: {relative}")
            require(sha256_bytes(blobs[prefix + relative]) == digest, f"internal SHA mismatch at job {job_index}: {relative}")
            checksum_names.add(relative)
        require(checksum_names == {"inner_crossfit.tsv", "structure_result.json"}, f"internal checksum coverage mismatch at job {job_index}")

        result_bytes = blobs[prefix + "structure_result.json"]
        crossfit_bytes = blobs[prefix + "inner_crossfit.tsv"]
        result = json.loads(result_bytes)
        required_result = {
            "schema_version": 1,
            "status": "pass_structure_job_complete",
            "job_id": job_id,
            "repository_head": SCIENTIFIC_HEAD,
            "execution_provenance_mode": "transferred_manifest",
            "execution_provenance_sha256": payload["execution_provenance_sha256"],
            "execution_worker_sha256": WORKER_SHA256,
            "execution_optimizer_sha256": OPTIMIZER_SHA256,
            "outer_fold": outer_fold,
            "development_folds": [fold for fold in range(5) if fold != outer_fold],
            "category_id": category,
            "structure_id": structure,
            "category_family_id": f"{category}::{structure}",
            "outer_fold_used_for_selection": False,
            "validation_payloads_opened": 0,
            "test_payloads_opened": 0,
        }
        for key, expected in required_result.items():
            require(result.get(key) == expected, f"result {key} mismatch at job {job_index}, structure {structure}")
        family, suffix = structure.split("__category__", 1)
        expected_cuts = BASE_CUTS[family] | EXTRA_CUTS[suffix]
        actual_cuts = {
            (cut.get("variable"), cut.get("operator")) for cut in result["cuts"]
        }
        require(
            len(result["cuts"]) == len(actual_cuts) and actual_cuts == expected_cuts,
            f"cut registry mismatch at job {job_index}, structure {structure}",
        )
        thresholds = result["refit_thresholds"]
        require(
            isinstance(thresholds, dict)
            and set(thresholds) == {variable for variable, _ in expected_cuts}
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in thresholds.values()
            ),
            f"refit threshold registry mismatch at job {job_index}, structure {structure}",
        )
        canonical_payload = dict(result)
        claimed_canonical = canonical_payload.pop("canonical_payload_sha256", None)
        canonical_payload.pop("runtime_seconds", None)
        require(claimed_canonical == canonical_sha256(canonical_payload), f"canonical result SHA mismatch at job {job_index}, structure {structure}")
        runtime_seconds = float(result.get("runtime_seconds", -1.0))
        require(math.isfinite(runtime_seconds) and runtime_seconds >= 0.0, f"invalid structure runtime at job {job_index}, structure {structure}")

        crossfit_rows = list(csv.DictReader(io.StringIO(crossfit_bytes.decode()), delimiter="\t"))
        require(len(crossfit_rows) == 4, f"crossfit row count mismatch at job {job_index}, structure {structure}")
        development_folds = set(range(5)) - {outer_fold}
        require({int(row["inner_heldout_fold"]) for row in crossfit_rows} == development_folds, f"crossfit fold coverage mismatch at job {job_index}, structure {structure}")
        for row in crossfit_rows:
            require(int(row["outer_fold"]) == outer_fold, f"crossfit outer fold mismatch at job {job_index}")
            require(row["category_id"] == category and row["structure_id"] == structure, f"crossfit identity mismatch at job {job_index}")

        metric = result["pooled_inner_oof_metric"]
        signal_eff = float(metric["signal_efficiency"])
        background_eff = float(metric["background_efficiency"])
        require(math.isfinite(signal_eff) and 0.0 <= signal_eff <= 1.0, f"signal efficiency invalid at job {job_index}")
        require(math.isfinite(background_eff) and 0.0 <= background_eff <= 1.0, f"background efficiency invalid at job {job_index}")
        feasible = bool(result["pooled_inner_oof_feasible"])
        pooled_support = bool(result["pooled_support_pass"])
        all_inner_support = bool(result["all_inner_support_pass"])
        support_failures = result["pooled_support_failures"]
        require(
            isinstance(support_failures, list)
            and all(isinstance(value, str) for value in support_failures)
            and (not pooled_support or support_failures == []),
            f"pooled support diagnostics mismatch at job {job_index}, structure {structure}",
        )
        feasible_count += int(feasible)
        support_count += int(pooled_support)
        structure_rows.append({
            "job_index": job_index,
            "replica": replica,
            "outer_fold": outer_fold,
            "category_id": category,
            "payload_key": payload_key,
            "structure_id": structure,
            "category_family_id": result["category_family_id"],
            "cut_count": len(result["cuts"]),
            "pooled_signal_efficiency": signal_eff,
            "pooled_background_efficiency": background_eff,
            "pooled_inner_oof_feasible": feasible,
            "all_inner_support_pass": all_inner_support,
            "pooled_support_pass": pooled_support,
            "pooled_support_failures_json": canonical_json(result["pooled_support_failures"]),
            "refit_thresholds_json": canonical_json(result["refit_thresholds"]),
            "canonical_payload_sha256": claimed_canonical,
            "structure_result_sha256": sha256_bytes(result_bytes),
            "inner_crossfit_sha256": sha256_bytes(crossfit_bytes),
            "execution_provenance_sha256": result["execution_provenance_sha256"],
            "runtime_seconds": runtime_seconds,
            "bundle_sha256": bundle_sha,
            "bundle_member": prefix + "structure_result.json",
        })

    require(receipt.get("pooled_feasible_results") == feasible_count, f"receipt feasible count mismatch at job {job_index}")
    require(receipt.get("pooled_support_pass_results") == support_count, f"receipt support count mismatch at job {job_index}")
    job_row = {
        "job_index": job_index,
        "replica": replica,
        "outer_fold": outer_fold,
        "category_id": category,
        "payload_key": payload_key,
        "receipt_sha256": receipt_sha,
        "runner_log_sha256": runner_sha,
        "bundle_sha256": bundle_sha,
        "elapsed_seconds": int(receipt["elapsed_seconds"]),
        "structures": 27,
        "pooled_support_pass_results": support_count,
        "pooled_feasible_results": feasible_count,
        "receipt_bundle_sha_was_null": True,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    return job_row, structure_rows


def validate_scheduler_snapshots(args: argparse.Namespace, package: Path) -> dict[str, object]:
    required_paths = (
        args.final_queue_snapshot,
        args.final_queue_stderr,
        args.final_history_snapshot,
        args.final_history_stderr,
    )
    require(all(path is not None for path in required_paths), "final scheduler snapshot paths are required")
    for path in required_paths:
        require(path.is_file() and not path.is_symlink(), f"bad scheduler snapshot path: {path}")
    require(args.final_queue_snapshot.read_bytes() == b"", "final queue snapshot is not empty")
    require(args.final_queue_stderr.read_bytes() == b"", "final queue stderr is not empty")
    require(args.final_history_stderr.read_bytes() == b"", "final history stderr is not empty")
    expected_transfer = f"{package / 'submission/run_escalation800_category_fold_job.sh'},{RUNTIME}"
    seen = set()
    rows = 0
    for line_number, line in enumerate(args.final_history_snapshot.read_text().splitlines(), start=1):
        fields = line.split()
        require(len(fields) == 9, f"history line {line_number} field count mismatch: {fields!r}")
        cluster, proc, status, exit_code, exit_signal, starts, cmd, transfer_input, iwd = fields
        proc_id = int(proc)
        require(cluster == str(CLUSTER_ID), f"history cluster mismatch at line {line_number}")
        require(0 <= proc_id < EXPECTED_JOBS and proc_id not in seen, f"history proc mismatch/duplicate: {proc_id}")
        require(status == "4" and exit_code == "0", f"non-clean history completion at proc {proc_id}")
        require(exit_signal.lower() in {"false", "0"}, f"signal exit at proc {proc_id}")
        require(starts == "1", f"unexpected executable start count at proc {proc_id}: {starts}")
        require(cmd == str(RECOVERY_CMD), f"history executable mismatch at proc {proc_id}")
        require(transfer_input == expected_transfer, f"history transfer input mismatch at proc {proc_id}")
        require(iwd == str(package / f"returns/job_{proc_id:04d}"), f"history Iwd mismatch at proc {proc_id}")
        seen.add(proc_id)
        rows += 1
    require(rows == EXPECTED_JOBS and seen == set(range(EXPECTED_JOBS)), f"history coverage mismatch: {rows}")
    return {
        "queue_snapshot_sha256": sha256_file(args.final_queue_snapshot),
        "queue_stderr_sha256": sha256_file(args.final_queue_stderr),
        "history_snapshot_sha256": sha256_file(args.final_history_snapshot),
        "history_stderr_sha256": sha256_file(args.final_history_stderr),
        "history_rows": rows,
    }


def output_shard_name(shard: int) -> str:
    low = REPLICA_MIN + shard * 100
    high = low + 99
    return f"structure_result_inventory_replicas_{low:04d}_{high:04d}.tsv"


def run_full_audit(
    args: argparse.Namespace,
    package: Path,
    queue_items: dict[int, dict[str, object]],
    payload_manifest: dict[tuple[int, str], dict[str, object]],
    payload_manifest_sha: str,
) -> int:
    require(args.output_root is not None, "--output-root is required for a full audit")
    output_root = args.output_root.resolve()
    require(not output_root.exists() and not output_root.is_symlink(), f"output root already exists: {output_root}")
    require(output_root.parent.is_dir() and not output_root.parent.is_symlink(), f"output parent is invalid: {output_root.parent}")
    scheduler = validate_scheduler_snapshots(args, package)
    build_root = output_root.parent / f".{output_root.name}.build.{os.getpid()}"
    require(not build_root.exists() and not build_root.is_symlink(), f"build root already exists: {build_root}")
    build_root.mkdir(mode=0o755)
    shutil.copyfile(args.final_queue_snapshot, build_root / "final_queue_snapshot.txt")
    shutil.copyfile(args.final_queue_stderr, build_root / "final_queue_snapshot.txt.stderr")
    shutil.copyfile(args.final_history_snapshot, build_root / "final_history_snapshot.txt")
    shutil.copyfile(args.final_history_stderr, build_root / "final_history_snapshot.txt.stderr")

    shard_handles = []
    shard_writers = []
    shard_rows = [0] * 8
    for shard in range(8):
        handle = (build_root / output_shard_name(shard)).open("w", newline="")
        writer = csv.DictWriter(handle, fieldnames=STRUCTURE_HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        shard_handles.append(handle)
        shard_writers.append(writer)
    job_path = build_root / "job_return_audit.tsv"
    job_handle = job_path.open("w", newline="")
    job_writer = csv.DictWriter(job_handle, fieldnames=JOB_HEADER, delimiter="\t", lineterminator="\n")
    job_writer.writeheader()

    feasible_total = 0
    support_total = 0
    all_inner_support_total = 0
    receipt_shas = set()
    bundle_shas = set()
    try:
        for job_index in range(EXPECTED_JOBS):
            job_row, structure_rows = audit_one_job(
                package, job_index, queue_items[job_index], payload_manifest
            )
            require(job_row["receipt_sha256"] not in receipt_shas, f"duplicate receipt SHA at job {job_index}")
            require(job_row["bundle_sha256"] not in bundle_shas, f"duplicate bundle SHA at job {job_index}")
            receipt_shas.add(job_row["receipt_sha256"])
            bundle_shas.add(job_row["bundle_sha256"])
            job_writer.writerow(job_row)
            shard = (int(job_row["replica"]) - REPLICA_MIN) // 100
            for row in structure_rows:
                shard_writers[shard].writerow(row)
                shard_rows[shard] += 1
                feasible_total += int(bool(row["pooled_inner_oof_feasible"]))
                support_total += int(bool(row["pooled_support_pass"]))
                all_inner_support_total += int(bool(row["all_inner_support_pass"]))
            if (job_index + 1) % 100 == 0:
                print(f"AUDITED_JOBS={job_index + 1}", flush=True)
    finally:
        job_handle.close()
        for handle in shard_handles:
            handle.close()

    require(sum(shard_rows) == EXPECTED_STRUCTURES, f"structure row total mismatch: {sum(shard_rows)}")
    require(all(rows == 27000 for rows in shard_rows), f"structure shard row mismatch: {shard_rows}")
    shard_manifest = []
    for shard in range(8):
        path = build_root / output_shard_name(shard)
        low = REPLICA_MIN + shard * 100
        shard_manifest.append({
            "path": path.name,
            "replica_min": low,
            "replica_max": low + 99,
            "rows_excluding_header": shard_rows[shard],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    inventory_manifest = {
        "schema_version": 1,
        "status": "pass_escalation800_structure_result_inventory_sharded",
        "header": STRUCTURE_HEADER,
        "total_rows_excluding_headers": EXPECTED_STRUCTURES,
        "shard_count": 8,
        "shards": shard_manifest,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    inventory_manifest_path = build_root / "structure_result_inventory_manifest.json"
    inventory_manifest_path.write_text(json.dumps(inventory_manifest, indent=2, sort_keys=True) + "\n")
    job_sha = sha256_file(job_path)
    audit = {
        "schema_version": 1,
        "status": "pass_complete_escalation800_8000_job_216000_structure_return_audit",
        "cluster_id": CLUSTER_ID,
        "authoritative_schedd": AUTHORITATIVE_SCHEDD,
        "jobs_expected": EXPECTED_JOBS,
        "jobs_audited": EXPECTED_JOBS,
        "structure_results_expected": EXPECTED_STRUCTURES,
        "structure_results_audited": EXPECTED_STRUCTURES,
        "structures_per_job": 27,
        "replica_count": 800,
        "replicas": [REPLICA_MIN, REPLICA_MAX],
        "categories": ["exact3tag", "ge4tag"],
        "outer_folds": [0, 1, 2, 3, 4],
        "audit_repository_head": args.audit_repository_head,
        "scientific_package_head": SCIENTIFIC_HEAD,
        "payload_manifest_sha256": payload_manifest_sha,
        "job_return_audit_sha256": job_sha,
        "structure_result_inventory_manifest_sha256": sha256_file(inventory_manifest_path),
        "receipt_result_bundle_sha256_null_count": EXPECTED_JOBS,
        "receipt_result_bundle_sha256_present_count": 0,
        "all_job_receipts_clean": True,
        "all_runner_logs_match_receipt_sha256": True,
        "all_structure_internal_sha256s_verified": True,
        "all_structure_canonical_payload_sha256s_recomputed": True,
        "all_inner_crossfit_four_fold_coverage_verified": True,
        "all_execution_provenance_sha256s_match_frozen_payload_manifest": True,
        "all_execution_worker_sha256s_match": True,
        "all_execution_optimizer_sha256s_match": True,
        "all_outer_fold_used_for_selection_false": True,
        "all_validation_payloads_opened_zero": True,
        "all_test_payloads_opened_zero": True,
        "bundle_sha256s_recomputed_in_audit_inventory": True,
        "exact_structure_set_per_category_fold_verified": True,
        "scientific_outcome_diagnostics": {
            "pooled_inner_oof_feasible_structure_results": feasible_total,
            "pooled_inner_oof_infeasible_structure_results": EXPECTED_STRUCTURES - feasible_total,
            "pooled_support_pass_structure_results": support_total,
            "pooled_support_fail_structure_results": EXPECTED_STRUCTURES - support_total,
            "all_inner_support_pass_structure_results": all_inner_support_total,
            "all_inner_support_fail_structure_results": EXPECTED_STRUCTURES - all_inner_support_total,
            "infeasible_or_support_failed_structure_can_be_valid_execution_result": True,
            "these_counts_are_not_a_return_audit_failure_gate": True,
        },
        "scheduler_snapshot_audit": scheduler,
        "aggregation_performed": False,
        "fold_level_winners_selected": False,
        "replica_level_stability_statistics_computed": False,
        "pilot_results_enter_stability_aggregation": False,
        "nominal_selection_changed": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "never_resubmit_cluster": True,
        "next": "freeze_complete_escalation800_return_audit_before_all1000_predeclared_aggregation",
    }
    audit_path = build_root / "complete_return_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    audit_lines = [
        "PASS=TRUE",
        "COMPLETE_ESCALATION800_RETURN_AUDIT=PASS",
        f"CLUSTER_ID={CLUSTER_ID}",
        f"AUTHORITATIVE_SCHEDD={AUTHORITATIVE_SCHEDD}",
        f"JOBS_AUDITED={EXPECTED_JOBS}",
        f"STRUCTURE_RESULTS_AUDITED={EXPECTED_STRUCTURES}",
        "EXACT_27_STRUCTURE_SET_PER_JOB=PASS",
        "ALL_STRUCTURE_INTERNAL_SHA256S=PASS",
        "ALL_CANONICAL_PAYLOAD_SHA256S=PASS",
        "ALL_INNER_CROSSFIT_FOLD_COVERAGE=PASS",
        "ALL_EXECUTION_PROVENANCE=PASS",
        "ALL_OUTER_FOLD_USED_FOR_SELECTION=FALSE",
        f"POOLED_INNER_OOF_FEASIBLE_STRUCTURE_RESULTS={feasible_total}",
        f"POOLED_INNER_OOF_INFEASIBLE_STRUCTURE_RESULTS={EXPECTED_STRUCTURES - feasible_total}",
        f"POOLED_SUPPORT_PASS_STRUCTURE_RESULTS={support_total}",
        f"POOLED_SUPPORT_FAIL_STRUCTURE_RESULTS={EXPECTED_STRUCTURES - support_total}",
        "SCIENTIFIC_FEASIBILITY_SUPPORT_COUNTS_USED_AS_INFRASTRUCTURE_GATE=FALSE",
        "AGGREGATION_PERFORMED=FALSE",
        "PILOT_RESULTS_ENTER_STABILITY_AGGREGATION=FALSE",
        "NOMINAL_SELECTION_CHANGED=FALSE",
        "VALIDATION_PAYLOADS_OPENED=0",
        "TEST_PAYLOADS_OPENED=0",
        "DO_NOT_RESUBMIT_CLUSTER_30020809=TRUE",
        "NEXT=FREEZE_COMPLETE_ESCALATION800_RETURN_AUDIT_BEFORE_ALL1000_AGGREGATION",
    ]
    (build_root / "complete_return_audit.txt").write_text("\n".join(audit_lines) + "\n")
    os.replace(build_root, output_root)
    print("ESCALATION800_COMPLETE_RETURN_AUDIT=PASS")
    print(f"JOBS_AUDITED={EXPECTED_JOBS}")
    print(f"STRUCTURE_RESULTS_AUDITED={EXPECTED_STRUCTURES}")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")
    return 0


def main() -> int:
    args = parse_args()
    package = args.package_root.resolve()
    require(package == DEFAULT_PACKAGE, f"unexpected package root: {package}")
    require(package.is_dir() and not package.is_symlink(), f"package root is invalid: {package}")
    require(len(STRUCTURE_IDS) == 27 and tuple(sorted(STRUCTURE_IDS)) == STRUCTURE_IDS, "structure registry invariant failed")
    require(sha256_file(RUNTIME) == RUNTIME_SHA256, "runtime archive SHA changed")
    queue_items = load_queue_items(package)
    load_matrix(package, queue_items)
    payload_manifest, payload_manifest_sha = load_payload_manifest(package)
    if args.check_job is not None:
        require(0 <= args.check_job < EXPECTED_JOBS, "--check-job out of range")
        job_row, structure_rows = audit_one_job(
            package, args.check_job, queue_items[args.check_job], payload_manifest
        )
        print(json.dumps({
            "status": "pass_single_escalation800_complete_return_check",
            "job": job_row,
            "structure_results": structure_rows,
            "validation_payloads_opened": 0,
            "test_payloads_opened": 0,
        }, indent=2, sort_keys=True))
        return 0
    return run_full_audit(
        args, package, queue_items, payload_manifest, payload_manifest_sha
    )


if __name__ == "__main__":
    raise SystemExit(main())
