#!/usr/bin/env python3
"""Independent fail-closed audit of the escalation-800 recovery canary."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


PACKAGE = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/baselines/"
    "multivariate_cut_selection_stability_escalation800_production_package_v1_20260809"
)
REPO = Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines")
SCHEDD = "lpcschedd5.fnal.gov"
CLUSTER = 30020809
PROC = 0
EXECUTION_HEAD = "275a2aaebe83b2f1f5e7403a245b52daf30560b4"
RECOVERY = REPO / "scripts/production/run_hh4b_stability_escalation800_eos_recovery.sh"
RUNNER = PACKAGE / "submission/run_escalation800_category_fold_job.sh"
RUNTIME = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/baselines/"
    "multivariate_cut_bounded_six_job_transfer_pilot_v8_r1_20260807T004537Z/"
    "archives/hh4b_python_runtime_v8_r1.tar.gz"
)
PAYLOAD = Path(
    "/eos/uscms/store/user/iturkmen/hh4b_cut_baseline/"
    "selection_stability_escalation800_production_package_v1_20260809/archives/"
    "escalation800_payload_replica_0200__exact3tag.tar.gz"
)
RETURN_DIR = PACKAGE / "returns/job_0000"
EVENT_LOG = PACKAGE / "logs/job_0.log"

EXPECTED = {
    "recovery_sha256": "ed687c9cda7dc27c51c65b1a789479c1612ce402dcb2ebd59a7cdeca67bd4821",
    "runner_sha256": "b5a0f96950f1b2cba99e1ed173b33a70630f10f2e1c516090920dea7137c15e5",
    "runtime_sha256": "4a9c4061e1d975f276326022454e578c5f0de3c094b9b56345f192cccc654112",
    "payload_sha256": "3c847483807a7e00599b45a784832acf56f0fa10acf67218c2d1dd6c3e47d135",
}
RECOVERY_TRANSFER_INPUT = f"{RUNNER},{RUNTIME}"


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


def checked(command: list[str]) -> bytes:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise AuditError(
            f"command failed rc={result.returncode}: {command!r}; "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}"
        )
    return result.stdout


def scheduler_audit() -> dict[str, object]:
    queue_raw = checked([
        "/usr/bin/bash", "/usr/local/bin/condor_q", "-name", SCHEDD,
        f"{CLUSTER}.{PROC}", "-json", "-attributes",
        "ClusterId,ProcId,JobStatus,Cmd,TransferInput,Iwd",
    ])
    queue = json.loads(queue_raw) if queue_raw.strip() else []
    require(queue == [], f"completed canary remains in queue: {queue!r}")

    history = json.loads(checked([
        "/usr/bin/bash", "/usr/local/bin/condor_history", "-name", SCHEDD,
        "-constraint", f"ClusterId == {CLUSTER} && ProcId == {PROC}",
        "-match", "20", "-json", "-attributes",
        "ClusterId,ProcId,JobStatus,ExitCode,ExitBySignal,CompletionDate,"
        "Cmd,TransferInput,Iwd,NumJobStarts,NumShadowStarts",
    ]))
    require(len(history) == 1, f"expected exactly one history ad: {history!r}")
    row = history[0]
    required = {
        "ClusterId": CLUSTER,
        "ProcId": PROC,
        "JobStatus": 4,
        "ExitCode": 0,
        "ExitBySignal": False,
        "Cmd": str(RECOVERY),
        "TransferInput": RECOVERY_TRANSFER_INPUT,
        "Iwd": str(RETURN_DIR),
    }
    for key, expected in required.items():
        require(row.get(key) == expected, f"history {key} mismatch: {row.get(key)!r} != {expected!r}")
    require(int(row.get("CompletionDate", 0)) > 0, "history CompletionDate is absent")

    event_text = EVENT_LOG.read_text()
    require("Job executing on host" in event_text, "event log has no execution event")
    require("Job terminated" in event_text, "event log has no termination event")
    require("Normal termination (return value 0)" in event_text, "event log has no clean return-value-zero termination")
    return {
        "queue_rows_for_canary": 0,
        "history_rows_for_canary": 1,
        "history_ad": row,
        "event_log_sha256": sha256_file(EVENT_LOG),
    }


def payload_contract() -> dict[str, object]:
    hashes = {
        "recovery_sha256": sha256_file(RECOVERY),
        "runner_sha256": sha256_file(RUNNER),
        "runtime_sha256": sha256_file(RUNTIME),
        "payload_sha256": sha256_file(PAYLOAD),
    }
    require(hashes == EXPECTED, f"immutable input hashes changed: {hashes!r}")
    with tarfile.open(PAYLOAD, "r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
        required_names = {
            "payload/structure_ids.txt",
            "payload/execution_provenance.json",
            "payload/code/hh4b_multivariate_cut_structure_worker.py",
            "payload/code/hh4b_multivariate_cut_optimizer.py",
        }
        require(required_names <= names, f"payload contract members absent: {required_names - names}")
        blobs = {}
        for name in required_names:
            handle = archive.extractfile(name)
            require(handle is not None, f"cannot read payload member {name}")
            blobs[name] = handle.read()
    structures = tuple(line for line in blobs["payload/structure_ids.txt"].decode().splitlines() if line)
    require(len(structures) == 27 and len(set(structures)) == 27, "payload does not declare 27 unique structures")
    provenance = json.loads(blobs["payload/execution_provenance.json"])
    require(provenance.get("repository_head") == EXECUTION_HEAD, "payload execution head mismatch")
    require(provenance.get("transferred_category_id") == "exact3tag", "payload category mismatch")
    require(int(provenance.get("bootstrap_replica", -1)) == 200, "payload replica mismatch")
    require(provenance.get("results_may_enter_all1000_stability_aggregation") is True, "payload aggregation role mismatch")
    require(provenance.get("pilot_results_may_enter_stability_aggregation") is False, "payload pilot role mismatch")
    require(int(provenance.get("validation_payloads_opened", -1)) == 0, "payload validation counter nonzero")
    require(int(provenance.get("test_payloads_opened", -1)) == 0, "payload test counter nonzero")
    return {
        "hashes": hashes,
        "structure_ids": structures,
        "execution_provenance_sha256": sha256_bytes(blobs["payload/execution_provenance.json"]),
        "execution_worker_sha256": sha256_bytes(blobs["payload/code/hh4b_multivariate_cut_structure_worker.py"]),
        "execution_optimizer_sha256": sha256_bytes(blobs["payload/code/hh4b_multivariate_cut_optimizer.py"]),
    }


def normalized_member_name(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    return name.rstrip("/")


def canonical_result_sha(payload: dict[str, object]) -> str:
    canonical = dict(payload)
    claimed = canonical.pop("canonical_payload_sha256", None)
    canonical.pop("runtime_seconds", None)
    observed = sha256_bytes(json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode())
    require(claimed == observed, f"canonical result SHA mismatch: {claimed!r} != {observed}")
    return observed


def return_audit(contract: dict[str, object]) -> dict[str, object]:
    expected_names = {"job_receipt.json", "result_bundle.tar.gz", "runner.log"}
    actual_names = {path.name for path in RETURN_DIR.iterdir()}
    require(actual_names == expected_names, f"return file set mismatch: {actual_names!r}")
    paths = {name: RETURN_DIR / name for name in expected_names}
    for name, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"return is not a regular file: {name}")
        require(path.stat().st_size > 0, f"empty return file: {name}")
    return_hashes = {name: sha256_file(path) for name, path in paths.items()}

    receipt = json.loads(paths["job_receipt.json"].read_text())
    required_receipt = {
        "schema_version": 1,
        "status": "selection_stability_escalation800_category_fold_job_complete",
        "job_index": 0,
        "bootstrap_replica": 200,
        "outer_fold": 0,
        "category_id": "exact3tag",
        "expected_repository_head": EXECUTION_HEAD,
        "runner_rc": "0",
        "structures_expected": 27,
        "structures_attempted": 27,
        "structures_completed_shell": 27,
        "valid_structure_results": 27,
        "expected_payload_archive_sha256": EXPECTED["payload_sha256"],
        "expected_runtime_archive_sha256": EXPECTED["runtime_sha256"],
        "result_bundle_sha256": None,
        "runner_log_sha256": return_hashes["runner.log"],
        "escalation_200_999_production_authorized": True,
        "results_may_enter_all1000_stability_aggregation": True,
        "pilot_results_may_enter_stability_aggregation": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    for key, expected in required_receipt.items():
        require(receipt.get(key) == expected, f"receipt {key} mismatch: {receipt.get(key)!r} != {expected!r}")
    require(int(receipt.get("elapsed_seconds", 0)) > 0, "receipt elapsed time is invalid")

    runner_text = paths["runner.log"].read_text()
    require("SELECTION_STABILITY_ESCALATION800_CATEGORY_FOLD_START" in runner_text, "runner start marker absent")
    require(f"ACTUAL_PAYLOAD_SHA={EXPECTED['payload_sha256']}" in runner_text, "runner payload hash marker mismatch")
    require(f"ACTUAL_RUNTIME_SHA={EXPECTED['runtime_sha256']}" in runner_text, "runner runtime hash marker mismatch")
    require(runner_text.count("RUN_STRUCTURE_INDEX=") == 27, "runner does not record exactly 27 structures")
    require("STRUCTURE_WORKER_FAILURE" not in runner_text, "runner records a structure failure")
    require("STRUCTURES_ATTEMPTED=27" in runner_text, "runner attempted count mismatch")
    require("STRUCTURES_COMPLETED=27" in runner_text, "runner completed count mismatch")
    require("RUNNER_RC=0" in runner_text, "runner RC marker mismatch")

    expected_structures = set(contract["structure_ids"])
    result_rows = []
    with tarfile.open(paths["result_bundle.tar.gz"], "r:gz") as archive:
        members = archive.getmembers()
        blobs: dict[str, bytes] = {}
        for member in members:
            normalized = normalized_member_name(member.name)
            pure = PurePosixPath(normalized)
            require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe bundle member: {member.name}")
            require(not member.issym() and not member.islnk(), f"link in bundle: {member.name}")
            if member.isfile():
                handle = archive.extractfile(member)
                require(handle is not None, f"cannot read bundle member: {member.name}")
                blobs[normalized] = handle.read()

    expected_files = {"runtime_config.json"}
    for structure in expected_structures:
        job_id = f"outer0__exact3tag__{structure}"
        expected_files.update({
            f"results/{job_id}/structure_result.json",
            f"results/{job_id}/inner_crossfit.tsv",
            f"results/{job_id}/SHA256SUMS",
        })
    require(set(blobs) == expected_files, f"bundle file inventory mismatch: missing={expected_files-set(blobs)}, extra={set(blobs)-expected_files}")

    for structure in sorted(expected_structures):
        job_id = f"outer0__exact3tag__{structure}"
        prefix = f"results/{job_id}/"
        checksums = blobs[prefix + "SHA256SUMS"].decode().splitlines()
        observed_checksum_names = set()
        for line in checksums:
            digest, relative = line.split("  ./", 1)
            require(relative in {"inner_crossfit.tsv", "structure_result.json"}, f"unexpected internal checksum target: {relative}")
            observed_checksum_names.add(relative)
            require(sha256_bytes(blobs[prefix + relative]) == digest, f"internal checksum mismatch for {structure}/{relative}")
        require(observed_checksum_names == {"inner_crossfit.tsv", "structure_result.json"}, f"incomplete internal checksums for {structure}")

        result = json.loads(blobs[prefix + "structure_result.json"])
        required_result = {
            "schema_version": 1,
            "status": "pass_structure_job_complete",
            "job_id": job_id,
            "repository_head": EXECUTION_HEAD,
            "execution_provenance_mode": "transferred_manifest",
            "execution_provenance_sha256": contract["execution_provenance_sha256"],
            "execution_worker_sha256": contract["execution_worker_sha256"],
            "execution_optimizer_sha256": contract["execution_optimizer_sha256"],
            "outer_fold": 0,
            "development_folds": [1, 2, 3, 4],
            "category_id": "exact3tag",
            "structure_id": structure,
            "outer_fold_used_for_selection": False,
            "validation_payloads_opened": 0,
            "test_payloads_opened": 0,
        }
        for key, expected in required_result.items():
            require(result.get(key) == expected, f"result {structure} {key} mismatch: {result.get(key)!r} != {expected!r}")
        canonical_result_sha(result)
        require(float(result.get("runtime_seconds", 0.0)) > 0.0, f"result runtime invalid for {structure}")

        crossfit_text = blobs[prefix + "inner_crossfit.tsv"].decode()
        crossfit_rows = list(csv.DictReader(io.StringIO(crossfit_text), delimiter="\t"))
        require(len(crossfit_rows) == 4, f"inner crossfit row count mismatch for {structure}")
        heldout_key = "inner_heldout_fold" if "inner_heldout_fold" in crossfit_rows[0] else "heldout_fold"
        heldout = {int(row[heldout_key]) for row in crossfit_rows}
        require(heldout == {1, 2, 3, 4}, f"inner crossfit coverage mismatch for {structure}: {heldout}")
        result_rows.append({
            "structure_id": structure,
            "canonical_payload_sha256": result["canonical_payload_sha256"],
            "pooled_inner_oof_feasible": bool(result["pooled_inner_oof_feasible"]),
            "pooled_support_pass": bool(result["pooled_support_pass"]),
        })

    require(len(result_rows) == 27, "audited structure count mismatch")
    require(receipt.get("pooled_support_pass_results") == sum(row["pooled_support_pass"] for row in result_rows), "receipt support count mismatch")
    require(receipt.get("pooled_feasible_results") == sum(row["pooled_inner_oof_feasible"] for row in result_rows), "receipt feasibility count mismatch")
    return {
        "return_file_sha256": return_hashes,
        "receipt": receipt,
        "bundle_regular_file_count": len(expected_files),
        "structure_results_audited": len(result_rows),
        "pooled_support_pass_results": sum(row["pooled_support_pass"] for row in result_rows),
        "pooled_feasible_results": sum(row["pooled_inner_oof_feasible"] for row in result_rows),
        "structure_results": result_rows,
    }


def main() -> int:
    contract = payload_contract()
    scheduler = scheduler_audit()
    returned = return_audit(contract)
    audit = {
        "schema_version": 1,
        "status": "pass_escalation800_proc0_eos_xrootd_recovery_canary_complete_return_audit",
        "cluster_id": CLUSTER,
        "proc_id": PROC,
        "authoritative_schedd": SCHEDD,
        "condor_submit_called": False,
        "never_resubmit_cluster": True,
        "scientific_execution_head": EXECUTION_HEAD,
        "transport_recovery_only": True,
        "nominal_selection_changed": False,
        "pilot_results_enter_stability_aggregation": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "payload_contract": contract,
        "scheduler_audit": scheduler,
        "return_audit": returned,
    }
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
