#!/usr/bin/env python3
"""Capture the final read-only scheduler evidence for cluster 30020809."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


CLUSTER_ID = 30020809
SCHEDD = "lpcschedd5.fnal.gov"
EXPECTED_JOBS = 8000
PACKAGE_ROOT = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/baselines/"
    "multivariate_cut_selection_stability_escalation800_production_package_v1_20260809"
)
RECOVERY_CMD = Path(
    "/uscms_data/d3/iturkmen/repos/hh-bbww-baselines/scripts/production/"
    "run_hh4b_stability_escalation800_eos_recovery.sh"
)
RUNTIME = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/baselines/"
    "multivariate_cut_bounded_six_job_transfer_pilot_v8_r1_20260807T004537Z/"
    "archives/hh4b_python_runtime_v8_r1.tar.gz"
)
ATTRIBUTES = (
    "ClusterId",
    "ProcId",
    "JobStatus",
    "ExitCode",
    "ExitBySignal",
    "NumJobStarts",
    "Cmd",
    "TransferInput",
    "Iwd",
)


class CaptureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CaptureError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_history_text(payload: bytes) -> None:
    lines = payload.decode("utf-8").splitlines()
    require(len(lines) == EXPECTED_JOBS, f"history rows={len(lines)}, expected 8000")
    seen = set()
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        require(len(fields) == len(ATTRIBUTES), f"history field count changed at line {line_number}")
        cluster, proc, status, exit_code, exit_signal, starts, cmd, transfer_input, iwd = fields
        cluster, proc = int(cluster), int(proc)
        require(cluster == CLUSTER_ID, f"history cluster changed at line {line_number}")
        require(0 <= proc < EXPECTED_JOBS and proc not in seen, f"history proc duplicate/out of range: {proc}")
        require(status == "4" and exit_code == "0", f"non-clean completion at proc {proc}")
        require(exit_signal.lower() in {"false", "0"}, f"signal exit at proc {proc}")
        require(starts == "1", f"unexpected executable start count at proc {proc}")
        require(cmd == str(RECOVERY_CMD), f"history executable changed at proc {proc}")
        expected_transfer = f"{PACKAGE_ROOT / 'submission/run_escalation800_category_fold_job.sh'},{RUNTIME}"
        require(transfer_input == expected_transfer, f"history transfer input changed at proc {proc}")
        require(iwd == str(PACKAGE_ROOT / f"returns/job_{proc:04d}"), f"history Iwd changed at proc {proc}")
        seen.add(proc)
    require(seen == set(range(EXPECTED_JOBS)), "history proc coverage changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(output.parent.is_dir() and not output.exists(), "scheduler evidence output path is invalid/exists")
    output.mkdir()
    constraint = f"ClusterId == {CLUSTER_ID}"
    queue_command = [
        "/usr/bin/bash",
        "/usr/local/bin/condor_q",
        "-name",
        SCHEDD,
        "-constraint",
        constraint,
        "-af",
        *ATTRIBUTES,
    ]
    history_command = [
        "/usr/bin/bash",
        "/usr/local/bin/condor_history",
        "-name",
        SCHEDD,
        "-constraint",
        constraint,
        "-match",
        "10000",
        "-af",
        *ATTRIBUTES,
    ]
    queue = subprocess.run(queue_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    history = subprocess.run(history_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    files = {
        "final_queue_snapshot.txt": queue.stdout,
        "final_queue_snapshot.txt.stderr": queue.stderr,
        "final_history_snapshot.txt": history.stdout,
        "final_history_snapshot.txt.stderr": history.stderr,
    }
    for name, payload in files.items():
        (output / name).write_bytes(payload)
    require(queue.returncode == 0, f"final queue query failed rc={queue.returncode}")
    require(history.returncode == 0, f"final history query failed rc={history.returncode}")
    require(queue.stdout == b"", "cluster still has live queue rows")
    require(queue.stderr == b"", "final queue query wrote stderr")
    require(history.stderr == b"", "final history query wrote stderr")
    validate_history_text(history.stdout)
    summary = {
        "schema_version": 1,
        "status": "pass_final_read_only_escalation800_scheduler_evidence_capture",
        "cluster_id": CLUSTER_ID,
        "authoritative_schedd": SCHEDD,
        "queue_rows": 0,
        "history_rows": EXPECTED_JOBS,
        "clean_history_rows": EXPECTED_JOBS,
        "bad_history_rows": 0,
        "single_start_history_rows": EXPECTED_JOBS,
        "attributes": list(ATTRIBUTES),
        "files": {
            name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
            for name, payload in files.items()
        },
        "scheduler_mutation_performed": False,
        "cluster_resubmitted": False,
        "do_not_resubmit_cluster": True,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    (output / "scheduler_evidence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    print("ESCALATION800_FINAL_SCHEDULER_EVIDENCE=PASS")
    print("QUEUE_ROWS=0")
    print("HISTORY_ROWS=8000")
    print("SCHEDULER_MUTATION_PERFORMED=FALSE")
    print("DO_NOT_RESUBMIT_CLUSTER_30020809=TRUE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
