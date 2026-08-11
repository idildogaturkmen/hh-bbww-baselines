#!/usr/bin/env python3
"""Harvest and audit the one-time fixed-cut HH->4b validation returns.

Only additive summaries, fixed-bin distributions, receipts, and durable access
markers are downloaded.  Validation ROOT payloads and the sealed test split are
never accessed by this auditor.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import pandas as pd

from prepare_hh4b_cut_baseline_validation_campaign import sha256, truthy


EXPECTED_JOBS = 116
EXPECTED_DISTRIBUTION_ROWS = 1360
SCHEDD_RE = re.compile(r"^lpcschedd[0-9]+\.fnal\.gov$")
NOMINAL_THRESHOLDS = {
    "exact3tag": {
        "r_hh_125_125": 36.40814019639858,
        "ht_candidate_jets": 176.5458068847656,
    },
    "ge4tag": {
        "r_hh_125_125": 33.92808917804956,
        "mhh": 164.73708096689654,
        "abs_h_delta_eta": 6.904302164473993,
    },
}


class ReturnAuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReturnAuditError(message)


def run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def parse_json_output(result: subprocess.CompletedProcess[bytes], label: str) -> list[dict[str, Any]]:
    require(
        result.returncode == 0,
        f"{label} failed rc={result.returncode}: stdout={result.stdout!r}; stderr={result.stderr!r}",
    )
    try:
        payload = json.loads(result.stdout.decode("utf-8") or "[]")
    except Exception as error:
        raise ReturnAuditError(f"{label} returned invalid JSON") from error
    require(isinstance(payload, list), f"{label} JSON is not a list")
    return payload


def query_scheduler(schedd: str, cluster: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    require(bool(SCHEDD_RE.fullmatch(schedd)), f"invalid schedd: {schedd}")
    constraint = f"ClusterId == {cluster}"
    queue_result = run(
        [
            "/usr/bin/bash",
            "/usr/local/bin/condor_q",
            "-name",
            schedd,
            "-constraint",
            constraint,
            "-long",
            "-json",
        ]
    )
    queue = parse_json_output(queue_result, "validation queue query")
    require(not queue, f"validation cluster still has {len(queue)} live jobs")
    history_result = run(
        [
            "/usr/bin/bash",
            "/usr/local/bin/condor_history",
            "-name",
            schedd,
            "-constraint",
            constraint,
            "-match",
            "1000",
            "-long",
            "-json",
        ]
    )
    history = parse_json_output(history_result, "validation history query")
    evidence = {
        "queue_stdout_sha256": hashlib.sha256(queue_result.stdout).hexdigest(),
        "queue_stderr_sha256": hashlib.sha256(queue_result.stderr).hexdigest(),
        "history_stdout_sha256": hashlib.sha256(history_result.stdout).hexdigest(),
        "history_stderr_sha256": hashlib.sha256(history_result.stderr).hexdigest(),
    }
    return queue, history, evidence


def validate_history(history: list[dict[str, Any]], cluster: int) -> list[dict[str, Any]]:
    require(len(history) == EXPECTED_JOBS, f"validation history rows={len(history)}, expected 116")
    identities = {(int(row.get("ClusterId", -1)), int(row.get("ProcId", -1))) for row in history}
    require(identities == {(cluster, proc) for proc in range(EXPECTED_JOBS)}, "validation history identity closure failed")
    for row in history:
        proc = int(row["ProcId"])
        require(int(row.get("JobStatus", -1)) == 4, f"history job is not completed: {proc}")
        require(int(row.get("ExitCode", -1)) == 0, f"history exit code is not zero: {proc}")
        require(row.get("ExitBySignal") is False, f"history exit-by-signal changed: {proc}")
        require(int(row.get("NumJobStarts", 0)) >= 1, f"history job never started: {proc}")
    return sorted(history, key=lambda row: int(row["ProcId"]))


def xrdfs_ls(remote_directory: str) -> list[str]:
    result = run(["xrdfs", "root://cmseos.fnal.gov", "ls", remote_directory])
    require(
        result.returncode == 0,
        f"remote validation inventory failed for {remote_directory}: {result.stderr!r}",
    )
    return sorted(line.strip() for line in result.stdout.decode("utf-8").splitlines() if line.strip())


def expected_remote_files(
    remote_root: str,
    physical: pd.DataFrame,
) -> dict[str, list[str]]:
    output = {name: [] for name in ("markers", "summaries", "distributions", "receipts")}
    for row in physical.sort_values("production_row_index").itertuples(index=False):
        index = int(row.production_row_index)
        output["markers"].append(
            f"{remote_root}/markers/source_{index:04d}_VALIDATION_OPEN_DO_NOT_RERUN.json"
        )
        output["summaries"].append(
            f"{remote_root}/summaries/source_{index:04d}_summary.json"
        )
        output["distributions"].append(
            f"{remote_root}/distributions/source_{index:04d}_distributions.tsv"
        )
        output["receipts"].append(
            f"{remote_root}/receipts/source_{index:04d}_job_receipt.json"
        )
    return output


def validate_remote_inventory(remote_root: str, expected: dict[str, list[str]]) -> list[dict[str, Any]]:
    inventory = []
    for role, expected_paths in expected.items():
        observed = xrdfs_ls(f"{remote_root}/{role}")
        require(observed == expected_paths, f"remote validation {role} file set changed")
        for path in observed:
            inventory.append({"role": role, "remote_path": path})
    temporary = xrdfs_ls(f"{remote_root}/.upload_tmp")
    require(not temporary, f"validation temporary uploads remain: {temporary[:5]}")
    require(len(inventory) == 4 * EXPECTED_JOBS, "remote validation inventory count changed")
    return inventory


def download_one(remote_path: str, destination: Path) -> dict[str, Any]:
    require(not destination.exists(), f"download destination already exists: {destination}")
    result = run(
        [
            "xrdcp",
            "--nopbar",
            "--cksum",
            "adler32",
            f"root://cmseos.fnal.gov/{remote_path}",
            str(destination),
        ]
    )
    require(
        result.returncode == 0,
        f"validation result download failed for {remote_path}: stdout={result.stdout!r}; stderr={result.stderr!r}",
    )
    require(destination.is_file() and not destination.is_symlink(), f"download missing: {destination}")
    return {
        "remote_path": remote_path,
        "local_path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "xrdcp_stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "xrdcp_stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
    }


def download_inventory(
    inventory: list[dict[str, Any]],
    source_outputs: Path,
    receipts: Path,
    workers: int,
) -> list[dict[str, Any]]:
    require(1 <= workers <= 32, "download worker count must be 1..32")
    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for item in inventory:
            destination_root = receipts if item["role"] == "receipts" else source_outputs
            destination = destination_root / Path(item["remote_path"]).name
            tasks.append(pool.submit(download_one, item["remote_path"], destination))
        results = []
        for future in as_completed(tasks):
            results.append(future.result())
    require(len(results) == 4 * EXPECTED_JOBS, "downloaded validation file count changed")
    return sorted(results, key=lambda row: row["remote_path"])


def validate_source_returns(
    physical: pd.DataFrame,
    queue_items: list[tuple[int, str]],
    source_outputs: Path,
    receipts: Path,
    authorization: dict[str, Any],
    authorization_sha: str,
    execution_head: str,
    cluster: int,
) -> list[dict[str, Any]]:
    require(len(queue_items) == EXPECTED_JOBS, "validation queue identity count changed")
    proc_by_identity = {identity: proc for proc, identity in enumerate(queue_items)}
    audit_rows = []
    for row in physical.sort_values("production_row_index").itertuples(index=False):
        index = int(row.production_row_index)
        identity = (index, str(row.source_uid))
        require(identity in proc_by_identity, f"source missing from validation queue: {identity}")
        proc = proc_by_identity[identity]
        marker_path = source_outputs / f"source_{index:04d}_VALIDATION_OPEN_DO_NOT_RERUN.json"
        summary_path = source_outputs / f"source_{index:04d}_summary.json"
        distribution_path = source_outputs / f"source_{index:04d}_distributions.tsv"
        receipt_path = receipts / f"source_{index:04d}_job_receipt.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        require(marker.get("status") == "validation_source_open_attempt_durable_do_not_rerun", f"marker status failed: {index}")
        require(marker.get("repository_head") == execution_head, f"marker head changed: {index}")
        require(marker.get("authorization_repository_head") == authorization.get("repository_head"), f"marker authorization head changed: {index}")
        require(marker.get("authorization_sha256") == authorization_sha, f"marker authorization changed: {index}")
        require(marker.get("production_row_index") == index and marker.get("source_uid") == row.source_uid, f"marker identity changed: {index}")
        require(marker.get("condor_cluster_id") == cluster and marker.get("condor_proc_id") == proc, f"marker scheduler identity changed: {index}")
        require(marker.get("source_payload_access_may_begin") is True, f"marker access flag changed: {index}")
        require(marker.get("rerun_forbidden_even_if_downstream_bookkeeping_fails") is True, f"marker rerun flag changed: {index}")
        require(marker.get("test_payloads_opened") == 0, f"marker test count changed: {index}")

        require(summary.get("status") == "pass_one_time_fixed_nominal_validation_source_evaluation", f"source status failed: {index}")
        require(summary.get("repository_head") == execution_head, f"source head changed: {index}")
        require(summary.get("authorization_sha256") == authorization_sha, f"source authorization changed: {index}")
        require(summary.get("production_row_index") == index and summary.get("source_uid") == row.source_uid, f"source identity changed: {index}")
        require(summary.get("nominal_thresholds") == NOMINAL_THRESHOLDS, f"nominal threshold drift: {index}")
        require(summary.get("cut_scan_performed") is False, f"validation scan detected: {index}")
        require(summary.get("threshold_adjustment_performed") is False, f"threshold adjustment detected: {index}")
        require(summary.get("family_adjustment_performed") is False, f"family adjustment detected: {index}")
        require(summary.get("source_payload_opened_once") is True, f"source not opened once: {index}")
        require(summary.get("source_payload_rerun_performed") is False, f"source rerun detected: {index}")
        require(summary.get("validation_payloads_opened") == 1, f"source validation count changed: {index}")
        require(summary.get("test_payloads_opened") == 0, f"source test count changed: {index}")
        require(summary.get("source_open_attempt_marker_sha256") == sha256(marker_path), f"marker SHA mismatch: {index}")
        require(summary.get("distribution_sha256") == sha256(distribution_path), f"distribution SHA mismatch: {index}")
        distributions = pd.read_csv(distribution_path, sep="\t", keep_default_na=False)
        require(len(distributions) == EXPECTED_DISTRIBUTION_ROWS, f"distribution row count changed: {index}")
        require(set(distributions["source_uid"].astype(str)) == {row.source_uid}, f"distribution UID changed: {index}")
        require(distributions["validation_payloads_opened"].eq(1).all(), f"distribution validation count changed: {index}")
        require(distributions["test_payloads_opened"].eq(0).all(), f"distribution test count changed: {index}")

        require(receipt.get("status") == "pass_one_time_fixed_nominal_validation_source_job", f"job receipt failed: {index}")
        require(receipt.get("repository_head") == execution_head, f"receipt head changed: {index}")
        require(receipt.get("authorization_sha256") == authorization_sha, f"receipt authorization changed: {index}")
        require(receipt.get("production_row_index") == index and receipt.get("source_uid") == row.source_uid, f"receipt identity changed: {index}")
        require(receipt.get("condor_cluster_id") == cluster and receipt.get("condor_proc_id") == proc, f"receipt scheduler identity changed: {index}")
        require(receipt.get("attempt_marker_sha256") == sha256(marker_path), f"receipt marker SHA changed: {index}")
        require(receipt.get("source_summary_sha256") == sha256(summary_path), f"receipt summary SHA changed: {index}")
        require(receipt.get("source_distributions_sha256") == sha256(distribution_path), f"receipt distributions SHA changed: {index}")
        require(receipt.get("validation_payloads_opened") == 1 and receipt.get("test_payloads_opened") == 0, f"receipt access counts changed: {index}")
        require(receipt.get("rerun_authorized") is False, f"receipt authorized rerun: {index}")
        audit_rows.append(
            {
                "condor_proc_id": proc,
                "production_row_index": index,
                "source_uid": row.source_uid,
                "sample_class": row.sample_class,
                "generated_events": int(row.generated_events),
                "attempt_marker_sha256": sha256(marker_path),
                "source_summary_sha256": sha256(summary_path),
                "source_distributions_sha256": sha256(distribution_path),
                "job_receipt_sha256": sha256(receipt_path),
                "source_payload_opened_once": True,
                "source_payload_rerun_performed": False,
                "validation_payloads_opened": 1,
                "test_payloads_opened": 0,
            }
        )
    require(len(audit_rows) == EXPECTED_JOBS, "validation source audit row count changed")
    return audit_rows


def validate_logs(log_root: Path, cluster: int, queue_items: list[tuple[int, str]]) -> list[dict[str, Any]]:
    require(log_root.is_dir() and not log_root.is_symlink(), "validation log root is invalid")
    expected_out = {
        log_root / f"validation.{cluster}.{proc}.row{row_index}.out"
        for proc, (row_index, _) in enumerate(queue_items)
    }
    expected_err = {
        log_root / f"validation.{cluster}.{proc}.row{row_index}.err"
        for proc, (row_index, _) in enumerate(queue_items)
    }
    observed_out = set(log_root.glob(f"validation.{cluster}.*.row*.out"))
    observed_err = set(log_root.glob(f"validation.{cluster}.*.row*.err"))
    require(observed_out == expected_out and observed_err == expected_err, "validation stdout/stderr file set changed")
    event_log = log_root / f"validation.{cluster}.log"
    require(event_log.is_file(), "validation event log is missing")
    rows = []
    for path in sorted(expected_out | expected_err | {event_log}):
        text = path.read_text(encoding="utf-8", errors="replace")
        require("VALIDATION_RUNNER_STATUS=FAIL" not in text, f"validation failure signature in {path}")
        if path.suffix == ".out":
            require("VALIDATION_RUNNER_STATUS=PASS" in text, f"validation pass signature missing: {path}")
        rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--source-access-manifest", type=Path, required=True)
    parser.add_argument("--campaign-package", type=Path, required=True)
    parser.add_argument("--submission-checkpoint", type=Path, required=True)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    submission_checkpoint = args.submission_checkpoint.resolve()
    require(submission_checkpoint.is_dir() and not submission_checkpoint.is_symlink(), "validation submission checkpoint is invalid")
    checkpoint_sums = submission_checkpoint / "SHA256SUMS"
    require(checkpoint_sums.is_file() and not checkpoint_sums.is_symlink(), "validation submission checkpoint lacks SHA256SUMS")
    checkpoint_check = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"], cwd=submission_checkpoint,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    require(checkpoint_check.returncode == 0, f"validation submission checkpoint checksum failure: {checkpoint_check.stdout}{checkpoint_check.stderr}")
    submission_freeze_path = submission_checkpoint / "validation_submission_freeze.json"
    args.submission_receipt = (
        submission_checkpoint / "evidence/submission/validation_submission_receipt.json"
    )

    for path in (args.authorization, args.source_access_manifest, args.submission_receipt):
        require(path.is_file() and not path.is_symlink(), f"missing/nonregular audit input: {path}")
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    require(
        authorization.get("authorized_sha256", {}).get("validation_return_auditor")
        == sha256(Path(__file__).resolve()),
        "validation return auditor is not authorized code",
    )
    submission_freeze = json.loads(submission_freeze_path.read_text(encoding="utf-8"))
    require(
        submission_freeze.get("status")
        == "pass_exactly_once_validation_submission_frozen_before_monitoring",
        "validation submission checkpoint freeze status changed",
    )
    require(
        submission_freeze.get("implementation_sha256")
        == authorization.get("authorized_sha256", {}).get("validation_submission_freezer"),
        "validation submission checkpoint was not produced by the master-authorized freezer",
    )
    require(submission_freeze.get("do_not_resubmit_cluster") is True, "validation submission checkpoint permits resubmission")
    require(submission_freeze.get("validation_payloads_opened_before_submission") == 0, "validation opened before submission")
    require(submission_freeze.get("test_payloads_opened") == 0, "test opened in submission checkpoint")
    authorization_sha = sha256(args.authorization)
    campaign_summary_path = args.campaign_package / "campaign_summary.json"
    queue_path = args.campaign_package / "validation_queue.items"
    require(campaign_summary_path.is_file() and queue_path.is_file(), "campaign package is incomplete")
    campaign = json.loads(campaign_summary_path.read_text(encoding="utf-8"))
    execution_head = campaign["repository_head"]
    submission = json.loads(args.submission_receipt.read_text(encoding="utf-8"))
    require(
        submission.get("status")
        == "pass_one_time_cut_baseline_validation_condor_submission_accepted_and_parsed",
        "validation submission receipt status changed",
    )
    require(submission.get("expected_job_count") == EXPECTED_JOBS, "submission job count changed")
    require(submission.get("do_not_resubmit_cluster") is True, "submission do-not-resubmit gate changed")
    require(submission.get("test_payloads_opened") == 0, "test was opened")
    cluster = int(submission["cluster_id"])
    schedd = str(submission["schedd"])
    require(cluster == int(submission_freeze["cluster_id"]), "submission checkpoint cluster changed")
    require(schedd == str(submission_freeze["authoritative_schedd"]), "submission checkpoint schedd changed")
    queue_items = []
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        row_index, source_uid = line.split()
        queue_items.append((int(row_index), source_uid))
    require(len(queue_items) == len(set(queue_items)) == EXPECTED_JOBS, "campaign queue closure changed")

    _, history, scheduler_evidence = query_scheduler(schedd, cluster)
    history = validate_history(history, cluster)
    access = pd.read_csv(args.source_access_manifest, sep="\t", keep_default_na=False)
    physical = access.loc[access["physical_evaluation_eligible"].map(truthy)].copy()
    auxiliary = access.loc[~access["physical_evaluation_eligible"].map(truthy)].copy()
    require(len(access) == 121 and len(physical) == 116 and len(auxiliary) == 5, "validation source partition changed")
    require(not access["test_content_opened"].map(truthy).any(), "test metadata reports opened content")
    remote_root = campaign["remote_output_root"]
    expected = expected_remote_files(remote_root, physical)
    remote_inventory = validate_remote_inventory(remote_root, expected)

    output = args.output_dir.resolve()
    require(output.parent.is_dir() and not output.exists(), "return audit output path is invalid or exists")
    output.mkdir()
    source_outputs = output / "source_outputs"
    receipts = output / "receipts"
    source_outputs.mkdir()
    receipts.mkdir()
    downloads = download_inventory(remote_inventory, source_outputs, receipts, args.download_workers)
    source_rows = validate_source_returns(
        physical,
        queue_items,
        source_outputs,
        receipts,
        authorization,
        authorization_sha,
        execution_head,
        cluster,
    )
    logs = validate_logs(Path(campaign["local_log_root"]), cluster, queue_items)

    pd.DataFrame(history).to_json(output / "condor_history.json", orient="records", indent=2)
    pd.DataFrame(remote_inventory).to_csv(
        output / "remote_output_inventory.tsv", sep="\t", index=False, lineterminator="\n"
    )
    pd.DataFrame(downloads).to_csv(
        output / "download_inventory.tsv", sep="\t", index=False, lineterminator="\n"
    )
    pd.DataFrame(source_rows).to_csv(
        output / "validation_source_return_audit.tsv", sep="\t", index=False, lineterminator="\n"
    )
    pd.DataFrame(logs).to_csv(
        output / "scheduler_log_inventory.tsv", sep="\t", index=False, lineterminator="\n"
    )
    report = {
        "schema_version": 1,
        "status": "pass_complete_one_time_cut_baseline_validation_return_audit",
        "repository_head": execution_head,
        "authorization_sha256": authorization_sha,
        "cluster_id": cluster,
        "authoritative_schedd": schedd,
        "clean_history_jobs": len(history),
        "source_attempt_markers": len(source_rows),
        "source_summaries": len(source_rows),
        "source_distributions": len(source_rows),
        "job_receipts": len(source_rows),
        "distribution_rows": len(source_rows) * EXPECTED_DISTRIBUTION_ROWS,
        "physical_validation_sources_opened_once": len(source_rows),
        "physical_validation_generated_events": int(
            pd.to_numeric(physical["generated_events"], errors="raise").sum()
        ),
        "auxiliary_qcd_validation_sources_opened": 0,
        "auxiliary_qcd_validation_sources_remain_sealed": len(auxiliary),
        "validation_evaluation_cycles": 1,
        "validation_payloads_opened": 1,
        "source_payload_reruns": 0,
        "cut_scan_performed": False,
        "threshold_adjustment_performed": False,
        "family_adjustment_performed": False,
        "nominal_thresholds": NOMINAL_THRESHOLDS,
        "test_payloads_opened": 0,
        "scheduler_query_evidence": scheduler_evidence,
        "do_not_resubmit_cluster": True,
        "next": "run_authorized_fixed_cut_validation_aggregation_once_from_source_outputs",
    }
    (output / "validation_return_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    top_files = sorted(path for path in output.iterdir() if path.is_file())
    nested_files = sorted(path for path in source_outputs.iterdir() if path.is_file())
    nested_files += sorted(path for path in receipts.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(output).as_posix()}\n"
            for path in [*top_files, *nested_files]
        ),
        encoding="utf-8",
    )
    print("CUT_BASELINE_VALIDATION_COMPLETE_RETURN_AUDIT=PASS")
    print("CLEAN_HISTORY_JOBS=116")
    print("PHYSICAL_VALIDATION_SOURCES_OPENED_ONCE=116")
    print("SOURCE_PAYLOAD_RERUNS=0")
    print("VALIDATION_EVALUATION_CYCLES=1")
    print("VALIDATION_PAYLOADS_OPENED=1")
    print("TEST_PAYLOADS_OPENED=0")
    print("DO_NOT_RESUBMIT=TRUE")


if __name__ == "__main__":
    main()
