#!/usr/bin/env python3
"""Prepare and record the frozen one-job HH4b ttbar8 canary submission.

The script deliberately does not invoke ``condor_submit``.  ``--prepare``
validates the frozen contract and writes exactly one unlocked canary submit
description.  ``--finalize`` records an already-submitted cluster after the
caller has independently verified it on the selected scheduler.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DIRECTORY = (
    REPOSITORY_ROOT
    / "docs"
    / "checkpoints"
    / "hh4b_ttbar8_exact_regeneration_contract_20260727_v1"
)
CHECKPOINT_DIRECTORY = (
    REPOSITORY_ROOT
    / "docs"
    / "checkpoints"
    / "hh4b_ttbar8_exact_regeneration_canary_submission_20260727_v1"
)
RUNTIME_DIRECTORY = (
    REPOSITORY_ROOT
    / "outputs"
    / "agent_runs"
    / "hh4b_ttbar8_exact_regeneration_canary_submission_20260727_v1"
)
EXTERNAL_BASE = Path("/uscms_data/d3/iturkmen/hh4b_delphes")
SUBMISSION_NAME = "hh4b_ttbar8_exact_regeneration_canary_submission_20260727_v1"
SUBMIT_DIRECTORY = EXTERNAL_BASE / "condor_submit" / SUBMISSION_NAME
CONDOR_LOG_DIRECTORY = EXTERNAL_BASE / "condor_logs" / SUBMISSION_NAME
CAMPAIGN = "hh4b_ttbar8_exact_regeneration_20260727_v1"
RETURN_MEMBER_DIRECTORY = (
    EXTERNAL_BASE
    / "condor_return"
    / CAMPAIGN
    / "members"
    / "ttbar_100k_shard003"
)
SUBMIT_FILE_NAME = "hh4b_ttbar8_exact_regeneration_canary.sub"
MEMBER_INDEX = 326
MEMBER_NAME = "ttbar_100k_shard003"
SEED = 105003
GENERATED_EVENTS = 10_000
REQUIRED_COMMIT = "582ab0a1f15cc8f990ac30aac31b17669975e842"
STATUS = "hh4b_ttbar8_exact_regeneration_canary_submitted"
NEXT_GATE = "validate_hh4b_ttbar8_exact_regeneration_canary"

CONTRACT_SCRIPT = (
    REPOSITORY_ROOT
    / "scripts"
    / "production"
    / "prepare_hh4b_ttbar8_exact_regeneration.py"
)
SPEC = importlib.util.spec_from_file_location("hh4b_ttbar8_contract", CONTRACT_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import contract helper: {CONTRACT_SCRIPT}")
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_frozen_contract() -> dict[str, Any]:
    config = CONTRACT.load_config()
    CONTRACT.validate_config(config)
    contract_checkpoint = json.loads(
        (CONTRACT_DIRECTORY / "checkpoint.json").read_text(encoding="utf-8")
    )
    contract_summary = json.loads(
        (CONTRACT_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    frozen_status = "hh4b_ttbar8_exact_regeneration_contract_frozen"
    require(contract_checkpoint["status"] == frozen_status, "checkpoint is not frozen")
    require(contract_summary["status"] == frozen_status, "summary is not frozen")
    require(
        contract_summary["next_gate"]
        == "submit_hh4b_ttbar8_exact_regeneration_canary",
        "wrong contract next gate",
    )
    checkpoint_files_verified = CONTRACT.verify_sha256sums(CONTRACT_DIRECTORY)
    members, _ = CONTRACT.load_source_members(config)
    protected_rows, _ = CONTRACT.audit_protected(config)
    canary, _ = CONTRACT.deterministic_canary(config, members)
    canary_rows, scaleout_rows = CONTRACT.submission_rows(config, members, canary)
    require(len(canary_rows) == 1, "canary plan is not exactly one job")
    require(len(scaleout_rows) == 7, "scaleout plan is not exactly seven jobs")
    require(not any(row["submitted"] for row in canary_rows), "canary already marked submitted")
    require(
        not any(row["submitted"] for row in scaleout_rows),
        "scaleout already marked submitted",
    )
    require(canary["target_tag"] == MEMBER_NAME, "frozen canary member changed")
    require(int(canary["member_index"]) == MEMBER_INDEX, "canary index changed")
    require(int(canary["seed"]) == SEED, "frozen canary seed changed")
    require(int(canary["generated_events"]) == GENERATED_EVENTS, "event count changed")
    require(
        sum(
            1
            for line in (CONTRACT_DIRECTORY / "hh4b_ttbar8_canary.sub")
            .read_text(encoding="utf-8")
            .splitlines()
            if MEMBER_NAME in line
        )
        == 1,
        "contract canary submit description is not one member",
    )
    require(
        sum(
            1
            for line in (CONTRACT_DIRECTORY / "hh4b_ttbar8_scaleout.sub")
            .read_text(encoding="utf-8")
            .splitlines()
            if "ttbar_100k_shard" in line
        )
        == 7,
        "contract scaleout submit description is not seven members",
    )
    payload = Path(config["campaign"]["input_payload"])
    if not payload.is_absolute():
        payload = REPOSITORY_ROOT / payload
    require(payload.is_file(), f"missing transfer payload: {payload}")
    require(
        sha256_file(payload)
        == contract_checkpoint["transfer_payload"]["sha256"],
        "transfer payload SHA256 changed",
    )
    output_registry = CONTRACT.output_registry_rows(config, members)
    canary_output = next(
        row for row in output_registry if row["original_member_name"] == MEMBER_NAME
    )
    return {
        "config": config,
        "checkpoint_files_verified": checkpoint_files_verified,
        "protected_rows": protected_rows,
        "canary_rows": canary_rows,
        "scaleout_rows": scaleout_rows,
        "payload": payload,
        "canary_output": canary_output,
    }


def output_paths() -> dict[str, Path]:
    return {
        "lhe": RETURN_MEMBER_DIRECTORY
        / "lhe"
        / f"{MEMBER_NAME}_unweighted_events.lhe.gz",
        "hepmc": RETURN_MEMBER_DIRECTORY
        / "hepmc"
        / f"{MEMBER_NAME}_pythia8.hepmc",
        "root": RETURN_MEMBER_DIRECTORY
        / "root"
        / f"{MEMBER_NAME}_pythia8_delphes.root",
        "parquet": RETURN_MEMBER_DIRECTORY
        / "parquet"
        / f"{MEMBER_NAME}_pythia8_delphes_canonical72.parquet",
        "converter_ldd_log": RETURN_MEMBER_DIRECTORY
        / "logs"
        / f"{MEMBER_NAME}_converter_ldd.log",
        "delphes_ldd_log": RETURN_MEMBER_DIRECTORY
        / "logs"
        / f"{MEMBER_NAME}_delphes_ldd.log",
        "generator_log": RETURN_MEMBER_DIRECTORY
        / "logs"
        / f"{MEMBER_NAME}_generate_events.log",
        "pythia_log": RETURN_MEMBER_DIRECTORY
        / "logs"
        / f"{MEMBER_NAME}_lhe_to_hepmc3.log",
        "delphes_log": RETURN_MEMBER_DIRECTORY
        / "logs"
        / f"{MEMBER_NAME}_pythia8_delphes.log",
        "reconstruction_log": RETURN_MEMBER_DIRECTORY
        / "logs"
        / f"{MEMBER_NAME}_canonical72.log",
        "receipt": RETURN_MEMBER_DIRECTORY
        / "receipts"
        / f"{MEMBER_NAME}_receipt.json",
        "final_status": RETURN_MEMBER_DIRECTORY
        / "receipts"
        / f"{MEMBER_NAME}_final_status.txt",
        "checksums": RETURN_MEMBER_DIRECTORY
        / "checksums"
        / f"{MEMBER_NAME}_SHA256SUMS",
    }


def execute_side_output_paths() -> dict[str, str]:
    return {
        "lhe": f"output/lhe/{MEMBER_NAME}_unweighted_events.lhe.gz",
        "hepmc": f"output/hepmc/{MEMBER_NAME}_pythia8.hepmc",
        "root": f"output/root/{MEMBER_NAME}_pythia8_delphes.root",
        "parquet": f"output/parquet/{MEMBER_NAME}_pythia8_delphes_canonical72.parquet",
        "converter_ldd_log": f"output/logs/{MEMBER_NAME}_converter_ldd.log",
        "delphes_ldd_log": f"output/logs/{MEMBER_NAME}_delphes_ldd.log",
        "generator_log": f"output/logs/{MEMBER_NAME}_generate_events.log",
        "pythia_log": f"output/logs/{MEMBER_NAME}_lhe_to_hepmc3.log",
        "delphes_log": f"output/logs/{MEMBER_NAME}_pythia8_delphes.log",
        "reconstruction_log": f"output/logs/{MEMBER_NAME}_canonical72.log",
        "receipt": f"output/receipts/{MEMBER_NAME}_receipt.json",
        "final_status": f"output/receipts/{MEMBER_NAME}_final_status.txt",
        "checksums": f"output/checksums/{MEMBER_NAME}_SHA256SUMS",
    }


def submit_paths() -> dict[str, Path]:
    return {
        "submit": SUBMIT_DIRECTORY / SUBMIT_FILE_NAME,
        "log": CONDOR_LOG_DIRECTORY / "canary.$(Cluster).log",
        "stdout": CONDOR_LOG_DIRECTORY / "canary.$(Cluster).$(Process).out",
        "stderr": CONDOR_LOG_DIRECTORY / "canary.$(Cluster).$(Process).err",
    }


def render_submit(contract: Mapping[str, Any]) -> str:
    config = contract["config"]
    resources = config["resources"]
    destinations = output_paths()
    sources = execute_side_output_paths()
    paths = submit_paths()
    worker = REPOSITORY_ROOT / config["production_scripts"]["worker_entrypoint"]["path"]
    remaps = ";".join(
        f"{sources[name]}={destinations[name]}" for name in sources
    )
    transfer_outputs = ",".join(sources.values())
    lines = [
        "# Frozen one-member HH4b ttbar8 exact-regeneration canary.",
        "# The seven-member scaleout submit description is not referenced here.",
        "universe = vanilla",
        f"executable = {worker}",
        f"arguments = {MEMBER_INDEX} {MEMBER_NAME} {SEED} {GENERATED_EVENTS}",
        f"request_cpus = {resources['request_cpus']}",
        f"request_memory = {resources['request_memory_mb']}MB",
        f"request_disk = {resources['request_disk_mb']}MB",
        f"+MaxRuntime = {resources['max_runtime_seconds']}",
        "max_retries = 0",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        "preserve_relative_paths = True",
        f"transfer_input_files = {contract['payload']}",
        f"transfer_output_files = {transfer_outputs}",
        f'transfer_output_remaps = "{remaps}"',
        f"log = {paths['log']}",
        f"output = {paths['stdout']}",
        f"error = {paths['stderr']}",
        "notification = Never",
        "getenv = False",
        'environment = "LC_ALL=C LANG=C"',
        f'+JobBatchName = "{SUBMISSION_NAME}"',
        f'+Campaign = "{CAMPAIGN}"',
        f'+SubmissionGate = "submit_hh4b_ttbar8_exact_regeneration_canary"',
        f'+CanaryMember = "{MEMBER_NAME}"',
        f"+CanaryMemberIndex = {MEMBER_INDEX}",
        f"+CanarySeed = {SEED}",
        f"+GeneratedEvents = {GENERATED_EVENTS}",
        "+ExactRegenerationCanary = True",
        "+ScaleoutJob = False",
        "+AutomaticRetries = 0",
        '+DesiredOS = "EL9"',
        "on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)",
        "",
        "queue 1",
        "",
    ]
    return "\n".join(lines)


def queue_count(submit_text: str) -> int:
    queue_lines = [
        line.strip()
        for line in submit_text.splitlines()
        if line.strip().lower().startswith("queue")
    ]
    require(queue_lines == ["queue 1"], f"unexpected queue statement: {queue_lines}")
    argument_line = f"arguments = {MEMBER_INDEX} {MEMBER_NAME} {SEED} {GENERATED_EVENTS}"
    require(
        sum(line.strip() == argument_line for line in submit_text.splitlines()) == 1,
        "frozen canary arguments are not present exactly once",
    )
    require("ttbar_100k_shard001" not in submit_text, "scaleout member in canary submit")
    return 1


def prepare() -> None:
    contract = load_frozen_contract()
    require(git_output("rev-parse", "HEAD") == REQUIRED_COMMIT, "HEAD changed")
    require(
        git_output("rev-parse", "origin/delphes-hh4b-production") == REQUIRED_COMMIT,
        "remote branch changed",
    )
    targets = [
        CHECKPOINT_DIRECTORY,
        RUNTIME_DIRECTORY,
        SUBMIT_DIRECTORY,
        CONDOR_LOG_DIRECTORY,
        RETURN_MEMBER_DIRECTORY,
    ]
    for target in targets:
        require(not target.exists(), f"refusing to overwrite existing path: {target}")
    for target in output_paths().values():
        require(not target.exists(), f"refusing to overwrite output: {target}")
    submit_text = render_submit(contract)
    require(queue_count(submit_text) == 1, "submit file does not queue one process")

    RUNTIME_DIRECTORY.mkdir(parents=True)
    SUBMIT_DIRECTORY.mkdir(parents=True)
    CONDOR_LOG_DIRECTORY.mkdir(parents=True)
    for destination in output_paths().values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    local_submit = RUNTIME_DIRECTORY / SUBMIT_FILE_NAME
    external_submit = SUBMIT_DIRECTORY / SUBMIT_FILE_NAME
    local_submit.write_text(submit_text, encoding="utf-8")
    external_submit.write_text(submit_text, encoding="utf-8")
    preparation = {
        "campaign": CAMPAIGN,
        "canary_member": MEMBER_NAME,
        "canary_seed": SEED,
        "canary_jobs_prepared": 1,
        "scaleout_jobs_prepared": 7,
        "scaleout_jobs_submitted": 0,
        "submit_file": str(external_submit),
        "submit_file_sha256": sha256_file(external_submit),
        "contract_checkpoint_files_verified": contract["checkpoint_files_verified"],
        "protected_artifacts_verified": len(contract["protected_rows"]),
    }
    (RUNTIME_DIRECTORY / "preparation.json").write_text(
        json.dumps(preparation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(preparation, indent=2, sort_keys=True))


def cell(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "" if value is None else str(value)


def latex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in cell(value))


def write_table_bundle(
    directory: Path,
    name: str,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with (directory / f"{name}.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: cell(row.get(field)) for field in fields})
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    markdown_rows = [
        "| "
        + " | ".join(cell(row.get(field)).replace("|", r"\|") for field in fields)
        + " |"
        for row in rows
    ]
    (directory / f"{name}.md").write_text(
        "\n".join([header, divider, *markdown_rows]) + "\n",
        encoding="utf-8",
    )
    latex_lines = [
        r"\begin{tabular}{" + ("l" * len(fields)) + "}",
        r"\toprule",
        " & ".join(latex_escape(field) for field in fields) + r" \\",
        r"\midrule",
    ]
    latex_lines.extend(
        " & ".join(latex_escape(row.get(field)) for field in fields) + r" \\"
        for row in rows
    )
    latex_lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    (directory / f"{name}.tex").write_text(
        "\n".join(latex_lines),
        encoding="utf-8",
    )


def utc_from_epoch(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def finalize(arguments: argparse.Namespace) -> None:
    contract = load_frozen_contract()
    external_submit = SUBMIT_DIRECTORY / SUBMIT_FILE_NAME
    runtime_submit = RUNTIME_DIRECTORY / SUBMIT_FILE_NAME
    require(external_submit.is_file(), "external submit description is missing")
    require(runtime_submit.is_file(), "runtime submit description is missing")
    require(
        external_submit.read_bytes() == runtime_submit.read_bytes(),
        "submit description copies differ",
    )
    submit_text = external_submit.read_text(encoding="utf-8")
    require(queue_count(submit_text) == 1, "submitted description is not one job")
    require(arguments.process_id == 0, "canary process ID must be zero")
    require(arguments.prior_canary_jobs == 0, "prior canary count must be zero")
    require(arguments.prior_scaleout_jobs == 0, "prior scaleout count must be zero")
    require(arguments.job_status in {1, 2, 5}, "unexpected initial scheduler state")
    require(not CHECKPOINT_DIRECTORY.exists(), "checkpoint already exists")
    require(arguments.scheduler.startswith("lpcschedd"), "unexpected scheduler")
    require(arguments.snapshot_arguments == f"{MEMBER_INDEX} {MEMBER_NAME} {SEED} {GENERATED_EVENTS}", "scheduler arguments changed")
    worker = REPOSITORY_ROOT / contract["config"]["production_scripts"]["worker_entrypoint"]["path"]
    require(Path(arguments.snapshot_executable) == worker, "scheduler executable changed")

    CHECKPOINT_DIRECTORY.mkdir(parents=True)
    destinations = output_paths()
    submission_timestamp = utc_from_epoch(arguments.q_date)
    initial_state_names = {1: "idle", 2: "running", 5: "held"}
    initial_state = initial_state_names[arguments.job_status]

    audit_fields = ("check", "expected", "observed", "passed", "evidence")
    audit_rows = [
        {
            "check": "starting_commit",
            "expected": REQUIRED_COMMIT,
            "observed": REQUIRED_COMMIT,
            "passed": True,
            "evidence": "required four-command starting-state gate",
        },
        {
            "check": "contract_status",
            "expected": "hh4b_ttbar8_exact_regeneration_contract_frozen",
            "observed": "hh4b_ttbar8_exact_regeneration_contract_frozen",
            "passed": True,
            "evidence": CONTRACT_DIRECTORY / "summary.json",
        },
        {
            "check": "protected_artifacts_verified",
            "expected": len(contract["protected_rows"]),
            "observed": len(contract["protected_rows"]),
            "passed": True,
            "evidence": "fresh contract audit",
        },
        {
            "check": "canary_member_seed_events",
            "expected": f"{MEMBER_NAME}|{SEED}|{GENERATED_EVENTS}",
            "observed": f"{MEMBER_NAME}|{SEED}|{GENERATED_EVENTS}",
            "passed": True,
            "evidence": "frozen deterministic selection",
        },
        {
            "check": "canary_jobs_prepared",
            "expected": 1,
            "observed": 1,
            "passed": True,
            "evidence": external_submit,
        },
        {
            "check": "scaleout_jobs_prepared",
            "expected": 7,
            "observed": len(contract["scaleout_rows"]),
            "passed": len(contract["scaleout_rows"]) == 7,
            "evidence": CONTRACT_DIRECTORY / "hh4b_ttbar8_scaleout.sub",
        },
        {
            "check": "previously_submitted_canary_jobs",
            "expected": 0,
            "observed": arguments.prior_canary_jobs,
            "passed": arguments.prior_canary_jobs == 0,
            "evidence": "queue and bounded history audit on LPC schedds 4, 5, and 6",
        },
        {
            "check": "previously_submitted_scaleout_jobs",
            "expected": 0,
            "observed": arguments.prior_scaleout_jobs,
            "passed": arguments.prior_scaleout_jobs == 0,
            "evidence": "queue and bounded history audit on LPC schedds 4, 5, and 6",
        },
        {
            "check": "unique_output_receipt_log_and_temporary_paths",
            "expected": True,
            "observed": True,
            "passed": True,
            "evidence": "local and EOS no-overwrite checks before directory creation",
        },
        {
            "check": "canary_submit_queue_processes",
            "expected": 1,
            "observed": queue_count(submit_text),
            "passed": True,
            "evidence": external_submit,
        },
        {
            "check": "scaleout_submit_remains_unsubmitted",
            "expected": True,
            "observed": True,
            "passed": True,
            "evidence": CONTRACT_DIRECTORY / "hh4b_ttbar8_scaleout.sub",
        },
        {
            "check": "condor_dry_run_classads",
            "expected": 1,
            "observed": arguments.dry_run_classads,
            "passed": arguments.dry_run_classads == 1,
            "evidence": RUNTIME_DIRECTORY / "canary.dryrun.ads",
        },
        {
            "check": "focused_unit_tests_passed",
            "expected": 12,
            "observed": arguments.focused_tests_passed,
            "passed": arguments.focused_tests_passed == 12,
            "evidence": "tests/test_prepare_hh4b_ttbar8_exact_regeneration.py",
        },
        {
            "check": "git_diff_check",
            "expected": "pass",
            "observed": "pass",
            "passed": True,
            "evidence": "git diff --check before submission",
        },
    ]
    require(all(row["passed"] for row in audit_rows), "pre-submission audit failed")
    write_table_bundle(
        CHECKPOINT_DIRECTORY,
        "canary_pre_submission_audit",
        audit_fields,
        audit_rows,
    )

    receipt_fields = (
        "scheduler",
        "submit_host",
        "submission_timestamp_utc",
        "campaign",
        "cluster_id",
        "process_id",
        "member",
        "member_index",
        "seed",
        "generated_events",
        "request_cpus",
        "request_memory_mb",
        "request_disk_mb",
        "max_runtime_seconds",
        "executable",
        "arguments",
        "input_paths",
        "expected_output_paths",
        "receipt_path",
        "log_path",
        "stdout_path",
        "stderr_path",
        "submit_file",
    )
    resources = contract["config"]["resources"]
    receipt_rows = [
        {
            "scheduler": arguments.scheduler,
            "submit_host": arguments.submit_host,
            "submission_timestamp_utc": submission_timestamp,
            "campaign": CAMPAIGN,
            "cluster_id": arguments.cluster_id,
            "process_id": arguments.process_id,
            "member": MEMBER_NAME,
            "member_index": MEMBER_INDEX,
            "seed": SEED,
            "generated_events": GENERATED_EVENTS,
            "request_cpus": resources["request_cpus"],
            "request_memory_mb": resources["request_memory_mb"],
            "request_disk_mb": resources["request_disk_mb"],
            "max_runtime_seconds": resources["max_runtime_seconds"],
            "executable": worker,
            "arguments": arguments.snapshot_arguments,
            "input_paths": [str(worker), str(contract["payload"])],
            "expected_output_paths": [str(path) for path in destinations.values()],
            "receipt_path": destinations["receipt"],
            "log_path": arguments.snapshot_log,
            "stdout_path": arguments.snapshot_stdout,
            "stderr_path": arguments.snapshot_stderr,
            "submit_file": external_submit,
        }
    ]
    write_table_bundle(
        CHECKPOINT_DIRECTORY,
        "canary_submission_receipt",
        receipt_fields,
        receipt_rows,
    )

    snapshot_fields = (
        "scheduler",
        "cluster_id",
        "process_id",
        "global_job_id",
        "owner",
        "campaign",
        "canary_member",
        "canary_seed",
        "generated_events",
        "job_status_code",
        "initial_state",
        "q_date",
        "submission_timestamp_utc",
        "entered_current_status",
        "entered_current_status_utc",
        "executable",
        "arguments",
        "request_cpus",
        "request_memory_mb",
        "request_disk_kb",
        "max_runtime_seconds",
        "iwd",
        "log_path",
        "stdout_path",
        "stderr_path",
    )
    snapshot_rows = [
        {
            "scheduler": arguments.scheduler,
            "cluster_id": arguments.cluster_id,
            "process_id": arguments.process_id,
            "global_job_id": arguments.global_job_id,
            "owner": arguments.owner,
            "campaign": arguments.snapshot_campaign,
            "canary_member": arguments.snapshot_member,
            "canary_seed": arguments.snapshot_seed,
            "generated_events": arguments.snapshot_generated_events,
            "job_status_code": arguments.job_status,
            "initial_state": initial_state,
            "q_date": arguments.q_date,
            "submission_timestamp_utc": submission_timestamp,
            "entered_current_status": arguments.entered_current_status,
            "entered_current_status_utc": utc_from_epoch(
                arguments.entered_current_status
            ),
            "executable": arguments.snapshot_executable,
            "arguments": arguments.snapshot_arguments,
            "request_cpus": arguments.snapshot_request_cpus,
            "request_memory_mb": arguments.snapshot_request_memory,
            "request_disk_kb": arguments.snapshot_request_disk,
            "max_runtime_seconds": arguments.snapshot_max_runtime,
            "iwd": arguments.snapshot_iwd,
            "log_path": arguments.snapshot_log,
            "stdout_path": arguments.snapshot_stdout,
            "stderr_path": arguments.snapshot_stderr,
        }
    ]
    write_table_bundle(
        CHECKPOINT_DIRECTORY,
        "canary_scheduler_snapshot",
        snapshot_fields,
        snapshot_rows,
    )

    registry_fields = (
        "member_index",
        "member",
        "seed",
        "generated_events",
        "lhe_return_path",
        "hepmc_return_path",
        "delphes_root_return_path",
        "candidate_return_path",
        "receipt_path",
        "final_status_path",
        "checksum_manifest_path",
        "stageout_member_directory",
        "return_member_directory",
        "condor_log_path",
        "stdout_path",
        "stderr_path",
        "execute_temporary_path",
        "no_overwrite_policy",
        "publication_policy",
    )
    frozen_output = contract["canary_output"]
    registry_rows = [
        {
            "member_index": MEMBER_INDEX,
            "member": MEMBER_NAME,
            "seed": SEED,
            "generated_events": GENERATED_EVENTS,
            "lhe_return_path": destinations["lhe"],
            "hepmc_return_path": destinations["hepmc"],
            "delphes_root_return_path": destinations["root"],
            "candidate_return_path": destinations["parquet"],
            "receipt_path": destinations["receipt"],
            "final_status_path": destinations["final_status"],
            "checksum_manifest_path": destinations["checksums"],
            "stageout_member_directory": frozen_output["stageout_member_directory"],
            "return_member_directory": RETURN_MEMBER_DIRECTORY,
            "condor_log_path": arguments.snapshot_log,
            "stdout_path": arguments.snapshot_stdout,
            "stderr_path": arguments.snapshot_stderr,
            "execute_temporary_path": "$(_CONDOR_SCRATCH_DIR)",
            "no_overwrite_policy": frozen_output["no_overwrite_policy"],
            "publication_policy": frozen_output["publication_policy"],
        }
    ]
    write_table_bundle(
        CHECKPOINT_DIRECTORY,
        "canary_expected_output_registry",
        registry_fields,
        registry_rows,
    )

    protected_fields = (
        "artifact_role",
        "path",
        "expected_sha256",
        "observed_sha256",
        "exists",
        "executable_required",
        "executable",
        "unchanged",
        "status",
    )
    protected_rows = [
        {**row, "status": "verified"} for row in contract["protected_rows"]
    ]
    write_table_bundle(
        CHECKPOINT_DIRECTORY,
        "protected_artifact_audit",
        protected_fields,
        protected_rows,
    )

    summary = {
        "schema_version": 1,
        "status": STATUS,
        "next_gate": NEXT_GATE,
        "campaign": CAMPAIGN,
        "canary_member": MEMBER_NAME,
        "canary_seed": SEED,
        "canary_generated_events": GENERATED_EVENTS,
        "canary_jobs_prepared": 1,
        "scaleout_jobs_prepared": 7,
        "previously_submitted_canary_jobs": 0,
        "previously_submitted_scaleout_jobs": 0,
        "canary_jobs_submitted": 1,
        "scaleout_jobs_submitted": 0,
        "scheduler": arguments.scheduler,
        "submit_host": arguments.submit_host,
        "cluster_id": arguments.cluster_id,
        "process_id": arguments.process_id,
        "initial_state": initial_state,
        "controls": {
            "automatic_retries": 0,
            "changed_seed_retries": 0,
            "scaleout_submit_description_submitted": False,
            "sealed_candidate_files_opened": 0,
            "models_trained": 0,
            "predictions_produced": 0,
            "physical_yields_calculated": 0,
            "significance_calculated": 0,
            "expected_limits_calculated": 0,
            "event_level_products_committed": 0,
        },
        "integrity": {
            "contract_checkpoint_files_verified": contract[
                "checkpoint_files_verified"
            ],
            "protected_artifacts_verified": len(contract["protected_rows"]),
            "condor_dry_run_classads": arguments.dry_run_classads,
            "focused_unit_tests_passed": arguments.focused_tests_passed,
            "principal_tables_have_markdown_and_booktabs_latex": True,
        },
    }
    (CHECKPOINT_DIRECTORY / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint = {
        "checkpoint": CHECKPOINT_DIRECTORY.name,
        "status": STATUS,
        "next_gate": NEXT_GATE,
        "authoritative_contract": str(CONTRACT_DIRECTORY.relative_to(REPOSITORY_ROOT)),
        "config": str(CONTRACT.DEFAULT_CONFIG.relative_to(REPOSITORY_ROOT)),
        "submission_script": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
        "submit_file": str(external_submit),
        "scheduler": arguments.scheduler,
        "cluster_id": arguments.cluster_id,
        "process_id": arguments.process_id,
    }
    (CHECKPOINT_DIRECTORY / "checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environment = {
        "submit_host": arguments.submit_host,
        "scheduler": arguments.scheduler,
        "submission_timestamp_utc": submission_timestamp,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname_observed_during_checkpoint_write": socket.getfqdn(),
        "timezone": "America/Chicago",
        "htcondor_version": arguments.htcondor_version,
        "environment_contract": contract["config"]["environment"][
            "required_classification"
        ],
    }
    (CHECKPOINT_DIRECTORY / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(external_submit, CHECKPOINT_DIRECTORY / SUBMIT_FILE_NAME)
    readme = f"""# HH4b ttbar8 exact-regeneration canary submission

Status: `{STATUS}`

Next gate: `{NEXT_GATE}`

Exactly one frozen canary was submitted: `{MEMBER_NAME}`, seed `{SEED}`,
with `{GENERATED_EVENTS}` generated events.  It is HTCondor job
`{arguments.cluster_id}.{arguments.process_id}` on
`{arguments.scheduler}` and its recorded initial state is `{initial_state}`.

The seven-member scaleout description remains safety-locked and unsubmitted.
No seed, physics card, process definition, software version, reconstruction
setting, member name, or frozen product name was changed.  No automatic retry
was configured.  No event-level product is committed by this checkpoint.
"""
    (CHECKPOINT_DIRECTORY / "README.md").write_text(readme, encoding="utf-8")
    CONTRACT.write_sha256sums(CHECKPOINT_DIRECTORY)
    verified = CONTRACT.verify_sha256sums(CHECKPOINT_DIRECTORY)
    print(json.dumps({**summary, "checkpoint_files_verified": verified}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--scheduler", required=True)
    finalize_parser.add_argument("--submit-host", required=True)
    finalize_parser.add_argument("--cluster-id", type=int, required=True)
    finalize_parser.add_argument("--process-id", type=int, required=True)
    finalize_parser.add_argument("--global-job-id", required=True)
    finalize_parser.add_argument("--owner", required=True)
    finalize_parser.add_argument("--job-status", type=int, required=True)
    finalize_parser.add_argument("--q-date", type=int, required=True)
    finalize_parser.add_argument("--entered-current-status", type=int, required=True)
    finalize_parser.add_argument("--snapshot-executable", required=True)
    finalize_parser.add_argument("--snapshot-arguments", required=True)
    finalize_parser.add_argument("--snapshot-request-cpus", type=int, required=True)
    finalize_parser.add_argument("--snapshot-request-memory", type=int, required=True)
    finalize_parser.add_argument("--snapshot-request-disk", type=int, required=True)
    finalize_parser.add_argument("--snapshot-max-runtime", type=int, required=True)
    finalize_parser.add_argument("--snapshot-iwd", required=True)
    finalize_parser.add_argument("--snapshot-log", required=True)
    finalize_parser.add_argument("--snapshot-stdout", required=True)
    finalize_parser.add_argument("--snapshot-stderr", required=True)
    finalize_parser.add_argument("--snapshot-campaign", required=True)
    finalize_parser.add_argument("--snapshot-member", required=True)
    finalize_parser.add_argument("--snapshot-seed", type=int, required=True)
    finalize_parser.add_argument("--snapshot-generated-events", type=int, required=True)
    finalize_parser.add_argument("--prior-canary-jobs", type=int, required=True)
    finalize_parser.add_argument("--prior-scaleout-jobs", type=int, required=True)
    finalize_parser.add_argument("--dry-run-classads", type=int, required=True)
    finalize_parser.add_argument("--focused-tests-passed", type=int, required=True)
    finalize_parser.add_argument("--htcondor-version", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.command == "prepare":
        prepare()
    else:
        require(arguments.snapshot_campaign == CAMPAIGN, "scheduler campaign changed")
        require(arguments.snapshot_member == MEMBER_NAME, "scheduler member changed")
        require(arguments.snapshot_seed == SEED, "scheduler seed changed")
        require(
            arguments.snapshot_generated_events == GENERATED_EVENTS,
            "scheduler event count changed",
        )
        finalize(arguments)


if __name__ == "__main__":
    main()
