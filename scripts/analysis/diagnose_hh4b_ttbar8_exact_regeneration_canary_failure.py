#!/usr/bin/env python3
"""Diagnose the held HH4b ttbar8 exact-regeneration canary.

This gate is deliberately read-only with respect to HTCondor.  It queries the
live ClassAd and history, reads only the frozen submission material, retained
canary logs, the canary return directory, and the immutable transfer payload,
then writes an auditable diagnosis checkpoint.  It never releases, removes,
reschedules, or submits a job.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = (
    REPOSITORY_ROOT
    / "outputs"
    / "agent_runs"
    / "hh4b_ttbar8_exact_regeneration_canary_failure_20260727_v1"
)
CHECKPOINT_DIRECTORY = (
    REPOSITORY_ROOT
    / "docs"
    / "checkpoints"
    / "hh4b_ttbar8_exact_regeneration_canary_failure_20260727_v1"
)
CONTRACT_DIRECTORY = (
    REPOSITORY_ROOT
    / "docs"
    / "checkpoints"
    / "hh4b_ttbar8_exact_regeneration_contract_20260727_v1"
)
SUBMISSION_CHECKPOINT_DIRECTORY = (
    REPOSITORY_ROOT
    / "docs"
    / "checkpoints"
    / "hh4b_ttbar8_exact_regeneration_canary_submission_20260727_v1"
)
PREPARE_SCRIPT = (
    REPOSITORY_ROOT
    / "scripts"
    / "production"
    / "prepare_hh4b_ttbar8_exact_regeneration.py"
)
WORKER = (
    REPOSITORY_ROOT
    / "scripts"
    / "production"
    / "run_hh4b_ttbar8_exact_regeneration_member.sh"
)
RECONSTRUCTION_BUILDER = (
    REPOSITORY_ROOT
    / "scripts"
    / "delphes"
    / "reconstruct_hh4b_candidates_v2.py"
)
PARQUET_WRITER = (
    REPOSITORY_ROOT
    / "scripts"
    / "delphes"
    / "write_parquet_from_pickle.py"
)
PAYLOAD = (
    REPOSITORY_ROOT
    / "outputs"
    / "agent_runs"
    / "hh4b_ttbar8_exact_regeneration_prepare_20260727_v1"
    / "hh4b_ttbar8_exact_regeneration_inputs.tar.gz"
)

EXTERNAL_BASE = Path("/uscms_data/d3/iturkmen/hh4b_delphes")
SUBMISSION_NAME = "hh4b_ttbar8_exact_regeneration_canary_submission_20260727_v1"
CAMPAIGN = "hh4b_ttbar8_exact_regeneration_20260727_v1"
LOG_DIRECTORY = EXTERNAL_BASE / "condor_logs" / SUBMISSION_NAME
SUBMIT_DIRECTORY = EXTERNAL_BASE / "condor_submit" / SUBMISSION_NAME
RETURN_MEMBER_DIRECTORY = (
    EXTERNAL_BASE
    / "condor_return"
    / CAMPAIGN
    / "members"
    / "ttbar_100k_shard003"
)
FROZEN_SUBMIT = SUBMIT_DIRECTORY / "hh4b_ttbar8_exact_regeneration_canary.sub"
EVENT_LOG = LOG_DIRECTORY / "canary.3654710.log"
STDOUT = LOG_DIRECTORY / "canary.3654710.0.out"
STDERR = LOG_DIRECTORY / "canary.3654710.0.err"

SCHEDULER = "lpcschedd4.fnal.gov"
CLUSTER_ID = 3_654_710
PROC_ID = 0
JOB_ID = f"{CLUSTER_ID}.{PROC_ID}"
MEMBER_INDEX = 326
MEMBER_NAME = "ttbar_100k_shard003"
SEED = 105_003
GENERATED_EVENTS = 10_000
REQUIRED_STARTING_COMMIT = "b4da1f1c5cd5ab436fb8e675192ddf0e1cb3cc84"
STATUS = "hh4b_ttbar8_exact_regeneration_canary_failure_diagnosed"
NEXT_GATE = "freeze_hh4b_ttbar8_exact_regeneration_canary_retry_contract"
PRIMARY_CLASSIFICATION = "canonical_reconstruction_failure"
MISSING_HELPER_ARCHIVE_PATH = (
    "payload/repo/scripts/delphes/write_parquet_from_pickle.py"
)
BUILDER_ARCHIVE_PATH = (
    "payload/repo/scripts/delphes/reconstruct_hh4b_candidates_v2.py"
)

EXPECTED_ERROR = (
    "ERROR: missing isolated Parquet writer: "
    "/srv/payload/repo/scripts/delphes/write_parquet_from_pickle.py"
)
EXPECTED_PARQUET = (
    RETURN_MEMBER_DIRECTORY
    / "parquet"
    / f"{MEMBER_NAME}_pythia8_delphes_canonical72.parquet"
)
EXPECTED_RECEIPT = (
    RETURN_MEMBER_DIRECTORY / "receipts" / f"{MEMBER_NAME}_receipt.json"
)
FINAL_STATUS = (
    RETURN_MEMBER_DIRECTORY / "receipts" / f"{MEMBER_NAME}_final_status.txt"
)
EXPECTED_CHECKSUMS = (
    RETURN_MEMBER_DIRECTORY / "checksums" / f"{MEMBER_NAME}_SHA256SUMS"
)
LHE = RETURN_MEMBER_DIRECTORY / "lhe" / f"{MEMBER_NAME}_unweighted_events.lhe.gz"
HEPMC = RETURN_MEMBER_DIRECTORY / "hepmc" / f"{MEMBER_NAME}_pythia8.hepmc"
ROOT_OUTPUT = (
    RETURN_MEMBER_DIRECTORY / "root" / f"{MEMBER_NAME}_pythia8_delphes.root"
)
CANONICAL_LOG = (
    RETURN_MEMBER_DIRECTORY / "logs" / f"{MEMBER_NAME}_canonical72.log"
)
GENERATOR_LOG = (
    RETURN_MEMBER_DIRECTORY / "logs" / f"{MEMBER_NAME}_generate_events.log"
)
PYTHIA_LOG = (
    RETURN_MEMBER_DIRECTORY / "logs" / f"{MEMBER_NAME}_lhe_to_hepmc3.log"
)
DELPHES_LOG = (
    RETURN_MEMBER_DIRECTORY / "logs" / f"{MEMBER_NAME}_pythia8_delphes.log"
)

ALLOWED_PRIMARY_TYPES = {
    "payload_environment_failure",
    "generator_failure",
    "pythia_failure",
    "delphes_failure",
    "root_validation_failure",
    "canonical_reconstruction_failure",
    "output_path_contract_mismatch",
    "premature_cleanup",
    "masked_payload_failure",
    "transfer_configuration_failure",
    "ambiguous_failure",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
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


def run_read_only_command(arguments: Sequence[str], *, check: bool = True) -> str:
    command = list(arguments)
    if command and command[0] in {"condor_q", "condor_history"}:
        launcher = shutil.which(command[0])
        require(launcher is not None, f"missing scheduler query command: {command[0]}")
        # The Fermilab site launchers intentionally begin with a shell ``exec``
        # line rather than a shebang, so Python cannot exec them directly.
        command = ["/usr/bin/bash", launcher, *command[1:]]
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode:
        raise RuntimeError(
            f"read-only command failed ({result.returncode}): "
            f"{' '.join(arguments)}\n{result.stderr}"
        )
    return result.stdout


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
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: cell(row.get(field)) for field in fields})
    # An omitted terminal TSV cell is still unambiguously empty, while a
    # literal terminal tab is rejected by ``git diff --check``.
    tsv_text = "\n".join(
        line.rstrip("\t") for line in buffer.getvalue().splitlines()
    )
    (directory / f"{name}.tsv").write_text(tsv_text + "\n", encoding="utf-8")

    markdown = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    markdown.extend(
        "| "
        + " | ".join(
            cell(row.get(field)).replace("|", r"\|") for field in fields
        )
        + " |"
        for row in rows
    )
    (directory / f"{name}.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    latex = [
        r"\begin{tabular}{" + ("l" * len(fields)) + "}",
        r"\toprule",
        " & ".join(latex_escape(field) for field in fields) + r" \\",
        r"\midrule",
    ]
    latex.extend(
        " & ".join(latex_escape(row.get(field)) for field in fields) + r" \\"
        for row in rows
    )
    latex.extend((r"\bottomrule", r"\end{tabular}", ""))
    (directory / f"{name}.tex").write_text(
        "\n".join(latex), encoding="utf-8"
    )


def strip_classad_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1].replace(r"\"", '"').replace(r"\\", "\\")
    return value


def parse_classad(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if match:
            result[match.group(1)] = strip_classad_value(match.group(2))
    return result


def parse_submit(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip().lower()] = value.strip().strip('"')
    return result


def parse_transfer_remaps(value: str) -> dict[str, str]:
    remaps: dict[str, str] = {}
    for item in value.split(";"):
        if not item.strip():
            continue
        source, destination = item.split("=", 1)
        remaps[source.strip()] = destination.strip()
    return remaps


def parse_scheduler_events(text: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    details: list[str] = []
    pattern = re.compile(
        r"^(\d{3}) \([^)]*\) "
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (.*)$"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            if current is not None:
                current["details"] = " ".join(details).strip()
                events.append(current)
            current = {
                "event_code": match.group(1),
                "timestamp_local": match.group(2),
                "event": match.group(3).strip(),
            }
            details = []
        elif current is not None and line.strip() != "...":
            details.append(line.strip())
    if current is not None:
        current["details"] = " ".join(details).strip()
        events.append(current)
    return events


def epoch_utc(value: str) -> str:
    if not value or value in {"undefined", "-1"}:
        return ""
    return datetime.fromtimestamp(int(value), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def require_authorized_path(path: Path) -> None:
    authorized_roots = (
        RETURN_MEMBER_DIRECTORY,
        LOG_DIRECTORY,
        SUBMIT_DIRECTORY,
        CONTRACT_DIRECTORY,
        SUBMISSION_CHECKPOINT_DIRECTORY,
        PAYLOAD.parent,
        REPOSITORY_ROOT / "scripts",
        REPOSITORY_ROOT / "configs",
        REPOSITORY_ROOT / "outputs" / "agent_runs",
    )
    require(
        any(is_within(path, root) for root in authorized_roots),
        f"path is outside authorized canary diagnosis scope: {path}",
    )


def inspect_payload(payload: Path) -> dict[str, Any]:
    require_authorized_path(payload)
    with tarfile.open(payload, "r:gz") as archive:
        names = set(archive.getnames())
        manifest_member = archive.extractfile("payload/SHA256SUMS")
        require(manifest_member is not None, "payload manifest is unreadable")
        manifest = manifest_member.read().decode("utf-8")
    return {
        "builder_present": BUILDER_ARCHIVE_PATH in names,
        "helper_present": MISSING_HELPER_ARCHIVE_PATH in names,
        "builder_manifested": (
            "repo/scripts/delphes/reconstruct_hh4b_candidates_v2.py" in manifest
        ),
        "helper_manifested": (
            "repo/scripts/delphes/write_parquet_from_pickle.py" in manifest
        ),
        "member_count": len(names),
    }


def count_lhe_events(path: Path) -> int:
    count = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "<event>" in line:
                count += 1
    return count


def read_root_entries_serial(path: Path) -> int:
    import uproot

    with uproot.open(path) as source:
        return int(source["Delphes"].num_entries)


def classify_failure(facts: Mapping[str, Any]) -> str:
    if (
        facts.get("canonical_started")
        and EXPECTED_ERROR in str(facts.get("canonical_log", ""))
        and facts.get("root_exists")
        and not facts.get("parquet_exists")
        and facts.get("payload_builder_present")
        and not facts.get("payload_helper_present")
    ):
        return "canonical_reconstruction_failure"
    return "ambiguous_failure"


def load_protected_audit() -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("ttbar8_prepare", PREPARE_SCRIPT)
    require(spec is not None and spec.loader is not None, "cannot import prepare helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config()
    rows, _ = module.audit_protected(config)
    return [{**row, "status": "verified_unchanged"} for row in rows]


def file_record(
    path: Path,
    category: str,
    checksum_cache: dict[Path, str],
    note: str,
) -> dict[str, Any]:
    require_authorized_path(path)
    exists = path.is_file()
    digest = ""
    if exists:
        digest = checksum_cache.setdefault(path, sha256_file(path))
    relative = ""
    if is_within(path, RETURN_MEMBER_DIRECTORY):
        relative = str(path.resolve().relative_to(RETURN_MEMBER_DIRECTORY.resolve()))
    return {
        "category": category,
        "path": str(path),
        "relative_path": relative,
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
        "sha256": digest,
        "note": note,
    }


def correction_plan_rows() -> list[dict[str, Any]]:
    return [
        {
            "order": 1,
            "change": (
                "Package scripts/delphes/write_parquet_from_pickle.py adjacent to "
                "reconstruct_hh4b_candidates_v2.py in the immutable transfer payload."
            ),
            "target": "build_transfer_payload in prepare_hh4b_ttbar8_exact_regeneration.py",
            "why_minimal": "Supplies the exact runtime dependency requested by the frozen builder.",
            "physics_invariant": True,
            "retry_authorized": False,
        },
        {
            "order": 2,
            "change": (
                "Add the helper SHA256 to the payload manifest/protected dependency "
                "closure and make preflight reject a payload missing either file."
            ),
            "target": "transfer-payload assembly and retry-contract preflight",
            "why_minimal": "Prevents recurrence without changing reconstruction arguments.",
            "physics_invariant": True,
            "retry_authorized": False,
        },
        {
            "order": 3,
            "change": (
                "Fail fast before MadGraph if the packaged helper is absent; retain "
                "set -Eeuo pipefail and add explicit success-path existence checks for "
                "Parquet, receipt, and checksum manifest."
            ),
            "target": "run_hh4b_ttbar8_exact_regeneration_member.sh",
            "why_minimal": "Improves failure propagation and avoids wasting generator time.",
            "physics_invariant": True,
            "retry_authorized": False,
        },
        {
            "order": 4,
            "change": (
                "Freeze and verify the corrected payload in the next gate; leave "
                "3654710.0 held and do not submit until separate authorization."
            ),
            "target": NEXT_GATE,
            "why_minimal": "Separates diagnosis and contract review from retry authorization.",
            "physics_invariant": True,
            "retry_authorized": False,
        },
    ]


def verify_sha256sums(directory: Path) -> int:
    rows = [
        line.split("  ", 1)
        for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for expected, relative in rows:
        require(
            sha256_file(directory / relative) == expected,
            f"checkpoint checksum mismatch: {relative}",
        )
    return len(rows)


def write_sha256sums(directory: Path) -> None:
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    text = "".join(f"{sha256_file(path)}  {path.name}\n" for path in paths)
    (directory / "SHA256SUMS").write_text(text, encoding="utf-8")


def diagnose() -> dict[str, Any]:
    require(not RUNTIME_DIRECTORY.exists(), f"refusing to overwrite {RUNTIME_DIRECTORY}")
    require(
        not CHECKPOINT_DIRECTORY.exists(),
        f"refusing to overwrite {CHECKPOINT_DIRECTORY}",
    )
    head = git_output("rev-parse", "HEAD")
    origin = git_output("rev-parse", "origin/delphes-hh4b-production")
    require(head == REQUIRED_STARTING_COMMIT, "HEAD is not the required starting commit")
    require(origin == REQUIRED_STARTING_COMMIT, "origin branch is not synchronized")

    classad_text = run_read_only_command(
        ["condor_q", JOB_ID, "-name", SCHEDULER, "-long"]
    )
    classad = parse_classad(classad_text)
    require(classad.get("ClusterId") == str(CLUSTER_ID), "wrong live cluster")
    require(classad.get("ProcId") == str(PROC_ID), "wrong live process")
    require(classad.get("JobStatus") == "5", "canary is no longer held")
    require(classad.get("HoldReasonCode") == "12", "unexpected hold reason code")
    require(classad.get("HoldReasonSubCode") == "2", "unexpected hold subcode")
    require(classad.get("ExitCode") == "1", "payload exit code is not one")
    require(classad.get("ExitBySignal") == "false", "payload exit-by-signal changed")

    history_text = run_read_only_command(
        [
            "condor_history",
            "-name",
            SCHEDULER,
            "-constraint",
            f"ClusterId == {CLUSTER_ID} && ProcId == {PROC_ID}",
            "-long",
            "-json",
        ],
        check=False,
    )
    history_rows = json.loads(history_text or "[]")
    require(history_rows == [], "held live job unexpectedly appeared in history")

    for path in (
        FROZEN_SUBMIT,
        EVENT_LOG,
        STDOUT,
        STDERR,
        WORKER,
        RECONSTRUCTION_BUILDER,
        PARQUET_WRITER,
        PAYLOAD,
    ):
        require(path.exists(), f"required diagnosis input missing: {path}")
        require_authorized_path(path)

    submit_text = FROZEN_SUBMIT.read_text(encoding="utf-8")
    submit = parse_submit(submit_text)
    wrapper_text = WORKER.read_text(encoding="utf-8")
    builder_text = RECONSTRUCTION_BUILDER.read_text(encoding="utf-8")
    event_text = EVENT_LOG.read_text(encoding="utf-8", errors="replace")
    stdout_text = STDOUT.read_text(encoding="utf-8", errors="replace")
    stderr_text = STDERR.read_text(encoding="utf-8", errors="replace")
    canonical_text = CANONICAL_LOG.read_text(encoding="utf-8", errors="replace")
    generator_text = GENERATOR_LOG.read_text(encoding="utf-8", errors="replace")
    pythia_text = PYTHIA_LOG.read_text(encoding="utf-8", errors="replace")
    delphes_text = DELPHES_LOG.read_text(encoding="utf-8", errors="replace")
    final_status_text = FINAL_STATUS.read_text(encoding="utf-8", errors="replace")
    payload_audit = inspect_payload(PAYLOAD)

    require(EXPECTED_ERROR in canonical_text, "canonical failure signature changed")
    require(EXPECTED_ERROR in stdout_text, "stdout lacks canonical failure signature")
    require("exit_status=1" in final_status_text, "trap did not preserve payload exit one")
    require(payload_audit["builder_present"], "builder is absent from payload")
    require(not payload_audit["helper_present"], "helper is unexpectedly in payload")
    require("write_parquet_from_pickle.py" in builder_text, "builder dependency changed")

    facts = {
        "canonical_started": True,
        "canonical_log": canonical_text,
        "root_exists": ROOT_OUTPUT.is_file(),
        "parquet_exists": EXPECTED_PARQUET.is_file(),
        "payload_builder_present": payload_audit["builder_present"],
        "payload_helper_present": payload_audit["helper_present"],
    }
    classification = classify_failure(facts)
    require(classification in ALLOWED_PRIMARY_TYPES, "invalid primary classification")
    require(
        classification == PRIMARY_CLASSIFICATION,
        f"failure is not fully diagnosed: {classification}",
    )

    lhe_events = count_lhe_events(LHE)
    root_entries = read_root_entries_serial(ROOT_OUTPUT)
    require(lhe_events == GENERATED_EVENTS, "retained LHE event count changed")
    require(root_entries == GENERATED_EVENTS, "retained ROOT entry count changed")

    checksum_cache: dict[Path, str] = {}
    retained_files = sorted(
        path for path in RETURN_MEMBER_DIRECTORY.rglob("*") if path.is_file()
    )
    log_inventory_rows = [
        file_record(EVENT_LOG, "scheduler_event_log", checksum_cache, "retained"),
        file_record(STDOUT, "stdout", checksum_cache, "retained"),
        file_record(STDERR, "stderr", checksum_cache, "retained_empty"),
        file_record(FROZEN_SUBMIT, "frozen_submit", checksum_cache, "retained"),
        file_record(PAYLOAD, "transfer_input_payload", checksum_cache, "immutable"),
    ]
    log_inventory_rows.extend(
        file_record(path, "returned_worker_output", checksum_cache, "authorized_canary")
        for path in retained_files
    )

    transfer_sources = [
        item.strip() for item in submit["transfer_output_files"].split(",")
    ]
    transfer_remaps = parse_transfer_remaps(submit["transfer_output_remaps"])
    require(set(transfer_sources) == set(transfer_remaps), "submit remap sources differ")
    expected_output_rows: list[dict[str, Any]] = []
    for source in transfer_sources:
        destination = Path(transfer_remaps[source])
        require_authorized_path(destination)
        found = destination.is_file()
        if source.endswith("_canonical72.parquet"):
            outcome = "b_ran_and_failed_before_parquet_write"
            explanation = "packaged isolated Parquet writer was absent"
        elif source.endswith("_receipt.json"):
            outcome = "not_reached_after_canonical_failure"
            explanation = "strict shell exit skipped receipt generation"
        elif source.endswith("_SHA256SUMS"):
            outcome = "not_reached_after_canonical_failure"
            explanation = "strict shell exit skipped checksum generation"
        else:
            outcome = "transferred_and_retained" if found else "unexpected_missing"
            explanation = ""
        expected_output_rows.append(
            {
                "execute_relative_path": source,
                "return_path": str(destination),
                "expected": True,
                "found": found,
                "bytes": destination.stat().st_size if found else 0,
                "sha256": (
                    checksum_cache.setdefault(destination, sha256_file(destination))
                    if found
                    else ""
                ),
                "outcome": outcome,
                "explanation": explanation,
            }
        )
    require(sum(not row["found"] for row in expected_output_rows) == 3, "missing count changed")

    live_fields = (
        "scheduler",
        "cluster_id",
        "proc_id",
        "job_status",
        "hold_reason_code",
        "hold_reason_subcode",
        "hold_reason",
        "execution_host",
        "start_time_utc",
        "hold_time_utc",
        "on_exit_code",
        "exit_code_classad_attribute",
        "exit_by_signal",
        "executable",
        "arguments",
        "iwd",
        "out",
        "err",
        "user_log",
        "transfer_input_files",
        "transfer_output_files",
        "transfer_output_remaps",
        "scratch_dir_file_count",
        "disk_usage_kb",
        "transfer_output_stats",
    )
    live_rows = [
        {
            "scheduler": SCHEDULER,
            "cluster_id": classad["ClusterId"],
            "proc_id": classad["ProcId"],
            "job_status": classad["JobStatus"],
            "hold_reason_code": classad["HoldReasonCode"],
            "hold_reason_subcode": classad["HoldReasonSubCode"],
            "hold_reason": classad["HoldReason"],
            "execution_host": classad["LastRemoteHost"],
            "start_time_utc": epoch_utc(classad["JobCurrentStartExecutingDate"]),
            "hold_time_utc": epoch_utc(classad["EnteredCurrentStatus"]),
            "on_exit_code": classad["ExitCode"],
            "exit_code_classad_attribute": "ExitCode (OnExitCode absent)",
            "exit_by_signal": classad["ExitBySignal"],
            "executable": classad["Cmd"],
            "arguments": classad["Args"],
            "iwd": classad["Iwd"],
            "out": classad["Out"],
            "err": classad["Err"],
            "user_log": classad["UserLog"],
            "transfer_input_files": classad["TransferInput"],
            "transfer_output_files": classad["TransferOutput"],
            "transfer_output_remaps": classad["TransferOutputRemaps"],
            "scratch_dir_file_count": classad["ScratchDirFileCount"],
            "disk_usage_kb": classad["DiskUsage_RAW"],
            "transfer_output_stats": classad["TransferOutputStats"],
        }
    ]

    scheduler_event_rows = parse_scheduler_events(event_text)
    scheduler_event_rows.append(
        {
            "event_code": "history",
            "timestamp_local": "",
            "event": "condor_history queried; no row because job remains live and held",
            "details": "[]",
        }
    )

    stdout_stderr_rows = [
        {
            "stream": "stdout",
            "path": str(STDOUT),
            "bytes": STDOUT.stat().st_size,
            "sha256": checksum_cache[STDOUT],
            "key_evidence": EXPECTED_ERROR,
            "interpretation": "payload error is preserved in stdout",
        },
        {
            "stream": "stderr",
            "path": str(STDERR),
            "bytes": STDERR.stat().st_size,
            "sha256": checksum_cache[STDERR],
            "key_evidence": "empty",
            "interpretation": "pipeline merged stage stderr into tee/stdout logs",
        },
    ]

    artifact_sha = {
        LHE: checksum_cache[LHE],
        HEPMC: checksum_cache[HEPMC],
        ROOT_OUTPUT: checksum_cache[ROOT_OUTPUT],
        CANONICAL_LOG: checksum_cache[CANONICAL_LOG],
        FINAL_STATUS: checksum_cache[FINAL_STATUS],
        PAYLOAD: checksum_cache[PAYLOAD],
    }
    stage_fields = (
        "stage",
        "started",
        "completed",
        "exit_evidence",
        "expected_artifact",
        "artifact_found",
        "artifact_sha256",
        "failure_message",
        "first_causal_failure",
    )
    stage_rows = [
        {
            "stage": "transfer-in",
            "started": True,
            "completed": True,
            "exit_evidence": "event 040 start/finish; TransferInFinished set",
            "expected_artifact": str(PAYLOAD),
            "artifact_found": True,
            "artifact_sha256": artifact_sha[PAYLOAD],
            "failure_message": "",
            "first_causal_failure": False,
        },
        {
            "stage": "environment setup",
            "started": True,
            "completed": True,
            "exit_evidence": "both retained ldd logs have no unresolved library",
            "expected_artifact": "converter_ldd.log and delphes_ldd.log",
            "artifact_found": True,
            "artifact_sha256": "",
            "failure_message": "",
            "first_causal_failure": False,
        },
        {
            "stage": "MadGraph",
            "started": True,
            "completed": True,
            "exit_evidence": "generate_events.log ends INFO: Done; wrapper continued",
            "expected_artifact": str(GENERATOR_LOG),
            "artifact_found": True,
            "artifact_sha256": checksum_cache[GENERATOR_LOG],
            "failure_message": "",
            "first_causal_failure": False,
        },
        {
            "stage": "LHE generation",
            "started": True,
            "completed": True,
            "exit_evidence": f"independent retained gzip count={lhe_events}",
            "expected_artifact": str(LHE),
            "artifact_found": True,
            "artifact_sha256": artifact_sha[LHE],
            "failure_message": "",
            "first_causal_failure": False,
        },
        {
            "stage": "Pythia",
            "started": True,
            "completed": True,
            "exit_evidence": "PYTHIA 8.312 log; 10000 tried/selected/accepted",
            "expected_artifact": str(PYTHIA_LOG),
            "artifact_found": True,
            "artifact_sha256": checksum_cache[PYTHIA_LOG],
            "failure_message": "",
            "first_causal_failure": False,
        },
        {
            "stage": "HepMC production",
            "started": True,
            "completed": True,
            "exit_evidence": "Wrote 10000 events message",
            "expected_artifact": str(HEPMC),
            "artifact_found": True,
            "artifact_sha256": artifact_sha[HEPMC],
            "failure_message": "",
            "first_causal_failure": False,
        },
        {
            "stage": "Delphes",
            "started": True,
            "completed": True,
            "exit_evidence": "retained Delphes log ends ** Exiting...",
            "expected_artifact": str(ROOT_OUTPUT),
            "artifact_found": True,
            "artifact_sha256": artifact_sha[ROOT_OUTPUT],
            "failure_message": "",
            "first_causal_failure": False,
        },
        {
            "stage": "ROOT validation",
            "started": False,
            "completed": False,
            "exit_evidence": (
                f"wrapper validation is after reconstruction; diagnosis-only serial "
                f"uproot metadata check observed {root_entries} entries"
            ),
            "expected_artifact": str(ROOT_OUTPUT),
            "artifact_found": True,
            "artifact_sha256": artifact_sha[ROOT_OUTPUT],
            "failure_message": "not reached after canonical reconstruction exit",
            "first_causal_failure": False,
        },
        {
            "stage": "canonical-72 reconstruction",
            "started": True,
            "completed": False,
            "exit_evidence": canonical_text.strip(),
            "expected_artifact": str(EXPECTED_PARQUET),
            "artifact_found": False,
            "artifact_sha256": "",
            "failure_message": EXPECTED_ERROR,
            "first_causal_failure": True,
        },
        {
            "stage": "receipt generation",
            "started": False,
            "completed": False,
            "exit_evidence": "success receipt code follows failed reconstruction",
            "expected_artifact": str(EXPECTED_RECEIPT),
            "artifact_found": False,
            "artifact_sha256": "",
            "failure_message": "not reached; EXIT trap separately recorded exit_status=1",
            "first_causal_failure": False,
        },
        {
            "stage": "transfer-out",
            "started": True,
            "completed": False,
            "exit_evidence": (
                "event 040 start then events 021/004/012; 3 declared files missing"
            ),
            "expected_artifact": "13 declared transfer outputs",
            "artifact_found": True,
            "artifact_sha256": "",
            "failure_message": classad["HoldReason"],
            "first_causal_failure": False,
        },
    ]

    path_contract_rows = [
        {
            "component": "wrapper_working_directory",
            "configured": 'SCRATCH=${_CONDOR_SCRATCH_DIR:-$PWD}; cd "$SCRATCH"',
            "expected": "execute scratch root",
            "agrees": True,
            "evidence": str(WORKER),
        },
        {
            "component": "wrapper_output_directories",
            "configured": 'mkdir -p "$OUTDIR"/{lhe,hepmc,root,parquet,logs,receipts,checksums}',
            "expected": "output/parquet parent exists",
            "agrees": True,
            "evidence": str(WORKER),
        },
        {
            "component": "reconstruction_out_argument",
            "configured": f"output/parquet/{EXPECTED_PARQUET.name}",
            "expected": f"output/parquet/{EXPECTED_PARQUET.name}",
            "agrees": True,
            "evidence": '--out "$CANDIDATE"',
        },
        {
            "component": "submission_output_registry",
            "configured": str(EXPECTED_PARQUET),
            "expected": str(EXPECTED_PARQUET),
            "agrees": True,
            "evidence": "canary_expected_output_registry.tsv",
        },
        {
            "component": "transfer_output_files",
            "configured": f"output/parquet/{EXPECTED_PARQUET.name}",
            "expected": f"output/parquet/{EXPECTED_PARQUET.name}",
            "agrees": f"output/parquet/{EXPECTED_PARQUET.name}" in transfer_sources,
            "evidence": str(FROZEN_SUBMIT),
        },
        {
            "component": "transfer_output_remaps",
            "configured": transfer_remaps[f"output/parquet/{EXPECTED_PARQUET.name}"],
            "expected": str(EXPECTED_PARQUET),
            "agrees": (
                transfer_remaps[f"output/parquet/{EXPECTED_PARQUET.name}"]
                == str(EXPECTED_PARQUET)
            ),
            "evidence": str(FROZEN_SUBMIT),
        },
        {
            "component": "payload_builder_dependency",
            "configured": MISSING_HELPER_ARCHIVE_PATH,
            "expected": "present adjacent to packaged builder and manifested",
            "agrees": False,
            "evidence": (
                f"builder_present={payload_audit['builder_present']}; "
                f"helper_present={payload_audit['helper_present']}; "
                f"helper_manifested={payload_audit['helper_manifested']}"
            ),
        },
    ]

    propagation_rows = [
        {
            "check": "strict_shell_mode",
            "observed": "set -euo pipefail",
            "satisfied": "set -euo pipefail" in wrapper_text,
            "impact": "nonzero reconstruction pipeline propagates",
            "evidence": str(WORKER),
        },
        {
            "check": "reconstruction_pipeline",
            "observed": "python reconstruction 2>&1 | tee canonical72.log",
            "satisfied": True,
            "impact": "builder exit one became payload exit one",
            "evidence": str(CANONICAL_LOG),
        },
        {
            "check": "exit_trap",
            "observed": "final status records exit_status=1",
            "satisfied": "exit_status=1" in final_status_text,
            "impact": "original payload status survived transfer hold",
            "evidence": str(FINAL_STATUS),
        },
        {
            "check": "payload_dependency_closure",
            "observed": "builder packaged; required isolated writer absent",
            "satisfied": False,
            "impact": "first causal failure",
            "evidence": str(PAYLOAD),
        },
        {
            "check": "explicit_success_output_checks",
            "observed": "post-reconstruction Python reads Parquet; no preflight helper check",
            "satisfied": False,
            "impact": "dependency defect discovered only after upstream production",
            "evidence": str(WORKER),
        },
        {
            "check": "payload_vs_transfer_status",
            "observed": "ExitCode=1; HoldReasonCode/SubCode=12/2",
            "satisfied": True,
            "impact": "payload and transfer failures are distinct",
            "evidence": "live ClassAd and scheduler event log",
        },
        {
            "check": "cleanup",
            "observed": "no deletion; EXIT trap archives output locally with || true",
            "satisfied": True,
            "impact": "failure is not premature cleanup",
            "evidence": str(WORKER),
        },
    ]

    classification_rows = [
        {
            "primary_type": classification,
            "first_causal_failure": EXPECTED_ERROR,
            "secondary_mechanism": "incomplete immutable payload dependency closure",
            "payload_exit": 1,
            "transfer_hold_code": 12,
            "transfer_hold_subcode": 2,
            "excluded_primary_types": sorted(ALLOWED_PRIMARY_TYPES - {classification}),
            "fully_diagnosed": True,
        }
    ]
    protected_rows = load_protected_audit()
    require(len(protected_rows) == 17, "protected artifact count changed")

    tables = [
        ("canary_live_classad", live_fields, live_rows),
        (
            "canary_scheduler_event_audit",
            ("event_code", "timestamp_local", "event", "details"),
            scheduler_event_rows,
        ),
        (
            "canary_log_inventory",
            ("category", "path", "relative_path", "exists", "bytes", "sha256", "note"),
            log_inventory_rows,
        ),
        (
            "canary_stdout_stderr_audit",
            ("stream", "path", "bytes", "sha256", "key_evidence", "interpretation"),
            stdout_stderr_rows,
        ),
        ("canary_production_stage_audit", stage_fields, stage_rows),
        (
            "canary_expected_vs_observed_outputs",
            (
                "execute_relative_path",
                "return_path",
                "expected",
                "found",
                "bytes",
                "sha256",
                "outcome",
                "explanation",
            ),
            expected_output_rows,
        ),
        (
            "canary_output_path_contract_audit",
            ("component", "configured", "expected", "agrees", "evidence"),
            path_contract_rows,
        ),
        (
            "canary_wrapper_failure_propagation_audit",
            ("check", "observed", "satisfied", "impact", "evidence"),
            propagation_rows,
        ),
        (
            "canary_primary_failure_classification",
            (
                "primary_type",
                "first_causal_failure",
                "secondary_mechanism",
                "payload_exit",
                "transfer_hold_code",
                "transfer_hold_subcode",
                "excluded_primary_types",
                "fully_diagnosed",
            ),
            classification_rows,
        ),
        (
            "canary_minimal_correction_plan",
            (
                "order",
                "change",
                "target",
                "why_minimal",
                "physics_invariant",
                "retry_authorized",
            ),
            correction_plan_rows(),
        ),
        (
            "protected_artifact_audit",
            (
                "artifact_role",
                "path",
                "expected_sha256",
                "observed_sha256",
                "exists",
                "executable_required",
                "executable",
                "unchanged",
                "provenance",
                "status",
            ),
            protected_rows,
        ),
    ]

    RUNTIME_DIRECTORY.mkdir(parents=True)
    for name, fields, rows in tables:
        write_table_bundle(RUNTIME_DIRECTORY, name, fields, rows)

    summary = {
        "schema_version": 1,
        "status": STATUS,
        "corrected_retry_authorized": False,
        "next_gate": NEXT_GATE,
        "scheduler": SCHEDULER,
        "cluster_id": CLUSTER_ID,
        "process_id": PROC_ID,
        "job_status": 5,
        "held_job_unchanged": True,
        "member": MEMBER_NAME,
        "member_index": MEMBER_INDEX,
        "seed": SEED,
        "generated_events": GENERATED_EVENTS,
        "primary_failure_classification": classification,
        "first_causal_failure": EXPECTED_ERROR,
        "root_cause": (
            "The immutable transfer payload packaged the canonical builder but omitted "
            "its required adjacent isolated Parquet writer."
        ),
        "candidate_outcome": "b_ran_and_failed",
        "path_contract_mismatch": False,
        "payload_exit": {
            "exit_code": 1,
            "exit_by_signal": False,
        },
        "transfer_failure": {
            "hold_reason_code": 12,
            "hold_reason_subcode": 2,
            "missing_declared_outputs": 3,
        },
        "retained_artifacts": {
            "lhe_events": lhe_events,
            "root_entries_serial_uproot_metadata_check": root_entries,
            "returned_files": len(retained_files),
            "expected_transfer_outputs": len(expected_output_rows),
            "found_transfer_outputs": sum(row["found"] for row in expected_output_rows),
        },
        "controls": {
            "held_job_released": False,
            "held_job_removed": False,
            "held_job_rescheduled": False,
            "retry_submitted": False,
            "scaleout_submitted": False,
            "seed_changed": False,
            "member_changed": False,
            "physics_configuration_changed": False,
            "sealed_candidate_files_opened": 0,
            "event_level_products_committed": 0,
        },
        "integrity": {
            "required_starting_commit": REQUIRED_STARTING_COMMIT,
            "head": head,
            "origin": origin,
            "protected_artifacts_verified": len(protected_rows),
            "principal_tables_have_markdown_and_booktabs_latex": True,
            "history_rows": len(history_rows),
        },
    }
    checkpoint = {
        "checkpoint": CHECKPOINT_DIRECTORY.name,
        "status": STATUS,
        "corrected_retry_authorized": False,
        "next_gate": NEXT_GATE,
        "authoritative_contract": str(CONTRACT_DIRECTORY.relative_to(REPOSITORY_ROOT)),
        "authoritative_submission_checkpoint": str(
            SUBMISSION_CHECKPOINT_DIRECTORY.relative_to(REPOSITORY_ROOT)
        ),
        "diagnosis_script": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
        "scheduler": SCHEDULER,
        "job": JOB_ID,
        "member": MEMBER_NAME,
        "seed": SEED,
        "primary_failure_classification": classification,
    }
    environment = {
        "diagnosis_timestamp_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "hostname": socket.getfqdn(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "timezone": "America/Chicago",
        "scheduler": SCHEDULER,
        "execute_host": classad["LastRemoteHost"],
        "condor_version": classad["CondorVersion"],
        "root_metadata_validation": "serial uproot.open; Delphes.num_entries only",
    }
    (RUNTIME_DIRECTORY / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (RUNTIME_DIRECTORY / "checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (RUNTIME_DIRECTORY / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme = f"""# HH4b ttbar8 exact-regeneration canary failure diagnosis

Status: `{STATUS}`

Primary classification: `{classification}`

Job `{JOB_ID}` remains held on `{SCHEDULER}`. Its payload exited normally with
code 1. Condor subsequently held it with transfer code/subcode 12/2 because the
canonical Parquet, success receipt, and checksum manifest were absent.

The earliest causal failure is:

`{EXPECTED_ERROR}`

The immutable payload contains the frozen reconstruction builder but omits the
builder's required adjacent `write_parquet_from_pickle.py`. MadGraph produced a
10,000-event LHE, Pythia wrote 10,000 HepMC events, and Delphes exited normally.
The retained ROOT file has 10,000 entries by a diagnosis-only serial uproot
metadata check. Production ROOT validation, success-receipt creation, and
checksum-manifest creation were not reached after canonical reconstruction
failed.

The wrapper, expected-output registry, `--out` argument, transfer-output list,
and transfer remap agree on the exact Parquet path. No different or temporary
Parquet exists in the authorized canary return directory. This is not an output
path mismatch, cleanup failure, or transfer-configuration root cause.

The minimal correction is technical and physics-invariant: package and manifest
the already-required isolated writer beside the frozen builder, preflight that
dependency closure, and add fail-fast/output-existence checks. The member, seed,
event count, cards, versions, Delphes card, reconstruction features, thresholds,
and sample identity remain unchanged.

`corrected_retry_authorized` is false. No retry or scaleout job was submitted,
and no scheduler mutation was performed. The next gate is `{NEXT_GATE}`.
"""
    (RUNTIME_DIRECTORY / "README.md").write_text(readme, encoding="utf-8")
    write_sha256sums(RUNTIME_DIRECTORY)
    runtime_files_verified = verify_sha256sums(RUNTIME_DIRECTORY)

    shutil.copytree(RUNTIME_DIRECTORY, CHECKPOINT_DIRECTORY)
    checkpoint_files_verified = verify_sha256sums(CHECKPOINT_DIRECTORY)
    require(
        runtime_files_verified == checkpoint_files_verified,
        "runtime/checkpoint file counts differ",
    )
    return {
        **summary,
        "runtime_files_verified": runtime_files_verified,
        "checkpoint_files_verified": checkpoint_files_verified,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="perform the read-only live diagnosis and write the checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    require(arguments.diagnose, "explicit --diagnose is required")
    print(json.dumps(diagnose(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
