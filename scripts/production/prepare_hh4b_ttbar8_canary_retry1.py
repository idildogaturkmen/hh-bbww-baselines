#!/usr/bin/env python3
"""Freeze the reconstruction-only retry contract for HH4b ttbar8 canary job 3654710.0.

This preparation gate validates the returned ROOT, builds and extracts the
corrected payload, runs no-physics environment/writer smoke tests, and emits
one inert Condor submit description. It never runs reconstruction, submits a
job, or changes the held source job.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import uproot
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "production"
    / "hh4b_ttbar8_canary_retry1_v1.yaml"
)
SUBMIT_FILE_NAME = "hh4b_ttbar8_exact_regeneration_canary_retry1.sub"
PAYLOAD_FILE_NAME = (
    "hh4b_ttbar8_exact_regeneration_canary_retry1_payload.tar.gz"
)
JOB_ID = "3654710.0"
MEMBER = "ttbar_100k_shard003"
SEED = 105003
REQUIRED_COMMIT = "4038995254c2bcbcad97d778c63e8e6b0b4c7916"
FROZEN_STATUS = "hh4b_ttbar8_exact_regeneration_canary_retry_contract_frozen"
NEXT_GATE = "submit_hh4b_ttbar8_exact_regeneration_canary_retry1"

TABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "retry_returned_root_input_audit": (
        "check",
        "expected",
        "observed",
        "passed",
        "evidence",
    ),
    "retry_payload_manifest": (
        "archive_path",
        "source_path",
        "role",
        "bytes",
        "sha256",
        "executable",
    ),
    "retry_payload_extraction_audit": (
        "archive_path",
        "extracted_path",
        "exact_worker_path",
        "exists",
        "sha256",
        "expected_sha256",
        "checksum_match",
        "syntax_check",
    ),
    "retry_writer_smoke_test": (
        "check",
        "expected",
        "observed",
        "passed",
        "environment",
    ),
    "retry_environment_contract": (
        "component",
        "expected",
        "observed",
        "passed",
        "evidence",
    ),
    "retry_reconstruction_command": (
        "field",
        "frozen_value",
        "verification",
    ),
    "retry_expected_output_registry": (
        "role",
        "relative_path",
        "required_on_success",
        "preserved_on_failure",
        "placeholder_forbidden",
    ),
    "retry_transfer_and_failure_policy": (
        "requirement",
        "frozen_value",
        "verification",
    ),
    "retry_legacy_comparison_contract": (
        "column",
        "role",
        "canonical_column",
        "comparison",
        "rtol",
        "atol",
        "exact_required",
        "nonfinite_mask_compared",
    ),
    "retry_held_job_removal_policy": (
        "order",
        "action",
        "execute_in_this_gate",
        "required_before_next_action",
    ),
    "retry_condor_submission_plan": (
        "campaign",
        "member",
        "seed",
        "source_job",
        "retry_mode",
        "submit_file",
        "jobs_described",
        "retry_jobs_submitted",
        "scaleout_jobs_submitted",
        "requirements",
        "initially_held",
        "transfer_inputs",
        "transfer_output",
    ),
    "retry_resource_request": (
        "request_cpus",
        "request_memory_mb",
        "request_disk_mb",
        "max_runtime_seconds",
        "successful_wall_seconds",
        "determinism_wall_seconds",
        "successful_input_bytes",
        "current_input_bytes",
        "prior_peak_memory_mb",
        "evidence_path",
        "evidence_sha256",
        "rationale",
    ),
    "retry_pass_conditions": (
        "condition",
        "required",
        "contract_frozen",
        "verification_stage",
        "failure_action",
    ),
    "protected_artifact_audit": (
        "role",
        "path",
        "before_sha256",
        "after_sha256",
        "expected_sha256",
        "unchanged",
        "matches_expected",
    ),
}

BOOTSTRAP_TEXT = '''"""Force the validated one-worker uproot source mechanism."""

import uproot


uproot.open.defaults["handler"] = uproot.source.file.MultithreadedFileSource
uproot.open.defaults["num_workers"] = 1
'''

ENVIRONMENT_PROBE_TEXT = """\
import importlib.util
import json
from pathlib import Path
import sys

import awkward
import numpy
import pandas
import pyarrow
import uproot

imported_scripts = []
for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    spec = importlib.util.spec_from_file_location(
        f"retry_import_check_{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import-check {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    imported_scripts.append(path.name)

print(json.dumps({
    "awkward": awkward.__version__,
    "imported_scripts": imported_scripts,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "parquet_backend": "pyarrow",
    "pyarrow": pyarrow.__version__,
    "python": sys.version.split()[0],
    "uproot": uproot.__version__,
}, sort_keys=True))
"""

WRITER_SMOKE_TEXT = """\
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pyarrow.parquet as pq

writer = Path(sys.argv[1])
work = Path(sys.argv[2])
pickle_path = work / "synthetic_contract_only.pickle"
parquet_path = work / "synthetic_contract_only.parquet"
expected_columns = ["contract_id", "ordinal", "measurement"]
expected_types = {
    "contract_id": "string",
    "ordinal": "int64",
    "measurement": "double",
}
frame = pd.DataFrame({
    "contract_id": pd.Series(["alpha", "beta"], dtype="string"),
    "ordinal": pd.Series([7, 11], dtype="int64"),
    "measurement": pd.Series([1.25, 2.5], dtype="float64"),
})
frame.to_pickle(pickle_path)
completed = subprocess.run(
    [
        sys.executable,
        str(writer),
        "--input",
        str(pickle_path),
        "--output",
        str(parquet_path),
    ],
    check=False,
    capture_output=True,
    text=True,
)
if completed.returncode:
    raise SystemExit(
        f"writer failed ({completed.returncode}): {completed.stderr}"
    )
table = pq.read_table(parquet_path)
observed_types = {field.name: str(field.type) for field in table.schema}
if table.column_names != expected_columns:
    raise SystemExit("synthetic writer changed provided column order")
if observed_types != expected_types:
    raise SystemExit("synthetic writer changed provided schema")
round_trip = table.to_pandas()
if len(round_trip) != 2:
    raise SystemExit("synthetic Parquet row count changed")
print(json.dumps({
    "column_order_preserved": True,
    "columns": table.column_names,
    "parquet_readable": True,
    "rows": table.num_rows,
    "schema_preserved": True,
    "types": observed_types,
    "writer_returncode": completed.returncode,
}, sort_keys=True))
"""


class RetryContractError(RuntimeError):
    """Raised when the retry contract cannot be frozen."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RetryContractError(message)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    require(isinstance(config, dict), "configuration is not a mapping")
    return config


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_sha256sums(directory: Path) -> int:
    manifest = directory / "SHA256SUMS"
    require(manifest.is_file(), f"missing SHA256SUMS: {manifest}")
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        target = directory / relative.strip()
        require(target.is_file(), f"checkpoint member missing: {target}")
        require(sha256_file(target) == expected, f"checksum mismatch: {target}")
        count += 1
    return count


def validate_config(config: Mapping[str, Any]) -> None:
    retry = config["retry"]
    root = config["returned_root"]
    require(config["required_starting_commit"] == REQUIRED_COMMIT, "starting commit")
    require(retry["campaign"].endswith("_retry1_20260727_v1"), "retry campaign")
    require(retry["member"] == MEMBER, "retry member changed")
    require(int(retry["seed_provenance"]) == SEED, "seed provenance changed")
    require(retry["source_job"] == JOB_ID, "source job changed")
    require(retry["jobs_described"] == 1, "retry must describe one job")
    require(retry["jobs_submitted"] == 0, "retry already marked submitted")
    require(retry["scaleout_jobs_submitted"] == 0, "scaleout marked submitted")
    require(not retry["submit_in_this_gate"], "submission enabled")
    require(not retry["remove_source_job_in_this_gate"], "removal enabled")
    require(not retry["release_source_job"], "release enabled")
    require(root["entries"] == 10000, "ROOT entry contract changed")
    require(len(root["required_branches"]) == 6, "required branch contract changed")
    require(config["reconstruction"]["expected_columns"] == 72, "schema count")
    require(config["environment"]["container_image"] == "none", "new container")
    require(config["transfer_policy"]["transfer_output_files"] == "output", "transfer")
    require(config["legacy_validation"]["role"].startswith("validation_only"), "legacy")
    require(config["resources"]["request_cpus"] == 1, "serial CPU contract changed")
    require(config["resources"]["max_runtime_seconds"] == 2700, "runtime changed")


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
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = TABLE_FIELDS[name]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: cell(row.get(field)) for field in fields})
    tsv = "\n".join(line.rstrip("\t") for line in buffer.getvalue().splitlines())
    (directory / f"{name}.tsv").write_text(tsv + "\n", encoding="utf-8")

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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate_starting_state() -> dict[str, str]:
    head = git_output("rev-parse", "HEAD")
    origin = git_output("rev-parse", "origin/delphes-hh4b-production")
    branch = git_output("branch", "--show-current")
    require(head == REQUIRED_COMMIT, f"HEAD is not {REQUIRED_COMMIT}")
    require(origin == REQUIRED_COMMIT, "origin branch changed")
    require(branch == "delphes-hh4b-production", "wrong branch")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "4038995", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    require(ancestor.returncode == 0, "diagnosis commit is not in HEAD")
    return {"branch": branch, "head": head, "origin": origin}


def validate_authoritative_checkpoints(
    config: Mapping[str, Any],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in (
        "authoritative_contract",
        "authoritative_submission",
        "authoritative_failure",
    ):
        directory = resolve_path(config["paths"][key])
        counts[key] = verify_sha256sums(directory)
    failure = resolve_path(config["paths"]["authoritative_failure"])
    summary = json.loads((failure / "summary.json").read_text(encoding="utf-8"))
    require(
        summary["status"]
        == "hh4b_ttbar8_exact_regeneration_canary_failure_diagnosed",
        "failure checkpoint status changed",
    )
    require(summary["primary_failure_classification"] == "canonical_reconstruction_failure", "failure classification changed")
    require(summary["cluster_id"] == 3654710, "source cluster changed")
    require(summary["member"] == MEMBER and summary["seed"] == SEED, "identity changed")
    return counts


def live_held_job() -> dict[str, Any]:
    launcher = shutil.which("condor_q")
    require(launcher is not None, "condor_q unavailable")
    completed = subprocess.run(
        [
            "/usr/bin/bash",
            launcher,
            "-name",
            "lpcschedd4.fnal.gov",
            JOB_ID,
            "-af",
            "ClusterId",
            "ProcId",
            "JobStatus",
            "HoldReasonCode",
            "HoldReasonSubCode",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    require(completed.returncode == 0, f"condor_q failed: {completed.stderr}")
    fields = completed.stdout.strip().split()
    require(len(fields) == 5, f"job {JOB_ID} is not one live queue row")
    cluster, proc, status, reason, subcode = (int(value) for value in fields)
    require((cluster, proc, status) == (3654710, 0, 5), "source job is not held")
    return {
        "cluster": cluster,
        "proc": proc,
        "status": status,
        "hold_reason_code": reason,
        "hold_reason_subcode": subcode,
    }


def audit_returned_root(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_contract = config["returned_root"]
    path = Path(root_contract["path"])
    require(path.is_file(), f"returned ROOT missing: {path}")
    size = path.stat().st_size
    digest = sha256_file(path)
    require(size == root_contract["bytes"], "returned ROOT size changed")
    require(digest == root_contract["sha256"], "returned ROOT SHA256 changed")
    with uproot.open(
        path,
        handler=uproot.source.file.MultithreadedFileSource,
        num_workers=1,
    ) as source:
        tree = source[root_contract["tree"]]
        entries = int(tree.num_entries)
        branches = {
            name: name in tree for name in root_contract["required_branches"]
        }
    require(entries == root_contract["entries"], "returned ROOT entries changed")
    require(all(branches.values()), "required Delphes branch missing")

    failure = resolve_path(config["paths"]["authoritative_failure"])
    classad_rows = read_tsv(failure / "canary_live_classad.tsv")
    require(len(classad_rows) == 1, "frozen classad row count changed")
    classad = classad_rows[0]
    require(classad["cluster_id"] == "3654710" and classad["proc_id"] == "0", "classad identity")
    require(classad["arguments"] == "326 ttbar_100k_shard003 105003 10000", "classad arguments")
    require(str(path) in classad["transfer_output_remaps"], "ROOT not tied by transfer remap")
    stages = read_tsv(failure / "canary_production_stage_audit.tsv")
    delphes = next(row for row in stages if row["stage"] == "Delphes")
    require(delphes["artifact_sha256"] == digest, "stage audit ROOT SHA changed")
    require(delphes["completed"] == "True", "Delphes was not completed")

    member_dir = path.parents[1]
    generator_log = member_dir / "logs" / f"{MEMBER}_generate_events.log"
    pythia_log = member_dir / "logs" / f"{MEMBER}_lhe_to_hepmc3.log"
    delphes_log = member_dir / "logs" / f"{MEMBER}_pythia8_delphes.log"
    canonical_log = member_dir / "logs" / f"{MEMBER}_canonical72.log"
    require("seed offset = 105003" in generator_log.read_text(encoding="utf-8"), "seed log evidence missing")
    require("Wrote 10000 events" in pythia_log.read_text(encoding="utf-8"), "Pythia event evidence missing")
    require("** Exiting..." in delphes_log.read_text(encoding="utf-8"), "Delphes completion marker missing")
    require("missing isolated Parquet writer" in canonical_log.read_text(encoding="utf-8"), "writer failure evidence missing")

    audit = {
        "path": str(path),
        "bytes": size,
        "sha256": digest,
        "tree": root_contract["tree"],
        "entries": entries,
        "branches": branches,
        "readable": True,
        "source_job": JOB_ID,
        "source_member": MEMBER,
        "source_seed": SEED,
    }
    checks = [
        ("path", root_contract["path"], path, path.is_file(), classad["transfer_output_remaps"]),
        ("file_size_bytes", root_contract["bytes"], size, size == root_contract["bytes"], path),
        ("sha256_immutable_input", root_contract["sha256"], digest, digest == root_contract["sha256"], delphes["artifact_sha256"]),
        ("source_job", JOB_ID, JOB_ID, True, failure / "canary_live_classad.tsv"),
        ("source_member", MEMBER, MEMBER, True, classad["arguments"]),
        ("source_seed", SEED, SEED, True, generator_log),
        ("tree_name", root_contract["tree"], root_contract["tree"], True, path),
        ("tree_entries", 10000, entries, entries == 10000, path),
        ("required_branches", root_contract["required_branches"], branches, all(branches.values()), path),
        ("readability", True, True, True, "serial uproot open"),
        ("failed_canary_log_provenance", True, True, True, canonical_log),
    ]
    rows = [
        {
            "check": check,
            "expected": expected,
            "observed": observed,
            "passed": passed,
            "evidence": evidence,
        }
        for check, expected, observed, passed, evidence in checks
    ]
    require(all(row["passed"] for row in rows), "returned ROOT input audit failed")
    return audit, rows


def protected_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": Path(item["archive_path"]).name,
            "path": str(item["source"]),
            "expected_sha256": item["sha256"],
        }
        for item in config["payload"]["required_files"]
    ]


def protected_fingerprints(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    specs = protected_specs(config)
    values: dict[str, str] = {}
    for spec in specs:
        path = resolve_path(spec["path"])
        require(path.is_file(), f"protected artifact missing: {path}")
        digest = sha256_file(path)
        require(digest == spec["expected_sha256"], f"protected artifact changed: {path}")
        values[str(path)] = digest
    return specs, values


def write_payload_sha256sums(payload: Path) -> None:
    files = sorted(
        path
        for path in payload.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    text = "\n".join(
        f"{sha256_file(path)}  {path.relative_to(payload).as_posix()}"
        for path in files
    )
    (payload / "SHA256SUMS").write_text(text + "\n", encoding="utf-8")


def create_reproducible_tar(payload: Path, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    with partial.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", dereference=False) as archive:
                for path in [payload, *sorted(payload.rglob("*"))]:
                    arcname = Path("payload") / path.relative_to(payload)
                    info = archive.gettarinfo(str(path), arcname.as_posix())
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    if path.is_file():
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
                    else:
                        archive.addfile(info)
    os.replace(partial, destination)


def build_payload(
    config: Mapping[str, Any],
    runtime: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    destination = resolve_path(config["paths"]["payload"])
    require(destination.parent == runtime, "payload must be inside runtime directory")
    with tempfile.TemporaryDirectory(prefix="payload_stage.", dir=runtime) as temporary:
        payload = Path(temporary) / "payload"
        payload.mkdir()
        source_by_archive: dict[str, str] = {}
        for item in config["payload"]["required_files"]:
            source = resolve_path(item["source"])
            target = Path(temporary) / item["archive_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_by_archive[item["archive_path"]] = str(item["source"])
        bootstrap = Path(temporary) / config["payload"]["bootstrap_archive_path"]
        bootstrap.parent.mkdir(parents=True, exist_ok=True)
        bootstrap.write_text(BOOTSTRAP_TEXT, encoding="utf-8")
        source_by_archive[config["payload"]["bootstrap_archive_path"]] = "generated validated serial-uproot bootstrap"
        manifest = {
            "campaign": config["retry"]["campaign"],
            "member": MEMBER,
            "mode": config["retry"]["mode"],
            "source_job": JOB_ID,
            "source_root_sha256": config["returned_root"]["sha256"],
            "seed_provenance": SEED,
        }
        manifest_path = payload / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_by_archive["payload/manifest.json"] = "generated retry manifest"
        write_payload_sha256sums(payload)
        source_by_archive["payload/SHA256SUMS"] = "generated payload checksum manifest"
        create_reproducible_tar(payload, destination)

    rows: list[dict[str, Any]] = []
    with tarfile.open(destination, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            require(extracted is not None, f"cannot read payload member: {member.name}")
            content = extracted.read()
            rows.append(
                {
                    "archive_path": member.name,
                    "source_path": source_by_archive.get(member.name, "generated"),
                    "role": (
                        "checksum_manifest"
                        if member.name.endswith("/SHA256SUMS")
                        else "payload_file"
                    ),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "executable": bool(member.mode & 0o111),
                }
            )
    require(
        any(
            row["archive_path"]
            == "payload/repo/scripts/delphes/write_parquet_from_pickle.py"
            for row in rows
        ),
        "corrected payload still omits writer",
    )
    return destination, rows


def run_lcg_python(setup: Path, script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/usr/bin/bash",
            "-c",
            'set +u; source "$1"; set -u; shift; python3 "$@"',
            "bash",
            str(setup),
            str(script),
            *arguments,
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def extract_and_smoke_test(
    config: Mapping[str, Any],
    payload_path: Path,
    runtime: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    setup = Path(config["environment"]["lcg_setup"])
    require(setup.is_file(), f"LCG setup missing: {setup}")
    require(sha256_file(setup) == config["environment"]["lcg_setup_sha256"], "LCG setup changed")
    with tempfile.TemporaryDirectory(prefix="payload_dryrun.", dir=runtime) as temporary:
        extraction_root = Path(temporary) / "extract"
        extraction_root.mkdir()
        with tarfile.open(payload_path, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                target = (extraction_root / member.name).resolve()
                require(
                    target == extraction_root.resolve()
                    or extraction_root.resolve() in target.parents,
                    f"unsafe archive path: {member.name}",
                )
            archive.extractall(extraction_root)
        payload = extraction_root / "payload"
        verify_sha256sums(payload)

        extraction_rows: list[dict[str, Any]] = []
        syntax_by_path: dict[str, str] = {}
        for item in config["payload"]["required_files"]:
            path = Path(temporary) / "extract" / item["archive_path"]
            syntax = "not_python"
            if path.suffix == ".py":
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
                syntax = "pass"
            syntax_by_path[item["archive_path"]] = syntax
            observed = sha256_file(path) if path.is_file() else ""
            extraction_rows.append(
                {
                    "archive_path": item["archive_path"],
                    "extracted_path": str(path),
                    "exact_worker_path": item["exact_worker_path"],
                    "exists": path.is_file(),
                    "sha256": observed,
                    "expected_sha256": item["sha256"],
                    "checksum_match": observed == item["sha256"],
                    "syntax_check": syntax,
                }
            )
        bootstrap_path = payload / "bootstrap" / "sitecustomize.py"
        compile(bootstrap_path.read_text(encoding="utf-8"), str(bootstrap_path), "exec")
        extraction_rows.append(
            {
                "archive_path": "payload/bootstrap/sitecustomize.py",
                "extracted_path": str(bootstrap_path),
                "exact_worker_path": "/srv/payload/bootstrap/sitecustomize.py",
                "exists": True,
                "sha256": sha256_file(bootstrap_path),
                "expected_sha256": sha256_file(bootstrap_path),
                "checksum_match": True,
                "syntax_check": "pass",
            }
        )
        require(all(row["exists"] and row["checksum_match"] for row in extraction_rows), "payload extraction audit failed")

        probe = Path(temporary) / "environment_probe.py"
        probe.write_text(ENVIRONMENT_PROBE_TEXT, encoding="utf-8")
        builder = (
            payload
            / "repo"
            / "scripts"
            / "delphes"
            / "reconstruct_hh4b_candidates_v2.py"
        )
        writer = (
            payload
            / "repo"
            / "scripts"
            / "delphes"
            / "write_parquet_from_pickle.py"
        )
        environment_result = run_lcg_python(
            setup, probe, str(builder), str(writer)
        )
        require(environment_result.returncode == 0, f"LCG import preflight failed: {environment_result.stderr}")
        environment = json.loads(environment_result.stdout.strip().splitlines()[-1])
        for component in ("python", "numpy", "awkward", "uproot", "pandas", "pyarrow"):
            require(
                str(environment[component]) == str(config["environment"][component]),
                f"frozen environment version changed: {component}",
            )

        smoke_driver = Path(temporary) / "writer_smoke.py"
        smoke_driver.write_text(WRITER_SMOKE_TEXT, encoding="utf-8")
        smoke_result = run_lcg_python(setup, smoke_driver, str(writer), temporary)
        (runtime / "writer_smoke.stdout").write_text(
            smoke_result.stdout, encoding="utf-8"
        )
        (runtime / "writer_smoke.stderr").write_text(
            smoke_result.stderr, encoding="utf-8"
        )
        require(smoke_result.returncode == 0, f"writer smoke failed: {smoke_result.stderr}")
        smoke = json.loads(smoke_result.stdout.strip().splitlines()[-1])
        require(smoke["parquet_readable"], "synthetic Parquet unreadable")
        require(smoke["schema_preserved"], "synthetic schema changed")
        require(smoke["column_order_preserved"], "synthetic column order changed")

    environment_rows = [
        {
            "component": component,
            "expected": config["environment"].get(component, ""),
            "observed": environment.get(component, ""),
            "passed": (
                str(environment.get(component, ""))
                == str(config["environment"].get(component, ""))
            ),
            "evidence": config["environment"]["lcg_setup"],
        }
        for component in (
            "python",
            "numpy",
            "awkward",
            "uproot",
            "pandas",
            "parquet_backend",
            "pyarrow",
        )
    ]
    environment_rows.extend(
        [
            {
                "component": "container_image",
                "expected": "none",
                "observed": "none",
                "passed": True,
                "evidence": config["environment"]["container_note"],
            },
            {
                "component": "serial_uproot_bootstrap",
                "expected": "num_workers=1",
                "observed": "num_workers=1",
                "passed": True,
                "evidence": "payload/bootstrap/sitecustomize.py",
            },
            {
                "component": "local_python_module_dependencies",
                "expected": [],
                "observed": config["payload"]["local_python_modules_imported"],
                "passed": config["payload"]["local_python_modules_imported"] == [],
                "evidence": "static imports of protected builder and writer",
            },
            {
                "component": "payload_python_import_checks",
                "expected": [
                    "reconstruct_hh4b_candidates_v2.py",
                    "write_parquet_from_pickle.py",
                ],
                "observed": environment["imported_scripts"],
                "passed": environment["imported_scripts"]
                == [
                    "reconstruct_hh4b_candidates_v2.py",
                    "write_parquet_from_pickle.py",
                ],
                "evidence": config["environment"]["lcg_setup"],
            },
        ]
    )
    smoke_rows = [
        {
            "check": key,
            "expected": True if isinstance(value, bool) else value,
            "observed": value,
            "passed": value is True if isinstance(value, bool) else True,
            "environment": config["environment"]["lcg_setup"],
        }
        for key, value in (
            ("writer_exit_zero", smoke["writer_returncode"] == 0),
            ("synthetic_parquet_readable", smoke["parquet_readable"]),
            ("provided_schema_preserved", smoke["schema_preserved"]),
            ("provided_column_order_preserved", smoke["column_order_preserved"]),
            ("synthetic_rows", smoke["rows"]),
            ("synthetic_columns", smoke["columns"]),
            ("synthetic_types", smoke["types"]),
            ("physics_candidate_data_used", False),
        )
    ]
    smoke_rows[-1]["expected"] = False
    smoke_rows[-1]["passed"] = smoke_rows[-1]["observed"] is False
    return extraction_rows, environment, environment_rows, smoke_rows


def exact_reconstruction_command(config: Mapping[str, Any]) -> str:
    root = f"/srv/{config['returned_root']['basename']}"
    output = f"/srv/output/{config['outputs']['parquet']}"
    reconstruction = config["reconstruction"]
    return " ".join(
        [
            "env",
            "PYTHONPATH=/srv/payload/bootstrap",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            "VECLIB_MAXIMUM_THREADS=1",
            "PYTHONUNBUFFERED=1",
            "timeout",
            "--signal=TERM",
            "--kill-after=60s",
            "45m",
            "python3",
            "/srv/payload/repo/scripts/delphes/reconstruct_hh4b_candidates_v2.py",
            "--input",
            root,
            "--out",
            output,
            "--sample",
            reconstruction["sample"],
            "--target-mass",
            str(reconstruction["target_mass"]),
            "--jet-pt-min",
            str(reconstruction["jet_pt_min"]),
            "--jet-eta-max",
            str(reconstruction["jet_eta_max"]),
            "--btag-min",
            str(reconstruction["btag_min"]),
            "--max-bjets-for-pairing",
            str(reconstruction["max_bjets_for_pairing"]),
            "--higgs-ordering",
            reconstruction["higgs_ordering"],
        ]
    )


def render_submit(
    config: Mapping[str, Any],
    payload_path: Path,
) -> str:
    retry = config["retry"]
    paths = config["paths"]
    root = config["returned_root"]
    resources = config["resources"]
    wrapper = REPOSITORY_ROOT / "scripts/production/run_hh4b_ttbar8_canary_retry1.sh"
    arguments = f"{root['basename']} {root['sha256']} {MEMBER} {SEED}"
    lines = [
        "# INERT CONTRACT-ONLY reconstruction retry; do not submit in this gate.",
        "# Exactly one job consumes only the corrected payload and returned ROOT.",
        "universe = vanilla",
        f"executable = {wrapper}",
        f"arguments = {arguments}",
        f"request_cpus = {resources['request_cpus']}",
        f"request_memory = {resources['request_memory_mb']}MB",
        f"request_disk = {resources['request_disk_mb']}MB",
        f"+MaxRuntime = {resources['max_runtime_seconds']}",
        "max_retries = 0",
        "requirements = False",
        "hold = True",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        "preserve_relative_paths = True",
        f"transfer_input_files = {payload_path},{root['path']}",
        "transfer_output_files = output",
        f'transfer_output_remaps = "output={paths["return_directory"]}"',
        f"log = {paths['log_directory']}/retry1.$(Cluster).log",
        f"output = {paths['log_directory']}/retry1.$(Cluster).$(Process).out",
        f"error = {paths['log_directory']}/retry1.$(Cluster).$(Process).err",
        "notification = Never",
        "getenv = False",
        'environment = "LC_ALL=C LANG=C"',
        f'+JobBatchName = "{retry["campaign"]}"',
        f'+Campaign = "{retry["campaign"]}"',
        f'+RetryMember = "{MEMBER}"',
        f"+RetrySeedProvenance = {SEED}",
        f'+SourceJob = "{JOB_ID}"',
        f'+RetryMode = "{retry["mode"]}"',
        "+ReconstructionOnlyRetry = True",
        "+ScaleoutJob = False",
        "+RetryJobsSubmittedAtFreeze = 0",
        "+ScaleoutJobsSubmittedAtFreeze = 0",
        '+DesiredOS = "EL9"',
        "on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)",
        "",
        "queue 1",
        "",
    ]
    return "\n".join(lines)


def validate_submit_text(text: str, config: Mapping[str, Any]) -> None:
    queue = [
        line.strip()
        for line in text.splitlines()
        if line.strip().lower().startswith("queue")
    ]
    require(queue == ["queue 1"], f"unexpected queue contract: {queue}")
    require("requirements = False" in text and "hold = True" in text, "submit not inert")
    require("transfer_output_files = output" in text, "output directory not transferred")
    require(config["returned_root"]["path"] in text, "returned ROOT input absent")
    require(str(resolve_path(config["paths"]["payload"])) in text, "payload input absent")
    for forbidden in ("generate_events", "lhe_to_hepmc3", "DelphesHepMC3", "condor_release", "condor_rm"):
        require(forbidden not in text, f"forbidden command in submit: {forbidden}")
    for member in (
        "ttbar_100k_shard001",
        "ttbar_100k_shard002",
        "ttbar_100k_shard004",
        "ttbar_100k_shard005",
        "ttbar_100k_shard007",
        "ttbar_100k_shard008",
        "ttbar_100k_shard009",
    ):
        require(member not in text, f"scaleout member in submit: {member}")


def condor_dry_run_and_dump(submit_path: Path, runtime: Path) -> dict[str, Any]:
    executable = Path("/usr/bin/original_condor_submit")
    require(executable.is_file(), f"native condor_submit missing: {executable}")
    outputs = {}
    for mode, filename in (
        ("-dry-run", "retry1.dryrun.ads"),
        ("-dump", "retry1.dump.ads"),
    ):
        target = runtime / filename
        completed = subprocess.run(
            [str(executable), mode, str(target), str(submit_path)],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        (runtime / f"{filename}.stdout").write_text(completed.stdout, encoding="utf-8")
        (runtime / f"{filename}.stderr").write_text(completed.stderr, encoding="utf-8")
        require(completed.returncode == 0, f"Condor {mode} failed: {completed.stderr}")
        require(target.is_file() and target.stat().st_size > 0, f"empty Condor {mode}")
        text = target.read_text(encoding="utf-8", errors="replace")
        compact_lines = [
            line.replace(" ", "") for line in text.splitlines()
        ]
        job_ads = sum(line == 'MyType="Job"' for line in compact_lines)
        if not job_ads:
            job_ads = sum(
                line.startswith("JobStatus=") for line in compact_lines
            )
        outputs[mode] = {
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "classads": job_ads,
        }
    require(outputs["-dry-run"]["classads"] == 1, "dry-run did not emit one job ad")
    require(outputs["-dump"]["classads"] == 1, "dump did not emit one job ad")
    return outputs


def legacy_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    legacy = config["legacy_validation"]
    rows = [
        {
            "column": "sample",
            "role": "join_identity",
            "canonical_column": "sample",
            "comparison": "exact_after_remove_trailing__pythia8_delphes",
            "rtol": 0,
            "atol": 0,
            "exact_required": True,
            "nonfinite_mask_compared": False,
        },
        {
            "column": "event",
            "role": "join_key",
            "canonical_column": "event",
            "comparison": "integer_exact",
            "rtol": 0,
            "atol": 0,
            "exact_required": True,
            "nonfinite_mask_compared": False,
        },
        {
            "column": "n_selected_bjets",
            "role": "acceptance",
            "canonical_column": "n_selected_bjets",
            "comparison": "integer_exact",
            "rtol": 0,
            "atol": 0,
            "exact_required": True,
            "nonfinite_mask_compared": False,
        },
        {
            "column": "pairing",
            "role": "diagnostic_categorical",
            "canonical_column": "pairing",
            "comparison": "categorical_exact_with_predeclared_policy_adjudication",
            "rtol": 0,
            "atol": 0,
            "exact_required": False,
            "nonfinite_mask_compared": False,
        },
    ]
    rows.extend(
        {
            "column": column,
            "role": "diagnostic_float",
            "canonical_column": column,
            "comparison": "float_tolerance_and_nonfinite_mask",
            "rtol": legacy["floating_rtol"],
            "atol": legacy["floating_atol"],
            "exact_required": False,
            "nonfinite_mask_compared": True,
        }
        for column in legacy["floating"]
    )
    return rows


def build_tables(
    config: Mapping[str, Any],
    root_rows: list[dict[str, Any]],
    payload_rows: list[dict[str, Any]],
    extraction_rows: list[dict[str, Any]],
    smoke_rows: list[dict[str, Any]],
    environment_rows: list[dict[str, Any]],
    submit_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    retry = config["retry"]
    resources = config["resources"]
    command = exact_reconstruction_command(config)
    command_rows = [
        {
            "field": "exact_command",
            "frozen_value": command,
            "verification": "protected builder plus explicit frozen arguments",
        },
        {
            "field": "input",
            "frozen_value": f"/srv/{config['returned_root']['basename']}",
            "verification": config["returned_root"]["sha256"],
        },
        {
            "field": "policy",
            "frozen_value": config["reconstruction"]["policy_path"],
            "verification": config["reconstruction"]["policy_sha256"],
        },
        {
            "field": "serial_executor",
            "frozen_value": config["environment"]["uproot_mechanism"],
            "verification": "successful retained-ROOT mechanism",
        },
        {
            "field": "isolated_writer",
            "frozen_value": "/srv/payload/repo/scripts/delphes/write_parquet_from_pickle.py",
            "verification": next(
                item["sha256"]
                for item in config["payload"]["required_files"]
                if item["source"].endswith("write_parquet_from_pickle.py")
            ),
        },
    ]
    output_rows = [
        {
            "role": role,
            "relative_path": relative,
            "required_on_success": True,
            "preserved_on_failure": role != "canonical_parquet",
            "placeholder_forbidden": role == "canonical_parquet",
        }
        for role, relative in (
            ("canonical_parquet", config["outputs"]["parquet"]),
            ("reconstruction_stdout", config["outputs"]["reconstruction_stdout"]),
            ("reconstruction_stderr", config["outputs"]["reconstruction_stderr"]),
            ("canonical_reconstruction_log", config["outputs"]["canonical_log"]),
            ("environment_report", config["outputs"]["environment_report"]),
            ("receipt_json", config["outputs"]["receipt"]),
            ("final_status", config["outputs"]["final_status"]),
            ("sha256_manifest", config["outputs"]["checksums"]),
            ("schema_audit", config["outputs"]["schema_audit"]),
            ("entry_candidate_accounting", config["outputs"]["entry_candidate_accounting"]),
        )
    ]
    transfer_rows = [
        {
            "requirement": requirement,
            "frozen_value": value,
            "verification": verification,
        }
        for requirement, value, verification in (
            ("top_level_output_created_before_checks", True, "worker mkdir precedes payload/input checks"),
            ("shell_failure_mode", "set -Eeuo pipefail", "worker line 3"),
            ("final_status_always_written", True, "EXIT trap"),
            ("failure_receipt_always_written", True, "EXIT trap"),
            ("transfer_existing_directory", "output", "submit transfer_output_files"),
            ("mandatory_individual_parquet_transfer", False, "directory transfer"),
            ("placeholder_parquet", "forbidden", "wrapper never creates placeholder"),
            ("wrapper_nonzero_without_valid_parquet", True, "EXIT trap and schema audit"),
            ("LHE_HepMC_new_ROOT_transfer", "forbidden", "output registry"),
        )
    ]
    held_rows = [
        {
            "order": index,
            "action": action,
            "execute_in_this_gate": False,
            "required_before_next_action": "yes" if index < 6 else "terminal submission step",
        }
        for index, action in enumerate(
            config["held_job_later_gate_policy"]["ordered_steps"], start=1
        )
    ]
    submission_rows = [
        {
            "campaign": retry["campaign"],
            "member": MEMBER,
            "seed": SEED,
            "source_job": JOB_ID,
            "retry_mode": retry["mode"],
            "submit_file": submit_path,
            "jobs_described": 1,
            "retry_jobs_submitted": 0,
            "scaleout_jobs_submitted": 0,
            "requirements": "False",
            "initially_held": True,
            "transfer_inputs": [
                config["paths"]["payload"],
                config["returned_root"]["path"],
            ],
            "transfer_output": "output directory",
        }
    ]
    resource_rows = [
        {
            "request_cpus": resources["request_cpus"],
            "request_memory_mb": resources["request_memory_mb"],
            "request_disk_mb": resources["request_disk_mb"],
            "max_runtime_seconds": resources["max_runtime_seconds"],
            "successful_wall_seconds": resources["successful_reconstruction_wall_seconds"],
            "determinism_wall_seconds": resources["successful_determinism_rerun_wall_seconds"],
            "successful_input_bytes": resources["successful_reconstruction_input_bytes"],
            "current_input_bytes": resources["current_input_bytes"],
            "prior_peak_memory_mb": resources["prior_canary_peak_memory_mb"],
            "evidence_path": resources["evidence_path"],
            "evidence_sha256": resources["evidence_sha256"],
            "rationale": resources["rationale"],
        }
    ]
    pass_conditions = [
        ("returned_ROOT_SHA256_matches_frozen_input", "preparation_and_worker"),
        ("ROOT_entries_exactly_10000", "preparation_and_worker"),
        ("missing_writer_exact_path_present", "payload_dry_run_and_worker"),
        ("writer_synthetic_smoke_passes", "preparation"),
        ("canonical_reconstruction_exits_zero", "retry_execution"),
        ("canonical_Parquet_exists_nonempty", "retry_execution"),
        ("canonical_72_column_names_and_order_exact", "retry_execution"),
        ("Arrow_logical_types_compatible", "retry_execution"),
        ("required_values_finite", "retry_execution"),
        ("source_member_identity_exact", "retry_execution"),
        ("candidate_keys_unique", "retry_execution"),
        ("duplicate_candidate_keys_zero", "retry_execution"),
        ("candidate_row_count_recorded", "retry_execution"),
        ("receipt_complete", "retry_execution"),
        ("checksum_manifest_complete", "retry_execution"),
        ("legacy_comparison_passes_frozen_contract", "post_return_validation"),
    ]
    pass_rows = [
        {
            "condition": condition,
            "required": True,
            "contract_frozen": True,
            "verification_stage": stage,
            "failure_action": "nonzero_retry_or_block_scaleout_no_publish",
        }
        for condition, stage in pass_conditions
    ]
    return {
        "retry_returned_root_input_audit": root_rows,
        "retry_payload_manifest": payload_rows,
        "retry_payload_extraction_audit": extraction_rows,
        "retry_writer_smoke_test": smoke_rows,
        "retry_environment_contract": environment_rows,
        "retry_reconstruction_command": command_rows,
        "retry_expected_output_registry": output_rows,
        "retry_transfer_and_failure_policy": transfer_rows,
        "retry_legacy_comparison_contract": legacy_rows(config),
        "retry_held_job_removal_policy": held_rows,
        "retry_condor_submission_plan": submission_rows,
        "retry_resource_request": resource_rows,
        "retry_pass_conditions": pass_rows,
    }


def write_sha256sums(directory: Path) -> None:
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    text = "\n".join(
        f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}"
        for path in files
    )
    (directory / "SHA256SUMS").write_text(text + "\n", encoding="utf-8")


def prepare(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config)
    starting_state = validate_starting_state()
    checkpoint_counts = validate_authoritative_checkpoints(config)
    held = live_held_job()
    root_audit, root_rows = audit_returned_root(config)
    specs, before = protected_fingerprints(config)

    runtime = resolve_path(config["paths"]["runtime_directory"])
    checkpoint = resolve_path(config["paths"]["checkpoint_directory"])
    require(not runtime.exists(), f"runtime directory already exists: {runtime}")
    require(not checkpoint.exists(), f"checkpoint directory already exists: {checkpoint}")
    require(
        not Path(config["paths"]["return_directory"]).exists(),
        "unique retry return directory already exists",
    )
    runtime.mkdir(parents=True)
    checkpoint.mkdir(parents=True)

    payload_path, payload_rows = build_payload(config, runtime)
    (runtime / "payload_archive_members.txt").write_text(
        "\n".join(row["archive_path"] for row in payload_rows) + "\n",
        encoding="utf-8",
    )
    extraction_rows, environment, environment_rows, smoke_rows = (
        extract_and_smoke_test(config, payload_path, runtime)
    )

    submit_text = render_submit(config, payload_path)
    validate_submit_text(submit_text, config)
    submit_path = resolve_path(config["paths"]["submit_description"])
    submit_path.write_text(submit_text, encoding="utf-8")
    (checkpoint / SUBMIT_FILE_NAME).write_text(submit_text, encoding="utf-8")
    condor_outputs = condor_dry_run_and_dump(submit_path, runtime)

    tables = build_tables(
        config,
        root_rows,
        payload_rows,
        extraction_rows,
        smoke_rows,
        environment_rows,
        submit_path,
    )
    for name, rows in tables.items():
        write_table_bundle(checkpoint, name, rows)

    _, after = protected_fingerprints(config)
    require(before == after, "protected artifacts changed during preparation")
    protected_rows = [
        {
            "role": spec["role"],
            "path": spec["path"],
            "before_sha256": before[str(resolve_path(spec["path"]))],
            "after_sha256": after[str(resolve_path(spec["path"]))],
            "expected_sha256": spec["expected_sha256"],
            "unchanged": before[str(resolve_path(spec["path"]))] == after[str(resolve_path(spec["path"]))],
            "matches_expected": after[str(resolve_path(spec["path"]))] == spec["expected_sha256"],
        }
        for spec in specs
    ]
    write_table_bundle(checkpoint, "protected_artifact_audit", protected_rows)

    summary = {
        "schema_version": config["schema_version"],
        "contract_name": config["contract_name"],
        "status": FROZEN_STATUS,
        "next_gate": NEXT_GATE,
        "retry_campaign": config["retry"]["campaign"],
        "retry_member": MEMBER,
        "retry_seed_provenance": SEED,
        "retry_mode": config["retry"]["mode"],
        "source_job": JOB_ID,
        "source_job_status": "held",
        "returned_root": root_audit,
        "payload": {
            "path": str(payload_path),
            "bytes": payload_path.stat().st_size,
            "sha256": sha256_file(payload_path),
            "file_members": len(payload_rows),
            "missing_writer_path_present": True,
        },
        "preflight": {
            "environment": environment,
            "writer_smoke_passed": all(row["passed"] for row in smoke_rows),
            "required_payload_paths_verified": len(extraction_rows),
            "condor_dry_run_classads": condor_outputs["-dry-run"]["classads"],
            "condor_dump_classads": condor_outputs["-dump"]["classads"],
        },
        "counts": {
            "retry_jobs_described": 1,
            "retry_jobs_submitted": 0,
            "scaleout_jobs_submitted": 0,
        },
        "prohibitions_observed": {
            "source_job_released": False,
            "source_job_removed": False,
            "retry_submitted": False,
            "scaleout_submitted": False,
            "madgraph_rerun": False,
            "pythia_rerun": False,
            "delphes_rerun": False,
            "candidate_content_opened": False,
        },
    }
    checkpoint_record = {
        "checkpoint": checkpoint.name,
        "status": FROZEN_STATUS,
        "next_gate": NEXT_GATE,
        "authoritative_checkpoints": config["paths"],
        "retry_campaign": config["retry"]["campaign"],
        "source_job": JOB_ID,
        "member": MEMBER,
        "seed": SEED,
        "retry_jobs_submitted": 0,
        "scaleout_jobs_submitted": 0,
    }
    environment_record = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "preparation_python": sys.version,
        "frozen_retry_environment": environment,
        "container_image": "none",
        "lcg_setup": config["environment"]["lcg_setup"],
        "git": starting_state,
        "authoritative_checkpoint_files_verified": checkpoint_counts,
        "live_held_job": held,
    }
    (checkpoint / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (checkpoint / "checkpoint.json").write_text(
        json.dumps(checkpoint_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (checkpoint / "environment.json").write_text(
        json.dumps(environment_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = f"""# HH4b ttbar8 canary reconstruction retry1 contract

Status: `{FROZEN_STATUS}`.

This checkpoint freezes exactly one reconstruction-only retry for `{MEMBER}`
using only the immutable ROOT returned by held job `{JOB_ID}`. The ROOT SHA256
is `{config['returned_root']['sha256']}` and its `Delphes` tree has 10,000
entries. The corrected payload contains the previously omitted isolated
Parquet writer at the required `/srv/payload/repo/scripts/delphes/` path.

The payload was created, fully listed, extracted, checksum-verified,
syntax/import checked, and exercised with non-physics synthetic data under the
frozen LCG 106 environment. Both Condor dry-run and dump contain exactly one
inert job (`requirements=False`, `hold=True`). No retry or scaleout job was
submitted, and source job `{JOB_ID}` remains held.

The later submission gate must archive and remove the old held job exactly once
before submitting this single retry. `condor_release` is forbidden.
"""
    (checkpoint / "README.md").write_text(readme, encoding="utf-8")
    write_sha256sums(checkpoint)
    verified = verify_sha256sums(checkpoint)
    summary["checkpoint_files_verified"] = verified
    (runtime / "preparation.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "condor": condor_outputs,
                "checkpoint": str(checkpoint),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    print(json.dumps(prepare(arguments.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
