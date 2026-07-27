#!/usr/bin/env python3
"""Freeze the eight-member HH4b ttbar exact-regeneration production contract.

This is a metadata-only preparation gate.  It verifies protected provenance
and writes reviewable production descriptions; it never submits a job or opens
legacy candidate content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "production"
    / "hh4b_ttbar8_exact_regeneration_v1.yaml"
)

MISSING_TAGS = {
    "ttbar_100k_shard001",
    "ttbar_100k_shard002",
    "ttbar_100k_shard003",
    "ttbar_100k_shard004",
    "ttbar_100k_shard005",
    "ttbar_100k_shard007",
    "ttbar_100k_shard008",
    "ttbar_100k_shard009",
}

TABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "ttbar8_exact_member_inventory": (
        "member_index",
        "source_member_id",
        "original_member_name",
        "campaign",
        "dataset_split",
        "expected_generated_events",
        "exact_generator_seed",
        "exact_process_definition",
        "exact_run_card_overrides",
        "parameter_card_sha256",
        "generator_version",
        "pythia_version",
        "pythia_settings",
        "delphes_version",
        "delphes_card_path",
        "delphes_card_sha256",
        "expected_delphes_root_filename",
        "expected_candidate_filename",
        "legacy_15_column_summary_path",
        "legacy_candidate_rows",
        "common_column_comparison_fields",
        "exposure_class",
        "sealed_test",
        "normalization_metadata",
        "provenance_sources",
    ),
    "ttbar8_seed_and_card_contract": (
        "member_index",
        "original_member_name",
        "run_name",
        "seed",
        "generated_events",
        "process_definition",
        "process_card_path",
        "process_card_sha256",
        "run_card_path",
        "run_card_template_sha256",
        "run_card_overrides",
        "parameter_card_path",
        "parameter_card_sha256",
        "generator_version",
        "pythia_version",
        "pythia_settings",
        "delphes_version",
        "delphes_card_path",
        "delphes_card_sha256",
        "reconstruction_builder_sha256",
        "reconstruction_policy_sha256",
    ),
    "ttbar8_environment_contract": (
        "stage",
        "original_environment_available",
        "environment_classification",
        "host_environment_type",
        "operating_system_or_container",
        "compiler_runtime_dependencies",
        "lhapdf_configuration",
        "executable_paths",
        "setup_commands",
        "relevant_environment_variables",
        "equivalence_validation_required",
        "equivalence_validation",
        "provenance",
    ),
    "ttbar8_protected_artifact_inventory": (
        "artifact_role",
        "path",
        "expected_sha256",
        "observed_sha256",
        "exists",
        "executable_required",
        "executable",
        "unchanged",
        "provenance",
    ),
    "ttbar8_provenance_inventory": (
        "member_index",
        "original_member_name",
        "field",
        "frozen_value",
        "source_path",
        "source_locator",
        "verification",
    ),
    "ttbar8_expected_output_registry": (
        "member_index",
        "original_member_name",
        "seed",
        "lhe_filename",
        "hepmc_filename",
        "delphes_root_filename",
        "candidate_filename",
        "stageout_member_directory",
        "return_member_directory",
        "receipt_path",
        "checksum_manifest_path",
        "no_overwrite_policy",
        "publication_policy",
    ),
    "ttbar8_legacy_common_column_contract": (
        "column",
        "canonical_column",
        "role",
        "comparison_type",
        "rtol",
        "atol",
        "exact_required",
        "permitted_difference",
        "failure_policy",
    ),
    "ttbar8_exactness_validation_contract": (
        "level",
        "level_name",
        "validation",
        "required",
        "comparison",
        "pass_condition",
        "failure_action",
        "byte_identity_claimed",
    ),
    "ttbar8_canary_selection": (
        "rank",
        "member_index",
        "original_member_name",
        "seed",
        "selection_salt",
        "selection_digest",
        "selected_canary",
        "submitted",
    ),
    "ttbar8_canary_submission_plan": (
        "campaign",
        "submission_file",
        "jobs",
        "member_index",
        "original_member_name",
        "seed",
        "request_cpus",
        "request_memory_mb",
        "request_disk_mb",
        "max_runtime_seconds",
        "automatic_retries",
        "initially_held",
        "requirements_guard",
        "submitted",
        "unlock_gate",
    ),
    "ttbar8_scaleout_submission_plan": (
        "campaign",
        "submission_file",
        "job_order",
        "member_index",
        "original_member_name",
        "seed",
        "request_cpus",
        "request_memory_mb",
        "request_disk_mb",
        "max_runtime_seconds",
        "automatic_retries",
        "initially_held",
        "requirements_guard",
        "submitted",
        "submission_precondition",
    ),
    "ttbar8_resource_request": (
        "scope",
        "request_cpus",
        "request_memory_mb",
        "request_disk_mb",
        "max_runtime_seconds",
        "historical_jobs",
        "historical_total_wall_seconds",
        "historical_mean_wall_seconds_per_member",
        "historical_maximum_resident_mb",
        "historical_mean_write_gib_per_member",
        "historical_retained_root_gib",
        "evidence_path",
        "evidence_sha256",
        "rationale",
    ),
    "ttbar8_failure_policy": (
        "failure_class",
        "detection",
        "job_action",
        "retry_policy",
        "seed_policy",
        "publication_policy",
        "scaleout_effect",
    ),
    "ttbar8_normalization_readiness": (
        "member_index",
        "original_member_name",
        "generated_events",
        "event_summary_path",
        "candidate_regeneration_ready",
        "candidate_regeneration_blockers",
        "physics_normalization_ready",
        "physical_normalization_blockers",
        "physics_yield_authorized",
    ),
    "unresolved_exact_regeneration_inputs": (
        "input",
        "scope",
        "status",
        "blocking",
        "required_resolution",
        "evidence",
    ),
    "protected_artifact_audit": (
        "artifact_role",
        "path",
        "before_sha256",
        "after_sha256",
        "expected_sha256",
        "unchanged",
        "matches_expected",
        "status",
    ),
}

LEGACY_COLUMNS = (
    "sample",
    "event",
    "n_selected_bjets",
    "mbb1",
    "mbb2",
    "avg_mbb",
    "delta_mbb",
    "mhh",
    "drbb1",
    "drbb2",
    "pairing",
    "j1_pt",
    "j2_pt",
    "j3_pt",
    "j4_pt",
)


class ContractError(RuntimeError):
    """Raised when the exact-regeneration contract cannot be frozen."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def process_directory_manifest_sha256(path: Path) -> str:
    """Match `find ... | sort -z | xargs -0 sha256sum | sha256sum`."""
    lines: list[str] = []
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path)
        if not item.is_file() or item.is_symlink():
            continue
        if relative.parts and relative.parts[0] in {"Events", "HTML"}:
            continue
        if item.name.endswith(".log"):
            continue
        lines.append(f"{sha256_file(item)}  ./{relative.as_posix()}\n")
    return sha256_text("".join(lines))


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    require(isinstance(config, dict), "configuration is not a mapping")
    return config


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def verify_sha256sums(directory: Path) -> int:
    sums = directory / "SHA256SUMS"
    require(sums.is_file(), f"missing checkpoint SHA256SUMS: {sums}")
    checked = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.strip()
        target = directory / relative
        require(target.is_file(), f"checkpoint file missing: {target}")
        require(
            sha256_file(target) == expected,
            f"checkpoint checksum mismatch: {target}",
        )
        checked += 1
    return checked


def required_start_is_ancestor(commit: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def validate_config(config: Mapping[str, Any]) -> None:
    status = config["status_contract"]
    campaign = config["campaign"]
    require(status["expected_members"] == 8, "expected member count must be eight")
    require(status["expected_legacy_rows"] == 307, "legacy row count must be 307")
    require(status["sealed_test_members"] == 0, "sealed-test count must be zero")
    require(campaign["jobs"] == 8, "campaign must contain eight jobs")
    require(campaign["members_per_job"] == 1, "one member per job is required")
    require(campaign["canary_jobs"] == 1, "canary must contain one job")
    require(campaign["scaleout_jobs"] == 7, "scaleout must contain seven jobs")
    require(campaign["automatic_retries"] == 0, "automatic retries are forbidden")
    require(not campaign["changed_seed_retries_allowed"], "seed changes are forbidden")
    require(not campaign["submit_in_this_gate"], "this gate must not submit")
    require(config["generator"]["event_count_per_member"] == 10000, "event count")
    require(config["reconstruction"]["schema_columns"] == 72, "schema is not 72")
    require(not config["normalization"]["physics_yield_authorized"], "yield enabled")
    require(len(config["exactness_levels"]) == 4, "four exactness levels required")
    require(
        [int(row["level"]) for row in config["exactness_levels"]] == [1, 2, 3, 4],
        "exactness levels must be 1 through 4",
    )


def load_source_members(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = resolve_path(config["paths"]["source_checkpoint"])
    checked = verify_sha256sums(source)
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    require(
        source_summary["status"] == config["status_contract"]["source_status"],
        "source checkpoint status changed",
    )
    unresolved = read_tsv(source / "unresolved_ttbar_members.tsv")
    holds = read_tsv(source / "ttbar27_hold_inventory.tsv")
    normalization = read_tsv(source / "ttbar27_normalization_readiness.tsv")
    source_search = read_tsv(source / "ttbar27_source_search.tsv")
    hold_by_tag = {row["target_tag"]: row for row in holds}
    normalization_by_tag = {row["target_tag"]: row for row in normalization}
    search_by_tag = {row["target_tag"]: row for row in source_search}
    require(len(unresolved) == 8, "source unresolved inventory is not exactly eight")
    require(
        {row["target_tag"] for row in unresolved} == MISSING_TAGS,
        "source unresolved member identities changed",
    )
    members: list[dict[str, Any]] = []
    for row in unresolved:
        tag = row["target_tag"]
        require(tag in hold_by_tag, f"hold inventory missing {tag}")
        require(tag in normalization_by_tag, f"normalization inventory missing {tag}")
        require(tag in search_by_tag, f"source search missing {tag}")
        hold = hold_by_tag[tag]
        search = search_by_tag[tag]
        require(not as_bool(hold["sealed_test"]), f"sealed-test conflict for {tag}")
        require(as_bool(search["generation_seed_log_match"]), f"seed log mismatch: {tag}")
        require(as_bool(search["pythia_8312_log_match"]), f"Pythia log mismatch: {tag}")
        require(
            as_bool(search["delphes_completion_log_match"]),
            f"Delphes log mismatch: {tag}",
        )
        require(int(row["seed"]) == int(hold["job_seed"]), f"seed mismatch: {tag}")
        require(
            int(row["generated_events"]) == int(hold["generated_events"]) == 10000,
            f"event-count mismatch: {tag}",
        )
        members.append(
            {
                **row,
                "source_member_id": hold["source_member_id"],
                "campaign": hold["campaign"],
                "legacy_candidate_path": hold["legacy_candidate_path"],
                "legacy_schema_sha256": hold["legacy_schema_sha256"],
                "event_summary_path": normalization_by_tag[tag]["event_summary_path"],
                "generator_log": search["generator_log"],
                "pipeline_log": search["pipeline_log"],
                "delphes_log": (
                    f"/uscms_data/d3/iturkmen/hh4b_delphes/logs/"
                    f"{tag}_pythia8_delphes.log"
                ),
                "sealed_test": False,
            }
        )
    members.sort(key=lambda item: int(item["member_index"]))
    require(
        len({item["source_member_id"] for item in members}) == 8,
        "source member IDs are not unique",
    )
    require(len({int(item["seed"]) for item in members}) == 8, "seeds are not unique")
    require(
        sum(int(item["legacy_candidate_rows"]) for item in members) == 307,
        "legacy row total is not 307",
    )
    return members, {"checkpoint_files_verified": checked, "summary": source_summary}


def protected_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    generator = config["generator"]
    pythia = config["pythia"]
    delphes = config["delphes"]
    reconstruction = config["reconstruction"]
    scripts = config["production_scripts"]
    environment = config["environment"]
    schema = config["paths"]["canonical72_schema"]
    specs = [
        ("generator_process_card", generator["process_card"], False),
        ("generator_parameter_card", generator["parameter_card"], False),
        ("generator_run_card_template", generator["run_card_template"], False),
        ("generator_me5_configuration", generator["me5_configuration"], False),
        ("exact_regeneration_worker", scripts["worker_entrypoint"], True),
        ("production_campaign_wrapper", scripts["campaign_wrapper"], True),
        ("production_local_member_wrapper", scripts["local_member_wrapper"], True),
        ("pythia_delphes_wrapper", scripts["shower_delphes_wrapper"], True),
        ("pythia_converter", pythia["converter"], True),
        ("delphes_executable", delphes["executable"], True),
        ("delphes_card", delphes["card"], False),
        ("canonical_reconstruction_builder", reconstruction["builder"], True),
        ("canonical_reconstruction_policy", reconstruction["policy"], False),
        ("lcg_environment_setup", environment["lcg_view"], False),
        (
            "madgraph_cvmfs_configuration",
            environment["madgraph_cvmfs_configuration"],
            False,
        ),
        ("canonical72_schema", schema, False),
    ]
    return [
        {
            "artifact_role": role,
            "path": str(spec["path"]),
            "expected_sha256": str(spec["sha256"]),
            "executable_required": executable,
        }
        for role, spec, executable in specs
    ]


def audit_protected(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    for spec in protected_specs(config):
        path = resolve_path(spec["path"])
        exists = path.is_file()
        observed = sha256_file(path) if exists else ""
        executable = os.access(path, os.X_OK) if exists else False
        unchanged = exists and observed == spec["expected_sha256"]
        require(unchanged, f"protected artifact changed: {path}")
        if spec["executable_required"]:
            require(executable, f"protected executable bit missing: {path}")
        fingerprints[str(path)] = observed
        rows.append(
            {
                **spec,
                "observed_sha256": observed,
                "exists": exists,
                "executable": executable,
                "unchanged": unchanged,
                "provenance": (
                    "this exact-regeneration contract"
                    if spec["artifact_role"] == "exact_regeneration_worker"
                    else config["paths"]["source_checkpoint"]
                ),
            }
        )
    process_path = Path(config["generator"]["process_directory"])
    expected_manifest = config["generator"]["process_directory_observed_manifest_sha256"]
    observed_manifest = process_directory_manifest_sha256(process_path)
    require(observed_manifest == expected_manifest, "MG5 process directory changed")
    fingerprints[str(process_path)] = observed_manifest
    rows.insert(
        0,
        {
            "artifact_role": "generator_process_directory_manifest",
            "path": str(process_path),
            "expected_sha256": expected_manifest,
            "observed_sha256": observed_manifest,
            "exists": process_path.is_dir(),
            "executable_required": False,
            "executable": False,
            "unchanged": observed_manifest == expected_manifest,
            "provenance": "observed immutable manifest excluding Events, HTML, and logs",
        },
    )
    return rows, fingerprints


def deterministic_canary(
    config: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    salt = config["campaign"]["canary_selection_salt"]
    ranked: list[tuple[str, Mapping[str, Any]]] = []
    for member in members:
        digest = sha256_text(f"{salt}|{member['source_member_id']}")
        ranked.append((digest, member))
    ranked.sort(key=lambda item: item[0])
    rows = []
    for rank, (digest, member) in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": rank,
                "member_index": int(member["member_index"]),
                "original_member_name": member["target_tag"],
                "seed": int(member["seed"]),
                "selection_salt": salt,
                "selection_digest": digest,
                "selected_canary": rank == 1,
                "submitted": False,
            }
        )
    return dict(ranked[0][1]), rows


def json_cell(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return "True" if value else "False"
    return "" if value is None else str(value)


def latex_escape(value: Any) -> str:
    text = json_cell(value)
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
    return "".join(replacements.get(char, char) for char in text)


def write_table_bundle(
    directory: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = TABLE_FIELDS[name]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{name}.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json_cell(row.get(field, "")) for field in fields})
    md_lines = [
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in rows:
        values = [json_cell(row.get(field, "")).replace("|", r"\|") for field in fields]
        md_lines.append("| " + " | ".join(values) + " |")
    (directory / f"{name}.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    align = "l" * len(fields)
    latex_lines = [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(field) for field in fields) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        latex_lines.append(
            " & ".join(latex_escape(row.get(field, "")) for field in fields)
            + r" \\"
        )
    latex_lines.extend((r"\bottomrule", r"\end{tabular}"))
    (directory / f"{name}.tex").write_text(
        "\n".join(latex_lines) + "\n", encoding="utf-8"
    )


def member_inventory_rows(
    config: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    generator = config["generator"]
    pythia = config["pythia"]
    delphes = config["delphes"]
    rows = []
    for member in members:
        tag = member["target_tag"]
        overrides = dict(generator["run_card_overrides"])
        overrides["iseed"] = int(member["seed"])
        rows.append(
            {
                "member_index": int(member["member_index"]),
                "source_member_id": member["source_member_id"],
                "original_member_name": tag,
                "campaign": member["campaign"],
                "dataset_split": member["dataset_split"],
                "expected_generated_events": int(member["generated_events"]),
                "exact_generator_seed": int(member["seed"]),
                "exact_process_definition": generator["process"],
                "exact_run_card_overrides": overrides,
                "parameter_card_sha256": generator["parameter_card"]["sha256"],
                "generator_version": generator["version"],
                "pythia_version": pythia["version"],
                "pythia_settings": pythia["settings"],
                "delphes_version": delphes["version"],
                "delphes_card_path": delphes["card"]["path"],
                "delphes_card_sha256": delphes["card"]["sha256"],
                "expected_delphes_root_filename": f"{tag}_pythia8_delphes.root",
                "expected_candidate_filename": (
                    f"{tag}_pythia8_delphes_canonical72.parquet"
                ),
                "legacy_15_column_summary_path": member["legacy_candidate_path"],
                "legacy_candidate_rows": int(member["legacy_candidate_rows"]),
                "common_column_comparison_fields": list(LEGACY_COLUMNS),
                "exposure_class": f"development_{member['dataset_split']}_non_test",
                "sealed_test": False,
                "normalization_metadata": {
                    "event_summary_path": member["event_summary_path"],
                    "physics_weight_status": config["normalization"][
                        "physics_weight_status"
                    ],
                    "physics_yield_authorized": False,
                },
                "provenance_sources": [
                    f"{config['paths']['source_checkpoint']}/"
                    "unresolved_ttbar_members.tsv",
                    f"{config['paths']['source_checkpoint']}/ttbar27_hold_inventory.tsv",
                    member["generator_log"],
                    member["pipeline_log"],
                    member["delphes_log"],
                ],
            }
        )
    return rows


def seed_card_rows(
    config: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    generator = config["generator"]
    pythia = config["pythia"]
    delphes = config["delphes"]
    reconstruction = config["reconstruction"]
    rows = []
    for member in members:
        overrides = dict(generator["run_card_overrides"])
        overrides["iseed"] = int(member["seed"])
        shard = member["target_tag"].removeprefix("ttbar_100k_shard")
        rows.append(
            {
                "member_index": int(member["member_index"]),
                "original_member_name": member["target_tag"],
                "run_name": f"run_ttbar_100k_{shard}",
                "seed": int(member["seed"]),
                "generated_events": int(member["generated_events"]),
                "process_definition": generator["process_definition_detail"],
                "process_card_path": generator["process_card"]["path"],
                "process_card_sha256": generator["process_card"]["sha256"],
                "run_card_path": generator["run_card_template"]["path"],
                "run_card_template_sha256": generator["run_card_template"]["sha256"],
                "run_card_overrides": overrides,
                "parameter_card_path": generator["parameter_card"]["path"],
                "parameter_card_sha256": generator["parameter_card"]["sha256"],
                "generator_version": generator["version"],
                "pythia_version": pythia["version"],
                "pythia_settings": pythia["settings"],
                "delphes_version": delphes["version"],
                "delphes_card_path": delphes["card"]["path"],
                "delphes_card_sha256": delphes["card"]["sha256"],
                "reconstruction_builder_sha256": reconstruction["builder"]["sha256"],
                "reconstruction_policy_sha256": reconstruction["policy"]["sha256"],
            }
        )
    return rows


def environment_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    environment = config["environment"]
    common = {
        "original_environment_available": environment[
            "original_environment_available"
        ],
        "environment_classification": environment["required_classification"],
        "host_environment_type": environment["host_type"],
        "operating_system_or_container": environment["operating_system_contract"],
        "compiler_runtime_dependencies": environment["compiler_runtime"],
        "lhapdf_configuration": environment["lhapdf"],
        "setup_commands": environment["setup_commands"],
        "relevant_environment_variables": environment["required_variables"],
        "equivalence_validation_required": True,
        "equivalence_validation": environment["equivalence_validation"],
        "provenance": (
            "source checkpoint production config and original timing logs; "
            "original production had no container"
        ),
    }
    return [
        {
            **common,
            "stage": "MadGraph_generation",
            "executable_paths": [
                config["generator"]["process_directory"] + "/bin/generate_events",
                environment["madgraph_cvmfs_configuration"]["path"],
            ],
        },
        {
            **common,
            "stage": "Pythia_showering_hadronization",
            "executable_paths": [config["pythia"]["converter"]["path"]],
        },
        {
            **common,
            "stage": "Delphes_simulation",
            "executable_paths": [
                config["delphes"]["executable"]["path"],
                config["delphes"]["card"]["path"],
            ],
        },
        {
            **common,
            "stage": "canonical_candidate_reconstruction",
            "executable_paths": [
                config["reconstruction"]["builder"]["path"],
                config["reconstruction"]["policy"]["path"],
            ],
        },
    ]


def provenance_rows(
    config: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source = config["paths"]["source_checkpoint"]
    generator = config["generator"]
    shared = [
        ("process_definition", generator["process"], config["paths"]["source_config"]),
        ("generator_version", generator["version"], config["paths"]["source_config"]),
        (
            "process_card_sha256",
            generator["process_card"]["sha256"],
            generator["process_card"]["path"],
        ),
        (
            "parameter_card_sha256",
            generator["parameter_card"]["sha256"],
            generator["parameter_card"]["path"],
        ),
        (
            "run_card_template_sha256",
            generator["run_card_template"]["sha256"],
            generator["run_card_template"]["path"],
        ),
        ("pythia_version", config["pythia"]["version"], config["paths"]["source_config"]),
        (
            "pythia_settings",
            config["pythia"]["settings"],
            config["production_scripts"]["shower_delphes_wrapper"]["path"],
        ),
        ("delphes_version", config["delphes"]["version"], config["paths"]["source_config"]),
        (
            "delphes_card_sha256",
            config["delphes"]["card"]["sha256"],
            config["delphes"]["card"]["path"],
        ),
        (
            "reconstruction_builder_sha256",
            config["reconstruction"]["builder"]["sha256"],
            config["reconstruction"]["builder"]["path"],
        ),
        (
            "reconstruction_policy_sha256",
            config["reconstruction"]["policy"]["sha256"],
            config["reconstruction"]["policy"]["path"],
        ),
    ]
    rows = []
    for member in members:
        member_fields = [
            ("source_member_id", member["source_member_id"], f"{source}/ttbar27_hold_inventory.tsv"),
            ("member_name", member["target_tag"], f"{source}/unresolved_ttbar_members.tsv"),
            ("seed", int(member["seed"]), member["generator_log"]),
            ("event_count", int(member["generated_events"]), member["generator_log"]),
            ("legacy_rows", int(member["legacy_candidate_rows"]), f"{source}/unresolved_ttbar_members.tsv"),
            ("sealed_test", False, f"{source}/ttbar27_hold_inventory.tsv"),
        ]
        for field, value, source_path in [*member_fields, *shared]:
            rows.append(
                {
                    "member_index": int(member["member_index"]),
                    "original_member_name": member["target_tag"],
                    "field": field,
                    "frozen_value": value,
                    "source_path": source_path,
                    "source_locator": "full_file_or_member_row",
                    "verification": "checksum_or_exact_value_verified",
                }
            )
    return rows


def output_registry_rows(
    config: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    campaign = config["campaign"]
    rows = []
    for member in members:
        tag = member["target_tag"]
        stageout = f"{campaign['stageout_base']}/members/{tag}"
        returned = f"{campaign['return_base']}/members/{tag}"
        rows.append(
            {
                "member_index": int(member["member_index"]),
                "original_member_name": tag,
                "seed": int(member["seed"]),
                "lhe_filename": f"{tag}_unweighted_events.lhe.gz",
                "hepmc_filename": f"{tag}_pythia8.hepmc",
                "delphes_root_filename": f"{tag}_pythia8_delphes.root",
                "candidate_filename": f"{tag}_pythia8_delphes_canonical72.parquet",
                "stageout_member_directory": stageout,
                "return_member_directory": returned,
                "receipt_path": f"{returned}/receipts/{tag}_receipt.json",
                "checksum_manifest_path": f"{returned}/checksums/{tag}_SHA256SUMS",
                "no_overwrite_policy": campaign["no_overwrite"][
                    "existing_target_action"
                ],
                "publication_policy": campaign["no_overwrite"]["publication"],
            }
        )
    return rows


def legacy_contract_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    legacy = config["legacy_validation"]
    integer = set(legacy["integer_exact"])
    floating = set(legacy["floating_columns"])
    rows = []
    for column in LEGACY_COLUMNS:
        if column == "sample":
            role = "joining_key"
            comparison = "categorical_after_suffix_normalization"
            exact = True
            permitted = "none"
        elif column == "event":
            role = "joining_key"
            comparison = "integer_exact"
            exact = True
            permitted = "none"
        elif column in integer:
            role = "acceptance"
            comparison = "integer_exact"
            exact = True
            permitted = "none"
        elif column in floating:
            role = "diagnostic_physics_comparison"
            comparison = "floating_at_declared_tolerance"
            exact = False
            permitted = legacy["permitted_differences"]
        else:
            role = "diagnostic_categorical_comparison"
            comparison = "categorical"
            exact = False
            permitted = legacy["permitted_differences"]
        rows.append(
            {
                "column": column,
                "canonical_column": column,
                "role": role,
                "comparison_type": comparison,
                "rtol": legacy["floating_rtol"] if column in floating else 0,
                "atol": legacy["floating_atol"] if column in floating else 0,
                "exact_required": exact,
                "permitted_difference": permitted,
                "failure_policy": legacy["failure_policy"],
            }
        )
    return rows


def validation_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    validations = {
        1: [
            ("generated_event_count", "integer_exact", "exactly 10000"),
            ("requested_seed", "integer_exact", "receipt and log equal frozen seed"),
            ("generator_completion", "log_and_exit", "exit zero and completion marker"),
            ("pythia_completion", "log_and_exit", "exit zero and Pythia 8.312 marker"),
            ("delphes_completion", "log_and_exit", "exit zero and Delphes marker"),
            ("root_entry_count", "integer_exact", "Delphes entries exactly 10000"),
            ("root_readability", "structural", "ROOT and Delphes tree readable"),
            ("canonical72_schema", "schema_exact", "72 columns in frozen order"),
            ("required_finite_values", "numeric", "all required values finite"),
            ("unique_event_keys", "key_exact", "no duplicate sample/event keys"),
            ("source_member_identity", "categorical_exact", "member identity exact"),
            ("output_checksums", "sha256", "all artifacts in receipt"),
        ],
        2: [
            (
                "generator_event_identity",
                "stable_event_comparator",
                "event identity where stable comparison is available",
            )
        ],
        3: [
            (
                "delphes_branch_event_identity",
                "stable_branch_comparator",
                "branch identity where stable comparison is available",
            )
        ],
        4: [
            (
                "legacy_common_columns",
                "predeclared_15_column_contract",
                "row/key/exact fields pass and differences are explained",
            ),
            (
                "cross_member_duplicate_keys",
                "aggregate_key_exact",
                "zero duplicates across regenerated members",
            ),
        ],
    }
    names = {int(row["level"]): row for row in config["exactness_levels"]}
    rows = []
    for level, checks in validations.items():
        for validation, comparison, condition in checks:
            rows.append(
                {
                    "level": level,
                    "level_name": names[level]["name"],
                    "validation": validation,
                    "required": names[level]["required"],
                    "comparison": comparison,
                    "pass_condition": condition,
                    "failure_action": "fail_member_block_scaleout_no_publish",
                    "byte_identity_claimed": False,
                }
            )
    return rows


def submission_rows(
    config: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    canary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    campaign = config["campaign"]
    resources = config["resources"]
    common = {
        "campaign": campaign["name"],
        "request_cpus": resources["request_cpus"],
        "request_memory_mb": resources["request_memory_mb"],
        "request_disk_mb": resources["request_disk_mb"],
        "max_runtime_seconds": resources["max_runtime_seconds"],
        "automatic_retries": campaign["automatic_retries"],
        "initially_held": True,
        "requirements_guard": "False_until_explicit_next_gate_authorization",
        "submitted": False,
    }
    canary_rows = [
        {
            **common,
            "submission_file": "hh4b_ttbar8_canary.sub",
            "jobs": 1,
            "member_index": int(canary["member_index"]),
            "original_member_name": canary["target_tag"],
            "seed": int(canary["seed"]),
            "unlock_gate": config["status_contract"]["frozen_next_gate"],
        }
    ]
    remaining = [row for row in members if row["target_tag"] != canary["target_tag"]]
    scaleout_rows = []
    for order, member in enumerate(remaining, start=1):
        scaleout_rows.append(
            {
                **common,
                "submission_file": "hh4b_ttbar8_scaleout.sub",
                "job_order": order,
                "member_index": int(member["member_index"]),
                "original_member_name": member["target_tag"],
                "seed": int(member["seed"]),
                "submission_precondition": (
                    "single canary passes every frozen canary requirement; "
                    "manual authorization required"
                ),
            }
        )
    require(len(canary_rows) == 1 and len(scaleout_rows) == 7, "submission split")
    return canary_rows, scaleout_rows


def resource_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    resources = config["resources"]
    evidence = resources["historical_evidence"]
    mean_wall = float(evidence["total_wall_seconds"]) / int(evidence["jobs"])
    peak_mb = float(evidence["maximum_resident_kb"]) / 1024.0
    mean_write_gib = (
        float(evidence["filesystem_output_blocks"]) * 512.0
        / int(evidence["jobs"])
        / 1024.0**3
    )
    root_gib = float(evidence["retained_root_bytes"]) / 1024.0**3
    return [
        {
            "scope": "per_member_job",
            "request_cpus": resources["request_cpus"],
            "request_memory_mb": resources["request_memory_mb"],
            "request_disk_mb": resources["request_disk_mb"],
            "max_runtime_seconds": resources["max_runtime_seconds"],
            "historical_jobs": evidence["jobs"],
            "historical_total_wall_seconds": evidence["total_wall_seconds"],
            "historical_mean_wall_seconds_per_member": round(mean_wall, 3),
            "historical_maximum_resident_mb": round(peak_mb, 3),
            "historical_mean_write_gib_per_member": round(mean_write_gib, 3),
            "historical_retained_root_gib": round(root_gib, 3),
            "evidence_path": evidence["path"],
            "evidence_sha256": evidence["sha256"],
            "rationale": resources["rationale"],
        }
    ]


def failure_rows() -> list[dict[str, Any]]:
    policies = [
        ("input_checksum_mismatch", "preflight SHA256", "exit before generation"),
        ("existing_output", "exclusive target preflight", "exit before generation"),
        ("generator_failure", "nonzero exit or missing completion", "hold for review"),
        ("event_count_mismatch", "LHE event count", "hold for review"),
        ("pythia_failure", "nonzero exit or missing marker", "hold for review"),
        ("delphes_failure", "nonzero exit or missing marker", "hold for review"),
        ("root_validation_failure", "readability or entries", "hold for review"),
        ("reconstruction_failure", "nonzero exit or schema/content check", "hold"),
        ("identity_disagreement", "exactness levels 1-4", "hold and adjudicate"),
        ("transfer_failure", "size/SHA256 mismatch", "hold and retain partial logs"),
        ("condor_held_job", "HoldReason/ClassAd", "manual inspection only"),
    ]
    rows = []
    for failure, detection, action in policies:
        rows.append(
            {
                "failure_class": failure,
                "detection": detection,
                "job_action": action,
                "retry_policy": "no_automatic_retry",
                "seed_policy": "seed_never_changes",
                "publication_policy": "no_canonical_publication_on_failure",
                "scaleout_effect": "block_scaleout_if_canary_or_unexplained_physics",
            }
        )
    return rows


def normalization_rows(
    config: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalization = config["normalization"]
    return [
        {
            "member_index": int(member["member_index"]),
            "original_member_name": member["target_tag"],
            "generated_events": int(member["generated_events"]),
            "event_summary_path": member["event_summary_path"],
            "candidate_regeneration_ready": False,
            "candidate_regeneration_blockers": [
                normalization["candidate_regeneration_blocker"]
            ],
            "physics_normalization_ready": False,
            "physical_normalization_blockers": normalization[
                "physical_normalization_blockers"
            ],
            "physics_yield_authorized": False,
        }
        for member in members
    ]


def protected_audit_rows(
    protected: Sequence[Mapping[str, Any]],
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for item in protected:
        path = str(resolve_path(item["path"]))
        before_sha = before[path]
        after_sha = after[path]
        expected = item["expected_sha256"]
        rows.append(
            {
                "artifact_role": item["artifact_role"],
                "path": item["path"],
                "before_sha256": before_sha,
                "after_sha256": after_sha,
                "expected_sha256": expected,
                "unchanged": before_sha == after_sha,
                "matches_expected": after_sha == expected,
                "status": "verified",
            }
        )
    return rows


def _copy_payload_file(source: Path, target: Path) -> None:
    require(source.is_file(), f"payload source missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _write_payload_sha256sums(payload: Path) -> None:
    files = sorted(
        path
        for path in payload.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(payload).as_posix()}"
        for path in files
    ]
    (payload / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _create_reproducible_tar(source: Path, destination: Path) -> None:
    import gzip

    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", dereference=False) as archive:
                paths = [source, *sorted(source.rglob("*"))]
                for path in paths:
                    arcname = Path("payload") / path.relative_to(source)
                    info = archive.gettarinfo(str(path), arcname.as_posix())
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    if path.is_file() and not path.is_symlink():
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
                    else:
                        archive.addfile(info)
    os.replace(temporary, destination)


def build_transfer_payload(
    config: Mapping[str, Any],
    runtime: Path,
) -> dict[str, Any]:
    runtime.mkdir(parents=True, exist_ok=True)
    output = resolve_path(config["campaign"]["input_payload"])
    require(output.parent == runtime, "input payload must be under the runtime directory")
    generator = config["generator"]
    delphes = config["delphes"]
    reconstruction = config["reconstruction"]
    with tempfile.TemporaryDirectory(
        prefix="ttbar8_payload_stage.", dir=runtime
    ) as temporary:
        payload = Path(temporary) / "payload"
        payload.mkdir()
        shutil.copytree(
            Path(generator["process_directory"]),
            payload / "mg5_template",
            symlinks=True,
            ignore=shutil.ignore_patterns("Events", "HTML", "*.log"),
        )
        repo_payload = payload / "repo"
        _copy_payload_file(
            resolve_path(delphes["card"]["path"]),
            repo_payload / "cards" / "delphes" / "delphes_card_CMS_lpc.tcl",
        )
        _copy_payload_file(
            resolve_path(reconstruction["builder"]["path"]),
            repo_payload
            / "scripts"
            / "delphes"
            / "reconstruct_hh4b_candidates_v2.py",
        )
        _copy_payload_file(
            resolve_path(reconstruction["policy"]["path"]),
            repo_payload
            / "configs"
            / "production"
            / "hh4b_rich_v2_reconstruction_policy_v1.yaml",
        )
        _copy_payload_file(
            resolve_path(config["paths"]["canonical72_schema"]["path"]),
            repo_payload / "schema" / "canonical72_columns.tsv",
        )
        _copy_payload_file(
            resolve_path(config["pythia"]["converter"]["path"]),
            payload / "software" / "bin" / "lhe_to_hepmc3",
        )
        delphes_source = Path(delphes["executable"]["path"]).parent
        for name in (
            "DelphesHepMC3",
            "libDelphes.so",
            "ClassesDict_rdict.pcm",
            "ExRootAnalysisDict_rdict.pcm",
            "ModulesDict_rdict.pcm",
            "ModulesFastJetDict_rdict.pcm",
            "DelphesEnv.sh",
        ):
            _copy_payload_file(delphes_source / name, payload / "Delphes" / name)
        manifest = {
            "campaign": config["campaign"]["name"],
            "contract_config": str(
                DEFAULT_CONFIG.relative_to(REPOSITORY_ROOT)
            ),
            "generator_process_directory_manifest_sha256": generator[
                "process_directory_observed_manifest_sha256"
            ],
            "member_count": config["campaign"]["jobs"],
            "physics_inputs": {
                "process_card_sha256": generator["process_card"]["sha256"],
                "parameter_card_sha256": generator["parameter_card"]["sha256"],
                "run_card_template_sha256": generator["run_card_template"]["sha256"],
                "pythia_converter_sha256": config["pythia"]["converter"]["sha256"],
                "delphes_executable_sha256": delphes["executable"]["sha256"],
                "delphes_card_sha256": delphes["card"]["sha256"],
                "reconstruction_builder_sha256": reconstruction["builder"]["sha256"],
                "reconstruction_policy_sha256": reconstruction["policy"]["sha256"],
            },
        }
        (payload / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_payload_sha256sums(payload)
        _create_reproducible_tar(payload, output)
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "contents_checksum_manifest": "payload/SHA256SUMS",
    }


def render_submit(
    name: str,
    plan: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> str:
    resources = config["resources"]
    lines = [
        "# CONTRACT-ONLY HTCondor description.",
        "# This gate forbids submission; requirements=False and hold=True are safety locks.",
        "universe = vanilla",
        "executable = scripts/production/run_hh4b_ttbar8_exact_regeneration_member.sh",
        "arguments = $(member_index) $(member_name) $(seed) 10000",
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
        f"transfer_input_files = {config['campaign']['input_payload']}",
        f"log = condor_logs/{name}.$(Cluster).log",
        f"output = condor_logs/{name}.$(Cluster).$(Process).out",
        f"error = condor_logs/{name}.$(Cluster).$(Process).err",
        "",
        "queue member_index, member_name, seed from (",
    ]
    for row in plan:
        lines.append(
            f"{row['member_index']} {row['original_member_name']} {row['seed']}"
        )
    lines.extend((")", ""))
    return "\n".join(lines)


def write_sha256sums(directory: Path) -> None:
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}" for path in files
    ]
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config)
    require(
        required_start_is_ancestor(config["required_starting_commit"]),
        "required starting commit is not an ancestor of HEAD",
    )
    members, source_evidence = load_source_members(config)
    protected, before = audit_protected(config)
    canonical_v1 = resolve_path(config["paths"]["canonical_v1_checkpoint"])
    require(
        sha256_file(canonical_v1 / "SHA256SUMS")
        == config["paths"]["canonical_v1_sha256sums_sha256"],
        "canonical-v1 SHA256SUMS changed",
    )
    canonical_v1_verified = verify_sha256sums(canonical_v1)
    canary, canary_selection = deterministic_canary(config, members)
    inventory = member_inventory_rows(config, members)
    canary_plan, scaleout_plan = submission_rows(config, members, canary)
    tables: dict[str, list[dict[str, Any]]] = {
        "ttbar8_exact_member_inventory": inventory,
        "ttbar8_seed_and_card_contract": seed_card_rows(config, members),
        "ttbar8_environment_contract": environment_rows(config),
        "ttbar8_protected_artifact_inventory": protected,
        "ttbar8_provenance_inventory": provenance_rows(config, members),
        "ttbar8_expected_output_registry": output_registry_rows(config, members),
        "ttbar8_legacy_common_column_contract": legacy_contract_rows(config),
        "ttbar8_exactness_validation_contract": validation_rows(config),
        "ttbar8_canary_selection": canary_selection,
        "ttbar8_canary_submission_plan": canary_plan,
        "ttbar8_scaleout_submission_plan": scaleout_plan,
        "ttbar8_resource_request": resource_rows(config),
        "ttbar8_failure_policy": failure_rows(),
        "ttbar8_normalization_readiness": normalization_rows(config, members),
        "unresolved_exact_regeneration_inputs": [],
    }
    runtime = resolve_path(config["paths"]["runtime_directory"])
    checkpoint = resolve_path(config["paths"]["checkpoint_directory"])
    runtime.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    payload_contract = build_transfer_payload(config, runtime)
    for name, rows in tables.items():
        write_table_bundle(checkpoint, name, rows)
    canary_sub = render_submit("hh4b_ttbar8_canary", canary_plan, config)
    scaleout_sub = render_submit("hh4b_ttbar8_scaleout", scaleout_plan, config)
    for directory in (runtime, checkpoint):
        (directory / "hh4b_ttbar8_canary.sub").write_text(
            canary_sub, encoding="utf-8"
        )
        (directory / "hh4b_ttbar8_scaleout.sub").write_text(
            scaleout_sub, encoding="utf-8"
        )
    _, after = audit_protected(config)
    require(before == after, "protected artifacts changed during preparation")
    tables["protected_artifact_audit"] = protected_audit_rows(
        protected, before, after
    )
    write_table_bundle(
        checkpoint,
        "protected_artifact_audit",
        tables["protected_artifact_audit"],
    )
    summary = {
        "contract_name": config["contract_name"],
        "schema_version": config["schema_version"],
        "base_commit": config["required_starting_commit"],
        "campaign": config["campaign"]["name"],
        "status": config["status_contract"]["frozen_status"],
        "next_gate": config["status_contract"]["frozen_next_gate"],
        "inventory": {
            "exact_unresolved_members": len(members),
            "exact_member_ids_frozen": len({row["source_member_id"] for row in inventory}),
            "unique_seeds_frozen": len({row["exact_generator_seed"] for row in inventory}),
            "generated_event_counts_frozen": len(inventory),
            "legacy_candidate_rows_validation_only": sum(
                row["legacy_candidate_rows"] for row in inventory
            ),
            "sealed_test_members": sum(row["sealed_test"] for row in inventory),
        },
        "contracts": {
            "complete_generator_contracts": len(tables["ttbar8_seed_and_card_contract"]),
            "complete_pythia_contracts": len(tables["ttbar8_seed_and_card_contract"]),
            "complete_delphes_contracts": len(tables["ttbar8_seed_and_card_contract"]),
            "complete_candidate_reconstruction_contracts": len(
                tables["ttbar8_seed_and_card_contract"]
            ),
            "unresolved_exact_regeneration_inputs": len(
                tables["unresolved_exact_regeneration_inputs"]
            ),
            "original_environment_available": config["environment"][
                "original_environment_available"
            ],
            "validated_equivalent_environment_required": True,
            "byte_identical_root_claimed": False,
        },
        "canary": {
            "selection_rule": config["campaign"]["canary_selection_rule"],
            "member_index": int(canary["member_index"]),
            "member_name": canary["target_tag"],
            "seed": int(canary["seed"]),
            "jobs_described": 1,
            "jobs_submitted": 0,
        },
        "scaleout": {
            "jobs_described": 7,
            "jobs_submitted": 0,
            "manual_authorization_required_after_canary": True,
        },
        "controls": {
            "sealed_candidate_files_opened": 0,
            "models_trained": 0,
            "predictions_produced": 0,
            "physical_yields_calculated": 0,
            "significance_calculated": 0,
            "expected_limits_calculated": 0,
            "event_level_products_committed": 0,
            "protected_artifacts_changed": 0,
            "canonical_v1_artifacts_changed": 0,
            "canary_jobs_submitted": 0,
            "scaleout_jobs_submitted": 0,
        },
        "integrity": {
            "source_checkpoint_files_verified": source_evidence[
                "checkpoint_files_verified"
            ],
            "protected_artifacts_verified": len(protected),
            "canonical_v1_artifacts_verified": canonical_v1_verified,
            "principal_tables_have_markdown_and_booktabs_latex": True,
            "transfer_payload_prepared": True,
            "transfer_payload_sha256": payload_contract["sha256"],
        },
        "normalization": {
            "metadata_carried_forward_for_members": len(members),
            "candidate_regeneration_blockers_separate": True,
            "physical_normalization_blockers_separate": True,
            "normalization_ready_members": 0,
        },
    }
    (checkpoint / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint_record = {
        "checkpoint": checkpoint.name,
        "status": summary["status"],
        "next_gate": summary["next_gate"],
        "authoritative_source_checkpoint": config["paths"]["source_checkpoint"],
        "config": str(config_path.relative_to(REPOSITORY_ROOT)),
        "preparation_script": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
        "submission_descriptions_are_safety_locked": True,
        "transfer_payload": payload_contract,
    }
    (checkpoint / "checkpoint.json").write_text(
        json.dumps(checkpoint_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environment_record = {
        "host_type": config["environment"]["host_type"],
        "operating_system_contract": config["environment"][
            "operating_system_contract"
        ],
        "original_environment_available": False,
        "classification": config["environment"]["required_classification"],
        "lcg_view": config["environment"]["lcg_view"],
        "runtime_observation": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "equivalence_validation": config["environment"]["equivalence_validation"],
    }
    (checkpoint / "environment.json").write_text(
        json.dumps(environment_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = f"""# HH4b ttbar8 exact-regeneration contract

Status: `{summary['status']}`

Next gate: `{summary['next_gate']}`

This checkpoint freezes the complete provenance and production contract for the
eight missing original `ttbar_100k` source members.  It describes one
deterministically selected canary ({canary['target_tag']}, seed
{canary['seed']}) and a seven-member scaleout.  Both HTCondor descriptions are
safety-locked with `requirements = False` and `hold = True`; no jobs were
submitted in this gate.

The original production was not containerized, so the environment is classified
as `validated_equivalent_environment`, not as an available exact original
environment.  No regenerated output may be called exact until the canary passes
the frozen equivalence and exactness checks.  Byte-identical ROOT output is not
claimed.

The 307 legacy 15-column rows are validation-only.  They are not canonical-v2
inputs; only newly reconstructed, validated canonical-72 rows may enter v2.
No sealed candidate content, model training, predictions, yields, significance,
or limits were accessed or produced.
"""
    (checkpoint / "README.md").write_text(readme, encoding="utf-8")
    runtime_record = {
        "campaign": config["campaign"]["name"],
        "checkpoint": str(checkpoint),
        "canary_member": canary["target_tag"],
        "canary_seed": int(canary["seed"]),
        "canary_submitted": False,
        "scaleout_submitted": False,
        "contract_only": True,
        "transfer_payload": payload_contract,
    }
    (runtime / "preparation_manifest.json").write_text(
        json.dumps(runtime_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_sha256sums(checkpoint)
    verify_sha256sums(checkpoint)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate configuration, source inventory, and protected artifacts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.validate_only:
        validate_config(config)
        load_source_members(config)
        audit_protected(config)
        print("hh4b ttbar8 exact-regeneration inputs: valid")
        return 0
    summary = prepare(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
