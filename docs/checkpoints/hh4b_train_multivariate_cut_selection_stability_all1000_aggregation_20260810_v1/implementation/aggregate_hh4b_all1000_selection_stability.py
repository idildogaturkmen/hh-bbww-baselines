#!/usr/bin/env python3
"""Deterministically aggregate frozen HH4b selection stability over all 1000 replicas.

This program implements only the predeclared train-side selection-stability
protocol.  It never reads validation or test payloads and never evaluates outer
fold physics results.  The sole scientific input is the normalized, audited
initial-200 inventory plus the audited, sharded escalation-800 inventories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_INITIAL_INVENTORY_SHA256 = "47df422f39a50ac2f008031d3222ef04556760526abf160db57485861f3db056"
EXPECTED_INITIAL_AUDIT_JSON_SHA256 = "1e0984279f9e35c33acb9534681508d38bba0c88726ad8b78a1c4ca98d22b876"
EXPECTED_PROTOCOL_SHA256 = "a856d9027cc1d2e11a9d60e72b4104b5b8c24ce61acf43def0d4d487ab867eea"
EXPECTED_AUDIT_FREEZE_SHA256 = "9c3b937968def05917ff280ad4438ecba591fdde9f2251e04401d988af4e52df"
EXPECTED_INITIAL_SUMMARY_SHA256 = "449300786a15517d6783a9df34936bc6f9fc346bcc1825f8d5ea34b5ceb73d77"
TARGET_SIGNAL_EFFICIENCY = 0.585957
CATEGORIES = ("exact3tag", "ge4tag")
OUTER_FOLDS = (0, 1, 2, 3, 4)
REPLICAS = tuple(range(1000))
REPLICA_COUNT = 1000
FOLD_WINNERS_PER_CATEGORY = REPLICA_COUNT * 5
REPLICA_CATEGORIES_PER_CATEGORY = REPLICA_COUNT
EXPECTED_ALL1000_ROWS = REPLICA_COUNT * 2 * 5 * 27
RANKED_REPLICAS_PER_SHARD = 100
RANKED_ROWS_PER_REPLICA = 2 * 5 * 27
OPTIONAL_VARIABLES = (
    "abs_h_delta_eta",
    "h2_pt",
    "ht_candidate_jets",
    "max_drbb",
    "mhh",
)
NOMINAL_STRUCTURES = {
    "exact3tag": "radial_mass__category__plus_ht_candidate_jets",
    "ge4tag": "radial_mass__category__plus_mhh_and_abs_h_delta_eta",
}
NOMINAL_THRESHOLDS = {
    "exact3tag": {
        "ht_candidate_jets": 176.5458068847656,
        "r_hh_125_125": 36.40814019639858,
    },
    "ge4tag": {
        "abs_h_delta_eta": 6.904302164473993,
        "mhh": 164.73708096689654,
        "r_hh_125_125": 33.92808917804956,
    },
}
EXPECTED_STRUCTURE_IDS = (
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


class AggregationError(RuntimeError):
    """Fail-closed aggregation contract violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregation_input_label(repo: Path, path: Path) -> str:
    if path.is_relative_to(repo):
        return str(path.relative_to(repo))
    return f"external_escalation_complete_return_audit/{path.name}"


def aggregation_input_sha256s(repo: Path, paths: Sequence[Path]) -> dict[str, str]:
    labels = [aggregation_input_label(repo, path) for path in paths]
    require(len(labels) == len(set(labels)), "aggregation input labels are not unique")
    return {label: sha256_file(path) for label, path in zip(labels, paths)}


def verify_bound_external_artifacts(
    source_root: Path,
    bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Path]:
    require(source_root.is_dir() and not source_root.is_symlink(), "external audit source is invalid")
    require(bool(bindings), "external audit source bindings are empty")
    actual = {path.name for path in source_root.iterdir() if path.is_file()}
    require(actual == set(bindings), "external audit source file set changed")
    verified: dict[str, Path] = {}
    for name, binding in sorted(bindings.items()):
        path = source_root / name
        require(path.is_file() and not path.is_symlink(), f"invalid external audit artifact: {name}")
        require(path.stat().st_size == binding.get("bytes"), f"external audit artifact size changed: {name}")
        require(sha256_file(path) == binding.get("sha256"), f"external audit artifact SHA changed: {name}")
        verified[name] = path
    return verified


def canonical_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value, pretty=True), encoding="utf-8")


def parse_bool(value: str, field: str) -> bool:
    require(value in {"True", "False"}, f"invalid {field}: {value!r}")
    return value == "True"


def parse_int(value: str, field: str) -> int:
    require(value.isdigit(), f"invalid {field}: {value!r}")
    return int(value)


def parse_float(value: str, field: str) -> float:
    parsed = float(value)
    require(math.isfinite(parsed), f"non-finite {field}: {value!r}")
    return parsed


def tsv_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        require(math.isfinite(value), "attempted to serialize a non-finite float")
        return repr(value)
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(value)
    return str(value)


def write_tsv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: tsv_scalar(row[field]) for field in fieldnames})


@dataclass(frozen=True)
class Candidate:
    job_index: int
    replica: int
    outer_fold: int
    category_id: str
    payload_key: str
    structure_id: str
    category_family_id: str
    cut_count: int
    signal_efficiency: float
    background_efficiency: float
    feasible: bool
    all_inner_support_pass: bool
    pooled_support_pass: bool
    refit_thresholds: Mapping[str, float]
    canonical_payload_sha256: str
    structure_result_sha256: str
    inner_crossfit_sha256: str
    execution_provenance_sha256: str
    bundle_sha256: str


@dataclass(frozen=True)
class RankedCandidate:
    candidate: Candidate
    rank: int
    ranking_key: tuple[Any, ...]


@dataclass(frozen=True)
class ReplicaCategoryResult:
    replica: int
    category_id: str
    winners: tuple[Candidate, ...]
    winner_counts: Mapping[str, int]
    maximum_winner_count: int
    maximum_winner_frequency: float
    unique_modal_family: bool
    tie_resolution_applied: bool
    tied_structure_ids: tuple[str, ...]
    selected_structure_id: str
    supporting_outer_folds: tuple[int, ...]
    median_thresholds: Mapping[str, float]


def selection_ranking_key(candidate: Candidate) -> tuple[Any, ...]:
    """Return the exact predeclared fold-level ranking tuple."""

    if candidate.feasible:
        return (
            0,
            candidate.background_efficiency,
            candidate.signal_efficiency - TARGET_SIGNAL_EFFICIENCY,
            candidate.cut_count,
            candidate.structure_id,
        )
    return (
        1,
        -candidate.signal_efficiency,
        candidate.background_efficiency,
        candidate.cut_count,
        candidate.structure_id,
    )


def rank_fold_candidates(candidates: Sequence[Candidate]) -> tuple[RankedCandidate, ...]:
    require(len(candidates) == 27, f"fold has {len(candidates)} candidates, expected 27")
    require(
        {candidate.structure_id for candidate in candidates} == set(EXPECTED_STRUCTURE_IDS),
        "fold does not contain the exact frozen 27-structure set",
    )
    ordered = sorted(candidates, key=selection_ranking_key)
    return tuple(
        RankedCandidate(candidate=candidate, rank=index, ranking_key=selection_ranking_key(candidate))
        for index, candidate in enumerate(ordered, start=1)
    )


def coordinatewise_median(thresholds: Sequence[Mapping[str, float]]) -> dict[str, float]:
    require(bool(thresholds), "cannot reduce an empty threshold collection")
    keys = set(thresholds[0])
    require(bool(keys), "threshold object is empty")
    require(all(set(item) == keys for item in thresholds), "threshold coordinate mismatch")
    return {
        key: float(statistics.median(item[key] for item in thresholds))
        for key in sorted(keys)
    }


def reduce_replica_category(winners: Sequence[Candidate]) -> ReplicaCategoryResult:
    require(len(winners) == 5, f"replica/category has {len(winners)} winners, expected 5")
    ordered = tuple(sorted(winners, key=lambda candidate: candidate.outer_fold))
    require(tuple(candidate.outer_fold for candidate in ordered) == OUTER_FOLDS, "outer-fold coverage mismatch")
    require(len({candidate.replica for candidate in ordered}) == 1, "replica mismatch in reduction")
    require(len({candidate.category_id for candidate in ordered}) == 1, "category mismatch in reduction")

    counts = Counter(candidate.structure_id for candidate in ordered)
    maximum_count = max(counts.values())
    tied = tuple(sorted(structure_id for structure_id, count in counts.items() if count == maximum_count))
    selected = tied[0]
    supporting = tuple(candidate.outer_fold for candidate in ordered if candidate.structure_id == selected)
    selected_thresholds = [candidate.refit_thresholds for candidate in ordered if candidate.structure_id == selected]

    return ReplicaCategoryResult(
        replica=ordered[0].replica,
        category_id=ordered[0].category_id,
        winners=ordered,
        winner_counts=dict(sorted(counts.items())),
        maximum_winner_count=maximum_count,
        maximum_winner_frequency=maximum_count / 5.0,
        unique_modal_family=len(tied) == 1,
        tie_resolution_applied=len(tied) > 1,
        tied_structure_ids=tied,
        selected_structure_id=selected,
        supporting_outer_folds=supporting,
        median_thresholds=coordinatewise_median(selected_thresholds),
    )


def optional_variables(structure_id: str) -> frozenset[str]:
    require(structure_id in EXPECTED_STRUCTURE_IDS, f"unknown structure ID: {structure_id}")
    suffix = structure_id.split("__category__", 1)[1]
    if suffix == "mass_only":
        return frozenset()
    require(suffix.startswith("plus_"), f"invalid structure suffix: {suffix}")
    payload = suffix.removeprefix("plus_")
    variables: set[str] = set()
    for variable in OPTIONAL_VARIABLES:
        if payload == variable or payload.startswith(f"{variable}_and_") or payload.endswith(f"_and_{variable}"):
            variables.add(variable)
    require(variables, f"could not decode optional variables from {structure_id}")
    return frozenset(variables)


def linear_quantile(values: Sequence[float], probability: float) -> float:
    require(bool(values), "cannot calculate a quantile of an empty sequence")
    require(0.0 <= probability <= 1.0, "quantile probability out of range")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def expected_job_identity(job_index: int, replica_base: int) -> tuple[int, int, str, str]:
    require(job_index >= 0, f"job index out of range: {job_index}")
    replica = replica_base + job_index // 10
    within_replica = job_index % 10
    if within_replica < 5:
        category = "exact3tag"
        outer_fold = within_replica
    else:
        category = "ge4tag"
        outer_fold = within_replica - 5
    return replica, outer_fold, category, f"replica_{replica:04d}__{category}"


def load_candidates(
    inventory_sources: Sequence[tuple[Path, int]],
    expected_feasible_count: int,
    expected_support_pass_count: int,
    expected_replicas: Sequence[int] = REPLICAS,
    expected_row_count: int = EXPECTED_ALL1000_ROWS,
) -> dict[tuple[int, str, int], list[Candidate]]:
    expected_header = [
        "job_index", "replica", "outer_fold", "category_id", "payload_key", "structure_id",
        "category_family_id", "cut_count", "pooled_signal_efficiency",
        "pooled_background_efficiency", "pooled_inner_oof_feasible", "all_inner_support_pass",
        "pooled_support_pass", "pooled_support_failures_json", "refit_thresholds_json",
        "canonical_payload_sha256", "structure_result_sha256", "inner_crossfit_sha256",
        "execution_provenance_sha256", "runtime_seconds", "bundle_sha256", "bundle_member",
    ]
    groups: dict[tuple[int, str, int], list[Candidate]] = defaultdict(list)
    row_count = 0
    feasible_count = 0
    support_pass_count = 0

    for inventory_path, replica_base in inventory_sources:
        with inventory_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require(reader.fieldnames == expected_header, f"inventory header mismatch: {inventory_path}")
            for row in reader:
                require(None not in row and all(value is not None for value in row.values()), "malformed inventory row")
                job_index = parse_int(row["job_index"], "job_index")
                replica = parse_int(row["replica"], "replica")
                outer_fold = parse_int(row["outer_fold"], "outer_fold")
                category = row["category_id"]
                expected_replica, expected_fold, expected_category, expected_payload = expected_job_identity(
                    job_index, replica_base
                )
                require(
                    (replica, outer_fold, category, row["payload_key"])
                    == (expected_replica, expected_fold, expected_category, expected_payload),
                    f"job mapping mismatch for source job {job_index} in {inventory_path}",
                )
                structure_id = row["structure_id"]
                require(structure_id in EXPECTED_STRUCTURE_IDS, f"unexpected structure ID: {structure_id}")
                require(row["category_family_id"] == f"{category}::{structure_id}", "category family mismatch")
                feasible = parse_bool(row["pooled_inner_oof_feasible"], "feasible")
                signal_efficiency = parse_float(row["pooled_signal_efficiency"], "signal efficiency")
                background_efficiency = parse_float(row["pooled_background_efficiency"], "background efficiency")
                require(0.0 <= signal_efficiency <= 1.0, "signal efficiency out of range")
                require(0.0 <= background_efficiency <= 1.0, "background efficiency out of range")
                if feasible:
                    require(
                        signal_efficiency >= TARGET_SIGNAL_EFFICIENCY,
                        f"feasible result below target at replica {replica}: {signal_efficiency}",
                    )
                else:
                    require(
                        signal_efficiency < TARGET_SIGNAL_EFFICIENCY,
                        f"infeasible result at/above target at replica {replica}: {signal_efficiency}",
                    )
                all_inner_support = parse_bool(row["all_inner_support_pass"], "all-inner support")
                pooled_support = parse_bool(row["pooled_support_pass"], "pooled support")
                support_failures = json.loads(row["pooled_support_failures_json"])
                require(
                    isinstance(support_failures, list)
                    and all(isinstance(value, str) for value in support_failures)
                    and (not pooled_support or support_failures == []),
                    "support diagnostics are malformed",
                )
                thresholds_raw = json.loads(row["refit_thresholds_json"])
                require(isinstance(thresholds_raw, dict) and thresholds_raw, "invalid refit threshold object")
                thresholds = {str(key): float(value) for key, value in thresholds_raw.items()}
                require(all(math.isfinite(value) for value in thresholds.values()), "non-finite refit threshold")
                cut_count = parse_int(row["cut_count"], "cut_count")
                require(len(thresholds) == cut_count, "cut-count/threshold-coordinate mismatch")
                for field in (
                    "canonical_payload_sha256", "structure_result_sha256", "inner_crossfit_sha256",
                    "execution_provenance_sha256", "bundle_sha256",
                ):
                    require(len(row[field]) == 64 and all(char in "0123456789abcdef" for char in row[field]),
                            f"invalid SHA-256 in {field}")

                within_replica = job_index % 10
                global_job_index = replica * 10 + within_replica
                candidate = Candidate(
                    job_index=global_job_index,
                    replica=replica,
                    outer_fold=outer_fold,
                    category_id=category,
                    payload_key=row["payload_key"],
                    structure_id=structure_id,
                    category_family_id=row["category_family_id"],
                    cut_count=cut_count,
                    signal_efficiency=signal_efficiency,
                    background_efficiency=background_efficiency,
                    feasible=feasible,
                    all_inner_support_pass=all_inner_support,
                    pooled_support_pass=pooled_support,
                    refit_thresholds=dict(sorted(thresholds.items())),
                    canonical_payload_sha256=row["canonical_payload_sha256"],
                    structure_result_sha256=row["structure_result_sha256"],
                    inner_crossfit_sha256=row["inner_crossfit_sha256"],
                    execution_provenance_sha256=row["execution_provenance_sha256"],
                    bundle_sha256=row["bundle_sha256"],
                )
                groups[(replica, category, outer_fold)].append(candidate)
                row_count += 1
                feasible_count += int(feasible)
                support_pass_count += int(pooled_support)

    require(row_count == expected_row_count, f"inventory row count is {row_count}, expected {expected_row_count}")
    require(feasible_count == expected_feasible_count, f"feasible count mismatch: {feasible_count}")
    require(support_pass_count == expected_support_pass_count, f"support-pass count mismatch: {support_pass_count}")
    expected_groups = {
        (replica, category, fold)
        for replica in expected_replicas
        for category in CATEGORIES
        for fold in OUTER_FOLDS
    }
    require(set(groups) == expected_groups, "replica/category/fold coverage mismatch")
    return groups


def validate_environment(
    repo: Path,
    initial_checkpoint: Path,
    escalation_checkpoint: Path,
    escalation_audit_output: Path,
    protocol: Path,
    initial_summary_path: Path,
) -> tuple[str, list[tuple[Path, int]], int, int, list[Path]]:
    require(repo.resolve() == repo, f"repository path must be canonical: {repo}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", "origin/delphes-hh4b-production"], cwd=repo, text=True
    ).strip()
    require(head == remote, f"local/remote HEAD mismatch: {head} != {remote}")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    require(branch == "delphes-hh4b-production", f"unexpected branch: {branch}")

    inventory = initial_checkpoint / "structure_result_inventory.tsv"
    audit_json = initial_checkpoint / "complete_return_audit.json"
    audit_freeze = initial_checkpoint / "complete_return_audit_freeze.json"
    expected_hashes = {
        inventory: EXPECTED_INITIAL_INVENTORY_SHA256,
        audit_json: EXPECTED_INITIAL_AUDIT_JSON_SHA256,
        audit_freeze: EXPECTED_AUDIT_FREEZE_SHA256,
        protocol: EXPECTED_PROTOCOL_SHA256,
        initial_summary_path: EXPECTED_INITIAL_SUMMARY_SHA256,
    }
    for path, expected in expected_hashes.items():
        require(path.is_file() and not path.is_symlink(), f"missing/non-regular input: {path}")
        actual = sha256_file(path)
        require(actual == expected, f"SHA-256 mismatch for {path}: {actual}")

    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    require(audit.get("status") == "pass_complete_initial200_2000_job_54000_structure_return_audit",
            "return-audit status mismatch")
    require(audit.get("jobs_audited") == 2000 and audit.get("structure_results_audited") == 54000,
            "return-audit counts mismatch")
    require(audit.get("aggregation_performed") is False, "input audit says aggregation already occurred")
    require(audit.get("fold_level_winners_selected") is False, "input audit says winners already selected")
    require(audit.get("replica_level_stability_statistics_computed") is False,
            "input audit says replica statistics already computed")
    require(audit.get("validation_payloads_opened") == 0 and audit.get("test_payloads_opened") == 0,
            "sealed-data count mismatch")
    initial_diagnostics = audit["scientific_outcome_diagnostics"]

    require(
        escalation_checkpoint.is_dir()
        and not escalation_checkpoint.is_symlink()
        and (escalation_checkpoint / "SHA256SUMS").is_file(),
        f"invalid escalation return-audit checkpoint: {escalation_checkpoint}",
    )
    checksum = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=escalation_checkpoint,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(
        checksum.returncode == 0,
        f"escalation checkpoint checksum failure: {checksum.stdout}{checksum.stderr}",
    )
    escalation_audit_path = escalation_checkpoint / "complete_return_audit.json"
    escalation_manifest_path = escalation_checkpoint / "structure_result_inventory_manifest.json"
    escalation_freeze_path = escalation_checkpoint / "complete_return_audit_freeze.json"
    escalation_audit = json.loads(escalation_audit_path.read_text(encoding="utf-8"))
    escalation_freeze = json.loads(escalation_freeze_path.read_text(encoding="utf-8"))
    require(
        escalation_freeze.get("status")
        == "pass_escalation800_complete_return_audit_frozen_before_all1000_aggregation",
        "escalation complete-return freeze status mismatch",
    )
    require(escalation_freeze.get("large_source_artifacts_bound_by_sha256_not_copied") is True,
            "escalation compact-source binding gate changed")
    require(
        escalation_freeze.get("queue_rows") == 0
        and escalation_freeze.get("clean_history_rows") == 8000
        and escalation_freeze.get("bad_history_rows") == 0
        and escalation_freeze.get("complete_triplets") == 8000
        and escalation_freeze.get("partial_triplets") == 0,
        "escalation frozen scheduler/return counts changed",
    )
    external_artifacts = verify_bound_external_artifacts(
        escalation_audit_output,
        escalation_freeze.get("source_artifacts", {}),
    )
    require(
        escalation_audit.get("status")
        == "pass_complete_escalation800_8000_job_216000_structure_return_audit",
        "escalation return-audit status mismatch",
    )
    require(
        escalation_audit.get("jobs_audited") == 8000
        and escalation_audit.get("structure_results_audited") == 216000,
        "escalation return-audit counts mismatch",
    )
    for key in (
        "aggregation_performed",
        "fold_level_winners_selected",
        "replica_level_stability_statistics_computed",
        "pilot_results_enter_stability_aggregation",
        "nominal_selection_changed",
    ):
        require(escalation_audit.get(key) is False, f"escalation audit false gate failed: {key}")
    require(
        escalation_audit.get("validation_payloads_opened") == 0
        and escalation_audit.get("test_payloads_opened") == 0,
        "escalation audit sealed-data count mismatch",
    )
    inventory_manifest = json.loads(escalation_manifest_path.read_text(encoding="utf-8"))
    require(
        inventory_manifest.get("status")
        == "pass_escalation800_structure_result_inventory_sharded"
        and inventory_manifest.get("total_rows_excluding_headers") == 216000
        and inventory_manifest.get("shard_count") == 8,
        "escalation structure inventory manifest mismatch",
    )
    inventory_sources: list[tuple[Path, int]] = [(inventory, 0)]
    initial_summary = json.loads(initial_summary_path.read_text(encoding="utf-8"))
    require(
        initial_summary.get("status")
        == "pass_initial200_predeclared_selection_stability_aggregation"
        and initial_summary.get("validation_payloads_opened") == 0
        and initial_summary.get("test_payloads_opened") == 0,
        "frozen initial-200 aggregation summary mismatch",
    )
    input_paths = [
        inventory,
        audit_json,
        audit_freeze,
        protocol,
        initial_summary_path,
        escalation_audit_path,
        escalation_manifest_path,
        escalation_freeze_path,
    ]
    for shard in inventory_manifest["shards"]:
        path = external_artifacts[shard["path"]]
        require(path.is_file() and not path.is_symlink(), f"missing escalation inventory shard: {path}")
        require(path.stat().st_size == shard["bytes"], f"escalation inventory shard size mismatch: {path}")
        require(sha256_file(path) == shard["sha256"], f"escalation inventory shard SHA mismatch: {path}")
        require(shard["rows_excluding_header"] == 27000, f"escalation inventory shard row mismatch: {path}")
        inventory_sources.append((path, 200))
        input_paths.append(path)
    require(len(inventory_sources) == 9, "combined inventory source count mismatch")

    escalation_diagnostics = escalation_audit["scientific_outcome_diagnostics"]
    expected_feasible = int(initial_diagnostics["pooled_inner_oof_feasible_structure_results"]) + int(
        escalation_diagnostics["pooled_inner_oof_feasible_structure_results"]
    )
    expected_support_pass = int(initial_diagnostics["pooled_support_pass_structure_results"]) + int(
        escalation_diagnostics["pooled_support_pass_structure_results"]
    )

    protocol_data = json.loads(protocol.read_text(encoding="utf-8"))
    require(protocol_data.get("status") ==
            "selection_stability_replica_aggregation_protocol_frozen_before_initial200_production",
            "aggregation protocol status mismatch")
    require(protocol_data.get("fold_level_winner_rule", {}).get("outer_fold_used_for_selection") is False,
            "protocol outer-fold selection gate mismatch")
    require(protocol_data.get("stability_diagnostic_interpretation", {}).get(
        "pilot_replicas_2_3_from_transfer_qualification_enter_aggregation") is False,
        "protocol pilot-exclusion gate mismatch")
    require(protocol_data.get("stability_diagnostic_interpretation", {}).get(
        "nominal_selection_can_be_changed_by_stability_results") is False,
        "protocol nominal-selection gate mismatch")
    require(
        protocol_data.get("scope", {}).get("predeclared_total_replicas") == REPLICA_COUNT
        and protocol_data.get("scope", {}).get("pretriggered_escalation_replicas") == [200, 999],
        "protocol all-1000 scope mismatch",
    )
    require(
        protocol_data.get("stability_diagnostic_interpretation", {}).get(
            "final_report_after_escalation_uses_all_1000_replicas"
        ) is True,
        "protocol final all-1000 report gate mismatch",
    )
    return head, inventory_sources, expected_feasible, expected_support_pass, input_paths


def ranked_row(ranked: RankedCandidate) -> dict[str, Any]:
    candidate = ranked.candidate
    return {
        "replica": candidate.replica,
        "category_id": candidate.category_id,
        "outer_fold": candidate.outer_fold,
        "job_index": candidate.job_index,
        "rank": ranked.rank,
        "selected_as_fold_winner": ranked.rank == 1,
        "structure_id": candidate.structure_id,
        "category_family_id": candidate.category_family_id,
        "feasibility_class": 0 if candidate.feasible else 1,
        "pooled_inner_oof_feasible": candidate.feasible,
        "pooled_signal_efficiency": candidate.signal_efficiency,
        "pooled_background_efficiency": candidate.background_efficiency,
        "signal_efficiency_overshoot_above_target": candidate.signal_efficiency - TARGET_SIGNAL_EFFICIENCY,
        "infeasible_negative_signal_efficiency_sort_value": -candidate.signal_efficiency,
        "cut_count": candidate.cut_count,
        "ranking_key_json": list(ranked.ranking_key),
        "all_inner_support_pass": candidate.all_inner_support_pass,
        "pooled_support_pass": candidate.pooled_support_pass,
        "refit_thresholds_json": candidate.refit_thresholds,
        "canonical_payload_sha256": candidate.canonical_payload_sha256,
        "structure_result_sha256": candidate.structure_result_sha256,
        "inner_crossfit_sha256": candidate.inner_crossfit_sha256,
        "execution_provenance_sha256": candidate.execution_provenance_sha256,
        "bundle_sha256": candidate.bundle_sha256,
        "outer_fold_used_for_selection": False,
    }


RANKED_FIELDS = (
    "replica", "category_id", "outer_fold", "job_index", "rank", "selected_as_fold_winner",
    "structure_id", "category_family_id", "feasibility_class", "pooled_inner_oof_feasible",
    "pooled_signal_efficiency", "pooled_background_efficiency",
    "signal_efficiency_overshoot_above_target", "infeasible_negative_signal_efficiency_sort_value",
    "cut_count", "ranking_key_json", "all_inner_support_pass", "pooled_support_pass",
    "refit_thresholds_json", "canonical_payload_sha256", "structure_result_sha256",
    "inner_crossfit_sha256", "execution_provenance_sha256", "bundle_sha256",
    "outer_fold_used_for_selection",
)


def aggregate(groups: Mapping[tuple[int, str, int], Sequence[Candidate]], output_dir: Path,
              repo: Path, repository_head: str, script_path: Path,
              input_paths: Sequence[Path], protocol_path: Path,
              initial_summary_path: Path) -> dict[str, Any]:
    tables = output_dir / "tables"
    figure_data = output_dir / "figure_data"
    manifests = output_dir / "manifests"
    tables.mkdir(parents=True)
    figure_data.mkdir()
    manifests.mkdir()

    all_ranked: list[RankedCandidate] = []
    fold_winners: list[Candidate] = []
    for replica in REPLICAS:
        for category in CATEGORIES:
            for outer_fold in OUTER_FOLDS:
                ranked = rank_fold_candidates(groups[(replica, category, outer_fold)])
                all_ranked.extend(ranked)
                fold_winners.append(ranked[0].candidate)

    require(len(all_ranked) == EXPECTED_ALL1000_ROWS, "ranked result count mismatch")
    require(len(fold_winners) == REPLICA_COUNT * 2 * 5, "fold winner count mismatch")
    ranked_shards: list[dict[str, Any]] = []
    for replica_start in range(0, REPLICA_COUNT, RANKED_REPLICAS_PER_SHARD):
        replica_stop = min(replica_start + RANKED_REPLICAS_PER_SHARD, REPLICA_COUNT)
        row_start = replica_start * RANKED_ROWS_PER_REPLICA
        row_stop = replica_stop * RANKED_ROWS_PER_REPLICA
        shard_items = all_ranked[row_start:row_stop]
        require(
            len(shard_items) == (replica_stop - replica_start) * RANKED_ROWS_PER_REPLICA,
            f"ranked shard row-count mismatch for replicas {replica_start}--{replica_stop - 1}",
        )
        require(
            {item.candidate.replica for item in shard_items}
            == set(range(replica_start, replica_stop)),
            f"ranked shard replica coverage mismatch for replicas {replica_start}--{replica_stop - 1}",
        )
        shard_path = tables / (
            f"ranked_structure_results_replicas_{replica_start:04d}_{replica_stop - 1:04d}.tsv"
        )
        write_tsv(shard_path, RANKED_FIELDS, (ranked_row(item) for item in shard_items))
        ranked_shards.append({
            "bytes": shard_path.stat().st_size,
            "path": str(shard_path.relative_to(output_dir)),
            "replica_first_inclusive": replica_start,
            "replica_last_inclusive": replica_stop - 1,
            "rows_excluding_header": len(shard_items),
            "sha256": sha256_file(shard_path),
        })
    expected_ranked_shards = math.ceil(REPLICA_COUNT / RANKED_REPLICAS_PER_SHARD)
    require(
        len(ranked_shards) == expected_ranked_shards,
        "ranked structure-result shard count mismatch",
    )
    write_json(
        manifests / "ranked_structure_results_manifest.json",
        {
            "schema_version": 1,
            "shard_count": len(ranked_shards),
            "shards": ranked_shards,
            "status": "pass_all1000_ranked_structure_results_sharded",
            "total_rows_excluding_headers": sum(
                int(shard["rows_excluding_header"]) for shard in ranked_shards
            ),
        },
    )

    winner_path = tables / "fold_level_winners.tsv"
    winner_rows = [ranked_row(item) for item in all_ranked if item.rank == 1]
    write_tsv(winner_path, RANKED_FIELDS, winner_rows)

    winners_by_replica_category: dict[tuple[int, str], list[Candidate]] = defaultdict(list)
    for winner in fold_winners:
        winners_by_replica_category[(winner.replica, winner.category_id)].append(winner)
    replica_results = [
        reduce_replica_category(winners_by_replica_category[(replica, category)])
        for replica in REPLICAS
        for category in CATEGORIES
    ]
    require(len(replica_results) == REPLICA_COUNT * 2, "replica/category result count mismatch")

    replica_fields = (
        "replica", "category_id", "five_fold_winner_structure_ids_json",
        "five_fold_winner_counts_by_structure_json", "maximum_winner_count",
        "maximum_winner_frequency", "unique_modal_family", "tie_resolution_applied",
        "tied_maximum_frequency_structure_ids_json", "replica_selected_structure_id",
        "replica_selected_structure_supporting_outer_folds_json",
        "replica_selected_structure_coordinatewise_median_thresholds_json",
        "five_fold_pooled_support_pass_json", "five_fold_all_inner_support_pass_json",
        "five_fold_feasibility_outcomes_json", "support_failure_count",
        "infeasible_fold_winner_count", "nominal_structure_recovered",
        "pilot_result_used", "nominal_deployment_candidate_changed",
        "validation_payloads_opened", "test_payloads_opened",
    )
    replica_rows: list[dict[str, Any]] = []
    for result in replica_results:
        replica_rows.append({
            "replica": result.replica,
            "category_id": result.category_id,
            "five_fold_winner_structure_ids_json": [winner.structure_id for winner in result.winners],
            "five_fold_winner_counts_by_structure_json": result.winner_counts,
            "maximum_winner_count": result.maximum_winner_count,
            "maximum_winner_frequency": result.maximum_winner_frequency,
            "unique_modal_family": result.unique_modal_family,
            "tie_resolution_applied": result.tie_resolution_applied,
            "tied_maximum_frequency_structure_ids_json": result.tied_structure_ids,
            "replica_selected_structure_id": result.selected_structure_id,
            "replica_selected_structure_supporting_outer_folds_json": result.supporting_outer_folds,
            "replica_selected_structure_coordinatewise_median_thresholds_json": result.median_thresholds,
            "five_fold_pooled_support_pass_json": [winner.pooled_support_pass for winner in result.winners],
            "five_fold_all_inner_support_pass_json": [winner.all_inner_support_pass for winner in result.winners],
            "five_fold_feasibility_outcomes_json": [winner.feasible for winner in result.winners],
            "support_failure_count": sum(not winner.pooled_support_pass for winner in result.winners),
            "infeasible_fold_winner_count": sum(not winner.feasible for winner in result.winners),
            "nominal_structure_recovered": result.selected_structure_id == NOMINAL_STRUCTURES[result.category_id],
            "pilot_result_used": False,
            "nominal_deployment_candidate_changed": False,
            "validation_payloads_opened": 0,
            "test_payloads_opened": 0,
        })
    write_tsv(tables / "replica_category_results.tsv", replica_fields, replica_rows)

    frequency_fields = (
        "category_id", "aggregation_level", "structure_id", "count", "denominator",
        "frequency", "frequency_rank", "is_nominal_structure",
    )
    frequency_rows: list[dict[str, Any]] = []
    frequency_by_category_level: dict[tuple[str, str], Counter[str]] = {}
    for category in CATEGORIES:
        fold_counts = Counter(winner.structure_id for winner in fold_winners if winner.category_id == category)
        replica_counts = Counter(
            result.selected_structure_id for result in replica_results if result.category_id == category
        )
        for level, counts, denominator in (
            ("fold_winner", fold_counts, FOLD_WINNERS_PER_CATEGORY),
            ("replica_selected", replica_counts, REPLICA_CATEGORIES_PER_CATEGORY),
        ):
            frequency_by_category_level[(category, level)] = counts
            rank_order = sorted(EXPECTED_STRUCTURE_IDS, key=lambda structure: (-counts[structure], structure))
            ranks = {structure: rank for rank, structure in enumerate(rank_order, start=1)}
            for structure in EXPECTED_STRUCTURE_IDS:
                frequency_rows.append({
                    "category_id": category,
                    "aggregation_level": level,
                    "structure_id": structure,
                    "count": counts[structure],
                    "denominator": denominator,
                    "frequency": counts[structure] / denominator,
                    "frequency_rank": ranks[structure],
                    "is_nominal_structure": structure == NOMINAL_STRUCTURES[category],
                })
    write_tsv(tables / "structure_frequencies.tsv", frequency_fields, frequency_rows)
    write_tsv(figure_data / "selection_structure_frequencies.tsv", frequency_fields, frequency_rows)

    optional_fields = (
        "category_id", "aggregation_level", "optional_variable", "inclusion_count",
        "denominator", "inclusion_frequency", "ambiguous_interval_inclusive_0p30_0p70",
    )
    optional_rows: list[dict[str, Any]] = []
    for category in CATEGORIES:
        for level, structures, denominator in (
            ("fold_winner", [winner.structure_id for winner in fold_winners if winner.category_id == category], FOLD_WINNERS_PER_CATEGORY),
            ("replica_selected", [result.selected_structure_id for result in replica_results if result.category_id == category], REPLICA_CATEGORIES_PER_CATEGORY),
        ):
            for variable in OPTIONAL_VARIABLES:
                count = sum(variable in optional_variables(structure) for structure in structures)
                frequency = count / denominator
                optional_rows.append({
                    "category_id": category,
                    "aggregation_level": level,
                    "optional_variable": variable,
                    "inclusion_count": count,
                    "denominator": denominator,
                    "inclusion_frequency": frequency,
                    "ambiguous_interval_inclusive_0p30_0p70": 0.30 <= frequency <= 0.70,
                })
    write_tsv(tables / "optional_variable_inclusion_frequencies.tsv", optional_fields, optional_rows)
    write_tsv(figure_data / "optional_variable_inclusion_frequencies.tsv", optional_fields, optional_rows)

    modal_fields = (
        "category_id", "maximum_winner_count", "replica_category_count", "denominator", "frequency",
    )
    modal_rows: list[dict[str, Any]] = []
    for category in CATEGORIES:
        counts = Counter(
            result.maximum_winner_count for result in replica_results if result.category_id == category
        )
        for maximum_count in range(1, 6):
            modal_rows.append({
                "category_id": category,
                "maximum_winner_count": maximum_count,
                "replica_category_count": counts[maximum_count],
                "denominator": REPLICA_CATEGORIES_PER_CATEGORY,
                "frequency": counts[maximum_count] / REPLICA_CATEGORIES_PER_CATEGORY,
            })
    write_tsv(tables / "modal_count_distribution.tsv", modal_fields, modal_rows)
    write_tsv(figure_data / "modal_count_distribution.tsv", modal_fields, modal_rows)

    conditional_fields = (
        "replica", "category_id", "selected_structure_id", "threshold_variable", "threshold_value",
        "supporting_outer_fold_count", "supporting_outer_folds_json", "is_nominal_structure",
    )
    conditional_rows: list[dict[str, Any]] = []
    for result in replica_results:
        for variable, value in sorted(result.median_thresholds.items()):
            conditional_rows.append({
                "replica": result.replica,
                "category_id": result.category_id,
                "selected_structure_id": result.selected_structure_id,
                "threshold_variable": variable,
                "threshold_value": value,
                "supporting_outer_fold_count": len(result.supporting_outer_folds),
                "supporting_outer_folds_json": result.supporting_outer_folds,
                "is_nominal_structure": result.selected_structure_id == NOMINAL_STRUCTURES[result.category_id],
            })
    write_tsv(tables / "conditional_replica_thresholds.tsv", conditional_fields, conditional_rows)
    write_tsv(figure_data / "conditional_replica_thresholds.tsv", conditional_fields, conditional_rows)

    nominal_conditional = [row for row in conditional_rows if row["is_nominal_structure"]]
    write_tsv(tables / "conditional_nominal_structure_thresholds.tsv", conditional_fields, nominal_conditional)

    threshold_summary_fields = (
        "category_id", "selected_structure_id", "threshold_variable", "replica_count",
        "mean", "population_standard_deviation", "minimum", "p16_linear", "median",
        "p84_linear", "maximum", "is_nominal_structure",
    )
    threshold_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in conditional_rows:
        threshold_groups[(row["category_id"], row["selected_structure_id"], row["threshold_variable"])].append(
            row["threshold_value"]
        )
    threshold_summary_rows: list[dict[str, Any]] = []
    for (category, structure, variable), values in sorted(threshold_groups.items()):
        threshold_summary_rows.append({
            "category_id": category,
            "selected_structure_id": structure,
            "threshold_variable": variable,
            "replica_count": len(values),
            "mean": statistics.fmean(values),
            "population_standard_deviation": statistics.pstdev(values),
            "minimum": min(values),
            "p16_linear": linear_quantile(values, 0.16),
            "median": float(statistics.median(values)),
            "p84_linear": linear_quantile(values, 0.84),
            "maximum": max(values),
            "is_nominal_structure": structure == NOMINAL_STRUCTURES[category],
        })
    write_tsv(tables / "conditional_threshold_distribution_summary.tsv",
              threshold_summary_fields, threshold_summary_rows)

    category_diagnostic_fields = (
        "category_id", "replica_category_count", "fold_winner_count",
        "modal_replica_selected_structure_id", "modal_replica_selected_count",
        "modal_replica_selected_frequency", "second_replica_selected_count",
        "second_replica_selected_frequency", "top_two_frequency_gap",
        "modal_frequency_below_0p70", "top_two_frequency_gap_below_0p15",
        "nominal_structure_id", "nominal_structure_recovery_count",
        "nominal_structure_recovery_frequency", "unique_modal_replica_category_count",
        "unique_modal_replica_category_fraction", "tie_resolution_replica_category_count",
        "tie_resolution_replica_category_fraction", "no_unique_five_fold_modal_structure_count",
        "no_unique_five_fold_modal_structure_fraction", "infeasible_fold_winner_count",
        "infeasible_fold_winner_frequency", "fold_winner_support_failure_count",
        "fold_winner_support_failure_frequency", "all1000_aggregation_complete",
        "initial200_retained_separately", "nominal_deployment_candidate_changed",
        "validation_payloads_opened", "test_payloads_opened",
    )
    category_rows: list[dict[str, Any]] = []
    summary_categories: dict[str, Any] = {}
    for category in CATEGORIES:
        results = [result for result in replica_results if result.category_id == category]
        winners = [winner for winner in fold_winners if winner.category_id == category]
        selected_counts = frequency_by_category_level[(category, "replica_selected")]
        ordered_counts = sorted(selected_counts.items(), key=lambda item: (-item[1], item[0]))
        modal_structure, modal_count = ordered_counts[0]
        second_count = ordered_counts[1][1] if len(ordered_counts) > 1 else 0
        modal_frequency = modal_count / REPLICA_CATEGORIES_PER_CATEGORY
        second_frequency = second_count / REPLICA_CATEGORIES_PER_CATEGORY
        gap = modal_frequency - second_frequency
        nominal_count = selected_counts[NOMINAL_STRUCTURES[category]]
        unique_count = sum(result.unique_modal_family for result in results)
        tie_count = sum(result.tie_resolution_applied for result in results)
        infeasible_winners = sum(not winner.feasible for winner in winners)
        support_failures = sum(not winner.pooled_support_pass for winner in winners)
        ambiguous_variables = [
            row["optional_variable"]
            for row in optional_rows
            if row["category_id"] == category
            and row["aggregation_level"] == "replica_selected"
            and row["ambiguous_interval_inclusive_0p30_0p70"]
        ]
        row = {
            "category_id": category,
            "replica_category_count": REPLICA_CATEGORIES_PER_CATEGORY,
            "fold_winner_count": FOLD_WINNERS_PER_CATEGORY,
            "modal_replica_selected_structure_id": modal_structure,
            "modal_replica_selected_count": modal_count,
            "modal_replica_selected_frequency": modal_frequency,
            "second_replica_selected_count": second_count,
            "second_replica_selected_frequency": second_frequency,
            "top_two_frequency_gap": gap,
            "modal_frequency_below_0p70": modal_frequency < 0.70,
            "top_two_frequency_gap_below_0p15": gap < 0.15,
            "nominal_structure_id": NOMINAL_STRUCTURES[category],
            "nominal_structure_recovery_count": nominal_count,
            "nominal_structure_recovery_frequency": nominal_count / REPLICA_CATEGORIES_PER_CATEGORY,
            "unique_modal_replica_category_count": unique_count,
            "unique_modal_replica_category_fraction": unique_count / REPLICA_CATEGORIES_PER_CATEGORY,
            "tie_resolution_replica_category_count": tie_count,
            "tie_resolution_replica_category_fraction": tie_count / REPLICA_CATEGORIES_PER_CATEGORY,
            "no_unique_five_fold_modal_structure_count": tie_count,
            "no_unique_five_fold_modal_structure_fraction": tie_count / REPLICA_CATEGORIES_PER_CATEGORY,
            "infeasible_fold_winner_count": infeasible_winners,
            "infeasible_fold_winner_frequency": infeasible_winners / FOLD_WINNERS_PER_CATEGORY,
            "fold_winner_support_failure_count": support_failures,
            "fold_winner_support_failure_frequency": support_failures / FOLD_WINNERS_PER_CATEGORY,
            "all1000_aggregation_complete": True,
            "initial200_retained_separately": True,
            "nominal_deployment_candidate_changed": False,
            "validation_payloads_opened": 0,
            "test_payloads_opened": 0,
        }
        category_rows.append(row)
        summary_categories[category] = {
            **row,
            "ambiguous_optional_variables_replica_level": ambiguous_variables,
            "modal_count_distribution": {
                str(maximum_count): sum(
                    result.maximum_winner_count == maximum_count for result in results
                )
                for maximum_count in range(1, 6)
            },
            "nominal_deployment_thresholds_unchanged": NOMINAL_THRESHOLDS[category],
            "replica_selected_structure_counts": dict(sorted(selected_counts.items())),
        }
    write_tsv(tables / "category_stability_diagnostics.tsv", category_diagnostic_fields, category_rows)

    initial_summary = json.loads(initial_summary_path.read_text(encoding="utf-8"))
    comparison_fields = (
        "category_id",
        "metric",
        "initial200_value",
        "all1000_value",
        "all1000_minus_initial200",
    )
    comparison_metrics = (
        "modal_replica_selected_frequency",
        "nominal_structure_recovery_frequency",
        "top_two_frequency_gap",
        "unique_modal_replica_category_fraction",
        "tie_resolution_replica_category_fraction",
        "infeasible_fold_winner_frequency",
        "fold_winner_support_failure_frequency",
    )
    comparison_rows: list[dict[str, Any]] = []
    for category in CATEGORIES:
        for metric in comparison_metrics:
            initial_value = float(initial_summary["categories"][category][metric])
            all1000_value = float(summary_categories[category][metric])
            comparison_rows.append({
                "category_id": category,
                "metric": metric,
                "initial200_value": initial_value,
                "all1000_value": all1000_value,
                "all1000_minus_initial200": all1000_value - initial_value,
            })
    write_tsv(
        tables / "initial200_vs_all1000_category_diagnostics.tsv",
        comparison_fields,
        comparison_rows,
    )
    write_tsv(
        figure_data / "initial200_vs_all1000_category_diagnostics.tsv",
        comparison_fields,
        comparison_rows,
    )

    summary = {
        "aggregation_input": aggregation_input_sha256s(repo, input_paths),
        "aggregation_performed": True,
        "categories": summary_categories,
        "fold_level_winner_count": REPLICA_COUNT * 2 * 5,
        "fold_level_winner_rule": {
            "feasible": [
                "feasibility_class_0", "minimum_pooled_inner_oof_background_efficiency",
                "minimum_signal_efficiency_overshoot_above_target", "fewer_continuous_cuts",
                "lexicographic_structure_id",
            ],
            "infeasible": [
                "feasibility_class_1", "maximum_pooled_inner_oof_signal_efficiency",
                "minimum_pooled_inner_oof_background_efficiency", "fewer_continuous_cuts",
                "lexicographic_structure_id",
            ],
            "outer_fold_used_for_selection": False,
            "selection_information": "pooled true inner out-of-fold metrics only",
            "target_signal_efficiency": TARGET_SIGNAL_EFFICIENCY,
        },
        "initial200_results_retained_separately": True,
        "next": "freeze_all1000_aggregation_before_final_train_side_cut_performance",
        "nominal_deployment_candidate_changed": False,
        "pilot_results_used": False,
        "ranked_structure_result_count": EXPECTED_ALL1000_ROWS,
        "ranked_structure_result_shard_count": len(ranked_shards),
        "replica_category_result_count": REPLICA_COUNT * 2,
        "repository_head_at_aggregation": repository_head,
        "schema_version": 1,
        "status": "pass_all1000_predeclared_selection_stability_aggregation",
        "test_payloads_opened": 0,
        "validation_payloads_opened": 0,
    }
    summary_path = output_dir / "all1000_stability_summary.json"
    write_json(summary_path, summary)

    # Machine-readable plot sidecars must be exact copies of their authoritative tables.
    require(sha256_file(tables / "structure_frequencies.tsv") ==
            sha256_file(figure_data / "selection_structure_frequencies.tsv"),
            "selection-frequency plot-data copy mismatch")
    require(sha256_file(tables / "optional_variable_inclusion_frequencies.tsv") ==
            sha256_file(figure_data / "optional_variable_inclusion_frequencies.tsv"),
            "optional-variable plot-data copy mismatch")
    require(sha256_file(tables / "modal_count_distribution.tsv") ==
            sha256_file(figure_data / "modal_count_distribution.tsv"),
            "modal-count plot-data copy mismatch")
    require(sha256_file(tables / "conditional_replica_thresholds.tsv") ==
            sha256_file(figure_data / "conditional_replica_thresholds.tsv"),
            "conditional-threshold plot-data copy mismatch")
    require(sha256_file(tables / "initial200_vs_all1000_category_diagnostics.tsv") ==
            sha256_file(figure_data / "initial200_vs_all1000_category_diagnostics.tsv"),
            "initial200/all1000 comparison plot-data copy mismatch")

    output_files = sorted(
        path for path in output_dir.rglob("*")
        if path.is_file() and path.name not in {"aggregation_manifest.json", "SHA256SUMS"}
    )
    manifest = {
        "aggregation_implementation": {
            "path": str(script_path.relative_to(repo)),
            "sha256": sha256_file(script_path),
        },
        "inputs": aggregation_input_sha256s(repo, input_paths),
        "output_files": {
            str(path.relative_to(output_dir)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_files
        },
        "pilot_results_used": False,
        "repository_head_at_aggregation": repository_head,
        "schema_version": 1,
        "status": "pass_deterministic_all1000_aggregation_manifest",
        "test_payloads_opened": 0,
        "validation_payloads_opened": 0,
    }
    manifest_path = manifests / "aggregation_manifest.json"
    write_json(manifest_path, manifest)

    checksum_paths = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    with (output_dir / "SHA256SUMS").open("w", encoding="utf-8", newline="") as handle:
        for path in checksum_paths:
            handle.write(f"{sha256_file(path)}  ./{path.relative_to(output_dir).as_posix()}\n")

    return summary


def self_check_initial200(repo: Path) -> None:
    """Prove that replicas 0--199 are reduced identically to the frozen implementation."""

    original_path = repo / "scripts/analysis/aggregate_hh4b_initial200_selection_stability.py"
    spec = importlib.util.spec_from_file_location("hh4b_frozen_initial200_aggregator", original_path)
    require(spec is not None and spec.loader is not None, "cannot load frozen initial-200 implementation")
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    inventory = repo / (
        "docs/checkpoints/"
        "hh4b_train_multivariate_cut_selection_stability_initial200_complete_return_audit_20260809_v1/"
        "structure_result_inventory.tsv"
    )
    require(sha256_file(inventory) == EXPECTED_INITIAL_INVENTORY_SHA256, "initial inventory SHA mismatch")
    original_groups = original.load_candidates(inventory)
    all1000_groups = load_candidates(
        [(inventory, 0)],
        expected_feasible_count=6561,
        expected_support_pass_count=54000,
        expected_replicas=tuple(range(200)),
        expected_row_count=54000,
    )
    require(set(original_groups) == set(all1000_groups), "initial-200 self-check group coverage mismatch")
    original_winners: dict[tuple[int, str], list[Any]] = defaultdict(list)
    all1000_winners: dict[tuple[int, str], list[Candidate]] = defaultdict(list)
    ranked_count = 0
    for key in sorted(original_groups):
        original_ranked = original.rank_fold_candidates(original_groups[key])
        all1000_ranked = rank_fold_candidates(all1000_groups[key])
        original_rows = [original.ranked_row(item) for item in original_ranked]
        all1000_rows = [ranked_row(item) for item in all1000_ranked]
        require(
            canonical_json(original_rows) == canonical_json(all1000_rows),
            f"initial-200 fold-ranking self-check mismatch: {key}",
        )
        replica, category, _ = key
        original_winners[(replica, category)].append(original_ranked[0].candidate)
        all1000_winners[(replica, category)].append(all1000_ranked[0].candidate)
        ranked_count += len(all1000_ranked)
    require(ranked_count == 54000, "initial-200 ranked-result self-check count mismatch")
    for key in sorted(original_winners):
        original_reduction = original.reduce_replica_category(original_winners[key])
        all1000_reduction = reduce_replica_category(all1000_winners[key])
        require(
            canonical_json(asdict(original_reduction)) == canonical_json(asdict(all1000_reduction)),
            f"initial-200 replica/category self-check mismatch: {key}",
        )
    require(len(original_winners) == 400, "initial-200 reduction self-check count mismatch")
    print("INITIAL200_BYTE_IDENTICAL_ALGORITHM_SELF_CHECK=PASS")
    print("INITIAL200_RANKED_RESULTS_COMPARED=54000")
    print("INITIAL200_REPLICA_CATEGORY_REDUCTIONS_COMPARED=400")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/hh4b_cut_baseline/all1000_stability"),
        help="Repository-relative output directory; it must not already exist.",
    )
    parser.add_argument(
        "--escalation-audit-checkpoint",
        type=Path,
        default=Path(
            "docs/checkpoints/"
            "hh4b_train_multivariate_cut_selection_stability_escalation800_complete_return_audit_20260810_v1"
        ),
        help="Repository-relative frozen escalation-800 complete-return audit checkpoint.",
    )
    parser.add_argument(
        "--escalation-audit-output",
        type=Path,
        default=Path(
            "/tmp/"
            "hh4b_train_multivariate_cut_selection_stability_escalation800_"
            "complete_return_audit_20260810_v1"
        ),
        help="External raw audit directory bound byte-for-byte by the compact checkpoint.",
    )
    parser.add_argument(
        "--self-check-initial200",
        action="store_true",
        help="Compare all initial-200 fold rankings and reductions to the frozen implementation, then exit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    if args.self_check_initial200:
        self_check_initial200(repo)
        return 0
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output_dir = output_dir.absolute()
    require(not output_dir.exists() and not output_dir.is_symlink(), f"output directory already exists: {output_dir}")
    require(output_dir.is_relative_to(repo), "output directory must be inside the repository")

    initial_checkpoint = repo / (
        "docs/checkpoints/"
        "hh4b_train_multivariate_cut_selection_stability_initial200_complete_return_audit_20260809_v1"
    )
    escalation_checkpoint = (
        args.escalation_audit_checkpoint
        if args.escalation_audit_checkpoint.is_absolute()
        else repo / args.escalation_audit_checkpoint
    ).resolve()
    require(escalation_checkpoint.is_relative_to(repo), "escalation checkpoint must be inside repository")
    escalation_audit_output = args.escalation_audit_output.resolve()
    protocol = repo / (
        "docs/checkpoints/"
        "hh4b_train_multivariate_cut_selection_stability_aggregation_protocol_20260808_v1/"
        "selection_stability_aggregation_protocol.json"
    )
    initial_summary = repo / (
        "artifacts/hh4b_cut_baseline/initial200_stability/initial200_stability_summary.json"
    )
    script_path = Path(__file__).resolve()

    (
        repository_head,
        inventory_sources,
        expected_feasible_count,
        expected_support_pass_count,
        input_paths,
    ) = validate_environment(
        repo,
        initial_checkpoint,
        escalation_checkpoint,
        escalation_audit_output,
        protocol,
        initial_summary,
    )
    groups = load_candidates(
        inventory_sources,
        expected_feasible_count,
        expected_support_pass_count,
    )
    try:
        summary = aggregate(
            groups,
            output_dir,
            repo,
            repository_head,
            script_path,
            input_paths,
            protocol,
            initial_summary,
        )
    except Exception:
        # A partially written aggregation must never be mistaken for a valid product.
        if output_dir.is_dir() and not output_dir.is_symlink():
            failure_marker = output_dir / "AGGREGATION_FAILED_DO_NOT_USE.txt"
            failure_marker.write_text(
                "AGGREGATION_STATUS=FAIL\nDO_NOT_USE_PARTIAL_OUTPUT=TRUE\n"
                "VALIDATION_PAYLOADS_OPENED=0\nTEST_PAYLOADS_OPENED=0\n",
                encoding="utf-8",
            )
        raise

    print("AGGREGATION_STATUS=PASS")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"FOLD_LEVEL_WINNERS={summary['fold_level_winner_count']}")
    print(f"REPLICA_CATEGORY_RESULTS={summary['replica_category_result_count']}")
    for category in CATEGORIES:
        values = summary["categories"][category]
        print(f"{category.upper()}_MODAL_STRUCTURE={values['modal_replica_selected_structure_id']}")
        print(f"{category.upper()}_MODAL_FREQUENCY={values['modal_replica_selected_frequency']}")
        print(f"{category.upper()}_NOMINAL_RECOVERY_FREQUENCY={values['nominal_structure_recovery_frequency']}")
    print("ALL1000_AGGREGATION_COMPLETE=TRUE")
    print("INITIAL200_RETAINED_SEPARATELY=TRUE")
    print("NOMINAL_DEPLOYMENT_CANDIDATE_CHANGED=FALSE")
    print("PILOT_RESULTS_USED=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AggregationError as exc:
        print(f"AGGREGATION_STATUS=FAIL\nERROR={exc}", file=sys.stderr)
        print("VALIDATION_PAYLOADS_OPENED=0\nTEST_PAYLOADS_OPENED=0", file=sys.stderr)
        raise SystemExit(1) from exc
