#!/usr/bin/env python3
"""Deterministically aggregate the frozen HH4b initial-200 stability inventory.

This program implements only the predeclared train-side selection-stability
protocol.  It never reads validation or test payloads and never evaluates outer
fold physics results.  The sole scientific input is the normalized, audited
54,000-row structure-result inventory frozen in Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_REPOSITORY_HEAD = "fd2557fffee1c3f1a5a5d903c7fd624cad5c92c9"
EXPECTED_INVENTORY_SHA256 = "47df422f39a50ac2f008031d3222ef04556760526abf160db57485861f3db056"
EXPECTED_AUDIT_JSON_SHA256 = "1e0984279f9e35c33acb9534681508d38bba0c88726ad8b78a1c4ca98d22b876"
EXPECTED_PROTOCOL_SHA256 = "a856d9027cc1d2e11a9d60e72b4104b5b8c24ce61acf43def0d4d487ab867eea"
EXPECTED_AUDIT_FREEZE_SHA256 = "9c3b937968def05917ff280ad4438ecba591fdde9f2251e04401d988af4e52df"
TARGET_SIGNAL_EFFICIENCY = 0.585957
CATEGORIES = ("exact3tag", "ge4tag")
OUTER_FOLDS = (0, 1, 2, 3, 4)
REPLICAS = tuple(range(200))
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


def expected_job_identity(job_index: int) -> tuple[int, int, str, str]:
    require(0 <= job_index < 2000, f"job index out of range: {job_index}")
    replica = job_index // 10
    within_replica = job_index % 10
    if within_replica < 5:
        category = "exact3tag"
        outer_fold = within_replica
    else:
        category = "ge4tag"
        outer_fold = within_replica - 5
    return replica, outer_fold, category, f"replica_{replica:04d}__{category}"


def load_candidates(inventory_path: Path) -> dict[tuple[int, str, int], list[Candidate]]:
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

    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames == expected_header, "inventory header mismatch")
        for row in reader:
            require(None not in row and all(value is not None for value in row.values()), "malformed inventory row")
            job_index = parse_int(row["job_index"], "job_index")
            replica = parse_int(row["replica"], "replica")
            outer_fold = parse_int(row["outer_fold"], "outer_fold")
            category = row["category_id"]
            expected_replica, expected_fold, expected_category, expected_payload = expected_job_identity(job_index)
            require(
                (replica, outer_fold, category, row["payload_key"])
                == (expected_replica, expected_fold, expected_category, expected_payload),
                f"job mapping mismatch for job {job_index}",
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
                    f"feasible result below target at job {job_index}: {signal_efficiency}",
                )
            else:
                require(
                    signal_efficiency < TARGET_SIGNAL_EFFICIENCY,
                    f"infeasible result at/above target at job {job_index}: {signal_efficiency}",
                )
            all_inner_support = parse_bool(row["all_inner_support_pass"], "all-inner support")
            pooled_support = parse_bool(row["pooled_support_pass"], "pooled support")
            require(json.loads(row["pooled_support_failures_json"]) == [], "support failure list is not empty")
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

            candidate = Candidate(
                job_index=job_index,
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

    require(row_count == 54000, f"inventory row count is {row_count}, expected 54000")
    require(feasible_count == 6561, f"feasible count is {feasible_count}, expected 6561")
    require(row_count - feasible_count == 47439, "infeasible count mismatch")
    require(support_pass_count == 54000, "pooled support-pass count mismatch")
    expected_groups = {(replica, category, fold) for replica in REPLICAS for category in CATEGORIES for fold in OUTER_FOLDS}
    require(set(groups) == expected_groups, "replica/category/fold coverage mismatch")
    return groups


def validate_environment(repo: Path, inventory: Path, audit_json: Path, audit_freeze: Path, protocol: Path) -> str:
    require(repo.resolve() == repo, f"repository path must be canonical: {repo}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    require(head == EXPECTED_REPOSITORY_HEAD, f"unexpected repository HEAD: {head}")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    require(branch == "delphes-hh4b-production", f"unexpected branch: {branch}")
    require(subprocess.run(["git", "diff", "--quiet", "--ignore-submodules", "--"], cwd=repo).returncode == 0,
            "tracked worktree is dirty")
    require(subprocess.run(["git", "diff", "--cached", "--quiet", "--ignore-submodules", "--"], cwd=repo).returncode == 0,
            "index is dirty")

    expected_hashes = {
        inventory: EXPECTED_INVENTORY_SHA256,
        audit_json: EXPECTED_AUDIT_JSON_SHA256,
        audit_freeze: EXPECTED_AUDIT_FREEZE_SHA256,
        protocol: EXPECTED_PROTOCOL_SHA256,
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
    return head


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
              inventory_path: Path, audit_json_path: Path, audit_freeze_path: Path,
              protocol_path: Path) -> dict[str, Any]:
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

    require(len(all_ranked) == 54000, "ranked result count mismatch")
    require(len(fold_winners) == 2000, "fold winner count mismatch")
    require(all(candidate.pooled_support_pass for candidate in fold_winners), "fold winner support failure")

    ranked_path = tables / "ranked_structure_results.tsv"
    write_tsv(ranked_path, RANKED_FIELDS, (ranked_row(item) for item in all_ranked))

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
    require(len(replica_results) == 400, "replica/category result count mismatch")

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
            ("fold_winner", fold_counts, 1000),
            ("replica_selected", replica_counts, 200),
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
            ("fold_winner", [winner.structure_id for winner in fold_winners if winner.category_id == category], 1000),
            ("replica_selected", [result.selected_structure_id for result in replica_results if result.category_id == category], 200),
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
                "denominator": 200,
                "frequency": counts[maximum_count] / 200.0,
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
        "fold_winner_support_failure_frequency", "pretriggered_escalation_remains_required",
        "initial200_can_cancel_escalation", "nominal_deployment_candidate_changed",
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
        modal_frequency = modal_count / 200.0
        second_frequency = second_count / 200.0
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
            "replica_category_count": 200,
            "fold_winner_count": 1000,
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
            "nominal_structure_recovery_frequency": nominal_count / 200.0,
            "unique_modal_replica_category_count": unique_count,
            "unique_modal_replica_category_fraction": unique_count / 200.0,
            "tie_resolution_replica_category_count": tie_count,
            "tie_resolution_replica_category_fraction": tie_count / 200.0,
            "no_unique_five_fold_modal_structure_count": tie_count,
            "no_unique_five_fold_modal_structure_fraction": tie_count / 200.0,
            "infeasible_fold_winner_count": infeasible_winners,
            "infeasible_fold_winner_frequency": infeasible_winners / 1000.0,
            "fold_winner_support_failure_count": support_failures,
            "fold_winner_support_failure_frequency": support_failures / 1000.0,
            "pretriggered_escalation_remains_required": True,
            "initial200_can_cancel_escalation": False,
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

    summary = {
        "aggregation_input": {
            "complete_return_audit_freeze_sha256": sha256_file(audit_freeze_path),
            "complete_return_audit_json_sha256": sha256_file(audit_json_path),
            "selection_stability_aggregation_protocol_sha256": sha256_file(protocol_path),
            "structure_result_inventory_sha256": sha256_file(inventory_path),
        },
        "aggregation_performed": True,
        "categories": summary_categories,
        "fold_level_winner_count": 2000,
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
        "initial200_results_can_cancel_pretriggered_escalation": False,
        "next": "freeze_initial200_aggregation_then_explicitly_authorize_pretriggered_replicas_200_999",
        "nominal_deployment_candidate_changed": False,
        "pilot_results_used": False,
        "replica_category_result_count": 400,
        "repository_head_at_aggregation": repository_head,
        "schema_version": 1,
        "status": "pass_initial200_predeclared_selection_stability_aggregation",
        "test_payloads_opened": 0,
        "validation_payloads_opened": 0,
    }
    summary_path = output_dir / "initial200_stability_summary.json"
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

    output_files = sorted(
        path for path in output_dir.rglob("*")
        if path.is_file() and path.name not in {"aggregation_manifest.json", "SHA256SUMS"}
    )
    manifest = {
        "aggregation_implementation": {
            "path": str(script_path.relative_to(repo)),
            "sha256": sha256_file(script_path),
        },
        "inputs": {
            str(inventory_path.relative_to(repo)): sha256_file(inventory_path),
            str(audit_json_path.relative_to(repo)): sha256_file(audit_json_path),
            str(audit_freeze_path.relative_to(repo)): sha256_file(audit_freeze_path),
            str(protocol_path.relative_to(repo)): sha256_file(protocol_path),
        },
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
        "status": "pass_deterministic_initial200_aggregation_manifest",
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
        default=Path("artifacts/hh4b_cut_baseline/initial200_stability"),
        help="Repository-relative output directory; it must not already exist.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output_dir = output_dir.absolute()
    require(not output_dir.exists() and not output_dir.is_symlink(), f"output directory already exists: {output_dir}")
    require(output_dir.is_relative_to(repo), "output directory must be inside the repository")

    audit_checkpoint = repo / (
        "docs/checkpoints/"
        "hh4b_train_multivariate_cut_selection_stability_initial200_complete_return_audit_20260809_v1"
    )
    inventory = audit_checkpoint / "structure_result_inventory.tsv"
    audit_json = audit_checkpoint / "complete_return_audit.json"
    audit_freeze = audit_checkpoint / "complete_return_audit_freeze.json"
    protocol = repo / (
        "docs/checkpoints/"
        "hh4b_train_multivariate_cut_selection_stability_aggregation_protocol_20260808_v1/"
        "selection_stability_aggregation_protocol.json"
    )
    script_path = Path(__file__).resolve()

    repository_head = validate_environment(repo, inventory, audit_json, audit_freeze, protocol)
    groups = load_candidates(inventory)
    try:
        summary = aggregate(
            groups,
            output_dir,
            repo,
            repository_head,
            script_path,
            inventory,
            audit_json,
            audit_freeze,
            protocol,
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
    print("PRETRIGGERED_ESCALATION_REMAINS_REQUIRED=TRUE")
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
