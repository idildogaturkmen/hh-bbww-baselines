#!/usr/bin/env python3
"""Independently audit and deterministically reproduce the HH4b all-1000 aggregation.

The row-level audit below deliberately does not import the aggregation
implementation.  Only the final deterministic-reproduction step loads that
implementation, after the independent ranking and reduction checks have
passed.
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
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


TARGET_SIGNAL_EFFICIENCY = 0.585957
CATEGORIES = ("exact3tag", "ge4tag")
OUTER_FOLDS = (0, 1, 2, 3, 4)
REPLICA_COUNT = 1000
STRUCTURES_PER_FOLD = 27
RANKED_ROWS = REPLICA_COUNT * len(CATEGORIES) * len(OUTER_FOLDS) * STRUCTURES_PER_FOLD
FOLD_WINNERS = REPLICA_COUNT * len(CATEGORIES) * len(OUTER_FOLDS)
REPLICA_CATEGORY_RESULTS = REPLICA_COUNT * len(CATEGORIES)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
NOMINAL_STRUCTURES = {
    "exact3tag": "radial_mass__category__plus_ht_candidate_jets",
    "ge4tag": "radial_mass__category__plus_mhh_and_abs_h_delta_eta",
}
EXPECTED_STRUCTURE_IDS = {
    f"{mass}__category__{suffix}"
    for mass in ("asymmetric_rectangular_mass", "radial_mass", "symmetric_rectangular_mass")
    for suffix in (
        "mass_only",
        "plus_abs_h_delta_eta",
        "plus_h2_pt",
        "plus_h2_pt_and_max_drbb",
        "plus_ht_candidate_jets",
        "plus_ht_candidate_jets_and_max_drbb",
        "plus_max_drbb",
        "plus_mhh",
        "plus_mhh_and_abs_h_delta_eta",
    )
}
RANKED_FIELDS = (
    "replica", "category_id", "outer_fold", "job_index", "rank",
    "selected_as_fold_winner", "structure_id", "category_family_id",
    "feasibility_class", "pooled_inner_oof_feasible", "pooled_signal_efficiency",
    "pooled_background_efficiency", "signal_efficiency_overshoot_above_target",
    "infeasible_negative_signal_efficiency_sort_value", "cut_count", "ranking_key_json",
    "all_inner_support_pass", "pooled_support_pass", "refit_thresholds_json",
    "canonical_payload_sha256", "structure_result_sha256", "inner_crossfit_sha256",
    "execution_provenance_sha256", "bundle_sha256", "outer_fold_used_for_selection",
)
REPLICA_FIELDS = (
    "replica", "category_id", "five_fold_winner_structure_ids_json",
    "five_fold_winner_counts_by_structure_json", "maximum_winner_count",
    "maximum_winner_frequency", "unique_modal_family", "tie_resolution_applied",
    "tied_maximum_frequency_structure_ids_json", "replica_selected_structure_id",
    "replica_selected_structure_supporting_outer_folds_json",
    "replica_selected_structure_coordinatewise_median_thresholds_json",
    "five_fold_pooled_support_pass_json", "five_fold_all_inner_support_pass_json",
    "five_fold_feasibility_outcomes_json", "support_failure_count",
    "infeasible_fold_winner_count", "nominal_structure_recovered", "pilot_result_used",
    "nominal_deployment_candidate_changed", "validation_payloads_opened",
    "test_payloads_opened",
)


class All1000AuditError(RuntimeError):
    """A fail-closed all-1000 audit contract was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise All1000AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def parse_bool(value: str, field: str) -> bool:
    require(value in {"True", "False"}, f"invalid Boolean {field}: {value!r}")
    return value == "True"


def parse_int(value: str, field: str) -> int:
    require(value.isdigit(), f"invalid integer {field}: {value!r}")
    return int(value)


def parse_float(value: str, field: str) -> float:
    result = float(value)
    require(math.isfinite(result), f"non-finite float {field}: {value!r}")
    return result


def ranking_key(row: Mapping[str, str]) -> tuple[Any, ...]:
    feasible = parse_bool(row["pooled_inner_oof_feasible"], "pooled_inner_oof_feasible")
    signal = parse_float(row["pooled_signal_efficiency"], "pooled_signal_efficiency")
    background = parse_float(row["pooled_background_efficiency"], "pooled_background_efficiency")
    cuts = parse_int(row["cut_count"], "cut_count")
    if feasible:
        return (0, background, signal - TARGET_SIGNAL_EFFICIENCY, cuts, row["structure_id"])
    return (1, -signal, background, cuts, row["structure_id"])


def coordinatewise_median(thresholds: Sequence[Mapping[str, float]]) -> dict[str, float]:
    require(bool(thresholds), "cannot reduce empty threshold collection")
    keys = set(thresholds[0])
    require(bool(keys) and all(set(item) == keys for item in thresholds), "threshold coordinate mismatch")
    return {
        key: float(statistics.median(item[key] for item in thresholds))
        for key in sorted(keys)
    }


def reduce_winners(winners: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(winners) == 5, "replica/category does not have five winners")
    ordered = sorted(winners, key=lambda row: row["outer_fold"])
    require(tuple(row["outer_fold"] for row in ordered) == OUTER_FOLDS, "outer-fold coverage mismatch")
    counts = Counter(row["structure_id"] for row in ordered)
    maximum = max(counts.values())
    tied = sorted(structure for structure, count in counts.items() if count == maximum)
    selected = tied[0]
    supporting = [row["outer_fold"] for row in ordered if row["structure_id"] == selected]
    selected_thresholds = [row["thresholds"] for row in ordered if row["structure_id"] == selected]
    return {
        "structure_ids": [row["structure_id"] for row in ordered],
        "counts": dict(sorted(counts.items())),
        "maximum": maximum,
        "maximum_frequency": maximum / 5.0,
        "unique": len(tied) == 1,
        "tie": len(tied) > 1,
        "tied": tied,
        "selected": selected,
        "supporting": supporting,
        "median_thresholds": coordinatewise_median(selected_thresholds),
        "pooled_support": [row["pooled_support"] for row in ordered],
        "all_inner_support": [row["all_inner_support"] for row in ordered],
        "feasible": [row["feasible"] for row in ordered],
    }


def read_tsv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames == list(fields), f"TSV header changed: {path}")
        rows = list(reader)
    require(all(None not in row and all(value is not None for value in row.values()) for row in rows),
            f"malformed TSV row: {path}")
    return rows


def validate_ranked_row(row: Mapping[str, str]) -> dict[str, Any]:
    replica = parse_int(row["replica"], "replica")
    category = row["category_id"]
    fold = parse_int(row["outer_fold"], "outer_fold")
    rank = parse_int(row["rank"], "rank")
    require(0 <= replica < REPLICA_COUNT, "replica out of range")
    require(category in CATEGORIES and fold in OUTER_FOLDS, "category/fold out of range")
    require(1 <= rank <= STRUCTURES_PER_FOLD, "rank out of range")
    expected_job = replica * 10 + fold + (0 if category == "exact3tag" else 5)
    require(parse_int(row["job_index"], "job_index") == expected_job, "job mapping mismatch")
    require(parse_bool(row["selected_as_fold_winner"], "selected_as_fold_winner") == (rank == 1),
            "fold-winner marker/rank mismatch")
    structure = row["structure_id"]
    require(structure in EXPECTED_STRUCTURE_IDS, "unexpected structure ID")
    require(row["category_family_id"] == f"{category}::{structure}", "category family mismatch")
    feasible = parse_bool(row["pooled_inner_oof_feasible"], "pooled_inner_oof_feasible")
    require(parse_int(row["feasibility_class"], "feasibility_class") == (0 if feasible else 1),
            "feasibility class mismatch")
    signal = parse_float(row["pooled_signal_efficiency"], "pooled_signal_efficiency")
    background = parse_float(row["pooled_background_efficiency"], "pooled_background_efficiency")
    require(0.0 <= signal <= 1.0 and 0.0 <= background <= 1.0, "efficiency out of range")
    require((signal >= TARGET_SIGNAL_EFFICIENCY) == feasible, "feasibility/target mismatch")
    require(parse_float(row["signal_efficiency_overshoot_above_target"], "signal overshoot")
            == signal - TARGET_SIGNAL_EFFICIENCY, "signal overshoot mismatch")
    require(parse_float(row["infeasible_negative_signal_efficiency_sort_value"], "negative signal")
            == -signal, "negative signal sort value mismatch")
    key = ranking_key(row)
    require(json.loads(row["ranking_key_json"]) == list(key), "serialized ranking key mismatch")
    thresholds = json.loads(row["refit_thresholds_json"])
    require(isinstance(thresholds, dict) and bool(thresholds), "invalid threshold JSON")
    thresholds = {str(name): float(value) for name, value in thresholds.items()}
    require(all(math.isfinite(value) for value in thresholds.values()), "non-finite threshold")
    require(len(thresholds) == parse_int(row["cut_count"], "cut_count"), "cut/threshold mismatch")
    for field in (
        "canonical_payload_sha256", "structure_result_sha256", "inner_crossfit_sha256",
        "execution_provenance_sha256", "bundle_sha256",
    ):
        require(SHA256_RE.fullmatch(row[field]) is not None, f"invalid {field}")
    require(parse_bool(row["outer_fold_used_for_selection"], "outer_fold_used_for_selection") is False,
            "outer fold was used for selection")
    return {
        "replica": replica,
        "category_id": category,
        "outer_fold": fold,
        "rank": rank,
        "structure_id": structure,
        "ranking_key": key,
        "thresholds": thresholds,
        "feasible": feasible,
        "all_inner_support": parse_bool(row["all_inner_support_pass"], "all_inner_support_pass"),
        "pooled_support": parse_bool(row["pooled_support_pass"], "pooled_support_pass"),
    }


def compare_initial200(source: Path, initial: Path) -> tuple[int, int]:
    initial_ranked = read_tsv(initial / "tables/ranked_structure_results.tsv", RANKED_FIELDS)
    first_shard = read_tsv(
        source / "tables/ranked_structure_results_replicas_0000_0099.tsv", RANKED_FIELDS
    )
    second_shard = read_tsv(
        source / "tables/ranked_structure_results_replicas_0100_0199.tsv", RANKED_FIELDS
    )
    require(first_shard + second_shard == initial_ranked,
            "replicas 0--199 ranked rows differ from the frozen initial-200 product")
    initial_reduced = read_tsv(initial / "tables/replica_category_results.tsv", REPLICA_FIELDS)
    all_reduced = read_tsv(source / "tables/replica_category_results.tsv", REPLICA_FIELDS)
    require(all_reduced[:400] == initial_reduced,
            "replicas 0--199 reductions differ from the frozen initial-200 product")
    return len(initial_ranked), len(initial_reduced)


def compare_trees(reference: Path, reproduced: Path) -> dict[str, int]:
    reference_files = sorted(path.relative_to(reference) for path in reference.rglob("*") if path.is_file())
    reproduced_files = sorted(path.relative_to(reproduced) for path in reproduced.rglob("*") if path.is_file())
    require(reference_files == reproduced_files, "deterministic rerun file set changed")
    byte_count = 0
    for relative in reference_files:
        left = reference / relative
        right = reproduced / relative
        require(left.stat().st_size == right.stat().st_size, f"rerun size mismatch: {relative}")
        require(sha256_file(left) == sha256_file(right), f"rerun byte mismatch: {relative}")
        byte_count += left.stat().st_size
    return {"file_count": len(reference_files), "byte_count": byte_count}


def load_aggregator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("hh4b_all1000_aggregator_for_reproduction", path)
    require(spec is not None and spec.loader is not None, "cannot load aggregation implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def deterministic_rerun(repo: Path, source: Path, summary: Mapping[str, Any]) -> dict[str, int]:
    script = repo / "scripts/analysis/aggregate_hh4b_all1000_selection_stability.py"
    aggregator = load_aggregator(script)
    initial_checkpoint = repo / (
        "docs/checkpoints/"
        "hh4b_train_multivariate_cut_selection_stability_initial200_complete_return_audit_20260809_v1"
    )
    escalation_checkpoint = repo / (
        "docs/checkpoints/"
        "hh4b_train_multivariate_cut_selection_stability_escalation800_complete_return_audit_20260810_v1"
    )
    external_audit = Path(
        "/tmp/hh4b_train_multivariate_cut_selection_stability_escalation800_"
        "complete_return_audit_20260810_v1"
    )
    protocol = repo / (
        "docs/checkpoints/"
        "hh4b_train_multivariate_cut_selection_stability_aggregation_protocol_20260808_v1/"
        "selection_stability_aggregation_protocol.json"
    )
    initial_summary = repo / "artifacts/hh4b_cut_baseline/initial200_stability/initial200_stability_summary.json"
    _, inventory_sources, feasible_count, support_count, input_paths = aggregator.validate_environment(
        repo, initial_checkpoint, escalation_checkpoint, external_audit, protocol, initial_summary
    )
    groups = aggregator.load_candidates(inventory_sources, feasible_count, support_count)
    with tempfile.TemporaryDirectory(prefix="hh4b_all1000_deterministic_rerun_") as directory:
        reproduced = Path(directory) / "all1000_stability"
        aggregator.aggregate(
            groups,
            reproduced,
            repo,
            str(summary["repository_head_at_aggregation"]),
            script,
            input_paths,
            protocol,
            initial_summary,
        )
        return compare_trees(source, reproduced)


def verify_source_checksums(source: Path) -> int:
    sums = source / "SHA256SUMS"
    require(sums.is_file() and not sums.is_symlink(), "source SHA256SUMS is missing")
    checked = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        require(SHA256_RE.fullmatch(digest) is not None and relative.startswith("./"),
                "malformed source checksum line")
        path = source / relative[2:]
        require(path.is_file() and not path.is_symlink(), f"missing checksummed product: {relative}")
        require(sha256_file(path) == digest, f"source checksum mismatch: {relative}")
        checked += 1
    actual = {path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()}
    listed = {line.split("  ", 1)[1][2:] for line in sums.read_text(encoding="utf-8").splitlines()}
    require(actual == listed | {"SHA256SUMS"}, "source checksum manifest file set mismatch")
    return checked


def audit_ranked_and_reductions(source: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    ranked_manifest_path = source / "manifests/ranked_structure_results_manifest.json"
    manifest = json.loads(ranked_manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("status") == "pass_all1000_ranked_structure_results_sharded",
            "ranked manifest status changed")
    shards = manifest.get("shards")
    require(isinstance(shards, list) and len(shards) == 10, "ranked shard count changed")
    require(manifest.get("total_rows_excluding_headers") == RANKED_ROWS, "ranked manifest row count changed")

    winner_raw: list[dict[str, str]] = []
    winner_parsed: list[dict[str, Any]] = []
    ranked_count = 0
    for index, shard in enumerate(shards):
        first = index * 100
        last = first + 99
        expected_path = f"tables/ranked_structure_results_replicas_{first:04d}_{last:04d}.tsv"
        require(shard.get("path") == expected_path, "ranked shard path/range changed")
        require(shard.get("replica_first_inclusive") == first
                and shard.get("replica_last_inclusive") == last
                and shard.get("rows_excluding_header") == 27000, "ranked shard metadata changed")
        path = source / expected_path
        require(path.stat().st_size == shard.get("bytes") and sha256_file(path) == shard.get("sha256"),
                f"ranked shard binding mismatch: {expected_path}")
        rows = read_tsv(path, RANKED_FIELDS)
        require(len(rows) == 27000, f"ranked shard row count changed: {expected_path}")
        for local_index in range(0, len(rows), STRUCTURES_PER_FOLD):
            raw_group = rows[local_index:local_index + STRUCTURES_PER_FOLD]
            parsed_group = [validate_ranked_row(row) for row in raw_group]
            expected_replica = first + local_index // 270
            within_replica = (local_index % 270) // 27
            expected_category = CATEGORIES[within_replica // 5]
            expected_fold = within_replica % 5
            require({(row["replica"], row["category_id"], row["outer_fold"])
                     for row in parsed_group} == {(expected_replica, expected_category, expected_fold)},
                    "ranked row ordering/coverage mismatch")
            require([row["rank"] for row in parsed_group] == list(range(1, 28)), "rank coverage changed")
            require({row["structure_id"] for row in parsed_group} == EXPECTED_STRUCTURE_IDS,
                    "frozen 27-structure coverage changed")
            require([row["ranking_key"] for row in parsed_group]
                    == sorted(row["ranking_key"] for row in parsed_group), "rank ordering changed")
            winner_raw.append(raw_group[0])
            winner_parsed.append(parsed_group[0])
        ranked_count += len(rows)
    require(ranked_count == RANKED_ROWS and len(winner_parsed) == FOLD_WINNERS,
            "ranked/winner total changed")

    authoritative_winners = read_tsv(source / "tables/fold_level_winners.tsv", RANKED_FIELDS)
    require(authoritative_winners == winner_raw, "fold-winner table differs from independently selected rank-1 rows")

    reductions = read_tsv(source / "tables/replica_category_results.tsv", REPLICA_FIELDS)
    require(len(reductions) == REPLICA_CATEGORY_RESULTS, "replica/category table row count changed")
    selected_by_category: dict[str, Counter[str]] = {category: Counter() for category in CATEGORIES}
    tie_by_category = Counter()
    infeasible_by_category = Counter()
    support_failures_by_category = Counter()
    for row_index, row in enumerate(reductions):
        replica = row_index // 2
        category = CATEGORIES[row_index % 2]
        require((parse_int(row["replica"], "replica"), row["category_id"]) == (replica, category),
                "replica/category reduction ordering changed")
        start = row_index * 5
        expected = reduce_winners(winner_parsed[start:start + 5])
        require(json.loads(row["five_fold_winner_structure_ids_json"]) == expected["structure_ids"],
                "five-fold structure sequence mismatch")
        require(json.loads(row["five_fold_winner_counts_by_structure_json"]) == expected["counts"],
                "winner-count reduction mismatch")
        require(parse_int(row["maximum_winner_count"], "maximum_winner_count") == expected["maximum"],
                "maximum winner count mismatch")
        require(parse_float(row["maximum_winner_frequency"], "maximum_winner_frequency")
                == expected["maximum_frequency"], "maximum winner frequency mismatch")
        require(parse_bool(row["unique_modal_family"], "unique_modal_family") == expected["unique"],
                "unique-modal marker mismatch")
        require(parse_bool(row["tie_resolution_applied"], "tie_resolution_applied") == expected["tie"],
                "tie-resolution marker mismatch")
        require(json.loads(row["tied_maximum_frequency_structure_ids_json"]) == expected["tied"],
                "tied-structure list mismatch")
        require(row["replica_selected_structure_id"] == expected["selected"],
                "lexicographic tie reduction mismatch")
        require(json.loads(row["replica_selected_structure_supporting_outer_folds_json"])
                == expected["supporting"], "supporting-fold list mismatch")
        require(json.loads(row["replica_selected_structure_coordinatewise_median_thresholds_json"])
                == expected["median_thresholds"], "coordinatewise threshold median mismatch")
        require(json.loads(row["five_fold_pooled_support_pass_json"]) == expected["pooled_support"],
                "pooled-support sequence mismatch")
        require(json.loads(row["five_fold_all_inner_support_pass_json"]) == expected["all_inner_support"],
                "inner-support sequence mismatch")
        require(json.loads(row["five_fold_feasibility_outcomes_json"]) == expected["feasible"],
                "feasibility sequence mismatch")
        failures = sum(not value for value in expected["pooled_support"])
        infeasible = sum(not value for value in expected["feasible"])
        require(parse_int(row["support_failure_count"], "support_failure_count") == failures,
                "support failure count mismatch")
        require(parse_int(row["infeasible_fold_winner_count"], "infeasible_fold_winner_count") == infeasible,
                "infeasible winner count mismatch")
        require(parse_bool(row["nominal_structure_recovered"], "nominal_structure_recovered")
                == (expected["selected"] == NOMINAL_STRUCTURES[category]), "nominal recovery mismatch")
        require(parse_bool(row["pilot_result_used"], "pilot_result_used") is False
                and parse_bool(row["nominal_deployment_candidate_changed"],
                               "nominal_deployment_candidate_changed") is False
                and row["validation_payloads_opened"] == "0" and row["test_payloads_opened"] == "0",
                "sealed-data/nominal/pilot reduction gate changed")
        selected_by_category[category][expected["selected"]] += 1
        tie_by_category[category] += int(expected["tie"])
        infeasible_by_category[category] += infeasible
        support_failures_by_category[category] += failures

    category_audit: dict[str, Any] = {}
    for category in CATEGORIES:
        payload = summary["categories"][category]
        counts = selected_by_category[category]
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        modal_structure, modal_count = ordered[0]
        require(payload["replica_selected_structure_counts"] == dict(sorted(counts.items())),
                f"summary selected-structure counts changed for {category}")
        require(payload["modal_replica_selected_structure_id"] == modal_structure
                and payload["modal_replica_selected_count"] == modal_count,
                f"summary modal structure changed for {category}")
        require(payload["nominal_structure_recovery_count"] == counts[NOMINAL_STRUCTURES[category]],
                f"summary nominal recovery changed for {category}")
        require(payload["tie_resolution_replica_category_count"] == tie_by_category[category],
                f"summary tie count changed for {category}")
        require(payload["infeasible_fold_winner_count"] == infeasible_by_category[category]
                and payload["fold_winner_support_failure_count"] == support_failures_by_category[category],
                f"summary winner diagnostic changed for {category}")
        require(payload["nominal_deployment_candidate_changed"] is False
                and payload["validation_payloads_opened"] == 0 and payload["test_payloads_opened"] == 0,
                f"summary seal changed for {category}")
        category_audit[category] = {
            "modal_replica_selected_structure_id": modal_structure,
            "modal_replica_selected_count": modal_count,
            "modal_replica_selected_frequency": modal_count / 1000.0,
            "nominal_structure_recovery_count": counts[NOMINAL_STRUCTURES[category]],
            "nominal_structure_recovery_frequency": counts[NOMINAL_STRUCTURES[category]] / 1000.0,
            "tie_resolution_replica_category_count": tie_by_category[category],
            "infeasible_fold_winner_count": infeasible_by_category[category],
            "fold_winner_support_failure_count": support_failures_by_category[category],
        }
    return {
        "ranked_structure_results_audited": ranked_count,
        "fold_level_winners_independently_selected": len(winner_parsed),
        "replica_category_reductions_independently_recomputed": len(reductions),
        "categories": category_audit,
    }


def write_audit_output(output: Path, payload: Mapping[str, Any]) -> None:
    require(not output.exists() and not output.is_symlink(), f"audit output exists: {output}")
    require(output.parent.is_dir(), f"audit output parent is missing: {output.parent}")
    build = output.parent / f".{output.name}.build.{os.getpid()}"
    require(not build.exists(), f"audit build path exists: {build}")
    build.mkdir()
    audit_path = build / "all1000_stability_independent_audit.json"
    audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                          encoding="utf-8")
    (build / "SHA256SUMS").write_text(
        f"{sha256_file(audit_path)}  ./all1000_stability_independent_audit.json\n",
        encoding="utf-8",
    )
    os.replace(build, output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"))
    parser.add_argument("--source-dir", type=Path,
                        default=Path("artifacts/hh4b_cut_baseline/all1000_stability"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/hh4b_cut_baseline/all1000_stability_independent_audit"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    require(repo == args.repo, "repository path is not canonical")
    require(subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
            == "delphes-hh4b-production", "branch changed")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", "origin/delphes-hh4b-production"], cwd=repo, text=True
    ).strip()
    require(head == remote, "local/remote HEAD mismatch")
    source = args.source_dir if args.source_dir.is_absolute() else repo / args.source_dir
    source = source.resolve()
    require(source.is_relative_to(repo) and source.is_dir() and not source.is_symlink(),
            "source artifact is invalid")
    output = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output = output.resolve()
    require(output.is_relative_to(repo), "audit output must be inside repository")

    checked_files = verify_source_checksums(source)
    summary = json.loads((source / "all1000_stability_summary.json").read_text(encoding="utf-8"))
    require(summary.get("status") == "pass_all1000_predeclared_selection_stability_aggregation",
            "all1000 summary status changed")
    require(summary.get("ranked_structure_result_count") == RANKED_ROWS
            and summary.get("fold_level_winner_count") == FOLD_WINNERS
            and summary.get("replica_category_result_count") == REPLICA_CATEGORY_RESULTS,
            "all1000 summary count changed")
    require(summary.get("pilot_results_used") is False
            and summary.get("nominal_deployment_candidate_changed") is False
            and summary.get("validation_payloads_opened") == 0
            and summary.get("test_payloads_opened") == 0,
            "all1000 summary seal/nominal/pilot gate changed")
    independent = audit_ranked_and_reductions(source, summary)
    initial = repo / "artifacts/hh4b_cut_baseline/initial200_stability"
    initial_ranked, initial_reduced = compare_initial200(source, initial)
    rerun = deterministic_rerun(repo, source, summary)
    payload = {
        "schema_version": 1,
        "status": "pass_independent_all1000_row_level_audit_and_byte_identical_rerun",
        "repository_head_at_audit": head,
        "source_artifact": str(source.relative_to(repo)),
        "source_sha256s_sha256": sha256_file(source / "SHA256SUMS"),
        "source_files_checked_by_sha256": checked_files,
        **independent,
        "initial200_ranked_rows_proven_identical": initial_ranked,
        "initial200_replica_category_reductions_proven_identical": initial_reduced,
        "deterministic_rerun": {
            "status": "pass_every_output_file_byte_identical",
            **rerun,
        },
        "pilot_results_used": False,
        "nominal_deployment_candidate_changed": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "next": "freeze_all1000_aggregation_and_independent_audit_before_train_performance",
    }
    write_audit_output(output, payload)
    print("HH4B_ALL1000_INDEPENDENT_AUDIT=PASS")
    print(f"RANKED_STRUCTURE_RESULTS_AUDITED={independent['ranked_structure_results_audited']}")
    print(f"FOLD_LEVEL_WINNERS_RECOMPUTED={independent['fold_level_winners_independently_selected']}")
    print(f"REPLICA_CATEGORY_REDUCTIONS_RECOMPUTED={independent['replica_category_reductions_independently_recomputed']}")
    print(f"DETERMINISTIC_RERUN_FILES_BYTE_IDENTICAL={rerun['file_count']}")
    print("NOMINAL_DEPLOYMENT_CANDIDATE_CHANGED=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
