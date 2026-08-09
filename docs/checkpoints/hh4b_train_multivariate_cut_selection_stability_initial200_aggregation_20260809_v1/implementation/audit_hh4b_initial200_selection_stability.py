#!/usr/bin/env python3
"""Independently audit the frozen HH4b initial-200 aggregation products."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


EXPECTED_INPUT_SHA256 = "47df422f39a50ac2f008031d3222ef04556760526abf160db57485861f3db056"
EXPECTED_PROTOCOL_SHA256 = "a856d9027cc1d2e11a9d60e72b4104b5b8c24ce61acf43def0d4d487ab867eea"
EXPECTED_AGGREGATOR_SHA256 = "db501c98f24e0a5b2725acbee45705925e19a45c9330af74389856df4a4126c8"
TARGET_SIGNAL_EFFICIENCY = 0.585957
CATEGORIES = ("exact3tag", "ge4tag")
NOMINAL_STRUCTURES = {
    "exact3tag": "radial_mass__category__plus_ht_candidate_jets",
    "ge4tag": "radial_mass__category__plus_mhh_and_abs_h_delta_eta",
}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_bool(value: str, field: str) -> bool:
    require(value in {"True", "False"}, f"invalid boolean {field}: {value!r}")
    return value == "True"


def parse_float(value: str, field: str) -> float:
    parsed = float(value)
    require(math.isfinite(parsed), f"non-finite {field}")
    return parsed


def independent_key(row: Mapping[str, str]) -> tuple[Any, ...]:
    feasible = parse_bool(row["pooled_inner_oof_feasible"], "feasible")
    signal = parse_float(row["pooled_signal_efficiency"], "signal efficiency")
    background = parse_float(row["pooled_background_efficiency"], "background efficiency")
    cut_count = int(row["cut_count"])
    if feasible:
        return (0, background, signal - TARGET_SIGNAL_EFFICIENCY, cut_count, row["structure_id"])
    return (1, -signal, background, cut_count, row["structure_id"])


def coordinate_median(thresholds: Sequence[Mapping[str, float]]) -> dict[str, float]:
    require(bool(thresholds), "empty threshold reduction")
    keys = set(thresholds[0])
    require(all(set(item) == keys for item in thresholds), "threshold-coordinate mismatch")
    return {key: float(statistics.median(item[key] for item in thresholds)) for key in sorted(keys)}


def verify_original_checksums(output_dir: Path) -> dict[str, str]:
    checksum_path = output_dir / "SHA256SUMS"
    require(checksum_path.is_file(), "missing SHA256SUMS")
    checksums: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        require(relative.startswith("./"), f"invalid checksum path: {relative}")
        path = output_dir / relative[2:]
        require(path.is_file() and not path.is_symlink(), f"missing checksum target: {relative}")
        actual = sha256_file(path)
        require(actual == expected, f"checksum mismatch for {relative}: {actual}")
        checksums[relative[2:]] = expected
    return checksums


def verify_reproduction(output_dir: Path, reproduction_dir: Path, checksums: Mapping[str, str]) -> str:
    require(reproduction_dir.is_dir() and not reproduction_dir.is_symlink(), "missing reproduction directory")
    for relative, expected in checksums.items():
        path = reproduction_dir / relative
        require(path.is_file(), f"reproduction missing {relative}")
        require(sha256_file(path) == expected, f"reproduction differs for {relative}")
    require(
        sha256_file(output_dir / "SHA256SUMS") == sha256_file(reproduction_dir / "SHA256SUMS"),
        "reproduction SHA256SUMS differs",
    )
    return sha256_file(reproduction_dir / "SHA256SUMS")


def audit(output_dir: Path, repo: Path, reproduction_dir: Path) -> dict[str, Any]:
    checksums = verify_original_checksums(output_dir)
    reproduction_sums_sha = verify_reproduction(output_dir, reproduction_dir, checksums)

    manifest_path = output_dir / "manifests/aggregation_manifest.json"
    summary_path = output_dir / "initial200_stability_summary.json"
    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    require(manifest.get("status") == "pass_deterministic_initial200_aggregation_manifest",
            "manifest status mismatch")
    require(summary.get("status") == "pass_initial200_predeclared_selection_stability_aggregation",
            "summary status mismatch")
    require(summary.get("pilot_results_used") is False, "pilot leakage flag")
    require(summary.get("nominal_deployment_candidate_changed") is False, "nominal-change flag")
    require(summary.get("validation_payloads_opened") == 0, "validation count nonzero")
    require(summary.get("test_payloads_opened") == 0, "test count nonzero")
    require(summary.get("fold_level_winner_rule", {}).get("outer_fold_used_for_selection") is False,
            "outer fold selection flag")

    require(manifest["inputs"].values(), "manifest input map empty")
    require(EXPECTED_INPUT_SHA256 in manifest["inputs"].values(), "input inventory SHA missing")
    require(EXPECTED_PROTOCOL_SHA256 in manifest["inputs"].values(), "protocol SHA missing")
    require(manifest["aggregation_implementation"]["sha256"] == EXPECTED_AGGREGATOR_SHA256,
            "aggregator SHA mismatch")
    require(sha256_file(repo / manifest["aggregation_implementation"]["path"]) == EXPECTED_AGGREGATOR_SHA256,
            "current aggregator bytes differ from manifest")
    for relative, identity in manifest["output_files"].items():
        path = output_dir / relative
        require(path.stat().st_size == identity["bytes"], f"manifest size mismatch for {relative}")
        require(sha256_file(path) == identity["sha256"], f"manifest SHA mismatch for {relative}")

    ranked = load_tsv(output_dir / "tables/ranked_structure_results.tsv")
    winners = load_tsv(output_dir / "tables/fold_level_winners.tsv")
    replicas = load_tsv(output_dir / "tables/replica_category_results.tsv")
    require(len(ranked) == 54000, f"ranked row count mismatch: {len(ranked)}")
    require(len(winners) == 2000, f"winner row count mismatch: {len(winners)}")
    require(len(replicas) == 400, f"replica row count mismatch: {len(replicas)}")

    ranked_groups: dict[tuple[int, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in ranked:
        key = (int(row["replica"]), row["category_id"], int(row["outer_fold"]))
        ranked_groups[key].append(row)
        require(parse_bool(row["outer_fold_used_for_selection"], "outer-fold selection") is False,
                "outer fold used for selection")
        require(parse_bool(row["pooled_support_pass"], "pooled support") is True,
                "support-fail result found")
        stored_key = tuple(load_json_string(row["ranking_key_json"]))
        calculated_key = independent_key(row)
        require(stored_key == calculated_key, f"ranking-key mismatch in group {key}")

    require(len(ranked_groups) == 2000, "ranked group count mismatch")
    rank_one_by_group: dict[tuple[int, str, int], dict[str, str]] = {}
    for key, rows in ranked_groups.items():
        rows.sort(key=lambda row: int(row["rank"]))
        require([int(row["rank"]) for row in rows] == list(range(1, 28)), f"rank coverage mismatch {key}")
        keys = [independent_key(row) for row in rows]
        require(keys == sorted(keys), f"ranking order mismatch {key}")
        require(sum(parse_bool(row["selected_as_fold_winner"], "selected") for row in rows) == 1,
                f"winner-flag count mismatch {key}")
        require(parse_bool(rows[0]["selected_as_fold_winner"], "selected"), f"rank 1 not selected {key}")
        rank_one_by_group[key] = rows[0]

    winner_by_group: dict[tuple[int, str, int], dict[str, str]] = {}
    for row in winners:
        key = (int(row["replica"]), row["category_id"], int(row["outer_fold"]))
        require(key not in winner_by_group, f"duplicate winner group {key}")
        require(row == rank_one_by_group[key], f"winner table differs from ranking ledger at {key}")
        winner_by_group[key] = row
    require(set(winner_by_group) == set(ranked_groups), "winner group coverage mismatch")

    replica_by_key: dict[tuple[int, str], dict[str, str]] = {}
    selected_counts: dict[str, Counter[str]] = {category: Counter() for category in CATEGORIES}
    tie_counts = Counter()
    infeasible_winner_counts = Counter()
    for row in replicas:
        key = (int(row["replica"]), row["category_id"])
        require(key not in replica_by_key, f"duplicate replica/category row {key}")
        replica_by_key[key] = row
        fold_rows = [winner_by_group[(key[0], key[1], fold)] for fold in range(5)]
        structure_ids = [item["structure_id"] for item in fold_rows]
        require(load_json_string(row["five_fold_winner_structure_ids_json"]) == structure_ids,
                f"five-fold structure list mismatch {key}")
        counts = Counter(structure_ids)
        stored_counts = load_json_string(row["five_fold_winner_counts_by_structure_json"])
        require(stored_counts == dict(sorted(counts.items())), f"winner count map mismatch {key}")
        maximum = max(counts.values())
        tied = sorted(structure for structure, count in counts.items() if count == maximum)
        selected = tied[0]
        supporting = [fold for fold, structure in enumerate(structure_ids) if structure == selected]
        thresholds = [load_json_string(fold_rows[fold]["refit_thresholds_json"]) for fold in supporting]
        medians = coordinate_median(thresholds)
        require(int(row["maximum_winner_count"]) == maximum, f"maximum count mismatch {key}")
        require(math.isclose(float(row["maximum_winner_frequency"]), maximum / 5.0,
                             rel_tol=0.0, abs_tol=1e-15), f"maximum frequency mismatch {key}")
        require(parse_bool(row["unique_modal_family"], "unique mode") == (len(tied) == 1),
                f"unique-mode mismatch {key}")
        require(parse_bool(row["tie_resolution_applied"], "tie flag") == (len(tied) > 1),
                f"tie flag mismatch {key}")
        require(load_json_string(row["tied_maximum_frequency_structure_ids_json"]) == tied,
                f"tied IDs mismatch {key}")
        require(row["replica_selected_structure_id"] == selected, f"selected structure mismatch {key}")
        require(load_json_string(row["replica_selected_structure_supporting_outer_folds_json"]) == supporting,
                f"supporting folds mismatch {key}")
        stored_medians = load_json_string(
            row["replica_selected_structure_coordinatewise_median_thresholds_json"]
        )
        require(stored_medians == medians, f"coordinate median mismatch {key}")
        require(parse_bool(row["pilot_result_used"], "pilot flag") is False, f"pilot flag {key}")
        require(parse_bool(row["nominal_deployment_candidate_changed"], "nominal flag") is False,
                f"nominal flag {key}")
        require(int(row["validation_payloads_opened"]) == 0 and int(row["test_payloads_opened"]) == 0,
                f"sealed-data count mismatch {key}")
        selected_counts[key[1]][selected] += 1
        tie_counts[key[1]] += int(len(tied) > 1)
        infeasible_winner_counts[key[1]] += sum(
            not parse_bool(item["pooled_inner_oof_feasible"], "winner feasible") for item in fold_rows
        )
    require(set(replica_by_key) == {(replica, category) for replica in range(200) for category in CATEGORIES},
            "replica/category coverage mismatch")

    expected_summary = {
        "exact3tag": {
            "modal": "asymmetric_rectangular_mass__category__plus_abs_h_delta_eta",
            "modal_count": 37,
            "second_count": 29,
            "nominal_count": 24,
            "ties": 79,
            "infeasible_winners": 538,
        },
        "ge4tag": {
            "modal": "asymmetric_rectangular_mass__category__mass_only",
            "modal_count": 52,
            "second_count": 26,
            "nominal_count": 5,
            "ties": 72,
            "infeasible_winners": 156,
        },
    }
    for category in CATEGORIES:
        ordered = sorted(selected_counts[category].items(), key=lambda item: (-item[1], item[0]))
        expected = expected_summary[category]
        require(ordered[0] == (expected["modal"], expected["modal_count"]), f"modal mismatch {category}")
        require(ordered[1][1] == expected["second_count"], f"second count mismatch {category}")
        require(selected_counts[category][NOMINAL_STRUCTURES[category]] == expected["nominal_count"],
                f"nominal recovery mismatch {category}")
        require(tie_counts[category] == expected["ties"], f"tie count mismatch {category}")
        require(infeasible_winner_counts[category] == expected["infeasible_winners"],
                f"infeasible winner count mismatch {category}")
        summary_category = summary["categories"][category]
        require(summary_category["modal_replica_selected_structure_id"] == expected["modal"],
                f"summary modal mismatch {category}")
        require(summary_category["modal_replica_selected_count"] == expected["modal_count"],
                f"summary modal count mismatch {category}")
        require(summary_category["nominal_structure_recovery_count"] == expected["nominal_count"],
                f"summary nominal count mismatch {category}")

    for table, sidecar in (
        ("tables/structure_frequencies.tsv", "figure_data/selection_structure_frequencies.tsv"),
        ("tables/optional_variable_inclusion_frequencies.tsv",
         "figure_data/optional_variable_inclusion_frequencies.tsv"),
        ("tables/modal_count_distribution.tsv", "figure_data/modal_count_distribution.tsv"),
        ("tables/conditional_replica_thresholds.tsv", "figure_data/conditional_replica_thresholds.tsv"),
    ):
        require((output_dir / table).read_bytes() == (output_dir / sidecar).read_bytes(),
                f"plot sidecar differs from table: {sidecar}")

    return {
        "aggregation_manifest_sha256": sha256_file(manifest_path),
        "aggregation_summary_sha256": sha256_file(summary_path),
        "aggregator_sha256": EXPECTED_AGGREGATOR_SHA256,
        "byte_for_byte_reproduction_verified": True,
        "category_results": {
            category: {
                "infeasible_fold_winner_count": infeasible_winner_counts[category],
                "modal_replica_selected_count": expected_summary[category]["modal_count"],
                "modal_replica_selected_structure_id": expected_summary[category]["modal"],
                "nominal_structure_recovery_count": expected_summary[category]["nominal_count"],
                "tie_resolution_count": tie_counts[category],
            }
            for category in CATEGORIES
        },
        "fold_level_winners_audited": 2000,
        "nominal_deployment_candidate_changed": False,
        "pilot_results_used": False,
        "ranked_structure_results_audited": 54000,
        "replica_category_results_audited": 400,
        "reproduction_original_sha256sums_sha256": reproduction_sums_sha,
        "schema_version": 1,
        "status": "pass_initial200_stability_aggregation_independent_audit",
        "test_payloads_opened": 0,
        "validation_payloads_opened": 0,
    }


def load_json_string(value: str) -> Any:
    return json.loads(value)


def regenerate_checksums(output_dir: Path) -> None:
    paths = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    with (output_dir / "SHA256SUMS").open("w", encoding="utf-8", newline="") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  ./{path.relative_to(output_dir).as_posix()}\n")


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
    )
    parser.add_argument(
        "--reproduction-dir",
        type=Path,
        default=Path("/tmp/hh4b_initial200_stability_reprocheck_20260809"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output_dir = output_dir.resolve()
    reproduction_dir = args.reproduction_dir.resolve()
    require(output_dir.is_relative_to(repo), "output directory is outside repository")
    require(output_dir.is_dir() and not output_dir.is_symlink(), "missing output directory")
    audit_json_path = output_dir / "manifests/aggregation_validation.json"
    audit_txt_path = output_dir / "manifests/aggregation_validation.txt"
    require(not audit_json_path.exists() and not audit_txt_path.exists(), "audit outputs already exist")

    result = audit(output_dir, repo, reproduction_dir)
    write_json(audit_json_path, result)
    audit_txt_path.write_text(
        "AUDIT_STATUS=PASS\n"
        "RANKED_STRUCTURE_RESULTS_AUDITED=54000\n"
        "FOLD_LEVEL_WINNERS_AUDITED=2000\n"
        "REPLICA_CATEGORY_RESULTS_AUDITED=400\n"
        "BYTE_FOR_BYTE_REPRODUCTION=PASS\n"
        "PILOT_RESULTS_USED=FALSE\n"
        "NOMINAL_DEPLOYMENT_CANDIDATE_CHANGED=FALSE\n"
        "VALIDATION_PAYLOADS_OPENED=0\n"
        "TEST_PAYLOADS_OPENED=0\n",
        encoding="utf-8",
    )
    regenerate_checksums(output_dir)
    verify_original_checksums(output_dir)
    print(audit_txt_path.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
