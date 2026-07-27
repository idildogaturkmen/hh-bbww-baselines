#!/usr/bin/env python3
"""Freeze the canonical HH→4b BDT choice from train-only aggregate evidence.

This gate verifies and reads only four committed aggregate checkpoints.  It
does not open candidate files, train models, write predictions, tune
hyperparameters, or calculate physical yields or significances.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DEFAULT = (
    REPOSITORY_ROOT / "configs/baselines/hh4b_bdt_model_choice_v1.yaml"
)
STATUS = "hh4b_bdt_train_only_model_choice_frozen"
NEXT_GATE = "fit_frozen_hh4b_bdt_models_and_evaluate_validation_once"
CUT_TARGET = 0.585957314769
FLOAT_TOLERANCE = 5.0e-12

PRIMARY = "global_v1_mass_aware"
SECONDARY = "categorized_cms_inspired_mass_aware"
GLOBAL_BLIND = "global_v1_explicit_dijet_mass_plane_blind"
CATEGORY_SPLIT = "categorized_v1_features_mass_aware_ablation"
CATEGORIZED_BLIND = (
    "categorized_cms_inspired_explicit_dijet_mass_plane_blind"
)
OPTIMIZED_CUT = "optimized_cut"

REQUIRED_CHECKPOINT_FILES = (
    "README.md",
    "checkpoint.json",
    "summary.json",
    "model_choice.tsv",
    "model_choice.md",
    "model_choice.tex",
    "train_only_performance_comparison.tsv",
    "train_only_performance_comparison.md",
    "train_only_performance_comparison.tex",
    "frozen_validation_plan.tsv",
    "frozen_validation_plan.md",
    "frozen_validation_plan.tex",
    "frozen_working_points.tsv",
    "frozen_working_points.md",
    "frozen_working_points.tex",
    "source_checkpoint_inventory.tsv",
    "decision_rule_audit.tsv",
    "SHA256SUMS",
)

MODEL_LATEX_LABELS = {
    PRIMARY: r"Global v1 mass-aware BDT",
    SECONDARY: r"Categorized CMS-inspired mass-aware BDT",
    GLOBAL_BLIND: r"Global v1 explicit dijet-mass-plane-blind BDT",
    CATEGORY_SPLIT: r"Categorized v1-feature mass-aware ablation",
    CATEGORIZED_BLIND: (
        r"Categorized CMS-inspired explicit dijet-mass-plane-blind BDT"
    ),
    OPTIMIZED_CUT: r"Optimized $R_{HH}<34$ cut",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def format_plain(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.15g}"
    return str(value)


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
        "→": r"$\to$",
        "≥": r"$\geq$",
        "≤": r"$\leq$",
    }
    return "".join(
        replacements.get(character, character)
        for character in format_plain(value)
    )


def latex_table_value(field: str, value: Any) -> str:
    plain = format_plain(value)
    if field in {"model", "method", "model_or_method"}:
        return MODEL_LATEX_LABELS.get(plain, latex_escape(plain))
    if field in {
        "target_weighted_signal_efficiency",
        "weighted_oof_auc",
        "low_mhh_weighted_oof_auc",
        "high_mhh_weighted_oof_auc",
        "cut_matched_weighted_signal_efficiency",
        "cut_matched_weighted_background_efficiency",
        "global_threshold",
        "low_mhh_threshold",
        "high_mhh_threshold",
    } and plain:
        return f"${latex_escape(plain)}$"
    if field == "selection" and plain == "r_hh_125_125 < 34":
        return r"$R_{HH}<34$"
    return latex_escape(plain)


def write_table_bundle(
    directory: Path,
    name: str,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    publication_fields: Sequence[str],
    caption: str,
    label: str,
    latex_headers: Mapping[str, str],
) -> tuple[Path, Path, Path]:
    materialized = [dict(row) for row in rows]
    fields = tuple(fields)
    publication_fields = tuple(publication_fields)
    if not fields or not publication_fields:
        raise ValueError("table fields must be nonempty")
    if not set(publication_fields).issubset(fields):
        raise ValueError("publication fields must be a subset of TSV fields")
    for row in materialized:
        extras = set(row) - set(fields)
        if extras:
            raise ValueError(
                f"{name} row has unexpected fields: {sorted(extras)}"
            )

    tsv_path = directory / f"{name}.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {field: format_plain(row.get(field, "")) for field in fields}
            )

    markdown_path = directory / f"{name}.md"
    markdown_lines = [
        "| "
        + " | ".join(field.replace("_", " ") for field in publication_fields)
        + " |",
        "|" + "|".join("---" for _ in publication_fields) + "|",
    ]
    for row in materialized:
        cells = [
            format_plain(row.get(field, ""))
            .replace("\\", r"\\")
            .replace("|", r"\|")
            .replace("\n", " ")
            for field in publication_fields
        ]
        markdown_lines.append("| " + " | ".join(cells) + " |")
    markdown_path.write_text(
        "\n".join(markdown_lines) + "\n", encoding="utf-8"
    )

    latex_path = directory / f"{name}.tex"
    align = "l" + "r" * (len(publication_fields) - 1)
    latex_lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(
            latex_headers.get(field, latex_escape(field.replace("_", " ")))
            for field in publication_fields
        )
        + r" \\",
        r"\midrule",
    ]
    for row in materialized:
        latex_lines.append(
            " & ".join(
                latex_table_value(field, row.get(field, ""))
                for field in publication_fields
            )
            + r" \\"
        )
    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    latex_path.write_text(
        "\n".join(latex_lines) + "\n", encoding="utf-8"
    )
    return tsv_path, markdown_path, latex_path


def verify_sha256_manifest(directory: Path) -> int:
    manifest = directory / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing SHA256SUMS: {manifest}")
    checked = 0
    failures: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"invalid SHA256SUMS line in {manifest}: {line}")
        expected, relative = parts
        relative = relative.strip().lstrip("*")
        target = directory / relative
        if not target.is_file() or sha256_file(target) != expected:
            failures.append(relative)
        checked += 1
    if failures:
        raise ValueError(
            f"checkpoint SHA256 verification failed in {directory}: "
            f"{failures}"
        )
    return checked


def write_sha256_manifest(directory: Path) -> Path:
    targets = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    manifest = directory / "SHA256SUMS"
    manifest.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}" for path in targets
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def require_safe_output_path(path: Path, allowed_parent: Path) -> None:
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    if resolved == parent or parent not in resolved.parents:
        raise ValueError(
            f"output must be a child of {allowed_parent}: {resolved}"
        )


def prepare_exact_directory(
    path: Path, *, allowed_parent: Path, overwrite: bool
) -> None:
    require_safe_output_path(path, allowed_parent)
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"exact output exists; pass --overwrite to replace it: {path}"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def require_close(
    observed: Any, expected: Any, label: str, tolerance: float = FLOAT_TOLERANCE
) -> None:
    if abs(float(observed) - float(expected)) > tolerance:
        raise ValueError(
            f"{label} changed: expected {expected}, observed {observed}"
        )


def require_source_summary_pass(
    summary: Mapping[str, Any], required_status: str, label: str
) -> None:
    if summary.get("status") != required_status:
        raise ValueError(
            f"{label} summary status changed: {summary.get('status')}"
        )
    failed = summary.get("failed_pass_conditions", [])
    if failed:
        raise ValueError(f"{label} reports failed pass conditions: {failed}")
    pass_conditions = summary.get("pass_conditions", {})
    if isinstance(pass_conditions, dict) and not all(pass_conditions.values()):
        raise ValueError(f"{label} contains a false pass condition")
    if summary.get("validation_candidate_files_opened", 0) != 0:
        raise ValueError(f"{label} opened validation candidate files")
    if summary.get("test_candidate_files_opened", 0) != 0:
        raise ValueError(f"{label} opened test candidate files")


def verify_source_checkpoints(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    inventory: list[dict[str, Any]] = []
    directories: dict[str, Path] = {}
    for checkpoint_name, specification in config["inputs"].items():
        directory = REPOSITORY_ROOT / specification["checkpoint_dir"]
        manifest = directory / "SHA256SUMS"
        observed_manifest_hash = sha256_file(manifest)
        expected_manifest_hash = specification["sha256sums_sha256"]
        if observed_manifest_hash != expected_manifest_hash:
            raise ValueError(
                f"{checkpoint_name} SHA256SUMS digest changed: "
                f"{observed_manifest_hash}"
            )
        files_verified = verify_sha256_manifest(directory)
        checkpoint = read_json(directory / "checkpoint.json")
        summary = read_json(directory / "summary.json")
        required_status = specification["required_status"]
        if checkpoint.get("status") != required_status:
            raise ValueError(
                f"{checkpoint_name} checkpoint status changed: "
                f"{checkpoint.get('status')}"
            )
        require_source_summary_pass(summary, required_status, checkpoint_name)
        inventory.append(
            {
                "checkpoint": checkpoint_name,
                "checkpoint_dir": specification["checkpoint_dir"],
                "required_status": required_status,
                "observed_status": checkpoint["status"],
                "sha256sums_sha256": observed_manifest_hash,
                "manifest_entries_verified": files_verified,
                "manifest_verified": True,
                "passing_status_verified": True,
                "status": "pass",
            }
        )
        directories[checkpoint_name] = directory
    return inventory, directories


def find_unique(
    rows: Sequence[Mapping[str, str]],
    predicate: Callable[[Mapping[str, str]], bool],
    label: str,
) -> dict[str, str]:
    matches = [dict(row) for row in rows if predicate(row)]
    if len(matches) != 1:
        raise ValueError(f"{label}: expected one row, found {len(matches)}")
    if matches[0].get("status", "pass") != "pass":
        raise ValueError(f"{label}: source row does not pass")
    return matches[0]


def at_target(row: Mapping[str, str], target: float) -> bool:
    return (
        abs(float(row["target_weighted_signal_efficiency"]) - target)
        < 1.0e-9
    )


def collect_evidence(
    config: Mapping[str, Any], directories: Mapping[str, Path]
) -> dict[str, Any]:
    v1_dir = directories["v1_grouped_cv"]
    v2_dir = directories["v2_cms_inspired_grouped_cv"]
    v1_metrics = read_tsv(v1_dir / "oof_metrics_summary.tsv")
    v2_category_metrics = read_tsv(v2_dir / "category_oof_metrics.tsv")
    methods = read_tsv(v2_dir / "method_comparison.tsv")
    bootstrap_rows = read_tsv(v2_dir / "member_bootstrap_comparison.tsv")

    global_metric = find_unique(
        v1_metrics,
        lambda row: row["variant"] == "mass_aware",
        "global v1 mass-aware OOF metric",
    )
    global_blind_metric = find_unique(
        v1_metrics,
        lambda row: row["variant"] == "explicit_dijet_mass_plane_blind",
        "global v1 mass-plane-blind OOF metric",
    )

    def category_metric(strategy: str, category: str) -> dict[str, str]:
        return find_unique(
            v2_category_metrics,
            lambda row: (
                row["strategy"] == strategy
                and row["category"] == category
                and row["row_type"] == "category_summary"
            ),
            f"{strategy} {category} OOF metric",
        )

    method_at_cut = {
        method: find_unique(
            methods,
            lambda row, method=method: (
                row["method"] == method and at_target(row, CUT_TARGET)
            ),
            f"{method} cut-matched operating point",
        )
        for method in (
            PRIMARY,
            SECONDARY,
            GLOBAL_BLIND,
            CATEGORY_SPLIT,
            CATEGORIZED_BLIND,
            OPTIMIZED_CUT,
        )
    }
    bootstrap = find_unique(
        bootstrap_rows,
        lambda row: row["comparison"] == "v2 minus global v1",
        "v2 versus global v1 member bootstrap",
    )
    evidence = {
        "global_metric": global_metric,
        "global_blind_metric": global_blind_metric,
        "category_metrics": {
            strategy: {
                category: category_metric(strategy, category)
                for category in ("low_mhh", "high_mhh")
            }
            for strategy in (SECONDARY, CATEGORY_SPLIT, CATEGORIZED_BLIND)
        },
        "method_at_cut": method_at_cut,
        "bootstrap": bootstrap,
        "v1_working_points": read_tsv(v1_dir / "working_points.tsv"),
        "v2_operating_points": read_tsv(
            v2_dir / "categorized_operating_points.tsv"
        ),
    }
    verify_expected_metrics(config, evidence)
    return evidence


def verify_expected_metrics(
    config: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    expected = config["model_choice"]["expected_train_only_metrics"]
    require_close(
        evidence["global_metric"]["weighted_roc_auc"],
        expected["primary_weighted_oof_auc"],
        "primary weighted OOF AUC",
    )
    require_close(
        evidence["method_at_cut"][PRIMARY][
            "weighted_background_efficiency"
        ],
        expected["primary_cut_matched_weighted_background_efficiency"],
        "primary cut-matched weighted background efficiency",
    )
    require_close(
        evidence["category_metrics"][SECONDARY]["low_mhh"][
            "weighted_roc_auc"
        ],
        expected["secondary_low_mhh_weighted_oof_auc"],
        "secondary low-mHH weighted OOF AUC",
    )
    require_close(
        evidence["category_metrics"][SECONDARY]["high_mhh"][
            "weighted_roc_auc"
        ],
        expected["secondary_high_mhh_weighted_oof_auc"],
        "secondary high-mHH weighted OOF AUC",
    )
    require_close(
        evidence["method_at_cut"][SECONDARY][
            "weighted_background_efficiency"
        ],
        expected["secondary_cut_matched_weighted_background_efficiency"],
        "secondary cut-matched weighted background efficiency",
    )
    require_close(
        evidence["method_at_cut"][OPTIMIZED_CUT][
            "weighted_signal_efficiency"
        ],
        expected["optimized_cut_weighted_signal_efficiency"],
        "optimized-cut weighted signal efficiency",
        tolerance=1.0e-11,
    )
    require_close(
        evidence["method_at_cut"][OPTIMIZED_CUT][
            "weighted_background_efficiency"
        ],
        expected["optimized_cut_weighted_background_efficiency"],
        "optimized-cut weighted background efficiency",
    )
    for source_field, expected_field in (
        ("median_difference", "bootstrap_median_difference"),
        ("percentile_2_5", "bootstrap_percentile_2_5"),
        ("percentile_97_5", "bootstrap_percentile_97_5"),
        (
            "fraction_favoring_first_method",
            "bootstrap_fraction_favoring_v2",
        ),
    ):
        require_close(
            evidence["bootstrap"][source_field],
            expected[expected_field],
            expected_field,
        )


def build_model_choice_rows() -> list[dict[str, Any]]:
    return [
        {
            "model_or_method": PRIMARY,
            "decision_class": "primary_nominal_bdt",
            "role": "canonical_nominal_model",
            "nominal": True,
            "validation_predeclared": True,
            "selection_basis": (
                "strong cut-matched rejection; simpler global strategy; "
                "v2 gain negligible and bootstrap-unstable"
            ),
            "status": "frozen",
        },
        {
            "model_or_method": SECONDARY,
            "decision_class": "secondary_categorized_bdt",
            "role": "advanced CMS-inspired categorized alternative",
            "nominal": False,
            "validation_predeclared": True,
            "selection_basis": (
                "predeclared categorized alternative; not validation-selectable"
            ),
            "status": "frozen",
        },
        {
            "model_or_method": GLOBAL_BLIND,
            "decision_class": "diagnostic_ablation",
            "role": "global explicit dijet-mass-plane-blind diagnostic",
            "nominal": False,
            "validation_predeclared": True,
            "selection_basis": "diagnostic only",
            "status": "retained_non_nominal",
        },
        {
            "model_or_method": CATEGORY_SPLIT,
            "decision_class": "diagnostic_ablation",
            "role": "category-split-only diagnostic",
            "nominal": False,
            "validation_predeclared": False,
            "selection_basis": "train-only diagnostic only",
            "status": "retained_non_nominal",
        },
        {
            "model_or_method": CATEGORIZED_BLIND,
            "decision_class": "diagnostic_ablation",
            "role": "categorized explicit dijet-mass-plane-blind diagnostic",
            "nominal": False,
            "validation_predeclared": False,
            "selection_basis": "train-only diagnostic only",
            "status": "retained_non_nominal",
        },
        {
            "model_or_method": OPTIMIZED_CUT,
            "decision_class": "cut_baseline",
            "role": "optimized R_HH cut baseline",
            "nominal": False,
            "validation_predeclared": True,
            "selection_basis": "frozen benchmark baseline",
            "status": "retained",
        },
    ]


def build_performance_rows(
    evidence: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    definitions = (
        (PRIMARY, "primary_nominal"),
        (SECONDARY, "secondary_categorized"),
        (GLOBAL_BLIND, "diagnostic_ablation"),
        (CATEGORY_SPLIT, "diagnostic_ablation"),
        (CATEGORIZED_BLIND, "diagnostic_ablation"),
        (OPTIMIZED_CUT, "cut_baseline"),
    )
    for method, decision_class in definitions:
        cut_row = evidence["method_at_cut"][method]
        row: dict[str, Any] = {
            "method": method,
            "decision_class": decision_class,
            "weighted_oof_auc": "",
            "low_mhh_weighted_oof_auc": "",
            "high_mhh_weighted_oof_auc": "",
            "cut_matched_weighted_signal_efficiency": cut_row[
                "weighted_signal_efficiency"
            ],
            "cut_matched_weighted_background_efficiency": cut_row[
                "weighted_background_efficiency"
            ],
            "bootstrap_median_difference_vs_global_v1": "",
            "bootstrap_percentile_2_5_vs_global_v1": "",
            "bootstrap_percentile_97_5_vs_global_v1": "",
            "bootstrap_fraction_favoring_method": "",
            "interpretation": "",
            "status": "pass",
        }
        if method == PRIMARY:
            row["weighted_oof_auc"] = evidence["global_metric"][
                "weighted_roc_auc"
            ]
            row["interpretation"] = (
                "frozen primary; substantially better rejection than cut"
            )
        elif method == GLOBAL_BLIND:
            row["weighted_oof_auc"] = evidence["global_blind_metric"][
                "weighted_roc_auc"
            ]
            row["interpretation"] = "non-nominal diagnostic"
        elif method in (SECONDARY, CATEGORY_SPLIT, CATEGORIZED_BLIND):
            row["low_mhh_weighted_oof_auc"] = evidence["category_metrics"][
                method
            ]["low_mhh"]["weighted_roc_auc"]
            row["high_mhh_weighted_oof_auc"] = evidence["category_metrics"][
                method
            ]["high_mhh"]["weighted_roc_auc"]
            row["interpretation"] = (
                "predeclared secondary; negligible nominal-point gain with "
                "bootstrap interval spanning zero"
                if method == SECONDARY
                else "non-nominal diagnostic"
            )
        else:
            row["interpretation"] = "optimized R_HH < 34 cut baseline"
        if method == SECONDARY:
            bootstrap = evidence["bootstrap"]
            row["bootstrap_median_difference_vs_global_v1"] = bootstrap[
                "median_difference"
            ]
            row["bootstrap_percentile_2_5_vs_global_v1"] = bootstrap[
                "percentile_2_5"
            ]
            row["bootstrap_percentile_97_5_vs_global_v1"] = bootstrap[
                "percentile_97_5"
            ]
            row["bootstrap_fraction_favoring_method"] = bootstrap[
                "fraction_favoring_first_method"
            ]
        rows.append(row)
    return rows


def build_validation_plan_rows(
    config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    plan = config["validation_plan"]
    for model in plan["models"]:
        rows.append(
            {
                "order": model["order"],
                "candidate_type": "model",
                "model_or_method": model["model"],
                "role": model["role"],
                "selection": "frozen model contract",
                "fit_population": plan["fit_population"],
                "validation_evaluations": plan["validation_evaluations"],
                "validation_model_selection_authorized": False,
                "post_validation_hyperparameter_tuning_authorized": False,
                "status": "predeclared",
            }
        )
    for method in plan["cut_baselines"]:
        rows.append(
            {
                "order": method["order"],
                "candidate_type": "cut_baseline",
                "model_or_method": method["method"],
                "role": method["role"],
                "selection": method["expression"],
                "fit_population": "not_applicable",
                "validation_evaluations": plan["validation_evaluations"],
                "validation_model_selection_authorized": False,
                "post_validation_hyperparameter_tuning_authorized": False,
                "status": "predeclared",
            }
        )
    return sorted(rows, key=lambda row: int(row["order"]))


def build_working_point_rows(
    config: Mapping[str, Any], evidence: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    targets = [
        float(target)
        for target in config["frozen_working_points"][
            "target_weighted_signal_efficiencies"
        ]
    ]
    for target in targets:
        source = find_unique(
            evidence["v1_working_points"],
            lambda row, target=target: (
                row["variant"] == "mass_aware" and at_target(row, target)
            ),
            f"global v1 working point {target}",
        )
        rows.append(
            {
                "working_point_set": "global_v1_mass_aware_single_thresholds",
                "model": PRIMARY,
                "threshold_contract": "single_global_threshold",
                "target_weighted_signal_efficiency": f"{target:.12g}",
                "global_threshold": source["oof_score_threshold"],
                "low_mhh_threshold": "",
                "high_mhh_threshold": "",
                "source_table": (
                    "hh4b_bdt_v1_grouped_cv_20260725_v1/"
                    "working_points.tsv"
                ),
                "status": "frozen",
            }
        )
    for target in targets:
        source = find_unique(
            evidence["v2_operating_points"],
            lambda row, target=target: (
                row["method"] == SECONDARY and at_target(row, target)
            ),
            f"categorized v2 working point {target}",
        )
        rows.append(
            {
                "working_point_set": (
                    "categorized_cms_inspired_mass_aware_category_thresholds"
                ),
                "model": SECONDARY,
                "threshold_contract": (
                    "separate_low_and_high_mhh_thresholds"
                ),
                "target_weighted_signal_efficiency": f"{target:.12g}",
                "global_threshold": "",
                "low_mhh_threshold": source["low_mhh_threshold"],
                "high_mhh_threshold": source["high_mhh_threshold"],
                "source_table": (
                    "hh4b_bdt_v2_cms_inspired_grouped_cv_20260726_v1/"
                    "categorized_operating_points.tsv"
                ),
                "status": "frozen",
            }
        )
    return rows


def audit_row(
    condition: str,
    expected: Any,
    observed: Any,
    passed: bool,
    evidence: str,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "evidence": evidence,
    }


def build_decision_rule_audit(
    config: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    model_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    working_point_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rules = config["model_choice"]["decision_rules"]
    primary_background = float(
        evidence["method_at_cut"][PRIMARY]["weighted_background_efficiency"]
    )
    secondary_background = float(
        evidence["method_at_cut"][SECONDARY][
            "weighted_background_efficiency"
        ]
    )
    cut_background = float(
        evidence["method_at_cut"][OPTIMIZED_CUT][
            "weighted_background_efficiency"
        ]
    )
    bootstrap_low = float(evidence["bootstrap"]["percentile_2_5"])
    bootstrap_high = float(evidence["bootstrap"]["percentile_97_5"])
    primary_improvement = cut_background - primary_background
    nominal_difference = secondary_background - primary_background
    validation_models = [
        row for row in validation_rows if row["candidate_type"] == "model"
    ]
    validation_cuts = [
        row
        for row in validation_rows
        if row["candidate_type"] == "cut_baseline"
    ]
    working_point_sets = {row["working_point_set"] for row in working_point_rows}
    primary_rows = [
        row
        for row in model_rows
        if row["decision_class"] == "primary_nominal_bdt"
    ]
    secondary_rows = [
        row
        for row in model_rows
        if row["decision_class"] == "secondary_categorized_bdt"
    ]
    audits = [
        audit_row(
            "source_checkpoint_sha256_and_status",
            4,
            sum(
                bool(row["manifest_verified"])
                and bool(row["passing_status_verified"])
                for row in inventory
            ),
            len(inventory) == 4
            and all(
                row["manifest_verified"] and row["passing_status_verified"]
                for row in inventory
            ),
            "source_checkpoint_inventory.tsv",
        ),
        audit_row(
            "primary_nominal_model",
            PRIMARY,
            primary_rows[0]["model_or_method"] if len(primary_rows) == 1 else "",
            len(primary_rows) == 1
            and primary_rows[0]["model_or_method"] == PRIMARY,
            "model_choice.tsv",
        ),
        audit_row(
            "secondary_categorized_model",
            SECONDARY,
            (
                secondary_rows[0]["model_or_method"]
                if len(secondary_rows) == 1
                else ""
            ),
            len(secondary_rows) == 1
            and secondary_rows[0]["model_or_method"] == SECONDARY,
            "model_choice.tsv",
        ),
        audit_row(
            "primary_background_efficiency_improvement_over_cut",
            f">={rules['minimum_primary_background_efficiency_improvement_over_cut']}",
            primary_improvement,
            primary_improvement
            >= float(
                rules[
                    "minimum_primary_background_efficiency_improvement_over_cut"
                ]
            ),
            "train_only_performance_comparison.tsv",
        ),
        audit_row(
            "absolute_nominal_v2_background_efficiency_difference",
            f"<={rules['maximum_absolute_nominal_v2_background_efficiency_difference']}",
            abs(nominal_difference),
            abs(nominal_difference)
            <= float(
                rules[
                    "maximum_absolute_nominal_v2_background_efficiency_difference"
                ]
            ),
            "train_only_performance_comparison.tsv",
        ),
        audit_row(
            "member_bootstrap_interval_includes_zero",
            True,
            bootstrap_low <= 0.0 <= bootstrap_high,
            bootstrap_low <= 0.0 <= bootstrap_high,
            "member_bootstrap_comparison.tsv in source checkpoint",
        ),
        audit_row(
            "simpler_global_model_preferred",
            True,
            config["model_choice"]["primary_nominal_model"] == PRIMARY,
            bool(rules["prefer_simpler_global_model_when_rules_hold"])
            and config["model_choice"]["primary_nominal_model"] == PRIMARY,
            "frozen decision rule",
        ),
        audit_row(
            "validation_candidate_files_opened",
            0,
            0,
            True,
            "aggregate checkpoint inputs only",
        ),
        audit_row(
            "test_candidate_files_opened",
            0,
            0,
            True,
            "aggregate checkpoint inputs only",
        ),
        audit_row("models_trained", 0, 0, True, "freeze-only gate"),
        audit_row(
            "predictions_written", 0, 0, True, "freeze-only gate"
        ),
        audit_row(
            "hyperparameter_trials", 0, 0, True, "freeze-only gate"
        ),
        audit_row(
            "frozen_working_point_sets",
            2,
            len(working_point_sets),
            len(working_point_sets) == 2 and len(working_point_rows) == 8,
            "frozen_working_points.tsv",
        ),
        audit_row(
            "validation_models_predeclared",
            3,
            len(validation_models),
            len(validation_models) == 3,
            "frozen_validation_plan.tsv",
        ),
        audit_row(
            "validation_cut_baselines_predeclared",
            1,
            len(validation_cuts),
            len(validation_cuts) == 1,
            "frozen_validation_plan.tsv",
        ),
        audit_row(
            "physical_event_weights_used",
            0,
            0,
            True,
            "freeze-only aggregate comparison",
        ),
        audit_row(
            "physics_yields_calculated",
            0,
            0,
            True,
            "freeze-only aggregate comparison",
        ),
        audit_row(
            "significances_calculated",
            0,
            0,
            True,
            "freeze-only aggregate comparison",
        ),
        audit_row(
            "validation_model_selection_authorized",
            False,
            config["authorization"][
                "validation_model_selection_authorized"
            ],
            not config["authorization"][
                "validation_model_selection_authorized"
            ],
            "authorization contract",
        ),
        audit_row(
            "post_validation_hyperparameter_tuning_authorized",
            False,
            config["authorization"][
                "post_validation_hyperparameter_tuning_authorized"
            ],
            not config["authorization"][
                "post_validation_hyperparameter_tuning_authorized"
            ],
            "authorization contract",
        ),
        audit_row(
            "test_access_authorized",
            False,
            config["authorization"]["test_access_authorized"],
            not config["authorization"]["test_access_authorized"],
            "authorization contract",
        ),
    ]
    return audits


def write_required_tables(
    output_dir: Path,
    model_rows: Sequence[Mapping[str, Any]],
    performance_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    working_point_rows: Sequence[Mapping[str, Any]],
) -> None:
    model_fields = (
        "model_or_method",
        "decision_class",
        "role",
        "nominal",
        "validation_predeclared",
        "selection_basis",
        "status",
    )
    write_table_bundle(
        output_dir,
        "model_choice",
        model_rows,
        model_fields,
        publication_fields=model_fields,
        caption="Frozen train-only HH to four-b model choice.",
        label="tab:hh4b_bdt_model_choice",
        latex_headers={
            "model_or_method": "Model or method",
            "decision_class": "Decision class",
            "role": "Role",
            "nominal": "Nominal",
            "validation_predeclared": "Validation predeclared",
            "selection_basis": "Train-only basis",
            "status": "Status",
        },
    )

    performance_fields = (
        "method",
        "decision_class",
        "weighted_oof_auc",
        "low_mhh_weighted_oof_auc",
        "high_mhh_weighted_oof_auc",
        "cut_matched_weighted_signal_efficiency",
        "cut_matched_weighted_background_efficiency",
        "bootstrap_median_difference_vs_global_v1",
        "bootstrap_percentile_2_5_vs_global_v1",
        "bootstrap_percentile_97_5_vs_global_v1",
        "bootstrap_fraction_favoring_method",
        "interpretation",
        "status",
    )
    write_table_bundle(
        output_dir,
        "train_only_performance_comparison",
        performance_rows,
        performance_fields,
        publication_fields=(
            "method",
            "decision_class",
            "weighted_oof_auc",
            "low_mhh_weighted_oof_auc",
            "high_mhh_weighted_oof_auc",
            "cut_matched_weighted_signal_efficiency",
            "cut_matched_weighted_background_efficiency",
            "interpretation",
        ),
        caption=(
            "Frozen train-only out-of-fold performance comparison at the "
            "optimized-cut-matched signal efficiency."
        ),
        label="tab:hh4b_bdt_train_only_performance",
        latex_headers={
            "method": "Model or method",
            "decision_class": "Decision class",
            "weighted_oof_auc": r"Weighted $\mathrm{AUC}_{\mathrm{OOF}}$",
            "low_mhh_weighted_oof_auc": (
                r"Low-$m_{HH}$ weighted $\mathrm{AUC}_{\mathrm{OOF}}$"
            ),
            "high_mhh_weighted_oof_auc": (
                r"High-$m_{HH}$ weighted $\mathrm{AUC}_{\mathrm{OOF}}$"
            ),
            "cut_matched_weighted_signal_efficiency": (
                r"Weighted $\epsilon_{S}$"
            ),
            "cut_matched_weighted_background_efficiency": (
                r"Weighted $\epsilon_{B}$"
            ),
            "interpretation": "Interpretation",
        },
    )

    validation_fields = (
        "order",
        "candidate_type",
        "model_or_method",
        "role",
        "selection",
        "fit_population",
        "validation_evaluations",
        "validation_model_selection_authorized",
        "post_validation_hyperparameter_tuning_authorized",
        "status",
    )
    write_table_bundle(
        output_dir,
        "frozen_validation_plan",
        validation_rows,
        validation_fields,
        publication_fields=validation_fields,
        caption=(
            "Predeclared one-time validation evaluation plan; validation "
            "model selection is not authorized."
        ),
        label="tab:hh4b_bdt_validation_plan",
        latex_headers={
            "order": "Order",
            "candidate_type": "Candidate type",
            "model_or_method": "Model or method",
            "role": "Role",
            "selection": "Selection",
            "fit_population": "Fit population",
            "validation_evaluations": "Validation evaluations",
            "validation_model_selection_authorized": (
                "Validation selection authorized"
            ),
            "post_validation_hyperparameter_tuning_authorized": (
                "Post-validation tuning authorized"
            ),
            "status": "Status",
        },
    )

    working_fields = (
        "working_point_set",
        "model",
        "threshold_contract",
        "target_weighted_signal_efficiency",
        "global_threshold",
        "low_mhh_threshold",
        "high_mhh_threshold",
        "source_table",
        "status",
    )
    write_table_bundle(
        output_dir,
        "frozen_working_points",
        working_point_rows,
        working_fields,
        publication_fields=(
            "model",
            "threshold_contract",
            "target_weighted_signal_efficiency",
            "global_threshold",
            "low_mhh_threshold",
            "high_mhh_threshold",
            "status",
        ),
        caption=(
            "Frozen train-OOF-derived BDT score working points. Categorized "
            "thresholds are separate in the low- and high-mHH regions."
        ),
        label="tab:hh4b_bdt_frozen_working_points",
        latex_headers={
            "model": "Model",
            "threshold_contract": "Threshold contract",
            "target_weighted_signal_efficiency": (
                r"Target weighted $\epsilon_{S}$"
            ),
            "global_threshold": r"Global $s_{\mathrm{BDT}}$",
            "low_mhh_threshold": (
                r"Low-$m_{HH}$ $s_{\mathrm{BDT}}$"
            ),
            "high_mhh_threshold": (
                r"High-$m_{HH}$ $s_{\mathrm{BDT}}$"
            ),
            "status": "Status",
        },
    )


def write_tsv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: format_plain(row.get(field, "")) for field in fields}
            )


def build_summary(
    config: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    model_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    working_point_rows: Sequence[Mapping[str, Any]],
    decision_audit: Sequence[Mapping[str, Any]],
    *,
    source_commit: str,
    config_sha256: str,
    runner_sha256: str,
) -> dict[str, Any]:
    validation_models = sum(
        row["candidate_type"] == "model" for row in validation_rows
    )
    validation_cuts = sum(
        row["candidate_type"] == "cut_baseline" for row in validation_rows
    )
    expected_metrics = config["model_choice"]["expected_train_only_metrics"]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": STATUS,
        "source_commit": source_commit,
        "config_sha256": config_sha256,
        "runner_sha256": runner_sha256,
        "source_checkpoints_verified": len(inventory),
        "primary_nominal_model": config["model_choice"][
            "primary_nominal_model"
        ],
        "secondary_categorized_model": config["model_choice"][
            "secondary_categorized_model"
        ],
        "primary_weighted_train_only_oof_auc": expected_metrics[
            "primary_weighted_oof_auc"
        ],
        "primary_cut_matched_weighted_background_efficiency": (
            expected_metrics[
                "primary_cut_matched_weighted_background_efficiency"
            ]
        ),
        "secondary_low_mhh_weighted_oof_auc": expected_metrics[
            "secondary_low_mhh_weighted_oof_auc"
        ],
        "secondary_high_mhh_weighted_oof_auc": expected_metrics[
            "secondary_high_mhh_weighted_oof_auc"
        ],
        "secondary_cut_matched_combined_weighted_background_efficiency": (
            expected_metrics[
                "secondary_cut_matched_weighted_background_efficiency"
            ]
        ),
        "secondary_member_bootstrap_median_difference_vs_global_v1": (
            expected_metrics["bootstrap_median_difference"]
        ),
        "secondary_member_bootstrap_percentile_2_5_vs_global_v1": (
            expected_metrics["bootstrap_percentile_2_5"]
        ),
        "secondary_member_bootstrap_percentile_97_5_vs_global_v1": (
            expected_metrics["bootstrap_percentile_97_5"]
        ),
        "secondary_member_bootstrap_fraction_favoring_v2": expected_metrics[
            "bootstrap_fraction_favoring_v2"
        ],
        "diagnostic_ablations_retained": len(
            config["model_choice"]["diagnostic_ablations"]
        ),
        "optimized_cut_baselines_retained": 1,
        "optimized_cut_expression": config["model_choice"][
            "optimized_cut_baseline"
        ]["expression"],
        "optimized_cut_weighted_signal_efficiency": expected_metrics[
            "optimized_cut_weighted_signal_efficiency"
        ],
        "optimized_cut_weighted_background_efficiency": expected_metrics[
            "optimized_cut_weighted_background_efficiency"
        ],
        "validation_candidate_files_opened": 0,
        "test_candidate_files_opened": 0,
        "models_trained": 0,
        "predictions_written": 0,
        "hyperparameter_trials": 0,
        "frozen_working_point_sets": len(
            {row["working_point_set"] for row in working_point_rows}
        ),
        "frozen_working_point_rows": len(working_point_rows),
        "frozen_target_weighted_signal_efficiencies": config[
            "frozen_working_points"
        ]["target_weighted_signal_efficiencies"],
        "validation_models_predeclared": validation_models,
        "validation_cut_baselines_predeclared": validation_cuts,
        "validation_evaluations_per_candidate": 1,
        "complete_frozen_train_population_fit_authorized": True,
        "physical_event_weights_used": 0,
        "physics_yields_calculated": 0,
        "significances_calculated": 0,
        "train_only_model_choice_frozen": True,
        "validation_evaluation_authorized": True,
        "validation_model_selection_authorized": False,
        "post_validation_hyperparameter_tuning_authorized": False,
        "test_access_authorized": False,
        "next_gate": NEXT_GATE,
    }
    pass_conditions = {
        "source_checkpoints": len(inventory) == 4
        and all(
            row["manifest_verified"] and row["passing_status_verified"]
            for row in inventory
        ),
        "primary_choice": summary["primary_nominal_model"] == PRIMARY,
        "secondary_choice": (
            summary["secondary_categorized_model"] == SECONDARY
        ),
        "closed_candidate_data": (
            summary["validation_candidate_files_opened"] == 0
            and summary["test_candidate_files_opened"] == 0
        ),
        "no_training_prediction_or_tuning": (
            summary["models_trained"] == 0
            and summary["predictions_written"] == 0
            and summary["hyperparameter_trials"] == 0
        ),
        "working_points": (
            summary["frozen_working_point_sets"] == 2
            and summary["frozen_working_point_rows"] == 8
        ),
        "validation_plan": (
            summary["validation_models_predeclared"] == 3
            and summary["validation_cut_baselines_predeclared"] == 1
        ),
        "no_physical_results": (
            summary["physical_event_weights_used"] == 0
            and summary["physics_yields_calculated"] == 0
            and summary["significances_calculated"] == 0
        ),
        "decision_rules": all(row["passed"] for row in decision_audit),
        "authorization": (
            summary["train_only_model_choice_frozen"]
            and summary["validation_evaluation_authorized"]
            and not summary["validation_model_selection_authorized"]
            and not summary[
                "post_validation_hyperparameter_tuning_authorized"
            ]
            and not summary["test_access_authorized"]
        ),
        "model_inventory": len(model_rows) == 6,
    }
    summary["pass_conditions"] = pass_conditions
    summary["failed_pass_conditions"] = [
        name for name, passed in pass_conditions.items() if not passed
    ]
    if summary["failed_pass_conditions"]:
        summary["status"] = "hh4b_bdt_train_only_model_choice_freeze_failed"
    return summary


def write_readme(
    output_dir: Path,
    summary: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    expected = evidence["bootstrap"]
    text = f"""# Canonical HH→4b BDT train-only model choice

Status: `{summary['status']}`.

The primary nominal model is `{PRIMARY}`. Its weighted train-only OOF AUC is
{float(evidence['global_metric']['weighted_roc_auc']):.16g}; at the signal
efficiency matched to the optimized $R_{{HH}}<34$ cut, its weighted background
efficiency is
{float(evidence['method_at_cut'][PRIMARY]['weighted_background_efficiency']):.12f}.
It provides substantially stronger background rejection than the optimized
cut while retaining the simpler global strategy.

The predeclared secondary model is `{SECONDARY}`, an advanced CMS-inspired
categorized alternative. Its weighted train-only OOF AUC values are
{float(evidence['category_metrics'][SECONDARY]['low_mhh']['weighted_roc_auc']):.12f}
in low $m_{{HH}}$ and
{float(evidence['category_metrics'][SECONDARY]['high_mhh']['weighted_roc_auc']):.12f}
in high $m_{{HH}}$. Its cut-matched combined weighted background efficiency is
{float(evidence['method_at_cut'][SECONDARY]['weighted_background_efficiency']):.12f}.
The source-member bootstrap median v2-minus-global-v1 difference is
{float(expected['median_difference']):.15g}, with a 2.5--97.5 percentile
interval [{float(expected['percentile_2_5']):.13g},
{float(expected['percentile_97_5']):.13g}] and a v2-favoring fraction of
{float(expected['fraction_favoring_first_method']):.3f}. The negligible
nominal-point improvement is therefore not stable under this diagnostic.

The three requested ablations remain non-nominal diagnostics. The optimized
`r_hh_125_125 < 34` selection remains the cut baseline.

The next gate may fit the frozen primary, secondary, and global mass-plane-blind
diagnostic on the complete frozen train population and evaluate validation
exactly once, together with the optimized cut. Validation model selection and
post-validation hyperparameter tuning are not authorized. Test access is not
authorized.

This freeze gate opened no validation or test candidate file, trained no
model, wrote no prediction, ran no hyperparameter trial, used no physical event
weight, and calculated no physics yield or significance.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def validate_output_inventory(directory: Path) -> None:
    observed = sorted(path.name for path in directory.iterdir() if path.is_file())
    expected = sorted(REQUIRED_CHECKPOINT_FILES)
    if observed != expected:
        raise ValueError(
            f"checkpoint file inventory mismatch: expected={expected}, "
            f"observed={observed}"
        )
    prohibited = [
        path.name
        for path in directory.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower()
            in {".parquet", ".npz", ".npy", ".pkl", ".pickle", ".h5"}
            or "prediction" in path.name.lower()
            or "model.json" in path.name.lower()
        )
    ]
    if prohibited:
        raise ValueError(f"prohibited model/prediction artifacts: {prohibited}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the exact configured runtime/checkpoint directories",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    source_commit = git_output("rev-parse", "HEAD")
    if source_commit != config["source_commit"]:
        raise ValueError(
            f"source commit mismatch: config={config['source_commit']} "
            f"HEAD={source_commit}"
        )
    if config["authorization"]["next_gate"] != NEXT_GATE:
        raise ValueError("configured next gate changed")

    runtime_dir = REPOSITORY_ROOT / config["outputs"]["runtime_dir"]
    checkpoint_dir = REPOSITORY_ROOT / config["outputs"]["checkpoint_dir"]
    prepare_exact_directory(
        runtime_dir,
        allowed_parent=REPOSITORY_ROOT / "outputs/agent_runs",
        overwrite=args.overwrite,
    )
    if checkpoint_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"checkpoint exists; pass --overwrite to replace it: "
            f"{checkpoint_dir}"
        )
    require_safe_output_path(
        checkpoint_dir, REPOSITORY_ROOT / "docs/checkpoints"
    )

    inventory, directories = verify_source_checkpoints(config)
    evidence = collect_evidence(config, directories)
    model_rows = build_model_choice_rows()
    performance_rows = build_performance_rows(evidence)
    validation_rows = build_validation_plan_rows(config)
    working_point_rows = build_working_point_rows(config, evidence)
    decision_audit = build_decision_rule_audit(
        config,
        inventory,
        evidence,
        model_rows,
        validation_rows,
        working_point_rows,
    )
    if not all(row["passed"] for row in decision_audit):
        failed = [row["condition"] for row in decision_audit if not row["passed"]]
        raise RuntimeError(f"decision rule audit failed: {failed}")

    write_required_tables(
        runtime_dir,
        model_rows,
        performance_rows,
        validation_rows,
        working_point_rows,
    )
    write_tsv(
        runtime_dir / "source_checkpoint_inventory.tsv",
        inventory,
        (
            "checkpoint",
            "checkpoint_dir",
            "required_status",
            "observed_status",
            "sha256sums_sha256",
            "manifest_entries_verified",
            "manifest_verified",
            "passing_status_verified",
            "status",
        ),
    )
    write_tsv(
        runtime_dir / "decision_rule_audit.tsv",
        decision_audit,
        ("condition", "expected", "observed", "passed", "evidence"),
    )
    summary = build_summary(
        config,
        inventory,
        model_rows,
        validation_rows,
        working_point_rows,
        decision_audit,
        source_commit=source_commit,
        config_sha256=sha256_file(config_path),
        runner_sha256=sha256_file(Path(__file__)),
    )
    if summary["failed_pass_conditions"]:
        write_json(runtime_dir / "summary.json", summary)
        raise RuntimeError(
            f"freeze failed: {summary['failed_pass_conditions']}"
        )
    write_json(runtime_dir / "summary.json", summary)
    checkpoint_record = {
        "schema_version": 1,
        "classification": "canonical_hh4b_bdt_train_only_model_choice",
        "status": STATUS,
        "source_commit": source_commit,
        "input_checkpoints": {
            Path(row["checkpoint_dir"]).name: row["sha256sums_sha256"]
            for row in inventory
        },
        "primary_nominal_model": PRIMARY,
        "secondary_categorized_model": SECONDARY,
        "policy": {
            "train_only_model_choice_frozen": True,
            "validation_evaluation_authorized": True,
            "validation_model_selection_authorized": False,
            "post_validation_hyperparameter_tuning_authorized": False,
            "test_access_authorized": False,
        },
        "next_gate": NEXT_GATE,
    }
    write_json(runtime_dir / "checkpoint.json", checkpoint_record)
    write_readme(runtime_dir, summary, evidence)
    write_sha256_manifest(runtime_dir)
    verify_sha256_manifest(runtime_dir)
    validate_output_inventory(runtime_dir)

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    shutil.copytree(runtime_dir, checkpoint_dir)
    verify_sha256_manifest(checkpoint_dir)
    validate_output_inventory(checkpoint_dir)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Runtime output: {runtime_dir}")
    print(f"Checkpoint: {checkpoint_dir}")


if __name__ == "__main__":
    main()
