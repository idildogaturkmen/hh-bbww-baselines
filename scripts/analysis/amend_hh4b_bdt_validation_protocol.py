#!/usr/bin/env python3
"""Freeze the amended HH→4b BDT validation model-selection protocol.

Only committed aggregate checkpoint metadata are read. This amendment opens
no candidate file, trains no model, writes no prediction, and changes no
frozen feature, label, fold, weight, hyperparameter, category, OOF prediction,
or train-OOF working point.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "analysis"))

from freeze_hh4b_bdt_model_choice import (  # noqa: E402
    format_plain,
    git_output,
    load_yaml,
    prepare_exact_directory,
    read_json,
    read_tsv,
    require_safe_output_path,
    sha256_file,
    verify_sha256_manifest,
    verify_source_checkpoints,
    write_json,
    write_sha256_manifest,
    write_table_bundle,
    write_tsv,
)


CONFIG_DEFAULT = (
    REPOSITORY_ROOT
    / "configs/baselines/hh4b_bdt_validation_selection_protocol_v2.yaml"
)
TEST_PATH = (
    REPOSITORY_ROOT / "tests/test_amend_hh4b_bdt_validation_protocol.py"
)
STATUS = "hh4b_bdt_validation_model_selection_protocol_frozen"
NEXT_GATE = "fit_frozen_hh4b_bdt_candidates_and_select_on_validation_once"
SUPERSEDED_POLICY = (
    "docs/checkpoints/hh4b_bdt_model_choice_20260727_v1/"
)
PRIMARY_TARGET = 0.585957314769
PRIMARY = "global_v1_mass_aware"
SECONDARY = "categorized_cms_inspired_mass_aware"
DIAGNOSTIC = "global_v1_explicit_dijet_mass_plane_blind"
OPTIMIZED_CUT = "optimized_cut"

REQUIRED_CHECKPOINT_FILES = (
    "README.md",
    "checkpoint.json",
    "summary.json",
    "candidate_models.tsv",
    "candidate_models.md",
    "candidate_models.tex",
    "validation_role.tsv",
    "validation_role.md",
    "validation_role.tex",
    "model_selection_rule.tsv",
    "model_selection_rule.md",
    "model_selection_rule.tex",
    "validation_metric_contract.tsv",
    "validation_metric_contract.md",
    "validation_metric_contract.tex",
    "test_policy.tsv",
    "test_policy.md",
    "test_policy.tex",
    "superseded_policy_inventory.tsv",
    "source_checkpoint_inventory.tsv",
    "decision_rule_audit.tsv",
    "SHA256SUMS",
)


def run_unittests() -> tuple[int, bool]:
    result = subprocess.run(
        [sys.executable, str(TEST_PATH), "-v"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
    )
    combined = result.stdout + result.stderr
    if combined.strip():
        print(combined, end="" if combined.endswith("\n") else "\n")
    matches = re.findall(r"Ran (\d+) tests?", combined)
    tests_run = int(matches[-1]) if matches else 0
    passed = result.returncode == 0 and tests_run > 0
    return tests_run, passed


def require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(
            f"{label} changed: expected {expected!r}, observed {observed!r}"
        )


def only_passing_row(
    rows: Sequence[Mapping[str, str]],
    *,
    field: str,
    value: str,
    label: str,
) -> dict[str, str]:
    matches = [
        dict(row)
        for row in rows
        if row.get(field) == value and row.get("status") == "pass"
    ]
    if len(matches) != 1:
        raise ValueError(f"{label}: expected one passing row, found {len(matches)}")
    return matches[0]


def verify_frozen_contracts(
    config: Mapping[str, Any], directories: Mapping[str, Path]
) -> dict[str, Any]:
    expected = config["population"]["expected"]
    v1_input = read_json(directories["v1_input_contract"] / "summary.json")
    require_equal(v1_input["train_members"], expected["train_members"], "train members")
    require_equal(
        v1_input["train_candidate_rows"], expected["train_rows"], "train rows"
    )
    require_equal(
        v1_input["validation_members"],
        expected["validation_members"],
        "validation members",
    )
    require_equal(
        v1_input["validation_candidate_rows_from_manifest"],
        expected["validation_rows_from_frozen_manifest_metadata"],
        "validation rows from frozen manifest metadata",
    )
    require_equal(
        v1_input["test_members_considered"],
        expected["test_members_considered"],
        "test members considered",
    )
    require_equal(v1_input["mass_aware_features"], 34, "v1 mass-aware features")
    require_equal(
        v1_input["mass_plane_blind_features"],
        30,
        "v1 mass-plane-blind features",
    )

    validation_opened = []
    test_opened = []
    for name, directory in directories.items():
        summary = read_json(directory / "summary.json")
        validation_opened.append(
            int(summary.get("validation_candidate_files_opened", 0))
        )
        test_opened.append(int(summary.get("test_candidate_files_opened", 0)))
    require_equal(max(validation_opened), 0, "prior validation candidate access")
    require_equal(max(test_opened), 0, "prior test candidate access")

    v1_hyperparameters = read_tsv(
        directories["v1_grouped_cv"] / "selected_hyperparameters.tsv"
    )
    global_hyper = only_passing_row(
        v1_hyperparameters,
        field="variant",
        value="mass_aware",
        label="global v1 selected hyperparameters",
    )
    blind_hyper = only_passing_row(
        v1_hyperparameters,
        field="variant",
        value="explicit_dijet_mass_plane_blind",
        label="global diagnostic selected hyperparameters",
    )
    require_equal(global_hyper["trial_id"], "18", "global v1 trial")
    require_equal(blind_hyper["trial_id"], "18", "global diagnostic trial")

    v2_hyperparameters = read_tsv(
        directories["v2_cms_inspired_grouped_cv"]
        / "selected_category_hyperparameters.tsv"
    )
    low_hyper = only_passing_row(
        v2_hyperparameters,
        field="category",
        value="low_mhh",
        label="low-mHH selected hyperparameters",
    )
    high_hyper = only_passing_row(
        v2_hyperparameters,
        field="category",
        value="high_mhh",
        label="high-mHH selected hyperparameters",
    )
    require_equal(low_hyper["trial_id"], "19", "low-mHH selected trial")
    require_equal(high_hyper["trial_id"], "18", "high-mHH selected trial")

    feature_rows = read_tsv(
        directories["v2_cms_inspired_contract"] / "feature_contracts.tsv"
    )
    feature_counts = Counter(row["variant"] for row in feature_rows)
    require_equal(
        feature_counts["global_v1_mass_aware_reference"],
        34,
        "global reference feature count",
    )
    require_equal(
        feature_counts["low_mhh_cms_inspired_mass_aware"],
        52,
        "low-mHH feature count",
    )
    require_equal(
        feature_counts["high_mhh_cms_inspired_mass_aware"],
        52,
        "high-mHH feature count",
    )

    category_rows = read_tsv(
        directories["v2_cms_inspired_contract"] / "category_definitions.tsv"
    )
    low_category = only_passing_row(
        category_rows,
        field="category",
        value="low_mhh",
        label="low-mHH category",
    )
    high_category = only_passing_row(
        category_rows,
        field="category",
        value="high_mhh",
        label="high-mHH category",
    )
    require_equal(low_category["boundary_GeV"], "450", "low-mHH boundary")
    require_equal(high_category["boundary_GeV"], "450", "high-mHH boundary")
    require_equal(low_category["boundary_optimized"], "false", "boundary optimization")
    require_equal(high_category["boundary_optimized"], "false", "boundary optimization")

    model_choice_dir = directories["train_only_model_choice"]
    previous_summary = read_json(model_choice_dir / "summary.json")
    require_equal(
        previous_summary["primary_nominal_model"], PRIMARY, "train-only primary"
    )
    require_equal(
        previous_summary["secondary_categorized_model"],
        SECONDARY,
        "train-only secondary",
    )
    require_equal(
        previous_summary["frozen_working_point_sets"],
        2,
        "frozen working-point sets",
    )
    working_points = read_tsv(model_choice_dir / "frozen_working_points.tsv")
    require_equal(len(working_points), 8, "frozen working-point rows")
    require_equal(
        {row["status"] for row in working_points},
        {"frozen"},
        "working-point status",
    )
    observed_targets = {
        round(float(row["target_weighted_signal_efficiency"]), 12)
        for row in working_points
    }
    expected_targets = {
        round(float(target), 12)
        for target in config["metric_contract"][
            "target_weighted_signal_efficiencies"
        ]
    }
    require_equal(observed_targets, expected_targets, "working-point targets")

    return {
        "train_members": v1_input["train_members"],
        "train_rows": v1_input["train_candidate_rows"],
        "validation_members": v1_input["validation_members"],
        "validation_rows_from_frozen_manifest_metadata": v1_input[
            "validation_candidate_rows_from_manifest"
        ],
        "test_members_considered": v1_input["test_members_considered"],
        "validation_candidate_files_opened_previously_by_bdt_gates": max(
            validation_opened
        ),
        "test_candidate_files_opened_previously_by_bdt_gates": max(test_opened),
        "global_v1_features": 34,
        "categorized_v2_features_per_model": 52,
        "mass_plane_blind_features": 30,
        "mhh_category_boundary_GeV": 450,
        "global_v1_trial_id": int(global_hyper["trial_id"]),
        "categorized_low_mhh_trial_id": int(low_hyper["trial_id"]),
        "categorized_high_mhh_trial_id": int(high_hyper["trial_id"]),
        "mass_plane_blind_trial_id": int(blind_hyper["trial_id"]),
        "frozen_working_point_sets": 2,
        "frozen_working_point_rows": 8,
    }


def build_candidate_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = config["candidate_bdt_methods"]
    fitting = {
        row["model"]: row
        for row in config["validation_fitting_contract"]["strategies"]
    }
    rows = []
    for method in (PRIMARY, SECONDARY):
        specification = candidates[method]
        rows.append(
            {
                "method": method,
                "candidate_label": specification["candidate_label"],
                "candidate_bdt": True,
                "validation_role": "final_nominal_selection_candidate",
                "architecture": specification["architecture"],
                "feature_contract": specification["feature_contract"],
                "features_per_model": specification["features_per_model"],
                "fitted_models_next_gate": fitting[method]["fitted_models"],
                "score_contract": specification["score_contract"],
                "category_contract": (
                    "all_train_global"
                    if method == PRIMARY
                    else "low_mhh_below_450_GeV;high_mhh_at_or_above_450_GeV"
                ),
                "hyperparameter_source": specification[
                    "hyperparameter_source"
                ],
                "fit_population": config["validation_fitting_contract"][
                    "fit_population"
                ],
                "fitting_weight_contract": config[
                    "validation_fitting_contract"
                ]["fitting_weights"],
                "scaler_fitted": False,
                "final_nominal_selected_in_amendment": False,
                "presentation_role": specification["presentation_role"],
                "status": "predeclared",
            }
        )
    diagnostic_fit = fitting[DIAGNOSTIC]
    rows.append(
        {
            "method": DIAGNOSTIC,
            "candidate_label": "diagnostic",
            "candidate_bdt": False,
            "validation_role": "predeclared_mass_plane_blind_diagnostic",
            "architecture": "one_global_xgboost_classifier",
            "feature_contract": "frozen_v1_explicit_dijet_mass_plane_blind",
            "features_per_model": 30,
            "fitted_models_next_gate": diagnostic_fit["fitted_models"],
            "score_contract": "one_global_score",
            "category_contract": "all_train_global",
            "hyperparameter_source": (
                "hh4b_bdt_v1_grouped_cv_20260725_v1/"
                "selected_hyperparameters.tsv:explicit_dijet_mass_plane_blind"
            ),
            "fit_population": config["validation_fitting_contract"][
                "fit_population"
            ],
            "fitting_weight_contract": config["validation_fitting_contract"][
                "fitting_weights"
            ],
            "scaler_fitted": False,
            "final_nominal_selected_in_amendment": False,
            "presentation_role": "diagnostic_only",
            "status": "predeclared",
        }
    )
    return rows


def build_validation_role_rows(
    config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    validation = config["validation"]
    rows = [
        {
            "scope": "validation_split",
            "activity": "split_role",
            "authorization": "frozen",
            "frequency": validation["access_frequency"],
            "constraint": validation["role"],
            "status": "predeclared",
        },
        {
            "scope": "model_fitting",
            "activity": "use_validation_during_fitting",
            "authorization": "prohibited",
            "frequency": "never",
            "constraint": "fit_only_on_complete_frozen_train_population",
            "status": "frozen",
        },
    ]
    for activity in validation["permitted_uses"]:
        rows.append(
            {
                "scope": "validation_split",
                "activity": activity,
                "authorization": "permitted",
                "frequency": validation["access_frequency"],
                "constraint": "predeclared_protocol_only",
                "status": "frozen",
            }
        )
    for activity in validation["prohibited_uses"]:
        rows.append(
            {
                "scope": "validation_split",
                "activity": activity,
                "authorization": "prohibited",
                "frequency": "never",
                "constraint": "no_post_inspection_reoptimization",
                "status": "frozen",
            }
        )
    return rows


def build_selection_rule_rows() -> list[dict[str, Any]]:
    return [
        {
            "order": 1,
            "condition": "lower_point_estimate_background_efficiency",
            "metric": "epsilon_B_categorized_v2_minus_epsilon_B_global_v1",
            "operator": "<",
            "threshold": 0,
            "all_conditions_required": True,
            "failure_action": f"select_{PRIMARY}",
            "status": "predeclared",
        },
        {
            "order": 2,
            "condition": "minimum_relative_background_efficiency_reduction",
            "metric": (
                "(epsilon_B_global_v1_minus_epsilon_B_categorized_v2)"
                "_divided_by_epsilon_B_global_v1"
            ),
            "operator": ">=",
            "threshold": 0.02,
            "all_conditions_required": True,
            "failure_action": f"select_{PRIMARY}",
            "status": "predeclared",
        },
        {
            "order": 3,
            "condition": "bootstrap_fraction_favoring_categorized_v2",
            "metric": "fraction_delta_epsilon_B_lt_0",
            "operator": ">=",
            "threshold": 0.84,
            "all_conditions_required": True,
            "failure_action": f"select_{PRIMARY}",
            "status": "predeclared",
        },
        {
            "order": 4,
            "condition": "signal_mode_equalization",
            "metric": (
                "max_abs_categorized_v2_ggf_or_vbf_epsilon_S_minus_target"
            ),
            "operator": "<=",
            "threshold": 0.05,
            "all_conditions_required": True,
            "failure_action": f"select_{PRIMARY}",
            "status": "predeclared",
        },
        {
            "order": 5,
            "condition": "integrity_category_and_model_application",
            "metric": "failure_count",
            "operator": "==",
            "threshold": 0,
            "all_conditions_required": True,
            "failure_action": f"select_{PRIMARY}",
            "status": "predeclared",
        },
        {
            "order": 6,
            "condition": "categorized_v2_selection_outcome",
            "metric": "all_five_conditions",
            "operator": "==",
            "threshold": True,
            "all_conditions_required": True,
            "failure_action": f"select_{PRIMARY}",
            "status": f"if_true_select_{SECONDARY}",
        },
    ]


def build_metric_contract_rows(
    config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    targets = config["metric_contract"]["target_weighted_signal_efficiencies"]
    report = (
        "weighted_signal_efficiency;weighted_background_efficiency;"
        "weighted_background_rejection;weighted_ggf_efficiency;"
        "weighted_vbf_efficiency;every_background_family_efficiency"
    )
    rows: list[dict[str, Any]] = []
    for method in (PRIMARY, SECONDARY):
        threshold_contract = (
            "frozen_single_train_oof_threshold"
            if method == PRIMARY
            else "frozen_separate_low_and_high_mhh_train_oof_thresholds"
        )
        for target in targets:
            rows.append(
                {
                    "comparison_kind": "fixed_train_oof_working_point_generalization",
                    "method": method,
                    "target_weighted_signal_efficiency": target,
                    "threshold_contract": threshold_contract,
                    "threshold_provenance": (
                        "hh4b_bdt_model_choice_20260727_v1/"
                        "frozen_working_points.tsv"
                    ),
                    "reported_metrics": report,
                    "primary_selection_metric": False,
                    "may_replace_frozen_train_oof_thresholds": False,
                    "normalization": "development_balancing_not_physical",
                    "status": "predeclared",
                }
            )
    for method in (PRIMARY, SECONDARY):
        threshold_contract = (
            "derive_one_validation_threshold"
            if method == PRIMARY
            else (
                "derive_separate_low_and_high_mhh_validation_thresholds_"
                "with_same_target_inside_each_category"
            )
        )
        for target in targets:
            is_primary = abs(float(target) - PRIMARY_TARGET) < 1.0e-12
            rows.append(
                {
                    "comparison_kind": "equalized_validation_efficiency",
                    "method": method,
                    "target_weighted_signal_efficiency": target,
                    "threshold_contract": threshold_contract,
                    "threshold_provenance": (
                        "one_time_validation_selection_diagnostic"
                    ),
                    "reported_metrics": report,
                    "primary_selection_metric": is_primary,
                    "may_replace_frozen_train_oof_thresholds": False,
                    "normalization": "development_balancing_not_physical",
                    "status": "predeclared",
                }
            )
    rows.extend(
        [
            {
                "comparison_kind": "validation_member_bootstrap",
                "method": "categorized_v2_minus_global_v1",
                "target_weighted_signal_efficiency": PRIMARY_TARGET,
                "threshold_contract": "equalized_validation_efficiency",
                "threshold_provenance": (
                    "source_member_seed_20260727_replicates_2000"
                ),
                "reported_metrics": (
                    "median;percentile_16;percentile_84;percentile_2_5;"
                    "percentile_97_5;fraction_favoring_categorized_v2"
                ),
                "primary_selection_metric": True,
                "may_replace_frozen_train_oof_thresholds": False,
                "normalization": "development_balancing_not_physical",
                "status": "predeclared",
            },
            {
                "comparison_kind": "secondary_performance_review",
                "method": "both_candidate_bdts",
                "target_weighted_signal_efficiency": "",
                "threshold_contract": "report_only",
                "threshold_provenance": "validation_once",
                "reported_metrics": (
                    "raw_metrics;average_precision;signal_mode_efficiencies;"
                    "background_family_efficiencies;score_mass_correlations"
                ),
                "primary_selection_metric": False,
                "may_replace_frozen_train_oof_thresholds": False,
                "normalization": "development_balancing_not_physical",
                "status": "predeclared",
            },
            {
                "comparison_kind": "category_roc_review",
                "method": SECONDARY,
                "target_weighted_signal_efficiency": "",
                "threshold_contract": "category_local_scores_only",
                "threshold_provenance": "validation_once",
                "reported_metrics": (
                    "low_mhh_weighted_and_raw_roc_auc;"
                    "high_mhh_weighted_and_raw_roc_auc;"
                    "category_average_precision"
                ),
                "primary_selection_metric": False,
                "may_replace_frozen_train_oof_thresholds": False,
                "normalization": "development_balancing_not_physical",
                "status": "predeclared",
            },
            {
                "comparison_kind": "pooled_uncalibrated_category_roc_auc",
                "method": SECONDARY,
                "target_weighted_signal_efficiency": "",
                "threshold_contract": "prohibited",
                "threshold_provenance": "not_applicable",
                "reported_metrics": "none",
                "primary_selection_metric": False,
                "may_replace_frozen_train_oof_thresholds": False,
                "normalization": "not_applicable",
                "status": "prohibited",
            },
        ]
    )
    return rows


def build_test_policy_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    test = config["test"]
    return [
        {
            "phase": "amendment_gate",
            "item": "test_split_access",
            "authorization": "prohibited",
            "frequency": "zero",
            "policy": test["role"],
            "status": "frozen",
        },
        {
            "phase": "validation_selection_gate",
            "item": "test_split_access",
            "authorization": "prohibited",
            "frequency": "zero",
            "policy": "test_inaccessible_while_validation_selects_nominal",
            "status": "frozen",
        },
        {
            "phase": "later_final_evaluation_gate",
            "item": "test_split_access",
            "authorization": "deferred",
            "frequency": test["later_access_frequency"],
            "policy": "only_after_validation_selects_final_nominal",
            "status": "predeclared",
        },
        {
            "phase": "all_test_reporting",
            "item": "final_nominal_model_selection_or_change",
            "authorization": "prohibited",
            "frequency": "never",
            "policy": "test_results_must_not_choose_or_change_nominal",
            "status": "frozen",
        },
        {
            "phase": "later_final_evaluation_gate",
            "item": "final_validation_selected_nominal_bdt",
            "authorization": "predeclared_for_reporting",
            "frequency": "exactly_once",
            "policy": "primary_test_report",
            "status": "predeclared",
        },
        {
            "phase": "later_final_evaluation_gate",
            "item": "non_selected_bdt_as_secondary_comparison",
            "authorization": "predeclared_for_reporting",
            "frequency": "exactly_once",
            "policy": "secondary_test_report_not_selection",
            "status": "predeclared",
        },
        {
            "phase": "later_final_evaluation_gate",
            "item": "optimized_r_hh_125_125_lt_34_cut_baseline",
            "authorization": "predeclared_for_reporting",
            "frequency": "exactly_once",
            "policy": "cut_baseline_test_report",
            "status": "predeclared",
        },
    ]


def build_superseded_policy_rows() -> list[dict[str, Any]]:
    return [
        {
            "policy": "source_checkpoint_contents",
            "previous_value": "frozen",
            "amended_value": "unchanged",
            "disposition": "preserved",
            "rationale": "amend_policy_only",
            "status": "pass",
        },
        {
            "policy": "train_only_nominal_benchmark",
            "previous_value": PRIMARY,
            "amended_value": PRIMARY,
            "disposition": "preserved_as_train_only_benchmark",
            "rationale": "presentation_role_retained",
            "status": "pass",
        },
        {
            "policy": "secondary_categorized_model",
            "previous_value": SECONDARY,
            "amended_value": SECONDARY,
            "disposition": "retained_as_validation_selection_candidate",
            "rationale": "both_bdt_candidates_retained",
            "status": "pass",
        },
        {
            "policy": "final_nominal_model",
            "previous_value": "not_validation_selected",
            "amended_value": "not_selected_in_amendment_gate",
            "disposition": "deferred_to_one_time_validation_selection",
            "rationale": "validation_is_selection_split",
            "status": "pass",
        },
        {
            "policy": "validation_model_selection_authorized",
            "previous_value": False,
            "amended_value": True,
            "disposition": "superseded",
            "rationale": "validation_selection_policy_amendment",
            "status": "pass",
        },
        {
            "policy": "validation_evaluation_authorized",
            "previous_value": True,
            "amended_value": True,
            "disposition": "preserved",
            "rationale": "one_time_validation_gate",
            "status": "pass",
        },
        {
            "policy": "post_validation_reoptimization_authorized",
            "previous_value": False,
            "amended_value": False,
            "disposition": "preserved",
            "rationale": "no_validation_reoptimization",
            "status": "pass",
        },
        {
            "policy": "test_access_authorized",
            "previous_value": False,
            "amended_value": False,
            "disposition": "preserved",
            "rationale": "test_is_final_unbiased_split",
            "status": "pass",
        },
        {
            "policy": "frozen_train_oof_working_points",
            "previous_value": "two_sets_eight_rows",
            "amended_value": "unchanged",
            "disposition": "preserved",
            "rationale": "fixed_threshold_generalization_contract",
            "status": "pass",
        },
        {
            "policy": "category_specific_roc_and_feature_importance_plots",
            "previous_value": "retained",
            "amended_value": "retained",
            "disposition": "preserved",
            "rationale": "presentation_policy",
            "status": "pass",
        },
        {
            "policy": "next_gate",
            "previous_value": "fit_frozen_hh4b_bdt_models_and_evaluate_validation_once",
            "amended_value": NEXT_GATE,
            "disposition": "superseded",
            "rationale": "add_predeclared_validation_selection",
            "status": "pass",
        },
    ]


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


def build_decision_audit(
    config: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    facts: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    unittest_tests_run: int,
    unittests_passed: bool,
    table_failures: int,
) -> list[dict[str, Any]]:
    candidate_count = sum(bool(row["candidate_bdt"]) for row in candidate_rows)
    validation_models = len(candidate_rows)
    validation_cuts = len(
        config["validation_fitting_contract"]["cut_baselines"]
    )
    final_selected = sum(
        bool(row["final_nominal_selected_in_amendment"])
        for row in candidate_rows
    )
    fixed_rows = [
        row
        for row in metric_rows
        if row["comparison_kind"]
        == "fixed_train_oof_working_point_generalization"
    ]
    equalized_rows = [
        row
        for row in metric_rows
        if row["comparison_kind"] == "equalized_validation_efficiency"
    ]
    prohibited_validation_uses = {
        row["activity"]
        for row in validation_rows
        if row["authorization"] == "prohibited"
    }
    selection_thresholds = {
        row["condition"]: row["threshold"] for row in selection_rows
    }
    audits = [
        audit_row(
            "source_checkpoint_sha256_and_status",
            5,
            len(inventory),
            len(inventory) == 5
            and all(
                row["manifest_verified"] and row["passing_status_verified"]
                for row in inventory
            ),
            "source_checkpoint_inventory.tsv",
        ),
        audit_row("train_members", 458, facts["train_members"], facts["train_members"] == 458, "v1 input summary"),
        audit_row("train_rows", 57326, facts["train_rows"], facts["train_rows"] == 57326, "v1 input summary"),
        audit_row("validation_members", 123, facts["validation_members"], facts["validation_members"] == 123, "frozen manifest metadata"),
        audit_row(
            "validation_rows_from_frozen_manifest_metadata",
            12745,
            facts["validation_rows_from_frozen_manifest_metadata"],
            facts["validation_rows_from_frozen_manifest_metadata"] == 12745,
            "v1 input summary",
        ),
        audit_row(
            "test_members_considered",
            0,
            facts["test_members_considered"],
            facts["test_members_considered"] == 0,
            "v1 input summary",
        ),
        audit_row("candidate_bdt_models", 2, candidate_count, candidate_count == 2, "candidate_models.tsv"),
        audit_row("final_nominal_model_selected", 0, final_selected, final_selected == 0, "candidate_models.tsv"),
        audit_row("validation_models_predeclared", 3, validation_models, validation_models == 3, "candidate_models.tsv"),
        audit_row("validation_cut_baselines_predeclared", 1, validation_cuts, validation_cuts == 1, "validation fitting contract"),
        audit_row("validation_candidate_files_opened", 0, 0, True, "aggregate checkpoint metadata only"),
        audit_row("test_candidate_files_opened", 0, 0, True, "aggregate checkpoint metadata only"),
        audit_row("models_trained", 0, 0, True, "amendment-only gate"),
        audit_row("predictions_written", 0, 0, True, "amendment-only gate"),
        audit_row("hyperparameter_trials", 0, 0, True, "amendment-only gate"),
        audit_row(
            "validation_selection_authorized",
            True,
            config["authorization"]["validation_model_selection_authorized"],
            bool(config["authorization"]["validation_model_selection_authorized"]),
            "authorization amendment",
        ),
        audit_row(
            "validation_reoptimization_authorized",
            False,
            config["authorization"]["post_validation_reoptimization_authorized"],
            not config["authorization"]["post_validation_reoptimization_authorized"],
            "authorization amendment",
        ),
        audit_row(
            "test_selection_authorized",
            False,
            config["authorization"]["test_selection_authorized"],
            not config["authorization"]["test_selection_authorized"],
            "test policy",
        ),
        audit_row(
            "test_access_authorized",
            False,
            config["authorization"]["test_access_authorized"],
            not config["authorization"]["test_access_authorized"],
            "test policy",
        ),
        audit_row(
            "bootstrap_replicates_predeclared",
            2000,
            config["bootstrap"]["replicates"],
            config["bootstrap"]["replicates"] == 2000,
            "validation metric contract",
        ),
        audit_row(
            "bootstrap_seed",
            20260727,
            config["bootstrap"]["seed"],
            config["bootstrap"]["seed"] == 20260727,
            "validation metric contract",
        ),
        audit_row(
            "fixed_working_point_comparisons",
            8,
            len(fixed_rows),
            len(fixed_rows) == 8
            and all(
                not row["may_replace_frozen_train_oof_thresholds"]
                for row in fixed_rows
            ),
            "validation_metric_contract.tsv",
        ),
        audit_row(
            "equalized_validation_comparisons",
            8,
            len(equalized_rows),
            len(equalized_rows) == 8
            and sum(bool(row["primary_selection_metric"]) for row in equalized_rows)
            == 2,
            "validation_metric_contract.tsv",
        ),
        audit_row(
            "relative_reduction_threshold",
            0.02,
            selection_thresholds[
                "minimum_relative_background_efficiency_reduction"
            ],
            selection_thresholds[
                "minimum_relative_background_efficiency_reduction"
            ]
            == 0.02,
            "model_selection_rule.tsv",
        ),
        audit_row(
            "bootstrap_fraction_threshold",
            0.84,
            selection_thresholds[
                "bootstrap_fraction_favoring_categorized_v2"
            ],
            selection_thresholds[
                "bootstrap_fraction_favoring_categorized_v2"
            ]
            == 0.84,
            "model_selection_rule.tsv",
        ),
        audit_row(
            "signal_mode_deviation_threshold",
            0.05,
            selection_thresholds["signal_mode_equalization"],
            selection_thresholds["signal_mode_equalization"] == 0.05,
            "model_selection_rule.tsv",
        ),
        audit_row(
            "no_validation_reopening",
            True,
            "reopening_validation_after_inspecting_results"
            in prohibited_validation_uses,
            "reopening_validation_after_inspecting_results"
            in prohibited_validation_uses,
            "validation_role.tsv",
        ),
        audit_row(
            "pooled_uncalibrated_category_auc_authorized",
            False,
            config["metric_contract"][
                "pooled_uncalibrated_categorized_roc_auc_authorized"
            ],
            not config["metric_contract"][
                "pooled_uncalibrated_categorized_roc_auc_authorized"
            ],
            "validation_metric_contract.tsv",
        ),
        audit_row(
            "test_policy_rows",
            7,
            len(test_rows),
            len(test_rows) == 7
            and any(
                row["item"] == "final_nominal_model_selection_or_change"
                and row["authorization"] == "prohibited"
                for row in test_rows
            ),
            "test_policy.tsv",
        ),
        audit_row(
            "frozen_contract_dimensions",
            "34_global;52_categorized;450_GeV_boundary;8_working_point_rows",
            (
                f"{facts['global_v1_features']}_global;"
                f"{facts['categorized_v2_features_per_model']}_categorized;"
                f"{facts['mhh_category_boundary_GeV']}_GeV_boundary;"
                f"{facts['frozen_working_point_rows']}_working_point_rows"
            ),
            facts["global_v1_features"] == 34
            and facts["categorized_v2_features_per_model"] == 52
            and facts["mhh_category_boundary_GeV"] == 450
            and facts["frozen_working_point_rows"] == 8,
            "verified source aggregate contracts",
        ),
        audit_row(
            "all_unit_tests_pass",
            True,
            unittests_passed,
            unittests_passed and unittest_tests_run > 0,
            f"standard-library unittest ({unittest_tests_run} tests)",
        ),
        audit_row("table_failures", 0, table_failures, table_failures == 0, "five table bundles"),
        audit_row(
            "physical_normalization_authorized",
            False,
            config["authorization"]["physical_normalization_authorized"],
            not config["authorization"]["physical_normalization_authorized"],
            "development balancing; not physical normalization",
        ),
        audit_row(
            "presentation_policy",
            True,
            all(config["presentation_policy"].values()),
            all(config["presentation_policy"].values()),
            "both BDTs, comparisons, plots, final label, secondary benchmark",
        ),
    ]
    return audits


def write_protocol_tables(
    output_dir: Path,
    candidate_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
) -> int:
    table_failures = 0
    try:
        candidate_fields = (
            "method",
            "candidate_label",
            "candidate_bdt",
            "validation_role",
            "architecture",
            "feature_contract",
            "features_per_model",
            "fitted_models_next_gate",
            "score_contract",
            "category_contract",
            "hyperparameter_source",
            "fit_population",
            "fitting_weight_contract",
            "scaler_fitted",
            "final_nominal_selected_in_amendment",
            "presentation_role",
            "status",
        )
        write_table_bundle(
            output_dir,
            "candidate_models",
            candidate_rows,
            candidate_fields,
            publication_fields=(
                "method",
                "candidate_label",
                "candidate_bdt",
                "architecture",
                "features_per_model",
                "fitted_models_next_gate",
                "category_contract",
                "final_nominal_selected_in_amendment",
                "presentation_role",
                "status",
            ),
            caption=(
                "Frozen HH to four-b BDT candidates for one-time validation "
                "model selection."
            ),
            label="tab:hh4b_bdt_validation_candidates",
            latex_headers={
                "method": "Method",
                "candidate_label": "Candidate",
                "candidate_bdt": "BDT candidate",
                "architecture": "Architecture",
                "features_per_model": "Features per model",
                "fitted_models_next_gate": "Full-train models",
                "category_contract": r"$m_{HH}$ category contract",
                "final_nominal_selected_in_amendment": "Final nominal now",
                "presentation_role": "Presentation role",
                "status": "Status",
            },
        )

        validation_fields = (
            "scope",
            "activity",
            "authorization",
            "frequency",
            "constraint",
            "status",
        )
        write_table_bundle(
            output_dir,
            "validation_role",
            validation_rows,
            validation_fields,
            publication_fields=validation_fields,
            caption=(
                "Frozen role and allowed uses of the validation model-selection "
                "and generalization split."
            ),
            label="tab:hh4b_bdt_validation_role",
            latex_headers={
                "scope": "Scope",
                "activity": "Activity",
                "authorization": "Authorization",
                "frequency": "Frequency",
                "constraint": "Constraint",
                "status": "Status",
            },
        )

        selection_fields = (
            "order",
            "condition",
            "metric",
            "operator",
            "threshold",
            "all_conditions_required",
            "failure_action",
            "status",
        )
        write_table_bundle(
            output_dir,
            "model_selection_rule",
            selection_rows,
            selection_fields,
            publication_fields=selection_fields,
            caption=(
                "Predeclared validation rule for selecting the final nominal "
                "BDT at weighted signal efficiency 0.585957314769."
            ),
            label="tab:hh4b_bdt_model_selection_rule",
            latex_headers={
                "order": "Order",
                "condition": "Condition",
                "metric": r"Validation metric involving $\epsilon_{B}$",
                "operator": "Operator",
                "threshold": "Threshold",
                "all_conditions_required": "All required",
                "failure_action": "Failure action",
                "status": "Outcome",
            },
        )

        metric_fields = (
            "comparison_kind",
            "method",
            "target_weighted_signal_efficiency",
            "threshold_contract",
            "threshold_provenance",
            "reported_metrics",
            "primary_selection_metric",
            "may_replace_frozen_train_oof_thresholds",
            "normalization",
            "status",
        )
        write_table_bundle(
            output_dir,
            "validation_metric_contract",
            metric_rows,
            metric_fields,
            publication_fields=metric_fields,
            caption=(
                "Fixed train-OOF and equalized validation-efficiency metric "
                "contracts; all weights are development balancing, not "
                "physical normalization."
            ),
            label="tab:hh4b_bdt_validation_metric_contract",
            latex_headers={
                "comparison_kind": "Comparison",
                "method": "Method",
                "target_weighted_signal_efficiency": (
                    r"Target weighted $\epsilon_{S}$"
                ),
                "threshold_contract": r"$s_{\mathrm{BDT}}$ contract",
                "threshold_provenance": "Threshold provenance",
                "reported_metrics": (
                    r"Reported $\epsilon_{S}$, $\epsilon_{B}$, and diagnostics"
                ),
                "primary_selection_metric": "Primary selection",
                "may_replace_frozen_train_oof_thresholds": (
                    "May replace frozen thresholds"
                ),
                "normalization": "Normalization",
                "status": "Status",
            },
        )

        test_fields = (
            "phase",
            "item",
            "authorization",
            "frequency",
            "policy",
            "status",
        )
        write_table_bundle(
            output_dir,
            "test_policy",
            test_rows,
            test_fields,
            publication_fields=test_fields,
            caption=(
                "Frozen test policy: final unbiased evaluation only, never "
                "model selection."
            ),
            label="tab:hh4b_bdt_test_policy",
            latex_headers={
                "phase": "Phase",
                "item": "Item",
                "authorization": "Authorization",
                "frequency": "Frequency",
                "policy": "Policy",
                "status": "Status",
            },
        )
    except Exception:
        table_failures += 1
        raise
    return table_failures


def build_summary(
    config: Mapping[str, Any],
    facts: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    decision_audit: Sequence[Mapping[str, Any]],
    *,
    source_commit: str,
    config_sha256: str,
    runner_sha256: str,
    unittest_tests_run: int,
    unittests_passed: bool,
    table_failures: int,
) -> dict[str, Any]:
    candidate_count = sum(bool(row["candidate_bdt"]) for row in candidate_rows)
    final_selected = sum(
        bool(row["final_nominal_selected_in_amendment"])
        for row in candidate_rows
    )
    summary: dict[str, Any] = {
        "schema_version": 2,
        "status": STATUS,
        "source_commit": source_commit,
        "config_sha256": config_sha256,
        "runner_sha256": runner_sha256,
        "supersedes_policy_from": SUPERSEDED_POLICY,
        "next_gate": NEXT_GATE,
        "source_checkpoints_verified": len(inventory),
        **facts,
        "candidate_bdt_models": candidate_count,
        "final_nominal_model_selected": final_selected,
        "final_nominal_model": None,
        "validation_models_predeclared": len(candidate_rows),
        "validation_cut_baselines_predeclared": len(
            config["validation_fitting_contract"]["cut_baselines"]
        ),
        "full_train_model_artifacts_predeclared_next_gate": sum(
            int(row["fitted_models_next_gate"]) for row in candidate_rows
        ),
        "scalers_fitted_next_gate": 0,
        "validation_role": config["validation"]["role"],
        "test_role": config["test"]["role"],
        "validation_candidate_files_opened": 0,
        "test_candidate_files_opened": 0,
        "models_trained": 0,
        "predictions_written": 0,
        "hyperparameter_trials": 0,
        "validation_selection_authorized": True,
        "validation_model_selection_authorized": True,
        "validation_reoptimization_authorized": False,
        "post_validation_reoptimization_authorized": False,
        "test_selection_authorized": False,
        "test_access_authorized": False,
        "both_bdt_candidates_retained": True,
        "both_bdt_candidates_presented": True,
        "category_specific_roc_plots_retained": True,
        "category_specific_feature_importance_plots_retained": True,
        "non_selected_bdt_secondary_benchmark_retained": True,
        "final_validation_selected_nominal_label_required": True,
        "bootstrap_seed_predeclared": config["bootstrap"]["seed"],
        "bootstrap_replicates_predeclared": config["bootstrap"]["replicates"],
        "primary_target_weighted_validation_signal_efficiency": (
            config["metric_contract"]["primary"][
                "target_weighted_signal_efficiency"
            ]
        ),
        "primary_model_selection_metric": config["metric_contract"]["primary"][
            "metric"
        ],
        "physical_event_weights_used": 0,
        "physics_yields_calculated": 0,
        "significances_calculated": 0,
        "pooled_uncalibrated_category_auc_calculated": 0,
        "unittest_tests_run": unittest_tests_run,
        "unittest_tests_passed": unittest_tests_run if unittests_passed else 0,
        "unittest_failures": 0 if unittests_passed else 1,
        "table_failures": table_failures,
        "table_bundles_written": 5,
        "tsv_markdown_latex_table_files_written": 15,
    }
    pass_conditions = {
        "source_checkpoints": len(inventory) == 5,
        "population": (
            summary["train_members"] == 458
            and summary["train_rows"] == 57326
            and summary["validation_members"] == 123
            and summary["validation_rows_from_frozen_manifest_metadata"]
            == 12745
            and summary["test_members_considered"] == 0
        ),
        "candidate_models": (
            summary["candidate_bdt_models"] == 2
            and summary["final_nominal_model_selected"] == 0
        ),
        "validation_predeclaration": (
            summary["validation_models_predeclared"] == 3
            and summary["validation_cut_baselines_predeclared"] == 1
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
        "authorization": (
            summary["validation_selection_authorized"]
            and not summary["validation_reoptimization_authorized"]
            and not summary["test_selection_authorized"]
            and not summary["test_access_authorized"]
        ),
        "bootstrap": summary["bootstrap_replicates_predeclared"] == 2000,
        "tests": unittests_passed and unittest_tests_run > 0,
        "tables": table_failures == 0,
        "decision_audit": all(row["passed"] for row in decision_audit),
        "no_physical_results": (
            summary["physical_event_weights_used"] == 0
            and summary["physics_yields_calculated"] == 0
            and summary["significances_calculated"] == 0
        ),
    }
    summary["pass_conditions"] = pass_conditions
    summary["failed_pass_conditions"] = [
        name for name, passed in pass_conditions.items() if not passed
    ]
    if summary["failed_pass_conditions"]:
        summary["status"] = "hh4b_bdt_validation_protocol_amendment_failed"
    return summary


def write_readme(output_dir: Path, summary: Mapping[str, Any]) -> None:
    text = f"""# HH→4b BDT validation model-selection protocol amendment

Status: `{summary['status']}`.

This checkpoint preserves
`{SUPERSEDED_POLICY}` unchanged and supersedes only its validation-selection
policy. The global v1 mass-aware BDT remains the canonical global train-only
benchmark, and the categorized CMS-inspired mass-aware BDT remains the
categorized benchmark. Neither is selected as the final nominal BDT here.

Validation is frozen as the `model_selection_and_generalization_split`. In the
next gate, the two BDT candidates and the global mass-plane-blind diagnostic
are fit on the complete frozen train population without validation input.
Validation is then opened exactly once to compare the two candidates, select
the final nominal BDT under the predeclared five-condition rule, check the
optimized $R_{{HH}}<34$ cut, and evaluate the diagnostic. No validation-driven
feature, category, hyperparameter, weight, background-family, or reconstruction
change is authorized, and validation may not be reopened after inspection.

Both fixed train-OOF threshold generalization and equalized validation
signal-efficiency comparisons are required at targets 0.30, 0.50,
0.585957314769, and 0.70. The primary metric is combined
development-weighted validation background efficiency at target
0.585957314769; lower is better. Equalized validation thresholds are selection
diagnostics and do not replace the frozen train-OOF thresholds.

The source-member bootstrap uses seed 20260727 and 2000 replicas for
$\Delta\epsilon_B=\epsilon_B(\mathrm{{categorized\ v2}})
-\epsilon_B(\mathrm{{global\ v1}})$. Categorized v2 is selected only if its
point estimate is lower, its relative reduction is at least 0.02, at least
0.84 of replicas favor it, ggF and VBF efficiencies are each within 0.05 of
the target, and no integrity, category, or application failure occurs.
Otherwise global v1 is selected.

Test is frozen as the `final_unbiased_evaluation_split`. It remains inaccessible
during validation selection. A later gate may evaluate test exactly once after
validation selects the nominal model, but test results may never choose or
change that model.

Both BDT approaches, their train-only and validation comparisons, and the
category-specific ROC and feature-importance plots remain part of the
presentation. The final validation-selected nominal BDT must be labeled
explicitly, while the non-selected BDT remains a secondary benchmark.

All weights are development balancing, not physical normalization. No expected
yield, luminosity, significance, or expected limit is part of model selection.

This amendment opened zero validation and test candidate files, trained zero
models, wrote zero predictions, and ran zero hyperparameter trials.

Next gate: `{summary['next_gate']}`.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def validate_output_inventory(directory: Path) -> None:
    observed = sorted(
        path.name for path in directory.iterdir() if path.is_file()
    )
    expected = sorted(REQUIRED_CHECKPOINT_FILES)
    if observed != expected:
        raise ValueError(
            f"checkpoint inventory mismatch: expected={expected}, "
            f"observed={observed}"
        )
    prohibited = [
        path.name
        for path in directory.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower()
            in {
                ".parquet",
                ".npz",
                ".npy",
                ".pkl",
                ".pickle",
                ".h5",
                ".hdf5",
            }
            or "prediction" in path.name.lower()
            or path.name.endswith("model.json")
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
    if git_output("rev-parse", "origin/delphes-hh4b-production") != source_commit:
        raise ValueError("HEAD no longer equals origin/delphes-hh4b-production")
    if config["authorization"]["next_gate"] != NEXT_GATE:
        raise ValueError("configured next gate changed")
    if config["supersession"]["supersedes_policy_from"] != SUPERSEDED_POLICY:
        raise ValueError("superseded policy path changed")

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

    unittest_tests_run, unittests_passed = run_unittests()
    if not unittests_passed:
        raise RuntimeError("standard-library unit tests failed")

    inventory, directories = verify_source_checkpoints(config)
    facts = verify_frozen_contracts(config, directories)
    candidate_rows = build_candidate_rows(config)
    validation_rows = build_validation_role_rows(config)
    selection_rows = build_selection_rule_rows()
    metric_rows = build_metric_contract_rows(config)
    test_rows = build_test_policy_rows(config)
    superseded_rows = build_superseded_policy_rows()

    table_failures = write_protocol_tables(
        runtime_dir,
        candidate_rows,
        validation_rows,
        selection_rows,
        metric_rows,
        test_rows,
    )
    decision_audit = build_decision_audit(
        config,
        inventory,
        facts,
        candidate_rows,
        validation_rows,
        selection_rows,
        metric_rows,
        test_rows,
        unittest_tests_run=unittest_tests_run,
        unittests_passed=unittests_passed,
        table_failures=table_failures,
    )
    if not all(row["passed"] for row in decision_audit):
        failed = [row["condition"] for row in decision_audit if not row["passed"]]
        raise RuntimeError(f"decision audit failed: {failed}")

    write_tsv(
        runtime_dir / "superseded_policy_inventory.tsv",
        superseded_rows,
        (
            "policy",
            "previous_value",
            "amended_value",
            "disposition",
            "rationale",
            "status",
        ),
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
        facts,
        inventory,
        candidate_rows,
        decision_audit,
        source_commit=source_commit,
        config_sha256=sha256_file(config_path),
        runner_sha256=sha256_file(Path(__file__)),
        unittest_tests_run=unittest_tests_run,
        unittests_passed=unittests_passed,
        table_failures=table_failures,
    )
    if summary["failed_pass_conditions"]:
        write_json(runtime_dir / "summary.json", summary)
        raise RuntimeError(
            f"protocol amendment failed: {summary['failed_pass_conditions']}"
        )
    write_json(runtime_dir / "summary.json", summary)
    checkpoint_record = {
        "schema_version": 2,
        "classification": (
            "canonical_hh4b_bdt_validation_model_selection_protocol_amendment"
        ),
        "status": STATUS,
        "source_commit": source_commit,
        "supersedes_policy_from": SUPERSEDED_POLICY,
        "input_checkpoints": {
            Path(row["checkpoint_dir"]).name: row["sha256sums_sha256"]
            for row in inventory
        },
        "policy": {
            "both_bdt_candidates_retained": True,
            "final_nominal_model_selected": False,
            "validation_model_selection_authorized": True,
            "post_validation_reoptimization_authorized": False,
            "test_access_authorized": False,
            "test_selection_authorized": False,
        },
        "next_gate": NEXT_GATE,
    }
    write_json(runtime_dir / "checkpoint.json", checkpoint_record)
    write_readme(runtime_dir, summary)
    write_sha256_manifest(runtime_dir)
    verify_sha256_manifest(runtime_dir)
    validate_output_inventory(runtime_dir)

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    shutil.copytree(runtime_dir, checkpoint_dir)
    verify_sha256_manifest(checkpoint_dir)
    validate_output_inventory(checkpoint_dir)

    # Prove that every source manifest, especially the superseded policy
    # checkpoint, still has its configured digest after output generation.
    for row in inventory:
        source_manifest = REPOSITORY_ROOT / row["checkpoint_dir"] / "SHA256SUMS"
        if sha256_file(source_manifest) != row["sha256sums_sha256"]:
            raise RuntimeError(
                f"source checkpoint changed during amendment: "
                f"{row['checkpoint_dir']}"
            )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Runtime output: {runtime_dir}")
    print(f"Checkpoint: {checkpoint_dir}")


if __name__ == "__main__":
    main()
