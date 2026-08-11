from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from build_hh4b_cut_baseline_final_report import (  # noqa: E402
    FinalReportError,
    NOMINAL_THRESHOLDS,
    validate_final_summaries,
)


def valid_summaries():
    sealed = {"validation_payloads_opened": 0, "test_payloads_opened": 0}
    opened = {"validation_payloads_opened": 1, "test_payloads_opened": 0}
    return {
        "initial200": {**sealed, "status": "pass_initial200_predeclared_selection_stability_aggregation"},
        "all1000": {**sealed, "status": "pass_all1000_predeclared_selection_stability_aggregation", "nominal_deployment_candidate_changed": False},
        "train": {
            **sealed,
            "status": "pass_hh4b_cut_baseline_train_performance_and_fixed_comparator",
            "primary_generalization_estimate": "pooled_nested_outer_oof",
            "historical_comparator_threshold_scan_performed": False,
            "historical_comparator_threshold": {"r_hh_125_125_lt": 34.0},
            "official_cms_result": False,
        },
        "publication_train": {
            **sealed,
            "status": "pass_hh4b_cut_baseline_train_only_publication_figures",
            "official_cms_status_claimed": False,
        },
        "master": {**sealed, "status": "pass_master_train_only_hh4b_cut_baseline_freeze", "nominal_thresholds": NOMINAL_THRESHOLDS},
        "authorization": {
            "status": "authorized_one_time_cut_baseline_validation",
            "validation_payloads_opened": 0,
            "test_payloads_opened": 0,
        },
        "campaign": {**sealed, "status": "pass_exactly_once_validation_campaign_frozen_before_submission"},
        "submission": {
            "status": "pass_exactly_once_validation_submission_frozen_before_monitoring",
            "validation_payloads_opened_before_submission": 0,
            "test_payloads_opened": 0,
        },
        "validation_returns": {
            **opened,
            "status": "pass_complete_one_time_cut_baseline_validation_return_audit",
            "physical_validation_sources_opened_once": 116,
            "source_payload_reruns": 0,
        },
        "validation": {
            **opened,
            "status": "pass_one_time_hh4b_cut_baseline_validation_aggregation",
            "nominal_selection_changed": False,
            "validation_payload_reruns": 0,
            "official_cms_result": False,
        },
        "publication_validation": {**opened, "status": "pass_cut_baseline_validation_publication_figure"},
    }


def test_fully_sealed_final_contract_passes() -> None:
    validate_final_summaries(valid_summaries())


def test_retuning_rerun_or_test_open_fails_final_report() -> None:
    for role, field, value in (
        ("master", "nominal_thresholds", {}),
        ("validation_returns", "source_payload_reruns", 1),
        ("validation", "nominal_selection_changed", True),
        ("publication_validation", "test_payloads_opened", 1),
    ):
        summaries = valid_summaries()
        summaries[role][field] = value
        try:
            validate_final_summaries(summaries)
        except FinalReportError:
            pass
        else:
            raise AssertionError(f"unsafe final-report input passed: {role}/{field}")


def test_repository_gate_targets_bound_artifacts_only() -> None:
    source = (SCRIPT_DIR / "build_hh4b_cut_baseline_final_report.py").read_text(
        encoding="utf-8"
    )
    main_body = source.split("def main()", 1)[1]
    assert "status\", \"--porcelain" not in main_body
    assert "require_committed_unchanged(repo, root)" in main_body


if __name__ == "__main__":
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, function in tests:
        function()
        print(f"{name}=PASS")
    print(f"TEST_COUNT={len(tests)}")
