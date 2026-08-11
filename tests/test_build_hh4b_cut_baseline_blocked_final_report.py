from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from build_hh4b_cut_baseline_blocked_final_report import (  # noqa: E402
    BlockedFinalReportError,
    NOMINAL_THRESHOLDS,
    render_markdown,
    validate_blocked_summaries,
)


def valid_summaries():
    sealed = {"validation_payloads_opened": 0, "test_payloads_opened": 0}
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
            "figures": 16,
        },
        "master": {
            **sealed,
            "status": "pass_master_train_only_hh4b_cut_baseline_freeze",
            "nominal_thresholds": NOMINAL_THRESHOLDS,
            "nominal_selection_changed": False,
        },
        "authorization": {**sealed, "status": "authorized_one_time_cut_baseline_validation"},
        "campaign": {**sealed, "status": "pass_exactly_once_validation_campaign_frozen_before_submission"},
        "submission": {
            "status": "pass_exactly_once_validation_submission_frozen_before_monitoring",
            "cluster_id": 3795859,
            "authoritative_schedd": "lpcschedd4.fnal.gov",
            "do_not_resubmit_cluster": True,
            "jobs_submitted": 116,
            "validation_payloads_opened_before_submission": 0,
            "test_payloads_opened": 0,
        },
        "transport": {
            "status": "pass_validation_cluster_in_place_transport_recovery_checkpoint_freeze",
            "cluster_id": 3795859,
            "authoritative_schedd": "lpcschedd4.fnal.gov",
            "do_not_resubmit_cluster": True,
            "condor_submit_called": False,
            "scientific_arguments_changed": False,
            "validation_payloads_opened_before_release": 0,
            "test_payloads_opened": 0,
        },
        "failure": {
            **sealed,
            "status": "blocked_validation_campaign_failed_before_payload_access",
            "cluster_id": 3795859,
            "authoritative_schedd": "lpcschedd4.fnal.gov",
            "do_not_resubmit_cluster": True,
            "queue_jobs": 0,
            "history_jobs": 116,
            "exit_code_1_jobs": 116,
            "runtime_probe_failure_jobs": 116,
            "durable_source_attempt_markers_created": 0,
            "validation_workers_started": 0,
            "validation_sources_opened": 0,
            "validation_evaluation_cycles": 0,
            "validation_metrics_available": False,
            "remote_result_namespace_absent": True,
            "second_validation_submission_authorized": False,
            "nominal_selection_changed": False,
            "failure_stage": "runtime_probe_before_marker_and_worker",
            "failure_exception": "ModuleNotFoundError: No module named 'awkward'",
            "root_cause": "frozen runner exported $SCRATCH/runtime/site-packages while the authorized archive stores packages under lib/python3.9/site-packages",
        },
    }


def test_blocked_preaccess_contract_passes() -> None:
    validate_blocked_summaries(valid_summaries())


def test_payload_metric_or_retry_claim_fails_closed() -> None:
    for field, value in (
        ("validation_payloads_opened", 1),
        ("validation_metrics_available", True),
        ("second_validation_submission_authorized", True),
        ("validation_workers_started", 1),
    ):
        summaries = valid_summaries()
        summaries["failure"][field] = value
        try:
            validate_blocked_summaries(summaries)
        except BlockedFinalReportError:
            pass
        else:
            raise AssertionError(f"unsafe validation state passed: {field}")


def test_test_access_or_cut_change_fails_closed() -> None:
    for role, field, value in (
        ("failure", "test_payloads_opened", 1),
        ("master", "nominal_thresholds", {}),
        ("all1000", "nominal_deployment_candidate_changed", True),
    ):
        summaries = valid_summaries()
        summaries[role][field] = value
        try:
            validate_blocked_summaries(summaries)
        except BlockedFinalReportError:
            pass
        else:
            raise AssertionError(f"unsafe final state passed: {role}/{field}")


def test_markdown_explicitly_omits_validation_figure() -> None:
    metric = {
        "signal_physical_efficiency": 0.6,
        "background_physical_efficiency": 0.3,
        "background_rejection": 3.3,
        "signal_over_background": 4e-6,
        "asimov_significance_stat_only": 0.027,
        "signal_selected_effective_events": 14000.0,
        "background_selected_effective_events": 63.0,
    }
    report = {
        "train_only": {
            "pooled_nested_outer_oof": {"exact3tag": metric, "ge4tag": metric, "combined": metric},
            "historical_rhh125125_lt34": {"combined": metric},
            "paired_bootstrap_combined_za_difference": {
                "difference_median": "0.0",
                "difference_p2p5": "-1.0",
                "difference_p97p5": "1.0",
            },
            "all1000_selection_stability": {
                "categories": {
                    "exact3tag": {
                        "modal_replica_selected_structure_id": "a",
                        "modal_replica_selected_frequency": 0.2,
                        "nominal_structure_recovery_frequency": 0.1,
                        "top_two_frequency_gap": 0.05,
                        "tie_resolution_replica_category_fraction": 0.38,
                        "infeasible_fold_winner_frequency": 0.55,
                    }
                }
            },
        },
        "validation": {
            "cluster_id": 3795859,
            "authoritative_schedd": "lpcschedd4.fnal.gov",
            "failure_audit_checkpoint": "failure",
            "failure_audit_checkpoint_sha256s_sha256": "1" * 64,
            "frozen_runner_sha256": "2" * 64,
            "runtime_bundle_sha256": "3" * 64,
        },
        "clusters": [],
        "checkpoints": [],
        "next_decision_required": "separate decision",
    }
    text = render_markdown(report)
    assert "Train-only cut-baseline study: **COMPLETE**" in text
    assert "Validation scientific result: **NOT AVAILABLE**" in text
    assert "Figure 17 / validation-comparison products were intentionally not produced" in text
    assert "CORRECTED_VALIDATION_RETRY_AUTHORIZED=FALSE" in text


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
