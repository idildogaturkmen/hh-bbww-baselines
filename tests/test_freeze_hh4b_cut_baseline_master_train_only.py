from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from freeze_hh4b_cut_baseline_master_train_only import (  # noqa: E402
    MasterFreezeError,
    NOMINAL_THRESHOLDS,
    validate_summary_contracts,
)


def sealed(status: str):
    return {
        "status": status,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }


def valid_contracts():
    full270 = sealed("pass_full_270_returned_result_integrity_audit")
    nested = {
        **sealed("pass_outer_fold_category_family_selection"),
        "winner_count": 10,
        "outer_fold_results_used_for_selection": False,
    }
    deployment = {
        **sealed("train_only_deployment_candidate_frozen"),
        "nominal_selection_is_changed_by_stability_escalation": False,
        "categories": {
            category: {
                "deployment_thresholds_coordinatewise_median": dict(thresholds)
            }
            for category, thresholds in NOMINAL_THRESHOLDS.items()
        },
    }
    initial200 = {
        **sealed("pass_initial200_predeclared_selection_stability_aggregation"),
        "nominal_deployment_candidate_changed": False,
    }
    escalation800 = {
        **sealed("pass_complete_escalation800_8000_job_216000_structure_return_audit"),
        "jobs_audited": 8000,
        "structure_results_audited": 216000,
        "nominal_selection_changed": False,
    }
    all1000 = {
        **sealed("pass_all1000_predeclared_selection_stability_aggregation"),
        "fold_level_winner_count": 10000,
        "replica_category_result_count": 2000,
        "ranked_structure_result_count": 270000,
        "nominal_deployment_candidate_changed": False,
    }
    train = {
        **sealed("pass_hh4b_cut_baseline_train_performance_and_fixed_comparator"),
        "primary_generalization_estimate": "pooled_nested_outer_oof",
        "fixed_deployment_cut_called_primary_generalization_estimate": False,
        "historical_comparator_threshold_scan_performed": False,
        "historical_comparator_threshold": {"r_hh_125_125_lt": 34.0},
        "official_cms_result": False,
    }
    publication = {
        **sealed("pass_hh4b_cut_baseline_train_only_publication_figures"),
        "figures": 16,
        "pdfs": 16,
        "pngs": 16,
        "figure_data_sidecars": 16,
        "official_cms_status_claimed": False,
    }
    validation_metadata = {
        **sealed("pass_metadata_only"),
        "event_payload_files_opened": 0,
        "validation_access_authorized": False,
        "physical_evaluation_sources": 116,
        "auxiliary_qcd_sources": 5,
        "remote_bundle_checksum_closure": "pass_116_of_116",
    }
    return {
        "full270": full270,
        "nested": nested,
        "deployment": deployment,
        "initial200": initial200,
        "escalation800": escalation800,
        "all1000": all1000,
        "train": train,
        "publication": publication,
        "validation_metadata": validation_metadata,
    }


def test_complete_train_only_contract_passes() -> None:
    validate_summary_contracts(**valid_contracts())


def test_nominal_threshold_drift_fails_master_freeze() -> None:
    contracts = valid_contracts()
    contracts["deployment"]["categories"]["exact3tag"][
        "deployment_thresholds_coordinatewise_median"
    ]["r_hh_125_125"] = 36.0
    try:
        validate_summary_contracts(**contracts)
    except MasterFreezeError:
        pass
    else:
        raise AssertionError("changed nominal threshold passed master freeze")


def test_validation_or_test_access_fails_master_freeze() -> None:
    for role, field in (("train", "validation_payloads_opened"), ("publication", "test_payloads_opened")):
        contracts = valid_contracts()
        contracts[role][field] = 1
        try:
            validate_summary_contracts(**contracts)
        except MasterFreezeError:
            pass
        else:
            raise AssertionError(f"opened sealed payload passed master freeze: {role}/{field}")


def test_repository_gate_targets_only_master_inputs() -> None:
    source = (SCRIPT_DIR / "freeze_hh4b_cut_baseline_master_train_only.py").read_text(
        encoding="utf-8"
    )
    verify_body = source.split("def verify_repo", 1)[1].split("def verify_checkpoint", 1)[0]
    assert "status\", \"--porcelain" not in verify_body
    assert "require_committed_unchanged(repo, checkpoint)" in source
    assert "require_committed_unchanged(repo, path)" in source


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
