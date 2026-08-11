from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from freeze_hh4b_escalation800_complete_return_audit import (  # noqa: E402
    CHECKPOINT_COPY_FILES,
    FreezeError,
    SHARD_NAMES,
    validate_audit_summary,
)


def valid_payloads():
    audit = {
        "status": "pass_complete_escalation800_8000_job_216000_structure_return_audit",
        "cluster_id": 30020809,
        "authoritative_schedd": "lpcschedd5.fnal.gov",
        "jobs_audited": 8000,
        "structure_results_audited": 216000,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    for key in (
        "all_job_receipts_clean",
        "all_runner_logs_match_receipt_sha256",
        "all_structure_internal_sha256s_verified",
        "all_structure_canonical_payload_sha256s_recomputed",
        "all_inner_crossfit_four_fold_coverage_verified",
        "all_execution_provenance_sha256s_match_frozen_payload_manifest",
        "all_outer_fold_used_for_selection_false",
        "all_validation_payloads_opened_zero",
        "all_test_payloads_opened_zero",
        "exact_structure_set_per_category_fold_verified",
        "never_resubmit_cluster",
    ):
        audit[key] = True
    for key in (
        "aggregation_performed",
        "fold_level_winners_selected",
        "replica_level_stability_statistics_computed",
        "pilot_results_enter_stability_aggregation",
        "nominal_selection_changed",
    ):
        audit[key] = False
    manifest = {
        "status": "pass_escalation800_structure_result_inventory_sharded",
        "total_rows_excluding_headers": 216000,
        "shard_count": 8,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    return audit, manifest


def test_complete_audit_contract_passes() -> None:
    validate_audit_summary(*valid_payloads())


def test_feasibility_is_not_required_but_integrity_is() -> None:
    audit, manifest = valid_payloads()
    audit["scientific_outcome_diagnostics"] = {
        "pooled_inner_oof_feasible_structure_results": 0,
        "pooled_inner_oof_infeasible_structure_results": 216000,
    }
    validate_audit_summary(audit, manifest)
    audit["all_structure_internal_sha256s_verified"] = False
    try:
        validate_audit_summary(audit, manifest)
    except FreezeError:
        pass
    else:
        raise AssertionError("failed structural integrity passed freeze")


def test_checkpoint_is_compact_and_binds_large_tables_by_manifest() -> None:
    assert CHECKPOINT_COPY_FILES == {
        "complete_return_audit.json",
        "complete_return_audit.txt",
        "structure_result_inventory_manifest.json",
    }
    assert not CHECKPOINT_COPY_FILES.intersection(SHARD_NAMES)
    assert "job_return_audit.tsv" not in CHECKPOINT_COPY_FILES
    assert "final_history_snapshot.txt" not in CHECKPOINT_COPY_FILES


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
