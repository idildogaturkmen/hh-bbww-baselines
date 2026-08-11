from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from freeze_hh4b_cut_baseline_train_artifact_checkpoint import (  # noqa: E402
    ALL1000_AUDIT_STATUS,
    ArtifactFreezeError,
    ROLE_CONTRACTS,
    should_copy_all1000_file,
    validate_all1000_audit,
    validate_role_summary,
)


def sealed(status: str):
    return {"status": status, "validation_payloads_opened": 0, "test_payloads_opened": 0}


def test_all_three_role_contracts_pass() -> None:
    validate_role_summary(
        "all1000",
        {
            **sealed(ROLE_CONTRACTS["all1000"]["status"]),
            "fold_level_winner_count": 10000,
            "replica_category_result_count": 2000,
            "ranked_structure_result_count": 270000,
            "nominal_deployment_candidate_changed": False,
        },
    )
    validate_role_summary(
        "train_performance",
        {
            **sealed(ROLE_CONTRACTS["train_performance"]["status"]),
            "primary_generalization_estimate": "pooled_nested_outer_oof",
            "fixed_deployment_cut_called_primary_generalization_estimate": False,
            "historical_comparator_threshold_scan_performed": False,
            "historical_comparator_threshold": {"r_hh_125_125_lt": 34.0},
            "official_cms_result": False,
        },
    )
    validate_role_summary(
        "publication_train_only",
        {
            **sealed(ROLE_CONTRACTS["publication_train_only"]["status"]),
            "figures": 16,
            "pdfs": 16,
            "pngs": 16,
            "figure_data_sidecars": 16,
            "official_cms_status_claimed": False,
        },
    )


def test_opened_validation_fails_artifact_freeze() -> None:
    payload = {
        **sealed(ROLE_CONTRACTS["all1000"]["status"]),
        "fold_level_winner_count": 10000,
        "replica_category_result_count": 2000,
        "ranked_structure_result_count": 270000,
        "nominal_deployment_candidate_changed": False,
        "validation_payloads_opened": 1,
    }
    try:
        validate_role_summary("all1000", payload)
    except ArtifactFreezeError:
        pass
    else:
        raise AssertionError("opened validation passed train artifact freeze")


def test_all1000_audit_requires_complete_independent_and_rerun_proof() -> None:
    source_sums = "a" * 64
    audit = {
        "status": ALL1000_AUDIT_STATUS,
        "ranked_structure_results_audited": 270000,
        "fold_level_winners_independently_selected": 10000,
        "replica_category_reductions_independently_recomputed": 2000,
        "initial200_ranked_rows_proven_identical": 54000,
        "initial200_replica_category_reductions_proven_identical": 400,
        "deterministic_rerun": {
            "status": "pass_every_output_file_byte_identical",
            "file_count": 29,
        },
        "source_sha256s_sha256": source_sums,
        "pilot_results_used": False,
        "nominal_deployment_candidate_changed": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    validate_all1000_audit(audit, source_sums)
    audit["deterministic_rerun"]["file_count"] = 28
    try:
        validate_all1000_audit(audit, source_sums)
    except ArtifactFreezeError:
        pass
    else:
        raise AssertionError("incomplete deterministic rerun passed all1000 freeze")


def test_all1000_compact_copy_excludes_only_source_sums_and_ranked_shards() -> None:
    assert should_copy_all1000_file("all1000_stability_summary.json") is True
    assert should_copy_all1000_file("tables/structure_frequencies.tsv") is True
    assert should_copy_all1000_file("SHA256SUMS") is False
    assert should_copy_all1000_file(
        "tables/ranked_structure_results_replicas_0000_0099.tsv"
    ) is False


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
