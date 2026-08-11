from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from freeze_hh4b_cut_baseline_validation_artifact_checkpoint import (  # noqa: E402
    ROLE_CONTRACTS,
    ValidationArtifactFreezeError,
    validate_role_summary,
)


def sealed(status: str):
    return {"status": status, "validation_payloads_opened": 1, "test_payloads_opened": 0}


def valid_returns():
    return {
        **sealed(ROLE_CONTRACTS["validation_returns"]["status"]),
        "clean_history_jobs": 116,
        "source_attempt_markers": 116,
        "source_summaries": 116,
        "source_distributions": 116,
        "job_receipts": 116,
        "distribution_rows": 116 * 1360,
        "physical_validation_sources_opened_once": 116,
        "auxiliary_qcd_validation_sources_opened": 0,
        "auxiliary_qcd_validation_sources_remain_sealed": 5,
        "validation_evaluation_cycles": 1,
        "source_payload_reruns": 0,
        "cut_scan_performed": False,
        "threshold_adjustment_performed": False,
        "family_adjustment_performed": False,
        "do_not_resubmit_cluster": True,
    }


def test_all_validation_artifact_contracts_pass() -> None:
    validate_role_summary("validation_returns", valid_returns())
    validate_role_summary(
        "validation_performance",
        {
            **sealed(ROLE_CONTRACTS["validation_performance"]["status"]),
            "validation_sources_opened": 116,
            "validation_source_payloads_each_opened_exactly_once": True,
            "validation_payload_reruns": 0,
            "auxiliary_qcd_validation_sources_excluded_and_unopened": 5,
            "cut_scan_performed": False,
            "threshold_adjustment_performed": False,
            "family_adjustment_performed": False,
            "nominal_selection_changed": False,
            "official_cms_result": False,
            "pooled_metrics": {"combined": {}},
        },
    )
    validate_role_summary(
        "publication_validation",
        {
            **sealed(ROLE_CONTRACTS["publication_validation"]["status"]),
            "outputs": {"figure.pdf": {}, "figure.png": {}, "figure.tsv": {}},
            "official_cms_result": False,
        },
    )


def test_rerun_retuning_or_test_access_fails_validation_freeze() -> None:
    for field, value in (
        ("source_payload_reruns", 1),
        ("threshold_adjustment_performed", True),
        ("test_payloads_opened", 1),
    ):
        payload = valid_returns()
        payload[field] = value
        try:
            validate_role_summary("validation_returns", payload)
        except ValidationArtifactFreezeError:
            pass
        else:
            raise AssertionError(f"unsafe validation artifact passed freeze: {field}")


def test_repository_gate_targets_only_the_source_artifact() -> None:
    source = (
        SCRIPT_DIR / "freeze_hh4b_cut_baseline_validation_artifact_checkpoint.py"
    ).read_text(encoding="utf-8")
    main_body = source.split("def main()", 1)[1]
    assert "status\", \"--porcelain" not in main_body
    assert '["git", "diff", "--quiet", "--", relative_source]' in main_body
    assert '["git", "diff", "--cached", "--quiet", "--", relative_source]' in main_body


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
