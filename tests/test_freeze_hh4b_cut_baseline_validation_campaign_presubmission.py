from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from freeze_hh4b_cut_baseline_validation_campaign_presubmission import (  # noqa: E402
    ValidationCampaignFreezeError,
    validate_campaign_freeze_inputs,
)


def valid_inputs():
    package = {
        "status": "pass_prepared_exactly_once_cut_baseline_validation_campaign",
        "condor_jobs_prepared": 116,
        "authorized_physical_validation_sources": 116,
        "auxiliary_qcd_validation_sources_in_queue": 0,
        "durable_marker_created_before_each_source_access": True,
        "automatic_source_rerun_authorized": False,
        "production_submission_performed": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    audit = {
        "status": "pass_exactly_once_cut_baseline_validation_campaign_pre_submission_audit",
        "condor_jobs": 116,
        "durable_marker_precedes_source_access": True,
        "automatic_validation_rerun_authorized": False,
        "remote_output_namespace_absent": True,
        "production_submission_performed": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    return package, audit


def test_exactly_once_presubmission_contract_passes() -> None:
    validate_campaign_freeze_inputs(*valid_inputs())


def test_marker_or_seal_drift_fails_presubmission_freeze() -> None:
    for target, field, value in (
        ("package", "durable_marker_created_before_each_source_access", False),
        ("audit", "automatic_validation_rerun_authorized", True),
        ("audit", "remote_output_namespace_absent", False),
        ("package", "validation_payloads_opened", 1),
        ("audit", "test_payloads_opened", 1),
    ):
        package, audit = valid_inputs()
        (package if target == "package" else audit)[field] = value
        try:
            validate_campaign_freeze_inputs(package, audit)
        except ValidationCampaignFreezeError:
            pass
        else:
            raise AssertionError(f"unsafe pre-submission drift passed: {target}/{field}")


def test_repository_gate_allows_unrelated_prepared_worktree() -> None:
    source = (
        SCRIPT_DIR / "freeze_hh4b_cut_baseline_validation_campaign_presubmission.py"
    ).read_text(encoding="utf-8")
    main_body = source.split("def main()", 1)[1]
    assert "status\", \"--porcelain" not in main_body
    assert "head == remote" in main_body


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
