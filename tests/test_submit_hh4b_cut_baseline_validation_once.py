from __future__ import annotations

from pathlib import Path
import sys
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from submit_hh4b_cut_baseline_validation_once import (  # noqa: E402
    NEVER_RESUBMIT_EXISTING,
    SubmissionGateError,
    durable_write_exclusive,
    parse_submit_output,
    validate_presubmission_freeze,
)


def test_existing_never_resubmit_registry_is_complete() -> None:
    assert NEVER_RESUBMIT_EXISTING == [
        3754344,
        3755882,
        3768139,
        30002685,
        30020809,
    ]


def test_parse_exact_116_job_dynamic_schedd_output() -> None:
    raw = (
        b"30009999.0 - 30009999.115\n"
        b"Attempting to submit jobs to lpcschedd6.fnal.gov\n"
    )
    assert parse_submit_output(raw, ["lpcschedd5.fnal.gov", "lpcschedd6.fnal.gov"]) == {
        "cluster_id": 30009999,
        "first_proc": 0,
        "last_proc": 115,
        "parsed_job_count": 116,
        "schedd": "lpcschedd6.fnal.gov",
    }


def test_wrong_job_count_fails_after_acceptance_parse() -> None:
    raw = (
        b"30009999.0 - 30009999.114\n"
        b"Attempting to submit jobs to lpcschedd6.fnal.gov\n"
    )
    try:
        parse_submit_output(raw, ["lpcschedd6.fnal.gov"])
    except SubmissionGateError:
        pass
    else:
        raise AssertionError("wrong validation job count passed parser")


def test_submission_marker_write_is_exclusive() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "attempt.txt"
        durable_write_exclusive(path, b"first\n")
        try:
            durable_write_exclusive(path, b"second\n")
        except FileExistsError:
            pass
        else:
            raise AssertionError("durable submission marker was overwritten")


def test_condor_submit_call_has_no_name_override() -> None:
    source = (SCRIPT_DIR / "submit_hh4b_cut_baseline_validation_once.py").read_text(
        encoding="utf-8"
    )
    call = '["/usr/bin/bash", "/usr/local/bin/condor_submit", "-terse", str(submit_file)]'
    assert call in source
    assert '"condor_submit", "-terse", "-name"' not in source


def test_presubmission_freeze_contract_is_required() -> None:
    package_sha = "a" * 64
    audit = {"campaign_package_sha256s_sha256": package_sha}
    freeze = {
        "status": "pass_exactly_once_validation_campaign_frozen_before_submission",
        "implementation_sha256": "b" * 64,
        "condor_jobs_prepared": 116,
        "durable_marker_precedes_source_access": True,
        "automatic_validation_source_rerun_authorized": False,
        "remote_output_namespace_absent": True,
        "production_submission_performed": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "campaign_package_sha256s_sha256": package_sha,
    }
    validate_presubmission_freeze(freeze, audit)
    freeze["automatic_validation_source_rerun_authorized"] = True
    try:
        validate_presubmission_freeze(freeze, audit)
    except SubmissionGateError:
        pass
    else:
        raise AssertionError("validation rerun authorization passed submission freeze gate")


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
