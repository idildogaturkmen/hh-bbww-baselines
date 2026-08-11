from __future__ import annotations

from pathlib import Path
import sys
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from freeze_hh4b_cut_baseline_validation_submission import (  # noqa: E402
    EXISTING_CLUSTERS,
    ValidationSubmissionFreezeError,
    parse_marker,
    validate_submission_payloads,
)


def test_durable_marker_parser_rejects_duplicate_keys() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "marker.txt"
        path.write_text("status=one\nvalue=two\n", encoding="utf-8")
        assert parse_marker(path) == {"status": "one", "value": "two"}
        path.write_text("status=one\nstatus=two\n", encoding="utf-8")
        try:
            parse_marker(path)
        except ValidationSubmissionFreezeError:
            pass
        else:
            raise AssertionError("duplicate durable marker key passed")


def valid_payloads():
    cluster = 30030000
    common = {"submit_file_sha256": "a", "queue_items_sha256": "b", "runner_sha256": "c"}
    gate = {
        "status": "pass_exactly_once_cut_baseline_validation_submission_preflight",
        "expected_jobs": 116,
        "no_prior_scheduler_association": True,
        "remote_output_namespace_absent": True,
        "production_submission_performed": False,
        "validation_payloads_opened_before_submission": 0,
        "test_payloads_opened": 0,
        **common,
    }
    parsed = {
        "status": "pass_single_contiguous_validation_116_condor_submission_parse",
        "cluster_id": cluster,
        "schedd": "lpcschedd6.fnal.gov",
        "first_proc": 0,
        "last_proc": 115,
        "parsed_job_count": 116,
    }
    receipt = {
        "status": "pass_one_time_cut_baseline_validation_condor_submission_accepted_and_parsed",
        "cluster_id": cluster,
        "schedd": "lpcschedd6.fnal.gov",
        "expected_job_count": 116,
        "do_not_resubmit_cluster": True,
        "never_resubmit_clusters": [*EXISTING_CLUSTERS, cluster],
        "validation_payloads_opened_before_submission": 0,
        "validation_execution_authorized": True,
        "test_access_authorized": False,
        "test_payloads_opened": 0,
        **common,
    }
    return gate, parsed, receipt


def test_exactly_once_validation_submission_contract_passes() -> None:
    validate_submission_payloads(*valid_payloads())


def test_prior_association_or_bad_registry_fails_submission_freeze() -> None:
    for target, field, value in (
        ("gate", "no_prior_scheduler_association", False),
        ("parsed", "last_proc", 114),
        ("receipt", "do_not_resubmit_cluster", False),
        ("receipt", "never_resubmit_clusters", []),
        ("receipt", "test_payloads_opened", 1),
    ):
        gate, parsed, receipt = valid_payloads()
        {"gate": gate, "parsed": parsed, "receipt": receipt}[target][field] = value
        try:
            validate_submission_payloads(gate, parsed, receipt)
        except ValidationSubmissionFreezeError:
            pass
        else:
            raise AssertionError(f"unsafe validation submission freeze passed: {target}/{field}")


def test_repository_gate_allows_unrelated_prepared_worktree() -> None:
    source = (SCRIPT_DIR / "freeze_hh4b_cut_baseline_validation_submission.py").read_text(
        encoding="utf-8"
    )
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
