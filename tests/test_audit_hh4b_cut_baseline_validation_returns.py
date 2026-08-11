from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from audit_hh4b_cut_baseline_validation_returns import (  # noqa: E402
    ReturnAuditError,
    expected_remote_files,
    validate_history,
)


def clean_history(cluster: int = 30009999):
    return [
        {
            "ClusterId": cluster,
            "ProcId": proc,
            "JobStatus": 4,
            "ExitCode": 0,
            "ExitBySignal": False,
            "NumJobStarts": 1 if proc else 2,
        }
        for proc in range(116)
    ]


def test_clean_116_job_history_closes_and_allows_pre_marker_restart() -> None:
    observed = validate_history(clean_history(), 30009999)
    assert len(observed) == 116
    assert observed[0]["NumJobStarts"] == 2


def test_nonclean_history_fails_closed() -> None:
    history = clean_history()
    history[7]["ExitCode"] = 1
    try:
        validate_history(history, 30009999)
    except ReturnAuditError:
        pass
    else:
        raise AssertionError("nonclean validation history passed")


def test_remote_inventory_names_are_source_specific() -> None:
    physical = pd.DataFrame(
        {
            "production_row_index": [2, 9],
            "source_uid": ["a", "b"],
        }
    )
    expected = expected_remote_files("/store/user/iturkmen/validation", physical)
    assert expected["markers"] == [
        "/store/user/iturkmen/validation/markers/source_0002_VALIDATION_OPEN_DO_NOT_RERUN.json",
        "/store/user/iturkmen/validation/markers/source_0009_VALIDATION_OPEN_DO_NOT_RERUN.json",
    ]
    assert expected["summaries"][1].endswith("source_0009_summary.json")
    assert expected["distributions"][0].endswith("source_0002_distributions.tsv")
    assert expected["receipts"][1].endswith("source_0009_job_receipt.json")


def test_return_auditor_never_calls_validation_source_extractor() -> None:
    source = (SCRIPT_DIR / "audit_hh4b_cut_baseline_validation_returns.py").read_text(
        encoding="utf-8"
    )
    assert "stage_source(" not in source
    assert "build_table(" not in source
    assert "Validation ROOT payloads" in source
    assert 'parser.add_argument("--submission-checkpoint"' in source
    assert 'parser.add_argument("--submission-receipt"' not in source


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
