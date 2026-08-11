from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from freeze_hh4b_cut_baseline_validation_metadata import (  # noqa: E402
    ValidationMetadataFreezeError,
    validate_metadata_freeze,
)


def inputs():
    access_rows = []
    coefficient_rows = []
    for index in range(121):
        physical = index >= 5
        access_rows.append(
            {
                "source_uid": f"source_{index}",
                "physical_evaluation_eligible": physical,
                "auxiliary_qcd": not physical,
                "validation_content_opened": False,
                "test_content_opened": False,
            }
        )
        coefficient_rows.append(
            {
                "source_uid": f"source_{index}",
                "run2_yield_coefficient_per_generator_weight": 2.0 if physical else "",
            }
        )
    remote = {
        "status": "pass_metadata_only_validation_remote_bundle_checksum_closure",
        "remote_bundles": 116,
        "independent_checksum_queries_per_bundle": 2,
        "all_independent_queries_match": True,
        "new_metadata_checksum_closures": 97,
        "preexisting_frozen_checksums_verified": 19,
    }
    metadata = {
        "status": "pass_metadata_only",
        "physical_evaluation_sources": 116,
        "auxiliary_qcd_sources": 5,
        "validation_sources": 121,
        "validation_generated_events": 1_002_584,
        "remote_bundle_checksum_closure": "pass_116_of_116",
        "event_payload_files_opened": 0,
        "validation_access_authorized": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    return remote, metadata, pd.DataFrame(access_rows), pd.DataFrame(coefficient_rows)


def test_metadata_only_freeze_contract_passes() -> None:
    validate_metadata_freeze(*inputs())


def test_opened_payload_or_checksum_drift_fails_metadata_freeze() -> None:
    for target, field, value in (
        ("remote", "all_independent_queries_match", False),
        ("metadata", "validation_access_authorized", True),
        ("metadata", "test_payloads_opened", 1),
    ):
        remote, metadata, access, coefficients = inputs()
        (remote if target == "remote" else metadata)[field] = value
        try:
            validate_metadata_freeze(remote, metadata, access, coefficients)
        except ValidationMetadataFreezeError:
            pass
        else:
            raise AssertionError(f"unsafe metadata freeze passed: {target}/{field}")


def test_repository_gate_allows_unrelated_prepared_worktree() -> None:
    source = (SCRIPT_DIR / "freeze_hh4b_cut_baseline_validation_metadata.py").read_text(
        encoding="utf-8"
    )
    main_body = source.split("def main()", 1)[1]
    assert "status\", \"--porcelain" not in main_body
    assert "require_committed_unchanged(repo, path)" in main_body


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
