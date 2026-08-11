from __future__ import annotations

import copy

import pytest

from scripts.analysis import recover_hh4b_cut_baseline_validation_transport_in_place as recovery


def synthetic_rows_and_ads() -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    rows: dict[int, dict[str, object]] = {}
    ads: list[dict[str, object]] = []
    for proc in range(recovery.TARGET_COUNT):
        arguments = f"{proc} source_{proc} {recovery.CLUSTER} {proc}"
        rows[proc] = {"arguments": arguments}
        ads.append(
            {
                "ClusterId": recovery.CLUSTER,
                "ProcId": proc,
                "Owner": "iturkmen",
                "JobStatus": 5,
                "HoldReasonCode": 13,
                "HoldReasonSubCode": 2,
                "NumJobStarts": 0,
                "Cmd": str(recovery.ORIGINAL_RUNNER),
                "TransferInput": recovery.ORIGINAL_TRANSFER_INPUT,
                "Arguments": arguments,
            }
        )
    return rows, ads


def test_original_held_validation_closes_all_116_unstarted_jobs() -> None:
    rows, ads = synthetic_rows_and_ads()
    summary = recovery.validate_original_held(ads, rows)
    assert summary["queued_jobs"] == recovery.TARGET_COUNT
    assert summary["scientific_executable_starts"] == 0
    assert summary["hold_reason_13_2_jobs"] == recovery.TARGET_COUNT


def test_original_held_validation_rejects_any_prior_start() -> None:
    rows, ads = synthetic_rows_and_ads()
    ads[17]["NumJobStarts"] = 1
    with pytest.raises(recovery.RecoveryGateError, match="already started"):
        recovery.validate_original_held(ads, rows)


def test_edited_held_validation_preserves_arguments_and_checks_shared_paths() -> None:
    rows, original = synthetic_rows_and_ads()
    edited = copy.deepcopy(original)
    for ad in edited:
        ad["Cmd"] = str(recovery.FROZEN_RUNNER)
        ad["TransferInput"] = recovery.RECOVERY_TRANSFER_INPUT
    recovery.validate_edited_held(edited, rows)
    edited[23]["Arguments"] = "changed"
    with pytest.raises(recovery.RecoveryGateError, match="arguments changed"):
        recovery.validate_edited_held(edited, rows)
