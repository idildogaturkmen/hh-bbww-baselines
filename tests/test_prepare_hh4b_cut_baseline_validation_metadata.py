from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_hh4b_cut_baseline_validation_metadata import (  # noqa: E402
    build_metadata,
    hard_qcd_source_tag,
    remote_uri,
)


def synthetic_metadata() -> tuple[pd.DataFrame, ...]:
    development_rows = []
    root_rows = []
    total_events = 0

    for index in range(121):
        if index < 5:
            process = "qcd_bbbb_general"
            sample_class = "background"
        elif index == 5:
            process = "qcd_hardqcd"
            sample_class = "background"
        else:
            process = "ordinary_process"
            sample_class = "background" if index < 102 else "signal"

        events = 10_000 if index < 100 else (124 if index == 100 else 123)
        total_events += events
        basename = (
            "qcd_hardqcd_campaign_bin00_shard0005_seed5_bundle.tar.gz"
            if index == 5
            else f"source_{index:04d}_bundle.tar.gz"
        )
        lfn = f"/store/frozen/validation/{basename}"
        candidate = f"/sealed/candidate_{index:04d}.parquet"
        development_rows.append(
            {
                "sample_class": sample_class,
                "source_index": index,
                "member_id": f"bundle:{lfn}",
                "process_or_mode": process,
                "final_split": "validation",
                "group_id": f"group_{index:04d}",
                "generated_events": events,
                "candidate_rows_metadata": 0,
                "source_locator": lfn,
                "candidate_path": candidate,
                "candidate_content_opened_in_this_step": False,
                "candidate_content_access_authorized_now": False,
                "physical_normalization_authorized": False,
            }
        )
        root_rows.append(
            {
                "candidate_path": candidate,
                "remote_bundle_path": lfn,
                "resolution_status": "resolved",
                "root_archive_member": f"root/source_{index:04d}.root",
                "root_basename": f"source_{index:04d}.root",
                "layout_rule": "standard_bundle_root_target_tag",
                "bundle_adler32_expected": f"{index:08x}",
                "bundle_size_bytes": 1000 + index,
            }
        )

    assert total_events == 1_002_584
    development = pd.DataFrame(development_rows)
    root_map = pd.DataFrame(root_rows)
    ordinary = pd.DataFrame(
        [
            {
                "sample_class": sample_class,
                "process_or_mode": "ordinary_process",
                "run2_yield_coefficient_per_generator_weight": coefficient,
            }
            for sample_class, coefficient in [("background", 2.0), ("signal", 3.0)]
        ]
    )
    hard_qcd = pd.DataFrame(
        [
            {
                "source_tag": "qcd_hardqcd_campaign_bin00_shard0005_seed5",
                "campaign": "qcd_hardqcd_campaign",
                "run2_yield_coefficient_per_generator_weight": 4.0,
            }
        ]
    )
    ttbar = pd.DataFrame(
        columns=["member_id", "final_split", "generated_events_frozen", "root_sha256"]
    )
    return development, root_map, ordinary, hard_qcd, ttbar


def test_locator_helpers() -> None:
    assert remote_uri("/store/user/example.tar.gz") == (
        "root://cmseos.fnal.gov//store/user/example.tar.gz"
    )
    assert hard_qcd_source_tag(
        "/store/qcd_campaign_bin00_shard1_seed1_bundle.tar.gz"
    ) == "qcd_campaign_bin00_shard1_seed1"


def test_metadata_only_contract_closes() -> None:
    access, coefficients, summary = build_metadata(
        *synthetic_metadata(),
        legacy_root=Path("/not/used"),
    )
    assert len(access) == 121
    assert access["source_uid"].is_unique
    assert access["physical_evaluation_eligible"].sum() == 116
    assert access["auxiliary_qcd"].sum() == 5
    assert coefficients["run2_yield_coefficient_per_generator_weight"].isna().sum() == 5
    assert summary["validation_generated_events"] == 1_002_584
    assert summary["event_payload_files_opened"] == 0
    assert summary["validation_payloads_opened"] == 0
    assert summary["validation_access_authorized"] is False
    assert summary["test_payloads_opened"] == 0


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
