from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_hh4b_cut_baseline_validation_metadata import (  # noqa: E402
    build_metadata,
    hard_qcd_source_tag,
    remote_uri,
)


def synthetic_metadata(legacy_root: Path) -> tuple[pd.DataFrame, ...]:
    development_rows = []
    root_rows = []
    total_events = 0

    for index in range(121):
        if index < 5:
            process = "qcd_bbbb_general"
            sample_class = "background"
        elif 5 <= index < 10:
            process = "ttbar_inclusive"
            sample_class = "background"
        elif index == 10:
            process = "qcd_hardqcd"
            sample_class = "background"
        else:
            process = "ordinary_process"
            sample_class = "background" if index < 102 else "signal"

        events = 10_000 if index < 100 else (124 if index == 100 else 123)
        total_events += events
        basename = (
            "qcd_hardqcd_campaign_bin00_shard0010_seed10_bundle.tar.gz"
            if index == 10
            else f"source_{index:04d}_bundle.tar.gz"
        )
        lfn = f"/store/frozen/validation/{basename}"
        candidate = f"/sealed/candidate_{index:04d}.parquet"
        local_tokens = {
            5: "ttbar_100k_shard000",
            6: "ttbar_100k_shard004",
            7: "ttbar_100k_shard005",
            8: "ttbar_200k_shard008",
            9: "ttbar_200k_shard015",
        }
        local_token = local_tokens.get(index)
        member_id = (
            f"event_parquet:/fake/{local_token}_event_summary.parquet"
            if local_token
            else f"bundle:{lfn}"
        )
        development_rows.append(
            {
                "sample_class": sample_class,
                "source_index": index,
                "member_id": member_id,
                "process_or_mode": process,
                "final_split": "validation",
                "group_id": f"group_{index:04d}",
                "generated_events": events,
                "candidate_rows_metadata": 0,
                "source_locator": candidate if local_token else lfn,
                "candidate_path": candidate,
                "candidate_content_opened_in_this_step": False,
                "candidate_content_access_authorized_now": False,
                "physical_normalization_authorized": False,
            }
        )
        if local_token:
            if local_token in {"ttbar_100k_shard004", "ttbar_100k_shard005"}:
                root_path = (
                    legacy_root
                    / "condor_return/hh4b_ttbar7_exact_regeneration_scaleout_20260728_v1"
                    / "members"
                    / local_token
                    / "root"
                    / f"{local_token}_pythia8_delphes.root"
                )
            else:
                root_path = legacy_root / "root" / f"{local_token}_pythia8_delphes.root"
            root_path.parent.mkdir(parents=True, exist_ok=True)
            root_path.write_bytes(f"synthetic-{local_token}".encode())
        else:
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
        + [
            {
                "sample_class": "background",
                "process_or_mode": "ttbar_inclusive",
                "run2_yield_coefficient_per_generator_weight": 5.0,
            }
        ]
    )
    hard_qcd = pd.DataFrame(
        [
            {
                "source_tag": "qcd_hardqcd_campaign_bin00_shard0010_seed10",
                "campaign": "qcd_hardqcd_campaign",
                "run2_yield_coefficient_per_generator_weight": 4.0,
            }
        ]
    )
    ttbar = pd.DataFrame(
        [
            {
                "member_id": token,
                "final_split": "validation",
                "generated_events_frozen": development_rows[index]["generated_events"],
                "root_sha256": f"{index:064x}",
            }
            for index, token in {
                5: "ttbar_100k_shard000",
                6: "ttbar_100k_shard004",
                7: "ttbar_100k_shard005",
                8: "ttbar_200k_shard008",
                9: "ttbar_200k_shard015",
            }.items()
        ]
    )
    remote_checksums = pd.DataFrame(
        [
            {
                "source_uid": "::".join(
                    [
                        row["group_id"],
                        row["sample_class"],
                        str(row["source_index"]),
                        row["member_id"],
                    ]
                ),
                "archive_locator": "root://cmseos.fnal.gov/" + row["source_locator"],
                "source_size_bytes": 1000 + row["source_index"],
                "checksum_kind": "adler32",
                "checksum": f"{row['source_index']:08x}",
                "independent_query_passes": 2,
                "independent_queries_match": True,
                "validation_event_payload_opened": False,
                "test_event_payload_opened": False,
                "status": "pass_metadata_only_remote_checksum_closure",
            }
            for row in development_rows
            if str(row["member_id"]).startswith("bundle:")
        ]
    )
    return development, root_map, ordinary, hard_qcd, ttbar, remote_checksums


def test_locator_helpers() -> None:
    assert remote_uri("/store/user/example.tar.gz") == (
        "root://cmseos.fnal.gov//store/user/example.tar.gz"
    )
    assert hard_qcd_source_tag(
        "/store/qcd_campaign_bin00_shard1_seed1_bundle.tar.gz"
    ) == "qcd_campaign_bin00_shard1_seed1"


def test_metadata_only_contract_closes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        legacy_root = Path(directory)
        development, root_map, ordinary, hard_qcd, ttbar, checksums = (
            synthetic_metadata(legacy_root)
        )
        access, coefficients, summary = build_metadata(
            development,
            root_map,
            ordinary,
            hard_qcd,
            ttbar,
            legacy_root,
            checksums,
        )
    assert len(access) == 121
    assert access["source_uid"].is_unique
    assert access["physical_evaluation_eligible"].sum() == 116
    assert access["auxiliary_qcd"].sum() == 5
    assert coefficients["run2_yield_coefficient_per_generator_weight"].isna().sum() == 5
    assert summary["validation_generated_events"] == 1_002_584
    assert summary["event_payload_files_opened"] == 0
    assert summary["remote_bundle_checksum_closure"] == "pass_116_of_116"
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
