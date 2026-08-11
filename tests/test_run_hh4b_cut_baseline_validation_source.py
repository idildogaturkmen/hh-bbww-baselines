from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys
import tempfile

import pyarrow as pa
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from run_hh4b_cut_baseline_validation_source import (  # noqa: E402
    build_source_products,
    durable_exclusive_json,
    empty_summary,
    finalize_summary,
    fixed_nominal_pass,
    install_preexisting_attempt_marker,
    update_summary,
)


class ReconstructionStub:
    @staticmethod
    def reconstruct_generalized(jets):
        tags = sum(float(jet["btag"]) > 0.0 for jet in jets)
        return {
            "candidate_tagged_jet_count": tags,
            "r_hh_125_125": 35.0 if tags == 3 else 33.0,
            "ht_candidate_jets": 200.0,
            "mhh": 200.0,
            "h2_pt": 100.0,
            "h_delta_eta": -1.0,
            "drbb1": 0.8,
            "drbb2": 1.2,
        }


def test_strict_nominal_boundaries() -> None:
    assert fixed_nominal_pass(
        "exact3tag", {"r_hh_125_125": 36.0, "ht_candidate_jets": 177.0}
    )
    assert not fixed_nominal_pass(
        "exact3tag",
        {"r_hh_125_125": 36.40814019639858, "ht_candidate_jets": 177.0},
    )
    assert not fixed_nominal_pass(
        "ge4tag",
        {
            "r_hh_125_125": 33.0,
            "mhh": 164.73708096689654,
            "abs_h_delta_eta": 1.0,
        },
    )


def test_signed_source_summary() -> None:
    summary = empty_summary()
    update_summary(summary, 2.0, True)
    update_summary(summary, -1.0, False)
    result = finalize_summary(summary)
    assert result["total_rows"] == 2
    assert result["selected_rows"] == 1
    assert result["negative_weight_rows"] == 1
    assert result["total_signed_yield"] == 1.0
    assert result["selected_signed_yield"] == 2.0
    assert result["total_sumw2"] == 5.0
    assert result["selected_sumw2"] == 4.0


def test_durable_attempt_marker_is_exclusive() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "DO_NOT_RERUN.json"
        durable_exclusive_json(path, {"status": "attempt"})
        assert path.is_file()
        try:
            durable_exclusive_json(path, {"status": "second_attempt"})
        except FileExistsError:
            pass
        else:
            raise AssertionError("exclusive validation marker was overwritten")


def test_preexisting_durable_attempt_marker_is_validated_and_copied() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "remote_marker.json"
        destination = root / "source_0007_VALIDATION_OPEN_DO_NOT_RERUN.json"
        authorization_sha = hashlib.sha256(b"authorization").hexdigest()
        uri = "root://cmseos.fnal.gov//store/user/example/marker.json"
        payload = {
            "schema_version": 1,
            "status": "validation_source_open_attempt_durable_do_not_rerun",
            "repository_head": "1" * 40,
            "authorization_repository_head": "0" * 40,
            "authorization_sha256": authorization_sha,
            "production_row_index": 7,
            "source_uid": "source",
            "durable_marker_uri": uri,
            "source_payload_access_may_begin": True,
            "rerun_forbidden_even_if_downstream_bookkeeping_fails": True,
            "test_payloads_opened": 0,
        }
        source.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        observed = install_preexisting_attempt_marker(
            source,
            destination,
            source_uid="source",
            row_index=7,
            authorization_sha256=authorization_sha,
            durable_marker_uri=uri,
            execution_head="1" * 40,
            authorization_head="0" * 40,
        )
        assert observed == payload
        assert destination.read_bytes() == source.read_bytes()

        second = root / "second.json"
        try:
            install_preexisting_attempt_marker(
                source,
                second,
                source_uid="changed",
                row_index=7,
                authorization_sha256=authorization_sha,
                durable_marker_uri=uri,
                execution_head="1" * 40,
                authorization_head="0" * 40,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("mismatched durable attempt marker passed")


def test_source_products_use_disjoint_categories_and_fixed_cut() -> None:
    masks = [[True] * 4, [True] * 4, [True] * 4]
    table = pa.table(
        {
            "source_uid": ["source"] * 3,
            "source_entry": [0, 1, 2],
            "event_uid": ["event0", "event1", "event2"],
            "raw_event_weight_available": [False, True, True],
            "raw_event_weight": [None, 2.0, -1.0],
            "broad_event_eligible": [False, True, True],
            "n_selected_jets": [4, 4, 4],
            "jet_pt": [[100.0, 90.0, 80.0, 70.0]] * 3,
            "jet_eta": [[0.1, -0.1, 0.2, -0.2]] * 3,
            "jet_phi": [[0.0, 1.0, 2.0, 3.0]] * 3,
            "jet_mass": [[10.0, 10.0, 10.0, 10.0]] * 3,
            "jet_btag": [
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            "jet_mask": masks,
        }
    )
    source = pd.Series(
        {
            "source_uid": "source",
            "production_row_index": 7,
            "group_id": "group",
            "sample_class": "background",
            "process_or_mode": "process",
            "generated_events": 3,
        }
    )
    summary, distributions = build_source_products(
        table, source, 10.0, ReconstructionStub
    )
    assert summary["broad_event_rows"] == 2
    assert summary["category_rows"] == 2
    assert summary["categories"]["exact3tag"]["total_rows"] == 1
    assert summary["categories"]["exact3tag"]["selected_signed_yield"] == 20.0
    assert summary["categories"]["ge4tag"]["total_rows"] == 1
    assert summary["categories"]["ge4tag"]["selected_signed_yield"] == -10.0
    assert len(distributions) == 1360
    assert {row["validation_payloads_opened"] for row in distributions} == {1}
    assert {row["test_payloads_opened"] for row in distributions} == {0}


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
