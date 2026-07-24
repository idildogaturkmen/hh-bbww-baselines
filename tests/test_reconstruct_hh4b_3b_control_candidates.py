from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot


REPO = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    REPO
    / "scripts"
    / "delphes"
    / "reconstruct_hh4b_3b_control_candidates.py"
)
FROZEN_BUILDER_PATH = (
    REPO
    / "scripts"
    / "delphes"
    / "reconstruct_hh4b_candidates_v2.py"
)


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "hh4b_3b_builder",
        BUILDER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_builder()


def event_inputs(
    *,
    pts,
    btags,
    etas=None,
    phis=None,
    masses=None,
    flavors=None,
):
    size = len(pts)
    return {
        "pts": pts,
        "etas": etas if etas is not None else [0.0] * size,
        "phis": phis
        if phis is not None
        else [0.55 * index for index in range(size)],
        "masses": masses if masses is not None else [12.0] * size,
        "btags": btags,
        "flavors": flavors if flavors is not None else [5] * size,
    }


def reconstruct_threeb(**event):
    return builder.reconstruct_event(
        sample="synthetic",
        event=7,
        mode=builder.MODE_THREEB,
        **event,
    )


def make_pair(mass: float, pt: float):
    return {"mass": mass, "pt": pt}


def write_synthetic_root(path: Path, events: list[dict]) -> None:
    data = {
        "Jet.PT": ak.values_astype(
            ak.Array([event["pts"] for event in events]),
            np.float32,
        ),
        "Jet.Eta": ak.values_astype(
            ak.Array([event["etas"] for event in events]),
            np.float32,
        ),
        "Jet.Phi": ak.values_astype(
            ak.Array([event["phis"] for event in events]),
            np.float32,
        ),
        "Jet.Mass": ak.values_astype(
            ak.Array([event["masses"] for event in events]),
            np.float32,
        ),
        "Jet.BTag": ak.values_astype(
            ak.Array([event["btags"] for event in events]),
            np.uint32,
        ),
        "Jet.Flavor": ak.values_astype(
            ak.Array([event["flavors"] for event in events]),
            np.uint32,
        ),
    }
    with uproot.recreate(path) as root_file:
        root_file.mktree("Delphes", data)


def run_builder(
    script: Path,
    input_path: Path,
    output_path: Path,
    *,
    mode: str | None = None,
) -> None:
    command = [
        sys.executable,
        str(script),
    ]
    if mode is not None:
        command.extend(["--mode", mode])
    command.extend(
        [
            "--input",
            str(input_path),
            "--out",
            str(output_path),
            "--sample",
            "synthetic_parity",
        ]
    )
    subprocess.run(
        command,
        cwd=REPO,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def assert_parity(
    frozen_frame: pd.DataFrame,
    parity_frame: pd.DataFrame,
) -> None:
    assert list(frozen_frame.columns) == list(parity_frame.columns)
    assert len(frozen_frame) == len(parity_frame)
    for column in frozen_frame.columns:
        frozen = frozen_frame[column]
        parity = parity_frame[column]
        if pd.api.types.is_float_dtype(frozen.dtype):
            np.testing.assert_allclose(
                frozen.to_numpy(),
                parity.to_numpy(),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )
        else:
            pd.testing.assert_series_equal(
                frozen,
                parity,
                check_names=True,
                check_dtype=True,
            )


def test_output_schema_contract_counts_and_frozen_prefix():
    assert len(builder.FROZEN_CANDIDATE_COLUMNS) == 72
    assert len(builder.THREEB_CANDIDATE_COLUMNS) == 83
    assert (
        builder.THREEB_CANDIDATE_COLUMNS[:72]
        == builder.FROZEN_CANDIDATE_COLUMNS
    )
    assert (
        builder.THREEB_CANDIDATE_COLUMNS[72:]
        == builder.THREEB_PROVENANCE_COLUMNS
    )


def test_strict_jet_pt_boundary():
    selected = builder.select_jets(
        [30.0, 30.0001],
        [0.0, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],
        [1.0, 1.0],
        [5, 5],
        jet_pt_min=30.0,
        jet_eta_max=2.5,
    )
    assert [jet["raw_index"] for jet in selected] == [1]


def test_strict_absolute_eta_boundary():
    selected = builder.select_jets(
        [50.0, 50.0, 50.0],
        [2.5, -2.5, 2.4999],
        [0.0, 0.1, 0.2],
        [10.0, 10.0, 10.0],
        [1.0, 1.0, 1.0],
        [5, 5, 5],
        jet_pt_min=30.0,
        jet_eta_max=2.5,
    )
    assert [jet["raw_index"] for jet in selected] == [2]


def test_exactly_three_tag_event_is_accepted():
    row, diagnostics = reconstruct_threeb(
        **event_inputs(
            pts=[100.0, 90.0, 80.0, 70.0],
            btags=[1.0, 1.0, 1.0, 0.0],
        )
    )
    assert row is not None
    assert diagnostics["n_selected_bjets"] == 3


def test_two_tag_event_is_rejected():
    row, diagnostics = reconstruct_threeb(
        **event_inputs(
            pts=[100.0, 90.0, 80.0, 70.0],
            btags=[1.0, 1.0, 0.0, 0.0],
        )
    )
    assert row is None
    assert diagnostics["n_selected_bjets"] == 2


def test_four_tag_event_is_rejected_in_threeb_mode():
    row, diagnostics = reconstruct_threeb(
        **event_inputs(
            pts=[100.0, 90.0, 80.0, 70.0],
            btags=[1.0, 1.0, 1.0, 1.0],
        )
    )
    assert row is None
    assert diagnostics["n_selected_bjets"] == 4


def test_fewer_than_four_selected_jets_is_rejected():
    row, diagnostics = reconstruct_threeb(
        **event_inputs(
            pts=[100.0, 90.0, 80.0, 30.0],
            btags=[1.0, 1.0, 1.0, 0.0],
        )
    )
    assert row is None
    assert diagnostics["n_selected_jets"] == 3


def test_highest_pt_selected_untagged_jet_is_promoted():
    row, diagnostics = reconstruct_threeb(
        **event_inputs(
            pts=[120.0, 110.0, 100.0, 95.0, 75.0],
            btags=[1.0, 1.0, 1.0, 0.0, 0.0],
        )
    )
    assert row is not None
    assert diagnostics["promoted_jet"]["raw_index"] == 3
    assert row["promoted_jet_pt"] == 95.0
    assert row["n_selected_untagged_jets"] == 2


def test_promoted_jet_is_fixed_before_pairing_even_if_lower_pt_is_better():
    event = event_inputs(
        pts=[
            137.53165021282805,
            65.09154802769801,
            120.55686155919507,
            192.6559429186077,
            96.90796249599994,
        ],
        etas=[
            -0.8699699715293331,
            0.11989005186037094,
            1.147368320426101,
            0.41631954317938913,
            -0.5724014101250243,
        ],
        phis=[
            -2.6258949960338165,
            2.7617935752335705,
            -0.16266160145177455,
            3.1169284278723617,
            -2.25070715690555,
        ],
        masses=[
            21.24408504496128,
            21.267562434981773,
            23.399500449952804,
            5.356686328766056,
            12.443856831148697,
        ],
        btags=[1.0, 1.0, 1.0, 0.0, 0.0],
    )
    selected = builder.select_jets(
        **event,
        jet_pt_min=30.0,
        jet_eta_max=2.5,
    )
    bjets = builder.assign_bjet_ranks(selected, btag_min=0.0)
    untagged = [jet for jet in selected if jet["btag"] <= 0.0]
    scores = []
    for jet in untagged:
        jet["bjet_rank"] = -1
        best, _ = builder.evaluate_pairings([*bjets, jet], 125.0)
        scores.append((jet["pt"], best["sort_key"][0]))
    assert scores[1][1] < scores[0][1]

    row, diagnostics = reconstruct_threeb(**event)
    assert row is not None
    assert diagnostics["promoted_jet"]["pt"] == scores[0][0]
    assert row["promoted_jet_raw_index"] == 3


def test_promoted_and_tagged_bjet_ranks_are_explicit():
    row, _ = reconstruct_threeb(
        **event_inputs(
            pts=[140.0, 130.0, 120.0, 150.0],
            btags=[1.0, 1.0, 1.0, 0.0],
        )
    )
    assert row is not None
    ranks = [row[f"j{index}_bjet_rank"] for index in range(1, 5)]
    assert sorted(ranks) == [-1, 0, 1, 2]
    promoted_slot = ranks.index(-1) + 1
    assert row[f"j{promoted_slot}_btag"] <= 0.0


def test_candidate_jets_are_output_in_descending_pt():
    row, _ = reconstruct_threeb(
        **event_inputs(
            pts=[95.0, 150.0, 105.0, 130.0],
            btags=[1.0, 1.0, 1.0, 0.0],
        )
    )
    assert row is not None
    output_pts = [row[f"j{index}_pt"] for index in range(1, 5)]
    assert output_pts == sorted(output_pts, reverse=True)


def test_all_three_fixed_jet_pairings_are_considered():
    row, diagnostics = reconstruct_threeb(
        **event_inputs(
            pts=[110.0, 100.0, 90.0, 80.0],
            btags=[1.0, 1.0, 1.0, 0.0],
        )
    )
    assert row is not None
    assert diagnostics["combinations_evaluated"] == 1
    assert diagnostics["pairings_evaluated"] == 3


def test_pairing_primary_objective_matches_frozen_definition():
    first = make_pair(120.0, 50.0)
    second = make_pair(132.0, 60.0)
    key = builder.pairing_sort_key(first, second, 125.0)
    assert key[0] == abs(120.0 - 125.0) + abs(132.0 - 125.0)


def test_pairing_first_tiebreaker_minimizes_mass_difference():
    balanced = builder.pairing_sort_key(
        make_pair(120.0, 50.0),
        make_pair(120.0, 50.0),
        125.0,
    )
    unbalanced = builder.pairing_sort_key(
        make_pair(115.0, 50.0),
        make_pair(125.0, 50.0),
        125.0,
    )
    assert balanced[0] == unbalanced[0]
    assert balanced < unbalanced


def test_pairing_second_tiebreaker_maximizes_higgs_pt_sum():
    lower_pt = builder.pairing_sort_key(
        make_pair(120.0, 40.0),
        make_pair(130.0, 50.0),
        125.0,
    )
    higher_pt = builder.pairing_sort_key(
        make_pair(120.0, 70.0),
        make_pair(130.0, 80.0),
        125.0,
    )
    assert lower_pt[:2] == higher_pt[:2]
    assert higher_pt < lower_pt


def test_higgs_candidates_are_ordered_by_descending_pt():
    low = {"pt": 70.0, "mass": 125.0}
    high = {"pt": 90.0, "mass": 120.0}
    h1, h2 = builder.choose_h1_h2(
        low,
        high,
        125.0,
        "pt",
    )
    assert h1 is high
    assert h2 is low


def test_r_hh_formulas_match_frozen_implementation():
    row, _ = reconstruct_threeb(
        **event_inputs(
            pts=[115.0, 105.0, 95.0, 85.0],
            btags=[1.0, 1.0, 1.0, 0.0],
        )
    )
    assert row is not None
    expected_125_125 = math.sqrt(
        (row["mbb1"] - 125.0) ** 2
        + (row["mbb2"] - 125.0) ** 2
    )
    expected_125_120 = math.sqrt(
        (row["mbb1"] - 125.0) ** 2
        + (row["mbb2"] - 120.0) ** 2
    )
    assert row["r_hh"] == pytest.approx(expected_125_125)
    assert row["r_hh_125_125"] == pytest.approx(expected_125_125)
    assert row["r_hh_125_120"] == pytest.approx(expected_125_120)


def test_threeb_constants_and_promoted_provenance_are_complete():
    row, _ = reconstruct_threeb(
        **event_inputs(
            pts=[120.0, 110.0, 100.0, 90.0],
            btags=[1.0, 1.0, 1.0, 0.0],
            flavors=[5, 5, 5, 1],
        )
    )
    assert row is not None
    assert row["candidate_category"] == builder.CANDIDATE_CATEGORY
    assert row["promoted_jet_rule"] == builder.PROMOTED_JET_RULE
    assert row["promoted_jet_flavor"] == 1
    assert list(row) == builder.THREEB_CANDIDATE_COLUMNS


def test_threeb_typed_zero_row_output_has_deterministic_schema(
    tmp_path: Path,
):
    output = tmp_path / "threeb_empty.parquet"
    assert list(builder.typed_empty_output(builder.MODE_THREEB)) == (
        builder.THREEB_CANDIDATE_COLUMNS
    )
    builder.write_parquet_output(
        output,
        [],
        mode=builder.MODE_THREEB,
    )
    frame = pd.read_parquet(output)
    assert len(frame) == 0
    assert list(frame.columns) == builder.THREEB_CANDIDATE_COLUMNS


def test_fourb_typed_zero_row_output_is_exact_frozen_schema(
    tmp_path: Path,
):
    output = tmp_path / "fourb_empty.parquet"
    builder.write_parquet_output(
        output,
        [],
        mode=builder.MODE_FOURB,
    )
    frame = pd.read_parquet(output)
    assert len(frame) == 0
    assert list(frame.columns) == builder.FROZEN_CANDIDATE_COLUMNS
    assert len(frame.columns) == 72


def test_fourb_pairing_pool_is_limited_to_eight_highest_pt_bjets():
    event = event_inputs(
        pts=[200.0 - 10.0 * index for index in range(10)],
        btags=[1.0] * 10,
    )
    _, diagnostics = builder.reconstruct_event(
        sample="synthetic",
        event=0,
        mode=builder.MODE_FOURB,
        **event,
    )
    assert diagnostics["combinations_evaluated"] == math.comb(8, 4)
    assert diagnostics["pairings_evaluated"] == 3 * math.comb(8, 4)


def test_mode_must_be_explicit_on_command_line(tmp_path: Path):
    completed = subprocess.run(
        [sys.executable, str(BUILDER_PATH), "--help"],
        cwd=REPO,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert "--mode" in completed.stdout
    assert "threeb-control" in completed.stdout
    assert "fourb-parity" in completed.stdout


def test_fourb_parity_matches_frozen_builder_on_synthetic_root(
    tmp_path: Path,
):
    symmetric_phis = [
        0.0,
        0.5 * math.pi,
        math.pi,
        -0.5 * math.pi,
    ]
    events = [
        event_inputs(
            pts=[100.0, 90.0, 80.0, 70.0],
            btags=[1.0, 1.0, 1.0, 0.0],
        ),
        event_inputs(
            pts=[120.0, 110.0, 100.0, 90.0],
            btags=[1.0, 1.0, 1.0, 1.0],
        ),
        event_inputs(
            pts=[150.0, 140.0, 130.0, 120.0, 110.0, 100.0],
            btags=[1.0] * 6,
            phis=[0.0, 0.4, 1.1, 1.8, 2.6, -2.2],
        ),
        event_inputs(
            pts=[220.0 - 12.0 * index for index in range(10)],
            btags=[1.0] * 10,
            phis=[-2.8 + 0.55 * index for index in range(10)],
        ),
        event_inputs(
            pts=[210.0, 175.0, 160.0, 145.0, 120.0, 95.0, 75.0],
            btags=[1.0] * 7,
            etas=[0.2, -0.4, 0.7, -1.0, 1.1, -0.8, 0.3],
            phis=[0.0, 0.9, 1.7, 2.8, -2.4, -1.2, 0.4],
            masses=[15.0, 18.0, 12.0, 20.0, 10.0, 14.0, 11.0],
        ),
        event_inputs(
            pts=[100.0, 100.0, 100.0, 100.0],
            btags=[1.0] * 4,
            phis=symmetric_phis,
            masses=[0.0] * 4,
        ),
        event_inputs(
            pts=[31.0, 30.0, 125.0, 115.0, 105.0, 95.0, 85.0],
            btags=[1.0] * 7,
            etas=[0.0, 0.0, 2.5, 0.2, -0.3, 0.4, -0.5],
        ),
    ]
    main_root = tmp_path / "synthetic_parity.root"
    write_synthetic_root(main_root, events)
    frozen_output = tmp_path / "frozen.parquet"
    parity_output = tmp_path / "parity.parquet"
    run_builder(
        FROZEN_BUILDER_PATH,
        main_root,
        frozen_output,
    )
    run_builder(
        BUILDER_PATH,
        main_root,
        parity_output,
        mode=builder.MODE_FOURB,
    )
    frozen_frame = pd.read_parquet(frozen_output)
    parity_frame = pd.read_parquet(parity_output)
    assert_parity(frozen_frame, parity_frame)
    assert list(parity_frame.columns) == builder.FROZEN_CANDIDATE_COLUMNS
    assert parity_frame["event"].tolist() == [1, 2, 3, 4, 5, 6]

    zero_events = [
        event_inputs(
            pts=[100.0, 90.0, 80.0, 70.0],
            btags=[1.0, 1.0, 1.0, 0.0],
        ),
        event_inputs(
            pts=[100.0, 90.0, 80.0, 70.0],
            btags=[1.0, 1.0, 0.0, 0.0],
        ),
    ]
    zero_root = tmp_path / "synthetic_zero.root"
    write_synthetic_root(zero_root, zero_events)
    frozen_zero = tmp_path / "frozen_zero.parquet"
    parity_zero = tmp_path / "parity_zero.parquet"
    run_builder(
        FROZEN_BUILDER_PATH,
        zero_root,
        frozen_zero,
    )
    run_builder(
        BUILDER_PATH,
        zero_root,
        parity_zero,
        mode=builder.MODE_FOURB,
    )
    frozen_zero_frame = pd.read_parquet(frozen_zero)
    parity_zero_frame = pd.read_parquet(parity_zero)
    assert_parity(frozen_zero_frame, parity_zero_frame)
    assert len(parity_zero_frame) == 0
    assert list(parity_zero_frame.columns) == builder.FROZEN_CANDIDATE_COLUMNS
