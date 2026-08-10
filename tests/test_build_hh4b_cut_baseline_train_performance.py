from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from build_hh4b_cut_baseline_train_performance import (  # noqa: E402
    HISTORICAL_ROLE,
    apply_cuts,
    asimov_significance,
    combine_summaries,
    effective_events,
    performance_row,
    relative_mc_uncertainty,
    summarize_frame,
)


def test_asimov_formula_matches_frozen_historical_benchmark() -> None:
    assert np.isclose(
        asimov_significance(417.46273293569186, 20240622849.775902),
        0.0029343085318647984,
        rtol=2.0e-15,
    )


def test_effective_count_and_relative_uncertainty() -> None:
    assert effective_events(10.0, 4.0) == 25.0
    assert relative_mc_uncertainty(10.0, 4.0) == 0.2


def test_strict_multivariate_cut_boundaries() -> None:
    frame = pd.DataFrame({
        "r_hh_125_125": [33.0, 34.0, 33.0],
        "mhh": [200.0, 200.0, 164.0],
    })
    mask = apply_cuts(
        frame,
        [
            {"variable": "r_hh_125_125", "operator": "<"},
            {"variable": "mhh", "operator": ">"},
        ],
        {"r_hh_125_125": 34.0, "mhh": 164.0},
    )
    assert mask.tolist() == [True, False, False]


def test_summary_combination_and_derived_metrics() -> None:
    frame = pd.DataFrame({
        "sample_class": ["signal", "signal", "background", "background"],
        "resolved_selection_contribution_weight": [2.0, 1.0, 10.0, 5.0],
    })
    first = summarize_frame(frame, np.array([True, False, True, False]))
    combined = combine_summaries([first, first])
    row = performance_row(
        "historical_rhh125125_lt34",
        HISTORICAL_ROLE,
        "combined",
        "pooled",
        combined,
    )
    assert row["signal_physical_efficiency"] == 2.0 / 3.0
    assert row["background_physical_efficiency"] == 2.0 / 3.0
    assert row["background_rejection"] == 1.5
    assert row["signal_selected_effective_events"] == 2.0
    assert row["background_selected_effective_events"] == 2.0
    assert row["validation_payloads_opened"] == 0
    assert row["test_payloads_opened"] == 0


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
