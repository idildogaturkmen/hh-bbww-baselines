from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from plot_hh4b_cut_baseline_validation import (  # noqa: E402
    SCOPES,
    build_figure_rows,
)


def metric_row(selection_id: str, scope: str, fold: str, scale: float):
    return {
        "selection_id": selection_id,
        "scope": scope,
        "outer_fold": fold,
        "signal_physical_efficiency": 0.6 * scale,
        "background_physical_efficiency": 0.2 * scale,
        "background_rejection": 5.0 / scale,
        "signal_selected_signed_yield": 100.0 * scale,
        "background_selected_signed_yield": 1000.0 * scale,
        "signal_over_background": 0.1,
        "asimov_significance_stat_only": 3.0 * scale,
    }


def test_figure_data_has_train_oof_fixed_train_and_validation() -> None:
    train = pd.DataFrame(
        [
            metric_row(selection, scope, "pooled", scale)
            for selection, scale in [
                ("nested_outer_oof", 1.0),
                ("fixed_nominal_deployment_cut", 0.9),
            ]
            for scope in SCOPES
        ]
    )
    validation = pd.DataFrame(
        [metric_row("fixed_nominal_deployment_cut", scope, "validation", 0.8) for scope in SCOPES]
    )
    rows = build_figure_rows(train, validation)
    assert len(rows) == 63
    assert {row["series_id"] for row in rows} == {
        "nested_train_oof",
        "fixed_train_diagnostic",
        "fixed_validation",
    }
    assert {row["scope"] for row in rows} == set(SCOPES)
    assert {row["validation_payloads_opened"] for row in rows} == {1}
    assert {row["test_payloads_opened"] for row in rows} == {0}


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
