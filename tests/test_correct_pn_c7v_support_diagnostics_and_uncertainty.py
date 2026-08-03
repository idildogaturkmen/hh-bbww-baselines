from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from correct_pn_c7v_support_diagnostics_and_uncertainty import (  # noqa: E402
    _plot_supported_segments,
    add_uncertainty_tables,
    format_asymmetric,
    format_paired_difference,
    scan_with_bootstrap,
)
from pn_c7_bootstrap_common import (  # noqa: E402
    BOOTSTRAP_METRICS,
    evaluate_frozen_baselines,
    paired_model_differences,
)
from pn_c7_ml_common import source_member_bootstrap_draws  # noqa: E402
from publish_pn_c7v_results_and_figures import load_inputs  # noqa: E402


def test_all_models_use_identical_common_replica_ids() -> None:
    inputs = load_inputs()
    draws, _ = source_member_bootstrap_draws(inputs["members"], replicates=4, seed=20260802)
    nominal, replicas = evaluate_frozen_baselines(inputs, draws)
    expected = {0, 1, 2, 3}
    assert set(nominal.model) == {"cut", "bdt", "dense_dnn", "lbn_dnn"}
    for _, group in replicas.groupby("model"):
        assert set(group.replica_id) == expected


def test_every_pairwise_comparison_uses_aligned_draws_and_asymmetric_intervals() -> None:
    models = ["cut", "bdt", "dense_dnn", "lbn_dnn"]
    nominal_rows = []
    replica_rows = []
    for model_index, model in enumerate(models):
        nominal_rows.append({"model": model, **{metric: float(model_index) for metric in BOOTSTRAP_METRICS}})
        for replica in range(5):
            replica_rows.append({
                "model": model,
                "replica_id": replica,
                **{metric: float(model_index + replica * replica) for metric in BOOTSTRAP_METRICS},
            })
    summary, replicas = paired_model_differences(pd.DataFrame(nominal_rows), pd.DataFrame(replica_rows))
    assert summary.comparison.nunique() == 6
    assert len(summary) == 6 * len(BOOTSTRAP_METRICS)
    assert replicas.groupby(["comparison", "metric"]).replica_id.nunique().eq(5).all()
    assert (summary.bootstrap_p16_difference <= summary.bootstrap_median_difference).all()
    assert (summary.bootstrap_median_difference <= summary.bootstrap_p84_difference).all()


def test_support_scan_keeps_invalid_replicas_undefined_and_records_exact_reasons() -> None:
    inputs = load_inputs()
    draws, _ = source_member_bootstrap_draws(inputs["members"], replicates=6, seed=20260802)
    scan = scan_with_bootstrap(inputs, draws, points=5)
    required = {
        "selected_transferred_qcd_rows", "selected_transferred_qcd_sources", "selected_transferred_qcd_neff",
        "selected_direct_qcd_closure_rows", "selected_direct_qcd_closure_sources", "selected_direct_qcd_closure_neff",
        "support_pass", "support_failure_reason", "systematic_ZA_valid_replicas",
    }
    assert required.issubset(scan.columns)
    unsupported = scan[~scan.support_pass]
    assert not unsupported.empty
    assert unsupported.support_failure_reason.str.len().gt(0).all()
    assert not (unsupported.systematic_ZA_bootstrap_median.fillna(np.inf) == 0.0).any()
    supported = scan[scan.support_pass]
    assert (supported.raw_systematic_aware_asimov_ZA > 0.0).all()


def test_unsupported_ranges_are_plotted_as_separate_segments() -> None:
    class Axis:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray]] = []

        def plot(self, x: np.ndarray, y: np.ndarray, **_: object) -> None:
            self.calls.append((x.copy(), y.copy()))

    axis = Axis()
    x = np.arange(6, dtype=float)
    _plot_supported_segments(axis, x, x + 1, np.array([True, True, False, True, True, False]), color="black")
    assert len(axis.calls) == 2
    assert axis.calls[0][0].tolist() == [0.0, 1.0]
    assert axis.calls[1][0].tolist() == [3.0, 4.0]


def test_primary_latex_uncertainty_is_asymmetric() -> None:
    row = pd.Series({"bootstrap_median": 0.06370, "bootstrap_p16": 0.05813, "bootstrap_p84": 0.07028})
    rendered = format_asymmetric(row)
    assert rendered == r"$0.06370^{+0.00658}_{-0.00557}$"
    assert "+/-" not in rendered


def test_paired_latex_uses_mathematical_scientific_notation() -> None:
    rendered = format_paired_difference(3.06e-5, 1.85e-5, 4.56e-5, scientific=True, digits=2)
    assert rendered == r"$(3.06^{+1.50}_{-1.21})\times10^{-5}$"
    assert "e-" not in rendered


def test_uncertainty_tables_reproduce_machine_summary_values(tmp_path: Path) -> None:
    metrics = {
        "weighted_auc": (0.75, 0.74, 0.73, 0.76),
        "nominal_asimov_ZA": (0.06, 0.061, 0.055, 0.069),
        "systematic_aware_asimov_ZA": (3.0e-5, 3.1e-5, 2.2e-5, 4.4e-5),
        "signal_over_background": (8.0e-5, 8.2e-5, 7.0e-5, 9.8e-5),
        "background_effective_events": (30.0, 31.0, 25.0, 38.0),
        "transferred_qcd_fraction": (0.85, 0.86, 0.82, 0.89),
    }
    rows = []
    for model in ("cut", "bdt", "dense_dnn", "lbn_dnn"):
        for metric, (nominal, median, p16, p84) in metrics.items():
            rows.append({"model": model, "metric": metric, "nominal_full_oof": nominal,
                         "bootstrap_median": median, "bootstrap_p16": p16, "bootstrap_p84": p84})
    paired = pd.DataFrame([{
        "model": "bdt", "reference_model": "cut", "metric": "signal_over_background",
        "bootstrap_median_difference": 3.06e-5, "bootstrap_p16_difference": 1.85e-5,
        "bootstrap_p84_difference": 4.56e-5, "fraction_valid_difference_greater_than_zero": 0.997,
        "interval_includes_zero_68": False,
    }])
    earlier = pd.DataFrame([{
        "baseline": "bdt", "metric": "signal_over_background", "earlier_snapshot": 4.0e-5,
    }])
    (tmp_path / "tables").mkdir()
    add_uncertainty_tables(tmp_path, pd.DataFrame(rows), paired, earlier)
    unified = (tmp_path / "tables" / "tab05_baseline_metrics.tex").read_text()
    historical = (tmp_path / "tables" / "tab06_earlier_current.tex").read_text()
    differences = (tmp_path / "tables" / "tab11_paired_improvements.tex").read_text()
    assert r"$0.06000$" in unified
    assert r"$0.06100^{+0.00800}_{-0.00600}$" in unified
    assert r"$(3.06^{+1.50}_{-1.21})\times10^{-5}$" in differences
    assert re.search(r"\d(?:\.\d+)?e[+-]\d+", differences) is None
    assert r"$S/B$" in differences
    assert "no paired improvement is claimed" in historical
    assert r"$8.00\times10^{-5}$; $(8.20^{+1.60}_{-1.20})\times10^{-5}$" in historical
    assert re.search(r"\d(?:\.\d+)?e[+-]\d+", historical) is None
