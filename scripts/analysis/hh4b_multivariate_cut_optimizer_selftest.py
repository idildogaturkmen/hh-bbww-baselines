#!/usr/bin/env python3
"""Bounded synthetic self-test for the repository-native cut optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hh4b_multivariate_cut_optimizer import (
    Configuration,
    CutSpec,
    OptimizationSettings,
    optimize_configuration,
    selection_mask,
    weighted_metric,
)


def main() -> None:
    rng = np.random.default_rng(20260806)
    signal_x = rng.normal(0.0, 1.0, 600)
    background_x = rng.normal(2.0, 1.2, 2400)
    signal_y = rng.normal(2.0, 0.8, 600)
    background_y = rng.normal(0.0, 1.0, 2400)
    frame = pd.DataFrame(
        {
            "sample_class": ["signal"] * 600 + ["background"] * 2400,
            "x": np.concatenate([signal_x, background_x]),
            "y": np.concatenate([signal_y, background_y]),
            "w": np.ones(3000, dtype=float),
        }
    )
    configuration = Configuration(
        category_family_id="synthetic",
        category_id="synthetic",
        structure_id="synthetic_xy",
        cuts=(CutSpec("x", "<"), CutSpec("y", ">")),
    )
    settings = OptimizationSettings(
        target_signal_efficiency=0.60,
        quantile_count=15,
        beam_width=32,
        refinement_starts=8,
        max_coordinate_passes=4,
        minimum_inner_selected_signal_rows=25,
        minimum_inner_selected_background_rows=50,
        minimum_pooled_background_rows=50,
        minimum_pooled_background_neff=20.0,
    )
    first = optimize_configuration(frame, configuration, "w", settings)
    second = optimize_configuration(frame, configuration, "w", settings)
    if first.thresholds != second.thresholds:
        raise RuntimeError("synthetic thresholds are nondeterministic")
    if first.metric != second.metric:
        raise RuntimeError("synthetic metrics are nondeterministic")
    if not first.metric.feasible:
        raise RuntimeError("synthetic result did not reach target")
    mask = selection_mask(frame, configuration, first.thresholds)
    metric = weighted_metric(frame, mask, "w", 0.60)
    if metric != first.metric:
        raise RuntimeError("selected-mask recomputation mismatch")
    print("SYNTHETIC_SELFTEST=PASS")
    print("DETERMINISTIC_RERUN=PASS")
    print("SELECTED_MASK_RECOMPUTATION=PASS")
    print("RESULT=HH4B_MULTIVARIATE_CUT_OPTIMIZER_SELFTEST_PASS")


if __name__ == "__main__":
    main()
