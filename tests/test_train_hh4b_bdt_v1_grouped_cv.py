#!/usr/bin/env python3
"""Synthetic-only tests for canonical HH4b BDT-v1 grouped CV."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "analysis"))

from train_hh4b_bdt_v1_grouped_cv import (  # noqa: E402
    PARAMETER_NAMES,
    aggregate_gain_importance,
    binary_metrics,
    check_oof_integrity,
    cut_baseline_metrics,
    generate_hyperparameter_space,
    load_yaml,
    rank_hyperparameter_trials,
    score_mass_diagnostic_rows,
    weighted_signal_threshold,
)

import numpy as np  # noqa: E402
from scripts.plotting.hh4b_cms_style import (  # noqa: E402
    add_delphes_header,
    apply_cms_style,
    save_png_pdf,
)


CONFIG = load_yaml(
    REPOSITORY_ROOT / "configs/baselines/hh4b_bdt_v1_grouped_cv.yaml"
)


class HyperparameterContractTests(unittest.TestCase):
    def test_deterministic_24_configuration_generation(self) -> None:
        first = generate_hyperparameter_space(CONFIG["hyperparameter_search"])
        second = generate_hyperparameter_space(CONFIG["hyperparameter_search"])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual(
            len(
                {
                    tuple(row[name] for name in PARAMETER_NAMES)
                    for row in first
                }
            ),
            24,
        )

    def test_baseline_configuration_inclusion(self) -> None:
        rows = generate_hyperparameter_space(CONFIG["hyperparameter_search"])
        baseline = rows[0]
        self.assertTrue(baseline["is_baseline"])
        for name, value in CONFIG["hyperparameter_search"]["baseline"].items():
            self.assertEqual(baseline[name], value)

    def test_identical_search_space_for_both_variants(self) -> None:
        mass_aware = generate_hyperparameter_space(
            CONFIG["hyperparameter_search"]
        )
        mass_blind = generate_hyperparameter_space(
            CONFIG["hyperparameter_search"]
        )
        self.assertEqual(mass_aware, mass_blind)

    def test_exact_model_selection_tie_break_order(self) -> None:
        def row(
            trial_id: int,
            mean: float,
            worst: float,
            std: float,
            complexity: int,
        ) -> dict[str, float | int]:
            return {
                "trial_id": trial_id,
                "mean_weighted_roc_auc": mean,
                "worst_fold_weighted_roc_auc": worst,
                "std_weighted_roc_auc": std,
                "complexity_proxy": complexity,
            }

        rows = [
            row(1, 0.80, 0.99, 0.01, 1),
            row(2, 0.90, 0.60, 0.01, 1),
            row(3, 0.90, 0.70, 0.20, 1),
            row(4, 0.90, 0.70, 0.10, 200),
            row(6, 0.90, 0.70, 0.10, 100),
            row(0, 0.90, 0.70, 0.10, 100),
        ]
        ranked = rank_hyperparameter_trials(rows)
        self.assertEqual(
            [entry["trial_id"] for entry in ranked],
            [0, 6, 4, 3, 2, 1],
        )
        self.assertTrue(ranked[0]["selected"])
        self.assertEqual(ranked[0]["selection_rank"], 1)


class MetricAndWorkingPointTests(unittest.TestCase):
    def test_weighted_and_raw_metric_calculations(self) -> None:
        target = np.asarray([0, 0, 1, 1])
        score = np.asarray([0.1, 0.2, 0.8, 0.9])
        weight = np.asarray([1.0, 3.0, 2.0, 4.0])
        metrics = binary_metrics(target, score, weight)
        self.assertEqual(metrics["weighted_roc_auc"], 1.0)
        self.assertEqual(metrics["raw_roc_auc"], 1.0)
        self.assertEqual(metrics["weighted_average_precision"], 1.0)
        self.assertEqual(metrics["raw_average_precision"], 1.0)

    def test_working_point_threshold_construction(self) -> None:
        threshold = weighted_signal_threshold(
            [0.9, 0.8, 0.7],
            [1.0, 1.0, 2.0],
            0.5,
        )
        self.assertEqual(threshold, 0.8)

    def test_cut_baseline_comparison(self) -> None:
        rhh = np.asarray([20.0, 40.0, 10.0, 50.0, 25.0, 60.0])
        labels = np.asarray([1, 1, 0, 0, 1, 0])
        weights = np.ones(6)
        modes = np.asarray(
            ["ggf_hh4b", "vbf_hh4b", "", "", "ggf_hh4b", ""],
            dtype=object,
        )
        families = np.asarray(
            ["", "", "family_a", "family_a", "", "family_b"],
            dtype=object,
        )
        result = cut_baseline_metrics(
            rhh,
            labels,
            weights,
            modes,
            families,
            threshold=34.0,
        )
        self.assertAlmostEqual(result["raw_signal_efficiency"], 2.0 / 3.0)
        self.assertAlmostEqual(result["raw_background_efficiency"], 1.0 / 3.0)
        self.assertEqual(
            result["raw_background_family_efficiencies"]["family_b"],
            0.0,
        )


class OofIntegrityTests(unittest.TestCase):
    def test_oof_completeness_detection(self) -> None:
        result = check_oof_integrity(
            [0, 1, 2, 3],
            [0, 1, 3],
            [set()],
            [set()],
        )
        self.assertEqual(result["missing_oof_rows"], 1)

    def test_oof_duplicate_detection(self) -> None:
        result = check_oof_integrity(
            [0, 1, 2],
            [0, 1, 1, 2],
            [set()],
            [set()],
        )
        self.assertEqual(result["duplicate_oof_rows"], 1)

    def test_member_leakage_detection(self) -> None:
        result = check_oof_integrity(
            [0, 1],
            [0, 1],
            [{1, 2}, {3}],
            [{2, 4}, {5}],
        )
        self.assertEqual(result["member_leakage_rows"], 1)


class DiagnosticTests(unittest.TestCase):
    def test_feature_importance_aggregation(self) -> None:
        rows = aggregate_gain_importance(
            [[2.0, 1.0, 0.0], [1.0, 1.0, 2.0]],
            ["mhh", "hh_pt", "mbb1"],
            "synthetic",
        )
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(
            sum(row["mean_normalized_gain_importance"] for row in rows),
            1.0,
        )
        self.assertEqual(rows[2]["minimum_normalized_gain_importance"], 0.0)

    def test_score_quintile_diagnostics(self) -> None:
        score = np.linspace(0.001, 0.999, 100)
        rhh = np.linspace(1.0, 200.0, 100)
        weights = np.linspace(0.5, 1.5, 100)
        rows = score_mass_diagnostic_rows(
            score,
            rhh,
            weights,
            "synthetic",
        )
        self.assertEqual(len(rows), 6)
        quintiles = [
            row
            for row in rows
            if row["row_type"] == "weighted_background_score_quintile"
        ]
        self.assertEqual(len(quintiles), 5)
        self.assertAlmostEqual(
            sum(row["weighted_fraction"] for row in quintiles),
            1.0,
        )
        self.assertTrue(
            all(not row["used_in_model_selection"] for row in rows)
        )


class PlotSmokeTests(unittest.TestCase):
    def test_png_pdf_writing_and_temporary_cleanup(self) -> None:
        import matplotlib.pyplot as plt

        metadata = apply_cms_style()
        self.assertTrue(metadata["cms_inspired_style_applied"])
        with tempfile.TemporaryDirectory(prefix="hh4b_bdt_cv_plot_") as tmp:
            directory = Path(tmp)
            fig, ax = plt.subplots(figsize=(4.0, 3.0))
            ax.plot([0.0, 1.0], [1.0, 0.0], color="#0072B2")
            ax.set_xlabel("Signal efficiency")
            ax.set_ylabel("Background rejection")
            add_delphes_header(
                ax,
                "Train-only grouped cross-validation",
            )
            png, pdf = save_png_pdf(
                fig,
                directory / "grouped_cv_smoke",
                dpi=300,
            )
            self.assertGreater(png.stat().st_size, 0)
            self.assertGreater(pdf.stat().st_size, 0)
            retained = directory
        self.assertFalse(retained.exists())


if __name__ == "__main__":
    unittest.main()
