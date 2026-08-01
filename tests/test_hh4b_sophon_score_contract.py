#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO / "scripts" / "analysis"

sys.path.insert(
    0,
    str(ANALYSIS_DIR),
)

import hh4b_sophon_score_contract as score_contract


def score_matrix(
    rows,
):
    array = np.zeros(
        (
            len(rows),
            138,
        ),
        dtype=np.float64,
    )

    for row_index, mapping in enumerate(
        rows
    ):
        for class_index, probability in mapping.items():
            array[
                row_index,
                class_index,
            ] = probability

    return array


class SophonScoreContractTests(
    unittest.TestCase
):
    def test_validate_accepts_normalized_scores(
        self,
    ):
        scores = score_matrix(
            [
                {
                    0: 0.7,
                    136: 0.2,
                    137: 0.1,
                },
                {
                    135: 0.5,
                    136: 0.25,
                    137: 0.25,
                },
            ]
        )

        validated = score_contract.validate_score_matrix(
            scores
        )

        np.testing.assert_allclose(
            validated,
            scores,
        )

    def test_validate_rejects_bad_shape_nonfinite_and_normalization(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "138 classes",
        ):
            score_contract.validate_score_matrix(
                np.ones(
                    (2, 137)
                )
            )

        nonfinite = score_matrix(
            [
                {
                    0: 1.0,
                },
            ]
        )

        nonfinite[0, 0] = np.nan

        with self.assertRaisesRegex(
            ValueError,
            "non-finite",
        ):
            score_contract.validate_score_matrix(
                nonfinite
            )

        with self.assertRaisesRegex(
            ValueError,
            "not normalized",
        ):
            score_contract.validate_score_matrix(
                score_matrix(
                    [
                        {
                            0: 0.8,
                        },
                    ]
                )
            )

    def test_ensemble_is_exact_arithmetic_mean(
        self,
    ):
        first = score_matrix(
            [
                {
                    0: 0.8,
                    136: 0.2,
                },
                {
                    1: 0.4,
                    137: 0.6,
                },
            ]
        )

        second = score_matrix(
            [
                {
                    0: 0.4,
                    136: 0.6,
                },
                {
                    1: 0.8,
                    137: 0.2,
                },
            ]
        )

        ensemble = score_contract.ensemble_score_matrices(
            [
                first,
                second,
            ]
        )

        np.testing.assert_allclose(
            ensemble,
            (
                first
                + second
            )
            / 2.0,
        )

    def test_ensemble_rejects_shape_mismatch(
        self,
    ):
        first = score_matrix(
            [
                {
                    0: 1.0,
                },
            ]
        )

        second = score_matrix(
            [
                {
                    0: 1.0,
                },
                {
                    0: 1.0,
                },
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "expected",
        ):
            score_contract.ensemble_score_matrices(
                [
                    first,
                    second,
                ]
            )

    def test_score_components_and_strict_threshold_summary(
        self,
    ):
        scores = score_matrix(
            [
                {
                    0: 0.5,
                    136: 0.3,
                    137: 0.2,
                },
                {
                    1: 0.9,
                    136: 0.05,
                    137: 0.05,
                },
                {
                    2: 0.998,
                    136: 0.001,
                    137: 0.001,
                },
            ]
        )

        components = score_contract.score_components(
            scores
        )

        np.testing.assert_allclose(
            components["signal"],
            [
                0.5,
                0.9,
                0.998,
            ],
        )

        summary = score_contract.summarize_score_matrix(
            scores
        )

        self.assertEqual(
            summary[
                "signal_thresholds_strict_greater_than"
            ]["0.5"]["count"],
            2,
        )

        self.assertEqual(
            summary[
                "signal_thresholds_strict_greater_than"
            ]["0.9"]["count"],
            1,
        )

        self.assertEqual(
            summary[
                "signal_thresholds_strict_greater_than"
            ]["0.997"]["count"],
            1,
        )

        self.assertEqual(
            summary["top1_class_counts"],
            {
                "0": 1,
                "1": 1,
                "2": 1,
            },
        )

    def test_top1_agreement_fraction(
        self,
    ):
        first = score_matrix(
            [
                {
                    0: 0.8,
                    136: 0.2,
                },
                {
                    1: 0.8,
                    137: 0.2,
                },
            ]
        )

        second = score_matrix(
            [
                {
                    0: 0.7,
                    136: 0.3,
                },
                {
                    137: 0.7,
                    1: 0.3,
                },
            ]
        )

        agreement = score_contract.model_top1_agreement_fraction(
            [
                first,
                second,
            ]
        )

        self.assertEqual(
            agreement,
            0.5,
        )

    def test_signal_grid_summary_maps_classes_to_mass_bins(
        self,
    ):
        scores = score_matrix(
            [
                {
                    0: 0.8,
                    136: 0.1,
                    137: 0.1,
                },
                {
                    135: 0.6,
                    136: 0.2,
                    137: 0.2,
                },
            ]
        )

        summary = score_contract.summarize_signal_grid(
            scores,
            top_k=2,
        )

        raw = summary[
            "top_cells_by_mean_raw_probability"
        ]

        self.assertEqual(
            raw[0]["class_index"],
            0,
        )

        self.assertEqual(
            raw[0]["first_mass_bin"],
            [
                40.0,
                50.0,
            ],
        )

        self.assertEqual(
            raw[0]["second_mass_bin"],
            [
                40.0,
                50.0,
            ],
        )

        self.assertEqual(
            raw[1]["class_index"],
            135,
        )

        self.assertEqual(
            raw[1]["first_mass_bin"],
            [
                190.0,
                200.0,
            ],
        )

        self.assertEqual(
            raw[1]["second_mass_bin"],
            [
                190.0,
                200.0,
            ],
        )

    def test_signal_grid_threshold_and_empty_selection(
        self,
    ):
        scores = score_matrix(
            [
                {
                    0: 0.5,
                    136: 0.5,
                },
                {
                    1: 0.9,
                    136: 0.1,
                },
            ]
        )

        selected = score_contract.summarize_signal_grid(
            scores,
            minimum_signal_probability=0.5,
        )

        self.assertEqual(
            selected["n_selected_events"],
            1,
        )

        empty = score_contract.summarize_signal_grid(
            scores,
            minimum_signal_probability=0.95,
        )

        self.assertEqual(
            empty["n_selected_events"],
            0,
        )

        self.assertEqual(
            empty[
                "top_cells_by_mean_raw_probability"
            ],
            [],
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
