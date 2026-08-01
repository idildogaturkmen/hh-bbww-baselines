#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import uproot


REPO = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO / "scripts" / "analysis"

sys.path.insert(
    0,
    str(ANALYSIS_DIR),
)

import hh4b_sophon_root_contract as root_contract


def score_matrix(
    rows,
):
    scores = np.zeros(
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
            scores[
                row_index,
                class_index,
            ] = probability

    return scores


class SophonRootContractTests(
    unittest.TestCase
):
    def test_required_branch_contracts(
        self,
    ):
        self.assertEqual(
            len(
                root_contract.REQUIRED_INPUT_BRANCHES
            ),
            21,
        )

        self.assertEqual(
            len(
                root_contract.REQUIRED_EVENTS_BRANCHES
            ),
            140,
        )

        root_contract.validate_required_branches(
            list(
                root_contract.REQUIRED_INPUT_BRANCHES
            )
            + [
                "unrelated_extra_branch",
            ],
            required_branches=(
                root_contract.REQUIRED_INPUT_BRANCHES
            ),
            context="synthetic input",
        )

        missing = list(
            root_contract.REQUIRED_INPUT_BRANCHES[
                1:
            ]
        )

        with self.assertRaisesRegex(
            KeyError,
            "pass_selection",
        ):
            root_contract.validate_required_branches(
                missing,
                required_branches=(
                    root_contract.REQUIRED_INPUT_BRANCHES
                ),
                context="synthetic input",
            )

    def test_selection_validation_and_eligible_entries(
        self,
    ):
        eligible = root_contract.eligible_entry_indices(
            [
                1,
                0,
                1,
                1,
                1,
                0,
                1,
            ],
            [
                1,
                1,
                0,
                1,
                1,
                0,
                1,
            ],
        )

        np.testing.assert_array_equal(
            eligible,
            [
                0,
                3,
                4,
                6,
            ],
        )

        self.assertEqual(
            eligible.dtype,
            np.int64,
        )

    def test_selection_validation_rejects_bad_inputs(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "only 0 or 1",
        ):
            root_contract.validate_selection_flags(
                [
                    0,
                    2,
                ],
                name="selection",
            )

        with self.assertRaisesRegex(
            ValueError,
            "different lengths",
        ):
            root_contract.eligible_entry_indices(
                [
                    1,
                    1,
                ],
                [
                    1,
                ],
            )

        with self.assertRaisesRegex(
            ValueError,
            "one-dimensional",
        ):
            root_contract.validate_selection_flags(
                np.ones(
                    (
                        2,
                        2,
                    )
                ),
                name="selection",
            )

    def test_spread_positions_contract(
        self,
    ):
        np.testing.assert_array_equal(
            root_contract.spread_positions(
                4,
                3,
            ),
            [
                0,
                1,
                3,
            ],
        )

        np.testing.assert_array_equal(
            root_contract.spread_positions(
                10,
                1,
            ),
            [
                4,
            ],
        )

        np.testing.assert_array_equal(
            root_contract.spread_positions(
                3,
                10,
            ),
            [
                0,
                1,
                2,
            ],
        )

        self.assertEqual(
            len(
                root_contract.spread_positions(
                    0,
                    3,
                )
            ),
            0,
        )

        with self.assertRaises(
            ValueError
        ):
            root_contract.spread_positions(
                -1,
                2,
            )

    def test_spread_selected_entries_contract(
        self,
    ):
        selected = root_contract.spread_selected_entries(
            [
                1,
                0,
                1,
                1,
                1,
                0,
                1,
            ],
            [
                1,
                1,
                0,
                1,
                1,
                0,
                1,
            ],
            requested=3,
        )

        np.testing.assert_array_equal(
            selected,
            [
                0,
                3,
                6,
            ],
        )

    def test_events_payload_contract(
        self,
    ):
        scores = score_matrix(
            [
                {
                    0: 0.998,
                    136: 0.001,
                    137: 0.001,
                },
                {
                    50: 0.9,
                    136: 0.05,
                    137: 0.05,
                },
            ]
        )

        payload = root_contract.build_events_payload(
            scores
        )

        self.assertEqual(
            set(
                payload
            ),
            set(
                root_contract.REQUIRED_EVENTS_BRANCHES
            ),
        )

        self.assertEqual(
            payload[
                "pass_selection"
            ].dtype,
            np.int32,
        )

        self.assertEqual(
            payload[
                "score_0"
            ].dtype,
            np.float32,
        )

        np.testing.assert_allclose(
            payload[
                "score_0"
            ],
            [
                0.998,
                0.0,
            ],
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_events_root_roundtrip(
        self,
    ):
        scores = score_matrix(
            [
                {
                    0: 0.998,
                    136: 0.001,
                    137: 0.001,
                },
                {
                    50: 0.9,
                    136: 0.05,
                    137: 0.05,
                },
                {
                    135: 0.5,
                    136: 0.3,
                    137: 0.2,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "events.root"
            )

            receipt = root_contract.write_events_tree(
                path,
                scores,
            )

            self.assertEqual(
                receipt[
                    "tree_class"
                ],
                "TTree",
            )

            self.assertEqual(
                receipt[
                    "entries"
                ],
                3,
            )

            self.assertEqual(
                receipt[
                    "required_branch_count"
                ],
                140,
            )

            self.assertGreater(
                receipt[
                    "file_size_bytes"
                ],
                0,
            )

            with uproot.open(
                path
            ) as root_file:
                tree = root_file[
                    "Events"
                ]

                self.assertEqual(
                    tree.classname,
                    "TTree",
                )

                self.assertEqual(
                    set(
                        tree.keys()
                    ),
                    set(
                        root_contract.REQUIRED_EVENTS_BRANCHES
                    ),
                )

            read_scores = (
                root_contract.read_events_score_matrix(
                    path
                )
            )

            np.testing.assert_allclose(
                read_scores,
                scores,
                rtol=0.0,
                atol=1.0e-7,
            )

    def test_output_refusal_and_score_validation(
        self,
    ):
        scores = score_matrix(
            [
                {
                    0: 1.0,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "events.root"
            )

            root_contract.write_events_tree(
                path,
                scores,
            )

            with self.assertRaises(
                FileExistsError
            ):
                root_contract.write_events_tree(
                    path,
                    scores,
                )

        bad_scores = np.zeros(
            (
                1,
                138,
            ),
            dtype=np.float64,
        )

        bad_scores[
            0,
            0,
        ] = 0.8

        with self.assertRaisesRegex(
            ValueError,
            "not normalized",
        ):
            root_contract.build_events_payload(
                bad_scores
            )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
