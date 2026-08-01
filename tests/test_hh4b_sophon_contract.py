#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[1]

MODULE_PATH = (
    REPO
    / "scripts"
    / "analysis"
    / "hh4b_sophon_contract.py"
)


def load_contract():
    spec = importlib.util.spec_from_file_location(
        "hh4b_sophon_contract",
        MODULE_PATH,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "could not load {}".format(
                MODULE_PATH
            )
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


contract = load_contract()


def particles(
    count=2,
):
    base = np.arange(
        count,
        dtype=np.float64,
    )

    return {
        "part_px": 3.0 + base,
        "part_py": 4.0 + base,
        "part_pz": 5.0 + base,
        "part_energy": 10.0 + base,
        "part_eta": 0.1 + 0.01 * base,
        "part_phi": -0.2 + 0.02 * base,
        "part_deta": 0.03 + 0.001 * base,
        "part_dphi": -0.04 - 0.001 * base,
        "part_charge": np.where(
            base.astype(int) % 2 == 0,
            1.0,
            0.0,
        ),
        "part_pid": np.full(
            count,
            211,
            dtype=np.int64,
        ),
        "part_d0val": 0.2 + 0.01 * base,
        "part_d0err": 0.1 + 0.01 * base,
        "part_dzval": -0.3 - 0.01 * base,
        "part_dzerr": 0.2 + 0.01 * base,
    }


class SophonContractTests(
    unittest.TestCase
):
    def test_shapes_order_and_zero_padding(
        self,
    ):
        result = contract.build_sophon_inputs(
            particles(2),
            pfcand_sum_pt=20.0,
            pfcand_sum_energy=30.0,
        )

        self.assertEqual(
            result["pf_features"].shape,
            (1, 19, 256),
        )

        self.assertEqual(
            result["pf_vectors"].shape,
            (1, 4, 256),
        )

        self.assertEqual(
            result["pf_mask"].shape,
            (1, 1, 256),
        )

        self.assertEqual(
            result["pf_features"].dtype,
            np.float32,
        )

        self.assertEqual(
            result["pf_vectors"].dtype,
            np.float32,
        )

        self.assertEqual(
            result["pf_mask"].dtype,
            np.float32,
        )

        expected_pt_log0 = np.clip(
            (
                np.log(5.0)
                - 1.7
            )
            * 0.7,
            -5.0,
            5.0,
        )

        expected_erel0 = np.clip(
            (
                np.log(
                    10.0 / 30.0
                )
                + 4.7
            )
            * 0.7,
            -5.0,
            5.0,
        )

        expected_dr0 = np.clip(
            (
                np.hypot(
                    0.03,
                    -0.04,
                )
                - 0.2
            )
            * 4.0,
            -5.0,
            5.0,
        )

        np.testing.assert_allclose(
            result["pf_features"][
                0,
                0,
                0,
            ],
            expected_pt_log0,
        )

        np.testing.assert_allclose(
            result["pf_features"][
                0,
                3,
                0,
            ],
            expected_erel0,
        )

        np.testing.assert_allclose(
            result["pf_features"][
                0,
                4,
                0,
            ],
            expected_dr0,
        )

        np.testing.assert_allclose(
            result["pf_vectors"][
                0,
                :,
                0,
            ],
            [
                3.0,
                4.0,
                5.0,
                10.0,
            ],
        )

        np.testing.assert_array_equal(
            result["pf_mask"][
                0,
                0,
                :4,
            ],
            [
                1.0,
                1.0,
                0.0,
                0.0,
            ],
        )

        self.assertEqual(
            np.count_nonzero(
                result["pf_features"][
                    0,
                    :,
                    2:,
                ]
            ),
            0,
        )

        self.assertEqual(
            np.count_nonzero(
                result["pf_vectors"][
                    0,
                    :,
                    2:,
                ]
            ),
            0,
        )

    def test_pid_category_contract(
        self,
    ):
        event = particles(5)

        event["part_pid"] = np.asarray(
            [
                11,
                -13,
                22,
                211,
                130,
            ]
        )

        event["part_charge"] = np.asarray(
            [
                -1.0,
                1.0,
                0.0,
                1.0,
                0.0,
            ]
        )

        features = contract.build_sophon_inputs(
            event,
            pfcand_sum_pt=100.0,
            pfcand_sum_energy=100.0,
        )["pf_features"][0]

        np.testing.assert_array_equal(
            features[9, :5],
            [1, 0, 0, 0, 0],
        )

        np.testing.assert_array_equal(
            features[10, :5],
            [0, 1, 0, 0, 0],
        )

        np.testing.assert_array_equal(
            features[8, :5],
            [0, 0, 1, 0, 0],
        )

        np.testing.assert_array_equal(
            features[6, :5],
            [0, 0, 0, 1, 0],
        )

        np.testing.assert_array_equal(
            features[7, :5],
            [0, 0, 0, 0, 1],
        )

    def test_error_clipping_and_tanh(
        self,
    ):
        event = particles(2)

        event["part_d0val"] = np.asarray(
            [100.0, -100.0]
        )

        event["part_d0err"] = np.asarray(
            [-0.5, 1.5]
        )

        event["part_dzval"] = np.asarray(
            [100.0, -100.0]
        )

        event["part_dzerr"] = np.asarray(
            [2.0, -2.0]
        )

        features = contract.build_sophon_inputs(
            event,
            pfcand_sum_pt=20.0,
            pfcand_sum_energy=30.0,
        )["pf_features"][0]

        np.testing.assert_allclose(
            features[11, :2],
            [1.0, -1.0],
            atol=1.0e-6,
        )

        np.testing.assert_array_equal(
            features[12, :2],
            [0.0, 1.0],
        )

        np.testing.assert_allclose(
            features[13, :2],
            [1.0, -1.0],
            atol=1.0e-6,
        )

        np.testing.assert_array_equal(
            features[14, :2],
            [1.0, 0.0],
        )

    def test_truncation_preserves_first_256(
        self,
    ):
        event = particles(300)

        result = contract.build_sophon_inputs(
            event,
            pfcand_sum_pt=100000.0,
            pfcand_sum_energy=100000.0,
        )

        self.assertEqual(
            int(result["particle_count"]),
            300,
        )

        self.assertEqual(
            int(
                result[
                    "used_particle_count"
                ]
            ),
            256,
        )

        self.assertTrue(
            bool(
                result["was_truncated"]
            )
        )

        np.testing.assert_allclose(
            result["pf_vectors"][
                0,
                0,
            ],
            event["part_px"][:256],
        )

        self.assertEqual(
            np.count_nonzero(
                result["pf_mask"]
            ),
            256,
        )

    def test_input_validation(
        self,
    ):
        event = particles(2)

        event["part_py"] = np.asarray(
            [4.0]
        )

        with self.assertRaisesRegex(
            ValueError,
            "inconsistent lengths",
        ):
            contract.build_sophon_inputs(
                event,
                pfcand_sum_pt=20.0,
                pfcand_sum_energy=30.0,
            )

        with self.assertRaisesRegex(
            ValueError,
            "pfcand_sum_pt",
        ):
            contract.build_sophon_inputs(
                particles(2),
                pfcand_sum_pt=0.0,
                pfcand_sum_energy=30.0,
            )

    def test_mass_grid_spans_all_136_classes_and_is_symmetric(
        self,
    ):
        observed = set()

        centers = [
            45.0 + 10.0 * index
            for index in range(16)
        ]

        for first_index, first_mass in enumerate(
            centers
        ):
            for second_mass in centers[
                first_index:
            ]:
                forward = contract.calculate_class_index(
                    first_mass,
                    second_mass,
                    999,
                )

                reverse = contract.calculate_class_index(
                    second_mass,
                    first_mass,
                    999,
                )

                self.assertEqual(
                    forward,
                    reverse,
                )

                observed.add(
                    forward
                )

        self.assertEqual(
            observed,
            set(range(136)),
        )

    def test_mass_boundary_and_background_contract(
        self,
    ):
        self.assertEqual(
            contract.calculate_class_index(
                40.0,
                40.0,
                999,
            ),
            0,
        )

        self.assertEqual(
            contract.calculate_class_index(
                50.0,
                50.0,
                999,
            ),
            16,
        )

        self.assertEqual(
            contract.calculate_class_index(
                200.0,
                200.0,
                999,
            ),
            135,
        )

        self.assertEqual(
            contract.calculate_class_index(
                39.99,
                125.0,
                0,
            ),
            136,
        )

        self.assertEqual(
            contract.calculate_class_index(
                39.99,
                125.0,
                7,
            ),
            137,
        )

        self.assertEqual(
            contract.calculate_class_index(
                125.0,
                200.01,
                0,
            ),
            136,
        )

        self.assertEqual(
            contract.calculate_class_index(
                125.0,
                200.01,
                7,
            ),
            137,
        )

    def test_class_index_inverse_endpoints(
        self,
    ):
        self.assertEqual(
            contract.class_index_to_mass_bins(
                0
            ),
            (
                (40.0, 50.0),
                (40.0, 50.0),
            ),
        )

        self.assertEqual(
            contract.class_index_to_mass_bins(
                135
            ),
            (
                (190.0, 200.0),
                (190.0, 200.0),
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            contract.class_index_to_mass_bins(
                136
            )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
