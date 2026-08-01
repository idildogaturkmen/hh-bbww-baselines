#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import awkward as ak
import numpy as np
import uproot


REPO = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO / "scripts" / "analysis"

sys.path.insert(
    0,
    str(
        ANALYSIS_DIR
    ),
)

import hh4b_sophon_execution_backend as backend
import hh4b_sophon_root_contract as root_contract
import hh4b_sophon_score_contract as score_contract


class FakeNode:
    def __init__(
        self,
        name,
        shape,
        type_name="tensor(float)",
    ):
        self.name = name
        self.shape = shape
        self.type = type_name


class FakeSession:
    def __init__(
        self,
        distribution,
        *,
        input_names=backend.MODEL_INPUT_NAMES,
        output_name=backend.MODEL_OUTPUT_NAME,
        output_shape=(
            "N",
            138,
        ),
    ):
        self.distribution = np.asarray(
            distribution,
            dtype=np.float32,
        )

        self.input_names = tuple(
            input_names
        )

        self.output_name = output_name
        self.output_shape = output_shape

        self.run_called = False
        self.last_feed_shapes = None

    def get_inputs(
        self,
    ):
        shapes = {
            "pf_features": (
                "N",
                19,
                "n_pf",
            ),
            "pf_vectors": (
                "N",
                4,
                "n_pf",
            ),
            "pf_mask": (
                "N",
                1,
                "n_pf",
            ),
        }

        return [
            FakeNode(
                name,
                shapes.get(
                    name,
                    (
                        "N",
                        1,
                        "n_pf",
                    ),
                ),
            )
            for name in self.input_names
        ]

    def get_outputs(
        self,
    ):
        return [
            FakeNode(
                self.output_name,
                self.output_shape,
            ),
        ]

    def run(
        self,
        output_names,
        feed,
    ):
        self.run_called = True

        if output_names != [
            backend.MODEL_OUTPUT_NAME,
        ]:
            raise AssertionError(
                "unexpected requested outputs"
            )

        self.last_feed_shapes = {
            name: tuple(
                values.shape
            )
            for name, values in feed.items()
        }

        event_count = int(
            feed[
                "pf_features"
            ].shape[
                0
            ]
        )

        return [
            np.repeat(
                self.distribution[
                    np.newaxis,
                    :
                ],
                event_count,
                axis=0,
            ),
        ]


def probability_vector(
    mapping,
    *,
    classes=138,
):
    values = np.zeros(
        classes,
        dtype=np.float32,
    )

    for class_index, probability in mapping.items():
        values[
            class_index
        ] = probability

    return values


def particle_event(
    count,
    *,
    offset,
):
    index = np.arange(
        count,
        dtype=np.float32,
    )

    px = (
        1.0
        + offset
        + 0.25
        * index
    )

    py = (
        0.5
        + 0.10
        * index
    )

    pz = (
        0.2
        + 0.15
        * index
    )

    energy = (
        np.sqrt(
            px * px
            + py * py
            + pz * pz
        )
        + 1.0
    )

    eta = (
        -1.0
        + 0.1
        * index
    )

    phi = (
        -0.5
        + 0.05
        * index
    )

    deta = (
        0.01
        * (
            index
            + 1.0
        )
    )

    dphi = (
        -0.02
        * (
            index
            + 1.0
        )
    )

    pid_cycle = np.asarray(
        [
            211,
            -211,
            22,
            11,
            13,
            130,
        ],
        dtype=np.int32,
    )

    pid = np.resize(
        pid_cycle,
        count,
    )

    charge = np.where(
        np.isin(
            pid,
            [
                22,
                130,
            ],
        ),
        0.0,
        np.where(
            pid > 0,
            1.0,
            -1.0,
        ),
    ).astype(
        np.float32
    )

    return {
        "part_px": px,
        "part_py": py,
        "part_pz": pz,
        "part_energy": energy.astype(
            np.float32
        ),
        "part_eta": eta,
        "part_phi": phi,
        "part_deta": deta,
        "part_dphi": dphi,
        "part_charge": charge,
        "part_pid": pid,
        "part_d0val": (
            0.001
            * index
        ),
        "part_d0err": np.full(
            count,
            0.01,
            dtype=np.float32,
        ),
        "part_dzval": (
            0.002
            * index
        ),
        "part_dzerr": np.full(
            count,
            0.02,
            dtype=np.float32,
        ),
    }


def write_synthetic_input(
    path,
    *,
    pass_selection=None,
    pass_4j3b_selection=None,
):
    lengths = [
        2,
        4,
        1,
        3,
        5,
        2,
        4,
    ]

    event_particles = [
        particle_event(
            count,
            offset=0.2
            * event_index,
        )
        for event_index, count in enumerate(
            lengths
        )
    ]

    if pass_selection is None:
        pass_selection = [
            1,
            0,
            1,
            1,
            1,
            0,
            1,
        ]

    if pass_4j3b_selection is None:
        pass_4j3b_selection = [
            1,
            1,
            0,
            1,
            1,
            0,
            1,
        ]

    branch_types = {
        "pass_selection": np.int32,
        "pass_4j3b_selection": np.int32,
        "pfcand_sum_pt": np.float32,
        "pfcand_sum_energy": np.float32,
        "gen_higgs1_mass": np.float32,
        "gen_higgs2_mass": np.float32,
        "process_index": np.int32,
    }

    for branch_name in (
        root_contract.INPUT_PARTICLE_BRANCHES
    ):
        if branch_name == "part_pid":
            branch_types[
                branch_name
            ] = "var * int32"
        else:
            branch_types[
                branch_name
            ] = "var * float32"

    payload = {
        "pass_selection": np.asarray(
            pass_selection,
            dtype=np.int32,
        ),
        "pass_4j3b_selection": np.asarray(
            pass_4j3b_selection,
            dtype=np.int32,
        ),
        "gen_higgs1_mass": np.full(
            len(
                lengths
            ),
            91.0,
            dtype=np.float32,
        ),
        "gen_higgs2_mass": np.full(
            len(
                lengths
            ),
            125.0,
            dtype=np.float32,
        ),
        "process_index": np.full(
            len(
                lengths
            ),
            16,
            dtype=np.int32,
        ),
    }

    sum_pt = []
    sum_energy = []

    for event in event_particles:
        sum_pt.append(
            float(
                np.sum(
                    np.hypot(
                        event[
                            "part_px"
                        ],
                        event[
                            "part_py"
                        ],
                    )
                )
                * 1.2
            )
        )

        sum_energy.append(
            float(
                np.sum(
                    event[
                        "part_energy"
                    ]
                )
                * 1.2
            )
        )

    payload[
        "pfcand_sum_pt"
    ] = np.asarray(
        sum_pt,
        dtype=np.float32,
    )

    payload[
        "pfcand_sum_energy"
    ] = np.asarray(
        sum_energy,
        dtype=np.float32,
    )

    for branch_name in (
        root_contract.INPUT_PARTICLE_BRANCHES
    ):
        payload[
            branch_name
        ] = ak.Array(
            [
                event[
                    branch_name
                ].tolist()
                for event in event_particles
            ]
        )

    with uproot.recreate(
        path
    ) as root_file:
        tree = root_file.mktree(
            root_contract.INPUT_TREE_NAME,
            branch_types,
        )

        tree.extend(
            payload
        )


def standard_sessions():
    return [
        FakeSession(
            probability_vector(
                {
                    10: 0.70,
                    136: 0.20,
                    137: 0.10,
                }
            )
        ),
        FakeSession(
            probability_vector(
                {
                    10: 0.60,
                    20: 0.10,
                    136: 0.20,
                    137: 0.10,
                }
            )
        ),
        FakeSession(
            probability_vector(
                {
                    10: 0.50,
                    20: 0.20,
                    136: 0.20,
                    137: 0.10,
                }
            )
        ),
    ]


class SophonExecutionBackendTests(
    unittest.TestCase
):
    def test_prepare_batch_spread_and_shapes(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "input.root"
            )

            write_synthetic_input(
                path
            )

            with uproot.open(
                path
            ) as root_file:
                prepared = backend.prepare_batch_from_tree(
                    root_file[
                        "tree"
                    ],
                    requested_events=3,
                )

            np.testing.assert_array_equal(
                prepared.selected_entries,
                [
                    0,
                    3,
                    6,
                ],
            )

            self.assertEqual(
                prepared.model_inputs[
                    "pf_features"
                ].shape,
                (
                    3,
                    19,
                    256,
                ),
            )

            self.assertEqual(
                prepared.model_inputs[
                    "pf_vectors"
                ].shape,
                (
                    3,
                    4,
                    256,
                ),
            )

            self.assertEqual(
                prepared.model_inputs[
                    "pf_mask"
                ].shape,
                (
                    3,
                    1,
                    256,
                ),
            )

            self.assertEqual(
                prepared.metadata[
                    "eligible_entries"
                ],
                4,
            )

            self.assertEqual(
                prepared.metadata[
                    "truncated_event_count"
                ],
                0,
            )

    def test_no_eligible_events_are_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "input.root"
            )

            write_synthetic_input(
                path,
                pass_selection=[
                    0,
                ]
                * 7,
                pass_4j3b_selection=[
                    0,
                ]
                * 7,
            )

            with uproot.open(
                path
            ) as root_file:
                with self.assertRaisesRegex(
                    ValueError,
                    "no eligible entries",
                ):
                    backend.prepare_batch_from_tree(
                        root_file[
                            "tree"
                        ],
                        requested_events=3,
                    )

    def test_model_input_validation_rejects_bad_dtype(
        self,
    ):
        valid = {
            "pf_features": np.ones(
                (
                    2,
                    19,
                    256,
                ),
                dtype=np.float32,
            ),
            "pf_vectors": np.ones(
                (
                    2,
                    4,
                    256,
                ),
                dtype=np.float32,
            ),
            "pf_mask": np.ones(
                (
                    2,
                    1,
                    256,
                ),
                dtype=np.float32,
            ),
        }

        bad = dict(
            valid
        )

        bad[
            "pf_features"
        ] = bad[
            "pf_features"
        ].astype(
            np.float64
        )

        with self.assertRaisesRegex(
            TypeError,
            "float32",
        ):
            backend.validate_model_input_batch(
                bad
            )

    def test_three_mock_sessions_and_exact_ensemble(
        self,
    ):
        model_inputs = {
            "pf_features": np.ones(
                (
                    2,
                    19,
                    256,
                ),
                dtype=np.float32,
            ),
            "pf_vectors": np.ones(
                (
                    2,
                    4,
                    256,
                ),
                dtype=np.float32,
            ),
            "pf_mask": np.ones(
                (
                    2,
                    1,
                    256,
                ),
                dtype=np.float32,
            ),
        }

        sessions = standard_sessions()

        model_scores, interfaces = (
            backend.run_model_sessions(
                sessions,
                model_inputs,
            )
        )

        ensemble = (
            score_contract.ensemble_score_matrices(
                model_scores
            )
        )

        expected = np.mean(
            np.stack(
                [
                    session.distribution
                    for session in sessions
                ],
                axis=0,
            ),
            axis=0,
        )

        np.testing.assert_allclose(
            ensemble[
                0
            ],
            expected,
            rtol=0.0,
            atol=1.0e-7,
        )

        self.assertEqual(
            len(
                interfaces
            ),
            3,
        )

        self.assertTrue(
            all(
                session.run_called
                for session in sessions
            )
        )

    def test_interface_mismatch_is_rejected_before_run(
        self,
    ):
        bad_session = FakeSession(
            probability_vector(
                {
                    0: 1.0,
                }
            ),
            input_names=(
                "pf_features",
                "pf_vectors",
                "wrong_mask",
            ),
        )

        model_inputs = {
            "pf_features": np.ones(
                (
                    1,
                    19,
                    256,
                ),
                dtype=np.float32,
            ),
            "pf_vectors": np.ones(
                (
                    1,
                    4,
                    256,
                ),
                dtype=np.float32,
            ),
            "pf_mask": np.ones(
                (
                    1,
                    1,
                    256,
                ),
                dtype=np.float32,
            ),
        }

        sessions = [
            bad_session,
            standard_sessions()[
                1
            ],
            standard_sessions()[
                2
            ],
        ]

        with self.assertRaisesRegex(
            ValueError,
            "input names",
        ):
            backend.run_model_sessions(
                sessions,
                model_inputs,
            )

        self.assertFalse(
            bad_session.run_called
        )

    def test_bad_model_output_is_rejected(
        self,
    ):
        bad_distribution = probability_vector(
            {
                0: 0.8,
            }
        )

        sessions = [
            FakeSession(
                bad_distribution
            ),
            standard_sessions()[
                1
            ],
            standard_sessions()[
                2
            ],
        ]

        model_inputs = {
            "pf_features": np.ones(
                (
                    1,
                    19,
                    256,
                ),
                dtype=np.float32,
            ),
            "pf_vectors": np.ones(
                (
                    1,
                    4,
                    256,
                ),
                dtype=np.float32,
            ),
            "pf_mask": np.ones(
                (
                    1,
                    1,
                    256,
                ),
                dtype=np.float32,
            ),
        }

        with self.assertRaisesRegex(
            ValueError,
            "not normalized",
        ):
            backend.run_model_sessions(
                sessions,
                model_inputs,
            )

    def test_execute_local_canary_receipt(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "input.root"
            )

            write_synthetic_input(
                path
            )

            result = backend.execute_local_canary(
                path,
                standard_sessions(),
                requested_events=3,
            )

            np.testing.assert_array_equal(
                result.selected_entries,
                [
                    0,
                    3,
                    6,
                ],
            )

            self.assertEqual(
                result.ensemble_scores.shape,
                (
                    3,
                    138,
                ),
            )

            self.assertEqual(
                result.receipt[
                    "inference"
                ][
                    "released_model_count"
                ],
                3,
            )

            self.assertEqual(
                result.receipt[
                    "inference"
                ][
                    "model_top1_agreement_fraction"
                ],
                1.0,
            )

            self.assertIsNone(
                result.receipt[
                    "events_tree"
                ]
            )

            self.assertFalse(
                result.receipt[
                    "operations"
                ][
                    "network_accessed"
                ]
            )

    def test_events_tree_roundtrip_and_overwrite_refusal(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(
                directory
            )

            input_path = (
                directory_path
                / "input.root"
            )

            output_path = (
                directory_path
                / "events.root"
            )

            write_synthetic_input(
                input_path
            )

            result = backend.execute_local_canary(
                input_path,
                standard_sessions(),
                requested_events=3,
                events_output_path=output_path,
            )

            roundtrip = (
                root_contract.read_events_score_matrix(
                    output_path
                )
            )

            np.testing.assert_allclose(
                roundtrip,
                result.ensemble_scores,
                rtol=0.0,
                atol=1.0e-7,
            )

            self.assertTrue(
                result.receipt[
                    "operations"
                ][
                    "events_tree_written"
                ]
            )

            with self.assertRaises(
                FileExistsError
            ):
                backend.execute_local_canary(
                    input_path,
                    standard_sessions(),
                    requested_events=3,
                    events_output_path=output_path,
                )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
