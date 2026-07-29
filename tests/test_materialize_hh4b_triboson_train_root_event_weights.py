#!/usr/bin/env python3
"""Targeted tests for train-only triboson weight sidecars."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts"
    / "analysis"
    / "materialize_hh4b_triboson_train_root_event_weights.py"
)

spec = importlib.util.spec_from_file_location(
    "triboson_train_sidecars",
    SCRIPT,
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import train sidecar materializer")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class TribosonTrainWeightSidecarTests(unittest.TestCase):
    def test_campaign_from_bundle_path(self) -> None:
        self.assertEqual(
            module.campaign_from_bundle_path(
                "/store/user/a/bundles/campaign_x/file.tar.gz"
            ),
            "campaign_x",
        )

    def test_split_event_weight_selected(self) -> None:
        self.assertEqual(
            module.select_nominal_weight_branch(
                [
                    "Event/Event.Weight",
                    "Weight/Weight.Weight",
                ]
            ),
            "Event/Event.Weight",
        )

    def test_one_weight_per_entry(self) -> None:
        values = module.one_weight_per_entry(
            ak.Array([[1.0], [-1.0], [1.0]]),
            3,
        )
        np.testing.assert_array_equal(values, [1.0, -1.0, 1.0])

    def test_atomic_sidecar_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.parquet"
            module.write_sidecar_atomic(
                path,
                member_index=1,
                process="wwz_zbb",
                campaign="campaign",
                root_weight_branch="Event/Event.Weight",
                weights=np.asarray([0.5, -0.5], dtype=np.float64),
            )
            table = pq.read_table(path)
            self.assertEqual(
                table.schema.names,
                [
                    "event",
                    "generator_nominal_weight",
                    "generator_weight_sign",
                ],
            )
            self.assertEqual(table.num_rows, 2)

    def test_source_candidate_never_written(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        forbidden = (
            "write_table(table, candidate_path",
            "candidate_path.unlink",
            "candidate_path.replace",
            "candidate_path.rename",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
