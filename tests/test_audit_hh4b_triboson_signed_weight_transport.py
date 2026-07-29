#!/usr/bin/env python3
"""Targeted tests for the triboson signed-weight transport auditor."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import awkward as ak
import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts"
    / "analysis"
    / "audit_hh4b_triboson_signed_weight_transport.py"
)

spec = importlib.util.spec_from_file_location(
    "triboson_weight_transport",
    SCRIPT,
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import triboson auditor")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class TribosonSignedWeightTransportTests(unittest.TestCase):
    def test_campaign_derived_from_remote_bundle_path(self) -> None:
        row = {
            "remote_bundle_path": (
                "/store/user/iturkmen/hh4b_delphes/run2_13tev/frozen_v2/"
                "bundles/final_background_triboson_zbb_scaleout40k_20260722_v1/"
                "wwz_zbb_run2_frozen_v2_train_shard9901_5200_bundle.tar.gz"
            )
        }
        self.assertEqual(
            module.campaign_from_source_row(row),
            "final_background_triboson_zbb_scaleout40k_20260722_v1",
        )

    def test_campaign_explicit_value_must_agree(self) -> None:
        row = {
            "bundle_campaign": "wrong_campaign",
            "remote_bundle_path": (
                "/store/user/iturkmen/hh4b_delphes/run2_13tev/frozen_v2/"
                "bundles/actual_campaign/file_bundle.tar.gz"
            ),
        }
        with self.assertRaises(RuntimeError):
            module.campaign_from_source_row(row)

    def test_split_event_weight_beats_weight_collection(self) -> None:
        self.assertEqual(
            module.select_nominal_weight_branch(
                [
                    "Event/Event.Weight",
                    "Weight/Weight.Weight",
                    "Jet/Jet.PT",
                ]
            ),
            "Event/Event.Weight",
        )

    def test_flat_event_weight_is_supported(self) -> None:
        self.assertEqual(
            module.select_nominal_weight_branch(
                ["Weight.Weight", "Event.Weight", "Jet.PT"]
            ),
            "Event.Weight",
        )

    def test_generic_weight_collection_is_not_accepted(self) -> None:
        with self.assertRaises(RuntimeError):
            module.select_nominal_weight_branch(
                ["Weight/Weight.Weight", "Jet/Jet.PT"]
            )

    def test_signed_uniform_classification(self) -> None:
        classification, ngen = module.classify_weights(
            np.asarray([1.0, -1.0, 1.0], dtype=np.float64)
        )
        self.assertEqual(classification, "signed_uniform_magnitude")
        self.assertFalse(ngen)

    def test_one_weight_per_entry_flatten(self) -> None:
        values = module.awkward_to_one_weight_per_entry(
            ak.Array([[1.0], [-1.0], [1.0]]),
            3,
        )
        np.testing.assert_allclose(values, [1.0, -1.0, 1.0])

    def test_no_parquet_reader(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        forbidden = (
            "read_parquet",
            "pyarrow",
            "ParquetFile",
            "pandas.read_parquet",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
