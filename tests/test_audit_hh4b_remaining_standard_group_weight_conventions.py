#!/usr/bin/env python3
"""Targeted tests for remaining standard-group weight audit."""

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
    / "audit_hh4b_remaining_standard_group_weight_conventions.py"
)

spec = importlib.util.spec_from_file_location(
    "remaining_standard_weight_audit",
    SCRIPT,
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import remaining-standard auditor")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class RemainingStandardWeightAuditTests(unittest.TestCase):
    def test_shared_campaign_labels_count_per_process(self) -> None:
        group_map = {
            "process_a": {
                "campaigns": "shared_campaign,a_only",
            },
            "process_b": {
                "campaigns": "shared_campaign,b_only",
            },
        }
        pairs = module.process_campaign_pairs(
            group_map,
            ["process_a", "process_b"],
        )
        self.assertEqual(
            pairs,
            {
                ("process_a", "shared_campaign"),
                ("process_a", "a_only"),
                ("process_b", "shared_campaign"),
                ("process_b", "b_only"),
            },
        )
        self.assertEqual(
            {campaign for _, campaign in pairs},
            {"shared_campaign", "a_only", "b_only"},
        )

    def test_campaign_from_bundle_path(self) -> None:
        self.assertEqual(
            module.campaign_from_bundle_path(
                "/store/user/a/bundles/campaign/file.tar.gz"
            ),
            "campaign",
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

    def test_uniform_positive_classification(self) -> None:
        classification, ngen = module.classify_weights(
            np.asarray([0.5, 0.5, 0.5], dtype=np.float64)
        )
        self.assertEqual(classification, "uniform_positive")
        self.assertTrue(ngen)

    def test_variable_positive_is_fail_closed(self) -> None:
        classification, ngen = module.classify_weights(
            np.asarray([0.5, 0.6], dtype=np.float64)
        )
        self.assertEqual(classification, "variable_positive")
        self.assertFalse(ngen)

    def test_one_weight_per_entry(self) -> None:
        values = module.one_weight_per_entry(
            ak.Array([[1.0], [1.0], [1.0]]),
            3,
        )
        np.testing.assert_array_equal(values, [1.0, 1.0, 1.0])

    def test_no_candidate_reader(self) -> None:
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
