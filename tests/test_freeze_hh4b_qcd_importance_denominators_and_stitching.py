#!/usr/bin/env python3
"""Targeted tests for hard-QCD denominator and stitching freeze."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts"
    / "analysis"
    / "freeze_hh4b_qcd_importance_denominators_and_stitching.py"
)

spec = importlib.util.spec_from_file_location(
    "qcd_denominator_stitching",
    SCRIPT,
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import QCD denominator freezer")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class QcdDenominatorStitchingTests(unittest.TestCase):
    def test_canonical_field(self) -> None:
        self.assertEqual(
            module.canonical_field("generator.sum_event_weights"),
            "sum_event_weights",
        )

    def test_open_bound(self) -> None:
        self.assertEqual(module.parse_bound("Inf"), "Inf")
        self.assertEqual(module.parse_bound("infinity"), "Inf")
        self.assertEqual(module.parse_bound("300"), 300)

    def test_expected_interval_closure(self) -> None:
        self.assertTrue(
            module.validate_exclusive_intervals(
                list(module.EXPECTED_EXCLUSIVE_BINS)
            )
        )

    def test_gap_is_rejected(self) -> None:
        intervals = list(module.EXPECTED_EXCLUSIVE_BINS)
        intervals[2] = (101, 200)
        self.assertFalse(
            module.validate_exclusive_intervals(intervals)
        )

    def test_duplicate_metadata_values_resolve(self) -> None:
        rows = [
            {"field": "sum_event_weights", "value": "10000.0"},
            {
                "field": "generator.sum_event_weights",
                "value": "10000",
            },
            {
                "field": "event_weight_convention",
                "value": module.REQUIRED_CONVENTION,
            },
        ]
        resolved, conflicts = module.resolve_metadata_values(rows)
        self.assertEqual(resolved["sum_event_weights"], 10000.0)
        self.assertEqual(conflicts, [])

    def test_synthetic_coefficient_closure(self) -> None:
        n = [1000.0, 2000.0]
        sigma = [10.0, 12.0]
        sumw = [1000.0, 2000.0]
        total_n = sum(n)
        coefficients = [
            (n_i / total_n) * sigma_i / sumw_i
            for n_i, sigma_i, sumw_i in zip(n, sigma, sumw)
        ]
        observed = sum(
            coefficient * sumw_i
            for coefficient, sumw_i in zip(coefficients, sumw)
        )
        expected = sum(
            n_i * sigma_i
            for n_i, sigma_i in zip(n, sigma)
        ) / total_n
        self.assertAlmostEqual(observed, expected)

    def test_testfill_role(self) -> None:
        self.assertEqual(
            module.campaign_role(
                "qcd_hardqcd_importance_testfill_bin3_10k_20260717"
            ),
            "diagnostic_testfill_excluded",
        )

    def test_incomplete_metadata_resolves_without_numeric_cast(self) -> None:
        rows = [
            {"field": "n_events", "value": "10000"},
            {"field": "seed", "value": "1201034"},
            {"field": "pthat_min_GeV", "value": "50"},
            {"field": "pthat_max_GeV", "value": "75"},
        ]
        resolved, conflicts = module.resolve_metadata_values(rows)
        self.assertEqual(conflicts, [])
        self.assertEqual(resolved["n_events"], 10000)
        self.assertNotIn("sigma_gen_pb", resolved)
        self.assertNotIn("sum_event_weights", resolved)

    def test_no_event_payload_reader(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        forbidden = (
            "uproot",
            "pyhepmc",
            "read_parquet",
            "pyarrow",
            "tarfile.open",
            "xrdcp",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
