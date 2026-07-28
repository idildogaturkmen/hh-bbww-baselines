#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts/analysis/"
    "validate_hh4b_ttbar8_canary_retry2.py"
)

SPEC = importlib.util.spec_from_file_location(
    "retry2_validation",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Retry2ValidationTests(unittest.TestCase):
    def test_exact_integer_comparison(self) -> None:
        result = MODULE.compare_series(
            pd.Series([1, 2, 3], dtype="int64"),
            pd.Series([1, 2, 3], dtype="int64"),
        )

        self.assertTrue(result["all_values_match"])
        self.assertEqual(result["mismatch_count"], 0)
        self.assertEqual(
            result["comparison_mode"],
            "exact",
        )

    def test_float_comparison_tolerance(self) -> None:
        result = MODULE.compare_series(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0 + 1.0e-9, 2.0]),
        )

        self.assertTrue(result["all_values_match"])
        self.assertEqual(result["mismatch_count"], 0)

    def test_float_comparison_detects_mismatch(self) -> None:
        result = MODULE.compare_series(
            pd.Series([1.0, 2.0]),
            pd.Series([1.1, 2.0]),
        )

        self.assertFalse(result["all_values_match"])
        self.assertEqual(result["mismatch_count"], 1)

    def test_string_comparison(self) -> None:
        result = MODULE.compare_series(
            pd.Series(["a", "b"]),
            pd.Series(["a", "b"]),
        )

        self.assertTrue(result["all_values_match"])
        self.assertEqual(
            result["comparison_mode"],
            "exact_string",
        )

    def test_event_key_detection(self) -> None:
        self.assertEqual(
            MODULE.find_event_key(
                ["sample", "event", "m_hh"]
            ),
            "event",
        )

    def test_event_key_detection_rejects_unknown(self) -> None:
        with self.assertRaises(MODULE.ValidationError):
            MODULE.find_event_key(
                ["sample", "m_hh"]
            )

    def test_table_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "table"

            MODULE.write_table_bundle(
                base,
                [
                    {
                        "column": "m_hh",
                        "match": True,
                    }
                ],
            )

            self.assertTrue(
                base.with_suffix(".tsv").is_file()
            )
            self.assertTrue(
                base.with_suffix(".md").is_file()
            )
            self.assertTrue(
                base.with_suffix(".tex").is_file()
            )

    def test_frozen_retry_identity(self) -> None:
        self.assertEqual(
            MODULE.JOB_ID,
            "3655099.0",
        )
        self.assertEqual(
            MODULE.MEMBER,
            "ttbar_100k_shard003",
        )
        self.assertEqual(
            MODULE.SEED,
            105003,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
