#!/usr/bin/env python3
"""Targeted tests for the standard LHE-weight canary."""

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
    / "run_hh4b_standard_lhe_weight_canary.py"
)

spec = importlib.util.spec_from_file_location("lhe_weight_canary", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import LHE-weight canary module")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class StandardLheWeightCanaryTests(unittest.TestCase):
    def test_remote_uri(self) -> None:
        self.assertEqual(
            module.remote_uri(
                "root://cmseos.fnal.gov",
                "/store/user/test/file.tar.gz",
            ),
            "root://cmseos.fnal.gov//store/user/test/file.tar.gz",
        )

    def test_uniform_positive_classification(self) -> None:
        classification, candidate = module.classify_weights(
            event_count=3,
            positive=3,
            negative=0,
            zero=0,
            minimum=1.0,
            maximum=1.0,
            sum_weights=3.0,
            sum_squared_weights=3.0,
        )
        self.assertEqual(classification, "uniform_positive")
        self.assertTrue(candidate)

    def test_signed_classification(self) -> None:
        classification, candidate = module.classify_weights(
            event_count=3,
            positive=2,
            negative=1,
            zero=0,
            minimum=-1.0,
            maximum=1.0,
            sum_weights=1.0,
            sum_squared_weights=3.0,
        )
        self.assertEqual(classification, "signed_uniform_magnitude")
        self.assertFalse(candidate)

    def test_canary_process_count(self) -> None:
        self.assertEqual(len(module.CANARY_PROCESSES), 6)
        self.assertIn("vbf_hh4b", module.CANARY_PROCESSES)
        self.assertIn("wwz_zbb", module.CANARY_PROCESSES)

    def test_no_root_or_parquet_reader(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        forbidden = (
            "read_parquet",
            "pyarrow",
            "uproot",
            "ROOT.TFile",
            "pyhepmc",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
