#!/usr/bin/env python3
"""Targeted tests for source-aware generator-configuration review."""

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
    / "review_hh4b_generator_configuration.py"
)

spec = importlib.util.spec_from_file_location("configuration_review", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import configuration-review module")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ConfigurationReviewTests(unittest.TestCase):
    def test_run_card_value_first_with_bang_comment(self) -> None:
        self.assertEqual(
            module.parse_assignment(
                "6500.0 = ebeam1 ! beam 1 energy in GeV"
            ),
            ("ebeam1", "6500.0"),
        )

    def test_pythia_key_first(self) -> None:
        self.assertEqual(
            module.parse_assignment("25:onMode = off"),
            ("25:onmode", "off"),
        )

    def test_mg_pythia_value_first(self) -> None:
        self.assertEqual(
            module.parse_assignment("10041 = lhaid"),
            ("lhaid", "10041"),
        )

    def test_source_classification(self) -> None:
        self.assertEqual(
            module.source_kind("./cards/run_card.dat", "run_card"),
            "run_card",
        )
        self.assertEqual(
            module.source_kind(
                "./cards/pythia8_card_default.dat",
                "parton_shower_card",
            ),
            "pythia",
        )

    def test_no_event_payload_readers(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        forbidden = (
            "read_parquet",
            "pyarrow",
            "uproot",
            "ROOT.TFile",
            "pyhepmc",
            "lhe_parser",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
