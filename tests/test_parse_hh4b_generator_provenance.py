#!/usr/bin/env python3
"""Targeted tests for the HH4b generator-provenance parser."""

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
    / "parse_hh4b_generator_provenance.py"
)

spec = importlib.util.spec_from_file_location("generator_parse", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import generator-provenance parser")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class GeneratorProvenanceParseTests(unittest.TestCase):
    def test_run_card_value_first(self) -> None:
        self.assertEqual(
            module.parse_run_card_line("6500.0 = ebeam1"),
            ("ebeam1", "6500.0"),
        )

    def test_run_card_key_first(self) -> None:
        self.assertEqual(
            module.parse_run_card_line("nevents = 10000"),
            ("nevents", "10000"),
        )

    def test_cross_section_conversion(self) -> None:
        self.assertAlmostEqual(
            module.convert_xsec_to_pb(1000.0, "fb"),
            1.0,
        )
        self.assertAlmostEqual(
            module.convert_xsec_to_pb(1.0, "nb"),
            1000.0,
        )

    def test_process_definition(self) -> None:
        self.assertTrue(
            module.process_definition_line("generate p p > t t~")
        )
        self.assertTrue(
            module.process_definition_line("add process p p > t t~ j")
        )

    def test_no_event_payload_reader(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        forbidden = (
            "read_parquet",
            "pyarrow",
            "uproot",
            "ROOT.TFile",
            "import pyhepmc",
            "from pyhepmc",
            "pyhepmc.open",
            "lhe_parser",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
