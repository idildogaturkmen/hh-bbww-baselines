#!/usr/bin/env python3
# Targeted tests for card-only HH4b convention freezing.

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
    / "freeze_hh4b_card_only_configuration_conventions.py"
)

spec = importlib.util.spec_from_file_location("card_only_freeze", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import card-only freeze module")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class CardOnlyFreezeTests(unittest.TestCase):
    def test_run_card_is_value_first(self) -> None:
        settings = module.parse_run_card(
            "6500.0 = ebeam1 ! beam energy\n"
            "10000 = nevents\n"
            "tag_1 = run_tag\n"
        )
        self.assertEqual(settings["ebeam1"], ["6500.0"])
        self.assertEqual(settings["nevents"], ["10000"])
        self.assertEqual(settings["run_tag"], ["tag_1"])

    def test_pythia8_is_key_first(self) -> None:
        settings = module.parse_pythia8_card(
            "25:onMode = off\n"
            "Beams:eCM = 13000.\n"
        )
        self.assertEqual(settings["25:onmode"], ["off"])
        self.assertEqual(settings["beams:ecm"], ["13000."])

    def test_pythia6_is_value_first(self) -> None:
        settings = module.parse_pythia6_card(
            "10041 = lhaid\n"
            "1 = MSTP(61)\n"
        )
        self.assertEqual(settings["lhaid"], ["10041"])
        self.assertEqual(settings["mstp_61"], ["1"])

    def test_volatile_fields_removed(self) -> None:
        canonical = module.canonical_run_card(
            {
                "ebeam1": ["6500.0"],
                "nevents": ["10000"],
                "iseed": ["1234"],
            }
        )
        self.assertIn("ebeam1=6500.0", canonical)
        self.assertNotIn("nevents", canonical)
        self.assertNotIn("iseed", canonical)

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
