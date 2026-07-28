#!/usr/bin/env python3

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
    / "index_hh4b_representative_bundle_canary.py"
)

spec = importlib.util.spec_from_file_location("bundle_canary", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import bundle canary")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class BundleCanaryTests(unittest.TestCase):
    def test_canaries(self) -> None:
        self.assertEqual(len(module.CANARIES), 6)
        self.assertEqual(
            {row[1] for row in module.CANARIES},
            {
                "ggf_hh4b",
                "vbf_hh4b",
                "ttbar_inclusive",
                "qcd_bbbb_general",
                "qcd_hardqcd",
                "ttz_zbb",
            },
        )

    def test_payload(self) -> None:
        payload, kinds = module.classify("bundle/events.hepmc.gz")
        self.assertTrue(payload)
        self.assertEqual(kinds, [])

    def test_run_card(self) -> None:
        payload, kinds = module.classify("bundle/cards/run_card.dat")
        self.assertFalse(payload)
        self.assertIn("run_card", kinds)

    def test_safe_name(self) -> None:
        self.assertTrue(module.safe_name("bundle/cards/run_card.dat"))
        self.assertFalse(module.safe_name("../../escape.txt"))
        self.assertFalse(module.safe_name("/absolute.txt"))

    def test_no_extraction(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for token in (
            ".extract(",
            ".extractall(",
            "archive.extract",
            "archive.extractall",
            "read_parquet",
            "uproot",
            "ROOT.TFile",
        ):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
