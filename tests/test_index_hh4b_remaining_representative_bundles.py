#!/usr/bin/env python3
"""Targeted tests for full HH4b representative-bundle indexing."""

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
    / "index_hh4b_remaining_representative_bundles.py"
)
BASE = (
    REPO
    / "scripts"
    / "analysis"
    / "index_hh4b_representative_bundle_canary.py"
)

spec = importlib.util.spec_from_file_location("full_bundle_index", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import full bundle-index module")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

base = module.load_base(BASE)


class FullBundleIndexTests(unittest.TestCase):
    def test_payload_classification(self) -> None:
        payload, nested, kinds = module.classify(
            base,
            "./source/events.lhe.gz",
        )
        self.assertTrue(payload)
        self.assertFalse(nested)
        self.assertEqual(kinds, [])

    def test_nested_archive_not_payload(self) -> None:
        payload, nested, _ = module.classify(
            base,
            "./products/reconstruction.tar.gz",
        )
        self.assertFalse(payload)
        self.assertTrue(nested)

    def test_provenance_json_classification(self) -> None:
        payload, nested, kinds = module.classify(
            base,
            "./metadata/sample_provenance.json",
        )
        self.assertFalse(payload)
        self.assertFalse(nested)
        self.assertIn("generation_provenance_metadata", kinds)

    def test_run_card_classification(self) -> None:
        payload, nested, kinds = module.classify(
            base,
            "./cards/run_card.dat",
        )
        self.assertFalse(payload)
        self.assertFalse(nested)
        self.assertIn("run_card", kinds)

    def test_no_extraction_calls(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        forbidden = (
            ".extract(",
            ".extractall(",
            "archive.extract",
            "archive.extractall",
            "read_parquet",
            "pyarrow",
            "uproot",
            "ROOT.TFile",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
