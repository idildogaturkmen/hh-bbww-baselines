#!/usr/bin/env python3
"""Targeted tests for normalization-provenance extraction."""

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
    / "extract_hh4b_normalization_provenance.py"
)

spec = importlib.util.spec_from_file_location("provenance_extract", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import provenance extraction module")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ProvenanceExtractionTests(unittest.TestCase):
    def test_safe_member_names(self) -> None:
        self.assertTrue(
            module.safe_member_name("./cards/run_card.dat")
        )
        self.assertFalse(
            module.safe_member_name("../../escape.txt")
        )
        self.assertFalse(
            module.safe_member_name("/absolute/path.txt")
        )

    def test_forbidden_payload_suffixes(self) -> None:
        for name in (
            "events.root",
            "events.parquet",
            "events.hepmc.gz",
            "events.lhe.gz",
            "nested.tar.gz",
        ):
            self.assertTrue(
                any(
                    name.endswith(suffix)
                    for suffix in module.FORBIDDEN_SUFFIXES
                )
            )

    def test_allowed_artifact_contract(self) -> None:
        self.assertIn(
            "generator_banner_or_lhe_init",
            module.ALLOWED_ARTIFACT_TYPES,
        )
        self.assertIn(
            "proc_card",
            module.ALLOWED_ARTIFACT_TYPES,
        )
        self.assertIn(
            "run_card",
            module.ALLOWED_ARTIFACT_TYPES,
        )

    def test_allowlist_size_contract(self) -> None:
        self.assertEqual(module.EXPECTED_CAMPAIGNS, 53)
        self.assertEqual(module.EXPECTED_ALLOWLIST_ROWS, 547)
        self.assertEqual(module.MAX_MEMBER_BYTES, 5 * 1024 * 1024)

    def test_no_candidate_or_event_readers(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        forbidden = (
            "read_parquet",
            "pyarrow",
            "uproot",
            "ROOT.TFile",
            ".extractall(",
            "archive.extractall",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
