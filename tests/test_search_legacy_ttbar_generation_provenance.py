#!/usr/bin/env python3
"""Targeted tests for the legacy ttbar provenance search."""

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
    / "search_legacy_ttbar_generation_provenance.py"
)

spec = importlib.util.spec_from_file_location("legacy_ttbar_search", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import legacy ttbar search")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class LegacyTtbarSearchTests(unittest.TestCase):
    def test_anchor_contract(self) -> None:
        self.assertIn("ttbar_100k_shard000", module.ANCHORS)
        self.assertIn("ttbar_100k", module.ANCHORS)

    def test_payload_protection(self) -> None:
        self.assertTrue(
            module.is_payload(
                Path("ttbar_100k_shard000_hh4b_candidates.parquet")
            )
        )
        self.assertTrue(module.is_payload(Path("events.root")))
        self.assertTrue(module.is_payload(Path("events.lhe.gz")))

    def test_artifact_classification(self) -> None:
        self.assertIn(
            "proc_card",
            module.artifact_types(Path("proc_card_mg5.dat")),
        )
        self.assertIn(
            "run_card",
            module.artifact_types(Path("run_card.dat")),
        )
        self.assertIn(
            "parton_shower_card",
            module.artifact_types(Path("pythia8_card.dat")),
        )

    def test_history_and_credentials_excluded(self) -> None:
        self.assertIn(".bash_history", module.EXCLUDED_FILE_NAMES)
        self.assertIn(".ssh", module.EXCLUDED_DIRECTORY_NAMES)

    def test_no_event_readers(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        forbidden = (
            "read_parquet",
            "pyarrow",
            "uproot",
            "ROOT.TFile",
            "tarfile.open",
            "gzip.open",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
