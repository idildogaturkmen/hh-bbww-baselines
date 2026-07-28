#!/usr/bin/env python3
"""Targeted tests for the generator-artifact location audit."""

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
    / "audit_hh4b_generator_artifact_locations.py"
)

spec = importlib.util.spec_from_file_location("artifact_location_audit", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import artifact-location audit")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ArtifactLocationAuditTests(unittest.TestCase):
    def test_card_classification(self) -> None:
        self.assertIn(
            "proc_card",
            module.classify_path("/tmp/proc_card.dat"),
        )
        self.assertIn(
            "run_card",
            module.classify_path("/tmp/run_card.dat"),
        )
        self.assertIn(
            "param_card",
            module.classify_path("/tmp/param_card.dat"),
        )

    def test_payload_classification(self) -> None:
        self.assertIn(
            "event_payload",
            module.classify_path("/tmp/events.root"),
        )
        self.assertIn(
            "bundle_archive",
            module.classify_path("/tmp/member_bundle.tar.gz"),
        )

    def test_qcd_metadata_classification(self) -> None:
        types = module.classify_path(
            "/tmp/qcd_importance_statistical_decision.json"
        )
        self.assertIn("importance_sampling_record", types)

    def test_required_artifact_contract(self) -> None:
        self.assertEqual(len(module.REQUIRED_ARTIFACT_TYPES), 8)
        self.assertEqual(len(module.CONDITIONAL_ARTIFACT_TYPES), 2)

    def test_no_payload_transfer_or_reader(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        forbidden = (
            "xrdcp",
            "tarfile.open",
            "read_parquet",
            "pyarrow",
            "uproot",
            "ROOT.TFile",
            "gzip.open",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
