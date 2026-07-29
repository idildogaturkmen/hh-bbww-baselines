#!/usr/bin/env python3
"""Targeted tests for denominator-evidence and source-recovery audit."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts"
    / "analysis"
    / "audit_hh4b_denominator_evidence_and_sources.py"
)

spec = importlib.util.spec_from_file_location("denominator_audit", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import denominator audit module")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class DenominatorEvidenceAuditTests(unittest.TestCase):
    def test_denominator_json_key(self) -> None:
        self.assertEqual(
            module.relevant_json_class(
                "normalization.sum_generator_weights",
                10000.0,
            ),
            "denominator_candidate",
        )

    def test_qcd_json_key(self) -> None:
        self.assertEqual(
            module.relevant_json_class(
                "sampling.pthat_bin_low",
                50,
            ),
            "qcd_sampling_or_overlap_candidate",
        )

    def test_run_card_patterns(self) -> None:
        text = (
            "10000 = nevents ! events\n"
            "average = event_norm\n"
        )
        self.assertEqual(
            module.RUN_CARD_NEVENTS_PATTERN.findall(text),
            ["10000"],
        )
        self.assertEqual(
            [
                value.strip()
                for value in module.RUN_CARD_EVENT_NORM_PATTERN.findall(text)
            ],
            ["average"],
        )

    def test_binary_payload_is_not_searchable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "sample.parquet"
            parquet.write_bytes(b"campaign")
            self.assertFalse(module.is_searchable_file(parquet))

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
