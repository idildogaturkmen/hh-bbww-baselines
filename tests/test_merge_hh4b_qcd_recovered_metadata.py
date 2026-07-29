#!/usr/bin/env python3
"""Targeted tests for the hard-QCD recovered-metadata merger."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "merge_hh4b_qcd_recovered_metadata.py"
spec = importlib.util.spec_from_file_location("qcd_metadata_merge", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import QCD metadata merger")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class QcdMetadataMergeTests(unittest.TestCase):
    def test_canonical_field(self) -> None:
        self.assertEqual(module.canonical_field("generator.sigma_gen_pb"), "sigma_gen_pb")

    def test_open_bound_aliases(self) -> None:
        self.assertEqual(module.parse_bound("Inf"), "Inf")
        self.assertEqual(module.parse_bound("Infinity"), "Inf")
        self.assertEqual(module.parse_bound("null"), "Inf")
        self.assertEqual(module.parse_bound("1000"), 1000)

    def test_equivalent_numeric_duplicates(self) -> None:
        rows = [
            {
                "process_or_mode": "qcd_hardqcd",
                "source_tag": "tag",
                "field": "sum_event_weights",
                "value": "10000.0",
            },
            {
                "process_or_mode": "qcd_hardqcd",
                "source_tag": "tag",
                "field": "generator.sum_event_weights",
                "value": "10000",
            },
        ]
        _, resolved, conflicts = module.collect_values(rows)
        self.assertEqual(conflicts, {})
        self.assertEqual(float(resolved["tag"]["sum_event_weights"]), 10000.0)

    def test_conflicting_duplicates_fail_closed(self) -> None:
        rows = [
            {
                "process_or_mode": "qcd_hardqcd",
                "source_tag": "tag",
                "field": "sigma_gen_pb",
                "value": "10",
            },
            {
                "process_or_mode": "qcd_hardqcd",
                "source_tag": "tag",
                "field": "generator.sigma_gen_pb",
                "value": "11",
            },
        ]
        _, _, conflicts = module.collect_values(rows)
        self.assertEqual(conflicts, {"tag": ["sigma_gen_pb"]})

    def test_recovered_rows_supersede_partial_base_required_fields(self) -> None:
        base = [
            {
                "process_or_mode": "qcd_hardqcd",
                "source_tag": "recovered",
                "field": "pthat_max_GeV",
                "value": "0",
            },
            {
                "process_or_mode": "qcd_hardqcd",
                "source_tag": "other",
                "field": "pthat_max_GeV",
                "value": "Inf",
            },
        ]
        retained, superseded = module.split_base_rows_for_recovered_tags(
            base,
            {"recovered"},
        )
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]["source_tag"], "other")
        self.assertEqual(len(superseded), 1)
        self.assertEqual(superseded[0]["canonical_field"], "pthat_max_GeV")

    def test_no_event_payload_reader(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for token in ("uproot", "pyhepmc", "read_parquet", "pyarrow", "tarfile.open", "xrdcp"):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
