#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "analysis" / "make_hh4b_expanded_cut_baseline_report.py"
SPEC = importlib.util.spec_from_file_location("expanded_cut_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExpandedCutReportTests(unittest.TestCase):
    def test_working_point_order(self) -> None:
        self.assertEqual(
            MODULE.WORKING_POINT_ORDER,
            (
                "cms_reference_rhh30",
                "higher_purity_rhh31p5",
                "optimized_nominal_rhh34",
                "higher_efficiency_rhh35p5",
            ),
        )

    def test_aggregate_regions(self) -> None:
        rows = []
        for region, signal, background in (
            ("cms_signal_region", 5103, 8126),
            ("cms_control_region", 2599, 11284),
            ("outside_geometry", 1635, 24415),
        ):
            rows.extend(
                [
                    {
                        "region": region,
                        "population_level": "class",
                        "population": "signal",
                        "selected_rows": str(signal),
                        "weighted_fraction": "0.3",
                    },
                    {
                        "region": region,
                        "population_level": "class",
                        "population": "background",
                        "selected_rows": str(background),
                        "weighted_fraction": "0.2",
                    },
                ]
            )
        aggregated = MODULE.aggregate_region_rows(rows)
        self.assertEqual(sum(row["total_rows"] for row in aggregated), 53162)
        self.assertEqual(aggregated[0]["signal_rows"], 5103)
        self.assertEqual(aggregated[0]["background_rows"], 8126)

    def test_latex_labels(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn(r"R_{HH}^{125,120}", source)
        self.assertIn(r"\epsilon_{S}", source)
        self.assertIn(r"\epsilon_{B}", source)

    def test_cms_style_helpers_used(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("apply_cms_style()", source)
        self.assertIn("add_delphes_header", source)
        self.assertIn("save_png_pdf", source)

    def test_no_candidate_or_validation_reader(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("read_parquet", source)
        self.assertNotIn("development_validation_cache", source)
        self.assertIn('"validation_candidate_files_opened": 0', source)
        self.assertIn('"evaluation_candidate_files_opened": 0', source)


if __name__ == "__main__":
    unittest.main()
