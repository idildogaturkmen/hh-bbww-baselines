#!/usr/bin/env python3
"""Targeted tests for the HH4b physical-normalization provenance inventory."""

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
    / "inventory_hh4b_physical_normalization_provenance.py"
)

spec = importlib.util.spec_from_file_location("provenance_inventory", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import provenance inventory")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ProvenanceInventoryTests(unittest.TestCase):
    def test_process_contract(self) -> None:
        self.assertEqual(len(module.EXPECTED_BACKGROUND_PROCESSES), 23)
        self.assertEqual(len(module.EXPECTED_SIGNAL_PROCESSES), 2)
        self.assertEqual(
            module.QCD_PROCESSES,
            {
                "qcd_bbbb_general",
                "qcd_bbbb_iht400to600",
                "qcd_hardqcd",
            },
        )

    def test_campaign_from_bundle_locator(self) -> None:
        locator = (
            "/store/user/example/run2_13tev/frozen_v2/bundles/"
            "qcd_campaign_v1/member_bundle.tar.gz"
        )
        self.assertEqual(
            module.campaign_from_locator(locator),
            "qcd_campaign_v1",
        )

    def test_energy_tag(self) -> None:
        self.assertEqual(
            module.energy_tag_from_locators(
                ["/store/example/run2_13tev/bundle.tar.gz"]
            ),
            "run2_13tev",
        )

    def test_qcd_strategy_is_not_direct_mc_primary(self) -> None:
        strategy, _ = module.normalization_strategy(
            "background",
            "qcd_hardqcd",
        )
        self.assertEqual(
            strategy,
            "data_driven_control_region_primary_mc_projection_secondary",
        )

    def test_no_candidate_reader_dependencies(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        forbidden = (
            "read_parquet",
            "pyarrow",
            "uproot",
            "ROOT.TFile",
            "candidate_path).open",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
