#!/usr/bin/env python3
"""Targeted tests for the ggF generation/decay/denominator freezer."""

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
    / "freeze_hh4b_ggf_generation_decay_denominators.py"
)

spec = importlib.util.spec_from_file_location("ggf_freezer", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not import ggF freezer")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class GgfFreezeTests(unittest.TestCase):
    def test_tag_parser(self) -> None:
        campaign, shard, seed = module.parse_tag(
            "HH4b_ggf_ggf_hh4b_ml_ext80k_frozen_v2_20260721_v1_"
            "shard020_seed86020"
        )
        self.assertEqual(
            (
                campaign,
                shard,
                seed,
            ),
            (
                "ggf_hh4b_ml_ext80k_frozen_v2_20260721_v1",
                20,
                86020,
            ),
        )

    def test_active_card_lines_strip_comments(self) -> None:
        observed = module.active_card_lines(
            "import model /path/heft # comment\n"
            "\n"
            "generate p p > h h, (h > b b~), (h > b b~)\n"
        )
        self.assertEqual(
            observed,
            [
                "import model /path/heft",
                "generate p p > h h, (h > b b~), (h > b b~)",
            ],
        )

    def test_run_card_assignment(self) -> None:
        text = " average = event_norm ! normalization\n"
        self.assertEqual(
            module.parse_assignment(text, "event_norm"),
            "average",
        )

    def test_expected_shard_partition(self) -> None:
        first = module.GGF_CAMPAIGNS[
            "ggf_hh4b_ml_ext20k_frozen_v2_20260716"
        ]["expected_shards"]
        second = module.GGF_CAMPAIGNS[
            "ggf_hh4b_ml_ext80k_frozen_v2_20260721_v1"
        ]["expected_shards"]
        self.assertEqual(first | second, set(range(100)))
        self.assertFalse(first & second)

    def test_no_event_payload_reader(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for token in (
            "uproot",
            "pyhepmc",
            "read_parquet",
            "pyarrow",
            "lhe_parser",
        ):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
