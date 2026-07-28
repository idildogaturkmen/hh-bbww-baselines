#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts"
    / "analysis"
    / "run_hh4b_expanded_cut_baseline.py"
)

SPEC = importlib.util.spec_from_file_location(
    "expanded_cut_runner",
    RUNNER_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to import expanded cut runner")

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExpandedCutBaselineTests(unittest.TestCase):
    def test_rhh_formula(self) -> None:
        values = MODULE.cms_rhh_125_120(
            [125.0, 128.0],
            [120.0, 124.0],
        )
        self.assertAlmostEqual(
            float(values[0]),
            0.0,
        )
        self.assertAlmostEqual(
            float(values[1]),
            5.0,
        )

    def test_region_boundaries(self) -> None:
        labels = MODULE.cms_region_labels(
            [
                0.0,
                29.999,
                30.0,
                54.999,
                55.0,
            ]
        )
        self.assertEqual(
            labels.tolist(),
            [
                "cms_signal_region",
                "cms_signal_region",
                "cms_control_region",
                "cms_control_region",
                "outside_geometry",
            ],
        )

    def test_weighted_metrics(self) -> None:
        metrics = MODULE.selection_metrics(
            selected=[
                True,
                False,
                True,
                False,
            ],
            labels=[1, 1, 0, 0],
            weights=[1.0, 3.0, 2.0, 2.0],
            signal_modes=[
                "ggf_hh4b",
                "vbf_hh4b",
                "",
                "",
            ],
            background_families=[
                "",
                "",
                "qcd_multijet",
                "qcd_multijet",
            ],
        )
        self.assertAlmostEqual(
            metrics[
                "weighted_signal_efficiency"
            ],
            0.25,
        )
        self.assertAlmostEqual(
            metrics[
                "weighted_background_efficiency"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            metrics[
                "weighted_inverse_background_efficiency"
            ],
            2.0,
        )

    def test_protocol_v2(self) -> None:
        path = (
            ROOT
            / "configs"
            / "baselines"
            / "hh4b_expanded_cut_bdt_protocol_v2.json"
        )
        protocol = json.loads(
            path.read_text(encoding="utf-8")
        )
        MODULE.validate_protocol(protocol)
        self.assertEqual(
            protocol["cut_baseline"]["variable"],
            "r_hh_125_120",
        )

    def test_protocol_v1_preserved(self) -> None:
        v1_path = (
            ROOT
            / "configs"
            / "baselines"
            / "hh4b_expanded_cut_bdt_protocol_v1.json"
        )
        v2_path = (
            ROOT
            / "configs"
            / "baselines"
            / "hh4b_expanded_cut_bdt_protocol_v2.json"
        )
        v1 = json.loads(
            v1_path.read_text(encoding="utf-8")
        )
        v2 = json.loads(
            v2_path.read_text(encoding="utf-8")
        )
        self.assertEqual(
            v1["cut_baseline"]["variable"],
            "r_hh_125_125",
        )
        self.assertEqual(
            v2["supersedes"][
                "legacy_cut_definition"
            ]["variable"],
            "r_hh_125_125",
        )
        self.assertEqual(
            v1["models"],
            v2["models"],
        )

    def test_runner_mentions_train_cache_only(self) -> None:
        source = RUNNER_PATH.read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "development_train_cache",
            source,
        )
        self.assertNotIn(
            '["development_validation_cache"]',
            source,
        )


if __name__ == "__main__":
    unittest.main()
