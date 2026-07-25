#!/usr/bin/env python3
"""Synthetic-only tests for the canonical HH4b BDT-v1 input contract."""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "analysis"))

_MPL_CONFIG = Path(tempfile.mkdtemp(prefix="hh4b_bdt_v1_test_mpl_"))
os.environ["MPLCONFIGDIR"] = str(_MPL_CONFIG)
atexit.register(shutil.rmtree, _MPL_CONFIG, ignore_errors=True)

from hh4b_bdt_v1_common import (  # noqa: E402
    COMMON_FEATURES,
    MASS_AWARE_FEATURES,
    MASS_PLANE_BLIND_FEATURES,
    SOURCE_COLUMNS,
    add_derived_features,
    assign_member_folds,
    build_hierarchical_weights,
    feature_quality_rows,
    validate_feature_names,
)
from scripts.plotting.hh4b_cms_style import (  # noqa: E402
    add_delphes_header,
    apply_cms_style,
    save_png_pdf,
)


class FeatureContractTests(unittest.TestCase):
    def synthetic_source_frame(self) -> pd.DataFrame:
        data = {}
        for offset, column in enumerate(SOURCE_COLUMNS, start=1):
            data[column] = np.asarray(
                [-2.0 * offset, 0.5 * offset, 3.0 * offset],
                dtype=np.float64,
            )
        return pd.DataFrame(data)

    def test_deterministic_derived_feature_construction(self) -> None:
        source = self.synthetic_source_frame()
        first = add_derived_features(source)
        second = add_derived_features(source)
        pd.testing.assert_frame_equal(first, second)
        np.testing.assert_array_equal(
            first["abs_hh_eta"].to_numpy(),
            np.abs(source["hh_eta"].to_numpy()),
        )
        np.testing.assert_array_equal(
            first["abs_h_delta_phi"].to_numpy(),
            np.abs(source["h_delta_phi"].to_numpy()),
        )

    def test_exact_feature_order_and_counts(self) -> None:
        self.assertEqual(len(COMMON_FEATURES), 30)
        self.assertEqual(len(MASS_PLANE_BLIND_FEATURES), 30)
        self.assertEqual(len(MASS_AWARE_FEATURES), 34)
        self.assertEqual(MASS_PLANE_BLIND_FEATURES, COMMON_FEATURES)
        self.assertEqual(
            MASS_AWARE_FEATURES[-4:],
            ("mbb1", "mbb2", "delta_mbb", "r_hh_125_125"),
        )

    def test_forbidden_feature_rejection(self) -> None:
        for forbidden in (
            "event",
            "j1_flavor",
            "j2_raw_index",
            "pairing",
            "source_root",
            "promoted_jet_pt",
            "hh_phi",
            "j1_btag",
            "r_hh",
            "avg_mbb",
        ):
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ValueError):
                    validate_feature_names([COMMON_FEATURES[0], forbidden])

    def test_nonfinite_feature_rejection_by_quality_audit(self) -> None:
        frame = pd.DataFrame({"mhh": [250.0, np.inf, 400.0]})
        rows = feature_quality_rows(frame, ["mhh"])
        self.assertEqual(rows[0]["nonfinite_rows"], 1)
        self.assertEqual(rows[0]["status"], "fail")


class FoldContractTests(unittest.TestCase):
    def synthetic_members(self) -> list[dict[str, object]]:
        members = []
        member_index = 1
        for sample_class, stratum in (
            ("signal", "ggf_hh4b"),
            ("signal", "vbf_hh4b"),
            ("background", "family_a"),
            ("background", "family_b"),
        ):
            for local_index in range(7):
                members.append(
                    {
                        "member_index": member_index,
                        "sample_class": sample_class,
                        "stratum": stratum,
                        "candidate_rows": (local_index + 1) * (member_index % 4),
                    }
                )
                member_index += 1
        return members

    def test_deterministic_member_fold_assignment(self) -> None:
        members = self.synthetic_members()
        first = assign_member_folds(members, n_folds=5)
        second = assign_member_folds(list(reversed(members)), n_folds=5)
        self.assertEqual(first.assignment, second.assignment)
        self.assertEqual(set(first.assignment), {m["member_index"] for m in members})
        self.assertTrue(all(0 <= fold < 5 for fold in first.assignment.values()))

    def test_no_member_leakage_between_folds(self) -> None:
        result = assign_member_folds(self.synthetic_members(), n_folds=5)
        pairs = [(member, fold) for member, fold in result.assignment.items()]
        self.assertEqual(len(pairs), len({member for member, _ in pairs}))


class HierarchicalWeightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.members = [
            {
                "member_index": 1,
                "sample_class": "signal",
                "stratum": "ggf_hh4b",
                "candidate_rows": 2,
            },
            {
                "member_index": 2,
                "sample_class": "signal",
                "stratum": "ggf_hh4b",
                "candidate_rows": 4,
            },
            {
                "member_index": 3,
                "sample_class": "signal",
                "stratum": "vbf_hh4b",
                "candidate_rows": 3,
            },
            {
                "member_index": 4,
                "sample_class": "signal",
                "stratum": "vbf_hh4b",
                "candidate_rows": 1,
            },
            {
                "member_index": 5,
                "sample_class": "background",
                "stratum": "family_a",
                "candidate_rows": 2,
            },
            {
                "member_index": 6,
                "sample_class": "background",
                "stratum": "family_a",
                "candidate_rows": 5,
            },
            {
                "member_index": 7,
                "sample_class": "background",
                "stratum": "family_b",
                "candidate_rows": 3,
            },
            {
                "member_index": 8,
                "sample_class": "background",
                "stratum": "family_b",
                "candidate_rows": 0,
            },
        ]
        self.row_members = np.concatenate(
            [
                np.full(member["candidate_rows"], member["member_index"], dtype=int)
                for member in self.members
                if member["candidate_rows"] > 0
            ]
        )
        self.result = build_hierarchical_weights(
            self.row_members,
            self.members,
            signal_modes=["ggf_hh4b", "vbf_hh4b"],
            background_families=["family_a", "family_b"],
        )

    def aggregate(self, member_indices: set[int]) -> float:
        mask = np.isin(self.row_members, list(member_indices))
        return float(np.sum(self.result.weights[mask]))

    def test_positive_finite_normalized_weights(self) -> None:
        self.assertTrue(np.all(np.isfinite(self.result.weights)))
        self.assertTrue(np.all(self.result.weights > 0.0))
        self.assertAlmostEqual(float(np.mean(self.result.weights)), 1.0, places=12)

    def test_class_aggregate_balancing(self) -> None:
        signal = {1, 2, 3, 4}
        background = {5, 6, 7, 8}
        self.assertAlmostEqual(
            self.aggregate(signal), self.aggregate(background), places=12
        )

    def test_signal_mode_balancing(self) -> None:
        self.assertAlmostEqual(
            self.aggregate({1, 2}), self.aggregate({3, 4}), places=12
        )

    def test_background_family_balancing(self) -> None:
        self.assertAlmostEqual(
            self.aggregate({5, 6}), self.aggregate({7, 8}), places=12
        )

    def test_equal_candidate_bearing_member_totals(self) -> None:
        self.assertAlmostEqual(self.aggregate({1}), self.aggregate({2}), places=12)
        self.assertAlmostEqual(self.aggregate({5}), self.aggregate({6}), places=12)
        zero_member = next(
            row for row in self.result.member_rows if row["member_index"] == 8
        )
        self.assertEqual(zero_member["aggregate_training_weight"], 0.0)
        self.assertEqual(
            zero_member["weight_eligibility"],
            "no_candidate_rows_not_row_weightable",
        )


class CmsStyleTests(unittest.TestCase):
    def test_cms_style_smoke_png_pdf_and_cleanup(self) -> None:
        import matplotlib.pyplot as plt

        metadata = apply_cms_style()
        self.assertTrue(metadata["cms_inspired_style_applied"])
        self.assertFalse(metadata["official_cms_status_claimed"])
        with tempfile.TemporaryDirectory(prefix="hh4b_bdt_v1_plot_") as tmp:
            tmp_path = Path(tmp)
            fig, ax = plt.subplots(figsize=(4.0, 3.0))
            ax.plot([0.0, 1.0], [0.0, 1.0], color="#0072B2")
            ax.set_xlabel(r"$p_{\mathrm{T}}$")
            add_delphes_header(ax, "Synthetic test")
            png, pdf = save_png_pdf(fig, tmp_path / "style_smoke", dpi=300)
            self.assertGreater(png.stat().st_size, 0)
            self.assertGreater(pdf.stat().st_size, 0)
            retained_path = tmp_path
        self.assertFalse(retained_path.exists())


if __name__ == "__main__":
    unittest.main()
