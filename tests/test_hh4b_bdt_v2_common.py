#!/usr/bin/env python3
"""Synthetic-only tests for the CMS-inspired HH4b BDT-v2 contract."""

from __future__ import annotations

import atexit
import math
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

_MPL_CONFIG = Path(tempfile.mkdtemp(prefix="hh4b_bdt_v2_test_mpl_"))
os.environ["MPLCONFIGDIR"] = str(_MPL_CONFIG)
atexit.register(shutil.rmtree, _MPL_CONFIG, ignore_errors=True)

from hh4b_bdt_v1_common import MASS_AWARE_FEATURES  # noqa: E402
from hh4b_bdt_v2_common import (  # noqa: E402
    CATEGORIZED_EXPLICIT_DIJET_MASS_PLANE_BLIND_FEATURES,
    CATEGORIZED_MASS_AWARE_FEATURES,
    CMS_INSPIRED_DERIVED_FEATURES,
    FEATURE_VARIANTS,
    GLOBAL_V1_MASS_AWARE_REFERENCE,
    LorentzVector,
    assign_mhh_category,
    cross_higgs_pair_indices,
    delta_r,
    derive_cms_inspired_features,
    pairwise_delta_r_summary,
    pairwise_delta_rs,
    reconstruct_higgs_pairs,
    require_finite_features,
    rest_frame_cosine,
    validate_feature_names,
    wrapped_delta_phi,
    write_table_bundle,
)
from scripts.plotting.hh4b_cms_style import (  # noqa: E402
    add_delphes_header,
    apply_cms_style,
    save_png_pdf,
)


def vector(pt: float, eta: float, phi: float, mass: float = 10.0) -> LorentzVector:
    return LorentzVector.from_pt_eta_phi_mass(pt, eta, phi, mass)


class AngularFeatureTests(unittest.TestCase):
    def test_wrapped_delta_phi(self) -> None:
        self.assertAlmostEqual(
            wrapped_delta_phi(math.pi - 0.1, -math.pi + 0.1),
            -0.2,
            places=12,
        )
        self.assertAlmostEqual(wrapped_delta_phi(0.3, 0.1), 0.2, places=12)

    def test_all_six_pairwise_delta_r_calculations(self) -> None:
        etas = [0.0, 1.0, -1.0, 0.5]
        phis = [math.pi - 0.1, -math.pi + 0.1, 0.0, 0.5]
        values = pairwise_delta_rs(etas, phis)
        self.assertEqual(
            set(values),
            {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)},
        )
        self.assertAlmostEqual(values[(0, 1)], math.hypot(1.0, 0.2))
        for (first, second), observed in values.items():
            self.assertAlmostEqual(
                observed,
                delta_r(etas[first], phis[first], etas[second], phis[second]),
            )

    def test_min_max_mean_pairwise_delta_r(self) -> None:
        values = pairwise_delta_rs(
            [0.0, 0.2, 1.0, -0.4],
            [0.0, 0.1, 1.2, -0.8],
        )
        minimum, maximum, mean = pairwise_delta_r_summary(values)
        array = np.asarray(list(values.values()))
        self.assertEqual(minimum, float(array.min()))
        self.assertEqual(maximum, float(array.max()))
        self.assertEqual(mean, float(array.mean()))

    def test_cross_higgs_pair_selection(self) -> None:
        self.assertEqual(
            cross_higgs_pair_indices((0, 2), (1, 3)),
            ((0, 1), (0, 3), (1, 2), (2, 3)),
        )
        with self.assertRaises(ValueError):
            cross_higgs_pair_indices((0, 1), (1, 3))


class LorentzConventionTests(unittest.TestCase):
    def test_lorentz_vector_construction_and_invariant_mass(self) -> None:
        item = vector(50.0, 0.7, -1.2, 12.0)
        self.assertAlmostEqual(item.pt, 50.0, places=12)
        self.assertAlmostEqual(item.mass, 12.0, places=10)
        self.assertGreater(item.e, item.momentum)
        signed_roundoff = vector(50.0, 0.7, -1.2, -0.1)
        self.assertAlmostEqual(signed_roundoff.mass, 0.1, places=9)

    def test_lorentz_boost_sends_parent_to_rest(self) -> None:
        parent = vector(120.0, 0.8, 0.5, 300.0)
        rested = parent.boost(parent.beta)
        self.assertAlmostEqual(rested.px, 0.0, places=10)
        self.assertAlmostEqual(rested.py, 0.0, places=10)
        self.assertAlmostEqual(rested.pz, 0.0, places=10)
        self.assertAlmostEqual(rested.e, parent.mass, places=10)

    def test_cos_theta_star_convention_and_range(self) -> None:
        beta = np.asarray([0.0, 0.0, 0.4])
        parent_rest = LorentzVector(300.0, 0.0, 0.0, 0.0)
        child_rest = LorentzVector(160.0, 0.0, 0.0, 100.0)
        parent_lab = parent_rest.boost(-beta)
        child_lab = child_rest.boost(-beta)
        observed = rest_frame_cosine(child_lab, parent_lab)
        self.assertAlmostEqual(observed, 1.0, places=12)
        self.assertGreaterEqual(observed, -1.0)
        self.assertLessEqual(observed, 1.0)

    def test_jet_decay_angle_conventions(self) -> None:
        jets = [
            vector(110.0, 0.2, 0.0),
            vector(70.0, -0.2, math.pi),
            vector(90.0, 0.3, 1.4),
            vector(60.0, -0.3, 1.4 + math.pi),
        ]
        h1_pair, h2_pair = reconstruct_higgs_pairs(jets)
        self.assertEqual(set(h1_pair) | set(h2_pair), {0, 1, 2, 3})
        h1 = jets[h1_pair[0]] + jets[h1_pair[1]]
        h2 = jets[h2_pair[0]] + jets[h2_pair[1]]
        for child, parent in ((jets[h1_pair[0]], h1), (jets[h2_pair[0]], h2)):
            cosine = rest_frame_cosine(child, parent)
            self.assertTrue(math.isfinite(cosine))
            self.assertGreaterEqual(cosine, -1.0 - 1.0e-12)
            self.assertLessEqual(cosine, 1.0 + 1.0e-12)


class CategoryAndFeatureContractTests(unittest.TestCase):
    def test_deterministic_category_assignment_and_exact_boundary(self) -> None:
        self.assertEqual(assign_mhh_category(449.999999), "low_mhh")
        self.assertEqual(assign_mhh_category(450.0), "high_mhh")
        self.assertEqual(assign_mhh_category(900.0), "high_mhh")
        with self.assertRaises(ValueError):
            assign_mhh_category(float("nan"))

    def test_category_exclusivity_and_exhaustiveness(self) -> None:
        values = np.linspace(0.0, 1000.0, 1001)
        assigned = [assign_mhh_category(value) for value in values]
        self.assertEqual(len(assigned), len(values))
        self.assertTrue(set(assigned).issubset({"low_mhh", "high_mhh"}))
        self.assertEqual(sum(value == "low_mhh" for value in assigned), 450)

    def test_exact_feature_order(self) -> None:
        self.assertEqual(GLOBAL_V1_MASS_AWARE_REFERENCE, MASS_AWARE_FEATURES)
        self.assertEqual(len(GLOBAL_V1_MASS_AWARE_REFERENCE), 34)
        self.assertEqual(len(CMS_INSPIRED_DERIVED_FEATURES), 18)
        self.assertEqual(len(CATEGORIZED_MASS_AWARE_FEATURES), 52)
        self.assertEqual(
            CATEGORIZED_MASS_AWARE_FEATURES[-18:],
            CMS_INSPIRED_DERIVED_FEATURES,
        )
        self.assertEqual(
            len(CATEGORIZED_EXPLICIT_DIJET_MASS_PLANE_BLIND_FEATURES),
            48,
        )
        self.assertEqual(
            FEATURE_VARIANTS["low_mhh_cms_inspired_mass_aware"],
            FEATURE_VARIANTS["high_mhh_cms_inspired_mass_aware"],
        )
        self.assertIn(
            "mhh",
            FEATURE_VARIANTS["categorized_explicit_dijet_mass_plane_blind"],
        )

    def test_forbidden_feature_rejection(self) -> None:
        for forbidden in (
            "j1_phi",
            "j1_flavor",
            "j2_btag",
            "member_index",
            "pairing",
            "source_root",
        ):
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ValueError):
                    validate_feature_names(["mhh", forbidden])

    def test_finite_feature_rejection(self) -> None:
        with self.assertRaises(ValueError):
            require_finite_features(
                pd.DataFrame({"mhh": [300.0, np.inf]}),
                ["mhh"],
            )
        with self.assertRaises(ValueError):
            require_finite_features(
                pd.DataFrame({"mhh": [300.0, np.nan]}),
                ["mhh"],
            )

    def test_derived_feature_physical_ranges(self) -> None:
        row = {}
        for index, name in enumerate(MASS_AWARE_FEATURES, start=1):
            source = {
                "abs_hh_eta": "hh_eta",
                "abs_h1_eta": "h1_eta",
                "abs_h2_eta": "h2_eta",
                "abs_h_delta_eta": "h_delta_eta",
                "abs_h_delta_phi": "h_delta_phi",
                "abs_j1_eta": "j1_eta",
                "abs_j2_eta": "j2_eta",
                "abs_j3_eta": "j3_eta",
                "abs_j4_eta": "j4_eta",
            }.get(name, name)
            row.setdefault(source, float(index + 1))
        jet_values = (
            (120.0, 0.3, 0.1, 12.0),
            (95.0, -0.2, 2.8, 11.0),
            (80.0, 0.7, -1.2, 9.0),
            (55.0, -0.8, 1.4, 8.0),
        )
        for jet, (pt, eta, phi, mass) in enumerate(jet_values, start=1):
            row[f"j{jet}_pt"] = pt
            row[f"j{jet}_eta"] = eta
            row[f"j{jet}_phi"] = phi
            row[f"j{jet}_mass"] = mass
        derived = derive_cms_inspired_features(pd.DataFrame([row]))
        for feature in CMS_INSPIRED_DERIVED_FEATURES:
            self.assertTrue(math.isfinite(float(derived.loc[0, feature])))
        self.assertGreaterEqual(float(derived.loc[0, "min_candidate_pair_dr"]), 0.0)
        self.assertLessEqual(
            float(derived.loc[0, "min_candidate_pair_dr"]),
            float(derived.loc[0, "max_candidate_pair_dr"]),
        )
        for feature in (
            "cos_theta_star_h1_in_hh_rest",
            "cos_theta_j1_in_h1_rest",
            "cos_theta_j_h2_leading_in_h2_rest",
        ):
            self.assertGreaterEqual(float(derived.loc[0, feature]), -1.0 - 1e-12)
            self.assertLessEqual(float(derived.loc[0, feature]), 1.0 + 1e-12)


class OutputUtilityTests(unittest.TestCase):
    def test_tsv_markdown_latex_table_writing_and_cleanup(self) -> None:
        retained: Path
        with tempfile.TemporaryDirectory(prefix="hh4b_bdt_v2_table_") as tmp:
            directory = Path(tmp)
            paths = write_table_bundle(
                directory,
                "synthetic",
                [
                    {
                        "name": "a_b & c",
                        "value": 1.25,
                        "latex_label": r"$m_{HH}$",
                    }
                ],
                ("name", "value", "latex_label"),
                caption="Synthetic & escaped",
                label="tab:synthetic_test",
                latex_raw_fields=("latex_label",),
            )
            for path in paths:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
            latex = paths[2].read_text(encoding="utf-8")
            self.assertIn(r"\toprule", latex)
            self.assertIn(r"\bottomrule", latex)
            self.assertIn(r"a\_b \& c", latex)
            self.assertIn(r"$m_{HH}$", latex)
            retained = directory
        self.assertFalse(retained.exists())

    def test_cms_style_png_pdf_smoke_and_cleanup(self) -> None:
        import matplotlib.pyplot as plt

        metadata = apply_cms_style()
        self.assertTrue(metadata["cms_inspired_style_applied"])
        self.assertFalse(metadata["official_cms_status_claimed"])
        with tempfile.TemporaryDirectory(prefix="hh4b_bdt_v2_plot_") as tmp:
            directory = Path(tmp)
            fig, ax = plt.subplots(figsize=(4.0, 3.0))
            ax.plot([0.0, 1.0], [1.0, 0.0], color="#0072B2")
            ax.set_xlabel(r"$m_{HH}$")
            add_delphes_header(ax, "CMS-inspired categorization")
            png, pdf = save_png_pdf(fig, directory / "style_smoke", dpi=300)
            self.assertGreater(png.stat().st_size, 0)
            self.assertGreater(pdf.stat().st_size, 0)
            retained = directory
        self.assertFalse(retained.exists())


if __name__ == "__main__":
    unittest.main()
