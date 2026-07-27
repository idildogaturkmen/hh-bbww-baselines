#!/usr/bin/env python3
"""Synthetic tests for the one-time HH4b BDT validation-selection runner."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import inspect
import json
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

import fit_hh4b_bdt_candidates_and_select_validation as gate
from scripts.plotting.hh4b_cms_style import (
    add_delphes_header,
    apply_cms_style,
    save_png_pdf,
)


class ValidationSelectionTests(unittest.TestCase):
    def test_relative_config_path_resolution(self) -> None:
        relative = Path("configs/baselines/hh4b_bdt_validation_selection_v1.yaml")
        self.assertEqual(
            gate.resolve_config_path(relative),
            gate.CONFIG_DEFAULT,
        )
        with self.assertRaises(ValueError):
            gate.resolve_config_path(Path("/tmp/outside-repository.yaml"))

    def test_full_train_model_configuration_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            global_path = root / "global.tsv"
            category_path = root / "category.tsv"
            parameter_header = "\t".join(
                ("variant", "trial_id", *gate.PARAMETER_NAMES)
            )
            values = "\t".join(
                ("400", "3", "0.05", "3", "0.9", "0.9", "5", "0.1", "0")
            )
            global_path.write_text(
                parameter_header
                + "\n"
                + f"mass_aware\t18\t{values}\n"
                + f"explicit_dijet_mass_plane_blind\t18\t{values}\n",
                encoding="utf-8",
            )
            category_path.write_text(
                "\t".join(("category", "trial_id", *gate.PARAMETER_NAMES))
                + "\n"
                + f"low_mhh\t19\t{values}\n"
                + f"high_mhh\t18\t{values}\n",
                encoding="utf-8",
            )
            config = {
                "models": {
                    "global_selected_hyperparameters": str(global_path),
                    "categorized_selected_hyperparameters": str(category_path),
                    "estimator_fixed_parameters": {
                        "objective": "binary:logistic",
                        "tree_method": "hist",
                        "n_jobs": 8,
                    },
                    "global_random_state": 1,
                    "categorized_random_state": 2,
                    "fitted_models": [
                        gate.GLOBAL_MASS,
                        gate.GLOBAL_BLIND,
                        gate.CATEGORIZED_LOW,
                        gate.CATEGORIZED_HIGH,
                    ],
                }
            }
            specifications = gate.load_frozen_model_config(config, root=root)
            self.assertEqual(list(specifications), config["models"]["fitted_models"])
            self.assertEqual(len(specifications[gate.GLOBAL_MASS]["features"]), 34)
            self.assertEqual(len(specifications[gate.GLOBAL_BLIND]["features"]), 30)
            self.assertEqual(len(specifications[gate.CATEGORIZED_LOW]["features"]), 52)
            self.assertEqual(
                specifications[gate.CATEGORIZED_LOW]["parameters"]["n_estimators"],
                400,
            )

    def test_frozen_feature_order_enforcement(self) -> None:
        frame = pd.DataFrame({"second": [2.0], "first": [1.0]})
        matrix = gate.matrix_for_features(frame, ("first", "second"))
        np.testing.assert_array_equal(matrix, np.asarray([[1.0, 2.0]]))
        with self.assertRaises(ValueError):
            gate.matrix_for_features(frame, ("first", "first"))
        with self.assertRaises(ValueError):
            gate.matrix_for_features(frame, ("missing",))

    def test_train_weight_handling(self) -> None:
        normalized = gate.normalize_train_weights([0.2, 0.4, 1.4])
        self.assertAlmostEqual(float(np.mean(normalized)), 1.0, places=14)
        with self.assertRaises(ValueError):
            gate.normalize_train_weights([1.0, 0.0])

        target = np.asarray([1, 1, 1, 1, 0, 0, 0, 0])
        member = np.arange(8)
        mode = np.asarray(["ggf", "ggf", "vbf", "vbf", "", "", "", ""])
        family = np.asarray(["", "", "", "", "qcd", "qcd", "top", "top"])
        selected = np.ones(8, dtype=bool)
        weights = gate.category_local_train_weights(
            target, member, mode, family, selected
        )
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=14)
        self.assertAlmostEqual(float(np.sum(weights[target == 1])), 4.0)
        self.assertAlmostEqual(float(np.sum(weights[target == 0])), 4.0)

    def test_validation_development_weight_hierarchy(self) -> None:
        target = np.asarray([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        member = np.asarray([1, 1, 2, 3, 4, 4, 5, 6, 7, 7])
        mode = np.asarray(["ggf", "ggf", "ggf", "vbf", "", "", "", "", "", ""])
        family = np.asarray(
            ["", "", "", "", "qcd", "qcd", "qcd", "top", "top", "top"]
        )
        weights, audit = gate.build_validation_development_weights(
            target, member, mode, family
        )
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=14)
        total = float(np.sum(weights))
        self.assertAlmostEqual(float(np.sum(weights[target == 1])) / total, 0.5)
        self.assertAlmostEqual(float(np.sum(weights[target == 0])) / total, 0.5)
        self.assertAlmostEqual(float(np.sum(weights[mode == "ggf"])) / total, 0.25)
        self.assertAlmostEqual(float(np.sum(weights[family == "qcd"])) / total, 0.25)
        self.assertTrue(all(row["status"] == "pass" for row in audit))

    def test_category_and_fixed_threshold_application(self) -> None:
        score = np.asarray([0.2, 0.7, 0.6, 0.9])
        category = np.asarray(["low_mhh", "low_mhh", "high_mhh", "high_mhh"])
        selected = gate.apply_fixed_threshold(
            score,
            categories=category,
            category_thresholds={"low_mhh": 0.5, "high_mhh": 0.8},
        )
        np.testing.assert_array_equal(selected, [False, True, False, True])
        np.testing.assert_array_equal(
            gate.apply_fixed_threshold(score, global_threshold=0.6),
            [False, True, True, True],
        )
        with self.assertRaises(ValueError):
            gate.apply_fixed_threshold(
                score,
                categories=["test"] * 4,
                category_thresholds={"low_mhh": 0.5, "high_mhh": 0.8},
            )

    def test_equalized_global_threshold_construction(self) -> None:
        score = np.asarray([0.9, 0.8, 0.7, 0.6, 0.4, 0.2])
        target = np.asarray([1, 1, 1, 1, 0, 0])
        thresholds, selected = gate.equalized_threshold_selection(
            score, target, np.ones(6), 0.5
        )
        self.assertEqual(thresholds, {"global": 0.8})
        np.testing.assert_array_equal(selected, score >= 0.8)

    def test_equalized_category_threshold_construction(self) -> None:
        score = np.asarray([0.9, 0.7, 0.8, 0.6, 0.5, 0.4])
        target = np.asarray([1, 1, 1, 1, 0, 0])
        category = np.asarray(
            ["low_mhh", "low_mhh", "high_mhh", "high_mhh", "low_mhh", "high_mhh"]
        )
        thresholds, selected = gate.equalized_threshold_selection(
            score, target, np.ones(6), 0.5, category
        )
        self.assertEqual(thresholds, {"low_mhh": 0.9, "high_mhh": 0.8})
        np.testing.assert_array_equal(
            selected, [True, False, True, False, False, False]
        )

    def test_paired_member_bootstrap_reproducibility(self) -> None:
        members = np.repeat(np.arange(5), 4)
        target = np.tile([1, 1, 0, 0], 5)
        categories = np.tile(
            ["low_mhh", "high_mhh", "low_mhh", "high_mhh"], 5
        )
        weights = np.ones(20)
        global_scores = np.linspace(0.05, 0.95, 20)
        categorized_scores = global_scores[::-1]
        first = gate.paired_member_bootstrap_primary(
            members,
            target,
            categories,
            weights,
            global_scores,
            categorized_scores,
            target_efficiency=0.5,
            seed=17,
            replicates=25,
        )
        second = gate.paired_member_bootstrap_primary(
            members,
            target,
            categories,
            weights,
            global_scores,
            categorized_scores,
            target_efficiency=0.5,
            seed=17,
            replicates=25,
        )
        np.testing.assert_equal(first, second)

    def test_bootstrap_percentile_intervals(self) -> None:
        interval = gate.percentile_interval(np.arange(100, dtype=float))
        self.assertEqual(interval["replicates_defined"], 100)
        self.assertAlmostEqual(interval["median"], 49.5)
        self.assertLess(interval["percentile_2_5"], interval["percentile_16"])
        self.assertLess(interval["percentile_84"], interval["percentile_97_5"])

    def test_model_selection_rule_pass_and_fail(self) -> None:
        selected, rows, _ = gate.final_model_selection(
            global_background_efficiency=0.20,
            categorized_background_efficiency=0.15,
            bootstrap_fraction_favoring_v2=0.90,
            categorized_ggf_efficiency=0.59,
            categorized_vbf_efficiency=0.58,
            target_signal_efficiency=0.585957314769,
            integrity_failures=0,
        )
        self.assertEqual(selected, gate.CATEGORIZED)
        self.assertTrue(all(row["passed"] for row in rows))
        selected, rows, _ = gate.final_model_selection(
            global_background_efficiency=0.20,
            categorized_background_efficiency=0.19,
            bootstrap_fraction_favoring_v2=0.50,
            categorized_ggf_efficiency=0.59,
            categorized_vbf_efficiency=0.58,
            target_signal_efficiency=0.585957314769,
            integrity_failures=0,
        )
        self.assertEqual(selected, gate.GLOBAL_MASS)
        self.assertFalse(rows[2]["passed"])

    def test_no_manual_selection_override(self) -> None:
        parameters = inspect.signature(gate.final_model_selection).parameters
        self.assertNotIn("override", parameters)
        self.assertNotIn("selected_model", parameters)

    def test_oof_and_category_integrity_helpers(self) -> None:
        clean = gate.check_prediction_integrity(
            [0, 1, 2], [0, 1, 2], ["low", "high", "low"], ["low", "high", "low"]
        )
        self.assertEqual(sum(clean.values()), 0)
        failed = gate.check_prediction_integrity([0, 1], [0, 0])
        self.assertEqual(failed["missing_predictions"], 1)
        self.assertEqual(failed["duplicate_predictions"], 1)

    def test_test_access_rejection(self) -> None:
        with self.assertRaises(PermissionError):
            gate.reject_test_candidate_access(
                {"dataset_split": "test", "test_member": "true"}
            )
        gate.reject_test_candidate_access(
            {"dataset_split": "validation", "test_member": "false"}
        )

    def test_pre_recovery_freeze_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.json"
            prediction = root / "prediction.parquet"
            bootstrap = root / "bootstrap.parquet"
            model.write_text("model", encoding="utf-8")
            prediction.write_text("prediction", encoding="utf-8")
            bootstrap.write_text("bootstrap", encoding="utf-8")
            freeze = {
                "audit_status": "immutable_pre_recovery_freeze",
                "source_commit": "ee051137f04f73d5ecb2438b58b22cbdba68363f",
                "recovery_read_started": False,
                "model_artifacts": {str(model): gate.sha256_file(model)},
                "prediction_artifacts": {
                    str(prediction): gate.sha256_file(prediction)
                },
                "bootstrap_artifacts": {
                    str(bootstrap): gate.sha256_file(bootstrap)
                },
            }
            (root / "validation_pre_recovery_freeze.json").write_text(
                json.dumps(freeze), encoding="utf-8"
            )
            _, digest, rows = gate.verify_pre_recovery_freeze(root)
            self.assertEqual(len(digest), 64)
            self.assertTrue(all(row["passed"] for row in rows))
            model.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                gate.verify_pre_recovery_freeze(root)

    def test_recovery_read_scope_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.parquet"
            pd.DataFrame(
                {
                    "event": [10, 11],
                    "r_hh_125_125": [20.0, 40.0],
                    "mhh": [440.0, 460.0],
                    "prohibited_extra": [1.0, 2.0],
                }
            ).to_parquet(candidate, index=False)
            validation = {
                "validation_manifest": [
                    {
                        "member_index": "1",
                        "sample_class": "signal",
                        "process_or_mode": "ggf_hh4b",
                        "candidate_rows": "2",
                        "local_path": str(candidate),
                        "candidate_sha256": "synthetic",
                        "dataset_split": "validation",
                        "test_member": "false",
                    }
                ]
            }
            recovered, audit, counters = gate.recover_validation_columns_once(
                validation,
                root,
                expected_files=1,
                expected_rows=2,
            )
            self.assertEqual(
                list(recovered.columns),
                ["member_index", "event", "r_hh_125_125", "mhh"],
            )
            self.assertEqual(audit[0]["columns_requested"], "event,r_hh_125_125,mhh")
            self.assertEqual(counters["recovery_validation_files_opened"], 1)
            candidate.unlink()
            cached, _, cached_counters = gate.recover_validation_columns_once(
                validation,
                root,
                expected_files=1,
                expected_rows=2,
            )
            pd.testing.assert_frame_equal(recovered, cached)
            self.assertEqual(cached_counters["missing_recovered_rows"], 0)

    def test_recovered_join_and_category_invariants(self) -> None:
        validation = {
            "member_index": np.asarray([1, 1]),
            "event": np.asarray([10, 11]),
            "category": np.asarray(["low_mhh", "high_mhh"], dtype=object),
        }
        recovered = pd.DataFrame(
            {
                "member_index": [1, 1],
                "event": [10, 11],
                "r_hh_125_125": [20.0, 40.0],
                "mhh": [449.9, 450.0],
            }
        )
        counters = gate.align_recovered_columns(validation, recovered)
        self.assertEqual(sum(counters.values()), 0)
        np.testing.assert_array_equal(
            validation["features"]["r_hh_125_125"], [20.0, 40.0]
        )
        recovered.loc[1, "mhh"] = 449.0
        with self.assertRaises(ValueError):
            gate.align_recovered_columns(validation, recovered)

    def test_markdown_latex_table_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = gate.write_table_bundle(
                Path(temporary),
                "synthetic",
                [{"method": "global_v1", "value": 0.25}],
                ("method", "value"),
                caption="Synthetic physics table",
                label="tab:synthetic",
            )
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertIn("| method | value |", paths[1].read_text(encoding="utf-8"))
            latex = paths[2].read_text(encoding="utf-8")
            self.assertIn(r"\toprule", latex)
            self.assertIn(r"\caption{", latex)

    def test_cms_style_png_pdf_writing(self) -> None:
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as temporary:
            style = apply_cms_style()
            figure, axis = plt.subplots()
            axis.plot([0, 1], [0, 1])
            add_delphes_header(axis, "Validation; Source-member bootstrap")
            png, pdf = save_png_pdf(
                figure, Path(temporary) / "synthetic", dpi=300
            )
            self.assertTrue(style["cms_inspired_style_applied"])
            self.assertGreater(png.stat().st_size, 0)
            self.assertGreater(pdf.stat().st_size, 0)

    def test_temporary_cleanup(self) -> None:
        parent = Path(tempfile.mkdtemp())
        child = parent / "nested"
        child.mkdir()
        (child / "artifact.txt").write_text("temporary", encoding="utf-8")
        shutil.rmtree(parent)
        self.assertFalse(parent.exists())


if __name__ == "__main__":
    unittest.main()
