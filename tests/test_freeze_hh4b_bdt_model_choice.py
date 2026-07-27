#!/usr/bin/env python3
"""Standard-library unittest coverage for the HH4b BDT choice freeze."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from freeze_hh4b_bdt_model_choice import (  # noqa: E402
    CONFIG_DEFAULT,
    PRIMARY,
    SECONDARY,
    build_decision_rule_audit,
    build_model_choice_rows,
    build_summary,
    build_validation_plan_rows,
    build_working_point_rows,
    collect_evidence,
    find_unique,
    load_yaml,
    require_safe_output_path,
    verify_sha256_manifest,
    verify_source_checkpoints,
    write_table_bundle,
)


class ManifestTests(unittest.TestCase):
    def test_manifest_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = directory / "payload.txt"
            payload.write_text("frozen\n", encoding="utf-8")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            (directory / "SHA256SUMS").write_text(
                f"{digest}  payload.txt\n", encoding="utf-8"
            )
            self.assertEqual(verify_sha256_manifest(directory), 1)

    def test_manifest_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = directory / "payload.txt"
            payload.write_text("first\n", encoding="utf-8")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            (directory / "SHA256SUMS").write_text(
                f"{digest}  payload.txt\n", encoding="utf-8"
            )
            payload.write_text("changed\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_sha256_manifest(directory)


class SourceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_yaml(CONFIG_DEFAULT)
        cls.inventory, cls.directories = verify_source_checkpoints(cls.config)
        cls.evidence = collect_evidence(cls.config, cls.directories)

    def test_four_passing_source_checkpoints(self) -> None:
        self.assertEqual(len(self.inventory), 4)
        self.assertTrue(all(row["manifest_verified"] for row in self.inventory))
        self.assertTrue(
            all(row["passing_status_verified"] for row in self.inventory)
        )

    def test_frozen_primary_evidence(self) -> None:
        self.assertEqual(
            float(self.evidence["global_metric"]["weighted_roc_auc"]),
            0.7737142628735474,
        )
        self.assertAlmostEqual(
            float(
                self.evidence["method_at_cut"][PRIMARY][
                    "weighted_background_efficiency"
                ]
            ),
            0.184305471151,
            places=12,
        )

    def test_frozen_secondary_evidence(self) -> None:
        categories = self.evidence["category_metrics"][SECONDARY]
        self.assertAlmostEqual(
            float(categories["low_mhh"]["weighted_roc_auc"]),
            0.753744498261,
            places=12,
        )
        self.assertAlmostEqual(
            float(categories["high_mhh"]["weighted_roc_auc"]),
            0.805324916529,
            places=12,
        )
        bootstrap = self.evidence["bootstrap"]
        self.assertLess(float(bootstrap["percentile_2_5"]), 0.0)
        self.assertGreater(float(bootstrap["percentile_97_5"]), 0.0)
        self.assertEqual(
            float(bootstrap["fraction_favoring_first_method"]), 0.533
        )

    def test_unique_row_requires_exactly_one(self) -> None:
        with self.assertRaises(ValueError):
            find_unique([], lambda row: True, "missing")
        with self.assertRaises(ValueError):
            find_unique(
                [{"status": "pass"}, {"status": "pass"}],
                lambda row: True,
                "duplicate",
            )


class FreezeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_yaml(CONFIG_DEFAULT)
        cls.inventory, directories = verify_source_checkpoints(cls.config)
        cls.evidence = collect_evidence(cls.config, directories)
        cls.model_rows = build_model_choice_rows()
        cls.validation_rows = build_validation_plan_rows(cls.config)
        cls.working_rows = build_working_point_rows(
            cls.config, cls.evidence
        )
        cls.audit = build_decision_rule_audit(
            cls.config,
            cls.inventory,
            cls.evidence,
            cls.model_rows,
            cls.validation_rows,
            cls.working_rows,
        )

    def test_exact_model_choice(self) -> None:
        primary = [
            row
            for row in self.model_rows
            if row["decision_class"] == "primary_nominal_bdt"
        ]
        secondary = [
            row
            for row in self.model_rows
            if row["decision_class"] == "secondary_categorized_bdt"
        ]
        self.assertEqual([row["model_or_method"] for row in primary], [PRIMARY])
        self.assertEqual(
            [row["model_or_method"] for row in secondary], [SECONDARY]
        )

    def test_validation_plan_is_predeclared(self) -> None:
        models = [
            row
            for row in self.validation_rows
            if row["candidate_type"] == "model"
        ]
        cuts = [
            row
            for row in self.validation_rows
            if row["candidate_type"] == "cut_baseline"
        ]
        self.assertEqual(len(models), 3)
        self.assertEqual(len(cuts), 1)
        self.assertTrue(
            all(
                not row["validation_model_selection_authorized"]
                and not row[
                    "post_validation_hyperparameter_tuning_authorized"
                ]
                for row in self.validation_rows
            )
        )

    def test_two_working_point_sets_and_eight_rows(self) -> None:
        self.assertEqual(len(self.working_rows), 8)
        self.assertEqual(
            len({row["working_point_set"] for row in self.working_rows}), 2
        )
        global_rows = [
            row for row in self.working_rows if row["model"] == PRIMARY
        ]
        categorized_rows = [
            row for row in self.working_rows if row["model"] == SECONDARY
        ]
        self.assertTrue(all(row["global_threshold"] for row in global_rows))
        self.assertTrue(
            all(
                row["low_mhh_threshold"] and row["high_mhh_threshold"]
                for row in categorized_rows
            )
        )

    def test_all_decision_rules_pass(self) -> None:
        self.assertTrue(all(row["passed"] for row in self.audit))

    def test_summary_passes_exact_contract(self) -> None:
        summary = build_summary(
            self.config,
            self.inventory,
            self.model_rows,
            self.validation_rows,
            self.working_rows,
            self.audit,
            source_commit=self.config["source_commit"],
            config_sha256="config",
            runner_sha256="runner",
        )
        self.assertEqual(summary["primary_nominal_model"], PRIMARY)
        self.assertEqual(summary["secondary_categorized_model"], SECONDARY)
        self.assertEqual(summary["models_trained"], 0)
        self.assertEqual(summary["predictions_written"], 0)
        self.assertEqual(summary["hyperparameter_trials"], 0)
        self.assertFalse(summary["failed_pass_conditions"])

    def test_output_paths_are_scoped(self) -> None:
        parent = ROOT / "outputs/agent_runs"
        require_safe_output_path(parent / "exact_run", parent)
        with self.assertRaises(ValueError):
            require_safe_output_path(parent, parent)
        with self.assertRaises(ValueError):
            require_safe_output_path(ROOT / "docs", parent)

    def test_latex_table_is_booktabs_with_physics_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, _, latex = write_table_bundle(
                directory,
                "physics",
                [
                    {
                        "model": PRIMARY,
                        "target_weighted_signal_efficiency": 0.5,
                    }
                ],
                ("model", "target_weighted_signal_efficiency"),
                publication_fields=(
                    "model",
                    "target_weighted_signal_efficiency",
                ),
                caption="Physics labels",
                label="tab:physics",
                latex_headers={
                    "model": "Model",
                    "target_weighted_signal_efficiency": (
                        r"Target weighted $\epsilon_{S}$"
                    ),
                },
            )
            text = latex.read_text(encoding="utf-8")
            self.assertIn(r"\toprule", text)
            self.assertIn(r"\bottomrule", text)
            self.assertIn(r"$\epsilon_{S}$", text)


if __name__ == "__main__":
    unittest.main()
