#!/usr/bin/env python3
"""Standard-library tests for the HH4b validation protocol amendment."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from amend_hh4b_bdt_validation_protocol import (  # noqa: E402
    CONFIG_DEFAULT,
    DIAGNOSTIC,
    NEXT_GATE,
    PRIMARY,
    PRIMARY_TARGET,
    SECONDARY,
    STATUS,
    SUPERSEDED_POLICY,
    build_candidate_rows,
    build_decision_audit,
    build_metric_contract_rows,
    build_selection_rule_rows,
    build_summary,
    build_superseded_policy_rows,
    build_test_policy_rows,
    build_validation_role_rows,
    load_yaml,
    require_safe_output_path,
    verify_frozen_contracts,
    verify_source_checkpoints,
    write_table_bundle,
)


class ProtocolFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_yaml(CONFIG_DEFAULT)
        cls.inventory, cls.directories = verify_source_checkpoints(cls.config)
        cls.facts = verify_frozen_contracts(cls.config, cls.directories)
        cls.candidates = build_candidate_rows(cls.config)
        cls.validation = build_validation_role_rows(cls.config)
        cls.selection = build_selection_rule_rows()
        cls.metrics = build_metric_contract_rows(cls.config)
        cls.test_policy = build_test_policy_rows(cls.config)
        cls.superseded = build_superseded_policy_rows()
        cls.audit = build_decision_audit(
            cls.config,
            cls.inventory,
            cls.facts,
            cls.candidates,
            cls.validation,
            cls.selection,
            cls.metrics,
            cls.test_policy,
            unittest_tests_run=15,
            unittests_passed=True,
            table_failures=0,
        )


class SourcePreconditionTests(ProtocolFixture):
    def test_five_source_checkpoints_pass(self) -> None:
        self.assertEqual(len(self.inventory), 5)
        self.assertTrue(all(row["manifest_verified"] for row in self.inventory))
        self.assertTrue(
            all(row["passing_status_verified"] for row in self.inventory)
        )

    def test_frozen_population_metadata(self) -> None:
        self.assertEqual(self.facts["train_members"], 458)
        self.assertEqual(self.facts["train_rows"], 57326)
        self.assertEqual(self.facts["validation_members"], 123)
        self.assertEqual(
            self.facts["validation_rows_from_frozen_manifest_metadata"], 12745
        )
        self.assertEqual(self.facts["test_members_considered"], 0)
        self.assertEqual(
            self.facts[
                "validation_candidate_files_opened_previously_by_bdt_gates"
            ],
            0,
        )
        self.assertEqual(
            self.facts[
                "test_candidate_files_opened_previously_by_bdt_gates"
            ],
            0,
        )

    def test_frozen_contract_dimensions_and_trials(self) -> None:
        self.assertEqual(self.facts["global_v1_features"], 34)
        self.assertEqual(self.facts["categorized_v2_features_per_model"], 52)
        self.assertEqual(self.facts["mass_plane_blind_features"], 30)
        self.assertEqual(self.facts["mhh_category_boundary_GeV"], 450)
        self.assertEqual(self.facts["global_v1_trial_id"], 18)
        self.assertEqual(self.facts["categorized_low_mhh_trial_id"], 19)
        self.assertEqual(self.facts["categorized_high_mhh_trial_id"], 18)
        self.assertEqual(self.facts["frozen_working_point_sets"], 2)
        self.assertEqual(self.facts["frozen_working_point_rows"], 8)


class CandidateAndFittingTests(ProtocolFixture):
    def test_two_candidates_and_no_final_nominal(self) -> None:
        candidates = [row for row in self.candidates if row["candidate_bdt"]]
        self.assertEqual(
            [row["method"] for row in candidates], [PRIMARY, SECONDARY]
        )
        self.assertTrue(
            all(
                not row["final_nominal_selected_in_amendment"]
                for row in self.candidates
            )
        )

    def test_three_validation_strategies_and_four_models(self) -> None:
        self.assertEqual(
            [row["method"] for row in self.candidates],
            [PRIMARY, SECONDARY, DIAGNOSTIC],
        )
        self.assertEqual(
            sum(int(row["fitted_models_next_gate"]) for row in self.candidates),
            4,
        )
        self.assertTrue(all(not row["scaler_fitted"] for row in self.candidates))
        self.assertTrue(
            all(
                row["fit_population"] == "complete_frozen_train_population"
                for row in self.candidates
            )
        )


class ValidationSelectionTests(ProtocolFixture):
    def test_validation_role_and_no_reopening(self) -> None:
        permitted = {
            row["activity"]
            for row in self.validation
            if row["authorization"] == "permitted"
        }
        prohibited = {
            row["activity"]
            for row in self.validation
            if row["authorization"] == "prohibited"
        }
        self.assertIn("select_final_nominal_bdt", permitted)
        self.assertIn("check_optimized_cut_baseline", permitted)
        self.assertIn(
            "evaluate_predeclared_mass_plane_blind_diagnostic", permitted
        )
        self.assertIn(
            "reopening_validation_after_inspecting_results", prohibited
        )
        self.assertIn("changing_model_hyperparameters", prohibited)

    def test_exact_selection_rule(self) -> None:
        thresholds = {
            row["condition"]: row["threshold"] for row in self.selection
        }
        self.assertEqual(
            thresholds["minimum_relative_background_efficiency_reduction"],
            0.02,
        )
        self.assertEqual(
            thresholds["bootstrap_fraction_favoring_categorized_v2"], 0.84
        )
        self.assertEqual(thresholds["signal_mode_equalization"], 0.05)
        self.assertEqual(
            thresholds["integrity_category_and_model_application"], 0
        )
        self.assertTrue(
            all(row["all_conditions_required"] for row in self.selection)
        )
        self.assertTrue(
            all(
                row["failure_action"] == f"select_{PRIMARY}"
                for row in self.selection
            )
        )

    def test_fixed_and_equalized_metric_contracts(self) -> None:
        fixed = [
            row
            for row in self.metrics
            if row["comparison_kind"]
            == "fixed_train_oof_working_point_generalization"
        ]
        equalized = [
            row
            for row in self.metrics
            if row["comparison_kind"] == "equalized_validation_efficiency"
        ]
        self.assertEqual(len(fixed), 8)
        self.assertEqual(len(equalized), 8)
        expected_targets = {0.3, 0.5, PRIMARY_TARGET, 0.7}
        self.assertEqual(
            {float(row["target_weighted_signal_efficiency"]) for row in fixed},
            expected_targets,
        )
        self.assertEqual(
            {
                float(row["target_weighted_signal_efficiency"])
                for row in equalized
            },
            expected_targets,
        )
        self.assertTrue(
            all(
                not row["may_replace_frozen_train_oof_thresholds"]
                for row in fixed + equalized
            )
        )
        self.assertEqual(
            sum(bool(row["primary_selection_metric"]) for row in equalized), 2
        )

    def test_bootstrap_contract(self) -> None:
        self.assertEqual(self.config["bootstrap"]["seed"], 20260727)
        self.assertEqual(self.config["bootstrap"]["replicates"], 2000)
        self.assertEqual(
            self.config["bootstrap"]["delta_definition"],
            "epsilon_background_categorized_v2_minus_global_v1",
        )
        self.assertEqual(
            self.config["bootstrap"]["negative_values_favor"], SECONDARY
        )

    def test_pooled_uncalibrated_category_auc_is_prohibited(self) -> None:
        rows = [
            row
            for row in self.metrics
            if row["comparison_kind"]
            == "pooled_uncalibrated_category_roc_auc"
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "prohibited")
        self.assertFalse(
            self.config["metric_contract"][
                "pooled_uncalibrated_categorized_roc_auc_authorized"
            ]
        )


class PolicyBoundaryTests(ProtocolFixture):
    def test_test_split_is_not_a_selection_split(self) -> None:
        self.assertEqual(
            self.config["test"]["role"], "final_unbiased_evaluation_split"
        )
        self.assertFalse(
            self.config["test"]["accessible_during_validation_selection"]
        )
        self.assertFalse(
            self.config["test"]["selection_using_test_authorized"]
        )
        selection_rows = [
            row
            for row in self.test_policy
            if row["item"] == "final_nominal_model_selection_or_change"
        ]
        self.assertEqual(selection_rows[0]["authorization"], "prohibited")

    def test_only_validation_selection_policy_is_superseded(self) -> None:
        self.assertEqual(
            self.config["supersession"]["supersedes_policy_from"],
            SUPERSEDED_POLICY,
        )
        self.assertTrue(
            self.config["supersession"][
                "preserve_source_checkpoint_unchanged"
            ]
        )
        changed = [
            row
            for row in self.superseded
            if row["disposition"] == "superseded"
        ]
        self.assertEqual(
            {row["policy"] for row in changed},
            {"validation_model_selection_authorized", "next_gate"},
        )
        preserved = {
            row["policy"]
            for row in self.superseded
            if row["disposition"].startswith("preserved")
        }
        self.assertIn("source_checkpoint_contents", preserved)
        self.assertIn("frozen_train_oof_working_points", preserved)
        self.assertIn(
            "category_specific_roc_and_feature_importance_plots", preserved
        )

    def test_all_decision_audits_pass(self) -> None:
        self.assertTrue(all(row["passed"] for row in self.audit))

    def test_summary_counters_and_authorization(self) -> None:
        summary = build_summary(
            self.config,
            self.facts,
            self.inventory,
            self.candidates,
            self.audit,
            source_commit=self.config["source_commit"],
            config_sha256="config",
            runner_sha256="runner",
            unittest_tests_run=15,
            unittests_passed=True,
            table_failures=0,
        )
        self.assertEqual(summary["status"], STATUS)
        self.assertEqual(summary["candidate_bdt_models"], 2)
        self.assertEqual(summary["final_nominal_model_selected"], 0)
        self.assertEqual(summary["validation_models_predeclared"], 3)
        self.assertEqual(summary["validation_cut_baselines_predeclared"], 1)
        self.assertEqual(summary["models_trained"], 0)
        self.assertEqual(summary["predictions_written"], 0)
        self.assertEqual(summary["hyperparameter_trials"], 0)
        self.assertTrue(summary["validation_selection_authorized"])
        self.assertFalse(summary["validation_reoptimization_authorized"])
        self.assertFalse(summary["test_selection_authorized"])
        self.assertEqual(summary["bootstrap_replicates_predeclared"], 2000)
        self.assertTrue(summary["both_bdt_candidates_presented"])
        self.assertTrue(summary["category_specific_roc_plots_retained"])
        self.assertTrue(
            summary["category_specific_feature_importance_plots_retained"]
        )
        self.assertTrue(summary["non_selected_bdt_secondary_benchmark_retained"])
        self.assertEqual(summary["next_gate"], NEXT_GATE)
        self.assertFalse(summary["failed_pass_conditions"])


class OutputTests(ProtocolFixture):
    def test_scoped_output_paths(self) -> None:
        parent = ROOT / "outputs/agent_runs"
        require_safe_output_path(parent / "exact_protocol_run", parent)
        with self.assertRaises(ValueError):
            require_safe_output_path(parent, parent)
        with self.assertRaises(ValueError):
            require_safe_output_path(ROOT / "docs", parent)

    def test_booktabs_latex_with_physics_labels_and_escaping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, _, latex = write_table_bundle(
                directory,
                "selection",
                [
                    {
                        "method": SECONDARY,
                        "target_weighted_signal_efficiency": PRIMARY_TARGET,
                        "rule": "epsilon_B & mHH",
                    }
                ],
                (
                    "method",
                    "target_weighted_signal_efficiency",
                    "rule",
                ),
                publication_fields=(
                    "method",
                    "target_weighted_signal_efficiency",
                    "rule",
                ),
                caption="Validation selection",
                label="tab:validation_selection",
                latex_headers={
                    "method": "Method",
                    "target_weighted_signal_efficiency": (
                        r"Target weighted $\epsilon_{S}$"
                    ),
                    "rule": r"Rule in $m_{HH}$",
                },
            )
            text = latex.read_text(encoding="utf-8")
            self.assertIn(r"\toprule", text)
            self.assertIn(r"\bottomrule", text)
            self.assertIn(r"$\epsilon_{S}$", text)
            self.assertIn(r"$m_{HH}$", text)
            self.assertIn(r"\&", text)


if __name__ == "__main__":
    unittest.main()
