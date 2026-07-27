from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts"
    / "production"
    / "prepare_hh4b_ttbar8_exact_regeneration.py"
)
CONFIG = (
    REPO
    / "configs"
    / "production"
    / "hh4b_ttbar8_exact_regeneration_v1.yaml"
)
SPEC = importlib.util.spec_from_file_location("ttbar8_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)


class Ttbar8ExactRegenerationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = CONTRACT.load_config(CONFIG)
        CONTRACT.validate_config(cls.config)
        cls.members, cls.source = CONTRACT.load_source_members(cls.config)

    def test_source_inventory_is_exactly_eight_nonsealed_members(self) -> None:
        self.assertEqual(len(self.members), 8)
        self.assertEqual(
            {row["target_tag"] for row in self.members},
            CONTRACT.MISSING_TAGS,
        )
        self.assertEqual(
            len({row["source_member_id"] for row in self.members}),
            8,
        )
        self.assertFalse(any(row["sealed_test"] for row in self.members))
        self.assertEqual(
            sum(int(row["legacy_candidate_rows"]) for row in self.members),
            307,
        )

    def test_seeds_and_generated_counts_are_frozen_and_unique(self) -> None:
        expected = {
            "ttbar_100k_shard001": 105001,
            "ttbar_100k_shard002": 105002,
            "ttbar_100k_shard003": 105003,
            "ttbar_100k_shard004": 105004,
            "ttbar_100k_shard005": 105005,
            "ttbar_100k_shard007": 105007,
            "ttbar_100k_shard008": 105008,
            "ttbar_100k_shard009": 105009,
        }
        observed = {
            row["target_tag"]: int(row["seed"]) for row in self.members
        }
        self.assertEqual(observed, expected)
        self.assertEqual(len(set(observed.values())), 8)
        self.assertTrue(
            all(int(row["generated_events"]) == 10_000 for row in self.members)
        )

    def test_every_exact_member_contract_is_complete(self) -> None:
        rows = CONTRACT.member_inventory_rows(self.config, self.members)
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertEqual(row["exact_process_definition"], "p p > t t~")
            self.assertEqual(
                row["exact_run_card_overrides"]["iseed"],
                row["exact_generator_seed"],
            )
            self.assertEqual(row["generator_version"], "MadGraph5_aMC@NLO_3.5.3")
            self.assertEqual(row["pythia_version"], "8.312")
            self.assertEqual(row["delphes_version"], "3.5.2pre01-33-ga3b0385")
            self.assertEqual(len(row["common_column_comparison_fields"]), 15)
            self.assertFalse(row["sealed_test"])

    def test_protected_artifacts_and_process_manifest_match(self) -> None:
        rows, fingerprints = CONTRACT.audit_protected(self.config)
        self.assertGreaterEqual(len(rows), 15)
        self.assertEqual(len(rows), len(fingerprints))
        self.assertTrue(all(row["exists"] for row in rows))
        self.assertTrue(all(row["unchanged"] for row in rows))

    def test_canary_selection_is_deterministic_and_exactly_one(self) -> None:
        selected, rows = CONTRACT.deterministic_canary(
            self.config, self.members
        )
        selected_again, rows_again = CONTRACT.deterministic_canary(
            self.config, list(reversed(self.members))
        )
        self.assertEqual(selected["target_tag"], "ttbar_100k_shard003")
        self.assertEqual(int(selected["seed"]), 105003)
        self.assertEqual(selected_again["target_tag"], selected["target_tag"])
        self.assertEqual(rows, rows_again)
        self.assertEqual(sum(row["selected_canary"] for row in rows), 1)
        self.assertFalse(any(row["submitted"] for row in rows))

    def test_canary_and_scaleout_plans_are_bounded_and_locked(self) -> None:
        canary, _ = CONTRACT.deterministic_canary(self.config, self.members)
        canary_rows, scaleout_rows = CONTRACT.submission_rows(
            self.config, self.members, canary
        )
        self.assertEqual(len(canary_rows), 1)
        self.assertEqual(len(scaleout_rows), 7)
        self.assertEqual(
            len(
                {
                    row["original_member_name"]
                    for row in [*canary_rows, *scaleout_rows]
                }
            ),
            8,
        )
        self.assertTrue(
            all(row["automatic_retries"] == 0 for row in scaleout_rows)
        )
        self.assertTrue(all(row["initially_held"] for row in scaleout_rows))
        self.assertFalse(any(row["submitted"] for row in scaleout_rows))

    def test_expected_outputs_are_unique_and_no_overwrite(self) -> None:
        rows = CONTRACT.output_registry_rows(self.config, self.members)
        for field in (
            "delphes_root_filename",
            "candidate_filename",
            "stageout_member_directory",
            "receipt_path",
            "checksum_manifest_path",
        ):
            self.assertEqual(len({row[field] for row in rows}), 8)
        self.assertTrue(
            all(row["no_overwrite_policy"] == "fail_before_generation" for row in rows)
        )

    def test_legacy_comparison_contract_has_all_fifteen_columns(self) -> None:
        rows = CONTRACT.legacy_contract_rows(self.config)
        self.assertEqual([row["column"] for row in rows], list(CONTRACT.LEGACY_COLUMNS))
        by_column = {row["column"]: row for row in rows}
        for exact in ("sample", "event", "n_selected_bjets"):
            self.assertTrue(by_column[exact]["exact_required"])
        self.assertEqual(by_column["mbb1"]["rtol"], 0.0)
        self.assertEqual(by_column["mbb1"]["atol"], 1.0e-9)
        self.assertFalse(by_column["pairing"]["exact_required"])

    def test_validation_levels_are_separate_and_never_claim_bytes(self) -> None:
        rows = CONTRACT.validation_rows(self.config)
        self.assertEqual({row["level"] for row in rows}, {1, 2, 3, 4})
        self.assertFalse(any(row["byte_identity_claimed"] for row in rows))
        self.assertIn(
            "cross_member_duplicate_keys",
            {row["validation"] for row in rows},
        )

    def test_resources_are_derived_from_historical_ttbar_run(self) -> None:
        row = CONTRACT.resource_rows(self.config)[0]
        self.assertEqual(row["historical_jobs"], 10)
        self.assertAlmostEqual(
            row["historical_mean_wall_seconds_per_member"],
            341.048,
            places=3,
        )
        self.assertGreater(row["request_memory_mb"], row["historical_maximum_resident_mb"])
        self.assertGreater(
            row["request_disk_mb"] / 1024.0,
            row["historical_mean_write_gib_per_member"],
        )

    def test_submit_description_is_inert_and_has_exact_queue(self) -> None:
        canary, _ = CONTRACT.deterministic_canary(self.config, self.members)
        canary_rows, scaleout_rows = CONTRACT.submission_rows(
            self.config, self.members, canary
        )
        canary_submit = CONTRACT.render_submit(
            "canary", canary_rows, self.config
        )
        scaleout_submit = CONTRACT.render_submit(
            "scaleout", scaleout_rows, self.config
        )
        for text in (canary_submit, scaleout_submit):
            self.assertIn("requirements = False", text)
            self.assertIn("hold = True", text)
            self.assertIn("max_retries = 0", text)
        self.assertEqual(canary_submit.count("ttbar_100k_shard"), 1)
        self.assertEqual(scaleout_submit.count("ttbar_100k_shard"), 7)

    def test_table_bundle_has_tsv_markdown_and_booktabs_latex(self) -> None:
        name = "unresolved_exact_regeneration_inputs"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            CONTRACT.write_table_bundle(output, name, [])
            self.assertTrue((output / f"{name}.tsv").is_file())
            self.assertTrue((output / f"{name}.md").is_file())
            latex = (output / f"{name}.tex").read_text(encoding="utf-8")
        self.assertIn(r"\toprule", latex)
        self.assertIn(r"\midrule", latex)
        self.assertIn(r"\bottomrule", latex)


if __name__ == "__main__":
    unittest.main()
