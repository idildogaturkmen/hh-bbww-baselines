from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts"
    / "analysis"
    / "recover_hh4b_ttbar27_canonical72_candidates.py"
)
CONFIG = (
    REPO
    / "configs"
    / "baselines"
    / "hh4b_ttbar27_canonical72_recovery_v1.yaml"
)
SPEC = importlib.util.spec_from_file_location("ttbar27_recovery", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


class Ttbar27RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = RECOVERY.load_config(CONFIG)
        cls.inputs = RECOVERY.load_inputs(cls.config)
        cls.names, cls.types = RECOVERY.exact_schema_contract(
            cls.inputs["canonical72_columns"]
        )

    def make_table(self, rows: int, *, nonfinite: bool = False) -> pa.Table:
        arrays = []
        expected_sample = "fixture_pythia8_delphes"
        for name in self.names:
            arrow_type = pa.type_for_alias(self.types[name])
            if pa.types.is_string(arrow_type):
                if name == "sample":
                    values = [expected_sample] * rows
                else:
                    values = ["fixture"] * rows
            elif pa.types.is_integer(arrow_type):
                values = list(range(rows))
            else:
                values = [float(index + 1) for index in range(rows)]
                if nonfinite and name == "mbb1" and rows:
                    values[0] = np.nan
            arrays.append(pa.array(values, type=arrow_type))
        return pa.Table.from_arrays(arrays, names=self.names)

    def test_frozen_inventory_contract(self) -> None:
        RECOVERY.validate_config_only(self.config)
        coverage = self.inputs["member_coverage"]
        holds = coverage[
            coverage["development_disposition"] == "legacy_ttbar_schema_hold"
        ]
        self.assertEqual(len(holds), 27)
        self.assertEqual(holds["member_id"].nunique(), 27)
        self.assertEqual(int(holds["generated_events"].astype(int).sum()), 270_000)
        self.assertFalse(holds["sealed_test"].map(RECOVERY.as_bool).any())

    def test_exact_schema_accepts_typed_zero_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zero.parquet"
            pq.write_table(self.make_table(0), path)
            result, frame = RECOVERY.verify_candidate(
                path,
                self.names,
                self.types,
                0,
                "fixture_pythia8_delphes",
            )
        self.assertEqual(len(frame), 0)
        self.assertEqual(result["observed_columns"], 72)
        self.assertTrue(result["exact_column_order"])
        self.assertTrue(result["zero_row_schema_readable"])

    def test_nonfinite_feature_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.parquet"
            pq.write_table(self.make_table(1, nonfinite=True), path)
            with self.assertRaisesRegex(
                RECOVERY.RecoveryError, "content check failed"
            ):
                RECOVERY.verify_candidate(
                    path,
                    self.names,
                    self.types,
                    1,
                    "fixture_pythia8_delphes",
                )

    def test_unexpected_or_reordered_columns_are_rejected(self) -> None:
        table = self.make_table(1)
        reordered = table.select(list(reversed(self.names)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reordered.parquet"
            pq.write_table(reordered, path)
            with self.assertRaisesRegex(
                RECOVERY.RecoveryError, "column order mismatch"
            ):
                RECOVERY.verify_candidate(
                    path,
                    self.names,
                    self.types,
                    1,
                    "fixture_pythia8_delphes",
                )

    def test_deterministic_metadata_rank_is_stable(self) -> None:
        args = ("fixed-salt", "background", "ttbar_inclusive", "member-a")
        self.assertEqual(
            RECOVERY.deterministic_rank(*args),
            RECOVERY.deterministic_rank(*args),
        )
        self.assertNotEqual(
            RECOVERY.deterministic_rank(*args),
            RECOVERY.deterministic_rank(
                "fixed-salt", "background", "ttbar_inclusive", "member-b"
            ),
        )

    def test_test_extension_preserves_22_and_opens_no_content(self) -> None:
        extension, composition, stats = RECOVERY.proposed_test_extension(
            self.config, self.inputs["member_coverage"]
        )
        preserved = [
            row
            for row in extension
            if row["disposition"] == "preserve_existing_sealed"
        ]
        added = [
            row
            for row in extension
            if row["disposition"] == "proposed_add_metadata_only"
        ]
        self.assertEqual(len(preserved), 22)
        self.assertGreater(len(added), 0)
        self.assertEqual(stats["existing"], 22)
        self.assertEqual(stats["combined"], 22 + len(added))
        self.assertEqual(stats["candidate_content_opened"], 0)
        self.assertFalse(stats["finalized"])
        self.assertTrue(
            all(row["candidate_content_opened"] is False for row in extension)
        )
        self.assertTrue(
            all(row["representative_coverage_achievable"] for row in composition)
        )

    def test_reconstruction_command_explicitly_encodes_policy(self) -> None:
        command = RECOVERY.build_reconstruction_command(
            self.config, Path("/tmp/fixture.parquet")
        )
        rendered = " ".join(command)
        for expected in (
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            "VECLIB_MAXIMUM_THREADS=1",
            "PYTHONUNBUFFERED=1",
            "timeout --signal=TERM --kill-after=60s 45m",
        ):
            self.assertIn(expected, rendered)
        for expected in (
            "--target-mass 125.0",
            "--jet-pt-min 30.0",
            "--jet-eta-max 2.5",
            "--btag-min 0.0",
            "--max-bjets-for-pairing 8",
            "--higgs-ordering pt",
        ):
            self.assertIn(expected, rendered)

    def test_determinism_comparison_is_row_level_and_tolerant(self) -> None:
        primary = self.make_table(2).to_pandas()
        rerun = primary.copy(deep=True)
        primary.loc[0, "mbb1"] = np.nan
        rerun.loc[0, "mbb1"] = np.nan
        rerun.loc[1, "mbb1"] += 5.0e-13
        result = RECOVERY.compare_determinism_frames(
            primary,
            rerun,
            self.names,
            self.types,
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertTrue(result["determinism_exact_schema_and_column_order"])
        self.assertTrue(result["exact_event_ordering"])
        self.assertTrue(result["exact_integer_and_string_values"])
        self.assertTrue(result["matching_nonfinite_masks"])
        self.assertTrue(result["floating_values_within_tolerance"])
        self.assertTrue(result["deterministic_row_level_equal"])

    def test_determinism_comparison_rejects_identity_or_mask_change(self) -> None:
        primary = self.make_table(2).to_pandas()
        rerun = primary.copy(deep=True)
        rerun.loc[0, "sample"] = "different_pythia8_delphes"
        rerun.loc[0, "mbb1"] = np.nan
        result = RECOVERY.compare_determinism_frames(
            primary,
            rerun,
            self.names,
            self.types,
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertFalse(result["exact_integer_and_string_values"])
        self.assertFalse(result["matching_nonfinite_masks"])
        self.assertFalse(result["deterministic_row_level_equal"])

    def test_execution_attempt_audit_preserves_stall_and_successes(self) -> None:
        attempts = RECOVERY.execution_attempt_rows(self.config)
        self.assertEqual(
            [row["result_classification"] for row in attempts],
            [
                "aborted_stalled_no_output",
                "successful_production_reconstruction",
                "successful_determinism_validation_rerun",
            ],
        )
        self.assertEqual(attempts[0]["output_bytes"], 0)
        self.assertEqual(attempts[1]["candidate_rows"], 29)
        self.assertEqual(attempts[2]["candidate_rows"], 29)

    def test_table_bundle_has_markdown_and_booktabs_latex(self) -> None:
        fields = RECOVERY.TABLE_FIELDS["ttbar27_duplicate_audit"]
        row = {field: "value" for field in fields}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            RECOVERY.write_table_bundle(
                output, "ttbar27_duplicate_audit", [row]
            )
            self.assertTrue((output / "ttbar27_duplicate_audit.tsv").is_file())
            self.assertTrue((output / "ttbar27_duplicate_audit.md").is_file())
            latex = (output / "ttbar27_duplicate_audit.tex").read_text(
                encoding="utf-8"
            )
        self.assertIn(r"\toprule", latex)
        self.assertIn(r"\midrule", latex)
        self.assertIn(r"\bottomrule", latex)

    def test_missing_57_columns_are_never_synthesized(self) -> None:
        missing = self.inputs["missing_source_adjudication"]
        self.assertEqual(len(missing), 8)
        self.assertTrue(
            (missing["event_summary_columns"].astype(int) == 13).all()
        )
        self.assertFalse(
            missing["has_raw_nested_jet_payload"].map(RECOVERY.as_bool).any()
        )
        self.assertEqual(
            set(missing["target_tag"]),
            RECOVERY.MISSING_TAGS,
        )


if __name__ == "__main__":
    unittest.main()
