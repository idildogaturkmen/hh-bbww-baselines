from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts"
    / "analysis"
    / "audit_hh4b_full_5m_background_and_signal_coverage.py"
)
CONFIG = (
    REPO
    / "configs"
    / "baselines"
    / "hh4b_full_5m_background_and_signal_coverage_v1.json"
)

SPEC = importlib.util.spec_from_file_location("full_coverage_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class FullCoverageAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        original_open = Path.open
        forbidden_suffixes = {
            ".joblib",
            ".parquet",
            ".pickle",
            ".pkl",
            ".root",
            ".ubj",
        }

        def guarded_open(path: Path, *args, **kwargs):
            if path.suffix.lower() in forbidden_suffixes:
                raise AssertionError(f"candidate/model content opened: {path}")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            cls.result = AUDIT.run_audit(REPO, CONFIG)

    def test_full_source_accounting(self) -> None:
        summary = self.result["summary"]
        self.assertEqual(
            summary["status"],
            "hh4b_full_5m_background_and_signal_coverage_audit_pass",
        )
        self.assertEqual(summary["background"]["members"], 520)
        self.assertEqual(summary["background"]["generated_events"], 5_000_000)
        self.assertEqual(summary["signal"]["members"], 110)
        self.assertEqual(summary["signal"]["unique_member_ids"], 110)
        self.assertEqual(summary["signal"]["generated_events"], 200_000)
        self.assertEqual(summary["combined"]["members"], 630)
        self.assertEqual(summary["combined"]["generated_events"], 5_200_000)
        self.assertEqual(summary["combined"]["source_covered_members"], 630)
        self.assertEqual(
            summary["candidate_reconstruction_coverage_status"],
            "blocked_on_27_ttbar_schema_holds",
        )
        self.assertEqual(summary["unresolved_coverage_count"], 27)
        self.assertFalse(summary["physical_normalization_ready"])
        self.assertEqual(
            summary["next_gate"],
            "reconstruct_hh4b_ttbar27_canonical72_candidates",
        )

    def test_development_exclusions_are_exact(self) -> None:
        disposition = {
            row["disposition"]: row["members"]
            for row in self.result["development_disposition"]
        }
        self.assertEqual(
            disposition,
            {
                "development_manifest_member": 581,
                "legacy_ttbar_schema_hold": 27,
                "sealed_background_test": 19,
                "sealed_signal_test": 3,
            },
        )
        self.assertEqual(self.result["summary"]["development"]["candidate_rows"], 70_071)

    def test_sealed_content_and_model_operations_are_zero(self) -> None:
        controls = self.result["summary"]["controls"]
        self.assertEqual(controls["candidate_files_opened"], 0)
        self.assertEqual(controls["candidate_rows_read"], 0)
        self.assertEqual(controls["sealed_test_candidate_files_opened"], 0)
        self.assertEqual(controls["sealed_test_candidate_rows_read"], 0)
        self.assertEqual(controls["models_trained"], 0)
        self.assertEqual(controls["models_scored"], 0)
        self.assertEqual(controls["models_evaluated"], 0)
        self.assertEqual(controls["predictions_produced"], 0)
        self.assertEqual(controls["scoring_performed"], 0)
        self.assertEqual(controls["candidate_content_opened"], 0)
        self.assertEqual(controls["sealed_test_candidate_content_opened"], 0)
        self.assertEqual(
            self.result["summary"]["combined"]["sealed_test_members"],
            22,
        )

    def test_canonical_validation_checkpoint_is_unchanged(self) -> None:
        validation = self.result["summary"]["validation_checkpoint"]
        self.assertTrue(validation["unchanged"])
        self.assertEqual(
            validation["sha256sums_sha256"],
            "127d59487078411a15e43e694f2671e900bf9c9a77a4bd8e39610094c22a92dc",
        )
        self.assertEqual(validation["artifacts_verified"], 88)
        self.assertTrue(
            all(row["matched"] for row in self.result["validation_integrity"])
        )

    def test_duplicate_signal_reference_is_rejected(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        signal_path = REPO / config["inputs"]["signal_registry"]["path"]
        background_path = REPO / config["inputs"]["background_registry"]["path"]
        signal_summary_path = (
            REPO / config["inputs"]["signal_registry_summary"]["path"]
        )
        signals = AUDIT.load_tsv(signal_path, AUDIT.SIGNAL_FIELDS)
        backgrounds = AUDIT.load_tsv(
            background_path,
            AUDIT.BACKGROUND_REGISTRY_FIELDS,
        )
        signals[1] = dict(signals[1])
        signals[1]["candidate_parquet"] = signals[0]["candidate_parquet"]
        with self.assertRaisesRegex(AUDIT.AuditError, "duplicate keys"):
            AUDIT.audit_signal(
                config,
                signals,
                AUDIT.load_json(signal_summary_path),
                backgrounds,
            )

    def test_output_bundle_is_self_consistent(self) -> None:
        environment = AUDIT.environment_value(
            REPO,
            SCRIPT,
            CONFIG,
            self.result,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "checkpoint"
            AUDIT.write_artifacts(output, self.result, environment, overwrite=False)
            listed = []
            with (output / "SHA256SUMS").open("r", encoding="utf-8") as handle:
                for line in handle:
                    expected, relative = line.rstrip("\n").split(None, 1)
                    listed.append(relative)
                    self.assertEqual(
                        AUDIT.sha256_file(output / relative),
                        expected,
                    )
            self.assertEqual(set(listed), set(AUDIT.GENERATED_FILES))
            with (output / "member_coverage.tsv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                members = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(members), 630)
            self.assertTrue(
                all(
                    row["candidate_content_opened_by_this_audit"] == "False"
                    for row in members
                )
            )


if __name__ == "__main__":
    unittest.main()
