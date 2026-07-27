from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts"
    / "analysis"
    / "diagnose_hh4b_ttbar8_exact_regeneration_canary_failure.py"
)
SPEC = importlib.util.spec_from_file_location("ttbar8_canary_diagnosis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)


class Ttbar8CanaryFailureDiagnosisTests(unittest.TestCase):
    def test_frozen_job_identity(self) -> None:
        self.assertEqual(DIAG.JOB_ID, "3654710.0")
        self.assertEqual(DIAG.MEMBER_NAME, "ttbar_100k_shard003")
        self.assertEqual(DIAG.SEED, 105003)
        self.assertEqual(DIAG.GENERATED_EVENTS, 10000)

    def test_parse_classad_preserves_relevant_values(self) -> None:
        parsed = DIAG.parse_classad(
            'ClusterId = 3654710\n'
            'JobStatus = 5\n'
            'HoldReason = "missing output"\n'
            'ExitBySignal = false\n'
        )
        self.assertEqual(parsed["ClusterId"], "3654710")
        self.assertEqual(parsed["JobStatus"], "5")
        self.assertEqual(parsed["HoldReason"], "missing output")
        self.assertEqual(parsed["ExitBySignal"], "false")

    def test_submit_output_and_remap_parser(self) -> None:
        parsed = DIAG.parse_submit(
            "transfer_output_files = output/a,output/b\n"
            'transfer_output_remaps = "output/a=/return/a;output/b=/return/b"\n'
        )
        self.assertEqual(parsed["transfer_output_files"], "output/a,output/b")
        self.assertEqual(
            DIAG.parse_transfer_remaps(parsed["transfer_output_remaps"]),
            {"output/a": "/return/a", "output/b": "/return/b"},
        )

    def test_scheduler_event_parser(self) -> None:
        rows = DIAG.parse_scheduler_events(
            "001 (3654710.000.000) 2026-07-27 18:04:04 Job executing\n"
            "\tSlotName: slot1_4@cmswn2224.fnal.gov\n"
            "...\n"
            "012 (3654710.000.000) 2026-07-27 18:20:57 Job was held.\n"
            "\tCode 12 Subcode 2\n"
            "...\n"
        )
        self.assertEqual([row["event_code"] for row in rows], ["001", "012"])
        self.assertIn("cmswn2224", rows[0]["details"])
        self.assertIn("Subcode 2", rows[1]["details"])

    def test_payload_dependency_defect_is_reproducible(self) -> None:
        audit = DIAG.inspect_payload(DIAG.PAYLOAD)
        self.assertTrue(audit["builder_present"])
        self.assertTrue(audit["builder_manifested"])
        self.assertFalse(audit["helper_present"])
        self.assertFalse(audit["helper_manifested"])

    def test_builder_declares_the_missing_sibling_dependency(self) -> None:
        text = DIAG.RECONSTRUCTION_BUILDER.read_text(encoding="utf-8")
        self.assertIn(
            'Path(__file__).with_name("write_parquet_from_pickle.py")',
            text,
        )
        self.assertIn("missing isolated Parquet writer", text)

    def test_failure_classifier_selects_exactly_canonical_reconstruction(self) -> None:
        facts = {
            "canonical_started": True,
            "canonical_log": DIAG.EXPECTED_ERROR,
            "root_exists": True,
            "parquet_exists": False,
            "payload_builder_present": True,
            "payload_helper_present": False,
        }
        self.assertEqual(
            DIAG.classify_failure(facts),
            "canonical_reconstruction_failure",
        )
        self.assertIn(
            DIAG.classify_failure(facts),
            DIAG.ALLOWED_PRIMARY_TYPES,
        )

    def test_failure_classifier_is_ambiguous_without_full_evidence(self) -> None:
        self.assertEqual(DIAG.classify_failure({}), "ambiguous_failure")

    def test_frozen_path_contract_is_exact(self) -> None:
        submit = DIAG.parse_submit(DIAG.FROZEN_SUBMIT.read_text(encoding="utf-8"))
        sources = submit["transfer_output_files"].split(",")
        relative = f"output/parquet/{DIAG.EXPECTED_PARQUET.name}"
        remaps = DIAG.parse_transfer_remaps(submit["transfer_output_remaps"])
        self.assertIn(relative, sources)
        self.assertEqual(remaps[relative], str(DIAG.EXPECTED_PARQUET))
        worker = DIAG.WORKER.read_text(encoding="utf-8")
        self.assertIn('--out "$CANDIDATE"', worker)
        self.assertIn('"$OUTDIR"/{lhe,hepmc,root,parquet,logs,receipts,checksums}', worker)

    def test_wrapper_propagates_pipeline_failure(self) -> None:
        worker = DIAG.WORKER.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", worker)
        self.assertIn('tee "$OUTDIR/logs/${MEMBER_NAME}_canonical72.log"', worker)
        self.assertIn("trap finalize EXIT", worker)

    def test_correction_plan_is_physics_invariant_and_not_authorized(self) -> None:
        rows = DIAG.correction_plan_rows()
        self.assertGreaterEqual(len(rows), 3)
        self.assertTrue(all(row["physics_invariant"] for row in rows))
        self.assertTrue(all(not row["retry_authorized"] for row in rows))
        self.assertIn("write_parquet_from_pickle.py", rows[0]["change"])

    def test_table_bundle_writes_tsv_markdown_and_booktabs_latex(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            DIAG.write_table_bundle(
                destination,
                "audit",
                ("field", "ok"),
                [{"field": "value", "ok": True}],
            )
            self.assertTrue((destination / "audit.tsv").is_file())
            self.assertTrue((destination / "audit.md").is_file())
            latex = (destination / "audit.tex").read_text(encoding="utf-8")
            self.assertIn(r"\toprule", latex)
            self.assertIn(r"\bottomrule", latex)

    def test_authorized_path_guard_rejects_unrelated_location(self) -> None:
        with self.assertRaises(RuntimeError):
            DIAG.require_authorized_path(Path("/etc/passwd"))


if __name__ == "__main__":
    unittest.main()
