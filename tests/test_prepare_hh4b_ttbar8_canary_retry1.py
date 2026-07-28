from __future__ import annotations

import copy
import importlib.util
import tarfile
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts"
    / "production"
    / "prepare_hh4b_ttbar8_canary_retry1.py"
)
CONFIG = (
    REPO
    / "configs"
    / "production"
    / "hh4b_ttbar8_canary_retry1_v1.yaml"
)
WRAPPER = (
    REPO
    / "scripts"
    / "production"
    / "run_hh4b_ttbar8_canary_retry1.sh"
)
SPEC = importlib.util.spec_from_file_location("ttbar8_canary_retry1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RETRY)


class Ttbar8CanaryRetry1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = RETRY.load_config(CONFIG)
        RETRY.validate_config(cls.config)

    def test_retry_identity_is_exact_and_unsubmitted(self) -> None:
        retry = self.config["retry"]
        self.assertEqual(
            retry["campaign"],
            "hh4b_ttbar8_exact_regeneration_canary_retry1_20260727_v1",
        )
        self.assertEqual(retry["member"], "ttbar_100k_shard003")
        self.assertEqual(retry["seed_provenance"], 105003)
        self.assertEqual(retry["source_job"], "3654710.0")
        self.assertEqual(retry["jobs_described"], 1)
        self.assertEqual(retry["jobs_submitted"], 0)
        self.assertEqual(retry["scaleout_jobs_submitted"], 0)

    def test_returned_root_is_immutable_readable_and_tied_to_job(self) -> None:
        audit, rows = RETRY.audit_returned_root(self.config)
        self.assertEqual(audit["bytes"], 913864894)
        self.assertEqual(
            audit["sha256"],
            "127c87510087d164038b7629ea9d5210008cc4cb1473aa9b069a5cd4b859629e",
        )
        self.assertEqual(audit["entries"], 10000)
        self.assertEqual(audit["source_job"], "3654710.0")
        self.assertTrue(audit["readable"])
        self.assertTrue(all(audit["branches"].values()))
        self.assertTrue(all(row["passed"] for row in rows))

    def test_payload_contract_contains_missing_writer_exact_path(self) -> None:
        required = {
            row["exact_worker_path"]: row
            for row in self.config["payload"]["required_files"]
        }
        writer = (
            "/srv/payload/repo/scripts/delphes/"
            "write_parquet_from_pickle.py"
        )
        self.assertIn(writer, required)
        self.assertEqual(
            required[writer]["sha256"],
            "abef7e4f5d1b82fe72837834b0b0794b59bb31ff16032b1b5a7f1519a6edbcfb",
        )

    def test_payload_archive_has_every_required_file_and_checksums(self) -> None:
        config = copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            config["paths"]["payload"] = str(runtime / RETRY.PAYLOAD_FILE_NAME)
            payload, rows = RETRY.build_payload(config, runtime)
            names = {row["archive_path"] for row in rows}
            with tarfile.open(payload, "r:gz") as archive:
                archive_names = set(archive.getnames())
        self.assertIn("payload/SHA256SUMS", names)
        self.assertIn("payload/bootstrap/sitecustomize.py", names)
        for item in config["payload"]["required_files"]:
            self.assertIn(item["archive_path"], names)
            self.assertIn(item["archive_path"], archive_names)

    def test_protected_builder_writer_policy_and_wrapper_match(self) -> None:
        specs, fingerprints = RETRY.protected_fingerprints(self.config)
        self.assertEqual(len(specs), 5)
        self.assertEqual(len(fingerprints), 5)
        self.assertTrue(all(spec["expected_sha256"] in fingerprints.values() for spec in specs))

    def test_exact_command_freezes_policy_and_physics_arguments(self) -> None:
        command = RETRY.exact_reconstruction_command(self.config)
        self.assertIn("PYTHONPATH=/srv/payload/bootstrap", command)
        self.assertIn("timeout --signal=TERM --kill-after=60s 45m", command)
        self.assertIn(
            "/srv/payload/repo/scripts/delphes/"
            "reconstruct_hh4b_candidates_v2.py",
            command,
        )
        for fragment in (
            "--sample ttbar_100k_shard003_pythia8_delphes",
            "--target-mass 125.0",
            "--jet-pt-min 30.0",
            "--jet-eta-max 2.5",
            "--btag-min 0.0",
            "--max-bjets-for-pairing 8",
            "--higgs-ordering pt",
        ):
            self.assertIn(fragment, command)

    def test_wrapper_preserves_failures_without_placeholder_parquet(self) -> None:
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("set -Eeuo pipefail", text)
        self.assertIn("trap finalize EXIT", text)
        self.assertLess(text.index('mkdir -p "$OUTDIR"'), text.index('tar -xzf'))
        self.assertIn(
            'WRITER="$REPO_PAYLOAD/scripts/delphes/write_parquet_from_pickle.py"',
            text,
        )
        self.assertIn("transferable destination", text)
        self.assertNotIn("touch \"$CANDIDATE\"", text)
        self.assertNotIn("generate_events", text)
        self.assertNotIn("DelphesHepMC3", text)
        self.assertNotIn("lhe_to_hepmc3", text)

    def test_submit_description_is_inert_exactly_one_and_root_only(self) -> None:
        payload = RETRY.resolve_path(self.config["paths"]["payload"])
        text = RETRY.render_submit(self.config, payload)
        RETRY.validate_submit_text(text, self.config)
        self.assertEqual(
            [
                line.strip()
                for line in text.splitlines()
                if line.strip().lower().startswith("queue")
            ],
            ["queue 1"],
        )
        self.assertIn("requirements = False", text)
        self.assertIn("hold = True", text)
        self.assertIn("transfer_output_files = output", text)
        self.assertNotIn("output/lhe", text.lower())
        self.assertNotIn("output/hepmc", text.lower())
        self.assertEqual(text.count("ttbar_100k_shard003"), 6)

    def test_legacy_comparison_is_validation_only_and_predeclared(self) -> None:
        legacy = self.config["legacy_validation"]
        rows = RETRY.legacy_rows(self.config)
        self.assertEqual(len(rows), 15)
        self.assertEqual(legacy["expected_rows"], 39)
        self.assertEqual(legacy["floating_rtol"], 0.0)
        self.assertEqual(legacy["floating_atol"], 1.0e-9)
        self.assertTrue(legacy["compare_nonfinite_masks"])
        self.assertTrue(legacy["role"].startswith("validation_only"))
        self.assertEqual(
            {row["column"] for row in rows if row["exact_required"]},
            {"sample", "event", "n_selected_bjets"},
        )

    def test_resources_come_from_successful_retained_root_evidence(self) -> None:
        resources = self.config["resources"]
        evidence = RETRY.resolve_path(resources["evidence_path"])
        self.assertEqual(RETRY.sha256_file(evidence), resources["evidence_sha256"])
        self.assertEqual(resources["request_cpus"], 1)
        self.assertEqual(resources["max_runtime_seconds"], 2700)
        self.assertEqual(resources["successful_reconstruction_wall_seconds"], 20.1217)
        self.assertGreater(resources["request_memory_mb"], resources["prior_canary_peak_memory_mb"])
        self.assertGreater(
            resources["request_disk_mb"] * 1024 * 1024,
            3 * resources["current_input_bytes"],
        )

    def test_output_registry_has_no_generated_event_products(self) -> None:
        outputs = self.config["outputs"]
        rendered = "\n".join(str(value) for value in outputs.values())
        self.assertNotIn(".lhe", rendered.lower())
        self.assertNotIn(".hepmc", rendered.lower())
        self.assertNotIn(".root", rendered.lower())
        self.assertEqual(
            self.config["transfer_policy"]["transfer_output_files"],
            "output",
        )

    def test_later_held_job_policy_forbids_release_and_orders_removal(self) -> None:
        policy = self.config["held_job_later_gate_policy"]
        self.assertTrue(policy["condor_release_forbidden"])
        self.assertFalse(policy["execute_in_this_gate"])
        self.assertEqual(policy["ordered_steps"][2], "call_condor_rm_exactly_once")
        self.assertEqual(
            policy["ordered_steps"][-1],
            "submit_exactly_one_reconstruction_only_retry",
        )

    def test_table_bundle_writes_tsv_markdown_and_booktabs_latex(self) -> None:
        row = {
            "check": "fixture",
            "expected": True,
            "observed": True,
            "passed": True,
            "evidence": "unit-test",
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            RETRY.write_table_bundle(
                directory,
                "retry_returned_root_input_audit",
                [row],
            )
            self.assertTrue(
                (directory / "retry_returned_root_input_audit.tsv").is_file()
            )
            self.assertTrue(
                (directory / "retry_returned_root_input_audit.md").is_file()
            )
            latex = (
                directory / "retry_returned_root_input_audit.tex"
            ).read_text(encoding="utf-8")
        self.assertIn(r"\toprule", latex)
        self.assertIn(r"\midrule", latex)
        self.assertIn(r"\bottomrule", latex)

    def test_bootstrap_and_writer_smoke_are_no_physics(self) -> None:
        self.assertIn(
            'uproot.open.defaults["num_workers"] = 1',
            RETRY.BOOTSTRAP_TEXT,
        )
        self.assertIn("synthetic_contract_only", RETRY.WRITER_SMOKE_TEXT)
        self.assertNotIn("mbb", RETRY.WRITER_SMOKE_TEXT)
        self.assertNotIn("candidate", RETRY.WRITER_SMOKE_TEXT.lower())


if __name__ == "__main__":
    unittest.main()
