#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
import re
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
USER = os.environ["USER"]

WRAPPER = (
    REPO
    / "scripts/production/run_hh4b_ttbar8_canary_retry2.sh"
)

CONFIG = (
    REPO
    / "configs/production/hh4b_ttbar8_canary_retry2_v1.yaml"
)

CHECKPOINT = (
    REPO
    / "docs/checkpoints/hh4b_ttbar8_canary_retry2_contract_20260727_v1"
)

SUBMIT = (
    CHECKPOINT
    / "hh4b_ttbar8_exact_regeneration_canary_retry2.sub"
)

INPUT_DIR = Path(
    f"/uscms_data/d3/{USER}/hh4b_delphes/condor_inputs/"
    "hh4b_ttbar8_exact_regeneration_canary_retry2_20260727_v1"
)

RETURN_DIR = Path(
    f"/uscms_data/d3/{USER}/hh4b_delphes/condor_return/"
    "hh4b_ttbar8_exact_regeneration_canary_retry2_20260727_v1/"
    "members/ttbar_100k_shard003"
)

EXPECTED_ENV_SHA = (
    "6f47dbb493681b424342045f38cd514a56dba056e41ce2cce04790d99ce515cd"
)

EXPECTED_PAYLOAD_SHA = (
    "3cc059b75cb63d27b57c51dfbd6643397fcb794e8e184ce689b4d905f53fad97"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


class Retry2ContractTests(unittest.TestCase):
    def test_staged_environment_sha(self) -> None:
        path = INPUT_DIR / "python39_site_packages.tar.gz"
        self.assertTrue(path.is_file())
        self.assertEqual(sha256(path), EXPECTED_ENV_SHA)

    def test_staged_payload_sha(self) -> None:
        path = (
            INPUT_DIR
            / "hh4b_ttbar8_exact_regeneration_canary_retry1_payload.tar.gz"
        )
        self.assertTrue(path.is_file())
        self.assertEqual(sha256(path), EXPECTED_PAYLOAD_SHA)

    def test_wrapper_uses_portable_environment_first(self) -> None:
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn(
            'export PYTHONPATH="$PYTHON_SITE:$PAYLOAD/bootstrap"',
            text,
        )
        self.assertIn("export PYTHONNOUSERSITE=1", text)
        self.assertIn('PYTHON_BIN="/usr/bin/python3"', text)
        self.assertNotIn('source "$LCG_SETUP"', text)

    def test_wrapper_is_reconstruction_only(self) -> None:
        text = WRAPPER.read_text(encoding="utf-8")

        for forbidden in (
            "generate_events",
            "lhe_to_hepmc",
            "DelphesHepMC",
            "pythia8_delphes.sh",
        ):
            self.assertNotIn(forbidden, text)

    def test_submit_has_one_scalar_queue(self) -> None:
        text = SUBMIT.read_text(encoding="utf-8")
        queue_lines = [
            line.strip()
            for line in text.splitlines()
            if re.match(r"^\s*queue\b", line, re.IGNORECASE)
        ]
        self.assertEqual(queue_lines, ["queue 1"])

    def test_submit_is_not_inert(self) -> None:
        text = SUBMIT.read_text(encoding="utf-8").lower()
        self.assertNotIn("hold = true", text)
        self.assertNotIn("requirements = false", text)

    def test_submit_has_exact_three_inputs(self) -> None:
        text = SUBMIT.read_text(encoding="utf-8")
        line = next(
            row
            for row in text.splitlines()
            if row.startswith("transfer_input_files = ")
        )
        inputs = line.split("=", 1)[1].strip().split(",")
        self.assertEqual(len(inputs), 3)
        self.assertTrue(
            any(item.endswith("python39_site_packages.tar.gz") for item in inputs)
        )
        self.assertTrue(
            any(item.endswith(".root") for item in inputs)
        )

    def test_submit_is_retry2_and_not_scaleout(self) -> None:
        text = SUBMIT.read_text(encoding="utf-8")
        self.assertIn("+RetryAttempt = 2", text)
        self.assertIn("+ScaleoutJob = False", text)
        self.assertIn("+ReconstructionOnlyRetry = True", text)

    def test_return_directories_preexist(self) -> None:
        for name in (
            "parquet",
            "logs",
            "receipts",
            "checksums",
            "audits",
        ):
            path = RETURN_DIR / name
            self.assertTrue(path.is_dir())
            self.assertTrue(os.access(path, os.W_OK))

    def test_config_is_unsubmitted(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        self.assertIn("retry_jobs_submitted: 0", text)
        self.assertIn("scaleout_jobs_submitted: 0", text)
        self.assertIn(
            "next_gate: submit_hh4b_ttbar8_exact_regeneration_canary_retry2",
            text,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
