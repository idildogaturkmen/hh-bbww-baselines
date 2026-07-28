#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts/production/submit_hh4b_ttbar8_canary_retry1.py"
)

SPEC = importlib.util.spec_from_file_location("retry_submit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RetrySubmissionToolTests(unittest.TestCase):
    def test_scalar_queue_one(self) -> None:
        text = """
        universe = vanilla
        queue 1
        """
        self.assertEqual(MODULE.count_queued_jobs(text), 1)

    def test_bare_queue_is_one(self) -> None:
        self.assertEqual(MODULE.count_queued_jobs("queue\n"), 1)

    def test_multiple_queues_are_counted(self) -> None:
        self.assertEqual(
            MODULE.count_queued_jobs("queue 1\nqueue 2\n"),
            3,
        )

    def test_nonscalar_queue_is_rejected(self) -> None:
        with self.assertRaises(MODULE.GateError):
            MODULE.count_queued_jobs("queue member from members.tsv\n")

    def test_parse_terse_submission(self) -> None:
        self.assertEqual(
            MODULE.parse_terse_submission("4000123.0 - 4000123.0\n"),
            (4000123, 0),
        )

    def test_parse_terse_submission_rejects_missing_job(self) -> None:
        with self.assertRaises(MODULE.GateError):
            MODULE.parse_terse_submission("submission failed")

    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            path.write_bytes(b"hh4b-retry")
            expected = hashlib.sha256(b"hh4b-retry").hexdigest()
            self.assertEqual(MODULE.sha256_file(path), expected)

    def test_confirmation_token_is_explicit(self) -> None:
        self.assertEqual(
            MODULE.CONFIRM_TOKEN,
            "REMOVE_3654710_AND_SUBMIT_ONE_RETRY",
        )

    def test_frozen_retry_identity(self) -> None:
        self.assertEqual(MODULE.OLD_JOB, "3654710.0")
        self.assertEqual(MODULE.OLD_SEED, "105003")
        self.assertEqual(
            MODULE.RETURNED_ROOT.name,
            "ttbar_100k_shard003_pythia8_delphes.root",
        )

    def test_payload_requires_missing_writer(self) -> None:
        self.assertIn(
            "repo/scripts/delphes/write_parquet_from_pickle.py",
            MODULE.REQUIRED_PAYLOAD_SUFFIXES,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
