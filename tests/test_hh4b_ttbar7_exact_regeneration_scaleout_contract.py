#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
import re
import tarfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
USER = os.environ["USER"]

CAMPAIGN = (
    "hh4b_ttbar7_exact_regeneration_scaleout_20260728_v1"
)

WORKER = (
    REPO
    / "scripts/production/"
    "run_hh4b_ttbar7_exact_regeneration_scaleout_member.sh"
)

CONFIG = (
    REPO
    / "configs/production/"
    "hh4b_ttbar7_exact_regeneration_scaleout_v1.yaml"
)

CHECKPOINT = (
    REPO
    / "docs/checkpoints/"
    "hh4b_ttbar7_exact_regeneration_scaleout_contract_20260728_v1"
)

SUBMIT = (
    CHECKPOINT
    / "hh4b_ttbar7_exact_regeneration_scaleout.sub"
)

INPUT_DIR = Path(
    f"/uscms_data/d3/{USER}/hh4b_delphes/condor_inputs/"
    f"{CAMPAIGN}"
)

PAYLOAD = (
    INPUT_DIR
    / "hh4b_ttbar7_exact_regeneration_scaleout_inputs.tar.gz"
)

ENVIRONMENT = (
    INPUT_DIR
    / "python39_site_packages.tar.gz"
)

RETURN_BASE = Path(
    f"/uscms_data/d3/{USER}/hh4b_delphes/condor_return/"
    f"{CAMPAIGN}"
)

EXPECTED_ENV_SHA = (
    "6f47dbb493681b424342045f38cd514a56dba056e41ce2cce04790d99ce515cd"
)

EXPECTED = [
    ("324", "ttbar_100k_shard001", "105001", "10000"),
    ("325", "ttbar_100k_shard002", "105002", "10000"),
    ("347", "ttbar_100k_shard004", "105004", "10000"),
    ("348", "ttbar_100k_shard005", "105005", "10000"),
    ("327", "ttbar_100k_shard007", "105007", "10000"),
    ("328", "ttbar_100k_shard008", "105008", "10000"),
    ("329", "ttbar_100k_shard009", "105009", "10000"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


class ScaleoutContractTests(unittest.TestCase):
    def test_environment_sha(self) -> None:
        self.assertEqual(
            sha256(ENVIRONMENT),
            EXPECTED_ENV_SHA,
        )

    def test_worker_exact_members(self) -> None:
        text = WORKER.read_text(encoding="utf-8")

        observed = re.findall(
            r"^\s+(\d+):(ttbar_100k_shard\d+):"
            r"(\d+):(10000)\) ;;$",
            text,
            flags=re.MULTILINE,
        )

        self.assertEqual(observed, EXPECTED)

    def test_payload_members(self) -> None:
        required = {
            "payload/bootstrap/sitecustomize.py",
            "payload/repo/scripts/delphes/"
            "reconstruct_hh4b_candidates_v2.py",
            "payload/repo/scripts/delphes/"
            "write_parquet_from_pickle.py",
            "payload/repo/scripts/production/"
            "run_hh4b_ttbar7_exact_regeneration_scaleout_member.sh",
            "payload/repo/schema/canonical72_columns.tsv",
        }

        with tarfile.open(PAYLOAD, "r:gz") as archive:
            names = set(archive.getnames())

        self.assertTrue(required.issubset(names))

    def test_submit_exact_rows(self) -> None:
        text = SUBMIT.read_text(encoding="utf-8")

        block = re.search(
            r"queue MEMBER_INDEX,MEMBER_NAME,SEED,"
            r"GENERATED_EVENTS from \(\n"
            r"(.*?)\n\)",
            text,
            flags=re.DOTALL,
        )

        self.assertIsNotNone(block)

        rows = [
            tuple(line.split())
            for line in block.group(1).splitlines()
            if line.strip()
        ]

        self.assertEqual(rows, EXPECTED)

    def test_submit_active(self) -> None:
        text = SUBMIT.read_text(
            encoding="utf-8"
        ).lower()

        self.assertNotIn("hold = true", text)
        self.assertNotIn("requirements = false", text)
        self.assertIn("+scaleoutjob = true", text)
        self.assertIn(
            "+fullchainregeneration = true",
            text,
        )

    def test_config_unsubmitted(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")

        self.assertIn("jobs_described: 7", text)
        self.assertIn("jobs_submitted: 0", text)
        self.assertIn(
            "scheduler_actions_performed: false",
            text,
        )

    def test_return_directories(self) -> None:
        for _, member, _, _ in EXPECTED:
            base = RETURN_BASE / "members" / member

            for child in (
                "lhe",
                "hepmc",
                "root",
                "parquet",
                "logs",
                "receipts",
                "checksums",
            ):
                path = base / child

                self.assertTrue(path.is_dir())
                self.assertTrue(
                    os.access(path, os.W_OK)
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
