#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import os
import re
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
USER = os.environ["USER"]

WORKER = (
    REPO
    / "scripts/production/"
    "run_hh4b_ttbar7_reconstruction_only_local_member.sh"
)

CONFIG = (
    REPO
    / "configs/production/"
    "hh4b_ttbar7_reconstruction_only_recovery_v1.yaml"
)

CHECKPOINT = (
    REPO
    / "docs/checkpoints/"
    "hh4b_ttbar7_reconstruction_only_recovery_contract_20260728_v1"
)

INVENTORY = (
    REPO
    / "outputs/agent_runs/"
    "hh4b_ttbar7_reconstruction_envfix_canary_20260728_v1/"
    "root_inventory.tsv"
)

OUTPUT_BASE = Path(
    f"/uscms_data/d3/{USER}/hh4b_delphes/"
    "reconstruction_return/"
    "hh4b_ttbar7_reconstruction_only_recovery_20260728_v1"
)

EXPECTED = [
    (
        "ttbar_100k_shard001",
        "105001",
        "cd812441e240f0fd7b137129aadf4def052907f44830784879ca4e5a0d6f395c",
    ),
    (
        "ttbar_100k_shard002",
        "105002",
        "3328999ac5bc867fc29b29e220d382d468076a4f85b8f6c8193679a702992d61",
    ),
    (
        "ttbar_100k_shard004",
        "105004",
        "fc7d89902695b64ba84ee39c183dd89b549840e32cf1de9f086f96df7b2dad4c",
    ),
    (
        "ttbar_100k_shard005",
        "105005",
        "5a405c983c1fea331dfd853caef09b05f1c092b543d0d80dedf7f303fac03111",
    ),
    (
        "ttbar_100k_shard007",
        "105007",
        "181529274199a5f38c795590dd471ed3d8e0dbe4676d5678c3b744ebf09643a2",
    ),
    (
        "ttbar_100k_shard008",
        "105008",
        "04f3142f14807b77601b735a4901bfe411102bcdc29f17c27189a75869800c50",
    ),
    (
        "ttbar_100k_shard009",
        "105009",
        "62fa589354ecfb8415331393e83c01475a2fe19f30d96bf46cad11bdfca2d50f",
    ),
]

CANARY_SHA = (
    "f42a8bb61007940b71c3e83870cb234d39806dc7f6ac361adbe20842e3118db3"
)


class ReconstructionRecoveryContractTests(
    unittest.TestCase
):
    def test_inventory_exact(self) -> None:
        with INVENTORY.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            rows = list(
                csv.DictReader(
                    handle,
                    delimiter="\t",
                )
            )

        observed = [
            (
                row["member"],
                row["seed"],
                row["root_sha256"],
            )
            for row in rows
        ]

        self.assertEqual(observed, EXPECTED)

        self.assertTrue(
            all(
                int(row["delphes_entries"]) == 10000
                for row in rows
            )
        )

    def test_worker_exact_members(self) -> None:
        text = WORKER.read_text(encoding="utf-8")

        observed = re.findall(
            r"^\s{2}(ttbar_100k_shard\d+):"
            r"(\d+)\)$",
            text,
            flags=re.MULTILINE,
        )

        self.assertEqual(
            observed,
            [
                (member, seed)
                for member, seed, _ in EXPECTED
            ],
        )

    def test_worker_is_reconstruction_only(self) -> None:
        text = WORKER.read_text(encoding="utf-8")

        for forbidden in (
            "generate_events",
            "lhe_to_hepmc3",
            "DelphesHepMC3",
        ):
            self.assertNotIn(forbidden, text)

        self.assertIn(
            "reconstruct_hh4b_candidates_v2.py",
            text,
        )

    def test_worker_has_environment_fix(self) -> None:
        text = WORKER.read_text(encoding="utf-8")

        self.assertIn("--cleanenv", text)
        self.assertIn(
            "--bind /cvmfs:/cvmfs:ro",
            text,
        )
        self.assertIn("-u PYTHONHOME", text)
        self.assertIn("-u PYTHONPATH", text)
        self.assertIn(
            'PYTHONPATH="$PYTHON_SITE:$BOOTSTRAP"',
            text,
        )

    def test_worker_freezes_canary_sha(self) -> None:
        text = WORKER.read_text(encoding="utf-8")

        self.assertIn(CANARY_SHA, text)

    def test_config_is_unexecuted(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")

        for fragment in (
            "status: "
            "hh4b_ttbar7_reconstruction_only_recovery_contract_frozen",
            "mode: local_exact_image_reconstruction_only",
            "runs_described: 7",
            "runs_executed: 0",
            "scheduler_actions_performed: false",
            "maximum_parallel_members: 1",
            "overwrite_existing_outputs: false",
            "madgraph: false",
            "pythia: false",
            "delphes: false",
            "sealed_test_members_opened: 0",
            "physical_normalization: false",
            "next_gate: "
            "commit_hh4b_ttbar7_reconstruction_only_recovery_contract",
        ):
            self.assertIn(fragment, text)

    def test_output_trees_are_empty(self) -> None:
        for member, _, _ in EXPECTED:
            base = OUTPUT_BASE / "members" / member

            for child in (
                "parquet",
                "logs",
                "receipts",
                "checksums",
            ):
                self.assertTrue(
                    (base / child).is_dir()
                )

            files = [
                path
                for path in base.rglob("*")
                if path.is_file()
            ]

            self.assertEqual(files, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
