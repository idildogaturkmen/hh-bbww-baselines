#!/usr/bin/env python3

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO / "scripts" / "analysis"

sys.path.insert(
    0,
    str(ANALYSIS_DIR),
)

import run_hh4b_sophon_remote_canary as canary_runner


def write_valid_fixture(
    directory: Path,
    *,
    execution_mode: str = "dry_run",
):
    model_paths = []

    for model_index in range(
        3
    ):
        model_path = (
            directory
            / "model{}.onnx".format(
                model_index
            )
        )

        model_path.write_bytes(
            (
                "synthetic-onnx-model-{}".format(
                    model_index
                )
            ).encode(
                "utf-8"
            )
        )

        model_paths.append(
            str(
                model_path
            )
        )

    temporary_root = (
        directory
        / "runtime"
    )

    raw = {
        "schema_version": 1,
        "sample_name": "zz_control_canary",
        "sample_role": "control",
        "endpoint": "root://cceos.ihep.ac.cn:1094",
        "remote_path": (
            "/eos/ihep/cms/store/user/coli/"
            "datasets/hh4b/inference/ZZ/example.root"
        ),
        "tree_policy": "highest_cycle",
        "requested_events": 64,
        "model_paths": model_paths,
        "temporary_root": str(
            temporary_root
        ),
        "execution_mode": execution_mode,
        "allow_independent_hh_execution": False,
        "write_events_tree": True,
    }

    config_path = (
        directory
        / "canary.json"
    )

    config_path.write_text(
        json.dumps(
            raw,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return (
        config_path,
        raw,
    )


class SophonRemoteCanaryRunnerTests(
    unittest.TestCase
):
    def test_dry_run_receipt_contract(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            config_path, raw = write_valid_fixture(
                Path(directory)
            )

            receipt = (
                canary_runner.build_dry_run_receipt(
                    config_path
                )
            )

            self.assertEqual(
                receipt["status"],
                "dry_run_validated",
            )

            self.assertFalse(
                receipt[
                    "execution_plan"
                ][
                    "execution_permitted"
                ]
            )

            self.assertEqual(
                receipt[
                    "execution_plan"
                ][
                    "requested_events"
                ],
                64,
            )

            self.assertEqual(
                receipt[
                    "execution_plan"
                ][
                    "xrootd_url"
                ],
                (
                    raw["endpoint"]
                    + raw["remote_path"]
                ),
            )

            self.assertEqual(
                len(
                    receipt[
                        "model_receipts"
                    ]
                ),
                3,
            )

            self.assertFalse(
                receipt[
                    "operations"
                ][
                    "network_accessed"
                ]
            )

            self.assertFalse(
                receipt[
                    "operations"
                ][
                    "onnx_inference_run"
                ]
            )

    def test_model_hash_receipts_are_exact(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            config_path, raw = write_valid_fixture(
                Path(directory)
            )

            receipt = (
                canary_runner.build_dry_run_receipt(
                    config_path
                )
            )

            for model_index, model_receipt in enumerate(
                receipt["model_receipts"]
            ):
                model_bytes = Path(
                    raw["model_paths"][
                        model_index
                    ]
                ).read_bytes()

                expected_sha = hashlib.sha256(
                    model_bytes
                ).hexdigest()

                self.assertEqual(
                    model_receipt[
                        "model_index"
                    ],
                    model_index,
                )

                self.assertEqual(
                    model_receipt[
                        "sha256"
                    ],
                    expected_sha,
                )

                self.assertEqual(
                    model_receipt[
                        "size_bytes"
                    ],
                    len(
                        model_bytes
                    ),
                )

    def test_config_hash_tracks_exact_file_bytes(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(
                directory
            )

            config_path, raw = write_valid_fixture(
                directory_path
            )

            first = (
                canary_runner.build_dry_run_receipt(
                    config_path
                )
            )

            raw["sample_name"] = (
                "zz_control_canary_v2"
            )

            config_path.write_text(
                json.dumps(
                    raw,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            second = (
                canary_runner.build_dry_run_receipt(
                    config_path
                )
            )

            self.assertNotEqual(
                first["config_sha256"],
                second["config_sha256"],
            )

            self.assertEqual(
                second["config_sha256"],
                hashlib.sha256(
                    config_path.read_bytes()
                ).hexdigest(),
            )

    def test_execute_mode_is_refused_before_backend_use(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            config_path, unused_raw = write_valid_fixture(
                Path(directory),
                execution_mode="execute",
            )

            with self.assertRaisesRegex(
                PermissionError,
                "supports dry_run only",
            ):
                canary_runner.build_dry_run_receipt(
                    config_path
                )

    def test_receipt_roundtrip_and_overwrite_refusal(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(
                directory
            )

            config_path, raw = write_valid_fixture(
                directory_path
            )

            receipt_path = (
                Path(
                    raw["temporary_root"]
                )
                / "receipts"
                / "dry_run.json"
            )

            receipt = canary_runner.run_dry_run(
                config_path,
                receipt_path=receipt_path,
            )

            loaded = json.loads(
                receipt_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                loaded,
                receipt,
            )

            with self.assertRaises(
                FileExistsError
            ):
                canary_runner.run_dry_run(
                    config_path,
                    receipt_path=receipt_path,
                )

    def test_receipt_path_cannot_escape_temporary_root(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(
                directory
            )

            config_path, raw = write_valid_fixture(
                directory_path
            )

            outside_path = (
                directory_path
                / "outside.json"
            )

            with self.assertRaisesRegex(
                ValueError,
                "under temporary_root",
            ):
                canary_runner.run_dry_run(
                    config_path,
                    receipt_path=outside_path,
                )

            self.assertFalse(
                outside_path.exists()
            )

    def test_cli_prints_valid_canonical_receipt(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            config_path, unused_raw = write_valid_fixture(
                Path(directory)
            )

            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(
                stdout
            ), redirect_stderr(
                stderr
            ):
                return_code = canary_runner.main(
                    [
                        "--config",
                        str(
                            config_path
                        ),
                    ]
                )

            self.assertEqual(
                return_code,
                0,
            )

            self.assertEqual(
                stderr.getvalue(),
                "",
            )

            parsed = json.loads(
                stdout.getvalue()
            )

            self.assertEqual(
                parsed["status"],
                "dry_run_validated",
            )

            self.assertEqual(
                stdout.getvalue().strip(),
                canary_runner.canonical_json(
                    parsed
                ),
            )

    def test_missing_model_fails_without_receipt(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(
                directory
            )

            config_path, raw = write_valid_fixture(
                directory_path
            )

            Path(
                raw["model_paths"][1]
            ).unlink()

            receipt_path = (
                Path(
                    raw["temporary_root"]
                )
                / "failed.json"
            )

            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(
                stdout
            ), redirect_stderr(
                stderr
            ):
                return_code = canary_runner.main(
                    [
                        "--config",
                        str(
                            config_path
                        ),
                        "--receipt",
                        str(
                            receipt_path
                        ),
                    ]
                )

            self.assertEqual(
                return_code,
                2,
            )

            self.assertEqual(
                stdout.getvalue(),
                "",
            )

            self.assertIn(
                "FileNotFoundError",
                stderr.getvalue(),
            )

            self.assertFalse(
                receipt_path.exists()
            )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
