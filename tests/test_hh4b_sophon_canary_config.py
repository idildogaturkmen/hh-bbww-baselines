#!/usr/bin/env python3

from __future__ import annotations

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

import hh4b_sophon_canary_config as canary_config


def valid_raw(
    directory: Path,
):
    model_paths = []

    for model_index in range(
        3
    ):
        model_paths.append(
            str(
                directory
                / "model{}.onnx".format(
                    model_index
                )
            )
        )

    return {
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
        "temporary_root": (
            "/tmp/iturkmen_track_b_canary"
        ),
        "execution_mode": "dry_run",
        "allow_independent_hh_execution": False,
        "write_events_tree": True,
    }


class SophonCanaryConfigTests(
    unittest.TestCase
):
    def test_valid_control_configuration_and_plan(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            raw = valid_raw(
                Path(directory)
            )

            config = canary_config.parse_canary_config(
                raw
            )

            self.assertEqual(
                config.sample_name,
                "zz_control_canary",
            )

            self.assertEqual(
                config.requested_events,
                64,
            )

            self.assertEqual(
                config.xrootd_url,
                (
                    "root://cceos.ihep.ac.cn:1094"
                    "/eos/ihep/cms/store/user/coli/"
                    "datasets/hh4b/inference/ZZ/example.root"
                ),
            )

            self.assertFalse(
                config.execution_permitted
            )

            plan = canary_config.build_execution_plan(
                config
            )

            self.assertEqual(
                plan["released_model_count"],
                3,
            )

            self.assertFalse(
                plan["execution_permitted"]
            )

    def test_json_roundtrip_loader(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(
                directory
            )

            raw = valid_raw(
                directory_path
            )

            config_path = (
                directory_path
                / "config.json"
            )

            config_path.write_text(
                json.dumps(
                    raw,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            loaded = canary_config.load_canary_config(
                config_path
            )

            self.assertEqual(
                loaded.remote_path,
                raw["remote_path"],
            )

            self.assertEqual(
                loaded.model_paths,
                tuple(
                    raw["model_paths"]
                ),
            )

    def test_missing_and_unknown_keys_are_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            raw = valid_raw(
                Path(directory)
            )

            raw.pop(
                "requested_events"
            )

            with self.assertRaisesRegex(
                KeyError,
                "missing keys",
            ):
                canary_config.parse_canary_config(
                    raw
                )

            raw = valid_raw(
                Path(directory)
            )

            raw["request_events"] = 64

            with self.assertRaisesRegex(
                KeyError,
                "unknown keys",
            ):
                canary_config.parse_canary_config(
                    raw
                )

    def test_endpoint_and_remote_path_validation(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            raw = valid_raw(
                Path(directory)
            )

            raw["endpoint"] = (
                "https://cceos.ihep.ac.cn"
            )

            with self.assertRaisesRegex(
                ValueError,
                "root://",
            ):
                canary_config.parse_canary_config(
                    raw
                )

            raw = valid_raw(
                Path(directory)
            )

            raw["remote_path"] = (
                "relative/example.root"
            )

            with self.assertRaisesRegex(
                ValueError,
                "absolute",
            ):
                canary_config.parse_canary_config(
                    raw
                )

            raw = valid_raw(
                Path(directory)
            )

            raw["remote_path"] = (
                "/eos/ihep/example.txt"
            )

            with self.assertRaisesRegex(
                ValueError,
                r"\.root",
            ):
                canary_config.parse_canary_config(
                    raw
                )

    def test_requested_event_limits(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            for invalid in (
                0,
                -1,
                10001,
            ):
                with self.subTest(
                    invalid=invalid
                ):
                    raw = valid_raw(
                        Path(directory)
                    )

                    raw[
                        "requested_events"
                    ] = invalid

                    with self.assertRaisesRegex(
                        ValueError,
                        "requested_events",
                    ):
                        canary_config.parse_canary_config(
                            raw
                        )

    def test_released_model_list_contract(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            raw = valid_raw(
                Path(directory)
            )

            raw["model_paths"] = raw[
                "model_paths"
            ][
                :2
            ]

            with self.assertRaisesRegex(
                ValueError,
                "exactly 3",
            ):
                canary_config.parse_canary_config(
                    raw
                )

            raw = valid_raw(
                Path(directory)
            )

            raw["model_paths"][2] = raw[
                "model_paths"
            ][0]

            with self.assertRaisesRegex(
                ValueError,
                "distinct",
            ):
                canary_config.parse_canary_config(
                    raw
                )

    def test_temporary_root_and_independent_execution_gate(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            raw = valid_raw(
                Path(directory)
            )

            raw["temporary_root"] = (
                "/uscms_data/d3/iturkmen/track_b_canary"
            )

            with self.assertRaisesRegex(
                ValueError,
                "under /tmp",
            ):
                canary_config.parse_canary_config(
                    raw
                )

            raw = valid_raw(
                Path(directory)
            )

            raw["sample_name"] = (
                "independent_hh"
            )

            raw["sample_role"] = (
                "independent_hh"
            )

            raw["execution_mode"] = (
                "execute"
            )

            with self.assertRaisesRegex(
                PermissionError,
                "execution is disabled",
            ):
                canary_config.parse_canary_config(
                    raw
                )

            raw[
                "allow_independent_hh_execution"
            ] = True

            config = canary_config.parse_canary_config(
                raw
            )

            self.assertTrue(
                config.execution_permitted
            )

    def test_model_file_receipts_and_missing_file_refusal(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(
                directory
            )

            raw = valid_raw(
                directory_path
            )

            for model_index, path_text in enumerate(
                raw["model_paths"]
            ):
                Path(
                    path_text
                ).write_bytes(
                    (
                        "synthetic-model-{}".format(
                            model_index
                        )
                    ).encode(
                        "utf-8"
                    )
                )

            config = canary_config.parse_canary_config(
                raw
            )

            receipts = canary_config.validate_model_files(
                config
            )

            self.assertEqual(
                len(receipts),
                3,
            )

            self.assertEqual(
                [
                    receipt[
                        "model_index"
                    ]
                    for receipt in receipts
                ],
                [
                    0,
                    1,
                    2,
                ],
            )

            for receipt in receipts:
                self.assertGreater(
                    receipt["size_bytes"],
                    0,
                )

                self.assertEqual(
                    len(
                        receipt["sha256"]
                    ),
                    64,
                )

            Path(
                raw["model_paths"][1]
            ).unlink()

            with self.assertRaises(
                FileNotFoundError
            ):
                canary_config.validate_model_files(
                    config
                )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
