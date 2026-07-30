#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("freeze_hh4b_qcd_train_generator_weight_transport.py")
if not MODULE_PATH.is_file():
    MODULE_PATH = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analysis"
        / "freeze_hh4b_qcd_train_generator_weight_transport.py"
    )
spec = importlib.util.spec_from_file_location("pn_c4j_freezer", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class FreezeContractTests(unittest.TestCase):
    def test_expected_population_contract(self) -> None:
        self.assertEqual(module.EXPECTED["production"], 260)
        self.assertEqual(module.EXPECTED["train"], 186)
        self.assertEqual(module.EXPECTED["variable_train"], 127)
        self.assertEqual(module.EXPECTED["uniform_train"], 59)
        self.assertEqual(module.EXPECTED["validation"], 56)
        self.assertEqual(module.EXPECTED["test"], 18)

    def test_freezer_is_payload_reader_free(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        forbidden = (
            "import uproot",
            "import pyarrow",
            "read_parquet",
            "tarfile.open",
            "xrdcp",
            "xrdfs",
        )
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_physical_application_remains_unauthorized(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('"qcd_train_physical_weight_application_authorized": False', source)
        self.assertIn('"physical_luminosity_weights_calculated": 0', source)
        self.assertIn('"physical_yields_calculated": 0', source)

    def test_checksum_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "a.txt").write_text("a\n", encoding="utf-8")
            (directory / "b.tsv").write_text("x\ty\n1\t2\n", encoding="utf-8")
            module.write_checksum_manifest(directory)
            module.verify_checksum_manifest(directory)

    def test_boolean_parser_is_fail_closed(self) -> None:
        self.assertTrue(module.parse_bool("True"))
        self.assertFalse(module.parse_bool("false"))
        with self.assertRaises(RuntimeError):
            module.parse_bool("yes")


if __name__ == "__main__":
    unittest.main()
