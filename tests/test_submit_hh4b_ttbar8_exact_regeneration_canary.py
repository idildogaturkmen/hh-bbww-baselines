from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts"
    / "production"
    / "submit_hh4b_ttbar8_exact_regeneration_canary.py"
)
SPEC = importlib.util.spec_from_file_location("ttbar8_canary_submit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUBMIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBMIT)


class Ttbar8ExactRegenerationCanarySubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = SUBMIT.load_frozen_contract()
        cls.text = SUBMIT.render_submit(cls.contract)

    def test_submit_description_queues_exactly_one_process(self) -> None:
        self.assertEqual(SUBMIT.queue_count(self.text), 1)
        self.assertEqual(
            [
                line.strip()
                for line in self.text.splitlines()
                if line.strip().lower().startswith("queue")
            ],
            ["queue 1"],
        )

    def test_only_frozen_canary_identity_is_present(self) -> None:
        self.assertIn(
            "arguments = 326 ttbar_100k_shard003 105003 10000",
            self.text,
        )
        for member in ("001", "002", "004", "005", "007", "008", "009"):
            self.assertNotIn(f"ttbar_100k_shard{member}", self.text)

    def test_resources_and_no_retry_policy_are_frozen(self) -> None:
        self.assertIn("request_cpus = 1", self.text)
        self.assertIn("request_memory = 2048MB", self.text)
        self.assertIn("request_disk = 12288MB", self.text)
        self.assertIn("+MaxRuntime = 7200", self.text)
        self.assertIn("max_retries = 0", self.text)
        self.assertIn("+AutomaticRetries = 0", self.text)

    def test_failed_worker_is_held_without_resubmission(self) -> None:
        self.assertIn(
            "on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)",
            self.text,
        )
        self.assertNotIn("requirements = False", self.text)
        self.assertNotIn("hold = True", self.text)

    def test_frozen_outputs_are_explicit_and_remapped(self) -> None:
        for source in SUBMIT.execute_side_output_paths().values():
            self.assertIn(source, self.text)
        for destination in SUBMIT.output_paths().values():
            self.assertIn(str(destination), self.text)

    def test_scaleout_classad_is_explicitly_false(self) -> None:
        self.assertIn("+ExactRegenerationCanary = True", self.text)
        self.assertIn("+ScaleoutJob = False", self.text)


if __name__ == "__main__":
    unittest.main()
