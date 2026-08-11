from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_hh4b_all1000_selection_stability import (  # noqa: E402
    AggregationError,
    aggregation_input_label,
    verify_bound_external_artifacts,
)


def test_external_input_label_is_stable_and_not_tmp_path_dependent() -> None:
    repo = Path("/repo")
    assert aggregation_input_label(repo, repo / "docs/checkpoint.json") == "docs/checkpoint.json"
    assert (
        aggregation_input_label(repo, Path("/tmp/audit/structure.tsv"))
        == "external_escalation_complete_return_audit/structure.tsv"
    )


def test_external_artifacts_are_bound_by_exact_set_size_and_sha() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        payload = b"frozen escalation inventory\n"
        path = root / "structure.tsv"
        path.write_bytes(payload)
        bindings = {
            path.name: {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        }
        assert verify_bound_external_artifacts(root, bindings) == {path.name: path}
        path.write_bytes(payload + b"changed\n")
        try:
            verify_bound_external_artifacts(root, bindings)
        except AggregationError:
            pass
        else:
            raise AssertionError("changed external escalation artifact passed")


if __name__ == "__main__":
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, function in tests:
        function()
        print(f"{name}=PASS")
    print(f"TEST_COUNT={len(tests)}")
