from __future__ import annotations

from pathlib import Path
import sys
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from audit_hh4b_all1000_selection_stability import (  # noqa: E402
    All1000AuditError,
    compare_trees,
    coordinatewise_median,
    ranking_key,
    reduce_winners,
)


def test_independent_ranking_key_contract() -> None:
    feasible = {
        "pooled_inner_oof_feasible": "True",
        "pooled_signal_efficiency": "0.6",
        "pooled_background_efficiency": "0.1",
        "cut_count": "2",
        "structure_id": "z",
    }
    infeasible = {
        "pooled_inner_oof_feasible": "False",
        "pooled_signal_efficiency": "0.5",
        "pooled_background_efficiency": "0.2",
        "cut_count": "1",
        "structure_id": "a",
    }
    assert ranking_key(feasible) == (0, 0.1, 0.6 - 0.585957, 2, "z")
    assert ranking_key(infeasible) == (1, -0.5, 0.2, 1, "a")


def test_independent_tie_reduction_and_coordinatewise_median() -> None:
    winners = [
        {"outer_fold": 0, "structure_id": "z", "thresholds": {"x": 10.0},
         "pooled_support": True, "all_inner_support": True, "feasible": True},
        {"outer_fold": 1, "structure_id": "a", "thresholds": {"x": 1.0},
         "pooled_support": True, "all_inner_support": True, "feasible": True},
        {"outer_fold": 2, "structure_id": "z", "thresholds": {"x": 20.0},
         "pooled_support": True, "all_inner_support": True, "feasible": True},
        {"outer_fold": 3, "structure_id": "a", "thresholds": {"x": 3.0},
         "pooled_support": True, "all_inner_support": True, "feasible": True},
        {"outer_fold": 4, "structure_id": "q", "thresholds": {"x": 7.0},
         "pooled_support": True, "all_inner_support": True, "feasible": False},
    ]
    result = reduce_winners(winners)
    assert result["tie"] is True
    assert result["tied"] == ["a", "z"]
    assert result["selected"] == "a"
    assert result["supporting"] == [1, 3]
    assert result["median_thresholds"] == {"x": 2.0}
    assert coordinatewise_median([{"x": 1.0, "y": 8.0}, {"x": 3.0, "y": 4.0}]) == {
        "x": 2.0,
        "y": 6.0,
    }


def test_tree_comparison_is_byte_sensitive() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        left = root / "left"
        right = root / "right"
        left.mkdir()
        right.mkdir()
        (left / "payload").write_bytes(b"same")
        (right / "payload").write_bytes(b"same")
        assert compare_trees(left, right) == {"file_count": 1, "byte_count": 4}
        (right / "payload").write_bytes(b"diff")
        try:
            compare_trees(left, right)
        except All1000AuditError:
            pass
        else:
            raise AssertionError("byte-different tree passed deterministic comparison")


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
