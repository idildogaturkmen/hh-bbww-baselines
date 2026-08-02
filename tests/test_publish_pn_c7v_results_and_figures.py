from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from publish_pn_c7v_results_and_figures import (  # noqa: E402
    histogram_rows,
    inspect_torch_checkpoint,
    load_inputs,
    require_fresh_directory,
    require_prediction_alignment,
)


REPO = Path(__file__).resolve().parents[1]
PUBLICATION_SCRIPT = SCRIPT_DIR / "publish_pn_c7v_results_and_figures.py"


def aligned() -> pd.DataFrame:
    return pd.DataFrame({
        "registry_group_id": ["a", "b"],
        "registry_member_index": [0, 1],
        "registry_oof_fold": [0, 1],
        "event": [10, 11],
    })


def test_prediction_alignment_accepts_exact_copy() -> None:
    frame = aligned()
    require_prediction_alignment(frame, frame.copy(), "synthetic")


def test_prediction_alignment_rejects_duplicate_or_reordered_source() -> None:
    frame = aligned()
    changed = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(RuntimeError, match="alignment drift"):
        require_prediction_alignment(frame, changed, "synthetic")


def test_prediction_alignment_rejects_missing_column() -> None:
    with pytest.raises(RuntimeError, match="missing required columns"):
        require_prediction_alignment(aligned().drop(columns="event"), aligned(), "synthetic")


def test_histogram_is_deterministic_and_normalized() -> None:
    one = histogram_rows(np.array([0.1, 0.2, 0.8]), np.array([1.0, 2.0, 1.0]), np.array([0, 0.5, 1]),
                         figure="x", panel="p", series="s", normalize=True)
    two = histogram_rows(np.array([0.1, 0.2, 0.8]), np.array([1.0, 2.0, 1.0]), np.array([0, 0.5, 1]),
                         figure="x", panel="p", series="s", normalize=True)
    assert one == two
    assert sum(row["value"] for row in one) == pytest.approx(1.0)


def test_empty_histogram_rejected() -> None:
    with pytest.raises(RuntimeError, match="empty normalized histogram"):
        histogram_rows(np.array([]), np.array([]), np.array([0, 1]), figure="x", panel="p", series="s", normalize=True)


def test_output_overwrite_rejected(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        require_fresh_directory(output)


def test_publication_module_import_succeeds_when_torch_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("publish_pn_c7v_without_torch", PUBLICATION_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "torch", None)
    spec.loader.exec_module(module)
    assert callable(module.make_figures)


def test_torch_checkpoint_inspection_fails_closed_without_torch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match=r"PyTorch is required only.*frozen c7t interpreter"):
        inspect_torch_checkpoint(tmp_path / "synthetic.pt")


def test_torch_checkpoint_inspection_uses_lazy_torch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []

    def fake_load(path: Path, **kwargs: object) -> dict[str, object]:
        calls.append((path, kwargs))
        return {"model": {}, "fold": 0}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=fake_load))
    path = tmp_path / "synthetic.pt"
    assert inspect_torch_checkpoint(path) == "torch_keys=fold,model"
    assert calls == [(path, {"map_location": "cpu", "weights_only": False})]


def test_load_inputs_accesses_only_frozen_train_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[Path] = []
    original = pd.read_parquet

    def audited_read_parquet(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        opened.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", audited_read_parquet)
    payload = load_inputs()
    assert payload["training"]["registry_final_split"].eq("train").all()
    assert opened
    assert all(path.name.startswith("train_") for path in opened)
    assert not any(part.lower() in {"validation", "test", "evaluation"} for path in opened for part in path.parts)


def test_repository_contains_no_hig_reference_pdf_and_script_contains_no_push_command() -> None:
    tracked = subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).splitlines()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO, text=True
    ).splitlines()
    candidates = tracked + [line[3:] for line in status]
    assert not any("HIG-24-015" in path and path.lower().endswith(".pdf") for path in candidates)
    assert "git push" not in PUBLICATION_SCRIPT.read_text().lower()
