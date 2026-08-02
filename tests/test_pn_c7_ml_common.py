import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from pn_c7_ml_common import (  # noqa: E402
    asimov_za,
    correlated_multijet_profile_za,
    prepare_staging,
    require_columns,
    require_group_fold_integrity,
    require_train_only,
    threshold_at_efficiency,
)


def test_missing_column_fails_closed():
    with pytest.raises(RuntimeError, match="missing required columns"):
        require_columns(pd.DataFrame({"a": [1]}), ["a", "b"], "synthetic")


def test_validation_or_test_split_fails_closed():
    with pytest.raises(RuntimeError, match="non-train"):
        require_train_only(pd.DataFrame({"split": ["train", "validation"]}), "split", "synthetic")


def test_source_fold_leakage_fails_closed():
    frame = pd.DataFrame({
        "registry_group_id": ["a", "a", "b", "c", "d", "e"],
        "registry_oof_fold": [0, 1, 1, 2, 3, 4],
    })
    with pytest.raises(RuntimeError, match="multiple OOF folds"):
        require_group_fold_integrity(frame)


def test_complete_group_fold_registry_passes():
    frame = pd.DataFrame({
        "registry_group_id": list("abcde"),
        "registry_oof_fold": [0, 1, 2, 3, 4],
    })
    require_group_fold_integrity(frame)


def test_output_overwrite_is_rejected(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(RuntimeError, match="overwrite"):
        prepare_staging(output)


def test_threshold_is_deterministic_and_weighted():
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    signal = np.array([True, True, True, False])
    weights = np.array([1.0, 2.0, 1.0, 1.0])
    assert threshold_at_efficiency(scores, signal, weights, 0.75) == pytest.approx(0.8)


def test_correlated_profile_reduces_to_statistical_without_qcd_uncertainty():
    signal = np.array([2.0, 3.0])
    ordinary = np.array([20.0, 30.0])
    qcd = np.zeros(2)
    uncertainty = np.zeros(2)
    observed = correlated_multijet_profile_za(signal, ordinary, qcd, uncertainty)
    expected = np.sqrt(asimov_za(2.0, 20.0) ** 2 + asimov_za(3.0, 30.0) ** 2)
    assert observed == pytest.approx(expected, rel=1.0e-8)


def test_correlated_qcd_uncertainty_cannot_improve_sensitivity():
    signal = np.array([2.0, 3.0])
    ordinary = np.array([5.0, 10.0])
    qcd = np.array([20.0, 30.0])
    statistical = np.sqrt(asimov_za(2.0, 25.0) ** 2 + asimov_za(3.0, 40.0) ** 2)
    profiled = correlated_multijet_profile_za(signal, ordinary, qcd, np.array([0.5, 0.5]))
    assert 0.0 <= profiled <= statistical
