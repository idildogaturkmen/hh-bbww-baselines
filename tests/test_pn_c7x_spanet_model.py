from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import pn_c7x_spanet_model as model


REPO = Path(__file__).resolve().parents[1]


def synthetic_frame(rows: int = 12) -> pd.DataFrame:
    data = {}
    for position in range(1, 5):
        data[f"j{position}_pt"] = np.linspace(40.0 + position, 160.0 + position, rows)
        data[f"j{position}_eta"] = np.linspace(-2.0, 2.0, rows) + position * 0.01
        data[f"j{position}_phi"] = np.linspace(-3.0, 3.0, rows) + position * 0.02
        data[f"j{position}_mass"] = np.linspace(5.0 + position, 30.0 + position, rows)
        data[f"j{position}_btag"] = (np.arange(rows) + position) % 2
        data[f"j{position}_flavor"] = 5
    return pd.DataFrame(data)


def test_jet_inputs_have_frozen_shape_and_exclude_flavor():
    frame = synthetic_frame()
    inputs = model.jet_feature_array(frame)
    assert inputs.shape == (12, 4, 5)
    assert np.isfinite(inputs).all()
    source = (REPO / "scripts/analysis/pn_c7x_spanet_model.py").read_text()
    assert "_flavor\"]" not in source
    assert model.JET_FEATURES == ("log1p_pt", "eta", "sin_phi", "cos_phi", "log1p_mass")


def test_normalizer_is_fit_only_from_supplied_partition():
    frame = synthetic_frame()
    first = model.fit_normalizer(frame.iloc[:8])
    second = model.fit_normalizer(frame.iloc[4:])
    assert first["mean"].shape == (5,)
    assert first["scale"].shape == (5,)
    assert not np.allclose(first["mean"], second["mean"])
    normalized = model.normalize_jets(frame.iloc[:8], first)
    assert np.allclose(normalized.reshape(-1, 5).mean(axis=0), 0.0, atol=2e-6)


def test_tiny_negative_mass_is_clipped_but_larger_negative_fails():
    frame = synthetic_frame()
    frame.loc[0, "j3_mass"] = -1.2e-6
    inputs = model.jet_feature_array(frame)
    assert inputs[0, 2, 4] == 0.0
    frame.loc[0, "j3_mass"] = -2.1e-6
    with pytest.raises(RuntimeError, match="frozen numerical tolerance"):
        model.jet_feature_array(frame)


def test_torch_absence_has_clear_fail_closed_environment(monkeypatch):
    original_import = __import__

    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("blocked torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    with pytest.raises(RuntimeError, match=model.TORCH_ENVIRONMENT):
        model.require_torch()


def test_real_torch_forward_backward_and_permutation_contract():
    code = r'''
import numpy as np
import pandas as pd
import torch
from scripts.analysis import pn_c7x_spanet_model as m

torch.manual_seed(7)
net = m.build_model(7)
net.eval()
x = torch.randn(5, 4, 5)
permutation = (2, 0, 3, 1)
with torch.no_grad():
    original, _ = net(x)
    permuted, _ = net(x[:, permutation, :])
mapping = m.matching_label_permutation(permutation)
assert torch.allclose(permuted, original[:, mapping], atol=2e-6, rtol=0.0)

rows = 15
data = {}
for position in range(1, 5):
    data[f"j{position}_pt"] = np.linspace(40 + position, 120 + position, rows)
    data[f"j{position}_eta"] = np.linspace(-2, 2, rows) + 0.03 * position
    data[f"j{position}_phi"] = np.linspace(-3, 3, rows) + 0.04 * position
    data[f"j{position}_mass"] = np.linspace(5 + position, 25 + position, rows)
    data[f"j{position}_btag"] = (np.arange(rows) + position) % 2
frame = pd.DataFrame(data)
normalizer = m.fit_normalizer(frame)
trained, curve, elapsed = m.train_assignment_model(
    frame,
    np.arange(rows) % 3,
    np.ones(rows),
    normalizer,
    seed=19,
    smoke=True,
)
features, prediction, maximum, entropy = m.invariant_features(trained, frame, normalizer)
assert len(curve) == 2 and elapsed > 0
assert features.shape == (rows, 133)
assert prediction.shape == maximum.shape == entropy.shape == (rows,)
assert np.isfinite(features).all()
assert m.parameter_count(trained) > 0
print(torch.__version__, m.parameter_count(trained))
'''
    completed = subprocess.run(
        [model.TORCH_ENVIRONMENT, "-c", code],
        cwd=REPO,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO)},
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout
    assert "2.5.1+cpu" in completed.stdout
