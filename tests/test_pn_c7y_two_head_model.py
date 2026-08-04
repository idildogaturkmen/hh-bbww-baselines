from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts.analysis.pn_c7x_spanet_model import matching_label_permutation
from scripts.analysis.pn_c7y_two_head_model import (
    LOSS_WEIGHTS,
    joint_selection_utility,
)


REPO = Path(__file__).resolve().parents[1]
TORCH_PYTHON = "/tmp/hh4b_pn_c7t_baseline_env_20260802_v1/bin/python"


def test_frozen_loss_weights_and_joint_utility() -> None:
    assert LOSS_WEIGHTS == (0.25, 1.0, 4.0)
    assert joint_selection_utility(0.7, 0.9) == pytest.approx(0.8)
    with pytest.raises(RuntimeError, match="unit interval"):
        joint_selection_utility(1.1, 0.9)


def test_matching_label_permutation_remains_bijective() -> None:
    for permutation in ((0, 1, 2, 3), (2, 0, 3, 1), (3, 2, 1, 0)):
        assert sorted(matching_label_permutation(permutation).tolist()) == [0, 1, 2]


def test_two_head_forward_backward_and_permutation_symmetry() -> None:
    code = r'''
import numpy as np
import torch
from scripts.analysis.pn_c7x_spanet_model import matching_label_permutation
from scripts.analysis.pn_c7y_two_head_model import build_two_head_model

torch.manual_seed(7)
model = build_two_head_model(7)
model.eval()
jets = torch.randn(8, 4, 5)
assignment, classification, embedding = model(jets)
assert assignment.shape == (8, 3)
assert classification.shape == (8,)
assert embedding.shape == (8, 4, 64)
loss = torch.nn.functional.cross_entropy(assignment, torch.tensor([0,1,2,0,1,2,0,1]))
loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
    classification, torch.tensor([0.,1.,0.,1.,0.,1.,0.,1.])
)
loss.backward()
assert all(parameter.grad is not None for parameter in model.parameters())

permutation = (2, 0, 3, 1)
with torch.no_grad():
    original_assignment, original_classification, _ = model(jets)
    permuted_assignment, permuted_classification, _ = model(jets[:, permutation, :])
mapping = matching_label_permutation(permutation)
np.testing.assert_allclose(
    permuted_assignment.numpy(),
    original_assignment[:, mapping].numpy(),
    rtol=0.0,
    atol=2e-5,
)
np.testing.assert_allclose(
    permuted_classification.numpy(),
    original_classification.numpy(),
    rtol=0.0,
    atol=2e-5,
)
print("two_head_symmetry_pass")
'''
    completed = subprocess.run(
        [TORCH_PYTHON, "-c", code],
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO)},
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout
    assert "two_head_symmetry_pass" in completed.stdout


def test_model_module_is_importable_without_torch() -> None:
    source = REPO / "scripts/analysis/pn_c7y_two_head_model.py"
    text = source.read_text()
    assert "\nimport torch\n" not in text
    assert "require_torch()" in text
