from __future__ import annotations

import ast
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/analysis/run_pn_c7x_single_head_spanet.py"
TORCH_PYTHON = "/tmp/hh4b_pn_c7t_baseline_env_20260802_v1/bin/python"


def test_training_script_is_explicitly_train_only_and_not_bootstrap_complete():
    source = SCRIPT.read_text()
    assert "tables/train_fourb_model_development.parquet" in source
    assert "tables/train_primary_physical_projection.parquet" in source
    assert "tables/train_direct_qcd_secondary_projection.parquet" in source
    assert '"validation_payload_files_opened": 0' in source
    assert '"test_or_evaluation_payload_files_opened": 0' in source
    assert '"bootstrap_evaluation_complete": False' in source
    assert "git push" not in source.lower()
    assert "validation.parquet" not in source.lower()
    assert "test.parquet" not in source.lower()


def test_training_has_one_assignment_head_and_distinct_downstream_classifier():
    model_source = (REPO / "scripts/analysis/pn_c7x_spanet_model.py").read_text()
    training_source = SCRIPT.read_text()
    assert model_source.count("self.pair_scorer =") == 1
    assert "classification_head" not in model_source
    assert "XGBClassifier" in training_source
    assert "logically_distinct_downstream_xgboost" in training_source
    assert "truth_partition_label" not in training_source.split("def fit_downstream", 1)[1].split("def source_split_masks", 1)[0]


def test_torch_environment_checks_splits_masses_and_frozen_xgboost_parameters():
    code = r'''
import numpy as np
import pandas as pd
from scripts.analysis import run_pn_c7x_single_head_spanet as r

frame = pd.DataFrame({
    "registry_group_id": ["a", "a", "b", "b", "c", "c"],
    "registry_oof_fold": [0, 0, 1, 1, 2, 2],
})
fit, held = r.source_split_masks(frame, 1)
assert set(frame.loc[fit, "registry_group_id"]) == {"a", "c"}
assert set(frame.loc[held, "registry_group_id"]) == {"b"}

jets = {}
for position, phi in enumerate((0.0, np.pi, 0.5, 0.5 + np.pi), start=1):
    jets[f"j{position}_pt"] = [50.0]
    jets[f"j{position}_eta"] = [0.0]
    jets[f"j{position}_phi"] = [phi]
    jets[f"j{position}_mass"] = [0.0]
mass = r.pairing_masses(pd.DataFrame(jets), np.array([0], dtype=np.int8))
assert mass.shape == (1, 2)
assert np.allclose(mass, 100.0, atol=1e-10)

params = r.xgb_parameters(17, False)
assert params["n_estimators"] == 400
assert params["max_depth"] == 5
assert params["random_state"] == 17
assert r.TARGET_SIGNAL_EFFICIENCY == 0.6156418377159854
assert r.PAIRING_LABELS == {
    "((0, 1), (2, 3))": 0,
    "((0, 2), (1, 3))": 1,
    "((0, 3), (1, 2))": 2,
}
print("helper-contract-pass")
'''
    completed = subprocess.run(
        [TORCH_PYTHON, "-c", code],
        cwd=REPO,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO)},
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout
    assert "helper-contract-pass" in completed.stdout


def test_outer_threshold_is_derived_from_inner_oof_only_by_ast():
    tree = ast.parse(SCRIPT.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_nested")
    text = ast.unparse(function)
    threshold_call = text.split("threshold = threshold_at_efficiency", 1)[1].split("seed =", 1)[0]
    assert "inner_oof[outer_train]" in threshold_call
    assert "outer_held" not in threshold_call
    assert "projection" not in threshold_call
