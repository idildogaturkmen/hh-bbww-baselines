from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd

from scripts.analysis.pn_c7y_two_head_model import select_loss_weight


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/analysis/run_pn_c7y_two_head_spanet.py"
TORCH_PYTHON = "/tmp/hh4b_pn_c7t_baseline_env_20260802_v1/bin/python"


def test_loss_weight_selection_is_inner_only_and_ties_choose_smaller_weight() -> None:
    rows = [
        {"assignment_loss_weight": 0.25, "joint_selection_utility": 0.75},
        {"assignment_loss_weight": 1.0, "joint_selection_utility": 0.80},
        {"assignment_loss_weight": 4.0, "joint_selection_utility": 0.80},
    ]
    assert select_loss_weight(rows)["assignment_loss_weight"] == 1.0


def test_runner_has_no_validation_test_or_push_path() -> None:
    source = SCRIPT.read_text().lower()
    assert "git push" not in source
    assert "validation.parquet" not in source
    assert "test.parquet" not in source
    assert '"validation_payload_files_opened": 0' in source
    assert '"test_or_evaluation_payload_files_opened": 0' in source
    assert '"observed_data_opened": false' in source


def test_smoke_pipeline_is_nested_masked_and_checksum_sealed(tmp_path: Path) -> None:
    output = tmp_path / "two_head_smoke"
    completed = subprocess.run(
        [TORCH_PYTHON, str(SCRIPT), "--output", str(output), "--smoke"],
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO)},
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout
    assert (output / "COMPLETE").read_text().strip() == "pn_c7y_two_head_spanet_smoke_pass"
    candidates = pd.read_csv(output / "inner_loss_weight_candidates.tsv", sep="\t")
    assert len(candidates) == 3
    assert set(candidates.assignment_loss_weight) == {0.25, 1.0, 4.0}
    assert not candidates.outer_held_used_for_selection.any()
    selected = pd.read_csv(
        output / "loss_weight_and_operating_point_stability.tsv",
        sep="\t",
    )
    assert len(selected) == 1
    assert not selected.outer_held_used_for_selection.any()
    curves = pd.read_csv(output / "loss_curves.tsv", sep="\t")
    assert {
        "weighted_classification_loss",
        "weighted_assignment_loss",
        "weighted_combined_loss",
        "weighted_assignment_to_classification_loss_ratio",
    }.issubset(curves.columns)
    gradients = pd.read_csv(
        output / "gradient_and_loss_balance_diagnostics.tsv",
        sep="\t",
    )
    assert (gradients.classification_gradient_l2 > 0).all()
    assert (gradients.assignment_gradient_l2 > 0).all()
    prediction = pd.read_parquet(output / "predictions/train_fourb_two_head_oof.parquet")
    completed_rows = prediction.two_head_score.notna()
    assert completed_rows.equals(prediction.registry_oof_fold.eq(0))
    assert prediction.loc[completed_rows, "selected_at_nested_operating_point"].notna().all()
