from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from pn_c7w_categorized_bdt_common import (  # noqa: E402
    assign_categories,
    bootstrap_fixed_categories,
    evaluate_categories,
    summarize_replicas,
    weighted_quantile,
)
from run_pn_c7w_nested_categorized_bdt import (  # noqa: E402
    REPO,
    VARIANTS,
    build_outer_selected_predictions,
    paired_difference_summary,
    standardize_source_rows,
)
from finalize_pn_c7w_reviewed_checkpoint import (  # noqa: E402
    EXPECTED_FIGURES,
    read_inspection_report,
    verify_source,
)


def sparse_synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    projection = pd.DataFrame({
        "registry_member_index": [0, 1, 2],
        "registry_group_id": ["signal", "ordinary", "qcd"],
        "registry_training_target": [1, 0, 0],
        "analysis_population_role": ["signal", "fourb_ordinary_background", "primary_transferred_multijet_template"],
        "primary_projection_physical_weight_inclusive": [2.0, 10.0, 20.0],
    })
    direct = pd.DataFrame({
        "registry_member_index": [2],
        "registry_group_id": ["qcd"],
        "population_kind": ["hard_qcd"],
        "direct_projection_physical_weight": [18.0],
    })
    members = pd.DataFrame({
        "member_index": [0, 1, 2],
        "transport_id": ["signal", "ordinary", "qcd"],
    })
    return projection, direct, members, np.array([0, 0, 1], dtype=np.int8), np.array([1], dtype=np.int8)


def test_weighted_quantile_and_exclusive_category_geometry() -> None:
    values = np.array([0.1, 0.4, 0.8, 0.9])
    weights = np.array([1.0, 1.0, 5.0, 1.0])
    assert weighted_quantile(values, weights, [0.5])[0] > 0.4
    categories = assign_categories(
        np.array([0.1, 0.6, 0.6, 0.9]), np.array([0.2, 0.2, 0.7, 0.95]), 0.5, [0.4, 0.8]
    )
    assert categories.tolist() == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="unordered"):
        assign_categories(np.array([0.8]), np.array([0.5]), 0.5, [0.7, 0.6])


def test_selection_support_and_frozen_evaluation_validity_are_separate() -> None:
    projection, direct, _, projection_categories, direct_categories = sparse_synthetic_inputs()
    selected, _ = evaluate_categories(
        projection, direct, projection_categories, direct_categories, [0.1, 0.09, 0.11], 0.05,
    )
    evaluated, _ = evaluate_categories(
        projection, direct, projection_categories, direct_categories, [0.1, 0.09, 0.11], 0.05,
        require_category_support_for_systematic=False,
    )
    assert not selected["support_pass"]
    assert np.isnan(selected["combined_systematic_aware_asimov_ZA"])
    assert not evaluated["support_pass"]
    assert evaluated["systematic_evaluation_pass"]
    assert np.isfinite(evaluated["combined_systematic_aware_asimov_ZA"])


def test_bootstrap_resamples_sources_and_does_not_censor_support_fluctuations() -> None:
    projection, direct, members, projection_categories, direct_categories = sparse_synthetic_inputs()
    combined, categories = bootstrap_fixed_categories(
        projection, direct, projection_categories, direct_categories, members,
        np.array([[1, 1, 1], [2, 1, 1]], dtype=np.int16), [0.1, 0.09, 0.11], 0.05, label="synthetic",
    )
    assert combined.replica_id.tolist() == [0, 1]
    assert combined.valid.all()
    assert np.isfinite(combined.combined_systematic_aware_asimov_ZA).all()
    assert not categories.support_pass.all()
    assert np.isfinite(categories.systematic_aware_ZA_standalone_contribution).all()


def test_outer_selected_predictions_use_each_held_folds_own_inner_choice() -> None:
    folds = np.arange(5, dtype=np.int8)
    inputs = {"training": pd.DataFrame({"registry_oof_fold": folds}),
              "projection": pd.DataFrame({"registry_oof_fold": folds})}
    predictions = {}
    rows = []
    for variant in VARIANTS:
        predictions[variant] = {}
        for count in range(2, 7):
            predictions[variant][count] = {
                "train_stage1": np.full(5, count, dtype=float), "train_stage2": np.full(5, count + 0.1),
                "train_category": np.full(5, count - 1, dtype=np.int16),
                "projection_stage1": np.full(5, count, dtype=float),
                "projection_stage2": np.full(5, count + 0.1),
                "projection_category": np.full(5, count - 1, dtype=np.int16),
            }
        for fold, count in enumerate((2, 3, 4, 5, 6)):
            rows.append({"variant": variant, "outer_fold": fold, "selection_role": "outer_selected",
                         "category_count": count})
    selected, counts = build_outer_selected_predictions(
        inputs, {"predictions": predictions, "selections": pd.DataFrame(rows)}
    )
    for variant in VARIANTS:
        assert selected[variant]["projection_stage1"].tolist() == [2, 3, 4, 5, 6]
        assert counts[variant] == {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}


def test_invalid_replicas_are_preserved_and_paired_quantiles_are_asymmetric() -> None:
    frame = pd.DataFrame({"label": ["x"] * 4, "metric_value": [0.0, 1.0, 4.0, np.nan],
                          "failure_reason": ["", "", "", "undefined nuisance"]})
    summary = summarize_replicas(frame, ["metric_value"], ["label"]).iloc[0]
    assert summary.valid_replicas == 3
    assert summary.invalid_replicas == 1
    assert summary.bootstrap_p84 - summary.bootstrap_median != pytest.approx(
        summary.bootstrap_median - summary.bootstrap_p16
    )
    paired = paired_difference_summary(np.array([-1.0, 0.2, 2.5, np.nan]), ["", "", "", "undefined"])
    assert paired["valid_replicas"] == 3
    assert paired["invalid_replicas"] == 1
    assert 0.0 < paired["fraction_valid_difference_greater_than_zero"] < 1.0


def test_plot_source_rows_always_expose_uncertainty_and_support_fields() -> None:
    row = standardize_source_rows([{"central_value": 1.0}])[0]
    assert {"uncertainty_central", "uncertainty_lower_68", "uncertainty_upper_68",
            "uncertainty_valid_replicas", "uncertainty_support_flag", "uncertainty_kind"}.issubset(row)


def test_review_finalizer_requires_every_full_1000_replica_figure_review(tmp_path: Path) -> None:
    report = REPO / "docs/paper/jhep_hh4b_ml/c7w_visual_inspection_report.tsv"
    rows = read_inspection_report(report)
    assert {row["asset"] for row in rows} == EXPECTED_FIGURES
    malformed = tmp_path / "review.tsv"
    malformed.write_text(report.read_text().replace("full_review_pass", "pending", 1))
    with pytest.raises(RuntimeError, match="not fully reviewed"):
        read_inspection_report(malformed)


def test_review_finalizer_fails_closed_on_source_manifest_drift(tmp_path: Path) -> None:
    source = tmp_path / "review_source"
    source.mkdir()
    (source / "SHA256SUMS").write_text("")
    with pytest.raises(RuntimeError, match="manifest identity drift"):
        verify_source(source, "0" * 64)


def test_c7w_code_has_no_validation_test_payload_or_push_command_and_git_has_no_reference_pdf() -> None:
    sources = [SCRIPT_DIR / "pn_c7w_categorized_bdt_common.py",
               SCRIPT_DIR / "run_pn_c7w_nested_categorized_bdt.py",
               SCRIPT_DIR / "finalize_pn_c7w_reviewed_checkpoint.py"]
    text = "\n".join(path.read_text().lower() for path in sources)
    assert "git push" not in text
    assert "validation.parquet" not in text
    assert "test.parquet" not in text
    tracked = subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).splitlines()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO, text=True
    ).splitlines()
    candidates = tracked + [line[3:] for line in status]
    assert not any(path.lower().endswith(".pdf") and "hig-24-015" in path.lower() for path in candidates)
