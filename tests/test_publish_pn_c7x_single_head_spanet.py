from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/analysis/publish_pn_c7x_single_head_spanet.py"
PYTHON = "/uscms_data/d3/iturkmen/venvs/hh4b-bdt-v1/bin/python"


def test_smoke_publication_has_paired_source_uncertainties_for_every_primary_value(tmp_path):
    output = tmp_path / "review"
    completed = subprocess.run(
        [PYTHON, str(SCRIPT), "--output", str(output), "--smoke"],
        cwd=REPO,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": f"{REPO}:{REPO / 'scripts/analysis'}",
            "MPLCONFIGDIR": str(tmp_path / "mpl"),
            "MPLBACKEND": "Agg",
        },
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout
    summary = pd.read_csv(output / "bootstrap_metric_summary.tsv", sep="\t")
    assert set(summary.model) == {"bdt", "single_head"}
    assert summary.groupby("model").metric.nunique().eq(16).all()
    non_systematic = summary.metric.ne("systematic_aware_asimov_ZA")
    assert summary.loc[non_systematic, "valid_replicas"].eq(20).all()
    assert summary.loc[~non_systematic, "valid_replicas"].between(1, 20).all()
    assert (summary.valid_replicas + summary.invalid_replicas).eq(20).all()
    assert summary[["bootstrap_p16", "bootstrap_median", "bootstrap_p84"]].notna().all().all()
    paired = pd.read_csv(output / "paired_model_differences.tsv", sep="\t")
    assert set(paired.comparison) == {"single_head_minus_bdt"}
    assert paired.metric.nunique() == 16
    assert paired.loc[paired.metric.ne("systematic_aware_asimov_ZA"), "valid_replicas"].eq(20).all()
    assert paired.loc[paired.metric.eq("systematic_aware_asimov_ZA"), "valid_replicas"].between(1, 20).all()
    assignment = pd.read_csv(output / "assignment_bootstrap_summary.tsv", sep="\t")
    assert assignment.valid_replicas.eq(20).all()
    assignment_paired = pd.read_csv(output / "assignment_paired_differences.tsv", sep="\t")
    assert assignment_paired.valid_replicas.eq(20).all()

    figures = pd.read_csv(output / "paper_figure_manifest.tsv", sep="\t")
    assert len(figures) == 12
    for row in figures.itertuples():
        assert (output / row.pdf).is_file()
        assert (output / row.png).is_file()
        source = pd.read_csv(output / row.source_data, sep="\t")
        if row.asset_id != "fig_c7x_12_selection_stability":
            columns = " ".join(source.columns).lower()
            assert "p16" in columns and "p84" in columns
            assert "valid" in columns


def test_publisher_never_opens_validation_test_or_mutates_remote():
    source = SCRIPT.read_text().lower()
    assert "validation.parquet" not in source
    assert "test.parquet" not in source
    assert '"validation_payload_files_opened":0' in source
    assert '"test_or_evaluation_payload_files_opened":0' in source
    assert "git push" not in source
    assert "observed_data_opened\":false" in source


def test_every_required_single_head_comparison_is_explicitly_paired():
    source = SCRIPT.read_text()
    for metric in (
        "weighted_auc",
        "nominal_asimov_ZA",
        "systematic_aware_asimov_ZA",
        "signal_over_background",
        "background_effective_events",
        "exact_event_pairing_accuracy",
    ):
        assert metric in source
    assert "paired_model_differences" in source
    assert "assignment_paired_differences" in source
    assert "COMMON_REGISTRY_SHA256" in source
