from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO / "scripts" / "analysis"
FINAL_CHECKPOINT = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/"
    "pn_c7x_single_head_spanet_20260803_v1"
)
PAPER_PACKAGE = REPO / "docs/paper/jhep_hh4b_ml/single_head_spanet"
sys.path.insert(0, str(SCRIPT_DIR))

from finalize_pn_c7x_reviewed_checkpoint import (  # noqa: E402
    EXPECTED_FIGURES,
    read_inspection_report,
    require_bootstrap_intervals,
    require_paired_intervals,
    verify_uncertainty_contract,
)


def test_reviewed_package_covers_every_png_and_vector_pdf() -> None:
    rows = read_inspection_report(PAPER_PACKAGE / "visual_inspection_report.tsv")
    assert {row["asset"] for row in rows} == EXPECTED_FIGURES
    for asset in EXPECTED_FIGURES:
        assert (PAPER_PACKAGE / "figures" / f"{asset}.png").is_file()
        assert (PAPER_PACKAGE / "figures" / f"{asset}.pdf").is_file()


def test_visual_review_fails_closed_on_any_pending_asset(tmp_path: Path) -> None:
    source = PAPER_PACKAGE / "visual_inspection_report.tsv"
    malformed = tmp_path / "visual_inspection_report.tsv"
    malformed.write_text(source.read_text().replace("full_review_pass", "pending", 1))
    with pytest.raises(RuntimeError, match="not fully reviewed"):
        read_inspection_report(malformed)


def test_interval_contract_rejects_incomplete_replica_accounting() -> None:
    bootstrap = [{
        "bootstrap_median": "1.0",
        "bootstrap_p16": "0.8",
        "bootstrap_p84": "1.3",
        "valid_replicas": "999",
        "invalid_replicas": "0",
    }]
    with pytest.raises(RuntimeError, match="replica accounting drift"):
        require_bootstrap_intervals(bootstrap, "synthetic")
    paired = [{
        "bootstrap_median_difference": "0.1",
        "bootstrap_p16_difference": "-0.1",
        "bootstrap_p84_difference": "0.2",
        "valid_replicas": "1000",
        "invalid_replicas": "0",
    }]
    require_paired_intervals(paired, "synthetic paired")


def test_sealed_checkpoint_retains_every_primary_and_paired_uncertainty() -> None:
    verify_uncertainty_contract(FINAL_CHECKPOINT)


def test_c7x_finalization_has_no_validation_test_payload_push_or_reference_pdf() -> None:
    sources = [
        SCRIPT_DIR / "extract_pn_c7x_spanet_signal_truth.py",
        SCRIPT_DIR / "run_pn_c7x_single_head_spanet.py",
        SCRIPT_DIR / "publish_pn_c7x_single_head_spanet.py",
        SCRIPT_DIR / "finalize_pn_c7x_reviewed_checkpoint.py",
    ]
    text = "\n".join(path.read_text().lower() for path in sources)
    assert "git push" not in text
    assert "validation.parquet" not in text
    assert "test.parquet" not in text
    tracked = subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).splitlines()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO,
        text=True,
    ).splitlines()
    candidates = tracked + [line[3:] for line in status]
    assert not any(
        path.lower().endswith(".pdf") and "hig-24-015" in path.lower()
        for path in candidates
    )
