#!/usr/bin/env python3
"""Seal the visually reviewed c7x package without repeating model training."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from pn_c7_ml_common import (
    artifact_rows,
    prepare_staging,
    require,
    seal_checkpoint,
    sha256,
    write_json,
    write_tsv,
)


EXPECTED_REVIEW_ARTIFACT_MANIFEST_SHA256 = (
    "74cdd54fd17bbbefa52483ce9ec1f7987a592b600f4a9fabf0afe2d4e0c34b0f"
)
EXPECTED_COMPILE_AUX_SHA256 = "c2bd959c0f4a5c9ae506dfb0625b5446550f331e19e5ac51f14bda19cb7de79b"
EXPECTED_COMPILE_LOG_SHA256 = "56b6c5d30a639ab6d28a3a56f9a55361bb73f4111668e1148b4175b63880f033"
EXPECTED_COMPILE_PDF_SHA256 = "1d5c4d81e11d581c375ef536ea6105f1afb009075a82508022102e35aef17a67"
EXPECTED_BOOTSTRAP_REGISTRY_SHA256 = (
    "37218e5531018ea2a79473bfbee43df657b309cf06d89751509e74b586d9dc29"
)
EXPECTED_FIGURES = {
    f"fig_c7x_{index:02d}_{suffix}"
    for index, suffix in (
        (1, "weighted_roc"),
        (2, "unweighted_roc"),
        (3, "efficiency_rejection"),
        (4, "weighted_auc"),
        (5, "sensitivity"),
        (6, "sb_neff"),
        (7, "background_absolute"),
        (8, "background_fraction"),
        (9, "pairing_accuracy"),
        (10, "pairing_binned"),
        (11, "mass_resolution"),
        (12, "selection_stability"),
    )
}
PRIMARY_CLASSIFICATION_METRICS = {
    "weighted_auc",
    "nominal_asimov_ZA",
    "systematic_aware_asimov_ZA",
    "signal_over_background",
    "background_effective_events",
}
SOURCE_EXTRAS = {
    "artifact_manifest.tsv",
    "REVIEW_REQUIRED",
    "paper/compile_fragments.aux",
    "paper/compile_fragments.log",
    "paper/compile_fragments.pdf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--source-artifact-manifest-sha256",
        default=EXPECTED_REVIEW_ARTIFACT_MANIFEST_SHA256,
    )
    parser.add_argument("--inspection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-output", type=Path, required=True)
    parser.add_argument("--repo-paper-output", type=Path, required=True)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"required TSV missing: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_inspection_report(path: Path) -> list[dict[str, str]]:
    rows = read_tsv(path)
    require(len(rows) == len(EXPECTED_FIGURES), "visual-inspection report must cover all 12 figures")
    require({row["asset"] for row in rows} == EXPECTED_FIGURES, "visual-inspection figure inventory drift")
    for row in rows:
        asset = row["asset"]
        require(row["status"] == "full_review_pass", f"figure not fully reviewed: {asset}")
        require(
            row["inspection_basis"] == "full_1000_replica_png_and_pdf_raster",
            f"figure review basis drift: {asset}",
        )
        for field in ("clipping", "missing_glyphs", "legend_overlap"):
            require(row[field] == "none", f"figure review failure ({field}): {asset}")
        for field in ("scales", "source_consistency"):
            require(row[field] == "pass", f"figure review failure ({field}): {asset}")
    return rows


def require_bootstrap_intervals(rows: list[dict[str, str]], label: str) -> None:
    require(rows, f"empty uncertainty table: {label}")
    required = {"bootstrap_median", "bootstrap_p16", "bootstrap_p84", "valid_replicas", "invalid_replicas"}
    require(required.issubset(rows[0]), f"{label} lacks bootstrap interval fields")
    for row in rows:
        for field in ("bootstrap_median", "bootstrap_p16", "bootstrap_p84"):
            value = row[field].strip().lower()
            require(value not in {"", "nan", "none"}, f"{label} has undefined {field}")
        total = int(float(row["valid_replicas"])) + int(float(row["invalid_replicas"]))
        require(total == 1000, f"{label} replica accounting drift")


def require_paired_intervals(rows: list[dict[str, str]], label: str) -> None:
    require(rows, f"empty paired-difference table: {label}")
    required = {
        "bootstrap_median_difference",
        "bootstrap_p16_difference",
        "bootstrap_p84_difference",
        "valid_replicas",
        "invalid_replicas",
    }
    require(required.issubset(rows[0]), f"{label} lacks paired interval fields")
    for row in rows:
        for field in (
            "bootstrap_median_difference",
            "bootstrap_p16_difference",
            "bootstrap_p84_difference",
        ):
            value = row[field].strip().lower()
            require(value not in {"", "nan", "none"}, f"{label} has undefined {field}")
        total = int(float(row["valid_replicas"])) + int(float(row["invalid_replicas"]))
        require(total == 1000, f"{label} replica accounting drift")


def verify_uncertainty_contract(source: Path) -> None:
    classification = read_tsv(source / "bootstrap_metric_summary.tsv")
    primary = [
        row for row in classification
        if row["model"] in {"bdt", "single_head"}
        and row["metric"] in PRIMARY_CLASSIFICATION_METRICS
    ]
    require(len(primary) == 2 * len(PRIMARY_CLASSIFICATION_METRICS),
            "primary classification uncertainty inventory drift")
    require_bootstrap_intervals(primary, "primary classification summary")

    classification_paired = [
        row for row in read_tsv(source / "paired_model_differences.tsv")
        if row["comparison"] == "single_head_minus_bdt"
        and row["metric"] in PRIMARY_CLASSIFICATION_METRICS
    ]
    require(len(classification_paired) == len(PRIMARY_CLASSIFICATION_METRICS),
            "primary classification paired-comparison inventory drift")
    require_paired_intervals(classification_paired, "primary classification comparisons")

    require_bootstrap_intervals(
        read_tsv(source / "assignment_bootstrap_summary.tsv"),
        "assignment summary",
    )
    require_paired_intervals(
        read_tsv(source / "assignment_paired_differences.tsv"),
        "assignment comparisons",
    )
    require_bootstrap_intervals(
        read_tsv(source / "binned_pairing_bootstrap_summary.tsv"),
        "binned assignment summary",
    )
    require_paired_intervals(
        read_tsv(source / "binned_pairing_paired_differences.tsv"),
        "binned assignment comparisons",
    )

    figure_rows = read_tsv(source / "paper_figure_manifest.tsv")
    require({row["asset_id"] for row in figure_rows} == EXPECTED_FIGURES,
            "paper figure manifest inventory drift")
    for row in figure_rows:
        require(
            row["uncertainty_contract"]
            == "source_member_bootstrap_68_percent_or_selection_stability_as_captioned",
            f"figure uncertainty contract drift: {row['asset_id']}",
        )


def verify_review_source(source: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    require(source.is_absolute() and source.is_dir(), f"review source missing: {source}")
    require(not source.is_symlink(), "review source may not be a symlink")
    manifest = source / "artifact_manifest.tsv"
    require(sha256(manifest) == expected_manifest_sha256, "review-source artifact manifest identity drift")
    rows = read_tsv(manifest)
    require(rows, "empty review-source artifact manifest")
    manifested: set[str] = set()
    for row in rows:
        relative = row["path"]
        rel = Path(relative)
        require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe artifact member: {relative}")
        require(relative not in manifested, f"duplicate artifact member: {relative}")
        manifested.add(relative)
        member = source / rel
        require(member.is_file() and not member.is_symlink(), f"review artifact missing or linked: {relative}")
        require(member.stat().st_size == int(row["bytes"]), f"review artifact size drift: {relative}")
        require(sha256(member) == row["sha256"], f"review artifact checksum drift: {relative}")

    actual = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
    }
    require(actual == manifested | SOURCE_EXTRAS, "review-source file inventory drift")
    require(
        (source / "REVIEW_REQUIRED").read_text().strip()
        == "visual_and_latex_review_required_before_checkpoint_sealing",
        "review marker drift",
    )
    compiled = source / "paper"
    require(sha256(compiled / "compile_fragments.aux") == EXPECTED_COMPILE_AUX_SHA256,
            "compiled LaTeX auxiliary identity drift")
    require(sha256(compiled / "compile_fragments.log") == EXPECTED_COMPILE_LOG_SHA256,
            "compiled LaTeX log identity drift")
    require(sha256(compiled / "compile_fragments.pdf") == EXPECTED_COMPILE_PDF_SHA256,
            "compiled LaTeX PDF identity drift")
    log_text = (compiled / "compile_fragments.log").read_text()
    for forbidden in ("Overfull", "Underfull", "LaTeX Warning", "Undefined control sequence"):
        require(forbidden not in log_text, f"compiled LaTeX contains {forbidden}")

    summary = json.loads((source / "summary.json").read_text())
    require(summary["status"] == "pn_c7x_single_head_review_package_pass", "review-source status drift")
    require(summary["replicas"] == 1000, "review-source replica count drift")
    require(summary["common_bootstrap_registry_sha256"] == EXPECTED_BOOTSTRAP_REGISTRY_SHA256,
            "review-source bootstrap registry drift")
    require(summary["figure_count"] == 12 and summary["table_count"] == 4,
            "review-source paper inventory drift")
    require(summary["review_required_before_sealing"] is True, "review gate was not active")
    for field in ("validation_payload_files_opened", "test_or_evaluation_payload_files_opened"):
        require(summary[field] == 0, f"review-source train-only contract failed: {field}")
    require(summary["observed_data_opened"] is False, "review-source observed-data contract failed")
    require(summary["full_run2_prediction"] is False, "review-source scope drift")
    verify_uncertainty_contract(source)
    return summary


def copy_review_source(source: Path, staging: Path) -> None:
    excluded = {"artifact_manifest.tsv", "README.md", "RUN_CONTRACT.txt", "summary.json", "REVIEW_REQUIRED"}
    for member in sorted(source.iterdir()):
        if member.name in excluded:
            continue
        require(not member.is_symlink(), f"refusing review-source symlink: {member}")
        destination = staging / member.name
        if member.name == "paper":
            shutil.copytree(
                member,
                destination,
                ignore=shutil.ignore_patterns("compile_fragments.aux", "compile_fragments.log"),
            )
        elif member.is_dir():
            shutil.copytree(member, destination)
        else:
            shutil.copy2(member, destination)


def make_writable(root: Path) -> None:
    for member in sorted(root.rglob("*"), reverse=True):
        member.chmod(0o644 if member.is_file() else 0o755)
    root.chmod(0o755)


def finalize(
    source: Path,
    inspection_report: Path,
    output: Path,
    paper_output: Path,
    repo_paper_output: Path,
    *,
    source_artifact_manifest_sha256: str,
) -> str:
    require(len({output, paper_output, repo_paper_output}) == 3, "output paths must differ")
    for path in (output, paper_output, repo_paper_output):
        require(path.is_absolute(), f"output must be absolute: {path}")
        require(path.parent.is_dir(), f"output parent missing: {path.parent}")
        require(not path.exists(), f"refusing output overwrite: {path}")

    inspection_rows = read_inspection_report(inspection_report)
    summary = verify_review_source(source, source_artifact_manifest_sha256)
    staging = prepare_staging(output)
    copy_review_source(source, staging)
    make_writable(staging)

    write_tsv(staging / "visual_inspection_report.tsv", inspection_rows)
    write_tsv(staging / "paper" / "visual_inspection_report.tsv", inspection_rows)
    paper_readme = staging / "paper" / "README.md"
    paper_readme.write_text(
        paper_readme.read_text().rstrip()
        + "\n\nAll 12 PNG and vector-PDF figures passed the final 1000-replica visual review. "
        f"Reviewed artifact manifest: {source_artifact_manifest_sha256}.\n"
    )
    summary.update({
        "schema_version": 2,
        "status": "pn_c7x_single_head_spanet_reviewed_complete",
        "review_source": str(source),
        "review_source_artifact_manifest_sha256": source_artifact_manifest_sha256,
        "visual_inspection_status": "full_review_pass",
        "visual_inspection_figures": len(inspection_rows),
        "review_required_before_sealing": False,
    })
    write_json(staging / "summary.json", summary)
    (staging / "README.md").write_text(
        "# c7x train-only single-head SPA-Net: reviewed checkpoint\n\n"
        "Authoritative checksum-sealed finalization of the nested source-group OOF study. "
        "All 12 PNG and vector-PDF figures passed visual review after 1000 common "
        "source-member replicas; no validation/test payload or observed data was opened.\n"
    )
    source_contract = (source / "RUN_CONTRACT.txt").read_text().rstrip()
    (staging / "RUN_CONTRACT.txt").write_text(
        source_contract
        + "\n"
        + f"finalize_command={sys.executable} {' '.join(sys.argv)}\n"
        + f"review_source={source}\n"
        + f"review_source_artifact_manifest_sha256={source_artifact_manifest_sha256}\n"
        + f"visual_inspection_report_sha256={sha256(inspection_report)}\n"
        + "finalization_retrained_models=0\nremote_updated=0\n"
    )
    write_tsv(
        staging / "artifact_manifest.tsv",
        artifact_rows(staging, exclude={"artifact_manifest.tsv", "SHA256SUMS", "COMPLETE"}),
    )
    manifest_sha = seal_checkpoint(
        staging,
        output,
        "pn_c7x train-only single-head SPA-Net reviewed complete",
    )

    shutil.copytree(output / "paper", paper_output)
    shutil.copytree(output / "paper", repo_paper_output)
    make_writable(repo_paper_output)
    for member in sorted(paper_output.rglob("*"), reverse=True):
        member.chmod(0o444 if member.is_file() else 0o555)
    paper_output.chmod(0o555)
    return manifest_sha


def main() -> None:
    args = parse_args()
    manifest_sha = finalize(
        args.source.resolve(),
        args.inspection_report.resolve(),
        args.output.resolve(),
        args.paper_output.resolve(),
        args.repo_paper_output.resolve(),
        source_artifact_manifest_sha256=args.source_artifact_manifest_sha256,
    )
    print(json.dumps({
        "manifest_sha256": manifest_sha,
        "models_retrained": 0,
        "output": str(args.output.resolve()),
        "paper_output": str(args.paper_output.resolve()),
        "repo_paper_output": str(args.repo_paper_output.resolve()),
        "source_artifact_manifest_sha256": args.source_artifact_manifest_sha256,
        "visual_inspection_status": "full_review_pass",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
