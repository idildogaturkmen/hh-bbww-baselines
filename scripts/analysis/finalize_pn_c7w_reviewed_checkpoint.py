#!/usr/bin/env python3
"""Seal the visually reviewed c7w checkpoint without repeating model training."""

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
    parse_sha256sums,
    prepare_staging,
    require,
    seal_checkpoint,
    sha256,
    write_json,
    write_tsv,
)


EXPECTED_SOURCE_MANIFEST_SHA256 = "dfd7c1423f609ef521af2db3210973c0efc1bf5ce78743a9ee7ee7f8390e4210"
EXPECTED_BOOTSTRAP_REGISTRY_SHA256 = "37218e5531018ea2a79473bfbee43df657b309cf06d89751509e74b586d9dc29"
EXPECTED_FIGURES = {
    "fig_c7w_01_selected_score_plane",
    "fig_c7w_02_population_score_densities",
    "fig_c7w_03_selected_and_benchmark_boundaries",
    "fig_c7w_04_category_yields",
    "fig_c7w_05_category_signal_over_background",
    "fig_c7w_06_category_background_neff",
    "fig_c7w_07_category_nominal_za",
    "fig_c7w_08_category_systematic_za",
    "fig_c7w_09_inclusive_categorized_comparison",
    "fig_c7w_09b_paired_differences",
    "fig_c7w_10_category_count_scan",
    "fig_c7w_11_category_support",
    "fig_c7w_12_mass_sculpting",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", default=EXPECTED_SOURCE_MANIFEST_SHA256)
    parser.add_argument("--inspection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-output", type=Path, required=True)
    return parser.parse_args()


def read_inspection_report(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"visual-inspection report missing: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(len(rows) == len(EXPECTED_FIGURES), "visual-inspection report must cover all 13 figures")
    require({row["asset"] for row in rows} == EXPECTED_FIGURES, "visual-inspection figure inventory drift")
    for row in rows:
        require(row["status"] == "full_review_pass", f"figure not fully reviewed: {row['asset']}")
        require(row["inspection_basis"] == "full_1000_replica_png_and_pdf_raster",
                f"figure review basis drift: {row['asset']}")
        for field in ("clipping", "missing_glyphs", "legend_overlap"):
            require(row[field] == "none", f"figure review failure ({field}): {row['asset']}")
        for field in ("scales", "source_consistency"):
            require(row[field] == "pass", f"figure review failure ({field}): {row['asset']}")
    return rows


def verify_source(source: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    require(source.is_absolute() and source.is_dir(), f"source checkpoint missing: {source}")
    require(not source.is_symlink(), "source checkpoint may not be a symlink")
    manifest = source / "SHA256SUMS"
    require(manifest.is_file(), f"source SHA256SUMS missing: {manifest}")
    require(sha256(manifest) == expected_manifest_sha256, "review-source manifest identity drift")
    entries = parse_sha256sums(manifest)
    require(entries, "empty review-source manifest")
    for relative, expected in entries.items():
        rel = Path(relative)
        require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe manifest member: {relative}")
        member = source / rel
        require(member.is_file() and not member.is_symlink(), f"source member missing or linked: {relative}")
        require(sha256(member) == expected, f"source member checksum drift: {relative}")
    require((source / "COMPLETE").read_text().strip() == "pn_c7w nested categorized BDT complete",
            "review-source completion marker drift")
    summary = json.loads((source / "summary.json").read_text())
    require(summary["status"] == "pn_c7w_nested_categorized_bdt_complete", "review-source status drift")
    require(summary["smoke"] is False, "refusing to finalize a smoke run")
    require(summary["bootstrap_replicates"] == 1000, "review-source replica count drift")
    require(summary["figures"] == 13 and summary["tables"] == 6, "review-source paper inventory drift")
    require(summary["common_bootstrap_registry_sha256"] == EXPECTED_BOOTSTRAP_REGISTRY_SHA256,
            "review-source bootstrap registry drift")
    for field in ("validation_payload_files_opened", "test_or_evaluation_payload_files_opened", "observed_data_opened"):
        require(summary[field] == 0, f"review-source train-only contract failed: {field}")
    require((source / "paper").is_dir(), "review-source paper package missing")
    return summary


def copy_verified_source(source: Path, staging: Path) -> None:
    excluded = {
        "paper", "SHA256SUMS", "COMPLETE", "artifact_manifest.tsv", "summary.json",
        "README.md", "RUN_CONTRACT.txt", "visual_inspection_report.tsv",
    }
    for member in sorted(source.iterdir()):
        if member.name in excluded:
            continue
        require(not member.is_symlink(), f"refusing source symlink: {member}")
        destination = staging / member.name
        if member.is_dir():
            shutil.copytree(member, destination)
        else:
            shutil.copy2(member, destination)
    shutil.copytree(source / "paper", staging / "paper")
    # The sealed review source is read-only.  Make only this private staging
    # copy writable before replacing the reviewed report and paper README.
    for member in sorted((staging / "paper").rglob("*"), reverse=True):
        member.chmod(0o644 if member.is_file() else 0o755)
    (staging / "paper").chmod(0o755)


def finalize(source: Path, inspection_report: Path, output: Path, paper_output: Path,
             *, source_manifest_sha256: str) -> str:
    require(output != paper_output, "checkpoint and paper output paths must differ")
    require(output.is_absolute() and paper_output.is_absolute(), "outputs must be absolute")
    require(paper_output.parent.is_dir(), f"paper-output parent missing: {paper_output.parent}")
    require(not paper_output.exists(), f"refusing paper-output overwrite: {paper_output}")
    inspection_rows = read_inspection_report(inspection_report)
    summary = verify_source(source, source_manifest_sha256)
    staging = prepare_staging(output)
    copy_verified_source(source, staging)

    write_tsv(staging / "visual_inspection_report.tsv", inspection_rows)
    write_tsv(staging / "paper" / "visual_inspection_report.tsv", inspection_rows)
    paper_readme = staging / "paper" / "README.md"
    paper_readme.write_text(
        paper_readme.read_text().rstrip() + "\n\n"
        f"All 13 PNG and PDF figures passed the final 1000-replica visual review. "
        f"Reviewed source manifest: `{source_manifest_sha256}`.\n"
    )
    summary.update({
        "schema_version": 2,
        "status": "pn_c7w_nested_categorized_bdt_reviewed_complete",
        "review_source_checkpoint": str(source),
        "review_source_manifest_sha256": source_manifest_sha256,
        "visual_inspection_status": "full_review_pass",
        "visual_inspection_figures": len(inspection_rows),
    })
    write_json(staging / "summary.json", summary)
    (staging / "README.md").write_text(
        "# c7w train-only nested categorized BDT: reviewed checkpoint\n\n"
        "Authoritative, checksum-sealed finalization of the full nested categorized-BDT run. "
        "All 13 PNG and PDF figures passed visual review after the 1000 common source-member replicas; "
        "no validation/test payload or observed data was opened.\n"
    )
    source_contract = (source / "RUN_CONTRACT.txt").read_text().rstrip()
    (staging / "RUN_CONTRACT.txt").write_text(
        source_contract + "\n"
        f"finalize_command={sys.executable} {' '.join(sys.argv)}\n"
        f"review_source={source}\nreview_source_manifest_sha256={source_manifest_sha256}\n"
        f"visual_inspection_report_sha256={sha256(inspection_report)}\n"
        "finalization_retrained_models=0\nremote_updated=0\n"
    )
    write_tsv(staging / "artifact_manifest.tsv", artifact_rows(
        staging, exclude={"artifact_manifest.tsv", "SHA256SUMS", "COMPLETE"},
    ))
    manifest_sha = seal_checkpoint(staging, output, "pn_c7w nested categorized BDT reviewed complete")
    shutil.copytree(output / "paper", paper_output)
    for member in sorted(paper_output.rglob("*"), reverse=True):
        member.chmod(0o444 if member.is_file() else 0o555)
    paper_output.chmod(0o555)
    return manifest_sha


def main() -> None:
    args = parse_args()
    manifest_sha = finalize(
        args.source.resolve(), args.inspection_report.resolve(), args.output.resolve(), args.paper_output.resolve(),
        source_manifest_sha256=args.source_manifest_sha256,
    )
    print(json.dumps({
        "output": str(args.output.resolve()), "paper_output": str(args.paper_output.resolve()),
        "manifest_sha256": manifest_sha, "source_manifest_sha256": args.source_manifest_sha256,
        "visual_inspection_status": "full_review_pass", "models_retrained": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
