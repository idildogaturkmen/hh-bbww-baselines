#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_tsv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(
        newline="",
        encoding="utf-8",
        errors="strict",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    require(fields, f"{path}: missing TSV header")
    return fields, rows


def markdown_rows(
    rows: list[dict[str, str]],
) -> str:
    lines = [
        "| Point | "
        r"\(p_{\mathrm{T}}^{\min}\) [GeV] | "
        r"\(|\eta|^{\max}\) | "
        r"\(R_{HH}^{\max}\) | "
        "Signal | Background | "
        r"Balanced \(S/\sqrt{B}\)-like proxy | "
        r"Balanced \(S/B\)-like proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"`{row['point_id']}` | "
            f"{float(row['jet_pt_min_GeV']):.1f} | "
            f"{float(row['jet_abs_eta_max']):.2f} | "
            f"{float(row['rhh_sr_max']):.1f} | "
            f"{int(row['selected_signal']):,} | "
            f"{int(row['selected_background']):,} | "
            f"{float(row['balanced_proxy_s_over_sqrt_b']):.6f} | "
            f"{float(row['balanced_proxy_s_over_b']):.6f} |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--named-points",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--shortlist",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--plateau",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--source-commit",
        required=True,
    )

    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo = script_path.parents[2]

    def resolve(path: Path) -> Path:
        if path.is_absolute():
            return path.resolve()
        return (repo / path).resolve()

    summary_path = resolve(args.summary)
    named_path = resolve(args.named_points)
    shortlist_path = resolve(args.shortlist)
    plateau_path = resolve(args.plateau)
    output_dir = resolve(args.output_dir)

    require(
        re.fullmatch(
            r"[0-9a-f]{40}",
            args.source_commit.lower(),
        )
        is not None,
        "source commit must be a full SHA",
    )

    require(
        not output_dir.exists(),
        f"refusing to overwrite {output_dir}",
    )

    temp = output_dir.with_name(
        output_dir.name + "_incomplete"
    )

    require(
        not temp.exists(),
        f"temporary directory exists: {temp}",
    )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8",
        )
    )

    require(
        summary.get("status")
        == "hh4b_train_only_fine_cut_scan_pass",
        "fine scan did not pass",
    )
    require(
        summary.get("grid_points") == 783,
        "fine grid does not contain 783 points",
    )
    require(
        summary.get("validation_rows_evaluated") == 0,
        "validation was used",
    )
    require(
        summary.get("test_candidate_files_opened") == 0,
        "test candidates were accessed",
    )
    require(
        summary.get("physical_significance_computed")
        is False,
        "physical significance was computed",
    )

    _, named_rows = read_tsv(named_path)
    _, shortlist_rows = read_tsv(shortlist_path)
    _, plateau_rows = read_tsv(plateau_path)

    require(
        len(named_rows) == 3,
        f"expected 3 named points, found {len(named_rows)}",
    )
    require(
        len(shortlist_rows) == 11,
        f"expected 11 shortlist points, found {len(shortlist_rows)}",
    )
    require(
        len(plateau_rows) == 13,
        f"expected 13 plateau points, found {len(plateau_rows)}",
    )

    best = summary["best_point"]
    coarse = summary["coarse_anchor_point"]
    reference = summary["reference_point"]

    require(
        best["point_id"] == "pt30_eta2p5_rhh34",
        f"unexpected fine optimum: {best['point_id']}",
    )
    require(
        coarse["point_id"] == best["point_id"],
        "fine and coarse optima differ unexpectedly",
    )
    require(
        math.isclose(
            summary["proxy_change_from_coarse_percent"],
            0.0,
            abs_tol=1.0e-12,
        ),
        "fine proxy changed from coarse",
    )

    temp.mkdir(
        parents=True,
        exist_ok=False,
    )

    for source, destination_name in (
        (named_path, "fine_named_points.tsv"),
        (shortlist_path, "fine_scan_shortlist.tsv"),
        (plateau_path, "fine_scan_plateau.tsv"),
    ):
        shutil.copy2(
            source,
            temp / destination_name,
        )

    svg_paths = sorted({
        Path(path).resolve()
        for path in summary["plots"]
        if Path(path).suffix.lower() == ".svg"
    })

    require(
        len(svg_paths) == 7,
        f"expected 7 SVG figures, found {len(svg_paths)}",
    )

    copied_figures: list[str] = []

    for source in svg_paths:
        require(
            source.is_file(),
            f"missing SVG figure: {source}",
        )
        shutil.copy2(
            source,
            temp / source.name,
        )
        copied_figures.append(source.name)

    checkpoint = {
        "schema_version": 1,
        "status":
            "hh4b_fine_cut_scan_checkpoint",
        "classification":
            "provisional_train_only_development_result",
        "source_commit":
            args.source_commit.lower(),
        "source_summary":
            str(summary_path),
        "source_summary_sha256":
            sha256_file(summary_path),
        "source_cache":
            summary["source_cache"],
        "source_cache_sha256":
            summary["source_cache_sha256"],
        "configuration":
            summary["configuration"],
        "configuration_sha256":
            summary["configuration_sha256"],
        "grid_points":
            summary["grid_points"],
        "pareto_frontier_points":
            summary["pareto_frontier_points"],
        "near_optimal_plateau_points":
            summary["near_optimal_plateau_points"],
        "shortlist_points":
            summary["shortlist_points"],
        "fine_proxy_optimum":
            best,
        "coarse_optimum_reproduced":
            True,
        "proxy_change_from_coarse_percent":
            summary["proxy_change_from_coarse_percent"],
        "proxy_change_from_reference_percent":
            summary["proxy_change_from_reference_percent"],
        "validation_rows_evaluated":
            0,
        "test_candidate_files_opened":
            0,
        "physical_background_normalization_frozen":
            False,
        "physical_significance_computed":
            False,
        "figures":
            copied_figures,
        "next_gate":
            "exact_observed_rhh_threshold_scan",
    }

    (
        temp / "checkpoint.json"
    ).write_text(
        json.dumps(
            checkpoint,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    shortlist_markdown = markdown_rows(
        shortlist_rows
    )

    readme = f"""# HH4b fine cut-scan checkpoint

## Status

This checkpoint records a **provisional train-only fine-scan result**. It is not yet the final physical optimized cut baseline.

- Grid shape: 9 jet-threshold values × 3 eta values × 29 RHH values
- Grid points: {summary['grid_points']}
- Pareto-frontier points: {summary['pareto_frontier_points']}
- Points within 1% of the optimum: {summary['near_optimal_plateau_points']}
- Shortlisted Pareto points: {summary['shortlist_points']}
- Validation rows evaluated: 0
- Test candidate files opened: 0
- Physical significance computed: no
- Source implementation commit: `{args.source_commit.lower()}`

## Fine proxy optimum

\\[
p_{{\\mathrm{{T}}}}^{{\\min}}=30~\\mathrm{{GeV}},
\\qquad
|\\eta|^{{\\max}}=2.5,
\\qquad
R_{{HH}}<34.
\\]

The train-only family- and mode-balanced proxy is

\\[
\\frac{{\\epsilon_S^{{\\mathrm{{bal}}}}}}
{{\\sqrt{{\\epsilon_B^{{\\mathrm{{fam}}}}}}}}
=
{float(best['balanced_proxy_s_over_sqrt_b']):.6f}.
\\]

The fine scan reproduces the coarse optimum exactly, with a proxy change of {summary['proxy_change_from_coarse_percent']:+.3f}% from the coarse result and {summary['proxy_change_from_reference_percent']:+.2f}% from the fixed reference point.

## Main interpretation

1. Raising the common jet threshold above 30 GeV reduces the best achievable proxy. No additional common jet-\\(p_{{\\mathrm{{T}}}}\\) cut is favored beyond the reconstruction requirement.
2. The proxy improves monotonically from \\(|\\eta|<2.4\\) to \\(|\\eta|<2.5\\), favoring the full reconstructed acceptance.
3. The meaningful interior optimization is in \\(R_{{HH}}\\). The global fine-grid maximum remains at 34.
4. The objective is step-like because the selected sample changes only when a threshold crosses an observed event value.
5. Thirteen points lie within 1% of the maximum. This supports a stable efficiency–purity plateau rather than a uniquely determined physical working point.
6. The final point may move after process cross sections, QCD overlap removal, finite-MC uncertainty, and background systematics are included.

## Named-point comparison

| Quantity | Reference | Coarse/fine proxy optimum |
|---|---:|---:|
| Point | `{reference['point_id']}` | `{best['point_id']}` |
| Selected signal rows | {int(reference['selected_signal']):,} | {int(best['selected_signal']):,} |
| Selected background rows | {int(reference['selected_background']):,} | {int(best['selected_background']):,} |
| Balanced signal efficiency | {float(reference['balanced_signal_efficiency']):.6f} | {float(best['balanced_signal_efficiency']):.6f} |
| Family-balanced background efficiency | {float(reference['family_balanced_background_efficiency']):.6f} | {float(best['family_balanced_background_efficiency']):.6f} |
| Balanced \\(S/\\sqrt{{B}}\\)-like proxy | {float(reference['balanced_proxy_s_over_sqrt_b']):.6f} | {float(best['balanced_proxy_s_over_sqrt_b']):.6f} |
| Balanced \\(S/B\\)-like proxy | {float(reference['balanced_proxy_s_over_b']):.6f} | {float(best['balanced_proxy_s_over_b']):.6f} |

## Fine-grid shortlist

{shortlist_markdown}

## Figures

- [Best proxy versus jet threshold](fine_best_proxy_vs_jet_pt_min_GeV.svg)
- [Best proxy versus eta acceptance](fine_best_proxy_vs_jet_abs_eta_max.svg)
- [Best proxy versus RHH threshold](fine_best_proxy_vs_rhh_sr_max.svg)
- [Heatmap for eta 2.4](fine_proxy_heatmap_eta2p4.svg)
- [Heatmap for eta 2.45](fine_proxy_heatmap_eta2p45.svg)
- [Heatmap for eta 2.5](fine_proxy_heatmap_eta2p5.svg)
- [Fine-grid Pareto plane](fine_scan_pareto.svg)

## Reproducibility

- Cache SHA-256: `{summary['source_cache_sha256']}`
- Configuration SHA-256: `{summary['configuration_sha256']}`
- Fine summary SHA-256: `{sha256_file(summary_path)}`
- Fine shortlist SHA-256: `{sha256_file(shortlist_path)}`
- Fine plateau SHA-256: `{sha256_file(plateau_path)}`

The event-level cache is intentionally not committed.

## Next gate

Fix the acceptance at the reconstruction-level values

\\[
p_{{\\mathrm{{T}}}}^{{\\min}}=30~\\mathrm{{GeV}},
\\qquad
|\\eta|^{{\\max}}=2.5,
\\]

and scan every distinct train-sample \\(R_{{HH}}\\) transition in the fine plateau. This provides the exact discrete equivalent of a continuous threshold optimization.

Physical ranking, validation confirmation, and test evaluation remain separate later gates.
"""

    (
        temp / "README.md"
    ).write_text(
        readme,
        encoding="utf-8",
    )

    checksum_paths = sorted(
        path
        for path in temp.iterdir()
        if path.is_file()
        and path.name != "SHA256SUMS"
    )

    (
        temp / "SHA256SUMS"
    ).write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in checksum_paths
        )
        + "\n",
        encoding="utf-8",
    )

    temp.rename(output_dir)

    print("HH4B_FINE_CUT_SCAN_CHECKPOINT_PASS")
    print(
        "HH4B_FINE_PROXY_OPTIMUM="
        "pt30_eta2p5_rhh34"
    )
    print(
        "HH4B_COARSE_OPTIMUM_REPRODUCED=True"
    )
    print(
        "HH4B_NEAR_OPTIMAL_PLATEAU_POINTS=13"
    )
    print("HH4B_SVG_FIGURES_COPIED=7")
    print(
        "HH4B_PHYSICAL_SIGNIFICANCE_COMPUTED=False"
    )
    print("HH4B_VALIDATION_ROWS_EVALUATED=0")
    print("HH4B_TEST_CANDIDATE_FILES_OPENED=0")
    print(
        "NEXT_GATE="
        "exact_observed_rhh_threshold_scan"
    )
    print(f"checkpoint={output_dir}")


if __name__ == "__main__":
    main()
