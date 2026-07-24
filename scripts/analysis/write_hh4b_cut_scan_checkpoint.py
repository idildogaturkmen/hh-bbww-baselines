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


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(
    path: Path,
) -> str:
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

    require(
        bool(fields),
        f"{path}: missing TSV header",
    )

    return fields, rows


def write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def resolve_path(
    repo: Path,
    value: Path,
) -> Path:
    if value.is_absolute():
        return value.resolve()

    return (repo / value).resolve()


def percentage_change(
    new: float,
    old: float,
) -> float:
    require(
        old != 0.0,
        "cannot calculate change relative to zero",
    )

    return 100.0 * (new / old - 1.0)


def markdown_shortlist(
    rows: list[dict[str, str]],
) -> str:
    lines = [
        "| Point | "
        r"\(p_{\mathrm{T}}^{\min}\) [GeV] | "
        r"\(|\eta|^{\max}\) | "
        r"\(R_{HH}^{\max}\) | "
        "Signal rows | Background rows | "
        "Balanced proxy |",
        "|---|---:|---:|---:|---:|---:|---:|",
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
            f"{float(row['balanced_proxy_s_over_sqrt_b']):.6f} |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a compact, versioned checkpoint "
            "from an HH4b cut-scan result."
        )
    )

    parser.add_argument(
        "--summary",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--shortlist",
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

    arguments = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo = script_path.parents[2]

    summary_path = resolve_path(
        repo,
        arguments.summary,
    )

    shortlist_path = resolve_path(
        repo,
        arguments.shortlist,
    )

    output_dir = resolve_path(
        repo,
        arguments.output_dir,
    )

    source_commit = arguments.source_commit.lower()

    require(
        re.fullmatch(
            r"[0-9a-f]{40}",
            source_commit,
        )
        is not None,
        "source commit must be a full 40-character SHA",
    )

    require(
        summary_path.is_file(),
        f"missing summary: {summary_path}",
    )

    require(
        shortlist_path.is_file(),
        f"missing shortlist: {shortlist_path}",
    )

    require(
        not output_dir.exists(),
        f"refusing to overwrite {output_dir}",
    )

    temporary_dir = output_dir.with_name(
        output_dir.name + "_incomplete"
    )

    require(
        not temporary_dir.exists(),
        f"temporary directory exists: {temporary_dir}",
    )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8",
        )
    )

    require(
        summary.get("status")
        == "hh4b_train_only_coarse_cut_scan_pass",
        "coarse scan did not pass",
    )

    require(
        summary.get("grid_points") == 512,
        "coarse scan does not contain 512 points",
    )

    require(
        summary.get(
            "validation_rows_evaluated"
        ) == 0,
        "validation was used in coarse optimization",
    )

    require(
        summary.get(
            "test_candidate_files_opened"
        ) == 0,
        "test candidate files were opened",
    )

    require(
        summary.get(
            "physical_significance_computed"
        ) is False,
        "physical significance was unexpectedly computed",
    )

    shortlist_fields, shortlist_rows = read_tsv(
        shortlist_path
    )

    require(
        len(shortlist_rows) == 6,
        (
            "expected six near-optimal shortlist "
            f"points, found {len(shortlist_rows)}"
        ),
    )

    best = summary["best_point"]
    reference = summary["reference_point"]

    require(
        best["point_id"]
        == "pt30_eta2p5_rhh34",
        (
            "unexpected coarse best point: "
            f"{best['point_id']}"
        ),
    )

    require(
        math.isclose(
            float(best["jet_pt_min_GeV"]),
            30.0,
            abs_tol=1.0e-12,
        ),
        "unexpected best pT threshold",
    )

    require(
        math.isclose(
            float(best["jet_abs_eta_max"]),
            2.5,
            abs_tol=1.0e-12,
        ),
        "unexpected best eta threshold",
    )

    require(
        math.isclose(
            float(best["rhh_sr_max"]),
            34.0,
            abs_tol=1.0e-12,
        ),
        "unexpected best RHH threshold",
    )

    proxy_change_percent = percentage_change(
        float(
            best[
                "balanced_proxy_s_over_sqrt_b"
            ]
        ),
        float(
            reference[
                "balanced_proxy_s_over_sqrt_b"
            ]
        ),
    )

    selected_signal_change_percent = percentage_change(
        float(best["selected_signal"]),
        float(reference["selected_signal"]),
    )

    selected_background_change_percent = percentage_change(
        float(best["selected_background"]),
        float(reference["selected_background"]),
    )

    balanced_s_over_b_change_percent = percentage_change(
        float(best["balanced_proxy_s_over_b"]),
        float(
            reference[
                "balanced_proxy_s_over_b"
            ]
        ),
    )

    temporary_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    comparison_rows = [
        {
            "working_point":
                "reference",
            "point_id":
                reference["point_id"],
            "jet_pt_min_GeV":
                reference["jet_pt_min_GeV"],
            "jet_abs_eta_max":
                reference["jet_abs_eta_max"],
            "rhh_sr_max":
                reference["rhh_sr_max"],
            "selected_signal":
                reference["selected_signal"],
            "selected_background":
                reference["selected_background"],
            "balanced_signal_efficiency":
                reference[
                    "balanced_signal_efficiency"
                ],
            "family_balanced_background_efficiency":
                reference[
                    "family_balanced_background_efficiency"
                ],
            "balanced_proxy_s_over_sqrt_b":
                reference[
                    "balanced_proxy_s_over_sqrt_b"
                ],
            "balanced_proxy_s_over_b":
                reference[
                    "balanced_proxy_s_over_b"
                ],
            "status":
                "frozen_reference_cut_point",
        },
        {
            "working_point":
                "coarse_proxy_optimum",
            "point_id":
                best["point_id"],
            "jet_pt_min_GeV":
                best["jet_pt_min_GeV"],
            "jet_abs_eta_max":
                best["jet_abs_eta_max"],
            "rhh_sr_max":
                best["rhh_sr_max"],
            "selected_signal":
                best["selected_signal"],
            "selected_background":
                best["selected_background"],
            "balanced_signal_efficiency":
                best[
                    "balanced_signal_efficiency"
                ],
            "family_balanced_background_efficiency":
                best[
                    "family_balanced_background_efficiency"
                ],
            "balanced_proxy_s_over_sqrt_b":
                best[
                    "balanced_proxy_s_over_sqrt_b"
                ],
            "balanced_proxy_s_over_b":
                best[
                    "balanced_proxy_s_over_b"
                ],
            "status":
                "provisional_train_only_coarse_working_point",
        },
    ]

    comparison_path = (
        temporary_dir
        / "working_point_comparison.tsv"
    )

    comparison_fields = [
        "working_point",
        "point_id",
        "jet_pt_min_GeV",
        "jet_abs_eta_max",
        "rhh_sr_max",
        "selected_signal",
        "selected_background",
        "balanced_signal_efficiency",
        "family_balanced_background_efficiency",
        "balanced_proxy_s_over_sqrt_b",
        "balanced_proxy_s_over_b",
        "status",
    ]

    write_tsv(
        comparison_path,
        comparison_rows,
        comparison_fields,
    )

    shutil.copy2(
        shortlist_path,
        temporary_dir
        / "coarse_scan_shortlist.tsv",
    )

    svg_plots = sorted({
        Path(path).resolve()
        for path in summary["plots"]
        if Path(path).suffix.lower() == ".svg"
    })

    require(
        len(svg_plots) == 5,
        (
            "expected five distinct SVG figures, "
            f"found {len(svg_plots)}"
        ),
    )

    copied_plot_names = []

    for source_plot in svg_plots:
        require(
            source_plot.is_file(),
            f"missing SVG plot: {source_plot}",
        )

        destination = (
            temporary_dir
            / source_plot.name
        )

        shutil.copy2(
            source_plot,
            destination,
        )

        copied_plot_names.append(
            destination.name
        )

    checkpoint = {
        "schema_version": 1,
        "status":
            "hh4b_coarse_cut_scan_checkpoint",
        "classification":
            "provisional_train_only_development_result",
        "source_commit":
            source_commit,
        "source_summary":
            str(summary_path),
        "source_summary_sha256":
            sha256_file(summary_path),
        "source_shortlist":
            str(shortlist_path),
        "source_shortlist_sha256":
            sha256_file(shortlist_path),
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
            summary[
                "near_optimal_plateau_points"
            ],
        "provisional_coarse_working_point": {
            "point_id":
                best["point_id"],
            "jet_pt_min_GeV":
                best["jet_pt_min_GeV"],
            "jet_abs_eta_max":
                best["jet_abs_eta_max"],
            "rhh_sr_max":
                best["rhh_sr_max"],
            "balanced_proxy_s_over_sqrt_b":
                best[
                    "balanced_proxy_s_over_sqrt_b"
                ],
        },
        "comparison_percent_changes": {
            "balanced_proxy_s_over_sqrt_b":
                proxy_change_percent,
            "selected_signal":
                selected_signal_change_percent,
            "selected_background":
                selected_background_change_percent,
            "balanced_proxy_s_over_b":
                balanced_s_over_b_change_percent,
        },
        "physical_background_normalization_frozen":
            False,
        "physical_significance_authorized":
            False,
        "physical_significance_computed":
            False,
        "validation_rows_evaluated":
            0,
        "test_candidate_files_opened":
            0,
        "figures":
            copied_plot_names,
        "next_gate":
            "run_train_only_fine_cut_scan",
    }

    checkpoint_path = (
        temporary_dir
        / "checkpoint.json"
    )

    checkpoint_path.write_text(
        json.dumps(
            checkpoint,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    shortlist_markdown = markdown_shortlist(
        shortlist_rows
    )

    readme = f"""# HH4b coarse cut-scan checkpoint

## Status

This checkpoint records a **provisional train-only coarse-scan working point**. It is not the final optimized cut baseline.

- Scan points: {summary['grid_points']}
- Pareto-frontier points: {summary['pareto_frontier_points']}
- Near-optimal plateau points: {summary['near_optimal_plateau_points']}
- Validation rows used for optimization: 0
- Test files opened: 0
- Physical background normalization frozen: no
- Physical significance computed: no
- Source implementation commit: `{source_commit}`

## Provisional coarse working point

\\[
p_{{\\mathrm{{T}}}}^{{\\min}}=30~\\mathrm{{GeV}},
\\qquad
|\\eta|^{{\\max}}=2.5,
\\qquad
R_{{HH}}<34.
\\]

This point maximizes the train-only family-balanced development proxy

\\[
\\frac{{\\epsilon_S^{{\\mathrm{{bal}}}}}}
{{\\sqrt{{\\epsilon_B^{{\\mathrm{{fam}}}}}}}}
\\]

on the frozen 512-point coarse grid.

It should be described as the **coarse proxy optimum**, not as a physical-significance optimum.

## Comparison with the reference point

| Quantity | Reference | Coarse proxy optimum |
|---|---:|---:|
| Point | `{reference['point_id']}` | `{best['point_id']}` |
| Selected signal rows | {int(reference['selected_signal']):,} | {int(best['selected_signal']):,} |
| Selected background rows | {int(reference['selected_background']):,} | {int(best['selected_background']):,} |
| Balanced signal efficiency | {float(reference['balanced_signal_efficiency']):.6f} | {float(best['balanced_signal_efficiency']):.6f} |
| Family-balanced background efficiency | {float(reference['family_balanced_background_efficiency']):.6f} | {float(best['family_balanced_background_efficiency']):.6f} |
| Balanced \\(S/\\sqrt{{B}}\\)-like proxy | {float(reference['balanced_proxy_s_over_sqrt_b']):.6f} | {float(best['balanced_proxy_s_over_sqrt_b']):.6f} |
| Balanced \\(S/B\\)-like proxy | {float(reference['balanced_proxy_s_over_b']):.6f} | {float(best['balanced_proxy_s_over_b']):.6f} |

Relative to the reference point, the coarse proxy optimum has:

- {proxy_change_percent:+.2f}% change in the balanced \\(S/\\sqrt{{B}}\\)-like proxy;
- {selected_signal_change_percent:+.2f}% change in selected signal rows;
- {selected_background_change_percent:+.2f}% change in selected background rows;
- {balanced_s_over_b_change_percent:+.2f}% change in the balanced \\(S/B\\)-like proxy.

## Physical interpretation

1. The proxy decreases as the common jet threshold is raised. Within the current candidate reconstruction, an additional jet-\\(p_{{\\mathrm{{T}}}}\\) cut above 30 GeV is not favored.
2. The \\(|\\eta|<2.5\\) acceptance has a slightly stronger proxy than \\(|\\eta|<2.4\\). The tighter acceptance removes similar fractions of signal and background.
3. The useful discriminating parameter is \\(R_{{HH}}\\), with a broad coarse plateau around 30–40 and a maximum at 34.
4. The provisional point favors signal acceptance over purity. It improves the development \\(S/\\sqrt{{B}}\\)-like proxy but worsens the \\(S/B\\)-like proxy.
5. The final operating point may move tighter after QCD normalization, background stitching, and systematic uncertainties are included.

## Near-optimal coarse shortlist

{shortlist_markdown}

## Figures

### Best proxy versus jet threshold

![Best proxy versus jet threshold](coarse_best_proxy_vs_ptmin.svg)

### Best proxy versus \\(R_{{HH}}\\)

![Best proxy versus RHH](coarse_best_proxy_vs_rhh.svg)

### Coarse heatmap, \\(|\\eta|<2.4\\)

![Coarse heatmap eta 2.4](coarse_proxy_heatmap_eta2p4.svg)

### Coarse heatmap, \\(|\\eta|<2.5\\)

![Coarse heatmap eta 2.5](coarse_proxy_heatmap_eta2p5.svg)

### Pareto plane

![Coarse Pareto plane](coarse_scan_pareto.svg)

## Reproducibility

- Source cache SHA-256: `{summary['source_cache_sha256']}`
- Configuration SHA-256: `{summary['configuration_sha256']}`
- Source summary SHA-256: `{sha256_file(summary_path)}`
- Source shortlist SHA-256: `{sha256_file(shortlist_path)}`

The event-level cache is intentionally not committed.

## Next gate

Run a train-only fine scan around the stable coarse region:

- \\(p_{{\\mathrm{{T}}}}^{{\\min}}=30.0\\)–34.0 GeV in 0.5 GeV steps;
- \\(|\\eta|^{{\\max}}\\in\\{{2.40,2.45,2.50\\}}\\);
- \\(R_{{HH}}^{{\\max}}=28.0\\)–42.0 in 0.5-unit steps.

Physical ranking and validation confirmation remain separate later gates.
"""

    (
        temporary_dir
        / "README.md"
    ).write_text(
        readme,
        encoding="utf-8",
    )

    checksum_paths = sorted(
        path
        for path in temporary_dir.iterdir()
        if path.is_file()
        and path.name != "SHA256SUMS"
    )

    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in checksum_paths
    ]

    (
        temporary_dir
        / "SHA256SUMS"
    ).write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )

    temporary_dir.rename(
        output_dir
    )

    print(
        "HH4B_COARSE_CUT_SCAN_CHECKPOINT_PASS"
    )
    print(
        "HH4B_CHECKPOINT_CLASSIFICATION="
        "provisional_train_only_development_result"
    )
    print(
        "HH4B_PROVISIONAL_WORKING_POINT="
        "pt30_eta2p5_rhh34"
    )
    print("HH4B_SVG_FIGURES_COPIED=5")
    print("HH4B_PHYSICAL_SIGNIFICANCE_COMPUTED=False")
    print("HH4B_VALIDATION_ROWS_EVALUATED=0")
    print("HH4B_TEST_CANDIDATE_FILES_OPENED=0")
    print("NEXT_GATE=run_train_only_fine_cut_scan")
    print(f"checkpoint={output_dir}")


if __name__ == "__main__":
    main()
