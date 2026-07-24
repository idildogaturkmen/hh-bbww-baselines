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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(
        newline="",
        encoding="utf-8",
        errors="strict",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )
        require(
            bool(reader.fieldnames),
            f"{path}: missing header",
        )
        return list(reader)


def render_shortlist(
    rows: list[dict[str, str]],
) -> str:
    lines = [
        "| Role | Threshold | Signal | Background | "
        "Balanced proxy | Purity proxy |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['shortlist_role']} | "
            f"{float(row['rhh_threshold_value']):.6g} | "
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
        "--shortlist",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--anchors",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--freeze-config",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--index-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--source-commit",
        required=True,
    )

    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]

    def resolve(path: Path) -> Path:
        return (
            path.resolve()
            if path.is_absolute()
            else (repo / path).resolve()
        )

    summary_path = resolve(args.summary)
    shortlist_path = resolve(args.shortlist)
    anchors_path = resolve(args.anchors)
    checkpoint = resolve(args.checkpoint)
    freeze_path = resolve(args.freeze_config)
    index_dir = resolve(args.index_dir)

    require(
        re.fullmatch(
            r"[0-9a-f]{40}",
            args.source_commit.lower(),
        ) is not None,
        "source commit must be a full SHA",
    )

    for path in (
        summary_path,
        shortlist_path,
        anchors_path,
    ):
        require(
            path.is_file(),
            f"missing input: {path}",
        )

    for path in (
        checkpoint,
        freeze_path,
        index_dir,
    ):
        require(
            not path.exists(),
            f"refusing to overwrite {path}",
        )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8",
        )
    )

    require(
        summary.get("status")
        == "hh4b_train_only_exact_rhh_scan_pass",
        "exact scan did not pass",
    )
    require(
        summary.get("observed_transition_points")
        == 3633,
        "unexpected transition count",
    )
    require(
        summary.get("best_at_scan_window_boundary")
        is False,
        "best point lies at scan boundary",
    )
    require(
        summary.get("validation_rows_evaluated")
        == 0,
        "validation was accessed",
    )
    require(
        summary.get("test_candidate_files_opened")
        == 0,
        "test files were accessed",
    )
    require(
        summary.get("physical_significance_computed")
        is False,
        "physical significance was computed",
    )

    best = summary["exact_best_point"]
    rounded = summary["rounded_34_point"]

    require(
        math.isclose(
            float(best["rhh_threshold_value"]),
            34.12742053804616,
            abs_tol=1.0e-12,
        ),
        "unexpected exact optimum",
    )
    require(
        math.isclose(
            float(rounded["rhh_threshold_value"]),
            34.0,
            abs_tol=1.0e-12,
        ),
        "rounded nominal point is not 34",
    )
    require(
        float(
            summary[
                "exact_proxy_change_from_rounded_34_percent"
            ]
        ) < 0.25,
        "exact gain over rounded 34 is unexpectedly large",
    )
    require(
        float(
            summary[
                "stable_component_transition_min"
            ]
        ) < 34.0
        < float(
            summary[
                "stable_component_transition_max"
            ]
        ),
        "rounded 34 is not inside stable component",
    )

    shortlist_rows = read_tsv(shortlist_path)
    anchor_rows = read_tsv(anchors_path)

    require(
        len(shortlist_rows) == 5,
        f"expected 5 shortlist rows, found {len(shortlist_rows)}",
    )
    require(
        len(anchor_rows) == 3,
        f"expected 3 rounded anchors, found {len(anchor_rows)}",
    )

    checkpoint_tmp = checkpoint.with_name(
        checkpoint.name + "_incomplete"
    )

    checkpoint_tmp.mkdir(
        parents=True,
        exist_ok=False,
    )

    shutil.copy2(
        summary_path,
        checkpoint_tmp / "exact_rhh_scan_summary.json",
    )
    shutil.copy2(
        shortlist_path,
        checkpoint_tmp / "exact_rhh_scan_shortlist.tsv",
    )
    shutil.copy2(
        anchors_path,
        checkpoint_tmp / "exact_rhh_rounded_anchors.tsv",
    )

    figure_paths = sorted({
        Path(value).resolve()
        for value in summary["plots"]
        if Path(value).suffix.lower()
        in {".svg", ".pdf"}
    })

    require(
        len(figure_paths) == 8,
        f"expected 8 vector figure files, found {len(figure_paths)}",
    )

    copied_figures: list[str] = []

    for source in figure_paths:
        require(
            source.is_file(),
            f"missing figure: {source}",
        )
        destination = checkpoint_tmp / source.name
        shutil.copy2(source, destination)
        copied_figures.append(destination.name)

    freeze_document = {
        "schema_version": 1,
        "status":
            "hh4b_train_only_cut_working_points_frozen",
        "classification":
            "paper_quality_train_only_optimization_result",
        "fixed_acceptance": {
            "jet_pt_min_GeV": 30.0,
            "jet_abs_eta_max": 2.5
        },
        "nominal_working_point": {
            "rhh_sr_max": 34.0,
            "point_id": "rounded_rhh_34p0",
            "selection":
                "min_jet_pt > 30 GeV, "
                "max_abs_jet_eta < 2.5, "
                "r_hh_125_120 < 34",
            "rationale": (
                "Rounded threshold lies inside the "
                "99.5%-of-maximum stable component and "
                "retains 99.779% of the exact train proxy."
            )
        },
        "exact_observed_transition_optimum": {
            "rhh_sr_max":
                best["rhh_threshold_value"],
            "balanced_proxy_s_over_sqrt_b":
                best[
                    "balanced_proxy_s_over_sqrt_b"
                ],
            "selected_signal":
                best["selected_signal"],
            "selected_background":
                best["selected_background"]
        },
        "rounded_alternatives": [
            {
                "role": "higher_purity",
                "rhh_sr_max": 31.5
            },
            {
                "role": "higher_efficiency",
                "rhh_sr_max": 35.5
            }
        ],
        "stable_component": {
            "fraction_of_best": 0.995,
            "transition_min":
                summary[
                    "stable_component_transition_min"
                ],
            "transition_max":
                summary[
                    "stable_component_transition_max"
                ],
            "points":
                summary[
                    "stable_component_points"
                ]
        },
        "exact_gain_over_rounded_34_percent":
            summary[
                "exact_proxy_change_from_rounded_34_percent"
            ],
        "validation_rows_evaluated": 0,
        "test_candidate_files_opened": 0,
        "physical_background_normalization_frozen":
            False,
        "physical_significance_computed":
            False,
        "source_commit":
            args.source_commit.lower(),
        "source_summary_sha256":
            sha256_file(summary_path),
        "next_gate":
            "freeze_physical_background_normalization_and_systematics"
    }

    freeze_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    freeze_path.write_text(
        json.dumps(
            freeze_document,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    shortlist_markdown = render_shortlist(
        shortlist_rows
    )

    readme = f"""# HH4b exact observed-transition cut optimization

## Result classification

This is a **paper-quality train-only optimization result**.

It uses every distinct observed train-sample transition in
\\(R_{{HH}}\\) between 31 and 36.5, with the object acceptance fixed to

\\[
p_{{\\mathrm{{T}}}}^{{\\min}}=30~\\mathrm{{GeV}},
\\qquad
|\\eta|^{{\\max}}=2.5.
\\]

No validation or test events were accessed. Physical background
normalization and significance were not used.

## Exact optimum

The exact observed-transition maximum is

\\[
R_{{HH}} <
{float(best['rhh_threshold_value']):.9f},
\\]

with

\\[
\\frac{{\\epsilon_S^{{\\mathrm{{bal}}}}}}
{{\\sqrt{{\\epsilon_B^{{\\mathrm{{fam}}}}}}}}
=
{float(best['balanced_proxy_s_over_sqrt_b']):.6f}.
\\]

It selects {int(best['selected_signal']):,} signal rows and
{int(best['selected_background']):,} background rows.

## Nominal rounded train-only working point

The frozen nominal train-only point is

\\[
\\boxed{{
p_{{\\mathrm{{T}}}}^{{\\min}}=30~\\mathrm{{GeV}},
\\quad
|\\eta|^{{\\max}}=2.5,
\\quad
R_{{HH}}<34
}}.
\\]

The rounded threshold is inside the contiguous 99.5%-of-maximum interval

\\[
{float(summary['stable_component_transition_min']):.4f}
<
R_{{HH}}^{{\\max}}
<
{float(summary['stable_component_transition_max']):.4f}.
\\]

This stable component contains
{summary['stable_component_points']} observed transitions.

The exact transition improves the proxy over the rounded cut by only
{float(summary['exact_proxy_change_from_rounded_34_percent']):.3f}%.
The rounded threshold is therefore preferred for reproducibility,
portability across samples, and interpretability.

## Efficiency–purity shortlist

{shortlist_markdown}

## Interpretation

- The exact scan confirms a genuine interior optimum near 34.
- Signal and background efficiencies both rise as the threshold is loosened.
- Purity decreases with increasing threshold.
- \\(R_{{HH}}<31.5\\) is the higher-purity rounded alternative.
- \\(R_{{HH}}<35.5\\) is the higher-efficiency rounded alternative.
- \\(R_{{HH}}<34\\) is the nominal train-only compromise.

The final physical working point remains conditional on process-level
background normalization, QCD stitching, finite-MC uncertainty, and the
background-systematic model.

## Figures

- [Exact balanced-proxy scan](exact_rhh_balanced_proxy.svg)
- [Exact efficiency curves](exact_rhh_efficiencies.svg)
- [Exact purity curve](exact_rhh_purity_proxy.svg)
- [Exact Pareto trajectory](exact_rhh_pareto.svg)

PDF versions are included for direct paper use.

## Reproducibility

- Source implementation commit:
  `{args.source_commit.lower()}`
- Exact summary SHA-256:
  `{sha256_file(summary_path)}`
- Exact shortlist SHA-256:
  `{sha256_file(shortlist_path)}`
- Rounded anchors SHA-256:
  `{sha256_file(anchors_path)}`
- Cache SHA-256:
  `{summary['source_cache_sha256']}`

## Next gate

Freeze the physical background normalization and systematic model, then
re-rank the frozen rounded working points without reopening geometric
cut optimization.
"""

    (
        checkpoint_tmp / "README.md"
    ).write_text(
        readme,
        encoding="utf-8",
    )

    checkpoint_document = {
        "schema_version": 1,
        "status":
            "hh4b_exact_rhh_cut_scan_checkpoint",
        "classification":
            "paper_quality_train_only_optimization_result",
        "source_commit":
            args.source_commit.lower(),
        "exact_transition_points":
            summary["observed_transition_points"],
        "exact_best_point":
            best,
        "nominal_rounded_point":
            rounded,
        "stable_component": {
            "points":
                summary["stable_component_points"],
            "transition_min":
                summary[
                    "stable_component_transition_min"
                ],
            "transition_max":
                summary[
                    "stable_component_transition_max"
                ]
        },
        "figures":
            copied_figures,
        "validation_rows_evaluated": 0,
        "test_candidate_files_opened": 0,
        "physical_significance_computed": False,
        "freeze_config":
            str(freeze_path),
        "next_gate":
            "freeze_physical_background_normalization_and_systematics"
    }

    (
        checkpoint_tmp / "checkpoint.json"
    ).write_text(
        json.dumps(
            checkpoint_document,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    checksum_files = sorted(
        path
        for path in checkpoint_tmp.iterdir()
        if path.is_file()
        and path.name != "SHA256SUMS"
    )

    (
        checkpoint_tmp / "SHA256SUMS"
    ).write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in checksum_files
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoint_tmp.rename(checkpoint)

    index_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    index_text = """# HH4b train-only cut optimization index

This index links the complete paper-quality train-only cut-optimization
sequence.

## Stages

1. [Coarse grid](../hh4b_cut_scan_coarse_20260725_v1/README.md)
2. [Fine grid](../hh4b_cut_scan_fine_20260725_v1/README.md)
3. [Exact observed-transition scan](../hh4b_cut_scan_exact_rhh_20260725_v1/README.md)

## Frozen train-only conclusion

The nominal train-only working point is

\\[
p_{\\mathrm{T}}^{\\min}=30~\\mathrm{GeV},
\\qquad
|\\eta|^{\\max}=2.5,
\\qquad
R_{HH}<34.
\\]

The exact observed-transition optimum is approximately 34.1274, but its
gain over the rounded cut is only 0.221%. The rounded value is retained
as the nominal point because it lies within the stable optimum and is
more reproducible and portable.

This is not yet the final physical baseline. The next stage is physical
background normalization and systematic modeling.
"""

    (
        index_dir / "README.md"
    ).write_text(
        index_text,
        encoding="utf-8",
    )

    print("HH4B_EXACT_RHH_CHECKPOINT_PASS")
    print(
        "HH4B_RESULT_CLASSIFICATION="
        "paper_quality_train_only_optimization_result"
    )
    print(
        "HH4B_NOMINAL_TRAIN_ONLY_WORKING_POINT="
        "pt30_eta2p5_rhh34"
    )
    print("HH4B_VECTOR_FIGURES_COPIED=8")
    print(
        "HH4B_EXACT_GAIN_OVER_ROUNDED34_PERCENT="
        f"{summary['exact_proxy_change_from_rounded_34_percent']}"
    )
    print(
        "NEXT_GATE="
        "freeze_physical_background_normalization_and_systematics"
    )


if __name__ == "__main__":
    main()
