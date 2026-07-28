#!/usr/bin/env python3
"""Create committed tables and CMS-inspired figures for the expanded HH4b cut baseline.

The script reads only the already-frozen train-only cut-result products. It does
not open candidate Parquet files, validation content, or final-evaluation content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.plotting.hh4b_cms_style import (  # noqa: E402
    CATEGORY_COLORS,
    add_delphes_header,
    apply_cms_style,
    save_png_pdf,
)


WORKING_POINT_ORDER: tuple[str, ...] = (
    "cms_reference_rhh30",
    "higher_purity_rhh31p5",
    "optimized_nominal_rhh34",
    "higher_efficiency_rhh35p5",
)

WORKING_POINT_LABELS: Mapping[str, str] = {
    "cms_reference_rhh30": r"CMS reference: $R_{HH}^{125,120}<30$",
    "higher_purity_rhh31p5": r"Higher purity: $R_{HH}^{125,120}<31.5$",
    "optimized_nominal_rhh34": r"Frozen nominal: $R_{HH}^{125,120}<34$",
    "higher_efficiency_rhh35p5": r"Higher efficiency: $R_{HH}^{125,120}<35.5$",
}

SHORT_LABELS: Mapping[str, str] = {
    "cms_reference_rhh30": r"$R_{HH}<30$",
    "higher_purity_rhh31p5": r"$R_{HH}<31.5$",
    "optimized_nominal_rhh34": r"$R_{HH}<34$",
    "higher_efficiency_rhh35p5": r"$R_{HH}<35.5$",
}

REGION_ORDER: tuple[str, ...] = (
    "cms_signal_region",
    "cms_control_region",
    "outside_geometry",
)

REGION_LABELS: Mapping[str, str] = {
    "cms_signal_region": r"$R_{HH}^{125,120}<30$",
    "cms_control_region": r"$30\leq R_{HH}^{125,120}<55$",
    "outside_geometry": r"$R_{HH}^{125,120}\geq55$",
}


class ReportError(RuntimeError):
    """Raised when a frozen input or report invariant fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="strict") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(bool(reader.fieldnames), f"{path}: missing TSV header")
        return list(reader)


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty TSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def current_git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()


def verify_sha256_manifest(directory: Path) -> None:
    manifest = directory / "SHA256SUMS"
    require(manifest.is_file(), f"missing checksum manifest: {manifest}")
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, name = raw_line.split("  ", 1)
        path = directory / name
        require(path.is_file(), f"checksum target missing: {path}")
        require(sha256_file(path) == digest, f"checksum mismatch: {path}")


def parse_json_mapping(value: str, label: str) -> dict[str, float]:
    payload = json.loads(value)
    require(isinstance(payload, dict), f"{label} is not a JSON object")
    result = {str(key): float(number) for key, number in payload.items()}
    require(all(math.isfinite(number) for number in result.values()), f"{label} contains nonfinite values")
    return result


def validate_metric_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    require(len(rows) == 4, f"cut metric row count is {len(rows)}, expected 4")
    by_name = {row["baseline"]: row for row in rows}
    require(set(by_name) == set(WORKING_POINT_ORDER), "working-point set mismatch")

    expected_thresholds = {
        "cms_reference_rhh30": 30.0,
        "higher_purity_rhh31p5": 31.5,
        "optimized_nominal_rhh34": 34.0,
        "higher_efficiency_rhh35p5": 35.5,
    }

    normalized: list[dict[str, Any]] = []
    for name in WORKING_POINT_ORDER:
        row = by_name[name]
        require(row["variable"] == "r_hh_125_120", f"wrong variable for {name}")
        require(row["operator"] == "less_than", f"wrong operator for {name}")
        require(row["status"] == "pass", f"failed metric row for {name}")
        require(row["physical_normalization"] == "False", f"{name} claims physical normalization")
        threshold = float(row["threshold_GeV"])
        require(threshold == expected_thresholds[name], f"threshold mismatch for {name}")

        values: dict[str, Any] = {
            "baseline": name,
            "label": WORKING_POINT_LABELS[name],
            "short_label": SHORT_LABELS[name],
            "threshold_GeV": threshold,
            "selected_signal_rows": int(row["selected_signal_rows"]),
            "selected_background_rows": int(row["selected_background_rows"]),
            "weighted_signal_efficiency": float(row["weighted_signal_efficiency"]),
            "weighted_background_efficiency": float(row["weighted_background_efficiency"]),
            "weighted_background_rejection_fraction": float(row["weighted_background_rejection_fraction"]),
            "weighted_inverse_background_efficiency": float(row["weighted_inverse_background_efficiency"]),
            "raw_signal_efficiency": float(row["raw_signal_efficiency"]),
            "raw_background_efficiency": float(row["raw_background_efficiency"]),
            "balanced_efficiency_proxy": float(row["balanced_efficiency_proxy"]),
            "purity_proxy": float(row["purity_proxy"]),
            "weighted_signal_mode_efficiencies": parse_json_mapping(
                row["weighted_signal_mode_efficiencies"],
                f"{name} signal-mode efficiencies",
            ),
            "weighted_background_family_efficiencies": parse_json_mapping(
                row["weighted_background_family_efficiencies"],
                f"{name} background-family efficiencies",
            ),
        }
        for key, number in values.items():
            if isinstance(number, float):
                require(math.isfinite(number), f"nonfinite {name} {key}")
        require(
            math.isclose(
                values["weighted_inverse_background_efficiency"],
                1.0 / values["weighted_background_efficiency"],
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ),
            f"inverse background efficiency mismatch for {name}",
        )
        normalized.append(values)
    return normalized


def aggregate_region_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    class_rows = [row for row in rows if row["population_level"] == "class"]
    require(len(class_rows) == 6, f"class-level region rows={len(class_rows)}, expected 6")
    by_key = {(row["region"], row["population"]): row for row in class_rows}
    expected_keys = {(region, sample_class) for region in REGION_ORDER for sample_class in ("signal", "background")}
    require(set(by_key) == expected_keys, "class-level region key set mismatch")

    result: list[dict[str, Any]] = []
    total_rows = 0
    for region in REGION_ORDER:
        signal = by_key[(region, "signal")]
        background = by_key[(region, "background")]
        signal_rows = int(signal["selected_rows"])
        background_rows = int(background["selected_rows"])
        rows_total = signal_rows + background_rows
        total_rows += rows_total
        result.append(
            {
                "region": region,
                "latex_label": REGION_LABELS[region],
                "signal_rows": signal_rows,
                "background_rows": background_rows,
                "total_rows": rows_total,
                "raw_fraction_all_rows": rows_total / 53162.0,
                "weighted_signal_fraction": float(signal["weighted_fraction"]),
                "weighted_background_fraction": float(background["weighted_fraction"]),
                "balanced_overall_fraction": 0.5 * float(signal["weighted_fraction"])
                + 0.5 * float(background["weighted_fraction"]),
            }
        )
    require(total_rows == 53162, f"region totals account for {total_rows} rows, expected 53162")
    return result


def latex_escape_text(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def working_point_latex(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Train-only expanded $HH\to b\bar b b\bar b$ cut working points. The efficiencies use hierarchical development-balancing weights and are not physical yield normalizations.}",
        r"\label{tab:hh4b_expanded_cut_working_points}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Working point & $R_{HH}^{\max}$ [GeV] & $N_{S}$ & $N_{B}$ & $\epsilon_{S}$ & $\epsilon_{B}$ & $1/\epsilon_{B}$ \\",
        r"\midrule",
    ]
    for row in rows:
        label = {
            "cms_reference_rhh30": "CMS reference",
            "higher_purity_rhh31p5": "Higher purity",
            "optimized_nominal_rhh34": "Frozen nominal",
            "higher_efficiency_rhh35p5": "Higher efficiency",
        }[str(row["baseline"])]
        lines.append(
            f"{label} & {float(row['threshold_GeV']):.1f} & "
            f"{int(row['selected_signal_rows']):d} & {int(row['selected_background_rows']):d} & "
            f"{float(row['weighted_signal_efficiency']):.4f} & "
            f"{float(row['weighted_background_efficiency']):.4f} & "
            f"{float(row['weighted_inverse_background_efficiency']):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def region_latex(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Train-only candidate accounting in the CMS-aligned $R_{HH}^{125,120}$ regions.}",
        r"\label{tab:hh4b_expanded_cut_regions}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Region & Signal rows & Background rows & Total rows \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"${str(row['latex_label']).strip('$')}$ & "
            f"{int(row['signal_rows']):d} & {int(row['background_rows']):d} & {int(row['total_rows']):d} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def save_figure(fig: Any, base_path: Path, description: str, manifest: list[dict[str, Any]]) -> None:
    png, pdf = save_png_pdf(fig, base_path, dpi=300)
    for path in (png, pdf):
        manifest.append(
            {
                "figure": base_path.name,
                "format": path.suffix.lstrip("."),
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "description": description,
                "official_cms_status_claimed": False,
            }
        )


def figure_efficiencies(rows: Sequence[Mapping[str, Any]], output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    thresholds = np.asarray([row["threshold_GeV"] for row in rows], dtype=float)
    signal = np.asarray([row["weighted_signal_efficiency"] for row in rows], dtype=float)
    background = np.asarray([row["weighted_background_efficiency"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.plot(thresholds, signal, marker="o", linewidth=2.0, label=r"Signal $\epsilon_{S}$")
    ax.plot(thresholds, background, marker="s", linewidth=2.0, label=r"Background $\epsilon_{B}$")
    ax.axvline(34.0, linestyle="--", linewidth=1.2, color="black", label=r"Frozen nominal $R_{HH}<34$")
    ax.set_xlabel(r"$R_{HH}^{125,120}$ upper threshold [GeV]")
    ax.set_ylabel("Weighted efficiency")
    ax.set_ylim(0.0, 0.75)
    ax.legend(loc="best")
    add_delphes_header(ax, "Train-only; hierarchical development-balancing weights")
    fig.tight_layout()
    save_figure(
        fig,
        output_dir / "cut_weighted_efficiencies_vs_threshold",
        "Weighted signal and background efficiencies across the four predeclared cut working points.",
        manifest,
    )


def figure_efficiency_plane(rows: Sequence[Mapping[str, Any]], output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    for index, row in enumerate(rows):
        ax.scatter(
            row["weighted_background_efficiency"],
            row["weighted_signal_efficiency"],
            s=65,
            color=CATEGORY_COLORS[index],
            label=row["short_label"],
            zorder=3,
        )
        ax.annotate(
            f"{row['threshold_GeV']:.1f}",
            (row["weighted_background_efficiency"], row["weighted_signal_efficiency"]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel(r"Weighted background efficiency $\epsilon_{B}$")
    ax.set_ylabel(r"Weighted signal efficiency $\epsilon_{S}$")
    ax.set_xlim(0.18, 0.28)
    ax.set_ylim(0.52, 0.66)
    ax.legend(loc="lower right")
    add_delphes_header(ax, "Train-only working-point trade-off")
    fig.tight_layout()
    save_figure(
        fig,
        output_dir / "cut_signal_background_efficiency_plane",
        "Signal-versus-background efficiency trade-off for the four predeclared cut working points.",
        manifest,
    )


def figure_inverse_rejection(rows: Sequence[Mapping[str, Any]], output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    thresholds = np.asarray([row["threshold_GeV"] for row in rows], dtype=float)
    values = np.asarray([row["weighted_inverse_background_efficiency"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.plot(thresholds, values, marker="o", linewidth=2.0)
    ax.axvline(34.0, linestyle="--", linewidth=1.2, color="black")
    for x, y in zip(thresholds, values):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlabel(r"$R_{HH}^{125,120}$ upper threshold [GeV]")
    ax.set_ylabel(r"Inverse background efficiency $1/\epsilon_{B}$")
    ax.set_ylim(3.5, 5.4)
    add_delphes_header(ax, "Train-only; larger values indicate stronger rejection")
    fig.tight_layout()
    save_figure(
        fig,
        output_dir / "cut_inverse_background_efficiency_vs_threshold",
        "Inverse weighted background efficiency versus the cut threshold.",
        manifest,
    )


def figure_balanced_proxy(rows: Sequence[Mapping[str, Any]], output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    thresholds = np.asarray([row["threshold_GeV"] for row in rows], dtype=float)
    values = np.asarray([row["balanced_efficiency_proxy"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.plot(thresholds, values, marker="o", linewidth=2.0)
    ax.axvline(34.0, linestyle="--", linewidth=1.2, color="black", label="Frozen nominal")
    for x, y in zip(thresholds, values):
        ax.annotate(f"{y:.4f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlabel(r"$R_{HH}^{125,120}$ upper threshold [GeV]")
    ax.set_ylabel(r"Balanced proxy $\epsilon_{S}/\sqrt{\epsilon_{B}}$")
    ax.set_ylim(min(values) - 0.01, max(values) + 0.01)
    ax.legend(loc="best")
    add_delphes_header(ax, "Train-only proxy; not a physical significance")
    fig.tight_layout()
    save_figure(
        fig,
        output_dir / "cut_balanced_proxy_vs_threshold",
        "Development-balanced efficiency proxy across the predeclared working points.",
        manifest,
    )


def figure_purity_proxy(rows: Sequence[Mapping[str, Any]], output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    thresholds = np.asarray([row["threshold_GeV"] for row in rows], dtype=float)
    values = np.asarray([row["purity_proxy"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.plot(thresholds, values, marker="o", linewidth=2.0)
    ax.axvline(34.0, linestyle="--", linewidth=1.2, color="black")
    for x, y in zip(thresholds, values):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlabel(r"$R_{HH}^{125,120}$ upper threshold [GeV]")
    ax.set_ylabel(r"Development purity proxy $\epsilon_{S}/\epsilon_{B}$")
    ax.set_ylim(min(values) - 0.08, max(values) + 0.08)
    add_delphes_header(ax, "Train-only proxy; not a physical signal purity")
    fig.tight_layout()
    save_figure(
        fig,
        output_dir / "cut_purity_proxy_vs_threshold",
        "Development purity proxy across the predeclared working points.",
        manifest,
    )


def figure_signal_modes(rows: Sequence[Mapping[str, Any]], output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    thresholds = np.asarray([row["threshold_GeV"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for mode, label, marker in (
        ("ggf_hh4b", r"ggF $HH\to b\bar b b\bar b$", "o"),
        ("vbf_hh4b", r"VBF $HH\to b\bar b b\bar b$", "s"),
    ):
        values = [row["weighted_signal_mode_efficiencies"][mode] for row in rows]
        ax.plot(thresholds, values, marker=marker, linewidth=2.0, label=label)
    ax.axvline(34.0, linestyle="--", linewidth=1.2, color="black")
    ax.set_xlabel(r"$R_{HH}^{125,120}$ upper threshold [GeV]")
    ax.set_ylabel("Weighted signal-mode efficiency")
    ax.set_ylim(0.45, 0.70)
    ax.legend(loc="best")
    add_delphes_header(ax, "Train-only signal-mode comparison")
    fig.tight_layout()
    save_figure(
        fig,
        output_dir / "cut_signal_mode_efficiencies_vs_threshold",
        "Weighted ggF and VBF signal efficiencies across the predeclared cut working points.",
        manifest,
    )


def figure_background_heatmap(rows: Sequence[Mapping[str, Any]], output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    families = sorted(rows[0]["weighted_background_family_efficiencies"])
    matrix = np.asarray(
        [
            [row["weighted_background_family_efficiencies"][family] for row in rows]
            for family in families
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=max(0.4, float(np.max(matrix))))
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([row["short_label"] for row in rows])
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(
        [latex_escape_text(family).replace(r"\_", " ") for family in families]
    )
    ax.set_xlabel("Cut working point")
    ax.set_ylabel("Background family")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"Weighted background efficiency $\epsilon_{B}$")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7)
    add_delphes_header(ax, "Train-only family-balanced diagnostic")
    fig.tight_layout()
    save_figure(
        fig,
        output_dir / "cut_background_family_efficiency_heatmap",
        "Weighted background-family efficiencies for each predeclared cut working point.",
        manifest,
    )


def figure_regions(rows: Sequence[Mapping[str, Any]], output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    x = np.arange(len(rows))
    signal = np.asarray([row["signal_rows"] for row in rows], dtype=float)
    background = np.asarray([row["background_rows"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.bar(x, background, label="Background candidates")
    ax.bar(x, signal, bottom=background, label="Signal candidates")
    ax.set_xticks(x)
    ax.set_xticklabels([row["latex_label"] for row in rows])
    ax.set_ylabel("Train candidate rows")
    ax.legend(loc="best")
    for index, total in enumerate(signal + background):
        ax.text(index, total + 500, f"{int(total):,}", ha="center", va="bottom", fontsize=9)
    add_delphes_header(ax, "CMS-aligned mass-plane region accounting")
    fig.tight_layout()
    save_figure(
        fig,
        output_dir / "cms_aligned_region_candidate_accounting",
        "Signal and background candidate counts in the CMS-aligned mass-plane regions.",
        manifest,
    )


def build_signal_mode_rows(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in metric_rows:
        for mode in ("ggf_hh4b", "vbf_hh4b"):
            result.append(
                {
                    "baseline": row["baseline"],
                    "threshold_GeV": row["threshold_GeV"],
                    "signal_mode": mode,
                    "weighted_efficiency": row["weighted_signal_mode_efficiencies"][mode],
                }
            )
    return result


def build_background_family_rows(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    families = sorted(metric_rows[0]["weighted_background_family_efficiencies"])
    for row in metric_rows:
        for family in families:
            result.append(
                {
                    "baseline": row["baseline"],
                    "threshold_GeV": row["threshold_GeV"],
                    "background_family": family,
                    "weighted_efficiency": row["weighted_background_family_efficiencies"][family],
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    source_commit = args.source_commit.lower()
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None, "source commit must be a full SHA")
    require(current_git_head() == source_commit, "current HEAD differs from source commit")

    input_dir = args.input_dir if args.input_dir.is_absolute() else REPOSITORY_ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPOSITORY_ROOT / args.output_dir
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    temporary_dir = output_dir.with_name(output_dir.name + "_incomplete")

    require(input_dir.is_dir(), f"missing input directory: {input_dir}")
    require(not output_dir.exists(), f"refusing to overwrite: {output_dir}")
    require(not temporary_dir.exists(), f"temporary directory exists: {temporary_dir}")
    verify_sha256_manifest(input_dir)

    summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    require(summary.get("status") == "hh4b_expanded_cut_baseline_train_only_pass", "source cut result did not pass")
    require(summary.get("source_commit") == source_commit, "source cut result commit mismatch")
    controls = summary["controls"]
    require(controls.get("validation_candidate_rows_read") == 0, "validation rows were read")
    require(controls.get("evaluation_candidate_rows_read") == 0, "evaluation rows were read")
    require(controls.get("models_trained") == 0, "models were trained")
    require(controls.get("physical_normalization_performed") is False, "physical normalization was performed")

    metric_rows = validate_metric_rows(read_tsv(input_dir / "cut_baseline_metrics.tsv"))
    region_rows = aggregate_region_rows(read_tsv(input_dir / "cms_region_accounting.tsv"))

    style_record = apply_cms_style()
    temporary_dir.mkdir(parents=True, exist_ok=False)
    figures_dir = temporary_dir / "figures"
    tables_dir = temporary_dir / "tables"
    source_dir = temporary_dir / "source"
    figures_dir.mkdir()
    tables_dir.mkdir()
    source_dir.mkdir()

    for name in (
        "README.md",
        "summary.json",
        "cut_baseline_metrics.tsv",
        "cms_region_accounting.tsv",
        "member_weight_audit.tsv",
        "weight_summary.tsv",
    ):
        shutil.copy2(input_dir / name, source_dir / name)

    working_table_rows = [
        {
            "baseline": row["baseline"],
            "threshold_GeV": row["threshold_GeV"],
            "selected_signal_rows": row["selected_signal_rows"],
            "selected_background_rows": row["selected_background_rows"],
            "weighted_signal_efficiency": row["weighted_signal_efficiency"],
            "weighted_background_efficiency": row["weighted_background_efficiency"],
            "weighted_background_rejection_fraction": row["weighted_background_rejection_fraction"],
            "weighted_inverse_background_efficiency": row["weighted_inverse_background_efficiency"],
            "balanced_efficiency_proxy": row["balanced_efficiency_proxy"],
            "purity_proxy": row["purity_proxy"],
        }
        for row in metric_rows
    ]
    write_tsv(tables_dir / "cut_working_points.tsv", working_table_rows)
    write_tsv(tables_dir / "cms_region_summary.tsv", region_rows)
    write_tsv(tables_dir / "signal_mode_efficiencies.tsv", build_signal_mode_rows(metric_rows))
    write_tsv(tables_dir / "background_family_efficiencies.tsv", build_background_family_rows(metric_rows))
    (tables_dir / "cut_working_points.tex").write_text(working_point_latex(metric_rows), encoding="utf-8")
    (tables_dir / "cms_region_summary.tex").write_text(region_latex(region_rows), encoding="utf-8")

    figure_manifest: list[dict[str, Any]] = []
    figure_efficiencies(metric_rows, figures_dir, figure_manifest)
    figure_efficiency_plane(metric_rows, figures_dir, figure_manifest)
    figure_inverse_rejection(metric_rows, figures_dir, figure_manifest)
    figure_balanced_proxy(metric_rows, figures_dir, figure_manifest)
    figure_purity_proxy(metric_rows, figures_dir, figure_manifest)
    figure_signal_modes(metric_rows, figures_dir, figure_manifest)
    figure_background_heatmap(metric_rows, figures_dir, figure_manifest)
    figure_regions(region_rows, figures_dir, figure_manifest)

    for row in figure_manifest:
        row["path"] = str(Path(row["path"]).relative_to(temporary_dir))
    write_tsv(temporary_dir / "figure_manifest.tsv", figure_manifest)

    table_paths = sorted(tables_dir.iterdir())
    table_manifest = [
        {
            "table": path.stem,
            "format": path.suffix.lstrip("."),
            "path": str(path.relative_to(temporary_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in table_paths
        if path.is_file()
    ]
    write_tsv(temporary_dir / "table_manifest.tsv", table_manifest)

    source_audit = []
    for path in sorted(source_dir.iterdir()):
        source_audit.append(
            {
                "role": "train_only_cut_source_product",
                "path": str(path.relative_to(temporary_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    source_audit.extend(
        [
            {
                "role": "report_generator",
                "path": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
                "bytes": Path(__file__).stat().st_size,
                "sha256": sha256_file(Path(__file__)),
            },
            {
                "role": "cms_plotting_helper",
                "path": "scripts/plotting/hh4b_cms_style.py",
                "bytes": (REPOSITORY_ROOT / "scripts/plotting/hh4b_cms_style.py").stat().st_size,
                "sha256": sha256_file(REPOSITORY_ROOT / "scripts/plotting/hh4b_cms_style.py"),
            },
        ]
    )
    write_tsv(temporary_dir / "protected_source_audit.tsv", source_audit)

    by_name = {row["baseline"]: row for row in metric_rows}
    nominal = by_name["optimized_nominal_rhh34"]
    cms_reference = by_name["cms_reference_rhh30"]
    higher_purity = by_name["higher_purity_rhh31p5"]
    higher_efficiency = by_name["higher_efficiency_rhh35p5"]

    interpretation = {
        "nominal_cut": "r_hh_125_120 < 34",
        "nominal_weighted_signal_efficiency": nominal["weighted_signal_efficiency"],
        "nominal_weighted_background_efficiency": nominal["weighted_background_efficiency"],
        "nominal_inverse_background_efficiency": nominal["weighted_inverse_background_efficiency"],
        "nominal_balanced_proxy": nominal["balanced_efficiency_proxy"],
        "cms_reference_balanced_proxy": cms_reference["balanced_efficiency_proxy"],
        "higher_purity_balanced_proxy": higher_purity["balanced_efficiency_proxy"],
        "higher_efficiency_balanced_proxy": higher_efficiency["balanced_efficiency_proxy"],
        "largest_predeclared_balanced_proxy_point": "higher_efficiency_rhh35p5",
        "post_result_retuning_authorized": False,
    }

    report_summary = {
        "schema_version": 1,
        "status": "hh4b_expanded_cut_baseline_tables_figures_pass",
        "source_commit": source_commit,
        "source_result_status": summary["status"],
        "source_train_rows": 53162,
        "source_signal_rows": 9337,
        "source_background_rows": 43825,
        "working_points": 4,
        "figures": 8,
        "figure_files": 16,
        "tables": len(table_manifest),
        "plot_style": style_record,
        "interpretation": interpretation,
        "controls": {
            "candidate_parquet_files_opened": 0,
            "validation_candidate_files_opened": 0,
            "evaluation_candidate_files_opened": 0,
            "models_trained": 0,
            "thresholds_reoptimized": 0,
            "physical_normalization_performed": False,
            "official_cms_status_claimed": False,
        },
        "next_gate": "commit_expanded_cut_tables_figures_then_implement_bdt_runner",
    }
    (temporary_dir / "summary.json").write_text(
        json.dumps(report_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    readme = rf"""# Expanded HH4b train-only cut baseline

## Scope

This checkpoint freezes the tables and CMS-publication-inspired figures for the
expanded 5M-background/200k-signal source study. The plotted population is the
53,162-row train cache: 9,337 signal candidates and 43,825 background candidates.

The figures are explicitly labeled **Delphes simulation** and do not claim official
CMS status. All efficiencies use hierarchical development-balancing weights, not
cross-section or luminosity normalization.

## Frozen nominal result

The predeclared nominal selection is

\\[
R_{{HH}}^{{125,120}} < 34.
\\]

It selects {nominal['selected_signal_rows']:,} signal candidates and
{nominal['selected_background_rows']:,} background candidates, with

\\[
\\epsilon_S={nominal['weighted_signal_efficiency']:.6f},\qquad
\\epsilon_B={nominal['weighted_background_efficiency']:.6f},\qquad
1/\\epsilon_B={nominal['weighted_inverse_background_efficiency']:.3f}.
\\]

Its development-balanced proxy is
\\(\\epsilon_S/\\sqrt{{\\epsilon_B}}={nominal['balanced_efficiency_proxy']:.6f}\\).

## Working-point interpretation

- The CMS-reference point, \\(R_{{HH}}^{{125,120}}<30\\), has the strongest background
  rejection: \\(1/\\epsilon_B={cms_reference['weighted_inverse_background_efficiency']:.3f}\\).
- The higher-purity point, \\(R_{{HH}}^{{125,120}}<31.5\\), raises the balanced proxy to
  {higher_purity['balanced_efficiency_proxy']:.6f} while retaining less background than the nominal point.
- The higher-efficiency point, \\(R_{{HH}}^{{125,120}}<35.5\\), has the largest balanced proxy among
  the four predeclared points, {higher_efficiency['balanced_efficiency_proxy']:.6f}.
- The nominal threshold remains 34 because the protocol was frozen before this result;
  post-result retuning is not authorized.

These proxies are controlled development diagnostics, not physical significances or
expected limits.

## Contents

- `figures/`: eight figures in paired PNG and PDF formats.
- `tables/`: TSV and LaTeX tables for working points, regions, signal modes, and background families.
- `source/`: frozen train-only source products copied for reproducibility.
- `figure_manifest.tsv` and `table_manifest.tsv`: checksums and file inventory.
- `protected_source_audit.tsv`: source provenance.

## Next gate

Implement and commit the reusable expanded BDT runner, then derive grouped train-only
out-of-fold scores and thresholds without opening validation.
"""
    (temporary_dir / "README.md").write_text(readme, encoding="utf-8")

    checkpoint_files = sorted(
        path
        for path in temporary_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (temporary_dir / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(temporary_dir)}"
            for path in checkpoint_files
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(temporary_dir, output_dir)

    print(json.dumps(report_summary, indent=2, sort_keys=True))
    print()
    print("CUT_WORKING_POINT_TABLES_PASS")
    print("CMS_REGION_TABLES_PASS")
    print("SIGNAL_MODE_TABLE_PASS")
    print("BACKGROUND_FAMILY_TABLE_PASS")
    print("CMS_INSPIRED_LATEX_LABELED_FIGURES_PASS")
    print("PAIRED_PNG_PDF_FIGURES_PASS")
    print("NO_VALIDATION_CANDIDATE_CONTENT_OPENED")
    print("NO_EVALUATION_CANDIDATE_CONTENT_OPENED")
    print("NO_THRESHOLDS_REOPTIMIZED")
    print("HH4B_EXPANDED_CUT_TABLES_FIGURES_PASS")


if __name__ == "__main__":
    main()
