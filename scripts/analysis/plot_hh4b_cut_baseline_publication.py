#!/usr/bin/env python3
"""Render frozen train-only HH4b cut-baseline publication figures.

Every figure is written as vector PDF and 300-dpi PNG with an exact TSV
sidecar.  The labels are CMS-inspired for readability but make no official CMS
claim.  This script never reads validation or test payloads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hh4b_cut_baseline_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hh4b_plot_style import apply_hh4b_paper_style


BRANCH = "delphes-hh4b-production"
CATEGORIES = ("exact3tag", "ge4tag")
CATEGORY_LABELS = {"exact3tag": "exactly 3 tags", "ge4tag": r"$\geq 4$ tags"}
SELECTION_LABELS = {
    "nested_outer_oof": "nested outer-OOF",
    "historical_rhh125125_lt34": r"historical $R_{HH}<34$",
    "fixed_nominal_deployment_cut": "frozen nominal cut",
}
COLORS = {
    "exact3tag": "#0072B2",
    "ge4tag": "#D55E00",
    "signal": "#0072B2",
    "background": "#D55E00",
    "initial200": "#999999",
    "all1000": "#009E73",
    "nested_outer_oof": "#009E73",
    "historical_rhh125125_lt34": "#7F7F7F",
    "fixed_nominal_deployment_cut": "#CC79A7",
}


class PlotError(RuntimeError):
    """Fail-closed publication-plot contract violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlotError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path: Path) -> None:
    require(path.is_dir() and not path.is_symlink(), f"invalid checkpoint: {path}")
    require((path / "SHA256SUMS").is_file(), f"missing SHA256SUMS: {path}")
    check = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(check.returncode == 0, f"checkpoint checksum failure: {check.stdout}{check.stderr}")


def verify_repository(repo: Path) -> str:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", f"origin/{BRANCH}"], cwd=repo, text=True
    ).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    require(head == remote and branch == BRANCH, "repository gate failed")
    return head


def add_header(axes: plt.Axes, subtitle: str = "Train-only") -> None:
    figure = axes.get_figure()
    figure.text(
        0.06, 0.985, "Delphes simulation",
        ha="left", va="top", fontweight="bold", fontsize=11,
    )
    figure.text(
        0.94, 0.985, r"$\sqrt{s}=13$ TeV, 138 fb$^{-1}$ equivalent",
        ha="right", va="top", fontsize=10,
    )
    figure.text(
        0.06, 0.955, subtitle,
        ha="left", va="top", fontsize=9,
    )


def short_structure(value: str) -> str:
    family, suffix = value.split("__category__", 1)
    family_label = {
        "radial_mass": "radial",
        "symmetric_rectangular_mass": "sym. rect.",
        "asymmetric_rectangular_mass": "asym. rect.",
    }[family]
    suffix = suffix.removeprefix("plus_").replace("mass_only", "mass only")
    replacements = {
        "ht_candidate_jets": r"$H_T^{\mathrm{cand.}}$",
        "h2_pt": r"$p_T(H_2)$",
        "max_drbb": r"$\max\Delta R_{bb}$",
        "abs_h_delta_eta": r"$|\Delta\eta(H_1,H_2)|$",
        "mhh": r"$m_{HH}$",
        "_and_": " + ",
    }
    for source, target in replacements.items():
        suffix = suffix.replace(source, target)
    return f"{family_label}: {suffix}"


def write_sidecar(path: Path, frame: pd.DataFrame) -> None:
    require(not frame.empty, f"empty figure sidecar: {path.name}")
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def save_pair(fig: plt.Figure, figure_id: str, png_dir: Path, pdf_dir: Path) -> tuple[Path, Path]:
    png = png_dir / f"{figure_id}.png"
    pdf = pdf_dir / f"{figure_id}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    require(png.is_file() and png.stat().st_size > 0, f"PNG write failure: {figure_id}")
    require(pdf.is_file() and pdf.stat().st_size > 0, f"PDF write failure: {figure_id}")
    return png, pdf


def plot_nested_winners(frame: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    plot = frame[["outer_fold", "category_id", "structure_id"]].copy()
    plot["structure_label"] = plot["structure_id"].map(short_structure)
    labels = sorted(plot["structure_label"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    for ax, category in zip(axes, CATEGORIES):
        part = plot[plot.category_id == category]
        y = [labels.index(value) for value in part.structure_label]
        ax.scatter(part.outer_fold, y, s=95, color=COLORS[category], zorder=3)
        ax.set_xticks(range(5))
        ax.set_xlabel("Outer fold")
        ax.set_title(CATEGORY_LABELS[category])
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=8)
    add_header(axes[0], "Nested fold winner; outer fold not used for selection")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, plot


def plot_frequency_matrix(frame: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    plot = frame.copy()
    plot["column"] = plot["category_id"].map(CATEGORY_LABELS) + ", " + plot[
        "aggregation_level"
    ].map({"fold_winner": "fold winner", "replica_selected": "replica selected"})
    structures = list(dict.fromkeys(plot.sort_values(["category_id", "aggregation_level", "frequency_rank"])[
        "structure_id"
    ]))
    columns = [
        f"{CATEGORY_LABELS[c]}, {level}"
        for c in CATEGORIES for level in ("fold winner", "replica selected")
    ]
    matrix = np.zeros((len(structures), len(columns)))
    for row in plot.itertuples():
        matrix[structures.index(row.structure_id), columns.index(row.column)] = row.frequency
    order = np.argsort(-matrix.max(axis=1))
    matrix = matrix[order]
    structures = [structures[index] for index in order]
    fig, ax = plt.subplots(figsize=(9.4, 10.5))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.01, matrix.max()))
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=25, ha="right")
    ax.set_yticks(range(len(structures)))
    ax.set_yticklabels([short_structure(value) for value in structures], fontsize=7)
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("Selection frequency")
    add_header(ax, "All 1,000 selection-stability replicas")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, plot


def plot_initial_vs_all1000(frame: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    plot = frame.copy()
    metrics = list(dict.fromkeys(plot.metric))
    labels = [value.replace("_frequency", "").replace("_fraction", "").replace("_", " ") for value in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    width = 0.36
    x = np.arange(len(metrics))
    for ax, category in zip(axes, CATEGORIES):
        part = plot[plot.category_id == category].set_index("metric").loc[metrics]
        ax.bar(x - width / 2, part.initial200_value, width, color=COLORS["initial200"], label="initial 200")
        ax.bar(x + width / 2, part.all1000_value, width, color=COLORS["all1000"], label="all 1,000")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(CATEGORY_LABELS[category])
        ax.legend()
    axes[0].set_ylabel("Frequency / fraction")
    add_header(axes[0], "Initial-200 retained separately")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, plot


def plot_nominal_recovery(comparison: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    plot = comparison[comparison.metric == "nominal_structure_recovery_frequency"].copy()
    x = np.arange(2)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.bar(x - width / 2, plot.initial200_value, width, color=COLORS["initial200"], label="initial 200")
    ax.bar(x + width / 2, plot.all1000_value, width, color=COLORS["all1000"], label="all 1,000")
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in plot.category_id])
    ax.set_ylabel("Nominal-structure recovery frequency")
    ax.set_ylim(0.0, max(0.3, 1.15 * plot[["initial200_value", "all1000_value"]].to_numpy().max()))
    ax.legend()
    add_header(ax, "Nominal thresholds unchanged")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, plot


def plot_optional_inclusion(frame: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    plot = frame[frame.aggregation_level == "replica_selected"].copy()
    variables = list(dict.fromkeys(plot.optional_variable))
    labels = {
        "mhh": r"$m_{HH}$", "h2_pt": r"$p_T(H_2)$",
        "ht_candidate_jets": r"$H_T^{\mathrm{cand.}}$", "max_drbb": r"$\max\Delta R_{bb}$",
        "abs_h_delta_eta": r"$|\Delta\eta(H_1,H_2)|$",
    }
    x = np.arange(len(variables))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for offset, category in ((-width / 2, "exact3tag"), (width / 2, "ge4tag")):
        part = plot[plot.category_id == category].set_index("optional_variable").loc[variables]
        ax.bar(x + offset, part.inclusion_frequency, width, color=COLORS[category],
               label=CATEGORY_LABELS[category])
    ax.axhspan(0.30, 0.70, color="#F0E442", alpha=0.18, label="predeclared ambiguous interval")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[value] for value in variables])
    ax.set_ylabel("Replica-level inclusion frequency")
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=2)
    add_header(ax, "All 1,000 selection-stability replicas")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, plot


def plot_modal_tie(modal: pd.DataFrame, category: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    plot = modal.copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for cat in CATEGORIES:
        part = plot[plot.category_id == cat]
        axes[0].plot(part.maximum_winner_count.to_numpy(), part.frequency.to_numpy(), marker="o", lw=2,
                     color=COLORS[cat], label=CATEGORY_LABELS[cat])
    axes[0].set_xlabel("Maximum count among five fold winners")
    axes[0].set_ylabel("Replica fraction")
    axes[0].set_xticks(range(1, 6))
    axes[0].legend()
    metrics = ["unique_modal_replica_category_fraction", "tie_resolution_replica_category_fraction"]
    x = np.arange(len(metrics))
    width = 0.36
    for offset, cat in ((-width / 2, "exact3tag"), (width / 2, "ge4tag")):
        row = category[category.category_id == cat].iloc[0]
        axes[1].bar(x + offset, [row[m] for m in metrics], width, color=COLORS[cat],
                    label=CATEGORY_LABELS[cat])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["unique mode", "tie resolution"])
    axes[1].set_ylabel("Replica fraction")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    add_header(axes[0], "Five-fold replica reduction")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, pd.concat([plot.assign(record_type="modal_distribution"),
                           category.assign(record_type="category_diagnostics")], ignore_index=True)


def plot_conditional_thresholds(frame: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    plot = frame[frame.is_nominal_structure.astype(str).str.lower().isin(["true", "1"])].copy()
    require(not plot.empty, "no nominal conditional threshold summaries")
    threshold_labels = {
        "r_hh_125_125": r"$R_{HH}(125,125)$",
        "ht_candidate_jets": r"$H_T^{\mathrm{cand.}}$",
        "mhh": r"$m_{HH}$",
        "abs_h_delta_eta": r"$|\Delta\eta(H_1,H_2)|$",
    }
    plot["coordinate"] = [
        f"{CATEGORY_LABELS[category]}\n{threshold_labels[variable]}"
        for category, variable in zip(plot.category_id, plot.threshold_variable)
    ]
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    x = np.arange(len(plot))
    low = plot["median"].to_numpy(float) - plot.p16_linear.to_numpy(float)
    high = plot.p84_linear.to_numpy(float) - plot["median"].to_numpy(float)
    colors = [COLORS[value] for value in plot.category_id]
    ax.errorbar(x, plot["median"].to_numpy(float), yerr=np.vstack([low, high]), fmt="none", ecolor="#555555",
                capsize=4, lw=1.6)
    ax.scatter(x, plot["median"].to_numpy(float), c=colors, s=55, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(plot.coordinate, rotation=15, ha="right")
    ax.set_ylabel("Conditional median threshold")
    ax.set_yscale("symlog", linthresh=10.0)
    add_header(ax, "Nominal structure recovered conditionally; 16th--84th percentiles")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, plot


def plot_category_stability(frame: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    plot = frame.copy()
    metrics = [
        "modal_replica_selected_frequency", "nominal_structure_recovery_frequency",
        "top_two_frequency_gap", "unique_modal_replica_category_fraction",
        "tie_resolution_replica_category_fraction", "infeasible_fold_winner_frequency",
    ]
    labels = ["modal", "nominal recovery", "top-two gap", "unique mode", "tie", "infeasible winner"]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.3, 5.2))
    for offset, category in ((-width / 2, "exact3tag"), (width / 2, "ge4tag")):
        row = plot[plot.category_id == category].iloc[0]
        ax.bar(x + offset, [row[value] for value in metrics], width, color=COLORS[category],
               label=CATEGORY_LABELS[category])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Frequency / fraction")
    ax.set_ylim(0, 1)
    ax.legend()
    add_header(ax, "All 1,000 selection-stability replicas")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, plot


def pooled_performance(performance: pd.DataFrame) -> pd.DataFrame:
    return performance[performance.outer_fold.astype(str) == "pooled"].copy()


def plot_efficiency_plane(performance: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    plot = pooled_performance(performance)
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    markers = {"exact3tag": "o", "ge4tag": "s", "combined": "D"}
    for row in plot.itertuples():
        ax.scatter(row.background_physical_efficiency, row.signal_physical_efficiency,
                   marker=markers[row.scope], s=70, color=COLORS[row.selection_id],
                   edgecolor="black", linewidth=0.4)
    for selection in SELECTION_LABELS:
        ax.scatter([], [], s=55, color=COLORS[selection], label=SELECTION_LABELS[selection])
    ax.set_xlabel(r"$\epsilon_{\mathrm{bkg}}$")
    ax.set_ylabel(r"$\epsilon_{S}$")
    ax.legend(fontsize=8)
    add_header(ax, "Pooled train metrics; markers include category and combined scopes")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, plot


def plot_nested_folds(performance: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    plot = performance[(performance.selection_id == "nested_outer_oof")
                       & (performance.outer_fold.astype(str) != "pooled")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for category in (*CATEGORIES, "combined"):
        part = plot[plot.scope == category].sort_values("outer_fold")
        color = COLORS.get(category, "black")
        label = CATEGORY_LABELS.get(category, "combined")
        axes[0].plot(part.outer_fold.astype(int).to_numpy(), part.signal_physical_efficiency.to_numpy(),
                     marker="o", color=color, label=label)
        axes[1].plot(part.outer_fold.astype(int).to_numpy(), part.background_rejection.to_numpy(),
                     marker="o", color=color, label=label)
    axes[0].set_ylabel(r"$\epsilon_S$")
    axes[1].set_ylabel(r"$1/\epsilon_{\mathrm{bkg}}$")
    for ax in axes:
        ax.set_xlabel("Outer fold")
        ax.set_xticks(range(5))
        ax.legend(fontsize=8)
    add_header(axes[0], "Primary train-only generalization estimate")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, plot


def plot_historical_comparison(performance: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    plot = pooled_performance(performance)
    plot = plot[(plot.scope == "combined") & plot.selection_id.isin([
        "historical_rhh125125_lt34", "fixed_nominal_deployment_cut"
    ])].copy()
    metrics = ["signal_physical_efficiency", "background_physical_efficiency", "background_rejection"]
    labels = [r"$\epsilon_S$", r"$\epsilon_{\mathrm{bkg}}$", r"$1/\epsilon_{\mathrm{bkg}}$"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3))
    for ax, metric, label in zip(axes, metrics, labels):
        ax.bar(range(2), plot[metric], color=[COLORS[value] for value in plot.selection_id])
        ax.set_xticks(range(2))
        ax.set_xticklabels([SELECTION_LABELS[value] for value in plot.selection_id], rotation=18, ha="right")
        ax.set_ylabel(label)
    add_header(axes[0], "Fixed selections on identical category-matched train universe")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig, plot


def bar_metric(performance: pd.DataFrame, metric: str, ylabel: str, subtitle: str,
               logy: bool = False) -> tuple[plt.Figure, pd.DataFrame]:
    plot = pooled_performance(performance)
    plot = plot[plot.scope == "combined"].copy()
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.bar(range(len(plot)), plot[metric], color=[COLORS[value] for value in plot.selection_id])
    ax.set_xticks(range(len(plot)))
    ax.set_xticklabels([SELECTION_LABELS[value] for value in plot.selection_id], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    add_header(ax, subtitle)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig, plot


def plot_yields(performance: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    plot = pooled_performance(performance)
    plot = plot[plot.scope == "combined"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
    for ax, sample_class in zip(axes, ("signal", "background")):
        metric = f"{sample_class}_selected_signed_yield"
        ax.bar(range(len(plot)), plot[metric], color=[COLORS[value] for value in plot.selection_id])
        ax.set_xticks(range(len(plot)))
        ax.set_xticklabels([SELECTION_LABELS[value] for value in plot.selection_id], rotation=18, ha="right")
        ax.set_ylabel(f"{sample_class.capitalize()} expected yield")
        ax.set_ylim(bottom=0.0)
    add_header(axes[0], "Run-2 expected-yield projection")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig, plot


def plot_neff_uncertainty(performance: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    plot = pooled_performance(performance)
    plot = plot[plot.scope == "combined"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    x = np.arange(len(plot))
    width = 0.36
    for offset, sample_class in ((-width / 2, "signal"), (width / 2, "background")):
        axes[0].bar(x + offset, plot[f"{sample_class}_selected_effective_events"], width,
                    color=COLORS[sample_class], label=sample_class)
        axes[1].bar(x + offset, plot[f"{sample_class}_finite_mc_relative_uncertainty"], width,
                    color=COLORS[sample_class], label=sample_class)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([SELECTION_LABELS[value] for value in plot.selection_id], rotation=18, ha="right")
        ax.legend()
    axes[0].set_ylabel(r"Selected $N_{\mathrm{eff}}$")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Finite-MC relative uncertainty")
    axes[1].set_yscale("log")
    add_header(axes[0], "Pooled category-combined train metrics")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig, plot


def plot_pre_post_distributions(frame: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    plot = frame.copy()
    grouped = plot.groupby(
        ["sample_class", "selection_stage", "variable", "bin_index", "bin_low_inclusive",
         "bin_high_exclusive", "axis_label_latex"], sort=True, as_index=False
    )["rows"].sum()
    variables = list(dict.fromkeys(grouped.variable))
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.2))
    styles = {"preselection": "--", "postselection": "-"}
    for ax, variable in zip(axes.flat, variables):
        part = grouped[grouped.variable == variable]
        for sample_class in ("signal", "background"):
            for stage in ("preselection", "postselection"):
                line = part[(part.sample_class == sample_class) & (part.selection_stage == stage)]
                total = line.rows.sum()
                require(total > 0, f"empty distribution: {variable}/{sample_class}/{stage}")
                centers = 0.5 * (line.bin_low_inclusive + line.bin_high_exclusive)
                ax.step(centers.to_numpy(), (line.rows / total).to_numpy(), where="mid", color=COLORS[sample_class],
                        linestyle=styles[stage], label=f"{sample_class}, {stage}")
        ax.set_xlabel(part.axis_label_latex.iloc[0])
        ax.set_ylabel("Normalized raw-row fraction")
        ax.set_yscale("log")
    axes[0, 0].legend(fontsize=7, ncol=2)
    add_header(axes[0, 0], "Frozen nominal deployment selection; categories combined")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, grouped


def build_figures(repo: Path, all1000: Path, train: Path, output: Path) -> dict[str, Any]:
    head = verify_repository(repo)
    verify_checkpoint(all1000)
    verify_checkpoint(train)
    all1000_root = all1000 / "evidence/aggregation"
    train_root = train / "evidence/train_performance"
    summary = json.loads((all1000_root / "all1000_stability_summary.json").read_text())
    train_summary = json.loads((train_root / "train_performance_summary.json").read_text())
    require(summary["validation_payloads_opened"] == summary["test_payloads_opened"] == 0,
            "all1000 sealed-data gate failed")
    require(train_summary["validation_payloads_opened"] == train_summary["test_payloads_opened"] == 0,
            "train-performance sealed-data gate failed")
    require(train_summary["primary_generalization_estimate"] == "pooled_nested_outer_oof",
            "primary generalization estimate changed")

    nested_winners = pd.read_csv(
        repo / "docs/checkpoints/hh4b_train_multivariate_cut_train_only_nested_oof_aggregation_20260807_v1/"
        "evidence/outer_fold_category_winners.tsv", sep="\t"
    )
    frequency = pd.read_csv(all1000_root / "tables/structure_frequencies.tsv", sep="\t")
    comparison = pd.read_csv(all1000_root / "tables/initial200_vs_all1000_category_diagnostics.tsv", sep="\t")
    optional = pd.read_csv(all1000_root / "tables/optional_variable_inclusion_frequencies.tsv", sep="\t")
    modal = pd.read_csv(all1000_root / "tables/modal_count_distribution.tsv", sep="\t")
    category = pd.read_csv(all1000_root / "tables/category_stability_diagnostics.tsv", sep="\t")
    thresholds = pd.read_csv(all1000_root / "tables/conditional_threshold_distribution_summary.tsv", sep="\t")
    performance = pd.read_csv(train_root / "tables/train_performance_metrics.tsv", sep="\t")
    distributions = pd.read_csv(
        train_root / "figure_data/pre_post_nominal_variable_distributions.tsv", sep="\t"
    )

    output.mkdir(parents=True)
    pdf_dir = output / "figures/pdf"
    png_dir = output / "figures/png"
    data_dir = output / "figure_data"
    manifest_dir = output / "manifests"
    for path in (pdf_dir, png_dir, data_dir, manifest_dir):
        path.mkdir(parents=True)
    apply_hh4b_paper_style()

    builders: list[tuple[str, Callable[[], tuple[plt.Figure, pd.DataFrame]], str]] = [
        ("01_nested_outer_fold_winners", lambda: plot_nested_winners(nested_winners), "nested outer-fold winner summary"),
        ("02_all1000_cut_family_frequency_matrix", lambda: plot_frequency_matrix(frequency), "cut-family frequency matrix"),
        ("03_initial200_vs_all1000_stability", lambda: plot_initial_vs_all1000(comparison), "initial200 vs all1000 diagnostics"),
        ("04_nominal_structure_recovery", lambda: plot_nominal_recovery(comparison), "nominal recovery"),
        ("05_optional_variable_inclusion", lambda: plot_optional_inclusion(optional), "optional-variable inclusion"),
        ("06_modal_count_and_ties", lambda: plot_modal_tie(modal, category), "modal-count and tie diagnostics"),
        ("07_conditional_nominal_thresholds", lambda: plot_conditional_thresholds(thresholds), "conditional threshold distributions"),
        ("08_exact3tag_ge4tag_stability", lambda: plot_category_stability(category), "category stability comparison"),
        ("09_signal_background_efficiency_plane", lambda: plot_efficiency_plane(performance), "signal/background efficiency plane"),
        ("10_nested_outer_oof_fold_performance", lambda: plot_nested_folds(performance), "nested outer-OOF performance"),
        ("11_historical_rhh34_comparison", lambda: plot_historical_comparison(performance), "historical comparator"),
        ("12_run2_equivalent_yields", lambda: plot_yields(performance), "Run-2-equivalent yields"),
        ("13_signal_over_background", lambda: bar_metric(performance, "signal_over_background", "S/B", "Pooled category-combined train metrics"), "S/B comparison"),
        ("14_asimov_significance_stat_only", lambda: bar_metric(performance, "asimov_significance_stat_only", r"Stat-only $Z_A$", "No nuisance parameters included"), "stat-only Asimov significance"),
        ("15_neff_finite_mc", lambda: plot_neff_uncertainty(performance), "effective counts and finite-MC uncertainty"),
        ("16_pre_post_nominal_distributions", lambda: plot_pre_post_distributions(distributions), "pre/post variable distributions"),
    ]
    records: list[dict[str, Any]] = []
    for figure_id, builder, title in builders:
        fig, sidecar = builder()
        sidecar_path = data_dir / f"{figure_id}.tsv"
        write_sidecar(sidecar_path, sidecar)
        png, pdf = save_pair(fig, figure_id, png_dir, pdf_dir)
        records.append({
            "figure_id": figure_id,
            "title": title,
            "pdf_path": str(pdf.relative_to(repo)),
            "pdf_bytes": pdf.stat().st_size,
            "pdf_sha256": sha256_file(pdf),
            "png_path": str(png.relative_to(repo)),
            "png_bytes": png.stat().st_size,
            "png_sha256": sha256_file(png),
            "sidecar_path": str(sidecar_path.relative_to(repo)),
            "sidecar_rows": len(sidecar),
            "sidecar_sha256": sha256_file(sidecar_path),
            "header": "Delphes simulation",
            "official_cms_status_claimed": False,
            "validation_payloads_opened": 0,
            "test_payloads_opened": 0,
        })
    require(len(records) == 16, "publication figure count mismatch")
    fields = tuple(records[0])
    with (manifest_dir / "figure_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    result = {
        "schema_version": 1,
        "status": "pass_hh4b_cut_baseline_train_only_publication_figures",
        "repository_head_at_execution": head,
        "figures": len(records),
        "pdfs": len(records),
        "pngs": len(records),
        "figure_data_sidecars": len(records),
        "official_cms_status_claimed": False,
        "header": "Delphes simulation",
        "energy_luminosity": "sqrt(s)=13 TeV, 138 fb^-1 equivalent",
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "next": "bind_figures_in_master_train_only_freeze_before_validation",
    }
    (output / "publication_figure_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_paths = sorted(path for path in output.rglob("*") if path.is_file())
    with (output / "SHA256SUMS").open("w", encoding="utf-8", newline="") as handle:
        for path in checksum_paths:
            handle.write(f"{sha256_file(path)}  ./{path.relative_to(output).as_posix()}\n")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"))
    parser.add_argument("--all1000-checkpoint", type=Path, default=Path(
        "docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_all1000_aggregation_20260810_v1"
    ))
    parser.add_argument("--train-performance-checkpoint", type=Path, default=Path(
        "docs/checkpoints/hh4b_train_cut_baseline_performance_and_comparator_20260810_v1"
    ))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/hh4b_cut_baseline/publication_train_only"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    all1000 = (args.all1000_checkpoint if args.all1000_checkpoint.is_absolute()
               else repo / args.all1000_checkpoint).resolve()
    train = (args.train_performance_checkpoint if args.train_performance_checkpoint.is_absolute()
             else repo / args.train_performance_checkpoint).resolve()
    output = (args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir).absolute()
    require(all1000.is_relative_to(repo) and train.is_relative_to(repo) and output.is_relative_to(repo),
            "all inputs/outputs must be repository paths")
    require(not output.exists() and not output.is_symlink(), f"output already exists: {output}")
    try:
        result = build_figures(repo, all1000, train, output)
    except Exception:
        if output.is_dir() and not output.is_symlink():
            (output / "PLOTTING_FAILED_DO_NOT_USE.txt").write_text(
                "STATUS=FAIL\nDO_NOT_USE_PARTIAL_OUTPUT=TRUE\n"
                "VALIDATION_PAYLOADS_OPENED=0\nTEST_PAYLOADS_OPENED=0\n",
                encoding="utf-8",
            )
        raise
    print("HH4B_CUT_BASELINE_PUBLICATION_FIGURES=PASS")
    print(f"FIGURES={result['figures']} PDFS={result['pdfs']} PNGS={result['pngs']}")
    print("OFFICIAL_CMS_STATUS_CLAIMED=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlotError as exc:
        print(f"HH4B_CUT_BASELINE_PUBLICATION_FIGURES=FAIL\nERROR={exc}", file=sys.stderr)
        print("VALIDATION_PAYLOADS_OPENED=0\nTEST_PAYLOADS_OPENED=0", file=sys.stderr)
        raise SystemExit(1) from exc
