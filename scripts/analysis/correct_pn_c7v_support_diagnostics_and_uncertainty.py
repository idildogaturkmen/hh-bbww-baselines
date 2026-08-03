#!/usr/bin/env python3
"""Build the immutable c7v errata and paired-bootstrap uncertainty checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import publish_pn_c7v_results_and_figures as c7v
from hh4b_plot_style import apply_hh4b_paper_style
from pn_c7_bootstrap_common import (
    BOOTSTRAP_METRICS,
    _support_reasons,
    bootstrap_validity_summary,
    evaluate_frozen_baselines,
    group_count,
    group_sum,
    member_positions,
    paired_model_differences,
    summarize_metric_replicas,
)
from pn_c7_ml_common import (
    artifact_rows,
    asimov_za,
    asimov_za_with_uncertainty,
    bootstrap_quantile_summary,
    parse_sha256sums,
    prepare_staging,
    require,
    seal_checkpoint,
    sha256,
    source_member_bootstrap_draws,
    threshold_at_efficiency,
    verify_all_checkpoints,
    write_json,
    write_tsv,
)


REPO = Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines")
C7V_V2 = Path("/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/pn_c7v_jhep_results_and_figures_20260803_v2")
C7V_V2_MANIFEST_SHA256 = "5c7daba43f7aa6cf2bbc1d91411a819f68cee482a49ba4c635724e3ff5c6174c"
MODEL_ORDER = c7v.MODEL_ORDER
COLORS = c7v.COLORS
DISPLAY = c7v.DISPLAY
BOOTSTRAP_SEED = 20260802
BOOTSTRAP_REPLICATES = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-output", type=Path, required=True)
    parser.add_argument("--inspection-report", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def verify_c7v_v2() -> list[dict[str, Any]]:
    manifest = C7V_V2 / "SHA256SUMS"
    require(sha256(manifest) == C7V_V2_MANIFEST_SHA256, "c7v v2 manifest identity drift")
    rows = []
    for relative, expected in parse_sha256sums(manifest).items():
        path = C7V_V2 / relative
        require(path.is_file(), f"missing c7v v2 member: {path}")
        observed = sha256(path)
        require(observed == expected, f"c7v v2 checksum drift: {path}")
        rows.append({
            "checkpoint": "c7v_v2",
            "checkpoint_root": str(C7V_V2),
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": observed,
        })
    rows.append({
        "checkpoint": "c7v_v2",
        "checkpoint_root": str(C7V_V2),
        "relative_path": "SHA256SUMS",
        "bytes": manifest.stat().st_size,
        "sha256": sha256(manifest),
    })
    return rows


def replace_plot(
    provenance: list[dict[str, Any]],
    fig: Any,
    figures: Path,
    paper_figures: Path,
    stem: str,
    caption: str,
    rows: list[dict[str, Any]],
) -> None:
    provenance[:] = [row for row in provenance if row["stem"] != stem]
    c7v.save_plot(fig, figures, paper_figures, stem, caption, rows, provenance)


def quantile_fields(values: np.ndarray, *, prefix: str, reasons: list[str] | None = None) -> dict[str, Any]:
    finite = np.isfinite(np.asarray(values, dtype=float))
    if not finite.any():
        reason_counts: dict[str, int] = {}
        for reason in reasons or []:
            key = str(reason) or "nonfinite metric"
            reason_counts[key] = reason_counts.get(key, 0) + 1
        return {
            f"{prefix}_bootstrap_mean": float("nan"), f"{prefix}_bootstrap_median": float("nan"),
            f"{prefix}_bootstrap_p16": float("nan"), f"{prefix}_bootstrap_p84": float("nan"),
            f"{prefix}_bootstrap_p2p5": float("nan"), f"{prefix}_bootstrap_p97p5": float("nan"),
            f"{prefix}_valid_replicas": 0, f"{prefix}_invalid_replicas": len(values),
            f"{prefix}_invalid_reason_counts_json": json.dumps(reason_counts, sort_keys=True, separators=(",", ":")),
        }
    summary = bootstrap_quantile_summary(values, invalid_reasons=reasons)
    return {
        f"{prefix}_bootstrap_mean": summary["bootstrap_mean"],
        f"{prefix}_bootstrap_median": summary["bootstrap_median"],
        f"{prefix}_bootstrap_p16": summary["bootstrap_p16"],
        f"{prefix}_bootstrap_p84": summary["bootstrap_p84"],
        f"{prefix}_bootstrap_p2p5": summary["bootstrap_p2p5"],
        f"{prefix}_bootstrap_p97p5": summary["bootstrap_p97p5"],
        f"{prefix}_valid_replicas": summary["valid_replicas"],
        f"{prefix}_invalid_replicas": summary["invalid_replicas"],
        f"{prefix}_invalid_reason_counts_json": summary["invalid_reason_counts_json"],
    }


def scan_with_bootstrap(inputs: dict[str, Any], draws: np.ndarray, *, points: int) -> pd.DataFrame:
    training = inputs["training"]
    projection = inputs["projection"]
    direct = inputs["direct"]
    train_scores = inputs["training_scores"]
    projection_scores = inputs["projection_scores"]
    members = inputs["members"].reset_index(drop=True)
    n_members = len(members)
    train_pos = member_positions(training, members, "scan training")
    projection_pos = member_positions(projection, members, "scan projection")
    direct_pos = member_positions(direct, members, "scan direct closure")
    evaluation_draws = np.vstack([np.ones((1, n_members), dtype=np.int16), draws])

    signal_train = training["registry_training_target"].to_numpy(dtype=np.int8) == 1
    dev_weight = training["development_hierarchical_weight"].to_numpy(dtype=np.float64)
    signal_projection = projection["registry_training_target"].to_numpy(dtype=np.int8) == 1
    roles = projection["analysis_population_role"].astype(str).to_numpy()
    ordinary = roles == "fourb_ordinary_background"
    transferred = roles == "primary_transferred_multijet_template"
    direct_qcd = direct["population_kind"].astype(str).to_numpy() == "hard_qcd"
    physical = projection["primary_projection_physical_weight_inclusive"].to_numpy(dtype=np.float64)
    qcd_base_weight = projection["run2_candidate_physical_weight"].to_numpy(dtype=np.float64)
    direct_weight = direct["direct_projection_physical_weight"].to_numpy(dtype=np.float64)
    factor_values, factor_stat = c7v.factor_contract(inputs)
    factor_values_array = np.asarray(factor_values, dtype=np.float64)
    nominal_factor = factor_values[0]

    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        train_score = train_scores[f"{model}_score"].to_numpy(dtype=np.float64)
        projection_score = projection_scores[f"{model}_score"].to_numpy(dtype=np.float64)
        order = np.argsort(-train_score[signal_train], kind="mergesort")
        ordered_scores = train_score[signal_train][order]
        ordered_weights = dev_weight[signal_train][order]
        cumulative = np.cumsum(ordered_weights) / ordered_weights.sum()
        for target in np.linspace(0.10, 1.0, points):
            threshold_index = min(int(np.searchsorted(cumulative, target, side="left")), len(ordered_scores) - 1)
            threshold = float(ordered_scores[threshold_index])
            train_selected = train_score >= threshold
            projection_selected = projection_score >= threshold
            direct_selected = train_selected

            def gsum(frame_values: np.ndarray, mask: np.ndarray, positions: np.ndarray) -> np.ndarray:
                return evaluation_draws @ group_sum(frame_values, mask, positions, n_members)

            signal_rows = evaluation_draws @ group_count(projection_selected & signal_projection, projection_pos, n_members)
            ordinary_rows = evaluation_draws @ group_count(projection_selected & ordinary, projection_pos, n_members)
            transferred_row_group = group_count(projection_selected & transferred, projection_pos, n_members)
            direct_row_group = group_count(direct_selected & direct_qcd, direct_pos, n_members)
            transferred_rows = evaluation_draws @ transferred_row_group
            direct_rows = evaluation_draws @ direct_row_group
            signal_yield = gsum(physical, projection_selected & signal_projection, projection_pos)
            ordinary_yield = gsum(physical, projection_selected & ordinary, projection_pos)
            qcd_base = gsum(qcd_base_weight, projection_selected & transferred, projection_pos)
            qcd_yield = qcd_base * nominal_factor
            qcd_sumw2 = gsum(np.square(physical), projection_selected & transferred, projection_pos)
            direct_yield = gsum(direct_weight, direct_selected & direct_qcd, direct_pos)
            direct_sumw2 = gsum(np.square(direct_weight), direct_selected & direct_qcd, direct_pos)
            background_yield = ordinary_yield + qcd_yield
            background_sumw2 = (
                gsum(np.square(physical), projection_selected & ordinary, projection_pos) + qcd_sumw2
            )
            background_neff = np.divide(np.square(background_yield), background_sumw2,
                                        out=np.full_like(background_yield, np.nan), where=background_sumw2 > 0)
            transferred_neff = np.divide(np.square(qcd_yield), qcd_sumw2,
                                         out=np.full_like(qcd_yield, np.nan), where=qcd_sumw2 > 0)
            direct_neff = np.divide(np.square(direct_yield), direct_sumw2,
                                    out=np.full_like(direct_yield, np.nan), where=direct_sumw2 > 0)
            selected_background_dev = gsum(dev_weight, train_selected & ~signal_train, train_pos)
            total_background_dev = gsum(dev_weight, ~signal_train, train_pos)
            rejection = 1.0 - selected_background_dev / total_background_dev
            nominal_za = np.asarray([asimov_za(float(s), float(b)) for s, b in zip(signal_yield, background_yield)])

            raw_systematic = np.full(len(evaluation_draws), np.nan)
            support = np.zeros(len(evaluation_draws), dtype=bool)
            reasons: list[str] = []
            transferred_positions = np.flatnonzero(transferred_row_group > 0)
            direct_positions = np.flatnonzero(direct_row_group > 0)
            nonclosure_values = np.full(len(evaluation_draws), np.nan)
            envelope_values = np.full(len(evaluation_draws), np.nan)
            sigma_values = np.full(len(evaluation_draws), np.nan)
            transferred_sources_values = np.zeros(len(evaluation_draws), dtype=int)
            direct_sources_values = np.zeros(len(evaluation_draws), dtype=int)
            for index, multiplicities in enumerate(evaluation_draws):
                transferred_sources = int(np.count_nonzero(multiplicities[transferred_positions]))
                direct_sources = int(np.count_nonzero(multiplicities[direct_positions]))
                transferred_sources_values[index] = transferred_sources
                direct_sources_values[index] = direct_sources
                qcd = float(qcd_yield[index])
                truth = float(direct_yield[index])
                nonclosure = abs(qcd - truth) / abs(truth) if truth != 0.0 else float("nan")
                alternatives = qcd_base[index] * factor_values_array
                envelope = float(np.max(np.abs(alternatives - qcd)) / abs(qcd)) if qcd != 0.0 else float("nan")
                sigma = abs(qcd) * math.sqrt(factor_stat**2 + envelope**2 + nonclosure**2) if math.isfinite(nonclosure) and math.isfinite(envelope) else float("nan")
                raw = asimov_za_with_uncertainty(float(signal_yield[index]), float(background_yield[index]), sigma) if math.isfinite(sigma) and sigma > 0 else float("nan")
                point_reasons = _support_reasons(float(transferred_rows[index]), transferred_sources,
                                                  float(transferred_neff[index]), float(direct_rows[index]),
                                                  direct_sources, raw)
                nonclosure_values[index] = nonclosure
                envelope_values[index] = envelope
                sigma_values[index] = sigma
                raw_systematic[index] = raw
                support[index] = not point_reasons
                reasons.append("; ".join(point_reasons))
            qualified = raw_systematic.copy()
            qualified[~support] = np.nan
            row = {
                "model": model,
                "target_signal_efficiency": float(target),
                "classifier_threshold": threshold,
                "selected_signal_rows": int(signal_rows[0]),
                "selected_signal_yield": float(signal_yield[0]),
                "selected_ordinary_background_rows": int(ordinary_rows[0]),
                "selected_ordinary_background_yield": float(ordinary_yield[0]),
                "selected_transferred_qcd_rows": int(transferred_rows[0]),
                "selected_transferred_qcd_sources": int(transferred_sources_values[0]),
                "selected_transferred_qcd_yield": float(qcd_yield[0]),
                "selected_transferred_qcd_neff": float(transferred_neff[0]),
                "selected_direct_qcd_closure_rows": int(direct_rows[0]),
                "selected_direct_qcd_closure_sources": int(direct_sources_values[0]),
                "selected_direct_qcd_closure_yield": float(direct_yield[0]),
                "selected_direct_qcd_closure_neff": float(direct_neff[0]),
                "background_yield": float(background_yield[0]),
                "background_neff": float(background_neff[0]),
                "direct_qcd_nonclosure_factor": float(nonclosure_values[0]),
                "normalization_region_envelope": float(envelope_values[0]),
                "transfer_factor_statistical_relative": factor_stat,
                "multijet_nuisance_absolute": float(sigma_values[0]),
                "nominal_asimov_ZA": float(nominal_za[0]),
                "raw_systematic_aware_asimov_ZA": float(raw_systematic[0]),
                "raw_systematic_finite": bool(math.isfinite(raw_systematic[0])),
                "support_pass": bool(support[0]),
                "support_failure_reason": reasons[0],
                **quantile_fields(rejection[1:], prefix="background_rejection"),
                **quantile_fields(nominal_za[1:], prefix="nominal_ZA"),
                **quantile_fields(qualified[1:], prefix="systematic_ZA", reasons=reasons[1:]),
                "uncertainty_central": float(raw_systematic[0]),
                "uncertainty_lower_68": float(np.nanquantile(qualified[1:], 0.16)) if np.isfinite(qualified[1:]).any() else "",
                "uncertainty_upper_68": float(np.nanquantile(qualified[1:], 0.84)) if np.isfinite(qualified[1:]).any() else "",
                "uncertainty_valid_replicas": int(np.isfinite(qualified[1:]).sum()),
                "uncertainty_support_flag": bool(support[0]),
                "uncertainty_kind": "paired_source_member_bootstrap_p16_p84",
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_supported_segments(ax: Any, x: np.ndarray, y: np.ndarray, support: np.ndarray, **kwargs: Any) -> None:
    start = None
    for index, valid in enumerate(support):
        if valid and start is None:
            start = index
        if start is not None and (not valid or index == len(support) - 1):
            stop = index if not valid else index + 1
            ax.plot(x[start:stop], y[start:stop], **kwargs)
            start = None


def replace_scan_figures(
    scans: pd.DataFrame,
    provenance: list[dict[str, Any]],
    figures: Path,
    paper_figures: Path,
    *,
    min_band_fraction: float,
) -> None:
    apply_hh4b_paper_style()
    scan_rows = scans.to_dict("records")
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for model in MODEL_ORDER:
        sub = scans[scans.model == model]
        x = sub.target_signal_efficiency.to_numpy(float)
        central = sub.nominal_asimov_ZA.to_numpy(float)
        lo = sub.nominal_ZA_bootstrap_p16.to_numpy(float)
        hi = sub.nominal_ZA_bootstrap_p84.to_numpy(float)
        ax.plot(x, central, color=COLORS[model], lw=2, label=DISPLAY[model])
        ax.fill_between(x, lo, hi, color=COLORS[model], alpha=0.16, linewidth=0)
    ax.set(xlabel="Target signal efficiency", ylabel=r"Nominal $Z_A$", xlim=(0.1, 1.0))
    ax.grid(alpha=0.18); ax.legend(); c7v.paper_label(ax)
    replace_plot(provenance, fig, figures, paper_figures, "fig04_nominal_za_scan",
                 r"Nominal statistical-only Asimov sensitivity scan. Bands are paired source-member bootstrap 16th--84th percentile intervals evaluated at frozen thresholds.", scan_rows)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.1), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.35]})
    for model in MODEL_ORDER:
        sub = scans[scans.model == model]
        x = sub.target_signal_efficiency.to_numpy(float)
        y = sub.raw_systematic_aware_asimov_ZA.to_numpy(float)
        support = sub.support_pass.to_numpy(bool) & np.isfinite(y) & (y > 0)
        _plot_supported_segments(axes[0], x, y, support, color=COLORS[model], lw=1.6)
        axes[0].scatter(x[support], y[support], color=COLORS[model], s=17, label=DISPLAY[model])
        band = support & (sub.systematic_ZA_valid_replicas.to_numpy(int) >= min_band_fraction)
        _plot_supported_segments(axes[0], x, sub.systematic_ZA_bootstrap_p16.to_numpy(float), band,
                                 color=COLORS[model], lw=0.0)
        axes[0].fill_between(x, sub.systematic_ZA_bootstrap_p16.to_numpy(float),
                             sub.systematic_ZA_bootstrap_p84.to_numpy(float), where=band,
                             color=COLORS[model], alpha=0.15, interpolate=False)
        unsupported = ~support
        if unsupported.any():
            axes[0].scatter(x[unsupported], np.full(unsupported.sum(), 1.0e-8),
                            marker="x", color=COLORS[model], s=23)
        axes[1].plot(x, sub.selected_transferred_qcd_rows.to_numpy(float), color=COLORS[model], lw=1.4)
        axes[1].plot(x, sub.selected_direct_qcd_closure_rows.to_numpy(float), color=COLORS[model], lw=1.1, ls="--")
    axes[0].set_yscale("log"); axes[0].set_ylabel(r"Systematics-aware $Z_A$")
    axes[0].grid(alpha=0.18); axes[0].legend(ncol=2, fontsize=8); c7v.paper_label(axes[0])
    axes[1].axhline(5, color="0.5", ls=":", lw=1, label="Transferred-row minimum")
    axes[1].set_yscale("symlog", linthresh=1); axes[1].set_ylim(bottom=0); axes[1].set_ylabel("Selected QCD rows")
    axes[1].set_xlabel("Target signal efficiency"); axes[1].grid(alpha=0.18)
    axes[1].text(0.01, 0.96, r"solid: transferred $3b$; dashed: direct $\geq4b$ closure", transform=axes[1].transAxes,
                 va="top", fontsize=8)
    replace_plot(provenance, fig, figures, paper_figures, "fig05_systematic_za_scan",
                 r"Support-qualified systematics-aware Asimov scan. Markers are supported central points; source-member bootstrap 68\% bands are drawn only when at least 80\% of replicas pass the frozen support contract. Crosses denote unsupported points and no line or band is interpolated across them. The lower panel shows transferred exactly-$3b$ rows (solid) and direct $\geq4b$ QCD closure rows (dashed).", scan_rows)

    fig, ax = plt.subplots(figsize=(7.0, 5.3))
    for model in MODEL_ORDER:
        sub = scans[scans.model == model]
        y = sub.raw_systematic_aware_asimov_ZA.to_numpy(float)
        finite = np.isfinite(y) & (y > 0)
        ax.scatter(sub.target_signal_efficiency.to_numpy(float)[finite], y[finite], s=14,
                   color=COLORS[model], label=DISPLAY[model])
    ax.set(xlabel="Target signal efficiency", ylabel=r"Raw systematics-aware $Z_A$", yscale="log", xlim=(0.1, 1.0))
    ax.grid(alpha=0.18); ax.legend(); c7v.paper_label(ax, subtitle="Raw appendix diagnostic")
    replace_plot(provenance, fig, figures, paper_figures, "fig18_raw_systematic_za_scan",
                 r"Raw high-precision systematics-aware scan before support qualification. Points are not connected. This appendix diagnostic exposes sparse-support behavior without interpolation or zero filling.", scan_rows)


def roc_band_rows(inputs: dict[str, Any], draws: np.ndarray, *, weighted: bool, points: int) -> pd.DataFrame:
    training = inputs["training"]
    scores = inputs["training_scores"]
    members = inputs["members"].reset_index(drop=True)
    positions = member_positions(training, members, "ROC training")
    labels = training["registry_training_target"].to_numpy(dtype=np.int8)
    base_weight = training["development_hierarchical_weight"].to_numpy(float) if weighted else np.ones(len(training))
    rows = []
    for model in c7v.LEARNED_MODELS:
        score = scores[f"{model}_score"].to_numpy(float)
        fpr, tpr, thresholds = roc_curve(labels, score, sample_weight=base_weight)
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(points, len(fpr))).astype(int))
        for index in keep:
            selected = score >= thresholds[index]
            sig_num = draws @ group_sum(base_weight, selected & (labels == 1), positions, len(members))
            sig_den = draws @ group_sum(base_weight, labels == 1, positions, len(members))
            bkg_num = draws @ group_sum(base_weight, selected & (labels == 0), positions, len(members))
            bkg_den = draws @ group_sum(base_weight, labels == 0, positions, len(members))
            boot_tpr = sig_num / sig_den
            boot_fpr = bkg_num / bkg_den
            tpr_summary = bootstrap_quantile_summary(boot_tpr)
            fpr_summary = bootstrap_quantile_summary(boot_fpr)
            rows.append({
                "model": model, "threshold": float(thresholds[index]), "false_positive_rate": float(fpr[index]),
                "true_positive_rate": float(tpr[index]), "false_positive_rate_p16": fpr_summary["bootstrap_p16"],
                "false_positive_rate_p84": fpr_summary["bootstrap_p84"], "true_positive_rate_p16": tpr_summary["bootstrap_p16"],
                "true_positive_rate_p84": tpr_summary["bootstrap_p84"], "valid_replicas": tpr_summary["valid_replicas"],
                "support_flag": True, "weighting": "development_hierarchical_weight" if weighted else "unweighted",
                "uncertainty_central": float(tpr[index]), "uncertainty_lower_68": tpr_summary["bootstrap_p16"],
                "uncertainty_upper_68": tpr_summary["bootstrap_p84"], "uncertainty_valid_replicas": tpr_summary["valid_replicas"],
                "uncertainty_support_flag": True, "uncertainty_kind": "paired_source_member_bootstrap_p16_p84",
            })
    cut_selected = training["r_hh_125_120"].to_numpy(float) < 34.0
    sig_num = draws @ group_sum(base_weight, cut_selected & (labels == 1), positions, len(members))
    sig_den = draws @ group_sum(base_weight, labels == 1, positions, len(members))
    bkg_num = draws @ group_sum(base_weight, cut_selected & (labels == 0), positions, len(members))
    bkg_den = draws @ group_sum(base_weight, labels == 0, positions, len(members))
    boot_tpr = sig_num / sig_den; boot_fpr = bkg_num / bkg_den
    tpr_summary = bootstrap_quantile_summary(boot_tpr); fpr_summary = bootstrap_quantile_summary(boot_fpr)
    central_tpr = float(base_weight[cut_selected & (labels == 1)].sum() / base_weight[labels == 1].sum())
    central_fpr = float(base_weight[cut_selected & (labels == 0)].sum() / base_weight[labels == 0].sum())
    rows.append({
        "model": "cut_operating_point", "threshold": -34.0, "false_positive_rate": central_fpr,
        "true_positive_rate": central_tpr, "false_positive_rate_p16": fpr_summary["bootstrap_p16"],
        "false_positive_rate_p84": fpr_summary["bootstrap_p84"], "true_positive_rate_p16": tpr_summary["bootstrap_p16"],
        "true_positive_rate_p84": tpr_summary["bootstrap_p84"], "valid_replicas": len(draws), "support_flag": True,
        "weighting": "development_hierarchical_weight" if weighted else "unweighted",
        "uncertainty_central": central_tpr, "uncertainty_lower_68": tpr_summary["bootstrap_p16"],
        "uncertainty_upper_68": tpr_summary["bootstrap_p84"], "uncertainty_valid_replicas": len(draws),
        "uncertainty_support_flag": True, "uncertainty_kind": "paired_source_member_bootstrap_p16_p84",
    })
    return pd.DataFrame(rows)


def replace_roc_figure(rows: pd.DataFrame, provenance: list[dict[str, Any]], figures: Path,
                       paper_figures: Path, *, weighted: bool) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for model in c7v.LEARNED_MODELS:
        sub = rows[rows.model == model]
        x = sub.false_positive_rate.to_numpy(float)
        y = sub.true_positive_rate.to_numpy(float)
        lo = sub.true_positive_rate_p16.to_numpy(float)
        hi = sub.true_positive_rate_p84.to_numpy(float)
        ax.plot(x, y, color=COLORS[model], lw=2, label=DISPLAY[model])
        ax.fill_between(x, lo, hi,
                        color=COLORS[model], alpha=0.16, linewidth=0)
    cut = rows[rows.model == "cut_operating_point"].iloc[0]
    ax.errorbar([cut.false_positive_rate], [cut.true_positive_rate],
                xerr=[[cut.false_positive_rate-cut.false_positive_rate_p16], [cut.false_positive_rate_p84-cut.false_positive_rate]],
                yerr=[[cut.true_positive_rate-cut.true_positive_rate_p16], [cut.true_positive_rate_p84-cut.true_positive_rate]],
                marker="*", ms=10, color=COLORS["cut"], capsize=3, label=DISPLAY["cut"])
    ax.plot([0, 1], [0, 1], ls="--", color="0.65", lw=1)
    ax.set(xlabel="Background efficiency", ylabel="Signal efficiency", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.18); ax.legend(loc="lower right")
    c7v.paper_label(ax, subtitle="Train-only source-group OOF" if weighted else "Unweighted appendix diagnostic")
    stem = "fig01_weighted_roc" if weighted else "fig16_unweighted_roc_appendix"
    caption = ("Weighted" if weighted else "Unweighted appendix") + " train-only source-group OOF ROC curves. Shaded regions are pointwise paired source-member bootstrap 16th--84th percentile bands at the frozen score thresholds."
    replace_plot(provenance, fig, figures, paper_figures, stem, caption, rows.to_dict("records"))


def score_distribution_rows(inputs: dict[str, Any], draws: np.ndarray) -> pd.DataFrame:
    training = inputs["training"]; projection = inputs["projection"]; direct = inputs["direct"]
    members = inputs["members"].reset_index(drop=True); n_members = len(members)
    train_pos = member_positions(training, members, "score-shape training")
    projection_pos = member_positions(projection, members, "score-shape projection")
    direct_pos = member_positions(direct, members, "score-shape direct closure")
    train_score = inputs["training_scores"]["bdt_score"].to_numpy(float)
    projection_score = inputs["projection_scores"]["bdt_score"].to_numpy(float)
    labels = training["registry_training_target"].to_numpy(int)
    dev_weight = training["development_hierarchical_weight"].to_numpy(float)
    roles = projection["analysis_population_role"].astype(str).to_numpy()
    physical = projection["primary_projection_physical_weight_inclusive"].to_numpy(float)
    direct_weight = direct["direct_projection_physical_weight"].to_numpy(float)
    direct_mask = direct["population_kind"].astype(str).to_numpy() == "hard_qcd"
    factor_values, factor_stat = c7v.factor_contract(inputs)
    envelope = max(abs(value-factor_values[0]) for value in factor_values) / abs(factor_values[0])
    bins = np.linspace(0.0, 1.0, 31)
    rows = []

    definitions = [
        ("unit_shape", "signal", train_score, dev_weight, labels == 1, train_pos, True),
        ("unit_shape", "background", train_score, dev_weight, labels == 0, train_pos, True),
        ("physical_yield", "signal", projection_score, physical, roles == "fourb_signal", projection_pos, False),
        ("physical_yield", "ordinary", projection_score, physical, roles == "fourb_ordinary_background", projection_pos, False),
        ("physical_yield", "transferred", projection_score, physical, roles == "primary_transferred_multijet_template", projection_pos, False),
        ("physical_yield", "direct_closure", train_score, direct_weight, direct_mask, direct_pos, False),
    ]
    by_key: dict[tuple[str, str, int], tuple[np.ndarray, float, float]] = {}
    for panel, series, values, weights, base_mask, positions, normalize in definitions:
        central_bins = []
        replica_bins = []
        stat_bins = []
        for bin_index, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            include_high = values <= high if bin_index == len(bins)-2 else values < high
            mask = base_mask & (values >= low) & include_high
            group = group_sum(weights, mask, positions, n_members)
            boot = draws @ group
            central_bins.append(float(group.sum()))
            replica_bins.append(boot)
            stat_bins.append(math.sqrt(float(np.square(weights[mask]).sum())))
        central_array = np.asarray(central_bins)
        replica_array = np.stack(replica_bins, axis=1)
        stat_array = np.asarray(stat_bins)
        if normalize:
            central_total = central_array.sum(); require(central_total > 0, f"empty score shape: {series}")
            central_array = central_array / central_total
            stat_array = stat_array / central_total
            replica_array = replica_array / replica_array.sum(axis=1, keepdims=True)
        for bin_index, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            summary = bootstrap_quantile_summary(replica_array[:, bin_index])
            central = float(central_array[bin_index])
            row = {
                "figure": "fig02", "panel": panel, "series": series, "bin_low": low, "bin_high": high,
                "central_value": central, "weighted_statistical_uncertainty": float(stat_array[bin_index]),
                "bootstrap_median": summary["bootstrap_median"], "bootstrap_p16": summary["bootstrap_p16"],
                "bootstrap_p84": summary["bootstrap_p84"], "valid_replicas": summary["valid_replicas"],
                "multijet_systematic_lower": "", "multijet_systematic_upper": "", "multijet_systematic_support": "not_applicable",
                "uncertainty_central": central, "uncertainty_lower_68": summary["bootstrap_p16"],
                "uncertainty_upper_68": summary["bootstrap_p84"], "uncertainty_valid_replicas": summary["valid_replicas"],
                "uncertainty_support_flag": True, "uncertainty_kind": "paired_source_member_bootstrap_p16_p84",
            }
            rows.append(row); by_key[(panel, series, bin_index)] = (replica_array[:, bin_index], central, stat_array[bin_index])
    frame = pd.DataFrame(rows)
    transferred = frame[(frame.panel == "physical_yield") & (frame.series == "transferred")]
    closure = frame[(frame.panel == "physical_yield") & (frame.series == "direct_closure")]
    for transferred_index, closure_index in zip(transferred.index, closure.index):
        qcd = float(frame.loc[transferred_index, "central_value"]); truth = float(frame.loc[closure_index, "central_value"])
        if qcd > 0 and truth > 0:
            nonclosure = abs(qcd-truth)/truth
            relative = math.sqrt(factor_stat**2 + envelope**2 + nonclosure**2)
            frame.loc[transferred_index, "multijet_systematic_lower"] = max(0.0, qcd*(1.0-relative))
            frame.loc[transferred_index, "multijet_systematic_upper"] = qcd*(1.0+relative)
            frame.loc[transferred_index, "multijet_systematic_support"] = "supported_by_direct_closure_bin"
        else:
            frame.loc[transferred_index, "multijet_systematic_support"] = "unsupported_zero_direct_or_transferred_bin"
    return frame


def replace_score_distribution_figure(rows: pd.DataFrame, provenance: list[dict[str, Any]], figures: Path, paper_figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    labels = {"signal": "Signal", "background": "Background", "ordinary": "Non-QCD background",
              "transferred": "Transferred QCD", "direct_closure": r"Direct $\geq4b$ QCD closure"}
    styles = {"signal": (COLORS["signal"], "-"), "background": (COLORS["ordinary"], "-"),
              "ordinary": (COLORS["ordinary"], "-"), "transferred": (COLORS["transferred"], "-"),
              "direct_closure": (COLORS["direct"], "--")}
    for ax, panel, series_list in ((axes[0], "unit_shape", ["signal", "background"]),
                                   (axes[1], "physical_yield", ["signal", "ordinary", "transferred", "direct_closure"])):
        for series in series_list:
            sub = rows[(rows.panel == panel) & (rows.series == series)]
            x = np.r_[sub.bin_low.to_numpy(float), sub.bin_high.to_numpy(float)[-1]]
            central = np.r_[sub.central_value.to_numpy(float), sub.central_value.to_numpy(float)[-1]]
            lo = np.r_[sub.bootstrap_p16.to_numpy(float), sub.bootstrap_p16.to_numpy(float)[-1]]
            hi = np.r_[sub.bootstrap_p84.to_numpy(float), sub.bootstrap_p84.to_numpy(float)[-1]]
            color, ls = styles[series]
            ax.step(x, central, where="post", color=color, ls=ls, lw=1.8, label=labels[series])
            ax.fill_between(x, lo, hi, step="post", color=color, alpha=0.14, linewidth=0)
            centers = 0.5*(sub.bin_low.to_numpy(float)+sub.bin_high.to_numpy(float))
            ax.errorbar(centers, sub.central_value.to_numpy(float), yerr=sub.weighted_statistical_uncertainty.to_numpy(float),
                        fmt="none", ecolor=color, alpha=0.35, lw=0.7)
        ax.grid(alpha=0.18); c7v.paper_label(ax, subtitle="Train-only OOF")
    transferred = rows[(rows.panel == "physical_yield") & (rows.series == "transferred")]
    supported = transferred.multijet_systematic_support == "supported_by_direct_closure_bin"
    x = np.r_[transferred.bin_low.to_numpy(float), transferred.bin_high.to_numpy(float)[-1]]
    sys_lo = pd.to_numeric(transferred.multijet_systematic_lower, errors="coerce").to_numpy(float)
    sys_hi = pd.to_numeric(transferred.multijet_systematic_upper, errors="coerce").to_numpy(float)
    sys_lo = np.r_[sys_lo, sys_lo[-1]]; sys_hi = np.r_[sys_hi, sys_hi[-1]]; support_step = np.r_[supported.to_numpy(bool), supported.to_numpy(bool)[-1]]
    axes[1].fill_between(x, sys_lo, sys_hi, where=support_step, step="post", facecolor="none", edgecolor=COLORS["transferred"],
                         hatch="///", linewidth=0.0, label="Frozen multijet systematic")
    axes[0].set(xlabel="BDT OOF score", ylabel="Unit-normalized weighted events")
    axes[1].set(xlabel="BDT OOF score", ylabel="Train-partition physical yield", yscale="log")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=7, ncol=2)
    replace_plot(provenance, fig, figures, paper_figures, "fig02_oof_score_distributions",
                 r"BDT OOF score distributions. Thin error bars are weighted statistical uncertainties, translucent bands are paired source-member bootstrap 68\% intervals, and hatching shows the frozen multijet systematic where direct $\geq4b$ QCD closure supports a bin. Unsupported systematic bins are not filled.", rows.to_dict("records"))


def replace_efficiency_figure(scans: pd.DataFrame, provenance: list[dict[str, Any]], figures: Path, paper_figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for model in MODEL_ORDER:
        sub = scans[scans.model == model]
        x = sub.target_signal_efficiency.to_numpy(float)
        median = sub.background_rejection_bootstrap_median.to_numpy(float)
        lo = sub.background_rejection_bootstrap_p16.to_numpy(float)
        hi = sub.background_rejection_bootstrap_p84.to_numpy(float)
        ax.plot(x, median, color=COLORS[model], lw=2, label=DISPLAY[model])
        ax.fill_between(x, lo, hi,
                        color=COLORS[model], alpha=0.16, linewidth=0)
    ax.set(xlabel="Target signal efficiency", ylabel="Background rejection", xlim=(0.1, 1.0), ylim=(0, 1))
    ax.grid(alpha=0.18); ax.legend(); c7v.paper_label(ax)
    replace_plot(provenance, fig, figures, paper_figures, "fig03_efficiency_background_rejection",
                 "Signal efficiency versus development-weighted background rejection. Central curves are bootstrap medians and bands are paired source-member 16th--84th percentile intervals at frozen thresholds.", scans.to_dict("records"))


def replace_model_comparisons(summary: pd.DataFrame, nominal: pd.DataFrame, provenance: list[dict[str, Any]],
                              figures: Path, paper_figures: Path) -> None:
    metrics = [
        ("weighted_auc", "Weighted AUC"), ("nominal_asimov_ZA", r"Nominal $Z_A$"),
        ("systematic_aware_asimov_ZA", r"Systematics-aware $Z_A$"), ("signal_over_background", r"$S/B$"),
        ("background_effective_events", r"Background $N_{\rm eff}$"), ("transferred_qcd_fraction", "Transferred-QCD fraction"),
    ]
    lookup = summary.set_index(["model", "metric"])
    nominal_lookup = nominal.set_index("model")
    x = np.arange(len(MODEL_ORDER))
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.4))
    source_rows = []
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        med = np.asarray([lookup.loc[(m, metric), "bootstrap_median"] for m in MODEL_ORDER], float)
        lo = np.asarray([lookup.loc[(m, metric), "bootstrap_p16"] for m in MODEL_ORDER], float)
        hi = np.asarray([lookup.loc[(m, metric), "bootstrap_p84"] for m in MODEL_ORDER], float)
        point = np.asarray([nominal_lookup.loc[m, metric] for m in MODEL_ORDER], float)
        ax.errorbar(x, med, yerr=np.vstack([med - lo, hi - med]), fmt="o", color="#0072B2", capsize=4,
                    label="Bootstrap median and 68% interval")
        ax.scatter(x, point, marker="D", facecolors="none", edgecolors="#D55E00", label="Nominal full OOF")
        ax.set_xticks(x); ax.set_xticklabels([DISPLAY[m] for m in MODEL_ORDER], rotation=16, ha="right")
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.18)
        if metric in {"systematic_aware_asimov_ZA", "signal_over_background"}: ax.set_yscale("log")
        for model, median, lower, upper, nominal_value in zip(MODEL_ORDER, med, lo, hi, point):
            record = lookup.loc[(model, metric)].to_dict()
            source_rows.append({"model": model, "metric": metric, "nominal_full_oof": nominal_value,
                                **record, "uncertainty_central": median, "uncertainty_lower_68": lower,
                                "uncertainty_upper_68": upper, "uncertainty_valid_replicas": int(record["valid_replicas"]),
                                "uncertainty_support_flag": int(record["valid_replicas"]) > 0,
                                "uncertainty_kind": "paired_source_member_bootstrap_p16_p84"})
    axes[0, 0].legend(fontsize=7)
    fig.subplots_adjust(wspace=0.42, hspace=0.55, top=0.91, bottom=0.16)
    fig.text(0.01, 0.98, "Delphes simulation", ha="left", va="top", fontsize=12, fontweight="bold")
    fig.text(0.99, 0.98, "Train-only source-group OOF", ha="right", va="top", fontsize=9)
    replace_plot(provenance, fig, figures, paper_figures, "fig07_nominal_baseline_comparison",
                 "Frozen baseline comparison. Circles and asymmetric bars show source-member bootstrap medians and 16th--84th percentile intervals; open diamonds show nominal full-OOF values.", source_rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    metric = "nominal_asimov_ZA"
    med = np.asarray([lookup.loc[(m, metric), "bootstrap_median"] for m in MODEL_ORDER])
    lo = np.asarray([lookup.loc[(m, metric), "bootstrap_p16"] for m in MODEL_ORDER])
    hi = np.asarray([lookup.loc[(m, metric), "bootstrap_p84"] for m in MODEL_ORDER])
    ax.errorbar(x, med, yerr=np.vstack([med-lo, hi-med]), fmt="o", capsize=5, color="#0072B2")
    ax.scatter(x, [nominal_lookup.loc[m, metric] for m in MODEL_ORDER], marker="D", facecolors="none", edgecolors="#D55E00")
    ax.set_xticks(x); ax.set_xticklabels([DISPLAY[m] for m in MODEL_ORDER]); ax.set_ylabel(r"Nominal $Z_A$")
    ax.grid(axis="y", alpha=0.18); c7v.paper_label(ax, subtitle="1000 paired source-member replicas")
    rows = [row for row in source_rows if row["metric"] == metric]
    replace_plot(provenance, fig, figures, paper_figures, "fig06_bootstrap_za",
                 "Nominal train-only sensitivity with asymmetric paired source-member bootstrap 68% intervals. Models are evaluated on the same 1000 draws and are not retrained in each replica.", rows)


def replace_earlier_current_figure(inputs: dict[str, Any], summary: pd.DataFrame,
                                   provenance: list[dict[str, Any]], figures: Path,
                                   paper_figures: Path) -> None:
    earlier = inputs["earlier"]
    chosen = earlier[earlier.metric.isin([
        "weighted_auc", "asimov_ZA", "background_effective_events",
    ])].copy()
    metric_map = {
        "weighted_auc": "weighted_auc",
        "asimov_ZA": "nominal_asimov_ZA",
        "background_effective_events": "background_effective_events",
    }
    metric_display = {
        "weighted_auc": "Weighted AUC",
        "asimov_ZA": r"Nominal $Z_A$",
        "background_effective_events": r"Background $N_{\rm eff}$",
    }
    lookup = summary.set_index(["model", "metric"])
    rows: list[dict[str, Any]] = []
    for row in chosen.itertuples():
        current = lookup.loc[(row.baseline, metric_map[row.metric])]
        earlier_value = float(row.earlier_snapshot)
        require(earlier_value != 0.0, "zero historical aggregate prevents relative diagnostic")
        median = float(current.bootstrap_median / earlier_value - 1.0)
        lower = float(current.bootstrap_p16 / earlier_value - 1.0)
        upper = float(current.bootstrap_p84 / earlier_value - 1.0)
        rows.append({
            "model": row.baseline,
            "metric": row.metric,
            "display_label": f"{DISPLAY[row.baseline]}: {metric_display[row.metric]}",
            "earlier_aggregate_snapshot": earlier_value,
            "current_nominal_full_oof": float(current.nominal_full_oof),
            "nominal_relative_change": float(current.nominal_full_oof / earlier_value - 1.0),
            "uncertainty_central": median,
            "uncertainty_lower_68": lower,
            "uncertainty_upper_68": upper,
            "uncertainty_valid_replicas": int(current.valid_replicas),
            "uncertainty_support_flag": True,
            "uncertainty_kind": "current_source_member_bootstrap_relative_to_fixed_historical_aggregate",
            "paired_difference_available": False,
            "paired_difference_unavailable_reason": "earlier artifact contains aggregate values only, not source-member predictions",
        })
    frame = pd.DataFrame(rows)
    y = np.arange(len(frame))
    central = frame.uncertainty_central.to_numpy(float)
    lower = frame.uncertainty_lower_68.to_numpy(float)
    upper = frame.uncertainty_upper_68.to_numpy(float)
    nominal = frame.nominal_relative_change.to_numpy(float)
    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    ax.errorbar(central, y, xerr=np.vstack([central - lower, upper - central]), fmt="o", capsize=4,
                color="#0072B2", label="Current bootstrap median and 68% interval")
    ax.scatter(nominal, y, marker="D", facecolors="none", edgecolors="#D55E00",
               label="Current nominal full OOF")
    ax.axvline(0.0, color="0.3", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(frame.display_label)
    ax.set_xlabel("Relative change from fixed earlier aggregate snapshot")
    ax.grid(axis="x", alpha=0.18); ax.legend(loc="lower right", fontsize=8)
    fig.subplots_adjust(left=0.34, right=0.97, top=0.90, bottom=0.12)
    c7v.paper_label(ax, subtitle="Non-pairable provenance diagnostic")
    replace_plot(
        provenance, fig, figures, paper_figures, "fig08_earlier_current_metrics",
        "Historical-provenance diagnostic. The current result carries a source-member bootstrap 68% interval, "
        "while the earlier fixed aggregate snapshot has no source-member predictions. This is not a paired "
        "statistical comparison and supports no improvement claim.",
        rows,
    )


def composition_rows(inputs: dict[str, Any], draws: np.ndarray, nominal: pd.DataFrame) -> pd.DataFrame:
    projection = inputs["projection"]
    scores = inputs["projection_scores"]
    metrics = inputs["metrics"].set_index("baseline")
    members = inputs["members"].reset_index(drop=True)
    positions = member_positions(projection, members, "composition projection")
    roles = projection["analysis_population_role"].astype(str).to_numpy()
    background = projection["registry_training_target"].to_numpy(dtype=np.int8) == 0
    component = np.where(roles == "primary_transferred_multijet_template", "Transferred QCD",
                         projection["process_or_mode"].astype(str).str.replace("_", " ").str.title())
    physical = projection["primary_projection_physical_weight_inclusive"].to_numpy(float)
    all_rows = []
    for model in MODEL_ORDER:
        if model == "cut": selected = projection["r_hh_125_120"].to_numpy(float) < 34.0
        else: selected = scores[f"{model}_score"].to_numpy(float) >= float(metrics.loc[model, "operating_threshold"])
        mask = selected & background
        central_by_component = {name: float(physical[mask & (component == name)].sum()) for name in np.unique(component[mask])}
        ordered = sorted(central_by_component, key=central_by_component.get, reverse=True)
        retained = set(ordered[:7]) | {"Transferred QCD"}
        display_component = np.where(np.isin(component, list(retained)), component, "Other")
        total_group = group_sum(physical, mask, positions, len(members))
        total_boot = draws @ total_group
        total_central = float(physical[mask].sum())
        for name in sorted(set(display_component[mask])):
            group = group_sum(physical, mask & (display_component == name), positions, len(members))
            boot = draws @ group
            central = float(group.sum())
            summary = bootstrap_quantile_summary(boot)
            fraction = boot / total_boot
            fraction_summary = bootstrap_quantile_summary(fraction)
            all_rows.append({
                "model": model, "component": str(name), "central_yield": central,
                "bootstrap_median_yield": summary["bootstrap_median"], "bootstrap_p16_yield": summary["bootstrap_p16"],
                "bootstrap_p84_yield": summary["bootstrap_p84"], "central_fraction": central / total_central,
                "bootstrap_median_fraction": fraction_summary["bootstrap_median"],
                "bootstrap_p16_fraction": fraction_summary["bootstrap_p16"], "bootstrap_p84_fraction": fraction_summary["bootstrap_p84"],
                "total_central_yield": total_central, "total_bootstrap_p16": float(np.quantile(total_boot, .16)),
                "total_bootstrap_p84": float(np.quantile(total_boot, .84)), "valid_replicas": len(draws), "support_flag": True,
                "uncertainty_central": central, "uncertainty_lower_68": summary["bootstrap_p16"],
                "uncertainty_upper_68": summary["bootstrap_p84"], "uncertainty_valid_replicas": len(draws),
                "uncertainty_support_flag": True, "uncertainty_kind": "paired_source_member_bootstrap_p16_p84",
            })
    frame = pd.DataFrame(all_rows)
    bdt = frame[(frame.model == "bdt") & (frame.component == "Transferred QCD")].iloc[0]
    expected = float(nominal.set_index("model").loc["bdt", "transferred_qcd_fraction"])
    require(abs(float(bdt.central_fraction) - expected) < 1.0e-12, "BDT transferred-QCD fraction reconstruction drift")
    require(abs(expected - 0.8542) < 0.001, f"unexpected BDT transferred-QCD fraction: {expected}")
    return frame


def replace_composition_figure(rows: pd.DataFrame, provenance: list[dict[str, Any]], figures: Path, paper_figures: Path) -> None:
    components = list(rows.groupby("component").central_yield.sum().sort_values(ascending=False).index)
    palette = ["#CC79A7", "#0072B2", "#E69F00", "#009E73", "#56B4E9", "#D55E00", "#F0E442", "#999999"]
    colors = {component: palette[index % len(palette)] for index, component in enumerate(components)}
    x = np.arange(len(MODEL_ORDER)); fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    bottom_abs = np.zeros(len(MODEL_ORDER)); bottom_frac = np.zeros(len(MODEL_ORDER))
    for component in components:
        sub = rows[rows.component == component].set_index("model").reindex(MODEL_ORDER)
        absolute = sub.central_yield.fillna(0).to_numpy(float); fraction = sub.central_fraction.fillna(0).to_numpy(float)
        axes[0].bar(x, absolute / 1.0e6, bottom=bottom_abs / 1.0e6, color=colors[component], label=component)
        axes[1].bar(x, fraction, bottom=bottom_frac, color=colors[component], label=component)
        bottom_abs += absolute; bottom_frac += fraction
    totals = rows.groupby("model", sort=False).first().reindex(MODEL_ORDER)
    total = totals.total_central_yield.to_numpy(float)
    total_lo = totals.total_bootstrap_p16.to_numpy(float)
    total_hi = totals.total_bootstrap_p84.to_numpy(float)
    axes[0].errorbar(x, total / 1.0e6, yerr=np.vstack([total-total_lo, total_hi-total]) / 1.0e6,
                     fmt="none", ecolor="black", capsize=4, lw=1.3, label="Total 68% bootstrap interval")
    axes[0].set_ylim(bottom=0); axes[0].set_ylabel(r"Selected physical background yield [$10^6$]")
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("Fraction of selected background")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([DISPLAY[m] for m in MODEL_ORDER], rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(ncol=2, fontsize=7)
    fig.subplots_adjust(wspace=0.25, top=0.90, bottom=0.18)
    fig.text(0.01, 0.98, "Delphes simulation", ha="left", va="top", fontsize=12, fontweight="bold")
    fig.text(0.99, 0.98, "Train-only source-group OOF", ha="right", va="top", fontsize=9)
    replace_plot(provenance, fig, figures, paper_figures, "fig10_selected_process_composition",
                 r"Selected background composition at frozen operating points. The absolute stacked-yield axis starts at zero, total yields carry asymmetric source-member bootstrap 68\% intervals, and the right panel shows fractional composition. Transferred QCD is explicit; direct $\geq4b$ QCD is secondary closure and is not stacked into the primary prediction.", rows.to_dict("records"))


def closure_rows(inputs: dict[str, Any], draws: np.ndarray) -> pd.DataFrame:
    projection = inputs["projection"]; direct = inputs["direct"]; members = inputs["members"].reset_index(drop=True)
    projection_pos = member_positions(projection, members, "closure projection")
    direct_pos = member_positions(direct, members, "closure direct")
    transferred = projection["analysis_population_role"].astype(str).to_numpy() == "primary_transferred_multijet_template"
    direct_qcd = direct["population_kind"].astype(str).to_numpy() == "hard_qcd"
    qcd_weight = projection["primary_projection_physical_weight_inclusive"].to_numpy(float)
    direct_weight = direct["direct_projection_physical_weight"].to_numpy(float)
    pr_rhh = projection["r_hh_125_120"].to_numpy(float); dr_rhh = direct["r_hh_125_120"].to_numpy(float)
    regions = [
        ("Inclusive baseline", np.ones(len(projection), bool), np.ones(len(direct), bool)),
        (r"$R_{HH}<30$", pr_rhh < 30, dr_rhh < 30),
        (r"$30\leq R_{HH}<55$", (pr_rhh >= 30) & (pr_rhh < 55), (dr_rhh >= 30) & (dr_rhh < 55)),
        (r"$R_{HH}\geq55$", pr_rhh >= 55, dr_rhh >= 55),
        (r"Frozen $R_{HH}<34$", pr_rhh < 34, dr_rhh < 34),
        (r"$R_{HH}<31.5$", pr_rhh < 31.5, dr_rhh < 31.5),
        (r"$R_{HH}<35.5$", pr_rhh < 35.5, dr_rhh < 35.5),
    ]
    rows = []
    for region, projection_region, direct_region in regions:
        qcd_group = group_sum(qcd_weight, transferred & projection_region, projection_pos, len(members))
        direct_group = group_sum(direct_weight, direct_qcd & direct_region, direct_pos, len(members))
        qcd_boot = draws @ qcd_group; direct_boot = draws @ direct_group
        ratio = np.divide(qcd_boot, direct_boot, out=np.full_like(qcd_boot, np.nan), where=direct_boot > 0)
        qcd_summary = bootstrap_quantile_summary(qcd_boot); direct_summary = bootstrap_quantile_summary(direct_boot)
        ratio_summary = bootstrap_quantile_summary(ratio)
        qcd = float(qcd_group.sum()); truth = float(direct_group.sum())
        rows.append({
            "region": region, "transferred_qcd_yield": qcd, "direct_qcd_closure_yield": truth,
            "closure_ratio": qcd/truth if truth > 0 else float("nan"),
            "transferred_p16": qcd_summary["bootstrap_p16"], "transferred_p84": qcd_summary["bootstrap_p84"],
            "direct_p16": direct_summary["bootstrap_p16"], "direct_p84": direct_summary["bootstrap_p84"],
            "ratio_p16": ratio_summary["bootstrap_p16"], "ratio_p84": ratio_summary["bootstrap_p84"],
            "ratio_valid_replicas": ratio_summary["valid_replicas"], "ratio_invalid_replicas": ratio_summary["invalid_replicas"],
            "support_flag": ratio_summary["valid_replicas"] > 0,
            "uncertainty_central": qcd/truth if truth > 0 else float("nan"),
            "uncertainty_lower_68": ratio_summary["bootstrap_p16"], "uncertainty_upper_68": ratio_summary["bootstrap_p84"],
            "uncertainty_valid_replicas": ratio_summary["valid_replicas"], "uncertainty_support_flag": ratio_summary["valid_replicas"] > 0,
            "uncertainty_kind": "paired_source_member_bootstrap_p16_p84",
        })
    return pd.DataFrame(rows)


def replace_closure_figure(rows: pd.DataFrame, provenance: list[dict[str, Any]], figures: Path, paper_figures: Path) -> None:
    x = np.arange(len(rows)); width = 0.34
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.8), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    qcd = rows.transferred_qcd_yield.to_numpy(float); direct = rows.direct_qcd_closure_yield.to_numpy(float)
    axes[0].bar(x-width/2, qcd, width, color=COLORS["transferred"], label=r"$3b\rightarrow{\geq4b}$ prediction")
    axes[0].bar(x+width/2, direct, width, color="#777777", label=r"Direct $\geq4b$ QCD closure")
    axes[0].errorbar(x-width/2, qcd, yerr=np.vstack([qcd-rows.transferred_p16.to_numpy(float), rows.transferred_p84.to_numpy(float)-qcd]),
                     fmt="none", ecolor="black", capsize=3)
    axes[0].errorbar(x+width/2, direct, yerr=np.vstack([direct-rows.direct_p16.to_numpy(float), rows.direct_p84.to_numpy(float)-direct]),
                     fmt="none", ecolor="black", capsize=3)
    axes[0].set_yscale("log"); axes[0].set_ylabel("QCD physical yield"); axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.18)
    c7v.paper_label(axes[0], subtitle="Direct QCD is secondary closure")
    ratio = rows.closure_ratio.to_numpy(float); lo = rows.ratio_p16.to_numpy(float); hi = rows.ratio_p84.to_numpy(float)
    axes[1].axhline(1, color="0.3", lw=1); axes[1].errorbar(x, ratio, yerr=np.vstack([ratio-lo, hi-ratio]), fmt="o", capsize=3, color="#0072B2")
    axes[1].set_ylabel("Pred./closure"); axes[1].set_xticks(x); axes[1].set_xticklabels(rows.region, rotation=15, ha="right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.18)
    replace_plot(provenance, fig, figures, paper_figures, "fig12_transfer_closure",
                 r"Frozen $3b\rightarrow{\geq4b}$ prediction and secondary direct $\geq4b$ QCD closure. Asymmetric bars are paired source-member bootstrap 16th--84th percentile intervals; undefined zero-closure replicas are excluded and counted in source data.", rows.to_dict("records"))


def replace_multijet_figure(nominal: pd.DataFrame, replicas: pd.DataFrame, provenance: list[dict[str, Any]], figures: Path, paper_figures: Path) -> None:
    metrics = [
        ("transferred_qcd_fraction", "Transferred-QCD fraction of background [%]", 100.0),
        ("score_domain_qcd_nonclosure", r"Direct $\geq4b$ QCD nonclosure [%]", 100.0),
        ("multijet_relative", "Frozen multijet nuisance / background [%]", 100.0),
    ]
    local = replicas.copy(); local["multijet_relative"] = local.multijet_systematic_absolute / local.selected_background_yield
    central = nominal.copy(); central["multijet_relative"] = central.multijet_systematic_absolute / central.selected_background_yield
    x = np.arange(len(MODEL_ORDER)); fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0)); source_rows = []
    for ax, (metric, ylabel, scale) in zip(axes, metrics):
        med=[]; lo=[]; hi=[]; point=[]
        for model in MODEL_ORDER:
            values = local.loc[local.model == model, metric].to_numpy(float)
            summary = bootstrap_quantile_summary(values)
            nominal_value = float(central.set_index("model").loc[model, metric])
            med.append(scale*summary["bootstrap_median"]); lo.append(scale*summary["bootstrap_p16"]); hi.append(scale*summary["bootstrap_p84"]); point.append(scale*nominal_value)
            source_rows.append({"model": model, "metric": metric, "nominal_full_oof": nominal_value,
                                **summary, "uncertainty_central": summary["bootstrap_median"],
                                "uncertainty_lower_68": summary["bootstrap_p16"], "uncertainty_upper_68": summary["bootstrap_p84"],
                                "uncertainty_valid_replicas": summary["valid_replicas"], "uncertainty_support_flag": True,
                                "uncertainty_kind": "paired_source_member_bootstrap_p16_p84"})
        med=np.asarray(med);lo=np.asarray(lo);hi=np.asarray(hi)
        ax.errorbar(x, med, yerr=np.vstack([med-lo,hi-med]), fmt="o", capsize=4, color="#0072B2")
        ax.scatter(x, point, marker="D", facecolors="none", edgecolors="#D55E00")
        ax.set_xticks(x); ax.set_xticklabels([DISPLAY[m] for m in MODEL_ORDER], rotation=16, ha="right")
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.18)
    fig.subplots_adjust(wspace=0.48, top=0.88, bottom=0.22)
    fig.text(0.01, 0.98, "Delphes simulation", ha="left", va="top", fontsize=12, fontweight="bold")
    fig.text(0.99, 0.98, "Train-only frozen transfer", ha="right", va="top", fontsize=9)
    replace_plot(provenance, fig, figures, paper_figures, "fig17_multijet_systematic_nonclosure",
                 r"Transferred-QCD fraction, direct-$\geq4b$ QCD nonclosure, and frozen multijet nuisance. Circles and asymmetric bars are source-member bootstrap medians and 68\% intervals; open diamonds are nominal full-OOF values.", source_rows)


def format_asymmetric(row: pd.Series, *, scientific: bool = False, digits: int = 5) -> str:
    median = float(row.bootstrap_median)
    up = float(row.bootstrap_p84 - row.bootstrap_median)
    down = float(row.bootstrap_median - row.bootstrap_p16)
    if scientific and median != 0.0:
        exponent = int(math.floor(math.log10(abs(median))))
        scale = 10.0 ** exponent
        return rf"$({median/scale:.2f}^{{+{up/scale:.2f}}}_{{-{down/scale:.2f}}})\times10^{{{exponent}}}$"
    return rf"${median:.{digits}f}^{{+{up:.{digits}f}}}_{{-{down:.{digits}f}}}$"


def format_paired_difference(median: float, p16: float, p84: float, *, scientific: bool, digits: int) -> str:
    up = float(p84 - median)
    down = float(median - p16)
    if scientific:
        magnitude = max(abs(float(median)), abs(up), abs(down))
        if magnitude == 0.0:
            return "$0$"
        exponent = int(math.floor(math.log10(magnitude)))
        scale = 10.0 ** exponent
        return rf"$({median/scale:.2f}^{{+{up/scale:.2f}}}_{{-{down/scale:.2f}}})\times10^{{{exponent}}}$"
    return rf"${median:.{digits}f}^{{+{up:.{digits}f}}}_{{-{down:.{digits}f}}}$"


def add_uncertainty_tables(paper: Path, summary: pd.DataFrame, paired: pd.DataFrame,
                           earlier: pd.DataFrame | None = None) -> None:
    lookup = summary.set_index(["model", "metric"])
    classification_rows = []
    yield_rows = []
    for model in MODEL_ORDER:
        classification_rows.append([
            DISPLAY[model],
            format_asymmetric(lookup.loc[(model, "weighted_auc")], digits=4),
            format_asymmetric(lookup.loc[(model, "nominal_asimov_ZA")]),
            format_asymmetric(lookup.loc[(model, "systematic_aware_asimov_ZA")], scientific=True),
        ])
        yield_rows.append([
            DISPLAY[model],
            format_asymmetric(lookup.loc[(model, "signal_over_background")], scientific=True),
            format_asymmetric(lookup.loc[(model, "background_effective_events")], digits=1),
            format_asymmetric(lookup.loc[(model, "transferred_qcd_fraction")], digits=3),
        ])
    unified_rows = []
    for model in MODEL_ORDER:
        za_row = lookup.loc[(model, "nominal_asimov_ZA")]
        unified_rows.append([
            DISPLAY[model],
            f"${float(za_row.nominal_full_oof):.5f}$",
            format_asymmetric(za_row),
            format_asymmetric(lookup.loc[(model, "weighted_auc")], digits=4),
            format_asymmetric(lookup.loc[(model, "systematic_aware_asimov_ZA")], scientific=True),
        ])
    c7v.write_latex_table(
        paper / "tables" / "tab05_baseline_metrics.tex",
        "Unified train-only frozen-model comparison. The nominal full-OOF point is accompanied by the common paired source-member bootstrap 68\\% interval.",
        "tab:baseline-metrics",
        ["Model", r"Nominal full-OOF $Z_A$", r"Bootstrap $Z_A$", "Weighted AUC", r"Systematics-aware $Z_A$"],
        unified_rows,
        align="lrrrr",
    )
    if earlier is not None:
        earlier_metric_map = {
            "weighted_auc": "weighted_auc",
            "asimov_ZA": "nominal_asimov_ZA",
            "bootstrap_median_ZA": "nominal_asimov_ZA",
            "signal_over_background": "signal_over_background",
            "background_effective_events": "background_effective_events",
        }
        earlier_metric_display = {
            "weighted_auc": "Weighted AUC",
            "asimov_ZA": r"Nominal $Z_A$",
            "bootstrap_median_ZA": r"Historical bootstrap-median $Z_A$ row",
            "signal_over_background": r"$S/B$",
            "background_effective_events": r"Background $N_{\rm eff}$",
        }
        historical_rows = []
        for row in earlier[earlier.metric != "bootstrap_median_ZA"].itertuples():
            metric = earlier_metric_map[row.metric]
            current = lookup.loc[(row.baseline, metric)]
            scientific = metric == "signal_over_background"
            earlier_text = (c7v.scientific_latex(float(row.earlier_snapshot), digits=2)
                            if scientific else f"${float(row.earlier_snapshot):.5g}$")
            nominal_text = (c7v.scientific_latex(float(current.nominal_full_oof), digits=2)
                            if scientific else f"${float(current.nominal_full_oof):.5g}$")
            interval_text = format_asymmetric(current, scientific=scientific,
                                              digits=1 if metric == "background_effective_events" else 5)
            historical_rows.append([
                DISPLAY[row.baseline], earlier_metric_display[row.metric], earlier_text,
                f"{nominal_text}; {interval_text}",
            ])
        c7v.write_latex_table(
            paper / "tables" / "tab06_earlier_current.tex",
            "Historical aggregate provenance diagnostic. Current values include source-member bootstrap 68\\% "
            "intervals; the earlier artifact has no source-member predictions, so no paired improvement is claimed.",
            "tab:earlier-current",
            ["Model", "Metric", "Earlier aggregate", "Current nominal; bootstrap 68\\%"],
            historical_rows,
            align="llll",
        )
    c7v.write_latex_table(
        paper / "tables" / "tab10_uncertainty_summary.tex",
        "Train-only frozen-model results with paired source-member bootstrap 68\\% intervals.",
        "tab:baseline-uncertainty-summary",
        ["Model", "Weighted AUC", r"Nominal $Z_A$", r"Systematics-aware $Z_A$"],
        classification_rows,
        align="lrrr",
    )
    c7v.write_latex_table(
        paper / "tables" / "tab10b_selected_yield_uncertainty.tex",
        "Selected-yield diagnostics with paired source-member bootstrap 68\\% intervals.",
        "tab:selected-yield-uncertainty-summary",
        ["Model", r"$S/B$", r"Background $N_{\rm eff}$", "Transferred-QCD fraction"],
        yield_rows,
        align="lrrr",
    )
    comparison_rows = []
    probability_rows = []
    short_display = {"cut": "Frozen cut", "bdt": "BDT", "dense_dnn": "Dense DNN", "lbn_dnn": "LBN-DNN"}
    metric_display = {
        "weighted_auc": "Weighted AUC",
        "nominal_asimov_ZA": r"Nominal $Z_A$",
        "systematic_aware_asimov_ZA": r"Systematics-aware $Z_A$",
        "signal_over_background": r"$S/B$",
        "background_effective_events": r"Background $N_{\rm eff}$",
    }
    metric_format = {
        "weighted_auc": (False, 5),
        "nominal_asimov_ZA": (False, 5),
        "systematic_aware_asimov_ZA": (True, 2),
        "signal_over_background": (True, 2),
        "background_effective_events": (False, 2),
    }
    for row in paired[paired.metric.isin(["weighted_auc", "nominal_asimov_ZA", "systematic_aware_asimov_ZA", "signal_over_background", "background_effective_events"])].itertuples():
        median = row.bootstrap_median_difference
        scientific, digits = metric_format[row.metric]
        comparison_rows.append([
            f"{short_display[row.model]} minus {short_display[row.reference_model]}", metric_display[row.metric],
            format_paired_difference(median, row.bootstrap_p16_difference, row.bootstrap_p84_difference,
                                     scientific=scientific, digits=digits),
            "Yes" if row.interval_includes_zero_68 else "No",
        ])
        probability_rows.append([
            f"{short_display[row.model]} minus {short_display[row.reference_model]}",
            metric_display[row.metric], f"{row.fraction_valid_difference_greater_than_zero:.3f}",
        ])
    c7v.write_latex_table(
        paper / "tables" / "tab11_paired_improvements.tex",
        "Paired source-member bootstrap model differences. A larger point estimate is not called an improvement when the 68\\% interval includes zero.",
        "tab:paired-model-differences",
        ["Comparison", "Metric", "Median difference (68\\%)", "Includes zero"],
        comparison_rows,
        align="llll",
    )
    c7v.write_latex_table(
        paper / "tables" / "tab11b_paired_positive_fraction.tex",
        "Fraction of aligned source-member bootstrap replicas with a positive model-minus-reference difference.",
        "tab:paired-positive-fraction",
        ["Comparison", "Metric", r"$P(\Delta>0)$"], probability_rows, align="llr",
    )
    (paper / "uncertainty_table.tex").write_text(
        (paper / "tables" / "tab10_uncertainty_summary.tex").read_text()
        + (paper / "tables" / "tab10b_selected_yield_uncertainty.tex").read_text()
    )
    (paper / "paired_improvement_table.tex").write_text(
        (paper / "tables" / "tab11_paired_improvements.tex").read_text()
        + (paper / "tables" / "tab11b_paired_positive_fraction.tex").read_text()
    )


def operating_point_stability(inputs: dict[str, Any]) -> pd.DataFrame:
    training = inputs["training"]
    scores = inputs["training_scores"]
    labels = training["registry_training_target"].to_numpy(dtype=np.int8)
    weights = training["development_hierarchical_weight"].to_numpy(float)
    cut = training["r_hh_125_120"].to_numpy(float) < 34.0
    target = float(weights[cut & (labels == 1)].sum() / weights[labels == 1].sum())
    rows = []
    for model in MODEL_ORDER:
        for fold in range(5):
            held = training["registry_oof_fold"].to_numpy(int) == fold
            score = scores[f"{model}_score"].to_numpy(float)
            threshold = threshold_at_efficiency(score[held], labels[held] == 1, weights[held], target)
            rows.append({"model": model, "fold": fold, "target_signal_efficiency": target,
                         "fold_diagnostic_threshold": threshold,
                         "status": "selection-stability diagnostic; frozen c7t operating point remains primary",
                         "validation_used": False, "test_used": False})
    return pd.DataFrame(rows)


def issue_registry() -> list[dict[str, Any]]:
    return [
        {"issue_id": "C7V-ERR-001", "asset": "fig05_systematic_za_scan", "diagnosis": "Binary64 cancellation in the Cowan formula produced 40 spurious exact zeros; low-efficiency unsupported ranges additionally fail transferred-QCD support.", "correction": "extended/high-precision evaluation, explicit support flags, noninterpolated segments, source-bootstrap bands, and a lower support panel"},
        {"issue_id": "C7V-ERR-002", "asset": "fig10_selected_process_composition", "diagnosis": "logarithmic/truncated presentation obscured the dominant transferred-QCD component", "correction": "zero-based linear absolute stack, fractional stack, explicit transferred QCD, and asymmetric total-yield bootstrap bars"},
        {"issue_id": "C7V-ERR-003", "asset": "paper-facing labels", "diagnosis": "raw identifiers and prose b-tag labels appeared in plots/tables", "correction": r"LaTeX-quality exactly $3b$, $\geq4b$, $3b\rightarrow{\geq4b}$, $m_{HH}$, and $R_{HH}(125,120)$ labels"},
        {"issue_id": "C7V-ERR-004", "asset": "LaTeX tables", "diagnosis": "snake-case values, e notation, right-aligned text, and resizebox reduced readability", "correction": "human labels, mathematical scientific notation, booktabs, natural sizing, and textual l columns"},
        {"issue_id": "C7V-ERR-005", "asset": "baseline model comparison", "diagnosis": "point estimates and an unmaterialized per-model RNG stream did not satisfy paired uncertainty reporting", "correction": "one frozen 1000-replica source-member registry, 16 metric intervals, all pairwise differences, and uncertainty-aware figures/tables"},
        {"issue_id": "C7V-ERR-006", "asset": "earlier-current historical diagnostic", "diagnosis": "current resampleable values were displayed without intervals beside a non-resampleable aggregate snapshot", "correction": "current source-member bootstrap intervals are shown; the fixed historical side is explicitly non-pairable and excluded from improvement claims"},
    ]


def run(output: Path, paper_output: Path, inspection_report: Path | None, *, smoke: bool) -> None:
    require(output != paper_output, "checkpoint and paper export must differ")
    c7v.require_fresh_directory(paper_output)
    if not smoke:
        require(inspection_report is not None and inspection_report.is_file(), "full run requires a reviewed visual-inspection report")
    staging = prepare_staging(output)
    try:
        evidence = verify_all_checkpoints() + verify_c7v_v2()
        inputs = c7v.load_inputs()
        replicates = 20 if smoke else BOOTSTRAP_REPLICATES
        draws, draw_registry = source_member_bootstrap_draws(inputs["members"], replicates=replicates, seed=BOOTSTRAP_SEED)
        nominal, replicas = evaluate_frozen_baselines(inputs, draws)
        metric_summary = summarize_metric_replicas(nominal, replicas)
        paired_summary, paired_replicas = paired_model_differences(nominal, replicas)
        validity = bootstrap_validity_summary(replicas)
        stability = operating_point_stability(inputs)
        scans = scan_with_bootstrap(inputs, draws, points=13 if smoke else 91)

        if not smoke:
            frozen = inputs["metrics"].set_index("baseline")
            reconstructed = metric_summary[metric_summary.metric == "nominal_asimov_ZA"].set_index("model")
            for model in MODEL_ORDER:
                for source_field, summary_field in (("bootstrap_mean_ZA", "bootstrap_mean"),
                                                     ("bootstrap_median_ZA", "bootstrap_median"),
                                                     ("bootstrap_p16_ZA", "bootstrap_p16"),
                                                     ("bootstrap_p84_ZA", "bootstrap_p84")):
                    require(math.isclose(float(frozen.loc[model, source_field]), float(reconstructed.loc[model, summary_field]),
                                         rel_tol=0.0, abs_tol=2.0e-12), f"c7t bootstrap reproduction drift: {model}/{source_field}")

        figures = staging / "figures"; figures.mkdir()
        paper_output.mkdir(); paper_figures = paper_output / "figures"; paper_figures.mkdir()
        paper_source = paper_output / "source_data"; paper_source.mkdir()
        provenance = c7v.make_figures(inputs, figures, paper_figures, smoke=smoke)
        roc_weighted = roc_band_rows(inputs, draws, weighted=True, points=31 if smoke else 121)
        roc_unweighted = roc_band_rows(inputs, draws, weighted=False, points=31 if smoke else 121)
        replace_roc_figure(roc_weighted, provenance, figures, paper_figures, weighted=True)
        replace_roc_figure(roc_unweighted, provenance, figures, paper_figures, weighted=False)
        score_shapes = score_distribution_rows(inputs, draws)
        replace_score_distribution_figure(score_shapes, provenance, figures, paper_figures)
        replace_efficiency_figure(scans, provenance, figures, paper_figures)
        replace_scan_figures(scans, provenance, figures, paper_figures, min_band_fraction=0.8 * replicates)
        replace_model_comparisons(metric_summary, nominal, provenance, figures, paper_figures)
        replace_earlier_current_figure(inputs, metric_summary, provenance, figures, paper_figures)
        composition = composition_rows(inputs, draws, nominal)
        replace_composition_figure(composition, provenance, figures, paper_figures)
        closure = closure_rows(inputs, draws)
        replace_closure_figure(closure, provenance, figures, paper_figures)
        replace_multijet_figure(nominal, replicas, provenance, figures, paper_figures)
        provenance.sort(key=lambda row: row["stem"])
        for source in sorted(figures.glob("*.tsv")): shutil.copy2(source, paper_source / source.name)

        assets = c7v.inspect_assets(verify_all_checkpoints())
        tables, claims = c7v.make_paper_material(inputs, paper_output, provenance, assets, subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip())
        add_uncertainty_tables(paper_output, metric_summary, paired_summary, inputs["earlier"])
        for table in tables:
            if table["table_id"] == "tab05":
                table["caption"] = "Unified frozen-model comparison with nominal and paired-bootstrap uncertainty values."
                table["placement"] = "main_text"
            if table["table_id"] == "tab06":
                table["caption"] = "Historical aggregate diagnostic with uncertainty on every resampleable current value."
                table["placement"] = "appendix_diagnostic"
        tables.extend([
            {"table_id": "tab10", "path": "tables/tab10_uncertainty_summary.tex", "caption": "Baseline paired-bootstrap uncertainty summary.", "source_checkpoint": "c7q-c7v", "generation": "machine_generated", "placement": "main_text"},
            {"table_id": "tab10b", "path": "tables/tab10b_selected_yield_uncertainty.tex", "caption": "Selected-yield paired-bootstrap uncertainty summary.", "source_checkpoint": "c7q-c7v", "generation": "machine_generated", "placement": "main_text"},
            {"table_id": "tab11", "path": "tables/tab11_paired_improvements.tex", "caption": "All paired baseline-model differences.", "source_checkpoint": "c7q-c7v", "generation": "machine_generated", "placement": "appendix"},
            {"table_id": "tab11b", "path": "tables/tab11b_paired_positive_fraction.tex", "caption": "Positive paired-difference replica fractions.", "source_checkpoint": "c7q-c7v", "generation": "machine_generated", "placement": "appendix"},
        ])
        write_tsv(paper_output / "paper_table_manifest.tsv", tables)
        write_tsv(paper_output / "paper_figure_manifest.tsv", provenance)
        figure_requests = c7v.requested_figure_registry(provenance)
        write_tsv(paper_output / "requested_figure_registry.tsv", figure_requests)
        write_tsv(staging / "requested_figure_registry.tsv", figure_requests)

        compact = {
            "bootstrap_metric_summary.tsv": metric_summary,
            "paired_model_differences.tsv": paired_summary,
            "bootstrap_validity_summary.tsv": validity,
            "operating_point_stability.tsv": stability,
            "systematics_scan_diagnostics.tsv": scans,
            "selected_background_composition_uncertainty.tsv": composition,
            "transfer_closure_bootstrap_summary.tsv": closure,
        }
        for name, frame in compact.items():
            write_tsv(staging / name, frame.to_dict("records"))
            shutil.copy2(staging / name, paper_output / name)
        registry_path = staging / "common_source_bootstrap_registry.tsv"
        write_tsv(registry_path, draw_registry)
        registry_sha256 = sha256(registry_path)
        registry_provenance = [{
            "registry_relative_path": "common_source_bootstrap_registry.tsv",
            "registry_sha256": registry_sha256,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": replicates,
            "bootstrap_unit": "source_member",
            "draw_definition": "exact_c7t_numpy_rng_stratified_by_sample_class_and_process_stratum",
            "source_members": int(draws.shape[1]),
            "validation_payload_files_opened": 0,
            "test_or_evaluation_payload_files_opened": 0,
        }]
        write_tsv(staging / "bootstrap_registry_provenance.tsv", registry_provenance)
        write_tsv(paper_output / "bootstrap_registry_provenance.tsv", registry_provenance)
        write_tsv(staging / "category_boundary_stability.tsv", [{
            "status": "not_applicable_before_categorized_BDT",
            "reason": "c7v errata evaluates only frozen inclusive baselines",
            "validation_used": False,
            "test_used": False,
        }])
        replicas.to_parquet(staging / "bootstrap_metric_replicas.parquet", index=False)
        paired_replicas.to_parquet(staging / "paired_model_difference_replicas.parquet", index=False)
        write_tsv(staging / "issue_registry.tsv", issue_registry())
        write_tsv(staging / "old_new_asset_mapping.tsv", [
            {"old_checkpoint": "c7v_v2", "old_asset": row["stem"], "new_asset": row["stem"],
             "status": "regenerated_or_reviewed", "uncertainty_contract": "source-data fields present; model-performance assets use paired source bootstrap"}
            for row in provenance
        ])
        write_tsv(staging / "paper_figure_manifest.tsv", provenance)
        write_tsv(staging / "paper_table_manifest.tsv", tables)
        write_tsv(staging / "numerical_claim_registry.tsv", claims)
        write_tsv(staging / "source_evidence_manifest.tsv", evidence)
        write_tsv(staging / "results_asset_manifest.tsv", assets)
        if inspection_report:
            shutil.copy2(inspection_report, staging / "visual_inspection_report.tsv")
            shutil.copy2(inspection_report, paper_output / "figure_inspection_report.tsv")
        else:
            pending_inspection = [{"status": "smoke_pending_full_visual_review", "figures": len(provenance)}]
            write_tsv(staging / "visual_inspection_report.tsv", pending_inspection)
            write_tsv(paper_output / "figure_inspection_report.tsv", pending_inspection)
        (paper_output / "sections" / "uncertainty_contract.tex").write_text(
            r"All resampleable frozen-model performance values use one common 1000-replica source-member bootstrap registry. Primary intervals are asymmetric 16th--84th percentile ranges; model differences are paired replica by replica. Invalid sparse-closure replicas remain undefined and are never replaced by zero." + "\n"
        )
        compile_path = paper_output / "compile_fragments.tex"
        compile_text = compile_path.read_text()
        require(compile_text.count("\\end{document}") == 1, "LaTeX compile wrapper terminator drift")
        compile_path.write_text(compile_text.replace(
            "\\end{document}",
            "\\input{sections/uncertainty_contract.tex}\n"
            "\\input{tables/tab10_uncertainty_summary.tex}\n"
            "\\input{tables/tab10b_selected_yield_uncertainty.tex}\n"
            "\\input{tables/tab11_paired_improvements.tex}\n"
            "\\input{tables/tab11b_paired_positive_fraction.tex}\n"
            "\\end{document}",
        ))
        (paper_output / "README.md").write_text((paper_output / "README.md").read_text() +
            "\nThis export supersedes c7v v2 for support diagnostics, labels, and uncertainty presentation. "
            "All primary frozen-model comparisons use the common paired source-member registry; "
            f"its SHA-256 is `{registry_sha256}`.\n")
        shutil.copytree(paper_output, staging / "paper")
        (staging / "README.md").write_text(
            "# c7v support-diagnostic and uncertainty errata\n\n"
            "Supersedes c7v v2 without modifying it. Corrects numerical cancellation, support display, background composition, labels, tables, and paired source-member uncertainty reporting.\n"
        )
        (staging / "RUN_CONTRACT.txt").write_text(
            f"command={sys.executable} {' '.join(sys.argv)}\nbase_commit={subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()}\n"
            f"bootstrap_seed={BOOTSTRAP_SEED}\nbootstrap_replicates={replicates}\nbootstrap_unit=source_member\n"
            "validation_opened=0\ntest_opened=0\nobserved_data_opened=0\n"
        )
        summary_payload = {
            "schema_version": 1,
            "status": "pn_c7v_errata_support_diagnostics_and_paired_uncertainty_complete",
            "supersedes": str(C7V_V2),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": replicates,
            "bootstrap_unit": "source_member",
            "common_bootstrap_registry_sha256": registry_sha256,
            "models": list(MODEL_ORDER),
            "metrics_with_intervals_per_model": len(BOOTSTRAP_METRICS),
            "paired_comparisons": int(paired_summary.comparison.nunique()),
            "figures": len(provenance),
            "tables": len(tables),
            "bdt_transferred_qcd_fraction": float(nominal.set_index("model").loc["bdt", "transferred_qcd_fraction"]),
            "validation_payload_files_opened": 0,
            "test_or_evaluation_payload_files_opened": 0,
            "observed_data_opened": 0,
            "smoke": smoke,
        }
        write_json(staging / "summary.json", summary_payload)
        write_tsv(staging / "artifact_manifest.tsv", artifact_rows(staging, exclude={"artifact_manifest.tsv", "SHA256SUMS", "COMPLETE"}))
        manifest_sha = seal_checkpoint(staging, output, "pn_c7v errata and paired source-member uncertainty complete")
        print(json.dumps({"output": str(output), "paper_output": str(paper_output), "manifest_sha256": manifest_sha,
                          "figures": len(provenance), "tables": len(tables), "replicates": replicates}, sort_keys=True))
    except Exception as exc:
        failure = {"status": "failed_preserved_staging", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        if staging.exists(): write_json(staging / "FAILURE.json", failure)
        if paper_output.exists(): write_json(paper_output / "FAILURE.json", failure)
        raise


def main() -> None:
    args = parse_args()
    run(args.output.resolve(), args.paper_output.resolve(), args.inspection_report.resolve() if args.inspection_report else None,
        smoke=args.smoke)


if __name__ == "__main__":
    main()
