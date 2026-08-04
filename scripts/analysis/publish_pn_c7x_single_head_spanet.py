#!/usr/bin/env python3
"""Build the uncertainty-qualified PN-c7x single-head SPA-Net review package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from hh4b_plot_style import apply_hh4b_paper_style
from pn_c7_bootstrap_common import (
    BOOTSTRAP_METRICS,
    bootstrap_validity_summary,
    evaluate_frozen_baselines,
    member_positions,
    paired_model_differences,
    summarize_metric_replicas,
)
from pn_c7_ml_common import (
    CHECKPOINTS,
    artifact_rows,
    bootstrap_quantile_summary,
    parse_sha256sums,
    require,
    sha256,
    source_member_bootstrap_draws,
    verify_checkpoint,
    write_json,
    write_tsv,
)
from pn_c7x_spanet_evaluation import (
    assignment_paired_differences,
    evaluate_assignment_bootstrap,
    evaluate_binned_pairing,
    summarize_assignment_replicas,
    summarize_binned_pairing,
)


REPO = Path(__file__).resolve().parents[2]
C7R = CHECKPOINTS["c7r"]
C7S = CHECKPOINTS["c7s"]
C7T = CHECKPOINTS["c7t"]
TRAINING = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/"
    "pn_c7x_single_head_spanet_training_20260803_v1"
)
TRAINING_MANIFEST_SHA256 = "17ad12fab1b354e34624c97a488c359600ea9adff6b6334306a8ed9418a94b82"
CONTRACT = REPO / "docs/paper/jhep_hh4b_ml/HIG_24_015_SINGLE_HEAD_SPANET_CONTRACT.md"
COMMON_REGISTRY_SHA256 = "37218e5531018ea2a79473bfbee43df657b309cf06d89751509e74b586d9dc29"
BOOTSTRAP_SEED = 20260802
BOOTSTRAP_REPLICATES = 1000
MODEL_ORDER = ("bdt", "single_head")
DISPLAY = {"bdt": "Inclusive BDT", "single_head": "Single-head SPA-Net"}
PAIRING_DISPLAY = {"frozen_geometric": "Frozen geometric", "single_head_learned": "Single-head SPA-Net"}
COLORS = {"bdt": "#009E73", "single_head": "#0072B2", "frozen_geometric": "#D55E00", "single_head_learned": "#0072B2"}


def verify_external_checkpoint(path: Path, expected_manifest_sha: str, role: str) -> list[dict[str, Any]]:
    manifest = path / "SHA256SUMS"
    require(manifest.is_file() and sha256(manifest) == expected_manifest_sha, f"checkpoint manifest drift: {path}")
    evidence = []
    for relative, expected in parse_sha256sums(manifest).items():
        member = path / relative
        require(member.is_file() and sha256(member) == expected, f"checkpoint member drift: {member}")
        evidence.append({"path": str(member), "bytes": member.stat().st_size, "sha256": expected, "role": role})
    return evidence


def load_inputs() -> dict[str, Any]:
    evidence = []
    for tag in ("c7r", "c7s", "c7t"):
        evidence.extend({
            "path": row["checkpoint_root"] + "/" + row["relative_path"],
            "bytes": row["bytes"], "sha256": row["sha256"], "role": f"sealed_{tag}_input",
        } for row in verify_checkpoint(tag))
    evidence.extend(verify_external_checkpoint(TRAINING, TRAINING_MANIFEST_SHA256, "sealed_single_head_training_input"))
    require(CONTRACT.is_file(), "single-head contract missing")
    evidence.append({"path": str(CONTRACT), "bytes": CONTRACT.stat().st_size, "sha256": sha256(CONTRACT), "role": "frozen_contract"})

    training = pd.read_parquet(C7S / "tables/train_fourb_model_development.parquet")
    projection = pd.read_parquet(C7S / "tables/train_primary_physical_projection.parquet")
    direct = pd.read_parquet(C7S / "tables/train_direct_qcd_secondary_projection.parquet")
    members = pd.read_csv(C7S / "member_fold_registry.tsv", sep="\t")
    c7t_train = pd.read_parquet(C7T / "predictions/train_fourb_oof_scores.parquet")
    c7t_projection = pd.read_parquet(C7T / "predictions/train_primary_projection_oof_scores.parquet")
    single_train = pd.read_parquet(TRAINING / "predictions/train_fourb_single_head_oof.parquet")
    single_projection = pd.read_parquet(TRAINING / "predictions/train_primary_projection_single_head_oof.parquet")
    require((len(training), len(projection), len(direct), len(members)) == (30705, 31225, 30705, 441), "input row-count drift")
    identity = ["registry_group_id", "event"]
    require(training[identity].reset_index(drop=True).equals(c7t_train[identity].reset_index(drop=True)), "c7t training score identity drift")
    require(training[identity].reset_index(drop=True).equals(single_train[identity].reset_index(drop=True)), "single-head training score identity drift")
    require(projection[identity].reset_index(drop=True).equals(c7t_projection[identity].reset_index(drop=True)), "c7t projection score identity drift")
    require(projection[identity].reset_index(drop=True).equals(single_projection[identity].reset_index(drop=True)), "single-head projection score identity drift")
    training_scores = c7t_train.copy()
    projection_scores = c7t_projection.copy()
    training_scores["single_head_score"] = single_train.single_head_score.to_numpy()
    projection_scores["single_head_score"] = single_projection.single_head_score.to_numpy()
    require(bool(np.isfinite(training_scores[["bdt_score", "single_head_score"]].to_numpy()).all()), "nonfinite training score")
    require(bool(np.isfinite(projection_scores[["bdt_score", "single_head_score"]].to_numpy()).all()), "nonfinite projection score")
    selection_masks = {
        "single_head": {
            "training": single_train.selected_at_nested_operating_point.to_numpy(dtype=bool),
            "projection": single_projection.selected_at_nested_operating_point.to_numpy(dtype=bool),
            "direct": single_train.selected_at_nested_operating_point.to_numpy(dtype=bool),
            "threshold": "five_outer_fold_specific_thresholds_selected_by_inner_source_OOF",
        }
    }
    return {
        "training": training,
        "projection": projection,
        "direct": direct,
        "members": members,
        "training_scores": training_scores,
        "projection_scores": projection_scores,
        "metrics": pd.read_csv(C7T / "baseline_metrics.tsv", sep="\t"),
        "factor": pd.read_csv(C7R / "transfer_factor_registry.tsv", sep="\t"),
        "selection_masks": selection_masks,
        "single_train": single_train,
        "single_projection": single_projection,
        "fold_stability": pd.read_csv(TRAINING / "outer_fold_training_and_stability.tsv", sep="\t"),
        "runtime": pd.read_csv(TRAINING / "runtime.tsv", sep="\t"),
        "model_manifest": pd.read_csv(TRAINING / "model_artifact_manifest.tsv", sep="\t"),
        "evidence": evidence,
    }


def bootstrap_curve_data(
    training: pd.DataFrame,
    scores: pd.DataFrame,
    members: pd.DataFrame,
    draws: np.ndarray,
    *,
    weighted: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = member_positions(training, members.reset_index(drop=True), "ROC training table")
    labels = training.registry_training_target.to_numpy(dtype=np.int8)
    development = training.development_hierarchical_weight.to_numpy(dtype=np.float64)
    evaluation_draws = np.vstack((np.ones((1, len(members)), dtype=np.int16), draws))
    fpr_grid = np.linspace(0.0, 1.0, 101)
    signal_grid = np.linspace(0.05, 0.95, 19)
    roc_rows = []
    rejection_rows = []
    for model in MODEL_ORDER:
        score = scores[f"{model}_score"].to_numpy(dtype=np.float64)
        tpr_values = []
        rejection_values = []
        for multiplicities in evaluation_draws:
            row_weight = multiplicities[positions].astype(np.float64)
            if weighted:
                row_weight *= development
            fpr, tpr, _ = roc_curve(labels, score, sample_weight=row_weight)
            tpr_values.append(np.interp(fpr_grid, fpr, tpr))
            unique_tpr, unique_indices = np.unique(tpr, return_index=True)
            corresponding_fpr = fpr[unique_indices]
            rejection_values.append(1.0 - np.interp(signal_grid, unique_tpr, corresponding_fpr))
        tpr_values = np.asarray(tpr_values)
        rejection_values = np.asarray(rejection_values)
        for index, fpr_value in enumerate(fpr_grid):
            replicas = tpr_values[1:, index]
            roc_rows.append({
                "model": model, "weighting": "development_hierarchical" if weighted else "unweighted",
                "background_efficiency": fpr_value, "signal_efficiency_central": tpr_values[0, index],
                "signal_efficiency_p16": np.percentile(replicas, 16), "signal_efficiency_p84": np.percentile(replicas, 84),
                "signal_efficiency_p2p5": np.percentile(replicas, 2.5), "signal_efficiency_p97p5": np.percentile(replicas, 97.5),
                "valid_replicas": len(replicas), "support_pass": True,
            })
        for index, signal_value in enumerate(signal_grid):
            replicas = rejection_values[1:, index]
            rejection_rows.append({
                "model": model, "weighting": "development_hierarchical" if weighted else "unweighted",
                "signal_efficiency": signal_value, "background_rejection_central": rejection_values[0, index],
                "background_rejection_p16": np.percentile(replicas, 16), "background_rejection_p84": np.percentile(replicas, 84),
                "background_rejection_p2p5": np.percentile(replicas, 2.5), "background_rejection_p97p5": np.percentile(replicas, 97.5),
                "valid_replicas": len(replicas), "support_pass": True,
            })
    return pd.DataFrame(roc_rows), pd.DataFrame(rejection_rows)


def component_summary(nominal: pd.DataFrame, replicas: pd.DataFrame) -> pd.DataFrame:
    fields = (
        "selected_signal_yield", "selected_background_yield", "ordinary_background_yield",
        "transferred_qcd_yield", "transferred_qcd_fraction",
    )
    nominal_index = nominal.set_index("model")
    rows = []
    for model, group in replicas.groupby("model", sort=False):
        for field in fields:
            rows.append({
                "model": model, "component": field, "nominal_full_oof": float(nominal_index.loc[model, field]),
                **bootstrap_quantile_summary(group[field].to_numpy(dtype=np.float64)),
            })
    return pd.DataFrame(rows)


def asymmetric(summary: pd.DataFrame, model: str, metric: str) -> tuple[float, float, float]:
    row = summary[(summary.model == model) & (summary.metric == metric)].iloc[0]
    center = float(row.bootstrap_median)
    return center, center - float(row.bootstrap_p16), float(row.bootstrap_p84) - center


def paper_stamp(ax: Any) -> None:
    ax.text(0.0, 1.02, "CMS-style", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")
    ax.text(1.0, 1.02, "Delphes simulation · Work in progress · Train-only source-group OOF",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)


def paper_stamp_figure(
    figure: Any,
    *,
    wspace: float | None = None,
    hspace: float | None = None,
) -> None:
    adjustments: dict[str, float] = {"top": 0.88}
    if wspace is not None:
        adjustments["wspace"] = wspace
    if hspace is not None:
        adjustments["hspace"] = hspace
    figure.subplots_adjust(**adjustments)
    figure.text(0.01, 0.98, "CMS-style", ha="left", va="top", fontweight="bold")
    figure.text(
        0.99,
        0.98,
        "Delphes simulation · Work in progress · Train-only source-group OOF",
        ha="right",
        va="top",
        fontsize=8,
    )


def categorical_ticks(ax: Any, positions: np.ndarray, labels: Iterable[str], rotation: float = 0.0) -> None:
    ax.set_xticks(positions)
    ax.set_xticklabels(list(labels), rotation=rotation)


def write_figure(
    figure: Any,
    asset_id: str,
    title: str,
    data: pd.DataFrame,
    caption: str,
    root: Path,
    paper: Path,
) -> dict[str, Any]:
    pdf = root / "figures" / f"{asset_id}.pdf"
    png = root / "figures" / f"{asset_id}.png"
    source = root / "source_data" / f"{asset_id}.tsv"
    caption_path = root / "captions" / f"{asset_id}.tex"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    data.to_csv(source, sep="\t", index=False)
    caption_path.write_text(caption.strip() + "\n")
    for path, subdir in ((pdf, "figures"), (png, "figures"), (source, "source_data"), (caption_path, "captions")):
        shutil.copy2(path, paper / subdir / path.name)
    return {
        "asset_id": asset_id, "title": title,
        "pdf": pdf.relative_to(root).as_posix(), "pdf_sha256": sha256(pdf),
        "png": png.relative_to(root).as_posix(), "png_sha256": sha256(png),
        "source_data": source.relative_to(root).as_posix(), "source_data_sha256": sha256(source),
        "caption": caption_path.relative_to(root).as_posix(), "caption_sha256": sha256(caption_path),
        "uncertainty_contract": "source_member_bootstrap_68_percent_or_selection_stability_as_captioned",
    }


def make_figures(
    root: Path,
    paper: Path,
    metric_summary: pd.DataFrame,
    component: pd.DataFrame,
    roc_weighted: pd.DataFrame,
    roc_unweighted: pd.DataFrame,
    rejection: pd.DataFrame,
    assignment_summary: pd.DataFrame,
    binned_summary: pd.DataFrame,
    fold_stability: pd.DataFrame,
) -> list[dict[str, Any]]:
    figures = []
    for asset_id, title, data, weighted in (
        ("fig_c7x_01_weighted_roc", "Weighted source-group OOF ROC", roc_weighted, True),
        ("fig_c7x_02_unweighted_roc", "Unweighted source-group OOF ROC", roc_unweighted, False),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        for model in MODEL_ORDER:
            sub = data[data.model == model]
            ax.plot(sub.background_efficiency.to_numpy(), sub.signal_efficiency_central.to_numpy(), color=COLORS[model], label=DISPLAY[model])
            ax.fill_between(sub.background_efficiency.to_numpy(), sub.signal_efficiency_p16.to_numpy(), sub.signal_efficiency_p84.to_numpy(),
                            color=COLORS[model], alpha=0.22)
        ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
        ax.set(xlabel="Background efficiency", ylabel="Signal efficiency", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="lower right")
        paper_stamp(ax)
        caption = (
            ("Development-weighted" if weighted else "Unweighted")
            + " train-only source-group OOF ROC curves. Shaded regions are paired source-member-bootstrap 68\\% bands; central lines are full OOF curves."
        )
        figures.append(write_figure(fig, asset_id, title, data, caption, root, paper))

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for model in MODEL_ORDER:
        sub = rejection[rejection.model == model]
        ax.plot(sub.signal_efficiency.to_numpy(), sub.background_rejection_central.to_numpy(), color=COLORS[model], label=DISPLAY[model])
        ax.fill_between(sub.signal_efficiency.to_numpy(), sub.background_rejection_p16.to_numpy(), sub.background_rejection_p84.to_numpy(),
                        color=COLORS[model], alpha=0.22)
    ax.set(xlabel="Signal efficiency", ylabel="Background rejection $1-\\epsilon_B$", xlim=(0.05, 0.95), ylim=(0, 1))
    ax.legend(loc="lower left")
    paper_stamp(ax)
    figures.append(write_figure(
        fig, "fig_c7x_03_efficiency_rejection", "Signal efficiency versus background rejection", rejection,
        "Development-weighted signal efficiency versus background rejection. Bands are paired source-member-bootstrap 68\\% intervals.", root, paper,
    ))

    auc_data = metric_summary[metric_summary.metric == "weighted_auc"].copy()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    centers = []; lower = []; upper = []
    for model in MODEL_ORDER:
        center, down, up = asymmetric(metric_summary, model, "weighted_auc")
        centers.append(center); lower.append(down); upper.append(up)
    x = np.arange(2)
    ax.errorbar(x, centers, yerr=[lower, upper], fmt="o", color="#222222", capsize=5, markersize=7)
    ax.scatter(x, [auc_data[auc_data.model == model].nominal_full_oof.iloc[0] for model in MODEL_ORDER], marker="x", color="#D55E00", label="Full OOF")
    categorical_ticks(ax, x, [DISPLAY[model] for model in MODEL_ORDER]); ax.set_ylabel("Weighted AUC"); ax.set_ylim(0.55, 0.82)
    ax.legend(); paper_stamp(ax)
    figures.append(write_figure(fig, "fig_c7x_04_weighted_auc", "Weighted AUC comparison", auc_data,
                                "Weighted AUC bootstrap medians with asymmetric 68\\% source-member intervals; crosses show nominal full-OOF values.", root, paper))

    za_data = metric_summary[metric_summary.metric.isin(["nominal_asimov_ZA", "systematic_aware_asimov_ZA"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for axis, metric, label, log in (
        (axes[0], "nominal_asimov_ZA", "Nominal $Z_A$", False),
        (axes[1], "systematic_aware_asimov_ZA", "Systematics-aware $Z_A$", True),
    ):
        centers=[]; lower=[]; upper=[]
        for model in MODEL_ORDER:
            center, down, up = asymmetric(metric_summary, model, metric)
            centers.append(center); lower.append(down); upper.append(up)
        axis.errorbar(np.arange(2), centers, yerr=[lower, upper], fmt="o", color="#222222", capsize=5)
        categorical_ticks(axis, np.arange(2), [DISPLAY[m] for m in MODEL_ORDER], rotation=12)
        axis.set_ylabel(label)
        if log: axis.set_yscale("log")
    paper_stamp_figure(fig, wspace=0.35)
    figures.append(write_figure(fig, "fig_c7x_05_sensitivity", "Nominal and systematics-aware sensitivity", za_data,
                                "Nominal and frozen-multijet-systematics-aware $Z_A$ bootstrap medians with asymmetric 68\\% intervals. The shared transfer nuisance is recomputed in every source draw.", root, paper))

    sb_neff = metric_summary[metric_summary.metric.isin(["signal_over_background", "background_effective_events"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for axis, metric, label in (
        (axes[0], "signal_over_background", "$S/B$"),
        (axes[1], "background_effective_events", "Background $N_{\\mathrm{eff}}$"),
    ):
        centers=[]; lower=[]; upper=[]
        for model in MODEL_ORDER:
            center, down, up = asymmetric(metric_summary, model, metric)
            centers.append(center); lower.append(down); upper.append(up)
        axis.errorbar(np.arange(2), centers, yerr=[lower, upper], fmt="o", capsize=5, color="#222222")
        categorical_ticks(axis, np.arange(2), [DISPLAY[m] for m in MODEL_ORDER], rotation=12); axis.set_ylabel(label)
    paper_stamp_figure(fig, wspace=0.35)
    figures.append(write_figure(fig, "fig_c7x_06_sb_neff", "S/B and effective background statistics", sb_neff,
                                "$S/B$ and background effective-event bootstrap medians with asymmetric 68\\% source-member intervals.", root, paper))

    comp_abs = component[component.component.isin(["ordinary_background_yield", "transferred_qcd_yield", "selected_background_yield"])].copy()
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    x=np.arange(2); ordinary=[]; qcd=[]; total=[]; lo=[]; hi=[]
    for model in MODEL_ORDER:
        rows=component[component.model == model].set_index("component")
        ordinary.append(rows.loc["ordinary_background_yield", "bootstrap_median"])
        qcd.append(rows.loc["transferred_qcd_yield", "bootstrap_median"])
        total.append(rows.loc["selected_background_yield", "bootstrap_median"])
        lo.append(total[-1]-rows.loc["selected_background_yield", "bootstrap_p16"])
        hi.append(rows.loc["selected_background_yield", "bootstrap_p84"]-total[-1])
    yield_scale = 1.0e6
    ax.bar(x, np.asarray(ordinary) / yield_scale, label="Non-QCD background", color="#E69F00")
    ax.bar(x, np.asarray(qcd) / yield_scale, bottom=np.asarray(ordinary) / yield_scale, label="Transferred QCD", color="#56B4E9")
    ax.errorbar(x, np.asarray(total) / yield_scale, yerr=np.asarray([lo,hi]) / yield_scale, fmt="none", color="black", capsize=4, label="Total 68% interval")
    ax.set_ylim(bottom=0); categorical_ticks(ax,x,[DISPLAY[m] for m in MODEL_ORDER]); ax.set_ylabel("Selected background yield [$10^6$]"); ax.legend(); paper_stamp(ax)
    figures.append(write_figure(fig,"fig_c7x_07_background_absolute","Absolute selected-background composition",comp_abs,
                                "Absolute selected-background yields. The stacked linear axis starts at zero; total-yield bars are asymmetric 68\\% source-member intervals. Transferred QCD is the primary exactly-$3b$ to $\\geq4b$ prediction.",root,paper))

    comp_frac = component[component.component == "transferred_qcd_fraction"].copy()
    fig, ax = plt.subplots(figsize=(6.4,4.5)); centers=[];lower=[];upper=[]
    for model in MODEL_ORDER:
        row=comp_frac[comp_frac.model==model].iloc[0]; center=row.bootstrap_median
        centers.append(center); lower.append(center-row.bootstrap_p16); upper.append(row.bootstrap_p84-center)
    ax.errorbar(np.arange(2),centers,yerr=[lower,upper],fmt="o",capsize=5,color="#222222")
    categorical_ticks(ax,np.arange(2),[DISPLAY[m] for m in MODEL_ORDER]); ax.set_ylabel("Transferred-QCD fraction"); ax.set_ylim(0,1); paper_stamp(ax)
    figures.append(write_figure(fig,"fig_c7x_08_background_fraction","Fractional selected-background composition",comp_frac,
                                "Transferred-QCD fraction of selected background with asymmetric 68\\% source-member intervals.",root,paper))

    acc_metrics = assignment_summary[assignment_summary.metric.isin(["learned_exact_event_pairing_accuracy","geometric_exact_event_pairing_accuracy"])].copy()
    fig, ax=plt.subplots(figsize=(6.4,4.5)); labels=["geometric_exact_event_pairing_accuracy","learned_exact_event_pairing_accuracy"]
    centers=[];lower=[];upper=[]
    for metric in labels:
        row=assignment_summary[assignment_summary.metric==metric].iloc[0]; center=row.bootstrap_median
        centers.append(center); lower.append(center-row.bootstrap_p16); upper.append(row.bootstrap_p84-center)
    ax.errorbar(np.arange(2),centers,yerr=[lower,upper],fmt="o",capsize=5,color="#222222")
    categorical_ticks(ax,np.arange(2),["Frozen geometric","Single-head SPA-Net"]); ax.set_ylabel("Exact pairing accuracy"); ax.set_ylim(0.75,0.9); paper_stamp(ax)
    figures.append(write_figure(fig,"fig_c7x_09_pairing_accuracy","Exact pairing accuracy",acc_metrics,
                                "Exact truth-pairing accuracy on matchable signal with asymmetric 68\\% complete-source-member intervals.",root,paper))

    fig, axes=plt.subplots(2,2,figsize=(11.2,8.0)); variables=list(binned_summary.variable.drop_duplicates())
    for axis, variable in zip(axes.flat,variables):
        sub=binned_summary[binned_summary.variable==variable]
        for offset,model in ((-0.08,"frozen_geometric"),(0.08,"single_head_learned")):
            rows=sub[sub.model==model].sort_values("bin_index"); center=rows.bootstrap_median.to_numpy()
            axis.errorbar(rows.bin_index.to_numpy(dtype=float)+offset,center,yerr=[center-rows.bootstrap_p16.to_numpy(),rows.bootstrap_p84.to_numpy()-center],fmt="o-",capsize=3,label=PAIRING_DISPLAY[model],color=COLORS[model])
        labels_ticks=sub[sub.model=="frozen_geometric"].sort_values("bin_index").bin_label
        categorical_ticks(axis,np.arange(len(labels_ticks)),labels_ticks,rotation=20); axis.set_ylabel("Pairing accuracy"); axis.set_xlabel(sub.axis_label.iloc[0]); axis.set_ylim(0.6,1.0)
    axes[0,0].legend(loc="lower right")
    paper_stamp_figure(fig, wspace=0.20, hspace=0.42)
    figures.append(write_figure(fig,"fig_c7x_10_pairing_binned","Pairing accuracy by kinematics and activity",binned_summary,
                                "Pairing accuracy versus $m_{HH}$, truth-Higgs $p_T$, selected-jet multiplicity, and extra-jet activity. Error bars are asymmetric 68\\% complete-source-member intervals.",root,paper))

    mass_data=assignment_summary[assignment_summary.metric.isin(["learned_mass_residual_RMS_GeV","geometric_mass_residual_RMS_GeV"])].copy()
    fig,ax=plt.subplots(figsize=(6.4,4.5)); metrics=["geometric_mass_residual_RMS_GeV","learned_mass_residual_RMS_GeV"]
    centers=[];lower=[];upper=[]
    for metric in metrics:
        row=assignment_summary[assignment_summary.metric==metric].iloc[0]; center=row.bootstrap_median
        centers.append(center);lower.append(center-row.bootstrap_p16);upper.append(row.bootstrap_p84-center)
    ax.errorbar(np.arange(2),centers,yerr=[lower,upper],fmt="o",capsize=5,color="#222222")
    categorical_ticks(ax,np.arange(2),["Frozen geometric","Single-head SPA-Net"]);ax.set_ylabel("Higgs-mass residual RMS [GeV]");paper_stamp(ax)
    figures.append(write_figure(fig,"fig_c7x_11_mass_resolution","Reconstructed Higgs-mass resolution",mass_data,
                                "RMS of reconstructed Higgs-candidate mass residuals relative to 125 GeV on matchable signal. Error bars are asymmetric 68\\% source-member intervals; the geometric algorithm explicitly optimizes mass proximity.",root,paper))

    fig,axes=plt.subplots(1,2,figsize=(10.2,4.4)); x=fold_stability.outer_fold
    axes[0].plot(x.to_numpy(),fold_stability.inner_oof_operating_threshold.to_numpy(),"o-",color=COLORS["single_head"]);axes[0].set(xlabel="Outer fold",ylabel="Inner-OOF score threshold")
    axes[1].plot(x.to_numpy(),fold_stability.outer_matchable_assignment_accuracy.to_numpy(),"o-",label="Outer held",color=COLORS["single_head"])
    axes[1].plot(x.to_numpy(),fold_stability.inner_matchable_assignment_accuracy.to_numpy(),"s--",label="Inner OOF",color="#D55E00");axes[1].set(xlabel="Outer fold",ylabel="Pairing accuracy");axes[1].legend()
    paper_stamp_figure(fig, wspace=0.35)
    stability_data=fold_stability[["outer_fold","inner_oof_operating_threshold","inner_matchable_assignment_accuracy","outer_matchable_assignment_accuracy"]]
    figures.append(write_figure(fig,"fig_c7x_12_selection_stability","Operating-point and fold stability",stability_data,
                                "Selection-stability diagnostic across outer folds. Thresholds are selected only from inner OOF sources. These fold variations are kept separate from source-bootstrap evaluation uncertainty.",root,paper))
    return figures


def fmt_asym(row: pd.Series, digits: int = 4, scientific: bool = False) -> str:
    center=float(row.bootstrap_median); down=center-float(row.bootstrap_p16); up=float(row.bootstrap_p84)-center
    if scientific:
        if center == 0: return "$0$"
        exponent=int(math.floor(math.log10(abs(center)))); scale=10.0**exponent
        return f"$({center/scale:.{digits}f}^{{+{up/scale:.{digits}f}}}_{{-{down/scale:.{digits}f}}})\\times10^{{{exponent}}}$"
    return f"${center:.{digits}f}^{{+{up:.{digits}f}}}_{{-{down:.{digits}f}}}$"


def write_table(path: Path, content: str, paper: Path) -> dict[str, Any]:
    path.write_text(content)
    shutil.copy2(path, paper / "tables" / path.name)
    return {"asset_id": path.stem, "path": path.relative_to(path.parents[1]).as_posix(), "sha256": sha256(path)}


def make_tables(
    root: Path,
    paper: Path,
    metric_summary: pd.DataFrame,
    paired: pd.DataFrame,
    assignment_summary: pd.DataFrame,
    assignment_paired: pd.DataFrame,
    fold_stability: pd.DataFrame,
) -> list[dict[str, Any]]:
    tables=[]
    lines=[r"\begin{tabular}{lccc}",r"\toprule","Model & Weighted AUC & Nominal $Z_A$ & Syst. $Z_A$ \\\\",r"\midrule"]
    for model in MODEL_ORDER:
        rows=metric_summary[metric_summary.model==model].set_index("metric")
        lines.append(f"{DISPLAY[model]} & {fmt_asym(rows.loc['weighted_auc'])} & {fmt_asym(rows.loc['nominal_asimov_ZA'],5)} & {fmt_asym(rows.loc['systematic_aware_asimov_ZA'],3,True)} \\\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\par\smallskip",r"\begin{tabular}{lcc}",r"\toprule","Model & $S/B$ & Background $N_{\\rm eff}$ \\\\",r"\midrule"]
    for model in MODEL_ORDER:
        rows=metric_summary[metric_summary.model==model].set_index("metric")
        lines.append(f"{DISPLAY[model]} & {fmt_asym(rows.loc['signal_over_background'],3,True)} & {fmt_asym(rows.loc['background_effective_events'],2)} \\\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\par\medskip"]
    tables.append(write_table(root/"tables"/"tab_c7x_01_classification.tex","\n".join(lines)+"\n",paper))

    a=assignment_summary.set_index("metric")
    lines=[r"{\small",r"\begin{tabular}{lcccc}",r"\toprule","Pairing & Exact accuracy & Matchable efficiency & Ambiguity & Mass RMS [GeV] \\\\",r"\midrule",
           f"Frozen geometric & {fmt_asym(a.loc['geometric_exact_event_pairing_accuracy'])} & {fmt_asym(a.loc['matchable_event_efficiency'])} & {fmt_asym(a.loc['ambiguity_fraction'],3,True)} & {fmt_asym(a.loc['geometric_mass_residual_RMS_GeV'],2)} \\\\",
           f"Single-head SPA-Net & {fmt_asym(a.loc['learned_exact_event_pairing_accuracy'])} & {fmt_asym(a.loc['matchable_event_efficiency'])} & {fmt_asym(a.loc['ambiguity_fraction'],3,True)} & {fmt_asym(a.loc['learned_mass_residual_RMS_GeV'],2)} \\\\",r"\bottomrule",r"\end{tabular}"]
    lines += [r"}",r"\par\medskip"]
    tables.append(write_table(root/"tables"/"tab_c7x_02_assignment.tex","\n".join(lines)+"\n",paper))

    required=["weighted_auc","nominal_asimov_ZA","systematic_aware_asimov_ZA","signal_over_background","background_effective_events"]
    lines=[r"\begin{tabular}{lrrrr}",r"\toprule","Metric & Nominal $\\Delta$ & Median $\\Delta$ & 68\\% interval & $P(\\Delta>0)$ \\\\",r"\midrule"]
    for metric in required:
        row=paired[(paired.comparison=="single_head_minus_bdt")&(paired.metric==metric)].iloc[0]
        lines.append(f"{metric.replace('_',' ')} & {row.nominal_difference:.5g} & {row.bootstrap_median_difference:.5g} & [{row.bootstrap_p16_difference:.5g}, {row.bootstrap_p84_difference:.5g}] & {row.fraction_valid_difference_greater_than_zero:.3f} \\\\")
    row=assignment_paired[assignment_paired.metric=="exact_event_pairing_accuracy"].iloc[0]
    lines.append(f"Pairing accuracy & {row.nominal_difference:.5g} & {row.bootstrap_median_difference:.5g} & [{row.bootstrap_p16_difference:.5g}, {row.bootstrap_p84_difference:.5g}] & {row.fraction_valid_difference_greater_than_zero:.3f} \\\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\par\medskip"]
    tables.append(write_table(root/"tables"/"tab_c7x_03_paired_differences.tex","\n".join(lines)+"\n",paper))

    lines=[r"\begin{tabular}{lrrrr}",r"\toprule","Outer fold & Threshold & Inner accuracy & Outer accuracy & Assignment train rows \\\\",r"\midrule"]
    for row in fold_stability.itertuples(): lines.append(f"{row.outer_fold} & {row.inner_oof_operating_threshold:.5f} & {row.inner_matchable_assignment_accuracy:.4f} & {row.outer_matchable_assignment_accuracy:.4f} & {row.outer_assignment_train_rows} \\\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\par\medskip"]
    tables.append(write_table(root/"tables"/"tab_c7x_04_operating_stability.tex","\n".join(lines)+"\n",paper))
    return tables


def claims(metric_summary: pd.DataFrame, assignment_summary: pd.DataFrame, paired: pd.DataFrame, registry_sha: str) -> list[dict[str, Any]]:
    rows=[]; index=1
    for source, frame, keys in (
        ("bootstrap_metric_summary.tsv",metric_summary,["model","metric"]),
        ("assignment_bootstrap_summary.tsv",assignment_summary,["metric"]),
        ("paired_model_differences.tsv",paired,["comparison","metric"]),
    ):
        for item in frame.to_dict("records"):
            rows.append({
                "claim_id":f"C7X-{index:04d}","source_file":source,
                "source_row":"; ".join(f"{key}={item[key]}" for key in keys),
                "value_field":"bootstrap_median" if "bootstrap_median" in item else "bootstrap_median_difference",
                "common_bootstrap_registry_sha256":registry_sha,"train_only":True,"validation_used":False,
                "test_used":False,"observed_data_used":False,"uncertainty":"asymmetric_p16_p84",
            }); index+=1
    return rows


def run(output: Path, smoke: bool) -> None:
    require(output.is_absolute() and output.parent.is_dir(), "output must be new absolute path with existing parent")
    require(not output.exists(), f"refusing to overwrite review output: {output}")
    output.mkdir()
    for directory in ("figures","source_data","captions","tables","paper"):
        (output/directory).mkdir()
    for directory in ("figures","source_data","captions","tables","sections"):
        (output/"paper"/directory).mkdir()
    apply_hh4b_paper_style()
    inputs=load_inputs()
    replicas=20 if smoke else BOOTSTRAP_REPLICATES
    draws,registry_rows=source_member_bootstrap_draws(inputs["members"],replicates=replicas,seed=BOOTSTRAP_SEED)
    write_tsv(output/"common_source_bootstrap_registry.tsv",registry_rows)
    registry_sha=sha256(output/"common_source_bootstrap_registry.tsv")
    if not smoke: require(registry_sha==COMMON_REGISTRY_SHA256,"common bootstrap registry identity drift")
    write_tsv(output/"bootstrap_registry_provenance.tsv",[{"registry_sha256":registry_sha,"replicas":replicas,"seed":BOOTSTRAP_SEED,"unit":"source_member","same_draws_for_every_model":True,"validation_payload_files_opened":0,"test_or_evaluation_payload_files_opened":0}])

    nominal,metric_replicas=evaluate_frozen_baselines(inputs,draws,model_order=MODEL_ORDER)
    metric_summary=summarize_metric_replicas(nominal,metric_replicas)
    paired,paired_replicas=paired_model_differences(nominal,metric_replicas,model_order=MODEL_ORDER)
    validity=bootstrap_validity_summary(metric_replicas)
    assignment_eval=evaluate_assignment_bootstrap(inputs["single_train"],inputs["members"],draws)
    assignment_summary=summarize_assignment_replicas(assignment_eval)
    assignment_paired,assignment_paired_replicas=assignment_paired_differences(assignment_eval)
    binned_eval=evaluate_binned_pairing(inputs["single_train"],inputs["members"],draws)
    binned_summary,binned_paired=summarize_binned_pairing(binned_eval)
    components=component_summary(nominal,metric_replicas)
    roc_weighted,rejection=bootstrap_curve_data(inputs["training"],inputs["training_scores"],inputs["members"],draws,weighted=True)
    roc_unweighted,_=bootstrap_curve_data(inputs["training"],inputs["training_scores"],inputs["members"],draws,weighted=False)

    write_tsv(output/"bootstrap_metric_summary.tsv",metric_summary.to_dict("records"))
    metric_replicas.to_parquet(output/"bootstrap_metric_replicas.parquet",index=False,compression="zstd")
    write_tsv(output/"paired_model_differences.tsv",paired.to_dict("records"))
    paired_replicas.to_parquet(output/"paired_model_difference_replicas.parquet",index=False,compression="zstd")
    write_tsv(output/"bootstrap_validity_summary.tsv",validity.to_dict("records"))
    write_tsv(output/"assignment_bootstrap_summary.tsv",assignment_summary.to_dict("records"))
    assignment_eval.to_parquet(output/"assignment_bootstrap_replicas.parquet",index=False,compression="zstd")
    write_tsv(output/"assignment_paired_differences.tsv",assignment_paired.to_dict("records"))
    assignment_paired_replicas.to_parquet(output/"assignment_paired_difference_replicas.parquet",index=False,compression="zstd")
    write_tsv(output/"binned_pairing_bootstrap_summary.tsv",binned_summary.to_dict("records"))
    write_tsv(output/"binned_pairing_paired_differences.tsv",binned_paired.to_dict("records"))
    binned_eval.to_parquet(output/"binned_pairing_bootstrap_replicas.parquet",index=False,compression="zstd")
    write_tsv(output/"selected_composition_bootstrap_summary.tsv",components.to_dict("records"))
    write_tsv(output/"operating_point_stability.tsv",inputs["fold_stability"].to_dict("records"))
    write_tsv(output/"runtime_summary.tsv",inputs["runtime"].to_dict("records"))
    write_tsv(output/"model_artifact_manifest.tsv",inputs["model_manifest"].to_dict("records"))
    write_tsv(output/"source_evidence_manifest.tsv",inputs["evidence"])

    figure_manifest=make_figures(output,output/"paper",metric_summary,components,roc_weighted,roc_unweighted,rejection,assignment_summary,binned_summary,inputs["fold_stability"])
    table_manifest=make_tables(output,output/"paper",metric_summary,paired,assignment_summary,assignment_paired,inputs["fold_stability"])
    write_tsv(output/"paper_figure_manifest.tsv",figure_manifest)
    write_tsv(output/"paper_table_manifest.tsv",table_manifest)
    write_tsv(output/"numerical_claim_registry.tsv",claims(metric_summary,assignment_summary,paired,registry_sha))
    for name in ("bootstrap_metric_summary.tsv","paired_model_differences.tsv","bootstrap_validity_summary.tsv","assignment_bootstrap_summary.tsv","assignment_paired_differences.tsv","binned_pairing_bootstrap_summary.tsv","binned_pairing_paired_differences.tsv","operating_point_stability.tsv","runtime_summary.tsv","model_artifact_manifest.tsv"):
        shutil.copy2(output/name,output/"paper"/name)

    primary=paired[(paired.comparison=="single_head_minus_bdt")&paired.metric.isin(["weighted_auc","nominal_asimov_ZA","systematic_aware_asimov_ZA","signal_over_background","background_effective_events"])]
    pair_acc=assignment_paired[assignment_paired.metric=="exact_event_pairing_accuracy"].iloc[0]
    section=(
        "\\section{Train-only single-head SPA-Net}\n"
        "This candidate-four assignment study uses Delphes simulation and nested train-source-group OOF only. No validation, test, observed data, or Run-2 expected-sensitivity claim is made. "
        "Exactly $3b\\rightarrow{\\geq4b}$ is the primary QCD projection; direct $\\geq4b$ QCD is secondary closure only.\n\n"
        f"The learned-minus-geometric exact-pairing difference has median {pair_acc.bootstrap_median_difference:.5f}; its 68\\% interval is [{pair_acc.bootstrap_p16_difference:.5f}, {pair_acc.bootstrap_p84_difference:.5f}]. "
        "All resampleable values and comparisons use the same complete-source-member draw registry. Classification differences are reported without improvement claims whenever their paired interval includes zero.\\par\\medskip\n"
    )
    (output/"paper"/"sections"/"single_head_spanet.tex").write_text(section)
    (output/"paper"/"README.md").write_text(
        "# PN-c7x train-only single-head SPA-Net\n\nDelphes simulation; nested train-source-group OOF; no validation, test, observed data, or Run-2 expected-sensitivity claim. "
        f"All primary uncertainties use {replicas} paired complete-source-member replicas; common registry SHA-256 `{registry_sha}`.\n"
    )
    (output/"paper"/"RESULTS_INDEX.md").write_text(
        f"# PN-c7x results index\n\n- Figures: {len(figure_manifest)} vector PDF/300-dpi PNG pairs with source data and captions.\n- Tables: {len(table_manifest)} modular LaTeX tables.\n- Primary transfer: exactly $3b$ to $\\geq4b$; direct $\\geq4b$ QCD is secondary closure only.\n"
    )
    (output/"paper"/"uncertainty_table.tex").write_text("".join((output/"tables"/Path(row["path"]).name).read_text() for row in table_manifest[:2]))
    (output/"paper"/"paired_improvement_table.tex").write_text((output/"tables"/"tab_c7x_03_paired_differences.tex").read_text())
    compile_lines=[r"\documentclass[11pt]{article}",r"\usepackage[margin=0.65in]{geometry}",r"\usepackage{booktabs}",r"\usepackage{graphicx}",r"\usepackage{amsmath}",r"\begin{document}",r"\input{sections/single_head_spanet.tex}"]
    compile_lines.extend(rf"\input{{tables/{Path(row['path']).name}}}" for row in table_manifest);compile_lines.append(r"\end{document}")
    (output/"paper"/"compile_fragments.tex").write_text("\n".join(compile_lines)+"\n")

    summary={
        "schema_version":1,"status":"pn_c7x_single_head_review_smoke_pass" if smoke else "pn_c7x_single_head_review_package_pass",
        "created_utc":datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),"replicas":replicas,"common_bootstrap_registry_sha256":registry_sha,
        "classification_summary":metric_summary.to_dict("records"),"classification_paired_differences":primary.to_dict("records"),
        "assignment_summary":assignment_summary.to_dict("records"),"assignment_paired_differences":assignment_paired.to_dict("records"),
        "figure_count":len(figure_manifest),"table_count":len(table_manifest),"environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__},
        "validation_payload_files_opened":0,"test_or_evaluation_payload_files_opened":0,"observed_data_opened":False,"full_run2_prediction":False,
        "review_required_before_sealing":True,
    }
    write_json(output/"summary.json",summary)
    (output/"README.md").write_text("# PN-c7x single-head review package\n\nUnsealed review output. Inspect every figure and compile the LaTeX package before finalization. No validation/test or observed data was accessed.\n")
    (output/"RUN_CONTRACT.txt").write_text("All primary resampleable values use the exact common complete-source-member bootstrap draws. Every model comparison is replica-aligned and paired. Undefined systematics replicas remain invalid. This review directory is not a sealed checkpoint.\n")
    write_tsv(output/"artifact_manifest.tsv",artifact_rows(output,exclude={"artifact_manifest.tsv"}))
    (output/"REVIEW_REQUIRED").write_text("visual_and_latex_review_required_before_checkpoint_sealing\n")
    print(json.dumps({"review_output":str(output),"registry_sha256":registry_sha,"figures":len(figure_manifest),"tables":len(table_manifest),"status":summary["status"]},sort_keys=True),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--smoke",action="store_true");args=parser.parse_args()
    run(args.output,args.smoke)


if __name__=="__main__": main()
