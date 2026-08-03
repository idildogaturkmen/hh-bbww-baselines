#!/usr/bin/env python3
"""Publish the sealed c7q--c7u train-only inventory, tables, and figures.

The authoritative output is an immutable checkpoint outside git.  A separate
lightweight paper export is produced for mechanical review/copying into the
repository.  Validation and test paths are neither accepted nor discovered.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_curve

from hh4b_plot_style import apply_hh4b_paper_style
from pn_c7_ml_common import (
    CHECKPOINTS,
    artifact_rows,
    asimov_za,
    asimov_za_with_uncertainty,
    effective_events,
    prepare_staging,
    require,
    require_columns,
    require_group_fold_integrity,
    seal_checkpoint,
    sha256,
    verify_all_checkpoints,
    write_json,
    write_tsv,
)


REPO = Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines")
MODEL_ORDER = ("cut", "bdt", "dense_dnn", "lbn_dnn")
LEARNED_MODELS = ("bdt", "dense_dnn", "lbn_dnn")
COLORS = {
    "cut": "#7f7f7f",
    "bdt": "#0072B2",
    "dense_dnn": "#D55E00",
    "lbn_dnn": "#009E73",
    "signal": "#D55E00",
    "ordinary": "#0072B2",
    "transferred": "#CC79A7",
    "direct": "#000000",
}
DISPLAY = {
    "cut": r"$R_{HH}(125,120)$ cut",
    "bdt": "BDT",
    "dense_dnn": "Dense DNN",
    "lbn_dnn": "LBN-DNN",
}
PAPER_LABELS = {
    "inclusive_mhh": r"Inclusive $m_{HH}$",
    "low_mhh": r"Low-$m_{HH}$ region",
    "high_mhh": r"High-$m_{HH}$ region",
    "cms_cr_nominal": "Nominal control-region transfer",
    "baseline_global_alternative": "Global baseline alternative",
    "outside_ge55_alternative": r"$R_{HH}(125,120)\geq55\,\mathrm{GeV}$ alternative",
    "nominal_transfer_factor_statistical": "Transfer-factor statistical uncertainty",
    "normalization_region_factor_envelope": "Normalization-region factor envelope",
    "cms_sr_direct_qcd_nonclosure": r"Direct $\geq4b$ QCD nonclosure",
    "multijet_normalization": "Multijet normalization",
    "multijet_normalization_and_shape": "Multijet normalization and shape",
    "cms_reference_signal_region_multijet": "Reference signal-region multijet",
    "hard_qcd": "Transferred QCD",
    "ordinary": "Non-QCD background",
    "signal": "Signal",
    "baseline_all_candidates": "Inclusive baseline",
    "cms_reference_sr_rhh125120_lt30": r"Reference $R_{HH}(125,120)<30\,\mathrm{GeV}$ region",
    "cms_reference_cr_rhh125120_ge30_lt55": r"Nominal $30\leq R_{HH}(125,120)<55\,\mathrm{GeV}$ control region",
    "cms_reference_outside_rhh125120_ge55": r"$R_{HH}(125,120)\geq55\,\mathrm{GeV}$ region",
    "optimized_nominal_sr_rhh125120_lt34": r"Frozen $R_{HH}(125,120)<34\,\mathrm{GeV}$ region",
    "higher_purity_sr_rhh125120_lt31p5": r"$R_{HH}(125,120)<31.5\,\mathrm{GeV}$ diagnostic",
    "higher_efficiency_sr_rhh125120_lt35p5": r"$R_{HH}(125,120)<35.5\,\mathrm{GeV}$ diagnostic",
}
TABLE_REGION_LABELS = {
    "baseline_all_candidates": "Inclusive baseline",
    "cms_reference_sr_rhh125120_lt30": r"$R_{HH}<30\,\mathrm{GeV}$",
    "cms_reference_cr_rhh125120_ge30_lt55": "Nominal control region",
    "cms_reference_outside_rhh125120_ge55": r"$R_{HH}\geq55\,\mathrm{GeV}$",
    "optimized_nominal_sr_rhh125120_lt34": r"Frozen $R_{HH}<34\,\mathrm{GeV}$",
    "higher_purity_sr_rhh125120_lt31p5": r"$R_{HH}<31.5\,\mathrm{GeV}$",
    "higher_efficiency_sr_rhh125120_lt35p5": r"$R_{HH}<35.5\,\mathrm{GeV}$",
}
BOOTSTRAP_SEED = 20260802
BOOTSTRAP_REPLICATES = 1000
TORCH_RUNTIME = Path("/tmp/hh4b_pn_c7t_baseline_env_20260802_v1/bin/python")
REQUESTED_FIGURES = (
    ("REQ-FIG-01", "Weighted source-group OOF ROC curves", "fig01_weighted_roc"),
    ("REQ-FIG-02", "Unweighted ROC curves as appendix diagnostics", "fig16_unweighted_roc_appendix"),
    ("REQ-FIG-03", "Signal efficiency versus background rejection", "fig03_efficiency_background_rejection"),
    ("REQ-FIG-04", "OOF classifier-score distributions", "fig02_oof_score_distributions"),
    ("REQ-FIG-05", "Physical-yield score distributions", "fig02_oof_score_distributions"),
    ("REQ-FIG-06", "Nominal Asimov ZA versus threshold", "fig04_nominal_za_scan"),
    ("REQ-FIG-07", "Systematic-aware Asimov ZA versus threshold", "fig05_systematic_za_scan"),
    ("REQ-FIG-08", "Weighted AUC comparison", "fig07_nominal_baseline_comparison"),
    ("REQ-FIG-09", "Nominal ZA comparison", "fig07_nominal_baseline_comparison"),
    ("REQ-FIG-10", "Bootstrap median and 16--84% ZA comparison", "fig06_bootstrap_za"),
    ("REQ-FIG-11", "Signal-to-background comparison", "fig07_nominal_baseline_comparison"),
    ("REQ-FIG-12", "Background effective-event comparison", "fig07_nominal_baseline_comparison"),
    ("REQ-FIG-13", "Earlier-snapshot versus authoritative-result comparison", "fig08_earlier_current_metrics"),
    ("REQ-FIG-14", "Signal, non-QCD, and direct-QCD population changes", "fig09_population_change"),
    ("REQ-FIG-15", "Selected process composition", "fig10_selected_process_composition"),
    ("REQ-FIG-16", "Three-b-to-four-b transfer-factor results", "fig11_transfer_factors"),
    ("REQ-FIG-17", r"Transfer closure against secondary direct $\geq4b$ QCD", "fig12_transfer_closure"),
    ("REQ-FIG-18", "Multijet systematic and nonclosure comparison", "fig17_multijet_systematic_nonclosure"),
    ("REQ-FIG-19", "R_HH distributions", "fig13_rhh_distributions"),
    ("REQ-FIG-20", "m(H1)-m(H2) mass planes", "fig14_higgs_mass_planes"),
    ("REQ-FIG-21", "mHH and promoted-jet diagnostics", "fig15_mhh_promoted_jet_quality"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def require_fresh_directory(path: Path) -> None:
    require(path.is_absolute(), f"output must be absolute: {path}")
    require(path.parent.is_dir(), f"output parent missing: {path.parent}")
    require(not path.exists(), f"refusing to overwrite output: {path}")


def require_prediction_alignment(frame: pd.DataFrame, scores: pd.DataFrame, label: str) -> None:
    keys = ["registry_group_id", "registry_member_index", "registry_oof_fold", "event"]
    require_columns(frame, keys, label)
    require_columns(scores, keys, f"{label} predictions")
    require(len(frame) == len(scores), f"{label} row-count mismatch")
    for column in keys:
        left = frame[column].astype(str).to_numpy()
        right = scores[column].astype(str).to_numpy()
        require(bool(np.array_equal(left, right)), f"{label} prediction alignment drift: {column}")


def histogram_rows(
    values: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
    *,
    figure: str,
    panel: str,
    series: str,
    normalize: bool,
) -> list[dict[str, Any]]:
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    sumw2, _ = np.histogram(values, bins=bins, weights=np.square(weights))
    if normalize:
        norm = float(counts.sum())
        require(norm > 0.0, f"empty normalized histogram: {figure}/{panel}/{series}")
        counts = counts / norm
        sumw2 = sumw2 / (norm * norm)
    return [
        {
            "figure": figure,
            "panel": panel,
            "series": series,
            "bin_low": float(lo),
            "bin_high": float(hi),
            "value": float(value),
            "statistical_uncertainty": float(math.sqrt(max(variance, 0.0))),
            "normalization": "unit_area" if normalize else "physical_yield",
        }
        for lo, hi, value, variance in zip(bins[:-1], bins[1:], counts, sumw2)
    ]


def step_xy(rows: list[dict[str, Any]], series: str, panel: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["series"] == series and row["panel"] == panel]
    return (
        np.asarray([row["bin_low"] for row in selected] + [selected[-1]["bin_high"]]),
        np.asarray([row["value"] for row in selected] + [selected[-1]["value"]]),
    )


def paper_label(ax: Any, *, subtitle: str = "Train-only source-group OOF") -> None:
    ax.text(0.0, 1.02, "Delphes simulation", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, fontweight="bold")
    ax.text(1.0, 1.02, subtitle, transform=ax.transAxes, ha="right", va="bottom", fontsize=9)


def save_plot(
    fig: Any,
    figures: Path,
    paper_figures: Path,
    stem: str,
    caption: str,
    source_rows: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> None:
    require(bool(source_rows), f"empty source data for {stem}")
    standardized = []
    for source_row in source_rows:
        row = dict(source_row)
        central = row.get("uncertainty_central", row.get("central_value", row.get("value", "")))
        lower = row.get("uncertainty_lower_68", "")
        upper = row.get("uncertainty_upper_68", "")
        kind = row.get("uncertainty_kind", "not_applicable_non_model_diagnostic")
        if central != "" and lower == "" and "statistical_uncertainty" in row:
            uncertainty = float(row["statistical_uncertainty"])
            lower = max(0.0, float(central) - uncertainty)
            upper = float(central) + uncertainty
            kind = "weighted_statistical"
        if central != "" and lower == "" and "transfer_factor_statistical_uncertainty" in row:
            uncertainty = float(row["transfer_factor_statistical_uncertainty"])
            lower = max(0.0, float(central) - uncertainty)
            upper = float(central) + uncertainty
            kind = "propagated_transfer_statistical"
        row.update({
            "uncertainty_central": central,
            "uncertainty_lower_68": lower,
            "uncertainty_upper_68": upper,
            "uncertainty_valid_replicas": row.get("uncertainty_valid_replicas", 0),
            "uncertainty_support_flag": row.get("uncertainty_support_flag", "not_applicable"),
            "uncertainty_kind": kind,
        })
        standardized.append(row)
    source_rows = standardized
    data_path = figures / f"{stem}.tsv"
    write_tsv(data_path, source_rows)
    for suffix in ("pdf", "png"):
        path = figures / f"{stem}.{suffix}"
        kwargs = {"dpi": 300, "bbox_inches": "tight"}
        if suffix == "pdf":
            kwargs["metadata"] = {"CreationDate": None, "ModDate": None, "Creator": "publish_pn_c7v_results_and_figures.py"}
        fig.savefig(path, **kwargs)
        require(path.stat().st_size > 1000, f"empty figure output: {path}")
        shutil.copy2(path, paper_figures / path.name)
    caption_path = figures / f"{stem}_caption.tex"
    caption_path.write_text(caption.strip() + "\n")
    shutil.copy2(caption_path, paper_figures / caption_path.name)
    plt.close(fig)
    provenance.append({
        "figure_id": stem.split("_", 1)[0],
        "stem": stem,
        "authoritative_pdf": f"figures/{stem}.pdf",
        "png_companion": f"figures/{stem}.png",
        "source_data": f"figures/{stem}.tsv",
        "caption": f"figures/{stem}_caption.tex",
        "paper_source_data": f"source_data/{stem}.tsv",
        "paper_caption": f"captions/{stem}_caption.tex",
        "split": "train",
        "evaluation": "source_group_oof_or_frozen_projection",
        "branding": "Delphes simulation; no CMS approval implied",
    })


def requested_figure_registry(provenance: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map every requested publication plot to a supported artifact or fail closed."""

    available = {str(row["stem"]) for row in provenance}
    rows = []
    for request_id, requested_plot, stem in REQUESTED_FIGURES:
        require(stem in available, f"requested figure is unsupported without a recorded reason: {request_id}")
        rows.append({
            "request_id": request_id,
            "requested_plot": requested_plot,
            "status": "supported",
            "figure_stem": stem,
            "missing_column_or_artifact": "",
            "reason": "supported by the checksum-verified frozen c7q--c7u artifacts",
        })
    require(len(rows) == 21, "requested figure registry count drift")
    return rows


def inspect_torch_checkpoint(path: Path) -> str:
    """Inspect an optional PyTorch model artifact without requiring torch at import time."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required only to inspect .pt model artifacts; rerun this "
            f"code path with the frozen c7t interpreter {TORCH_RUNTIME} "
            "(verified with torch 2.5.1+cpu), or another compatible torch environment"
        ) from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict), f"unexpected torch checkpoint: {path}")
    return "torch_keys=" + ",".join(sorted(payload))


def inspect_assets(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in evidence:
        path = Path(source["checkpoint_root"]) / source["relative_path"]
        suffix = path.suffix.lower()
        details = ""
        rows_count: int | str = ""
        columns_count: int | str = ""
        if suffix == ".json":
            payload = json.loads(path.read_text())
            details = "json_keys=" + ",".join(sorted(payload)[:80]) if isinstance(payload, dict) else "json_array"
        elif suffix == ".tsv":
            frame = pd.read_csv(path, sep="\t")
            rows_count, columns_count = frame.shape
            details = "columns=" + ",".join(frame.columns)
        elif suffix == ".parquet":
            metadata = pq.ParquetFile(path).metadata
            rows_count, columns_count = metadata.num_rows, metadata.num_columns
            details = "columns=" + ",".join(pq.ParquetFile(path).schema.names)
        elif suffix == ".pt":
            details = inspect_torch_checkpoint(path)
        elif suffix == ".md":
            details = f"text_lines={len(path.read_text().splitlines())}"
        elif suffix in {".txt", ""}:
            details = f"text_lines={len(path.read_text().splitlines())}"
        rows.append({
            **source,
            "artifact_kind": suffix.lstrip(".") or "marker",
            "rows": rows_count,
            "columns": columns_count,
            "inspection": details,
        })
    for tag, root in CHECKPOINTS.items():
        manifest = root / "SHA256SUMS"
        rows.append({
            "checkpoint": tag,
            "checkpoint_root": str(root),
            "relative_path": "SHA256SUMS",
            "bytes": manifest.stat().st_size,
            "sha256": sha256(manifest),
            "artifact_kind": "checksum_manifest",
            "rows": len(manifest.read_text().splitlines()),
            "columns": 2,
            "inspection": "parsed and every listed member independently verified",
        })
    return sorted(rows, key=lambda row: (row["checkpoint"], row["relative_path"]))


def load_inputs() -> dict[str, Any]:
    c7s = CHECKPOINTS["c7s"]
    c7t = CHECKPOINTS["c7t"]
    training = pd.read_parquet(c7s / "tables/train_fourb_model_development.parquet")
    projection = pd.read_parquet(c7s / "tables/train_primary_physical_projection.parquet")
    direct = pd.read_parquet(c7s / "tables/train_direct_qcd_secondary_projection.parquet")
    training_scores = pd.read_parquet(c7t / "predictions/train_fourb_oof_scores.parquet")
    projection_scores = pd.read_parquet(c7t / "predictions/train_primary_projection_oof_scores.parquet")
    for label, frame in (("training", training), ("projection", projection), ("direct", direct)):
        require(set(frame["registry_final_split"].astype(str)) == {"train"}, f"{label} is not train-only")
        require_group_fold_integrity(frame)
    require_prediction_alignment(training, training_scores, "training")
    require_prediction_alignment(projection, projection_scores, "projection")
    require_prediction_alignment(training, direct, "direct-to-training")
    for column in [f"{model}_score" for model in MODEL_ORDER]:
        require_columns(training_scores, [column], "training scores")
        require_columns(projection_scores, [column], "projection scores")
        require(np.isfinite(training_scores[column]).all(), f"nonfinite train score: {column}")
        require(np.isfinite(projection_scores[column]).all(), f"nonfinite projection score: {column}")
    return {
        "training": training,
        "projection": projection,
        "direct": direct,
        "training_scores": training_scores,
        "projection_scores": projection_scores,
        "metrics": pd.read_csv(c7t / "baseline_metrics.tsv", sep="\t"),
        "earlier": pd.read_csv(CHECKPOINTS["c7u"] / "earlier_snapshot_metric_comparison.tsv", sep="\t"),
        "population": pd.read_csv(CHECKPOINTS["c7u"] / "train_population_and_weight_comparison.tsv", sep="\t"),
        "composition": pd.read_csv(CHECKPOINTS["c7u"] / "selected_process_composition.tsv", sep="\t"),
        "factor": pd.read_csv(CHECKPOINTS["c7r"] / "transfer_factor_registry.tsv", sep="\t"),
        "closure": pd.read_csv(CHECKPOINTS["c7r"] / "transfer_closure.tsv", sep="\t"),
        "systematic": pd.read_csv(CHECKPOINTS["c7r"] / "systematic_registry.tsv", sep="\t"),
        "members": pd.read_csv(c7s / "member_fold_registry.tsv", sep="\t"),
    }


def factor_contract(inputs: dict[str, Any]) -> tuple[list[float], float]:
    factor = inputs["factor"]
    inclusive = factor[factor["mhh_category"] == "inclusive_mhh"]
    nominal = inclusive[inclusive["factor_scheme"] == "cms_cr_nominal"].iloc[0]
    values = [float(nominal["transfer_factor"])] + [
        float(value) for value in inclusive["transfer_factor"] if float(value) != float(nominal["transfer_factor"])
    ]
    require(len(values) == 3, "inclusive transfer-factor contract drift")
    return values, float(nominal["transfer_factor_relative_statistical_uncertainty"])


def scan_rows(inputs: dict[str, Any], *, points: int) -> list[dict[str, Any]]:
    training = inputs["training"]
    projection = inputs["projection"]
    direct = inputs["direct"]
    train_scores = inputs["training_scores"]
    projection_scores = inputs["projection_scores"]
    factors, factor_stat = factor_contract(inputs)
    signal_train = training["registry_training_target"].to_numpy() == 1
    train_weight = training["development_hierarchical_weight"].to_numpy(dtype=float)
    signal_projection = projection["registry_training_target"].to_numpy() == 1
    transferred = projection["analysis_population_role"].to_numpy() == "primary_transferred_multijet_template"
    direct_qcd = direct["population_kind"].to_numpy() == "hard_qcd"
    phys = projection["primary_projection_physical_weight_inclusive"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    efficiencies = np.linspace(0.10, 1.0, points)
    for model in MODEL_ORDER:
        tr = train_scores[f"{model}_score"].to_numpy(dtype=float)
        pr = projection_scores[f"{model}_score"].to_numpy(dtype=float)
        order = np.argsort(-tr[signal_train], kind="mergesort")
        ordered_scores = tr[signal_train][order]
        ordered_weights = train_weight[signal_train][order]
        cumulative = np.cumsum(ordered_weights) / ordered_weights.sum()
        for target in efficiencies:
            index = min(int(np.searchsorted(cumulative, target, side="left")), len(ordered_scores) - 1)
            threshold = float(ordered_scores[index])
            selected = pr >= threshold
            direct_selected = tr >= threshold
            signal_yield = float(phys[selected & signal_projection].sum())
            background_yield = float(phys[selected & ~signal_projection].sum())
            qcd_base = float(projection.loc[selected & transferred, "run2_candidate_physical_weight"].sum())
            qcd = qcd_base * factors[0]
            alternatives = np.asarray([qcd_base * value for value in factors])
            direct_truth = float(direct.loc[direct_selected & direct_qcd, "direct_projection_physical_weight"].sum())
            transferred_rows = int(np.count_nonzero(selected & transferred))
            transferred_sources = int(projection.loc[selected & transferred, "registry_group_id"].nunique())
            direct_rows = int(np.count_nonzero(direct_selected & direct_qcd))
            nonclosure = abs(qcd - direct_truth) / abs(direct_truth) if direct_truth else 0.0
            envelope = max(abs(alternatives.min() - qcd), abs(alternatives.max() - qcd)) / abs(qcd) if qcd else 0.0
            sigma = abs(qcd) * math.sqrt(factor_stat**2 + envelope**2 + nonclosure**2)
            background_weights = phys[selected & ~signal_projection]
            background_neff = effective_events(background_weights)
            systematic_supported = (
                transferred_rows >= 5 and transferred_sources >= 3 and direct_rows >= 1 and background_neff >= 2.0
            )
            rows.append({
                "model": model,
                "target_signal_efficiency": float(target),
                "operating_threshold": threshold,
                "signal_yield": signal_yield,
                "background_yield": background_yield,
                "background_rejection": float(1.0 - train_weight[(tr >= threshold) & ~signal_train].sum() /
                                              train_weight[~signal_train].sum()),
                "background_effective_events": background_neff,
                "selected_transferred_qcd_rows": transferred_rows,
                "selected_transferred_qcd_sources": transferred_sources,
                "selected_direct_qcd_closure_rows": direct_rows,
                "systematic_supported": systematic_supported,
                "transferred_qcd_yield": qcd,
                "direct_qcd_closure_yield": direct_truth,
                "multijet_systematic_absolute": sigma,
                "nominal_asimov_ZA": asimov_za(signal_yield, background_yield),
                "systematic_aware_asimov_ZA": (
                    asimov_za_with_uncertainty(signal_yield, background_yield, sigma)
                    if systematic_supported else float("nan")
                ),
            })
    return rows


def bootstrap_rows(inputs: dict[str, Any], *, replicates: int) -> list[dict[str, Any]]:
    projection = inputs["projection"]
    scores = inputs["projection_scores"]
    metrics = inputs["metrics"].set_index("baseline")
    members = inputs["members"]
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        if model == "cut":
            selected = projection["r_hh_125_120"].to_numpy(dtype=float) < 34.0
        else:
            selected = scores[f"{model}_score"].to_numpy(dtype=float) >= float(metrics.loc[model, "operating_threshold"])
        selected_frame = projection.loc[selected]
        yields = selected_frame.groupby("registry_group_id", sort=False)[
            "primary_projection_physical_weight_inclusive"
        ].sum().to_dict()
        strata: dict[tuple[str, str], list[float]] = {}
        for member in members.to_dict("records"):
            key = (str(member["sample_class"]), str(member["stratum"]))
            strata.setdefault(key, []).append(float(yields.get(member["transport_id"], 0.0)))
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for replica in range(replicates):
            signal = background = 0.0
            for (sample_class, _), values_list in strata.items():
                values = np.asarray(values_list, dtype=float)
                total = float(values[rng.integers(0, len(values), size=len(values))].sum())
                if sample_class == "signal":
                    signal += total
                else:
                    background += total
            rows.append({"model": model, "replica": replica, "asimov_ZA": asimov_za(signal, background)})
    return rows


def make_figures(inputs: dict[str, Any], figures: Path, paper_figures: Path, *, smoke: bool) -> list[dict[str, Any]]:
    apply_hh4b_paper_style()
    provenance: list[dict[str, Any]] = []
    training = inputs["training"]
    projection = inputs["projection"]
    direct = inputs["direct"]
    train_scores = inputs["training_scores"]
    projection_scores = inputs["projection_scores"]
    labels = training["registry_training_target"].to_numpy(dtype=int)
    dev_weights = training["development_hierarchical_weight"].to_numpy(dtype=float)

    # 01 weighted ROC and cut operating point.
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    rows: list[dict[str, Any]] = []
    for model in LEARNED_MODELS:
        fpr, tpr, thresholds = roc_curve(labels, train_scores[f"{model}_score"], sample_weight=dev_weights)
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(500, len(fpr))).astype(int))
        for index in keep:
            rows.append({"model": model, "false_positive_rate": fpr[index], "true_positive_rate": tpr[index],
                         "threshold": thresholds[index], "weighting": "development_hierarchical_weight"})
        ax.plot(fpr, tpr, lw=2, color=COLORS[model], label=f"{DISPLAY[model]} (AUC={inputs['metrics'].set_index('baseline').loc[model, 'weighted_auc']:.3f})")
    cut_selected = training["r_hh_125_120"].to_numpy(dtype=float) < 34.0
    cut_tpr = float(dev_weights[cut_selected & (labels == 1)].sum() / dev_weights[labels == 1].sum())
    cut_fpr = float(dev_weights[cut_selected & (labels == 0)].sum() / dev_weights[labels == 0].sum())
    rows.append({"model": "cut_operating_point", "false_positive_rate": cut_fpr, "true_positive_rate": cut_tpr,
                 "threshold": -34.0, "weighting": "development_hierarchical_weight"})
    ax.scatter([cut_fpr], [cut_tpr], marker="*", s=110, color=COLORS["cut"], label=DISPLAY["cut"], zorder=5)
    ax.plot([0, 1], [0, 1], ls="--", color="0.65", lw=1)
    ax.set(xlabel="Background efficiency", ylabel="Signal efficiency", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.18); ax.legend(loc="lower right"); paper_label(ax)
    save_plot(fig, figures, paper_figures, "fig01_weighted_roc", r"Weighted train-only source-group OOF ROC curves. The star is the frozen $R_{HH}(125,120)<34\,\mathrm{GeV}$ operating point.", rows, provenance)

    # 02 OOF score shapes and physical-yield projection.
    bins = np.linspace(0.0, 1.0, 31)
    rows = []
    bdt_train = train_scores["bdt_score"].to_numpy(dtype=float)
    bdt_projection = projection_scores["bdt_score"].to_numpy(dtype=float)
    for series, mask in (("signal", labels == 1), ("background", labels == 0)):
        rows += histogram_rows(bdt_train[mask], dev_weights[mask], bins, figure="fig02", panel="unit_shape",
                               series=series, normalize=True)
    roles = projection["analysis_population_role"].astype(str).to_numpy()
    projection_weight = projection["primary_projection_physical_weight_inclusive"].to_numpy(dtype=float)
    for series, role in (("signal", "fourb_signal"), ("ordinary", "fourb_ordinary_background"),
                         ("transferred", "primary_transferred_multijet_template")):
        mask = roles == role
        rows += histogram_rows(bdt_projection[mask], projection_weight[mask], bins, figure="fig02",
                               panel="physical_yield", series=series, normalize=False)
    direct_mask = direct["population_kind"].astype(str).to_numpy() == "hard_qcd"
    rows += histogram_rows(bdt_train[direct_mask], direct.loc[direct_mask, "direct_projection_physical_weight"].to_numpy(dtype=float),
                           bins, figure="fig02", panel="physical_yield", series="direct_closure", normalize=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for series, color in (("signal", COLORS["signal"]), ("background", COLORS["ordinary"])):
        x, y = step_xy(rows, series, "unit_shape"); axes[0].step(x, y, where="post", lw=2, color=color, label=series.title())
    score_labels = {"signal": "Signal", "ordinary": "Non-QCD background", "transferred": "Transferred QCD",
                    "direct_closure": r"Direct $\geq4b$ QCD closure"}
    for series, color, ls in (("signal", COLORS["signal"], "-"), ("ordinary", COLORS["ordinary"], "-"),
                              ("transferred", COLORS["transferred"], "-"), ("direct_closure", COLORS["direct"], "--")):
        x, y = step_xy(rows, series, "physical_yield"); axes[1].step(x, y, where="post", lw=2, color=color, ls=ls, label=score_labels[series])
    axes[0].set(xlabel="BDT OOF score", ylabel="Unit-normalized weighted events")
    axes[1].set(xlabel="BDT OOF score", ylabel="Train-partition physical yield", yscale="log")
    axes[1].set_ylim(bottom=max(1e-2, axes[1].get_ylim()[0]))
    for ax in axes: ax.grid(alpha=0.18); paper_label(ax, subtitle="Train-only OOF")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.subplots_adjust(bottom=0.23)
    save_plot(fig, figures, paper_figures, "fig02_oof_score_distributions", r"BDT OOF score distributions. Direct $\geq4b$ QCD is shown only as a dashed secondary closure and is not added to the primary $3b\rightarrow{\geq4b}$ transferred-QCD prediction.", rows, provenance)

    scans = scan_rows(inputs, points=13 if smoke else 91)
    # 03 efficiency versus rejection.
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for model in MODEL_ORDER:
        sub = pd.DataFrame(scans); sub = sub[sub.model == model]
        ax.plot(sub.target_signal_efficiency.to_numpy(), sub.background_rejection.to_numpy(), color=COLORS[model], lw=2, label=DISPLAY[model])
    ax.set(xlabel="Target signal efficiency", ylabel="Background rejection", xlim=(0.1, 1.0), ylim=(0, 1))
    ax.grid(alpha=0.18); ax.legend(); paper_label(ax)
    save_plot(fig, figures, paper_figures, "fig03_efficiency_background_rejection",
              "Signal efficiency versus development-weighted background rejection for the frozen train-only discriminants.", scans, provenance)

    # 04/05 ZA scans.
    for number, column, ylabel, suffix in (
        (4, "nominal_asimov_ZA", r"Nominal $Z_A$", "nominal"),
        (5, "systematic_aware_asimov_ZA", r"Systematic-aware $Z_A$", "systematic"),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        for model in MODEL_ORDER:
            sub = pd.DataFrame(scans); sub = sub[sub.model == model]
            values = sub[column].to_numpy(dtype=float)
            if suffix == "systematic":
                values = np.where(values > 0.0, values, np.nan)
            ax.plot(sub.target_signal_efficiency.to_numpy(), values, color=COLORS[model], lw=2, label=DISPLAY[model])
        ax.set(xlabel="Target signal efficiency", ylabel=ylabel, xlim=(0.1, 1.0))
        if suffix == "systematic":
            ax.set_yscale("log")
        ax.grid(alpha=0.18); ax.legend(); paper_label(ax)
        caption = ("Nominal statistical-only Asimov sensitivity scan." if suffix == "nominal" else
                   "Systematic-aware Asimov sensitivity scan on a logarithmic axis using the frozen transfer-factor statistics, normalization-region envelope, and direct-QCD nonclosure. The isolated high dense-DNN point demonstrates transfer instability rather than robust sensitivity; exact support diagnostics accompany every point in the source data.")
        save_plot(fig, figures, paper_figures, f"fig{number:02d}_{suffix}_za_scan", caption, scans, provenance)

    # 06 exact c7t member-bootstrap reconstruction.
    bootstrap = bootstrap_rows(inputs, replicates=20 if smoke else BOOTSTRAP_REPLICATES)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    boot = pd.DataFrame(bootstrap)
    ax.boxplot([boot.loc[boot.model == model, "asimov_ZA"].to_numpy() for model in MODEL_ORDER], labels=[DISPLAY[m] for m in MODEL_ORDER],
               showfliers=False, patch_artist=True, boxprops={"facecolor": "#D9EAF7"})
    ax.set_ylabel(r"Source-member bootstrap $Z_A$"); ax.grid(axis="y", alpha=0.18); paper_label(ax, subtitle=f"{20 if smoke else 1000} replicas")
    save_plot(fig, figures, paper_figures, "fig06_bootstrap_za", "Source-member bootstrap distributions of nominal train-only sensitivity. Models are not retrained in each replica.", bootstrap, provenance)

    # 07 nominal baseline comparison.
    metrics = inputs["metrics"]
    rows = metrics.to_dict("records")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    fields = [("weighted_auc", "Weighted AUC"), ("asimov_ZA", r"Nominal $Z_A$"),
              ("signal_over_background", r"$S/B$"), ("background_effective_events", r"Background $N_\mathrm{eff}$")]
    x = np.arange(len(MODEL_ORDER))
    for ax, (field, ylabel) in zip(axes.flat, fields):
        values = [float(metrics.set_index("baseline").loc[m, field]) for m in MODEL_ORDER]
        ax.bar(x, values, color=[COLORS[m] for m in MODEL_ORDER]); ax.set_xticks(x); ax.set_xticklabels([DISPLAY[m] for m in MODEL_ORDER], rotation=18, ha="right")
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.18)
        if field == "signal_over_background": ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    paper_label(axes[0, 0])
    save_plot(fig, figures, paper_figures, "fig07_nominal_baseline_comparison", "Comparison of weighted AUC, nominal sensitivity, purity, and effective background statistics at the frozen operating points.", rows, provenance)

    # 08 earlier-current ratios.
    earlier = inputs["earlier"]
    chosen = earlier[earlier.metric.isin(["weighted_auc", "asimov_ZA", "bootstrap_median_ZA", "background_effective_events"])].copy()
    chosen["label"] = chosen.baseline + ": " + chosen.metric
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    y = np.arange(len(chosen)); ax.barh(y, chosen.relative_change.to_numpy(), color=np.where(chosen.relative_change.to_numpy() >= 0, "#009E73", "#D55E00"))
    ax.axvline(0, color="0.3", lw=1); ax.set_yticks(y); ax.set_yticklabels(chosen.label.str.replace("_", " "))
    ax.set_xlabel("Relative change from earlier numeric snapshot"); ax.grid(axis="x", alpha=0.18); paper_label(ax, subtitle="Snapshot comparison")
    save_plot(fig, figures, paper_figures, "fig08_earlier_current_metrics", "Relative changes between the earlier approximate numeric snapshot and the authoritative re-established train-only results.", chosen.to_dict("records"), provenance)

    # 09 population change.
    population = inputs["population"]
    chosen = population[(population.category == "population") & population.quantity.isin(["signal_rows", "non_qcd_background_rows", "direct_qcd_fourb_rows"])].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.0)); x = np.arange(len(chosen)); width = 0.36
    ax.bar(x - width/2, chosen.earlier_frozen_train_population.to_numpy(), width, label="Earlier frozen", color="#999999")
    ax.bar(x + width/2, chosen.current_441_source_population.to_numpy(), width, label="Authoritative 441-source", color="#0072B2")
    population_labels = {"signal_rows": "Signal", "non_qcd_background_rows": "Non-QCD background",
                         "direct_qcd_fourb_rows": r"Direct $\geq4b$ QCD closure"}
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([population_labels[value] for value in chosen.quantity], rotation=15, ha="right")
    ax.set_ylabel("Candidate rows"); ax.legend(); ax.grid(axis="y", alpha=0.18); paper_label(ax, subtitle="Population audit")
    save_plot(fig, figures, paper_figures, "fig09_population_change", r"Earlier versus authoritative candidate populations. The material population change is confined to direct $\geq4b$ QCD closure.", chosen.to_dict("records"), provenance)

    # 10 selected physical process composition.
    comp = inputs["composition"].copy()
    comp["display_group"] = np.where(comp.population_kind == "hard_qcd", "Transferred QCD", comp.process_or_mode)
    totals = comp.groupby("display_group").physical_yield.sum().sort_values(ascending=False)
    keep = set(totals.head(7).index)
    comp["display_group"] = np.where(comp.display_group.isin(keep), comp.display_group, "Other")
    pivot = comp.pivot_table(index="baseline", columns="display_group", values="physical_yield", aggfunc="sum", fill_value=0).reindex(MODEL_ORDER)
    order = list(pivot.sum().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(8.2, 5.4)); bottom = np.zeros(len(pivot))
    palette = ["#CC79A7", "#0072B2", "#E69F00", "#009E73", "#56B4E9", "#D55E00", "#F0E442", "#999999"]
    for color, column in zip(palette, order):
        values = pivot[column].to_numpy(); ax.bar(np.arange(len(pivot)), values, bottom=bottom, label=column.replace("_", " "), color=color); bottom += values
    ax.set_yscale("log"); ax.set_xticks(np.arange(len(pivot))); ax.set_xticklabels([DISPLAY[m] for m in pivot.index])
    ax.set_ylabel("Selected physical background yield"); ax.legend(ncol=2, fontsize=8); ax.grid(axis="y", alpha=0.18); paper_label(ax)
    save_plot(fig, figures, paper_figures, "fig10_selected_process_composition", "Selected background composition at each frozen operating point. Transferred QCD is primary; direct QCD is excluded from these stacks.", comp.to_dict("records"), provenance)

    # 11 transfer factors.
    factor = inputs["factor"].copy()
    fig, ax = plt.subplots(figsize=(8.0, 5.2)); schemes = list(factor.factor_scheme.unique()); categories = ["inclusive_mhh", "low_mhh", "high_mhh"]
    x = np.arange(len(categories)); width = 0.22
    for i, scheme in enumerate(schemes):
        sub = factor.set_index(["mhh_category", "factor_scheme"])
        values = [sub.loc[(cat, scheme), "transfer_factor"] for cat in categories]
        errors = [sub.loc[(cat, scheme), "transfer_factor_statistical_uncertainty"] for cat in categories]
        ax.errorbar(x + (i-1)*width, values, yerr=errors, fmt="o", capsize=3, label=PAPER_LABELS[scheme])
    ax.set_xticks(x); ax.set_xticklabels(["Inclusive", r"Low $m_{HH}$", r"High $m_{HH}$"]); ax.set_ylabel(r"Exactly $3b$ to $\geq4b$ transfer factor")
    ax.set_yscale("log"); ax.grid(axis="y", alpha=0.18); ax.legend(fontsize=8); paper_label(ax, subtitle="Frozen train transfer")
    save_plot(fig, figures, paper_figures, "fig11_transfer_factors", r"Frozen exactly-$3b$ to $\geq4b$ QCD transfer factors by normalization region and $m_{HH}$ category. Error bars show propagated statistical uncertainty.", factor.to_dict("records"), provenance)

    # 12 closure.
    closure = inputs["closure"]; chosen = closure[(closure.mhh_category == "inclusive_mhh")].copy()
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    x = np.arange(len(chosen)); axes[0].bar(x-0.18, chosen.transferred_qcd_prediction.to_numpy(), 0.36, label="Transferred prediction", color=COLORS["transferred"])
    axes[0].bar(x+0.18, chosen.direct_fourb_qcd_truth.to_numpy(), 0.36, label=r"Direct $\geq4b$ QCD closure", color="#777777")
    axes[0].set_yscale("log"); axes[0].set_ylabel("QCD physical yield"); axes[0].legend(); axes[0].grid(axis="y", alpha=0.18); paper_label(axes[0], subtitle="Direct QCD is secondary")
    axes[1].axhline(1, color="0.3", lw=1); axes[1].plot(x, chosen.prediction_over_truth.to_numpy(), "o", color="#0072B2")
    region_labels = ["Baseline", r"SR $<30$", r"CR $30$--$55$", r"Outside $\geq55$", r"SR $<34$", r"SR $<31.5$", r"SR $<35.5$"]
    require(len(region_labels) == len(chosen), "inclusive transfer-closure region count drift")
    axes[1].set_ylabel("Pred./truth"); axes[1].set_xticks(x); axes[1].set_xticklabels(region_labels, rotation=15, ha="right", fontsize=8); axes[1].grid(axis="y", alpha=0.18)
    save_plot(fig, figures, paper_figures, "fig12_transfer_closure", r"Frozen $3b\rightarrow{\geq4b}$ transferred-QCD prediction compared with the secondary direct $\geq4b$ QCD closure. The lower panel is prediction divided by direct closure truth.", chosen.to_dict("records"), provenance)

    # 13 R_HH population shapes.
    q3 = pd.read_parquet(CHECKPOINTS["c7q"] / "tables/train_exactly3b_promoted_run2_physical.parquet",
                         columns=["r_hh_125_120", "sample_class", "population_kind", "run2_candidate_physical_weight"])
    bins_rhh = np.geomspace(0.25, 2000.0, 61); rows = []
    definitions = [
        ("fourb_signal", projection["analysis_population_role"].to_numpy() == "fourb_signal", projection, "primary_projection_physical_weight_inclusive"),
        ("fourb_ordinary", projection["analysis_population_role"].to_numpy() == "fourb_ordinary_background", projection, "primary_projection_physical_weight_inclusive"),
        ("threeb_qcd_template", q3["population_kind"].to_numpy() == "hard_qcd", q3, "run2_candidate_physical_weight"),
        ("fourb_direct_qcd", direct["population_kind"].to_numpy() == "hard_qcd", direct, "direct_projection_physical_weight"),
    ]
    for series, mask, frame, weight_col in definitions:
        rows += histogram_rows(frame.loc[mask, "r_hh_125_120"].to_numpy(dtype=float), frame.loc[mask, weight_col].to_numpy(dtype=float), bins_rhh,
                               figure="fig13", panel="unit_shape", series=series, normalize=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    rhh_labels = {"fourb_signal": r"Signal $\geq4b$", "fourb_ordinary": "Non-QCD background",
                  "threeb_qcd_template": r"Exactly $3b$ QCD template",
                  "fourb_direct_qcd": r"Direct $\geq4b$ QCD closure"}
    for series, color, ls in (("fourb_signal", COLORS["signal"], "-"), ("fourb_ordinary", COLORS["ordinary"], "-"),
                              ("threeb_qcd_template", COLORS["transferred"], "-"), ("fourb_direct_qcd", COLORS["direct"], "--")):
        xh, yh = step_xy(rows, series, "unit_shape"); ax.step(xh, yh, where="post", lw=2, color=color, ls=ls, label=rhh_labels[series])
    ax.axvline(34, color="0.35", ls=":", label="Frozen cut"); ax.set(xlabel=r"$R_{HH}(125,120)$ [GeV]", ylabel="Unit-normalized weighted events", xlim=(0.25,2000), xscale="log")
    ax.legend(fontsize=8); ax.grid(alpha=0.18); paper_label(ax)
    save_plot(fig, figures, paper_figures, "fig13_rhh_distributions", r"Full-range weighted $R_{HH}(125,120)$ shapes for signal, non-QCD background, the exactly-$3b$ QCD template, and secondary direct $\geq4b$ QCD closure. Logarithmic binning includes every simulated event.", rows, provenance)

    # 14 mass planes.
    mass_bins = np.geomspace(15.0, 2000.0, 41); rows = []
    plane_defs = [("signal", roles == "fourb_signal"), ("ordinary_background", roles == "fourb_ordinary_background"),
                  ("transferred_qcd", roles == "primary_transferred_multijet_template")]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), sharex=True, sharey=True)
    for ax, (series, mask) in zip(axes, plane_defs):
        hist, xedges, yedges = np.histogram2d(projection.loc[mask, "mbb1"], projection.loc[mask, "mbb2"], bins=(mass_bins, mass_bins), weights=projection_weight[mask])
        total = hist.sum(); require(total > 0, f"empty mass plane: {series}"); hist /= total
        for i in range(len(mass_bins)-1):
            for j in range(len(mass_bins)-1):
                rows.append({"series": series, "mbb1_low_GeV": xedges[i], "mbb1_high_GeV": xedges[i+1],
                             "mbb2_low_GeV": yedges[j], "mbb2_high_GeV": yedges[j+1], "unit_normalized_weight": hist[i,j]})
        mesh = ax.pcolormesh(xedges, yedges, hist.T, shading="auto", cmap="cividis"); fig.colorbar(mesh, ax=ax, label="Weighted fraction/bin")
        title = {"signal": "Signal", "ordinary_background": "Ordinary background", "transferred_qcd": "Transferred QCD"}[series]
        ax.set_title(title); ax.set_xlabel(r"$m(H_1)$ [GeV]"); ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.08)
    axes[0].set_ylabel(r"$m(H_2)$ [GeV]")
    fig.text(0.01, 1.01, "Delphes simulation", ha="left", va="bottom", fontsize=12, fontweight="bold")
    fig.text(0.99, 1.01, "Train-only source-group OOF", ha="right", va="bottom", fontsize=9)
    save_plot(fig, figures, paper_figures, "fig14_higgs_mass_planes", r"Full-range reconstructed Higgs-candidate mass planes for signal, ordinary background, and the primary transferred-QCD template. Each panel is unit normalized with logarithmic mass binning and includes every observed event.", rows, provenance)

    # 15 mHH and promoted-jet quality.
    bins_mhh = np.geomspace(100.0, 6000.0, 55); bins_btag = np.linspace(0, 1, 41); bins_pt = np.geomspace(30.0, 2000.0, 49); rows = []
    for series, mask in (("signal", roles == "fourb_signal"), ("ordinary", roles == "fourb_ordinary_background"), ("transferred", roles == "primary_transferred_multijet_template")):
        rows += histogram_rows(projection.loc[mask, "mhh"].to_numpy(dtype=float), projection_weight[mask], bins_mhh,
                               figure="fig15", panel="mhh", series=series, normalize=True)
    promoted = roles == "primary_transferred_multijet_template"
    rows += histogram_rows(projection.loc[promoted, "promoted_jet_btag"].to_numpy(dtype=float), projection_weight[promoted], bins_btag,
                           figure="fig15", panel="promoted_btag", series="transferred", normalize=True)
    rows += histogram_rows(projection.loc[promoted, "promoted_jet_pt"].to_numpy(dtype=float), projection_weight[promoted], bins_pt,
                           figure="fig15", panel="promoted_pt", series="transferred", normalize=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for series, color in (("signal", COLORS["signal"]), ("ordinary", COLORS["ordinary"]), ("transferred", COLORS["transferred"])):
        xh,yh=step_xy(rows,series,"mhh"); axes[0].step(xh,yh,where="post",lw=2,color=color,label=series.title())
    xh,yh=step_xy(rows,"transferred","promoted_btag"); axes[1].step(xh,yh,where="post",lw=2,color=COLORS["transferred"])
    xh,yh=step_xy(rows,"transferred","promoted_pt"); axes[2].step(xh,yh,where="post",lw=2,color=COLORS["transferred"])
    axes[0].set(xlabel=r"$m_{HH}$ [GeV]", ylabel="Unit-normalized weighted events", xlim=(100,6000), xscale="log"); axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Promoted-jet b-tag discriminator", ylabel="Unit-normalized transferred QCD", xlim=(0,1))
    axes[2].set(xlabel="Promoted-jet $p_T$ [GeV]", ylabel="Unit-normalized transferred QCD", xlim=(30,2000), xscale="log")
    for ax in axes: ax.grid(alpha=0.18)
    fig.text(0.01, 1.01, "Delphes simulation", ha="left", va="bottom", fontsize=12, fontweight="bold")
    fig.text(0.99, 1.01, "Train-only source-group OOF", ha="right", va="bottom", fontsize=9)
    save_plot(fig, figures, paper_figures, "fig15_mhh_promoted_jet_quality", r"Full-range frozen $m_{HH}$ shapes and promoted-jet quality diagnostics supporting the lower-b-tag multijet transfer. The promoted-jet b-tag input is identically zero for all 576 transferred-template rows; this degenerate panel is retained as an explicit data-quality diagnostic. Logarithmic kinematic binning includes every observed event.", rows, provenance)

    # 16 unweighted ROC appendix diagnostic.
    fig, ax = plt.subplots(figsize=(6.4, 5.2)); rows = []
    for model in LEARNED_MODELS:
        fpr, tpr, thresholds = roc_curve(labels, train_scores[f"{model}_score"])
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(500, len(fpr))).astype(int))
        for index in keep:
            rows.append({"model": model, "false_positive_rate": fpr[index], "true_positive_rate": tpr[index],
                         "threshold": thresholds[index], "weighting": "unweighted_appendix_diagnostic"})
        auc = float(inputs["metrics"].set_index("baseline").loc[model, "unweighted_auc"])
        ax.plot(fpr, tpr, lw=2, color=COLORS[model], label=f"{DISPLAY[model]} (AUC={auc:.3f})")
    cut_tpr = float(np.mean(cut_selected[labels == 1])); cut_fpr = float(np.mean(cut_selected[labels == 0]))
    rows.append({"model": "cut_operating_point", "false_positive_rate": cut_fpr, "true_positive_rate": cut_tpr,
                 "threshold": -34.0, "weighting": "unweighted_appendix_diagnostic"})
    ax.scatter([cut_fpr], [cut_tpr], marker="*", s=110, color=COLORS["cut"], label=DISPLAY["cut"], zorder=5)
    ax.plot([0, 1], [0, 1], ls="--", color="0.65", lw=1)
    ax.set(xlabel="Background efficiency", ylabel="Signal efficiency", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.18); ax.legend(loc="lower right"); paper_label(ax, subtitle="Unweighted appendix diagnostic")
    save_plot(fig, figures, paper_figures, "fig16_unweighted_roc_appendix",
              "Unweighted train-only source-group OOF ROC curves, provided only as an appendix diagnostic. The weighted ROC is authoritative.",
              rows, provenance)

    # 17 selected multijet fraction, nonclosure, and frozen nuisance.
    rows = []
    for row in metrics.itertuples():
        rows.append({
            "model": row.baseline,
            "transferred_qcd_fraction_of_background": row.transferred_qcd_prediction / row.background_yield,
            "direct_qcd_relative_nonclosure": row.score_domain_qcd_relative_nonclosure,
            "multijet_systematic_relative_to_total_background": row.multijet_systematic_relative_to_total_background,
            "projection_status": "primary transfer; direct >=4b QCD is secondary closure only",
        })
    diagnostic = pd.DataFrame(rows).set_index("model").reindex(MODEL_ORDER)
    x = np.arange(len(MODEL_ORDER)); width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
    axes[0].bar(x, 100.0 * diagnostic.transferred_qcd_fraction_of_background.to_numpy(),
                color=COLORS["transferred"])
    axes[0].set_ylabel("Transferred-QCD fraction of background [%]")
    axes[0].set_ylim(0, 100)
    axes[1].bar(x - width / 2, 100.0 * diagnostic.direct_qcd_relative_nonclosure.to_numpy(), width,
                color="#E69F00", label="Direct-QCD nonclosure")
    axes[1].bar(x + width / 2, 100.0 * diagnostic.multijet_systematic_relative_to_total_background.to_numpy(), width,
                color="#CC79A7", label="Frozen multijet nuisance / background")
    axes[1].set_ylabel("Relative size [%]")
    axes[1].legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.01))
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([DISPLAY[model] for model in MODEL_ORDER], rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.18)
    paper_label(axes[0], subtitle="Train-only frozen transfer")
    save_plot(fig, figures, paper_figures, "fig17_multijet_systematic_nonclosure",
              r"Selected transferred-QCD fraction, score-domain direct-$\geq4b$ QCD nonclosure, and frozen multijet nuisance. Direct $\geq4b$ QCD is secondary closure only.",
              rows, provenance)
    return provenance


def latex_escape(value: Any) -> str:
    text = str(value)
    if "$" in text or "\\" in text:
        return text
    for old, new in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def write_latex_table(path: Path, caption: str, label: str, headers: list[str], rows: Iterable[Iterable[Any]], *, align: str | None = None) -> None:
    alignment = align or "l" * len(headers)
    require(len(alignment) == len(headers), f"LaTeX table alignment drift: {path}")
    lines = [r"\begin{table}[tb]", r"  \centering", r"  \footnotesize", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
             f"  \\begin{{tabular}}{{{alignment}}}", r"    \toprule", "    " + " & ".join(headers) + r" \\", r"    \midrule"]
    for row in rows:
        lines.append("    " + " & ".join(latex_escape(value) for value in row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def scientific_latex(value: float, digits: int = 2) -> str:
    if value == 0.0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(value))))
    coefficient = value / (10.0 ** exponent)
    return rf"${coefficient:.{digits}f}\times10^{{{exponent}}}$"


def make_paper_material(inputs: dict[str, Any], paper: Path, provenance: list[dict[str, Any]], assets: list[dict[str, Any]], head: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sections = paper / "sections"; tables = paper / "tables"; captions = paper / "captions"
    sections.mkdir(); tables.mkdir(); captions.mkdir()
    metrics = inputs["metrics"]
    c7q_summary = json.loads((CHECKPOINTS["c7q"] / "summary.json").read_text())
    c7r_summary = json.loads((CHECKPOINTS["c7r"] / "summary.json").read_text())
    c7s_summary = json.loads((CHECKPOINTS["c7s"] / "summary.json").read_text())

    table_specs: list[tuple[str, str, str, list[str], list[list[Any]], str]] = []
    table_specs.append(("tab01_train_source_inventory.tex", "Frozen train-source and candidate inventory.", "tab:train-inventory",
                        ["Quantity", "Value"], [["Train source members", 441], [r"Exactly $3b$ candidates", c7q_summary["threeb_rows"]],
                        [r"$\geq4b$ candidates", c7q_summary["fourb_rows"]], ["Composite identity overlap", c7q_summary["threeb_fourb_composite_identity_overlap"]]], "main_text"))
    table_specs.append(("tab02_fourb_population.tex", r"Authoritative $\geq4b$ development population.", "tab:fourb-population",
                        ["Population", "Rows"], [["Signal", 9337], ["Non-QCD background", 21312], [r"Direct $\geq4b$ QCD closure", 56], ["Total", 30705]], "main_text"))
    factor = inputs["factor"]
    table_specs.append(("tab03_transfer_factors.tex", "Frozen multijet transfer factors.", "tab:transfer-factors",
                        [r"$m_{HH}$ category", "Scheme", "Factor", "Stat. rel. unc."],
                        [[PAPER_LABELS[r.mhh_category], PAPER_LABELS[r.factor_scheme], scientific_latex(r.transfer_factor), f"{r.transfer_factor_relative_statistical_uncertainty:.3f}"] for r in factor.itertuples()], "main_text"))
    systematic = inputs["systematic"]
    table_specs.append(("tab04_multijet_nuisances.tex", "Frozen multijet systematic components.", "tab:multijet-nuisances",
                        ["Nuisance", "Scope", "Relative down", "Relative up"],
                        [[PAPER_LABELS[r.nuisance], PAPER_LABELS[r.scope], f"{r.relative_down:.3f}", f"{r.relative_up:.3f}"] for r in systematic.itertuples()], "main_text"))
    table_specs.append(("tab05_baseline_metrics.tex", "Train-only source-group OOF baseline metrics at frozen operating points.", "tab:baseline-metrics",
                        ["Model", "Weighted AUC", r"$Z_A$", r"Sys.-aware $Z_A$", r"$S/B$", r"Bkg. $N_{\rm eff}$"],
                        [[DISPLAY[r.baseline], f"{r.weighted_auc:.4f}", f"{r.asimov_ZA:.5f}", scientific_latex(r.systematic_aware_asimov_ZA), scientific_latex(r.signal_over_background), f"{r.background_effective_events:.1f}"] for r in metrics.itertuples()], "main_text"))
    earlier = inputs["earlier"]
    table_specs.append(("tab06_earlier_current.tex", "Earlier approximate snapshot versus current results.", "tab:earlier-current",
                        ["Model", "Metric", "Earlier", "Current", "Relative change"],
                        [[DISPLAY[r.baseline], r.metric.replace("_", " ").title(), f"{r.earlier_snapshot:.5g}", f"{r.current_reestablished:.5g}", f"{r.relative_change:.3f}"] for r in earlier.itertuples()], "appendix"))
    comp = inputs["composition"]
    grouped = comp.groupby(["baseline", "population_kind"], as_index=False).agg(selected_rows=("selected_rows", "sum"), physical_yield=("physical_yield", "sum"))
    table_specs.append(("tab07_process_composition.tex", "Selected background composition grouped by frozen population kind.", "tab:process-composition",
                        ["Model", "Population", "Rows", "Physical yield"],
                        [[DISPLAY[r.baseline], PAPER_LABELS.get(r.population_kind, str(r.population_kind).replace("_", " ").title()), int(r.selected_rows), scientific_latex(r.physical_yield)] for r in grouped.itertuples()], "appendix"))
    model_manifest = pd.read_csv(CHECKPOINTS["c7t"] / "model_artifact_manifest.tsv", sep="\t")
    table_specs.append(("tab08_model_artifacts.tex", "Frozen model artifact identities.", "tab:model-artifacts",
                        ["Model", "Fold", "File", "SHA-256 prefix"],
                        [[DISPLAY.get(r.model, str(r.model).replace("_", " ").upper()), int(r.fold), r.path, str(r.sha256)[:12]] for r in model_manifest.itertuples()], "appendix"))
    closure = inputs["closure"]
    closure = closure[closure.mhh_category == "inclusive_mhh"]
    table_specs.append(("tab09_transfer_closure.tex", r"Inclusive frozen multijet-transfer yields and secondary direct $\geq4b$ QCD closure.", "tab:transfer-closure-yields",
                        ["Target region", r"$3b$ rows", r"Direct $\geq4b$ rows", "Transferred yield", "Direct closure"],
                        [[TABLE_REGION_LABELS[r.target_region], int(r.threeb_qcd_rows), int(r.fourb_direct_qcd_rows),
                          scientific_latex(r.transferred_qcd_prediction), scientific_latex(r.direct_fourb_qcd_truth)]
                         for r in closure.itertuples()], "appendix"))
    table_specs.append(("tab09b_transfer_closure_ratios.tex", r"Inclusive frozen $3b\rightarrow{\geq4b}$ closure ratios.", "tab:transfer-closure-ratios",
                        ["Target region", "Prediction / closure", "Absolute nonclosure"],
                        [[TABLE_REGION_LABELS[r.target_region], f"{r.prediction_over_truth:.3f}", f"{r.relative_absolute_nonclosure:.3f}"]
                         for r in closure.itertuples()], "appendix"))
    table_manifest = []
    for filename, caption, label, headers, rows, placement in table_specs:
        write_latex_table(tables / filename, caption, label, headers, rows)
        table_manifest.append({"table_id": filename.split("_", 1)[0], "path": f"tables/{filename}", "caption": caption,
                               "source_checkpoint": "c7q-c7u", "generation": "machine_generated", "placement": placement})

    prominent = (
        "This is a Delphes-based simulation study evaluated only with train-source-group OOF predictions. "
        r"No validation/test result, observed data, or CMS approval is claimed. Direct $\geq4b$ QCD is secondary "
        "closure only; the primary multijet prediction is the frozen lower-b-tag transfer, whose uncertainty "
        "currently dominates the sensitivity."
    )
    section_text = {
        "analysis_strategy.tex": prominent + r" The analysis uses 441 frozen train sources and disjoint exactly $3b$ and $\geq4b$ candidate populations." + "\n",
        "event_reconstruction.tex": "Four selected candidate jets are paired geometrically into two Higgs candidates. The frozen baseline uses $R_{HH}(125,120)<34\\,\\mathrm{GeV}$.\n",
        "multijet_transfer.tex": rf"The primary multijet prediction uses the exactly $3b\rightarrow{{\geq4b}}$ QCD transfer with nominal inclusive factor {scientific_latex(c7r_summary['nominal_transfer_factor'])}. Direct $\geq4b$ QCD is shown only as secondary closure." + "\n",
        "machine_learning_methods.tex": "The BDT, dense DNN, and LBN-DNN use the same deterministic five source-group folds. Learned operating points reproduce the frozen cut's development-weighted signal efficiency.\n",
        "train_only_results.tex": "The strongest nominal baseline is the BDT, but nominal ordering must not be interpreted as a publication-model choice because the frozen multijet uncertainty reduces all systematic-aware sensitivities to approximately zero.\n",
        "systematic_limitations.tex": prominent + " The result is not a full Run-2 prediction.\n",
        "categorized_bdt.tex": "A HIG-24-015-inspired two-stage categorized BDT is specified in the accompanying adaptation plan. Its results must remain explicitly train-only nested OOF.\n",
        "spanet.tex": "Single-head and joint two-head permutation-symmetric assignment studies require unambiguous truth-matched signal labels. Background assignment loss is masked; classification loss may use all supported train events.\n",
        "conclusions_and_next_steps.tex": "The immediate publication blockers are a stable multijet estimate, validation/test authorization, full systematic treatment, and collaboration review.\n",
    }
    for filename, text in section_text.items(): (sections / filename).write_text(text)
    for source in sorted((paper / "figures").glob("*_caption.tex")):
        shutil.copy2(source, captions / source.name)
    require(len(list(captions.glob("*_caption.tex"))) == len(provenance), "paper caption export count drift")
    (paper / "references_hig_24_015.bib").write_text(
        "@techreport{CMS-PAS-HIG-24-015,\n  author = {{CMS Collaboration}},\n  title = {Search for Higgs boson pair production in final states with two photons and two bottom quarks},\n  institution = {CERN},\n  type = {CMS Physics Analysis Summary},\n  number = {CMS-PAS-HIG-24-015},\n  note = {Public PAS; bibliographic metadata to be verified before submission}\n}\n"
    )
    includes = "\n".join(f"\\input{{sections/{name}}}" for name in section_text)
    table_inputs = "\n".join(f"\\input{{tables/{spec[0]}}}" for spec in table_specs)
    (paper / "compile_fragments.tex").write_text(
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\usepackage{graphicx}\n\\usepackage{amsmath}\n\\usepackage{booktabs}\n"
        "\\begin{document}\n\\section*{HH to four-b ML train-only fragments}\n" + includes + "\n" + table_inputs + "\n\\end{document}\n"
    )

    claim_specs = [
        ("CLAIM-001", "441", "source members", "c7q", "summary.json", "source_rows", "main_text", "nominal", "train_population_inventory"),
        ("CLAIM-002", str(c7q_summary["threeb_rows"]), "candidate rows", "c7q", "summary.json", "threeb_rows", "main_text", "nominal", "primary_lower_b_tag_population"),
        ("CLAIM-003", str(c7q_summary["fourb_rows"]), "candidate rows", "c7q", "summary.json", "fourb_rows", "main_text", "nominal", "four_b_tag_development_population"),
        ("CLAIM-004", str(c7r_summary["nominal_transfer_factor"]), "dimensionless", "c7r", "summary.json", "nominal_transfer_factor", "main_text", "nominal", "primary_transferred_qcd_projection"),
        ("CLAIM-005", str(c7r_summary["nominal_transfer_factor_relative_statistical_uncertainty"]), "relative", "c7r", "summary.json", "nominal_transfer_factor_relative_statistical_uncertainty", "main_text", "systematic-aware", "primary_transferred_qcd_projection"),
    ]
    for row in metrics.itertuples():
        for field, unit, placement, status, projection_status in (
            ("weighted_auc", "dimensionless", "main_text", "nominal", "train_oof_development"),
            ("unweighted_auc", "dimensionless", "appendix", "nominal", "train_oof_development"),
            ("asimov_ZA", "sigma", "main_text", "nominal", "primary_transferred_qcd_projection"),
            ("systematic_aware_asimov_ZA", "sigma", "main_text", "systematic-aware", "primary_projection_with_secondary_direct_qcd_closure"),
            ("signal_over_background", "dimensionless", "main_text", "nominal", "primary_transferred_qcd_projection"),
            ("background_effective_events", "effective events", "appendix", "nominal", "primary_transferred_qcd_projection"),
            ("bootstrap_median_ZA", "sigma", "main_text", "nominal", "primary_projection_source_bootstrap"),
            ("bootstrap_p16_ZA", "sigma", "main_text", "nominal", "primary_projection_source_bootstrap"),
            ("bootstrap_p84_ZA", "sigma", "main_text", "nominal", "primary_projection_source_bootstrap"),
            ("score_domain_qcd_relative_nonclosure", "relative", "diagnostic", "systematic-aware", "secondary_direct_qcd_closure"),
            ("multijet_systematic_relative_to_total_background", "relative", "main_text", "systematic-aware", "primary_projection_with_secondary_direct_qcd_closure"),
        ):
            claim_specs.append((f"CLAIM-{len(claim_specs)+1:03d}", str(getattr(row, field)), unit, "c7t", "baseline_metrics.tsv",
                                f"row: baseline={row.baseline}; column: {field}", placement, status, projection_status))
    asset_lookup = {(row["checkpoint"], row["relative_path"]): row for row in assets}
    claims = []
    for claim_id, value, unit, checkpoint, source_file, field, placement, metric_status, projection_status in claim_specs:
        source = asset_lookup[(checkpoint, source_file)]
        claims.append({"claim_id": claim_id, "value": value, "units": unit, "source_checkpoint": checkpoint,
                       "source_file": str(Path(source["checkpoint_root"]) / source_file), "field_row_column": field,
                       "source_file_sha256": source["sha256"], "generating_script": "scripts/analysis/publish_pn_c7v_results_and_figures.py",
                       "current_commit": head, "data_status": "train-only",
                       "metric_status": metric_status, "projection_or_closure_status": projection_status,
                       "publication_placement": placement})
    write_tsv(paper / "numerical_claim_registry.tsv", claims)
    write_tsv(paper / "paper_table_manifest.tsv", table_manifest)
    write_tsv(paper / "paper_figure_manifest.tsv", provenance)
    write_tsv(paper / "results_asset_manifest.tsv", assets)
    (paper / "README.md").write_text(
        "# HH to four-b train-only paper assets\n\n" + prominent +
        "\n\nAll tables are machine generated from checksum-verified c7q--c7u inputs. Figure PDFs are authoritative; PNGs are review companions. Exact plotting inputs are in `source_data/`, caption fragments are in `captions/`, and `compile_fragments.tex` is a lightweight syntax wrapper, not a JHEP template.\n"
    )
    (paper / "limitations_and_publication_status.md").write_text(
        "# Limitations and publication status\n\n" + prominent +
        "\n\nThe direct-QCD sample is statistically sparse, score-domain nonclosure is large, and the frozen multijet nuisance is roughly 2.24--2.27 times the selected total background. Validation/test opening, a full systematic model, data treatment, and collaboration approval remain required.\n"
    )
    (paper / "RESULTS_INDEX.md").write_text(
        "# Authoritative results index\n\n"
        f"Generation base commit: `{head}`.\n\n"
        "- Frozen inputs: c7q--c7u under `/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/`.\n"
        "- Full inspected asset map: `results_asset_manifest.tsv`.\n"
        "- Numerical claims: `numerical_claim_registry.tsv`.\n"
        "- Paper tables: `tables/` and `paper_table_manifest.tsv`.\n"
        "- Paper figures: `figures/` and `paper_figure_manifest.tsv`.\n"
        "- Exact plotting data: `source_data/`; LaTeX captions: `captions/`.\n"
        "- Requested plot support and any unsupported-item reasons: `requested_figure_registry.tsv`.\n"
        "- Frozen c7t model weights and predictions remain in c7t; their exact identities are in the asset and model tables.\n"
        "- Categorized-BDT and SPA-Net sections are explicit method/status placeholders until their separate sealed checkpoints exist.\n"
    )
    return table_manifest, claims


def run(output: Path, paper_output: Path, *, smoke: bool) -> None:
    require(output != paper_output, "checkpoint and paper outputs must differ")
    require_fresh_directory(paper_output)
    staging = prepare_staging(output)
    try:
        evidence = verify_all_checkpoints()
        assets = inspect_assets(evidence)
        head = git_head()
        inputs = load_inputs()
        figures = staging / "figures"; figures.mkdir()
        paper_output.mkdir(); paper_figures = paper_output / "figures"; paper_figures.mkdir()
        paper_source_data = paper_output / "source_data"; paper_source_data.mkdir()
        provenance = make_figures(inputs, figures, paper_figures, smoke=smoke)
        for source in sorted(figures.glob("*.tsv")):
            shutil.copy2(source, paper_source_data / source.name)
        require(len(list(paper_source_data.glob("*.tsv"))) == len(provenance), "paper source-data export count drift")
        figure_requests = requested_figure_registry(provenance)
        tables, claims = make_paper_material(inputs, paper_output, provenance, assets, head)
        write_tsv(paper_output / "requested_figure_registry.tsv", figure_requests)
        write_tsv(staging / "results_asset_manifest.tsv", assets)
        write_tsv(staging / "paper_figure_manifest.tsv", provenance)
        write_tsv(staging / "paper_table_manifest.tsv", tables)
        write_tsv(staging / "numerical_claim_registry.tsv", claims)
        write_tsv(staging / "source_evidence_manifest.tsv", evidence)
        write_tsv(staging / "requested_figure_registry.tsv", figure_requests)
        (staging / "README.md").write_text(
            "# pn-c7v publication inventory and baseline figures\n\nChecksum-verified, train-only publication inventory and reproducible plot package for c7q--c7u.\n"
        )
        (staging / "RUN_CONTRACT.txt").write_text(
            f"command={sys.executable} {' '.join(sys.argv)}\nbase_commit={head}\nsplit=train\nvalidation_opened=0\ntest_opened=0\nsmoke={str(smoke).lower()}\n"
        )
        summary = {
            "schema_version": 1,
            "status": "pn_c7v_train_publication_inventory_and_baseline_figures_complete",
            "split": "train",
            "smoke": smoke,
            "input_checkpoints": {tag: str(path) for tag, path in CHECKPOINTS.items()},
            "inspected_input_artifacts": len(assets),
            "figures": len(provenance),
            "requested_figures": len(figure_requests),
            "supported_requested_figures": sum(row["status"] == "supported" for row in figure_requests),
            "unsupported_requested_figures": sum(row["status"] != "supported" for row in figure_requests),
            "paper_tables": len(tables),
            "numerical_claims": len(claims),
            "validation_payload_files_opened": 0,
            "test_or_evaluation_payload_files_opened": 0,
            "full_run2_prediction": False,
            "cms_approval_claimed": False,
            "paper_export": str(paper_output),
        }
        write_json(staging / "summary.json", summary)
        write_tsv(staging / "artifact_manifest.tsv", artifact_rows(staging, exclude={"artifact_manifest.tsv", "SHA256SUMS", "COMPLETE"}))
        manifest_sha = seal_checkpoint(staging, output, "pn_c7v publication inventory and baseline figures complete")
        print(json.dumps({"output": str(output), "paper_output": str(paper_output), "manifest_sha256": manifest_sha,
                          "figures": len(provenance), "tables": len(tables)}, sort_keys=True))
    except Exception as exc:
        failure = {"status": "failed_preserved_staging", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        if staging.exists(): write_json(staging / "FAILURE.json", failure)
        if paper_output.exists(): write_json(paper_output / "FAILURE.json", failure)
        raise


def main() -> None:
    args = parse_args()
    run(args.output.resolve(), args.paper_output.resolve(), smoke=args.smoke)


if __name__ == "__main__":
    main()
