#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import roc_curve

from hh4b_bdt_apples_to_apples_common import validate_training_weights
from run_hh4b_bdt_apples_to_apples_outer import (
    DEFAULT_AUTHORIZATION,
    DEFAULT_PLAN,
    estimator,
    feature_names,
    hierarchical_weights,
    load_registry,
    read_sources,
)
from scripts.plotting.hh4b_cms_style import (
    FEATURE_LATEX_LABELS,
    add_delphes_header,
    apply_cms_style,
    save_png_pdf,
)

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/"
    "bdt_apples_to_apples/production_audit/cluster_3799191"
)
OUT = REPO / "artifacts/hh4b_bdt_apples_to_apples/train_only_final_v1"
PROTOCOL = REPO / "configs/baselines/hh4b_bdt_apples_to_apples_v1.json"
VARIANTS = ("global_mass_plane_blind", "global_mass_aware")
LABELS = {
    "global_mass_plane_blind": "Global mass-plane-blind BDT",
    "global_mass_aware": "Global mass-aware BDT",
}
COLORS = {
    "global_mass_plane_blind": "#0072B2",
    "global_mass_aware": "#D55E00",
}
EXPECTED_OOF_SHA = {
    "global_mass_plane_blind":
        "dfee50e0a83ae3f0f3382f2e9e8205b39c47749c23b3e694c0fdce49e490c14a",
    "global_mass_aware":
        "1c17362c5cae14ab3ac980033bb3052803bc81f9b121b183390594dab0988725",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_dirs() -> None:
    for name in (
        "figures/png", "figures/pdf", "figure_data",
        "models", "tables", "manifests",
    ):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    fig.tight_layout()
    png, pdf = save_png_pdf(fig, OUT / "figures/png" / name)
    target = OUT / "figures/pdf" / pdf.name
    pdf.replace(target)
    require(png.is_file() and target.is_file(), f"figure missing: {name}")


def load_inputs():
    frames = {}
    for variant in VARIANTS:
        path = SOURCE / f"pooled_oof_{variant}.parquet"
        require(sha256(path) == EXPECTED_OOF_SHA[variant],
                f"OOF SHA changed: {variant}")
        frame = pd.read_parquet(path)
        require(len(frame) == 1_042_397, f"row count changed: {variant}")
        require(frame.event_uid.nunique() == 1_042_397,
                f"UID coverage changed: {variant}")
        require(set(frame.source_fold) == set(range(5)),
                f"fold coverage changed: {variant}")
        frames[variant] = frame

    require(
        set(frames[VARIANTS[0]].event_uid.astype(str))
        == set(frames[VARIANTS[1]].event_uid.astype(str)),
        "variant event universes differ",
    )
    metrics = pd.read_csv(SOURCE / "metrics.tsv", sep="\t")
    return frames, metrics


def collect_fold_evidence(metrics: pd.DataFrame):
    selection_rows, importance_parts = [], []
    for variant in VARIANTS:
        for fold in range(5):
            result = SOURCE / f"result_{fold}_{variant}" / "result"
            receipt = json.loads((result / "receipt.json").read_text())
            trials = json.loads((result / "hyperparameter_trials.json").read_text())
            freeze = json.loads((result / "selection_freeze.json").read_text())
            require(receipt["selected_candidate_id"] == "depth4",
                    "non-depth4 candidate found")
            require(receipt["validation_payloads_opened"] == 0, "validation opened")
            require(receipt["test_payloads_opened"] == 0, "test opened")
            require(freeze["outer_payloads_opened_before_freeze"] == 0,
                    "outer payload opened before freeze")
            require(len(trials) == 8, "candidate count changed")
            require(all(len(row["inner_fold_aucs"]) == 4 for row in trials),
                    "inner-fold count changed")

            metric = metrics[
                (metrics.method == variant)
                & (metrics.fold.astype(str) == str(fold))
            ].iloc[0]
            selection_rows.append({
                "variant": variant,
                "outer_fold": fold,
                "selected_candidate": receipt["selected_candidate_id"],
                "threshold": receipt["score_threshold"],
                "inner_oof_auc": trials[0]["pooled_inner_oof_weighted_roc_auc"],
                "held_outer_auc": metric.auc,
                "epsilon_B": metric.epsilon_B,
                "rejection": metric.rejection,
                "Neff_background": metric.Neff_background,
                "finite_mc_relative_uncertainty_background":
                    metric.finite_mc_rel_unc_background,
                "finite_mc_dominated": bool(metric.Neff_background < 100.0),
            })
            imp = pd.read_csv(result / "feature_importance_gain.tsv", sep="\t")
            imp["variant"] = variant
            imp["outer_fold"] = fold
            importance_parts.append(imp)

    selection = pd.DataFrame(selection_rows)
    importance = pd.concat(importance_parts, ignore_index=True)
    return selection, importance


def write_tables(metrics, selection, importance):
    tables = OUT / "tables"
    metrics.to_csv(tables / "all_metrics.tsv", sep="\t", index=False)
    selection.to_csv(tables / "fold_metrics_selection.tsv", sep="\t", index=False)
    importance.to_csv(tables / "fold_feature_importance.tsv", sep="\t", index=False)

    pooled = metrics[metrics.fold.astype(str) == "pooled"].copy()
    display = {
        "R_HH_125_125_lt34": "R_HH(125,125)<34",
        **LABELS,
    }
    pooled["display_method"] = pooled.method.map(display)
    columns = [
        "display_method", "auc", "epsilon_S", "epsilon_B", "rejection",
        "S", "B", "S_over_B", "ZA_stat", "Neff_background",
        "finite_mc_rel_unc_background",
    ]
    pooled[columns].to_csv(
        tables / "primary_comparison.tsv", sep="\t", index=False
    )

    cut = pooled[pooled.method == "R_HH_125_125_lt34"].iloc[0]
    blind = pooled[pooled.method == VARIANTS[0]].iloc[0]
    aware = pooled[pooled.method == VARIANTS[1]].iloc[0]

    def compare(first, second, name):
        return {
            "comparison": name,
            "delta_auc":
                first.auc - second.auc
                if pd.notna(first.auc) and pd.notna(second.auc)
                else np.nan,
            "delta_epsilon_B": first.epsilon_B - second.epsilon_B,
            "rejection_ratio": first.rejection / second.rejection,
            "S_over_B_ratio": first.S_over_B / second.S_over_B,
            "delta_ZA": first.ZA_stat - second.ZA_stat,
            "relative_ZA_change": first.ZA_stat / second.ZA_stat - 1.0,
        }

    pd.DataFrame([
        compare(blind, cut, "blind_vs_cut"),
        compare(aware, cut, "aware_vs_cut"),
        compare(aware, blind, "aware_vs_blind"),
    ]).to_csv(tables / "direct_improvements.tsv", sep="\t", index=False)

    return pooled


def make_figures(frames, metrics, selection, importance, pooled):
    apply_cms_style()
    sidecars = OUT / "figure_data"

    roc_data = {}
    for variant, frame in frames.items():
        fpr, tpr, threshold = roc_curve(frame.class_label, frame.score)
        data = pd.DataFrame({
            "signal_efficiency": tpr,
            "background_efficiency": fpr,
            "background_rejection": np.divide(
                1.0, fpr, out=np.full_like(fpr, np.nan), where=fpr > 0
            ),
            "threshold": threshold,
        })
        data.to_csv(sidecars / f"{variant}_roc.tsv", sep="\t", index=False)
        roc_data[variant] = data

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for variant in VARIANTS:
        data = roc_data[variant]
        ax.plot(
            data.background_efficiency.to_numpy(),
            data.signal_efficiency.to_numpy(),
            label=LABELS[variant], color=COLORS[variant],
        )
    ax.set(xlabel="Background efficiency", ylabel="Signal efficiency",
           xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend()
    add_delphes_header(ax, "Train-only nested outer-OOF")
    save(fig, "01_pooled_roc")

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for variant in VARIANTS:
        data = roc_data[variant]
        ax.plot(
            data.signal_efficiency.to_numpy(),
            data.background_rejection.to_numpy(),
            label=LABELS[variant], color=COLORS[variant],
        )
    ax.set(xlabel="Signal efficiency", ylabel="Background rejection",
           xlim=(0.2, 0.9), ylim=(1, 1000), yscale="log")
    ax.grid(alpha=0.25)
    ax.legend()
    add_delphes_header(ax, "Train-only nested outer-OOF")
    save(fig, "02_background_rejection")

    for number, variant in ((3, VARIANTS[0]), (4, VARIANTS[1])):
        frame = frames[variant]
        bins = np.linspace(0.0, 1.0, 51)
        rows = []
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        for sample, color in (("signal", "#009E73"), ("background", "#666666")):
            values = frame.loc[frame.sample_class == sample, "score"].to_numpy()
            density, edges = np.histogram(values, bins=bins, density=True)
            ax.stairs(density, edges, label=sample.capitalize(), color=color)
            rows.extend({
                "sample_class": sample,
                "bin_low": edges[index],
                "bin_high": edges[index + 1],
                "density": density[index],
            } for index in range(len(density)))
        pd.DataFrame(rows).to_csv(
            sidecars / f"{variant}_score_distribution.tsv",
            sep="\t", index=False,
        )
        ax.set(xlabel="BDT score", ylabel="Normalized density", yscale="log")
        ax.legend()
        add_delphes_header(ax, LABELS[variant])
        save(fig, f"{number:02d}_{variant}_score")

    plot_specs = (
        ("05_fixed_operating_point", "rejection", "Background rejection", False),
        ("08_signal_over_background", "S_over_B", "S/B", True),
        ("09_asimov_za", "ZA_stat", r"Stat-only Asimov $Z_A$", False),
        ("10_background_yield", "B", "Selected background yield", True),
        ("11_neff_background", "Neff_background", r"Background $N_{\rm eff}$", True),
    )
    names = pooled.display_method.tolist()
    colors = ["#777777", COLORS[VARIANTS[0]], COLORS[VARIANTS[1]]]
    for figure_name, column, ylabel, log_scale in plot_specs:
        data = pooled[["display_method", column]].copy()
        data.to_csv(sidecars / f"{figure_name}.tsv", sep="\t", index=False)
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.bar(np.arange(3), data[column].to_numpy(), color=colors)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        add_delphes_header(ax, "Run-2 expected-yield projection, 138 fb$^{-1}$")
        save(fig, figure_name)

    for figure_name, column, ylabel in (
        ("06_fold_auc", "auc", "Held-outer ROC AUC"),
        ("07_fold_rejection", "rejection", "Background rejection"),
    ):
        parts = []
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        for variant in VARIANTS:
            data = metrics[
                (metrics.method == variant)
                & (metrics.fold.astype(str) != "pooled")
            ].copy()
            data["fold"] = data.fold.astype(int)
            parts.append(data[["fold", column]].assign(variant=variant))
            ax.plot(
                data.fold.to_numpy(), data[column].to_numpy(), marker="o",
                label=LABELS[variant], color=COLORS[variant],
            )
        pd.concat(parts).to_csv(
            sidecars / f"{figure_name}.tsv", sep="\t", index=False
        )
        ax.set(xlabel="Outer fold", ylabel=ylabel, xticks=range(5))
        ax.grid(alpha=0.25)
        ax.legend()
        add_delphes_header(ax, "Train-only nested outer-OOF")
        save(fig, figure_name)

    physical = pooled[["display_method", "S", "B"]].copy()
    physical.to_csv(sidecars / "12_physical_yields.tsv", sep="\t", index=False)
    fig, ax_signal = plt.subplots(figsize=(7.0, 5.0))
    ax_background = ax_signal.twinx()
    x = np.arange(3)
    ax_signal.bar(x - 0.18, physical.S.to_numpy(), 0.36,
                  label="Signal", color="#009E73")
    ax_background.bar(x + 0.18, physical.B.to_numpy(), 0.36,
                      label="Background", color="#999999")
    ax_signal.set_xticks(x)
    ax_signal.set_xticklabels(
        physical.display_method.tolist(), rotation=15, ha="right"
    )
    ax_signal.set_ylabel("Selected signal yield")
    ax_background.set_ylabel("Selected background yield")
    ax_background.set_yscale("log")
    add_delphes_header(
        ax_signal, "Run-2 expected-yield projection, 138 fb$^{-1}$"
    )
    save(fig, "12_physical_yields")

    for number, variant in ((13, VARIANTS[0]), (14, VARIANTS[1])):
        data = (
            importance[importance.variant == variant]
            .groupby("feature").gain.mean()
            .sort_values().tail(15)
        )
        data.rename("mean_gain").to_csv(
            sidecars / f"{variant}_feature_importance.tsv", sep="\t"
        )
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        labels = [FEATURE_LATEX_LABELS.get(name, name) for name in data.index]
        ax.barh(labels, data.values, color=COLORS[variant])
        ax.set_xlabel("Mean XGBoost gain across outer folds")
        add_delphes_header(ax, LABELS[variant])
        save(fig, f"{number:02d}_{variant}_feature_importance")

    selection.to_csv(
        sidecars / "15_16_fold_candidate_threshold.tsv",
        sep="\t", index=False,
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.text(0.5, 0.5, "All 10 workloads selected depth4",
            ha="center", va="center", fontsize=16)
    ax.axis("off")
    add_delphes_header(ax, "Train-only model-selection stability")
    save(fig, "15_selected_hyperparameter")

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for variant in VARIANTS:
        data = selection[selection.variant == variant]
        ax.plot(
            data.outer_fold.to_numpy(), data.threshold.to_numpy(), marker="o",
            label=LABELS[variant], color=COLORS[variant],
        )
    ax.set(
        xlabel="Outer fold",
        ylabel="Development inner-OOF score threshold",
        xticks=range(5),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    add_delphes_header(ax, "Target signal efficiency = 0.585957")
    save(fig, "16_selected_threshold")


def write_summary(pooled, selection):
    records = pooled.replace({np.nan: None}).to_dict(orient="records")
    summary = {
        "status": "train_only_oof_finalized",
        "PRIMARY_PERFORMANCE_SOURCE": "five-fold nested outer-OOF",
        "train_rows": 1042397,
        "source_groups": 441,
        "production_cluster": 3799191,
        "production_return_audit": "PASS",
        "pooled_metrics": records,
        "all_outer_workloads_selected_candidate": "depth4",
        "threshold_min": float(selection.threshold.min()),
        "threshold_max": float(selection.threshold.max()),
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "categorized_bdt_started": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def refit_deployment():
    protocol = json.loads(PROTOCOL.read_text())
    require(xgboost.__version__ == protocol["implementation"]["xgboost"],
            "XGBoost version changed")
    require(sklearn.__version__ == protocol["implementation"]["sklearn"],
            "sklearn version changed")

    registry = load_registry(DEFAULT_PLAN, DEFAULT_AUTHORIZATION)
    aware_features = feature_names(protocol, "global_mass_aware")
    access_log = []
    frame = read_sources(
        registry, aware_features, access_log, "full_train_deployment"
    )
    require(len(frame) == 1_042_397, "deployment row count changed")
    require(frame.group_id.nunique() == 441, "deployment source count changed")
    weights = hierarchical_weights(frame)
    validate_training_weights(weights)
    row_counts_by_process = {
        str(key): int(value)
        for key, value in frame.process_or_mode.value_counts().sort_index().items()
    }
    source_counts_by_process = {
        str(key): int(value)
        for key, value in frame.groupby("process_or_mode").source_uid.nunique().sort_index().items()
    }
    source_uid_sha256 = hashlib.sha256(
        "\n".join(sorted(frame.source_uid.astype(str).unique())).encode()
    ).hexdigest()

    candidate = next(
        item for item in protocol["hyperparameter_candidates"]
        if item["id"] == "depth4"
    )
    model_records = []
    for variant in VARIANTS:
        features = feature_names(protocol, variant)
        model = estimator(protocol, candidate)
        model.fit(
            frame[features].to_numpy(np.float32),
            frame.class_label.to_numpy(np.int8),
            sample_weight=weights,
        )
        path = OUT / "models" / f"{variant}_full_train_xgboost.json"
        model.save_model(path)
        model_records.append({
            "variant": variant,
            "selected_candidate": "depth4",
            "selected_candidate_parameters": candidate,
            "features": features,
            "training_weight_recipe":
                protocol["weights"]["training"],
            "random_seed": protocol["implementation"]["random_seed"],
            "python": platform.python_version(),
            "xgboost": xgboost.__version__,
            "sklearn": sklearn.__version__,
            "train_rows": 1042397,
            "source_groups": 441,
            "train_row_counts_by_process": row_counts_by_process,
            "train_source_counts_by_process": source_counts_by_process,
            "sorted_source_uid_list_sha256": source_uid_sha256,
            "model_path": str(path.relative_to(REPO)),
            "model_sha256": sha256(path),
            "threshold_status":
                "score-only; no validation/test or resubstitution threshold; "
                "fold-specific OOF thresholds are not assumed transferable",
            "validation_used": False,
            "test_used": False,
        })

    contract = {
        "PRIMARY_PERFORMANCE_SOURCE": "five-fold nested outer-OOF",
        "FINAL_DEPLOYMENT_MODEL_TRAIN_ROWS": 1042397,
        "FINAL_DEPLOYMENT_MODEL_VALIDATION_USED": False,
        "FINAL_DEPLOYMENT_MODEL_TEST_USED": False,
        "selection_rule":
            "modal outer-fold candidate; all five folds selected depth4 "
            "for each variant",
        "deployment_performance_claimed": False,
        "models": model_records,
    }
    (OUT / "manifests" / "deployment_refit_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )


def write_manifest():
    contract_path = OUT / "manifests" / "deployment_refit_contract.json"
    require(contract_path.is_file(), "deployment contract missing")
    contract = json.loads(contract_path.read_text())
    final_summary = {
        "FINALIZATION_STATUS": "PASS",
        "PRIMARY_PERFORMANCE_SOURCE": "five-fold nested outer-OOF",
        "FINAL_DEPLOYMENT_MODEL_TRAIN_ROWS": 1042397,
        "FINAL_DEPLOYMENT_MODEL_VALIDATION_USED": False,
        "FINAL_DEPLOYMENT_MODEL_TEST_USED": False,
        "production_cluster_resubmitted": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "OOF_products": {
            variant: {
                "path": str(SOURCE / f"pooled_oof_{variant}.parquet"),
                "sha256": sha256(SOURCE / f"pooled_oof_{variant}.parquet"),
                "rows": 1042397,
            }
            for variant in VARIANTS
        },
        "deployment_models": contract["models"],
        "figure_inventory": {
            "png": sorted(path.name for path in (OUT / "figures/png").glob("*.png")),
            "pdf": sorted(path.name for path in (OUT / "figures/pdf").glob("*.pdf")),
            "sidecars": sorted(path.name for path in (OUT / "figure_data").glob("*")),
        },
        "fold_metrics_selection": "tables/fold_metrics_selection.tsv",
    }
    (OUT / "final_summary.json").write_text(
        json.dumps(final_summary, indent=2, sort_keys=True) + "\n"
    )
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.tsv":
            rows.append({
                "path": str(path.relative_to(OUT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    pd.DataFrame(rows).to_csv(
        OUT / "manifests" / "artifact_manifest.tsv",
        sep="\t", index=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("plots", "refit", "manifest", "all"))
    args = parser.parse_args()
    prepare_dirs()

    if args.stage in ("plots", "all"):
        frames, metrics = load_inputs()
        selection, importance = collect_fold_evidence(metrics)
        pooled = write_tables(metrics, selection, importance)
        make_figures(frames, metrics, selection, importance, pooled)
        write_summary(pooled, selection)

    if args.stage in ("refit", "all"):
        refit_deployment()

    if args.stage in ("manifest", "all"):
        write_manifest()

    print(f"FINALIZER_STAGE_{args.stage.upper()}=PASS")


if __name__ == "__main__":
    main()
