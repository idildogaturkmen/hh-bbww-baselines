import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve


DATA_DIR = Path("outputs/ttbar_classifier") 
OUTDIR = Path("outputs/ttbar_jet_activity_robustness")
PLOT_DIR = Path("outputs/plots/ttbar_jet_activity_robustness")
RESULTS_PATH = Path("RESULTS_TTBAR_JET_ACTIVITY_ROBUSTNESS.md")

TARGET_SIGNAL_EFFS = [0.90, 0.70, 0.50] 
MODEL_CONFIG = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "random_state": 42,
}

VARIANTS = [
    {
        # reference with all features and natural training events
        "variant": "all_features_nominal",
        "description": "All eight baseline features, no n_jets reweighting.",
        "drop_features": [],
        "train_reweight_njets": False,
    },
    {
        # test sensitivity to n_jets as an explicit feature, without changing the training event distribution
        "variant": "drop_n_jets",
        "description": "Remove n_jets as an explicit input feature.",
        "drop_features": ["n_jets"],
        "train_reweight_njets": False,
    },
    {
        # test sensitivity to n_jets differences in the training distribution, without removing it as a feature
        "variant": "njet_reweighted_training",
        "description": "All features, but training events are weighted so HH and ttbar have the same n_jets distribution.",
        "drop_features": [],
        "train_reweight_njets": True,
    },
    {
        # remove n_jets as a feature and also reweight training events to equalize the n_jets distribution. This is the most aggressive test of whether jet-activity differences are driving performance.
        "variant": "drop_n_jets_njet_reweighted_training",
        "description": "Remove n_jets and also reweight training events to equalize the n_jets distribution.",
        "drop_features": ["n_jets"],
        "train_reweight_njets": True,
    },
]


def ensure_dirs():
    # creates output folders 
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(split):
    # Loads features_train.npz, features_val.npz, or features_test.npz
    data = np.load(DATA_DIR / f"features_{split}.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(int)
    feature_names = [str(x) for x in data["feature_names"]]
    sample = data["sample"] if "sample" in data.files else np.array(["unknown"] * len(y))
    return X, y, feature_names, sample


def binomial_error(eff, n):
    # Standard binomial proportion confidence interval (Wald method).
    if n <= 0 or not np.isfinite(eff):
        return float("nan")
    return math.sqrt(eff * (1.0 - eff) / n)


def weighted_quantile(values, weights, quantile):
    '''
    Compute the weighted quantile of a set of values with corresponding weights.
    Parameters:
    - values: array-like, the data values.
    - weights: array-like, the weights corresponding to each value.
    - quantile: float in [0, 1], the desired quantile to compute (e.g., 0.5 for median).
    Returns:
    - The weighted quantile value.  
    '''

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(values) == 0:
        return float("nan")
    if np.sum(weights) <= 0:
        return float(np.quantile(values, quantile))

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = quantile * cumulative[-1]
    idx = np.searchsorted(cumulative, cutoff, side="left")
    idx = min(max(int(idx), 0), len(values) - 1)
    return float(values[idx])


def choose_threshold_for_signal_eff(scores_val, y_val, target_eff, weights_val=None):
    '''
    Choose a score threshold that achieves the target signal efficiency on the validation set, 
    optionally using weights for a weighted efficiency.
    '''
    sig = y_val == 1
    sig_scores = scores_val[sig]
    if weights_val is None:
        return float(np.quantile(sig_scores, 1.0 - target_eff))
    return weighted_quantile(sig_scores, weights_val[sig], 1.0 - target_eff)


def weighted_efficiency(pass_mask, total_mask, weights=None):
    '''
    Calculate the efficiency and its uncertainty, optionally using weights for a weighted efficiency.
    '''

    if weights is None:
        total = int(np.sum(total_mask))
        passed = int(np.sum(pass_mask & total_mask))
        eff = passed / total if total else float("nan")
        return passed, total, float(eff), binomial_error(eff, total)

    denom = float(np.sum(weights[total_mask]))
    numer = float(np.sum(weights[pass_mask & total_mask]))
    eff = numer / denom if denom > 0 else float("nan")

    # Effective-count approximation for an uncertainty on a weighted efficiency.
    w = weights[total_mask]
    n_eff = (float(np.sum(w)) ** 2 / float(np.sum(w**2))) if np.sum(w**2) > 0 else 0.0
    err = binomial_error(eff, n_eff)
    return numer, denom, float(eff), err


def evaluate_threshold(scores, y, threshold, weights=None, background_name="ttbar"):
    '''
    Evaluate the signal efficiency, background efficiency, and background rejection at the given threshold, optionally using weights for a weighted efficiency.
    '''
    sig = y == 1
    bkg = y == 0
    passed = scores >= threshold

    sig_pass, sig_total, sig_eff, sig_eff_err = weighted_efficiency(passed, sig, weights)
    bkg_pass, bkg_total, bkg_eff, bkg_eff_err = weighted_efficiency(passed, bkg, weights)
    bkg_rej = float("inf") if bkg_eff == 0 else 1.0 / bkg_eff

    return {
        "threshold": float(threshold),
        "signal_pass": float(sig_pass),
        "signal_total": float(sig_total),
        "signal_eff": float(sig_eff),
        "signal_eff_err": float(sig_eff_err),
        f"{background_name}_pass": float(bkg_pass),
        f"{background_name}_total": float(bkg_total),
        f"{background_name}_eff": float(bkg_eff),
        f"{background_name}_eff_err": float(bkg_eff_err),
        f"{background_name}_rejection": float(bkg_rej),
    }


def njet_weights_to_common_distribution(y, n_jets):
    """Return per-event weights that make the class-conditional n_jets spectra match.

    The target spectrum is the normalized overlap, bin by bin, of the HH and ttbar
    n_jets fractions. Bins with no support in either class receive zero weight.
    """
    y = np.asarray(y, dtype=int)
    n_jets = np.asarray(n_jets, dtype=int)
    weights = np.zeros(len(y), dtype=float)

    bins = np.array(sorted(set(int(x) for x in n_jets)))
    class_masks = {0: y == 0, 1: y == 1}
    class_totals = {cls: int(np.sum(mask)) for cls, mask in class_masks.items()}

    fractions = {}
    for cls, mask in class_masks.items():
        denom = class_totals[cls]
        fractions[cls] = {}
        for njet in bins:
            fractions[cls][njet] = (np.sum(mask & (n_jets == njet)) / denom) if denom else 0.0

    target = {njet: min(fractions[0][njet], fractions[1][njet]) for njet in bins}
    target_sum = float(sum(target.values()))
    if target_sum <= 0:
        return np.ones(len(y), dtype=float)
    target = {njet: value / target_sum for njet, value in target.items()}

    for cls, mask in class_masks.items():
        denom = class_totals[cls]
        if denom <= 0:
            continue
        for njet in bins:
            bin_mask = mask & (n_jets == njet)
            observed_fraction = fractions[cls][njet]
            if observed_fraction > 0:
                weights[bin_mask] = target[njet] / observed_fraction
            else:
                weights[bin_mask] = 0.0

        class_mean = np.mean(weights[mask]) if np.sum(mask) else 1.0
        if class_mean > 0:
            weights[mask] /= class_mean

    return weights


def njet_distribution_table(split, y, n_jets, weights=None):
    rows = []
    for cls, sample in [(1, "HH"), (0, "ttbar")]:
        mask = y == cls
        total = np.sum(mask) if weights is None else np.sum(weights[mask])
        for njet in sorted(set(int(x) for x in n_jets)):
            bin_mask = mask & (n_jets == njet)
            count = np.sum(bin_mask) if weights is None else np.sum(weights[bin_mask])
            rows.append(
                {
                    "split": split,
                    "sample": sample,
                    "n_jets": int(njet),
                    "count": float(count),
                    "fraction": float(count / total) if total > 0 else float("nan"),
                    "weighted": bool(weights is not None),
                }
            )
    return rows


def feature_indices(feature_names, drop_features):
    drop = set(drop_features)
    keep = [idx for idx, name in enumerate(feature_names) if name not in drop]
    return keep, [feature_names[idx] for idx in keep]


def auc_score(y, scores, weights=None):
    return float(roc_auc_score(y, scores, sample_weight=weights))


def train_and_evaluate_variant(
    variant,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_names,
    njet_train,
    njet_val,
    njet_test,
):
    keep, kept_names = feature_indices(feature_names, variant["drop_features"])
    X_train_v = X_train[:, keep]
    X_val_v = X_val[:, keep]
    X_test_v = X_test[:, keep]

    train_weights = (
        njet_weights_to_common_distribution(y_train, njet_train)
        if variant["train_reweight_njets"]
        else None
    )

    model = GradientBoostingClassifier(**MODEL_CONFIG)
    model.fit(X_train_v, y_train, sample_weight=train_weights)

    scores = {
        "train": model.predict_proba(X_train_v)[:, 1],
        "val": model.predict_proba(X_val_v)[:, 1],
        "test": model.predict_proba(X_test_v)[:, 1],
    }

    eval_weights = {
        "natural": {
            "train": None,
            "val": None,
            "test": None,
        },
        "njet_balanced": {
            "train": njet_weights_to_common_distribution(y_train, njet_train),
            "val": njet_weights_to_common_distribution(y_val, njet_val),
            "test": njet_weights_to_common_distribution(y_test, njet_test),
        },
    }

    metric_rows = []
    wp_rows = []
    roc_rows = []

    for eval_mode, weights_by_split in eval_weights.items():
        row = {
            "variant": variant["variant"],
            "description": variant["description"],
            "features": ", ".join(kept_names),
            "dropped_features": ", ".join(variant["drop_features"]),
            "train_reweight_njets": bool(variant["train_reweight_njets"]),
            "evaluation": eval_mode,
            "auc_train": auc_score(y_train, scores["train"], weights_by_split["train"]),
            "auc_val": auc_score(y_val, scores["val"], weights_by_split["val"]),
            "auc_test": auc_score(y_test, scores["test"], weights_by_split["test"]),
        }
        metric_rows.append(row)

        fpr, tpr, _ = roc_curve(
            y_test,
            scores["test"],
            sample_weight=weights_by_split["test"],
        )
        for x, yy in zip(fpr, tpr):
            roc_rows.append(
                {
                    "variant": variant["variant"],
                    "evaluation": eval_mode,
                    "ttbar_eff": float(x),
                    "signal_eff": float(yy),
                }
            )

        for target in TARGET_SIGNAL_EFFS:
            threshold = choose_threshold_for_signal_eff(
                scores["val"],
                y_val,
                target,
                weights_by_split["val"],
            )
            wp = evaluate_threshold(
                scores["test"],
                y_test,
                threshold,
                weights_by_split["test"],
                background_name="ttbar",
            )
            wp.update(
                {
                    "variant": variant["variant"],
                    "evaluation": eval_mode,
                    "target_signal_eff_val": target,
                }
            )
            wp_rows.append(wp)

    importance_rows = []
    for feature, importance in zip(kept_names, model.feature_importances_):
        importance_rows.append(
            {
                "variant": variant["variant"],
                "feature": feature,
                "importance": float(importance),
            }
        )

    return metric_rows, wp_rows, importance_rows, roc_rows


def plot_njet_distributions(njet_rows):
    df = pd.DataFrame(njet_rows)

    for split in ["train", "val", "test"]:
        natural = df[(df["split"] == split) & (~df["weighted"])]
        balanced = df[(df["split"] == split) & (df["weighted"])]

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharey=True)
        for ax, sub, title in [
            (axes[0], natural, "Natural"),
            (axes[1], balanced, "n_jets-balanced weights"),
        ]:
            for sample, linestyle in [("HH", "-"), ("ttbar", "--")]:
                one = sub[sub["sample"] == sample].sort_values("n_jets")
                ax.plot(one["n_jets"], one["fraction"], marker="o", linestyle=linestyle, label=sample)
            ax.set_xlabel("n_jets after MAX_JETS cap")
            ax.set_title(title)
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Fraction of events")
        axes[0].legend()
        fig.suptitle(f"{split} n_jets distributions")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"njet_distribution_{split}.png", dpi=200)
        plt.close(fig)


def plot_auc(metrics_df):
    test = metrics_df[metrics_df["evaluation"].isin(["natural", "njet_balanced"])].copy()
    pivot = test.pivot(index="variant", columns="evaluation", values="auc_test")
    pivot = pivot.loc[[v["variant"] for v in VARIANTS]]

    x = np.arange(len(pivot))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(x - width / 2, pivot["natural"], width, label="natural test")
    ax.bar(x + width / 2, pivot["njet_balanced"], width, label="n_jets-balanced test")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=25, ha="right")
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.45, max(0.8, float(np.nanmax(pivot.to_numpy())) + 0.03))
    ax.set_title("HH-vs-ttbar BDT robustness to n_jets treatment")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "auc_comparison.png", dpi=200)
    plt.close(fig)


def plot_working_points(wp_df):
    for target in TARGET_SIGNAL_EFFS:
        sub = wp_df[np.isclose(wp_df["target_signal_eff_val"], target)].copy()
        pivot = sub.pivot(index="variant", columns="evaluation", values="ttbar_rejection")
        pivot = pivot.loc[[v["variant"] for v in VARIANTS]]

        x = np.arange(len(pivot))
        width = 0.36

        fig, ax = plt.subplots(figsize=(9.0, 4.6))
        ax.bar(x - width / 2, pivot["natural"], width, label="natural test")
        ax.bar(x + width / 2, pivot["njet_balanced"], width, label="n_jets-balanced test")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=25, ha="right")
        ax.set_ylabel("ttbar rejection")
        ax.set_title(f"ttbar rejection at validation target {target:.0%} signal efficiency")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"ttbar_rejection_wp{int(target * 100)}.png", dpi=200)
        plt.close(fig)


def plot_rocs(roc_df):
    for eval_mode in ["natural", "njet_balanced"]:
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        for variant in [v["variant"] for v in VARIANTS]:
            sub = roc_df[(roc_df["variant"] == variant) & (roc_df["evaluation"] == eval_mode)]
            ax.plot(sub["ttbar_eff"], sub["signal_eff"], linewidth=1.4, label=variant)
        ax.set_xlabel("ttbar efficiency")
        ax.set_ylabel("HH signal efficiency")
        ax.set_title(f"ROC curves, {eval_mode.replace('_', ' ')} evaluation")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"roc_{eval_mode}.png", dpi=200)
        plt.close(fig)


def plot_feature_importances(importance_df):
    for variant in [v["variant"] for v in VARIANTS]:
        sub = importance_df[importance_df["variant"] == variant].sort_values("importance")
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.barh(sub["feature"], sub["importance"])
        ax.set_xlabel("Feature importance")
        ax.set_title(variant)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"feature_importance_{variant}.png", dpi=200)
        plt.close(fig)


def write_summary(metrics_df, wp_df, feature_importance_df, njet_df, args):
    natural = metrics_df[metrics_df["evaluation"] == "natural"].set_index("variant")
    balanced = metrics_df[metrics_df["evaluation"] == "njet_balanced"].set_index("variant")

    lines = [
        "# TTbar Jet-Activity Robustness Check",
        "",
        "This study addresses the question about whether the corrected HH-vs-semileptonic-ttbar BDT is still sensitive to jet multiplicity or to the fixed `MAX_JETS = 12` preprocessing choice.",
        "",
        "## Command",
        "",
        "```bash",
        f"/tmp/hh-bbww-venv/bin/python scripts/check_ttbar_jet_activity_robustness.py --seed {args.seed}",
        "```",
        "",
        "## Method",
        "",
        "- The script reuses the fixed `outputs/ttbar_classifier/features_{train,val,test}.npz` files; it does not rescan COLLIDE-1M.",
        "- The nominal model uses the same `GradientBoostingClassifier` configuration as the current ttbar baseline.",
        "- `drop_n_jets` removes the explicit jet-count input.",
        "- `njet_reweighted_training` gives training events weights so the HH and ttbar `n_jets` spectra match.",
        "- `njet_balanced` evaluation uses validation/test weights that equalize the HH and ttbar `n_jets` spectra. This asks how much performance remains after the test metric stops rewarding jet-count differences.",
        "",
        "## Test AUC Summary",
        "",
        "| Variant | Natural test AUC | n_jets-balanced test AUC | Change |",
        "|---|---:|---:|---:|",
    ]

    for variant in [v["variant"] for v in VARIANTS]:
        nat = natural.loc[variant, "auc_test"]
        bal = balanced.loc[variant, "auc_test"]
        lines.append(f"| {variant} | {nat:.4f} | {bal:.4f} | {bal - nat:+.4f} |")

    lines.extend(
        [
            "",
            "## Working Points",
            "",
            "| Variant | Evaluation | Target signal eff. | Test signal eff. | Test ttbar eff. | Test ttbar rejection |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )

    wp_order = {variant: idx for idx, variant in enumerate(v["variant"] for v in VARIANTS)}
    wp_sorted = wp_df.sort_values(
        by=["variant", "evaluation", "target_signal_eff_val"],
        key=lambda col: col.map(wp_order) if col.name == "variant" else col,
    )
    for _, row in wp_sorted.iterrows():
        lines.append(
            "| {variant} | {evaluation} | {target:.2f} | {sig:.4f} | {bkg:.4f} | {rej:.2f} |".format(
                variant=row["variant"],
                evaluation=row["evaluation"],
                target=row["target_signal_eff_val"],
                sig=row["signal_eff"],
                bkg=row["ttbar_eff"],
                rej=row["ttbar_rejection"],
            )
        )

    lines.extend(
        [
            "",
            "## Feature Importance Notes",
            "",
            "| Variant | Most important features |",
            "|---|---|",
        ]
    )
    for variant in [v["variant"] for v in VARIANTS]:
        sub = feature_importance_df[feature_importance_df["variant"] == variant]
        top = sub.sort_values("importance", ascending=False).head(4)
        text = ", ".join(f"{r.feature} ({r.importance:.3f})" for r in top.itertuples())
        lines.append(f"| {variant} | {text} |")

    njet_nat = njet_df[(njet_df["split"] == "test") & (~njet_df["weighted"])]
    hh_mean = np.average(
        njet_nat[njet_nat["sample"] == "HH"]["n_jets"],
        weights=njet_nat[njet_nat["sample"] == "HH"]["fraction"],
    )
    tt_mean = np.average(
        njet_nat[njet_nat["sample"] == "ttbar"]["n_jets"],
        weights=njet_nat[njet_nat["sample"] == "ttbar"]["fraction"],
    )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"On the natural test split, the mean capped jet multiplicity is about {hh_mean:.2f} for HH and {tt_mean:.2f} for semileptonic ttbar. Therefore, jet activity is a real difference between the samples, but it is also exactly the kind of difference that can make a classifier look better if not checked carefully.",
            "",
            "If the natural AUC and the n_jets-balanced AUC are close, the BDT is not mainly being rewarded for the class-level jet-count spectrum. If the balanced AUC is much lower, then the separation depends strongly on jet multiplicity or features correlated with it.",
            "",
            "This is different from the earlier `MAX_JETS` retention scan. That scan asked whether a 12-jet cap loses truth-matchable H->bb signal events. This robustness check asks whether the event-level classifier changes when the HH and ttbar jet-multiplicity spectra are made comparable.",
            "",
            "## Outputs",
            "",
            "- `outputs/ttbar_jet_activity_robustness/variant_metrics.csv`",
            "- `outputs/ttbar_jet_activity_robustness/working_points.csv`",
            "- `outputs/ttbar_jet_activity_robustness/feature_importance.csv`",
            "- `outputs/ttbar_jet_activity_robustness/njet_distributions.csv`",
            "- `outputs/ttbar_jet_activity_robustness/summary.json`",
            "- `outputs/plots/ttbar_jet_activity_robustness/*.png`",
        ]
    )

    RESULTS_PATH.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train HH-vs-ttbar BDT variants that test sensitivity to n_jets and "
            "jet-activity differences using fixed classifier feature files."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    X_train, y_train, feature_names, _ = load_split("train")
    X_val, y_val, _, _ = load_split("val")
    X_test, y_test, _, _ = load_split("test")

    if "n_jets" not in feature_names:
        raise RuntimeError("The feature files do not contain n_jets; cannot run this diagnostic.")

    MODEL_CONFIG["random_state"] = args.seed

    njet_idx = feature_names.index("n_jets")
    njet_train = X_train[:, njet_idx].astype(int)
    njet_val = X_val[:, njet_idx].astype(int)
    njet_test = X_test[:, njet_idx].astype(int)

    njet_rows = []
    for split, y, njet in [
        ("train", y_train, njet_train),
        ("val", y_val, njet_val),
        ("test", y_test, njet_test),
    ]:
        njet_rows.extend(njet_distribution_table(split, y, njet, weights=None))
        njet_rows.extend(njet_distribution_table(split, y, njet, weights=njet_weights_to_common_distribution(y, njet)))

    metric_rows = []
    wp_rows = []
    importance_rows = []
    roc_rows = []

    print("Dataset sizes:")
    print(f"  train: {X_train.shape}, HH={np.sum(y_train == 1)}, ttbar={np.sum(y_train == 0)}")
    print(f"  val:   {X_val.shape}, HH={np.sum(y_val == 1)}, ttbar={np.sum(y_val == 0)}")
    print(f"  test:  {X_test.shape}, HH={np.sum(y_test == 1)}, ttbar={np.sum(y_test == 0)}")

    for variant in VARIANTS:
        print(f"Training variant: {variant['variant']}", flush=True)
        metrics, wps, importances, rocs = train_and_evaluate_variant(
            variant,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_names,
            njet_train,
            njet_val,
            njet_test,
        )
        metric_rows.extend(metrics)
        wp_rows.extend(wps)
        importance_rows.extend(importances)
        roc_rows.extend(rocs)

    metrics_df = pd.DataFrame(metric_rows)
    wp_df = pd.DataFrame(wp_rows)
    importance_df = pd.DataFrame(importance_rows)
    roc_df = pd.DataFrame(roc_rows)
    njet_df = pd.DataFrame(njet_rows)

    metrics_df.to_csv(OUTDIR / "variant_metrics.csv", index=False)
    wp_df.to_csv(OUTDIR / "working_points.csv", index=False)
    importance_df.to_csv(OUTDIR / "feature_importance.csv", index=False)
    roc_df.to_csv(OUTDIR / "roc_curves.csv", index=False)
    njet_df.to_csv(OUTDIR / "njet_distributions.csv", index=False)

    summary = {
        "model": "GradientBoostingClassifier",
        "model_config": MODEL_CONFIG,
        "feature_names": feature_names,
        "variants": VARIANTS,
        "target_signal_efficiencies": TARGET_SIGNAL_EFFS,
        "outputs": {
            "metrics": str(OUTDIR / "variant_metrics.csv"),
            "working_points": str(OUTDIR / "working_points.csv"),
            "feature_importance": str(OUTDIR / "feature_importance.csv"),
            "roc_curves": str(OUTDIR / "roc_curves.csv"),
            "njet_distributions": str(OUTDIR / "njet_distributions.csv"),
            "plots": str(PLOT_DIR),
            "markdown": str(RESULTS_PATH),
        },
    }
    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_njet_distributions(njet_rows)
    plot_auc(metrics_df)
    plot_working_points(wp_df)
    plot_rocs(roc_df)
    plot_feature_importances(importance_df)
    write_summary(metrics_df, wp_df, importance_df, njet_df, args)

    print("\nTest AUC summary:")
    print(metrics_df[["variant", "evaluation", "auc_train", "auc_val", "auc_test"]].to_string(index=False))
    print(f"\nSaved outputs to {OUTDIR}")
    print(f"Saved plots to {PLOT_DIR}")
    print(f"Saved markdown summary to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
