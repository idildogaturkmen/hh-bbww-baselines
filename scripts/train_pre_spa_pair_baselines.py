import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


M_H = 125.0


def width68(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return 0.5 * (np.percentile(values, 84) - np.percentile(values, 16))


def summarize_selected(df):
    values = df["selected_mbb"].to_numpy(dtype=float)
    return {
        "n_events": int(len(df)),
        "pair_accuracy": float(np.mean(df["correct"])),
        "mbb_mean": float(np.mean(values)),
        "mbb_median": float(np.median(values)),
        "mbb_p16": float(np.percentile(values, 16)),
        "mbb_p84": float(np.percentile(values, 84)),
        "mbb_width68": float(width68(values)),
        "frac_90_140": float(np.mean((values >= 90.0) & (values <= 140.0))),
        "frac_100_150": float(np.mean((values >= 100.0) & (values <= 150.0))),
        "median_abs_offset_from_125": float(abs(np.median(values) - M_H)),
    }


def bootstrap_metrics(selected_df, n_boot=500, seed=12345):
    rng = np.random.default_rng(seed)
    event_ids = selected_df["event_id"].unique()
    rows = []

    for _ in range(n_boot):
        sampled_ids = rng.choice(event_ids, size=len(event_ids), replace=True)
        sampled = pd.concat(
            [selected_df[selected_df["event_id"] == eid] for eid in sampled_ids],
            ignore_index=True,
        )
        rows.append(summarize_selected(sampled))

    boot = pd.DataFrame(rows)
    out = {}
    for col in [
        "pair_accuracy",
        "mbb_median",
        "mbb_width68",
        "frac_90_140",
        "frac_100_150",
        "median_abs_offset_from_125",
    ]:
        out[f"{col}_err"] = float(boot[col].std())
        out[f"{col}_lo"] = float(np.percentile(boot[col], 16))
        out[f"{col}_hi"] = float(np.percentile(boot[col], 84))
    return out


def split_by_event_id(df):
    """
    Deterministic 60/20/20 split by event_id.
    All candidate pairs from the same event stay in the same split.
    """
    mod = df["event_id"] % 5
    df = df.copy()
    df["split"] = np.where(mod == 0, "test", np.where(mod == 1, "val", "train"))
    return df


def choose_by_score(df, score_col, selected_mbb_col):
    idx = df.groupby("event_id")[score_col].idxmax()
    chosen = df.loc[idx].copy()
    chosen["correct"] = chosen["is_truth_pair"].astype(bool)
    chosen["selected_mbb"] = chosen[selected_mbb_col]
    return chosen[["event_id", "correct", "selected_mbb"]]


def choose_min_absdiff(df, diff_col, selected_mbb_col):
    idx = df.groupby("event_id")[diff_col].idxmin()
    chosen = df.loc[idx].copy()
    chosen["correct"] = chosen["is_truth_pair"].astype(bool)
    chosen["selected_mbb"] = chosen[selected_mbb_col]
    return chosen[["event_id", "correct", "selected_mbb"]]


def evaluate_physics_selector(df, selector, lam=None):
    work = df.copy()

    if selector == "top2_btag":
        work["score"] = work["sum_btag"]
        return choose_by_score(work, "score", "mbb_uncorrected")

    if selector == "closest_mass_uncorrected":
        return choose_min_absdiff(work, "mbb_unc_absdiff", "mbb_uncorrected")

    if selector == "closest_mass_corrected":
        return choose_min_absdiff(work, "mbb_corr_absdiff", "mbb_corrected")

    if selector == "btag_mass_uncorrected":
        if lam is None:
            raise ValueError("lambda required")
        work["score"] = work["sum_btag"] - work["mbb_unc_absdiff"] / lam
        return choose_by_score(work, "score", "mbb_uncorrected")

    if selector == "btag_mass_corrected":
        if lam is None:
            raise ValueError("lambda required")
        work["score"] = work["sum_btag"] - work["mbb_corr_absdiff"] / lam
        return choose_by_score(work, "score", "mbb_corrected")

    raise ValueError(selector)


def add_pair_features(df):
    df = df.copy()

    df["max_btag"] = df[["btag_a", "btag_b"]].max(axis=1)
    df["min_btag"] = df[["btag_a", "btag_b"]].min(axis=1)
    df["max_pt"] = df[["pt_a", "pt_b"]].max(axis=1)
    df["min_pt"] = df[["pt_a", "pt_b"]].min(axis=1)
    df["pt_balance"] = df["min_pt"] / np.maximum(df["max_pt"], 1e-6)
    df["mean_corr"] = 0.5 * (df["corr_a"] + df["corr_b"])
    df["max_corr"] = df[["corr_a", "corr_b"]].max(axis=1)
    df["min_corr"] = df[["corr_a", "corr_b"]].min(axis=1)

    # Number of pairs per event approximates event jet multiplicity.
    n_pairs = df.groupby("event_id")["event_id"].transform("count").astype(float)
    df["n_pairs"] = n_pairs
    df["approx_n_jets"] = 0.5 * (1.0 + np.sqrt(1.0 + 8.0 * n_pairs))

    return df


def train_bdt(train_df, feature_cols):
    X = train_df[feature_cols].to_numpy(dtype=float)
    y = train_df["is_truth_pair"].astype(int).to_numpy()

    n_pos = max(np.sum(y == 1), 1)
    n_neg = max(np.sum(y == 0), 1)
    weights = np.where(y == 1, 0.5 / n_pos, 0.5 / n_neg)

    clf = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=12345,
    )
    clf.fit(X, y, sample_weight=weights)
    return clf


def evaluate_bdt(clf, df, feature_cols, selected_mbb_col, name):
    work = df.copy()
    X = work[feature_cols].to_numpy(dtype=float)
    y = work["is_truth_pair"].astype(int).to_numpy()
    proba = clf.predict_proba(X)[:, 1]
    work["score"] = proba

    try:
        auc = float(roc_auc_score(y, proba))
    except Exception:
        auc = np.nan

    try:
        ap = float(average_precision_score(y, proba))
    except Exception:
        ap = np.nan

    selected = choose_by_score(work, "score", selected_mbb_col)
    selected["selector"] = name
    return selected, auc, ap, proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs-csv",
        default="outputs/pairing_with_bjet_correction/all_candidate_pairs.csv",
    )
    parser.add_argument("--outdir", default="outputs/pre_spa_pair_baselines")
    parser.add_argument("--plot-dir", default="outputs/plots/pre_spa_pair_baselines")
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    plot_dir = Path(args.plot_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pairs_csv)
    df = add_pair_features(df)
    df = split_by_event_id(df)

    print("Loaded candidate pairs:", len(df))
    print("Events:", df["event_id"].nunique())
    print(df.groupby("split")["event_id"].nunique())

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    # ------------------------------------------------------------
    # 1. Tune lambda for btag+mass on validation.
    # ------------------------------------------------------------
    lambdas = [15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200]
    lambda_rows = []

    for lam in lambdas:
        for selector in ["btag_mass_uncorrected", "btag_mass_corrected"]:
            selected = evaluate_physics_selector(val_df, selector, lam=lam)
            s = summarize_selected(selected)
            lambda_rows.append({
                "selector": selector,
                "lambda": lam,
                **s,
            })

    lambda_scan = pd.DataFrame(lambda_rows)
    lambda_scan.to_csv(outdir / "lambda_scan_validation.csv", index=False)

    best_lambdas = {}
    for selector in ["btag_mass_uncorrected", "btag_mass_corrected"]:
        part = lambda_scan[lambda_scan["selector"] == selector]
        # Main criterion: pair accuracy. Tie-breakers: width68, then frac_90_140.
        part = part.sort_values(
            ["pair_accuracy", "mbb_width68", "frac_90_140"],
            ascending=[False, True, False],
        )
        best_lambdas[selector] = float(part.iloc[0]["lambda"])

    print("Best lambdas from validation:", best_lambdas)

    # ------------------------------------------------------------
    # 2. Evaluate physics selectors on test.
    # ------------------------------------------------------------
    selected_outputs = []
    metric_rows = []

    physics_specs = [
        ("top2_btag", None),
        ("closest_mass_uncorrected", None),
        ("closest_mass_corrected", None),
        ("btag_mass_uncorrected_tuned", best_lambdas["btag_mass_uncorrected"]),
        ("btag_mass_corrected_tuned", best_lambdas["btag_mass_corrected"]),
    ]

    for name, lam in physics_specs:
        base_name = name.replace("_tuned", "")
        selected = evaluate_physics_selector(test_df, base_name, lam=lam)
        selected["selector"] = name
        selected_outputs.append(selected)

        row = {
            "selector": name,
            "lambda": lam if lam is not None else np.nan,
            **summarize_selected(selected),
            **bootstrap_metrics(selected, n_boot=args.n_boot),
        }
        metric_rows.append(row)

    # ------------------------------------------------------------
    # 3. Train BDTs.
    # ------------------------------------------------------------
    features_uncorrected = [
        "sum_btag",
        "max_btag",
        "min_btag",
        "pt_a",
        "pt_b",
        "max_pt",
        "min_pt",
        "pt_balance",
        "mbb_uncorrected",
        "mbb_unc_absdiff",
        "n_pairs",
        "approx_n_jets",
    ]

    features_corrected = features_uncorrected + [
        "mbb_corrected",
        "mbb_corr_absdiff",
        "corr_a",
        "corr_b",
        "mean_corr",
        "max_corr",
        "min_corr",
    ]

    bdt_unc = train_bdt(train_df, features_uncorrected)
    bdt_corr = train_bdt(train_df, features_corrected)

    selected, auc_unc, ap_unc, score_unc = evaluate_bdt(
        bdt_unc,
        test_df,
        features_uncorrected,
        "mbb_uncorrected",
        "bdt_uncorrected_features",
    )
    selected_outputs.append(selected)
    metric_rows.append({
        "selector": "bdt_uncorrected_features",
        "lambda": np.nan,
        "pair_auc": auc_unc,
        "pair_average_precision": ap_unc,
        **summarize_selected(selected),
        **bootstrap_metrics(selected, n_boot=args.n_boot),
    })

    selected, auc_corr, ap_corr, score_corr = evaluate_bdt(
        bdt_corr,
        test_df,
        features_corrected,
        "mbb_corrected",
        "bdt_corrected_features",
    )
    selected_outputs.append(selected)
    metric_rows.append({
        "selector": "bdt_corrected_features",
        "lambda": np.nan,
        "pair_auc": auc_corr,
        "pair_average_precision": ap_corr,
        **summarize_selected(selected),
        **bootstrap_metrics(selected, n_boot=args.n_boot),
    })

    bdt_diagnostics = pd.DataFrame([
        {
            "model": "bdt_uncorrected_features",
            "roc_auc": auc_unc,
            "average_precision": ap_unc,
        },
        {
            "model": "bdt_corrected_features",
            "roc_auc": auc_corr,
            "average_precision": ap_corr,
        },
    ])
    bdt_diagnostics.to_csv(outdir / "bdt_diagnostics.csv", index=False)

    bdt_scores = test_df[
        [
            "event_id",
            "jet_a_idx",
            "jet_b_idx",
            "is_truth_pair",
            "sum_btag",
            "mbb_uncorrected",
            "mbb_corrected",
            "mbb_unc_absdiff",
            "mbb_corr_absdiff",
            "corr_a",
            "corr_b",
        ]
    ].copy()
    bdt_scores["bdt_uncorrected_score"] = score_unc
    bdt_scores["bdt_corrected_score"] = score_corr
    bdt_scores.to_csv(outdir / "bdt_test_pair_scores.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    selected_all = pd.concat(selected_outputs, ignore_index=True)

    metrics.to_csv(outdir / "test_metrics_with_bootstrap.csv", index=False)
    selected_all.to_csv(outdir / "selected_pairs_test.csv", index=False)

    with open(outdir / "feature_sets.json", "w") as f:
        json.dump(
            {
                "features_uncorrected": features_uncorrected,
                "features_corrected": features_corrected,
                "best_lambdas": best_lambdas,
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # 4. Markdown summary.
    # ------------------------------------------------------------
    lines = []
    lines.append("# Pre-SPA-Net H→bb Pair Baselines")
    lines.append("")
    lines.append("This study uses the candidate-pair table from the DNN-corrected b-jet response workflow.")
    lines.append("Events are split deterministically by event_id into train/validation/test.")
    lines.append("")
    lines.append("## Validation-tuned lambda values")
    lines.append("")
    for k, v in best_lambdas.items():
        lines.append(f"- {k}: lambda = {v:g}")

    lines.append("")
    lines.append("## Test metrics")
    lines.append("")
    lines.append("| Selector | Lambda | Events | Pair accuracy | Acc. err | m_bb median [GeV] | m_bb width68 [GeV] | Width err | Frac 90-140 | Frac 100-150 | Pair AUC | Pair AP |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for _, r in metrics.iterrows():
        lambda_value = r.get("lambda", np.nan)
        lambda_text = "" if pd.isna(lambda_value) else f"{lambda_value:.0f}"

        auc_value = r.get("pair_auc", np.nan)
        auc_text = "" if pd.isna(auc_value) else f"{auc_value:.4f}"

        ap_value = r.get("pair_average_precision", np.nan)
        ap_text = "" if pd.isna(ap_value) else f"{ap_value:.4f}"

        lines.append(
            f"| {r['selector']} | "
            f"{lambda_text} | "
            f"{int(r['n_events'])} | "
            f"{r['pair_accuracy']:.4f} | "
            f"{r['pair_accuracy_err']:.4f} | "
            f"{r['mbb_median']:.2f} | "
            f"{r['mbb_width68']:.2f} | "
            f"{r['mbb_width68_err']:.2f} | "
            f"{r['frac_90_140']:.4f} | "
            f"{r['frac_100_150']:.4f} | "
            f"{auc_text} | "
            f"{ap_text} |"
        )

    summary_text = "\n".join(lines) + "\n"
    (outdir / "RESULTS_PRE_SPA_PAIR_BASELINES.md").write_text(summary_text)

    summary_dir = Path("Results Summaries")
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "RESULTS_PRE_SPA_PAIR_BASELINES.md").write_text(summary_text)

    print("\n=== Test metrics ===")
    print(metrics.to_string(index=False))
    print(f"\nSaved outputs in: {outdir}")

    # ------------------------------------------------------------
    # 5. Plots.
    # ------------------------------------------------------------
    df_plot = metrics.sort_values("pair_accuracy")
    plt.figure(figsize=(10, 5))
    plt.barh(df_plot["selector"], df_plot["pair_accuracy"])
    plt.xlabel("Truth-pair selection accuracy")
    plt.title("Pre-SPA-Net pair-reconstruction baselines")
    plt.tight_layout()
    plt.savefig(plot_dir / "test_pair_accuracy.png", dpi=180)
    plt.close()

    df_plot = metrics.sort_values("mbb_width68")
    plt.figure(figsize=(10, 5))
    plt.barh(df_plot["selector"], df_plot["mbb_width68"])
    plt.xlabel("Selected-pair m_bb width68 [GeV]")
    plt.title("Selected H→bb candidate mass resolution")
    plt.tight_layout()
    plt.savefig(plot_dir / "test_mbb_width68.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for selector in selected_all["selector"].unique():
        part = selected_all[selected_all["selector"] == selector]
        plt.hist(
            part["selected_mbb"],
            bins=np.linspace(0, 250, 80),
            histtype="step",
            density=True,
            label=selector,
        )
    plt.axvline(M_H, linestyle="--", linewidth=1.0)
    plt.xlabel("Selected-pair m_bb [GeV]")
    plt.ylabel("Normalized events")
    plt.title("Selected H→bb candidate mass on test events")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(plot_dir / "test_selected_mbb_distributions.png", dpi=180)
    plt.close()

    # Pair-level ROC curves for the two BDTs.
    y_test = test_df["is_truth_pair"].astype(int).to_numpy()

    plt.figure(figsize=(7, 6))
    for label, score, auc_value in [
        ("BDT uncorrected features", score_unc, auc_unc),
        ("BDT corrected features", score_corr, auc_corr),
    ]:
        fpr, tpr, _ = roc_curve(y_test, score)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc_value:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Pair-level ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "roc_bdt_uncorrected_vs_corrected.png", dpi=180)
    plt.close()

    # Pair-level precision-recall curves.
    plt.figure(figsize=(7, 6))
    for label, score, ap_value in [
        ("BDT uncorrected features", score_unc, ap_unc),
        ("BDT corrected features", score_corr, ap_corr),
    ]:
        precision, recall, _ = precision_recall_curve(y_test, score)
        plt.plot(recall, precision, label=f"{label} (AP={ap_value:.4f})")
    positive_fraction = float(np.mean(y_test))
    plt.axhline(positive_fraction, linestyle="--", linewidth=1.0, label=f"Positive fraction={positive_fraction:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Pair-level precision-recall curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "pr_bdt_uncorrected_vs_corrected.png", dpi=180)
    plt.close()

    # Corrected-feature BDT score separation.
    plt.figure(figsize=(7, 5))
    bins = np.linspace(0, 1, 60)
    plt.hist(
        score_corr[y_test == 0],
        bins=bins,
        histtype="step",
        density=True,
        label="Wrong candidate pairs",
    )
    plt.hist(
        score_corr[y_test == 1],
        bins=bins,
        histtype="step",
        density=True,
        label="True H→bb pair",
    )
    plt.xlabel("Corrected-feature BDT score")
    plt.ylabel("Normalized candidate pairs")
    plt.title("Corrected-feature BDT score separation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "bdt_score_distribution_corrected.png", dpi=180)
    plt.close()

    print(f"Saved plots in: {plot_dir}")


if __name__ == "__main__":
    main()