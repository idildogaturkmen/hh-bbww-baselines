import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PAIR_DNN_PREDICTIONS = Path("outputs/hbb_pair_dnn/predictions.csv")
PAIR_DNN_METRICS = Path("outputs/hbb_pair_dnn/metrics.csv")
TTBAR_SCORES = Path("outputs/ttbar_classifier/scores.npz")
TTBAR_WORKING_POINTS = Path("outputs/ttbar_classifier/working_points.csv")
TTBAR_SEED_STABILITY = Path("outputs/ttbar_classifier/seed_stability_summary.json")
ZBB_SCORES = Path("outputs/zbb_rejection/scores.npz")
ZBB_WORKING_POINTS = Path("outputs/zbb_rejection/working_points.csv")
ZBB_METRICS = Path("outputs/zbb_rejection/metrics.csv")

OUTDIR = Path("outputs/resampling")
PLOT_DIR = Path("outputs/plots/resampling")
SUMMARY_MD = Path("RESULTS_RESAMPLING.md")

TARGET_SIGNAL_EFFS = [0.90, 0.70, 0.50]
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_SEED = 20260616

OUTDIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(name):
    '''
    Converts a metric name into a safe filename by replacing special characters with underscores and removing leading/trailing underscores.
     For example, "ttbar_bdt_wp90_ttbar_rejection" would become "ttbar_bdt_wp90_ttbar_rejection", and "zbb_full_bdt_auc"
    '''
    name = name.replace("~", "approx")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_")


def finite_percentile(values, q):
    '''
    Computes the q-th percentile of the finite values in the input array. If there are no finite values, returns NaN.
     For example, if values = [0.5, 0.7, inf, nan] and q = 50, it would compute the median of the finite values [0.5, 0.7] and return 0.6. If values = [inf, nan], it would return NaN.
     This is used to compute percentiles for bootstrap replicates while ignoring any non-finite values that may arise from certain bootstrap samples.
    '''
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, q))


def summarize_bootstrap(values):
    '''
    Summarizes the bootstrap replicates by computing the mean, standard deviation, and percentiles of the finite values.

    '''
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "n_bootstrap": int(len(values)),
            "n_finite": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p02_5": float("nan"),
            "p16": float("nan"),
            "p50": float("nan"),
            "p84": float("nan"),
            "p97_5": float("nan"),
        }

    return {
        "n_bootstrap": int(len(values)),
        "n_finite": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "p02_5": finite_percentile(finite, 2.5),
        "p16": finite_percentile(finite, 16.0),
        "p50": finite_percentile(finite, 50.0),
        "p84": finite_percentile(finite, 84.0),
        "p97_5": finite_percentile(finite, 97.5),
    }


def evaluate_auc(y, scores):
    '''
    Evaluates the AUC metric for the given true labels and predicted scores. I
    f there are fewer than 2 unique labels, returns NaN since AUC is not defined.
    '''
    labels = np.unique(y)
    if len(labels) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def evaluate_threshold(scores, y, threshold, background_name):
    '''
    Evaluates the signal efficiency, background efficiency, and background rejection at the given threshold.
     Signal efficiency is the fraction of signal events with scores above the threshold, 
     background efficiency is the fraction of background events with scores above the threshold, and
     background rejection is the inverse of background efficiency.
    '''
    sig = y == 1
    bkg = y == 0
    sig_n = int(np.sum(sig))
    bkg_n = int(np.sum(bkg))

    if sig_n == 0 or bkg_n == 0:
        return {
            "signal_eff": float("nan"),
            f"{background_name}_eff": float("nan"),
            f"{background_name}_rejection": float("nan"),
        }

    sig_eff = float(np.mean(scores[sig] >= threshold))
    bkg_eff = float(np.mean(scores[bkg] >= threshold))
    rejection = float("inf") if bkg_eff == 0 else 1.0 / bkg_eff
    return {
        "signal_eff": sig_eff,
        f"{background_name}_eff": bkg_eff,
        f"{background_name}_rejection": rejection,
    }


def load_pair_dnn_inputs():
    '''
    Loads the Pair-DNN test set predictions and nominal metrics. The predictions must include a "target_rank" column to compute the top-k accuracies.
     The nominal metrics are used to compare the bootstrap replicates against the original test-set values.
    '''
    predictions = pd.read_csv(PAIR_DNN_PREDICTIONS)
    test = predictions[predictions["split"] == "test"].copy().reset_index(drop=True)
    if "target_rank" not in test.columns:
        raise RuntimeError(
            "Pair-DNN predictions need a target_rank column to bootstrap top-k accuracies. "
            "Rerun scripts/train_pair_dnn.py with target_rank output enabled."
        )
    nominal = pd.read_csv(PAIR_DNN_METRICS)
    nominal = nominal[(nominal["split"] == "test") & (nominal["method"] == "pair_dnn")].iloc[0]
    return test, nominal


def load_ttbar_inputs():
    scores = np.load(TTBAR_SCORES, allow_pickle=True)
    y_test = scores["y_test"].astype(int) # convert from uint8 to int for boolean indexing in evaluate_threshold
    scores_test = scores["scores_test"].astype(float)
    wp = pd.read_csv(TTBAR_WORKING_POINTS)
    return y_test, scores_test, wp


def load_zbb_inputs():
    '''
    Loads the ZJetsTobb full-BDT test set scores and nominal metrics. 
    The scores must include a "scores_test_full_bdt" array to compute the AUC and working-point efficiencies.
    '''
    scores = np.load(ZBB_SCORES, allow_pickle=True)
    y_test = scores["y_test"].astype(int)
    key = "scores_test_full_bdt"
    if key not in scores.files:
        raise RuntimeError(
            "ZJetsTobb scores.npz is missing scores_test_full_bdt. "
            "Rerun scripts/run_zbb_rejection.py to save per-event full-BDT test scores."
        )
    scores_test = scores[key].astype(float)
    wp = pd.read_csv(ZBB_WORKING_POINTS)
    wp = wp[wp["method"] == "full_bdt"].copy()
    return y_test, scores_test, wp


def bootstrap_pair_dnn(rng, n_bootstrap, pair_test):
    '''
    Bootstraps the Pair-DNN test set by resampling the target ranks with replacement. 
    For each bootstrap replicate, computes the top-1, top-3, and top-5 pair accuracies.
    '''
    ranks = pair_test["target_rank"].to_numpy(dtype=int)
    n = len(ranks)
    rows = []
    for rep in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sampled = ranks[idx]
        rows.append(
            {
                "bootstrap_id": rep,
                "pair_dnn_top1_accuracy": float(np.mean(sampled <= 1)),
                "pair_dnn_top3_accuracy": float(np.mean(sampled <= 3)),
                "pair_dnn_top5_accuracy": float(np.mean(sampled <= 5)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_bdt(rng, n_bootstrap, y_test, scores_test, wp, prefix, background_name):
    '''
    Bootstraps a BDT test set by resampling the true labels and predicted scores with replacement.
        For each bootstrap replicate, computes the AUC and the signal efficiency, background efficiency, and background rejection at each validation-defined working-point threshold.
        The thresholds are taken from the provided working points DataFrame, which should have a "target_signal_eff_val" column for the target signal efficiency and a "threshold" column for the corresponding score threshold.
    '''
    n = len(y_test)
    rows = []
    thresholds = {
        float(row["target_signal_eff_val"]): float(row["threshold"])
        for _, row in wp.iterrows()
    }
    for rep in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y = y_test[idx]
        scores = scores_test[idx]
        row = {
            "bootstrap_id": rep,
            f"{prefix}_auc": evaluate_auc(y, scores),
        }
        for target in TARGET_SIGNAL_EFFS:
            threshold = thresholds[target]
            metrics = evaluate_threshold(scores, y, threshold, background_name)
            suffix = f"wp{int(round(target * 100))}"
            row[f"{prefix}_{suffix}_signal_eff"] = metrics["signal_eff"]
            row[f"{prefix}_{suffix}_{background_name}_eff"] = metrics[f"{background_name}_eff"]
            row[f"{prefix}_{suffix}_{background_name}_rejection"] = metrics[f"{background_name}_rejection"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_distributions(replicates, summary_df):
    '''
    Plots the distributions of the bootstrap replicates for each metric. Each metric gets its own histogram plot, with vertical lines indicating the nominal test value and the 16-84% interval of the bootstrap distribution.
    '''
    for _, row in summary_df.iterrows():
        metric = row["metric"]
        values = replicates[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        plt.figure(figsize=(7.0, 4.8))
        plt.hist(values, bins=40, histtype="stepfilled", alpha=0.6, edgecolor="black")
        plt.axvline(row["nominal"], color="black", linestyle="--", linewidth=1.2, label="nominal test value")
        plt.axvline(row["p16"], color="tab:orange", linestyle=":", linewidth=1.2, label="16-84% interval")
        plt.axvline(row["p84"], color="tab:orange", linestyle=":", linewidth=1.2)
        plt.xlabel(metric)
        plt.ylabel("Bootstrap replicas")
        plt.title(row["label"])
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{safe_name(metric)}.png", dpi=200)
        plt.close()


def nominal_pair_metrics(pair_nominal):
    '''
    Extracts the nominal Pair-DNN top-1, top-3, and top-5 pair accuracies from the provided nominal metrics DataFrame row.
     These nominal values are used to compare against the bootstrap replicates.
    '''
    return {
        "pair_dnn_top1_accuracy": float(pair_nominal["exact_pair_accuracy"]),
        "pair_dnn_top3_accuracy": float(pair_nominal["top3_pair_accuracy"]),
        "pair_dnn_top5_accuracy": float(pair_nominal["top5_pair_accuracy"]),
    }


def nominal_bdt_metrics(y_test, scores_test, wp, prefix, background_name):
    out = {f"{prefix}_auc": evaluate_auc(y_test, scores_test)}
    for _, row in wp.iterrows():
        target = float(row["target_signal_eff_val"])
        threshold = float(row["threshold"])
        suffix = f"wp{int(round(target * 100))}"
        metrics = evaluate_threshold(scores_test, y_test, threshold, background_name)
        out[f"{prefix}_{suffix}_signal_eff"] = metrics["signal_eff"]
        out[f"{prefix}_{suffix}_{background_name}_eff"] = metrics[f"{background_name}_eff"]
        out[f"{prefix}_{suffix}_{background_name}_rejection"] = metrics[f"{background_name}_rejection"]
    return out


def build_metric_labels():
    labels = {
        "pair_dnn_top1_accuracy": "Pair-DNN Top-1 exact pair accuracy",
        "pair_dnn_top3_accuracy": "Pair-DNN Top-3 pair accuracy",
        "pair_dnn_top5_accuracy": "Pair-DNN Top-5 pair accuracy",
        "ttbar_bdt_auc": "HH-vs-ttbar BDT test AUC",
        "zbb_full_bdt_auc": "HH-vs-ZJetsTobb full-BDT test AUC",
    }
    for prefix, background in [("ttbar_bdt", "ttbar"), ("zbb_full_bdt", "zbb")]:
        pretty_background = "ttbar" if background == "ttbar" else "ZJetsTobb"
        for target in TARGET_SIGNAL_EFFS:
            suffix = f"wp{int(round(target * 100))}"
            labels[f"{prefix}_{suffix}_signal_eff"] = f"{prefix}: HH efficiency at validation target {target:.0%}"
            labels[f"{prefix}_{suffix}_{background}_eff"] = f"{prefix}: {pretty_background} efficiency at validation target {target:.0%}"
            labels[f"{prefix}_{suffix}_{background}_rejection"] = f"{prefix}: {pretty_background} rejection at validation target {target:.0%}"
    return labels


def build_summary(replicates, nominal):
    labels = build_metric_labels()
    rows = []
    for metric in replicates.columns:
        if metric == "bootstrap_id":
            continue
        row = {
            "metric": metric,
            "label": labels.get(metric, metric),
            "nominal": float(nominal.get(metric, float("nan"))),
        }
        row.update(summarize_bootstrap(replicates[metric].to_numpy(dtype=float)))
        rows.append(row)
    return pd.DataFrame(rows)


def format_value(value, digits=4):
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def selected_summary_rows(summary_df):
    metrics = [
        "pair_dnn_top1_accuracy",
        "pair_dnn_top3_accuracy",
        "pair_dnn_top5_accuracy",
        "ttbar_bdt_auc",
        "ttbar_bdt_wp90_ttbar_rejection",
        "ttbar_bdt_wp70_ttbar_rejection",
        "ttbar_bdt_wp50_ttbar_rejection",
        "zbb_full_bdt_auc",
        "zbb_full_bdt_wp90_zbb_rejection",
        "zbb_full_bdt_wp70_zbb_rejection",
        "zbb_full_bdt_wp50_zbb_rejection",
    ]
    return summary_df[summary_df["metric"].isin(metrics)].set_index("metric").loc[metrics].reset_index()


def write_markdown(args, summary_df):
    selected = selected_summary_rows(summary_df)
    lines = [
        "# Bootstrap Test-Set Resampling",
        "",
        "This study estimates sampling uncertainty from the fixed held-out test sets by resampling test events with replacement. It does not retrain any model and does not change validation-defined working-point thresholds.",
        "",
        "## Command",
        "",
        "```bash",
        f"/tmp/hh-bbww-venv/bin/python scripts/bootstrap_resampling.py --n-bootstrap {args.n_bootstrap} --seed {args.seed}",
        "```",
        "",
        "## Inputs Used",
        "",
        "- Pair-DNN: `outputs/hbb_pair_dnn/predictions.csv`, using the test-set `target_rank` column.",
        "- HH-vs-ttbar BDT: `outputs/ttbar_classifier/scores.npz` and validation-defined thresholds in `working_points.csv`.",
        "- HH-vs-ZJetsTobb full BDT: `outputs/zbb_rejection/scores.npz` and validation-defined full-BDT thresholds in `working_points.csv`.",
        "",
        "## Main Results",
        "",
        "| Metric | Nominal | Bootstrap mean | Std | 16-84% | 95% interval |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"| {row['label']} | {format_value(row['nominal'])} | {format_value(row['mean'])} | "
            f"{format_value(row['std'])} | {format_value(row['p16'])}-{format_value(row['p84'])} | "
            f"{format_value(row['p02_5'])}-{format_value(row['p97_5'])} |"
        )

    lines.extend(
        [
            "",
            "## How To Interpret This",
            "",
            "The existing binomial uncertainties in the reconstruction and working-point tables estimate counting uncertainty for simple pass/fail efficiencies. They are useful for quantities like pair accuracy or a single signal/background efficiency.",
            "",
            "The ttbar BDT random-seed stability study varies the BDT `random_state` on fixed train/validation/test feature files. That probes model-training randomness for the corrected ttbar baseline, but it does not resample events or change the data split.",
            "",
            "This bootstrap study keeps all trained models, fixed splits, and validation-selected thresholds unchanged. It asks: if the finite test set were a slightly different sample drawn from the same population, how much would the reported test metrics fluctuate? This is a test-set sampling uncertainty, not a full analysis uncertainty.",
            "",
            "A future repeated split/retraining study would be stronger and more expensive: regenerate train/validation/test splits, retrain models, reselect thresholds on each validation split, and evaluate each corresponding test split. That would mix data-split variance, training variance, and test-set sampling variance.",
            "",
            "## Outputs",
            "",
            "- `outputs/resampling/bootstrap_replicates.csv`",
            "- `outputs/resampling/bootstrap_summary.csv`",
            "- `outputs/resampling/bootstrap_summary.json`",
            "- `outputs/plots/resampling/*.png`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap fixed test-set outputs for COLLIDE-1M HH -> bbWW baseline studies.")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("Loading saved per-event outputs...")
    pair_test, pair_nominal = load_pair_dnn_inputs()
    tt_y, tt_scores, tt_wp = load_ttbar_inputs()
    z_y, z_scores, z_wp = load_zbb_inputs()

    print(f"Pair-DNN test events: {len(pair_test)}")
    print(f"ttbar BDT test events: {len(tt_y)}")
    print(f"ZJetsTobb BDT test events: {len(z_y)}")

    print(f"Building {args.n_bootstrap} bootstrap replicas with seed {args.seed}...")
    pair_boot = bootstrap_pair_dnn(rng, args.n_bootstrap, pair_test)
    tt_boot = bootstrap_bdt(rng, args.n_bootstrap, tt_y, tt_scores, tt_wp, "ttbar_bdt", "ttbar")
    z_boot = bootstrap_bdt(rng, args.n_bootstrap, z_y, z_scores, z_wp, "zbb_full_bdt", "zbb")

    replicates = pair_boot.merge(tt_boot, on="bootstrap_id").merge(z_boot, on="bootstrap_id")

    nominal = {}
    nominal.update(nominal_pair_metrics(pair_nominal))
    nominal.update(nominal_bdt_metrics(tt_y, tt_scores, tt_wp, "ttbar_bdt", "ttbar"))
    nominal.update(nominal_bdt_metrics(z_y, z_scores, z_wp, "zbb_full_bdt", "zbb"))

    summary_df = build_summary(replicates, nominal)

    replicates.to_csv(OUTDIR / "bootstrap_replicates.csv", index=False)
    summary_df.to_csv(OUTDIR / "bootstrap_summary.csv", index=False)
    with open(OUTDIR / "bootstrap_summary.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "nominal": nominal,
                "summary": summary_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    plot_distributions(replicates, summary_df)
    write_markdown(args, summary_df)

    print("\nSelected bootstrap summary:")
    print(selected_summary_rows(summary_df).to_string(index=False))
    print(f"\nSaved tables to {OUTDIR}")
    print(f"Saved plots to {PLOT_DIR}")
    print(f"Saved markdown summary to {SUMMARY_MD}")


if __name__ == "__main__":
    main()
