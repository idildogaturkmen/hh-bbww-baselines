import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


DATA_DIR = Path("outputs/ttbar_classifier")
PLOT_DIR = Path("outputs/plots/ttbar_classifier")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2024, 2025, 2026]
TARGET_SIGNAL_EFFS = [0.90, 0.70, 0.50]

MODEL_CONFIG = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
}


def load_split(split):
    data = np.load(DATA_DIR / f"features_{split}.npz", allow_pickle=True)
    return data["X"], data["y"]


def binomial_error(eff, n):
    if n <= 0:
        return float("nan")
    return math.sqrt(eff * (1.0 - eff) / n)


def choose_threshold_for_signal_eff(scores_val, y_val, target_eff):
    sig_scores = scores_val[y_val == 1]
    return float(np.quantile(sig_scores, 1.0 - target_eff))


def evaluate_threshold(scores, y, threshold):
    sig = y == 1
    bkg = y == 0

    sig_n = int(np.sum(sig))
    bkg_n = int(np.sum(bkg))

    sig_pass = int(np.sum(scores[sig] >= threshold))
    bkg_pass = int(np.sum(scores[bkg] >= threshold))

    sig_eff = sig_pass / sig_n if sig_n else float("nan")
    bkg_eff = bkg_pass / bkg_n if bkg_n else float("nan")
    bkg_rej = float("inf") if bkg_eff == 0 else 1.0 / bkg_eff

    return {
        "threshold": float(threshold),
        "signal_pass": sig_pass,
        "signal_total": sig_n,
        "signal_eff": float(sig_eff),
        "signal_eff_err": binomial_error(sig_eff, sig_n),
        "ttbar_pass": bkg_pass,
        "ttbar_total": bkg_n,
        "ttbar_eff": float(bkg_eff),
        "ttbar_eff_err": binomial_error(bkg_eff, bkg_n),
        "ttbar_rejection": float(bkg_rej),
    }


def train_one_seed(seed, X_train, y_train, X_val, y_val, X_test, y_test):
    model = GradientBoostingClassifier(
        **MODEL_CONFIG,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    scores_train = model.predict_proba(X_train)[:, 1]
    scores_val = model.predict_proba(X_val)[:, 1]
    scores_test = model.predict_proba(X_test)[:, 1]

    row = {
        "seed": seed,
        "auc_train": float(roc_auc_score(y_train, scores_train)),
        "auc_val": float(roc_auc_score(y_val, scores_val)),
        "auc_test": float(roc_auc_score(y_test, scores_test)),
    }

    for target in TARGET_SIGNAL_EFFS:
        key = f"sig_eff_{int(target * 100)}"
        threshold = choose_threshold_for_signal_eff(scores_val, y_val, target)
        metrics = evaluate_threshold(scores_test, y_test, threshold)
        row[f"threshold_at_{key}_val"] = metrics["threshold"]
        row[f"test_signal_eff_at_{key}_val"] = metrics["signal_eff"]
        row[f"test_signal_eff_err_at_{key}_val"] = metrics["signal_eff_err"]
        row[f"test_ttbar_eff_at_{key}_val"] = metrics["ttbar_eff"]
        row[f"test_ttbar_eff_err_at_{key}_val"] = metrics["ttbar_eff_err"]
        row[f"test_ttbar_rejection_at_{key}_val"] = metrics["ttbar_rejection"]

    return row


def summarize(df):
    summary = {
        "seeds": [int(x) for x in df["seed"]],
        "model": "GradientBoostingClassifier",
        "model_config": MODEL_CONFIG,
        "note": (
            "This scan varies the BDT random_state on fixed train/validation/test "
            "feature files. Regenerate the feature files first if preprocessing changes."
        ),
    }

    metric_columns = [
        "auc_train",
        "auc_val",
        "auc_test",
    ]
    for target in TARGET_SIGNAL_EFFS:
        key = f"sig_eff_{int(target * 100)}"
        metric_columns.append(f"test_ttbar_rejection_at_{key}_val")

    for col in metric_columns:
        values = df[col].to_numpy(dtype=float)
        summary[col] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return summary


def plot_stability(df):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)

    axes[0].plot(df["seed"], df["auc_train"], marker="o", label="train")
    axes[0].plot(df["seed"], df["auc_val"], marker="o", label="validation")
    axes[0].plot(df["seed"], df["auc_test"], marker="o", label="test")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("BDT AUC stability across model random seeds")
    axes[0].legend()

    for target in TARGET_SIGNAL_EFFS:
        key = f"sig_eff_{int(target * 100)}"
        col = f"test_ttbar_rejection_at_{key}_val"
        axes[1].plot(df["seed"], df[col], marker="o", label=f"{target:.0%} signal eff.")

    axes[1].set_yscale("log")
    axes[1].set_xlabel("BDT random seed")
    axes[1].set_ylabel("Test ttbar rejection")
    axes[1].set_title("Working-point rejection stability")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "seed_stability.png", dpi=200)
    plt.close(fig)


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    print("Dataset sizes:")
    print(f"  train: {X_train.shape}, HH={np.sum(y_train == 1)}, ttbar={np.sum(y_train == 0)}")
    print(f"  val:   {X_val.shape}, HH={np.sum(y_val == 1)}, ttbar={np.sum(y_val == 0)}")
    print(f"  test:  {X_test.shape}, HH={np.sum(y_test == 1)}, ttbar={np.sum(y_test == 0)}")

    rows = []
    for seed in SEEDS:
        print(f"Training seed {seed}...", flush=True)
        rows.append(train_one_seed(seed, X_train, y_train, X_val, y_val, X_test, y_test))

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "seed_stability.csv", index=False)

    summary = summarize(df)
    with open(DATA_DIR / "seed_stability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_stability(df)

    print("\nSeed stability results:")
    print(df.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved table to {DATA_DIR / 'seed_stability.csv'}")
    print(f"Saved summary to {DATA_DIR / 'seed_stability_summary.json'}")
    print(f"Saved plot to {PLOT_DIR / 'seed_stability.png'}")


if __name__ == "__main__":
    main()
