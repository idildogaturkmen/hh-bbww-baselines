import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score


DATA_DIR = Path("outputs/ttbar_classifier")
PLOT_DIR = Path("outputs/plots/ttbar_classifier")
FEATURE_OVERLAY_DIR = PLOT_DIR / "feature_overlays"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = DATA_DIR / "bdt_model.joblib"


def load_split(split):
    data = np.load(DATA_DIR / f"features_{split}.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_names = [str(x) for x in data["feature_names"]]
    sample = data["sample"]
    return X, y, feature_names, sample


def binomial_error(eff, n):
    if n <= 0:
        return float("nan")
    return math.sqrt(eff * (1.0 - eff) / n)


def plot_feature_overlays(X_train, y_train, feature_names):
    for i, name in enumerate(feature_names):
        hh = X_train[y_train == 1, i]
        tt = X_train[y_train == 0, i]

        plt.figure()
        plt.hist(hh, bins=50, histtype="step", linewidth=1.5, density=True, label="HH → bbWW")
        plt.hist(tt, bins=50, histtype="step", linewidth=1.5, density=True, label="semileptonic ttbar")
        plt.xlabel(name)
        plt.ylabel("Normalized events")
        plt.title(f"Input feature: {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FEATURE_OVERLAY_DIR / f"{name}.png", dpi=200)
        plt.close()


def plot_roc(y_test, scores_test, auc_test):
    fpr, tpr, _ = roc_curve(y_test, scores_test)

    plt.figure()
    plt.plot(fpr, tpr, linewidth=1.5, label=f"Test AUC = {auc_test:.3f}")
    plt.xlabel("ttbar efficiency")
    plt.ylabel("HH signal efficiency")
    plt.title("HH → bbWW vs semileptonic ttbar ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc_curve_test.png", dpi=200)
    plt.close()

    rejection = np.divide(1.0, fpr, out=np.full_like(fpr, np.inf), where=fpr > 0)

    plt.figure()
    plt.plot(tpr, rejection, linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("HH signal efficiency")
    plt.ylabel("ttbar rejection = 1 / ttbar efficiency")
    plt.title("Background rejection vs signal efficiency")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "background_rejection_test.png", dpi=200)
    plt.close()


def plot_score_distributions(y_train, s_train, y_test, s_test):
    plt.figure()

    plt.hist(s_train[y_train == 1], bins=50, histtype="step", linewidth=1.3, density=True, label="HH train")
    plt.hist(s_train[y_train == 0], bins=50, histtype="step", linewidth=1.3, density=True, label="ttbar train")
    plt.hist(s_test[y_test == 1], bins=50, histtype="step", linewidth=1.3, density=True, linestyle="--", label="HH test")
    plt.hist(s_test[y_test == 0], bins=50, histtype="step", linewidth=1.3, density=True, linestyle="--", label="ttbar test")

    plt.xlabel("BDT score")
    plt.ylabel("Normalized events")
    plt.title("BDT score distributions: train vs test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "score_distribution_train_test.png", dpi=200)
    plt.close()


def choose_threshold_for_signal_eff(scores_val, y_val, target_eff):
    sig_scores = scores_val[y_val == 1]
    # Events pass if score >= threshold.
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
        "threshold": threshold,
        "signal_pass": sig_pass,
        "signal_total": sig_n,
        "ttbar_pass": bkg_pass,
        "ttbar_total": bkg_n,
        "signal_eff": sig_eff,
        "signal_eff_err": binomial_error(sig_eff, sig_n),
        "ttbar_eff": bkg_eff,
        "ttbar_eff_err": binomial_error(bkg_eff, bkg_n),
        "ttbar_rejection": bkg_rej,
    }


def make_working_points(scores_val, y_val, scores_test, y_test):
    rows = []

    for target in [0.90, 0.70, 0.50]:
        threshold = choose_threshold_for_signal_eff(scores_val, y_val, target)
        row = evaluate_threshold(scores_test, y_test, threshold)
        row["target_signal_eff_val"] = target
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[
        [
            "target_signal_eff_val",
            "threshold",
            "signal_pass",
            "signal_total",
            "signal_eff",
            "signal_eff_err",
            "ttbar_pass",
            "ttbar_total",
            "ttbar_eff",
            "ttbar_eff_err",
            "ttbar_rejection",
        ]
    ]
    df.to_csv(DATA_DIR / "working_points.csv", index=False)
    return df


def save_feature_importance(model, feature_names):
    imp = model.feature_importances_

    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values("importance", ascending=False)
    df.to_csv(DATA_DIR / "feature_importance.csv", index=False)

    plt.figure()
    plt.barh(df["feature"][::-1], df["importance"][::-1])
    plt.xlabel("Feature importance")
    plt.title("BDT feature importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "feature_importance.png", dpi=200)
    plt.close()

    return df


def main():
    X_train, y_train, feature_names, _ = load_split("train")
    X_val, y_val, _, _ = load_split("val")
    X_test, y_test, _, _ = load_split("test")

    print("Dataset sizes:")
    print(f"  train: {X_train.shape}, HH={np.sum(y_train == 1)}, ttbar={np.sum(y_train == 0)}")
    print(f"  val:   {X_val.shape}, HH={np.sum(y_val == 1)}, ttbar={np.sum(y_val == 0)}")
    print(f"  test:  {X_test.shape}, HH={np.sum(y_test == 1)}, ttbar={np.sum(y_test == 0)}")

    print("\nMaking feature overlay plots...")
    plot_feature_overlays(X_train, y_train, feature_names)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )

    print("Training BDT...")
    model.fit(X_train, y_train)

    scores_train = model.predict_proba(X_train)[:, 1]
    scores_val = model.predict_proba(X_val)[:, 1]
    scores_test = model.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, scores_train)
    auc_val = roc_auc_score(y_val, scores_val)
    auc_test = roc_auc_score(y_test, scores_test)

    print(f"AUC train: {auc_train:.4f}")
    print(f"AUC val:   {auc_val:.4f}")
    print(f"AUC test:  {auc_test:.4f}")

    plot_roc(y_test, scores_test, auc_test)
    plot_score_distributions(y_train, scores_train, y_test, scores_test)

    wp = make_working_points(scores_val, y_val, scores_test, y_test)
    print("\nWorking points evaluated on test:")
    print(wp.to_string(index=False))

    imp = save_feature_importance(model, feature_names)
    print("\nFeature importance:")
    print(imp.to_string(index=False))

    joblib.dump(model, MODEL_PATH)

    np.savez(
        DATA_DIR / "scores.npz",
        X_train=X_train,
        y_train=y_train,
        scores_train=scores_train,
        X_val=X_val,
        y_val=y_val,
        scores_val=scores_val,
        X_test=X_test,
        y_test=y_test,
        scores_test=scores_test,
        feature_names=np.array(feature_names),
    )

    metrics = {
        "auc_train": float(auc_train),
        "auc_val": float(auc_val),
        "auc_test": float(auc_test),
        "model": "GradientBoostingClassifier",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.8,
        "random_state": 42,
    }

    with open(DATA_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics and plots to {DATA_DIR} and {PLOT_DIR}")


if __name__ == "__main__":
    main()
