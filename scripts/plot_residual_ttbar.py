from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path("outputs/ttbar_classifier")
PLOT_DIR = Path("outputs/plots/ttbar_classifier/residual_ttbar")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_scores():
    data = np.load(DATA_DIR / "scores.npz", allow_pickle=True)
    return {
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "scores_test": data["scores_test"],
        "feature_names": [str(x) for x in data["feature_names"]],
    }


def get_threshold(target=0.70):
    wp = pd.read_csv(DATA_DIR / "working_points.csv")
    idx = (wp["target_signal_eff_val"] - target).abs().idxmin()
    row = wp.loc[idx]
    return float(row["threshold"]), row


def save_overlay(hh_values, tt_values, residual_values, name):
    plt.figure()
    plt.hist(
        hh_values,
        bins=50,
        histtype="step",
        linewidth=1.5,
        density=True,
        label="HH test",
    )
    plt.hist(
        tt_values,
        bins=50,
        histtype="step",
        linewidth=1.5,
        density=True,
        label="all ttbar test",
    )
    plt.hist(
        residual_values,
        bins=50,
        histtype="step",
        linewidth=1.5,
        density=True,
        label="residual ttbar",
    )

    plt.xlabel(name)
    plt.ylabel("Normalized events")
    plt.title(f"Residual ttbar topology: {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{name}.png", dpi=200)
    plt.close()


def main():
    data = load_scores()

    X_test = data["X_test"]
    y_test = data["y_test"]
    scores = data["scores_test"]
    feature_names = data["feature_names"]

    threshold, wp_row = get_threshold(target=0.70)

    hh = y_test == 1
    tt = y_test == 0
    residual_tt = tt & (scores >= threshold)

    print("Using threshold from validation target signal efficiency 0.70:")
    print(wp_row.to_string())
    print()
    print(f"HH test events: {np.sum(hh)}")
    print(f"ttbar test events: {np.sum(tt)}")
    print(f"residual ttbar events passing cut: {np.sum(residual_tt)}")
    print(f"residual ttbar fraction: {np.sum(residual_tt) / np.sum(tt):.4f}")

    for i, name in enumerate(feature_names):
        save_overlay(
            X_test[hh, i],
            X_test[tt, i],
            X_test[residual_tt, i],
            name,
        )

    summary = {
        "target_signal_eff_val": 0.70,
        "threshold": threshold,
        "hh_test_events": int(np.sum(hh)),
        "ttbar_test_events": int(np.sum(tt)),
        "residual_ttbar_events": int(np.sum(residual_tt)),
        "residual_ttbar_fraction": float(np.sum(residual_tt) / np.sum(tt)),
    }

    pd.DataFrame([summary]).to_csv(DATA_DIR / "residual_ttbar_summary.csv", index=False)

    print(f"\nSaved residual plots to {PLOT_DIR}")
    print(f"Saved residual summary to {DATA_DIR / 'residual_ttbar_summary.csv'}")


if __name__ == "__main__":
    main()
