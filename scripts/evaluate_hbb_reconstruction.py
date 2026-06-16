import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NPZ_DIR = Path("outputs/hbb_npz")
OUTDIR = Path("outputs/hbb_reconstruction")
PLOT_DIR = Path("outputs/plots/hbb_reconstruction")
M_H = 125.0

OUTDIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = ["pt", "eta", "phi", "mass", "btag", "btagPhys"]
METHOD_LABELS = {
    "top2_btag": "Top-2 b-tag",
    "closest_mass": "Closest m_jj to 125 GeV",
    "combined_btag_mass": "Combined b-tag + mass",
}
METHOD_ORDER = ["top2_btag", "closest_mass", "combined_btag_mass"]


def load_split(split):
    data = np.load(NPZ_DIR / f"{split}.npz")
    return data["jets"], data["mask"], data["labels"]


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi <= -np.pi:
        dphi += 2 * np.pi
    return dphi


def delta_r(j1, j2):
    return float(np.sqrt((j1[1] - j2[1]) ** 2 + delta_phi(j1[2], j2[2]) ** 2))


def invariant_mass(j1, j2):
    pt1, eta1, phi1, m1 = j1[:4]
    pt2, eta2, phi2, m2 = j2[:4]

    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)

    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m2_tot = e**2 - px**2 - py**2 - pz**2
    return float(np.sqrt(max(m2_tot, 0.0)))


def binomial_error(eff, n):
    if n <= 0:
        return float("nan")
    return math.sqrt(eff * (1.0 - eff) / n)


def pair_key(pair):
    return tuple(sorted((int(pair[0]), int(pair[1]))))


def ranked_pairs(jets, mask, method):
    real = np.where(mask > 0)[0]
    scored = []

    # Reproduce the legacy top-2-btag baseline exactly.  The COLLIDE btag field
    # is often binary, so ties are common; np.argsort tie-breaking changes the
    # selected pair and therefore the reported baseline accuracy.
    if method == "top2_btag":
        btags = jets[real, 4]
        order = list(real[np.argsort(btags)])
        jet_rank = {int(j): rank for rank, j in enumerate(order)}

    for i, j in itertools.combinations(real, 2):
        j1 = jets[i]
        j2 = jets[j]
        mbb = invariant_mass(j1, j2)
        dr = delta_r(j1, j2)
        btag_sum = float(j1[4] + j2[4])

        if method == "top2_btag":
            score = -(jet_rank[int(i)] + jet_rank[int(j)])
        elif method == "closest_mass":
            score = abs(mbb - M_H)
        elif method == "combined_btag_mass":
            score = abs(mbb - M_H) - 20.0 * btag_sum
        else:
            raise ValueError(f"Unknown method: {method}")

        scored.append(
            {
                "pair": pair_key((i, j)),
                "sort_score": score,
                "mbb": mbb,
                "deltaR": dr,
                "btag_sum": btag_sum,
            }
        )

    return sorted(scored, key=lambda row: row["sort_score"])


def evaluate_method(split, method, jets_all, mask_all, labels_all):
    event_rows = []
    n_events = len(labels_all)
    correct_at = {1: 0, 3: 0, 5: 0}

    for ev in range(n_events):
        truth = pair_key(labels_all[ev])
        ranking = ranked_pairs(jets_all[ev], mask_all[ev], method)
        pred = ranking[0]
        ranked_truth = [row["pair"] for row in ranking]

        for k in correct_at:
            if truth in ranked_truth[: min(k, len(ranked_truth))]:
                correct_at[k] += 1

        event_rows.append(
            {
                "split": split,
                "event": ev,
                "method": method,
                "truth_i": truth[0],
                "truth_j": truth[1],
                "pred_i": pred["pair"][0],
                "pred_j": pred["pair"][1],
                "correct": pred["pair"] == truth,
                "selected_mbb": pred["mbb"],
                "selected_deltaR": pred["deltaR"],
                "selected_btag_sum": pred["btag_sum"],
                "n_jets": int(np.sum(mask_all[ev] > 0)),
            }
        )

    selected_mbb = np.array([row["selected_mbb"] for row in event_rows], dtype=np.float64)
    metrics = {
        "split": split,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "n_events": n_events,
        "exact_pair_accuracy": correct_at[1] / n_events,
        "exact_pair_accuracy_err": binomial_error(correct_at[1] / n_events, n_events),
        "top3_pair_accuracy": correct_at[3] / n_events,
        "top3_pair_accuracy_err": binomial_error(correct_at[3] / n_events, n_events),
        "top5_pair_accuracy": correct_at[5] / n_events,
        "top5_pair_accuracy_err": binomial_error(correct_at[5] / n_events, n_events),
        "selected_mbb_mean": float(np.mean(selected_mbb)),
        "selected_mbb_median": float(np.median(selected_mbb)),
        "selected_mbb_p16": float(np.percentile(selected_mbb, 16)),
        "selected_mbb_p84": float(np.percentile(selected_mbb, 84)),
        "selected_mbb_frac_90_140": float(np.mean((selected_mbb >= 90.0) & (selected_mbb <= 140.0))),
        "selected_mbb_frac_100_150": float(np.mean((selected_mbb >= 100.0) & (selected_mbb <= 150.0))),
    }

    return metrics, event_rows


def accuracy_by_njets(event_df):
    rows = []
    for (split, method, n_jets), group in event_df.groupby(["split", "method", "n_jets"]):
        n = len(group)
        acc = float(np.mean(group["correct"]))
        rows.append(
            {
                "split": split,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_jets": int(n_jets),
                "n_events": n,
                "exact_pair_accuracy": acc,
                "exact_pair_accuracy_err": binomial_error(acc, n),
            }
        )
    return pd.DataFrame(rows)


def plot_test_accuracy(metrics_df):
    test = metrics_df[metrics_df["split"] == "test"].copy()
    test["method"] = pd.Categorical(test["method"], categories=METHOD_ORDER, ordered=True)
    test = test.sort_values("method")

    x = np.arange(len(test))
    width = 0.25

    plt.figure(figsize=(8.0, 4.8))
    for offset, col, label in [
        (-width, "exact_pair_accuracy", "Top-1"),
        (0.0, "top3_pair_accuracy", "Top-3"),
        (width, "top5_pair_accuracy", "Top-5"),
    ]:
        err_col = col + "_err"
        plt.bar(x + offset, test[col], width=width, yerr=test[err_col], capsize=3, label=label)

    plt.xticks(x, test["method_label"], rotation=20, ha="right")
    plt.ylabel("Truth-pair efficiency / accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("H->bb pair-assignment baseline comparison on test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "test_accuracy_comparison.png", dpi=200)
    plt.close()


def plot_test_mbb(event_df):
    test = event_df[event_df["split"] == "test"]
    bins = np.linspace(0, 250, 70)

    plt.figure(figsize=(7.5, 5.0))
    for method in METHOD_ORDER:
        vals = test.loc[test["method"] == method, "selected_mbb"]
        plt.hist(vals, bins=bins, histtype="step", linewidth=1.5, density=True, label=METHOD_LABELS[method])
    plt.axvline(M_H, color="black", linestyle="--", linewidth=1.0, label="125 GeV")
    plt.xlabel("Selected pair $m_{jj}$ [GeV]")
    plt.ylabel("Normalized events")
    plt.title("Selected H->bb candidate mass on test set")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "test_selected_mbb_comparison.png", dpi=200)
    plt.close()


def plot_accuracy_by_njets(acc_df):
    test = acc_df[acc_df["split"] == "test"]

    plt.figure(figsize=(7.5, 5.0))
    for method in METHOD_ORDER:
        rows = test[test["method"] == method].sort_values("n_jets")
        plt.errorbar(
            rows["n_jets"],
            rows["exact_pair_accuracy"],
            yerr=rows["exact_pair_accuracy_err"],
            marker="o",
            linewidth=1.4,
            capsize=2,
            label=METHOD_LABELS[method],
        )
    plt.xlabel("Number of retained AK4 jets")
    plt.ylabel("Exact pair accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("H->bb pair-assignment accuracy vs jet multiplicity")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "test_accuracy_vs_njets.png", dpi=200)
    plt.close()


def write_summary(metrics_df):
    test = metrics_df[metrics_df["split"] == "test"].copy()
    test["method"] = pd.Categorical(test["method"], categories=METHOD_ORDER, ordered=True)
    test = test.sort_values("method")

    lines = [
        "# H->bb Reconstruction Baseline Comparison",
        "",
        "This summary is generated by `scripts/evaluate_hbb_reconstruction.py` from the fixed-size HH -> bbWW arrays in `outputs/hbb_npz/`.",
        "",
        "Input format:",
        "",
        "- `jets`: event x jet x feature, shape `(N, 12, 6)`",
        "- `mask`: event x jet, shape `(N, 12)`",
        "- `labels`: event x 2 truth-matched H -> bb AK4 jet indices",
        "- jet features: `[pt, eta, phi, mass, btag, btagPhys]`",
        "",
        "## Test-set comparison",
        "",
        "| Method | Top-1 accuracy | Top-3 accuracy | Top-5 accuracy | Median selected m_jj | 90 < m_jj < 140 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in test.iterrows():
        lines.append(
            f"| {row['method_label']} | "
            f"{row['exact_pair_accuracy']:.4f} +/- {row['exact_pair_accuracy_err']:.4f} | "
            f"{row['top3_pair_accuracy']:.4f} +/- {row['top3_pair_accuracy_err']:.4f} | "
            f"{row['top5_pair_accuracy']:.4f} +/- {row['top5_pair_accuracy_err']:.4f} | "
            f"{row['selected_mbb_median']:.2f} GeV | "
            f"{row['selected_mbb_frac_90_140']:.3f} |"
        )

    lines.extend(
        [
            "",
            "The closest-to-125 mass baseline has a Higgs-like selected mass distribution by construction, but its truth-pair accuracy is low. For learned reconstruction, truth-pair accuracy should be the primary metric and selected m_jj should be treated as a diagnostic distribution.",
            "",
            "## Plots",
            "",
            "- `outputs/plots/hbb_reconstruction/test_accuracy_comparison.png`",
            "- `outputs/plots/hbb_reconstruction/test_selected_mbb_comparison.png`",
            "- `outputs/plots/hbb_reconstruction/test_accuracy_vs_njets.png`",
            "",
            "Future learned models, including the pair-DNN baseline, SPA-Net, and jet-embedded SPA-Net, should be added to this same comparison table using the same train/validation/test split and truth-pair metrics.",
        ]
    )

    Path("RESULTS_HBB_RECONSTRUCTION_COMPARISON.md").write_text("\n".join(lines) + "\n")


def main():
    metrics = []
    event_rows = []

    dataset_summary = {}
    for split in ["train", "val", "test"]:
        jets, mask, labels = load_split(split)
        dataset_summary[split] = {
            "jets_shape": list(jets.shape),
            "mask_shape": list(mask.shape),
            "labels_shape": list(labels.shape),
            "mean_n_jets": float(mask.sum(axis=1).mean()),
            "min_n_jets": int(mask.sum(axis=1).min()),
            "max_n_jets": int(mask.sum(axis=1).max()),
        }
        for method in METHOD_ORDER:
            row, rows = evaluate_method(split, method, jets, mask, labels)
            metrics.append(row)
            event_rows.extend(rows)

    metrics_df = pd.DataFrame(metrics)
    event_df = pd.DataFrame(event_rows)
    acc_njets_df = accuracy_by_njets(event_df)

    metrics_df.to_csv(OUTDIR / "baseline_metrics.csv", index=False)
    event_df.to_csv(OUTDIR / "baseline_event_predictions.csv", index=False)
    acc_njets_df.to_csv(OUTDIR / "baseline_accuracy_by_njets.csv", index=False)

    with open(OUTDIR / "hbb_npz_format_summary.json", "w") as f:
        json.dump(
            {
                "feature_names": FEATURE_NAMES,
                "max_jets": 12,
                "splits": dataset_summary,
            },
            f,
            indent=2,
        )

    plot_test_accuracy(metrics_df)
    plot_test_mbb(event_df)
    plot_accuracy_by_njets(acc_njets_df)
    write_summary(metrics_df)

    print("H->bb reconstruction baseline comparison")
    print("----------------------------------------")
    print(metrics_df[metrics_df["split"] == "test"].to_string(index=False))
    print(f"\nSaved tables to {OUTDIR}")
    print(f"Saved plots to {PLOT_DIR}")
    print("Saved markdown summary to RESULTS_HBB_RECONSTRUCTION_COMPARISON.md")


if __name__ == "__main__":
    main()
