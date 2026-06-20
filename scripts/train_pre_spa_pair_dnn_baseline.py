'''
Train pair-level neural network classifiers to select the correct b-jet pair for H→bb reconstruction, using the same candidate pair table and event split as the physics/BDT baselines. 
This serves as a strong non-physics baseline for comparison against SPA-Net.

Usage:
    python scripts/train_pre_spa_pair_dnn_baseline.py \
        --pairs-csv outputs/pairing_with_bjet_correction/all_candidate_pairs.csv \
        --outdir outputs/pre_spa_pair_baselines \
        --plot-dir outputs/plots/pre_spa_pair_baselines \
        --epochs 120 \
        --patience 15 \
        --batch-size 4096 \
        --hidden 128 \
        --dropout 0.10 \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --n-boot 500

'''
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


M_H = 125.0


def width68(values):
    '''
    Compute the width of the central 68% interval of the values, ignoring NaNs and infinities.
    '''
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return 0.5 * (np.percentile(values, 84) - np.percentile(values, 16))


def summarize_arrays(correct, selected_mbb):
    '''
    Summarize the selected pairs with various metrics, given arrays of correctness and selected m_bb values.
    '''
    correct = np.asarray(correct).astype(bool)
    values = np.asarray(selected_mbb, dtype=float)
    return {
        "n_events": int(len(values)),
        "pair_accuracy": float(np.mean(correct)),
        "mbb_mean": float(np.mean(values)),
        "mbb_median": float(np.median(values)),
        "mbb_p16": float(np.percentile(values, 16)),
        "mbb_p84": float(np.percentile(values, 84)),
        "mbb_width68": float(width68(values)),
        "frac_90_140": float(np.mean((values >= 90.0) & (values <= 140.0))),
        "frac_100_150": float(np.mean((values >= 100.0) & (values <= 150.0))),
        "median_abs_offset_from_125": float(abs(np.median(values) - M_H)),
    }


def summarize_selected(df):
    return summarize_arrays(df["correct"].to_numpy(), df["selected_mbb"].to_numpy())


def bootstrap_metrics(selected_df, n_boot=500, seed=12345):
    '''
    Perform bootstrap resampling to estimate uncertainties on the metrics of the selected pairs.
    '''
    rng = np.random.default_rng(seed)
    correct = selected_df["correct"].to_numpy(dtype=bool)
    values = selected_df["selected_mbb"].to_numpy(dtype=float)
    n = len(values)

    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rows.append(summarize_arrays(correct[idx], values[idx]))

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
    '''
    Split the DataFrame into train/val/test sets based on event_id modulo 5, ensuring all pairs from the same event are in the same split.
    '''
    mod = df["event_id"] % 5
    df = df.copy()
    df["split"] = np.where(mod == 0, "test", np.where(mod == 1, "val", "train"))
    return df


def add_pair_features(df):
    '''
    Add additional features to the candidate pair DataFrame, such as max/min b-tag, pt balance, mean/max/min correction, and approximate number of jets.
    '''
    df = df.copy()

    df["max_btag"] = df[["btag_a", "btag_b"]].max(axis=1)
    df["min_btag"] = df[["btag_a", "btag_b"]].min(axis=1)
    df["max_pt"] = df[["pt_a", "pt_b"]].max(axis=1)
    df["min_pt"] = df[["pt_a", "pt_b"]].min(axis=1)
    df["pt_balance"] = df["min_pt"] / np.maximum(df["max_pt"], 1e-6)

    df["mean_corr"] = 0.5 * (df["corr_a"] + df["corr_b"])
    df["max_corr"] = df[["corr_a", "corr_b"]].max(axis=1)
    df["min_corr"] = df[["corr_a", "corr_b"]].min(axis=1)

    n_pairs = df.groupby("event_id")["event_id"].transform("count").astype(float)
    df["n_pairs"] = n_pairs
    df["approx_n_jets"] = 0.5 * (1.0 + np.sqrt(1.0 + 8.0 * n_pairs))

    return df


def choose_by_score(df, score_col, selected_mbb_col):
    '''
    For each event, select the pair with the highest score, and return a DataFrame with one row per event containing the event_id, whether the selected pair is correct, and the selected m_bb value.
    '''
    idx = df.groupby("event_id")[score_col].idxmax()
    chosen = df.loc[idx].copy()
    chosen["correct"] = chosen["is_truth_pair"].astype(bool)
    chosen["selected_mbb"] = chosen[selected_mbb_col]
    return chosen[["event_id", "correct", "selected_mbb"]]

# Define the pair-level Multilayer Perceptron (MLP) model. 
class PairMLP(nn.Module):
    def __init__(self, n_features, hidden=128, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def standardize_train_val_test(train_df, val_df, test_df, feature_cols):
    '''
    Standardize the feature columns of the train/val/test DataFrames using the mean and std from the training set. 
    Returns the standardized feature arrays and the mean/std values.
    '''
    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std,
        mean,
        std,
    )

# Train the pair-level DNN model on the training set, using the validation set for early stopping. 
# Returns the best model, mean/std for standardization, training history, and best validation metrics.
def train_pair_dnn(
    train_df,
    val_df,
    feature_cols,
    hidden=128,
    dropout=0.10,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=4096,
    epochs=120,
    patience=15,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_val, _, mean, std = standardize_train_val_test(
        train_df, val_df, val_df, feature_cols
    )

    y_train = train_df["is_truth_pair"].astype(int).to_numpy(dtype=np.float32)
    y_val = val_df["is_truth_pair"].astype(int).to_numpy(dtype=np.float32)

    model = PairMLP(len(feature_cols), hidden=hidden, dropout=dropout).to(device)

    n_pos = max(float(np.sum(y_train == 1)), 1.0)
    n_neg = max(float(np.sum(y_train == 0)), 1.0)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    n_train = len(X_train_t)
    best_val = np.inf
    best_state = None
    best_epoch = -1
    wait = 0
    history = []

    rng = np.random.default_rng(12345)

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n_train)
        train_losses = []

        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(loss_fn(val_logits, y_val_t).detach().cpu())

            val_prob = torch.sigmoid(val_logits).detach().cpu().numpy()
            try:
                val_auc = float(roc_auc_score(y_val, val_prob))
            except Exception:
                val_auc = np.nan
            try:
                val_ap = float(average_precision_score(y_val, val_prob))
            except Exception:
                val_ap = np.nan

        train_loss = float(np.mean(train_losses))
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_average_precision": val_ap,
        })

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} "
            f"val_auc={val_auc:.5f} "
            f"val_ap={val_ap:.5f}",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
                break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    return model, mean, std, pd.DataFrame(history), {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val),
    }


def predict_pair_dnn(model, df, feature_cols, mean, std, batch_size=8192, device=None):
    '''
    Use the trained pair DNN model to predict scores for the candidate pairs in the given DataFrame.
    '''
    if device is None:
        device = next(model.parameters()).device

    X = df[feature_cols].to_numpy(dtype=np.float32)
    X = (X - mean) / std

    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            preds.append(prob)

    return np.concatenate(preds)


def evaluate_dnn_scores(df, score, selected_mbb_col, name):
    '''
    Evaluate the pair DNN scores by computing AUC and average precision, and selecting pairs based on the scores to compute selection metrics.
    '''
    work = df.copy()
    y = work["is_truth_pair"].astype(int).to_numpy()
    work["score"] = score

    try:
        auc = float(roc_auc_score(y, score))
    except Exception:
        auc = np.nan

    try:
        ap = float(average_precision_score(y, score))
    except Exception:
        ap = np.nan

    selected = choose_by_score(work, "score", selected_mbb_col)
    selected["selector"] = name

    return selected, auc, ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-csv", default="outputs/pairing_with_bjet_correction/all_candidate_pairs.csv")
    parser.add_argument("--outdir", default="outputs/pre_spa_pair_baselines")
    parser.add_argument("--plot-dir", default="outputs/plots/pre_spa_pair_baselines")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    plot_dir = Path(args.plot_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(args.pairs_csv)
    df = add_pair_features(df)
    df = split_by_event_id(df)

    print("Loaded candidate pairs:", len(df))
    print("Events:", df["event_id"].nunique())
    print(df.groupby("split")["event_id"].nunique())

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

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

    metric_rows = []
    selected_outputs = []
    curve_rows = []

    for model_name, feature_cols, selected_mbb_col in [
        ("pair_dnn_uncorrected_features", features_uncorrected, "mbb_uncorrected"),
        ("pair_dnn_corrected_features", features_corrected, "mbb_corrected"),
    ]:
        print(f"\nTraining {model_name}...")

        model, mean, std, history, model_info = train_pair_dnn(
            train_df,
            val_df,
            feature_cols,
            hidden=args.hidden,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            device=device,
        )

        history["model"] = model_name
        history.to_csv(outdir / f"{model_name}_training_history.csv", index=False)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_cols": feature_cols,
                "mean": mean,
                "std": std,
                "model_info": model_info,
                "args": vars(args),
            },
            outdir / f"{model_name}.pt",
        )

        score = predict_pair_dnn(model, test_df, feature_cols, mean, std, device=device)
        selected, auc, ap = evaluate_dnn_scores(test_df, score, selected_mbb_col, model_name)

        selected_outputs.append(selected)
        metric_rows.append({
            "selector": model_name,
            "lambda": np.nan,
            "pair_auc": auc,
            "pair_average_precision": ap,
            **model_info,
            **summarize_selected(selected),
            **bootstrap_metrics(selected, n_boot=args.n_boot),
        })

        score_df = test_df[
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
        score_df["score"] = score
        score_df["model"] = model_name
        curve_rows.append(score_df)

    dnn_metrics = pd.DataFrame(metric_rows)
    dnn_selected = pd.concat(selected_outputs, ignore_index=True)
    dnn_scores = pd.concat(curve_rows, ignore_index=True)

    dnn_metrics.to_csv(outdir / "pair_dnn_test_metrics_with_bootstrap.csv", index=False)
    dnn_selected.to_csv(outdir / "pair_dnn_selected_pairs_test.csv", index=False)
    dnn_scores.to_csv(outdir / "pair_dnn_test_pair_scores.csv", index=False)

    # Combine with existing BDT/physics metrics if available.
    existing_metrics_path = outdir / "test_metrics_with_bootstrap.csv"
    if existing_metrics_path.exists():
        existing = pd.read_csv(existing_metrics_path)
        combined = pd.concat([existing, dnn_metrics], ignore_index=True)
    else:
        combined = dnn_metrics.copy()

    combined.to_csv(outdir / "combined_pre_spa_metrics_with_pair_dnn.csv", index=False)

    # Combined markdown summary.
    lines = []
    lines.append("# Pre-SPA-Net H→bb Pair Baselines with Pair-DNN")
    lines.append("")
    lines.append("This comparison adds pair-level neural-network classifiers to the existing physics and BDT pre-SPA-Net baselines.")
    lines.append("")
    lines.append("## Test metrics")
    lines.append("")
    lines.append("| Selector | Events | Pair accuracy | Acc. err | m_bb median [GeV] | m_bb width68 [GeV] | Frac 90-140 | Frac 100-150 | Pair AUC | Pair AP |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in combined.iterrows():
        auc = r.get("pair_auc", np.nan)
        ap = r.get("pair_average_precision", np.nan)
        auc_text = "" if pd.isna(auc) else f"{auc:.4f}"
        ap_text = "" if pd.isna(ap) else f"{ap:.4f}"

        lines.append(
            f"| {r['selector']} | "
            f"{int(r['n_events'])} | "
            f"{r['pair_accuracy']:.4f} | "
            f"{r.get('pair_accuracy_err', np.nan):.4f} | "
            f"{r['mbb_median']:.2f} | "
            f"{r['mbb_width68']:.2f} | "
            f"{r['frac_90_140']:.4f} | "
            f"{r['frac_100_150']:.4f} | "
            f"{auc_text} | "
            f"{ap_text} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `pair_dnn_uncorrected_features` is the neural-network analogue of the uncorrected-feature BDT.")
    lines.append("- `pair_dnn_corrected_features` is the neural-network analogue of the corrected-feature BDT.")
    lines.append("- The fair comparison is BDT corrected features vs Pair-DNN corrected features, since both use the same candidate-pair table and deterministic event split.")
    lines.append("- SPA-Net should later be compared against the strongest pre-SPA-Net method from this table.")
    lines.append("")

    summary_text = "\n".join(lines) + "\n"
    (outdir / "RESULTS_PRE_SPA_PAIR_BASELINES_WITH_DNN.md").write_text(summary_text)

    summary_dir = Path("Results Summaries")
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "RESULTS_PRE_SPA_PAIR_BASELINES_WITH_DNN.md").write_text(summary_text)

    print("\n=== Pair-DNN metrics ===")
    print(dnn_metrics.to_string(index=False))
    print("\n=== Combined metrics ===")
    print(combined.to_string(index=False))

    # Training curves.
    plt.figure(figsize=(8, 5))
    for model_name in ["pair_dnn_uncorrected_features", "pair_dnn_corrected_features"]:
        h = pd.read_csv(outdir / f"{model_name}_training_history.csv")
        plt.plot(h["epoch"], h["val_loss"], label=f"{model_name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Validation BCE loss")
    plt.title("Pair-DNN validation loss")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plot_dir / "pair_dnn_validation_loss.png", dpi=180)
    plt.close()

    # Accuracy comparison.
    df_plot = combined.sort_values("pair_accuracy")
    plt.figure(figsize=(10, 6))
    plt.barh(df_plot["selector"], df_plot["pair_accuracy"])
    plt.xlabel("Truth-pair selection accuracy")
    plt.title("Pre-SPA-Net baselines including Pair-DNN")
    plt.tight_layout()
    plt.savefig(plot_dir / "combined_pair_accuracy_with_pair_dnn.png", dpi=180)
    plt.close()

    # ROC and PR curves for BDT + Pair-DNN if all score files exist.
    y_test = test_df["is_truth_pair"].astype(int).to_numpy()

    curves = []
    bdt_scores_path = outdir / "bdt_test_pair_scores.csv"
    if bdt_scores_path.exists():
        bdt_scores = pd.read_csv(bdt_scores_path)
        merged = test_df[
            ["event_id", "jet_a_idx", "jet_b_idx", "is_truth_pair"]
        ].merge(
            bdt_scores,
            on=["event_id", "jet_a_idx", "jet_b_idx", "is_truth_pair"],
            how="left",
        )
        if not merged["bdt_uncorrected_score"].isna().any():
            curves.append(("BDT uncorrected", merged["bdt_uncorrected_score"].to_numpy()))
        if not merged["bdt_corrected_score"].isna().any():
            curves.append(("BDT corrected", merged["bdt_corrected_score"].to_numpy()))

    for model_name in ["pair_dnn_uncorrected_features", "pair_dnn_corrected_features"]:
        part = dnn_scores[dnn_scores["model"] == model_name]
        merged = test_df[
            ["event_id", "jet_a_idx", "jet_b_idx", "is_truth_pair"]
        ].merge(
            part,
            on=["event_id", "jet_a_idx", "jet_b_idx", "is_truth_pair"],
            how="left",
        )
        curves.append((model_name, merged["score"].to_numpy()))

    plt.figure(figsize=(7, 6))
    for label, score in curves:
        auc = roc_auc_score(y_test, score)
        fpr, tpr, _ = roc_curve(y_test, score)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Pair-level ROC curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plot_dir / "roc_bdt_vs_pair_dnn.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    positive_fraction = float(np.mean(y_test))
    for label, score in curves:
        ap = average_precision_score(y_test, score)
        precision, recall, _ = precision_recall_curve(y_test, score)
        plt.plot(recall, precision, label=f"{label} (AP={ap:.4f})")
    plt.axhline(positive_fraction, linestyle="--", linewidth=1.0, label=f"Positive fraction={positive_fraction:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Pair-level precision-recall curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plot_dir / "pr_bdt_vs_pair_dnn.png", dpi=180)
    plt.close()

    print(f"\nSaved outputs in: {outdir}")
    print(f"Saved plots in: {plot_dir}")


if __name__ == "__main__":
    main()