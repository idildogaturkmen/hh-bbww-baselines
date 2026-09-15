'''
This script trains a simple dense neural network (DNN) on the same continuous resolved HH4b features as BDT v2. Binary category flags are not used as DNN inputs; they are used only for evaluation. The DNN test score file includes `bdt_score` as an alias for `dnn_score` so that the existing BDT stability/resampling script can be reused.
The DNN is trained with a weighted binary cross-entropy loss using the physics weights. The training weights are balanced to give equal total weight to signal and background. The DNN is trained with early stopping based on the raw validation AUC, and the best model is saved along with the scaler and evaluation metrics.
 The script also evaluates the DNN on the test set and summarizes the results in various categories, including fixed thresholds and quantile-based selections. The best nominal DNN category is identified based on Asimov significance and other metrics, and the background composition is summarized by process group and sample.
Usage:
    python scripts/hh4b_baseline/train_hh4b_resolved_dnn_v1.py --features <path_to_features_parquet> --outdir <output_directory> [options]
Options:
    --random-state: Random seed for reproducibility (default: 12345)
    --epochs: Maximum number of training epochs (default: 80)
    --patience: Number of epochs with no improvement to wait before early stopping (default: 12)
    --batch-size: Training batch size (default: 512)
    --learning-rate: Learning rate for the optimizer (default: 1e-3)
    --weight-decay: Weight decay (L2 regularization) for the optimizer (default: 1e-4)
    --dropout: Dropout rate for the DNN (default: 0.10)
'''

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def asimov_z(s: float, b: float) -> float:
    if s <= 0 or b <= 0:
        return 0.0
    return math.sqrt(max(0.0, 2.0 * ((s + b) * math.log(1.0 + s / b) - s)))


def neff(w) -> float:
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w)]
    if len(w) == 0 or np.sum(w * w) <= 0:
        return 0.0
    return float(np.sum(w) ** 2 / np.sum(w * w))


def two_b_over_s(s: float, b: float) -> float:
    if s <= 0:
        return float("inf")
    return float(2.0 * b / s)


def summarize_selection(
    df: pd.DataFrame,
    mask,
    selection: str,
    category_type: str = "",
    score_low: float | None = None,
    score_high: float | None = None,
) -> dict:
    sub = df.loc[mask].copy()

    if len(sub) == 0:
        return {
            "selection": selection,
            "category_type": category_type,
            "score_low": score_low,
            "score_high": score_high,
            "raw_signal": 0,
            "raw_background": 0,
            "raw_background_missing_weight": 0,
            "S_w": 0.0,
            "B_w": 0.0,
            "S_over_B": 0.0,
            "S_over_SplusB": 0.0,
            "S_over_sqrtB": 0.0,
            "S_over_sqrtSplusB": 0.0,
            "asimov_Z_A": 0.0,
            "N_eff_bkg": 0.0,
            "twoB_over_S": float("inf"),
        }

    sig = sub["is_signal"].to_numpy(dtype=bool)
    bkg = ~sig

    w = sub["physics_weight"].to_numpy(dtype=float)
    finite_w = np.isfinite(w)

    sw = float(np.sum(w[sig & finite_w]))
    bw = float(np.sum(w[bkg & finite_w]))
    bkg_w = w[bkg & finite_w]

    return {
        "selection": selection,
        "category_type": category_type,
        "score_low": score_low,
        "score_high": score_high,
        "raw_signal": int(np.sum(sig)),
        "raw_background": int(np.sum(bkg)),
        "raw_background_missing_weight": int(np.sum(bkg & ~finite_w)),
        "S_w": sw,
        "B_w": bw,
        "S_over_B": float(sw / bw) if bw > 0 else 0.0,
        "S_over_SplusB": float(sw / (sw + bw)) if (sw + bw) > 0 else 0.0,
        "S_over_sqrtB": float(sw / math.sqrt(bw)) if bw > 0 else 0.0,
        "S_over_sqrtSplusB": float(sw / math.sqrt(sw + bw)) if (sw + bw) > 0 else 0.0,
        "asimov_Z_A": asimov_z(sw, bw),
        "N_eff_bkg": neff(bkg_w),
        "twoB_over_S": two_b_over_s(sw, bw),
    }


def make_balanced_training_weights(y, physics_w):
    y = np.asarray(y).astype(int)
    w = np.asarray(physics_w).astype(float)

    out = np.zeros_like(w, dtype=float)

    for cls in [0, 1]:
        mask = y == cls
        total = np.sum(w[mask])
        if total > 0:
            out[mask] = 0.5 * w[mask] / total

    nonzero = out[out > 0]
    if len(nonzero) > 0:
        out = out / np.mean(nonzero)

    return out


class DenseDNN(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predict_scores(model, X_np, device, batch_size=4096):
    model.eval()

    scores = []

    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            xb = torch.tensor(X_np[start:start + batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            scores.append(probs)

    return np.concatenate(scores)


def summarize_backgrounds(df: pd.DataFrame, mask, category_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df.loc[mask].copy()
    sub = sub[(sub["is_signal"] == 0) & np.isfinite(sub["physics_weight"])].copy()

    if len(sub) == 0:
        return pd.DataFrame(), pd.DataFrame()

    group = (
        sub.groupby("process_group")
        .agg(
            raw_background=("physics_weight", "size"),
            B_w=("physics_weight", "sum"),
            B_w2=("physics_weight", lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2))),
        )
        .reset_index()
    )
    total_b = group["B_w"].sum()
    group["fraction_of_B_w"] = group["B_w"] / total_b if total_b > 0 else 0.0
    group["N_eff_group"] = group["B_w"] ** 2 / group["B_w2"]
    group["category"] = category_name
    group = group.sort_values("B_w", ascending=False)

    sample = (
        sub.groupby(["sample", "process_group"])
        .agg(
            raw_background=("physics_weight", "size"),
            B_w=("physics_weight", "sum"),
            B_w2=("physics_weight", lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2))),
        )
        .reset_index()
    )
    total_b = sample["B_w"].sum()
    sample["fraction_of_B_w"] = sample["B_w"] / total_b if total_b > 0 else 0.0
    sample["N_eff_sample"] = sample["B_w"] ** 2 / sample["B_w2"]
    sample["category"] = category_name
    sample = sample.sort_values("B_w", ascending=False)

    return group, sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="outputs/hh4b_baseline/full_features/hh4b_resolved_features.parquet",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/hh4b_baseline/dnn_v1",
    )
    parser.add_argument("--random-state", type=int, default=12345)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    args = parser.parse_args()

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.features)

    # Exclude missing/non-finite proxy weights from weighted DNN training/evaluation.
    df_trainable = df[np.isfinite(df["physics_weight"])].copy().reset_index(drop=True)

    print("[INFO] Loaded feature table:", args.features)
    print("[INFO] Total rows:", len(df))
    print("[INFO] Trainable finite-weight rows:", len(df_trainable))
    print("[INFO] Excluded missing-weight rows:", len(df) - len(df_trainable))
    print("[INFO] Signal/background in trainable table:")
    print(df_trainable["is_signal"].value_counts().rename(index={0: "background", 1: "signal"}).to_string())

    if df_trainable["is_signal"].nunique() < 2:
        raise RuntimeError("Need both signal and background in trainable sample.")

    # Same continuous feature set as BDT v2.
    # Binary category flags are not used for training, only evaluation.
    feature_cols = [
        "n_ak4",
        "n_btag",
        "n_ak8_200",
        "n_ak8_300",
        "ht",
        "vbf_mjj",
        "vbf_deta",
        "b1_pt",
        "b1_eta",
        "b1_mass",
        "b1_btag",
        "b2_pt",
        "b2_eta",
        "b2_mass",
        "b2_btag",
        "b3_pt",
        "b3_eta",
        "b3_mass",
        "b3_btag",
        "b4_pt",
        "b4_eta",
        "b4_mass",
        "b4_btag",
        "h1_mass",
        "h2_mass",
        "h1_pt",
        "h2_pt",
        "hh_mass",
        "hh_pt",
        "dr_h1_bb",
        "dr_h2_bb",
        "pairing_score",
        "h_mass_avg",
        "h_mass_diff",
        "h1_abs_m125",
        "h2_abs_m125",
    ]

    feature_cols = [c for c in feature_cols if c in df_trainable.columns]

    X_df = df_trainable[feature_cols].astype(float)
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(-999.0)

    y = df_trainable["is_signal"].astype(int).to_numpy()
    w_phys = df_trainable["physics_weight"].astype(float).to_numpy()

    idx_all = np.arange(len(df_trainable))

    idx_train, idx_temp = train_test_split(
        idx_all,
        test_size=0.40,
        random_state=args.random_state,
        stratify=y,
    )

    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=0.50,
        random_state=args.random_state,
        stratify=y[idx_temp],
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_df.iloc[idx_train].to_numpy(dtype=float))
    X_val = scaler.transform(X_df.iloc[idx_val].to_numpy(dtype=float))
    X_test = scaler.transform(X_df.iloc[idx_test].to_numpy(dtype=float))

    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    w_train_phys = w_phys[idx_train]
    w_val_phys = w_phys[idx_val]
    w_test_phys = w_phys[idx_test]

    w_train_balanced = make_balanced_training_weights(y_train, w_train_phys)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    w_train_t = torch.tensor(w_train_balanced, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t, w_train_t)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = DenseDNN(n_features=len(feature_cols), dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_val_auc = -np.inf
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            per_event_loss = loss_fn(logits, yb)
            loss = torch.mean(per_event_loss * wb)

            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))

        val_score = predict_scores(model, X_val, device)
        val_auc = roc_auc_score(y_val, val_score)
        val_auc_w = roc_auc_score(y_val, val_score, sample_weight=w_val_phys)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "raw_auc_val": float(val_auc),
            "weighted_auc_val": float(val_auc_w),
        }
        history.append(row)

        print(
            f"[INFO] epoch {epoch:03d} "
            f"loss={row['train_loss']:.6f} "
            f"raw_auc_val={val_auc:.4f} "
            f"weighted_auc_val={val_auc_w:.4f}"
        )

        # Use raw validation AUC for early stopping because weighted AUC can be noisy with high-weight tails.
        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_score = predict_scores(model, X_val, device)
    test_score = predict_scores(model, X_test, device)

    df_val = df_trainable.iloc[idx_val].copy()
    df_val["dnn_score"] = val_score
    df_val["bdt_score"] = val_score  # alias so existing stability script can be reused

    df_test = df_trainable.iloc[idx_test].copy()
    df_test["dnn_score"] = test_score
    df_test["bdt_score"] = test_score  # alias so existing stability script can be reused

    metrics = {
        "n_total_rows_original": int(len(df)),
        "n_total_trainable_finite_weight": int(len(df_trainable)),
        "n_excluded_missing_weight": int(len(df) - len(df_trainable)),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "n_signal_trainable": int(np.sum(y)),
        "n_background_trainable": int(np.sum(1 - y)),
        "feature_cols": feature_cols,
        "removed_from_training_but_used_for_evaluation": [
            "category_hmass_70_190",
            "category_hmass_90_160",
            "category_vbf_like",
            "category_boosted_1ak8_2b",
        ],
        "best_epoch": int(best_epoch),
        "raw_auc_val": float(roc_auc_score(y_val, val_score)),
        "raw_auc_test": float(roc_auc_score(y_test, test_score)),
        "weighted_auc_val": float(roc_auc_score(y_val, val_score, sample_weight=w_val_phys)),
        "weighted_auc_test": float(roc_auc_score(y_test, test_score, sample_weight=w_test_phys)),
        "raw_ap_val": float(average_precision_score(y_val, val_score)),
        "raw_ap_test": float(average_precision_score(y_test, test_score)),
        "weighted_ap_val": float(average_precision_score(y_val, val_score, sample_weight=w_val_phys)),
        "weighted_ap_test": float(average_precision_score(y_test, test_score, sample_weight=w_test_phys)),
        "score_min_val": float(np.min(val_score)),
        "score_max_val": float(np.max(val_score)),
        "score_min_test": float(np.min(test_score)),
        "score_max_test": float(np.max(test_score)),
    }

    # Test cut and DNN threshold categories.
    category_rows = []

    category_rows.append(
        summarize_selection(
            df_test,
            np.ones(len(df_test), dtype=bool),
            "test_cut_all_resolved_4b",
            "cut_baseline",
        )
    )

    for cat_col, name in [
        ("category_hmass_70_190", "test_cut_hmass_70_190"),
        ("category_hmass_90_160", "test_cut_hmass_90_160"),
        ("category_vbf_like", "test_cut_vbf_like"),
        ("category_boosted_1ak8_2b", "test_cut_boosted_1AK8_2b"),
    ]:
        if cat_col in df_test.columns:
            category_rows.append(
                summarize_selection(
                    df_test,
                    df_test[cat_col].astype(bool),
                    name,
                    "cut_baseline",
                )
            )

    fixed_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    for thr in fixed_thresholds:
        category_rows.append(
            summarize_selection(
                df_test,
                df_test["dnn_score"].to_numpy() >= thr,
                f"test_dnn_score_ge_{thr:.2f}",
                "dnn_fixed_cumulative",
                score_low=thr,
                score_high=None,
            )
        )

    # Quantile categories are useful because DNN score calibration may differ from BDT.
    for q in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        thr = float(np.quantile(test_score, q))
        category_rows.append(
            summarize_selection(
                df_test,
                df_test["dnn_score"].to_numpy() >= thr,
                f"test_dnn_top_{int(round((1-q)*100))}pct_score_ge_{thr:.6f}",
                "dnn_quantile_cumulative",
                score_low=thr,
                score_high=None,
            )
        )

    cats = pd.DataFrame(category_rows)

    # Background tables for the best nominal DNN cumulative category.
    dnn_rows = cats[cats["category_type"].str.contains("dnn", na=False)].copy()
    stable_like = dnn_rows[
        (dnn_rows["raw_signal"] >= 5)
        & (dnn_rows["raw_background"] >= 20)
        & (dnn_rows["N_eff_bkg"] >= 5)
    ].copy()

    if len(stable_like) > 0:
        best_row = stable_like.sort_values(
            ["asimov_Z_A", "S_over_B", "N_eff_bkg"],
            ascending=[False, False, False],
        ).iloc[0]
    else:
        best_row = dnn_rows.sort_values(
            ["asimov_Z_A", "S_over_B"],
            ascending=[False, False],
        ).iloc[0]

    best_name = str(best_row["selection"])
    best_thr = float(best_row["score_low"])

    best_mask = df_test["dnn_score"].to_numpy() >= best_thr
    group_bkg, sample_bkg = summarize_backgrounds(df_test, best_mask, best_name)

    metrics["best_nominal_dnn_selection"] = best_name
    metrics["best_nominal_dnn_threshold"] = best_thr

    # Save artifacts.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "metrics": metrics,
        },
        outdir / "hh4b_resolved_dnn_v1_model.pt",
    )

    joblib.dump(scaler, outdir / "hh4b_resolved_dnn_v1_scaler.joblib")

    pd.DataFrame(history).to_csv(outdir / "hh4b_resolved_dnn_v1_training_history.csv", index=False)

    with open(outdir / "hh4b_resolved_dnn_v1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    df_val[
        [
            "event_key",
            "sample",
            "process_group",
            "is_signal",
            "physics_weight",
            "dnn_score",
            "bdt_score",
            "category_hmass_70_190",
            "category_hmass_90_160",
            "category_vbf_like",
            "category_boosted_1ak8_2b",
        ]
    ].to_csv(outdir / "hh4b_resolved_dnn_v1_validation_scores.csv", index=False)

    df_test[
        [
            "event_key",
            "sample",
            "process_group",
            "is_signal",
            "physics_weight",
            "dnn_score",
            "bdt_score",
            "category_hmass_70_190",
            "category_hmass_90_160",
            "category_vbf_like",
            "category_boosted_1ak8_2b",
        ]
    ].to_csv(outdir / "hh4b_resolved_dnn_v1_test_scores.csv", index=False)

    cats.to_csv(outdir / "hh4b_resolved_dnn_v1_test_categories.csv", index=False)
    group_bkg.to_csv(outdir / "hh4b_resolved_dnn_v1_best_region_backgrounds_by_group.csv", index=False)
    sample_bkg.to_csv(outdir / "hh4b_resolved_dnn_v1_best_region_backgrounds_by_sample.csv", index=False)

    note_lines = []
    note_lines.append("# HH4b resolved DNN v1\n\n")
    note_lines.append("This is a simple dense neural network baseline trained on the same continuous resolved HH4b features as BDT v2.\n\n")
    note_lines.append("Binary category flags are not used as DNN inputs; they are used only for evaluation.\n\n")
    note_lines.append("The DNN test score file includes `bdt_score` as an alias for `dnn_score` so that the existing BDT stability/resampling script can be reused.\n\n")
    note_lines.append("## Metrics\n\n")
    note_lines.append("```json\n")
    note_lines.append(json.dumps(metrics, indent=2))
    note_lines.append("\n```\n\n")
    note_lines.append("## Initial test categories\n\n")
    note_lines.append(cats.to_markdown(index=False))
    note_lines.append("\n")

    (outdir / "hh4b_resolved_dnn_v1_interpretation.md").write_text(
        "".join(note_lines),
        encoding="utf-8",
    )

    print("\n[INFO] Metrics:")
    print(json.dumps(metrics, indent=2))

    print("\n[INFO] Test categories:")
    print(cats.to_string(index=False))

    print("\n[INFO] Best nominal DNN category:", best_name)
    print("\n[INFO] Backgrounds by process group in best nominal DNN category:")
    print(group_bkg.head(40).to_string(index=False))

    print("\n[INFO] Wrote outputs to:", outdir)


if __name__ == "__main__":
    main()
