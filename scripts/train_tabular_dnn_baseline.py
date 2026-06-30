"""
Train a binary tabular DNN baseline for HH->bbWW using corrected recoMET inputs.

Goal:
- Same input features/splits as the corrected recoMET BDT.
- Train binary classifier: HH_bbWW signal vs all backgrounds.
- Use class-balanced training weights by default.
- Evaluate with rough physics weights.
- Produce metrics, threshold scans, stable-threshold summary, and predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def asimov_z(s: float, b: float) -> float:
    if b <= 0:
        return math.sqrt(2.0 * s) if s > 0 else 0.0
    if s <= 0:
        return 0.0
    return float(math.sqrt(2.0 * ((s + b) * math.log(1.0 + s / b) - s)))


def neff(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    den = np.sum(weights * weights)
    if den <= 0:
        return 0.0
    return float(np.sum(weights) ** 2 / den)

# The DNN model is a simple feedforward MLP with ReLU activations, batch normalization, and optional dropout.
# The final layer outputs a single logit for binary classification.
class TabularMLP(nn.Module):
    def __init__(self, n_features: int, hidden: list[int], dropout: float):
        super().__init__()

        layers = []
        prev = n_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    logits = []

    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    for (xb,) in loader:
        xb = xb.to(device)
        out = model(xb).detach().cpu().numpy()
        logits.append(out)

    return np.concatenate(logits)


def safe_auc(y: np.ndarray, score: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score, sample_weight=sample_weight))


def safe_ap(y: np.ndarray, score: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score, sample_weight=sample_weight))


def compute_metrics(
    y: np.ndarray,
    score: np.ndarray,
    physics_weight: np.ndarray,
    split: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for s in ["train", "val", "test", "valtest"]:
        if s == "valtest":
            mask = np.isin(split, ["val", "test"])
        else:
            mask = split == s

        yy = y[mask]
        ss = score[mask]
        ww = physics_weight[mask]

        rows.append(
            {
                "split": s,
                "raw_events": int(mask.sum()),
                "signal_raw": int((yy == 1).sum()),
                "background_raw": int((yy == 0).sum()),
                "roc_auc_raw": safe_auc(yy, ss),
                "roc_auc_weighted": safe_auc(yy, ss, ww),
                "average_precision_raw": safe_ap(yy, ss),
                "average_precision_weighted": safe_ap(yy, ss, ww),
            }
        )

    return pd.DataFrame(rows)


def threshold_scan(
    y: np.ndarray,
    score: np.ndarray,
    physics_weight: np.ndarray,
    split: np.ndarray,
    n_thresholds: int,
) -> pd.DataFrame:
    '''
    Scan thresholds for a given dataset and compute metrics.
    '''
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    rows = []
    for split_name in ["train", "val", "test", "valtest"]:
        if split_name == "valtest":
            split_mask = np.isin(split, ["val", "test"])
        else:
            split_mask = split == split_name

        for thr in thresholds:
            sel = split_mask & (score >= thr)

            sig = sel & (y == 1)
            bkg = sel & (y == 0)

            s_raw = int(sig.sum())
            b_raw = int(bkg.sum())

            s_w = float(physics_weight[sig].sum())
            b_w = float(physics_weight[bkg].sum())

            split_sig = split_mask & (y == 1)
            split_bkg = split_mask & (y == 0)

            s_w_total = float(physics_weight[split_sig].sum())
            b_w_total = float(physics_weight[split_bkg].sum())

            rows.append(
                {
                    "split": split_name,
                    "threshold": float(thr),
                    "S_raw": s_raw,
                    "B_raw": b_raw,
                    "S_weighted": s_w,
                    "B_weighted": b_w,
                    "S_over_B_raw": float(s_raw / b_raw) if b_raw > 0 else np.inf,
                    "S_over_SplusB_raw": float(s_raw / (s_raw + b_raw)) if (s_raw + b_raw) > 0 else np.nan,
                    "asimovZ_raw": asimov_z(float(s_raw), float(b_raw)),
                    "S_over_B_weighted": float(s_w / b_w) if b_w > 0 else np.inf,
                    "S_over_SplusB_weighted": float(s_w / (s_w + b_w)) if (s_w + b_w) > 0 else np.nan,
                    "asimovZ_weighted": asimov_z(s_w, b_w),
                    "signal_eff_raw": float(s_raw / split_sig.sum()) if split_sig.sum() > 0 else np.nan,
                    "background_eff_raw": float(b_raw / split_bkg.sum()) if split_bkg.sum() > 0 else np.nan,
                    "signal_eff_weighted": float(s_w / s_w_total) if s_w_total > 0 else np.nan,
                    "background_eff_weighted": float(b_w / b_w_total) if b_w_total > 0 else np.nan,
                    "background_neff": neff(physics_weight[bkg]),
                }
            )

    return pd.DataFrame(rows)


def stable_threshold_summary(scan: pd.DataFrame) -> pd.DataFrame:
    val = scan[scan["split"] == "val"].copy()
    test = scan[scan["split"] == "test"].copy()

    merged = val.merge(test, on="threshold", suffixes=("_val", "_test"))

    rows = []
    for min_neff in [10, 25, 50]:
        good = merged[
            (merged["background_neff_val"] >= min_neff)
            & (merged["background_neff_test"] >= min_neff)
            & (merged["S_raw_val"] >= 10)
            & (merged["S_raw_test"] >= 10)
            & (merged["B_raw_val"] >= 25)
            & (merged["B_raw_test"] >= 25)
        ].copy()

        if len(good) == 0:
            rows.append(
                {
                    "min_neff_valtest": min_neff,
                    "status": "no_threshold_passes",
                }
            )
            continue

        best = good.sort_values("asimovZ_weighted_val", ascending=False).iloc[0]

        rows.append(
            {
                "min_neff_valtest": min_neff,
                "status": "ok",
                "threshold": best["threshold"],

                "val_S_raw": best["S_raw_val"],
                "val_B_raw": best["B_raw_val"],
                "val_S_weighted": best["S_weighted_val"],
                "val_B_weighted": best["B_weighted_val"],
                "val_S_over_B": best["S_over_B_weighted_val"],
                "val_Z": best["asimovZ_weighted_val"],
                "val_bkg_neff": best["background_neff_val"],

                "test_S_raw": best["S_raw_test"],
                "test_B_raw": best["B_raw_test"],
                "test_S_weighted": best["S_weighted_test"],
                "test_B_weighted": best["B_weighted_test"],
                "test_S_over_B": best["S_over_B_weighted_test"],
                "test_Z": best["asimovZ_weighted_test"],
                "test_bkg_neff": best["background_neff_test"],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/dnn_inputs_lepemu_rough_topology_recoMET/tabular_dnn_inputs.npz",
    )
    parser.add_argument(
        "--metadata",
        default="outputs/dnn_inputs_lepemu_rough_topology_recoMET/event_metadata.parquet",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/tabular_dnn_binary_recoMET_topology",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", default="128,64,32")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--loss-weight-mode",
        choices=["class", "physics", "class_x_physics"],
        default="class",
        help="Training loss weights. Evaluation always uses physics weights.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-thresholds", type=int, default=201)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if (torch.cuda.is_available() and not args.no_cuda)
        else "cpu"
    )
    print("Device:", device)

    data = np.load(args.input, allow_pickle=True)

    X = data["X"].astype(np.float32)
    y = data["y_binary"].astype(np.float32)
    y_int = y.astype(np.int64)
    physics_weight = data["physics_weight"].astype(np.float32)
    split = data["split"].astype(str)

    feature_names = data["feature_names"].astype(str).tolist()

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    print("Input:", args.input)
    print("Events:", len(X))
    print("Features:", X.shape[1])
    print("Train/val/test:", train_mask.sum(), val_mask.sum(), test_mask.sum())
    print("Signal raw train/val/test:", int(y[train_mask].sum()), int(y[val_mask].sum()), int(y[test_mask].sum()))

    train_weight_binary = data["train_weight_binary"].astype(np.float32)

    if args.loss_weight_mode == "class":
        loss_weight = train_weight_binary.copy()
    elif args.loss_weight_mode == "physics":
        loss_weight = physics_weight.copy()
    elif args.loss_weight_mode == "class_x_physics":
        loss_weight = train_weight_binary.copy() * physics_weight.copy()
    else:
        raise ValueError(args.loss_weight_mode)

    # Normalize the training weights to mean 1 on the training split.
    mean_train_weight = float(np.mean(loss_weight[train_mask]))
    if mean_train_weight <= 0 or not np.isfinite(mean_train_weight):
        raise RuntimeError("Bad training weight normalization.")
    loss_weight = loss_weight / mean_train_weight

    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    model = TabularMLP(X.shape[1], hidden=hidden, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    X_train = torch.tensor(X[train_mask], dtype=torch.float32)
    y_train = torch.tensor(y[train_mask], dtype=torch.float32)
    w_train = torch.tensor(loss_weight[train_mask], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train, w_train),
        batch_size=args.batch_size,
        shuffle=True,
    )

    history = []
    best_val_auc = -np.inf
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0

        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            per_event_loss = bce(logits, yb)
            loss = (per_event_loss * wb).sum() / wb.sum()
            loss.backward()
            optimizer.step()

            train_loss_sum += float((per_event_loss * wb).sum().detach().cpu())
            train_weight_sum += float(wb.sum().detach().cpu())

        train_loss = train_loss_sum / train_weight_sum

        logits_all = predict_logits(model, X, args.batch_size, device)
        score_all = sigmoid_np(logits_all)

        val_auc_raw = safe_auc(y_int[val_mask], score_all[val_mask])
        val_auc_weighted = safe_auc(y_int[val_mask], score_all[val_mask], physics_weight[val_mask])
        val_ap_raw = safe_ap(y_int[val_mask], score_all[val_mask])
        test_auc_weighted = safe_auc(y_int[test_mask], score_all[test_mask], physics_weight[test_mask])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc_raw": val_auc_raw,
                "val_auc_weighted": val_auc_weighted,
                "val_ap_raw": val_ap_raw,
                "test_auc_weighted_monitor": test_auc_weighted,
            }
        )

        improved = val_auc_weighted > best_val_auc + 1e-5
        if improved:
            best_val_auc = val_auc_weighted
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            print(
                f"epoch {epoch:03d} "
                f"loss={train_loss:.5f} "
                f"val_auc_raw={val_auc_raw:.4f} "
                f"val_auc_w={val_auc_weighted:.4f} "
                f"val_ap={val_ap_raw:.4f} "
                f"test_auc_w_monitor={test_auc_weighted:.4f}"
            )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch = {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best state saved.")

    model.load_state_dict(best_state)
    model.to(device)

    logits = predict_logits(model, X, args.batch_size, device)
    score = sigmoid_np(logits)

    metrics = compute_metrics(y_int, score, physics_weight, split)
    scan = threshold_scan(y_int, score, physics_weight, split, args.n_thresholds)
    stable = stable_threshold_summary(scan)

    metrics_path = outdir / "dnn_classification_metrics.csv"
    scan_path = outdir / "dnn_threshold_scan_rough_weighted.csv"
    stable_path = outdir / "dnn_stable_thresholds.csv"
    history_path = outdir / "dnn_training_history.csv"

    metrics.to_csv(metrics_path, index=False)
    scan.to_csv(scan_path, index=False)
    stable.to_csv(stable_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)

    # Raw validation-best threshold, for comparison with BDT behavior.
    val_scan = scan[scan["split"] == "val"].copy()
    best_val_row = val_scan.sort_values("asimovZ_weighted", ascending=False).iloc[0]

    test_scan = scan[scan["split"] == "test"].copy()
    test_nearest = test_scan.iloc[(test_scan["threshold"] - best_val_row["threshold"]).abs().argsort().iloc[0]]

    # Save predictions with metadata if available.
    metadata_path = Path(args.metadata)
    if metadata_path.exists():
        meta = pd.read_parquet(metadata_path)
        if len(meta) != len(score):
            print("Warning: metadata length mismatch; writing minimal predictions.")
            pred = pd.DataFrame()
        else:
            pred = meta.copy()
    else:
        pred = pd.DataFrame()

    if len(pred) == 0:
        pred = pd.DataFrame(
            {
                "split": split,
                "target": y_int,
                "physics_weight_nominal": physics_weight,
            }
        )

    pred["dnn_logit"] = logits
    pred["dnn_score"] = score

    predictions_path = outdir / "dnn_predictions.parquet"
    pred.to_parquet(predictions_path, index=False)

    model_path = outdir / "tabular_dnn_binary.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "n_features": X.shape[1],
            "hidden": hidden,
            "dropout": args.dropout,
            "feature_names": feature_names,
            "best_epoch": best_epoch,
            "best_val_auc_weighted": best_val_auc,
            "args": vars(args),
        },
        model_path,
    )

    summary = {
        "input": args.input,
        "metadata": args.metadata,
        "outdir": str(outdir),
        "device": str(device),
        "loss_weight_mode": args.loss_weight_mode,
        "best_epoch": int(best_epoch),
        "best_val_auc_weighted": float(best_val_auc),
        "metrics": metrics.to_dict(orient="records"),
        "best_validation_row_by_weighted_Z": best_val_row.to_dict(),
        "test_nearest_best_validation_threshold": test_nearest.to_dict(),
        "stable_thresholds": stable.to_dict(orient="records"),
    }

    summary_path = outdir / "dnn_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DNN classification metrics ===")
    print(metrics.to_string(index=False))

    print("\n=== Best validation threshold by rough weighted Z ===")
    print(best_val_row.to_string())

    print("\n=== Test row nearest best validation threshold ===")
    print(test_nearest.to_string())

    print("\n=== Stable thresholds ===")
    print(stable.to_string(index=False))

    print("\nWrote outputs to:", outdir)
    print("Model:", model_path)
    print("Predictions:", predictions_path)


if __name__ == "__main__":
    main()
