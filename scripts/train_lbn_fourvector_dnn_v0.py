"""
Train a v0 LBN-style four-vector multiclass DNN for HH->bbWW.

Inputs:
  X_obj  : object tensor, shape (N, n_objects, n_object_features)
  X_pair : manual-LBN pair features from object four-vectors
  X_aux  : auxiliary event-level features

Classes:
  HH_ggF_like vs Top_Higgs vs WJets_Other

This script reuses the CMS-style evaluation utilities from
train_tabular_multiclass_dnn_cmsstyle.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent))

from train_tabular_multiclass_dnn_cmsstyle import (
    category_yields,
    compute_metrics,
    hh_score_scan,
    safe_auc,
    softmax_np,
    stable_threshold_summary,
    weighted_accuracy,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_class_weights(y_class: np.ndarray, train_mask: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.array([(y_class[train_mask] == i).sum() for i in range(n_classes)], dtype=float)

    if np.any(counts <= 0):
        raise RuntimeError(f"Missing class in train split: counts={counts}")

    total = counts.sum()
    weights = total / (n_classes * counts)
    return weights.astype(np.float32)


class BranchMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float):
        super().__init__()

        layers = []
        prev = in_dim

        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev = h

        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LBNStyleDNN(nn.Module):
    def __init__(
        self,
        n_obj: int,
        n_obj_feat: int,
        n_pair: int,
        n_aux: int,
        n_classes: int,
        use_pair: bool = True,
        use_aux: bool = True,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.use_pair = use_pair
        self.use_aux = use_aux

        obj_in = n_obj * n_obj_feat

        self.obj_branch = BranchMLP(
            obj_in,
            hidden=[64, 32],
            dropout=dropout,
        )

        combine_dim = self.obj_branch.out_dim

        if self.use_pair:
            self.pair_branch = BranchMLP(
                n_pair,
                hidden=[64, 32],
                dropout=dropout,
            )
            combine_dim += self.pair_branch.out_dim
        else:
            self.pair_branch = None

        if self.use_aux:
            self.aux_branch = BranchMLP(
                n_aux,
                hidden=[96, 48],
                dropout=dropout,
            )
            combine_dim += self.aux_branch.out_dim
        else:
            self.aux_branch = None

        self.head = nn.Sequential(
            nn.Linear(combine_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x_obj: torch.Tensor, x_pair: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        batch = x_obj.shape[0]

        parts = []

        x_obj_flat = x_obj.reshape(batch, -1)
        parts.append(self.obj_branch(x_obj_flat))

        if self.use_pair:
            parts.append(self.pair_branch(x_pair))

        if self.use_aux:
            parts.append(self.aux_branch(x_aux))

        z = torch.cat(parts, dim=1)
        return self.head(z)


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    X_obj: np.ndarray,
    X_pair: np.ndarray,
    X_aux: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_obj, dtype=torch.float32),
            torch.tensor(X_pair, dtype=torch.float32),
            torch.tensor(X_aux, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    out = []

    for xb_obj, xb_pair, xb_aux in loader:
        xb_obj = xb_obj.to(device)
        xb_pair = xb_pair.to(device)
        xb_aux = xb_aux.to(device)

        logits = model(xb_obj, xb_pair, xb_aux)
        out.append(logits.detach().cpu().numpy())

    return np.concatenate(out, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="outputs/lbn_inputs_lepemu_rough_recoMET_v0/lbn_fourvector_inputs_v0.npz",
    )
    parser.add_argument(
        "--metadata",
        default="outputs/lbn_inputs_lepemu_rough_recoMET_v0/event_metadata.parquet",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/lbn_fourvector_dnn_v0_full",
    )

    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-thresholds", type=int, default=201)

    parser.add_argument("--no-pair", action="store_true")
    parser.add_argument("--no-aux", action="store_true")
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
    print("Input:", args.input)

    data = np.load(args.input, allow_pickle=True)

    X_obj = data["X_obj"].astype(np.float32)
    X_pair = data["X_pair"].astype(np.float32)
    X_aux = data["X_aux"].astype(np.float32)

    y_binary = data["y_binary"].astype(np.int64)
    y_class = data["y_multiclass"].astype(np.int64)
    physics_weight = data["physics_weight"].astype(np.float32)
    split = data["split"].astype(str)

    class_names = data["class_names"].astype(str).tolist()
    object_names = data["object_names"].astype(str).tolist()
    object_features = data["object_features"].astype(str).tolist()
    pair_feature_names = data["pair_feature_names"].astype(str).tolist()
    aux_feature_names = data["aux_feature_names"].astype(str).tolist()

    n_classes = len(class_names)
    hh_idx = 0

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    print("Shapes:")
    print("  X_obj:", X_obj.shape)
    print("  X_pair:", X_pair.shape)
    print("  X_aux:", X_aux.shape)

    print("Classes:", class_names)
    print("Objects:", object_names)
    print("Object features:", object_features)
    print("Use pair branch:", not args.no_pair)
    print("Use aux branch:", not args.no_aux)

    print("Train/val/test:", int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()))
    print("Signal raw train/val/test:", int(y_binary[train_mask].sum()), int(y_binary[val_mask].sum()), int(y_binary[test_mask].sum()))

    print("\nClass counts by split:")
    for s in ["train", "val", "test"]:
        m = split == s
        parts = []
        for i, name in enumerate(class_names):
            parts.append(f"{name}={int((y_class[m] == i).sum())}")
        print(s + ":", ", ".join(parts))

    class_weights = compute_class_weights(y_class, train_mask, n_classes)
    event_loss_weight = class_weights[y_class]
    event_loss_weight = event_loss_weight / np.mean(event_loss_weight[train_mask])

    print("\nTraining class weights:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weights[i]:.6g}")

    model = LBNStyleDNN(
        n_obj=X_obj.shape[1],
        n_obj_feat=X_obj.shape[2],
        n_pair=X_pair.shape[1],
        n_aux=X_aux.shape[1],
        n_classes=n_classes,
        use_pair=(not args.no_pair),
        use_aux=(not args.no_aux),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ce = nn.CrossEntropyLoss(reduction="none")

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_obj[train_mask], dtype=torch.float32),
            torch.tensor(X_pair[train_mask], dtype=torch.float32),
            torch.tensor(X_aux[train_mask], dtype=torch.float32),
            torch.tensor(y_class[train_mask], dtype=torch.long),
            torch.tensor(event_loss_weight[train_mask], dtype=torch.float32),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    history = []
    best_state = None
    best_epoch = -1
    best_val_hh_auc_weighted = -np.inf
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        loss_sum = 0.0
        weight_sum = 0.0

        for xb_obj, xb_pair, xb_aux, yb, wb in train_loader:
            xb_obj = xb_obj.to(device)
            xb_pair = xb_pair.to(device)
            xb_aux = xb_aux.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad()

            logits = model(xb_obj, xb_pair, xb_aux)
            per_event_loss = ce(logits, yb)
            loss = (per_event_loss * wb).sum() / wb.sum()

            loss.backward()
            optimizer.step()

            loss_sum += float((per_event_loss * wb).sum().detach().cpu())
            weight_sum += float(wb.sum().detach().cpu())

        train_loss = loss_sum / weight_sum

        logits_all = predict_logits(
            model,
            X_obj,
            X_pair,
            X_aux,
            args.batch_size,
            device,
        )

        probs_all = softmax_np(logits_all)

        val_hh_auc_raw = safe_auc(y_binary[val_mask], probs_all[val_mask, hh_idx])
        val_hh_auc_weighted = safe_auc(
            y_binary[val_mask],
            probs_all[val_mask, hh_idx],
            physics_weight[val_mask],
        )
        test_hh_auc_weighted = safe_auc(
            y_binary[test_mask],
            probs_all[test_mask, hh_idx],
            physics_weight[test_mask],
        )

        pred_val = np.argmax(probs_all[val_mask], axis=1)
        val_acc_raw = float(np.mean(pred_val == y_class[val_mask]))
        val_acc_weighted = weighted_accuracy(
            y_class[val_mask],
            pred_val,
            physics_weight[val_mask],
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_hh_auc_raw": val_hh_auc_raw,
                "val_hh_auc_weighted": val_hh_auc_weighted,
                "val_multiclass_accuracy_raw": val_acc_raw,
                "val_multiclass_accuracy_weighted": val_acc_weighted,
                "test_hh_auc_weighted_monitor": test_hh_auc_weighted,
            }
        )

        improved = val_hh_auc_weighted > best_val_hh_auc_weighted + 1e-5

        if improved:
            best_val_hh_auc_weighted = val_hh_auc_weighted
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            print(
                f"epoch {epoch:03d} "
                f"loss={train_loss:.5f} "
                f"val_hh_auc_raw={val_hh_auc_raw:.4f} "
                f"val_hh_auc_w={val_hh_auc_weighted:.4f} "
                f"val_acc_w={val_acc_weighted:.4f} "
                f"test_hh_auc_w_monitor={test_hh_auc_weighted:.4f}"
            )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch = {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best model state saved.")

    model.load_state_dict(best_state)
    model.to(device)

    logits = predict_logits(
        model,
        X_obj,
        X_pair,
        X_aux,
        args.batch_size,
        device,
    )

    probs = softmax_np(logits)
    pred_class = np.argmax(probs, axis=1)
    score_hh = probs[:, hh_idx]

    metrics = compute_metrics(
        y_binary=y_binary,
        y_class=y_class,
        probs=probs,
        pred_class=pred_class,
        physics_weight=physics_weight,
        split=split,
        class_names=class_names,
        hh_idx=hh_idx,
    )

    cats = category_yields(
        y_binary=y_binary,
        y_class=y_class,
        pred_class=pred_class,
        probs=probs,
        physics_weight=physics_weight,
        split=split,
        class_names=class_names,
        hh_idx=hh_idx,
    )

    scan_argmax = hh_score_scan(
        y_binary=y_binary,
        pred_class=pred_class,
        score_hh=score_hh,
        physics_weight=physics_weight,
        split=split,
        hh_idx=hh_idx,
        n_thresholds=args.n_thresholds,
        mode="argmax_hh",
    )

    scan_global = hh_score_scan(
        y_binary=y_binary,
        pred_class=pred_class,
        score_hh=score_hh,
        physics_weight=physics_weight,
        split=split,
        hh_idx=hh_idx,
        n_thresholds=args.n_thresholds,
        mode="global_hh_score",
    )

    scan = pd.concat([scan_argmax, scan_global], ignore_index=True)

    stable = pd.concat(
        [
            stable_threshold_summary(scan, "argmax_hh"),
            stable_threshold_summary(scan, "global_hh_score"),
        ],
        ignore_index=True,
    )

    metadata_path = Path(args.metadata)

    if metadata_path.exists():
        meta = pd.read_parquet(metadata_path)

        if len(meta) == len(X_obj):
            pred_df = meta.copy()
        else:
            print("Warning: metadata length mismatch; writing minimal prediction table.")
            pred_df = pd.DataFrame()
    else:
        pred_df = pd.DataFrame()

    if len(pred_df) == 0:
        pred_df = pd.DataFrame(
            {
                "split": split,
                "y_binary": y_binary,
                "y_multiclass": y_class,
                "physics_weight_nominal": physics_weight,
            }
        )

    pred_df["y_binary"] = y_binary
    pred_df["y_multiclass"] = y_class
    pred_df["predicted_class"] = pred_class
    pred_df["predicted_class_name"] = [class_names[i] for i in pred_class]
    pred_df["score_HH"] = score_hh

    for i, name in enumerate(class_names):
        safe_name = name.replace("/", "_").replace(" ", "_")
        pred_df[f"score_{safe_name}"] = probs[:, i]

    metrics_path = outdir / "lbn_dnn_metrics.csv"
    cats_path = outdir / "lbn_cmsstyle_category_yields.csv"
    scan_path = outdir / "lbn_hh_score_scan.csv"
    stable_path = outdir / "lbn_stable_thresholds.csv"
    history_path = outdir / "lbn_training_history.csv"
    predictions_path = outdir / "lbn_dnn_predictions.parquet"
    model_path = outdir / "lbn_fourvector_dnn_v0.pt"
    summary_path = outdir / "lbn_dnn_summary.json"

    metrics.to_csv(metrics_path, index=False)
    cats.to_csv(cats_path, index=False)
    scan.to_csv(scan_path, index=False)
    stable.to_csv(stable_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)
    pred_df.to_parquet(predictions_path, index=False)

    torch.save(
        {
            "model_state_dict": best_state,
            "class_names": class_names,
            "object_names": object_names,
            "object_features": object_features,
            "pair_feature_names": pair_feature_names,
            "aux_feature_names": aux_feature_names,
            "best_epoch": best_epoch,
            "best_val_hh_auc_weighted": best_val_hh_auc_weighted,
            "use_pair": not args.no_pair,
            "use_aux": not args.no_aux,
            "args": vars(args),
        },
        model_path,
    )

    summary = {
        "input": args.input,
        "metadata": args.metadata,
        "outdir": str(outdir),
        "device": str(device),
        "class_names": class_names,
        "best_epoch": int(best_epoch),
        "best_val_hh_auc_weighted": float(best_val_hh_auc_weighted),
        "use_pair": not args.no_pair,
        "use_aux": not args.no_aux,
        "metrics": metrics.to_dict(orient="records"),
        "stable_thresholds": stable.to_dict(orient="records"),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LBN-style DNN metrics ===")
    print(metrics.to_string(index=False))

    print("\n=== LBN-style CMS category yields, validation+test ===")
    print(cats[cats["split"].isin(["val", "test", "valtest"])].to_string(index=False))

    print("\n=== LBN-style stable HH score thresholds ===")
    print(stable.to_string(index=False))

    print("\nWrote outputs to:", outdir)
    print("Metrics:", metrics_path)
    print("Category yields:", cats_path)
    print("HH score scan:", scan_path)
    print("Stable thresholds:", stable_path)
    print("Predictions:", predictions_path)
    print("Model:", model_path)


if __name__ == "__main__":
    main()
