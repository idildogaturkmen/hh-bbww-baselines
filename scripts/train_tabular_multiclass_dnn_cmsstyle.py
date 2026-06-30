"""
CMS-inspired multiclass tabular DNN for HH->bbWW.

This is Stage 1 of the CMS-style path:
- Use the corrected recoMET tabular features.
- Train a multiclass classifier:
    HH_ggF_like vs Top_Higgs vs WJets_Other
- Evaluate CMS-style behavior:
    1. per-class scores
    2. argmax event categories
    3. HH-like category score scan
    4. stable-threshold summary
- Compare the HH-like category to the corrected recoMET BDT baseline.

This is not yet the LBN/four-vector network. It is the controlled multiclass
baseline before changing the input representation.
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
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


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


def safe_auc(y_true: np.ndarray, score: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score, sample_weight=sample_weight))


def safe_ap(y_true: np.ndarray, score: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score, sample_weight=sample_weight))


class MulticlassMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: list[int], dropout: float):
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

        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    out = []

    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    for (xb,) in loader:
        xb = xb.to(device)
        logits = model(xb).detach().cpu().numpy()
        out.append(logits)

    return np.concatenate(out, axis=0)


def get_npz_key(data: np.lib.npyio.NpzFile, candidates: list[str], required: bool = True):
    keys = set(data.files)
    for c in candidates:
        if c in keys:
            return c
    if required:
        raise KeyError(f"Could not find any of keys {candidates}. Available keys: {data.files}")
    return None


def weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    den = float(np.sum(weight))
    if den <= 0:
        return float("nan")
    return float(np.sum(weight[y_true == y_pred]) / den)


def compute_class_weights(y_class: np.ndarray, train_mask: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.array([(y_class[train_mask] == i).sum() for i in range(n_classes)], dtype=float)
    if np.any(counts <= 0):
        raise RuntimeError(f"At least one class is missing in train split: counts={counts}")

    total = counts.sum()
    weights = total / (n_classes * counts)
    return weights.astype(np.float32)


def compute_metrics(
    y_binary: np.ndarray,
    y_class: np.ndarray,
    probs: np.ndarray,
    pred_class: np.ndarray,
    physics_weight: np.ndarray,
    split: np.ndarray,
    class_names: list[str],
    hh_idx: int,
) -> pd.DataFrame:
    rows = []

    score_hh = probs[:, hh_idx]

    for split_name in ["train", "val", "test", "valtest"]:
        if split_name == "valtest":
            mask = np.isin(split, ["val", "test"])
        else:
            mask = split == split_name

        yy_bin = y_binary[mask]
        yy_cls = y_class[mask]
        pp = probs[mask]
        pred = pred_class[mask]
        ww = physics_weight[mask]

        row = {
            "split": split_name,
            "raw_events": int(mask.sum()),
            "signal_raw": int((yy_bin == 1).sum()),
            "background_raw": int((yy_bin == 0).sum()),
            "multiclass_accuracy_raw": float(np.mean(yy_cls == pred)),
            "multiclass_accuracy_weighted": weighted_accuracy(yy_cls, pred, ww),
            "hh_roc_auc_raw": safe_auc(yy_bin, score_hh[mask]),
            "hh_roc_auc_weighted": safe_auc(yy_bin, score_hh[mask], ww),
            "hh_average_precision_raw": safe_ap(yy_bin, score_hh[mask]),
            "hh_average_precision_weighted": safe_ap(yy_bin, score_hh[mask], ww),
        }

        for i, name in enumerate(class_names):
            y_ovr = (yy_cls == i).astype(int)
            row[f"auc_ovr_raw_{name}"] = safe_auc(y_ovr, pp[:, i])
            row[f"auc_ovr_weighted_{name}"] = safe_auc(y_ovr, pp[:, i], ww)

        rows.append(row)

    return pd.DataFrame(rows)


def category_yields(
    y_binary: np.ndarray,
    y_class: np.ndarray,
    pred_class: np.ndarray,
    probs: np.ndarray,
    physics_weight: np.ndarray,
    split: np.ndarray,
    class_names: list[str],
    hh_idx: int,
) -> pd.DataFrame:
    rows = []

    for split_name in ["train", "val", "test", "valtest"]:
        if split_name == "valtest":
            split_mask = np.isin(split, ["val", "test"])
        else:
            split_mask = split == split_name

        for i, name in enumerate(class_names):
            cat = split_mask & (pred_class == i)

            sig = cat & (y_binary == 1)
            bkg = cat & (y_binary == 0)

            s_raw = int(sig.sum())
            b_raw = int(bkg.sum())
            s_w = float(physics_weight[sig].sum())
            b_w = float(physics_weight[bkg].sum())

            rows.append(
                {
                    "split": split_name,
                    "predicted_category": name,
                    "category_index": i,
                    "S_raw": s_raw,
                    "B_raw": b_raw,
                    "S_weighted": s_w,
                    "B_weighted": b_w,
                    "S_over_B_raw": float(s_raw / b_raw) if b_raw > 0 else np.inf,
                    "S_over_B_weighted": float(s_w / b_w) if b_w > 0 else np.inf,
                    "asimovZ_weighted": asimov_z(s_w, b_w),
                    "background_neff": neff(physics_weight[bkg]),
                    "mean_score_HH": float(np.mean(probs[cat, hh_idx])) if cat.sum() > 0 else np.nan,
                }
            )

    return pd.DataFrame(rows)


def hh_score_scan(
    y_binary: np.ndarray,
    pred_class: np.ndarray,
    score_hh: np.ndarray,
    physics_weight: np.ndarray,
    split: np.ndarray,
    hh_idx: int,
    n_thresholds: int,
    mode: str,
) -> pd.DataFrame:
    rows = []
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    for split_name in ["train", "val", "test", "valtest"]:
        if split_name == "valtest":
            split_mask = np.isin(split, ["val", "test"])
        else:
            split_mask = split == split_name

        for thr in thresholds:
            if mode == "argmax_hh":
                sel = split_mask & (pred_class == hh_idx) & (score_hh >= thr)
            elif mode == "global_hh_score":
                sel = split_mask & (score_hh >= thr)
            else:
                raise ValueError(mode)

            sig = sel & (y_binary == 1)
            bkg = sel & (y_binary == 0)

            split_sig = split_mask & (y_binary == 1)
            split_bkg = split_mask & (y_binary == 0)

            s_raw = int(sig.sum())
            b_raw = int(bkg.sum())
            s_w = float(physics_weight[sig].sum())
            b_w = float(physics_weight[bkg].sum())

            total_s_w = float(physics_weight[split_sig].sum())
            total_b_w = float(physics_weight[split_bkg].sum())

            rows.append(
                {
                    "mode": mode,
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
                    "signal_eff_weighted": float(s_w / total_s_w) if total_s_w > 0 else np.nan,
                    "background_eff_weighted": float(b_w / total_b_w) if total_b_w > 0 else np.nan,
                    "background_neff": neff(physics_weight[bkg]),
                }
            )

    return pd.DataFrame(rows)


def stable_threshold_summary(scan: pd.DataFrame, mode: str) -> pd.DataFrame:
    sub = scan[scan["mode"] == mode].copy()

    val = sub[sub["split"] == "val"].copy()
    test = sub[sub["split"] == "test"].copy()

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
                    "mode": mode,
                    "min_neff_valtest": min_neff,
                    "status": "no_threshold_passes",
                }
            )
            continue

        best = good.sort_values("asimovZ_weighted_val", ascending=False).iloc[0]

        rows.append(
            {
                "mode": mode,
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
        default="outputs/tabular_dnn_multiclass_recoMET_topology_cmsstyle",
    )
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden", default="96,48")
    parser.add_argument("--dropout", type=float, default=0.20)
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
    print("Input:", args.input)
    print("Available NPZ keys:", list(data.files))

    X_key = get_npz_key(data, ["X"])
    split_key = get_npz_key(data, ["split"])
    weight_key = get_npz_key(data, ["physics_weight", "physics_weight_nominal"])
    ybin_key = get_npz_key(data, ["y_binary", "target", "y"])
    yclass_key = get_npz_key(
        data,
        ["y_multiclass", "y_class", "y_multiclass_label", "multiclass_label", "class_label"],
    )

    class_names_key = get_npz_key(
        data,
        ["class_names", "multiclass_class_names", "multiclass_names"],
        required=False,
    )

    feature_names_key = get_npz_key(data, ["feature_names"], required=False)

    X = data[X_key].astype(np.float32)
    split = data[split_key].astype(str)
    physics_weight = data[weight_key].astype(np.float32)
    y_binary = data[ybin_key].astype(np.int64)
    y_class = data[yclass_key].astype(np.int64)

    if class_names_key is not None:
        class_names = data[class_names_key].astype(str).tolist()
    else:
        # Expected convention from make_tabular_dnn_input.py
        class_names = ["HH_ggF_like", "Top_Higgs", "WJets_Other"]

    n_classes = len(class_names)

    if sorted(np.unique(y_class).tolist()) != list(range(n_classes)):
        print("Warning: y_class unique labels:", sorted(np.unique(y_class).tolist()))
        print("Class names:", class_names)

    if feature_names_key is not None:
        feature_names = data[feature_names_key].astype(str).tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    hh_candidates = [i for i, name in enumerate(class_names) if "HH" in name or "signal" in name.lower()]
    if len(hh_candidates) == 0:
        hh_idx = 0
        print("Warning: no HH-like class name found. Assuming class index 0 is HH-like.")
    else:
        hh_idx = hh_candidates[0]

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    print("Events:", len(X))
    print("Features:", X.shape[1])
    print("Classes:", class_names)
    print("HH class index:", hh_idx, class_names[hh_idx])
    print("Train/val/test:", train_mask.sum(), val_mask.sum(), test_mask.sum())
    print("Signal raw train/val/test:", int(y_binary[train_mask].sum()), int(y_binary[val_mask].sum()), int(y_binary[test_mask].sum()))

    print("\nClass counts by split:")
    for s in ["train", "val", "test"]:
        mask = split == s
        vals = []
        for i, name in enumerate(class_names):
            vals.append(f"{name}={int((y_class[mask] == i).sum())}")
        print(s + ":", ", ".join(vals))

    class_weights = compute_class_weights(y_class, train_mask, n_classes)
    print("\nTraining class weights:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weights[i]:.6g}")

    event_loss_weight = class_weights[y_class]
    event_loss_weight = event_loss_weight / np.mean(event_loss_weight[train_mask])

    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    model = MulticlassMLP(X.shape[1], n_classes, hidden=hidden, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss(reduction="none")

    X_train = torch.tensor(X[train_mask], dtype=torch.float32)
    y_train = torch.tensor(y_class[train_mask], dtype=torch.long)
    w_train = torch.tensor(event_loss_weight[train_mask], dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_train, y_train, w_train),
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

        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            per_event = ce(logits, yb)
            loss = (per_event * wb).sum() / wb.sum()
            loss.backward()
            optimizer.step()

            loss_sum += float((per_event * wb).sum().detach().cpu())
            weight_sum += float(wb.sum().detach().cpu())

        train_loss = loss_sum / weight_sum

        logits_all = predict_logits(model, X, args.batch_size, device)
        probs_all = softmax_np(logits_all)

        val_hh_auc_raw = safe_auc(y_binary[val_mask], probs_all[val_mask, hh_idx])
        val_hh_auc_weighted = safe_auc(y_binary[val_mask], probs_all[val_mask, hh_idx], physics_weight[val_mask])
        val_hh_ap_raw = safe_ap(y_binary[val_mask], probs_all[val_mask, hh_idx])
        test_hh_auc_weighted = safe_auc(y_binary[test_mask], probs_all[test_mask, hh_idx], physics_weight[test_mask])

        pred_val = np.argmax(probs_all[val_mask], axis=1)
        val_acc_raw = float(np.mean(pred_val == y_class[val_mask]))
        val_acc_weighted = weighted_accuracy(y_class[val_mask], pred_val, physics_weight[val_mask])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_hh_auc_raw": val_hh_auc_raw,
                "val_hh_auc_weighted": val_hh_auc_weighted,
                "val_hh_ap_raw": val_hh_ap_raw,
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
                f"val_hh_ap={val_hh_ap_raw:.4f} "
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

    logits = predict_logits(model, X, args.batch_size, device)
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

    # Confusion matrices for raw and weighted counts.
    cm_rows = []
    for split_name in ["train", "val", "test", "valtest"]:
        if split_name == "valtest":
            mask = np.isin(split, ["val", "test"])
        else:
            mask = split == split_name

        cm_raw = confusion_matrix(y_class[mask], pred_class[mask], labels=list(range(n_classes)))
        cm_weighted = confusion_matrix(
            y_class[mask],
            pred_class[mask],
            labels=list(range(n_classes)),
            sample_weight=physics_weight[mask],
        )

        for true_i, true_name in enumerate(class_names):
            for pred_i, pred_name in enumerate(class_names):
                cm_rows.append(
                    {
                        "split": split_name,
                        "true_class": true_name,
                        "predicted_class": pred_name,
                        "raw_events": int(cm_raw[true_i, pred_i]),
                        "weighted_events": float(cm_weighted[true_i, pred_i]),
                    }
                )

    confusion = pd.DataFrame(cm_rows)

    metadata_path = Path(args.metadata)
    if metadata_path.exists():
        meta = pd.read_parquet(metadata_path)
        if len(meta) == len(X):
            pred_df = meta.copy()
        else:
            print("Warning: metadata length mismatch. Writing minimal prediction table.")
            pred_df = pd.DataFrame()
    else:
        pred_df = pd.DataFrame()

    if len(pred_df) == 0:
        pred_df = pd.DataFrame(
            {
                "split": split,
                "target": y_binary,
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

    metrics_path = outdir / "multiclass_dnn_metrics.csv"
    cats_path = outdir / "cmsstyle_category_yields.csv"
    scan_path = outdir / "cmsstyle_hh_score_scan.csv"
    stable_path = outdir / "cmsstyle_stable_thresholds.csv"
    confusion_path = outdir / "multiclass_confusion_matrix.csv"
    history_path = outdir / "multiclass_training_history.csv"
    predictions_path = outdir / "multiclass_dnn_predictions.parquet"
    model_path = outdir / "tabular_multiclass_dnn.pt"
    summary_path = outdir / "multiclass_dnn_summary.json"

    metrics.to_csv(metrics_path, index=False)
    cats.to_csv(cats_path, index=False)
    scan.to_csv(scan_path, index=False)
    stable.to_csv(stable_path, index=False)
    confusion.to_csv(confusion_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)
    pred_df.to_parquet(predictions_path, index=False)

    torch.save(
        {
            "model_state_dict": best_state,
            "n_features": X.shape[1],
            "n_classes": n_classes,
            "class_names": class_names,
            "hh_idx": hh_idx,
            "hidden": hidden,
            "dropout": args.dropout,
            "feature_names": feature_names,
            "best_epoch": best_epoch,
            "best_val_hh_auc_weighted": best_val_hh_auc_weighted,
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
        "hh_idx": int(hh_idx),
        "hh_class_name": class_names[hh_idx],
        "best_epoch": int(best_epoch),
        "best_val_hh_auc_weighted": float(best_val_hh_auc_weighted),
        "metrics": metrics.to_dict(orient="records"),
        "stable_thresholds": stable.to_dict(orient="records"),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Multiclass DNN metrics ===")
    print(metrics.to_string(index=False))

    print("\n=== CMS-style argmax category yields, validation+test ===")
    print(cats[cats["split"].isin(["val", "test", "valtest"])].to_string(index=False))

    print("\n=== CMS-style stable HH score thresholds ===")
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
