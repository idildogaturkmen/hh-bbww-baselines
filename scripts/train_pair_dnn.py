import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


NPZ_DIR = Path("outputs/hbb_npz")
BASELINE_DIR = Path("outputs/hbb_reconstruction")
OUTDIR = Path("outputs/hbb_pair_dnn")
PLOT_DIR = Path("outputs/plots/hbb_pair_dnn")
COMPARISON_PLOT_DIR = Path("outputs/plots/hbb_reconstruction")
M_H = 125.0
MAX_JETS = 12

OUTDIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_PLOT_DIR.mkdir(parents=True, exist_ok=True)

JET_FEATURE_NAMES = ["pt", "eta", "phi", "mass", "btag", "btagPhys"]
PAIR_EXTRA_NAMES = [
    "mjj",
    "deltaR",
    "pt_sum",
    "pt_abs_diff",
    "mass_sum",
    "btag_sum",
    "btag_max",
    "btag_min",
]
METHOD_LABEL = "Pair-DNN"


class PairScorer(nn.Module):
    def __init__(self, n_features, hidden=96, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, pair_features, pair_mask):
        logits = self.net(pair_features).squeeze(-1)
        return logits.masked_fill(~pair_mask, -1.0e9)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_split(split):
    data = np.load(NPZ_DIR / f"{split}.npz")
    return data["jets"].astype(np.float32), data["mask"].astype(bool), data["labels"].astype(np.int64)


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


def pair_indices(max_jets=MAX_JETS):
    return list(itertools.combinations(range(max_jets), 2))


def build_pair_arrays(jets, mask, labels, pair_idx):
    n_events = jets.shape[0]
    n_pairs = len(pair_idx)
    n_features = 2 * jets.shape[2] + len(PAIR_EXTRA_NAMES)

    pair_features = np.zeros((n_events, n_pairs, n_features), dtype=np.float32)
    pair_mask = np.zeros((n_events, n_pairs), dtype=bool)
    target = np.full((n_events,), -1, dtype=np.int64)
    pair_mbb = np.zeros((n_events, n_pairs), dtype=np.float32)
    pair_delta_r = np.zeros((n_events, n_pairs), dtype=np.float32)

    pair_to_index = {pair_key(pair): idx for idx, pair in enumerate(pair_idx)}

    for ev in range(n_events):
        truth = pair_key(labels[ev])
        target[ev] = pair_to_index[truth]

        for pidx, (i, j) in enumerate(pair_idx):
            valid = bool(mask[ev, i] and mask[ev, j])
            pair_mask[ev, pidx] = valid
            if not valid:
                continue

            ji = jets[ev, i]
            jj = jets[ev, j]
            # Canonicalize the unordered pair by descending pT before concatenation.
            if jj[0] > ji[0]:
                ji, jj = jj, ji

            mbb = invariant_mass(ji, jj)
            dr = delta_r(ji, jj)
            extras = np.array(
                [
                    mbb,
                    dr,
                    ji[0] + jj[0],
                    abs(ji[0] - jj[0]),
                    ji[3] + jj[3],
                    ji[4] + jj[4],
                    max(ji[4], jj[4]),
                    min(ji[4], jj[4]),
                ],
                dtype=np.float32,
            )

            pair_features[ev, pidx] = np.concatenate([ji, jj, extras]).astype(np.float32)
            pair_mbb[ev, pidx] = mbb
            pair_delta_r[ev, pidx] = dr

    if np.any(target < 0):
        raise ValueError("Found event whose truth pair is not in the pair index list.")

    return pair_features, pair_mask, target, pair_mbb, pair_delta_r


def fit_scaler(train_features, train_mask):
    vals = train_features[train_mask]
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scaler(features, mean, std):
    return ((features - mean) / std).astype(np.float32)


def make_loader(features, pair_mask, target, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(pair_mask),
        torch.from_numpy(target),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total = 0
    correct_at = {1: 0, 3: 0, 5: 0}

    with torch.no_grad():
        for features, pair_mask, target in loader:
            features = features.to(device)
            pair_mask = pair_mask.to(device)
            target = target.to(device)

            logits = model(features, pair_mask)
            loss = loss_fn(logits, target)
            total_loss += float(loss.item()) * len(target)
            total += len(target)

            for k in correct_at:
                pred = torch.topk(logits, k=k, dim=1).indices
                correct_at[k] += int((pred == target[:, None]).any(dim=1).sum().item())

    return {
        "loss": total_loss / total,
        "top1": correct_at[1] / total,
        "top3": correct_at[3] / total,
        "top5": correct_at[5] / total,
    }


def predict_logits(model, features, pair_mask, batch_size, device):
    loader = make_loader(features, pair_mask, np.zeros(len(features), dtype=np.int64), batch_size, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch_features, batch_mask, _ in loader:
            logits = model(batch_features.to(device), batch_mask.to(device))
            outputs.append(logits.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def split_metrics(split, logits, target, pair_mbb):
    n = len(target)
    selected = np.argmax(logits, axis=1)
    selected_mbb = pair_mbb[np.arange(n), selected]
    rows = []
    metrics = {"split": split, "method": "pair_dnn", "method_label": METHOD_LABEL, "n_events": n}

    for k, name in [(1, "exact_pair_accuracy"), (3, "top3_pair_accuracy"), (5, "top5_pair_accuracy")]:
        pred = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
        acc = float(np.mean(np.any(pred == target[:, None], axis=1)))
        metrics[name] = acc
        metrics[name + "_err"] = binomial_error(acc, n)

    metrics.update(
        {
            "selected_mbb_mean": float(np.mean(selected_mbb)),
            "selected_mbb_median": float(np.median(selected_mbb)),
            "selected_mbb_p16": float(np.percentile(selected_mbb, 16)),
            "selected_mbb_p84": float(np.percentile(selected_mbb, 84)),
            "selected_mbb_frac_90_140": float(np.mean((selected_mbb >= 90.0) & (selected_mbb <= 140.0))),
            "selected_mbb_frac_100_150": float(np.mean((selected_mbb >= 100.0) & (selected_mbb <= 150.0))),
        }
    )

    for ev in range(n):
        order = np.argsort(-logits[ev])
        rows.append(
            {
                "split": split,
                "event": ev,
                "target_pair_index": int(target[ev]),
                "pred_pair_index": int(selected[ev]),
                "correct": bool(selected[ev] == target[ev]),
                "selected_mbb": float(selected_mbb[ev]),
                "target_rank": int(np.where(order == target[ev])[0][0] + 1),
            }
        )

    return metrics, rows


def accuracy_summary(logits, target):
    n = len(target)
    row = {"n_events": n}
    for k, name in [(1, "exact_pair_accuracy"), (3, "top3_pair_accuracy"), (5, "top5_pair_accuracy")]:
        pred = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
        acc = float(np.mean(np.any(pred == target[:, None], axis=1)))
        row[name] = acc
        row[name + "_err"] = binomial_error(acc, n)
    return row


def make_slice_metrics(split, logits, target, pair_mbb, pair_delta_r, jets, mask, labels):
    selected = np.argmax(logits, axis=1)
    selected_mbb = pair_mbb[np.arange(len(target)), selected]
    target_delta_r = pair_delta_r[np.arange(len(target)), target]
    n_jets = np.sum(mask, axis=1).astype(int)
    truth_min_pt = np.min(jets[np.arange(len(labels))[:, None], labels, 0], axis=1)

    rows = []

    def add_rows(variable, values, bins=None):
        if bins is None:
            bin_defs = [(float(v), float(v), values == v) for v in sorted(np.unique(values))]
        else:
            bin_defs = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                sel = (values >= lo) & (values < hi)
                bin_defs.append((float(lo), float(hi), sel))

        for lo, hi, sel in bin_defs:
            n = int(np.sum(sel))
            if n == 0:
                continue
            row = {
                "split": split,
                "method": "pair_dnn",
                "method_label": METHOD_LABEL,
                "variable": variable,
                "bin_low": lo,
                "bin_high": hi,
            }
            row.update(accuracy_summary(logits[sel], target[sel]))
            row.update(
                {
                    "selected_mbb_mean": float(np.mean(selected_mbb[sel])),
                    "selected_mbb_median": float(np.median(selected_mbb[sel])),
                    "truth_deltaR_mean": float(np.mean(target_delta_r[sel])),
                    "truth_min_bjet_pt_mean": float(np.mean(truth_min_pt[sel])),
                }
            )
            rows.append(row)

    add_rows("n_jets", n_jets)
    add_rows("truth_min_bjet_pt", truth_min_pt, bins=[0.0, 30.0, 50.0, 75.0, 100.0, 150.0, np.inf])
    add_rows("truth_deltaR_bb", target_delta_r, bins=[0.0, 0.4, 0.8, 1.2, 2.0, 3.0, np.inf])

    return rows


def plot_history(history):
    df = pd.DataFrame(history)
    plt.figure(figsize=(7.0, 4.8))
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Pair-DNN training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "training_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    plt.plot(df["epoch"], df["train_top1"], label="train top-1")
    plt.plot(df["epoch"], df["val_top1"], label="validation top-1")
    plt.plot(df["epoch"], df["val_top3"], label="validation top-3")
    plt.xlabel("Epoch")
    plt.ylabel("Truth-pair accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Pair-DNN accuracy curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "training_accuracy.png", dpi=200)
    plt.close()


def plot_dnn_mbb(test_predictions):
    vals = test_predictions["selected_mbb"].to_numpy()
    plt.figure(figsize=(7.0, 4.8))
    plt.hist(vals, bins=np.linspace(0, 250, 70), histtype="step", linewidth=1.6, density=True)
    plt.axvline(M_H, color="black", linestyle="--", linewidth=1.0, label="125 GeV")
    plt.xlabel("Pair-DNN selected $m_{jj}$ [GeV]")
    plt.ylabel("Normalized events")
    plt.title("Pair-DNN selected H->bb candidate mass on test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "test_selected_mbb.png", dpi=200)
    plt.close()


def write_summary(metrics_df, args, best_epoch):
    test = metrics_df[metrics_df["split"] == "test"].iloc[0]
    lines = [
        "# Pair-DNN H->bb Assignment Baseline",
        "",
        "This is a lightweight learned baseline between the hand-built heuristics and SPA-Net. It scores each unordered AK4 jet pair independently with a shared MLP.",
        "",
        "## Inputs",
        "",
        "- Up to 12 AK4 jets per event",
        "- Per-jet features: `[pt, eta, phi, mass, btag, btagPhys]`",
        "- Pair extras: `[mjj, deltaR, pt_sum, pt_abs_diff, mass_sum, btag_sum, btag_max, btag_min]`",
        "- Target: unordered truth-matched H -> bb jet pair",
        "",
        "## Training",
        "",
        f"- Seed: `{args.seed}`",
        f"- Epochs requested: `{args.epochs}`",
        f"- Best validation epoch: `{best_epoch}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Learning rate: `{args.lr}`",
        f"- Weight decay: `{args.weight_decay}`",
        "",
        "## Test Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Top-1 exact pair accuracy | {test['exact_pair_accuracy']:.4f} +/- {test['exact_pair_accuracy_err']:.4f} |",
        f"| Top-3 pair accuracy | {test['top3_pair_accuracy']:.4f} +/- {test['top3_pair_accuracy_err']:.4f} |",
        f"| Top-5 pair accuracy | {test['top5_pair_accuracy']:.4f} +/- {test['top5_pair_accuracy_err']:.4f} |",
        f"| Median selected m_jj | {test['selected_mbb_median']:.2f} GeV |",
        f"| Fraction with 90 < m_jj < 140 GeV | {test['selected_mbb_frac_90_140']:.3f} |",
        "",
        "The model is trained on truth-pair labels, not on closeness to 125 GeV. The selected m_jj distribution is therefore a diagnostic, not the optimization target.",
    ]
    Path("RESULTS_HBB_PAIR_DNN.md").write_text("\n".join(lines) + "\n")


def update_combined_comparison(dnn_metrics):
    baseline_path = BASELINE_DIR / "baseline_metrics.csv"
    if not baseline_path.exists():
        return

    baseline = pd.read_csv(baseline_path)
    combined = pd.concat([baseline, dnn_metrics], ignore_index=True, sort=False)
    combined.to_csv(BASELINE_DIR / "model_comparison_metrics.csv", index=False)

    test = combined[combined["split"] == "test"].copy()
    method_order = ["top2_btag", "closest_mass", "combined_btag_mass", "pair_dnn"]
    test["method"] = pd.Categorical(test["method"], categories=method_order, ordered=True)
    test = test.sort_values("method")

    x = np.arange(len(test))
    width = 0.25
    plt.figure(figsize=(9.0, 5.0))
    for offset, col, label in [
        (-width, "exact_pair_accuracy", "Top-1"),
        (0.0, "top3_pair_accuracy", "Top-3"),
        (width, "top5_pair_accuracy", "Top-5"),
    ]:
        plt.bar(
            x + offset,
            test[col],
            width=width,
            yerr=test[col + "_err"],
            capsize=3,
            label=label,
        )
    plt.xticks(x, test["method_label"], rotation=20, ha="right")
    plt.ylabel("Truth-pair efficiency / accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("H->bb pair-assignment comparison on test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT_DIR / "test_accuracy_comparison_with_pair_dnn.png", dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight pair-DNN H->bb assignment baseline.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pair_idx = pair_indices()
    with open(OUTDIR / "pair_indices.json", "w") as f:
        json.dump([list(pair) for pair in pair_idx], f, indent=2)

    raw = {}
    arrays = {}
    for split in ["train", "val", "test"]:
        jets, mask, labels = load_split(split)
        raw[split] = {"jets": jets, "mask": mask, "labels": labels}
        arrays[split] = build_pair_arrays(jets, mask, labels, pair_idx)
        print(f"{split}: jets={jets.shape}, pair_features={arrays[split][0].shape}")

    mean, std = fit_scaler(arrays["train"][0], arrays["train"][1])
    np.savez(OUTDIR / "pair_feature_scaler.npz", mean=mean, std=std)

    features = {}
    for split in ["train", "val", "test"]:
        pair_features, pair_mask, target, pair_mbb, pair_delta_r = arrays[split]
        features[split] = (
            apply_scaler(pair_features, mean, std),
            pair_mask,
            target,
            pair_mbb,
            pair_delta_r,
        )

    train_loader = make_loader(features["train"][0], features["train"][1], features["train"][2], args.batch_size, True)
    val_loader = make_loader(features["val"][0], features["val"][1], features["val"][2], args.batch_size, False)
    test_loader = make_loader(features["test"][0], features["test"][1], features["test"][2], args.batch_size, False)

    model = PairScorer(features["train"][0].shape[-1], hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val = -np.inf
    best_epoch = 0
    stale_epochs = 0
    history = []
    best_path = OUTDIR / "pair_dnn_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        train_correct = 0
        for batch_features, batch_mask, batch_target in train_loader:
            batch_features = batch_features.to(device)
            batch_mask = batch_mask.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features, batch_mask)
            loss = loss_fn(logits, batch_target)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch_target)
            total += len(batch_target)
            train_correct += int((torch.argmax(logits.detach(), dim=1) == batch_target).sum().item())

        train_loss = total_loss / total
        train_top1 = train_correct / total
        val_eval = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_loss": val_eval["loss"],
            "val_top1": val_eval["top1"],
            "val_top3": val_eval["top3"],
            "val_top5": val_eval["top5"],
        }
        history.append(row)

        print(
            f"epoch {epoch:03d} | train loss {train_loss:.4f} top1 {train_top1:.4f} | "
            f"val loss {val_eval['loss']:.4f} top1 {val_eval['top1']:.4f} top3 {val_eval['top3']:.4f}",
            flush=True,
        )

        if val_eval["top1"] > best_val:
            best_val = val_eval["top1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "pair_feature_names": [
                        *(f"jet1_{name}" for name in JET_FEATURE_NAMES),
                        *(f"jet2_{name}" for name in JET_FEATURE_NAMES),
                        *PAIR_EXTRA_NAMES,
                    ],
                    "pair_indices": pair_idx,
                    "scaler_mean": mean,
                    "scaler_std": std,
                    "best_epoch": best_epoch,
                    "best_val_top1": best_val,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs; best epoch {best_epoch}.")
                break

    pd.DataFrame(history).to_csv(OUTDIR / "training_history.csv", index=False)
    plot_history(history)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = []
    prediction_rows = []
    slice_rows = []
    logits_by_split = {}
    for split in ["train", "val", "test"]:
        split_features, split_mask, split_target, split_mbb, _ = features[split]
        logits = predict_logits(model, split_features, split_mask, args.batch_size, device)
        logits_by_split[split] = logits
        split_metrics_row, split_prediction_rows = split_metrics(split, logits, split_target, split_mbb)
        metrics.append(split_metrics_row)
        prediction_rows.extend(split_prediction_rows)
        slice_rows.extend(
            make_slice_metrics(
                split,
                logits,
                split_target,
                split_mbb,
                features[split][4],
                raw[split]["jets"],
                raw[split]["mask"],
                raw[split]["labels"],
            )
        )

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.DataFrame(prediction_rows)
    slice_df = pd.DataFrame(slice_rows)
    metrics_df.to_csv(OUTDIR / "metrics.csv", index=False)
    predictions_df.to_csv(OUTDIR / "predictions.csv", index=False)
    slice_df.to_csv(OUTDIR / "slice_metrics.csv", index=False)
    plot_dnn_mbb(predictions_df[predictions_df["split"] == "test"])
    update_combined_comparison(metrics_df)
    write_summary(metrics_df, args, best_epoch)

    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_top1": float(best_val),
                "args": vars(args),
                "device": str(device),
                "test_metrics": metrics_df[metrics_df["split"] == "test"].iloc[0].to_dict(),
            },
            f,
            indent=2,
        )

    print("\nPair-DNN metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved outputs to {OUTDIR}")
    print(f"Saved plots to {PLOT_DIR}")
    print("Updated combined comparison if baseline metrics were available.")


if __name__ == "__main__":
    main()
