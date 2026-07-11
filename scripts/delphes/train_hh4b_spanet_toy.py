#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


LUMI_PB = 450_000.0


def make_pair_indices(n_jets):
    pairs = []
    for i in range(n_jets):
        for j in range(i + 1, n_jets):
            pairs.append((i, j))
    return torch.tensor(pairs, dtype=torch.long)


def preprocess_features(X, mask, mean=None, std=None):
    pt = X[:, :, 0]
    eta = X[:, :, 1]
    phi = X[:, :, 2]
    mass = X[:, :, 3]
    btag = X[:, :, 4]

    F6 = np.stack(
        [
            np.log1p(np.maximum(pt, 0.0)),
            eta,
            np.sin(phi),
            np.cos(phi),
            np.log1p(np.maximum(mass, 0.0)),
            btag,
        ],
        axis=-1,
    ).astype(np.float32)

    if mean is None or std is None:
        real = F6[mask]
        mean = real.mean(axis=0)
        std = real.std(axis=0)
        std[std < 1e-6] = 1.0

    F6 = (F6 - mean) / std
    F6[~mask] = 0.0
    return F6.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


class HH4BDataset(Dataset):
    def __init__(self, npz_path, mean=None, std=None):
        d = np.load(npz_path, allow_pickle=True)
        self.raw_X = d["X_jets"].astype(np.float32)
        self.mask = d["jet_mask"].astype(bool)
        self.X, self.mean, self.std = preprocess_features(self.raw_X, self.mask, mean, std)

        self.y = d["y"].astype(np.float32)
        self.assignment = d["assignment"].astype(np.int64)
        self.assignment_mask = d["assignment_mask"].astype(np.float32)
        self.weight_pb = d["weight_pb"].astype(np.float64)
        self.sample_id = d["sample_id"].astype(np.int64)
        self.sample_names = d["sample_names"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "X": torch.tensor(self.X[idx], dtype=torch.float32),
            "mask": torch.tensor(self.mask[idx], dtype=torch.bool),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "assignment": torch.tensor(self.assignment[idx], dtype=torch.long),
            "assignment_mask": torch.tensor(self.assignment_mask[idx], dtype=torch.float32),
            "weight_pb": self.weight_pb[idx],
            "sample_id": self.sample_id[idx],
        }


class ToySPANet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=96, n_heads=4, n_layers=2, dropout=0.10):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.pair_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # H1 pair logit, H2 pair logit
        )

    def forward(self, X, mask, pair_indices):
        z = self.embed(X)
        z = self.encoder(z, src_key_padding_mask=~mask)

        masked_z = z * mask.unsqueeze(-1)
        pooled = masked_z.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        cls_logit = self.cls_head(pooled).squeeze(-1)

        i = pair_indices[:, 0]
        j = pair_indices[:, 1]
        zi = z[:, i, :]
        zj = z[:, j, :]

        pair_feat = torch.cat([zi, zj, torch.abs(zi - zj), zi * zj], dim=-1)
        pair_logits = self.pair_head(pair_feat)  # B, n_pairs, 2

        pair_mask = mask[:, i] & mask[:, j]
        pair_logits = pair_logits.masked_fill(~pair_mask[:, :, None], -1e9)

        return cls_logit, pair_logits


def assignment_targets(assignments, pair_indices):
    device = assignments.device
    B = assignments.shape[0]
    P = pair_indices.shape[0]

    pairs_sorted = torch.sort(pair_indices, dim=1).values
    targets = torch.full((B, 2), -1, dtype=torch.long, device=device)

    for h in range(2):
        truth_pair = torch.sort(assignments[:, h, :], dim=1).values
        eq = (pairs_sorted[None, :, :] == truth_pair[:, None, :]).all(dim=-1)
        any_match = eq.any(dim=1)
        if any_match.any():
            targets[any_match, h] = eq[any_match].float().argmax(dim=1).long()

    return targets


def assignment_loss_and_acc(pair_logits, assignments, assignment_mask, pair_indices):
    good = assignment_mask > 0.5
    if good.sum() == 0:
        return pair_logits.sum() * 0.0, float("nan")

    logits = pair_logits[good]
    assign = assignments[good]
    targets = assignment_targets(assign, pair_indices)

    valid = (targets[:, 0] >= 0) & (targets[:, 1] >= 0)
    if valid.sum() == 0:
        return pair_logits.sum() * 0.0, float("nan")

    logits = logits[valid]
    targets = targets[valid]

    loss_direct = (
        F.cross_entropy(logits[:, :, 0], targets[:, 0], reduction="none")
        + F.cross_entropy(logits[:, :, 1], targets[:, 1], reduction="none")
    )
    loss_swap = (
        F.cross_entropy(logits[:, :, 0], targets[:, 1], reduction="none")
        + F.cross_entropy(logits[:, :, 1], targets[:, 0], reduction="none")
    )
    loss = torch.minimum(loss_direct, loss_swap).mean()

    pred0 = logits[:, :, 0].argmax(dim=1)
    pred1 = logits[:, :, 1].argmax(dim=1)
    correct = (
        ((pred0 == targets[:, 0]) & (pred1 == targets[:, 1]))
        | ((pred0 == targets[:, 1]) & (pred1 == targets[:, 0]))
    )
    acc = correct.float().mean().item()

    return loss, acc


def predict(model, dataset, device, batch_size=2048):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pair_indices = make_pair_indices(dataset.X.shape[1]).to(device)

    scores = []
    ys = []
    weights = []
    sample_ids = []
    assignment_masks = []
    assignment_corrects = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)
            assignment = batch["assignment"].to(device)
            assignment_mask = batch["assignment_mask"].to(device)

            cls_logit, pair_logits = model(X, mask, pair_indices)
            score = torch.sigmoid(cls_logit)

            _, acc = assignment_loss_and_acc(pair_logits, assignment, assignment_mask, pair_indices)

            scores.append(score.cpu().numpy())
            ys.append(y.cpu().numpy())
            weights.append(batch["weight_pb"].numpy())
            sample_ids.append(batch["sample_id"].numpy())
            assignment_masks.append(batch["assignment_mask"].numpy())

    return {
        "score": np.concatenate(scores),
        "y": np.concatenate(ys).astype(int),
        "weight_pb": np.concatenate(weights),
        "sample_id": np.concatenate(sample_ids),
        "assignment_mask": np.concatenate(assignment_masks),
    }


def scaled_test_weights(test_pred, all_dataset):
    all_counts = pd.Series(all_dataset.sample_id).value_counts().to_dict()
    test_counts = pd.Series(test_pred["sample_id"]).value_counts().to_dict()

    scale = np.ones(len(test_pred["sample_id"]), dtype=float)
    for i, sid in enumerate(test_pred["sample_id"]):
        scale[i] = all_counts.get(sid, 1) / max(test_counts.get(sid, 1), 1)

    return test_pred["weight_pb"] * scale


def safe_auc(y, score, weight=None):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, score, sample_weight=weight)


def evaluate_and_write(model, train_ds, val_ds, test_ds, all_ds, outdir, device):
    test = predict(model, test_ds, device)
    w_scaled = scaled_test_weights(test, all_ds)

    sample_names = [str(x) for x in test_ds.sample_names]
    sid_to_name = {i: s for i, s in enumerate(sample_names)}

    is_signal = test["y"] == 1
    is_qcd = np.array([sid_to_name[int(sid)].startswith("qcd_") for sid in test["sample_id"]])
    is_top = np.array(["ttbar" in sid_to_name[int(sid)] for sid in test["sample_id"]])

    qcd_task = is_signal | is_qcd
    top_task = is_signal | is_top

    summary = {
        "test_auc_all_unweighted": safe_auc(test["y"], test["score"]),
        "test_auc_all_weighted": safe_auc(test["y"], test["score"], w_scaled),
        "test_qcd_weighted_auc": safe_auc(test["y"][qcd_task], test["score"][qcd_task], w_scaled[qcd_task]),
        "test_top_weighted_auc": safe_auc(test["y"][top_task], test["score"][top_task], w_scaled[top_task]),
        "test_assignment_mask_events": int(test["assignment_mask"].sum()),
    }

    rows = []
    for thr in np.linspace(0.01, 0.99, 99):
        sel = test["score"] >= thr
        s = w_scaled[sel & is_signal].sum() * LUMI_PB
        b = w_scaled[sel & ~is_signal].sum() * LUMI_PB
        rows.append({
            "threshold": thr,
            "signal_events_450fb": s,
            "background_events_450fb": b,
            "S_over_B": s / b if b > 0 else np.nan,
            "S_over_sqrtB": s / np.sqrt(b) if b > 0 else np.nan,
            "n_signal_test_rows": int((sel & is_signal).sum()),
            "n_background_test_rows": int((sel & ~is_signal).sum()),
        })

    scan = pd.DataFrame(rows)
    stable = scan[(scan["n_signal_test_rows"] >= 20) & (scan["n_background_test_rows"] >= 50)]
    best = stable.sort_values("S_over_sqrtB", ascending=False).iloc[0].to_dict()

    for k, v in best.items():
        summary[f"best_{k}"] = v

    comp_rows = []
    thr = best["threshold"]
    sel = test["score"] >= thr
    for sid, name in sid_to_name.items():
        m = sel & (test["sample_id"] == sid)
        expected = w_scaled[m].sum() * LUMI_PB
        comp_rows.append({
            "sample": name,
            "selected_test_rows": int(m.sum()),
            "expected_events_450fb_scaled": expected,
            "group": "signal" if "HH4b" in name else "background",
        })
    comp = pd.DataFrame(comp_rows)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(outdir / "spanet_toy_summary.csv", index=False)
    scan.to_csv(outdir / "spanet_toy_threshold_scan.csv", index=False)
    comp.to_csv(outdir / "spanet_toy_best_composition.csv", index=False)

    (outdir / "spanet_toy_summary.md").write_text(summary_df.to_markdown(index=False) + "\n")
    (outdir / "spanet_toy_best_composition.md").write_text(comp.to_markdown(index=False) + "\n")

    print("\n=== Test summary ===")
    print(summary_df.to_string(index=False))

    print("\n=== Best threshold composition ===")
    print(comp.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="nominal_btag4_v0")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--assign-loss-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    repo = Path(os.environ["HH4B_REPO"])
    store = Path(os.environ["HH4B_STORE"])
    data_dir = store / "spanet_npz" / args.tag

    outdir = repo / "outputs/tables" / f"hh4b_spanet_toy_{args.tag}_2026_07_10"
    outdir.mkdir(parents=True, exist_ok=True)

    train_raw = np.load(data_dir / "hh4b_spanet_leading8_train.npz", allow_pickle=True)
    train_X, mean, std = preprocess_features(train_raw["X_jets"], train_raw["jet_mask"])

    train_ds = HH4BDataset(data_dir / "hh4b_spanet_leading8_train.npz", mean, std)
    val_ds = HH4BDataset(data_dir / "hh4b_spanet_leading8_val.npz", mean, std)
    test_ds = HH4BDataset(data_dir / "hh4b_spanet_leading8_test.npz", mean, std)
    all_ds = HH4BDataset(data_dir / "hh4b_spanet_leading8_all.npz", mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = ToySPANet(input_dim=train_ds.X.shape[-1]).to(device)
    pair_indices = make_pair_indices(train_ds.X.shape[1]).to(device)

    n_pos = max(float(train_ds.y.sum()), 1.0)
    n_neg = max(float(len(train_ds.y) - train_ds.y.sum()), 1.0)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    print("pos_weight:", pos_weight.item())

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    history = []
    best_val_auc = -1
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        cls_losses = []
        ass_losses = []
        ass_accs = []

        for batch in train_loader:
            X = batch["X"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)
            assignment = batch["assignment"].to(device)
            assignment_mask = batch["assignment_mask"].to(device)

            opt.zero_grad()
            cls_logit, pair_logits = model(X, mask, pair_indices)

            cls_loss = F.binary_cross_entropy_with_logits(cls_logit, y, pos_weight=pos_weight)
            ass_loss, ass_acc = assignment_loss_and_acc(pair_logits, assignment, assignment_mask, pair_indices)

            loss = cls_loss + args.assign_loss_weight * ass_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            losses.append(loss.item())
            cls_losses.append(cls_loss.item())
            ass_losses.append(ass_loss.item())
            if not np.isnan(ass_acc):
                ass_accs.append(ass_acc)

        val = predict(model, val_ds, device)
        val_auc = safe_auc(val["y"], val["score"])

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_cls_loss": float(np.mean(cls_losses)),
            "train_assignment_loss": float(np.mean(ass_losses)),
            "train_assignment_acc_batch_mean": float(np.mean(ass_accs)) if ass_accs else np.nan,
            "val_auc_unweighted": val_auc,
        }
        history.append(row)

        print(row)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    hist = pd.DataFrame(history)
    hist.to_csv(outdir / "training_history.csv", index=False)
    (outdir / "training_history.md").write_text(hist.to_markdown(index=False) + "\n")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "args": vars(args),
        },
        outdir / "spanet_toy_model.pt",
    )

    evaluate_and_write(model, train_ds, val_ds, test_ds, all_ds, outdir, device)

    print("\nWrote:", outdir)


if __name__ == "__main__":
    main()
