#!/usr/bin/env python3

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader


LUMI_PB = 450_000.0
QCD_LIKE_SAMPLE_IDS = {3, 4, 5, 6, 7}  # Zbbbb + QCD slices
TOP_SAMPLE_IDS = {2}                   # ttbar
SIGNAL_SAMPLE_IDS = {0, 1}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_md(df, path):
    try:
        path.write_text(df.to_markdown(index=False) + "\n")
    except Exception:
        path.write_text(df.to_string(index=False) + "\n")


def safe_auc(y, score, weight=None):
    y = np.asarray(y)
    score = np.asarray(score)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, score, sample_weight=weight)


def delta_phi(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


class HH4BDeepSPANetDataset(Dataset):
    def __init__(self, path, jet_mean=None, jet_std=None, global_mean=None, global_std=None):
        z = np.load(path, allow_pickle=True)

        self.path = Path(path)
        self.raw_x = z["X_jets"].astype(np.float32)
        self.jet_mask = z["jet_mask"].astype(bool)
        self.y = z["y"].astype(np.float32)
        self.assignment = z["assignment"].astype(np.int64)
        self.assignment_mask = z["assignment_mask"].astype(np.int64)
        self.weight_pb = z["weight_pb"].astype(np.float32)
        self.sample_id = z["sample_id"].astype(np.int64)
        self.event_id = z["event_id"].astype(np.int64)
        self.n_selected_jets = z["n_selected_jets"].astype(np.int64)
        self.n_selected_btags = z["n_selected_btags"].astype(np.int64)
        self.feature_names = z["feature_names"].astype(str)
        self.sample_names = z["sample_names"].astype(str)
        self.sample_groups = z["sample_groups"].astype(str)

        self.x = self.make_jet_features(self.raw_x)
        self.global_x = self.make_global_features(self.raw_x, self.jet_mask)

        if jet_mean is None:
            valid = self.jet_mask
            flat = self.x[valid]
            jet_mean = flat.mean(axis=0)
            jet_std = flat.std(axis=0)
            jet_std[jet_std < 1e-6] = 1.0

        if global_mean is None:
            global_mean = self.global_x.mean(axis=0)
            global_std = self.global_x.std(axis=0)
            global_std[global_std < 1e-6] = 1.0

        self.jet_mean = jet_mean.astype(np.float32)
        self.jet_std = jet_std.astype(np.float32)
        self.global_mean = global_mean.astype(np.float32)
        self.global_std = global_std.astype(np.float32)

        self.x = (self.x - self.jet_mean) / self.jet_std
        self.x[~self.jet_mask] = 0.0

        self.global_x = (self.global_x - self.global_mean) / self.global_std

    @staticmethod
    def make_jet_features(raw):
        pt = raw[..., 0]
        eta = raw[..., 1]
        phi = raw[..., 2]
        mass = raw[..., 3]
        btag = raw[..., 4]

        return np.stack(
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

    @staticmethod
    def make_global_features(raw, mask):
        pt = raw[..., 0]
        eta = raw[..., 1]
        phi = raw[..., 2]
        mass = raw[..., 3]
        btag = raw[..., 4]

        masked_pt = np.where(mask, pt, 0.0)
        n_jets = mask.sum(axis=1).astype(np.float32)
        n_btags = ((btag > 0.5) & mask).sum(axis=1).astype(np.float32)
        ht = masked_pt.sum(axis=1)

        sorted_pt = np.sort(masked_pt, axis=1)[:, ::-1]
        pt1 = sorted_pt[:, 0]
        pt2 = sorted_pt[:, 1]
        pt3 = sorted_pt[:, 2]
        pt4 = sorted_pt[:, 3]

        mean_pt = ht / np.maximum(n_jets, 1.0)
        mean_btag = np.where(mask, btag, 0.0).sum(axis=1) / np.maximum(n_jets, 1.0)
        max_btag = np.where(mask, btag, -999.0).max(axis=1)
        max_btag[max_btag < -100] = 0.0

        # Pairwise summary features: min deltaR and median dijet mass among valid pairs.
        min_dr = []
        med_mjj = []
        for i in range(raw.shape[0]):
            pairs_dr = []
            pairs_m = []
            valid = np.where(mask[i])[0]
            for a_i in range(len(valid)):
                for b_i in range(a_i + 1, len(valid)):
                    a = valid[a_i]
                    b = valid[b_i]
                    deta = eta[i, a] - eta[i, b]
                    dphi = delta_phi(phi[i, a], phi[i, b])
                    dr = math.sqrt(float(deta * deta + dphi * dphi))
                    pairs_dr.append(dr)

                    e1 = math.sqrt(max(float(pt[i, a] ** 2 * math.cosh(eta[i, a]) ** 2 + mass[i, a] ** 2), 0.0))
                    e2 = math.sqrt(max(float(pt[i, b] ** 2 * math.cosh(eta[i, b]) ** 2 + mass[i, b] ** 2), 0.0))
                    px1 = float(pt[i, a] * math.cos(phi[i, a]))
                    py1 = float(pt[i, a] * math.sin(phi[i, a]))
                    pz1 = float(pt[i, a] * math.sinh(eta[i, a]))
                    px2 = float(pt[i, b] * math.cos(phi[i, b]))
                    py2 = float(pt[i, b] * math.sin(phi[i, b]))
                    pz2 = float(pt[i, b] * math.sinh(eta[i, b]))
                    m2 = (e1 + e2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2
                    pairs_m.append(math.sqrt(max(m2, 0.0)))

            min_dr.append(min(pairs_dr) if pairs_dr else 0.0)
            med_mjj.append(float(np.median(pairs_m)) if pairs_m else 0.0)

        return np.stack(
            [
                n_jets,
                n_btags,
                np.log1p(ht),
                np.log1p(pt1),
                np.log1p(pt2),
                np.log1p(pt3),
                np.log1p(pt4),
                mean_pt / 100.0,
                mean_btag,
                max_btag,
                np.asarray(min_dr, dtype=np.float32),
                np.log1p(np.asarray(med_mjj, dtype=np.float32)),
            ],
            axis=1,
        ).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx], dtype=torch.float32),
            "mask": torch.tensor(self.jet_mask[idx], dtype=torch.bool),
            "global_x": torch.tensor(self.global_x[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "assignment": torch.tensor(self.assignment[idx], dtype=torch.long),
            "assignment_mask": torch.tensor(self.assignment_mask[idx], dtype=torch.long),
            "weight_pb": torch.tensor(self.weight_pb[idx], dtype=torch.float32),
            "sample_id": torch.tensor(self.sample_id[idx], dtype=torch.long),
            "event_id": torch.tensor(self.event_id[idx], dtype=torch.long),
        }


class DeepTwoHeadSPANet(nn.Module):
    def __init__(
        self,
        jet_dim=6,
        global_dim=12,
        hidden_dim=128,
        n_heads=4,
        n_layers=3,
        dropout=0.10,
        max_jets=8,
    ):
        super().__init__()

        self.max_jets = max_jets
        self.hidden_dim = hidden_dim

        self.jet_embed = nn.Sequential(
            nn.Linear(jet_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.rank_embed = nn.Embedding(max_jets, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.global_embed = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        event_dim = 2 * hidden_dim

        self.qcd_head = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.top_head = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        pair_dim = 4 * hidden_dim
        self.pair_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        pairs = []
        for i in range(max_jets):
            for j in range(i + 1, max_jets):
                pairs.append((i, j))
        self.register_buffer("pairs", torch.tensor(pairs, dtype=torch.long))

    def forward(self, x, mask, global_x):
        bsz, n_jets, _ = x.shape
        h = self.jet_embed(x)

        ranks = torch.arange(n_jets, device=x.device).unsqueeze(0).expand(bsz, n_jets)
        h = h + self.rank_embed(ranks)

        cls = self.cls_token.expand(bsz, 1, self.hidden_dim)
        seq = torch.cat([cls, h], dim=1)

        pad_mask = torch.cat(
            [
                torch.zeros((bsz, 1), dtype=torch.bool, device=x.device),
                ~mask,
            ],
            dim=1,
        )

        enc = self.encoder(seq, src_key_padding_mask=pad_mask)
        cls_out = enc[:, 0]
        jet_out = enc[:, 1:]

        g = self.global_embed(global_x)
        event = torch.cat([cls_out, g], dim=1)

        qcd_logit = self.qcd_head(event).squeeze(-1)
        top_logit = self.top_head(event).squeeze(-1)

        i = self.pairs[:, 0]
        j = self.pairs[:, 1]
        hi = jet_out[:, i, :]
        hj = jet_out[:, j, :]
        pair_rep = torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=-1)
        pair_logits = self.pair_head(pair_rep).permute(0, 2, 1)  # B, 2 Higgs heads, n_pairs

        pair_valid = mask[:, i] & mask[:, j]

        return {
            "qcd_logit": qcd_logit,
            "top_logit": top_logit,
            "pair_logits": pair_logits,
            "pair_valid": pair_valid,
        }


def build_pair_to_index(max_jets=8):
    out = {}
    k = 0
    for i in range(max_jets):
        for j in range(i + 1, max_jets):
            out[(i, j)] = k
            k += 1
    return out


def assignment_loss_and_acc(pair_logits, pair_valid, assignment, assignment_mask, pair_to_index):
    device = pair_logits.device
    losses = []
    correct = 0
    total = 0

    masked_logits = pair_logits.masked_fill(~pair_valid[:, None, :], -1e9)

    for b in range(pair_logits.shape[0]):
        if int(assignment_mask[b].item()) != 1:
            continue

        true_pair_indices = []
        ok = True
        for h in range(2):
            a = int(assignment[b, h, 0].item())
            c = int(assignment[b, h, 1].item())
            if a < 0 or c < 0 or a == c:
                ok = False
                break
            key = tuple(sorted((a, c)))
            if key not in pair_to_index:
                ok = False
                break
            true_pair_indices.append(pair_to_index[key])

        if not ok:
            continue

        t0 = torch.tensor([true_pair_indices[0]], dtype=torch.long, device=device)
        t1 = torch.tensor([true_pair_indices[1]], dtype=torch.long, device=device)

        loss_no_swap = (
            F.cross_entropy(masked_logits[b, 0].unsqueeze(0), t0)
            + F.cross_entropy(masked_logits[b, 1].unsqueeze(0), t1)
        )
        loss_swap = (
            F.cross_entropy(masked_logits[b, 0].unsqueeze(0), t1)
            + F.cross_entropy(masked_logits[b, 1].unsqueeze(0), t0)
        )
        losses.append(torch.minimum(loss_no_swap, loss_swap))

        pred0 = int(torch.argmax(masked_logits[b, 0]).item())
        pred1 = int(torch.argmax(masked_logits[b, 1]).item())
        if {pred0, pred1} == set(true_pair_indices):
            correct += 1
        total += 1

    if not losses:
        return torch.tensor(0.0, device=device), np.nan, 0

    return torch.stack(losses).mean(), correct / max(total, 1), total


def task_masks(sample_id, y):
    qcd_like = torch.zeros_like(y, dtype=torch.bool)
    top_like = torch.zeros_like(y, dtype=torch.bool)

    for sid in QCD_LIKE_SAMPLE_IDS:
        qcd_like |= sample_id == sid
    for sid in TOP_SAMPLE_IDS:
        top_like |= sample_id == sid

    sig = y == 1

    qcd_task = sig | qcd_like
    top_task = sig | top_like

    return qcd_task, top_task


def compute_pos_weights(ds):
    sid = ds.sample_id
    y = ds.y

    sig = y == 1
    qcd_like = np.isin(sid, list(QCD_LIKE_SAMPLE_IDS))
    top_like = np.isin(sid, list(TOP_SAMPLE_IDS))

    qcd_pos = np.sum(sig)
    qcd_neg = np.sum(qcd_like)
    top_pos = np.sum(sig)
    top_neg = np.sum(top_like)

    qcd_pw = qcd_neg / max(qcd_pos, 1)
    top_pw = top_neg / max(top_pos, 1)

    return float(qcd_pw), float(top_pw)


def train_one_epoch(model, loader, optimizer, device, qcd_pos_weight, top_pos_weight, assign_weight, pair_to_index):
    model.train()

    qcd_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(qcd_pos_weight, device=device))
    top_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(top_pos_weight, device=device))

    rows = []

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        global_x = batch["global_x"].to(device)
        y = batch["y"].to(device)
        sid = batch["sample_id"].to(device)
        assignment = batch["assignment"].to(device)
        assignment_mask = batch["assignment_mask"].to(device)

        out = model(x, mask, global_x)
        qcd_task, top_task = task_masks(sid, y)

        loss_parts = []

        qcd_loss = qcd_bce(out["qcd_logit"][qcd_task], y[qcd_task]) if qcd_task.any() else torch.tensor(0.0, device=device)
        top_loss = top_bce(out["top_logit"][top_task], y[top_task]) if top_task.any() else torch.tensor(0.0, device=device)

        assign_loss, assign_acc, assign_n = assignment_loss_and_acc(
            out["pair_logits"],
            out["pair_valid"],
            assignment,
            assignment_mask,
            pair_to_index,
        )

        total_loss = qcd_loss + top_loss + assign_weight * assign_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        rows.append({
            "loss": float(total_loss.detach().cpu()),
            "qcd_loss": float(qcd_loss.detach().cpu()),
            "top_loss": float(top_loss.detach().cpu()),
            "assignment_loss": float(assign_loss.detach().cpu()),
            "assignment_acc": assign_acc,
            "assignment_n": assign_n,
        })

    df = pd.DataFrame(rows)
    out = {c: float(df[c].mean()) for c in ["loss", "qcd_loss", "top_loss", "assignment_loss"]}
    out["assignment_acc_batch_mean"] = float(df["assignment_acc"].dropna().mean()) if df["assignment_acc"].notna().any() else np.nan
    out["assignment_events_seen"] = int(df["assignment_n"].sum())
    return out


@torch.no_grad()
def predict(model, ds, device, batch_size=2048):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    rows = []
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        global_x = batch["global_x"].to(device)

        out = model(x, mask, global_x)

        qcd_score = torch.sigmoid(out["qcd_logit"]).cpu().numpy()
        top_score = torch.sigmoid(out["top_logit"]).cpu().numpy()
        combined_score = np.sqrt(np.clip(qcd_score * top_score, 0.0, 1.0))

        rows.append(pd.DataFrame({
            "qcd_score": qcd_score,
            "top_score": top_score,
            "combined_score": combined_score,
            "y": batch["y"].numpy(),
            "weight_pb": batch["weight_pb"].numpy(),
            "sample_id": batch["sample_id"].numpy(),
            "event_id": batch["event_id"].numpy(),
            "assignment_mask": batch["assignment_mask"].numpy(),
        }))

    return pd.concat(rows, ignore_index=True)


def scaled_weights_for_test(pred, all_ds):
    all_counts = pd.Series(all_ds.sample_id).value_counts().to_dict()
    test_counts = pred["sample_id"].value_counts().to_dict()

    factors = pred["sample_id"].map(
        lambda sid: all_counts.get(int(sid), 0) / max(test_counts.get(int(sid), 1), 1)
    ).astype(float)

    return pred["weight_pb"].astype(float) * factors


def evaluate_scores(pred):
    y = pred["y"].to_numpy().astype(int)
    w = pred["weight_pb_scaled"].to_numpy().astype(float)
    sid = pred["sample_id"].to_numpy().astype(int)

    qcd_mask = (y == 1) | np.isin(sid, list(QCD_LIKE_SAMPLE_IDS))
    top_mask = (y == 1) | np.isin(sid, list(TOP_SAMPLE_IDS))

    return {
        "auc_all_combined_unweighted": safe_auc(y, pred["combined_score"]),
        "auc_all_combined_weighted": safe_auc(y, pred["combined_score"], w),
        "auc_qcd_head_unweighted": safe_auc(y[qcd_mask], pred.loc[qcd_mask, "qcd_score"]),
        "auc_qcd_head_weighted": safe_auc(y[qcd_mask], pred.loc[qcd_mask, "qcd_score"], w[qcd_mask]),
        "auc_top_head_unweighted": safe_auc(y[top_mask], pred.loc[top_mask, "top_score"]),
        "auc_top_head_weighted": safe_auc(y[top_mask], pred.loc[top_mask, "top_score"], w[top_mask]),
    }


def threshold_scan(pred):
    thresholds = np.round(np.arange(0.50, 0.951, 0.025), 3)
    rows = []

    for tq in thresholds:
        for tt in thresholds:
            sel = (pred["qcd_score"] >= tq) & (pred["top_score"] >= tt)
            sig = sel & (pred["y"] == 1)
            bkg = sel & (pred["y"] == 0)

            s_pb = float(pred.loc[sig, "weight_pb_scaled"].sum())
            b_pb = float(pred.loc[bkg, "weight_pb_scaled"].sum())

            s_ev = s_pb * LUMI_PB
            b_ev = b_pb * LUMI_PB

            b_weights = pred.loc[bkg, "weight_pb_scaled"].to_numpy(dtype=float)
            neff_b = (b_weights.sum() ** 2 / np.sum(b_weights ** 2)) if np.sum(b_weights ** 2) > 0 else 0.0

            rows.append({
                "qcd_threshold": tq,
                "top_threshold": tt,
                "signal_events_450fb": s_ev,
                "background_events_450fb": b_ev,
                "S_over_B": s_ev / b_ev if b_ev > 0 else np.nan,
                "S_over_sqrtB": s_ev / np.sqrt(b_ev) if b_ev > 0 else np.nan,
                "S_over_sqrtB_10pct_syst": s_ev / np.sqrt(b_ev + (0.10 * b_ev) ** 2) if b_ev > 0 else np.nan,
                "n_signal_test_rows": int(sig.sum()),
                "n_background_test_rows": int(bkg.sum()),
                "neff_background_rows": neff_b,
            })

    return pd.DataFrame(rows)


def composition_at_best(pred, best_row, sample_names):
    sel = (pred["qcd_score"] >= best_row["qcd_threshold"]) & (pred["top_score"] >= best_row["top_threshold"])
    d = pred[sel].copy()

    rows = []
    for sid, name in enumerate(sample_names):
        m = d["sample_id"] == sid
        rows.append({
            "sample_id": sid,
            "sample": name,
            "selected_test_rows": int(m.sum()),
            "expected_events_450fb_scaled": float((d.loc[m, "weight_pb_scaled"] * LUMI_PB).sum()),
            "group": "signal" if sid in SIGNAL_SAMPLE_IDS else "background",
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qcdplus_btag4_v1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--assign-loss-weight", type=float, default=0.5)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    set_seed(args.seed)

    repo = Path(os.environ["HH4B_REPO"])
    store = Path(os.environ["HH4B_STORE"])
    npz_dir = store / f"spanet_npz/{args.tag}"

    outdir = repo / f"outputs/tables/hh4b_deepspanet_twohead_qcdplus_{args.tag}_2026_07_14"
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_ds = HH4BDeepSPANetDataset(npz_dir / "hh4b_spanet_leading8_train.npz")
    val_ds = HH4BDeepSPANetDataset(
        npz_dir / "hh4b_spanet_leading8_val.npz",
        train_ds.jet_mean,
        train_ds.jet_std,
        train_ds.global_mean,
        train_ds.global_std,
    )
    test_ds = HH4BDeepSPANetDataset(
        npz_dir / "hh4b_spanet_leading8_test.npz",
        train_ds.jet_mean,
        train_ds.jet_std,
        train_ds.global_mean,
        train_ds.global_std,
    )
    all_ds = HH4BDeepSPANetDataset(
        npz_dir / "hh4b_spanet_leading8_all.npz",
        train_ds.jet_mean,
        train_ds.jet_std,
        train_ds.global_mean,
        train_ds.global_std,
    )

    model = DeepTwoHeadSPANet(
        jet_dim=train_ds.x.shape[-1],
        global_dim=train_ds.global_x.shape[-1],
        hidden_dim=args.hidden_dim,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=args.dropout,
        max_jets=train_ds.x.shape[1],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print("parameters:", n_params)

    qcd_pw, top_pw = compute_pos_weights(train_ds)
    print("qcd_pos_weight:", qcd_pw)
    print("top_pos_weight:", top_pw)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    pair_to_index = build_pair_to_index(train_ds.x.shape[1])

    history = []
    best_metric = -np.inf
    best_epoch = -1
    best_path = outdir / "deepspanet_twohead_best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            qcd_pw,
            top_pw,
            args.assign_loss_weight,
            pair_to_index,
        )

        val_pred = predict(model, val_ds, device)
        val_pred["weight_pb_scaled"] = val_pred["weight_pb"]
        val_metrics = evaluate_scores(val_pred)

        metric_values = [
            val_metrics["auc_all_combined_unweighted"],
            val_metrics["auc_qcd_head_unweighted"],
            val_metrics["auc_top_head_unweighted"],
        ]
        metric_values = [v for v in metric_values if not np.isnan(v)]
        val_metric = float(np.mean(metric_values)) if metric_values else -np.inf
        scheduler.step(val_metric)

        row = {
            "epoch": epoch,
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "val_model_selection_metric": val_metric,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(row)

        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "jet_mean": train_ds.jet_mean,
                "jet_std": train_ds.jet_std,
                "global_mean": train_ds.global_mean,
                "global_std": train_ds.global_std,
                "sample_names": train_ds.sample_names,
                "sample_groups": train_ds.sample_groups,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
            }, best_path)

    hist = pd.DataFrame(history)
    hist.to_csv(outdir / "training_history.csv", index=False)
    write_md(hist, outdir / "training_history.md")

    print("Loading best checkpoint:", best_path)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_pred = predict(model, test_ds, device)
    test_pred["weight_pb_scaled"] = scaled_weights_for_test(test_pred, all_ds)
    sample_names = np.asarray(test_ds.sample_names).astype(str)
    test_pred["sample_name"] = [sample_names[int(i)] for i in test_pred["sample_id"]]
    test_pred.to_csv(outdir / "test_predictions.csv", index=False)

    test_metrics = evaluate_scores(test_pred)
    scan = threshold_scan(test_pred)
    scan.to_csv(outdir / "threshold_scan.csv", index=False)

    best = scan.sort_values("S_over_sqrtB", ascending=False).iloc[0].to_dict()
    comp = composition_at_best(test_pred, best, sample_names)
    comp.to_csv(outdir / "best_composition.csv", index=False)
    write_md(comp, outdir / "best_composition.md")

    summary = pd.DataFrame([{**test_metrics, **best, "best_epoch": best_epoch, "best_val_metric": best_metric, "n_parameters": n_params}])
    summary.to_csv(outdir / "summary.csv", index=False)
    write_md(summary, outdir / "summary.md")

    print("\n=== Deep two-head SPA-Net summary ===")
    print(summary.to_string(index=False))
    print("\n=== Best threshold composition ===")
    print(comp.to_string(index=False))
    print("\nWrote:", outdir)


if __name__ == "__main__":
    main()
