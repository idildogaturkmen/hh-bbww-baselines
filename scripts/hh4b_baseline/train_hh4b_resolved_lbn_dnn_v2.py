'''
Train a baseline LBN-inspired DNN for HH->4b resolved analysis.
This script trains a baseline LBN-inspired DNN for the HH->4b resolved analysis.
The model learns soft combinations of the four selected b-jet four-vectors and computes Lorentz
-inspired observables from those combinations. These learned Lorentz features are combined with scalar event variables and passed to a dense classifier.
The script saves the trained model, scaler, training history, metrics, and test scores to the specified output directory.

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


def pt_eta_phi_m_to_e_px_py_pz(pt, eta, phi, mass):
    pt = np.asarray(pt, dtype=float)
    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    mass = np.asarray(mass, dtype=float)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(np.maximum(px * px + py * py + pz * pz + mass * mass, 0.0))

    return np.stack([energy, px, py, pz], axis=-1)


def build_fourvectors(df: pd.DataFrame, fourvec_scale: float) -> np.ndarray:
    jets = []

    required = []

    for i in range(1, 5):
        required += [f"b{i}_pt", f"b{i}_eta", f"b{i}_phi", f"b{i}_mass"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Missing required four-vector columns: "
            + ", ".join(missing)
            + "\nRe-run the HH4b feature builder if phi columns were not saved."
        )

    for i in range(1, 5):
        pt = df[f"b{i}_pt"].astype(float).to_numpy()
        eta = df[f"b{i}_eta"].astype(float).to_numpy()
        phi = df[f"b{i}_phi"].astype(float).to_numpy()
        mass = df[f"b{i}_mass"].astype(float).to_numpy()

        vec = pt_eta_phi_m_to_e_px_py_pz(pt, eta, phi, mass)
        jets.append(vec)

    x = np.stack(jets, axis=1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x / float(fourvec_scale)

    return x.astype(np.float32)


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


def summarize_backgrounds(df: pd.DataFrame, mask, category_name: str):
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


class LorentzCombinationLayer(nn.Module):
    """
    LBN-inspired layer.

    Input: four-vectors with shape (batch, 4 jets, 4 components), components ordered as E, px, py, pz.

    The layer learns soft combinations of the four input b-jet four-vectors.
    From the learned combinations it computes Lorentz-inspired observables:
    mass, pt, eta, log-energy, and pairwise invariant masses.
    """

    def __init__(self, n_input_particles: int = 4, n_combinations: int = 8, eps: float = 1e-8):
        super().__init__()
        self.n_input_particles = n_input_particles
        self.n_combinations = n_combinations
        self.eps = eps

        self.combo_logits = nn.Parameter(0.01 * torch.randn(n_combinations, n_input_particles))

    def vector_features(self, vec):
        e = vec[..., 0]
        px = vec[..., 1]
        py = vec[..., 2]
        pz = vec[..., 3]

        pt2 = px * px + py * py
        pt = torch.sqrt(torch.clamp(pt2, min=self.eps))

        p2 = pt2 + pz * pz
        m2 = e * e - p2
        mass = torch.sqrt(torch.clamp(m2, min=self.eps))

        eta = torch.asinh(pz / torch.clamp(pt, min=self.eps))
        loge = torch.log(torch.clamp(e, min=self.eps))

        return mass, pt, eta, loge

    def invariant_mass(self, vec):
        e = vec[..., 0]
        px = vec[..., 1]
        py = vec[..., 2]
        pz = vec[..., 3]

        m2 = e * e - px * px - py * py - pz * pz
        return torch.sqrt(torch.clamp(m2, min=self.eps))

    def forward(self, x):
        # x: (batch, 4, 4)
        weights = torch.softmax(self.combo_logits, dim=-1)
        combos = torch.einsum("kp,bpc->bkc", weights, x)

        mass, pt, eta, loge = self.vector_features(combos)

        combo_features = torch.cat(
            [
                mass,
                pt,
                eta,
                loge,
            ],
            dim=-1,
        )

        pair_masses = []
        for i in range(self.n_combinations):
            for j in range(i + 1, self.n_combinations):
                pair = combos[:, i, :] + combos[:, j, :]
                pair_masses.append(self.invariant_mass(pair))

        pair_masses = torch.stack(pair_masses, dim=-1)

        return torch.cat([combo_features, pair_masses], dim=-1)


class LBNDNN(nn.Module):
    def __init__(
        self,
        n_aux_features: int,
        n_combinations: int = 8,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.lbn = LorentzCombinationLayer(
            n_input_particles=4,
            n_combinations=n_combinations,
        )

        n_lbn_features = 4 * n_combinations + (n_combinations * (n_combinations - 1)) // 2
        n_total = n_lbn_features + n_aux_features

        self.net = nn.Sequential(
            nn.BatchNorm1d(n_total),

            nn.Linear(n_total, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
        )

    def forward(self, fourvecs, aux):
        lbn_features = self.lbn(fourvecs)
        x = torch.cat([lbn_features, aux], dim=-1)
        return self.net(x).squeeze(-1)


def predict_scores(model, fourvec_np, aux_np, device, batch_size=4096):
    model.eval()
    scores = []

    with torch.no_grad():
        for start in range(0, len(aux_np), batch_size):
            f = torch.tensor(fourvec_np[start:start + batch_size], dtype=torch.float32, device=device)
            a = torch.tensor(aux_np[start:start + batch_size], dtype=torch.float32, device=device)
            logits = model(f, a)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            scores.append(probs)

    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="outputs/hh4b_baseline/full_features/hh4b_resolved_features.parquet",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/hh4b_baseline/lbn_dnn_v2",
    )
    parser.add_argument("--random-state", type=int, default=12345)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--n-combinations", type=int, default=8)
    parser.add_argument("--fourvec-scale", type=float, default=1000.0)
    args = parser.parse_args()

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.features)
    df_trainable = df[np.isfinite(df["physics_weight"])].copy().reset_index(drop=True)

    print("[INFO] Loaded feature table:", args.features)
    print("[INFO] Total rows:", len(df))
    print("[INFO] Trainable finite-weight rows:", len(df_trainable))
    print("[INFO] Excluded missing-weight rows:", len(df) - len(df_trainable))
    print("[INFO] Signal/background in trainable table:")
    print(df_trainable["is_signal"].value_counts().rename(index={0: "background", 1: "signal"}).to_string())

    aux_cols = [
        "n_ak4",
        "n_btag",
        "n_ak8_200",
        "n_ak8_300",
        "ht",
        "vbf_mjj",
        "vbf_deta",
        "b1_btag",
        "b2_btag",
        "b3_btag",
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

    aux_cols = [c for c in aux_cols if c in df_trainable.columns]

    fourvecs = build_fourvectors(df_trainable, fourvec_scale=args.fourvec_scale)

    aux_df = df_trainable[aux_cols].astype(float)
    aux_df = aux_df.replace([np.inf, -np.inf], np.nan).fillna(-999.0)

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

    aux_train = scaler.fit_transform(aux_df.iloc[idx_train].to_numpy(dtype=float))
    aux_val = scaler.transform(aux_df.iloc[idx_val].to_numpy(dtype=float))
    aux_test = scaler.transform(aux_df.iloc[idx_test].to_numpy(dtype=float))

    four_train = fourvecs[idx_train]
    four_val = fourvecs[idx_val]
    four_test = fourvecs[idx_test]

    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    w_train_phys = w_phys[idx_train]
    w_val_phys = w_phys[idx_val]
    w_test_phys = w_phys[idx_test]

    w_train_balanced = make_balanced_training_weights(y_train, w_train_phys)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    train_ds = TensorDataset(
        torch.tensor(four_train, dtype=torch.float32),
        torch.tensor(aux_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(w_train_balanced, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = LBNDNN(
        n_aux_features=len(aux_cols),
        n_combinations=args.n_combinations,
        dropout=args.dropout,
    ).to(device)

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

        for four_b, aux_b, y_b, w_b in train_loader:
            four_b = four_b.to(device)
            aux_b = aux_b.to(device)
            y_b = y_b.to(device)
            w_b = w_b.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(four_b, aux_b)
            per_event_loss = loss_fn(logits, y_b)
            loss = torch.mean(per_event_loss * w_b)

            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))

        val_score = predict_scores(model, four_val, aux_val, device)
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

    val_score = predict_scores(model, four_val, aux_val, device)
    test_score = predict_scores(model, four_test, aux_test, device)

    df_val = df_trainable.iloc[idx_val].copy()
    df_val["lbn_dnn_score"] = val_score
    df_val["dnn_score"] = val_score
    df_val["bdt_score"] = val_score

    df_test = df_trainable.iloc[idx_test].copy()
    df_test["lbn_dnn_score"] = test_score
    df_test["dnn_score"] = test_score
    df_test["bdt_score"] = test_score

    metrics = {
        "n_total_rows_original": int(len(df)),
        "n_total_trainable_finite_weight": int(len(df_trainable)),
        "n_excluded_missing_weight": int(len(df) - len(df_trainable)),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "n_signal_trainable": int(np.sum(y)),
        "n_background_trainable": int(np.sum(1 - y)),
        "aux_cols": aux_cols,
        "n_lbn_combinations": int(args.n_combinations),
        "fourvec_scale": float(args.fourvec_scale),
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

    fixed_thresholds = [0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.74, 0.76, 0.80, 0.85, 0.90]
    for thr in fixed_thresholds:
        category_rows.append(
            summarize_selection(
                df_test,
                df_test["lbn_dnn_score"].to_numpy() >= thr,
                f"test_lbn_dnn_score_ge_{thr:.2f}",
                "lbn_dnn_fixed_cumulative",
                score_low=thr,
                score_high=None,
            )
        )

    for q in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        thr = float(np.quantile(test_score, q))
        category_rows.append(
            summarize_selection(
                df_test,
                df_test["lbn_dnn_score"].to_numpy() >= thr,
                f"test_lbn_dnn_top_{int(round((1-q)*100))}pct_score_ge_{thr:.6f}",
                "lbn_dnn_quantile_cumulative",
                score_low=thr,
                score_high=None,
            )
        )

    cats = pd.DataFrame(category_rows)

    model_rows = cats[cats["category_type"].str.contains("lbn_dnn", na=False)].copy()

    stable_like = model_rows[
        (model_rows["raw_signal"] >= 5)
        & (model_rows["raw_background"] >= 20)
        & (model_rows["N_eff_bkg"] >= 5)
    ].copy()

    if len(stable_like) > 0:
        best_row = stable_like.sort_values(
            ["asimov_Z_A", "S_over_B", "N_eff_bkg"],
            ascending=[False, False, False],
        ).iloc[0]
    else:
        best_row = model_rows.sort_values(
            ["asimov_Z_A", "S_over_B"],
            ascending=[False, False],
        ).iloc[0]

    best_name = str(best_row["selection"])
    best_thr = float(best_row["score_low"])

    best_mask = df_test["lbn_dnn_score"].to_numpy() >= best_thr
    group_bkg, sample_bkg = summarize_backgrounds(df_test, best_mask, best_name)

    metrics["best_nominal_lbn_dnn_selection"] = best_name
    metrics["best_nominal_lbn_dnn_threshold"] = best_thr

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "aux_cols": aux_cols,
            "n_lbn_combinations": args.n_combinations,
            "fourvec_scale": args.fourvec_scale,
            "metrics": metrics,
        },
        outdir / "hh4b_resolved_lbn_dnn_v2_model.pt",
    )

    joblib.dump(scaler, outdir / "hh4b_resolved_lbn_dnn_v2_aux_scaler.joblib")

    pd.DataFrame(history).to_csv(outdir / "hh4b_resolved_lbn_dnn_v2_training_history.csv", index=False)

    with open(outdir / "hh4b_resolved_lbn_dnn_v2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    keep_cols = [
        "event_key",
        "sample",
        "process_group",
        "is_signal",
        "physics_weight",
        "lbn_dnn_score",
        "dnn_score",
        "bdt_score",
        "category_hmass_70_190",
        "category_hmass_90_160",
        "category_vbf_like",
        "category_boosted_1ak8_2b",
    ]

    keep_cols = [c for c in keep_cols if c in df_test.columns]

    df_val[keep_cols].to_csv(outdir / "hh4b_resolved_lbn_dnn_v2_validation_scores.csv", index=False)
    df_test[keep_cols].to_csv(outdir / "hh4b_resolved_lbn_dnn_v2_test_scores.csv", index=False)

    cats.to_csv(outdir / "hh4b_resolved_lbn_dnn_v2_test_categories.csv", index=False)
    group_bkg.to_csv(outdir / "hh4b_resolved_lbn_dnn_v2_best_region_backgrounds_by_group.csv", index=False)
    sample_bkg.to_csv(outdir / "hh4b_resolved_lbn_dnn_v2_best_region_backgrounds_by_sample.csv", index=False)

    note_lines = []
    note_lines.append("# HH4b resolved LBN-inspired DNN v2\n\n")
    note_lines.append("This model is an LBN-inspired four-vector neural-network baseline.\n\n")
    note_lines.append("It learns soft combinations of the four selected b-jet four-vectors and computes Lorentz-inspired observables from those combinations. These learned Lorentz features are combined with scalar event variables and passed to a dense classifier.\n\n")
    note_lines.append("The score file includes `bdt_score` as an alias for `lbn_dnn_score` so that the existing BDT stability/resampling script can be reused.\n\n")
    note_lines.append("## Metrics\n\n")
    note_lines.append("```json\n")
    note_lines.append(json.dumps(metrics, indent=2))
    note_lines.append("\n```\n\n")
    note_lines.append("## Initial test categories\n\n")
    note_lines.append(cats.to_markdown(index=False))
    note_lines.append("\n")

    (outdir / "hh4b_resolved_lbn_dnn_v2_interpretation.md").write_text(
        "".join(note_lines),
        encoding="utf-8",
    )

    print("\n[INFO] Metrics:")
    print(json.dumps(metrics, indent=2))

    print("\n[INFO] Test categories:")
    print(cats.to_string(index=False))

    print("\n[INFO] Best nominal LBN-DNN category:", best_name)
    print("\n[INFO] Backgrounds by process group in best nominal LBN-DNN category:")
    print(group_bkg.head(40).to_string(index=False))

    print("\n[INFO] Wrote outputs to:", outdir)


if __name__ == "__main__":
    main()
