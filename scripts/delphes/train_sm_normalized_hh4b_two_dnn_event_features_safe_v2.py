#!/usr/bin/env python3

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])

BASE_SCRIPT = REPO / "scripts/delphes/train_sm_normalized_hh4b_two_bdt_event_features_safe_v2.py"
OUTDIR = REPO / "outputs/tables/sm_normalized_hh4b_two_dnn_event_features_safe_v2_2026_07_10"
OUTDIR.mkdir(parents=True, exist_ok=True)

LUMI_PB = 450_000.0

spec = importlib.util.spec_from_file_location("bdt_v2_loader", BASE_SCRIPT)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def relu(x):
    return np.maximum(x, 0.0)


def class_balanced_weights(y):
    y = np.asarray(y).astype(int)
    n0 = max((y == 0).sum(), 1)
    n1 = max((y == 1).sum(), 1)
    w = np.ones(len(y), dtype=float)
    w[y == 0] = len(y) / (2.0 * n0)
    w[y == 1] = len(y) / (2.0 * n1)
    return w


def neff(weights):
    w = np.asarray(weights, dtype=float)
    denom = np.sum(w * w)
    return float(w.sum() * w.sum() / denom) if denom > 0 else 0.0


class TinyMLP:
    def __init__(self, n_in, hidden=(64, 32), seed=1, lr=1e-3, l2=1e-4):
        rng = np.random.default_rng(seed)
        h1, h2 = hidden

        self.lr = lr
        self.l2 = l2

        self.W1 = rng.normal(0, np.sqrt(2 / n_in), size=(n_in, h1))
        self.b1 = np.zeros(h1)
        self.W2 = rng.normal(0, np.sqrt(2 / h1), size=(h1, h2))
        self.b2 = np.zeros(h2)
        self.W3 = rng.normal(0, np.sqrt(2 / h2), size=(h2, 1))
        self.b3 = np.zeros(1)

        self.m = {k: np.zeros_like(getattr(self, k)) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
        self.t = 0

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3
        p = sigmoid(z3).ravel()
        return z1, a1, z2, a2, z3, p

    def fit(self, X, y, sample_weight, epochs=180, batch_size=512, seed=1):
        rng = np.random.default_rng(seed)
        y = y.astype(float)
        w = sample_weight.astype(float)
        w = w / np.mean(w)

        n = len(y)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(epochs):
            idx = rng.permutation(n)

            for start in range(0, n, batch_size):
                ii = idx[start:start + batch_size]
                Xb = X[ii]
                yb = y[ii]
                wb = w[ii]

                z1, a1, z2, a2, z3, p = self.forward(Xb)

                denom = np.sum(wb) + 1e-12
                dz3 = ((p - yb) * wb / denom).reshape(-1, 1)

                dW3 = a2.T @ dz3 + self.l2 * self.W3
                db3 = dz3.sum(axis=0)

                da2 = dz3 @ self.W3.T
                dz2 = da2 * (z2 > 0)
                dW2 = a1.T @ dz2 + self.l2 * self.W2
                db2 = dz2.sum(axis=0)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * (z1 > 0)
                dW1 = Xb.T @ dz1 + self.l2 * self.W1
                db1 = dz1.sum(axis=0)

                grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

                self.t += 1
                for k, g in grads.items():
                    self.m[k] = beta1 * self.m[k] + (1 - beta1) * g
                    self.v[k] = beta2 * self.v[k] + (1 - beta2) * (g * g)
                    mh = self.m[k] / (1 - beta1 ** self.t)
                    vh = self.v[k] / (1 - beta2 ** self.t)
                    setattr(self, k, getattr(self, k) - self.lr * mh / (np.sqrt(vh) + eps))

    def predict_proba(self, X):
        return self.forward(X)[-1]


def train_task(train_df, test_df, feature_cols, mask_col, seed):
    task_train = train_df[train_df["is_signal"] | train_df[mask_col]].copy()
    task_test = test_df[test_df["is_signal"] | test_df[mask_col]].copy()

    y_train = task_train["target"].to_numpy().astype(int)
    y_test = task_test["target"].to_numpy().astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(task_train[feature_cols].to_numpy(dtype=float))
    X_test_task = scaler.transform(task_test[feature_cols].to_numpy(dtype=float))
    X_test_all = scaler.transform(test_df[feature_cols].to_numpy(dtype=float))

    model = TinyMLP(n_in=X_train.shape[1], seed=seed)
    model.fit(
        X_train,
        y_train,
        sample_weight=class_balanced_weights(y_train),
        epochs=180,
        batch_size=512,
        seed=seed,
    )

    task_scores = model.predict_proba(X_test_task)
    all_scores = model.predict_proba(X_test_all)

    auc = roc_auc_score(y_test, task_scores)
    wauc = roc_auc_score(
        y_test,
        task_scores,
        sample_weight=task_test["weight_pb_scaled_to_full"],
    )

    return all_scores, auc, wauc, len(task_train), len(task_test)


def run_one_seed(df, feature_cols, seed):
    train_df, test_df = train_test_split(
        df,
        test_size=0.35,
        random_state=seed,
        stratify=df["target"],
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    total_counts = df.groupby("analysis_sample").size()
    test_counts = test_df.groupby("analysis_sample").size()
    scale_to_full = (total_counts / test_counts).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    test_df["weight_pb_scaled_to_full"] = test_df["weight_pb"] * test_df["analysis_sample"].map(scale_to_full)

    qcd_scores, qcd_auc, qcd_wauc, qcd_ntrain, qcd_ntest = train_task(
        train_df, test_df, feature_cols, "is_qcd", seed + 11
    )
    top_scores, top_auc, top_wauc, top_ntrain, top_ntest = train_task(
        train_df, test_df, feature_cols, "is_top", seed + 22
    )

    test_df["dnn_qcd_score"] = qcd_scores
    test_df["dnn_top_score"] = top_scores

    auc_row = {
        "seed": seed,
        "qcd_auc": qcd_auc,
        "qcd_weighted_auc": qcd_wauc,
        "top_auc": top_auc,
        "top_weighted_auc": top_wauc,
        "qcd_ntrain": qcd_ntrain,
        "qcd_ntest": qcd_ntest,
        "top_ntrain": top_ntrain,
        "top_ntest": top_ntest,
    }

    return test_df, auc_row


def scan_working_points(test_df, seed):
    thresholds_qcd = np.round(np.arange(0.50, 0.951, 0.025), 3)
    thresholds_top = np.round(np.arange(0.50, 0.951, 0.025), 3)

    rows = []
    for tq in thresholds_qcd:
        for tt in thresholds_top:
            sel = (test_df["dnn_qcd_score"] >= tq) & (test_df["dnn_top_score"] >= tt)
            selected = test_df[sel]
            sig = selected[selected["is_signal"]]
            bkg = selected[~selected["is_signal"]]

            s_pb = sig["weight_pb_scaled_to_full"].sum()
            b_pb = bkg["weight_pb_scaled_to_full"].sum()
            s_ev = s_pb * LUMI_PB
            b_ev = b_pb * LUMI_PB

            rows.append({
                "seed": seed,
                "qcd_threshold": tq,
                "top_threshold": tt,
                "signal_events_450fb": s_ev,
                "background_events_450fb": b_ev,
                "S_over_B": s_ev / b_ev if b_ev > 0 else np.nan,
                "S_over_sqrtB": s_ev / np.sqrt(b_ev) if b_ev > 0 else np.nan,
                "S_over_10pctB": s_ev / (0.1 * b_ev) if b_ev > 0 else np.nan,
                "n_signal_test_rows": len(sig),
                "n_background_test_rows": len(bkg),
                "neff_signal": neff(sig["weight_pb_scaled_to_full"]),
                "neff_background": neff(bkg["weight_pb_scaled_to_full"]),
            })

    return pd.DataFrame(rows)


def main():
    df, feature_cols = loader.load_all()

    seeds = list(range(5))

    auc_rows = []
    all_scan = []

    for seed in seeds:
        print(f"\n=== DNN seed {seed} ===")
        test_df, auc_row = run_one_seed(df, feature_cols, seed)
        auc_rows.append(auc_row)

        scan = scan_working_points(test_df, seed)
        all_scan.append(scan)

        stable = scan[
            (scan["n_background_test_rows"] >= 50)
            & (scan["neff_background"] >= 20)
            & (scan["n_signal_test_rows"] >= 30)
        ].sort_values("S_over_sqrtB", ascending=False).head(5)

        print(stable.to_string(index=False))

    auc = pd.DataFrame(auc_rows)
    scan = pd.concat(all_scan, ignore_index=True)

    stable_scan = scan[
        (scan["n_background_test_rows"] >= 50)
        & (scan["neff_background"] >= 20)
        & (scan["n_signal_test_rows"] >= 30)
    ].copy()

    best_per_seed = (
        stable_scan
        .sort_values(["seed", "S_over_sqrtB"], ascending=[True, False])
        .groupby("seed", as_index=False)
        .head(1)
    )

    summary = pd.DataFrame({
        "metric": [
            "qcd_weighted_auc_mean",
            "qcd_weighted_auc_std",
            "top_weighted_auc_mean",
            "top_weighted_auc_std",
            "best_per_seed_S_over_sqrtB_mean",
            "best_per_seed_S_over_sqrtB_std",
            "best_per_seed_S_over_B_mean",
        ],
        "value": [
            auc["qcd_weighted_auc"].mean(),
            auc["qcd_weighted_auc"].std(),
            auc["top_weighted_auc"].mean(),
            auc["top_weighted_auc"].std(),
            best_per_seed["S_over_sqrtB"].mean(),
            best_per_seed["S_over_sqrtB"].std(),
            best_per_seed["S_over_B"].mean(),
        ],
    })

    auc.to_csv(OUTDIR / "dnn_v2_auc_by_seed.csv", index=False)
    scan.to_csv(OUTDIR / "dnn_v2_threshold_scan.csv", index=False)
    best_per_seed.to_csv(OUTDIR / "dnn_v2_best_stable_per_seed.csv", index=False)
    summary.to_csv(OUTDIR / "dnn_v2_summary.csv", index=False)

    (OUTDIR / "dnn_v2_auc_by_seed.md").write_text(auc.to_markdown(index=False) + "\n")
    (OUTDIR / "dnn_v2_best_stable_per_seed.md").write_text(best_per_seed.to_markdown(index=False) + "\n")
    (OUTDIR / "dnn_v2_summary.md").write_text(summary.to_markdown(index=False) + "\n")

    readme = f"""# DNN-v2 HH4b baseline

Small NumPy MLP baseline using the same v2 reconstructed features as the safe BDT-v2.

Seeds: {seeds}

No truth-flavor or identifier columns are used as input features.
Training uses class-balanced weights; evaluation uses physics cross-section weights.
"""
    (OUTDIR / "README.md").write_text(readme)

    print("\n=== DNN-v2 AUC by seed ===")
    print(auc.to_string(index=False))

    print("\n=== DNN-v2 summary ===")
    print(summary.to_string(index=False))

    print("\n=== DNN-v2 best stable per seed ===")
    print(best_per_seed.to_string(index=False))

    print("\nWrote:", OUTDIR)


if __name__ == "__main__":
    main()
