#!/usr/bin/env python3

"""
LBN-DNN v3 qcdplus baseline.

This trains lightweight LBN-style models on the same HH→4b candidate set used
for the BDT-v3 and DNN-v3 qcdplus baselines.

Models:
- lbn_p4_only: four candidate jet four-vectors only
- lbn_p4_plus_topology: four-vectors plus topology-only auxiliary features
- lbn_p4_plus_topology_btag: four-vectors plus topology-only features plus four candidate b-tag scores

Tasks:
- signal vs QCD bbbb HT slices
- signal vs ttbar/top-like background
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])

DATA = STORE / "lbn_npz/v3_qcdplus/lbn_v3_qcdplus_candidates.npz"

OUTDIR = REPO / "outputs/tables/lbn_dnn_v3_qcdplus_2026_07_13"
OUTDIR.mkdir(parents=True, exist_ok=True)

LUMI_PB = 450_000.0
RANDOM_STATE = 12345

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def neff(weights):
    w = np.asarray(weights, dtype=float)
    if len(w) == 0 or np.sum(w * w) <= 0:
        return 0.0
    return float((np.sum(w) ** 2) / np.sum(w * w))


def class_balanced_weights(y):
    y = np.asarray(y).astype(int)
    n_pos = max((y == 1).sum(), 1)
    n_neg = max((y == 0).sum(), 1)
    return np.where(y == 1, 0.5 / n_pos, 0.5 / n_neg).astype("float32")


class LorentzCombinationLayer(nn.Module):
    def __init__(self, n_input=4, n_combos=8):
        super().__init__()

        init = torch.full((n_combos, n_input), -5.0)

        # First four combinations start as individual jets.
        for i in range(min(n_input, n_combos)):
            init[i, i] = 0.54

        # Next combinations start as simple dijet/symmetric combinations.
        pairs = [(0, 1), (2, 3), (0, 2), (1, 3)]
        for k, pair in enumerate(pairs, start=n_input):
            if k < n_combos:
                for j in pair:
                    init[k, j] = 0.54

        self.raw_weights = nn.Parameter(init)

    def forward(self, p4):
        # p4 shape: (batch, 4 jets, 4 components [E, px, py, pz])
        weights = torch.nn.functional.softplus(self.raw_weights)
        combos = torch.einsum("ki,bic->bkc", weights, p4)

        e = combos[..., 0]
        px = combos[..., 1]
        py = combos[..., 2]
        pz = combos[..., 3]

        pt = torch.sqrt(torch.clamp(px * px + py * py, min=1e-8))
        p = torch.sqrt(torch.clamp(px * px + py * py + pz * pz, min=1e-8))
        m2 = torch.clamp(e * e - p * p, min=0.0)
        mass = torch.sqrt(m2 + 1e-8)

        eta_arg_num = torch.clamp(p + pz, min=1e-6)
        eta_arg_den = torch.clamp(p - pz, min=1e-6)
        eta = 0.5 * torch.log(eta_arg_num / eta_arg_den)
        eta = torch.clamp(eta, min=-8.0, max=8.0)

        features = torch.stack([e, px, py, pz, pt, mass, eta], dim=-1)
        return features.flatten(start_dim=1)


class LBNDNN(nn.Module):
    def __init__(self, aux_dim=0, n_combos=8):
        super().__init__()

        self.lbn = LorentzCombinationLayer(n_input=4, n_combos=n_combos)
        lbn_dim = n_combos * 7
        in_dim = lbn_dim + aux_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, p4, aux=None):
        feats = self.lbn(p4)
        if aux is not None:
            feats = torch.cat([feats, aux], dim=1)
        return self.net(feats).squeeze(-1)


def train_one_classifier(arrays, mode, task):
    X_p4 = arrays["X_p4"].astype("float32") / 100.0
    y = arrays["y"].astype("int64")

    train_mask = arrays["train_mask"].astype(bool)
    test_mask = arrays["test_mask"].astype(bool)

    is_signal = arrays["is_signal"].astype(bool)
    is_qcd = arrays["is_qcd"].astype(bool)
    is_top = arrays["is_top"].astype(bool)

    if task == "qcd":
        task_mask = is_signal | is_qcd
    elif task == "top":
        task_mask = is_signal | is_top
    else:
        raise ValueError(task)

    train_task = train_mask & task_mask
    test_task = test_mask & task_mask

    if mode == "lbn_p4_only":
        aux_train = None
        aux_test_task = None
        aux_test_all = None
        aux_dim = 0
        aux_scaler = None
    elif mode == "lbn_p4_plus_topology":
        aux_all = arrays["X_aux_topology"].astype("float32")
        aux_scaler = StandardScaler()
        aux_train = aux_scaler.fit_transform(aux_all[train_task]).astype("float32")
        aux_test_task = aux_scaler.transform(aux_all[test_task]).astype("float32")
        aux_test_all = aux_scaler.transform(aux_all[test_mask]).astype("float32")
        aux_dim = aux_train.shape[1]
    elif mode == "lbn_p4_plus_topology_btag":
        aux_all = np.concatenate(
            [
                arrays["X_aux_topology"].astype("float32"),
                arrays["X_btag"].astype("float32"),
            ],
            axis=1,
        )
        aux_scaler = StandardScaler()
        aux_train = aux_scaler.fit_transform(aux_all[train_task]).astype("float32")
        aux_test_task = aux_scaler.transform(aux_all[test_task]).astype("float32")
        aux_test_all = aux_scaler.transform(aux_all[test_mask]).astype("float32")
        aux_dim = aux_train.shape[1]
    elif mode == "lbn_p4_plus_massaware_btag":
        aux_all = np.concatenate(
            [
                arrays["X_aux_mass_aware"].astype("float32"),
                arrays["X_btag"].astype("float32"),
            ],
            axis=1,
        )
        aux_scaler = StandardScaler()
        aux_train = aux_scaler.fit_transform(aux_all[train_task]).astype("float32")
        aux_test_task = aux_scaler.transform(aux_all[test_task]).astype("float32")
        aux_test_all = aux_scaler.transform(aux_all[test_mask]).astype("float32")
        aux_dim = aux_train.shape[1]
    else:
        raise ValueError(mode)

    y_train = y[train_task].astype("float32")
    y_test = y[test_task].astype("int64")

    w_train = class_balanced_weights(y_train)

    p4_train = X_p4[train_task]
    p4_test_task = X_p4[test_task]
    p4_test_all = X_p4[test_mask]

    if aux_train is None:
        train_ds = TensorDataset(
            torch.tensor(p4_train, dtype=torch.float32),
            torch.zeros((len(p4_train), 0), dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(w_train, dtype=torch.float32),
        )
    else:
        train_ds = TensorDataset(
            torch.tensor(p4_train, dtype=torch.float32),
            torch.tensor(aux_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(w_train, dtype=torch.float32),
        )

    loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    model = LBNDNN(aux_dim=aux_dim, n_combos=8).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    model.train()
    for epoch in range(140):
        for p4b, auxb, yb, wb in loader:
            p4b = p4b.to(DEVICE)
            auxb = auxb.to(DEVICE)
            yb = yb.to(DEVICE)
            wb = wb.to(DEVICE)

            opt.zero_grad()
            logits = model(p4b, auxb if aux_dim > 0 else None)
            loss = (loss_fn(logits, yb) * wb).sum()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        p4_task_t = torch.tensor(p4_test_task, dtype=torch.float32).to(DEVICE)
        p4_all_t = torch.tensor(p4_test_all, dtype=torch.float32).to(DEVICE)

        if aux_dim > 0:
            aux_task_t = torch.tensor(aux_test_task, dtype=torch.float32).to(DEVICE)
            aux_all_t = torch.tensor(aux_test_all, dtype=torch.float32).to(DEVICE)
        else:
            aux_task_t = None
            aux_all_t = None

        task_scores = torch.sigmoid(model(p4_task_t, aux_task_t)).cpu().numpy()
        all_scores = torch.sigmoid(model(p4_all_t, aux_all_t)).cpu().numpy()

    test_weights = arrays["weight_pb_scaled_to_full"][test_task]
    auc = roc_auc_score(y_test, task_scores)
    wauc = roc_auc_score(y_test, task_scores, sample_weight=test_weights)

    model_dir = OUTDIR / "models"
    model_dir.mkdir(exist_ok=True)

    save = {
        "model_state_dict": model.state_dict(),
        "mode": mode,
        "task": task,
        "aux_dim": aux_dim,
        "device": str(DEVICE),
    }
    if aux_scaler is not None:
        save["aux_scaler_mean"] = aux_scaler.mean_
        save["aux_scaler_scale"] = aux_scaler.scale_

    torch.save(save, model_dir / f"{mode}_{task}.pt")

    return all_scores, auc, wauc, int(train_task.sum()), int(test_task.sum())


def scan_rectangles(df, qkey, tkey):
    thresholds_qcd = np.round(np.arange(0.50, 0.951, 0.025), 3)
    thresholds_top = np.round(np.arange(0.50, 0.951, 0.025), 3)

    rows = []
    for tq in thresholds_qcd:
        for tt in thresholds_top:
            sel = (df[qkey] >= tq) & (df[tkey] >= tt)
            sig = df[sel & df["is_signal"]]
            bkg = df[sel & ~df["is_signal"]]

            s = sig["weight_events_450fb_scaled_to_full"].sum()
            b = bkg["weight_events_450fb_scaled_to_full"].sum()

            rows.append({
                "qcd_threshold": tq,
                "top_threshold": tt,
                "signal_events_450fb": s,
                "background_events_450fb": b,
                "S_over_B": s / b if b > 0 else np.nan,
                "S_over_sqrtB": s / np.sqrt(b) if b > 0 else np.nan,
                "S_over_10pctB": s / (0.10 * b) if b > 0 else np.nan,
                "n_signal_test_rows": len(sig),
                "n_background_test_rows": len(bkg),
                "neff_signal": neff(sig["weight_events_450fb_scaled_to_full"]),
                "neff_background": neff(bkg["weight_events_450fb_scaled_to_full"]),
            })

    scan = pd.DataFrame(rows)
    stable = scan[
        (scan["n_background_test_rows"] >= 20)
        & (scan["neff_background"] >= 5)
        & (scan["n_signal_test_rows"] >= 20)
    ].copy()

    return scan, stable.sort_values("S_over_sqrtB", ascending=False).head(20)


def assign_category(df, qkey, tkey):
    q = df[qkey]
    t = df[tkey]

    cat = np.full(len(df), "UNSELECTED", dtype=object)

    cat[(q >= 0.850) & (t >= 0.850)] = "CAT0_high_purity_diagnostic"
    cat[(q >= 0.850) & (t >= 0.500) & (t < 0.850)] = "CAT1_tight"
    cat[(q >= 0.800) & (q < 0.850) & (t >= 0.500)] = "CAT2_medium_tight"
    cat[(q >= 0.700) & (q < 0.800) & (t >= 0.500)] = "CAT3_medium"
    cat[(q >= 0.500) & (q < 0.700) & (t >= 0.500)] = "CAT4_loose"

    return cat


def make_category_tables(df, mode):
    cats = [
        "CAT0_high_purity_diagnostic",
        "CAT1_tight",
        "CAT2_medium_tight",
        "CAT3_medium",
        "CAT4_loose",
    ]

    rows = []
    for cat in cats:
        sub = df[df["category"] == cat]
        sig = sub[sub["is_signal"]]
        bkg = sub[~sub["is_signal"]]

        s = sig["weight_events_450fb_scaled_to_full"].sum()
        b = bkg["weight_events_450fb_scaled_to_full"].sum()

        rows.append({
            "mode": mode,
            "category": cat,
            "n_signal_test_rows": len(sig),
            "n_background_test_rows": len(bkg),
            "signal_events_450fb": s,
            "background_events_450fb": b,
            "S_over_B": s / b if b > 0 else np.nan,
            "S_over_sqrtB": s / np.sqrt(b) if b > 0 else np.nan,
            "S_over_10pctB": s / (0.10 * b) if b > 0 else np.nan,
            "neff_signal": neff(sig["weight_events_450fb_scaled_to_full"]),
            "neff_background": neff(bkg["weight_events_450fb_scaled_to_full"]),
        })

    yields = pd.DataFrame(rows)

    comp = (
        df[df["category"].isin(cats) & ~df["is_signal"]]
        .groupby(["category", "analysis_sample"], as_index=False)
        .agg(
            n_test_rows=("is_signal", "size"),
            background_events_450fb=("weight_events_450fb_scaled_to_full", "sum"),
            neff_sample=("weight_events_450fb_scaled_to_full", neff),
            mean_qcd_score=("lbn_qcd_score", "mean"),
            mean_top_score=("lbn_top_score", "mean"),
        )
    )

    comp["mode"] = mode
    total_bkg = comp.groupby("category")["background_events_450fb"].transform("sum")
    comp["fraction_of_category_background"] = comp["background_events_450fb"] / total_bkg
    comp = comp.sort_values(["category", "background_events_450fb"], ascending=[True, False])

    return yields, comp


def main():
    arrays = dict(np.load(DATA, allow_pickle=True))
    print("Using device:", DEVICE)
    print("Loaded:", DATA)
    print("X_p4:", arrays["X_p4"].shape)

    test_mask = arrays["test_mask"].astype(bool)
    test_indices = np.where(test_mask)[0]

    base_test = pd.DataFrame({
        "event_index": test_indices,
        "analysis_sample": arrays["analysis_sample"][test_mask].astype(str),
        "group": arrays["group"][test_mask].astype(str),
        "is_signal": arrays["is_signal"][test_mask].astype(bool),
        "is_qcd": arrays["is_qcd"][test_mask].astype(bool),
        "is_top": arrays["is_top"][test_mask].astype(bool),
        "weight_pb_scaled_to_full": arrays["weight_pb_scaled_to_full"][test_mask],
        "weight_events_450fb_scaled_to_full": arrays["weight_events_450fb_scaled_to_full"][test_mask],
    })

    all_auc = []
    all_scan = []
    all_best = []
    all_yields = []
    all_comp = []

    for mode in ["lbn_p4_only", "lbn_p4_plus_topology", "lbn_p4_plus_topology_btag", "lbn_p4_plus_massaware_btag"]:
        qcd_scores, qcd_auc, qcd_wauc, qcd_ntrain, qcd_ntest = train_one_classifier(arrays, mode, "qcd")
        top_scores, top_auc, top_wauc, top_ntrain, top_ntest = train_one_classifier(arrays, mode, "top")

        df = base_test.copy()
        df["lbn_qcd_score"] = qcd_scores
        df["lbn_top_score"] = top_scores
        df["category"] = assign_category(df, "lbn_qcd_score", "lbn_top_score")

        scored_path = OUTDIR / f"scored_test_events_{mode}.parquet"
        df.to_parquet(scored_path, index=False)

        auc = pd.DataFrame([
            {
                "mode": mode,
                "classifier": "LBN_QCD",
                "negative_class": "QCD bbbb HT slices",
                "unweighted_auc": qcd_auc,
                "physics_weighted_auc": qcd_wauc,
                "n_train_task": qcd_ntrain,
                "n_test_task": qcd_ntest,
            },
            {
                "mode": mode,
                "classifier": "LBN_top",
                "negative_class": "ttbar / top-like backgrounds",
                "unweighted_auc": top_auc,
                "physics_weighted_auc": top_wauc,
                "n_train_task": top_ntrain,
                "n_test_task": top_ntest,
            },
        ])

        scan, best = scan_rectangles(df, "lbn_qcd_score", "lbn_top_score")
        scan["mode"] = mode
        best["mode"] = mode

        yields, comp = make_category_tables(df, mode)

        all_auc.append(auc)
        all_scan.append(scan)
        all_best.append(best)
        all_yields.append(yields)
        all_comp.append(comp)

        print(f"\n=== {mode} AUC ===")
        print(auc.to_string(index=False))

        print(f"\n=== {mode} best stable rectangles ===")
        print(best.head(10).to_string(index=False))

        print(f"\n=== {mode} category yields ===")
        print(yields.to_string(index=False))

    outputs = {
        "lbn_auc_summary": pd.concat(all_auc, ignore_index=True),
        "lbn_rectangle_scan": pd.concat(all_scan, ignore_index=True),
        "lbn_best_stable_rectangles": pd.concat(all_best, ignore_index=True),
        "lbn_category_yields": pd.concat(all_yields, ignore_index=True),
        "lbn_category_background_composition": pd.concat(all_comp, ignore_index=True),
    }

    for name, df in outputs.items():
        df.to_csv(OUTDIR / f"{name}.csv", index=False)
        (OUTDIR / f"{name}.md").write_text(df.to_markdown(index=False) + "\n")

    (OUTDIR / "README.md").write_text(
        "# LBN-DNN v3 qcdplus baseline\n\n"
        "This directory contains lightweight LBN-style DNN results for the HH→4b Delphes analysis.\n\n"
        "Modes:\n"
        "- `lbn_p4_only`: four candidate jet four-vectors only.\n"
        "- `lbn_p4_plus_topology`: four-vectors plus topology-only auxiliary features.\n- `lbn_p4_plus_topology_btag`: four-vectors plus topology-only features plus candidate b-tag scores.\n- `lbn_p4_plus_massaware_btag`: four-vectors plus mass-aware scalar features plus candidate b-tag scores. This is an upper-bound, mass-aware LBN mode.\n\n"
        "The LBN layer learns non-negative combinations of the four candidate jet four-vectors and computes Lorentz features before a dense classifier.\n\n"
        "This is a physics-structured neural-network baseline between the plain DNN and SPA-Net.\n"
    )

    print("\nWrote:", OUTDIR)


if __name__ == "__main__":
    main()
