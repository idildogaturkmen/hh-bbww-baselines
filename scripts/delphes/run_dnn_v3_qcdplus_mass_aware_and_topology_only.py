#!/usr/bin/env python3

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO = Path(os.environ["HH4B_REPO"])

RUNS = {
    "mass_aware": REPO / "scripts/delphes/train_sm_normalized_hh4b_two_bdt_event_features_safe_v3_qcdplus.py",
    "topology_only": REPO / "scripts/delphes/train_sm_normalized_hh4b_two_bdt_event_features_safe_v3_qcdplus_topology_only.py",
}

OUTDIR = REPO / "outputs/tables/dnn_v3_qcdplus_mass_aware_and_topology_only_2026_07_13"
OUTDIR.mkdir(parents=True, exist_ok=True)

LUMI_PB = 450_000.0
RANDOM_STATE = 12345
TEST_SIZE = 0.35

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def import_wrapper(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def class_balanced_weights(y):
    y = np.asarray(y).astype(int)
    n_pos = max((y == 1).sum(), 1)
    n_neg = max((y == 0).sum(), 1)
    return np.where(y == 1, 0.5 / n_pos, 0.5 / n_neg).astype("float32")


def neff(weights):
    w = np.asarray(weights, dtype=float)
    if len(w) == 0 or np.sum(w * w) <= 0:
        return 0.0
    return float((np.sum(w) ** 2) / np.sum(w * w))


class SmallDNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_dnn_classifier(train_df, test_df, feature_cols, mask_col, label, run_name):
    task_train = train_df[train_df["is_signal"] | train_df[mask_col]].copy()
    task_test = test_df[test_df["is_signal"] | test_df[mask_col]].copy()

    y_train = task_train["is_signal"].astype(int).values
    y_test = task_test["is_signal"].astype(int).values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(task_train[feature_cols].values).astype("float32")
    x_test_task = scaler.transform(task_test[feature_cols].values).astype("float32")
    x_test_all = scaler.transform(test_df[feature_cols].values).astype("float32")

    weights = class_balanced_weights(y_train)

    ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=512, shuffle=True)

    model = SmallDNN(len(feature_cols))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    model.train()
    for epoch in range(120):
        for xb, yb, wb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = (loss_fn(logits, yb) * wb).sum()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        task_scores = torch.sigmoid(model(torch.tensor(x_test_task, dtype=torch.float32))).numpy()
        all_scores = torch.sigmoid(model(torch.tensor(x_test_all, dtype=torch.float32))).numpy()

    auc = roc_auc_score(y_test, task_scores)
    wauc = roc_auc_score(
        y_test,
        task_scores,
        sample_weight=task_test["weight_pb_scaled_to_full"],
    )

    model_dir = OUTDIR / "models"
    model_dir.mkdir(exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "run_name": run_name,
            "label": label,
        },
        model_dir / f"dnn_{run_name}_{label}.pt",
    )

    return all_scores, auc, wauc, len(task_train), len(task_test)


def assign_category(df):
    q = df["dnn_qcd_score"]
    t = df["dnn_top_score"]

    cat = np.full(len(df), "UNSELECTED", dtype=object)

    cat[(q >= 0.850) & (t >= 0.850)] = "CAT0_high_purity_diagnostic"
    cat[(q >= 0.850) & (t >= 0.500) & (t < 0.850)] = "CAT1_tight"
    cat[(q >= 0.800) & (q < 0.850) & (t >= 0.500)] = "CAT2_medium_tight"
    cat[(q >= 0.700) & (q < 0.800) & (t >= 0.500)] = "CAT3_medium"
    cat[(q >= 0.500) & (q < 0.700) & (t >= 0.500)] = "CAT4_loose"

    return cat


def scan_rectangles(test_df):
    thresholds_qcd = np.round(np.arange(0.50, 0.951, 0.025), 3)
    thresholds_top = np.round(np.arange(0.50, 0.951, 0.025), 3)

    rows = []
    for tq in thresholds_qcd:
        for tt in thresholds_top:
            sel = (test_df["dnn_qcd_score"] >= tq) & (test_df["dnn_top_score"] >= tt)

            sig = test_df[sel & test_df["is_signal"]]
            bkg = test_df[sel & ~test_df["is_signal"]]

            s_ev = sig["weight_events_450fb"].sum()
            b_ev = bkg["weight_events_450fb"].sum()

            rows.append({
                "qcd_threshold": tq,
                "top_threshold": tt,
                "signal_events_450fb": s_ev,
                "background_events_450fb": b_ev,
                "S_over_B": s_ev / b_ev if b_ev > 0 else np.nan,
                "S_over_sqrtB": s_ev / np.sqrt(b_ev) if b_ev > 0 else np.nan,
                "S_over_10pctB": s_ev / (0.10 * b_ev) if b_ev > 0 else np.nan,
                "n_signal_test_rows": len(sig),
                "n_background_test_rows": len(bkg),
                "neff_signal": neff(sig["weight_events_450fb"]),
                "neff_background": neff(bkg["weight_events_450fb"]),
            })

    scan = pd.DataFrame(rows)

    stable = scan[
        (scan["n_background_test_rows"] >= 20)
        & (scan["neff_background"] >= 5)
        & (scan["n_signal_test_rows"] >= 20)
    ].copy()

    best = stable.sort_values("S_over_sqrtB", ascending=False).head(20)

    return scan, best


def make_category_tables(test_df, run_name):
    cats = [
        "CAT0_high_purity_diagnostic",
        "CAT1_tight",
        "CAT2_medium_tight",
        "CAT3_medium",
        "CAT4_loose",
    ]

    yield_rows = []
    for cat in cats:
        sub = test_df[test_df["category"].eq(cat)]
        sig = sub[sub["is_signal"]]
        bkg = sub[~sub["is_signal"]]

        s = sig["weight_events_450fb"].sum()
        b = bkg["weight_events_450fb"].sum()

        yield_rows.append({
            "run": run_name,
            "category": cat,
            "n_signal_test_rows": len(sig),
            "n_background_test_rows": len(bkg),
            "signal_events_450fb": s,
            "background_events_450fb": b,
            "S_over_B": s / b if b > 0 else np.nan,
            "S_over_sqrtB": s / np.sqrt(b) if b > 0 else np.nan,
            "S_over_10pctB": s / (0.10 * b) if b > 0 else np.nan,
            "neff_signal": neff(sig["weight_events_450fb"]),
            "neff_background": neff(bkg["weight_events_450fb"]),
        })

    yields = pd.DataFrame(yield_rows)

    comp = (
        test_df[test_df["category"].isin(cats) & ~test_df["is_signal"]]
        .groupby(["category", "analysis_sample"], as_index=False)
        .agg(
            n_test_rows=("event", "size"),
            background_events_450fb=("weight_events_450fb", "sum"),
            neff_sample=("weight_events_450fb", neff),
            mean_qcd_score=("dnn_qcd_score", "mean"),
            mean_top_score=("dnn_top_score", "mean"),
        )
    )

    comp["run"] = run_name
    total_bkg = comp.groupby("category")["background_events_450fb"].transform("sum")
    comp["fraction_of_category_background"] = comp["background_events_450fb"] / total_bkg
    comp = comp.sort_values(["category", "background_events_450fb"], ascending=[True, False])

    return yields, comp


def run_one(run_name, wrapper_path):
    wrapper = import_wrapper(wrapper_path, f"wrapper_{run_name}")
    df, feature_cols = wrapper.load_all()

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    total_counts = df.groupby("analysis_sample").size()
    test_counts = test_df.groupby("analysis_sample").size()

    test_df = test_df.copy()
    test_df["weight_pb_scaled_to_full"] = test_df["weight_pb"] * test_df["analysis_sample"].map(
        lambda s: total_counts.loc[s] / test_counts.loc[s]
    )
    test_df["weight_events_450fb"] = test_df["weight_pb_scaled_to_full"] * LUMI_PB

    qcd_scores, qcd_auc, qcd_wauc, qcd_ntrain, qcd_ntest = train_dnn_classifier(
        train_df, test_df, feature_cols, "is_qcd", "qcd", run_name
    )

    top_scores, top_auc, top_wauc, top_ntrain, top_ntest = train_dnn_classifier(
        train_df, test_df, feature_cols, "is_top", "top", run_name
    )

    test_df["dnn_qcd_score"] = qcd_scores
    test_df["dnn_top_score"] = top_scores
    test_df["category"] = assign_category(test_df)

    scored_path = OUTDIR / f"scored_test_events_dnn_v3_qcdplus_{run_name}.parquet"
    test_df.to_parquet(scored_path, index=False)

    auc_rows = pd.DataFrame([
        {
            "run": run_name,
            "classifier": "DNN_QCD",
            "negative_class": "QCD bbbb HT slices",
            "unweighted_auc": qcd_auc,
            "physics_weighted_auc": qcd_wauc,
            "n_train_task": qcd_ntrain,
            "n_test_task": qcd_ntest,
            "n_features": len(feature_cols),
        },
        {
            "run": run_name,
            "classifier": "DNN_top",
            "negative_class": "ttbar / top-like backgrounds",
            "unweighted_auc": top_auc,
            "physics_weighted_auc": top_wauc,
            "n_train_task": top_ntrain,
            "n_test_task": top_ntest,
            "n_features": len(feature_cols),
        },
    ])

    scan, best = scan_rectangles(test_df)
    scan["run"] = run_name
    best["run"] = run_name

    yields, comp = make_category_tables(test_df, run_name)

    print(f"\n=== {run_name} DNN AUC ===")
    print(auc_rows.to_string(index=False))

    print(f"\n=== {run_name} best stable rectangles ===")
    print(best.head(10).to_string(index=False))

    print(f"\n=== {run_name} category yields ===")
    print(yields.to_string(index=False))

    return auc_rows, scan, best, yields, comp


def main():
    all_auc = []
    all_scan = []
    all_best = []
    all_yields = []
    all_comp = []

    for run_name, wrapper_path in RUNS.items():
        auc, scan, best, yields, comp = run_one(run_name, wrapper_path)
        all_auc.append(auc)
        all_scan.append(scan)
        all_best.append(best)
        all_yields.append(yields)
        all_comp.append(comp)

    outputs = {
        "dnn_auc_summary": pd.concat(all_auc, ignore_index=True),
        "dnn_rectangle_scan": pd.concat(all_scan, ignore_index=True),
        "dnn_best_stable_rectangles": pd.concat(all_best, ignore_index=True),
        "dnn_category_yields": pd.concat(all_yields, ignore_index=True),
        "dnn_category_background_composition": pd.concat(all_comp, ignore_index=True),
    }

    for name, df in outputs.items():
        df.to_csv(OUTDIR / f"{name}.csv", index=False)
        (OUTDIR / f"{name}.md").write_text(df.to_markdown(index=False) + "\n")

    (OUTDIR / "README.md").write_text(
        "# DNN-v3 qcdplus mass-aware and topology-only baselines\n\n"
        "This trains ordinary dense neural networks using the same samples and train/test split as BDT-v3 qcdplus.\n\n"
        "Two feature sets are evaluated:\n"
        "- mass_aware: same features as mass-aware BDT-v3 qcdplus\n"
        "- topology_only: direct mass variables removed\n\n"
        "Outputs include AUC summaries, rectangle scans, category yields, background composition, scored test events, and saved PyTorch models.\n"
    )

    print("\nWrote:", OUTDIR)


if __name__ == "__main__":
    main()
