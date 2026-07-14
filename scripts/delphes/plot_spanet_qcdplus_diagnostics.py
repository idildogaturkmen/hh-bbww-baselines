#!/usr/bin/env python3

import os
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, roc_auc_score

REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])
TAG = "qcdplus_btag4_v1"

TRAIN_SCRIPT = REPO / "scripts/delphes/train_hh4b_spanet_qcdplus.py"
NPZ_DIR = STORE / f"spanet_npz/{TAG}"

MODEL_DIR = REPO / "outputs/tables/hh4b_spanet_qcdplus_qcdplus_btag4_v1_2026_07_13"
MODEL_PATH = MODEL_DIR / "spanet_toy_model.pt"

OUT_TABLE = REPO / "outputs/tables/spanet_qcdplus_diagnostics_2026_07_14"
OUT_PLOT = REPO / "outputs/plots/spanet_qcdplus_diagnostics_2026_07_14"
OUT_TABLE.mkdir(parents=True, exist_ok=True)
OUT_PLOT.mkdir(parents=True, exist_ok=True)

LUMI_PB = 450_000.0

def analysis_label(ax, lumi_fb=450, extra=None):
    """Draw a non-CMS analysis label above the plotting area."""
    fig = ax.figure

    fig.text(
        0.105, 0.988, "Delphes simulation",
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.90, 0.988,
        rf"13 TeV, {lumi_fb:g} fb$^{{-1}}$",
        fontsize=10.5,
        ha="right",
        va="top",
    )


def load_train_module():
    spec = importlib.util.spec_from_file_location("spanet_train", TRAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def auc(y, s, w=None):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, s, sample_weight=w)

def make_roc(y, score, weight, label, outbase):
    fpr, tpr, _ = roc_curve(y, score, sample_weight=weight)
    roc_auc = auc(y, score, weight)

    df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    df.to_csv(OUT_TABLE / f"{outbase}.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(fpr, tpr, linewidth=2, label=f"{label} AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="0.5")
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    analysis_label(ax)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(OUT_PLOT / f"{outbase}.png", dpi=220)
    fig.savefig(OUT_PLOT / f"{outbase}.pdf")
    plt.close(fig)

    eps = np.clip(fpr, 1e-5, 1.0)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(tpr, 1.0 / eps, linewidth=2, label=f"{label} AUC={roc_auc:.3f}")
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection")
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", frameon=False)
    analysis_label(ax)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(OUT_PLOT / f"{outbase}_background_rejection.png", dpi=220)
    fig.savefig(OUT_PLOT / f"{outbase}_background_rejection.pdf")
    plt.close(fig)

    return roc_auc

def score_hist(df):
    groups = {
        "Signal": df["y"] == 1,
        "ttbar": df["sample_name"].str.contains("ttbar", case=False, regex=False),
        "QCD": df["sample_name"].str.contains("qcd", case=False, regex=False),
        "Zbbbb": df["sample_name"].str.contains("Zbbbb", case=False, regex=False),
    }

    bins = np.linspace(0, 1, 31)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for name, mask in groups.items():
        d = df[mask]
        if len(d) == 0:
            continue
        ax.hist(d["score"], bins=bins, weights=d["weight_pb_scaled"] * LUMI_PB,
                histtype="step", linewidth=2, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("SPA-Net event score")
    ax.set_ylabel(r"Expected events / bin")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    analysis_label(ax)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(OUT_PLOT / "spanet_score_distribution.png", dpi=220)
    fig.savefig(OUT_PLOT / "spanet_score_distribution.pdf")
    plt.close(fig)

def training_history():
    hist_path = MODEL_DIR / "training_history.csv"
    if not hist_path.exists():
        return
    h = pd.read_csv(hist_path)
    h.to_csv(OUT_TABLE / "training_history_copy.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(h["epoch"], h["train_loss"], label="total train loss", linewidth=2)
    ax.plot(h["epoch"], h["train_cls_loss"], label="classification loss", linewidth=2)
    ax.plot(h["epoch"], h["train_assignment_loss"], label="assignment loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    analysis_label(ax)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(OUT_PLOT / "spanet_training_losses.png", dpi=220)
    fig.savefig(OUT_PLOT / "spanet_training_losses.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(h["epoch"], h["val_auc_unweighted"], linewidth=2)
    best = h.iloc[h["val_auc_unweighted"].idxmax()]
    ax.scatter([best["epoch"]], [best["val_auc_unweighted"]], s=50)
    ax.text(best["epoch"], best["val_auc_unweighted"], f" best epoch {int(best['epoch'])}", va="bottom")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.grid(True, alpha=0.3)
    analysis_label(ax)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(OUT_PLOT / "spanet_validation_auc.png", dpi=220)
    fig.savefig(OUT_PLOT / "spanet_validation_auc.pdf")
    plt.close(fig)

def main():
    mod = load_train_module()
    device = torch.device("cpu")

    train_ds = mod.HH4BDataset(NPZ_DIR / "hh4b_spanet_leading8_train.npz")
    test_ds = mod.HH4BDataset(NPZ_DIR / "hh4b_spanet_leading8_test.npz", train_ds.mean, train_ds.std)
    all_ds = mod.HH4BDataset(NPZ_DIR / "hh4b_spanet_leading8_all.npz", train_ds.mean, train_ds.std)

    model = mod.HH4BSPANet()
    state = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)

    pred = mod.predict(model, test_ds, device, batch_size=2048)
    w_scaled = mod.scaled_test_weights(pred, all_ds)

    sample_names = np.asarray(test_ds.sample_names).astype(str)
    pred_df = pd.DataFrame({
        "score": pred["score"],
        "y": pred["y"],
        "weight_pb": pred["weight_pb"],
        "weight_pb_scaled": w_scaled,
        "sample_id": pred["sample_id"],
        "sample_name": sample_names[pred["sample_id"]],
        "assignment_mask": pred["assignment_mask"],
    })
    pred_df.to_csv(OUT_TABLE / "spanet_test_predictions.csv", index=False)

    y = pred_df["y"].to_numpy()
    s = pred_df["score"].to_numpy()
    w = pred_df["weight_pb_scaled"].to_numpy()

    rows = []
    rows.append({
        "roc": "signal_vs_all_background",
        "auc_unweighted": auc(y, s),
        "auc_weighted": make_roc(y, s, w, "Signal vs all backgrounds", "roc_signal_vs_all_background"),
    })

    qcd_mask = (pred_df["y"] == 1) | pred_df["sample_name"].str.contains("qcd", case=False, regex=False)
    rows.append({
        "roc": "signal_vs_qcd",
        "auc_unweighted": auc(y[qcd_mask], s[qcd_mask]),
        "auc_weighted": make_roc(y[qcd_mask], s[qcd_mask], w[qcd_mask], "Signal vs QCD", "roc_signal_vs_qcd"),
    })

    top_mask = (pred_df["y"] == 1) | pred_df["sample_name"].str.contains("ttbar", case=False, regex=False)
    rows.append({
        "roc": "signal_vs_ttbar",
        "auc_unweighted": auc(y[top_mask], s[top_mask]),
        "auc_weighted": make_roc(y[top_mask], s[top_mask], w[top_mask], "Signal vs ttbar", "roc_signal_vs_ttbar"),
    })

    roc_summary = pd.DataFrame(rows)
    roc_summary.to_csv(OUT_TABLE / "spanet_roc_summary.csv", index=False)
    (OUT_TABLE / "spanet_roc_summary.md").write_text(roc_summary.to_markdown(index=False) + "\n")

    score_hist(pred_df)
    training_history()

    print("Wrote tables:", OUT_TABLE)
    print("Wrote plots:", OUT_PLOT)
    print(roc_summary.to_string(index=False))

if __name__ == "__main__":
    main()
