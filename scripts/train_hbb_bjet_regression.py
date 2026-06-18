"""
Train and evaluate CMS-inspired H->bb b-jet response correction models.

Input
-----
Uses the dataset made by:
    scripts/make_hbb_bjet_regression_dataset.py

Expected files:
    outputs/hbb_bjet_regression_dataset/train.npz
    outputs/hbb_bjet_regression_dataset/val.npz
    outputs/hbb_bjet_regression_dataset/test.npz
    outputs/hbb_bjet_regression_dataset/feature_names.json
    outputs/hbb_bjet_regression_dataset/metadata.json

Models compared
---------------
1. uncorrected:
   corrected pT = reco pT

2. binned pT/eta correction:
   train-only median target in bins of reco pT and |eta|

3. DNN regression:
   predicts target = log(GenJetAK4 pT / RecoJetAK4 pT)

Application
-----------
For a prediction y_pred:

    corrected pT = reco pT * exp(y_pred)

For m_bb evaluation, the jet four-vector is scaled by the same factor:
    pT -> scale * pT
    mass -> scale * mass
    eta, phi unchanged

This is a simple simulation-level response correction study, not a full CMS
calibration/reconstruction chain.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


EPS = 1e-8
M_H = 125.0


METHOD_LABELS = {
    "uncorrected": "Uncorrected",
    "binned": "pT/eta binned correction",
    "dnn": "DNN regression",
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def ensure_dirs(outdir, plot_dir):
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_split(dataset_dir, split):
    data = np.load(dataset_dir / f"{split}.npz")
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    df = pd.DataFrame(
        {
            "event_id": data["event_id"].astype(np.int64),
            "pair_slot": data["pair_slot"].astype(np.int64),
            "reco_jet_idx": data["reco_jet_idx"].astype(np.int64),
            "genjet_idx": data["genjet_idx"].astype(np.int64),
            "reco_pt": data["reco_pt"].astype(np.float64),
            "reco_eta": data["reco_eta"].astype(np.float64),
            "reco_phi": data["reco_phi"].astype(np.float64),
            "reco_mass": data["reco_mass"].astype(np.float64),
            "genjet_pt": data["genjet_pt"].astype(np.float64),
            "genjet_eta": data["genjet_eta"].astype(np.float64),
            "genjet_phi": data["genjet_phi"].astype(np.float64),
            "genjet_mass": data["genjet_mass"].astype(np.float64),
            "genb_pt": data["genb_pt"].astype(np.float64),
            "genb_eta": data["genb_eta"].astype(np.float64),
            "genb_phi": data["genb_phi"].astype(np.float64),
            "response_genjet_pt_over_reco_pt": data["response_genjet_pt_over_reco_pt"].astype(np.float64),
            "matched_mbb_uncorrected_saved": data["matched_mbb_uncorrected"].astype(np.float64),
            "target": y.astype(np.float64),
            "split": split,
        }
    )

    return X, y, df


def clean_features(X):
    '''
    Replace NaN and infinite values with zero, and convert to float32.
    '''
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_standardizer(X_train):
    '''
    Compute mean and std for each feature in X_train, after cleaning. 
    Std values below EPS are set to 1.0 to avoid division by zero.
    '''
    X_train = clean_features(X_train)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std < EPS] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    '''
    Standardize features in X using the provided mean and std, after cleaning.
    '''
    X = clean_features(X)
    return ((X - mean) / std).astype(np.float32)

# Simple MLP regression model with two hidden layers, ReLU activations, batch normalization, and dropout.
class RegressionMLP(nn.Module):
    def __init__(self, n_features, hidden=128, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        '''
        Input: tensor of shape (batch_size, n_features)
        Output: tensor of shape (batch_size,) representing the predicted target value for each input row.
        '''
        return self.net(x).squeeze(-1)


def train_dnn(X_train, y_train, X_val, y_val, args):
    '''
    Train a DNN regression model to predict the target from input features, 
    using Huber/SmoothL1 loss and early stopping based on validation loss.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = RegressionMLP(
        n_features=X_train.shape[1],
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_x = torch.tensor(X_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(y_val, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # uses AdamW optimizer with weight decay for regularization; learning rate and weight decay are set by args.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_state = None
    best_val = float("inf")
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = [] # store per-batch losses for averaging at the end of the epoch

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = torch.nn.functional.smooth_l1_loss(pred, yb, beta=args.huber_beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))

        model.eval() # set model to evaluation mode for validation loss calculation (disables dropout, batchnorm uses running stats)
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = torch.nn.functional.smooth_l1_loss(
                val_pred,
                val_y,
                beta=args.huber_beta,
            )

        train_loss = float(np.mean(train_losses))
        val_loss = float(val_loss.detach().cpu())

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"epoch {epoch:04d} | train loss {train_loss:.6f} | val loss {val_loss:.6f}",
                flush=True,
            )

        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}; best val loss = {best_val:.6f}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, device


def predict_dnn(model, X, device, batch_size=8192):
    '''
    Use the trained DNN model to predict target values for input features X, in batches.
    '''
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds).astype(np.float64)


def parse_bins(text):
    '''
    Parse a comma-separated list of bin edges from a string, and return a sorted numpy array of floats.
    '''
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) < 2:
        raise ValueError(f"Need at least two bin edges, got: {text}")
    values = sorted(values)
    return np.array(values, dtype=np.float64)


def append_inf_edge(edges):
    '''
    If the last edge is not already infinity, append infinity to the edges array. 
    This ensures that all values above the last finite edge are included in the last bin.
    '''
    if not np.isinf(edges[-1]):
        edges = np.concatenate([edges, np.array([np.inf])])
    return edges


def bin_indices(values, edges):
    idx = np.searchsorted(edges, values, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return idx


def fit_binned_correction(train_df, pt_edges, eta_edges, min_bin_count):
    '''
    Fit a binned correction model by computing the median target in bins of reco pT and |eta|, using only the training split.

    The correction for each bin is the median target in that bin. 
    If a bin has fewer than min_bin_count training jets, it falls back to a global median correction computed from all training jets.
    '''
    pt_edges = append_inf_edge(pt_edges)
    eta_edges = append_inf_edge(eta_edges)

    global_median = float(np.median(train_df["target"]))
    pt_idx = bin_indices(train_df["reco_pt"].to_numpy(), pt_edges)
    eta_idx = bin_indices(np.abs(train_df["reco_eta"].to_numpy()), eta_edges)

    table = {}
    rows = []

    for i in range(len(pt_edges) - 1):
        for j in range(len(eta_edges) - 1):
            mask = (pt_idx == i) & (eta_idx == j)
            n = int(np.sum(mask))

            if n >= min_bin_count:
                correction = float(np.median(train_df.loc[mask, "target"]))
                source = "bin_median"
            else:
                correction = global_median
                source = "global_fallback"

            table[(i, j)] = correction
            rows.append(
                {
                    "pt_bin": i,
                    "eta_bin": j,
                    "pt_low": float(pt_edges[i]),
                    "pt_high": float(pt_edges[i + 1]) if np.isfinite(pt_edges[i + 1]) else "inf",
                    "abs_eta_low": float(eta_edges[j]),
                    "abs_eta_high": float(eta_edges[j + 1]) if np.isfinite(eta_edges[j + 1]) else "inf",
                    "n_train": n,
                    "target_log_median": correction,
                    "correction_factor": float(np.exp(correction)),
                    "source": source,
                }
            )

    return {
        "pt_edges": pt_edges,
        "eta_edges": eta_edges,
        "global_median": global_median,
        "table": table,
        "summary": pd.DataFrame(rows),
    }


def predict_binned_correction(df, binned_model):
    '''
    Predict binned correction values for the input dataframe df using the provided binned_model.
    '''
    pt_edges = binned_model["pt_edges"]
    eta_edges = binned_model["eta_edges"]
    pt_idx = bin_indices(df["reco_pt"].to_numpy(), pt_edges)
    eta_idx = bin_indices(np.abs(df["reco_eta"].to_numpy()), eta_edges)

    pred = np.zeros(len(df), dtype=np.float64)
    for k, (i, j) in enumerate(zip(pt_idx, eta_idx)):
        pred[k] = binned_model["table"].get((int(i), int(j)), binned_model["global_median"])

    return pred


def maybe_clip(pred, args):
    '''
    Optionally clip the predicted target values to a specified range defined by args.clip_pred_min and args.clip_pred_max.
     - If both are None, no clipping is applied.
     - If one is None, it means no limit in that direction (e.g. if clip_pred_min is None and clip_pred_max is 0.5, then values above 0.5 are clipped but there is no lower clipping).
     - If both are set, values outside the [clip_pred_min, clip_pred_max] range are clipped to the nearest edge of the range.
     This can be useful to prevent extreme correction factors that could arise from outlier predictions.
     The clipping is applied after all predictions are made, so it does not affect the training process, only the final evaluation and plots.
    '''
    if args.clip_pred_min is None and args.clip_pred_max is None:
        return pred
    lo = -np.inf if args.clip_pred_min is None else args.clip_pred_min
    hi = np.inf if args.clip_pred_max is None else args.clip_pred_max
    return np.clip(pred, lo, hi)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p16": float("nan"),
            "p84": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
        }

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p16": float(np.percentile(values, 16)),
        "p84": float(np.percentile(values, 84)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def perjet_metrics(df, pred, method):
    '''
    Compute per-jet metrics for the given dataframe df, predicted target values pred, and method name.
    '''
    target = df["target"].to_numpy()
    residual = pred - target
    closure = np.exp(residual)
    corr_factor = np.exp(pred)

    s_closure = summarize(closure)
    s_corr = summarize(corr_factor)

    return {
        "method": method,
        "n_jets": int(len(df)),
        "target_mae": float(np.mean(np.abs(residual))),
        "target_rmse": float(np.sqrt(np.mean(residual**2))),
        "target_median_abs_error": float(np.median(np.abs(residual))),
        "closure_mean": s_closure["mean"],
        "closure_median": s_closure["median"],
        "closure_p16": s_closure["p16"],
        "closure_p84": s_closure["p84"],
        "closure_width68": float(0.5 * (s_closure["p84"] - s_closure["p16"])),
        "correction_factor_mean": s_corr["mean"],
        "correction_factor_median": s_corr["median"],
        "correction_factor_p16": s_corr["p16"],
        "correction_factor_p84": s_corr["p84"],
    }


def four_vector(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return e, px, py, pz


def invariant_mass_two(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    e1, px1, py1, pz1 = four_vector(pt1, eta1, phi1, m1)
    e2, px2, py2, pz2 = four_vector(pt2, eta2, phi2, m2)

    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    mass2 = e**2 - px**2 - py**2 - pz**2
    return float(np.sqrt(max(mass2, 0.0)))


def build_event_mass_table(df, pred_dict):
    work = df.reset_index(drop=True).copy()
    for method, pred in pred_dict.items():
        work[f"pred_{method}"] = pred
        work[f"corr_{method}"] = np.exp(pred)

    rows = []
    skipped = 0

    for event_id, group in work.groupby("event_id"):
        if len(group) != 2:
            skipped += 1
            continue

        group = group.sort_values("pair_slot")
        a = group.iloc[0]
        b = group.iloc[1]

        out = {
            "event_id": int(event_id),
            "lower_reco_pt": float(min(a["reco_pt"], b["reco_pt"])),
            "higher_reco_pt": float(max(a["reco_pt"], b["reco_pt"])),
            "saved_uncorrected_mbb": float(a["matched_mbb_uncorrected_saved"]),
        }

        for method in pred_dict:
            ca = float(a[f"corr_{method}"])
            cb = float(b[f"corr_{method}"])

            pt1 = float(a["reco_pt"]) * ca
            pt2 = float(b["reco_pt"]) * cb
            m1 = max(float(a["reco_mass"]), 0.0) * ca
            m2 = max(float(b["reco_mass"]), 0.0) * cb

            mbb = invariant_mass_two(
                pt1,
                float(a["reco_eta"]),
                float(a["reco_phi"]),
                m1,
                pt2,
                float(b["reco_eta"]),
                float(b["reco_phi"]),
                m2,
            )
            out[f"mbb_{method}"] = mbb

        rows.append(out)

    return pd.DataFrame(rows), skipped


def mass_summary(event_df, method):
    values = event_df[f"mbb_{method}"].to_numpy(dtype=np.float64)
    s = summarize(values)

    return {
        "method": method,
        "n_events": s["n"],
        "mean": s["mean"],
        "std": s["std"],
        "median": s["median"],
        "p16": s["p16"],
        "p84": s["p84"],
        "width68": float(0.5 * (s["p84"] - s["p16"])),
        "frac_90_140": float(np.mean((values >= 90.0) & (values <= 140.0))),
        "frac_100_150": float(np.mean((values >= 100.0) & (values <= 150.0))),
        "median_abs_offset_from_125": float(abs(s["median"] - M_H)),
    }


def make_response_profile(df, pred_dict, pt_edges):
    pt_edges = append_inf_edge(pt_edges)
    rows = []

    for method, pred in pred_dict.items():
        closure = np.exp(pred - df["target"].to_numpy())
        pt = df["reco_pt"].to_numpy()
        idx = bin_indices(pt, pt_edges)

        for i in range(len(pt_edges) - 1):
            mask = idx == i
            if not np.any(mask):
                continue

            vals = closure[mask]
            rows.append(
                {
                    "method": method,
                    "pt_low": float(pt_edges[i]),
                    "pt_high": float(pt_edges[i + 1]) if np.isfinite(pt_edges[i + 1]) else "inf",
                    "pt_center": float(np.median(pt[mask])),
                    "n": int(np.sum(mask)),
                    "closure_median": float(np.median(vals)),
                    "closure_p16": float(np.percentile(vals, 16)),
                    "closure_p84": float(np.percentile(vals, 84)),
                }
            )

    return pd.DataFrame(rows)


def make_lower_pt_mbb_profile(event_df, methods, pt_edges):
    pt_edges = append_inf_edge(pt_edges)
    lower_pt = event_df["lower_reco_pt"].to_numpy()
    idx = bin_indices(lower_pt, pt_edges)

    rows = []
    for method in methods:
        values = event_df[f"mbb_{method}"].to_numpy()
        for i in range(len(pt_edges) - 1):
            mask = idx == i
            if not np.any(mask):
                continue

            rows.append(
                {
                    "method": method,
                    "pt_low": float(pt_edges[i]),
                    "pt_high": float(pt_edges[i + 1]) if np.isfinite(pt_edges[i + 1]) else "inf",
                    "pt_center": float(np.median(lower_pt[mask])),
                    "n": int(np.sum(mask)),
                    "mbb_median": float(np.median(values[mask])),
                    "mbb_p16": float(np.percentile(values[mask], 16)),
                    "mbb_p84": float(np.percentile(values[mask], 84)),
                }
            )
    return pd.DataFrame(rows)


def make_plots(outdir, plot_dir, history, test_df, pred_dict, event_df, response_profile, lower_pt_profile):
    # Training curve
    if history:
        hist = pd.DataFrame(history)
        plt.figure(figsize=(7.5, 5.0))
        plt.plot(hist["epoch"], hist["train_loss"], label="train")
        plt.plot(hist["epoch"], hist["val_loss"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Huber / SmoothL1 loss")
        plt.title("DNN training curve")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / "dnn_training_curve.png", dpi=200)
        plt.close()

    # DNN target vs prediction
    plt.figure(figsize=(6.0, 6.0))
    plt.scatter(test_df["target"], pred_dict["dnn"], s=4, alpha=0.25)
    lo = float(min(np.min(test_df["target"]), np.min(pred_dict["dnn"])))
    hi = float(max(np.max(test_df["target"]), np.max(pred_dict["dnn"])))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    plt.xlabel("True target log(GenJet pT / reco pT)")
    plt.ylabel("DNN predicted target")
    plt.title("DNN target prediction on test jets")
    plt.tight_layout()
    plt.savefig(plot_dir / "dnn_target_vs_prediction.png", dpi=200)
    plt.close()

    # Correction factor vs reco pT
    plt.figure(figsize=(7.5, 5.0))
    plt.scatter(test_df["reco_pt"], np.exp(pred_dict["dnn"]), s=4, alpha=0.25, label="DNN")
    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Reco AK4 jet pT [GeV]")
    plt.ylabel("Predicted correction factor")
    plt.title("DNN correction factor vs reco pT")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "dnn_correction_factor_vs_reco_pt.png", dpi=200)
    plt.close()

    # Response closure profile
    plt.figure(figsize=(7.5, 5.0))
    for method, label in METHOD_LABELS.items():
        part = response_profile[response_profile["method"] == method]
        if len(part) == 0:
            continue
        plt.plot(part["pt_center"], part["closure_median"], marker="o", label=label)

    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Reco AK4 jet pT [GeV]")
    plt.ylabel("Corrected pT / GenJetAK4 pT")
    plt.title("Response closure vs reco pT")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "response_closure_vs_reco_pt.png", dpi=200)
    plt.close()

    # m_bb histograms
    plt.figure(figsize=(7.5, 5.0))
    bins = np.linspace(0.0, 250.0, 80)
    for method, label in METHOD_LABELS.items():
        plt.hist(
            event_df[f"mbb_{method}"],
            bins=bins,
            histtype="step",
            linewidth=1.5,
            density=True,
            label=label,
        )
    plt.axvline(M_H, linestyle="--", linewidth=1.0, label="125 GeV")
    plt.xlabel("Matched H->bb m_bb [GeV]")
    plt.ylabel("Normalized events")
    plt.title("m_bb before and after b-jet response correction")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "mbb_before_after_corrections.png", dpi=200)
    plt.close()

    # median m_bb vs lower reco pT
    plt.figure(figsize=(7.5, 5.0))
    for method, label in METHOD_LABELS.items():
        part = lower_pt_profile[lower_pt_profile["method"] == method]
        if len(part) == 0:
            continue
        plt.plot(part["pt_center"], part["mbb_median"], marker="o", label=label)

    plt.axhline(M_H, linestyle="--", linewidth=1.0, label="125 GeV")
    plt.xlabel("Lower reco b-jet pT [GeV]")
    plt.ylabel("Median matched m_bb [GeV]")
    plt.title("Median m_bb vs lower b-jet pT")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "median_mbb_vs_lower_reco_pt.png", dpi=200)
    plt.close()


def write_markdown(summary_path, dataset_metadata, args, perjet_df, mass_df, skipped_events, history):
    lines = [
        "# H->bb B-Jet Response Regression",
        "",
        "This study trains simulation-level b-jet response corrections for truth-matched H->bb AK4 jets in COLLIDE-1M HH->bbWW.",
        "",
        "The target is:",
        "",
        "`target = log(GenJetAK4 pT / RecoJetAK4 pT)`",
        "",
        "The correction is applied as:",
        "",
        "`corrected pT = reco pT * exp(predicted target)`",
        "",
        "For the m_bb evaluation, the matched reco-jet four-vector is scaled by the same factor, keeping eta and phi unchanged.",
        "",
        "## Dataset configuration",
        "",
    ]

    matching = dataset_metadata.get("matching", {})
    for key, value in matching.items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Methods",
            "",
            "- **Uncorrected:** no response correction.",
            "- **pT/eta binned correction:** median target in reco pT and |eta| bins, fit using the training split only.",
            "- **DNN regression:** MLP trained with Huber/SmoothL1 loss and selected using validation loss.",
            "",
            "## Test-set per-jet response closure",
            "",
            "| Method | Jets | Target MAE | Target RMSE | Closure median | Closure 16% | Closure 84% | Closure width68 | Correction median |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in perjet_df.iterrows():
        label = METHOD_LABELS[row["method"]]
        lines.append(
            f"| {label} | {int(row['n_jets'])} | "
            f"{row['target_mae']:.4f} | {row['target_rmse']:.4f} | "
            f"{row['closure_median']:.4f} | {row['closure_p16']:.4f} | {row['closure_p84']:.4f} | "
            f"{row['closure_width68']:.4f} | {row['correction_factor_median']:.4f} |"
        )

    lines.extend(
        [
            "",
            "Here, response closure is `corrected pT / GenJetAK4 pT`; a perfect correction has median closure near 1.",
            "",
            "## Test-set event-level m_bb",
            "",
            f"Events skipped because they did not have exactly two rows in the test split: {skipped_events}",
            "",
            "| Method | Events | Mean [GeV] | Median [GeV] | 16% [GeV] | 84% [GeV] | Width68 [GeV] | Frac 90-140 | Frac 100-150 | |Median-125| [GeV] |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in mass_df.iterrows():
        label = METHOD_LABELS[row["method"]]
        lines.append(
            f"| {label} | {int(row['n_events'])} | "
            f"{row['mean']:.2f} | {row['median']:.2f} | {row['p16']:.2f} | {row['p84']:.2f} | "
            f"{row['width68']:.2f} | {row['frac_90_140']:.4f} | {row['frac_100_150']:.4f} | "
            f"{row['median_abs_offset_from_125']:.2f} |"
        )

    if history:
        best = min(history, key=lambda r: r["val_loss"])
        lines.extend(
            [
                "",
                "## DNN training",
                "",
                f"- best epoch: {best['epoch']}",
                f"- best validation loss: {best['val_loss']:.6f}",
                f"- hidden width: {args.hidden}",
                f"- dropout: {args.dropout}",
                f"- learning rate: {args.lr}",
                f"- weight decay: {args.weight_decay}",
                f"- Huber beta: {args.huber_beta}",
            ]
        )

    lines.extend(
        [
            "",
            "## Important caveat",
            "",
            "This is a simulation-level, CMS-inspired response correction study. It does not reproduce the full CMS jet energy calibration chain and does not include explicit secondary-vertex variables.",
            "",
            "## Outputs",
            "",
            "- `outputs/hbb_bjet_regression/results.json`",
            "- `outputs/hbb_bjet_regression/perjet_metrics_test.csv`",
            "- `outputs/hbb_bjet_regression/event_mass_summary_test.csv`",
            "- `outputs/hbb_bjet_regression/event_masses_test.csv`",
            "- `outputs/hbb_bjet_regression/predictions_test.csv`",
            "- `outputs/hbb_bjet_regression/binned_correction_table.csv`",
            "- `outputs/hbb_bjet_regression/dnn_training_history.csv`",
            "- `outputs/plots/hbb_bjet_regression/*.png`",
        ]
    )

    summary_path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train H->bb b-jet response regression models.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/hbb_bjet_regression_dataset"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/hbb_bjet_regression"))
    parser.add_argument("--plot-dir", type=Path, default=Path("outputs/plots/hbb_bjet_regression"))
    parser.add_argument("--summary-md", type=Path, default=Path("RESULTS_HBB_BJET_REGRESSION.md"))

    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--huber-beta", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1.0e-5)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument(
        "--pt-bins",
        type=str,
        default="20,30,40,50,65,80,100,140,200,400,1000",
        help="Comma-separated reco pT bin edges for binned correction and profiles.",
    )
    parser.add_argument(
        "--eta-bins",
        type=str,
        default="0,0.8,1.5,2.0,2.5",
        help="Comma-separated |eta| bin edges for binned correction.",
    )
    parser.add_argument("--min-bin-count", type=int, default=50)

    parser.add_argument("--clip-pred-min", type=float, default=None)
    parser.add_argument("--clip-pred-max", type=float, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs(args.outdir, args.plot_dir)

    feature_names = load_json(args.dataset_dir / "feature_names.json")
    dataset_metadata = load_json(args.dataset_dir / "metadata.json")

    print("Loading dataset...", flush=True)
    X_train_raw, y_train, train_df = load_split(args.dataset_dir, "train")
    X_val_raw, y_val, val_df = load_split(args.dataset_dir, "val")
    X_test_raw, y_test, test_df = load_split(args.dataset_dir, "test")

    print(f"train: X={X_train_raw.shape}, y={y_train.shape}")
    print(f"val:   X={X_val_raw.shape}, y={y_val.shape}")
    print(f"test:  X={X_test_raw.shape}, y={y_test.shape}")

    mean, std = fit_standardizer(X_train_raw)
    X_train = apply_standardizer(X_train_raw, mean, std)
    X_val = apply_standardizer(X_val_raw, mean, std)
    X_test = apply_standardizer(X_test_raw, mean, std)

    # Binned correction baseline
    print("Fitting pT/eta binned correction...", flush=True)
    pt_edges = parse_bins(args.pt_bins)
    eta_edges = parse_bins(args.eta_bins)

    binned_model = fit_binned_correction(
        train_df=train_df,
        pt_edges=pt_edges,
        eta_edges=eta_edges,
        min_bin_count=args.min_bin_count,
    )
    binned_model["summary"].to_csv(args.outdir / "binned_correction_table.csv", index=False)

    pred_binned_train = predict_binned_correction(train_df, binned_model)
    pred_binned_val = predict_binned_correction(val_df, binned_model)
    pred_binned_test = predict_binned_correction(test_df, binned_model)

    # DNN regression
    print("Training DNN regression...", flush=True)
    model, history, device = train_dnn(X_train, y_train, X_val, y_val, args)

    pred_dnn_train = predict_dnn(model, X_train, device)
    pred_dnn_val = predict_dnn(model, X_val, device)
    pred_dnn_test = predict_dnn(model, X_test, device)

    pred_dnn_train = maybe_clip(pred_dnn_train, args)
    pred_dnn_val = maybe_clip(pred_dnn_val, args)
    pred_dnn_test = maybe_clip(pred_dnn_test, args)

    # Predictions for test split
    pred_uncorrected_test = np.zeros(len(test_df), dtype=np.float64)
    pred_dict_test = {
        "uncorrected": pred_uncorrected_test,
        "binned": pred_binned_test,
        "dnn": pred_dnn_test,
    }

    # Save model/scaler
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": feature_names,
            "mean": mean,
            "std": std,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        args.outdir / "dnn_model.pt",
    )

    with open(args.outdir / "scaler.json", "w") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
            f,
            indent=2,
        )

    # Per-jet metrics
    perjet_rows = []
    for method, pred in pred_dict_test.items():
        perjet_rows.append(perjet_metrics(test_df, pred, method))
    perjet_df = pd.DataFrame(perjet_rows)
    perjet_df.to_csv(args.outdir / "perjet_metrics_test.csv", index=False)

    # Event-level m_bb evaluation
    event_df, skipped_events = build_event_mass_table(test_df, pred_dict_test)
    event_df.to_csv(args.outdir / "event_masses_test.csv", index=False)

    mass_rows = []
    for method in ["uncorrected", "binned", "dnn"]:
        mass_rows.append(mass_summary(event_df, method))
    mass_df = pd.DataFrame(mass_rows)
    mass_df.to_csv(args.outdir / "event_mass_summary_test.csv", index=False)

    # Profiles
    response_profile = make_response_profile(test_df, pred_dict_test, pt_edges)
    response_profile.to_csv(args.outdir / "response_closure_profile_vs_reco_pt_test.csv", index=False)

    lower_pt_profile = make_lower_pt_mbb_profile(
        event_df,
        methods=["uncorrected", "binned", "dnn"],
        pt_edges=pt_edges,
    )
    lower_pt_profile.to_csv(args.outdir / "lower_pt_mbb_profile_test.csv", index=False)

    # Save test predictions
    pred_test_df = test_df.copy()
    pred_test_df["pred_uncorrected"] = pred_uncorrected_test
    pred_test_df["pred_binned"] = pred_binned_test
    pred_test_df["pred_dnn"] = pred_dnn_test
    pred_test_df["corr_uncorrected"] = np.exp(pred_uncorrected_test)
    pred_test_df["corr_binned"] = np.exp(pred_binned_test)
    pred_test_df["corr_dnn"] = np.exp(pred_dnn_test)
    pred_test_df["closure_uncorrected"] = np.exp(pred_uncorrected_test - pred_test_df["target"].to_numpy())
    pred_test_df["closure_binned"] = np.exp(pred_binned_test - pred_test_df["target"].to_numpy())
    pred_test_df["closure_dnn"] = np.exp(pred_dnn_test - pred_test_df["target"].to_numpy())
    pred_test_df.to_csv(args.outdir / "predictions_test.csv", index=False)

    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(args.outdir / "dnn_training_history.csv", index=False)

    # Save compact results JSON
    results = {
        "dataset_metadata": dataset_metadata,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_features": int(X_train.shape[1]),
        "n_train_jets": int(len(train_df)),
        "n_val_jets": int(len(val_df)),
        "n_test_jets": int(len(test_df)),
        "n_test_events": int(len(event_df)),
        "skipped_test_events": int(skipped_events),
        "perjet_metrics_test": perjet_df.to_dict(orient="records"),
        "event_mass_summary_test": mass_df.to_dict(orient="records"),
        "best_epoch": int(min(history, key=lambda r: r["val_loss"])["epoch"]) if history else None,
        "best_val_loss": float(min(history, key=lambda r: r["val_loss"])["val_loss"]) if history else None,
    }
    with open(args.outdir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plots and markdown
    make_plots(
        outdir=args.outdir,
        plot_dir=args.plot_dir,
        history=history,
        test_df=test_df,
        pred_dict=pred_dict_test,
        event_df=event_df,
        response_profile=response_profile,
        lower_pt_profile=lower_pt_profile,
    )

    write_markdown(
        summary_path=args.summary_md,
        dataset_metadata=dataset_metadata,
        args=args,
        perjet_df=perjet_df,
        mass_df=mass_df,
        skipped_events=skipped_events,
        history=history,
    )

    print("\nTraining/evaluation complete.")
    print(f"Saved summary: {args.summary_md}")
    print(f"Saved outputs: {args.outdir}")
    print(f"Saved plots: {args.plot_dir}")
    print("\nPer-jet test metrics:")
    print(perjet_df.to_string(index=False))
    print("\nEvent-level m_bb test summary:")
    print(mass_df.to_string(index=False))


if __name__ == "__main__":
    main()