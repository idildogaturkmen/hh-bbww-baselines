import argparse
import json
import math
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve


REPO_ID = "fastmachinelearning/collide-1m"
HH_NPZ_DIR = Path("outputs/hbb_npz")
OUTDIR = Path("outputs/zbb_rejection")
PLOT_DIR = Path("outputs/plots/zbb_rejection")
FEATURE_OVERLAY_DIR = PLOT_DIR / "feature_overlays"
MAX_JETS = 12
RANDOM_SEED = 42
M_Z = 91.1876
M_H = 125.0

OUTDIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "n_jets",
    "leading_jet_pt",
    "subleading_jet_pt",
    "highest_btag",
    "second_highest_btag",
    "mbb_top2_btag",
    "deltaR_top2_btag",
    "HT",
]

MODEL_FEATURES = {
    "full_bdt": FEATURE_NAMES,
    "no_mbb_bdt": [name for name in FEATURE_NAMES if name != "mbb_top2_btag"],
}

ZBB_FILES = {
    "train": [
        "ZJetsTobb_13TeV-madgraphMLM-pythia8/"
        "ZJetsTobb_13TeV-madgraphMLM-pythia8-NEVENT10000-RS52000001.parquet",
    ],
    "val": [
        "ZJetsTobb_13TeV-madgraphMLM-pythia8/"
        "ZJetsTobb_13TeV-madgraphMLM-pythia8-NEVENT10000-RS52000002.parquet",
    ],
    "test": [
        "ZJetsTobb_13TeV-madgraphMLM-pythia8/"
        "ZJetsTobb_13TeV-madgraphMLM-pythia8-NEVENT10000-RS52000003.parquet",
    ],
}

JET_COLUMNS = [
    "FullReco_JetAK4_PT",
    "FullReco_JetAK4_Eta",
    "FullReco_JetAK4_Phi",
    "FullReco_JetAK4_Mass",
    "FullReco_JetAK4_BTag",
    "FullReco_JetAK4_BTagPhys",
]

TARGET_SIGNAL_EFFS = [0.90, 0.70, 0.50]
MODEL_CONFIG = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "random_state": RANDOM_SEED,
}


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi <= -math.pi:
        dphi += 2 * math.pi
    return dphi


def delta_r(j1, j2):
    return math.sqrt((j1[1] - j2[1]) ** 2 + delta_phi(j1[2], j2[2]) ** 2)


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
    return math.sqrt(max(float(m2_tot), 0.0))


def event_features_from_jets(jets):
    if len(jets) < 2:
        return None

    pts = jets[:, 0]
    btags = jets[:, 4]

    order_pt = np.argsort(pts)[::-1]
    leading_pt = pts[order_pt[0]]
    subleading_pt = pts[order_pt[1]]

    order_btag = np.argsort(btags)[::-1]
    j1 = jets[order_btag[0]]
    j2 = jets[order_btag[1]]

    highest_btag = btags[order_btag[0]]
    second_highest_btag = btags[order_btag[1]]
    mbb = invariant_mass(j1, j2)
    drbb = delta_r(j1, j2)
    ht = float(np.sum(pts))

    return np.array(
        [
            len(jets),
            leading_pt,
            subleading_pt,
            highest_btag,
            second_highest_btag,
            mbb,
            drbb,
            ht,
        ],
        dtype=np.float32,
    )


def load_hh_features(split):
    data = np.load(HH_NPZ_DIR / f"{split}.npz")
    jets_all = data["jets"]
    mask_all = data["mask"]

    if jets_all.shape[1] != MAX_JETS or mask_all.shape[1] != MAX_JETS:
        raise ValueError(
            f"Expected HH inputs padded to MAX_JETS={MAX_JETS}, got "
            f"jets shape {jets_all.shape} and mask shape {mask_all.shape}."
        )

    xs = []
    for jets, mask in zip(jets_all, mask_all):
        feats = event_features_from_jets(jets[mask > 0])
        if feats is not None:
            xs.append(feats)

    return np.stack(xs)


def iter_zbb_events(split):
    ds = load_dataset(
        REPO_ID,
        data_files={"train": ZBB_FILES[split]},
        split="train",
        streaming=True,
    )
    yield from ds


def load_zbb_features_once(split, target_n, max_scan=None, progress_every=1000):
    xs = []
    n_scanned = 0
    n_lt2jets = 0

    for event in iter_zbb_events(split):
        n_scanned += 1
        if max_scan is not None and n_scanned > max_scan:
            break

        n_jets_raw = len(event["FullReco_JetAK4_PT"])
        if n_jets_raw < 2:
            n_lt2jets += 1
            continue

        n_jets = min(n_jets_raw, MAX_JETS)
        if n_jets < 2:
            n_lt2jets += 1
            continue

        jets = np.zeros((n_jets, 6), dtype=np.float32)
        for idx, col in enumerate(JET_COLUMNS):
            jets[:, idx] = event[col][:n_jets]

        feats = event_features_from_jets(jets)
        if feats is not None:
            xs.append(feats)

        if len(xs) > 0 and len(xs) % progress_every == 0:
            print(f"{split} ZJetsTobb usable events: {len(xs)}; scanned: {n_scanned}", flush=True)

        if len(xs) >= target_n:
            break

    if len(xs) < target_n:
        raise RuntimeError(
            f"Only found {len(xs)} usable ZJetsTobb events for split={split}, target_n={target_n}."
        )

    summary = {
        f"{split}_zbb_scanned_events": int(n_scanned),
        f"{split}_zbb_lt2jets_rejected": int(n_lt2jets),
        f"{split}_zbb_usable_events": int(len(xs)),
        f"{split}_zbb_files": ZBB_FILES[split],
    }
    return np.stack(xs), summary


def load_zbb_features(split, target_n, max_scan=None, progress_every=1000, attempts=3):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return load_zbb_features_once(split, target_n, max_scan=max_scan, progress_every=progress_every)
        except Exception as exc:
            last_error = exc
            print(
                f"ZJetsTobb streaming failed for split={split} on attempt {attempt}/{attempts}: {exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(5)
    raise last_error


def existing_split_counts(split):
    path = OUTDIR / f"features_{split}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    y = data["y"]
    return {
        f"{split}_hh_events": int(np.sum(y == 1)),
        f"{split}_zbb_events": int(np.sum(y == 0)),
        f"{split}_total_events": int(len(y)),
    }


def existing_split_is_usable(split, required_each):
    counts = existing_split_counts(split)
    if counts is None:
        return False
    return (
        counts[f"{split}_hh_events"] >= required_each
        and counts[f"{split}_zbb_events"] >= required_each
    )


def save_split(split, X_hh, X_zbb, rng):
    n = min(len(X_hh), len(X_zbb))
    X_hh = X_hh[:n]
    X_zbb = X_zbb[:n]

    y_hh = np.ones(n, dtype=np.int64)
    y_zbb = np.zeros(n, dtype=np.int64)

    X = np.concatenate([X_hh, X_zbb], axis=0)
    y = np.concatenate([y_hh, y_zbb], axis=0)
    sample = np.array(["HH_bbWW"] * n + ["ZJetsTobb"] * n)

    idx = rng.permutation(len(y))
    X = X[idx]
    y = y[idx]
    sample = sample[idx]

    np.savez(
        OUTDIR / f"features_{split}.npz",
        X=X,
        y=y,
        sample=sample,
        feature_names=np.array(FEATURE_NAMES),
    )

    return {
        f"{split}_hh_events": int(n),
        f"{split}_zbb_events": int(n),
        f"{split}_total_events": int(len(y)),
    }


def binomial_error(eff, n):
    if n <= 0 or not np.isfinite(eff):
        return float("nan")
    return math.sqrt(eff * (1.0 - eff) / n)


def choose_threshold_for_signal_eff(scores_val, y_val, target_eff):
    sig_scores = scores_val[y_val == 1]
    return float(np.quantile(sig_scores, 1.0 - target_eff))


def evaluate_threshold(scores, y, threshold):
    sig = y == 1
    bkg = y == 0
    sig_n = int(np.sum(sig))
    bkg_n = int(np.sum(bkg))
    sig_pass = int(np.sum(scores[sig] >= threshold))
    bkg_pass = int(np.sum(scores[bkg] >= threshold))
    sig_eff = sig_pass / sig_n if sig_n else float("nan")
    bkg_eff = bkg_pass / bkg_n if bkg_n else float("nan")
    bkg_rej = float("inf") if bkg_eff == 0 else 1.0 / bkg_eff
    return {
        "threshold": float(threshold),
        "signal_pass": sig_pass,
        "signal_total": sig_n,
        "signal_eff": float(sig_eff),
        "signal_eff_err": binomial_error(sig_eff, sig_n),
        "zbb_pass": bkg_pass,
        "zbb_total": bkg_n,
        "zbb_eff": float(bkg_eff),
        "zbb_eff_err": binomial_error(bkg_eff, bkg_n),
        "zbb_rejection": float(bkg_rej),
    }


def make_working_points(method, scores_val, y_val, scores_test, y_test):
    rows = []
    for target in TARGET_SIGNAL_EFFS:
        threshold = choose_threshold_for_signal_eff(scores_val, y_val, target)
        row = evaluate_threshold(scores_test, y_test, threshold)
        row["method"] = method
        row["target_signal_eff_val"] = target
        rows.append(row)
    return rows


def feature_indices(feature_names):
    return {name: idx for idx, name in enumerate(feature_names)}


def train_bdt(method, features_for_model, arrays, feature_names):
    indices = [feature_indices(feature_names)[name] for name in features_for_model]
    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    X_test, y_test = arrays["test"]

    model = GradientBoostingClassifier(**MODEL_CONFIG)
    model.fit(X_train[:, indices], y_train)

    scores = {
        "train": model.predict_proba(X_train[:, indices])[:, 1],
        "val": model.predict_proba(X_val[:, indices])[:, 1],
        "test": model.predict_proba(X_test[:, indices])[:, 1],
    }
    aucs = {
        "auc_train": float(roc_auc_score(y_train, scores["train"])),
        "auc_val": float(roc_auc_score(y_val, scores["val"])),
        "auc_test": float(roc_auc_score(y_test, scores["test"])),
    }

    joblib.dump(model, OUTDIR / f"{method}.joblib")

    importance = pd.DataFrame(
        {
            "method": method,
            "feature": features_for_model,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(OUTDIR / f"{method}_feature_importance.csv", index=False)

    return model, scores, aucs, importance


def mass_score(X, feature_names):
    # Higher scores are treated as more HH-like. This intentionally tests the
    # simplest one-sided rejection suggested by the lower Z->bb reconstructed mass.
    return X[:, feature_indices(feature_names)["mbb_top2_btag"]]


def plot_roc_curves(y_test, score_map, auc_map):
    plt.figure(figsize=(7.0, 5.0))
    for method, scores in score_map.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        plt.plot(fpr, tpr, linewidth=1.5, label=f"{method} AUC={auc_map[method]['auc_test']:.3f}")
    plt.xlabel("ZJetsTobb efficiency")
    plt.ylabel("HH signal efficiency")
    plt.title("HH -> bbWW vs Z->bb ROC on test set")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc_curve_test.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.0, 5.0))
    for method, scores in score_map.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        rejection = np.divide(1.0, fpr, out=np.full_like(fpr, np.inf), where=fpr > 0)
        finite = np.isfinite(rejection)
        plt.plot(tpr[finite], rejection[finite], linewidth=1.5, label=method)
    plt.yscale("log")
    plt.xlabel("HH signal efficiency")
    plt.ylabel("ZJetsTobb rejection")
    plt.title("Z->bb rejection vs HH signal efficiency")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "background_rejection_test.png", dpi=200)
    plt.close()


def plot_feature_overlays(X_train, y_train, feature_names):
    for idx, name in enumerate(feature_names):
        hh = X_train[y_train == 1, idx]
        zbb = X_train[y_train == 0, idx]
        plt.figure(figsize=(7.0, 4.8))
        plt.hist(hh, bins=50, histtype="step", linewidth=1.5, density=True, label="HH -> bbWW")
        plt.hist(zbb, bins=50, histtype="step", linewidth=1.5, density=True, label="ZJetsTobb")
        if name == "mbb_top2_btag":
            plt.axvline(M_Z, color="tab:blue", linestyle="--", linewidth=1.0, label="Z mass")
            plt.axvline(M_H, color="black", linestyle="--", linewidth=1.0, label="H mass")
        plt.xlabel(name)
        plt.ylabel("Normalized events")
        plt.title(f"HH vs ZJetsTobb input feature: {name}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FEATURE_OVERLAY_DIR / f"{name}.png", dpi=200)
        plt.close()


def plot_score_distributions(y_train, train_scores, y_test, test_scores):
    for method in train_scores:
        plt.figure(figsize=(7.0, 4.8))
        plt.hist(train_scores[method][y_train == 1], bins=50, histtype="step", linewidth=1.3, density=True, label="HH train")
        plt.hist(train_scores[method][y_train == 0], bins=50, histtype="step", linewidth=1.3, density=True, label="ZJetsTobb train")
        plt.hist(test_scores[method][y_test == 1], bins=50, histtype="step", linewidth=1.3, density=True, linestyle="--", label="HH test")
        plt.hist(test_scores[method][y_test == 0], bins=50, histtype="step", linewidth=1.3, density=True, linestyle="--", label="ZJetsTobb test")
        plt.xlabel(f"{method} score")
        plt.ylabel("Normalized events")
        plt.title(f"HH vs ZJetsTobb score distributions: {method}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"score_distribution_{method}.png", dpi=200)
        plt.close()


def write_summary(dataset_summary, metrics_df, working_points_df, feature_importance):
    lines = [
        "# HH -> bbWW vs Z->bb Rejection Baseline",
        "",
        "This study responds to Harvey's suggestion to test whether a dedicated rejection of Z -> bb can help. It uses COLLIDE-1M HH -> bbWW as signal and `ZJetsTobb_13TeV-madgraphMLM-pythia8` as background.",
        "",
        "## Inputs",
        "",
        "- HH signal: existing fixed `outputs/hbb_npz/{train,val,test}.npz` split.",
        "- Z background: separate ZJetsTobb shards for train, validation, and test.",
        "- Both samples use the first `MAX_JETS = 12` AK4 jets before feature construction.",
        "- Labels: HH = 1, ZJetsTobb = 0.",
        "",
        "Features:",
        "",
    ]
    for name in FEATURE_NAMES:
        lines.append(f"- `{name}`")

    lines.extend(
        [
            "",
            "## Dataset Counts",
            "",
            "| Split | HH | ZJetsTobb | Total |",
            "|---|---:|---:|---:|",
        ]
    )
    for split in ["train", "val", "test"]:
        lines.append(
            f"| {split} | {dataset_summary[f'{split}_hh_events']} | "
            f"{dataset_summary[f'{split}_zbb_events']} | {dataset_summary[f'{split}_total_events']} |"
        )

    lines.extend(
        [
            "",
            "## Test AUC",
            "",
            "| Method | Train AUC | Validation AUC | Test AUC |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in metrics_df.iterrows():
        lines.append(f"| {row['method']} | {row['auc_train']:.4f} | {row['auc_val']:.4f} | {row['auc_test']:.4f} |")

    lines.extend(
        [
            "",
            "## Validation-Chosen Working Points Evaluated On Test",
            "",
            "| Method | Target HH eff on validation | Test HH eff | Test Z eff | Test Z rejection |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in working_points_df.iterrows():
        lines.append(
            f"| {row['method']} | {row['target_signal_eff_val']:.2f} | "
            f"{row['signal_eff']:.4f} +/- {row['signal_eff_err']:.4f} | "
            f"{row['zbb_eff']:.4f} +/- {row['zbb_eff_err']:.4f} | {row['zbb_rejection']:.2f} |"
        )

    full_imp = feature_importance.get("full_bdt")
    if full_imp is not None:
        lines.extend(
            [
                "",
                "## Full-BDT Feature Importance",
                "",
                "| Feature | Importance |",
                "|---|---:|",
            ]
        )
        for _, row in full_imp.iterrows():
            lines.append(f"| {row['feature']} | {row['importance']:.4f} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The mass-only score is a deliberately simple one-sided test using the top-2-btag dijet mass, with larger values treated as more HH-like. The no-mbb BDT tests whether topology and b-tag information alone can reject ZJetsTobb. The full BDT tests the combined event-level rejection available from the current baseline features.",
            "",
            "These numbers are COLLIDE-1M baseline results, not CMS expected limits. They use balanced class counts and do not include cross-section weights or systematics.",
            "",
            "## Outputs",
            "",
            "- `outputs/zbb_rejection/features_{train,val,test}.npz`",
            "- `outputs/zbb_rejection/metrics.csv`",
            "- `outputs/zbb_rejection/working_points.csv`",
            "- `outputs/zbb_rejection/dataset_summary.json`",
            "- `outputs/plots/zbb_rejection/roc_curve_test.png`",
            "- `outputs/plots/zbb_rejection/background_rejection_test.png`",
        ]
    )
    Path("RESULTS_ZBB_REJECTION.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate HH -> bbWW vs Z->bb rejection baselines.")
    parser.add_argument("--train-z", type=int, default=None, help="Number of Z events for training; default matches HH train count.")
    parser.add_argument("--val-z", type=int, default=None, help="Number of Z events for validation; default matches HH val count.")
    parser.add_argument("--test-z", type=int, default=None, help="Number of Z events for test; default matches HH test count.")
    parser.add_argument("--max-scan-train", type=int, default=-1)
    parser.add_argument("--max-scan-val", type=int, default=-1)
    parser.add_argument("--max-scan-test", type=int, default=-1)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--stream-attempts", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate feature splits even if matching output files already exist.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    for attr in ["max_scan_train", "max_scan_val", "max_scan_test"]:
        if getattr(args, attr) < 0:
            setattr(args, attr, None)
    return args


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    hh = {split: load_hh_features(split) for split in ["train", "val", "test"]}
    targets = {
        "train": args.train_z or len(hh["train"]),
        "val": args.val_z or len(hh["val"]),
        "test": args.test_z or len(hh["test"]),
    }
    max_scan = {
        "train": args.max_scan_train,
        "val": args.max_scan_val,
        "test": args.max_scan_test,
    }

    zbb = {}
    dataset_summary = {"feature_names": FEATURE_NAMES, "max_jets": MAX_JETS, "zbb_files": ZBB_FILES}
    for split in ["train", "val", "test"]:
        required_each = min(len(hh[split]), targets[split])
        print(f"Loading HH {split}: {hh[split].shape}")
        if not args.overwrite and existing_split_is_usable(split, required_each):
            print(f"Reusing existing {OUTDIR / f'features_{split}.npz'}")
            dataset_summary.update(existing_split_counts(split))
            continue

        print(f"Streaming ZJetsTobb {split}: target={targets[split]}")
        zbb[split], z_summary = load_zbb_features(
            split,
            target_n=targets[split],
            max_scan=max_scan[split],
            progress_every=args.progress_every,
            attempts=args.stream_attempts,
        )
        dataset_summary.update(z_summary)
        dataset_summary.update(save_split(split, hh[split], zbb[split], rng))

    with open(OUTDIR / "dataset_summary.json", "w") as f:
        json.dump(dataset_summary, f, indent=2)

    arrays = {}
    feature_names = FEATURE_NAMES
    samples = {}
    for split in ["train", "val", "test"]:
        data = np.load(OUTDIR / f"features_{split}.npz", allow_pickle=True)
        arrays[split] = (data["X"], data["y"])
        samples[split] = data["sample"]

    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    X_test, y_test = arrays["test"]

    print("Dataset sizes:")
    for split in ["train", "val", "test"]:
        X, y = arrays[split]
        print(f"  {split}: {X.shape}, HH={np.sum(y == 1)}, ZJetsTobb={np.sum(y == 0)}")

    plot_feature_overlays(X_train, y_train, feature_names)

    metrics_rows = []
    wp_rows = []
    score_map_test = {}
    score_map_train = {}
    auc_map = {}
    feature_importance = {}

    # Mass-only baseline: no training, threshold chosen on validation.
    mass_scores = {
        "train": mass_score(X_train, feature_names),
        "val": mass_score(X_val, feature_names),
        "test": mass_score(X_test, feature_names),
    }
    method = "mass_only_top2_btag_mbb"
    aucs = {
        "auc_train": float(roc_auc_score(y_train, mass_scores["train"])),
        "auc_val": float(roc_auc_score(y_val, mass_scores["val"])),
        "auc_test": float(roc_auc_score(y_test, mass_scores["test"])),
    }
    metrics_rows.append({"method": method, **aucs})
    wp_rows.extend(make_working_points(method, mass_scores["val"], y_val, mass_scores["test"], y_test))
    score_map_test[method] = mass_scores["test"]
    score_map_train[method] = mass_scores["train"]
    auc_map[method] = aucs

    for method, features_for_model in MODEL_FEATURES.items():
        print(f"Training {method} with features: {features_for_model}")
        _, scores, aucs, importance = train_bdt(method, features_for_model, arrays, feature_names)
        metrics_rows.append({"method": method, **aucs})
        wp_rows.extend(make_working_points(method, scores["val"], y_val, scores["test"], y_test))
        score_map_test[method] = scores["test"]
        score_map_train[method] = scores["train"]
        auc_map[method] = aucs
        feature_importance[method] = importance

    metrics_df = pd.DataFrame(metrics_rows)
    working_points_df = pd.DataFrame(wp_rows)
    metrics_df.to_csv(OUTDIR / "metrics.csv", index=False)
    working_points_df.to_csv(OUTDIR / "working_points.csv", index=False)

    np.savez(
        OUTDIR / "scores.npz",
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        sample_train=samples["train"],
        sample_val=samples["val"],
        sample_test=samples["test"],
        feature_names=np.array(feature_names),
        **{f"scores_train_{name}": score for name, score in score_map_train.items()},
        **{f"scores_test_{name}": score for name, score in score_map_test.items()},
    )

    plot_roc_curves(y_test, score_map_test, auc_map)
    plot_score_distributions(y_train, score_map_train, y_test, score_map_test)
    write_summary(dataset_summary, metrics_df, working_points_df, feature_importance)

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))
    print("\nWorking points evaluated on test:")
    print(working_points_df.to_string(index=False))
    print(f"\nSaved outputs to {OUTDIR}")
    print(f"Saved plots to {PLOT_DIR}")
    print("Saved markdown summary to RESULTS_ZBB_REJECTION.md")


if __name__ == "__main__":
    main()
