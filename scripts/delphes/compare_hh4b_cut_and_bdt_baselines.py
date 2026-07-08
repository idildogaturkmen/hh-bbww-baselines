'''
Combined BDT training for HH4b analysis, using ggF+VBF signal and nominal backgrounds.
This script is intended to be run in the `scripts/delphes` directory, with the HH4b parquet store available at the path specified by the HH4B_STORE environment variable.
The script will produce plots and tables in the outputs/plots/hh4b_bdt_YYYY_MM_DD and outputs/tables/hh4b_bdt_YYYY_MM_DD directories    
Compares the BDT performance to the cut-based baseline, and produces a table of yields at several thresholds.
Models compared:
- Cut-based baseline: mbb1 in [100, 150], mbb2 in [90, 140], r_hh < 40
- BDT baseline: trained on all features, with hyperparameters tuned for best s/sqrt(b) at 10% background systematic uncertainty 
- 
'''
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

store = Path(os.environ["HH4B_STORE"])

out_plot = Path("outputs/plots/hh4b_baseline_comparison_2026_07_08")
out_table = Path("outputs/tables/hh4b_baseline_comparison_2026_07_08")
out_plot.mkdir(parents=True, exist_ok=True)
out_table.mkdir(parents=True, exist_ok=True)

LUMI_FB = 450.0
PB_TO_FB = 1000.0

ALL_FEATURES = [
    "mbb1", "mbb2", "avg_mbb", "delta_mbb", "mhh",
    "drbb1", "drbb2",
    "j1_pt", "j2_pt", "j3_pt", "j4_pt",
    "r_hh", "pt_sum4", "pt_asym_12", "pt_asym_34",
]

TOPO_FEATURES = [
    "drbb1", "drbb2",
    "j1_pt", "j2_pt", "j3_pt", "j4_pt",
    "pt_sum4", "pt_asym_12", "pt_asym_34",
]

def add_features(df):
    df = df.copy()
    df["r_hh"] = np.sqrt((df["mbb1"] - 125.0)**2 + (df["mbb2"] - 120.0)**2)
    df["pt_sum4"] = df["j1_pt"] + df["j2_pt"] + df["j3_pt"] + df["j4_pt"]
    df["pt_asym_12"] = (df["j1_pt"] - df["j2_pt"]) / (df["j1_pt"] + df["j2_pt"] + 1e-6)
    df["pt_asym_34"] = (df["j3_pt"] - df["j4_pt"]) / (df["j3_pt"] + df["j4_pt"] + 1e-6)
    return df

def load_merged(tag, label, y, role):
    ev_path = store / "parquet" / f"{tag}_event_summary.parquet"
    cand_path = store / "parquet" / f"{tag}_hh4b_candidates.parquet"

    if not ev_path.exists():
        ev_candidates = list(store.glob(f"parquet/{tag}*event_summary.parquet"))
        cand_candidates = list(store.glob(f"parquet/{tag}*hh4b_candidates.parquet"))
        if len(ev_candidates) != 1 or len(cand_candidates) != 1:
            raise FileNotFoundError(f"Could not uniquely resolve {tag}")
        ev_path = ev_candidates[0]
        cand_path = cand_candidates[0]

    ev = pd.read_parquet(ev_path)
    cand = add_features(pd.read_parquet(cand_path))

    xsec_pb = float(ev["event_cross_section_pb"].median())
    n_generated = len(ev)
    event_weight_pb = xsec_pb / n_generated

    cand["label"] = label
    cand["y"] = y
    cand["role"] = role
    cand["xsec_pb"] = xsec_pb
    cand["n_generated"] = n_generated
    cand["event_weight_pb"] = event_weight_pb
    return cand

def load_qcd_iht_10k():
    meta = pd.read_csv(store / "metadata" / "qcd_bbbb_iht_slice_scan_10000.csv")
    frames = []

    for _, row in meta.iterrows():
        tag = row["tag"]
        cand = add_features(pd.read_parquet(store / "parquet" / f"{tag}_hh4b_candidates.parquet"))

        xsec_pb = float(row["xsec_pb"])
        n_generated = int(row["n_generated"])
        event_weight_pb = xsec_pb / n_generated

        cand["label"] = "QCD bbbb HT-sliced"
        cand["y"] = 0
        cand["role"] = "nominal_background"
        cand["xsec_pb"] = xsec_pb
        cand["n_generated"] = n_generated
        cand["event_weight_pb"] = event_weight_pb
        cand["qcd_slice"] = row.get("slice_label", tag)
        frames.append(cand)

    return pd.concat(frames, ignore_index=True)

frames = [
    load_merged("HH4b_ggf_hh4b_10k_transfer_cluster_84730223_merged_10k", "ggF HH4b", 1, "signal"),
    load_merged("HH4b_vbf_hh4b_10k_transfer_cluster_3473183_merged_10k", "VBF HH4b", 1, "signal"),
    load_qcd_iht_10k(),
    load_merged("ttbar_100k_merged", "inclusive ttbar", 0, "nominal_background"),
    load_merged("zbbbb_presel_100k_merged", "Zbbbb", 0, "nominal_background"),
]

df = pd.concat(frames, ignore_index=True)
df = df[ALL_FEATURES + ["label", "y", "role", "event_weight_pb"]].replace([np.inf, -np.inf], np.nan).dropna().copy()

train_df, test_df = train_test_split(
    df,
    test_size=0.35,
    random_state=42,
    stratify=df["y"],
)

def train_bdt(train_df, features):
    train_df = train_df.copy()
    train_df["train_weight"] = train_df["event_weight_pb"]

    sum_sig = train_df.loc[train_df["y"] == 1, "train_weight"].sum()
    sum_bkg = train_df.loc[train_df["y"] == 0, "train_weight"].sum()

    train_df.loc[train_df["y"] == 1, "train_weight"] *= 0.5 / sum_sig
    train_df.loc[train_df["y"] == 0, "train_weight"] *= 0.5 / sum_bkg

    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=0.01,
        random_state=42,
    )

    clf.fit(train_df[features], train_df["y"], sample_weight=train_df["train_weight"])
    return clf

def yields_for_mask(df, mask, name):
    rows = []
    for label, sub in df[mask].groupby("label"):
        xsec_pb = float(sub["event_weight_pb"].sum())
        rows.append({
            "selection": name,
            "label": label,
            "selected_rows": len(sub),
            "selected_xsec_pb": xsec_pb,
            "selected_events_450fb": xsec_pb * PB_TO_FB * LUMI_FB,
        })
    return rows

def summarize_selection(df, mask, name):
    s_pb = df.loc[mask & (df["y"] == 1), "event_weight_pb"].sum()
    b_pb = df.loc[mask & (df["y"] == 0), "event_weight_pb"].sum()
    s_ev = s_pb * PB_TO_FB * LUMI_FB
    b_ev = b_pb * PB_TO_FB * LUMI_FB

    return {
        "selection": name,
        "s_pb": s_pb,
        "b_pb": b_pb,
        "s_events_450fb": s_ev,
        "b_events_450fb": b_ev,
        "s_over_b": s_ev / b_ev if b_ev > 0 else np.nan,
        "s_over_sqrt_b": s_ev / np.sqrt(b_ev) if b_ev > 0 else np.nan,
        "s_over_sqrt_b_10pct_bkg_syst": s_ev / np.sqrt(b_ev + (0.10*b_ev)**2) if b_ev > 0 else np.nan,
        "s_over_sqrt_b_20pct_bkg_syst": s_ev / np.sqrt(b_ev + (0.20*b_ev)**2) if b_ev > 0 else np.nan,
    }

summary_rows = []
yield_rows = []

# Cut baselines on same held-out test set
cut_defs = {
    "cut_rhh_lt_30": test_df["r_hh"] < 30,
    "cut_rhh_lt_55": test_df["r_hh"] < 55,
    "cut_rhh_lt_30_mhh_gt_400": (test_df["r_hh"] < 30) & (test_df["mhh"] > 400),
    "cut_rhh_lt_55_mhh_gt_400": (test_df["r_hh"] < 55) & (test_df["mhh"] > 400),
}

for name, mask in cut_defs.items():
    summary_rows.append(summarize_selection(test_df, mask, name))
    yield_rows.extend(yields_for_mask(test_df, mask, name))

# BDTs
models = {
    "bdt_all_features": (train_bdt(train_df, ALL_FEATURES), ALL_FEATURES),
    "bdt_topology_only": (train_bdt(train_df, TOPO_FEATURES), TOPO_FEATURES),
}

roc_rows = []

for model_name, (clf, features) in models.items():
    scores = clf.predict_proba(test_df[features])[:, 1]

    fpr, tpr, thresholds = roc_curve(
        test_df["y"],
        scores,
        sample_weight=test_df["event_weight_pb"],
    )
    roc_auc = auc(fpr, tpr)

    roc_rows.append({"model": model_name, "weighted_auc": roc_auc})

    scan_rows = []
    for thr in np.linspace(0, 0.99, 100):
        mask = scores > thr
        row = summarize_selection(test_df, mask, f"{model_name}_thr_{thr:.2f}")
        row["model"] = model_name
        row["threshold"] = thr
        scan_rows.append(row)

    scan = pd.DataFrame(scan_rows)
    scan.to_csv(out_table / f"{model_name}_threshold_scan.csv", index=False)

    best = scan.sort_values("s_over_sqrt_b_10pct_bkg_syst", ascending=False).head(1).iloc[0]
    best_thr = float(best["threshold"])
    best_mask = scores > best_thr

    summary_rows.append(summarize_selection(test_df, best_mask, f"{model_name}_best10pct_thr_{best_thr:.2f}"))
    yield_rows.extend(yields_for_mask(test_df, best_mask, f"{model_name}_best10pct_thr_{best_thr:.2f}"))

    plt.figure()
    plt.plot(tpr, 1.0 / np.maximum(fpr, 1e-12), label=f"{model_name}, AUC={roc_auc:.3f}")
    plt.yscale("log")
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background rejection")
    plt.title(model_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot / f"{model_name}_roc.png", dpi=180)
    plt.close()

    plt.figure()
    tmp = test_df.copy()
    tmp["score"] = scores
    for label, sub in tmp.groupby("label"):
        plt.hist(
            sub["score"],
            bins=np.linspace(0, 1, 51),
            weights=sub["event_weight_pb"],
            histtype="step",
            label=label,
        )
    plt.yscale("log")
    plt.xlabel("BDT score")
    plt.ylabel("Cross section [pb/bin]")
    plt.title(model_name)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_plot / f"{model_name}_score_distributions.png", dpi=180)
    plt.close()

summary = pd.DataFrame(summary_rows)
yields = pd.DataFrame(yield_rows)
rocs = pd.DataFrame(roc_rows)

summary.to_csv(out_table / "cut_vs_bdt_summary.csv", index=False)
summary.to_markdown(out_table / "cut_vs_bdt_summary.md", index=False)

yields.to_csv(out_table / "cut_vs_bdt_process_yields.csv", index=False)
yields.to_markdown(out_table / "cut_vs_bdt_process_yields.md", index=False)

rocs.to_csv(out_table / "cut_vs_bdt_auc.csv", index=False)
rocs.to_markdown(out_table / "cut_vs_bdt_auc.md", index=False)

print("\nWeighted AUCs:")
print(rocs.to_string(index=False))

print("\nCut vs BDT summary:")
print(summary.to_string(index=False))

print("\nProcess yields:")
print(yields.to_string(index=False))

print("\nWrote:", out_table)
print("Wrote:", out_plot)
