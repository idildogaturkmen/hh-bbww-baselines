'''
BDT training for HH4b analysis, using combined ggF+VBF signal and nominal backgrounds.
This script is intended to be run in the `scripts/delphes` directory, with the
HH4b parquet store available at the path specified by the HH4B_STORE environment variable.
The script will produce plots and tables in the outputs/plots/hh4b_bdt_YYYY_MM_DD and outputs/tables/hh4b_bdt_YYYY_MM_DD directories    

'''
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

store = Path(os.environ["HH4B_STORE"])

out_plot = Path("outputs/plots/hh4b_bdt_2026_07_08")
out_table = Path("outputs/tables/hh4b_bdt_2026_07_08")
out_plot.mkdir(parents=True, exist_ok=True)
out_table.mkdir(parents=True, exist_ok=True)

LUMI_FB = 450.0
PB_TO_FB = 1000.0

FEATURES = [
    "mbb1", "mbb2", "avg_mbb", "delta_mbb", "mhh",
    "drbb1", "drbb2",
    "j1_pt", "j2_pt", "j3_pt", "j4_pt",
    "r_hh", "pt_sum4", "pt_asym_12", "pt_asym_34",
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
        # support files named ..._merged_10k_event_summary.parquet etc.
        ev_candidates = list(store.glob(f"parquet/{tag}*event_summary.parquet"))
        cand_candidates = list(store.glob(f"parquet/{tag}*hh4b_candidates.parquet"))
        if len(ev_candidates) != 1 or len(cand_candidates) != 1:
            raise FileNotFoundError(f"Could not uniquely resolve {tag}: {ev_candidates}, {cand_candidates}")
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
    meta_path = store / "metadata" / "qcd_bbbb_iht_slice_scan_10000.csv"
    meta = pd.read_csv(meta_path)

    frames = []
    for _, row in meta.iterrows():
        tag = row["tag"]
        cand_path = store / "parquet" / f"{tag}_hh4b_candidates.parquet"
        cand = add_features(pd.read_parquet(cand_path))

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

frames = []

# Signals
frames.append(load_merged(
    "HH4b_ggf_hh4b_10k_transfer_cluster_84730223_merged_10k",
    "ggF HH4b",
    1,
    "signal",
))

frames.append(load_merged(
    "HH4b_vbf_hh4b_10k_transfer_cluster_3473183_merged_10k",
    "VBF HH4b",
    1,
    "signal",
))

# Nominal backgrounds
frames.append(load_qcd_iht_10k())

frames.append(load_merged(
    "ttbar_100k_merged",
    "inclusive ttbar",
    0,
    "nominal_background",
))

frames.append(load_merged(
    "zbbbb_presel_100k_merged",
    "Zbbbb",
    0,
    "nominal_background",
))

# Diagnostic background, excluded from nominal training/evaluation
diag_ttbb = load_merged(
    "ttbb_50k_merged",
    "ttbb diagnostic",
    0,
    "diagnostic_background",
)

df_nominal = pd.concat(frames, ignore_index=True)
df_diag = diag_ttbb.copy()

needed = FEATURES + ["y", "label", "role", "event_weight_pb"]
df_nominal = df_nominal[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
df_diag = df_diag[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()

# Split nominal samples into train/test.
# Stratify by y to preserve signal/background balance.
train_df, test_df = train_test_split(
    df_nominal,
    test_size=0.35,
    random_state=42,
    stratify=df_nominal["y"],
)

# Training weights:
# - use physical relative weights within each class,
# - but rescale total signal and background training weight to be equal.
train_df = train_df.copy()
train_df["train_weight"] = train_df["event_weight_pb"]

sum_sig = train_df.loc[train_df["y"] == 1, "train_weight"].sum()
sum_bkg = train_df.loc[train_df["y"] == 0, "train_weight"].sum()

train_df.loc[train_df["y"] == 1, "train_weight"] *= 0.5 / sum_sig
train_df.loc[train_df["y"] == 0, "train_weight"] *= 0.5 / sum_bkg

X_train = train_df[FEATURES]
y_train = train_df["y"]
w_train = train_df["train_weight"]

X_test = test_df[FEATURES]
y_test = test_df["y"]
w_test_pb = test_df["event_weight_pb"]

clf = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.04,
    max_leaf_nodes=15,
    l2_regularization=0.01,
    random_state=42,
)

clf.fit(X_train, y_train, sample_weight=w_train)

test_df = test_df.copy()
test_df["bdt_score"] = clf.predict_proba(X_test)[:, 1]

df_diag = df_diag.copy()
df_diag["bdt_score"] = clf.predict_proba(df_diag[FEATURES])[:, 1]

# Weighted ROC on nominal test set
fpr, tpr, thresholds = roc_curve(
    y_test,
    test_df["bdt_score"],
    sample_weight=w_test_pb,
)
roc_auc = auc(fpr, tpr)

roc_table = pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr,
    "threshold": thresholds,
})
roc_table.to_csv(out_table / "combined_bdt_weighted_roc.csv", index=False)

plt.figure()
plt.plot(tpr, 1.0 / np.maximum(fpr, 1e-12), label=f"Weighted AUC = {roc_auc:.4f}")
plt.yscale("log")
plt.xlabel("Signal efficiency")
plt.ylabel("Background rejection")
plt.title("HH4b combined BDT: weighted ROC")
plt.legend()
plt.tight_layout()
plt.savefig(out_plot / "combined_bdt_weighted_roc.png", dpi=180)
plt.close()

# Score distributions by process
plot_df = pd.concat([
    test_df.assign(dataset="nominal_test"),
    df_diag.assign(dataset="diagnostic_ttbb"),
], ignore_index=True)

plt.figure()
for label in plot_df["label"].unique():
    sub = plot_df[plot_df["label"] == label]
    plt.hist(
        sub["bdt_score"],
        bins=np.linspace(0, 1, 51),
        weights=sub["event_weight_pb"],
        histtype="step",
        label=label,
    )
plt.yscale("log")
plt.xlabel("BDT score")
plt.ylabel("Cross section [pb/bin]")
plt.title("BDT score distributions")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(out_plot / "combined_bdt_score_distributions_pb.png", dpi=180)
plt.close()

# Threshold/yield scan on nominal test set only
rows = []
for thr in np.linspace(0.0, 0.99, 100):
    selected = test_df["bdt_score"] > thr

    s_pb = test_df.loc[selected & (test_df["y"] == 1), "event_weight_pb"].sum()
    b_pb = test_df.loc[selected & (test_df["y"] == 0), "event_weight_pb"].sum()

    # Convert pb to expected events at LUMI_FB.
    s_ev = s_pb * PB_TO_FB * LUMI_FB
    b_ev = b_pb * PB_TO_FB * LUMI_FB

    z_stat = s_ev / np.sqrt(b_ev) if b_ev > 0 else np.nan
    z_10 = s_ev / np.sqrt(b_ev + (0.10 * b_ev)**2) if b_ev > 0 else np.nan
    z_20 = s_ev / np.sqrt(b_ev + (0.20 * b_ev)**2) if b_ev > 0 else np.nan

    rows.append({
        "threshold": thr,
        "s_pb_test": s_pb,
        "b_pb_test": b_pb,
        "s_events_450fb": s_ev,
        "b_events_450fb": b_ev,
        "s_over_sqrt_b": z_stat,
        "s_over_sqrt_b_10pct_bkg_syst": z_10,
        "s_over_sqrt_b_20pct_bkg_syst": z_20,
    })

scan = pd.DataFrame(rows)
scan.to_csv(out_table / "combined_bdt_threshold_scan_nominal_test.csv", index=False)

best = scan.sort_values("s_over_sqrt_b_10pct_bkg_syst", ascending=False).head(10)
best.to_csv(out_table / "combined_bdt_best_thresholds_10pct_syst.csv", index=False)

# Process yields at a few useful thresholds
yield_rows = []
for thr in [0.50, 0.70, 0.80, 0.90, 0.95, 0.98]:
    for label, sub in plot_df.groupby("label"):
        mask = sub["bdt_score"] > thr
        xsec_pb = sub.loc[mask, "event_weight_pb"].sum()
        yield_rows.append({
            "threshold": thr,
            "label": label,
            "selected_rows": int(mask.sum()),
            "selected_xsec_pb": xsec_pb,
            "selected_events_450fb": xsec_pb * PB_TO_FB * LUMI_FB,
        })

yield_df = pd.DataFrame(yield_rows)
yield_df.to_csv(out_table / "combined_bdt_process_yields_by_threshold.csv", index=False)
(out_table / "combined_bdt_process_yields_by_threshold.md").write_text(yield_df.to_markdown(index=False) + "\n")

# Permutation importance on test set, using unweighted scoring for quick diagnostic
perm = permutation_importance(
    clf,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=1,
)

imp = pd.DataFrame({
    "feature": FEATURES,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)

imp.to_csv(out_table / "combined_bdt_permutation_importance.csv", index=False)
(out_table / "combined_bdt_permutation_importance.md").write_text(imp.to_markdown(index=False) + "\n")

plt.figure()
top = imp.head(12).iloc[::-1]
plt.barh(top["feature"], top["importance_mean"])
plt.xlabel("Permutation importance")
plt.title("Combined BDT feature importance")
plt.tight_layout()
plt.savefig(out_plot / "combined_bdt_feature_importance.png", dpi=180)
plt.close()

summary_rows = []
for name, d in [
    ("train", train_df),
    ("test_nominal", test_df),
    ("diagnostic_ttbb", df_diag),
]:
    for label, sub in d.groupby("label"):
        summary_rows.append({
            "dataset": name,
            "label": label,
            "rows": len(sub),
            "xsec_pb_sum": sub["event_weight_pb"].sum(),
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(out_table / "combined_bdt_sample_summary.csv", index=False)
(out_table / "combined_bdt_sample_summary.md").write_text(summary.to_markdown(index=False) + "\n")

print("\nNominal weighted ROC AUC:", roc_auc)
print("\nBest thresholds by S/sqrt(B + (10%B)^2):")
print(best.to_string(index=False))

print("\nSample summary:")
print(summary.to_string(index=False))

print("\nWrote plots to:", out_plot)
print("Wrote tables to:", out_table)
