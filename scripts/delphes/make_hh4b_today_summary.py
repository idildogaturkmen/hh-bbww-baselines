#!/usr/bin/env python3
from pathlib import Path
import os
import json
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

store = Path(os.environ["HH4B_STORE"])
plot_dir = store / "plots" / "harvey_update_2026_07_07"
meta_dir = store / "metadata"
plot_dir.mkdir(parents=True, exist_ok=True)
meta_dir.mkdir(parents=True, exist_ok=True)

status_path = meta_dir / "hh4b_signal_background_10k_status.csv"
train_summary_path = meta_dir / "hh4b_candidate_training_10k_v0_summary.csv"
train_table_path = store / "ml" / "hh4b_candidate_training_10k_v0.parquet"
bdt_report_path = store / "ml" / "hh4b_candidate_bdt_10k_v0_report.json"
bdt_scores_path = store / "ml" / "hh4b_candidate_bdt_10k_v0_scores.parquet"

status = pd.read_csv(status_path)
train_summary = pd.read_csv(train_summary_path)
train = pd.read_parquet(train_table_path)

with open(bdt_report_path) as f:
    bdt_report = json.load(f)

scores = pd.read_parquet(bdt_scores_path)

# ----------------------------------------------------------------------
# Helpful names
# ----------------------------------------------------------------------
pretty = {
    "VBF_HH4b_10k": "VBF HH4b",
    "ggF_HEFT_HH4b_10k": "ggF HEFT HH4b",
    "QCD_bbbb_presel_10k": "QCD bbbb",
    "Zbbbb_presel_10k": "Zbbbb",
    "ttbar_10k": "ttbar",
    "vbf_hh4b": "VBF HH4b",
    "ggf_heft_hh4b": "ggF HEFT HH4b",
    "qcd_bbbb_presel": "QCD bbbb",
    "zbbbb_presel": "Zbbbb",
    "ttbar": "ttbar",
}

def prettify_series(s):
    return s.map(lambda x: pretty.get(x, x))

# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
status_out = meta_dir / "hh4b_today_status_table.csv"
train_out = meta_dir / "hh4b_today_candidate_table.csv"

status.to_csv(status_out, index=False)
train_summary.to_csv(train_out, index=False)

# Markdown tables
status_md = status.copy()
status_md["sample"] = prettify_series(status_md["sample"])
train_md = train_summary.copy()
train_md["process"] = prettify_series(train_md["process"])

status_md_path = meta_dir / "hh4b_today_status_table.md"
train_md_path = meta_dir / "hh4b_today_candidate_table.md"
status_md_path.write_text(status_md.to_markdown(index=False) + "\n")
train_md_path.write_text(train_md.to_markdown(index=False) + "\n")

# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------
def savefig(name):
    out = plot_dir / name
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print("Wrote:", out)

def hist_by_process(df, column, filename, bins=50, density=True, title=None, xlabel=None):
    plt.figure()
    for proc, sub in df.groupby("process"):
        x = sub[column].dropna()
        if len(x) == 0:
            continue
        plt.hist(x, bins=bins, histtype="step", density=density, label=pretty.get(proc, proc))
    plt.xlabel(xlabel or column)
    plt.ylabel("Normalized candidates" if density else "Candidates")
    if title:
        plt.title(title)
    plt.legend(fontsize=8)
    savefig(filename)

def hist_sig_bkg(df, column, filename, bins=50, density=True, title=None, xlabel=None):
    plt.figure()
    for label, sub in df.groupby("label"):
        name = "signal" if label == 1 else "background"
        x = sub[column].dropna()
        if len(x) == 0:
            continue
        plt.hist(x, bins=bins, histtype="step", density=density, label=name)
    plt.xlabel(xlabel or column)
    plt.ylabel("Normalized candidates" if density else "Candidates")
    if title:
        plt.title(title)
    plt.legend()
    savefig(filename)

# ----------------------------------------------------------------------
# Candidate yields
# ----------------------------------------------------------------------
yield_df = train.groupby("process").size().reset_index(name="candidate_rows")
yield_df["process_pretty"] = prettify_series(yield_df["process"])
yield_df = yield_df.sort_values("candidate_rows", ascending=False)

plt.figure()
plt.bar(yield_df["process_pretty"], yield_df["candidate_rows"])
plt.ylabel("HH4b candidate rows")
plt.xticks(rotation=30, ha="right")
plt.title("Candidate yields after four selected b-tagged jets")
savefig("candidate_rows_by_process.png")

# ----------------------------------------------------------------------
# Core physics distributions
# ----------------------------------------------------------------------
for col, label in [
    ("mbb1", "m(bb) candidate 1 [GeV]"),
    ("mbb2", "m(bb) candidate 2 [GeV]"),
    ("avg_mbb", "average m(bb) [GeV]"),
    ("delta_mbb", "|m(bb1)-m(bb2)| [GeV]"),
    ("mhh", "m(HH candidate) [GeV]"),
    ("drbb1", "ΔR(bb) candidate 1"),
    ("drbb2", "ΔR(bb) candidate 2"),
    ("j1_pt", "leading selected b-jet pT [GeV]"),
    ("j2_pt", "second selected b-jet pT [GeV]"),
]:
    if col in train.columns:
        hist_by_process(
            train,
            col,
            f"{col}_by_process.png",
            bins=50,
            title=f"{label} by process",
            xlabel=label,
        )
        hist_sig_bkg(
            train,
            col,
            f"{col}_signal_vs_background.png",
            bins=50,
            title=f"{label}: signal vs background",
            xlabel=label,
        )

# ----------------------------------------------------------------------
# BDT score plots
# ----------------------------------------------------------------------
plt.figure()
for label, sub in scores.groupby("label"):
    name = "signal" if label == 1 else "background"
    plt.hist(sub["bdt_score"], bins=50, histtype="step", density=True, label=name)
plt.xlabel("BDT score")
plt.ylabel("Normalized candidates")
plt.title("BDT score: signal vs background")
plt.legend()
savefig("bdt_score_signal_vs_background.png")

plt.figure()
for proc, sub in scores.groupby("process"):
    plt.hist(sub["bdt_score"], bins=50, histtype="step", density=True, label=pretty.get(proc, proc))
plt.xlabel("BDT score")
plt.ylabel("Normalized candidates")
plt.title("BDT score by process")
plt.legend(fontsize=8)
savefig("bdt_score_by_process.png")

fpr, tpr, thresholds = roc_curve(scores["label"], scores["bdt_score"])
auc = roc_auc_score(scores["label"], scores["bdt_score"])

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Background efficiency")
plt.ylabel("Signal efficiency")
plt.title("Candidate-level BDT ROC curve")
plt.legend()
savefig("bdt_roc_curve.png")

# ----------------------------------------------------------------------
# Simple weighted-yield note
# ----------------------------------------------------------------------
weighted_rows = []
for proc, sub in train.groupby("process"):
    xsec = sub["xsec_pb"].median()
    n_gen = sub["n_generated"].median()
    n_cand = len(sub)
    weighted_rows.append({
        "process": pretty.get(proc, proc),
        "candidate_rows": n_cand,
        "xsec_pb": xsec,
        "n_generated": n_gen,
        "xsec_over_ngen_pb": xsec / n_gen,
        "sum_candidate_weight_pb": n_cand * xsec / n_gen,
    })

weighted = pd.DataFrame(weighted_rows)
weighted_path = meta_dir / "hh4b_today_weighted_candidate_yields.csv"
weighted_md_path = meta_dir / "hh4b_today_weighted_candidate_yields.md"
weighted.to_csv(weighted_path, index=False)
weighted_md_path.write_text(weighted.to_markdown(index=False) + "\n")

# ----------------------------------------------------------------------
# Markdown summary for Harvey
# ----------------------------------------------------------------------
summary_path = meta_dir / "hh4b_today_harvey_update.md"

summary_text = f"""
# HH→4b Delphes + ML progress summary — 2026-07-07

## Bottom-line messages

1. I now have a working LPC workflow for generating HH→4b signal and background samples through MG5/Pythia8/Delphes.
2. I validated two 10k signal samples: VBF HH→4b SM and ggF HH→4b HEFT/effective approximation.
3. I validated three 10k background samples: QCD bbbb, Zbbbb, and ttbar.
4. The candidate-level dataset contains {len(train)} HH4b candidate rows: {(train['label'] == 1).sum()} signal and {(train['label'] == 0).sum()} background.
5. A first candidate-level BDT sanity check gives an unweighted AUC of {bdt_report['auc_unweighted']:.3f}.
6. This is not yet a final analysis result; it is a validation milestone showing that the simulation, reconstruction, feature extraction, and first ML baseline are functioning.

## Current 10k signal/background samples

{status_md.to_markdown(index=False)}

## Candidate-level training table

{train_md.to_markdown(index=False)}

## First BDT sanity check

- Input table: `{bdt_report['input']}`
- Total candidates: {bdt_report['n_total']}
- Train candidates: {bdt_report['n_train']}
- Test candidates: {bdt_report['n_test']}
- Signal rows in test: {bdt_report['test_signal_rows']}
- Background rows in test: {bdt_report['test_background_rows']}
- Unweighted AUC: {bdt_report['auc_unweighted']:.3f}

Features used:
{', '.join(bdt_report['features'])}

## Caveats

- The ggF sample used for bulk production is HEFT/effective ggF, not full SM loop-induced ggF.
- A 20-event SM loop-induced ggF smoke test has passed, but it is not yet used for bulk production.
- QCD bbbb and Zbbbb samples are preselected/fiducial heavy-flavor samples, not inclusive QCD.
- ttbar has very low four-b-tag candidate yield at 10k, as expected for ordinary ttbar.
- Current BDT is only a candidate-level sanity check, not a final optimized classifier.
- For larger production, ROOT and HepMC cannot be kept for every event due to storage limits. Future 100k/1M production should keep compact parquet/features and only retain small ROOT validation subsets.

## Useful plots

Plots are saved in:

`{plot_dir}`

Recommended plots to show first:
- `candidate_rows_by_process.png`
- `avg_mbb_signal_vs_background.png`
- `delta_mbb_signal_vs_background.png`
- `mhh_signal_vs_background.png`
- `bdt_score_signal_vs_background.png`
- `bdt_roc_curve.png`

## Next steps

1. Produce storage-safe 100k samples using shard-level parquet output.
2. Add event-level features and possibly jet-level arrays, not only HH candidate features.
3. Compare cut baseline, BDT, DNN, and SPA-Net-style assignment-aware models.
4. Use simulated backgrounds to test ABCD closure before claiming any data-driven background estimate.
5. Keep SM loop-induced ggF as a validation/shape cross-check unless it scales efficiently.
"""

summary_path.write_text(textwrap.dedent(summary_text).strip() + "\n")

print("Wrote:", status_out)
print("Wrote:", train_out)
print("Wrote:", weighted_path)
print("Wrote:", summary_path)
print("Plot directory:", plot_dir)
