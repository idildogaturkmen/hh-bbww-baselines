"""
Detector/reconstruction diagnostic for the low truth-matched H->bb mass.

This script asks whether the reconstructed H->bb mass shift is correlated with:
  - low matched b-jet pT,
  - detector eta region,
  - reconstructed/gen pT response of the matched b jets.

It uses the same practical truth-matching convention as the existing HH diagnostics:
  - select status-23 generator b quarks,
  - match each to the nearest retained AK4 jet,
  - require DeltaR < 0.4 and two distinct matched jets.

This is not a new classifier. It is a detector/reconstruction cross-check.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SNAPSHOT_DIR = Path("outputs/dataset_cache/collide_1m/datasets--fastmachinelearning--collide-1m/snapshots")
OUTDIR = Path("outputs/hbb_detector_response")
PLOT_DIR = Path("outputs/plots/hbb_detector_response")
SUMMARY_MD = Path("RESULTS_HBB_DETECTOR_RESPONSE.md")

M_H = 125.0
MAX_JETS = 12
MATCH_DR = 0.4
ETA_CENTRAL = 1.5

JET_COLUMNS = [
    "FullReco_JetAK4_PT",
    "FullReco_JetAK4_Eta",
    "FullReco_JetAK4_Phi",
    "FullReco_JetAK4_Mass",
    "FullReco_JetAK4_BTag",
    "FullReco_JetAK4_BTagPhys",
]

GEN_COLUMNS = [
    "FullReco_GenPart_PT",
    "FullReco_GenPart_Eta",
    "FullReco_GenPart_Phi",
    "FullReco_GenPart_PID",
    "FullReco_GenPart_Status",
]

COLUMNS = JET_COLUMNS + GEN_COLUMNS

DEFAULT_MIN_PT_BINS = np.array([20, 30, 40, 50, 65, 80, 95, 110, 125, 140, 160], dtype=float)
DEFAULT_RECO_PT_BINS = np.array([20, 30, 40, 50, 65, 80, 100, 125, 160, 220, 320], dtype=float)
DEFAULT_ABS_ETA_BINS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=float)
MIN_PT_CUTS = [20, 30, 40, 50, 70, 90, 110]


def ensure_dirs():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def find_hh_files():
    files = sorted(SNAPSHOT_DIR.glob("*/HH_bbWW/*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"Could not find cached HH_bbWW parquet files under {SNAPSHOT_DIR}. "
            "Run the HH preprocessing/download step first."
        )
    return files


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2.0 * math.pi
    while dphi <= -math.pi:
        dphi += 2.0 * math.pi
    return dphi


def delta_r_eta_phi(eta1, phi1, eta2, phi2):
    return math.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2)


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
    return float(np.sqrt(max(m2_tot, 0.0)))


def event_from_batch(columns, row):
    return {name: columns[name][row] for name in COLUMNS}


def build_jets(event, max_jets=MAX_JETS):
    n_jets_raw = len(event["FullReco_JetAK4_PT"])
    n_jets = min(n_jets_raw, max_jets)
    jets = np.zeros((n_jets, 6), dtype=np.float32)
    for idx, col in enumerate(JET_COLUMNS):
        jets[:, idx] = event[col][:n_jets]
    return jets, n_jets_raw


def find_hbb_bquarks(event):
    """
    Practical COLLIDE-1M truth-labeling convention:
    choose the two highest-pT status-23 b/anti-b quarks.
    """
    pids = event["FullReco_GenPart_PID"]
    statuses = event["FullReco_GenPart_Status"]
    pts = event["FullReco_GenPart_PT"]

    candidates = []
    for idx, pid in enumerate(pids):
        if abs(int(pid)) == 5 and int(statuses[idx]) == 23:
            candidates.append(idx)

    return sorted(candidates, key=lambda idx: pts[idx], reverse=True)[:2]


def match_bquarks_to_retained_jets(event, b_indices, max_dr=MATCH_DR, max_jets=MAX_JETS):
    n_jets = min(len(event["FullReco_JetAK4_PT"]), max_jets)
    matches = []

    for bidx in b_indices:
        b_eta = float(event["FullReco_GenPart_Eta"][bidx])
        b_phi = float(event["FullReco_GenPart_Phi"][bidx])

        best_j = -1
        best_dr = float("inf")

        for jidx in range(n_jets):
            dr = delta_r_eta_phi(
                b_eta,
                b_phi,
                float(event["FullReco_JetAK4_Eta"][jidx]),
                float(event["FullReco_JetAK4_Phi"][jidx]),
            )
            if dr < best_dr:
                best_j = jidx
                best_dr = dr

        matches.append(
            {
                "bidx": int(bidx),
                "jet_idx": int(best_j),
                "deltaR": float(best_dr),
                "matched": bool(best_dr < max_dr),
            }
        )

    return matches


def eta_category(abs_eta0, abs_eta1, eta_central=ETA_CENTRAL):
    n_forward = int(abs_eta0 >= eta_central) + int(abs_eta1 >= eta_central)
    if n_forward == 0:
        return f"both_central_abs_eta_lt_{eta_central}"
    if n_forward == 1:
        return f"one_forward_abs_eta_ge_{eta_central}"
    return f"both_forward_abs_eta_ge_{eta_central}"


def process_event(event, file_label, local_event, global_event):
    b_indices = find_hbb_bquarks(event)
    if len(b_indices) != 2:
        return None, "no_two_status23_bquarks"

    matches = match_bquarks_to_retained_jets(event, b_indices)
    if len(matches) != 2 or not all(match["matched"] for match in matches):
        return None, "unmatched_bquark"

    jet_indices = [match["jet_idx"] for match in matches]
    if jet_indices[0] == jet_indices[1]:
        return None, "same_matched_jet"

    jets, n_jets_raw = build_jets(event)
    if max(jet_indices) >= len(jets):
        return None, "matched_beyond_max_jets"

    j0, j1 = jet_indices
    b0, b1 = b_indices

    jet0 = jets[j0]
    jet1 = jets[j1]

    mbb = invariant_mass(jet0, jet1)
    drbb = delta_r_eta_phi(float(jet0[1]), float(jet0[2]), float(jet1[1]), float(jet1[2]))

    gen0_pt = float(event["FullReco_GenPart_PT"][b0])
    gen1_pt = float(event["FullReco_GenPart_PT"][b1])
    gen0_eta = float(event["FullReco_GenPart_Eta"][b0])
    gen1_eta = float(event["FullReco_GenPart_Eta"][b1])
    gen0_phi = float(event["FullReco_GenPart_Phi"][b0])
    gen1_phi = float(event["FullReco_GenPart_Phi"][b1])

    reco0_pt = float(jet0[0])
    reco1_pt = float(jet1[0])
    reco0_eta = float(jet0[1])
    reco1_eta = float(jet1[1])
    reco0_phi = float(jet0[2])
    reco1_phi = float(jet1[2])
    reco0_mass = float(jet0[3])
    reco1_mass = float(jet1[3])

    response0 = reco0_pt / gen0_pt if gen0_pt > 0 else float("nan")
    response1 = reco1_pt / gen1_pt if gen1_pt > 0 else float("nan")

    lower_reco_pt = min(reco0_pt, reco1_pt)
    higher_reco_pt = max(reco0_pt, reco1_pt)

    if reco0_pt <= reco1_pt:
        lower_response = response0
        higher_response = response1
        lower_abs_eta = abs(reco0_eta)
        higher_abs_eta = abs(reco1_eta)
    else:
        lower_response = response1
        higher_response = response0
        lower_abs_eta = abs(reco1_eta)
        higher_abs_eta = abs(reco0_eta)

    max_abs_eta = max(abs(reco0_eta), abs(reco1_eta))
    both_central = bool(abs(reco0_eta) < ETA_CENTRAL and abs(reco1_eta) < ETA_CENTRAL)

    return {
        "file": file_label,
        "local_event": int(local_event),
        "global_event": int(global_event),
        "n_jets_raw": int(n_jets_raw),
        "n_jets_retained": int(len(jets)),

        "b0_gen_idx": int(b0),
        "b1_gen_idx": int(b1),
        "b0_gen_pt": gen0_pt,
        "b1_gen_pt": gen1_pt,
        "b0_gen_eta": gen0_eta,
        "b1_gen_eta": gen1_eta,
        "b0_gen_phi": gen0_phi,
        "b1_gen_phi": gen1_phi,

        "b0_jet_idx": int(j0),
        "b1_jet_idx": int(j1),
        "b0_match_deltaR": float(matches[0]["deltaR"]),
        "b1_match_deltaR": float(matches[1]["deltaR"]),

        "b0_reco_pt": reco0_pt,
        "b1_reco_pt": reco1_pt,
        "b0_reco_eta": reco0_eta,
        "b1_reco_eta": reco1_eta,
        "b0_reco_phi": reco0_phi,
        "b1_reco_phi": reco1_phi,
        "b0_reco_mass": reco0_mass,
        "b1_reco_mass": reco1_mass,
        "b0_response_reco_pt_over_gen_pt": float(response0),
        "b1_response_reco_pt_over_gen_pt": float(response1),

        "matched_mbb": float(mbb),
        "matched_deltaR_bb": float(drbb),
        "lower_matched_jet_pt": float(lower_reco_pt),
        "higher_matched_jet_pt": float(higher_reco_pt),
        "lower_response_reco_pt_over_gen_pt": float(lower_response),
        "higher_response_reco_pt_over_gen_pt": float(higher_response),
        "average_response_reco_pt_over_gen_pt": float(np.nanmean([response0, response1])),
        "lower_abs_eta": float(lower_abs_eta),
        "higher_abs_eta": float(higher_abs_eta),
        "max_abs_eta": float(max_abs_eta),
        "both_central_abs_eta_lt_1p5": both_central,
        "eta_category": eta_category(abs(reco0_eta), abs(reco1_eta)),
    }, "usable"


def scan_hh_events(args):
    rows = []
    counts = {
        "scanned": 0,
        "usable": 0,
        "no_two_status23_bquarks": 0,
        "unmatched_bquark": 0,
        "same_matched_jet": 0,
        "matched_beyond_max_jets": 0,
    }

    files = find_hh_files()[: args.n_files]
    global_event = 0

    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        file_label = str(file_path.relative_to(SNAPSHOT_DIR))
        local_event = 0
        print(f"Scanning {file_label}", flush=True)

        for batch in parquet_file.iter_batches(batch_size=args.batch_size, columns=COLUMNS):
            columns = batch.to_pydict()

            for row_idx in range(batch.num_rows):
                if args.max_events is not None and counts["scanned"] >= args.max_events:
                    return pd.DataFrame(rows), counts

                event = event_from_batch(columns, row_idx)
                event_row, status = process_event(event, file_label, local_event, global_event)

                counts["scanned"] += 1
                if event_row is not None:
                    rows.append(event_row)
                    counts["usable"] += 1
                else:
                    counts[status] = counts.get(status, 0) + 1

                if counts["scanned"] % args.progress_every == 0:
                    print(f"scanned={counts['scanned']} usable={counts['usable']}", flush=True)

                local_event += 1
                global_event += 1

    return pd.DataFrame(rows), counts


def bootstrap_median_interval(values, rng, n_bootstrap):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1 or n_bootstrap <= 0:
        med = float(np.median(values))
        return med, med, med

    medians = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = values[rng.integers(0, n, size=n)]
        medians[i] = np.median(sample)

    return (
        float(np.percentile(medians, 16)),
        float(np.median(medians)),
        float(np.percentile(medians, 84)),
    )


def profile_mbb_vs_min_pt(df, args):
    rng = np.random.default_rng(args.seed)
    rows = []

    for lo, hi in zip(DEFAULT_MIN_PT_BINS[:-1], DEFAULT_MIN_PT_BINS[1:]):
        part = df[(df["lower_matched_jet_pt"] >= lo) & (df["lower_matched_jet_pt"] < hi)]
        values = part["matched_mbb"].to_numpy(dtype=float)
        n = len(values)

        if n == 0:
            rows.append(
                {
                    "bin_low": lo,
                    "bin_high": hi,
                    "bin_center": 0.5 * (lo + hi),
                    "n_events": 0,
                    "median_mbb": float("nan"),
                    "mean_mbb": float("nan"),
                    "p16_mbb": float("nan"),
                    "p84_mbb": float("nan"),
                    "bootstrap_median_p16": float("nan"),
                    "bootstrap_median_p50": float("nan"),
                    "bootstrap_median_p84": float("nan"),
                    "median_err_low": float("nan"),
                    "median_err_high": float("nan"),
                }
            )
            continue

        boot16, boot50, boot84 = bootstrap_median_interval(values, rng, args.bootstrap)
        med = float(np.median(values))

        rows.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "bin_center": 0.5 * (lo + hi),
                "n_events": int(n),
                "median_mbb": med,
                "mean_mbb": float(np.mean(values)),
                "p16_mbb": float(np.percentile(values, 16)),
                "p84_mbb": float(np.percentile(values, 84)),
                "bootstrap_median_p16": boot16,
                "bootstrap_median_p50": boot50,
                "bootstrap_median_p84": boot84,
                "median_err_low": med - boot16,
                "median_err_high": boot84 - med,
            }
        )

    return pd.DataFrame(rows)


def response_profile(values_x, values_y, bins, x_name):
    rows = []
    x = np.asarray(values_x, dtype=float)
    y = np.asarray(values_y, dtype=float)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x >= lo) & (x < hi) & np.isfinite(y)
        vals = y[mask]
        rows.append(
            {
                f"{x_name}_low": float(lo),
                f"{x_name}_high": float(hi),
                f"{x_name}_center": float(0.5 * (lo + hi)),
                "n_jets": int(len(vals)),
                "median_response": float(np.median(vals)) if len(vals) else float("nan"),
                "mean_response": float(np.mean(vals)) if len(vals) else float("nan"),
                "p16_response": float(np.percentile(vals, 16)) if len(vals) else float("nan"),
                "p84_response": float(np.percentile(vals, 84)) if len(vals) else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def build_perjet_df(event_df):
    rows = []
    for _, row in event_df.iterrows():
        for idx in [0, 1]:
            rows.append(
                {
                    "global_event": int(row["global_event"]),
                    "b_index": idx,
                    "gen_pt": float(row[f"b{idx}_gen_pt"]),
                    "gen_eta": float(row[f"b{idx}_gen_eta"]),
                    "reco_pt": float(row[f"b{idx}_reco_pt"]),
                    "reco_eta": float(row[f"b{idx}_reco_eta"]),
                    "abs_reco_eta": float(abs(row[f"b{idx}_reco_eta"])),
                    "response_reco_pt_over_gen_pt": float(row[f"b{idx}_response_reco_pt_over_gen_pt"]),
                    "match_deltaR": float(row[f"b{idx}_match_deltaR"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_values(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p16": float("nan"),
            "p84": float("nan"),
            "frac_90_140": float("nan"),
            "frac_100_150": float("nan"),
        }

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p16": float(np.percentile(values, 16)),
        "p84": float(np.percentile(values, 84)),
        "frac_90_140": float(np.mean((values >= 90.0) & (values <= 140.0))),
        "frac_100_150": float(np.mean((values >= 100.0) & (values <= 150.0))),
    }


def eta_category_summary(df):
    rows = []
    for category, part in df.groupby("eta_category", sort=True):
        rows.append({"eta_category": category, **summarize_values(part["matched_mbb"].to_numpy())})
    return pd.DataFrame(rows)


def min_pt_cut_summary(df):
    rows = []
    n_total = len(df)
    for cut in MIN_PT_CUTS:
        part = df[df["lower_matched_jet_pt"] >= cut]
        row = {
            "min_lower_matched_jet_pt_cut": float(cut),
            "signal_retention": float(len(part) / n_total) if n_total else float("nan"),
        }
        row.update(summarize_values(part["matched_mbb"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_mbb_profile(profile_df):
    sub = profile_df[profile_df["n_events"] > 0].copy()

    plt.figure(figsize=(8.0, 5.4))
    yerr = np.vstack([sub["median_err_low"].to_numpy(), sub["median_err_high"].to_numpy()])
    plt.errorbar(
        sub["bin_center"],
        sub["median_mbb"],
        yerr=yerr,
        marker="o",
        capsize=3,
        linewidth=1.5,
        label="median with bootstrap 16-84%",
    )
    plt.fill_between(
        sub["bin_center"],
        sub["p16_mbb"],
        sub["p84_mbb"],
        alpha=0.2,
        label="event 16-84%",
    )
    plt.axhline(M_H, linestyle="--", linewidth=1.2, label="125 GeV")
    plt.xlabel("Lower matched-jet pT [GeV]")
    plt.ylabel("Truth-matched reconstructed m_bb [GeV]")
    plt.title("Matched m_bb vs lower matched-jet pT")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "matched_mbb_profile_vs_min_pt_with_errors.png", dpi=200)
    plt.close()


def plot_counts(profile_df):
    sub = profile_df.copy()
    widths = sub["bin_high"] - sub["bin_low"]
    plt.figure(figsize=(7.5, 4.8))
    plt.bar(sub["bin_center"], sub["n_events"], width=0.85 * widths, align="center")
    plt.xlabel("Lower matched-jet pT bin [GeV]")
    plt.ylabel("Number of usable HH events")
    plt.title("Event counts in lower matched-jet pT bins")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "n_events_vs_min_pt_bin.png", dpi=200)
    plt.close()


def plot_response_profile(profile_df, x_center_col, filename, title, xlabel):
    sub = profile_df[profile_df["n_jets"] > 0].copy()
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(sub[x_center_col], sub["median_response"], marker="o", label="median")
    plt.fill_between(
        sub[x_center_col],
        sub["p16_response"],
        sub["p84_response"],
        alpha=0.2,
        label="16-84%",
    )
    plt.axhline(1.0, linestyle="--", linewidth=1.0, label="response = 1")
    plt.xlabel(xlabel)
    plt.ylabel("Matched jet pT / gen b pT")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200)
    plt.close()


def plot_eta_box(df):
    groups = []
    labels = []
    for category, part in df.groupby("eta_category", sort=True):
        groups.append(part["matched_mbb"].to_numpy())
        labels.append(category.replace("_", " "))

    plt.figure(figsize=(8.2, 5.2))
    plt.boxplot(groups, tick_labels=labels, showfliers=False)
    plt.axhline(M_H, linestyle="--", linewidth=1.0, label="125 GeV")
    plt.ylabel("Truth-matched reconstructed m_bb [GeV]")
    plt.title("Matched m_bb by matched-jet eta category")
    plt.xticks(rotation=20, ha="right")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "matched_mbb_by_eta_category.png", dpi=200)
    plt.close()


def plot_min_pt_cuts(cut_df):
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(cut_df["min_lower_matched_jet_pt_cut"], cut_df["median"], marker="o", label="median m_bb")
    plt.plot(cut_df["min_lower_matched_jet_pt_cut"], cut_df["mean"], marker="s", label="mean m_bb")
    plt.axhline(M_H, linestyle="--", linewidth=1.0, label="125 GeV")
    plt.xlabel("Cut on lower matched-jet pT [GeV]")
    plt.ylabel("Matched m_bb [GeV]")
    plt.title("Matched m_bb after lower matched-jet pT cuts")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "matched_mbb_by_min_pt_cut.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(cut_df["min_lower_matched_jet_pt_cut"], cut_df["signal_retention"], marker="o")
    plt.xlabel("Cut on lower matched-jet pT [GeV]")
    plt.ylabel("Fraction of usable HH events retained")
    plt.ylim(0, 1.05)
    plt.title("Signal retention after lower matched-jet pT cuts")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "signal_retention_by_min_pt_cut.png", dpi=200)
    plt.close()


def write_markdown(counts, overall, profile_df, eta_df, cut_df, args):
    best_profile = profile_df[profile_df["n_events"] > 0].copy()
    profile_rows = []
    for _, row in best_profile.iterrows():
        profile_rows.append(
            f"| {row['bin_low']:.0f}-{row['bin_high']:.0f} | {int(row['n_events'])} | "
            f"{row['median_mbb']:.2f} | {row['bootstrap_median_p16']:.2f}-{row['bootstrap_median_p84']:.2f} | "
            f"{row['p16_mbb']:.2f}-{row['p84_mbb']:.2f} |"
        )

    lines = [
        "# H->bb Detector Response Diagnostic",
        "",
        "This diagnostic follows up on the low truth-matched reconstructed H->bb mass in the COLLIDE-1M HH->bbWW sample.",
        "",
        "It is a detector/reconstruction check, not a new classifier.",
        "",
        "## Command",
        "",
        "```bash",
        f"/tmp/hh-bbww-venv/bin/python scripts/diagnose_hbb_detector_response.py --n-files {args.n_files} --bootstrap {args.bootstrap}",
        "```",
        "",
        "## Event Selection",
        "",
        "- Select status-23 generator b quarks using the same practical convention as the existing HH preprocessing.",
        "- Match each generator b quark to the closest retained AK4 jet.",
        "- Require two distinct matched AK4 jets with DeltaR < 0.4.",
        f"- Use MAX_JETS = {MAX_JETS}.",
        "",
        "## Event Counts",
        "",
        f"- scanned events: {counts['scanned']}",
        f"- usable matched events: {counts['usable']}",
        f"- no two status-23 b quarks: {counts.get('no_two_status23_bquarks', 0)}",
        f"- unmatched b quark: {counts.get('unmatched_bquark', 0)}",
        f"- both b quarks matched to same jet: {counts.get('same_matched_jet', 0)}",
        "",
        "## Overall Matched m_bb",
        "",
        "| n | Mean [GeV] | Median [GeV] | 16-84% [GeV] | 90 < m_bb < 140 |",
        "|---:|---:|---:|---:|---:|",
        f"| {overall['n']} | {overall['mean']:.2f} | {overall['median']:.2f} | {overall['p16']:.2f}-{overall['p84']:.2f} | {overall['frac_90_140']:.3f} |",
        "",
        "## Matched m_bb vs Lower Matched-Jet pT",
        "",
        "| lower matched-jet pT bin [GeV] | n | median m_bb [GeV] | bootstrap median 16-84% [GeV] | event 16-84% [GeV] |",
        "|---:|---:|---:|---:|---:|",
        *profile_rows,
        "",
        "## Eta Category Summary",
        "",
        "| Eta category | n | Mean [GeV] | Median [GeV] | 16-84% [GeV] | 90 < m_bb < 140 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in eta_df.iterrows():
        lines.append(
            f"| {row['eta_category']} | {int(row['n'])} | {row['mean']:.2f} | {row['median']:.2f} | "
            f"{row['p16']:.2f}-{row['p84']:.2f} | {row['frac_90_140']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Lower Matched-Jet pT Cut Summary",
            "",
            "| pT cut [GeV] | Signal retention | n | Mean [GeV] | Median [GeV] | 16-84% [GeV] |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in cut_df.iterrows():
        lines.append(
            f"| {row['min_lower_matched_jet_pt_cut']:.0f} | {row['signal_retention']:.3f} | {int(row['n'])} | "
            f"{row['mean']:.2f} | {row['median']:.2f} | {row['p16']:.2f}-{row['p84']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If the median matched m_bb moves toward 125 GeV as the lower matched-jet pT increases, the low mass is correlated with low-pT matched b jets.",
            "- If reco/gen pT response is below 1, the matched reconstructed AK4 jet captures less pT than the generator-level b quark.",
            "- If the response is worse at low pT or high |eta|, this supports detector response, acceptance, thresholds, or AK4 containment as contributors to the low m_bb.",
            "- If pT cuts move m_bb closer to 125 GeV but remove many events, then this effect is relevant but cannot simply be fixed by a hard cut without losing signal.",
            "",
            "## Outputs",
            "",
            "- `outputs/hbb_detector_response/event_response_diagnostics.csv`",
            "- `outputs/hbb_detector_response/min_pt_profile.csv`",
            "- `outputs/hbb_detector_response/perjet_response_profile_vs_reco_pt.csv`",
            "- `outputs/hbb_detector_response/perjet_response_profile_vs_abs_eta.csv`",
            "- `outputs/hbb_detector_response/eta_category_summary.csv`",
            "- `outputs/hbb_detector_response/min_pt_cut_summary.csv`",
            "- `outputs/hbb_detector_response/summary.json`",
            "- `outputs/plots/hbb_detector_response/*.png`",
        ]
    )

    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose detector/reconstruction effects in truth-matched H->bb mass.")
    parser.add_argument("--n-files", type=int, default=2, help="Number of cached HH parquet files to scan.")
    parser.add_argument("--max-events", type=int, default=-1, help="Maximum raw events to scan; use -1 for all.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap replicas for median error bars.")
    parser.add_argument("--seed", type=int, default=20260617)
    args = parser.parse_args()
    if args.max_events < 0:
        args.max_events = None
    return args


def main():
    args = parse_args()
    ensure_dirs()

    print("Scanning HH raw events for detector-response diagnostic...")
    event_df, counts = scan_hh_events(args)
    if event_df.empty:
        raise RuntimeError("No usable H->bb matched events were found.")

    perjet_df = build_perjet_df(event_df)
    overall = summarize_values(event_df["matched_mbb"].to_numpy())

    min_pt_profile = profile_mbb_vs_min_pt(event_df, args)
    reco_pt_response = response_profile(
        perjet_df["reco_pt"].to_numpy(),
        perjet_df["response_reco_pt_over_gen_pt"].to_numpy(),
        DEFAULT_RECO_PT_BINS,
        "reco_pt",
    )
    abs_eta_response = response_profile(
        perjet_df["abs_reco_eta"].to_numpy(),
        perjet_df["response_reco_pt_over_gen_pt"].to_numpy(),
        DEFAULT_ABS_ETA_BINS,
        "abs_eta",
    )
    eta_df = eta_category_summary(event_df)
    cut_df = min_pt_cut_summary(event_df)

    event_df.to_csv(OUTDIR / "event_response_diagnostics.csv", index=False)
    perjet_df.to_csv(OUTDIR / "perjet_response_diagnostics.csv", index=False)
    min_pt_profile.to_csv(OUTDIR / "min_pt_profile.csv", index=False)
    reco_pt_response.to_csv(OUTDIR / "perjet_response_profile_vs_reco_pt.csv", index=False)
    abs_eta_response.to_csv(OUTDIR / "perjet_response_profile_vs_abs_eta.csv", index=False)
    eta_df.to_csv(OUTDIR / "eta_category_summary.csv", index=False)
    cut_df.to_csv(OUTDIR / "min_pt_cut_summary.csv", index=False)

    summary = {
        "args": vars(args),
        "counts": counts,
        "overall_mbb": overall,
        "outputs": {
            "event_response_diagnostics": str(OUTDIR / "event_response_diagnostics.csv"),
            "perjet_response_diagnostics": str(OUTDIR / "perjet_response_diagnostics.csv"),
            "min_pt_profile": str(OUTDIR / "min_pt_profile.csv"),
            "perjet_response_profile_vs_reco_pt": str(OUTDIR / "perjet_response_profile_vs_reco_pt.csv"),
            "perjet_response_profile_vs_abs_eta": str(OUTDIR / "perjet_response_profile_vs_abs_eta.csv"),
            "eta_category_summary": str(OUTDIR / "eta_category_summary.csv"),
            "min_pt_cut_summary": str(OUTDIR / "min_pt_cut_summary.csv"),
            "plots": str(PLOT_DIR),
            "markdown": str(SUMMARY_MD),
        },
    }

    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_mbb_profile(min_pt_profile)
    plot_counts(min_pt_profile)
    plot_response_profile(
        reco_pt_response,
        "reco_pt_center",
        "reco_over_gen_pt_response_vs_reco_pt.png",
        "Matched b-jet response vs reconstructed pT",
        "Matched reconstructed jet pT [GeV]",
    )
    plot_response_profile(
        abs_eta_response,
        "abs_eta_center",
        "reco_over_gen_pt_response_vs_abs_eta.png",
        "Matched b-jet response vs |eta|",
        "Matched reconstructed jet |eta|",
    )
    plot_eta_box(event_df)
    plot_min_pt_cuts(cut_df)
    write_markdown(counts, overall, min_pt_profile, eta_df, cut_df, args)

    print("\nOverall matched m_bb:")
    print(json.dumps(overall, indent=2))

    print("\nEvent counts:")
    print(json.dumps(counts, indent=2))

    print("\nMin-pT profile:")
    print(min_pt_profile.to_string(index=False))

    print("\nEta summary:")
    print(eta_df.to_string(index=False))

    print("\nMin-pT cut summary:")
    print(cut_df.to_string(index=False))

    print(f"\nSaved tables to {OUTDIR}")
    print(f"Saved plots to {PLOT_DIR}")
    print(f"Saved markdown summary to {SUMMARY_MD}")


if __name__ == "__main__":
    main()
