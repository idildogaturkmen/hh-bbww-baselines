"""
Diagnose whether semileptonic-like b decays correlate with the low reconstructed
truth-matched H->bb AK4 dijet mass in the COLLIDE-1M HH->bbWW sample.

The label is intentionally called "semileptonic-like": the flattened COLLIDE
truth record does not provide a clean b-hadron ancestry chain for every lepton
or neutrino.  We therefore classify each matched H->bb b jet by nearby generator
charged leptons and neutrinos in the matched AK4 jet cone.
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
OUTDIR = Path("outputs/hbb_decay_modes")
PLOT_DIR = Path("outputs/plots/hbb_decay_modes")
SUMMARY_MD = Path("RESULTS_HBB_DECAY_MODES.md")
M_H = 125.0
MAX_JETS = 12
MATCH_DR = 0.4
LEPTON_CONE = 0.4
MIN_LEPTON_PT = 1.0
MIN_NEUTRINO_PT = 1.0

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
CHARGED_LEPTON_PIDS = {11, 13, 15} # e, mu, tau
NEUTRINO_PIDS = {12, 14, 16} # nu_e, nu_mu, nu_tau

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

# makes phi between -pi and pi, then calculates the difference between two phi values in a way that accounts for the periodicity of phi
def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2.0 * math.pi
    while dphi <= -math.pi:
        dphi += 2.0 * math.pi
    return dphi

# calculates the delta R between two particles given their eta and phi values
def delta_r_eta_phi(eta1, phi1, eta2, phi2):
    return math.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2)

# calculates the delta R between two jets given their eta and phi values, where j1 and j2 are arrays containing the jet information (pt, eta, phi, mass, etc.) and the eta and phi values are located at indices 1 and 2 respectively
def delta_r(j1, j2):
    return delta_r_eta_phi(j1[1], j1[2], j2[1], j2[2])


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


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            "n_events": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p16": float("nan"),
            "p84": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
            "frac_90_140": float("nan"),
            "frac_100_150": float("nan"),
        }

    return {
        "n_events": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p16": float(np.percentile(values, 16)),
        "p84": float(np.percentile(values, 84)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "frac_90_140": float(np.mean((values >= 90.0) & (values <= 140.0))),
        "frac_100_150": float(np.mean((values >= 100.0) & (values <= 150.0))),
    }


def event_from_batch(columns, row):
    return {name: columns[name][row] for name in COLUMNS}


def build_jets(event, max_jets=MAX_JETS):
    '''
    builds the AK4 jet array for a given event, where each jet is represented as an array of its properties (pt, eta, phi, mass, b-tag score, etc.)
     and the number of jets is limited to max_jets. 
    The function returns the jet array and the raw number of jets before applying the max_jets limit.
    '''
    n_jets_raw = len(event["FullReco_JetAK4_PT"])
    n_jets = min(n_jets_raw, max_jets)
    jets = np.zeros((n_jets, 6), dtype=np.float32)
    for idx, col in enumerate(JET_COLUMNS):
        jets[:, idx] = event[col][:n_jets]
    return jets, n_jets_raw


def find_hbb_bquarks(event):
    '''
    Finds two generator b quarks from H->bb decay in the event by looking for status-23 b quarks, 
    which are after parton showering but before hadronization and detector effects. 
    The function returns the indices of the two b quarks sorted by descending pT.
    '''
    pids = event["FullReco_GenPart_PID"]
    statuses = event["FullReco_GenPart_Status"]
    pts = event["FullReco_GenPart_PT"]

    candidates = []
    for idx, pid in enumerate(pids):
        if abs(int(pid)) == 5 and int(statuses[idx]) == 23: # looking for status-23 b quarks, which are the practical choice for H->bb matching in COLLIDE because they are after parton showering but before hadronization and detector effects
            candidates.append(idx)

    return sorted(candidates, key=lambda idx: pts[idx], reverse=True)[:2]


def match_bquarks_to_retained_jets(event, b_indices, max_dr=MATCH_DR, max_jets=MAX_JETS):
    '''
    Matches each generator-level b quark to the closest retained AK4 jet within a specified deltaR<0.4 and returns a list of matches.
    '''

    n_jets = min(len(event["FullReco_JetAK4_PT"]), max_jets)
    matches = []

    for bidx in b_indices:
        b_eta = float(event["FullReco_GenPart_Eta"][bidx])
        b_phi = float(event["FullReco_GenPart_Phi"][bidx])
        best_j = -1 # initialize best_j to -1 to indicate no match found yet
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

        matches.append({"bidx": int(bidx), "jet_idx": int(best_j), "deltaR": float(best_dr), "matched": best_dr < max_dr})

    return matches


def nearby_truth_leptons(event, eta, phi, cone=LEPTON_CONE):
    '''
    Looks for generator-level charged leptons and neutrinos within a specified cone around the given eta and phi coordinates (which correspond to the matched AK4 jet axis).
    The function returns a dictionary with counts and pT sums of nearby charged leptons and neutrinos, 
    as well as lists of the nearby leptons and neutrinos with their properties (index, PID, pT, deltaR to the jet axis)
    '''
    pids = event["FullReco_GenPart_PID"]
    pts = event["FullReco_GenPart_PT"]
    etas = event["FullReco_GenPart_Eta"]
    phis = event["FullReco_GenPart_Phi"]

    charged_pts = []
    neutrino_pts = []
    charged_rows = []
    neutrino_rows = []

    for idx, pid_raw in enumerate(pids):
        pid = abs(int(pid_raw))
        # 0 semileptonic case
        if pid not in CHARGED_LEPTON_PIDS and pid not in NEUTRINO_PIDS:
            continue

        pt = float(pts[idx])
        # is a charged lepton and below pT threshold
        if pid in CHARGED_LEPTON_PIDS and pt < MIN_LEPTON_PT:
            continue
        # is a neutrino and below pT threshold
        if pid in NEUTRINO_PIDS and pt < MIN_NEUTRINO_PT:
            continue

        dr = delta_r_eta_phi(float(etas[idx]), float(phis[idx]), eta, phi)
        if dr >= cone:
            continue

        row = {"idx": int(idx), "pid": int(pid_raw), "pt": pt, "deltaR": float(dr)}
        if pid in CHARGED_LEPTON_PIDS:
            charged_pts.append(pt)
            charged_rows.append(row)
        else:
            neutrino_pts.append(pt)
            neutrino_rows.append(row)

    return {
        "charged_lepton_count": int(len(charged_pts)),
        "charged_lepton_pt_sum": float(np.sum(charged_pts)) if charged_pts else 0.0,
        "charged_lepton_max_pt": float(np.max(charged_pts)) if charged_pts else 0.0,
        "neutrino_count": int(len(neutrino_pts)),
        "neutrino_pt_sum": float(np.sum(neutrino_pts)) if neutrino_pts else 0.0,
        "neutrino_max_pt": float(np.max(neutrino_pts)) if neutrino_pts else 0.0,
        "charged_leptons": charged_rows,
        "neutrinos": neutrino_rows,
    }


def classify_matched_bjet(event, jet_idx):
    '''
    Labels a matched b jet as semileptonic-like if it has nearby charged lepton and neutrino activity.
    '''
    eta = float(event["FullReco_JetAK4_Eta"][jet_idx])
    phi = float(event["FullReco_JetAK4_Phi"][jet_idx])
    nearby = nearby_truth_leptons(event, eta, phi)

    has_charged = nearby["charged_lepton_count"] > 0
    has_neutrino = nearby["neutrino_count"] > 0
    strict_semileptonic_like = has_charged and has_neutrino

    return {
        **{k: v for k, v in nearby.items() if k not in {"charged_leptons", "neutrinos"}},
        "has_nearby_charged_lepton": bool(has_charged),
        "has_nearby_neutrino": bool(has_neutrino),
        "strict_semileptonic_like": bool(strict_semileptonic_like),
        "nearby_charged_leptons_json": json.dumps(nearby["charged_leptons"]),
        "nearby_neutrinos_json": json.dumps(nearby["neutrinos"]),
    }


def category_from_count(n, label):
    '''
    Turns the number of semileptonic-like b jets into labels like 0, 1, or 2.
    '''
    if n == 0:
        return f"0_{label}_bjets"
    if n == 1:
        return f"1_{label}_bjet"
    return f"2_{label}_bjets"


def process_event(event, file_label, local_event, global_event):
    '''
    1. find b quarks
    2. match jets
    3. compute m_bb
    4. classify the two matched b jets
    5. store one row of output.
    '''
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
    mbb = invariant_mass(jets[j0], jets[j1])
    drbb = delta_r(jets[j0], jets[j1])
    pts = [float(jets[j0, 0]), float(jets[j1, 0])]

    bjet0 = classify_matched_bjet(event, j0)
    bjet1 = classify_matched_bjet(event, j1)
    bjets = [bjet0, bjet1]

    n_strict = int(sum(bjet["strict_semileptonic_like"] for bjet in bjets))
    n_neutrino = int(sum(bjet["has_nearby_neutrino"] for bjet in bjets))
    n_charged = int(sum(bjet["has_nearby_charged_lepton"] for bjet in bjets))

    row = {
        "file": file_label,
        "local_event": int(local_event),
        "global_event": int(global_event),
        "n_jets_raw": int(n_jets_raw),
        "n_jets_retained": int(len(jets)),
        "b0_gen_idx": int(matches[0]["bidx"]),
        "b1_gen_idx": int(matches[1]["bidx"]),
        "b0_jet_idx": int(j0),
        "b1_jet_idx": int(j1),
        "b0_match_deltaR": float(matches[0]["deltaR"]),
        "b1_match_deltaR": float(matches[1]["deltaR"]),
        "matched_mbb": float(mbb),
        "matched_deltaR_bb": float(drbb),
        "matched_min_pt": float(min(pts)),
        "matched_max_pt": float(max(pts)),
        "matched_mean_pt": float(np.mean(pts)),
        "n_strict_semileptonic_like_bjets": n_strict,
        "n_neutrino_like_bjets": n_neutrino,
        "n_charged_lepton_like_bjets": n_charged,
        "strict_decay_category": category_from_count(n_strict, "strict_semileptonic_like"),
        "neutrino_category": category_from_count(n_neutrino, "neutrino_like"),
        "charged_lepton_category": category_from_count(n_charged, "charged_lepton_like"),
    }

    for idx, bjet in enumerate(bjets):
        for key, value in bjet.items():
            row[f"b{idx}_{key}"] = value

    return row, "usable"


def scan_hh_events(args):
    '''
    Loops over HH events and collects all usable matched events.
    '''
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


def build_group_summary(df, group_col):
    '''
    Summarized the matched m_bb distribution for each category in the specified grouping column 
    (e.g. strict semileptonic-like category or nearby neutrino category) and returns a list of summary dictionaries for each category.
    '''
    rows = []
    for group, part in df.groupby(group_col, sort=True):
        summary = summarize_values(part["matched_mbb"].to_numpy())
        rows.append({"grouping": group_col, "category": group, **summary})
    return rows


def save_overlay(df, group_col, filename, title):
    '''
    Makes histogram overlay plots. Each category in the specified grouping column gets its own histogram of the matched m_bb distribution, 
    and all histograms are overlaid on the same plot for comparison.
    '''
    bins = np.linspace(0.0, 240.0, 80)
    plt.figure(figsize=(7.5, 5.0))
    for category, part in df.groupby(group_col, sort=True):
        plt.hist(
            part["matched_mbb"],
            bins=bins,
            histtype="step",
            linewidth=1.5,
            density=True,
            label=f"{category} (n={len(part)})",
        )
    plt.axvline(M_H, color="black", linestyle="--", linewidth=1.0, label="125 GeV")
    plt.xlabel("Truth-matched reconstructed $m_{bb}$ [GeV]")
    plt.ylabel("Normalized events")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200)
    plt.close()


def save_boxplot(df, group_col, filename, title):
    '''
    Makes box-and-whisker plots to show the distribution of matched m_bb for each category in the specified grouping column.
     Each box represents the interquartile range (16th to 84th percentile) of the matched m_bb values for that category, with a line at the median. 
     The whiskers extend to the 5th and 95th percentiles, and outliers beyond that are not shown.
    '''
    groups = [(category, part["matched_mbb"].to_numpy()) for category, part in df.groupby(group_col, sort=True)]
    labels = [item[0] for item in groups]
    values = [item[1] for item in groups]
    plt.figure(figsize=(8.0, 5.2))
    plt.boxplot(values, tick_labels=labels, showfliers=False)
    plt.axhline(M_H, color="black", linestyle="--", linewidth=1.0, label="125 GeV")
    plt.ylabel("Truth-matched reconstructed $m_{bb}$ [GeV]")
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200)
    plt.close()


def save_scatter(df):
    '''
    Plots m_bb versus nearby neutrino pT sum.
    Each point corresponds to one event, with the x-coordinate given by the sum of the pT of generator neutrinos found within the matched AK4 jet cones of the two b jets,
     and the y-coordinate given by the truth-matched reconstructed m_bb.
    '''
    plt.figure(figsize=(7.2, 5.0))
    plt.hexbin(
        df["b0_neutrino_pt_sum"] + df["b1_neutrino_pt_sum"],
        df["matched_mbb"],
        gridsize=45,
        mincnt=1,
        cmap="viridis",
    )
    plt.axhline(M_H, color="white", linestyle="--", linewidth=1.0)
    plt.colorbar(label="Events")
    plt.xlabel("Nearby generator neutrino pT sum in matched b-jet cones [GeV]")
    plt.ylabel("Truth-matched reconstructed $m_{bb}$ [GeV]")
    plt.title("m_bb vs nearby neutrino activity")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "matched_mbb_vs_nearby_neutrino_pt_sum.png", dpi=200)
    plt.close()


def write_markdown(summary, counts, args):
    summary_df = pd.DataFrame(summary)
    strict = summary_df[summary_df["grouping"] == "strict_decay_category"].copy()
    neutrino = summary_df[summary_df["grouping"] == "neutrino_category"].copy()
    overall = summary_df[(summary_df["grouping"] == "overall") & (summary_df["category"] == "all")].iloc[0]

    lines = [
        "# H->bb Decay-Mode m_bb Diagnostic",
        "",
        "This diagnostic tests whether the low truth-matched reconstructed H->bb mass is correlated with semileptonic-like b decays.",
        "",
        "## Command",
        "",
        "```bash",
        f"/tmp/hh-bbww-venv/bin/python scripts/diagnose_hbb_decay_modes.py --n-files {args.n_files} --max-events {args.max_events if args.max_events is not None else -1}",
        "```",
        "",
        "## Definition",
        "",
        "- The H->bb b quarks are selected with the same practical convention as the current HH preprocessing: status-23 generator b quarks, matched to distinct retained AK4 jets with DeltaR < 0.4.",
        "- A matched b jet is called `strict_semileptonic_like` when both a generator charged lepton and a generator neutrino are found within DeltaR < 0.4 of the matched AK4 jet axis.",
        "- A matched b jet is called `neutrino_like` when a generator neutrino is found within DeltaR < 0.4 of the matched AK4 jet axis.",
        "- These are cone-based labels, not perfect b-hadron ancestry labels. They can be contaminated by nearby W-decay leptons/neutrinos, so the result should be interpreted as a diagnostic trend.",
        "",
        "## Event Counts",
        "",
        f"- scanned events: {counts['scanned']}",
        f"- usable matched events: {counts['usable']}",
        f"- no two status-23 b quarks: {counts.get('no_two_status23_bquarks', 0)}",
        f"- unmatched b quark: {counts.get('unmatched_bquark', 0)}",
        f"- both b quarks matched to the same AK4 jet: {counts.get('same_matched_jet', 0)}",
        "",
        "## Overall Matched m_bb",
        "",
        "| n | Mean [GeV] | Median [GeV] | 16-84% [GeV] | 90 < m_bb < 140 |",
        "|---:|---:|---:|---:|---:|",
        f"| {int(overall['n_events'])} | {overall['mean']:.2f} | {overall['median']:.2f} | {overall['p16']:.2f}-{overall['p84']:.2f} | {overall['frac_90_140']:.3f} |",
        "",
        "## Strict Semileptonic-Like Categories",
        "",
        "| Category | n | Mean [GeV] | Median [GeV] | 16-84% [GeV] | 90 < m_bb < 140 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in strict.iterrows():
        lines.append(
            f"| {row['category']} | {int(row['n_events'])} | {row['mean']:.2f} | {row['median']:.2f} | "
            f"{row['p16']:.2f}-{row['p84']:.2f} | {row['frac_90_140']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Nearby-Neutrino Categories",
            "",
            "| Category | n | Mean [GeV] | Median [GeV] | 16-84% [GeV] | 90 < m_bb < 140 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in neutrino.iterrows():
        lines.append(
            f"| {row['category']} | {int(row['n_events'])} | {row['mean']:.2f} | {row['median']:.2f} | "
            f"{row['p16']:.2f}-{row['p84']:.2f} | {row['frac_90_140']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If the hadronic-like category moves closer to 125 GeV while semileptonic-like categories shift lower, that supports semileptonic b-decay neutrino energy loss as one contributor to the low reconstructed m_bb. If all categories remain similarly low, then detector/jet response, AK4 containment, pT thresholds, or simplified COLLIDE calibration are likely also important.",
            "",
            "This diagnostic cannot prove the full cause because the available truth record does not provide a clean b-hadron decay chain for every nearby lepton/neutrino. It is still a useful first physics cross-check because it tests whether invisible energy near the matched b jets correlates with the mass shift.",
            "",
            "Z->bb contamination cannot explain the truth-matched HH signal m_bb shift because these events come from the HH signal sample and the jets are matched to H->bb truth b quarks. Z->bb remains a separate background-rejection issue.",
            "",
            "## Outputs",
            "",
            "- `outputs/hbb_decay_modes/hbb_decay_mode_event_diagnostics.csv`",
            "- `outputs/hbb_decay_modes/hbb_decay_mode_summary.csv`",
            "- `outputs/hbb_decay_modes/hbb_decay_mode_summary.json`",
            "- `outputs/plots/hbb_decay_modes/matched_mbb_by_strict_decay_category.png`",
            "- `outputs/plots/hbb_decay_modes/matched_mbb_by_neutrino_category.png`",
            "- `outputs/plots/hbb_decay_modes/matched_mbb_vs_nearby_neutrino_pt_sum.png`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Split HH truth-matched H->bb m_bb by semileptonic-like b-jet activity.")
    parser.add_argument("--n-files", type=int, default=2, help="Number of cached HH parquet files to scan.")
    parser.add_argument("--max-events", type=int, default=-1, help="Maximum raw events to scan; use -1 for all selected files.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=2000)
    args = parser.parse_args()
    if args.max_events < 0:
        args.max_events = None
    return args


def main():
    args = parse_args()
    print("Scanning HH raw events for semileptonic-like b-jet signatures...")
    event_df, counts = scan_hh_events(args)
    if event_df.empty:
        raise RuntimeError("No usable H->bb matched events were found.")

    summaries = [{"grouping": "overall", "category": "all", **summarize_values(event_df["matched_mbb"].to_numpy())}]
    for group_col in ["strict_decay_category", "neutrino_category", "charged_lepton_category"]:
        summaries.extend(build_group_summary(event_df, group_col))

    summary_df = pd.DataFrame(summaries)
    event_df.to_csv(OUTDIR / "hbb_decay_mode_event_diagnostics.csv", index=False)
    summary_df.to_csv(OUTDIR / "hbb_decay_mode_summary.csv", index=False)
    with open(OUTDIR / "hbb_decay_mode_summary.json", "w") as f:
        json.dump({"args": vars(args), "counts": counts, "summary": summaries}, f, indent=2)

    save_overlay(
        event_df,
        "strict_decay_category",
        "matched_mbb_by_strict_decay_category.png",
        "H->bb matched m_bb by strict semileptonic-like category",
    )
    save_overlay(
        event_df,
        "neutrino_category",
        "matched_mbb_by_neutrino_category.png",
        "H->bb matched m_bb by nearby-neutrino category",
    )
    save_boxplot(
        event_df,
        "strict_decay_category",
        "matched_mbb_box_by_strict_decay_category.png",
        "H->bb matched m_bb by strict semileptonic-like category",
    )
    save_scatter(event_df)
    write_markdown(summaries, counts, args)

    print("\nDecay-mode m_bb summary:")
    print(summary_df.to_string(index=False))
    print("\nCounts:")
    print(json.dumps(counts, indent=2))
    print(f"\nSaved tables to {OUTDIR}")
    print(f"Saved plots to {PLOT_DIR}")
    print(f"Saved markdown summary to {SUMMARY_MD}")


if __name__ == "__main__":
    main()
