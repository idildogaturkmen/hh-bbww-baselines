'''
This script inspects HH->bbWW events to classify the WW decay mode using status-23 hard-process particles, and to determine whether the hadronic W-side quarks can be matched to reconstructed AK4 jets after excluding the H->bb jets.
'''

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from make_hbb_bjet_regression_dataset import (
    COLUMNS,
    find_hh_files,
    event_from_batch,
    build_reco_jets,
    find_hbb_bquarks,
    match_bquarks_to_reco_jets,
)


def as_int(x, default=-999):
    try:
        return int(x)
    except Exception:
        return default


def arr(event, name):
    x = event.get(name, [])
    return [] if x is None else x


def delta_phi(a, b):
    d = float(a) - float(b)
    while d > math.pi:
        d -= 2.0 * math.pi
    while d <= -math.pi:
        d += 2.0 * math.pi
    return d


def delta_r(eta1, phi1, eta2, phi2):
    return math.sqrt((float(eta1) - float(eta2)) ** 2 + delta_phi(phi1, phi2) ** 2)


def status23_indices(event):
    pids = arr(event, "FullReco_GenPart_PID")
    statuses = arr(event, "FullReco_GenPart_Status")
    if len(pids) != len(statuses):
        return []
    return [i for i, s in enumerate(statuses) if as_int(s) == 23]


def classify_status23_ww(event):
    '''
    Classify the WW decay mode using status-23 hard-process particles, and identify the hadronic W-side quarks for matching to reco jets.
    The classification is based on the number of status-23 light quarks, charged leptons, and neutrinos after excluding the two H->bb b quarks.
    The hadronic W-side quarks are the first two or four status-23 light quarks found, depending on the decay mode.
    1. If there are at least 4 status-23 light quarks and no charged leptons or neutrinos, the mode is "had_had" and the first four light quarks are the hadronic targets.
    2. If there are at least 2 status-23 light quarks, at least 1 charged lepton, and at least 1 neutrino, the mode is "lep_had_*" (with the lepton flavor) and the first two light quarks are the hadronic targets.
    3. If there are no status-23 light quarks, at least 2 charged leptons, and at least 2 neutrinos, the mode is "lep_lep" and there are no hadronic targets.
    4. Otherwise, the mode is "unknown" and all status-23 light quarks are considered hadronic targets for matching.
    The function returns a dictionary with the WW mode, the indices of the H->bb b quarks, the indices and PIDs of the status-23 light quarks, charged leptons, neutrinos, and gluons, the indices of the hadronic target quarks, and the PIDs and PTs of the light quarks, charged leptons, and neutrinos.
    '''
    pids = arr(event, "FullReco_GenPart_PID")
    pts = arr(event, "FullReco_GenPart_PT")

    s23 = status23_indices(event)

    # The two H->bb quarks are handled by the existing convention.
    hbb_b = find_hbb_bquarks(event)
    hbb_set = set(hbb_b)

    light_quarks = []
    charged_leptons = []
    neutrinos = []
    gluons = []

    for i in s23:
        pid = as_int(pids[i])
        apid = abs(pid)

        if i in hbb_set:
            continue

        if apid == 21:
            gluons.append(i)
        elif 1 <= apid <= 4:
            light_quarks.append(i)
        elif apid in [11, 13, 15]:
            charged_leptons.append(i)
        elif apid in [12, 14, 16]:
            neutrinos.append(i)

    # Keep stable order by GenPart index. This appears meaningful in the debug printout.
    light_quarks = sorted(light_quarks)
    charged_leptons = sorted(charged_leptons)
    neutrinos = sorted(neutrinos)
    gluons = sorted(gluons)

    n_q = len(light_quarks)
    n_l = len(charged_leptons)
    n_nu = len(neutrinos)

    if n_q >= 4 and n_l == 0 and n_nu == 0:
        ww_mode = "had_had"
        hadronic_target_quarks = light_quarks[:4]
    elif n_q >= 2 and n_l >= 1 and n_nu >= 1:
        lep_flavors = sorted({abs(as_int(pids[i])) for i in charged_leptons})
        if 15 in lep_flavors:
            ww_mode = "lep_had_tau"
        elif 13 in lep_flavors:
            ww_mode = "lep_had_mu"
        elif 11 in lep_flavors:
            ww_mode = "lep_had_e"
        else:
            ww_mode = "lep_had"
        hadronic_target_quarks = light_quarks[:2]
    elif n_q == 0 and n_l >= 2 and n_nu >= 2:
        ww_mode = "lep_lep"
        hadronic_target_quarks = []
    else:
        ww_mode = "unknown"
        hadronic_target_quarks = light_quarks

    return {
        "ww_mode": ww_mode,
        "hbb_b_indices": hbb_b,
        "light_quark_indices": light_quarks,
        "charged_lepton_indices": charged_leptons,
        "neutrino_indices": neutrinos,
        "gluon_indices": gluons,
        "hadronic_target_quarks": hadronic_target_quarks,
        "light_quark_pids": [as_int(pids[i]) for i in light_quarks],
        "charged_lepton_pids": [as_int(pids[i]) for i in charged_leptons],
        "neutrino_pids": [as_int(pids[i]) for i in neutrinos],
        "gluon_pids": [as_int(pids[i]) for i in gluons],
        "light_quark_pts": [float(pts[i]) for i in light_quarks],
        "charged_lepton_pts": [float(pts[i]) for i in charged_leptons],
        "neutrino_pts": [float(pts[i]) for i in neutrinos],
    }


def match_truth_particles_to_reco_jets(event, truth_indices, reco_jets, exclude_reco_indices=None, dr_max=0.4):
    '''
    Matches truth particles to reconstructed jets based on delta-R distance.

    Parameters:
    - event: The event data containing truth particle information.
    - truth_indices: List of indices of truth particles to match.
    - reco_jets: List of reconstructed jets, each represented as a dictionary with 'idx', 'eta', and 'phi' keys.
    - exclude_reco_indices: Set of reco jet indices to exclude from matching (default: None).
    - dr_max: Maximum delta-R distance for a match to be considered valid (default: 0.4).
    
    '''
    if exclude_reco_indices is None:
        exclude_reco_indices = set()
    else:
        exclude_reco_indices = set(exclude_reco_indices)

    gen_eta = arr(event, "FullReco_GenPart_Eta")
    gen_phi = arr(event, "FullReco_GenPart_Phi")

    used = set()
    matches = []

    for tidx in truth_indices:
        best = None

        for jet in reco_jets:
            ridx = int(jet["idx"])
            if ridx in exclude_reco_indices:
                continue
            if ridx in used:
                continue

            dr = delta_r(gen_eta[tidx], gen_phi[tidx], jet["eta"], jet["phi"])
            if best is None or dr < best["dr"]:
                best = {
                    "truth_idx": int(tidx),
                    "reco_idx": ridx,
                    "dr": float(dr),
                    "matched": bool(dr < dr_max),
                }

        if best is None:
            matches.append({
                "truth_idx": int(tidx),
                "reco_idx": -1,
                "dr": np.nan,
                "matched": False,
            })
        elif best["matched"]:
            matches.append(best)
            used.add(best["reco_idx"])
        else:
            matches.append(best)

    all_matched = len(matches) == len(truth_indices) and all(m["matched"] for m in matches)
    return matches, bool(all_matched)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-files", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--min-reco-pt", type=float, default=20.0)
    parser.add_argument("--max-abs-eta", type=float, default=2.5)
    parser.add_argument("--max-reco-jets", type=int, default=-1)
    parser.add_argument("--dr-match", type=float, default=0.4)
    parser.add_argument("--outdir", default="outputs/hhbbww_status23_ww_targets")
    parser.add_argument("--plot-dir", default="outputs/plots/hhbbww_status23_ww_targets")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    plot_dir = Path(args.plot_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    files = find_hh_files(args.n_files)
    print(f"Using {len(files)} HH->bbWW files")

    rows = []
    counters = {
        "raw_events": 0,
        "events_with_two_status23_bquarks": 0,
        "events_with_truth_hbb_reco_pair": 0,
        "events_with_hadronic_ww_targets": 0,
        "events_with_all_hadronic_target_quarks_matched": 0,
    }

    stop = False

    for file_path in files:
        print(f"Reading {file_path}")
        pf = pq.ParquetFile(file_path)

        for batch in pf.iter_batches(columns=COLUMNS, batch_size=args.batch_size):
            cols = batch.to_pydict()
            n_rows = len(next(iter(cols.values())))

            for row_idx in range(n_rows):
                if args.max_events > 0 and counters["raw_events"] >= args.max_events:
                    stop = True
                    break

                event = event_from_batch(cols, row_idx)
                event_id = counters["raw_events"]
                counters["raw_events"] += 1

                reco_jets, n_raw_reco = build_reco_jets(
                    event,
                    max_reco_jets=args.max_reco_jets,
                    min_reco_pt=args.min_reco_pt,
                    max_abs_eta=args.max_abs_eta,
                )

                ww = classify_status23_ww(event)

                hbb_reco_indices = []
                hbb_reco_pair_matched = False

                if len(ww["hbb_b_indices"]) >= 2:
                    counters["events_with_two_status23_bquarks"] += 1

                    if len(reco_jets) >= 2:
                        hbb_matches = match_bquarks_to_reco_jets(event, ww["hbb_b_indices"], reco_jets)
                        if len(hbb_matches) == 2 and all(m["matched"] for m in hbb_matches):
                            hbb_reco_indices = [int(m["jet"]["idx"]) for m in hbb_matches]
                            hbb_reco_pair_matched = len(set(hbb_reco_indices)) == 2

                if hbb_reco_pair_matched:
                    counters["events_with_truth_hbb_reco_pair"] += 1

                hadronic_targets = ww["hadronic_target_quarks"]
                if len(hadronic_targets) > 0:
                    counters["events_with_hadronic_ww_targets"] += 1

                q_matches, all_q_matched = match_truth_particles_to_reco_jets(
                    event,
                    hadronic_targets,
                    reco_jets,
                    exclude_reco_indices=set(hbb_reco_indices),
                    dr_max=args.dr_match,
                )

                if len(hadronic_targets) > 0 and all_q_matched:
                    counters["events_with_all_hadronic_target_quarks_matched"] += 1

                rows.append({
                    "event_id": int(event_id),
                    "n_raw_reco_jets": int(n_raw_reco),
                    "n_selected_reco_jets": int(len(reco_jets)),
                    "ww_mode": ww["ww_mode"],
                    "hbb_b_indices": json.dumps(ww["hbb_b_indices"]),
                    "hbb_reco_pair_matched": bool(hbb_reco_pair_matched),
                    "hbb_reco_indices": json.dumps(hbb_reco_indices),
                    "n_status23_light_quarks": int(len(ww["light_quark_indices"])),
                    "n_status23_charged_leptons": int(len(ww["charged_lepton_indices"])),
                    "n_status23_neutrinos": int(len(ww["neutrino_indices"])),
                    "n_status23_gluons": int(len(ww["gluon_indices"])),
                    "light_quark_indices": json.dumps(ww["light_quark_indices"]),
                    "light_quark_pids": json.dumps(ww["light_quark_pids"]),
                    "light_quark_pts": json.dumps(ww["light_quark_pts"]),
                    "charged_lepton_indices": json.dumps(ww["charged_lepton_indices"]),
                    "charged_lepton_pids": json.dumps(ww["charged_lepton_pids"]),
                    "charged_lepton_pts": json.dumps(ww["charged_lepton_pts"]),
                    "neutrino_indices": json.dumps(ww["neutrino_indices"]),
                    "neutrino_pids": json.dumps(ww["neutrino_pids"]),
                    "neutrino_pts": json.dumps(ww["neutrino_pts"]),
                    "hadronic_target_quark_indices": json.dumps(hadronic_targets),
                    "n_hadronic_target_quarks": int(len(hadronic_targets)),
                    "n_hadronic_target_quarks_matched": int(sum(m["matched"] for m in q_matches)),
                    "all_hadronic_target_quarks_matched": bool(all_q_matched if len(hadronic_targets) > 0 else False),
                    "hadronic_target_quark_matches": json.dumps(q_matches),
                })

            if stop:
                break
        if stop:
            break

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "event_status23_ww_targets.csv", index=False)

    with open(outdir / "counters.json", "w") as f:
        json.dump(counters, f, indent=2)

    mode_counts = df["ww_mode"].value_counts().rename_axis("ww_mode").reset_index(name="n_events")
    mode_counts["fraction"] = mode_counts["n_events"] / max(len(df), 1)
    mode_counts.to_csv(outdir / "ww_mode_counts.csv", index=False)

    match_by_mode = (
        df.groupby("ww_mode")
        .agg(
            n_events=("event_id", "count"),
            hbb_reco_pair_matched_fraction=("hbb_reco_pair_matched", "mean"),
            mean_hadronic_target_quarks=("n_hadronic_target_quarks", "mean"),
            total_hadronic_target_quarks=("n_hadronic_target_quarks", "sum"),
            total_hadronic_target_quarks_matched=("n_hadronic_target_quarks_matched", "sum"),
            all_hadronic_target_quarks_matched_fraction=("all_hadronic_target_quarks_matched", "mean"),
        )
        .reset_index()
    )

    match_by_mode["per_quark_match_fraction"] = (
        match_by_mode["total_hadronic_target_quarks_matched"]
        / match_by_mode["total_hadronic_target_quarks"].replace(0, np.nan)
    )

    match_by_mode.to_csv(outdir / "ww_target_match_by_mode.csv", index=False)

    lines = []
    lines.append("# HH→bbWW status-23 WW-side target diagnostics")
    lines.append("")
    lines.append("This diagnostic classifies the WW side using status-23 hard-process particles, because the available GenPart parent/daughter links are not reliable for following W decays.")
    lines.append("")
    lines.append("## Counters")
    lines.append("")
    for k, v in counters.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## WW decay-mode counts")
    lines.append("")
    lines.append("| WW mode | Events | Fraction |")
    lines.append("|---|---:|---:|")
    for _, r in mode_counts.iterrows():
        lines.append(f"| {r['ww_mode']} | {int(r['n_events'])} | {r['fraction']:.4f} |")
    lines.append("")
    lines.append("## Hadronic target matching by mode")
    lines.append("")
    lines.append("| WW mode | Events | Hbb reco-pair matched fraction | Mean hadronic target quarks | Per-quark match fraction | All hadronic targets matched fraction |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in match_by_mode.iterrows():
        per_q = r["per_quark_match_fraction"]
        per_q_text = "" if pd.isna(per_q) else f"{per_q:.4f}"
        lines.append(
            f"| {r['ww_mode']} | {int(r['n_events'])} | "
            f"{r['hbb_reco_pair_matched_fraction']:.4f} | "
            f"{r['mean_hadronic_target_quarks']:.2f} | "
            f"{per_q_text} | "
            f"{r['all_hadronic_target_quarks_matched_fraction']:.4f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `had_had` means four status-23 light quarks were found after excluding the two H→bb b quarks.")
    lines.append("- `lep_had_*` means two status-23 light quarks plus a charged lepton and neutrino were found.")
    lines.append("- The hadronic target match fraction tells us whether W-side quarks can be associated with retained reco AK4 jets after excluding the truth H→bb jets.")
    lines.append("- This is the right diagnostic for deciding whether to build a semileptonic, fully hadronic, or mixed bbWW assignment target.")
    lines.append("")

    summary = "\n".join(lines) + "\n"
    (outdir / "RESULTS_HHBBWW_STATUS23_WW_TARGET_DIAGNOSTICS.md").write_text(summary)

    summary_dir = Path("Results Summaries")
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "RESULTS_HHBBWW_STATUS23_WW_TARGET_DIAGNOSTICS.md").write_text(summary)

    if len(mode_counts):
        plot_df = mode_counts.sort_values("n_events")
        plt.figure(figsize=(7, 4))
        plt.barh(plot_df["ww_mode"], plot_df["n_events"])
        plt.xlabel("Events")
        plt.title("Status-23 WW decay modes in HH→bbWW")
        plt.tight_layout()
        plt.savefig(plot_dir / "ww_status23_decay_mode_counts.png", dpi=180)
        plt.close()

    plot_df = match_by_mode.dropna(subset=["per_quark_match_fraction"]).copy()
    if len(plot_df):
        plot_df = plot_df.sort_values("per_quark_match_fraction")
        plt.figure(figsize=(7, 4))
        plt.barh(plot_df["ww_mode"], plot_df["per_quark_match_fraction"])
        plt.xlabel("Per-quark reco-match fraction")
        plt.title("Status-23 hadronic W-side target matchability")
        plt.tight_layout()
        plt.savefig(plot_dir / "ww_status23_quark_match_fraction_by_mode.png", dpi=180)
        plt.close()

    print("\n=== Counters ===")
    for k, v in counters.items():
        print(f"{k}: {v}")

    print("\n=== WW decay-mode counts ===")
    print(mode_counts.to_string(index=False))

    print("\n=== W target match by mode ===")
    print(match_by_mode.to_string(index=False))

    print(f"\nSaved outputs in: {outdir}")
    print(f"Saved plots in: {plot_dir}")


if __name__ == "__main__":
    main()