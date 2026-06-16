import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset


REPO_ID = "fastmachinelearning/collide-1m"
HH_NPZ_DIR = Path("outputs/hbb_npz")
OUTDIR = Path("outputs/zbb_diagnostics")
PLOT_DIR = Path("outputs/plots/zbb_diagnostics")
M_Z = 91.1876
M_H = 125.0
MAX_JETS = 12

ZBB_FILES = [
    f"ZJetsTobb_13TeV-madgraphMLM-pythia8/"
    f"ZJetsTobb_13TeV-madgraphMLM-pythia8-NEVENT10000-RS520000{i:02d}.parquet"
    for i in range(1, 26)
]

OUTDIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


JET_COLUMNS = [
    "FullReco_JetAK4_PT",
    "FullReco_JetAK4_Eta",
    "FullReco_JetAK4_Phi",
    "FullReco_JetAK4_Mass",
    "FullReco_JetAK4_BTag",
    "FullReco_JetAK4_BTagPhys",
]

GEN_COLUMNS = [
    "FullReco_GenPart_PID",
    "FullReco_GenPart_Status",
    "FullReco_GenPart_PT",
    "FullReco_GenPart_Eta",
    "FullReco_GenPart_Phi",
    "FullReco_GenPart_M1",
    "FullReco_GenPart_M2",
]


METHOD_LABELS = {
    "z_truth_matched": "Z->bb truth-matched jets",
    "hh_truth_matched": "H->bb truth-matched jets",
    "top2_btag": "top-2 b-tag jets",
    "closest_z_mass": "m_jj closest to Z mass",
    "closest_h_mass": "m_jj closest to Higgs mass",
}


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi <= -math.pi:
        dphi += 2 * math.pi
    return dphi


def delta_r_eta_phi(eta1, phi1, eta2, phi2):
    return math.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2)


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


def pair_key(pair):
    return tuple(sorted((int(pair[0]), int(pair[1]))))


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p16": float(np.percentile(values, 16)),
        "p84": float(np.percentile(values, 84)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "frac_70_110": float(np.mean((values >= 70.0) & (values <= 110.0))),
        "frac_80_120": float(np.mean((values >= 80.0) & (values <= 120.0))),
        "frac_90_140": float(np.mean((values >= 90.0) & (values <= 140.0))),
        "frac_100_150": float(np.mean((values >= 100.0) & (values <= 150.0))),
    }


def build_jets(event, max_jets=MAX_JETS):
    n_jets_raw = len(event["FullReco_JetAK4_PT"])
    n_jets = min(n_jets_raw, max_jets)
    jets = np.zeros((n_jets, 6), dtype=np.float32)
    for idx, col in enumerate(JET_COLUMNS):
        jets[:, idx] = event[col][:n_jets]
    return jets, n_jets_raw


def find_first_ancestor_with_abs_pid(event, start_idx, abs_pid, max_depth=30):
    pids = event["FullReco_GenPart_PID"]
    parents = [event["FullReco_GenPart_M1"], event["FullReco_GenPart_M2"]]
    stack = []
    for parent_col in parents:
        parent = int(parent_col[start_idx])
        if parent >= 0:
            stack.append(parent)

    seen = set()
    depth = 0
    while stack and depth < max_depth:
        idx = stack.pop()
        depth += 1
        if idx in seen or idx < 0 or idx >= len(pids):
            continue
        seen.add(idx)
        if abs(int(pids[idx])) == abs_pid:
            return idx
        for parent_col in parents:
            parent = int(parent_col[idx])
            if parent >= 0 and parent not in seen:
                stack.append(parent)
    return None


def find_zbb_bquarks(event):
    pids = event["FullReco_GenPart_PID"]
    statuses = event["FullReco_GenPart_Status"]
    pts = event["FullReco_GenPart_PT"]

    rows = []
    for idx, pid in enumerate(pids):
        if abs(int(pid)) != 5 or int(statuses[idx]) != 23:
            continue
        z_ancestor = find_first_ancestor_with_abs_pid(event, idx, 23)
        rows.append({"idx": idx, "pid": int(pid), "pt": float(pts[idx]), "z_ancestor": z_ancestor})

    by_z = {}
    for row in rows:
        if row["z_ancestor"] is None:
            continue
        by_z.setdefault(row["z_ancestor"], []).append(row)

    viable_groups = []
    for z_idx, group in by_z.items():
        if len(group) < 2:
            continue
        has_b = any(row["pid"] == 5 for row in group)
        has_bbar = any(row["pid"] == -5 for row in group)
        if has_b and has_bbar:
            viable_groups.append((z_idx, group))

    if viable_groups:
        _, group = max(viable_groups, key=lambda item: sum(row["pt"] for row in item[1]))
        selected = sorted(group, key=lambda row: row["pt"], reverse=True)[:2]
        return [row["idx"] for row in selected], "status23_with_z_ancestor"

    preferred = [row for row in rows if row["z_ancestor"] is not None]
    if len(preferred) >= 2:
        selected = sorted(preferred, key=lambda row: row["pt"], reverse=True)[:2]
        return [row["idx"] for row in selected], "status23_any_z_ancestor"

    if len(rows) >= 2:
        selected = sorted(rows, key=lambda row: row["pt"], reverse=True)[:2]
        return [row["idx"] for row in selected], "status23_no_z_ancestor_fallback"

    return [], "no_two_status23_bquarks"


def match_bquarks_to_jets(event, b_indices, max_dr, max_jets=MAX_JETS):
    matches = []
    n_jets = min(len(event["FullReco_JetAK4_PT"]), max_jets)

    for bidx in b_indices:
        b_eta = event["FullReco_GenPart_Eta"][bidx]
        b_phi = event["FullReco_GenPart_Phi"][bidx]

        best_j = None
        best_dr = float("inf")
        for j in range(n_jets):
            dr = delta_r_eta_phi(
                b_eta,
                b_phi,
                event["FullReco_JetAK4_Eta"][j],
                event["FullReco_JetAK4_Phi"][j],
            )
            if dr < best_dr:
                best_dr = dr
                best_j = j

        matches.append(best_j if best_dr < max_dr else -1)

    return matches


def ranked_pair(jets, method):
    if len(jets) < 2:
        return None

    if method == "top2_btag":
        btags = jets[:, 4]
        order = np.argsort(btags)[-2:]
        return pair_key(order)

    target_mass = M_Z if method == "closest_z_mass" else M_H
    best_pair = None
    best_score = float("inf")
    for i, j in itertools.combinations(range(len(jets)), 2):
        score = abs(invariant_mass(jets[i], jets[j]) - target_mass)
        if score < best_score:
            best_score = score
            best_pair = pair_key((i, j))
    return best_pair


def load_hh_matched_mbb():
    values = []
    for split in ["train", "val", "test"]:
        data = np.load(HH_NPZ_DIR / f"{split}.npz")
        jets_all = data["jets"]
        labels_all = data["labels"]
        for jets, labels in zip(jets_all, labels_all):
            i, j = int(labels[0]), int(labels[1])
            values.append(invariant_mass(jets[i], jets[j]))
    return np.asarray(values, dtype=np.float64)


def iter_zbb_events(n_files):
    files = ZBB_FILES[:n_files]
    ds = load_dataset(REPO_ID, data_files={"train": files}, split="train", streaming=True)
    yield from ds


def build_zbb_event_table(args):
    rows = []
    counts = {
        "scanned": 0,
        "two_bquarks": 0,
        "matched_two": 0,
        "matched_distinct": 0,
        "usable": 0,
        "lt2_retained_jets": 0,
        "same_matched_jet": 0,
        "unmatched": 0,
    }
    label_modes = {}

    for event in iter_zbb_events(args.n_files):
        counts["scanned"] += 1
        if args.max_scan is not None and counts["scanned"] > args.max_scan:
            break

        jets, n_jets_raw = build_jets(event, args.max_jets)
        if len(jets) < 2:
            counts["lt2_retained_jets"] += 1
            continue

        b_indices, label_mode = find_zbb_bquarks(event)
        label_modes[label_mode] = label_modes.get(label_mode, 0) + 1
        if len(b_indices) != 2:
            continue
        counts["two_bquarks"] += 1

        matches = match_bquarks_to_jets(event, b_indices, args.max_dr, args.max_jets)
        if len(matches) != 2 or matches[0] < 0 or matches[1] < 0:
            counts["unmatched"] += 1
            continue
        counts["matched_two"] += 1

        if matches[0] == matches[1]:
            counts["same_matched_jet"] += 1
            continue
        counts["matched_distinct"] += 1

        truth = pair_key(matches)
        truth_mbb = invariant_mass(jets[truth[0]], jets[truth[1]])
        truth_dr = delta_r(jets[truth[0]], jets[truth[1]])
        truth_min_pt = float(min(jets[truth[0], 0], jets[truth[1], 0]))

        method_pairs = {
            "top2_btag": ranked_pair(jets, "top2_btag"),
            "closest_z_mass": ranked_pair(jets, "closest_z_mass"),
            "closest_h_mass": ranked_pair(jets, "closest_h_mass"),
        }
        method_mbb = {
            name: invariant_mass(jets[pair[0]], jets[pair[1]]) if pair is not None else float("nan")
            for name, pair in method_pairs.items()
        }

        rows.append(
            {
                "event": counts["scanned"] - 1,
                "label_mode": label_mode,
                "n_jets_raw": int(n_jets_raw),
                "n_jets_retained": int(len(jets)),
                "b0_gen_idx": int(b_indices[0]),
                "b1_gen_idx": int(b_indices[1]),
                "matched_i": int(truth[0]),
                "matched_j": int(truth[1]),
                "matched_mbb": float(truth_mbb),
                "matched_deltaR": float(truth_dr),
                "matched_min_pt": truth_min_pt,
                "top2_btag_mbb": float(method_mbb["top2_btag"]),
                "top2_btag_correct": method_pairs["top2_btag"] == truth,
                "closest_z_mass_mbb": float(method_mbb["closest_z_mass"]),
                "closest_z_mass_correct": method_pairs["closest_z_mass"] == truth,
                "closest_h_mass_mbb": float(method_mbb["closest_h_mass"]),
                "closest_h_mass_correct": method_pairs["closest_h_mass"] == truth,
            }
        )
        counts["usable"] += 1

        if counts["usable"] % args.progress_every == 0:
            print(
                f"usable Z->bb events: {counts['usable']}; scanned: {counts['scanned']}",
                flush=True,
            )

        if counts["usable"] >= args.target_usable:
            break

    counts["label_modes"] = label_modes
    return pd.DataFrame(rows), counts


def build_comparison_table(hh_mbb, zbb_df):
    rows = []
    rows.append({"sample": "HH_bbWW", "method": "truth_matched", **summarize_values(hh_mbb)})
    rows.append({"sample": "ZJetsTobb", "method": "truth_matched", **summarize_values(zbb_df["matched_mbb"])})

    for method, col in [
        ("top2_btag", "top2_btag_mbb"),
        ("closest_z_mass", "closest_z_mass_mbb"),
        ("closest_h_mass", "closest_h_mass_mbb"),
    ]:
        rows.append({"sample": "ZJetsTobb", "method": method, **summarize_values(zbb_df[col])})

    return pd.DataFrame(rows)


def plot_hh_vs_zbb(hh_mbb, zbb_mbb):
    bins = np.linspace(0.0, 240.0, 80)
    plt.figure(figsize=(7.5, 5.0))
    plt.hist(hh_mbb, bins=bins, histtype="step", linewidth=1.6, density=True, label="HH truth-matched H->bb")
    plt.hist(zbb_mbb, bins=bins, histtype="step", linewidth=1.6, density=True, label="Z+jets truth-matched Z->bb")
    plt.axvline(M_Z, color="tab:blue", linestyle="--", linewidth=1.0, label="Z mass")
    plt.axvline(M_H, color="black", linestyle="--", linewidth=1.0, label="H mass")
    plt.xlabel("Truth-matched reconstructed $m_{bb}$ [GeV]")
    plt.ylabel("Normalized events")
    plt.title("Truth-matched reconstructed bb mass: HH vs Z->bb")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "hh_vs_zbb_truth_matched_mbb.png", dpi=200)
    plt.close()


def plot_zbb_methods(zbb_df):
    bins = np.linspace(0.0, 240.0, 80)
    plt.figure(figsize=(7.5, 5.0))
    for col, label in [
        ("matched_mbb", "truth-matched Z->bb"),
        ("top2_btag_mbb", "top-2 b-tag"),
        ("closest_z_mass_mbb", "closest to Z mass"),
        ("closest_h_mass_mbb", "closest to Higgs mass"),
    ]:
        plt.hist(zbb_df[col], bins=bins, histtype="step", linewidth=1.5, density=True, label=label)
    plt.axvline(M_Z, color="tab:blue", linestyle="--", linewidth=1.0, label="Z mass")
    plt.axvline(M_H, color="black", linestyle="--", linewidth=1.0, label="H mass")
    plt.xlabel("Selected pair $m_{bb}$ [GeV]")
    plt.ylabel("Normalized events")
    plt.title("ZJetsTobb selected pair mass diagnostics")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "zbb_pair_method_mbb.png", dpi=200)
    plt.close()


def write_markdown(summary, comparison_df, counts, args):
    hh = summary["hh_truth_matched_mbb"]
    z = summary["zbb_truth_matched_mbb"]
    delta_mean = z["mean"] - hh["mean"]
    delta_median = z["median"] - hh["median"]

    lines = [
        "# Z->bb m_bb Diagnostics",
        "",
        "This note compares the truth-matched reconstructed bb mass in the COLLIDE-1M `ZJetsTobb_13TeV-madgraphMLM-pythia8` sample to the current HH -> bbWW truth-matched H -> bb mass diagnostic.",
        "",
        "## Command",
        "",
        "```bash",
        f"python scripts/diagnose_zbb_mbb.py --n-files {args.n_files} --target-usable {args.target_usable} --max-scan {args.max_scan} --max-jets {args.max_jets} --max-dr {args.max_dr}",
        "```",
        "",
        "## Matching Definition",
        "",
        "- Z sample: status-23 generator b quarks are selected, preferring candidates with a Z-boson ancestor in the generator record.",
        "- The two selected generator b quarks are matched to distinct retained AK4 jets with DeltaR < max_dr.",
        "- The same MAX_JETS cap is used as in the HH preprocessing.",
        "- The reconstructed mass is computed from the matched AK4 jet four-vectors, not from the generator b quarks.",
        "",
        "## Event Counts",
        "",
        f"- scanned events: {counts['scanned']}",
        f"- usable matched Z->bb events: {counts['usable']}",
        f"- events with two selected b quarks: {counts['two_bquarks']}",
        f"- events with two matched b quarks: {counts['matched_two']}",
        f"- events with distinct matched AK4 jets: {counts['matched_distinct']}",
        f"- label-mode counts: `{counts.get('label_modes', {})}`",
        "",
        "Truth-record caveat: the script tries to identify b quarks with a resolvable Z-boson ancestor, but the flattened COLLIDE generator record does not always make `M1/M2` usable as direct array indices. When the ancestor cannot be resolved, it falls back to the same practical convention used for the HH preprocessing: the two status-23 b quarks.",
        "",
        "## Truth-Matched m_bb Comparison",
        "",
        "| Sample | Mean [GeV] | Median [GeV] | 16-84% [GeV] | Fraction 80-120 | Fraction 90-140 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| HH truth-matched H->bb | {hh['mean']:.2f} | {hh['median']:.2f} | {hh['p16']:.2f}-{hh['p84']:.2f} | {hh['frac_80_120']:.3f} | {hh['frac_90_140']:.3f} |",
        f"| ZJetsTobb truth-matched Z->bb | {z['mean']:.2f} | {z['median']:.2f} | {z['p16']:.2f}-{z['p84']:.2f} | {z['frac_80_120']:.3f} | {z['frac_90_140']:.3f} |",
        "",
        f"Relative to the current HH truth-matched distribution, the Z->bb matched mean shifts by {delta_mean:+.2f} GeV and the median shifts by {delta_median:+.2f} GeV.",
        "",
        "## Z Pair Selection Cross-Checks",
        "",
        "| Method | Mean [GeV] | Median [GeV] | 16-84% [GeV] | Truth-pair accuracy |",
        "|---|---:|---:|---:|---:|",
    ]

    z_rows = comparison_df[comparison_df["sample"] == "ZJetsTobb"]
    for _, row in z_rows.iterrows():
        method = row["method"]
        if method == "truth_matched":
            acc = 1.0
        else:
            acc = summary["zbb_method_accuracy"].get(method, float("nan"))
        lines.append(
            f"| {method} | {row['mean']:.2f} | {row['median']:.2f} | "
            f"{row['p16']:.2f}-{row['p84']:.2f} | {acc:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A Z-mass veto or Z-vs-H classifier should be evaluated as a signal-efficiency tradeoff, not assumed to be safe. The current HH truth-matched H->bb mass is already shifted below 125 GeV, so any rejection using only m_bb can remove genuine HH signal as well as Z->bb background.",
            "",
            "## Outputs",
            "",
            "- `outputs/zbb_diagnostics/zbb_event_diagnostics.csv`",
            "- `outputs/zbb_diagnostics/zbb_mbb_comparison.csv`",
            "- `outputs/zbb_diagnostics/zbb_mbb_summary.json`",
            "- `outputs/plots/zbb_diagnostics/hh_vs_zbb_truth_matched_mbb.png`",
            "- `outputs/plots/zbb_diagnostics/zbb_pair_method_mbb.png`",
        ]
    )

    Path("RESULTS_ZBB_MBB_DIAGNOSTICS.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose reconstructed m_bb in COLLIDE-1M Z->bb events.")
    parser.add_argument("--n-files", type=int, default=1, help="Number of ZJetsTobb parquet shards to stream.")
    parser.add_argument("--target-usable", type=int, default=3000, help="Stop after this many usable matched Z->bb events.")
    parser.add_argument("--max-scan", type=int, default=10000, help="Maximum number of Z events to scan; use -1 for no cap.")
    parser.add_argument("--max-jets", type=int, default=MAX_JETS)
    parser.add_argument("--max-dr", type=float, default=0.4)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    if args.max_scan is not None and args.max_scan < 0:
        args.max_scan = None
    if args.n_files < 1 or args.n_files > len(ZBB_FILES):
        raise ValueError(f"--n-files must be between 1 and {len(ZBB_FILES)}")
    return args


def main():
    args = parse_args()
    print("Loading current HH truth-matched m_bb values...")
    hh_mbb = load_hh_matched_mbb()
    print(f"HH events: {len(hh_mbb)}; mean={np.mean(hh_mbb):.2f} GeV; median={np.median(hh_mbb):.2f} GeV")

    print("Streaming ZJetsTobb events and matching Z->bb quarks to AK4 jets...")
    zbb_df, counts = build_zbb_event_table(args)
    if zbb_df.empty:
        raise RuntimeError("No usable matched Z->bb events found. Check the truth-label definition and input files.")

    zbb_df.to_csv(OUTDIR / "zbb_event_diagnostics.csv", index=False)
    comparison_df = build_comparison_table(hh_mbb, zbb_df)
    comparison_df.to_csv(OUTDIR / "zbb_mbb_comparison.csv", index=False)

    summary = {
        "args": vars(args),
        "counts": counts,
        "hh_truth_matched_mbb": summarize_values(hh_mbb),
        "zbb_truth_matched_mbb": summarize_values(zbb_df["matched_mbb"]),
        "zbb_method_accuracy": {
            "top2_btag": float(np.mean(zbb_df["top2_btag_correct"])),
            "closest_z_mass": float(np.mean(zbb_df["closest_z_mass_correct"])),
            "closest_h_mass": float(np.mean(zbb_df["closest_h_mass_correct"])),
        },
    }
    with open(OUTDIR / "zbb_mbb_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_hh_vs_zbb(hh_mbb, zbb_df["matched_mbb"].to_numpy())
    plot_zbb_methods(zbb_df)
    write_markdown(summary, comparison_df, counts, args)

    print("\nZ->bb m_bb comparison:")
    print(comparison_df.to_string(index=False))
    print("\nZ pair method accuracies:")
    print(json.dumps(summary["zbb_method_accuracy"], indent=2))
    print("\nCounts:")
    print(json.dumps(counts, indent=2))
    print(f"\nSaved tables to {OUTDIR}")
    print(f"Saved plots to {PLOT_DIR}")
    print("Saved markdown summary to RESULTS_ZBB_MBB_DIAGNOSTICS.md")


if __name__ == "__main__":
    main()
