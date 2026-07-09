'''
Make a table comparing the truth-matched reconstruction performance of different reco methods.
Methods:
- strict_4b_closest_mass: require 4 b-tagged jets, choose pairing closest to 125 GeV.
- CMS_inspired_top_four_btag_priority_mass_balance: choose the four leading b-tagged jets, choose pairing with most balanced masses.
- v2_all_bjet_leading8_closest_mass: choose the four leading b-tagged jets among the leading 8 b-tagged jets,
  choose pairing closest to 125 GeV.
The table includes per-sample and combined-sample statistics:
- candidate efficiency vs. events read
- fraction of candidates with all 4 jets matched to truth b quarks
- fraction of candidates with correct pairing
- median mbb1, mbb2, avg_mbb, mhh   
'''

import itertools
import os
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot


REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])

OUTDIR = REPO / "outputs/tables/hh4b_reconstruction_method_comparison_2026_07_09"
OUTDIR.mkdir(parents=True, exist_ok=True)

BRANCHES = [
    "Particle/Particle.PID",
    "Particle/Particle.M1",
    "Particle/Particle.M2",
    "Particle/Particle.D1",
    "Particle/Particle.D2",
    "Particle/Particle.PT",
    "Particle/Particle.Eta",
    "Particle/Particle.Phi",
    "Jet/Jet.PT",
    "Jet/Jet.Eta",
    "Jet/Jet.Phi",
    "Jet/Jet.Mass",
    "Jet/Jet.BTag",
    "Jet/Jet.Flavor",
]


SIGNAL_ROOTS = {
    "ggF_HH4b": sorted((STORE / "root").glob("HH4b_ggf_hh4b_10k*shard*_pythia8_delphes.root")),
    "VBF_HH4b": sorted((STORE / "root").glob("HH4b_vbf_hh4b_10k*shard*_pythia8_delphes.root")),
}

N_GENERATED = {
    "ggF_HH4b": 10000,
    "VBF_HH4b": 10000,
}


def dphi(phi1, phi2):
    return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))


def dr(eta1, phi1, eta2, phi2):
    return float(np.hypot(eta1 - eta2, dphi(phi1, phi2)))


def p4(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return np.array([e, px, py, pz], dtype=float)


def inv_mass(p4sum):
    e, px, py, pz = p4sum
    m2 = e * e - px * px - py * py - pz * pz
    return float(np.sqrt(max(m2, 0.0)))


def jet_p4(j):
    return p4(j["pt"], j["eta"], j["phi"], j["mass"])


def pair_masses(jets, pairing):
    a, b = pairing
    m1 = inv_mass(jet_p4(jets[a[0]]) + jet_p4(jets[a[1]]))
    m2 = inv_mass(jet_p4(jets[b[0]]) + jet_p4(jets[b[1]]))
    return m1, m2


def choose_pairing(jets, mode):
    pairings = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]

    best = None
    best_score = None

    for pairing in pairings:
        m1, m2 = pair_masses(jets, pairing)

        if mode == "closest_125":
            score = abs(m1 - 125.0) + abs(m2 - 125.0)
        elif mode == "mass_balance":
            score = abs(m1 - m2)
        else:
            raise ValueError(mode)

        if best_score is None or score < best_score:
            best_score = score
            best = pairing

    return best


def candidate_from_jets(jets4, pairing):
    m1, m2 = pair_masses(jets4, pairing)
    hh = sum((jet_p4(j) for j in jets4), start=np.zeros(4))
    return {
        "mbb1": m1,
        "mbb2": m2,
        "avg_mbb": 0.5 * (m1 + m2),
        "delta_mbb": abs(m1 - m2),
        "mhh": inv_mass(hh),
    }


def find_higgs_bb_groups(pid, d1, d2, m1, m2, pt, eta, phi):
    """
    Return list of truth groups:
      [{"h_index": h, "b_indices": [b1,b2]}]

    Primary method: use Higgs direct daughter range D1..D2.
    Fallback: use particles whose M1/M2 equals the Higgs index.
    """
    groups = []

    higgs_indices = np.where(pid == 25)[0]

    for h in higgs_indices:
        daughters = []

        lo = int(d1[h])
        hi = int(d2[h])

        if lo >= 0 and hi >= lo and hi < len(pid):
            for k in range(lo, hi + 1):
                if abs(int(pid[k])) == 5:
                    daughters.append(k)

        if len(daughters) < 2:
            daughters = []
            for k in range(len(pid)):
                if abs(int(pid[k])) == 5 and (int(m1[k]) == h or int(m2[k]) == h):
                    daughters.append(k)

        # Keep the two hardest direct b daughters if there are more than two copies.
        if len(daughters) >= 2:
            daughters = sorted(daughters, key=lambda i: float(pt[i]), reverse=True)[:2]
            groups.append({"h_index": int(h), "b_indices": [int(daughters[0]), int(daughters[1])]})

    # Sometimes there may be duplicate Higgs copies. Keep unique bb daughter pairs.
    unique = []
    seen = set()
    for g in groups:
        key = tuple(sorted(g["b_indices"]))
        if key not in seen:
            seen.add(key)
            unique.append(g)

    return unique[:2]


def match_jets_to_truth_groups(jets4, truth_groups, truth_eta, truth_phi, max_dr=0.4):
    """
    Greedy one-to-one matching of the four selected reco jets to the four H->bb truth b quarks.

    Returns list of group IDs for each jet:
      [0, 0, 1, 1] means jet0/jet1 match Higgs group 0, jet2/jet3 match group 1.
      None means unmatched.
    """
    truth_bs = []
    for gid, g in enumerate(truth_groups):
        for bidx in g["b_indices"]:
            truth_bs.append({
                "group": gid,
                "bidx": bidx,
                "eta": float(truth_eta[bidx]),
                "phi": float(truth_phi[bidx]),
            })

    candidates = []
    for ji, j in enumerate(jets4):
        for ti, tb in enumerate(truth_bs):
            delta = dr(j["eta"], j["phi"], tb["eta"], tb["phi"])
            if delta < max_dr:
                candidates.append((delta, ji, ti))

    candidates.sort()

    used_jets = set()
    used_truth = set()
    jet_group = [None] * len(jets4)

    for delta, ji, ti in candidates:
        if ji in used_jets or ti in used_truth:
            continue
        used_jets.add(ji)
        used_truth.add(ti)
        jet_group[ji] = truth_bs[ti]["group"]

    return jet_group


def pairing_is_correct(pairing, jet_group):
    if any(g is None for g in jet_group):
        return False

    p1, p2 = pairing

    g11, g12 = jet_group[p1[0]], jet_group[p1[1]]
    g21, g22 = jet_group[p2[0]], jet_group[p2[1]]

    return (g11 == g12) and (g21 == g22) and (g11 != g21)


def build_jets(jpt, jeta, jphi, jmass, jbtag, jflav):
    jets = []
    for i in range(len(jpt)):
        jets.append({
            "idx": i,
            "pt": float(jpt[i]),
            "eta": float(jeta[i]),
            "phi": float(jphi[i]),
            "mass": float(jmass[i]),
            "btag": float(jbtag[i]),
            "flavor": int(jflav[i]) if len(jflav) > i else 0,
        })
    return jets


def reco_strict_4b_closest_mass(jets):
    selected = [j for j in jets if j["pt"] > 30.0 and abs(j["eta"]) < 2.5 and j["btag"] > 0]
    selected = sorted(selected, key=lambda j: j["pt"], reverse=True)

    if len(selected) < 4:
        return None

    jets4 = selected[:4]
    pairing = choose_pairing(jets4, "closest_125")
    return jets4, pairing


def reco_cms_top_four_btag_priority_mass_balance(jets):
    selected = [j for j in jets if j["pt"] > 30.0 and abs(j["eta"]) < 2.5]

    if len(selected) < 4:
        return None

    # Delphes BTag is binary, not a DeepJet score.
    # This is therefore "CMS-inspired b-tag priority", not exact DeepJet ranking.
    selected = sorted(selected, key=lambda j: (j["btag"], j["pt"]), reverse=True)

    jets4 = selected[:4]
    pairing = choose_pairing(jets4, "mass_balance")
    return jets4, pairing


def reco_all_bjet_leading8_closest_mass(jets):
    selected = [j for j in jets if j["pt"] > 30.0 and abs(j["eta"]) < 2.5 and j["btag"] > 0]
    selected = sorted(selected, key=lambda j: j["pt"], reverse=True)[:8]

    if len(selected) < 4:
        return None

    best = None
    best_score = None

    for combo in itertools.combinations(range(len(selected)), 4):
        jets4 = [selected[i] for i in combo]
        pairing = choose_pairing(jets4, "closest_125")
        m1, m2 = pair_masses(jets4, pairing)
        score = abs(m1 - 125.0) + abs(m2 - 125.0)

        if best_score is None or score < best_score:
            best_score = score
            best = (jets4, pairing)

    return best


METHODS = {
    "strict_4b_closest_mass": reco_strict_4b_closest_mass,
    "CMS_inspired_top_four_btag_priority_mass_balance": reco_cms_top_four_btag_priority_mass_balance,
    "v2_all_bjet_leading8_closest_mass": reco_all_bjet_leading8_closest_mass,
}


def central68_half_width(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    q16, q84 = np.quantile(x, [0.16, 0.84])
    return 0.5 * (q84 - q16)


def process_signal_sample(sample_name, root_files):
    rows = []
    candidates = {m: [] for m in METHODS}

    total_events = 0
    truth_ok_events = 0

    counters = {
        m: {
            "candidate_events": 0,
            "truth_ok_candidates": 0,
            "all4_matched": 0,
            "correct_pairing": 0,
        }
        for m in METHODS
    }

    if not root_files:
        raise FileNotFoundError(f"No ROOT files for {sample_name}")

    for path in root_files:
        print(f"Processing {sample_name}: {path.name}")

        with uproot.open(path) as f:
            tree = f["Delphes"]
            arr = tree.arrays(BRANCHES, library="ak")

        n_events = len(arr["Jet/Jet.PT"])
        total_events += n_events

        for ievt in range(n_events):
            pid = ak.to_numpy(arr["Particle/Particle.PID"][ievt])
            m1 = ak.to_numpy(arr["Particle/Particle.M1"][ievt])
            m2 = ak.to_numpy(arr["Particle/Particle.M2"][ievt])
            d1 = ak.to_numpy(arr["Particle/Particle.D1"][ievt])
            d2 = ak.to_numpy(arr["Particle/Particle.D2"][ievt])
            ppt = ak.to_numpy(arr["Particle/Particle.PT"][ievt])
            peta = ak.to_numpy(arr["Particle/Particle.Eta"][ievt])
            pphi = ak.to_numpy(arr["Particle/Particle.Phi"][ievt])

            truth_groups = find_higgs_bb_groups(pid, d1, d2, m1, m2, ppt, peta, pphi)
            truth_ok = len(truth_groups) >= 2
            if truth_ok:
                truth_ok_events += 1

            jets = build_jets(
                ak.to_numpy(arr["Jet/Jet.PT"][ievt]),
                ak.to_numpy(arr["Jet/Jet.Eta"][ievt]),
                ak.to_numpy(arr["Jet/Jet.Phi"][ievt]),
                ak.to_numpy(arr["Jet/Jet.Mass"][ievt]),
                ak.to_numpy(arr["Jet/Jet.BTag"][ievt]),
                ak.to_numpy(arr["Jet/Jet.Flavor"][ievt]),
            )

            for method_name, reco_func in METHODS.items():
                reco = reco_func(jets)
                if reco is None:
                    continue

                jets4, pairing = reco
                counters[method_name]["candidate_events"] += 1

                cand = candidate_from_jets(jets4, pairing)
                cand["sample"] = sample_name
                cand["method"] = method_name
                candidates[method_name].append(cand)

                if not truth_ok:
                    continue

                counters[method_name]["truth_ok_candidates"] += 1

                jet_group = match_jets_to_truth_groups(jets4, truth_groups, peta, pphi, max_dr=0.4)

                if all(g is not None for g in jet_group):
                    counters[method_name]["all4_matched"] += 1

                    if pairing_is_correct(pairing, jet_group):
                        counters[method_name]["correct_pairing"] += 1

    for method_name in METHODS:
        c = counters[method_name]
        cand_df = pd.DataFrame(candidates[method_name])

        row = {
            "sample": sample_name,
            "method": method_name,
            "n_generated_expected": N_GENERATED[sample_name],
            "n_events_read": total_events,
            "truth_2hbb_events": truth_ok_events,
            "truth_2hbb_fraction": truth_ok_events / total_events if total_events else np.nan,
            "candidate_events": c["candidate_events"],
            "candidate_efficiency_vs_read": c["candidate_events"] / total_events if total_events else np.nan,
            "truth_ok_candidates": c["truth_ok_candidates"],
            "all4_truth_matched_candidates": c["all4_matched"],
            "all4_truth_matched_fraction_of_candidates": c["all4_matched"] / c["candidate_events"] if c["candidate_events"] else np.nan,
            "correct_pairing_candidates": c["correct_pairing"],
            "correct_pairing_over_all_candidates": c["correct_pairing"] / c["candidate_events"] if c["candidate_events"] else np.nan,
            "correct_pairing_given_all4matched": c["correct_pairing"] / c["all4_matched"] if c["all4_matched"] else np.nan,
            "median_mbb1": cand_df["mbb1"].median() if len(cand_df) else np.nan,
            "median_mbb2": cand_df["mbb2"].median() if len(cand_df) else np.nan,
            "median_avg_mbb": cand_df["avg_mbb"].median() if len(cand_df) else np.nan,
            "avg_mbb_central68_half_width": central68_half_width(cand_df["avg_mbb"]) if len(cand_df) else np.nan,
            "median_abs_avg_mbb_minus_125": np.median(np.abs(cand_df["avg_mbb"] - 125.0)) if len(cand_df) else np.nan,
            "median_mhh": cand_df["mhh"].median() if len(cand_df) else np.nan,
        }
        rows.append(row)

    return rows


def main():
    all_rows = []

    for sample_name, root_files in SIGNAL_ROOTS.items():
        all_rows.extend(process_signal_sample(sample_name, root_files))

    per_sample = pd.DataFrame(all_rows)

    combined_rows = []
    for method, sub in per_sample.groupby("method"):
        total_read = sub["n_events_read"].sum()
        total_candidates = sub["candidate_events"].sum()
        total_all4 = sub["all4_truth_matched_candidates"].sum()
        total_correct = sub["correct_pairing_candidates"].sum()

        combined_rows.append({
            "method": method,
            "samples_combined": ",".join(sub["sample"]),
            "n_events_read_total": total_read,
            "candidate_events_total": total_candidates,
            "candidate_efficiency_combined": total_candidates / total_read if total_read else np.nan,
            "all4_truth_matched_candidates_total": total_all4,
            "all4_truth_matched_fraction_of_candidates": total_all4 / total_candidates if total_candidates else np.nan,
            "correct_pairing_candidates_total": total_correct,
            "correct_pairing_over_all_candidates": total_correct / total_candidates if total_candidates else np.nan,
            "correct_pairing_given_all4matched": total_correct / total_all4 if total_all4 else np.nan,
            "median_candidate_efficiency_across_samples": sub["candidate_efficiency_vs_read"].median(),
            "median_correct_pairing_given_all4matched_across_samples": sub["correct_pairing_given_all4matched"].median(),
            "median_avg_mbb_across_samples": sub["median_avg_mbb"].median(),
            "median_avg_mbb_central68_half_width_across_samples": sub["avg_mbb_central68_half_width"].median(),
        })

    combined = pd.DataFrame(combined_rows).sort_values("method")

    per_sample.to_csv(OUTDIR / "truth_matched_reco_comparison_per_signal_sample.csv", index=False)
    combined.to_csv(OUTDIR / "truth_matched_reco_comparison_combined_signal.csv", index=False)

    (OUTDIR / "truth_matched_reco_comparison_per_signal_sample.md").write_text(
        per_sample.to_markdown(index=False) + "\n"
    )
    (OUTDIR / "truth_matched_reco_comparison_combined_signal.md").write_text(
        combined.to_markdown(index=False) + "\n"
    )

    print("\n=== Truth-matched reco comparison: per signal sample ===")
    print(per_sample.to_string(index=False))

    print("\n=== Truth-matched reco comparison: combined signal ===")
    print(combined.to_string(index=False))

    print(f"\nWrote outputs to: {OUTDIR}")


if __name__ == "__main__":
    main()
