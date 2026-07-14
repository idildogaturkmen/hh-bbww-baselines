#!/usr/bin/env python3

from pathlib import Path
import os
import itertools
import numpy as np
import pandas as pd
import awkward as ak
import uproot

HH4B_STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])

OUTDIR = REPO / "outputs/tables/ak4ak8_mass_plane_smoke_comparison_2026_07_14"
OUTDIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    (
        "ggf_hh4b_smoke20",
        HH4B_STORE / "root/HH4b_ggf_loop_sm_hh4b_smoke_20_pythia8_delphes.root",
        HH4B_STORE / "root_ak4ak8_smoke/ggf_hh4b_smoke20_ak4ak8_delphes.root",
    ),
    (
        "vbf_hh4b_smoke1000",
        HH4B_STORE / "root/HH4b_smoke_vbf_run_03_1000_pythia8_delphes.root",
        HH4B_STORE / "root_ak4ak8_smoke/vbf_hh4b_smoke1000_ak4ak8_delphes.root",
    ),
    (
        "ttbar_10k",
        HH4B_STORE / "root/ttbar_10k_pythia8_delphes.root",
        HH4B_STORE / "root_ak4ak8_smoke/ttbar_10k_ak4ak8_delphes.root",
    ),
    (
        "zbbbb_10k",
        HH4B_STORE / "root/zbbbb_presel_10k_pythia8_delphes.root",
        HH4B_STORE / "root_ak4ak8_smoke/zbbbb_10k_ak4ak8_delphes.root",
    ),
    (
        "qcd_bbbb_10k",
        HH4B_STORE / "root/qcd_bbbb_presel_10k_pythia8_delphes.root",
        HH4B_STORE / "root_ak4ak8_smoke/qcd_bbbb_10k_ak4ak8_delphes.root",
    ),
]


def p4(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return e, px, py, pz


def inv_mass(j1, j2):
    e1, px1, py1, pz1 = p4(*j1)
    e2, px2, py2, pz2 = p4(*j2)
    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    m2 = e * e - px * px - py * py - pz * pz
    return float(np.sqrt(max(m2, 0.0)))


PAIRINGS = [
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
]


def best_pair_masses(jets):
    best = None
    for (a, b), (c, d) in PAIRINGS:
        m1 = inv_mass(jets[a], jets[b])
        m2 = inv_mass(jets[c], jets[d])
        # HH-like objective: both masses close to 125.
        score = abs(m1 - 125.0) + abs(m2 - 125.0)
        if best is None or score < best[0]:
            best = (score, m1, m2)
    m1, m2 = best[1], best[2]
    if m1 < m2:
        return m1, m2
    return m2, m1


def read_candidates(path, label, sample):
    if not path.exists():
        print("[missing]", path)
        return pd.DataFrame()

    with uproot.open(path) as f:
        tree = f["Delphes"]
        arr = tree.arrays(["Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass", "Jet.BTag"], library="ak")

    rows = []
    for iev in range(tree.num_entries):
        pt = np.asarray(arr["Jet.PT"][iev], dtype=float)
        eta = np.asarray(arr["Jet.Eta"][iev], dtype=float)
        phi = np.asarray(arr["Jet.Phi"][iev], dtype=float)
        mass = np.asarray(arr["Jet.Mass"][iev], dtype=float)
        btag = np.asarray(arr["Jet.BTag"][iev], dtype=float)

        mask = (pt > 20.0) & (np.abs(eta) < 2.5) & (btag > 0)
        idx = np.where(mask)[0]

        if len(idx) < 4:
            continue

        # Delphes btag is mostly binary, so sort b-tagged jets by pT.
        idx = idx[np.argsort(pt[idx])[::-1]][:4]

        jets = [(pt[i], eta[i], phi[i], mass[i]) for i in idx]
        mbb1, mbb2 = best_pair_masses(jets)

        rows.append({
            "sample": sample,
            "version": label,
            "event": iev,
            "mbb1": mbb1,
            "mbb2": mbb2,
            "mbb_avg": 0.5 * (mbb1 + mbb2),
            "mass_distance_hh": abs(mbb1 - 125.0) + abs(mbb2 - 125.0),
            "in_hh_window_90_150": (90.0 <= mbb1 <= 150.0) and (90.0 <= mbb2 <= 150.0),
        })

    return pd.DataFrame(rows)


all_rows = []
for sample, old_path, new_path in PAIRS:
    all_rows.append(read_candidates(old_path, "R05", sample))
    all_rows.append(read_candidates(new_path, "AK4AK8", sample))

cand = pd.concat(all_rows, ignore_index=True)
cand.to_csv(OUTDIR / "r05_vs_ak4ak8_mass_plane_candidates.csv", index=False)

summary = (
    cand.groupby(["sample", "version"])
    .agg(
        n_candidates=("event", "count"),
        median_mbb1=("mbb1", "median"),
        median_mbb2=("mbb2", "median"),
        median_mbb_avg=("mbb_avg", "median"),
        median_mass_distance_hh=("mass_distance_hh", "median"),
        frac_hh_window_90_150=("in_hh_window_90_150", "mean"),
    )
    .reset_index()
)

summary.to_csv(OUTDIR / "r05_vs_ak4ak8_mass_plane_summary.csv", index=False)
(OUTDIR / "r05_vs_ak4ak8_mass_plane_summary.md").write_text(summary.to_markdown(index=False) + "\n")

print(summary.to_string(index=False))
print("Wrote:", OUTDIR)
