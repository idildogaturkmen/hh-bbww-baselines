#!/usr/bin/env python3

from pathlib import Path
import os
import numpy as np
import pandas as pd
import uproot

STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])

OUTDIR = REPO / "outputs/tables/zz_zh_hh_mass_hypothesis_validation_ak4ak8_2026_07_14"
OUTDIR.mkdir(parents=True, exist_ok=True)

INPUTS = [
    (
        "HH_ggF_smoke20",
        "HH",
        STORE / "root_ak4ak8_smoke/ggf_hh4b_smoke20_ak4ak8_delphes.root",
    ),
    (
        "HH_VBF_smoke1000",
        "HH",
        STORE / "root_ak4ak8_smoke/vbf_hh4b_smoke1000_ak4ak8_delphes.root",
    ),
    (
        "ZZ4b_pilot10k",
        "ZZ",
        STORE / "root/zz4b_ak4ak8_pilot10k_pythia8_delphes.root",
    ),
    (
        "ZH4b_pilot10k",
        "ZH",
        STORE / "root/zh4b_ak4ak8_pilot10k_pythia8_delphes.root",
    ),
]

MZ = 91.1876
MH = 125.0

HYPOTHESES = {
    "HH": [(MH, MH)],
    "ZZ": [(MZ, MZ)],
    "ZH": [(MZ, MH), (MH, MZ)],
}

PAIRINGS = [
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
]


def read_branch(tree, names):
    for name in names:
        try:
            return tree[name].array(library="ak")
        except Exception:
            pass
    raise KeyError(f"Could not find any of {names}")


def p4(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return np.array([e, px, py, pz], dtype=float)


def invmass(a, b):
    q = a + b
    m2 = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
    return float(np.sqrt(max(m2, 0.0)))


def best_pairing_for_hypothesis(jets, hypothesis):
    target_pairs = HYPOTHESES[hypothesis]
    best = None

    for pairing_index, ((a, b), (c, d)) in enumerate(PAIRINGS):
        m1 = invmass(jets[a], jets[b])
        m2 = invmass(jets[c], jets[d])

        for t1, t2 in target_pairs:
            dist = abs(m1 - t1) + abs(m2 - t2)
            if best is None or dist < best["mass_distance"]:
                best = {
                    "hypothesis": hypothesis,
                    "pairing_index": pairing_index,
                    "mbb1": m1,
                    "mbb2": m2,
                    "target1": t1,
                    "target2": t2,
                    "mass_distance": dist,
                }

    return best


def window_flags(m1, m2):
    masses = [m1, m2]
    hh = all(90.0 <= m <= 150.0 for m in masses)
    zz = all(70.0 <= m <= 110.0 for m in masses)
    zh = (
        (70.0 <= m1 <= 110.0 and 100.0 <= m2 <= 150.0)
        or (70.0 <= m2 <= 110.0 and 100.0 <= m1 <= 150.0)
    )
    return hh, zz, zh


rows = []

for sample, truth, path in INPUTS:
    print(f"Reading {sample}: {path}")
    if not path.exists():
        print(f"  MISSING: {path}")
        continue

    with uproot.open(path) as f:
        tree = f["Delphes"]
        jet_pt = read_branch(tree, ["Jet.PT", "Jet/Jet.PT"])
        jet_eta = read_branch(tree, ["Jet.Eta", "Jet/Jet.Eta"])
        jet_phi = read_branch(tree, ["Jet.Phi", "Jet/Jet.Phi"])
        jet_mass = read_branch(tree, ["Jet.Mass", "Jet/Jet.Mass"])
        jet_btag = read_branch(tree, ["Jet.BTag", "Jet/Jet.BTag"])

    n_events = len(jet_pt)

    for iev in range(n_events):
        pt = np.asarray(jet_pt[iev], dtype=float)
        eta = np.asarray(jet_eta[iev], dtype=float)
        phi = np.asarray(jet_phi[iev], dtype=float)
        mass = np.asarray(jet_mass[iev], dtype=float)
        btag = np.asarray(jet_btag[iev], dtype=float)

        keep = np.where((pt > 20.0) & (np.abs(eta) < 2.5) & (btag > 0))[0]
        if len(keep) < 4:
            continue

        # Use leading four selected b-tagged jets by pT.
        keep = keep[np.argsort(pt[keep])[::-1]][:4]

        jets = [
            p4(pt[i], eta[i], phi[i], mass[i])
            for i in keep
        ]

        for hyp in ["HH", "ZZ", "ZH"]:
            best = best_pairing_for_hypothesis(jets, hyp)
            hh_window, zz_window, zh_window = window_flags(best["mbb1"], best["mbb2"])
            rows.append({
                "sample": sample,
                "truth": truth,
                "event": iev,
                "hypothesis": hyp,
                "pairing_index": best["pairing_index"],
                "mbb1": best["mbb1"],
                "mbb2": best["mbb2"],
                "target1": best["target1"],
                "target2": best["target2"],
                "mass_distance": best["mass_distance"],
                "hh_window_90_150": hh_window,
                "zz_window_70_110": zz_window,
                "zh_window_z70_110_h100_150": zh_window,
            })

df = pd.DataFrame(rows)
df.to_csv(OUTDIR / "zz_zh_hh_mass_hypothesis_candidates.csv", index=False)

summary = (
    df.groupby(["sample", "truth", "hypothesis"], as_index=False)
    .agg(
        n_candidates=("event", "count"),
        median_mbb1=("mbb1", "median"),
        median_mbb2=("mbb2", "median"),
        median_mass_distance=("mass_distance", "median"),
        frac_hh_window_90_150=("hh_window_90_150", "mean"),
        frac_zz_window_70_110=("zz_window_70_110", "mean"),
        frac_zh_window=("zh_window_z70_110_h100_150", "mean"),
    )
)

summary.to_csv(OUTDIR / "zz_zh_hh_mass_hypothesis_summary.csv", index=False)
(OUTDIR / "zz_zh_hh_mass_hypothesis_summary.md").write_text(
    summary.to_markdown(index=False) + "\n"
)

# Per-event best hypothesis by distance.
best = (
    df.sort_values("mass_distance")
    .groupby(["sample", "truth", "event"], as_index=False)
    .first()
)

best_counts = (
    best.groupby(["sample", "truth", "hypothesis"], as_index=False)
    .agg(n_events=("event", "count"))
)

totals = best.groupby(["sample", "truth"], as_index=False).agg(total_events=("event", "count"))
best_counts = best_counts.merge(totals, on=["sample", "truth"], how="left")
best_counts["fraction"] = best_counts["n_events"] / best_counts["total_events"]

best_counts.to_csv(OUTDIR / "zz_zh_hh_best_hypothesis_fractions.csv", index=False)
(OUTDIR / "zz_zh_hh_best_hypothesis_fractions.md").write_text(
    best_counts.to_markdown(index=False) + "\n"
)

print("\n=== Hypothesis summary ===")
print(summary.to_string(index=False))

print("\n=== Best-hypothesis fractions ===")
print(best_counts.to_string(index=False))

print("\nWrote:", OUTDIR)
