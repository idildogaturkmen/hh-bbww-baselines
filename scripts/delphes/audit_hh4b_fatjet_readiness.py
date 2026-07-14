#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import awkward as ak

STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])

ROOT_DIR = STORE / "root"
OUTDIR = REPO / "outputs/tables/fatjet_readiness_2026_07_14"
OUTDIR.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "ggF_HH4b_SMnorm": ["HH4b_ggf_hh4b_10k_transfer_cluster_84730223_shard_*_pythia8_delphes.root"],
    "VBF_HH4b_SMnorm": ["HH4b_vbf_hh4b_10k_transfer_cluster_3473183_shard_*_pythia8_delphes.root"],
    "ttbar_200k": ["ttbar_200k_shard*_pythia8_delphes.root"],
    "Zbbbb_100k": ["zbbbb_presel_100k_shard*_pythia8_delphes.root"],
    "qcd_iht100to200": ["qcd_bbbb_iht100to200_20000_pythia8_delphes.root"],
    "qcd_iht200to400": [
        "qcd_bbbb_iht200to400_20000_pythia8_delphes.root",
        "qcd_bbbb_iht200to400_extra200k_rootkeep_v1_shard*_pythia8_delphes.root",
    ],
    "qcd_iht400to600": [
        "qcd_bbbb_iht400to600_20000_pythia8_delphes.root",
        "qcd_bbbb_iht400to600_extra100k_rootkeep_v1_shard*_pythia8_delphes.root",
    ],
    "qcd_iht600plus": ["qcd_bbbb_iht600plus_20000_pythia8_delphes.root"],
}

def collect(patterns):
    files = []
    for pat in patterns:
        files.extend(sorted(ROOT_DIR.glob(pat)))
    return files

rows = []

for sample, patterns in SAMPLES.items():
    files = collect(patterns)
    n_events = 0
    n_fat_ge1 = 0
    n_fat_pt200_ge1 = 0
    n_fat_pt300_ge1 = 0
    n_fat_pt300_btag_ge1 = 0
    n_fat_pt300_boosted_ge1 = 0
    fat_counts = []
    leading_pts = []
    leading_masses = []
    leading_sd_masses = []

    print(f"=== {sample}: {len(files)} files ===")

    for f in files:
        with uproot.open(f) as rf:
            t = rf["Delphes"]
            arrays = t.arrays(
                [
                    "FatJet/FatJet.PT",
                    "FatJet/FatJet.Eta",
                    "FatJet/FatJet.Mass",
                    "FatJet/FatJet.BTag",
                    "FatJet/FatJet.BoostedTag",
                    "FatJet/FatJet.SoftDroppedP4[5]",
                ],
                library="ak",
            )

        pt = arrays["FatJet/FatJet.PT"]
        eta = arrays["FatJet/FatJet.Eta"]
        mass = arrays["FatJet/FatJet.Mass"]
        btag = arrays["FatJet/FatJet.BTag"]
        boosted = arrays["FatJet/FatJet.BoostedTag"]

        sel = (pt > 200) & (abs(eta) < 2.5)
        sel300 = (pt > 300) & (abs(eta) < 2.5)
        sel300_b = sel300 & (btag > 0.5)
        sel300_boost = sel300 & (boosted > 0.5)

        n = len(pt)
        n_events += n
        n_fat = ak.num(pt, axis=1)

        n_fat_ge1 += int(ak.sum(n_fat >= 1))
        n_fat_pt200_ge1 += int(ak.sum(ak.any(sel, axis=1)))
        n_fat_pt300_ge1 += int(ak.sum(ak.any(sel300, axis=1)))
        n_fat_pt300_btag_ge1 += int(ak.sum(ak.any(sel300_b, axis=1)))
        n_fat_pt300_boosted_ge1 += int(ak.sum(ak.any(sel300_boost, axis=1)))

        fat_counts.extend(ak.to_numpy(n_fat).tolist())

        # Leading fat jet among pt>200, |eta|<2.5.
        for ev_pt, ev_mass, ev_sel in zip(pt, mass, sel):
            idx = np.where(np.asarray(ev_sel))[0]
            if len(idx) == 0:
                continue
            ev_pt_np = np.asarray(ev_pt)
            lead = idx[np.argmax(ev_pt_np[idx])]
            leading_pts.append(float(ev_pt[lead]))
            leading_masses.append(float(ev_mass[lead]))

    rows.append({
        "sample": sample,
        "n_files": len(files),
        "n_events": n_events,
        "frac_any_fatjet": n_fat_ge1 / n_events if n_events else np.nan,
        "frac_fat_pt200_eta25": n_fat_pt200_ge1 / n_events if n_events else np.nan,
        "frac_fat_pt300_eta25": n_fat_pt300_ge1 / n_events if n_events else np.nan,
        "frac_fat_pt300_btag": n_fat_pt300_btag_ge1 / n_events if n_events else np.nan,
        "frac_fat_pt300_boostedtag": n_fat_pt300_boosted_ge1 / n_events if n_events else np.nan,
        "mean_n_fatjets": float(np.mean(fat_counts)) if fat_counts else 0.0,
        "median_leading_fat_pt": float(np.median(leading_pts)) if leading_pts else np.nan,
        "median_leading_fat_mass": float(np.median(leading_masses)) if leading_masses else np.nan,
    })

df = pd.DataFrame(rows)
df.to_csv(OUTDIR / "fatjet_readiness_summary.csv", index=False)
(OUTDIR / "fatjet_readiness_summary.md").write_text(df.to_markdown(index=False) + "\n")

print(df.to_string(index=False))
print("Wrote:", OUTDIR)
