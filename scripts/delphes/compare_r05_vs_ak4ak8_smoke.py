#!/usr/bin/env python3

from pathlib import Path
import os
import numpy as np
import pandas as pd
import awkward as ak
import uproot

HH4B_STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])

OUTDIR = REPO / "outputs/tables/ak4ak8_smoke_comparison_2026_07_14"
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


def summarize_file(path):
    if not path.exists():
        return None

    with uproot.open(path) as f:
        tree = f["Delphes"]
        arr = tree.arrays(["Jet.PT", "Jet.Eta", "Jet.BTag", "FatJet.PT", "FatJet.Eta"], library="ak")

    jet_mask = (arr["Jet.PT"] > 20) & (abs(arr["Jet.Eta"]) < 2.5)
    fat_mask = (arr["FatJet.PT"] > 200) & (abs(arr["FatJet.Eta"]) < 2.5)

    njet = ak.to_numpy(ak.sum(jet_mask, axis=1))
    nbtag = ak.to_numpy(ak.sum(jet_mask & (arr["Jet.BTag"] > 0), axis=1))
    ht = ak.to_numpy(ak.sum(arr["Jet.PT"][jet_mask], axis=1))
    nfat = ak.to_numpy(ak.sum(fat_mask, axis=1))

    return {
        "n_events": int(len(njet)),
        "frac_ge4_jets": float(np.mean(njet >= 4)),
        "frac_ge4_btags": float(np.mean(nbtag >= 4)),
        "mean_njets": float(np.mean(njet)),
        "mean_nbtags": float(np.mean(nbtag)),
        "median_ht": float(np.median(ht)),
        "frac_any_fatjet": float(np.mean(nfat >= 1)),
    }


rows = []

for sample, old_path, new_path in PAIRS:
    old = summarize_file(old_path)
    new = summarize_file(new_path)

    if old is None:
        print("[skip old missing]", old_path)
        continue
    if new is None:
        print("[skip new missing]", new_path)
        continue

    row = {"sample": sample}
    for key, val in old.items():
        row[f"R05_{key}"] = val
    for key, val in new.items():
        row[f"AK4AK8_{key}"] = val

    row["delta_frac_ge4_jets"] = row["AK4AK8_frac_ge4_jets"] - row["R05_frac_ge4_jets"]
    row["delta_frac_ge4_btags"] = row["AK4AK8_frac_ge4_btags"] - row["R05_frac_ge4_btags"]
    row["delta_median_ht"] = row["AK4AK8_median_ht"] - row["R05_median_ht"]

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUTDIR / "r05_vs_ak4ak8_smoke_comparison.csv", index=False)
(OUTDIR / "r05_vs_ak4ak8_smoke_comparison.md").write_text(df.to_markdown(index=False) + "\n")

print(df.to_string(index=False))
print("Wrote:", OUTDIR)
