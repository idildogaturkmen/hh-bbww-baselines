#!/usr/bin/env python3

from pathlib import Path
import json
import os
import re

import awkward as ak
import numpy as np
import pandas as pd
import uproot

REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])

OUTDIR = REPO / "outputs/audits/ak4ak8_btag_information_2026_07_15"
OUTDIR.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "ggF_HH": STORE / "root/ggf_hh4b_ak4ak8_10k_pythia8_delphes.root",
    "QCD_bbbb": (
        STORE
        / "root/qcd_bbbb_ak4ak8_extra50k_v1_shard000_pythia8_delphes.root"
    ),
    "ttbar": (
        STORE
        / "root/ttbar_extra50k_ak4ak8_v1_shard000_pythia8_delphes.root"
    ),
    "Zbbbb": (
        STORE
        / "root/zbbbb_ak4ak8_extra50k_v1_shard000_pythia8_delphes.root"
    ),
}

rows = []
inventory = {}

for sample, path in SAMPLES.items():
    if not path.exists():
        print("Missing:", path)
        continue

    print("Reading:", sample, path)

    with uproot.open(path) as root_file:
        tree = root_file["Delphes"]

        # Delphes split branches commonly appear as:
        # Jet/Jet.BTag, FatJet/FatJet.BTag, etc.
        all_paths = sorted({
            str(key).split(";")[0]
            for key in tree.keys(recursive=True)
        })

        discovered = []

        for branch_path in all_paths:
            leaf = branch_path.rsplit("/", 1)[-1]

            if not re.match(r"^(Jet|FatJet)\.", leaf):
                continue

            if "BTag" not in leaf and "BoostedTag" not in leaf:
                continue

            discovered.append({
                "path": branch_path,
                "leaf": leaf,
            })

        inventory[sample] = discovered

        if not discovered:
            print("WARNING: no tag branches found")
            print("Other tag-like paths:")
            for key in all_paths:
                if "tag" in key.lower():
                    print(" ", repr(key))

        for item in discovered:
            branch_path = item["path"]
            leaf = item["leaf"]

            try:
                array = tree[branch_path].array(
                    entry_stop=10000,
                    library="ak",
                )

                flat = np.asarray(
                    ak.to_numpy(ak.flatten(array, axis=None))
                )

                if flat.size:
                    try:
                        flat = flat[np.isfinite(flat)]
                    except TypeError:
                        pass

                if flat.size == 0:
                    rows.append({
                        "sample": sample,
                        "branch_path": branch_path,
                        "branch": leaf,
                        "n_values": 0,
                        "n_unique": 0,
                        "minimum": np.nan,
                        "maximum": np.nan,
                        "mean": np.nan,
                        "q01": np.nan,
                        "q10": np.nan,
                        "q50": np.nan,
                        "q90": np.nan,
                        "q99": np.nan,
                        "fraction_nonzero": np.nan,
                        "unique_values_preview": "",
                    })
                    continue

                numeric = flat.astype(float)
                unique = np.unique(flat)

                if len(unique) <= 30:
                    preview = ", ".join(str(x) for x in unique)
                else:
                    preview = (
                        ", ".join(str(x) for x in unique[:30])
                        + ", ..."
                    )

                rows.append({
                    "sample": sample,
                    "branch_path": branch_path,
                    "branch": leaf,
                    "n_values": int(len(numeric)),
                    "n_unique": int(len(unique)),
                    "minimum": float(np.min(numeric)),
                    "maximum": float(np.max(numeric)),
                    "mean": float(np.mean(numeric)),
                    "q01": float(np.quantile(numeric, 0.01)),
                    "q10": float(np.quantile(numeric, 0.10)),
                    "q50": float(np.quantile(numeric, 0.50)),
                    "q90": float(np.quantile(numeric, 0.90)),
                    "q99": float(np.quantile(numeric, 0.99)),
                    "fraction_nonzero": float(
                        np.mean(numeric != 0.0)
                    ),
                    "unique_values_preview": preview,
                })

            except Exception as exc:
                rows.append({
                    "sample": sample,
                    "branch_path": branch_path,
                    "branch": leaf,
                    "n_values": -1,
                    "n_unique": -1,
                    "minimum": np.nan,
                    "maximum": np.nan,
                    "mean": np.nan,
                    "q01": np.nan,
                    "q10": np.nan,
                    "q50": np.nan,
                    "q90": np.nan,
                    "q99": np.nan,
                    "fraction_nonzero": np.nan,
                    "unique_values_preview": f"ERROR: {exc}",
                })

summary = pd.DataFrame(rows)

summary.to_csv(
    OUTDIR / "ak4ak8_btag_branch_summary.csv",
    index=False,
)

(
    OUTDIR / "ak4ak8_btag_branch_summary.md"
).write_text(summary.to_markdown(index=False) + "\n")

(
    OUTDIR / "ak4ak8_btag_branch_inventory.json"
).write_text(json.dumps(inventory, indent=2))

card = REPO / "cards/delphes/delphes_card_CMS_lpc_ak4ak8.tcl"
card_lines = []

if card.exists():
    for number, line in enumerate(
        card.read_text(errors="replace").splitlines(),
        start=1,
    ):
        low = line.lower()
        if (
            "btag" in low
            or "bitnumber" in low
            or "efficiencyformula" in low
        ):
            card_lines.append(f"{number}: {line}")

(
    OUTDIR / "delphes_btag_card_lines.txt"
).write_text("\n".join(card_lines) + "\n")

readme = """# AK4/AK8 b-tag information audit

This audit searches recursively through Delphes split-branch paths.

Interpretation:

- Values {0, 1}: one binary tagging decision.
- Several integer values: possible Delphes working-point bit mask.
- Many continuously distributed values: possible score-like variable.

Delphes BTag fields must not be described as DeepJet discriminators unless
a calibrated continuous discriminator has explicitly been implemented.
"""

(OUTDIR / "README.md").write_text(readme)

print("\n=== B-tag branch inventory ===")
print(json.dumps(inventory, indent=2))

print("\n=== B-tag branch summary ===")
if summary.empty:
    print("EMPTY: no matching branches were successfully discovered")
else:
    print(summary.to_string(index=False))

print("\n=== Card lines ===")
print("\n".join(card_lines))

print("\nWrote:", OUTDIR)
