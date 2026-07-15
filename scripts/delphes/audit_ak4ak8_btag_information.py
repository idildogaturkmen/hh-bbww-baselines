#!/usr/bin/env python3

from pathlib import Path
import json
import os

import awkward as ak
import numpy as np
import pandas as pd
import uproot

REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])

OUTDIR = REPO / "outputs/audits/ak4ak8_btag_information_2026_07_15"
OUTDIR.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "ggF_HH": (
        STORE
        / "root/ggf_hh4b_ak4ak8_10k_pythia8_delphes.root"
    ),
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
branch_inventory = {}

for sample, path in SAMPLES.items():
    if not path.exists():
        print("Missing:", path)
        continue

    print("Reading:", sample, path)

    with uproot.open(path) as root_file:
        tree = root_file["Delphes"]
        all_branches = list(tree.keys())

        btag_branches = sorted(
            branch
            for branch in all_branches
            if (
                branch.startswith("Jet.")
                or branch.startswith("FatJet.")
            )
            and "BTag" in branch
        )

        branch_inventory[sample] = btag_branches

        for branch in btag_branches:
            try:
                array = tree[branch].array(
                    entry_stop=10000,
                    library="ak",
                )

                flat = ak.to_numpy(
                    ak.flatten(array, axis=None)
                )

                flat = np.asarray(flat)
                flat = flat[np.isfinite(flat)]

                if len(flat) == 0:
                    rows.append({
                        "sample": sample,
                        "branch": branch,
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

                unique = np.unique(flat)

                if len(unique) <= 30:
                    preview = ", ".join(
                        str(value) for value in unique
                    )
                else:
                    preview = ", ".join(
                        str(value) for value in unique[:30]
                    ) + ", ..."

                rows.append({
                    "sample": sample,
                    "branch": branch,
                    "n_values": int(len(flat)),
                    "n_unique": int(len(unique)),
                    "minimum": float(np.min(flat)),
                    "maximum": float(np.max(flat)),
                    "mean": float(np.mean(flat)),
                    "q01": float(np.quantile(flat, 0.01)),
                    "q10": float(np.quantile(flat, 0.10)),
                    "q50": float(np.quantile(flat, 0.50)),
                    "q90": float(np.quantile(flat, 0.90)),
                    "q99": float(np.quantile(flat, 0.99)),
                    "fraction_nonzero": float(
                        np.mean(flat != 0)
                    ),
                    "unique_values_preview": preview,
                })

            except Exception as exc:
                rows.append({
                    "sample": sample,
                    "branch": branch,
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
).write_text(json.dumps(branch_inventory, indent=2))

card = (
    REPO
    / "cards/delphes/delphes_card_CMS_lpc_ak4ak8.tcl"
)

card_lines = []

if card.exists():
    for number, line in enumerate(
        card.read_text(errors="replace").splitlines(),
        start=1,
    ):
        if "btag" in line.lower() or "efficiencyformula" in line.lower():
            card_lines.append(f"{number}: {line}")

(
    OUTDIR / "delphes_btag_card_lines.txt"
).write_text("\n".join(card_lines) + "\n")

readme = """# AK4/AK8 b-tag information audit

This audit determines whether the current Delphes files contain only binary
tag decisions, multiple working-point bits, or a continuous discriminator.

Interpretation:

- Two unique values, usually 0 and 1: binary working-point decision.
- Several integer values or powers of two: possible working-point bit mask.
- Many continuously distributed values: continuous tag-score candidate.

The final production card must be frozen before large-scale sample production.
"""

(OUTDIR / "README.md").write_text(readme)

print("\n=== B-tag branch summary ===")
print(summary.to_string(index=False))

print("\n=== Card lines ===")
print("\n".join(card_lines))

print("\nWrote:", OUTDIR)
