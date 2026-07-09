#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd

store = Path(os.environ["HH4B_STORE"])
repo = Path(os.environ["HH4B_REPO"])

raw = store / "metadata/mg5_vbf_decay_diagnostics.csv"
outdir = repo / "outputs/tables/mg5_vbf_decay_diagnostics_2026_07_08"
outdir.mkdir(parents=True, exist_ok=True)

if not raw.exists():
    raise FileNotFoundError(raw)

rows = []
lines = raw.read_text().splitlines()

for i, line in enumerate(lines):
    if i == 0 or not line.strip():
        continue

    # Expected logical columns:
    # sample,xsec_pb,xsec_fb,process,banner
    # But process can contain commas, so normal CSV parsing fails.
    parts = line.split(",", 3)
    if len(parts) < 4:
        print(f"Skipping malformed line {i+1}: {line}")
        continue

    sample, xsec_pb, xsec_fb, rest = parts

    if ",/uscms_data/" in rest:
        process, banner_tail = rest.rsplit(",/uscms_data/", 1)
        banner = "/uscms_data/" + banner_tail
    else:
        process = rest
        banner = ""

    rows.append(
        {
            "sample": sample.strip(),
            "xsec_pb": float(xsec_pb),
            "xsec_fb": float(xsec_fb),
            "process": process.strip(),
            "banner": banner.strip(),
        }
    )

df = pd.DataFrame(rows)

print("\n=== Salvaged VBF decay diagnostics ===")
print(df[["sample", "xsec_pb", "xsec_fb", "process"]].to_string(index=False))

def has(name):
    return name in set(df["sample"])

def get(name):
    return float(df.loc[df["sample"] == name, "xsec_fb"].iloc[0])

ratio_rows = []

if has("check_vbf_hjj_bb_sm") and has("check_vbf_hjj_stable_sm"):
    ratio_rows.append(
        {
            "ratio": "VBF Hjj→bb / VBF Hjj stable",
            "value": get("check_vbf_hjj_bb_sm") / get("check_vbf_hjj_stable_sm"),
            "interpretation": "effective BR(H→bb) in the SM VBF-Hjj check",
        }
    )

if has("check_vbf_hhjj_decay_single_syntax_sm") and has("check_vbf_hhjj_stable_sm"):
    ratio_rows.append(
        {
            "ratio": "VBF HHjj single-decay syntax / stable HHjj",
            "value": get("check_vbf_hhjj_decay_single_syntax_sm") / get("check_vbf_hhjj_stable_sm"),
            "interpretation": "tests whether one H→bb decay is applied",
        }
    )

if has("check_vbf_hhjj_decay_double_syntax_sm") and has("check_vbf_hhjj_stable_sm"):
    ratio_rows.append(
        {
            "ratio": "VBF HHjj double-decay syntax / stable HHjj",
            "value": get("check_vbf_hhjj_decay_double_syntax_sm") / get("check_vbf_hhjj_stable_sm"),
            "interpretation": "tests explicit HH→4b syntax",
        }
    )

ratio_rows.append(
    {
        "ratio": "0.662² from HEFT single-H check",
        "value": 0.661767209579479**2,
        "interpretation": "expected HH→4b/stable-HH ratio if two H→bb BRs are applied",
    }
)

ratio_rows.append(
    {
        "ratio": "0.5824² reference",
        "value": 0.5824**2,
        "interpretation": "modern SM BR(H→bb)² reference used earlier",
    }
)

ratios = pd.DataFrame(ratio_rows)

print("\n=== Ratios ===")
print(ratios.to_string(index=False))

df.to_csv(outdir / "mg5_vbf_decay_diagnostics_salvaged.csv", index=False)
ratios.to_csv(outdir / "mg5_vbf_decay_diagnostics_ratios.csv", index=False)

(outdir / "mg5_vbf_decay_diagnostics_salvaged.md").write_text(df.to_markdown(index=False) + "\n")
(outdir / "mg5_vbf_decay_diagnostics_ratios.md").write_text(ratios.to_markdown(index=False) + "\n")

print(f"\nWrote outputs to: {outdir}")
