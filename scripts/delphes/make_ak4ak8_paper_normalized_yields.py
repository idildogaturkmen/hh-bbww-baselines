#!/usr/bin/env python3

from pathlib import Path
import os
import yaml
import json
import numpy as np
import pandas as pd

REPO = Path(os.environ["HH4B_REPO"])
CATALOG = REPO / "datasets/ak4ak8_v1/ak4ak8_v1_local_manifest.csv"
CONFIG = REPO / "config/ak4ak8_paper_cross_sections_13tev.yaml"
OUTDIR = REPO / "outputs/tables/ak4ak8_paper_normalization_2026_07_15"
OUTDIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

df = pd.read_csv(CATALOG)

# Collapse the local manifest to one process row.
proc = (
    df.groupby(["role", "process"], as_index=False)
    .agg(
        n_shards=("tag", "count"),
        n_generated=("events", "sum"),
        n_candidates=("candidate_rows", "sum"),
        median_generator_xsec_pb=("cross_section_median_pb", "median"),
    )
)

# Map process names to config names.
process_to_config = {
    "ggF_HH_4b": "ggF_HH_4b",
    "VBF_HH_4b": "VBF_HH_4b",
    "ttbar": "ttbar",
    "QCD_bbbb": "QCD_bbbb",
    "Zbbbb": "Zbbbb",
    "ZZ_4b": "ZZ_4b",
    "ZH_4b": "ZH_4b",
}

rows = []
for _, r in proc.iterrows():
    process = r["process"]
    key = process_to_config.get(process)
    if key is None:
        continue

    sample_cfg = cfg["samples"][key]
    xsec_pb = float(sample_cfg["xsec_pb"])
    eff_candidate = r["n_candidates"] / r["n_generated"] if r["n_generated"] > 0 else np.nan

    base = {
        "role": r["role"],
        "process": process,
        "normalization_key": key,
        "source_status": sample_cfg["source_status"],
        "n_generated": int(r["n_generated"]),
        "n_candidates": int(r["n_candidates"]),
        "candidate_efficiency": eff_candidate,
        "xsec_pb_used": xsec_pb,
        "median_generator_xsec_pb": r["median_generator_xsec_pb"],
        "note": sample_cfg.get("note", ""),
    }

    for lumi in cfg["metadata"]["luminosities_fb"]:
        expected_produced = xsec_pb * lumi * 1000.0
        expected_candidates = expected_produced * eff_candidate
        base[f"expected_produced_{lumi}fb"] = expected_produced
        base[f"expected_candidates_{lumi}fb"] = expected_candidates

    rows.append(base)

out = pd.DataFrame(rows)

out.to_csv(OUTDIR / "ak4ak8_paper_normalized_candidate_yields.csv", index=False)
(OUTDIR / "ak4ak8_paper_normalized_candidate_yields.md").write_text(out.to_markdown(index=False) + "\n")

# Also make a compact signal-only table that is safe to call paper-level.
signal = out[out["role"] == "signal"].copy()
signal.to_csv(OUTDIR / "ak4ak8_official_signal_yields.csv", index=False)
(OUTDIR / "ak4ak8_official_signal_yields.md").write_text(signal.to_markdown(index=False) + "\n")

# Flag non-final backgrounds.
caveats = out[out["source_status"].str.contains("placeholder|generator", case=False, na=False)].copy()
caveats.to_csv(OUTDIR / "ak4ak8_normalization_caveats.csv", index=False)
(OUTDIR / "ak4ak8_normalization_caveats.md").write_text(caveats.to_markdown(index=False) + "\n")

summary = {
    "important_caveat": (
        "Signal rows use official SM HH cross sections times BR(H->bb)^2. "
        "QCD_bbbb and Zbbbb are fiducial generator placeholders, not final CMS-recommended normalizations."
    ),
    "luminosities_fb": cfg["metadata"]["luminosities_fb"],
}
(OUTDIR / "README.md").write_text(
    "# AK4/AK8 paper-normalized yield tables\n\n"
    + summary["important_caveat"]
    + "\n\nFiles:\n"
    + "- `ak4ak8_official_signal_yields.*`: signal-only table using official SM signal normalization.\n"
    + "- `ak4ak8_paper_normalized_candidate_yields.*`: full table, with caveats for placeholder backgrounds.\n"
    + "- `ak4ak8_normalization_caveats.*`: samples that are not final CMS-recommended normalizations.\n"
)
(OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))

print("=== Full normalized table ===")
print(out.to_string(index=False))

print("\n=== Signal-only official table ===")
print(signal.to_string(index=False))

print("\n=== Caveats ===")
print(caveats[["process", "source_status", "note"]].to_string(index=False))

print("\nWrote:", OUTDIR)
