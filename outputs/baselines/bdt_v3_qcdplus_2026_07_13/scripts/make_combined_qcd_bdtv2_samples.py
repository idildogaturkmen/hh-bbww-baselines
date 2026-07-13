#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import pandas as pd

store = Path(os.environ["HH4B_STORE"])
outdir = store / "parquet_v2_nominal_plus_extra"
metadir = store / "metadata"
outdir.mkdir(parents=True, exist_ok=True)
metadir.mkdir(parents=True, exist_ok=True)

def add_derived(df):
    df = df.copy()
    eps = 1e-9

    required = [
        "j1_pt", "j2_pt", "j3_pt", "j4_pt",
        "ht_candidate_jets", "mhh",
        "drbb1", "drbb2",
    ]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise RuntimeError(f"Cannot make BDT-v2-ready file; missing required columns: {missing_required}")

    df["pt_sum4"] = df["j1_pt"] + df["j2_pt"] + df["j3_pt"] + df["j4_pt"]
    df["ht_over_mhh"] = df["ht_candidate_jets"] / np.maximum(df["mhh"], eps)
    df["pt_asym_12"] = np.abs(df["j1_pt"] - df["j2_pt"]) / np.maximum(df["j1_pt"] + df["j2_pt"], eps)
    df["pt_asym_34"] = np.abs(df["j3_pt"] - df["j4_pt"]) / np.maximum(df["j3_pt"] + df["j4_pt"], eps)
    df["avg_drbb"] = 0.5 * (df["drbb1"] + df["drbb2"])
    df["max_drbb"] = np.maximum(df["drbb1"], df["drbb2"])
    df["min_drbb"] = np.minimum(df["drbb1"], df["drbb2"])

    return df

def read_component(label, path, n_generated, xsec_pb):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    df = add_derived(df)

    df["source_component"] = label
    df["source_path"] = str(path)
    df["source_event"] = df["event"].values if "event" in df.columns else np.arange(len(df))

    return {
        "label": label,
        "path": path,
        "n_generated": int(n_generated),
        "xsec_pb": float(xsec_pb),
        "rows": len(df),
        "df": df,
    }

def combine(tag, phase_space, components):
    total_generated = sum(c["n_generated"] for c in components)

    # Same physical iHT slice, independent MC statistics.
    # Do not add the component cross sections as separate backgrounds.
    # Use one effective slice cross section, estimated by generated-event-weighted average.
    xsec_eff_pb = sum(c["xsec_pb"] * c["n_generated"] for c in components) / total_generated
    event_weight_pb = xsec_eff_pb / total_generated

    dfs = []
    for c in components:
        df = c["df"].copy()
        df["original_sample"] = df["sample"] if "sample" in df.columns else c["label"]
        df["sample"] = tag
        df["analysis_sample"] = tag
        df["phase_space"] = phase_space
        df["n_generated_equivalent"] = total_generated
        df["event_cross_section_pb"] = xsec_eff_pb
        df["weight_pb"] = event_weight_pb
        df["event_weight_pb"] = event_weight_pb
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined["event"] = np.arange(len(combined), dtype=np.int64)

    out = outdir / f"{tag}_hh4b_candidates.parquet"
    combined.to_parquet(out, index=False)

    summary_lines = []
    summary_lines.append(f"tag: {tag}")
    summary_lines.append(f"phase_space: {phase_space}")
    summary_lines.append(f"total_generated_events: {total_generated}")
    summary_lines.append(f"xsec_eff_pb: {xsec_eff_pb}")
    summary_lines.append(f"event_weight_pb: {event_weight_pb}")
    summary_lines.append(f"combined_candidate_rows: {len(combined)}")
    summary_lines.append(f"output: {out}")
    summary_lines.append("")
    summary_lines.append("components:")

    for c in components:
        summary_lines.append(f"  - label: {c['label']}")
        summary_lines.append(f"    path: {c['path']}")
        summary_lines.append(f"    n_generated: {c['n_generated']}")
        summary_lines.append(f"    xsec_pb: {c['xsec_pb']}")
        summary_lines.append(f"    candidate_rows: {c['rows']}")

    summary_path = metadir / f"{tag}_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))
    print("=" * 100)

    return {
        "tag": tag,
        "phase_space": phase_space,
        "total_generated_events": total_generated,
        "xsec_eff_pb": xsec_eff_pb,
        "event_weight_pb": event_weight_pb,
        "candidate_rows": len(combined),
        "output": str(out),
        "summary": str(summary_path),
    }

p200_nom = store / "parquet_v2_nominal/qcd_bbbb_iht200to400_combined120k_v2_hh4b_candidates.parquet"
p200_extra = store / "parquet/qcd_bbbb_iht200to400_extra100k_v3_merged_hh4b_candidates_bdtv2_ready.parquet"

p400_nom = store / "parquet_v2_nominal/qcd_bbbb_iht400to600_20000_v2_hh4b_candidates.parquet"
p400_extra = store / "parquet/qcd_bbbb_iht400to600_extra100k_rootkeep_v1_merged_hh4b_candidates_bdtv2_ready.parquet"

results = []

results.append(combine(
    tag="qcd_bbbb_iht200to400_combined220k_bdtv2",
    phase_space="200 <= iHT < 400 GeV",
    components=[
        read_component(
            "qcd_bbbb_iht200to400_combined120k_v2",
            p200_nom,
            n_generated=120000,
            xsec_pb=126.35226440429688,
        ),
        read_component(
            "qcd_bbbb_iht200to400_extra100k_v3",
            p200_extra,
            n_generated=100000,
            xsec_pb=126.40453720092773,
        ),
    ],
))

results.append(combine(
    tag="qcd_bbbb_iht400to600_combined120k_bdtv2",
    phase_space="400 <= iHT < 600 GeV",
    components=[
        read_component(
            "qcd_bbbb_iht400to600_20000_v2",
            p400_nom,
            n_generated=20000,
            xsec_pb=9.774867057800293,
        ),
        read_component(
            "qcd_bbbb_iht400to600_extra100k_rootkeep_v1",
            p400_extra,
            n_generated=100000,
            xsec_pb=9.783634185791016,
        ),
    ],
))

summary_csv = metadir / "combined_qcd_bdtv2_samples_summary.csv"
pd.DataFrame(results).to_csv(summary_csv, index=False)
print("Wrote:", summary_csv)
