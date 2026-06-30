"""
Add lepton and MET features to the one-lepton HH->bbWW classifier table.

Important MET priority:
1. FullReco_MET_MET / FullReco_MET_Phi
2. FullReco_PUPPIMET_MET / FullReco_PUPPIMET_Phi
3. FullReco_GenMissingET_MET as fallback

The script also overwrites the table's `met` column with the selected reconstructed
MET when available, while saving the original table value as `met_original_table`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ELECTRON_BRANCHES = {
    "pt": "FullReco_Electron_PT",
    "eta": "FullReco_Electron_Eta",
    "phi": "FullReco_Electron_Phi",
    "iso": "FullReco_Electron_IsolationVarRhoCorr",
}

MUON_BRANCHES = {
    "pt": "FullReco_MuonTight_PT",
    "eta": "FullReco_MuonTight_Eta",
    "phi": "FullReco_MuonTight_Phi",
    "iso": "FullReco_MuonTight_IsolationVarRhoCorr",
}

MET_CANDIDATES = [
    {
        "name": "FullReco_MET",
        "met": "FullReco_MET_MET",
        "phi": "FullReco_MET_Phi",
        "eta": "FullReco_MET_Eta",
        "code": 0,
    },
    {
        "name": "FullReco_PUPPIMET",
        "met": "FullReco_PUPPIMET_MET",
        "phi": "FullReco_PUPPIMET_Phi",
        "eta": "FullReco_PUPPIMET_Eta",
        "code": 1,
    },
    {
        "name": "FullReco_GenMissingET",
        "met": "FullReco_GenMissingET_MET",
        "phi": "FullReco_GenMissingET_Phi",
        "eta": "FullReco_GenMissingET_Eta",
        "code": 2,
    },
]

NEW_NUMERIC_FEATURES = [
    "leading_lepton_eta",
    "leading_lepton_phi",
    "leading_lepton_iso",
    "leading_lepton_reliso",
    "leading_lepton_is_muon",
    "leading_lepton_is_electron",
    "n_iso_leptons_reliso_lt_0p1",
    "n_iso_leptons_reliso_lt_0p2",
    "met_phi",
    "met_eta",
    "met_branch_code",
    "dphi_lep_met",
    "mt_lep_met",
    "met_over_lepton_pt",
]

NEW_METADATA_COLUMNS = [
    "met_source",
    "met_original_table",
]


def list_or_empty(x):
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def first_scalar(x):
    vals = list_or_empty(x)
    if len(vals) == 0:
        return np.nan
    try:
        return float(vals[0])
    except Exception:
        return np.nan


def dphi(phi1, phi2):
    if not np.isfinite(phi1) or not np.isfinite(phi2):
        return np.nan
    return float(np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2)))


def mt_lep_met(lep_pt, met, dphi_value):
    if not np.isfinite(lep_pt) or not np.isfinite(met) or not np.isfinite(dphi_value):
        return np.nan
    val = 2.0 * lep_pt * met * (1.0 - np.cos(dphi_value))
    return float(np.sqrt(max(0.0, val)))


def collect_leptons(row):
    leptons = []

    e_pts = list_or_empty(row.get(ELECTRON_BRANCHES["pt"], []))
    e_etas = list_or_empty(row.get(ELECTRON_BRANCHES["eta"], []))
    e_phis = list_or_empty(row.get(ELECTRON_BRANCHES["phi"], []))
    e_isos = list_or_empty(row.get(ELECTRON_BRANCHES["iso"], []))

    for i, pt in enumerate(e_pts):
        try:
            pt = float(pt)
            eta = float(e_etas[i])
            phi = float(e_phis[i])
            iso = float(e_isos[i]) if i < len(e_isos) else np.nan
        except Exception:
            continue

        if not np.isfinite(pt + eta + phi):
            continue

        leptons.append(
            {
                "pt": pt,
                "eta": eta,
                "phi": phi,
                "iso": iso,
                "is_muon": 0.0,
                "is_electron": 1.0,
            }
        )

    m_pts = list_or_empty(row.get(MUON_BRANCHES["pt"], []))
    m_etas = list_or_empty(row.get(MUON_BRANCHES["eta"], []))
    m_phis = list_or_empty(row.get(MUON_BRANCHES["phi"], []))
    m_isos = list_or_empty(row.get(MUON_BRANCHES["iso"], []))

    for i, pt in enumerate(m_pts):
        try:
            pt = float(pt)
            eta = float(m_etas[i])
            phi = float(m_phis[i])
            iso = float(m_isos[i]) if i < len(m_isos) else np.nan
        except Exception:
            continue

        if not np.isfinite(pt + eta + phi):
            continue

        leptons.append(
            {
                "pt": pt,
                "eta": eta,
                "phi": phi,
                "iso": iso,
                "is_muon": 1.0,
                "is_electron": 0.0,
            }
        )

    leptons = sorted(leptons, key=lambda x: x["pt"], reverse=True)
    return leptons


def choose_met(row):
    for cand in MET_CANDIDATES:
        met_col = cand["met"]
        if met_col not in row.index:
            continue

        met = first_scalar(row.get(met_col, np.nan))
        if not np.isfinite(met):
            continue

        phi = first_scalar(row.get(cand["phi"], np.nan)) if cand["phi"] in row.index else np.nan
        eta = first_scalar(row.get(cand["eta"], np.nan)) if cand["eta"] in row.index else np.nan

        return {
            "met": met,
            "met_phi": phi,
            "met_eta": eta,
            "met_source": cand["name"],
            "met_branch_code": float(cand["code"]),
        }

    return {
        "met": np.nan,
        "met_phi": np.nan,
        "met_eta": np.nan,
        "met_source": "missing",
        "met_branch_code": np.nan,
    }


def event_features(row):
    leptons = collect_leptons(row)
    met_info = choose_met(row)

    out = {
        "leading_lepton_eta": np.nan,
        "leading_lepton_phi": np.nan,
        "leading_lepton_iso": np.nan,
        "leading_lepton_reliso": np.nan,
        "leading_lepton_is_muon": np.nan,
        "leading_lepton_is_electron": np.nan,
        "n_iso_leptons_reliso_lt_0p1": 0,
        "n_iso_leptons_reliso_lt_0p2": 0,
        "met": met_info["met"],
        "met_phi": met_info["met_phi"],
        "met_eta": met_info["met_eta"],
        "met_source": met_info["met_source"],
        "met_branch_code": met_info["met_branch_code"],
        "dphi_lep_met": np.nan,
        "mt_lep_met": np.nan,
        "met_over_lepton_pt": np.nan,
    }

    relisos = []
    for lep in leptons:
        reliso = lep["iso"] / lep["pt"] if np.isfinite(lep["iso"]) and lep["pt"] > 0 else np.nan
        relisos.append(reliso)

    out["n_iso_leptons_reliso_lt_0p1"] = int(sum(np.isfinite(x) and x < 0.1 for x in relisos))
    out["n_iso_leptons_reliso_lt_0p2"] = int(sum(np.isfinite(x) and x < 0.2 for x in relisos))

    if len(leptons) > 0:
        lep = leptons[0]
        reliso = lep["iso"] / lep["pt"] if np.isfinite(lep["iso"]) and lep["pt"] > 0 else np.nan

        out["leading_lepton_eta"] = lep["eta"]
        out["leading_lepton_phi"] = lep["phi"]
        out["leading_lepton_iso"] = lep["iso"]
        out["leading_lepton_reliso"] = reliso
        out["leading_lepton_is_muon"] = lep["is_muon"]
        out["leading_lepton_is_electron"] = lep["is_electron"]

        dphi_lm = dphi(lep["phi"], met_info["met_phi"])
        out["dphi_lep_met"] = abs(dphi_lm) if np.isfinite(dphi_lm) else np.nan
        out["mt_lep_met"] = mt_lep_met(lep["pt"], met_info["met"], dphi_lm)
        out["met_over_lepton_pt"] = (
            met_info["met"] / lep["pt"]
            if np.isfinite(met_info["met"]) and lep["pt"] > 0
            else np.nan
        )

    return out


def extract_features_for_file(file_path: str, event_indices: list[int]) -> pd.DataFrame:
    schema = pq.read_schema(file_path)
    available = set(schema.names)

    requested = set()
    for group in [ELECTRON_BRANCHES, MUON_BRANCHES]:
        requested.update(group.values())
    for cand in MET_CANDIDATES:
        requested.add(cand["met"])
        requested.add(cand["phi"])
        requested.add(cand["eta"])

    columns = sorted(c for c in requested if c in available)

    if len(columns) == 0:
        raise RuntimeError(f"No requested lepton/MET branches found in {file_path}")

    table = pq.read_table(file_path, columns=columns).to_pandas()

    rows = []
    for idx in event_indices:
        idx = int(idx)
        feats = event_features(table.iloc[idx])
        feats["file"] = file_path
        feats["event_in_file"] = idx
        rows.append(feats)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/classifier_input_lepemu/onelep_pre_mbb_classifier_input.parquet",
    )
    parser.add_argument(
        "--features",
        default="outputs/classifier_input_lepemu/feature_columns.json",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/lepton_met_features_lepemu_rough_recoMET",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)

    all_extra = []
    for file_path, group in df.groupby("file"):
        event_indices = sorted(group["event_in_file"].astype(int).unique().tolist())
        print(f"Processing {file_path} with {len(event_indices)} selected events")
        extra = extract_features_for_file(file_path, event_indices)
        all_extra.append(extra)

    extra_df = pd.concat(all_extra, ignore_index=True)

    merged = df.merge(extra_df, on=["file", "event_in_file"], how="left", suffixes=("", "_new"))

    # Preserve the old table-level MET, then overwrite `met` with selected reco MET.
    if "met" in df.columns:
        merged["met_original_table"] = merged["met"]

    if "met_new" in merged.columns:
        merged["met"] = merged["met_new"]
        merged = merged.drop(columns=["met_new"])

    for c in NEW_NUMERIC_FEATURES + ["met"]:
        if c in merged.columns:
            merged[c] = merged[c].replace([np.inf, -np.inf], np.nan)

    out_table = outdir / "onelep_pre_mbb_classifier_input_with_lepton_met_features.parquet"
    merged.to_parquet(out_table, index=False)
    merged.head(500).to_csv(outdir / "lepton_met_feature_preview.csv", index=False)

    with open(args.features) as f:
        features = json.load(f)

    updated_features = list(features)

    # Ensure the corrected `met` remains a feature.
    if "met" not in updated_features and "met" in merged.columns:
        updated_features.append("met")

    for c in NEW_NUMERIC_FEATURES:
        if c in merged.columns and c not in updated_features:
            updated_features.append(c)

    out_features = outdir / "feature_columns_augmented.json"
    with open(out_features, "w") as f:
        json.dump(updated_features, f, indent=2)

    print("\nMET source counts:")
    print(merged["met_source"].value_counts(dropna=False).to_string())

    if "met_original_table" in merged.columns:
        diff = merged["met"] - merged["met_original_table"]
        print("\nCorrected MET minus original table MET:")
        print(diff.describe().to_string())

    print("\nWrote:", out_table)
    print("Wrote:", out_features)


if __name__ == "__main__":
    main()
