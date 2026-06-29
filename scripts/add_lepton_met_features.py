'''
Extracts lepton and MET features from parquet files.
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ELECTRON_BRANCHES = {
    "pt": ["FullReco_Electron_PT", "Electron_PT"],
    "eta": ["FullReco_Electron_Eta", "Electron_Eta"],
    "phi": ["FullReco_Electron_Phi", "Electron_Phi"],
    "iso": ["FullReco_Electron_IsolationVarRhoCorr", "Electron_IsolationVarRhoCorr"],
}

MUON_BRANCHES = {
    "pt": ["FullReco_MuonTight_PT", "FullReco_Muon_PT", "MuonTight_PT", "Muon_PT"],
    "eta": ["FullReco_MuonTight_Eta", "FullReco_Muon_Eta", "MuonTight_Eta", "Muon_Eta"],
    "phi": ["FullReco_MuonTight_Phi", "FullReco_Muon_Phi", "MuonTight_Phi", "Muon_Phi"],
    "iso": ["FullReco_MuonTight_IsolationVarRhoCorr", "FullReco_Muon_IsolationVarRhoCorr", "MuonTight_IsolationVarRhoCorr", "Muon_IsolationVarRhoCorr"],
}

MET_BRANCHES = {
    "met": [
        "FullReco_MissingET_MET",
        "MissingET_MET",
        "FullReco_GenMissingET_MET",
    ],
    "phi": [
        "FullReco_MissingET_Phi",
        "MissingET_Phi",
        "FullReco_GenMissingET_Phi",
    ],
}


def first_existing(names: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in names:
            return c
    return None


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


def safe_get(arr, i, default=np.nan):
    try:
        return arr[i]
    except Exception:
        return default


def dphi(phi1, phi2):
    if not np.isfinite(phi1) or not np.isfinite(phi2):
        return np.nan
    x = phi1 - phi2
    while x > np.pi:
        x -= 2 * np.pi
    while x <= -np.pi:
        x += 2 * np.pi
    return x


def mt(pt, met, dphi_value):
    if not np.isfinite(pt) or not np.isfinite(met) or not np.isfinite(dphi_value):
        return np.nan
    return float(np.sqrt(max(0.0, 2.0 * pt * met * (1.0 - np.cos(dphi_value)))))


def infer_branches(parquet_file: str) -> dict:
    schema = pq.read_schema(parquet_file)
    names = schema.names

    out = {}
    for prefix, branch_set in [("electron", ELECTRON_BRANCHES), ("muon", MUON_BRANCHES)]:
        for key, candidates in branch_set.items():
            out[f"{prefix}_{key}"] = first_existing(names, candidates)

    out["met"] = first_existing(names, MET_BRANCHES["met"])
    out["met_phi"] = first_existing(names, MET_BRANCHES["phi"])

    return out


def extract_for_file(file_path: str, event_indices: list[int], lepton_pt_min: float) -> pd.DataFrame:
    branches = infer_branches(file_path)
    needed = sorted({v for v in branches.values() if v is not None})

    if not needed:
        raise RuntimeError(f"No needed branches found in {file_path}")

    table = pq.read_table(file_path, columns=needed).to_pandas()

    rows = []
    for idx in event_indices:
        ev = table.iloc[int(idx)]

        leptons = []

        e_pts = list_or_empty(ev.get(branches.get("electron_pt"))) if branches.get("electron_pt") else []
        e_etas = list_or_empty(ev.get(branches.get("electron_eta"))) if branches.get("electron_eta") else []
        e_phis = list_or_empty(ev.get(branches.get("electron_phi"))) if branches.get("electron_phi") else []
        e_isos = list_or_empty(ev.get(branches.get("electron_iso"))) if branches.get("electron_iso") else []

        for i, pt in enumerate(e_pts):
            pt = float(pt)
            eta = float(safe_get(e_etas, i))
            phi = float(safe_get(e_phis, i))
            iso = float(safe_get(e_isos, i))
            if pt >= lepton_pt_min and abs(eta) < 2.5:
                leptons.append(("electron", pt, eta, phi, iso))

        m_pts = list_or_empty(ev.get(branches.get("muon_pt"))) if branches.get("muon_pt") else []
        m_etas = list_or_empty(ev.get(branches.get("muon_eta"))) if branches.get("muon_eta") else []
        m_phis = list_or_empty(ev.get(branches.get("muon_phi"))) if branches.get("muon_phi") else []
        m_isos = list_or_empty(ev.get(branches.get("muon_iso"))) if branches.get("muon_iso") else []

        for i, pt in enumerate(m_pts):
            pt = float(pt)
            eta = float(safe_get(m_etas, i))
            phi = float(safe_get(m_phis, i))
            iso = float(safe_get(m_isos, i))
            if pt >= lepton_pt_min and abs(eta) < 2.4:
                leptons.append(("muon", pt, eta, phi, iso))

        leptons = sorted(leptons, key=lambda x: x[1], reverse=True)

        if branches.get("met"):
            met_values = list_or_empty(ev.get(branches["met"]))
            met_value = float(met_values[0]) if len(met_values) else np.nan
        else:
            met_value = np.nan

        if branches.get("met_phi"):
            met_phi_values = list_or_empty(ev.get(branches["met_phi"]))
            met_phi_value = float(met_phi_values[0]) if len(met_phi_values) else np.nan
        else:
            met_phi_value = np.nan

        if leptons:
            lep_type, lep_pt, lep_eta, lep_phi, lep_iso = leptons[0]
            reliso = lep_iso / lep_pt if np.isfinite(lep_iso) and lep_pt > 0 else np.nan
            dphi_l_met = abs(dphi(lep_phi, met_phi_value))
            mt_l_met = mt(lep_pt, met_value, dphi_l_met)
        else:
            lep_type, lep_pt, lep_eta, lep_phi, lep_iso = "none", np.nan, np.nan, np.nan, np.nan
            reliso, dphi_l_met, mt_l_met = np.nan, np.nan, np.nan

        n_iso_025 = sum(
            1 for _, pt, _, _, iso in leptons
            if np.isfinite(iso) and pt > 0 and iso / pt < 0.25
        )
        n_iso_015 = sum(
            1 for _, pt, _, _, iso in leptons
            if np.isfinite(iso) and pt > 0 and iso / pt < 0.15
        )
        n_iso_010 = sum(
            1 for _, pt, _, _, iso in leptons
            if np.isfinite(iso) and pt > 0 and iso / pt < 0.10
        )

        rows.append(
            {
                "event_in_file": int(idx),
                "leading_lepton_type": lep_type,
                "leading_lepton_eta": lep_eta,
                "leading_lepton_phi": lep_phi,
                "leading_lepton_iso": lep_iso,
                "leading_lepton_reliso_proxy": reliso,
                "met_from_branch": met_value,
                "met_phi": met_phi_value,
                "dphi_lep_met": dphi_l_met,
                "mt_lep_met": mt_l_met,
                "n_iso_leptons_reliso_lt_0p25": n_iso_025,
                "n_iso_leptons_reliso_lt_0p15": n_iso_015,
                "n_iso_leptons_reliso_lt_0p10": n_iso_010,
                "met_branch_used": branches.get("met"),
                "met_phi_branch_used": branches.get("met_phi"),
            }
        )

    return pd.DataFrame(rows)


def counts(df: pd.DataFrame, mask) -> dict:
    sub = df[mask]
    sig = sub[sub["target"] == 1]
    bkg = sub[sub["target"] == 0]

    s = float(sig["physics_weight_nominal"].sum())
    b = float(bkg["physics_weight_nominal"].sum())

    return {
        "S_raw": int(len(sig)),
        "B_raw": int(len(bkg)),
        "S_weighted": s,
        "B_weighted": b,
        "S_over_B_weighted": s / b if b > 0 else np.nan,
        "S_over_SplusB_weighted": s / (s + b) if (s + b) > 0 else np.nan,
        "Z_s_over_sqrt_b_weighted": s / np.sqrt(b) if b > 0 else np.nan,
    }


def run_selection_scan(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []

    iso_cuts = [None, 0.25, 0.15, 0.10]
    met_cuts = [0, 30, 40, 50]
    mt_cuts = [0, 30, 50]

    regions = {
        "premass": np.ones(len(df), dtype=bool),
        "ht100_mbb_ptcorr_80_150": (
            (df["ht"] >= 100)
            & (df["mbb_pt_binned_scaled"] >= 80)
            & (df["mbb_pt_binned_scaled"] <= 150)
        ),
    }

    for region_name, region_mask in regions.items():
        for iso in iso_cuts:
            for met_min in met_cuts:
                for mt_min in mt_cuts:
                    mask = region_mask.copy()

                    if iso is not None:
                        mask = mask & (df["leading_lepton_reliso_proxy"] < iso)

                    mask = mask & (df["met"] >= met_min)
                    mask = mask & (df["mt_lep_met"] >= mt_min)

                    row = {
                        "region": region_name,
                        "iso_reliso_lt": "none" if iso is None else iso,
                        "met_min": met_min,
                        "mt_lep_met_min": mt_min,
                    }
                    row.update(counts(df, mask))

                    zbb = df[mask & df["sample"].astype(str).str.contains("ZJetsTobb", na=False)]
                    tt = df[mask & (df["process_group"] == "ttbar")]
                    row["ZJetsTobb_weighted"] = float(zbb["physics_weight_nominal"].sum())
                    row["ttbar_weighted"] = float(tt["physics_weight_nominal"].sum())

                    rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "isolation_met_mt_selection_scan.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/physics_weights_lepemu_rough/onelep_pre_mbb_classifier_input_with_weights.parquet",
    )
    parser.add_argument("--outdir", default="outputs/lepton_met_features_lepemu_rough")
    parser.add_argument("--lepton-pt-min", type=float, default=10.0)
    parser.add_argument("--features", default="outputs/classifier_input_lepemu/feature_columns.json")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)

    if "file" not in df.columns:
        raise RuntimeError("Input table must contain a 'file' column to map rows back to source parquet files.")

    feature_rows = []

    for file_path, group in df.groupby("file"):
        event_indices = sorted(group["event_in_file"].astype(int).unique().tolist())
        print(f"Processing {file_path} with {len(event_indices)} selected events")
        extra = extract_for_file(file_path, event_indices, args.lepton_pt_min)
        extra["file"] = file_path
        feature_rows.append(extra)

    extra_all = pd.concat(feature_rows, ignore_index=True)

    merged = df.merge(extra_all, on=["file", "event_in_file"], how="left")

    merged.to_parquet(outdir / "onelep_pre_mbb_classifier_input_with_lepton_met_features.parquet", index=False)
    merged.head(500).to_csv(outdir / "lepton_met_feature_preview.csv", index=False)

    scan = run_selection_scan(merged, outdir)

    print("\n=== Best rough-weighted scans by region ===")
    for region in scan["region"].unique():
        sub = scan[scan["region"] == region].copy()
        print(f"\n--- {region} ---")
        print(
            sub.sort_values("Z_s_over_sqrt_b_weighted", ascending=False)
            .head(20)
            .to_string(index=False)
        )

    with open(args.features) as f:
        features = json.load(f)

    augmented_features = list(features)
    for c in [
        "leading_lepton_eta",
        "leading_lepton_iso",
        "leading_lepton_reliso_proxy",
        "dphi_lep_met",
        "mt_lep_met",
        "n_iso_leptons_reliso_lt_0p25",
        "n_iso_leptons_reliso_lt_0p15",
        "n_iso_leptons_reliso_lt_0p10",
    ]:
        if c not in augmented_features:
            augmented_features.append(c)

    with open(outdir / "feature_columns_augmented.json", "w") as f:
        json.dump(augmented_features, f, indent=2)

    print("\nWrote outputs to:", outdir)
    print("Augmented table:")
    print(outdir / "onelep_pre_mbb_classifier_input_with_lepton_met_features.parquet")
    print("Augmented features:")
    print(outdir / "feature_columns_augmented.json")


if __name__ == "__main__":
    main()
