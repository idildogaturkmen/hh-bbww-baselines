'''
take downloaded COLLIDE-1M parquet files here: outputs/collide_selected_backgrounds/

and produce a compact analysis skim here: outputs/all_background_reco_skim/all_processes_reco_skim.parquet

First background-compatible choise: H→bb candidate = two reconstructed AK4 jets with the highest b-tag scores

Goal:
    - skimmed parquet file is small enough to be used for signal-vs-background plots and significance scans
    - skimmed parquet file contains only the most relevant reco-level variables for HH→bbWW analysis

each row is one event with:
sample
process_group
n_jets
n_leptons
ht
met
b1_pt, b2_pt
b1_btag, b2_btag
mbb_top2_btag
dr_bb_top2_btag
pass_ge2jets
pass_1lep_ge3jets
pass_2lep_ge2jets
'''


from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


JET_PT_MIN_DEFAULT = 25.0
JET_ABS_ETA_MAX_DEFAULT = 2.4


def process_group(sample: str) -> str:
    """
    Physics grouping for HH→bbWW signal-vs-background studies.

    Important:
      - HH_bbWW is the signal.
      - Other HH samples are not signal for this analysis; keep them as HH_other.
      - ttH/ttW/ttZ/tttt are separated from inclusive ttbar.
    """
    s = sample.lower()

    if sample == "HH_bbWW":
        return "signal"

    if sample.startswith("HH_"):
        return "HH_other"

    if s.startswith(("tth", "ttw", "ttz", "tttt")):
        return "ttV_ttH_tttt"

    if s.startswith("tt0123j") or s.startswith("tt"):
        return "ttbar"

    if s.startswith("wjets"):
        return "wjets"

    if s.startswith("dyjets") or s.startswith("zjets"):
        return "dy_zjets"

    if s.startswith(("ww_", "wz_", "zz_")):
        return "diboson"

    if s.startswith("vvv"):
        return "triboson"

    if s.startswith(("ggh", "vbfh", "vh")):
        return "single_higgs"

    if s.startswith("qcd"):
        return "qcd"

    if s.startswith("gamma"):
        return "gamma"

    if s.startswith("minbias"):
        return "minbias"

    if s.startswith("upsilon"):
        return "upsilon"

    if s.startswith(("st_", "single")) or s.startswith(("tw_", "tzq", "thq")):
        return "single_top_candidate"

    return "other"


def delta_phi(phi1: float, phi2: float) -> float:
    dphi = float(phi1) - float(phi2)
    while dphi > math.pi:
        dphi -= 2.0 * math.pi
    while dphi <= -math.pi:
        dphi += 2.0 * math.pi
    return dphi


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    return math.sqrt((float(eta1) - float(eta2)) ** 2 + delta_phi(phi1, phi2) ** 2)


def pair_mass(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2) -> float:
    px1 = pt1 * math.cos(phi1)
    py1 = pt1 * math.sin(phi1)
    pz1 = pt1 * math.sinh(eta1)
    e1 = math.sqrt(max(px1 * px1 + py1 * py1 + pz1 * pz1 + m1 * m1, 0.0))

    px2 = pt2 * math.cos(phi2)
    py2 = pt2 * math.sin(phi2)
    pz2 = pt2 * math.sinh(eta2)
    e2 = math.sqrt(max(px2 * px2 + py2 * py2 + pz2 * pz2 + m2 * m2, 0.0))

    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    return math.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0))


def list_or_empty(event: dict, name: str | None):
    if name is None:
        return []
    x = event.get(name, [])
    if x is None:
        return []
    return x


def first_existing_name(available: set[str], candidates: list[str]) -> str | None:
    for name in candidates:
        if name in available:
            return name
    return None


def scalar_or_nan(event: dict, possible_names: list[str]) -> float:
    for name in possible_names:
        if name not in event:
            continue
        x = event[name]
        try:
            if isinstance(x, (list, tuple, np.ndarray)):
                if len(x) == 0:
                    continue
                return float(x[0])
            return float(x)
        except Exception:
            continue
    return float("nan")


def parquet_columns(path: Path) -> list[str]:
    pf = pq.ParquetFile(path)
    try:
        return list(pf.schema_arrow.names)
    except Exception:
        return list(pf.schema.names)


def build_jets(
    event: dict,
    names: dict,
    jet_pt_min: float,
    jet_abs_eta_max: float,
):
    pts = list_or_empty(event, names["jet_pt"])
    etas = list_or_empty(event, names["jet_eta"])
    phis = list_or_empty(event, names["jet_phi"])
    masses = list_or_empty(event, names["jet_mass"])

    btags = list_or_empty(event, names.get("jet_btag"))
    btag_phys = list_or_empty(event, names.get("jet_btag_phys"))

    jets = []
    n_raw = len(pts)

    for i in range(n_raw):
        try:
            pt = float(pts[i])
            eta = float(etas[i])
            phi = float(phis[i])
            mass = float(masses[i])
        except Exception:
            continue

        if pt < jet_pt_min:
            continue
        if abs(eta) > jet_abs_eta_max:
            continue

        btag = float(btags[i]) if i < len(btags) else 0.0
        btag_phys_value = float(btag_phys[i]) if i < len(btag_phys) else float("nan")

        jets.append(
            {
                "idx": i,
                "pt": pt,
                "eta": eta,
                "phi": phi,
                "mass": mass,
                "btag": btag,
                "btag_phys": btag_phys_value,
            }
        )

    return jets, n_raw


def count_leptons(event: dict, names: dict, lepton_pt_min: float):
    ele_pts_all = [float(x) for x in list_or_empty(event, names.get("electron_pt"))]
    mu_pts_all = [float(x) for x in list_or_empty(event, names.get("muon_pt"))]

    ele_pts = [pt for pt in ele_pts_all if pt > lepton_pt_min]
    mu_pts = [pt for pt in mu_pts_all if pt > lepton_pt_min]

    n_ele = len(ele_pts)
    n_mu = len(mu_pts)
    leading_lepton_pt = max(ele_pts + mu_pts) if (ele_pts or mu_pts) else 0.0

    return n_ele, n_mu, leading_lepton_pt


def infer_branch_names(available: set[str]) -> dict:
    names = {
        "jet_pt": first_existing_name(available, ["FullReco_JetAK4_PT", "JetAK4_PT"]),
        "jet_eta": first_existing_name(available, ["FullReco_JetAK4_Eta", "JetAK4_Eta"]),
        "jet_phi": first_existing_name(available, ["FullReco_JetAK4_Phi", "JetAK4_Phi"]),
        "jet_mass": first_existing_name(available, ["FullReco_JetAK4_Mass", "JetAK4_Mass"]),
        "jet_btag": first_existing_name(
            available,
            [
                "FullReco_JetAK4_BTag",
                "FullReco_JetAK4_BTagAlgo",
                "FullReco_JetAK4_BTagDeepB",
                "FullReco_JetAK4_BTagCSV",
                "JetAK4_BTag",
            ],
        ),
        "jet_btag_phys": first_existing_name(
            available,
            [
                "FullReco_JetAK4_BTagPhys",
                "JetAK4_BTagPhys",
            ],
        ),
        "electron_pt": first_existing_name(
            available,
            [
                "FullReco_Electron_PT",
                "Electron_PT",
            ],
        ),
        "muon_pt": first_existing_name(
            available,
            [
                "FullReco_MuonTight_PT",
                "FullReco_Muon_PT",
                "MuonTight_PT",
                "Muon_PT",
            ],
        ),
    }

    return names


def needed_columns(names: dict, available: set[str]) -> list[str]:
    columns = set()

    for key in [
        "jet_pt",
        "jet_eta",
        "jet_phi",
        "jet_mass",
        "jet_btag",
        "jet_btag_phys",
        "electron_pt",
        "muon_pt",
    ]:
        if names.get(key) is not None:
            columns.add(names[key])

    for met_name in [
        "FullReco_MET_MET",
        "FullReco_MissingET_MET",
        "FullReco_PuppiMissingET_MET",
        "FullReco_PuppiMET_MET",
        "MissingET_MET",
    ]:
        if met_name in available:
            columns.add(met_name)

    return sorted(columns)


def process_file(
    path: Path,
    sample: str,
    file_index: int,
    max_events: int,
    batch_size: int,
    jet_pt_min: float,
    jet_abs_eta_max: float,
    lepton_pt_min: float,
):
    available = set(parquet_columns(path))
    names = infer_branch_names(available)

    required = ["jet_pt", "jet_eta", "jet_phi", "jet_mass"]
    missing = [key for key in required if names.get(key) is None]
    if missing:
        raise RuntimeError(f"Missing required jet branches {missing}; available columns include {sorted(list(available))[:20]}")

    columns = needed_columns(names, available)
    pf = pq.ParquetFile(path)

    rows = []
    seen = 0

    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        cols = batch.to_pydict()
        n_batch = batch.num_rows

        for i in range(n_batch):
            if max_events > 0 and seen >= max_events:
                return rows

            event = {k: v[i] for k, v in cols.items()}

            jets, n_jets_raw = build_jets(
                event,
                names=names,
                jet_pt_min=jet_pt_min,
                jet_abs_eta_max=jet_abs_eta_max,
            )

            jets_by_pt = sorted(jets, key=lambda j: j["pt"], reverse=True)
            jets_by_btag = sorted(jets, key=lambda j: (j["btag"], j["pt"]), reverse=True)

            mbb = float("nan")
            drbb = float("nan")
            b1_pt = b2_pt = float("nan")
            b1_eta = b2_eta = float("nan")
            b1_phi = b2_phi = float("nan")
            b1_mass = b2_mass = float("nan")
            b1_btag = b2_btag = float("nan")
            b1_btag_phys = b2_btag_phys = float("nan")

            if len(jets_by_btag) >= 2:
                j1, j2 = jets_by_btag[0], jets_by_btag[1]
                mbb = pair_mass(
                    j1["pt"], j1["eta"], j1["phi"], j1["mass"],
                    j2["pt"], j2["eta"], j2["phi"], j2["mass"],
                )
                drbb = delta_r(j1["eta"], j1["phi"], j2["eta"], j2["phi"])

                b1_pt, b2_pt = j1["pt"], j2["pt"]
                b1_eta, b2_eta = j1["eta"], j2["eta"]
                b1_phi, b2_phi = j1["phi"], j2["phi"]
                b1_mass, b2_mass = j1["mass"], j2["mass"]
                b1_btag, b2_btag = j1["btag"], j2["btag"]
                b1_btag_phys, b2_btag_phys = j1["btag_phys"], j2["btag_phys"]

            n_ele, n_mu, leading_lepton_pt = count_leptons(
                event,
                names=names,
                lepton_pt_min=lepton_pt_min,
            )
            n_lep = n_ele + n_mu

            ht = float(sum(j["pt"] for j in jets))
            met = scalar_or_nan(
                event,
                [
                    "FullReco_MET_MET",
                    "FullReco_MissingET_MET",
                    "FullReco_PuppiMissingET_MET",
                    "FullReco_PuppiMET_MET",
                    "MissingET_MET",
                ],
            )

            n_jets = len(jets)

            rows.append(
                {
                    "sample": sample,
                    "process_group": process_group(sample),
                    "file": str(path),
                    "file_index": file_index,
                    "event_in_file": seen,
                    "weight": 1.0,
                    "n_jets_raw": n_jets_raw,
                    "n_jets": n_jets,
                    "n_leptons": n_lep,
                    "n_electrons": n_ele,
                    "n_muons": n_mu,
                    "leading_lepton_pt": leading_lepton_pt,
                    "ht": ht,
                    "met": met,
                    "lead_jet_pt": jets_by_pt[0]["pt"] if len(jets_by_pt) > 0 else float("nan"),
                    "sublead_jet_pt": jets_by_pt[1]["pt"] if len(jets_by_pt) > 1 else float("nan"),
                    "b1_pt": b1_pt,
                    "b2_pt": b2_pt,
                    "b1_eta": b1_eta,
                    "b2_eta": b2_eta,
                    "b1_phi": b1_phi,
                    "b2_phi": b2_phi,
                    "b1_mass": b1_mass,
                    "b2_mass": b2_mass,
                    "b1_btag": b1_btag,
                    "b2_btag": b2_btag,
                    "b1_btag_phys": b1_btag_phys,
                    "b2_btag_phys": b2_btag_phys,
                    "mbb_top2_btag": mbb,
                    "dr_bb_top2_btag": drbb,
                    "pass_ge2jets": n_jets >= 2,
                    "pass_ge3jets": n_jets >= 3,
                    "pass_ge4jets": n_jets >= 4,
                    "pass_1lep_ge3jets": (n_lep == 1 and n_jets >= 3),
                    "pass_1lep_ge4jets": (n_lep == 1 and n_jets >= 4),
                    "pass_2lep_ge2jets": (n_lep == 2 and n_jets >= 2),
                    "pass_mbb_90_140": bool(np.isfinite(mbb) and 90.0 <= mbb <= 140.0),
                }
            )

            seen += 1

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact reco-level skim from local COLLIDE-1M signal and "
            "background parquet files. The H→bb candidate is defined using the "
            "two highest-btag reconstructed AK4 jets, so the output is usable for "
            "signal-vs-background plots and significance scans."
        )
    )
    parser.add_argument("--root", default="outputs/collide_selected_backgrounds")
    parser.add_argument("--outdir", default="outputs/all_background_reco_skim")
    parser.add_argument("--max-events-per-file", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--jet-pt-min", type=float, default=JET_PT_MIN_DEFAULT)
    parser.add_argument("--jet-abs-eta-max", type=float, default=JET_ABS_ETA_MAX_DEFAULT)
    parser.add_argument("--lepton-pt-min", type=float, default=10.0)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(root.glob("*/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root}")

    print(f"Using root: {root}")
    print(f"Found parquet files: {len(parquet_files)}")
    print(f"max_events_per_file: {args.max_events_per_file}")
    print(f"jet selection: pt > {args.jet_pt_min} GeV, |eta| < {args.jet_abs_eta_max}")
    print(f"lepton counting: pt > {args.lepton_pt_min} GeV")

    all_rows = []
    failures = []

    for file_index, path in enumerate(parquet_files):
        sample = path.parent.name
        group = process_group(sample)

        print(f"[{file_index + 1:4d}/{len(parquet_files)}] {sample:45s} group={group:16s} {path.name}", flush=True)

        try:
            rows = process_file(
                path=path,
                sample=sample,
                file_index=file_index,
                max_events=args.max_events_per_file,
                batch_size=args.batch_size,
                jet_pt_min=args.jet_pt_min,
                jet_abs_eta_max=args.jet_abs_eta_max,
                lepton_pt_min=args.lepton_pt_min,
            )
            all_rows.extend(rows)
            print(f"    kept rows: {len(rows)}", flush=True)
        except Exception as exc:
            print(f"    WARNING failed: {exc}", flush=True)
            failures.append(
                {
                    "sample": sample,
                    "file": str(path),
                    "error": repr(exc),
                }
            )

    df = pd.DataFrame(all_rows)

    out_parquet = outdir / "all_processes_reco_skim.parquet"
    out_csv_preview = outdir / "all_processes_reco_skim_preview.csv"
    out_summary_json = outdir / "skim_summary.json"
    out_failures = outdir / "skim_failures.csv"

    df.to_parquet(out_parquet, index=False)
    df.head(5000).to_csv(out_csv_preview, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(out_failures, index=False)

    process_counts = (
        df.groupby("process_group")
        .size()
        .sort_values(ascending=False)
        .to_dict()
        if len(df)
        else {}
    )

    sample_counts = (
        df.groupby(["process_group", "sample"])
        .size()
        .reset_index(name="n_events")
        .sort_values(["process_group", "sample"])
    )
    sample_counts.to_csv(outdir / "skim_sample_counts.csv", index=False)

    selection_counts = {}
    if len(df):
        for group, sub in df.groupby("process_group"):
            selection_counts[group] = {
                "n_events": int(len(sub)),
                "pass_ge2jets": int(sub["pass_ge2jets"].sum()),
                "pass_1lep_ge3jets": int(sub["pass_1lep_ge3jets"].sum()),
                "pass_2lep_ge2jets": int(sub["pass_2lep_ge2jets"].sum()),
                "pass_mbb_90_140": int(sub["pass_mbb_90_140"].sum()),
            }

    summary = {
        "root": str(root),
        "n_input_files": len(parquet_files),
        "n_output_events": int(len(df)),
        "max_events_per_file": args.max_events_per_file,
        "jet_pt_min": args.jet_pt_min,
        "jet_abs_eta_max": args.jet_abs_eta_max,
        "lepton_pt_min": args.lepton_pt_min,
        "process_counts": {k: int(v) for k, v in process_counts.items()},
        "selection_counts": selection_counts,
        "n_failures": len(failures),
        "outputs": {
            "skim_parquet": str(out_parquet),
            "preview_csv": str(out_csv_preview),
            "sample_counts_csv": str(outdir / "skim_sample_counts.csv"),
            "summary_json": str(out_summary_json),
            "failures_csv": str(out_failures) if failures else None,
        },
    }

    with open(out_summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nWrote:")
    print(f"  {out_parquet}")
    print(f"  {out_csv_preview}")
    print(f"  {outdir / 'skim_sample_counts.csv'}")
    print(f"  {out_summary_json}")
    if failures:
        print(f"  {out_failures}")

    print("\nEvent counts by process group:")
    if len(df):
        print(df.groupby("process_group").size().sort_values(ascending=False).to_string())
    else:
        print("No rows written.")

    print("\nSelection counts by process group:")
    if len(df):
        cols = ["pass_ge2jets", "pass_1lep_ge3jets", "pass_2lep_ge2jets", "pass_mbb_90_140"]
        print(df.groupby("process_group")[cols].sum().to_string())
    else:
        print("No rows written.")


if __name__ == "__main__":
    main()