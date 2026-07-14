#!/usr/bin/env python3

from pathlib import Path
import os
import numpy as np
import pandas as pd
import awkward as ak
import uproot

HH4B_STORE = Path(os.environ["HH4B_STORE"])
HH4B_REPO = Path(os.environ["HH4B_REPO"])

OUTDIR = HH4B_REPO / "outputs/tables/ak4ak8_constituent_cloud_feasibility_2026_07_14"
OUTDIR.mkdir(parents=True, exist_ok=True)

INPUTS = [
    (
        "ttbar_ak4ak8_smoke10k",
        HH4B_STORE / "root_ak4ak8_smoke/ttbar_10k_ak4ak8_delphes.root",
    ),
    (
        "ttbar_extra50k_ak4ak8_shard000",
        HH4B_STORE / "root/ttbar_extra50k_ak4ak8_v1_shard000_pythia8_delphes.root",
    ),
]

MAX_EVENTS = 1000


def read_branch(tree, names):
    for name in names:
        try:
            return tree[name].array(entry_stop=MAX_EVENTS, library="ak")
        except Exception:
            pass
    return None


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p90": np.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def build_particles(tree):
    # Tracks: use PT/Eta/Phi/Mass.
    trk_pt = read_branch(tree, ["EFlowTrack.PT", "EFlowTrack/EFlowTrack.PT"])
    trk_eta = read_branch(tree, ["EFlowTrack.Eta", "EFlowTrack/EFlowTrack.Eta"])
    trk_phi = read_branch(tree, ["EFlowTrack.Phi", "EFlowTrack/EFlowTrack.Phi"])
    trk_mass = read_branch(tree, ["EFlowTrack.Mass", "EFlowTrack/EFlowTrack.Mass"])
    trk_charge = read_branch(tree, ["EFlowTrack.Charge", "EFlowTrack/EFlowTrack.Charge"])
    trk_pid = read_branch(tree, ["EFlowTrack.PID", "EFlowTrack/EFlowTrack.PID"])

    # EFlow photons / neutral hadrons usually use ET rather than PT.
    pho_pt = read_branch(tree, ["EFlowPhoton.PT", "EFlowPhoton/EFlowPhoton.PT", "EFlowPhoton.ET", "EFlowPhoton/EFlowPhoton.ET"])
    pho_eta = read_branch(tree, ["EFlowPhoton.Eta", "EFlowPhoton/EFlowPhoton.Eta"])
    pho_phi = read_branch(tree, ["EFlowPhoton.Phi", "EFlowPhoton/EFlowPhoton.Phi"])

    neu_pt = read_branch(tree, ["EFlowNeutralHadron.PT", "EFlowNeutralHadron/EFlowNeutralHadron.PT", "EFlowNeutralHadron.ET", "EFlowNeutralHadron/EFlowNeutralHadron.ET"])
    neu_eta = read_branch(tree, ["EFlowNeutralHadron.Eta", "EFlowNeutralHadron/EFlowNeutralHadron.Eta"])
    neu_phi = read_branch(tree, ["EFlowNeutralHadron.Phi", "EFlowNeutralHadron/EFlowNeutralHadron.Phi"])

    n_events = len(trk_pt) if trk_pt is not None else 0
    out = []

    for iev in range(n_events):
        pts = []
        etas = []
        phis = []
        masses = []
        charges = []
        pids = []
        types = []

        if trk_pt is not None:
            n = len(trk_pt[iev])
            pts.extend(np.asarray(trk_pt[iev], dtype=float))
            etas.extend(np.asarray(trk_eta[iev], dtype=float))
            phis.extend(np.asarray(trk_phi[iev], dtype=float))
            masses.extend(np.asarray(trk_mass[iev], dtype=float) if trk_mass is not None else np.zeros(n))
            charges.extend(np.asarray(trk_charge[iev], dtype=float) if trk_charge is not None else np.ones(n))
            pids.extend(np.asarray(trk_pid[iev], dtype=float) if trk_pid is not None else np.zeros(n))
            types.extend(np.full(n, 0))  # charged track

        if pho_pt is not None:
            n = len(pho_pt[iev])
            pts.extend(np.asarray(pho_pt[iev], dtype=float))
            etas.extend(np.asarray(pho_eta[iev], dtype=float))
            phis.extend(np.asarray(pho_phi[iev], dtype=float))
            masses.extend(np.zeros(n))
            charges.extend(np.zeros(n))
            pids.extend(np.full(n, 22))
            types.extend(np.full(n, 1))  # photon

        if neu_pt is not None:
            n = len(neu_pt[iev])
            pts.extend(np.asarray(neu_pt[iev], dtype=float))
            etas.extend(np.asarray(neu_eta[iev], dtype=float))
            phis.extend(np.asarray(neu_phi[iev], dtype=float))
            masses.extend(np.zeros(n))
            charges.extend(np.zeros(n))
            pids.extend(np.zeros(n))
            types.extend(np.full(n, 2))  # neutral hadron

        out.append({
            "pt": np.asarray(pts, dtype=float),
            "eta": np.asarray(etas, dtype=float),
            "phi": np.asarray(phis, dtype=float),
            "mass": np.asarray(masses, dtype=float),
            "charge": np.asarray(charges, dtype=float),
            "pid": np.asarray(pids, dtype=float),
            "type": np.asarray(types, dtype=int),
        })

    return out


def audit_collection(sample, path, collection, radius, pt_min, eta_max, max_jets_per_event):
    if not path.exists():
        print("[skip missing]", path)
        return []

    rows = []

    with uproot.open(path) as f:
        tree = f["Delphes"]

        jet_pt = read_branch(tree, [f"{collection}.PT", f"{collection}/{collection}.PT"])
        jet_eta = read_branch(tree, [f"{collection}.Eta", f"{collection}/{collection}.Eta"])
        jet_phi = read_branch(tree, [f"{collection}.Phi", f"{collection}/{collection}.Phi"])
        jet_mass = read_branch(tree, [f"{collection}.Mass", f"{collection}/{collection}.Mass"])

        particles = build_particles(tree)

    n_events = min(len(jet_pt), len(particles), MAX_EVENTS)

    for iev in range(n_events):
        jpt = np.asarray(jet_pt[iev], dtype=float)
        jeta = np.asarray(jet_eta[iev], dtype=float)
        jphi = np.asarray(jet_phi[iev], dtype=float)
        jmass = np.asarray(jet_mass[iev], dtype=float)

        keep = np.where((jpt > pt_min) & (np.abs(jeta) < eta_max))[0]
        if len(keep) == 0:
            continue

        keep = keep[np.argsort(jpt[keep])[::-1]][:max_jets_per_event]

        p = particles[iev]
        ppt = p["pt"]
        peta = p["eta"]
        pphi = p["phi"]

        valid_particles = np.isfinite(ppt) & np.isfinite(peta) & np.isfinite(pphi) & (ppt > 0.05)

        for rank, ij in enumerate(keep):
            deta = peta - jeta[ij]
            dphi = delta_phi(pphi, jphi[ij])
            dr = np.sqrt(deta * deta + dphi * dphi)
            inside = valid_particles & (dr < radius)

            n_const = int(np.sum(inside))
            sum_const_pt = float(np.sum(ppt[inside])) if n_const else 0.0
            lead_const_pt = float(np.max(ppt[inside])) if n_const else 0.0

            rows.append({
                "sample": sample,
                "file": path.name,
                "collection": collection,
                "radius_used_for_matching": radius,
                "event": iev,
                "jet_rank": rank,
                "jet_pt": float(jpt[ij]),
                "jet_eta": float(jeta[ij]),
                "jet_mass": float(jmass[ij]),
                "n_constituents_dR": n_const,
                "sum_const_pt": sum_const_pt,
                "sum_const_pt_over_jet_pt": sum_const_pt / float(jpt[ij]) if jpt[ij] > 0 else np.nan,
                "lead_const_pt_frac": lead_const_pt / sum_const_pt if sum_const_pt > 0 else np.nan,
            })

    return rows


all_rows = []

for sample, path in INPUTS:
    print("Auditing", sample, path)

    all_rows.extend(
        audit_collection(
            sample=sample,
            path=path,
            collection="Jet",
            radius=0.4,
            pt_min=20.0,
            eta_max=2.5,
            max_jets_per_event=8,
        )
    )

    all_rows.extend(
        audit_collection(
            sample=sample,
            path=path,
            collection="FatJet",
            radius=0.8,
            pt_min=200.0,
            eta_max=2.5,
            max_jets_per_event=3,
        )
    )

df = pd.DataFrame(all_rows)
df.to_csv(OUTDIR / "ak4ak8_constituent_cloud_feasibility_per_jet.csv", index=False)

summary_rows = []
for (sample, collection), g in df.groupby(["sample", "collection"]):
    n_const = summarize(g["n_constituents_dR"])
    pt_ratio = summarize(g["sum_const_pt_over_jet_pt"])
    lead_frac = summarize(g["lead_const_pt_frac"])

    summary_rows.append({
        "sample": sample,
        "collection": collection,
        "n_jets_audited": len(g),
        "median_n_constituents": n_const["median"],
        "p10_n_constituents": n_const["p10"],
        "p90_n_constituents": n_const["p90"],
        "frac_ge5_constituents": float(np.mean(g["n_constituents_dR"] >= 5)),
        "frac_ge10_constituents": float(np.mean(g["n_constituents_dR"] >= 10)),
        "median_sum_const_pt_over_jet_pt": pt_ratio["median"],
        "p10_sum_const_pt_over_jet_pt": pt_ratio["p10"],
        "p90_sum_const_pt_over_jet_pt": pt_ratio["p90"],
        "median_lead_const_pt_frac": lead_frac["median"],
    })

summary = pd.DataFrame(summary_rows).sort_values(["sample", "collection"])
summary.to_csv(OUTDIR / "ak4ak8_constituent_cloud_feasibility_summary.csv", index=False)
(OUTDIR / "ak4ak8_constituent_cloud_feasibility_summary.md").write_text(summary.to_markdown(index=False) + "\n")

print(summary.to_string(index=False))
print("Wrote:", OUTDIR)
