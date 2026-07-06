#!/usr/bin/env python3
import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot


def first_or_default(array, default=np.nan):
    out = ak.firsts(array)
    return ak.to_numpy(ak.fill_none(out, default))


def main():
    parser = argparse.ArgumentParser(
        description="Create a simple event-level parquet summary from a Delphes ROOT file."
    )
    parser.add_argument("--input", required=True, help="Input Delphes ROOT file")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--sample", default="HH4b_smoke_vbf", help="Sample name")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with uproot.open(args.input) as f:
        tree = f["Delphes"]

        branches = [
            "Event/Event.Weight",
            "Event/Event.CrossSection",
            "Particle/Particle.PID",
            "Jet/Jet.PT",
            "Jet/Jet.Eta",
            "Jet/Jet.Phi",
            "Jet/Jet.Mass",
            "Jet/Jet.BTag",
            "GenJet/GenJet.PT",
            "Electron/Electron.PT",
            "Muon/Muon.PT",
            "MissingET/MissingET.MET",
        ]

        arr = tree.arrays(branches, library="ak")
        n_events = tree.num_entries

    jet_pt = arr["Jet/Jet.PT"]
    jet_eta = arr["Jet/Jet.Eta"]
    jet_btag = arr["Jet/Jet.BTag"]

    jet_sel = (jet_pt > 30.0) & (abs(jet_eta) < 2.5)
    bjet_sel = jet_sel & (jet_btag > 0)

    selected_jet_pt = jet_pt[jet_sel]

    summary = pd.DataFrame(
        {
            "sample": [args.sample] * n_events,
            "event": np.arange(n_events, dtype=np.int64),
            "event_weight": first_or_default(arr["Event/Event.Weight"], 1.0),
            "event_cross_section_pb": first_or_default(arr["Event/Event.CrossSection"], np.nan),
            "n_particle": ak.to_numpy(ak.num(arr["Particle/Particle.PID"], axis=1)),
            "n_jet": ak.to_numpy(ak.num(jet_pt, axis=1)),
            "n_jet_pt30_eta25": ak.to_numpy(ak.sum(jet_sel, axis=1)),
            "n_bjet_pt30_eta25": ak.to_numpy(ak.sum(bjet_sel, axis=1)),
            "n_genjet": ak.to_numpy(ak.num(arr["GenJet/GenJet.PT"], axis=1)),
            "n_electron": ak.to_numpy(ak.num(arr["Electron/Electron.PT"], axis=1)),
            "n_muon": ak.to_numpy(ak.num(arr["Muon/Muon.PT"], axis=1)),
            "ht_pt30_eta25": ak.to_numpy(ak.sum(selected_jet_pt, axis=1)),
            "met": first_or_default(arr["MissingET/MissingET.MET"], 0.0),
        }
    )

    output = outdir / f"{args.sample}_event_summary.parquet"
    summary.to_parquet(output, index=False)

    print(f"Wrote: {output}")
    print(summary)
    print()
    print("Selection counts:")
    print("  events:", len(summary))
    print("  events with >=4 selected jets:", int((summary["n_jet_pt30_eta25"] >= 4).sum()))
    print("  events with >=4 selected b-tagged jets:", int((summary["n_bjet_pt30_eta25"] >= 4).sum()))


if __name__ == "__main__":
    main()
