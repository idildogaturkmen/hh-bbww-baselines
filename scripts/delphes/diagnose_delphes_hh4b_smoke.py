#!/usr/bin/env python3
import argparse
import awkward as ak
import numpy as np
import pandas as pd
import uproot


def to_np(x):
    return ak.to_numpy(x)


def main():
    parser = argparse.ArgumentParser(description="Diagnose Delphes HH4b smoke sample jets, flavors, and b-tags.")
    parser.add_argument("--input", required=True, help="Input Delphes ROOT file")
    parser.add_argument("--out", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    branches = [
        "Particle/Particle.PID",
        "Particle/Particle.Status",
        "Jet/Jet.PT",
        "Jet/Jet.Eta",
        "Jet/Jet.Phi",
        "Jet/Jet.Mass",
        "Jet/Jet.BTag",
        "Jet/Jet.Flavor",
        "Jet/Jet.FlavorAlgo",
        "Jet/Jet.FlavorPhys",
        "GenJet/GenJet.PT",
        "GenJet/GenJet.Eta",
        "GenJet/GenJet.Flavor",
        "GenJet/GenJet.BTag",
    ]

    with uproot.open(args.input) as f:
        tree = f["Delphes"]
        arr = tree.arrays(branches, library="ak")
        n_events = tree.num_entries

    pid = arr["Particle/Particle.PID"]
    status = arr["Particle/Particle.Status"]

    jet_pt = arr["Jet/Jet.PT"]
    jet_eta = arr["Jet/Jet.Eta"]
    jet_btag = arr["Jet/Jet.BTag"]
    jet_flavor = arr["Jet/Jet.Flavor"]
    jet_flavor_algo = arr["Jet/Jet.FlavorAlgo"]
    jet_flavor_phys = arr["Jet/Jet.FlavorPhys"]

    genjet_pt = arr["GenJet/GenJet.PT"]
    genjet_eta = arr["GenJet/GenJet.Eta"]
    genjet_flavor = arr["GenJet/GenJet.Flavor"]
    genjet_btag = arr["GenJet/GenJet.BTag"]

    jet_sel = (jet_pt > 30.0) & (abs(jet_eta) < 2.5)
    genjet_sel = (genjet_pt > 30.0) & (abs(genjet_eta) < 2.5)

    df = pd.DataFrame({
        "event": np.arange(n_events),
        "n_higgs_particles": to_np(ak.sum(abs(pid) == 25, axis=1)),
        "n_b_particles": to_np(ak.sum(abs(pid) == 5, axis=1)),
        "n_status1_b_particles": to_np(ak.sum((abs(pid) == 5) & (status == 1), axis=1)),
        "n_jets": to_np(ak.num(jet_pt, axis=1)),
        "n_jets_pt30_eta25": to_np(ak.sum(jet_sel, axis=1)),
        "n_jets_flavor5": to_np(ak.sum(abs(jet_flavor) == 5, axis=1)),
        "n_jets_flavoralgo5": to_np(ak.sum(abs(jet_flavor_algo) == 5, axis=1)),
        "n_jets_flavorphys5": to_np(ak.sum(abs(jet_flavor_phys) == 5, axis=1)),
        "n_btagged_jets": to_np(ak.sum(jet_btag > 0, axis=1)),
        "n_btagged_jets_pt30_eta25": to_np(ak.sum((jet_btag > 0) & jet_sel, axis=1)),
        "n_genjets": to_np(ak.num(genjet_pt, axis=1)),
        "n_genjets_pt30_eta25": to_np(ak.sum(genjet_sel, axis=1)),
        "n_genjets_flavor5": to_np(ak.sum(abs(genjet_flavor) == 5, axis=1)),
        "n_genjets_btag": to_np(ak.sum(genjet_btag > 0, axis=1)),
    })

    print("Input:", args.input)
    print("Events:", n_events)
    print()
    print(df)
    print()
    print("Column sums:")
    for col in df.columns:
        if col != "event":
            print(f"  {col}: {df[col].sum()}")

    print()
    print("Events with >=4 reconstructed jets:", int((df["n_jets_pt30_eta25"] >= 4).sum()))
    print("Events with >=4 reconstructed b-tagged jets:", int((df["n_btagged_jets_pt30_eta25"] >= 4).sum()))
    print("Events with >=4 jet FlavorPhys b jets:", int((df["n_jets_flavorphys5"] >= 4).sum()))
    print("Events with >=4 generator-level b particles:", int((df["n_b_particles"] >= 4).sum()))

    if args.out:
        df.to_csv(args.out, index=False)
        print()
        print("Wrote:", args.out)


if __name__ == "__main__":
    main()
