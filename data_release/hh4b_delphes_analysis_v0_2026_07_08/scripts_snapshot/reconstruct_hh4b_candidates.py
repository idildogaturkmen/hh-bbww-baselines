#!/usr/bin/env python3

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot


PAIRINGS = [
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
]


def four_vector(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return e, px, py, pz


def invariant_mass(jets, pair):
    i, j = pair

    e1, px1, py1, pz1 = four_vector(
        jets[i]["pt"], jets[i]["eta"], jets[i]["phi"], jets[i]["mass"]
    )
    e2, px2, py2, pz2 = four_vector(
        jets[j]["pt"], jets[j]["eta"], jets[j]["phi"], jets[j]["mass"]
    )

    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m2 = e * e - px * px - py * py - pz * pz
    return float(np.sqrt(max(m2, 0.0)))


def delta_r(jets, pair):
    i, j = pair
    deta = jets[i]["eta"] - jets[j]["eta"]
    dphi = np.arctan2(
        np.sin(jets[i]["phi"] - jets[j]["phi"]),
        np.cos(jets[i]["phi"] - jets[j]["phi"]),
    )
    return float(np.sqrt(deta * deta + dphi * dphi))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Delphes ROOT file")
    parser.add_argument("--out", required=True, help="Output parquet file")
    parser.add_argument("--sample", required=True, help="Sample name")
    parser.add_argument("--target-mass", type=float, default=125.0)
    args = parser.parse_args()

    with uproot.open(args.input) as f:
        tree = f["Delphes"]
        arrays = tree.arrays(
            ["Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass", "Jet.BTag", "Jet.Flavor"],
            library="ak",
        )

    rows = []

    n_events = len(arrays["Jet.PT"])

    for iev in range(n_events):
        jets = []

        pts = ak.to_numpy(arrays["Jet.PT"][iev])
        etas = ak.to_numpy(arrays["Jet.Eta"][iev])
        phis = ak.to_numpy(arrays["Jet.Phi"][iev])
        masses = ak.to_numpy(arrays["Jet.Mass"][iev])
        btags = ak.to_numpy(arrays["Jet.BTag"][iev])
        flavors = ak.to_numpy(arrays["Jet.Flavor"][iev])

        for pt, eta, phi, mass, btag, flavor in zip(pts, etas, phis, masses, btags, flavors):
            if pt > 30.0 and abs(eta) < 2.5 and btag > 0:
                jets.append(
                    {
                        "pt": float(pt),
                        "eta": float(eta),
                        "phi": float(phi),
                        "mass": float(mass),
                        "btag": int(btag),
                        "flavor": int(flavor),
                    }
                )

        jets = sorted(jets, key=lambda x: x["pt"], reverse=True)

        if len(jets) < 4:
            continue

        jets4 = jets[:4]

        best = None

        for pairing in PAIRINGS:
            pair1, pair2 = pairing
            mbb1 = invariant_mass(jets4, pair1)
            mbb2 = invariant_mass(jets4, pair2)
            score = abs(mbb1 - args.target_mass) + abs(mbb2 - args.target_mass)

            if best is None or score < best["score"]:
                best = {
                    "pairing": str(pairing),
                    "mbb1": mbb1,
                    "mbb2": mbb2,
                    "score": score,
                    "drbb1": delta_r(jets4, pair1),
                    "drbb2": delta_r(jets4, pair2),
                }

        # HH four-body mass from the selected four b-tagged jets
        e_sum = px_sum = py_sum = pz_sum = 0.0
        for jet in jets4:
            e, px, py, pz = four_vector(jet["pt"], jet["eta"], jet["phi"], jet["mass"])
            e_sum += e
            px_sum += px
            py_sum += py
            pz_sum += pz

        mhh2 = e_sum * e_sum - px_sum * px_sum - py_sum * py_sum - pz_sum * pz_sum
        mhh = float(np.sqrt(max(mhh2, 0.0)))

        rows.append(
            {
                "sample": args.sample,
                "event": iev,
                "n_selected_bjets": len(jets),
                "mbb1": best["mbb1"],
                "mbb2": best["mbb2"],
                "avg_mbb": 0.5 * (best["mbb1"] + best["mbb2"]),
                "delta_mbb": abs(best["mbb1"] - best["mbb2"]),
                "mhh": mhh,
                "drbb1": best["drbb1"],
                "drbb2": best["drbb2"],
                "pairing": best["pairing"],
                "j1_pt": jets4[0]["pt"],
                "j2_pt": jets4[1]["pt"],
                "j3_pt": jets4[2]["pt"],
                "j4_pt": jets4[3]["pt"],
            }
        )

    df = pd.DataFrame(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"Input events: {n_events}")
    print(f"Events with >=4 selected b-tagged jets: {len(df)}")
    print(f"Wrote: {out}")

    if len(df) > 0:
        print()
        print(df[["mbb1", "mbb2", "avg_mbb", "delta_mbb", "mhh", "n_selected_bjets"]].describe())


if __name__ == "__main__":
    main()
