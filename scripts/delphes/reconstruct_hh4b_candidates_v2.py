#!/usr/bin/env python3

import argparse
from itertools import combinations
import os
from pathlib import Path
import pickle
import subprocess
import sys

import awkward as ak
import numpy as np
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
    return {"e": float(e), "px": float(px), "py": float(py), "pz": float(pz)}


def add_vec(vectors):
    out = {"e": 0.0, "px": 0.0, "py": 0.0, "pz": 0.0}
    for v in vectors:
        out["e"] += v["e"]
        out["px"] += v["px"]
        out["py"] += v["py"]
        out["pz"] += v["pz"]
    return out


def vec_mass(v):
    m2 = v["e"] * v["e"] - v["px"] * v["px"] - v["py"] * v["py"] - v["pz"] * v["pz"]
    return float(np.sqrt(max(m2, 0.0)))


def vec_pt(v):
    return float(np.sqrt(v["px"] * v["px"] + v["py"] * v["py"]))


def vec_phi(v):
    return float(np.arctan2(v["py"], v["px"]))


def vec_eta(v):
    pt = vec_pt(v)
    if pt <= 0:
        return 0.0
    return float(np.arcsinh(v["pz"] / pt))


def delta_phi(phi1, phi2):
    return float(np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2)))


def delta_r_eta_phi(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return float(np.sqrt(deta * deta + dphi * dphi))


def pair_kinematics(jets, pair):
    i, j = pair
    v1 = four_vector(jets[i]["pt"], jets[i]["eta"], jets[i]["phi"], jets[i]["mass"])
    v2 = four_vector(jets[j]["pt"], jets[j]["eta"], jets[j]["phi"], jets[j]["mass"])
    v = add_vec([v1, v2])

    return {
        "mass": vec_mass(v),
        "pt": vec_pt(v),
        "eta": vec_eta(v),
        "phi": vec_phi(v),
        "vec": v,
        "dr": delta_r_eta_phi(jets[i]["eta"], jets[i]["phi"], jets[j]["eta"], jets[j]["phi"]),
        "jet_indices": (jets[i]["selected_index"], jets[j]["selected_index"]),
        "bjet_ranks": (jets[i]["bjet_rank"], jets[j]["bjet_rank"]),
    }


def choose_h1_h2(pair_a, pair_b, target_mass, ordering):
    if ordering == "pt":
        if pair_b["pt"] > pair_a["pt"]:
            return pair_b, pair_a
        return pair_a, pair_b

    if ordering == "mass_closest":
        if abs(pair_b["mass"] - target_mass) < abs(pair_a["mass"] - target_mass):
            return pair_b, pair_a
        return pair_a, pair_b

    if ordering == "mass_high":
        if pair_b["mass"] > pair_a["mass"]:
            return pair_b, pair_a
        return pair_a, pair_b

    raise ValueError(f"Unknown ordering: {ordering}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Delphes ROOT file")
    parser.add_argument("--out", required=True, help="Output parquet file")
    parser.add_argument("--sample", required=True, help="Sample name")
    parser.add_argument("--target-mass", type=float, default=125.0)
    parser.add_argument("--jet-pt-min", type=float, default=30.0)
    parser.add_argument("--jet-eta-max", type=float, default=2.5)
    parser.add_argument("--btag-min", type=float, default=0.0)
    parser.add_argument("--max-bjets-for-pairing", type=int, default=8)
    parser.add_argument(
        "--higgs-ordering",
        choices=["pt", "mass_closest", "mass_high"],
        default="pt",
        help="Stable ordering convention for h1/h2 after best pairing is chosen",
    )
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
        pts = ak.to_numpy(arrays["Jet.PT"][iev])
        etas = ak.to_numpy(arrays["Jet.Eta"][iev])
        phis = ak.to_numpy(arrays["Jet.Phi"][iev])
        masses = ak.to_numpy(arrays["Jet.Mass"][iev])
        btags = ak.to_numpy(arrays["Jet.BTag"][iev])
        flavors = ak.to_numpy(arrays["Jet.Flavor"][iev])

        selected_jets = []
        for raw_index, (pt, eta, phi, mass, btag, flavor) in enumerate(
            zip(pts, etas, phis, masses, btags, flavors)
        ):
            if pt > args.jet_pt_min and abs(eta) < args.jet_eta_max:
                selected_jets.append(
                    {
                        "raw_index": int(raw_index),
                        "pt": float(pt),
                        "eta": float(eta),
                        "phi": float(phi),
                        "mass": float(mass),
                        "btag": float(btag),
                        "flavor": int(flavor),
                    }
                )

        selected_jets = sorted(selected_jets, key=lambda x: x["pt"], reverse=True)
        for idx, jet in enumerate(selected_jets):
            jet["selected_index"] = idx

        bjets = [j for j in selected_jets if j["btag"] > args.btag_min]
        bjets = sorted(bjets, key=lambda x: x["pt"], reverse=True)
        for idx, jet in enumerate(bjets):
            jet["bjet_rank"] = idx

        if len(bjets) < 4:
            continue

        pairing_pool = bjets[: args.max_bjets_for_pairing]

        best = None

        for combo in combinations(range(len(pairing_pool)), 4):
            jets4_local = [pairing_pool[i] for i in combo]

            for pairing in PAIRINGS:
                pair1, pair2 = pairing
                kin1 = pair_kinematics(jets4_local, pair1)
                kin2 = pair_kinematics(jets4_local, pair2)

                mass_score = abs(kin1["mass"] - args.target_mass) + abs(
                    kin2["mass"] - args.target_mass
                )
                delta_mbb = abs(kin1["mass"] - kin2["mass"])
                pt_sum = kin1["pt"] + kin2["pt"]

                # Primary: closest to two Higgs masses.
                # Tie-breakers: balanced dijet masses, then larger dijet pT sum.
                sort_key = (mass_score, delta_mbb, -pt_sum)

                if best is None or sort_key < best["sort_key"]:
                    best = {
                        "sort_key": sort_key,
                        "combo": combo,
                        "pairing": pairing,
                        "pair1": kin1,
                        "pair2": kin2,
                        "jets4_local": jets4_local,
                    }

        h1, h2 = choose_h1_h2(
            best["pair1"], best["pair2"], args.target_mass, args.higgs_ordering
        )

        jets4 = sorted(best["jets4_local"], key=lambda x: x["pt"], reverse=True)
        hh_vec = add_vec(
            [
                four_vector(j["pt"], j["eta"], j["phi"], j["mass"])
                for j in jets4
            ]
        )

        mbb1 = h1["mass"]
        mbb2 = h2["mass"]
        avg_mbb = 0.5 * (mbb1 + mbb2)
        delta_mbb = abs(mbb1 - mbb2)

        r_hh_125_125 = float(
            np.sqrt((mbb1 - args.target_mass) ** 2 + (mbb2 - args.target_mass) ** 2)
        )
        # Kept only for comparison with older CMS-like / previous diagnostic convention.
        r_hh_125_120 = float(np.sqrt((mbb1 - 125.0) ** 2 + (mbb2 - 120.0) ** 2))

        ht_selected_jets = float(sum(j["pt"] for j in selected_jets))
        ht_selected_bjets = float(sum(j["pt"] for j in bjets))
        ht_candidate_jets = float(sum(j["pt"] for j in jets4))

        h_deta = h1["eta"] - h2["eta"]
        h_dphi = delta_phi(h1["phi"], h2["phi"])
        h_dr = float(np.sqrt(h_deta * h_deta + h_dphi * h_dphi))
        h_pt_balance = float(
            abs(h1["pt"] - h2["pt"]) / (h1["pt"] + h2["pt"] + 1e-6)
        )

        row = {
            "sample": args.sample,
            "event": iev,

            # Event-level jet counts
            "n_selected_jets": len(selected_jets),
            "n_selected_bjets": len(bjets),
            "n_extra_selected_jets": max(len(selected_jets) - 4, 0),
            "n_extra_selected_bjets": max(len(bjets) - 4, 0),

            # Event-level scalar pT sums
            "ht_selected_jets": ht_selected_jets,
            "ht_selected_bjets": ht_selected_bjets,
            "ht_candidate_jets": ht_candidate_jets,

            # Pairing diagnostics
            "pairing": str(best["pairing"]),
            "pairing_combo_bjet_ranks": str(tuple(int(i) for i in best["combo"])),
            "pairing_score_125_125": float(best["sort_key"][0]),
            "higgs_ordering": args.higgs_ordering,

            # Higgs candidate masses
            "mbb1": mbb1,
            "mbb2": mbb2,
            "avg_mbb": avg_mbb,
            "delta_mbb": delta_mbb,

            # Consistent R_HH definitions
            "r_hh": r_hh_125_125,
            "r_hh_125_125": r_hh_125_125,
            "r_hh_125_120": r_hh_125_120,

            # HH system
            "mhh": vec_mass(hh_vec),
            "hh_pt": vec_pt(hh_vec),
            "hh_eta": vec_eta(hh_vec),
            "hh_phi": vec_phi(hh_vec),

            # Dijet/Higgs topology
            "h1_pt": h1["pt"],
            "h1_eta": h1["eta"],
            "h1_phi": h1["phi"],
            "h2_pt": h2["pt"],
            "h2_eta": h2["eta"],
            "h2_phi": h2["phi"],
            "h_delta_eta": float(h_deta),
            "h_delta_phi": float(h_dphi),
            "h_delta_r": h_dr,
            "h_pt_balance": h_pt_balance,

            # Within-Higgs angular structure
            "drbb1": h1["dr"],
            "drbb2": h2["dr"],

            # Original leading candidate-jet pT variables
            "j1_pt": jets4[0]["pt"],
            "j2_pt": jets4[1]["pt"],
            "j3_pt": jets4[2]["pt"],
            "j4_pt": jets4[3]["pt"],
        }

        # Candidate jet diagnostics
        for k, jet in enumerate(jets4, start=1):
            row[f"j{k}_eta"] = jet["eta"]
            row[f"j{k}_phi"] = jet["phi"]
            row[f"j{k}_mass"] = jet["mass"]
            row[f"j{k}_btag"] = jet["btag"]
            row[f"j{k}_flavor"] = jet["flavor"]
            row[f"j{k}_raw_index"] = jet["raw_index"]
            row[f"j{k}_selected_index"] = jet["selected_index"]
            row[f"j{k}_bjet_rank"] = jet["bjet_rank"]

        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    parquet_writer = Path(__file__).with_name("write_parquet_from_pickle.py")
    if not parquet_writer.is_file():
        raise SystemExit(f"ERROR: missing isolated Parquet writer: {parquet_writer}")

    temporary_pickle = out.with_name(f".{out.name}.{os.getpid()}.pickle")
    try:
        with temporary_pickle.open("wb") as handle:
            pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
        subprocess.run(
            [
                sys.executable,
                str(parquet_writer),
                "--input",
                str(temporary_pickle),
                "--output",
                str(out),
            ],
            check=True,
        )
    finally:
        temporary_pickle.unlink(missing_ok=True)

    print(f"Input events: {n_events}")
    print(f"Events with >=4 selected b-tagged jets: {len(rows)}")
    print(f"Wrote: {out}")

    # The EL9 LCG 106 environment can abort during C++ static-library
    # destruction after a valid zero-row, zero-column candidate Parquet
    # has already been written and round-trip verified by the isolated
    # writer. All durable work and temporary-file cleanup are complete
    # here. Skip only interpreter/library teardown on this verified
    # empty-output success path.
    if not rows:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
