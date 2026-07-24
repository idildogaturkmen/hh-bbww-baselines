#!/usr/bin/env python3

from __future__ import annotations

import argparse
from itertools import combinations
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any, Iterable, Sequence

import awkward as ak
import numpy as np
import uproot


MODE_THREEB = "threeb-control"
MODE_FOURB = "fourb-parity"
MODE_NAMES = (MODE_THREEB, MODE_FOURB)

REQUIRED_ROOT_BRANCHES = (
    "Jet.PT",
    "Jet.Eta",
    "Jet.Phi",
    "Jet.Mass",
    "Jet.BTag",
    "Jet.Flavor",
)

PAIRINGS = [
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
]

BASE_CANDIDATE_COLUMNS = [
    "sample",
    "event",
    "n_selected_jets",
    "n_selected_bjets",
    "n_extra_selected_jets",
    "n_extra_selected_bjets",
    "ht_selected_jets",
    "ht_selected_bjets",
    "ht_candidate_jets",
    "pairing",
    "pairing_combo_bjet_ranks",
    "pairing_score_125_125",
    "higgs_ordering",
    "mbb1",
    "mbb2",
    "avg_mbb",
    "delta_mbb",
    "r_hh",
    "r_hh_125_125",
    "r_hh_125_120",
    "mhh",
    "hh_pt",
    "hh_eta",
    "hh_phi",
    "h1_pt",
    "h1_eta",
    "h1_phi",
    "h2_pt",
    "h2_eta",
    "h2_phi",
    "h_delta_eta",
    "h_delta_phi",
    "h_delta_r",
    "h_pt_balance",
    "drbb1",
    "drbb2",
    "j1_pt",
    "j2_pt",
    "j3_pt",
    "j4_pt",
]

FROZEN_CANDIDATE_COLUMNS = (
    BASE_CANDIDATE_COLUMNS
    + [
        f"j{index}_{field}"
        for index in range(1, 5)
        for field in (
            "eta",
            "phi",
            "mass",
            "btag",
            "flavor",
            "raw_index",
            "selected_index",
            "bjet_rank",
        )
    ]
)

THREEB_PROVENANCE_COLUMNS = [
    "candidate_category",
    "promoted_jet_rule",
    "promoted_jet_raw_index",
    "promoted_jet_selected_index",
    "promoted_jet_pt",
    "promoted_jet_eta",
    "promoted_jet_phi",
    "promoted_jet_mass",
    "promoted_jet_btag",
    "promoted_jet_flavor",
    "n_selected_untagged_jets",
]

THREEB_CANDIDATE_COLUMNS = (
    FROZEN_CANDIDATE_COLUMNS + THREEB_PROVENANCE_COLUMNS
)

CANDIDATE_CATEGORY = "exactly_3b_plus_highest_pt_untagged"
PROMOTED_JET_RULE = "highest_pt_selected_untagged"


def output_columns(mode: str) -> list[str]:
    if mode == MODE_FOURB:
        return list(FROZEN_CANDIDATE_COLUMNS)
    if mode == MODE_THREEB:
        return list(THREEB_CANDIDATE_COLUMNS)
    raise ValueError(f"Unknown reconstruction mode: {mode}")


def typed_empty_output(mode: str) -> dict[str, list[Any]]:
    return {column: [] for column in output_columns(mode)}


def four_vector(
    pt: float,
    eta: float,
    phi: float,
    mass: float,
) -> dict[str, float]:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return {
        "e": float(energy),
        "px": float(px),
        "py": float(py),
        "pz": float(pz),
    }


def add_vec(vectors: Iterable[dict[str, float]]) -> dict[str, float]:
    result = {"e": 0.0, "px": 0.0, "py": 0.0, "pz": 0.0}
    for vector in vectors:
        result["e"] += vector["e"]
        result["px"] += vector["px"]
        result["py"] += vector["py"]
        result["pz"] += vector["pz"]
    return result


def vec_mass(vector: dict[str, float]) -> float:
    mass_squared = (
        vector["e"] * vector["e"]
        - vector["px"] * vector["px"]
        - vector["py"] * vector["py"]
        - vector["pz"] * vector["pz"]
    )
    return float(np.sqrt(max(mass_squared, 0.0)))


def vec_pt(vector: dict[str, float]) -> float:
    return float(
        np.sqrt(
            vector["px"] * vector["px"]
            + vector["py"] * vector["py"]
        )
    )


def vec_phi(vector: dict[str, float]) -> float:
    return float(np.arctan2(vector["py"], vector["px"]))


def vec_eta(vector: dict[str, float]) -> float:
    pt = vec_pt(vector)
    if pt <= 0:
        return 0.0
    return float(np.arcsinh(vector["pz"] / pt))


def delta_phi(phi1: float, phi2: float) -> float:
    return float(
        np.arctan2(
            np.sin(phi1 - phi2),
            np.cos(phi1 - phi2),
        )
    )


def delta_r_eta_phi(
    eta1: float,
    phi1: float,
    eta2: float,
    phi2: float,
) -> float:
    delta_eta = eta1 - eta2
    delta_phi_value = delta_phi(phi1, phi2)
    return float(
        np.sqrt(
            delta_eta * delta_eta
            + delta_phi_value * delta_phi_value
        )
    )


def pair_kinematics(
    jets: Sequence[dict[str, Any]],
    pair: tuple[int, int],
) -> dict[str, Any]:
    first, second = pair
    first_vector = four_vector(
        jets[first]["pt"],
        jets[first]["eta"],
        jets[first]["phi"],
        jets[first]["mass"],
    )
    second_vector = four_vector(
        jets[second]["pt"],
        jets[second]["eta"],
        jets[second]["phi"],
        jets[second]["mass"],
    )
    vector = add_vec([first_vector, second_vector])
    return {
        "mass": vec_mass(vector),
        "pt": vec_pt(vector),
        "eta": vec_eta(vector),
        "phi": vec_phi(vector),
        "vec": vector,
        "dr": delta_r_eta_phi(
            jets[first]["eta"],
            jets[first]["phi"],
            jets[second]["eta"],
            jets[second]["phi"],
        ),
        "jet_indices": (
            jets[first]["selected_index"],
            jets[second]["selected_index"],
        ),
        "bjet_ranks": (
            jets[first]["bjet_rank"],
            jets[second]["bjet_rank"],
        ),
    }


def pairing_sort_key(
    pair_a: dict[str, Any],
    pair_b: dict[str, Any],
    target_mass: float,
) -> tuple[float, float, float]:
    mass_score = (
        abs(pair_a["mass"] - target_mass)
        + abs(pair_b["mass"] - target_mass)
    )
    delta_mbb = abs(pair_a["mass"] - pair_b["mass"])
    pt_sum = pair_a["pt"] + pair_b["pt"]
    return (mass_score, delta_mbb, -pt_sum)


def evaluate_pairings(
    jets4_local: Sequence[dict[str, Any]],
    target_mass: float,
) -> tuple[dict[str, Any], int]:
    if len(jets4_local) != 4:
        raise ValueError("Exactly four fixed jets are required for pairing")
    best: dict[str, Any] | None = None
    pairings_evaluated = 0
    for pairing in PAIRINGS:
        pairings_evaluated += 1
        pair1, pair2 = pairing
        kinematics1 = pair_kinematics(jets4_local, pair1)
        kinematics2 = pair_kinematics(jets4_local, pair2)
        sort_key = pairing_sort_key(
            kinematics1,
            kinematics2,
            target_mass,
        )
        if best is None or sort_key < best["sort_key"]:
            best = {
                "sort_key": sort_key,
                "pairing": pairing,
                "pair1": kinematics1,
                "pair2": kinematics2,
                "jets4_local": list(jets4_local),
            }
    if best is None:
        raise RuntimeError("No pairing was evaluated")
    return best, pairings_evaluated


def choose_h1_h2(
    pair_a: dict[str, Any],
    pair_b: dict[str, Any],
    target_mass: float,
    ordering: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if ordering == "pt":
        if pair_b["pt"] > pair_a["pt"]:
            return pair_b, pair_a
        return pair_a, pair_b
    if ordering == "mass_closest":
        if abs(pair_b["mass"] - target_mass) < abs(
            pair_a["mass"] - target_mass
        ):
            return pair_b, pair_a
        return pair_a, pair_b
    if ordering == "mass_high":
        if pair_b["mass"] > pair_a["mass"]:
            return pair_b, pair_a
        return pair_a, pair_b
    raise ValueError(f"Unknown ordering: {ordering}")


def select_jets(
    pts: Sequence[float],
    etas: Sequence[float],
    phis: Sequence[float],
    masses: Sequence[float],
    btags: Sequence[float],
    flavors: Sequence[int],
    *,
    jet_pt_min: float,
    jet_eta_max: float,
) -> list[dict[str, Any]]:
    lengths = {
        len(pts),
        len(etas),
        len(phis),
        len(masses),
        len(btags),
        len(flavors),
    }
    if len(lengths) != 1:
        raise ValueError("Jet branch lengths disagree within one event")
    selected: list[dict[str, Any]] = []
    for raw_index, values in enumerate(
        zip(pts, etas, phis, masses, btags, flavors)
    ):
        pt, eta, phi, mass, btag, flavor = values
        if pt > jet_pt_min and abs(eta) < jet_eta_max:
            selected.append(
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
    selected.sort(key=lambda jet: jet["pt"], reverse=True)
    for selected_index, jet in enumerate(selected):
        jet["selected_index"] = selected_index
    return selected


def assign_bjet_ranks(
    selected_jets: Sequence[dict[str, Any]],
    *,
    btag_min: float,
) -> list[dict[str, Any]]:
    bjets = [
        jet
        for jet in selected_jets
        if jet["btag"] > btag_min
    ]
    bjets.sort(key=lambda jet: jet["pt"], reverse=True)
    for bjet_rank, jet in enumerate(bjets):
        jet["bjet_rank"] = bjet_rank
    return bjets


def choose_threeb_candidate_jets(
    selected_jets: Sequence[dict[str, Any]],
    bjets: Sequence[dict[str, Any]],
    *,
    btag_min: float,
    target_mass: float,
) -> dict[str, Any] | None:
    if len(bjets) != 3 or len(selected_jets) < 4:
        return None
    untagged = [
        jet
        for jet in selected_jets
        if jet["btag"] <= btag_min
    ]
    if not untagged:
        return None

    # Promotion is fixed before any pairing or mass calculation.
    promoted = untagged[0]
    promoted["bjet_rank"] = -1
    jets4_local = [*bjets, promoted]
    best, pairings_evaluated = evaluate_pairings(
        jets4_local,
        target_mass,
    )
    best["combo"] = (0, 1, 2, -1)
    return {
        "best": best,
        "promoted_jet": promoted,
        "n_selected_untagged_jets": len(untagged),
        "pairings_evaluated": pairings_evaluated,
    }


def choose_fourb_candidate_jets(
    bjets: Sequence[dict[str, Any]],
    *,
    max_bjets_for_pairing: int,
    target_mass: float,
) -> dict[str, Any] | None:
    if len(bjets) < 4:
        return None
    pairing_pool = list(bjets[:max_bjets_for_pairing])
    best: dict[str, Any] | None = None
    pairings_evaluated = 0
    combinations_evaluated = 0
    for combo in combinations(range(len(pairing_pool)), 4):
        combinations_evaluated += 1
        jets4_local = [pairing_pool[index] for index in combo]
        local_best, local_count = evaluate_pairings(
            jets4_local,
            target_mass,
        )
        pairings_evaluated += local_count
        if best is None or local_best["sort_key"] < best["sort_key"]:
            best = dict(local_best)
            best["combo"] = combo
    if best is None:
        raise RuntimeError("No four-b pairing candidate was evaluated")
    return {
        "best": best,
        "pairings_evaluated": pairings_evaluated,
        "combinations_evaluated": combinations_evaluated,
    }


def build_candidate_row(
    *,
    sample: str,
    event: int,
    mode: str,
    selected_jets: Sequence[dict[str, Any]],
    bjets: Sequence[dict[str, Any]],
    best: dict[str, Any],
    target_mass: float,
    higgs_ordering: str,
    promoted_jet: dict[str, Any] | None = None,
    n_selected_untagged_jets: int = 0,
) -> dict[str, Any]:
    h1, h2 = choose_h1_h2(
        best["pair1"],
        best["pair2"],
        target_mass,
        higgs_ordering,
    )
    jets4 = sorted(
        best["jets4_local"],
        key=lambda jet: jet["pt"],
        reverse=True,
    )
    hh_vector = add_vec(
        [
            four_vector(
                jet["pt"],
                jet["eta"],
                jet["phi"],
                jet["mass"],
            )
            for jet in jets4
        ]
    )
    mbb1 = h1["mass"]
    mbb2 = h2["mass"]
    avg_mbb = 0.5 * (mbb1 + mbb2)
    delta_mbb = abs(mbb1 - mbb2)
    r_hh_125_125 = float(
        np.sqrt(
            (mbb1 - target_mass) ** 2
            + (mbb2 - target_mass) ** 2
        )
    )
    r_hh_125_120 = float(
        np.sqrt(
            (mbb1 - 125.0) ** 2
            + (mbb2 - 120.0) ** 2
        )
    )
    ht_selected_jets = float(sum(jet["pt"] for jet in selected_jets))
    ht_selected_bjets = float(sum(jet["pt"] for jet in bjets))
    ht_candidate_jets = float(sum(jet["pt"] for jet in jets4))
    h_delta_eta = h1["eta"] - h2["eta"]
    h_delta_phi = delta_phi(h1["phi"], h2["phi"])
    h_delta_r = float(
        np.sqrt(
            h_delta_eta * h_delta_eta
            + h_delta_phi * h_delta_phi
        )
    )
    h_pt_balance = float(
        abs(h1["pt"] - h2["pt"])
        / (h1["pt"] + h2["pt"] + 1e-6)
    )

    row: dict[str, Any] = {
        "sample": sample,
        "event": event,
        "n_selected_jets": len(selected_jets),
        "n_selected_bjets": len(bjets),
        "n_extra_selected_jets": max(len(selected_jets) - 4, 0),
        "n_extra_selected_bjets": max(len(bjets) - 4, 0),
        "ht_selected_jets": ht_selected_jets,
        "ht_selected_bjets": ht_selected_bjets,
        "ht_candidate_jets": ht_candidate_jets,
        "pairing": str(best["pairing"]),
        "pairing_combo_bjet_ranks": str(
            tuple(int(index) for index in best["combo"])
        ),
        "pairing_score_125_125": float(best["sort_key"][0]),
        "higgs_ordering": higgs_ordering,
        "mbb1": mbb1,
        "mbb2": mbb2,
        "avg_mbb": avg_mbb,
        "delta_mbb": delta_mbb,
        "r_hh": r_hh_125_125,
        "r_hh_125_125": r_hh_125_125,
        "r_hh_125_120": r_hh_125_120,
        "mhh": vec_mass(hh_vector),
        "hh_pt": vec_pt(hh_vector),
        "hh_eta": vec_eta(hh_vector),
        "hh_phi": vec_phi(hh_vector),
        "h1_pt": h1["pt"],
        "h1_eta": h1["eta"],
        "h1_phi": h1["phi"],
        "h2_pt": h2["pt"],
        "h2_eta": h2["eta"],
        "h2_phi": h2["phi"],
        "h_delta_eta": float(h_delta_eta),
        "h_delta_phi": float(h_delta_phi),
        "h_delta_r": h_delta_r,
        "h_pt_balance": h_pt_balance,
        "drbb1": h1["dr"],
        "drbb2": h2["dr"],
        "j1_pt": jets4[0]["pt"],
        "j2_pt": jets4[1]["pt"],
        "j3_pt": jets4[2]["pt"],
        "j4_pt": jets4[3]["pt"],
    }
    for candidate_index, jet in enumerate(jets4, start=1):
        row[f"j{candidate_index}_eta"] = jet["eta"]
        row[f"j{candidate_index}_phi"] = jet["phi"]
        row[f"j{candidate_index}_mass"] = jet["mass"]
        row[f"j{candidate_index}_btag"] = jet["btag"]
        row[f"j{candidate_index}_flavor"] = jet["flavor"]
        row[f"j{candidate_index}_raw_index"] = jet["raw_index"]
        row[f"j{candidate_index}_selected_index"] = jet["selected_index"]
        row[f"j{candidate_index}_bjet_rank"] = jet["bjet_rank"]

    if mode == MODE_THREEB:
        if promoted_jet is None:
            raise ValueError("threeb-control output requires a promoted jet")
        row.update(
            {
                "candidate_category": CANDIDATE_CATEGORY,
                "promoted_jet_rule": PROMOTED_JET_RULE,
                "promoted_jet_raw_index": promoted_jet["raw_index"],
                "promoted_jet_selected_index": promoted_jet[
                    "selected_index"
                ],
                "promoted_jet_pt": promoted_jet["pt"],
                "promoted_jet_eta": promoted_jet["eta"],
                "promoted_jet_phi": promoted_jet["phi"],
                "promoted_jet_mass": promoted_jet["mass"],
                "promoted_jet_btag": promoted_jet["btag"],
                "promoted_jet_flavor": promoted_jet["flavor"],
                "n_selected_untagged_jets": n_selected_untagged_jets,
            }
        )
    observed_columns = list(row)
    if observed_columns != output_columns(mode):
        raise RuntimeError(
            "Candidate row schema differs from the mode contract"
        )
    return row


def reconstruct_event(
    *,
    sample: str,
    event: int,
    mode: str,
    pts: Sequence[float],
    etas: Sequence[float],
    phis: Sequence[float],
    masses: Sequence[float],
    btags: Sequence[float],
    flavors: Sequence[int],
    target_mass: float = 125.0,
    jet_pt_min: float = 30.0,
    jet_eta_max: float = 2.5,
    btag_min: float = 0.0,
    max_bjets_for_pairing: int = 8,
    higgs_ordering: str = "pt",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if mode not in MODE_NAMES:
        raise ValueError(f"Unknown reconstruction mode: {mode}")
    selected_jets = select_jets(
        pts,
        etas,
        phis,
        masses,
        btags,
        flavors,
        jet_pt_min=jet_pt_min,
        jet_eta_max=jet_eta_max,
    )
    bjets = assign_bjet_ranks(
        selected_jets,
        btag_min=btag_min,
    )
    diagnostics: dict[str, Any] = {
        "mode": mode,
        "n_selected_jets": len(selected_jets),
        "n_selected_bjets": len(bjets),
        "pairings_evaluated": 0,
        "combinations_evaluated": 0,
        "promoted_jet": None,
        "rejection_reason": "",
    }

    if mode == MODE_THREEB:
        choice = choose_threeb_candidate_jets(
            selected_jets,
            bjets,
            btag_min=btag_min,
            target_mass=target_mass,
        )
        if choice is None:
            diagnostics["rejection_reason"] = (
                "requires_exactly_three_selected_tagged_jets_and_four_selected_jets"
            )
            return None, diagnostics
        diagnostics["pairings_evaluated"] = choice[
            "pairings_evaluated"
        ]
        diagnostics["combinations_evaluated"] = 1
        diagnostics["promoted_jet"] = dict(choice["promoted_jet"])
        row = build_candidate_row(
            sample=sample,
            event=event,
            mode=mode,
            selected_jets=selected_jets,
            bjets=bjets,
            best=choice["best"],
            target_mass=target_mass,
            higgs_ordering=higgs_ordering,
            promoted_jet=choice["promoted_jet"],
            n_selected_untagged_jets=choice[
                "n_selected_untagged_jets"
            ],
        )
        return row, diagnostics

    choice = choose_fourb_candidate_jets(
        bjets,
        max_bjets_for_pairing=max_bjets_for_pairing,
        target_mass=target_mass,
    )
    if choice is None:
        diagnostics["rejection_reason"] = (
            "requires_at_least_four_selected_tagged_jets"
        )
        return None, diagnostics
    diagnostics["pairings_evaluated"] = choice[
        "pairings_evaluated"
    ]
    diagnostics["combinations_evaluated"] = choice[
        "combinations_evaluated"
    ]
    row = build_candidate_row(
        sample=sample,
        event=event,
        mode=mode,
        selected_jets=selected_jets,
        bjets=bjets,
        best=choice["best"],
        target_mass=target_mass,
        higgs_ordering=higgs_ordering,
    )
    return row, diagnostics


def read_required_root_arrays(input_path: Path) -> Any:
    with uproot.open(input_path) as root_file:
        tree = root_file["Delphes"]
        return tree.arrays(
            list(REQUIRED_ROOT_BRANCHES),
            library="ak",
        )


def reconstruct_arrays(
    arrays: Any,
    *,
    sample: str,
    mode: str,
    target_mass: float = 125.0,
    jet_pt_min: float = 30.0,
    jet_eta_max: float = 2.5,
    btag_min: float = 0.0,
    max_bjets_for_pairing: int = 8,
    higgs_ordering: str = "pt",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    event_count = len(arrays["Jet.PT"])
    for event in range(event_count):
        row, event_diagnostics = reconstruct_event(
            sample=sample,
            event=event,
            mode=mode,
            pts=ak.to_numpy(arrays["Jet.PT"][event]),
            etas=ak.to_numpy(arrays["Jet.Eta"][event]),
            phis=ak.to_numpy(arrays["Jet.Phi"][event]),
            masses=ak.to_numpy(arrays["Jet.Mass"][event]),
            btags=ak.to_numpy(arrays["Jet.BTag"][event]),
            flavors=ak.to_numpy(arrays["Jet.Flavor"][event]),
            target_mass=target_mass,
            jet_pt_min=jet_pt_min,
            jet_eta_max=jet_eta_max,
            btag_min=btag_min,
            max_bjets_for_pairing=max_bjets_for_pairing,
            higgs_ordering=higgs_ordering,
        )
        diagnostics.append(event_diagnostics)
        if row is not None:
            rows.append(row)
    return rows, diagnostics


def write_parquet_output(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    mode: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_columns = output_columns(mode)
    if rows:
        if any(list(row) != expected_columns for row in rows):
            raise RuntimeError("Candidate rows do not share the mode schema")
        payload: Any = rows
    else:
        payload = typed_empty_output(mode)

    parquet_writer = Path(__file__).with_name(
        "write_parquet_from_pickle.py"
    )
    if not parquet_writer.is_file():
        raise RuntimeError(
            f"Missing isolated Parquet writer: {parquet_writer}"
        )
    temporary_pickle = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.pickle"
    )
    try:
        with temporary_pickle.open("wb") as handle:
            pickle.dump(
                payload,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        subprocess.run(
            [
                sys.executable,
                str(parquet_writer),
                "--input",
                str(temporary_pickle),
                "--output",
                str(output_path),
            ],
            check=True,
        )
    finally:
        temporary_pickle.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct HH4b exactly-3b control candidates or run the "
            "frozen-algorithm four-b parity mode."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=MODE_NAMES,
        help="Explicit reconstruction mode; it is never inferred.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input Delphes ROOT file",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output Parquet file",
    )
    parser.add_argument("--sample", required=True, help="Sample name")
    parser.add_argument("--target-mass", type=float, default=125.0)
    parser.add_argument("--jet-pt-min", type=float, default=30.0)
    parser.add_argument("--jet-eta-max", type=float, default=2.5)
    parser.add_argument("--btag-min", type=float, default=0.0)
    parser.add_argument(
        "--max-bjets-for-pairing",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--higgs-ordering",
        choices=["pt", "mass_closest", "mass_high"],
        default="pt",
    )
    args = parser.parse_args()

    arrays = read_required_root_arrays(args.input)
    rows, _ = reconstruct_arrays(
        arrays,
        sample=args.sample,
        mode=args.mode,
        target_mass=args.target_mass,
        jet_pt_min=args.jet_pt_min,
        jet_eta_max=args.jet_eta_max,
        btag_min=args.btag_min,
        max_bjets_for_pairing=args.max_bjets_for_pairing,
        higgs_ordering=args.higgs_ordering,
    )
    write_parquet_output(
        args.out,
        rows,
        mode=args.mode,
    )
    print(f"Input events: {len(arrays['Jet.PT'])}")
    print(f"Selected candidates ({args.mode}): {len(rows)}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
