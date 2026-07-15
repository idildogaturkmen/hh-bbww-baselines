#!/usr/bin/env python3

from pathlib import Path
import json
import math
import os

import awkward as ak
import numpy as np
import pandas as pd
import uproot

REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])

CATALOG = REPO / "datasets/ak4ak8_v1/ak4ak8_v1_local_manifest.csv"
OUTDIR = REPO / "outputs/tables/cms_run2_regions_ak4ak8_2026_07_15"
OUTDIR.mkdir(parents=True, exist_ok=True)

# CMS HIG-20-005 reconstructed Higgs mass centers.
ANALYSIS_CENTER = (125.0, 120.0)
VALIDATION_CENTER = (179.0, 172.0)

K_PAIR = ANALYSIS_CENTER[0] / ANALYSIS_CENTER[1]
SQRT_ONE_PLUS_K2 = math.sqrt(1.0 + K_PAIR * K_PAIR)

PT_MIN = 40.0
ETA_MAX = 2.5

PAIRINGS = [
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
]


def p4(pt, eta, phi, mass):
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    momentum2 = px * px + py * py + pz * pz
    energy = math.sqrt(max(momentum2 + mass * mass, 0.0))
    return np.array([energy, px, py, pz], dtype=float)


def p4_pt(v):
    return math.hypot(v[1], v[2])


def p4_mass(v):
    value = v[0] * v[0] - np.dot(v[1:], v[1:])
    return math.sqrt(max(value, 0.0))


def p4_eta(v):
    p = math.sqrt(max(np.dot(v[1:], v[1:]), 0.0))
    denominator = p - v[3]
    numerator = p + v[3]

    if denominator <= 0 or numerator <= 0:
        return math.copysign(20.0, v[3])

    return 0.5 * math.log(numerator / denominator)


def p4_phi(v):
    return math.atan2(v[2], v[1])


def delta_phi(a, b):
    value = a - b
    while value > math.pi:
        value -= 2.0 * math.pi
    while value <= -math.pi:
        value += 2.0 * math.pi
    return value


def delta_r(v1, v2):
    deta = p4_eta(v1) - p4_eta(v2)
    dphi = delta_phi(p4_phi(v1), p4_phi(v2))
    return math.hypot(deta, dphi)


def boost_to_total_rest_frame(v, total):
    if total[0] <= 0:
        return v.copy()

    beta = total[1:] / total[0]
    beta2 = float(np.dot(beta, beta))

    if beta2 <= 0 or beta2 >= 1:
        return v.copy()

    gamma = 1.0 / math.sqrt(1.0 - beta2)
    beta_dot_p = float(np.dot(beta, v[1:]))

    energy_prime = gamma * (v[0] - beta_dot_p)
    coefficient = ((gamma - 1.0) * beta_dot_p / beta2) - gamma * v[0]
    momentum_prime = v[1:] + coefficient * beta

    return np.concatenate(([energy_prime], momentum_prime))


def construct_pairing(jets, pairing):
    (a, b), (c, d) = pairing

    first = jets[a]["p4"] + jets[b]["p4"]
    second = jets[c]["p4"] + jets[d]["p4"]

    first_indices = (a, b)
    second_indices = (c, d)

    # CMS defines H1 as the higher-pT Higgs candidate.
    if p4_pt(second) > p4_pt(first):
        first, second = second, first
        first_indices, second_indices = second_indices, first_indices

    m1 = p4_mass(first)
    m2 = p4_mass(second)

    distance = abs(m1 - K_PAIR * m2) / SQRT_ONE_PLUS_K2

    return {
        "h1": first,
        "h2": second,
        "h1_indices": first_indices,
        "h2_indices": second_indices,
        "m1": m1,
        "m2": m2,
        "distance": distance,
    }


def choose_pairing(jets):
    candidates = [construct_pairing(jets, p) for p in PAIRINGS]
    candidates.sort(key=lambda item: item["distance"])

    delta_distance = candidates[1]["distance"] - candidates[0]["distance"]

    if delta_distance > 30.0:
        chosen = candidates[0]
        mode = "minimum_distance"
    else:
        total = sum((jet["p4"] for jet in jets), np.zeros(4))

        def com_pt_score(candidate):
            h1_rest = boost_to_total_rest_frame(candidate["h1"], total)
            h2_rest = boost_to_total_rest_frame(candidate["h2"], total)
            return p4_pt(h1_rest) + p4_pt(h2_rest)

        chosen = max(candidates[:2], key=com_pt_score)
        mode = "four_jet_com_pt"

    chosen = dict(chosen)
    chosen["delta_distance"] = delta_distance
    chosen["pairing_mode"] = mode
    return chosen


def mass_region(m1, m2, center):
    chi = math.hypot(m1 - center[0], m2 - center[1])

    if chi < 25.0:
        region = "SR"
    elif chi < 50.0:
        region = "CR"
    else:
        region = "outside"

    return chi, region


def combined_region(prefix, btag_region, region):
    if region == "outside":
        return f"{prefix}_outside"

    return f"{prefix}{btag_region}{region}"


catalog = pd.read_csv(CATALOG)
catalog = catalog[catalog["root_exists"] == True].copy()  # noqa: E712

rows = []
file_summaries = []
observed_btag_values = set()

for _, entry in catalog.iterrows():
    root_path = Path(entry["root_path"])

    if not root_path.exists():
        print("Skipping missing:", root_path)
        continue

    print("Reading:", root_path)

    with uproot.open(root_path) as root_file:
        tree = root_file["Delphes"]
        available = set(tree.keys())

        branches = [
            "Jet.PT",
            "Jet.Eta",
            "Jet.Phi",
            "Jet.Mass",
            "Jet.BTag",
        ]

        optional = [
            "Electron.PT",
            "Electron.Eta",
            "Muon.PT",
            "Muon.Eta",
        ]

        branches.extend([name for name in optional if name in available])
        arrays = tree.arrays(branches, library="ak")

    n_events = len(arrays["Jet.PT"])
    n_ge4jets = 0
    n_ge3tags = 0
    n_rows_before_trigger = 0
    n_rows_trigger_proxy = 0

    for event_index in range(n_events):
        pts = np.asarray(ak.to_numpy(arrays["Jet.PT"][event_index]), dtype=float)
        etas = np.asarray(ak.to_numpy(arrays["Jet.Eta"][event_index]), dtype=float)
        phis = np.asarray(ak.to_numpy(arrays["Jet.Phi"][event_index]), dtype=float)
        masses = np.asarray(ak.to_numpy(arrays["Jet.Mass"][event_index]), dtype=float)
        btags = np.asarray(ak.to_numpy(arrays["Jet.BTag"][event_index]), dtype=float)

        observed_btag_values.update(float(x) for x in np.unique(btags))

        selected = np.where((pts >= PT_MIN) & (np.abs(etas) <= ETA_MAX))[0]

        if len(selected) < 4:
            continue

        n_ge4jets += 1

        selected_pts = pts[selected]
        selected_btags = btags[selected]

        # Approximation of selecting the four highest DeepJet-score jets:
        # primary ordering by Delphes BTag, then by pT for ties.
        ordering = np.lexsort((-selected_pts, -selected_btags))
        chosen_indices = selected[ordering[:4]]

        jets = []
        for index in chosen_indices:
            jets.append({
                "source_index": int(index),
                "pt": float(pts[index]),
                "eta": float(etas[index]),
                "phi": float(phis[index]),
                "mass": float(masses[index]),
                "btag": float(btags[index]),
                "p4": p4(
                    float(pts[index]),
                    float(etas[index]),
                    float(phis[index]),
                    float(masses[index]),
                ),
            })

        n_tags = int(sum(jet["btag"] > 0.5 for jet in jets))

        if n_tags < 3:
            continue

        n_ge3tags += 1
        btag_region = "4b" if n_tags == 4 else "3b"

        pairing = choose_pairing(jets)
        h1 = pairing["h1"]
        h2 = pairing["h2"]

        m1 = pairing["m1"]
        m2 = pairing["m2"]

        total = sum((jet["p4"] for jet in jets), np.zeros(4))
        mhh = p4_mass(total)

        analysis_chi, analysis_mass_region = mass_region(
            m1, m2, ANALYSIS_CENTER
        )
        validation_chi, validation_mass_region = mass_region(
            m1, m2, VALIDATION_CENTER
        )

        analysis_region = combined_region(
            "A", btag_region, analysis_mass_region
        )
        validation_region = combined_region(
            "V", btag_region, validation_mass_region
        )

        pts_desc = sorted(
            [float(x) for x in selected_pts],
            reverse=True,
        )

        # Conservative 2018-like trigger/offline proxy.
        trigger_proxy = (
            len(pts_desc) >= 4
            and pts_desc[0] >= 75.0
            and pts_desc[1] >= 60.0
            and pts_desc[2] >= 45.0
            and pts_desc[3] >= 40.0
            and float(np.sum(selected_pts)) >= 330.0
        )

        electron_veto = True
        if "Electron.PT" in arrays.fields:
            e_pt = np.asarray(
                ak.to_numpy(arrays["Electron.PT"][event_index]),
                dtype=float,
            )
            e_eta = np.asarray(
                ak.to_numpy(arrays["Electron.Eta"][event_index]),
                dtype=float,
            )
            electron_veto = not np.any(
                (e_pt > 15.0) & (np.abs(e_eta) < 2.4)
            )

        muon_veto = True
        if "Muon.PT" in arrays.fields:
            mu_pt = np.asarray(
                ak.to_numpy(arrays["Muon.PT"][event_index]),
                dtype=float,
            )
            mu_eta = np.asarray(
                ak.to_numpy(arrays["Muon.Eta"][event_index]),
                dtype=float,
            )
            muon_veto = not np.any(
                (mu_pt > 10.0) & (np.abs(mu_eta) < 2.4)
            )

        lepton_veto_proxy = electron_veto and muon_veto

        mk = (
            K_PAIR * m1 + m2
        ) / SQRT_ONE_PLUS_K2

        m_perp = (
            m1 - K_PAIR * m2
        ) / SQRT_ONE_PLUS_K2

        h1_i, h1_j = pairing["h1_indices"]
        h2_i, h2_j = pairing["h2_indices"]

        row = {
            "role": entry["role"],
            "process": entry["process"],
            "source_tag": entry["tag"],
            "source_root": str(root_path),
            "event_index": event_index,
            "n_selected_jets": int(len(selected)),
            "n_selected_btags_four_jets": n_tags,
            "btag_region": btag_region,
            "trigger_proxy_2018": bool(trigger_proxy),
            "lepton_veto_proxy": bool(lepton_veto_proxy),
            "analysis_chi": analysis_chi,
            "analysis_mass_region": analysis_mass_region,
            "analysis_region": analysis_region,
            "validation_chi": validation_chi,
            "validation_mass_region": validation_mass_region,
            "validation_region": validation_region,
            "mk": mk,
            "m_perp": m_perp,
            "pairing_mode": pairing["pairing_mode"],
            "pairing_delta_distance": pairing["delta_distance"],
            "mbb1": m1,
            "mbb2": m2,
            "avg_mbb": 0.5 * (m1 + m2),
            "delta_mbb": abs(m1 - m2),
            "mhh": mhh,
            "h1_pt": p4_pt(h1),
            "h2_pt": p4_pt(h2),
            "h1_eta": p4_eta(h1),
            "h2_eta": p4_eta(h2),
            "delta_eta_hh": abs(p4_eta(h1) - p4_eta(h2)),
            "delta_phi_hh": abs(delta_phi(p4_phi(h1), p4_phi(h2))),
            "drbb1": delta_r(jets[h1_i]["p4"], jets[h1_j]["p4"]),
            "drbb2": delta_r(jets[h2_i]["p4"], jets[h2_j]["p4"]),
        }

        for j, jet in enumerate(jets, start=1):
            row[f"j{j}_pt"] = jet["pt"]
            row[f"j{j}_eta"] = jet["eta"]
            row[f"j{j}_phi"] = jet["phi"]
            row[f"j{j}_mass"] = jet["mass"]
            row[f"j{j}_btag"] = jet["btag"]

        rows.append(row)
        n_rows_before_trigger += 1

        if trigger_proxy and lepton_veto_proxy:
            n_rows_trigger_proxy += 1

    file_summaries.append({
        "role": entry["role"],
        "process": entry["process"],
        "source_tag": entry["tag"],
        "n_generated_events": n_events,
        "n_events_ge4_selected_jets": n_ge4jets,
        "n_events_ge3_tags_in_chosen_four": n_ge3tags,
        "n_region_rows_before_trigger_proxy": n_rows_before_trigger,
        "n_region_rows_after_trigger_and_lepton_proxies": n_rows_trigger_proxy,
    })

events = pd.DataFrame(rows)
files = pd.DataFrame(file_summaries)

events.to_parquet(
    OUTDIR / "cms_run2_region_events_ak4ak8.parquet",
    index=False,
)
events.to_csv(
    OUTDIR / "cms_run2_region_events_ak4ak8.csv",
    index=False,
)

files.to_csv(
    OUTDIR / "cms_run2_region_file_summary.csv",
    index=False,
)
(
    OUTDIR / "cms_run2_region_file_summary.md"
).write_text(files.to_markdown(index=False) + "\n")

selected = events[
    events["trigger_proxy_2018"]
    & events["lepton_veto_proxy"]
].copy()

analysis_summary = (
    selected.groupby(
        ["role", "process", "analysis_region"],
        as_index=False,
    )
    .size()
    .rename(columns={"size": "n_events"})
)

validation_summary = (
    selected.groupby(
        ["role", "process", "validation_region"],
        as_index=False,
    )
    .size()
    .rename(columns={"size": "n_events"})
)

analysis_summary.to_csv(
    OUTDIR / "cms_run2_analysis_region_counts.csv",
    index=False,
)
(
    OUTDIR / "cms_run2_analysis_region_counts.md"
).write_text(analysis_summary.to_markdown(index=False) + "\n")

validation_summary.to_csv(
    OUTDIR / "cms_run2_validation_region_counts.csv",
    index=False,
)
(
    OUTDIR / "cms_run2_validation_region_counts.md"
).write_text(validation_summary.to_markdown(index=False) + "\n")

btag_audit = {
    "observed_Jet_BTag_values": sorted(observed_btag_values),
    "btag_interpretation": (
        "Four jets are ranked by Delphes Jet.BTag and then pT. "
        "If Jet.BTag is binary, this is a proxy rather than a continuous "
        "DeepJet-score ranking."
    ),
    "analysis_centers_GeV": list(ANALYSIS_CENTER),
    "validation_centers_GeV": list(VALIDATION_CENTER),
    "signal_region_radius_GeV": 25.0,
    "control_region_outer_radius_GeV": 50.0,
    "jet_pt_min_GeV": PT_MIN,
    "jet_abs_eta_max": ETA_MAX,
}

(
    OUTDIR / "cms_run2_region_definition.json"
).write_text(json.dumps(btag_audit, indent=2))

readme = """# CMS Run-2-inspired 3b/4b region tables

These tables implement the mass-region and 3b/4b structure of CMS
HIG-20-005 using AK4 Delphes objects.

Important limitations:

- Delphes Jet.BTag may be binary and is not a calibrated continuous DeepJet
  discriminator.
- The trigger and lepton selections are proxies.
- No CMS collision data are used.
- These tables support a simulation closure study; they are not themselves
  a CMS data-driven background measurement.
"""

(OUTDIR / "README.md").write_text(readme)

print("\n=== File summary ===")
print(files.to_string(index=False))

print("\n=== Analysis-region counts after trigger/lepton proxies ===")
print(analysis_summary.to_string(index=False))

print("\n=== Validation-region counts after trigger/lepton proxies ===")
print(validation_summary.to_string(index=False))

print("\n=== BTag audit ===")
print(json.dumps(btag_audit, indent=2))

print("\nWrote:", OUTDIR)
