#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import math
import os

import awkward as ak
import numpy as np
import pandas as pd
import uproot

REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])

OUTDIR = (
    REPO
    / "outputs/audits/run2_frozen_v2_smoke_validation_2026_07_15"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    {
        "sample": "ggF_HH",
        "old_root": (
            STORE
            / "root/ggf_hh4b_ak4ak8_10k_pythia8_delphes.root"
        ),
        "new_root": (
            STORE
            / "root_run2_frozen_v2_smoke/"
              "ggf_hh4b_ak4ak8_run2_frozen_v2_10k_delphes.root"
        ),
        "old_candidates": (
            STORE
            / "parquet/ggf_hh4b_ak4ak8_10k_hh4b_candidates.parquet"
        ),
        "new_candidates": (
            STORE
            / "parquet_run2_frozen_v2_smoke/"
              "ggf_hh4b_ak4ak8_run2_frozen_v2_10k_candidates.parquet"
        ),
    },
    {
        "sample": "ttbar",
        "old_root": (
            STORE
            / "root/"
              "ttbar_extra50k_ak4ak8_v1_shard000_pythia8_delphes.root"
        ),
        "new_root": (
            STORE
            / "root_run2_frozen_v2_smoke/"
              "ttbar_ak4ak8_run2_frozen_v2_shard000_10k_delphes.root"
        ),
        "old_candidates": (
            STORE
            / "parquet/"
              "ttbar_extra50k_ak4ak8_v1_shard000_hh4b_candidates.parquet"
        ),
        "new_candidates": (
            STORE
            / "parquet_run2_frozen_v2_smoke/"
              "ttbar_ak4ak8_run2_frozen_v2_shard000_10k_candidates.parquet"
        ),
    },
]

BRANCHES = [
    "Jet.PT",
    "Jet.Eta",
    "Jet.Phi",
    "Jet.Mass",
    "Jet.Flavor",
    "Jet.BTag",
]

PT_EDGES = [40.0, 60.0, 80.0, 120.0, 200.0, 400.0, np.inf]


def flavor_group(flavor):
    value = abs(int(flavor))
    if value == 5:
        return "b"
    if value == 4:
        return "c"
    return "light"


def expected_efficiency(group, pt):
    if group == "b":
        value = (
            0.85
            * math.tanh(0.0025 * pt)
            * (25.0 / (1.0 + 0.063 * pt))
        )
    elif group == "c":
        value = (
            0.25
            * math.tanh(0.018 * pt)
            * (1.0 / (1.0 + 0.0013 * pt))
        )
    else:
        value = 0.01 + 0.000038 * pt

    return float(np.clip(value, 0.0, 1.0))


def pt_bin_label(pt):
    for low, high in zip(PT_EDGES[:-1], PT_EDGES[1:]):
        if low <= pt < high:
            upper = "inf" if np.isinf(high) else f"{high:g}"
            return f"{low:g}-{upper}"
    return None


def load(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with uproot.open(path) as root_file:
        return root_file["Delphes"].arrays(BRANCHES, library="ak")


def candidate_summary(sample, version, path):
    frame = pd.read_parquet(path)

    return {
        "sample": sample,
        "version": version,
        "n_candidates": len(frame),
        "median_mbb1": frame["mbb1"].median(),
        "median_mbb2": frame["mbb2"].median(),
        "median_mhh": frame["mhh"].median(),
    }


kinematic_rows = []
migration_rows = []
multiplicity_rows = []
efficiency_stats = defaultdict(
    lambda: {
        "n_jets": 0,
        "n_tagged": 0,
        "expected_sum": 0.0,
    }
)
candidate_rows = []

for spec in SAMPLES:
    sample = spec["sample"]

    old = load(spec["old_root"])
    new = load(spec["new_root"])

    n_old = len(old["Jet.PT"])
    n_new = len(new["Jet.PT"])

    n_compared = min(n_old, n_new)
    n_same_multiplicity = 0
    n_identical_kinematics = 0

    max_pt_difference = 0.0
    max_eta_difference = 0.0
    max_phi_difference = 0.0
    max_mass_difference = 0.0

    jet_tag_migration = {
        "0_to_0": 0,
        "0_to_1": 0,
        "1_to_0": 0,
        "1_to_1": 0,
    }

    jet_flavor_changed = 0
    jet_flavor_compared = 0

    event_multiplicity = {
        "old": defaultdict(int),
        "new": defaultdict(int),
    }

    for event_index in range(n_compared):
        old_pt = np.asarray(
            ak.to_numpy(old["Jet.PT"][event_index]),
            dtype=float,
        )
        new_pt = np.asarray(
            ak.to_numpy(new["Jet.PT"][event_index]),
            dtype=float,
        )

        old_eta = np.asarray(
            ak.to_numpy(old["Jet.Eta"][event_index]),
            dtype=float,
        )
        new_eta = np.asarray(
            ak.to_numpy(new["Jet.Eta"][event_index]),
            dtype=float,
        )

        old_phi = np.asarray(
            ak.to_numpy(old["Jet.Phi"][event_index]),
            dtype=float,
        )
        new_phi = np.asarray(
            ak.to_numpy(new["Jet.Phi"][event_index]),
            dtype=float,
        )

        old_mass = np.asarray(
            ak.to_numpy(old["Jet.Mass"][event_index]),
            dtype=float,
        )
        new_mass = np.asarray(
            ak.to_numpy(new["Jet.Mass"][event_index]),
            dtype=float,
        )

        old_flavor = np.asarray(
            ak.to_numpy(old["Jet.Flavor"][event_index]),
            dtype=int,
        )
        new_flavor = np.asarray(
            ak.to_numpy(new["Jet.Flavor"][event_index]),
            dtype=int,
        )

        old_tag = np.asarray(
            ak.to_numpy(old["Jet.BTag"][event_index]),
            dtype=int,
        )
        new_tag = np.asarray(
            ak.to_numpy(new["Jet.BTag"][event_index]),
            dtype=int,
        )

        if len(old_pt) != len(new_pt):
            continue

        n_same_multiplicity += 1

        if len(old_pt):
            max_pt_difference = max(
                max_pt_difference,
                float(np.max(np.abs(old_pt - new_pt))),
            )
            max_eta_difference = max(
                max_eta_difference,
                float(np.max(np.abs(old_eta - new_eta))),
            )
            max_phi_difference = max(
                max_phi_difference,
                float(np.max(np.abs(old_phi - new_phi))),
            )
            max_mass_difference = max(
                max_mass_difference,
                float(np.max(np.abs(old_mass - new_mass))),
            )

        kinematics_identical = (
            np.allclose(old_pt, new_pt, rtol=0.0, atol=1e-10)
            and np.allclose(old_eta, new_eta, rtol=0.0, atol=1e-10)
            and np.allclose(old_phi, new_phi, rtol=0.0, atol=1e-10)
            and np.allclose(old_mass, new_mass, rtol=0.0, atol=1e-10)
        )

        if kinematics_identical:
            n_identical_kinematics += 1

        selected_old = (old_pt >= 40.0) & (np.abs(old_eta) <= 2.5)
        selected_new = (new_pt >= 40.0) & (np.abs(new_eta) <= 2.5)

        old_n_tags = int(np.sum(old_tag[selected_old] != 0))
        new_n_tags = int(np.sum(new_tag[selected_new] != 0))

        old_category = str(old_n_tags) if old_n_tags < 4 else "4+"
        new_category = str(new_n_tags) if new_n_tags < 4 else "4+"

        event_multiplicity["old"][old_category] += 1
        event_multiplicity["new"][new_category] += 1

        for version, pt, eta, flavor, tag in [
            ("old_dr0p5", old_pt, old_eta, old_flavor, old_tag),
            ("new_dr0p4", new_pt, new_eta, new_flavor, new_tag),
        ]:
            mask = (pt >= 40.0) & (np.abs(eta) <= 2.5)

            for jet_pt, jet_flavor, jet_tag in zip(
                pt[mask],
                flavor[mask],
                tag[mask],
            ):
                group = flavor_group(jet_flavor)
                bin_label = pt_bin_label(float(jet_pt))

                key = (sample, version, group, bin_label)
                efficiency_stats[key]["n_jets"] += 1
                efficiency_stats[key]["n_tagged"] += int(jet_tag != 0)
                efficiency_stats[key]["expected_sum"] += (
                    expected_efficiency(group, float(jet_pt))
                )

        common = min(
            len(old_tag),
            len(new_tag),
            len(old_flavor),
            len(new_flavor),
        )

        for index in range(common):
            old_value = int(old_tag[index] != 0)
            new_value = int(new_tag[index] != 0)
            jet_tag_migration[f"{old_value}_to_{new_value}"] += 1

            jet_flavor_compared += 1
            jet_flavor_changed += int(
                int(old_flavor[index]) != int(new_flavor[index])
            )

    kinematic_rows.append({
        "sample": sample,
        "n_old_events": n_old,
        "n_new_events": n_new,
        "n_events_compared": n_compared,
        "n_same_jet_multiplicity": n_same_multiplicity,
        "n_identical_jet_kinematics": n_identical_kinematics,
        "fraction_identical_jet_kinematics": (
            n_identical_kinematics / n_compared
            if n_compared
            else np.nan
        ),
        "max_abs_pt_difference": max_pt_difference,
        "max_abs_eta_difference": max_eta_difference,
        "max_abs_phi_difference": max_phi_difference,
        "max_abs_mass_difference": max_mass_difference,
    })

    migration_rows.append({
        "sample": sample,
        **jet_tag_migration,
        "n_flavors_compared": jet_flavor_compared,
        "n_flavors_changed": jet_flavor_changed,
        "fraction_flavors_changed": (
            jet_flavor_changed / jet_flavor_compared
            if jet_flavor_compared
            else np.nan
        ),
    })

    for version in ["old", "new"]:
        for category in ["0", "1", "2", "3", "4+"]:
            multiplicity_rows.append({
                "sample": sample,
                "version": version,
                "tag_multiplicity": category,
                "n_events": event_multiplicity[version][category],
            })

    candidate_rows.append(
        candidate_summary(
            sample,
            "old_dr0p5",
            spec["old_candidates"],
        )
    )
    candidate_rows.append(
        candidate_summary(
            sample,
            "new_dr0p4",
            spec["new_candidates"],
        )
    )

efficiency_rows = []

for key, stat in sorted(efficiency_stats.items()):
    sample, version, group, bin_label = key
    n_jets = stat["n_jets"]
    n_tagged = stat["n_tagged"]

    observed = n_tagged / n_jets if n_jets else np.nan
    expected = stat["expected_sum"] / n_jets if n_jets else np.nan

    uncertainty = (
        math.sqrt(observed * (1.0 - observed) / n_jets)
        if n_jets and 0.0 <= observed <= 1.0
        else np.nan
    )

    pull = (
        (observed - expected) / uncertainty
        if uncertainty and uncertainty > 0
        else np.nan
    )

    efficiency_rows.append({
        "sample": sample,
        "version": version,
        "flavor": group,
        "pt_bin_GeV": bin_label,
        "n_jets": n_jets,
        "n_tagged": n_tagged,
        "observed_efficiency": observed,
        "mean_formula_efficiency": expected,
        "binomial_uncertainty": uncertainty,
        "observed_minus_expected": observed - expected,
        "pull": pull,
    })

kinematics = pd.DataFrame(kinematic_rows)
migration = pd.DataFrame(migration_rows)
multiplicity = pd.DataFrame(multiplicity_rows)
efficiencies = pd.DataFrame(efficiency_rows)
candidates = pd.DataFrame(candidate_rows)

for name, frame in [
    ("kinematic_identity", kinematics),
    ("tag_and_flavor_migration", migration),
    ("event_tag_multiplicity", multiplicity),
    ("tag_efficiency_by_flavor_pt", efficiencies),
    ("candidate_comparison", candidates),
]:
    frame.to_csv(OUTDIR / f"{name}.csv", index=False)
    (OUTDIR / f"{name}.md").write_text(
        frame.to_markdown(index=False) + "\n"
    )

readme = """# Frozen-v2 Delphes-card smoke validation

The old and new ROOT files were produced from identical HepMC events.

Required validation conditions:

1. Jet multiplicities and kinematics should be identical.
2. Differences should be confined to flavor association and b tagging.
3. Observed tagging efficiencies should be statistically compatible with
   the formulas encoded in the Delphes card.
4. Candidate-count changes must be documented before the card is frozen.

The old card uses JetFlavorAssociation DeltaR=0.5.
The new candidate card uses DeltaR=0.4.
"""

(OUTDIR / "README.md").write_text(readme)

print("\n=== Kinematic identity ===")
print(kinematics.to_string(index=False))

print("\n=== Tag and flavor migration ===")
print(migration.to_string(index=False))

print("\n=== Event tag multiplicity ===")
print(multiplicity.to_string(index=False))

print("\n=== Candidate comparison ===")
print(candidates.to_string(index=False))

print("\n=== Tag efficiency by flavor and pT ===")
print(efficiencies.to_string(index=False))

print("\nWrote:", OUTDIR)
