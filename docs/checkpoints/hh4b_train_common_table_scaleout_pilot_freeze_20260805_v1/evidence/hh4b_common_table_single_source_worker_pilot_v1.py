#!/usr/bin/env python3
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PAIRINGS = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def clean(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def four_vector(
    pt: float,
    eta: float,
    phi: float,
    mass: float,
) -> dict[str, float]:
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    energy = math.sqrt(px * px + py * py + pz * pz + mass * mass)
    return {
        "e": float(energy),
        "px": float(px),
        "py": float(py),
        "pz": float(pz),
    }


def add_vec(
    vectors: Iterable[dict[str, float]],
) -> dict[str, float]:
    result = {"e": 0.0, "px": 0.0, "py": 0.0, "pz": 0.0}
    for vector in vectors:
        for key in result:
            result[key] += vector[key]
    return result


def vec_mass(vector: dict[str, float]) -> float:
    mass2 = (
        vector["e"] * vector["e"]
        - vector["px"] * vector["px"]
        - vector["py"] * vector["py"]
        - vector["pz"] * vector["pz"]
    )
    return float(math.sqrt(max(mass2, 0.0)))


def vec_pt(vector: dict[str, float]) -> float:
    return float(math.hypot(vector["px"], vector["py"]))


def vec_phi(vector: dict[str, float]) -> float:
    return float(math.atan2(vector["py"], vector["px"]))


def vec_eta(vector: dict[str, float]) -> float:
    pt = vec_pt(vector)
    if pt <= 0.0:
        return 0.0
    return float(math.asinh(vector["pz"] / pt))


def delta_phi(phi1: float, phi2: float) -> float:
    return float(
        math.atan2(
            math.sin(phi1 - phi2),
            math.cos(phi1 - phi2),
        )
    )


def delta_r(
    jet1: dict[str, Any],
    jet2: dict[str, Any],
) -> float:
    return float(
        math.hypot(
            jet1["eta"] - jet2["eta"],
            delta_phi(jet1["phi"], jet2["phi"]),
        )
    )


def pair_kinematics(
    jets: list[dict[str, Any]],
    pair: tuple[int, int],
) -> dict[str, Any]:
    first, second = pair
    vector = add_vec(
        [
            four_vector(
                jets[first]["pt"],
                jets[first]["eta"],
                jets[first]["phi"],
                jets[first]["mass"],
            ),
            four_vector(
                jets[second]["pt"],
                jets[second]["eta"],
                jets[second]["phi"],
                jets[second]["mass"],
            ),
        ]
    )
    return {
        "mass": vec_mass(vector),
        "pt": vec_pt(vector),
        "eta": vec_eta(vector),
        "phi": vec_phi(vector),
        "vec": vector,
        "dr": delta_r(jets[first], jets[second]),
    }


def finalize_reconstruction(best: dict[str, Any]) -> dict[str, Any]:
    pair_a = best["pair_a"]
    pair_b = best["pair_b"]

    if pair_b["pt"] > pair_a["pt"]:
        h1, h2 = pair_b, pair_a
    else:
        h1, h2 = pair_a, pair_b

    jets4 = sorted(
        best["jets4_local"],
        key=lambda jet: (
            -float(jet["pt"]),
            int(jet["selected_index"]),
        ),
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

    mbb1 = float(h1["mass"])
    mbb2 = float(h2["mass"])
    h_delta_eta = float(h1["eta"] - h2["eta"])
    h_delta_phi = delta_phi(h1["phi"], h2["phi"])

    output: dict[str, Any] = {
        "candidate_pool_size": int(best["pool_size"]),
        "candidate_tagged_jet_count": int(best["tagged_count"]),
        "candidate_selected_indices": json.dumps(
            [int(jet["selected_index"]) for jet in jets4]
        ),
        "pairing_index": int(best["pairing_index"]),
        "pairing": str(best["pairing"]),
        "pairing_score_125_125": float(
            abs(mbb1 - 125.0) + abs(mbb2 - 125.0)
        ),
        "mbb1": mbb1,
        "mbb2": mbb2,
        "delta_mbb": abs(mbb1 - mbb2),
        "r_hh_125_125": float(
            math.hypot(mbb1 - 125.0, mbb2 - 125.0)
        ),
        "r_hh_125_120": float(
            math.hypot(mbb1 - 125.0, mbb2 - 120.0)
        ),
        "mhh": vec_mass(hh_vector),
        "hh_pt": vec_pt(hh_vector),
        "hh_eta": vec_eta(hh_vector),
        "hh_phi": vec_phi(hh_vector),
        "h1_pt": float(h1["pt"]),
        "h1_eta": float(h1["eta"]),
        "h1_phi": float(h1["phi"]),
        "h2_pt": float(h2["pt"]),
        "h2_eta": float(h2["eta"]),
        "h2_phi": float(h2["phi"]),
        "h_delta_eta": h_delta_eta,
        "h_delta_phi": h_delta_phi,
        "h_delta_r": float(math.hypot(h_delta_eta, h_delta_phi)),
        "h_pt_balance": float(
            abs(h1["pt"] - h2["pt"])
            / (h1["pt"] + h2["pt"] + 1.0e-6)
        ),
        "drbb1": float(h1["dr"]),
        "drbb2": float(h2["dr"]),
    }

    for index, jet in enumerate(jets4, start=1):
        output[f"j{index}_pt"] = float(jet["pt"])
        output[f"j{index}_eta"] = float(jet["eta"])
        output[f"j{index}_phi"] = float(jet["phi"])
        output[f"j{index}_mass"] = float(jet["mass"])
        output[f"j{index}_btag"] = float(jet["btag"])
        output[f"j{index}_selected_index"] = int(
            jet["selected_index"]
        )

    output["ht_candidate_jets"] = float(
        sum(output[f"j{index}_pt"] for index in range(1, 5))
    )
    return output


def reconstruct_generalized(
    jets: list[dict[str, Any]],
) -> dict[str, Any]:
    require(len(jets) >= 4, "generalized reconstruction needs four jets")

    ordered_pool = sorted(
        jets,
        key=lambda jet: (
            -int(jet["btag"] > 0.0),
            -float(jet["pt"]),
            int(jet["selected_index"]),
        ),
    )[:8]

    best: dict[str, Any] | None = None

    for combo in combinations(range(len(ordered_pool)), 4):
        jets4_local = [ordered_pool[index] for index in combo]
        tagged_count = sum(
            jet["btag"] > 0.0 for jet in jets4_local
        )

        for pairing_index, pairing in enumerate(PAIRINGS):
            pair_a = pair_kinematics(jets4_local, pairing[0])
            pair_b = pair_kinematics(jets4_local, pairing[1])

            sort_key = (
                -int(tagged_count),
                abs(pair_a["mass"] - 125.0)
                + abs(pair_b["mass"] - 125.0),
                abs(pair_a["mass"] - pair_b["mass"]),
                -(pair_a["pt"] + pair_b["pt"]),
                tuple(int(value) for value in combo),
                int(pairing_index),
            )

            if best is None or sort_key < best["sort_key"]:
                best = {
                    "sort_key": sort_key,
                    "combo": combo,
                    "pairing_index": pairing_index,
                    "pairing": pairing,
                    "pair_a": pair_a,
                    "pair_b": pair_b,
                    "jets4_local": jets4_local,
                    "tagged_count": int(tagged_count),
                    "pool_size": len(ordered_pool),
                }

    require(best is not None, "generalized reconstruction found no candidate")
    return finalize_reconstruction(best)


def reconstruct_historical(
    jets: list[dict[str, Any]],
) -> dict[str, Any] | None:
    bjets = sorted(
        [jet for jet in jets if jet["btag"] > 0.0],
        key=lambda jet: (
            -float(jet["pt"]),
            int(jet["selected_index"]),
        ),
    )

    if len(bjets) < 4:
        return None

    pool = bjets[:8]
    best: dict[str, Any] | None = None

    for combo in combinations(range(len(pool)), 4):
        jets4_local = [pool[index] for index in combo]

        for pairing_index, pairing in enumerate(PAIRINGS):
            pair_a = pair_kinematics(jets4_local, pairing[0])
            pair_b = pair_kinematics(jets4_local, pairing[1])

            sort_key = (
                abs(pair_a["mass"] - 125.0)
                + abs(pair_b["mass"] - 125.0),
                abs(pair_a["mass"] - pair_b["mass"]),
                -(pair_a["pt"] + pair_b["pt"]),
                tuple(int(value) for value in combo),
                int(pairing_index),
            )

            if best is None or sort_key < best["sort_key"]:
                best = {
                    "sort_key": sort_key,
                    "combo": combo,
                    "pairing_index": pairing_index,
                    "pairing": pairing,
                    "pair_a": pair_a,
                    "pair_b": pair_b,
                    "jets4_local": jets4_local,
                    "tagged_count": 4,
                    "pool_size": len(pool),
                }

    require(best is not None, "historical reconstruction found no candidate")
    return finalize_reconstruction(best)


def deterministic_parquet(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    table = pa.Table.from_pandas(
        frame,
        preserve_index=False,
        safe=True,
    )
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        version="2.6",
    )


def build_products(
    source: pd.Series,
    frame: pd.DataFrame,
    coefficient: pd.Series,
    source_weights: pd.DataFrame,
    mass_aware_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    auxiliary = truthy(source["auxiliary_qcd"])
    source_fold = int(source["optimized_fold_k5"])

    coefficient_value = pd.to_numeric(
        pd.Series(
            [
                coefficient[
                    "run2_yield_coefficient_per_generator_weight"
                ]
            ]
        ),
        errors="coerce",
    ).iloc[0]

    weight_by_fold: dict[int, float | None] = {}
    for fold in range(5):
        rows = source_weights.loc[
            pd.to_numeric(
                source_weights["outer_holdout_fold"],
                errors="raise",
            ).astype(int)
            == fold
        ]
        require(len(rows) == 1, f"fold {fold} source-weight rows={len(rows)}")

        is_training = truthy(
            rows.iloc[0]["is_outer_training_source"]
        )
        value = pd.to_numeric(
            pd.Series(
                [rows.iloc[0]["per_row_training_weight_mean_one"]]
            ),
            errors="coerce",
        ).iloc[0]

        if is_training:
            require(
                np.isfinite(value) and float(value) > 0.0,
                f"fold {fold} training comparison weight is invalid",
            )
            weight_by_fold[fold] = float(value)
        else:
            require(
                pd.isna(value),
                f"fold {fold} held-out comparison weight is not null",
            )
            weight_by_fold[fold] = None

    require(
        weight_by_fold[source_fold] is None,
        "own source fold received a training weight",
    )
    require(
        sum(value is not None for value in weight_by_fold.values()) == 4,
        "source does not have exactly four outer-training weights",
    )

    accounting_rows: list[dict[str, Any]] = []
    resolved_rows: list[dict[str, Any]] = []
    historical_parity_rows = 0
    historical_max_difference = 0.0

    for event in frame.itertuples(index=False):
        nominal = event.generator_nominal_weight

        if auxiliary:
            require(
                pd.isna(nominal),
                "auxiliary event has a generator nominal weight",
            )
            require(
                pd.isna(coefficient_value),
                "auxiliary source has a physical coefficient",
            )
            physical_weight = np.nan
            selection_weight = np.nan
        else:
            require(
                not pd.isna(nominal),
                "primary event lacks a generator nominal weight",
            )
            require(
                np.isfinite(coefficient_value)
                and float(coefficient_value) > 0.0,
                "primary source coefficient is invalid",
            )
            physical_weight = float(nominal) * float(
                coefficient_value
            )
            require(
                math.isfinite(physical_weight),
                "diagnostic physical event weight is not finite",
            )
            selection_weight = (
                physical_weight
                if bool(event.broad_event_eligible)
                else 0.0
            )

        accounting = {
            "production_row_index": int(source["production_row_index"]),
            "source_uid": clean(event.source_uid),
            "group_id": clean(event.group_id),
            "source_fold": source_fold,
            "source_entry": int(event.source_entry),
            "event_uid": clean(event.event_uid),
            "sample_class": clean(event.sample_class),
            "process_or_mode": clean(event.process_or_mode),
            "workflow_population": clean(event.workflow_population),
            "class_label": int(event.class_label),
            "auxiliary_qcd": auxiliary,
            "physical_evaluation_eligible": bool(
                event.physical_evaluation_eligible
            ),
            "broad_event_eligible": bool(event.broad_event_eligible),
            "assignment_matchable": bool(event.assignment_matchable),
            "n_selected_jets": int(event.n_selected_jets),
            "n_selected_bjets": int(event.n_selected_bjets),
            "population_kind": clean(event.population_kind),
            "transport_id": clean(event.transport_id),
            "generator_weight_transport_class": clean(
                event.generator_weight_transport_class
            ),
            "generator_nominal_weight": (
                np.nan if pd.isna(nominal) else float(nominal)
            ),
            "generator_weight_sign": (
                np.nan
                if pd.isna(event.generator_weight_sign)
                else int(event.generator_weight_sign)
            ),
            "run2_yield_coefficient_per_generator_weight": (
                np.nan
                if pd.isna(coefficient_value)
                else float(coefficient_value)
            ),
            "run2_event_weight_diagnostic_not_authorized": (
                physical_weight
            ),
            "resolved_selection_contribution_weight": (
                selection_weight
            ),
            "physical_weight_application_authorized": False,
        }
        accounting_rows.append(accounting)

        if not bool(event.broad_event_eligible):
            continue

        pts = list(event.jet_pt)
        etas = list(event.jet_eta)
        phis = list(event.jet_phi)
        masses = list(event.jet_mass)
        btags = list(event.jet_btag)
        masks = list(event.jet_mask)

        require(
            len(pts)
            == len(etas)
            == len(phis)
            == len(masses)
            == len(btags)
            == len(masks),
            "selected-jet vector lengths differ",
        )

        jets: list[dict[str, Any]] = []
        for selected_index, values in enumerate(
            zip(pts, etas, phis, masses, btags, masks)
        ):
            pt, eta, phi, mass, btag, mask = values
            if not bool(mask):
                continue

            jets.append(
                {
                    "pt": float(pt),
                    "eta": float(eta),
                    "phi": float(phi),
                    "mass": float(mass),
                    "btag": float(btag),
                    "selected_index": int(selected_index),
                }
            )

        require(
            len(jets) == int(event.n_selected_jets),
            "jet-mask count differs from n_selected_jets",
        )
        require(len(jets) >= 4, "broad event has fewer than four jets")

        reconstruction = reconstruct_generalized(jets)
        historical = reconstruct_historical(jets)

        if historical is not None:
            historical_parity_rows += 1
            differences = [
                abs(
                    float(reconstruction[key])
                    - float(historical[key])
                )
                for key in [
                    "mbb1",
                    "mbb2",
                    "mhh",
                    "hh_pt",
                    "h1_pt",
                    "h2_pt",
                    "drbb1",
                    "drbb2",
                    "r_hh_125_125",
                    "r_hh_125_120",
                ]
            ]
            maximum = max(differences)
            historical_max_difference = max(
                historical_max_difference,
                maximum,
            )
            require(
                maximum <= 1.0e-9,
                (
                    "historical reconstruction parity failed for "
                    f"{event.event_uid}: {maximum}"
                ),
            )

        features: dict[str, Any] = {
            **accounting,
            "ht_selected_jets": float(
                sum(jet["pt"] for jet in jets)
            ),
            "ht_selected_bjets": float(
                sum(
                    jet["pt"]
                    for jet in jets
                    if jet["btag"] > 0.0
                )
            ),
            **reconstruction,
        }

        for target, source_name in [
            ("abs_hh_eta", "hh_eta"),
            ("abs_h1_eta", "h1_eta"),
            ("abs_h2_eta", "h2_eta"),
            ("abs_h_delta_eta", "h_delta_eta"),
            ("abs_h_delta_phi", "h_delta_phi"),
            ("abs_j1_eta", "j1_eta"),
            ("abs_j2_eta", "j2_eta"),
            ("abs_j3_eta", "j3_eta"),
            ("abs_j4_eta", "j4_eta"),
        ]:
            features[target] = abs(float(features[source_name]))

        for fold in range(5):
            features[
                f"comparison_weight_outer_fold_{fold}"
            ] = weight_by_fold[fold]

        for feature_name in mass_aware_features:
            require(
                feature_name in features,
                f"missing frozen feature: {feature_name}",
            )
            require(
                math.isfinite(float(features[feature_name])),
                f"non-finite frozen feature: {feature_name}",
            )

        resolved_rows.append(features)

    accounting_frame = pd.DataFrame(accounting_rows).sort_values(
        ["source_entry", "event_uid"]
    ).reset_index(drop=True)
    resolved_frame = pd.DataFrame(resolved_rows).sort_values(
        ["source_entry", "event_uid"]
    ).reset_index(drop=True)

    require(
        len(accounting_frame) == int(source["generated_events"]),
        "accounting row closure failed",
    )
    require(
        len(resolved_frame) == int(source["broad_eligible_rows"]),
        "resolved row closure failed",
    )
    require(
        accounting_frame["event_uid"].astype(str).is_unique,
        "accounting event_uid is not unique",
    )
    require(
        resolved_frame["event_uid"].astype(str).is_unique,
        "resolved event_uid is not unique",
    )
    require(
        set(resolved_frame["event_uid"]).issubset(
            set(accounting_frame["event_uid"])
        ),
        "resolved table is not a subset of accounting",
    )
    require(
        not accounting_frame[
            "physical_weight_application_authorized"
        ].astype(bool).any(),
        "physical authorization changed in accounting",
    )
    require(
        not resolved_frame[
            "physical_weight_application_authorized"
        ].astype(bool).any(),
        "physical authorization changed in resolved table",
    )

    summary = {
        "accounting_rows": len(accounting_frame),
        "resolved_rows": len(resolved_frame),
        "non_broad_rows": int(
            (~accounting_frame["broad_event_eligible"]).sum()
        ),
        "assignment_matchable_rows": int(
            accounting_frame["assignment_matchable"].sum()
        ),
        "historical_parity_rows": historical_parity_rows,
        "historical_maximum_absolute_difference": (
            historical_max_difference
        ),
    }

    return accounting_frame, resolved_frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--coefficient-registry", type=Path, required=True)
    parser.add_argument("--comparison-weight-registry", type=Path, required=True)
    parser.add_argument("--feature-schema", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--accounting-output", type=Path, required=True)
    parser.add_argument("--resolved-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()

    for path in [
        args.input_parquet,
        args.pilot_manifest,
        args.coefficient_registry,
        args.comparison_weight_registry,
        args.feature_schema,
        args.policy,
    ]:
        require(path.is_file(), f"missing worker input: {path}")

    require(
        sha256(args.input_parquet) == args.input_sha256,
        "transferred input Parquet checksum mismatch",
    )

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    require(policy["status"] == "pass_frozen", "policy status changed")
    require(
        policy["repository_head_before_contract_commit"]
        == "96f4e2893a31df80f81ada4851ed23a658bf046b",
        "policy pre-commit head changed",
    )
    require(
        policy["physical_coefficient_join_contract"][
            "application_authorized"
        ]
        is False,
        "policy unexpectedly authorizes physical weights",
    )
    require(
        policy["cut_contract"]["new_nominal_threshold"] is None,
        "policy unexpectedly selects a cut threshold",
    )

    manifest = pd.read_csv(
        args.pilot_manifest,
        sep="\t",
        keep_default_na=False,
    )
    matches = manifest.loc[
        pd.to_numeric(
            manifest["production_row_index"],
            errors="raise",
        ).astype(int)
        == args.row_index
    ]
    require(len(matches) == 1, "pilot row-index match is not unique")
    source = matches.iloc[0]

    require(
        clean(source["parquet_sha256"]) == args.input_sha256,
        "pilot manifest input checksum changed",
    )
    require(
        Path(clean(source["parquet_path"])).name
        == args.input_parquet.name,
        "transferred input basename changed",
    )

    coefficients = pd.read_csv(
        args.coefficient_registry,
        sep="\t",
        keep_default_na=False,
    )
    coefficient_rows = coefficients.loc[
        coefficients["source_uid"].astype(str)
        == clean(source["source_uid"])
    ]
    require(
        len(coefficient_rows) == 1,
        "source coefficient row is not unique",
    )
    coefficient = coefficient_rows.iloc[0]

    weights = pd.read_csv(
        args.comparison_weight_registry,
        sep="\t",
        keep_default_na=False,
    )
    source_weights = weights.loc[
        weights["source_uid"].astype(str)
        == clean(source["source_uid"])
    ].copy()
    require(
        len(source_weights) == 5,
        "source comparison-weight rows are not five",
    )

    feature_schema = pd.read_csv(
        args.feature_schema,
        sep="\t",
        keep_default_na=False,
    )
    mass_aware_features = (
        feature_schema.loc[
            feature_schema["feature_set"].astype(str)
            == "mass_aware_34"
        ]
        .sort_values("feature_index")["feature_name"]
        .astype(str)
        .tolist()
    )
    require(
        len(mass_aware_features) == 34,
        "mass-aware feature schema changed",
    )

    required_columns = [
        "source_uid",
        "group_id",
        "source_entry",
        "event_uid",
        "sample_class",
        "process_or_mode",
        "workflow_population",
        "class_label",
        "auxiliary_qcd",
        "physical_evaluation_eligible",
        "broad_event_eligible",
        "assignment_matchable",
        "n_selected_jets",
        "n_selected_bjets",
        "jet_pt",
        "jet_eta",
        "jet_phi",
        "jet_mass",
        "jet_btag",
        "jet_mask",
        "population_kind",
        "transport_id",
        "generator_weight_transport_class",
        "generator_nominal_weight",
        "generator_weight_sign",
        "physical_weight_application_authorized",
    ]

    frame = pd.read_parquet(
        args.input_parquet,
        columns=required_columns,
    ).sort_values("source_entry").reset_index(drop=True)

    require(
        len(frame) == int(source["generated_events"]),
        "input generated-event row closure failed",
    )
    require(
        int(frame["broad_event_eligible"].astype(bool).sum())
        == int(source["broad_eligible_rows"]),
        "input broad-row closure failed",
    )
    require(
        frame["event_uid"].astype(str).is_unique,
        "input event_uid is not unique",
    )
    require(
        set(frame["source_uid"].astype(str))
        == {clean(source["source_uid"])},
        "input source_uid changed",
    )
    require(
        not frame["physical_weight_application_authorized"]
        .astype(bool)
        .any(),
        "input physical authorization changed",
    )

    accounting_a, resolved_a, summary_a = build_products(
        source,
        frame,
        coefficient,
        source_weights,
        mass_aware_features,
    )
    accounting_b, resolved_b, summary_b = build_products(
        source,
        frame,
        coefficient,
        source_weights,
        mass_aware_features,
    )

    require(summary_a == summary_b, "deterministic summary mismatch")
    require(
        accounting_a.equals(accounting_b),
        "deterministic accounting DataFrame mismatch",
    )
    require(
        resolved_a.equals(resolved_b),
        "deterministic resolved DataFrame mismatch",
    )

    accounting_a_tmp = Path("accounting_run_a.parquet")
    accounting_b_tmp = Path("accounting_run_b.parquet")
    resolved_a_tmp = Path("resolved_run_a.parquet")
    resolved_b_tmp = Path("resolved_run_b.parquet")

    deterministic_parquet(accounting_a, accounting_a_tmp)
    deterministic_parquet(accounting_b, accounting_b_tmp)
    deterministic_parquet(resolved_a, resolved_a_tmp)
    deterministic_parquet(resolved_b, resolved_b_tmp)

    accounting_hash_a = sha256(accounting_a_tmp)
    accounting_hash_b = sha256(accounting_b_tmp)
    resolved_hash_a = sha256(resolved_a_tmp)
    resolved_hash_b = sha256(resolved_b_tmp)

    require(
        accounting_hash_a == accounting_hash_b,
        "accounting Parquet reruns are not byte-identical",
    )
    require(
        resolved_hash_a == resolved_hash_b,
        "resolved Parquet reruns are not byte-identical",
    )

    os.replace(accounting_a_tmp, args.accounting_output)
    os.replace(resolved_a_tmp, args.resolved_output)
    accounting_b_tmp.unlink(missing_ok=True)
    resolved_b_tmp.unlink(missing_ok=True)

    receipt = {
        "schema_version": 1,
        "status": "pass",
        "repository_head": args.expected_head,
        "production_row_index": args.row_index,
        "source_uid": clean(source["source_uid"]),
        "group_id": clean(source["group_id"]),
        "source_fold": int(source["optimized_fold_k5"]),
        "sample_class": clean(source["sample_class"]),
        "process_or_mode": clean(source["process_or_mode"]),
        "workflow_population": clean(source["workflow_population"]),
        "generator_weight_transport_class": clean(
            source["generator_weight_transport_class"]
        ),
        "input_parquet_basename": args.input_parquet.name,
        "input_parquet_sha256": args.input_sha256,
        "accounting_output_sha256": accounting_hash_a,
        "resolved_output_sha256": resolved_hash_a,
        "byte_identical_internal_rerun": True,
        **summary_a,
        "physical_weight_application_authorized": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "models_trained": 0,
        "runtime_seconds": time.time() - started,
    }
    args.receipt_output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"ROW_INDEX={args.row_index}")
    print(f"SOURCE_UID={clean(source['source_uid'])}")
    print(f"ACCOUNTING_ROWS={summary_a['accounting_rows']}")
    print(f"RESOLVED_ROWS={summary_a['resolved_rows']}")
    print(
        "HISTORICAL_PARITY_ROWS="
        f"{summary_a['historical_parity_rows']}"
    )
    print("BYTE_IDENTICAL_INTERNAL_RERUN=PASS")
    print("PHYSICAL_WEIGHT_APPLICATION_AUTHORIZED=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")
    print("MODELS_TRAINED=0")
    print("WORKER_RESULT=PASS")


if __name__ == "__main__":
    main()
