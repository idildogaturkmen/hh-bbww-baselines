#!/usr/bin/env python3
"""Evaluate one explicitly authorized HH->4b validation source exactly once.

The worker emits only additive source summaries and fixed-bin distribution
accumulators.  It never writes an event-level validation table, never scans a
cut, and never evaluates the sealed test split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd


NOMINAL_THRESHOLDS = {
    "exact3tag": {
        "r_hh_125_125": 36.40814019639858,
        "ht_candidate_jets": 176.5458068847656,
    },
    "ge4tag": {
        "r_hh_125_125": 33.92808917804956,
        "mhh": 164.73708096689654,
        "abs_h_delta_eta": 6.904302164473993,
    },
}
DISTRIBUTION_SPECS = {
    "r_hh_125_125": (0.0, 300.0, 60, r"$R_{HH}(125,125)$"),
    "mhh": (0.0, 3000.0, 60, r"$m_{HH}$ [GeV]"),
    "h2_pt": (0.0, 1500.0, 60, r"$p_T(H_2)$ [GeV]"),
    "ht_candidate_jets": (0.0, 3000.0, 60, r"$H_T^{\mathrm{cand.}}$ [GeV]"),
    "max_drbb": (0.0, 6.5, 52, r"$\max\Delta R_{bb}$"),
    "abs_h_delta_eta": (0.0, 12.0, 48, r"$|\Delta\eta(H_1,H_2)|$"),
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def clean(value: Any) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def durable_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def install_preexisting_attempt_marker(
    source_path: Path,
    destination_path: Path,
    *,
    source_uid: str,
    row_index: int,
    authorization_sha256: str,
    durable_marker_uri: str,
    execution_head: str,
    authorization_head: str,
) -> dict[str, Any]:
    """Install local evidence for a marker already made durable by the runner.

    Portable batch runners cannot rely on a shared POSIX filesystem.  They must
    create the marker atomically in durable remote storage, verify its checksum,
    and only then invoke this worker.  The worker validates those exact bytes and
    installs an exclusive local copy that is shipped with the source products.
    """

    require(source_path.is_file() and not source_path.is_symlink(), "preexisting attempt marker is invalid")
    require(clean(durable_marker_uri), "durable attempt marker URI is missing")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    require(
        payload.get("status")
        == "validation_source_open_attempt_durable_do_not_rerun",
        "preexisting attempt marker status changed",
    )
    require(payload.get("source_uid") == source_uid, "preexisting attempt marker UID changed")
    require(
        payload.get("production_row_index") == row_index,
        "preexisting attempt marker row changed",
    )
    require(
        payload.get("authorization_sha256") == authorization_sha256,
        "preexisting attempt marker authorization changed",
    )
    require(payload.get("repository_head") == execution_head, "preexisting attempt marker head changed")
    require(
        payload.get("authorization_repository_head") == authorization_head,
        "preexisting attempt marker authorization head changed",
    )
    require(
        payload.get("durable_marker_uri") == durable_marker_uri,
        "preexisting attempt marker URI changed",
    )
    require(
        payload.get("source_payload_access_may_begin") is True,
        "preexisting marker does not permit source access",
    )
    require(
        payload.get("rerun_forbidden_even_if_downstream_bookkeeping_fails") is True,
        "preexisting marker rerun gate changed",
    )
    require(payload.get("test_payloads_opened") == 0, "test marker count changed")
    encoded = source_path.read_bytes()
    with destination_path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    require(sha256(source_path) == sha256(destination_path), "attempt marker copy changed")
    return payload


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_nominal_pass(category: str, features: dict[str, Any]) -> bool:
    require(category in NOMINAL_THRESHOLDS, f"unknown category: {category}")
    values = {name: float(features[name]) for name in NOMINAL_THRESHOLDS[category]}
    require(all(math.isfinite(value) for value in values.values()), "non-finite cut input")
    if category == "exact3tag":
        return (
            values["r_hh_125_125"] < NOMINAL_THRESHOLDS[category]["r_hh_125_125"]
            and values["ht_candidate_jets"]
            > NOMINAL_THRESHOLDS[category]["ht_candidate_jets"]
        )
    return (
        values["r_hh_125_125"] < NOMINAL_THRESHOLDS[category]["r_hh_125_125"]
        and values["mhh"] > NOMINAL_THRESHOLDS[category]["mhh"]
        and values["abs_h_delta_eta"]
        < NOMINAL_THRESHOLDS[category]["abs_h_delta_eta"]
    )


def empty_summary() -> dict[str, Any]:
    return {
        "total_rows": 0,
        "selected_rows": 0,
        "negative_weight_rows": 0,
        "selected_negative_weight_rows": 0,
        "total_weights": [],
        "selected_weights": [],
    }


def update_summary(summary: dict[str, Any], weight: float, selected: bool) -> None:
    require(math.isfinite(weight), "non-finite physical event weight")
    summary["total_rows"] += 1
    summary["negative_weight_rows"] += int(weight < 0.0)
    summary["total_weights"].append(weight)
    if selected:
        summary["selected_rows"] += 1
        summary["selected_negative_weight_rows"] += int(weight < 0.0)
        summary["selected_weights"].append(weight)


def finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    total = [float(value) for value in summary.pop("total_weights")]
    selected = [float(value) for value in summary.pop("selected_weights")]
    return {
        **summary,
        "total_signed_yield": math.fsum(total),
        "selected_signed_yield": math.fsum(selected),
        "total_sumw2": math.fsum(value * value for value in total),
        "selected_sumw2": math.fsum(value * value for value in selected),
    }


def make_distribution_rows(
    category_events: dict[str, list[dict[str, Any]]],
    sample_class: str,
    source_uid: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category in ("exact3tag", "ge4tag"):
        events = category_events[category]
        weights = np.asarray([event["weight"] for event in events], dtype=np.float64)
        selected = np.asarray([event["selected"] for event in events], dtype=bool)
        for variable, (minimum, maximum, bins, label) in DISTRIBUTION_SPECS.items():
            values = np.asarray([event[variable] for event in events], dtype=np.float64)
            require(np.isfinite(values).all(), f"non-finite {variable}")
            edges = np.linspace(minimum, maximum, bins + 1)
            for stage, mask in (
                ("preselection", np.ones(len(events), dtype=bool)),
                ("postselection", selected),
            ):
                chosen_values = values[mask]
                chosen_weights = weights[mask]
                counts = np.histogram(chosen_values, bins=edges)[0]
                yields = np.histogram(chosen_values, bins=edges, weights=chosen_weights)[0]
                sumw2 = np.histogram(
                    chosen_values, bins=edges, weights=np.square(chosen_weights)
                )[0]
                underflow = chosen_values < minimum
                overflow = chosen_values >= maximum
                underflow_rows = int(np.count_nonzero(underflow))
                overflow_rows = int(np.count_nonzero(overflow))
                underflow_yield = float(chosen_weights[underflow].sum())
                overflow_yield = float(chosen_weights[overflow].sum())
                underflow_sumw2 = float(np.square(chosen_weights[underflow]).sum())
                overflow_sumw2 = float(np.square(chosen_weights[overflow]).sum())
                for bin_index in range(bins):
                    rows.append(
                        {
                            "source_uid": source_uid,
                            "category_id": category,
                            "sample_class": sample_class,
                            "selection_stage": stage,
                            "selection_id": "fixed_nominal_deployment_cut",
                            "variable": variable,
                            "axis_label_latex": label,
                            "bin_index": bin_index,
                            "bin_low_inclusive": float(edges[bin_index]),
                            "bin_high_exclusive": float(edges[bin_index + 1]),
                            "rows": int(counts[bin_index]),
                            "signed_yield": float(yields[bin_index]),
                            "sumw2": float(sumw2[bin_index]),
                            "underflow_rows_distribution_total": underflow_rows,
                            "overflow_rows_distribution_total": overflow_rows,
                            "underflow_signed_yield_distribution_total": underflow_yield,
                            "overflow_signed_yield_distribution_total": overflow_yield,
                            "underflow_sumw2_distribution_total": underflow_sumw2,
                            "overflow_sumw2_distribution_total": overflow_sumw2,
                            "validation_payloads_opened": 1,
                            "test_payloads_opened": 0,
                        }
                    )
    expected = sum(spec[2] for spec in DISTRIBUTION_SPECS.values()) * 2 * 2
    require(len(rows) == expected, f"distribution rows={len(rows)}, expected={expected}")
    return rows


def build_source_products(
    table,
    source: pd.Series,
    coefficient: float,
    reconstruction,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    columns = [
        "source_uid",
        "source_entry",
        "event_uid",
        "raw_event_weight_available",
        "raw_event_weight",
        "broad_event_eligible",
        "n_selected_jets",
        "jet_pt",
        "jet_eta",
        "jet_phi",
        "jet_mass",
        "jet_btag",
        "jet_mask",
    ]
    require(set(columns).issubset(table.column_names), "validation broad table schema changed")
    data = table.select(columns).to_pydict()
    expected_events = int(source["generated_events"])
    require(table.num_rows == expected_events, "validation source event closure failed")
    uid = clean(source["source_uid"])
    require(set(data["source_uid"]) == {uid}, "validation source UID drift")
    require(data["source_entry"] == list(range(expected_events)), "source entry drift")
    require(len(set(data["event_uid"])) == expected_events, "event UID collision")

    summaries = {"exact3tag": empty_summary(), "ge4tag": empty_summary()}
    category_events: dict[str, list[dict[str, Any]]] = {
        "exact3tag": [],
        "ge4tag": [],
    }
    broad_rows = 0
    below_three_tags = 0

    for index in range(expected_events):
        if not bool(data["broad_event_eligible"][index]):
            continue
        broad_rows += 1
        require(bool(data["raw_event_weight_available"][index]), "raw event weight unavailable")
        nominal = float(data["raw_event_weight"][index])
        require(math.isfinite(nominal), "raw event weight non-finite")

        arrays = [
            data[name][index]
            for name in ("jet_pt", "jet_eta", "jet_phi", "jet_mass", "jet_btag", "jet_mask")
        ]
        require(len({len(values) for values in arrays}) == 1, "jet vector length mismatch")
        jets = []
        for selected_index, values in enumerate(zip(*arrays)):
            pt, eta, phi, mass, btag, mask = values
            if bool(mask):
                jets.append(
                    {
                        "pt": float(pt),
                        "eta": float(eta),
                        "phi": float(phi),
                        "mass": float(mass),
                        "btag": float(btag),
                        "selected_index": selected_index,
                    }
                )
        require(len(jets) == int(data["n_selected_jets"][index]), "selected-jet count drift")
        features = reconstruction.reconstruct_generalized(jets)
        features["max_drbb"] = max(float(features["drbb1"]), float(features["drbb2"]))
        features["abs_h_delta_eta"] = abs(float(features["h_delta_eta"]))
        tags = int(features["candidate_tagged_jet_count"])
        if tags < 3:
            below_three_tags += 1
            continue
        category = "exact3tag" if tags == 3 else "ge4tag"
        require(tags == 3 or tags >= 4, "category mapping failure")
        selected = fixed_nominal_pass(category, features)
        weight = nominal * coefficient
        update_summary(summaries[category], weight, selected)
        category_events[category].append(
            {
                "weight": weight,
                "selected": selected,
                **{variable: float(features[variable]) for variable in DISTRIBUTION_SPECS},
            }
        )

    finalized = {category: finalize_summary(values) for category, values in summaries.items()}
    summary = {
        "source_uid": uid,
        "production_row_index": int(source["production_row_index"]),
        "group_id": clean(source["group_id"]),
        "sample_class": clean(source["sample_class"]),
        "process_or_mode": clean(source["process_or_mode"]),
        "generated_events": expected_events,
        "broad_event_rows": broad_rows,
        "broad_rows_below_three_candidate_tags": below_three_tags,
        "category_rows": sum(value["total_rows"] for value in finalized.values()),
        "run2_yield_coefficient_per_generator_weight": coefficient,
        "categories": finalized,
    }
    distribution_rows = make_distribution_rows(
        category_events, clean(source["sample_class"]), uid
    )
    return summary, distribution_rows


def validate_authorization(
    authorization: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    require(authorization.get("status") == "authorized_one_time_cut_baseline_validation", "authorization status mismatch")
    require(authorization.get("validation_access_authorized") is True, "validation not authorized")
    require(authorization.get("validation_payloads_opened_before_authorization") == 0, "validation was already opened")
    require(authorization.get("test_payloads_opened") == 0, "test is not sealed")
    require(authorization.get("authorized_validation_sources") == 116, "authorized source count changed")
    require(authorization.get("auxiliary_qcd_validation_sources_authorized") == 0, "auxiliary QCD was authorized")
    require(authorization.get("nominal_thresholds") == NOMINAL_THRESHOLDS, "authorized nominal thresholds changed")
    authorization_head = clean(authorization.get("repository_head"))
    require(
        len(authorization_head) == 40
        and all(character in "0123456789abcdef" for character in authorization_head),
        "authorization repository head is invalid",
    )
    require(
        len(args.expected_head) == 40
        and all(character in "0123456789abcdef" for character in args.expected_head),
        "execution repository head is invalid",
    )
    expected_hashes = authorization.get("authorized_sha256", {})
    actual = {
        "source_access_manifest": sha256(args.source_access_manifest),
        "physical_coefficient_registry": sha256(args.coefficient_registry),
        "broad_feature_extractor": sha256(args.extractor),
        "candidate_reconstruction_module": sha256(args.reconstruction_module),
        "validation_source_worker": sha256(Path(__file__).resolve()),
    }
    require(
        all(expected_hashes.get(name) == value for name, value in actual.items()),
        "authorized input/code SHA256 map mismatch",
    )
    require(clean(authorization.get("master_train_only_checkpoint_commit")), "master train-only checkpoint not bound")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--source-access-manifest", type=Path, required=True)
    parser.add_argument("--coefficient-registry", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--reconstruction-module", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preexisting-attempt-marker", type=Path)
    parser.add_argument("--durable-attempt-marker-uri", default="")
    args = parser.parse_args()

    for path in [
        args.source_access_manifest,
        args.coefficient_registry,
        args.extractor,
        args.reconstruction_module,
        args.authorization,
    ]:
        require(path.is_file(), f"missing authorized validation input: {path}")
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    validate_authorization(authorization, args)

    access = pd.read_csv(args.source_access_manifest, sep="\t", keep_default_na=False)
    coefficients = pd.read_csv(args.coefficient_registry, sep="\t", keep_default_na=False)
    require(len(access) == 121 and access["source_uid"].nunique() == 121, "source manifest closure changed")
    require(int(access["physical_evaluation_eligible"].map(truthy).sum()) == 116, "physical source closure changed")
    matches = access.loc[
        pd.to_numeric(access["production_row_index"], errors="raise").astype(int)
        == args.row_index
    ]
    require(len(matches) == 1, f"row index matched {len(matches)} sources")
    source = matches.iloc[0]
    require(clean(source["split"]) == "validation", "worker source is not validation")
    require(truthy(source["physical_evaluation_eligible"]), "nonphysical source must remain unopened")
    require(not truthy(source["auxiliary_qcd"]), "auxiliary QCD must remain unopened")
    coefficient_matches = coefficients.loc[
        coefficients["source_uid"].astype(str).map(clean) == clean(source["source_uid"])
    ]
    require(len(coefficient_matches) == 1, "coefficient row not unique")
    coefficient = float(
        coefficient_matches.iloc[0]["run2_yield_coefficient_per_generator_weight"]
    )
    require(math.isfinite(coefficient) and coefficient > 0.0, "coefficient invalid")

    output_dir = args.output_root.resolve()
    require(output_dir.is_dir() and not output_dir.is_symlink(), "output root is invalid")
    summary_path = output_dir / f"source_{args.row_index:04d}_summary.json"
    distribution_path = output_dir / f"source_{args.row_index:04d}_distributions.tsv"
    attempt_path = output_dir / f"source_{args.row_index:04d}_VALIDATION_OPEN_DO_NOT_RERUN.json"
    require(
        not summary_path.exists()
        and not distribution_path.exists()
        and not attempt_path.exists(),
        "validation source was already attempted or evaluated",
    )

    args.work_root.mkdir(parents=True, exist_ok=True)
    work = args.work_root / f"validation_source_{args.row_index:04d}_{os.getpid()}"
    require(not work.exists(), f"work directory already exists: {work}")
    work.mkdir()
    summary_tmp = output_dir / f".{summary_path.name}.tmp.{os.getpid()}"
    distribution_tmp = output_dir / f".{distribution_path.name}.tmp.{os.getpid()}"
    require(not summary_tmp.exists() and not distribution_tmp.exists(), "temporary output collision")
    started = time.time()

    try:
        extractor = load_module("hh4b_validation_broad_extractor", args.extractor)
        reconstruction = load_module("hh4b_validation_reconstruction", args.reconstruction_module)
        extractor.MAX_EVENTS_PER_SOURCE = int(source["generated_events"])
        marker_payload = {
                "schema_version": 1,
                "status": "validation_source_open_attempt_durable_do_not_rerun",
                "repository_head": args.expected_head,
                "authorization_repository_head": authorization["repository_head"],
                "authorization_sha256": sha256(args.authorization),
                "production_row_index": args.row_index,
                "source_uid": clean(source["source_uid"]),
                "source_payload_access_may_begin": True,
                "rerun_forbidden_even_if_downstream_bookkeeping_fails": True,
                "test_payloads_opened": 0,
                "created_unix_time": time.time(),
            }
        if args.preexisting_attempt_marker is None:
            require(
                not clean(args.durable_attempt_marker_uri),
                "durable marker URI supplied without a preexisting marker",
            )
            durable_exclusive_json(attempt_path, marker_payload)
        else:
            install_preexisting_attempt_marker(
                args.preexisting_attempt_marker,
                attempt_path,
                source_uid=clean(source["source_uid"]),
                row_index=args.row_index,
                authorization_sha256=sha256(args.authorization),
                durable_marker_uri=clean(args.durable_attempt_marker_uri),
                execution_head=args.expected_head,
                authorization_head=authorization["repository_head"],
            )
        root_path, resolution_mode, resolution_chain = extractor.stage_source(source, work)
        table, feature_metadata = extractor.build_table(source, root_path)
        source_summary, distributions = build_source_products(
            table, source, coefficient, reconstruction
        )
        source_summary.update(
            {
                "schema_version": 1,
                "status": "pass_one_time_fixed_nominal_validation_source_evaluation",
                "repository_head": args.expected_head,
                "authorization_repository_head": authorization["repository_head"],
                "master_train_only_checkpoint_commit": authorization[
                    "master_train_only_checkpoint_commit"
                ],
                "authorization_sha256": sha256(args.authorization),
                "root_resolution_mode": resolution_mode,
                "root_resolution_chain": resolution_chain,
                "feature_metadata": feature_metadata,
                "nominal_thresholds": NOMINAL_THRESHOLDS,
                "cut_scan_performed": False,
                "threshold_adjustment_performed": False,
                "family_adjustment_performed": False,
                "source_payload_opened_once": True,
                "source_payload_rerun_performed": False,
                "source_open_attempt_marker": attempt_path.name,
                "source_open_attempt_marker_sha256": sha256(attempt_path),
                "durable_attempt_marker_uri": clean(
                    args.durable_attempt_marker_uri
                ),
                "validation_payloads_opened": 1,
                "test_payloads_opened": 0,
                "elapsed_seconds": time.time() - started,
            }
        )
        summary_tmp.write_text(
            json.dumps(source_summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(distributions).to_csv(
            distribution_tmp, sep="\t", index=False, lineterminator="\n"
        )
        source_summary["distribution_sha256"] = sha256(distribution_tmp)
        source_summary["distribution_rows"] = len(distributions)
        summary_tmp.write_text(
            json.dumps(source_summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(distribution_tmp, distribution_path)
        os.replace(summary_tmp, summary_path)
        print("VALIDATION_SOURCE_EVALUATION=PASS")
        print(f"PRODUCTION_ROW_INDEX={args.row_index}")
        print(f"SOURCE_UID={clean(source['source_uid'])}")
        print("SOURCE_PAYLOAD_OPENED_ONCE=TRUE")
        print("CUT_SCAN_PERFORMED=FALSE")
        print("VALIDATION_PAYLOADS_OPENED=1")
        print("TEST_PAYLOADS_OPENED=0")
    finally:
        summary_tmp.unlink(missing_ok=True)
        distribution_tmp.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
