#!/usr/bin/env python3
"""Run one frozen outer-fold/variant HH4b global BDT workload.

Development source payloads are fully tuned and thresholded before held-outer
payloads are opened. Validation and test manifests are not inputs to this
program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
import xgboost
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from hh4b_bdt_apples_to_apples_common import (
    historical_rhh34_mask,
    select_fixed_efficiency_threshold,
    validate_feature_contract,
    validate_training_weights,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = REPO / "configs/baselines/hh4b_bdt_apples_to_apples_v1.json"
DEFAULT_PLAN = REPO / "docs/checkpoints/hh4b_train_common_table_full_output_freeze_20260805_v1/evidence/production_manifest/full_464_production_plan.tsv"
DEFAULT_AUTHORIZATION = REPO / "docs/checkpoints/hh4b_train_run2_138fb_physical_authorization_freeze_20260806_v1/evidence/authorization/physical_weight_authorization_registry_464.tsv"

BACKGROUND_FAMILY = {
    "bbh_hbb_4fs": "single_higgs", "ggh_hbb": "single_higgs",
    "qcd_hardqcd": "qcd_multijet", "schannel_single_top": "single_top",
    "tchannel_antitop": "single_top", "tchannel_top": "single_top",
    "ttbar_inclusive": "ttbar", "tth_hbb": "top_associated",
    "tttt": "top_associated", "ttw": "top_associated",
    "ttz_zbb": "top_associated", "tw_antitop": "single_top",
    "tw_top": "single_top", "vbf_hbb": "single_higgs",
    "wh_hbb": "single_higgs", "ww": "diboson", "wwz_zbb": "triboson",
    "wz_zbb": "diboson", "wzz_zbb": "triboson", "zbbbb": "zbbbb",
    "zzz_zbb": "triboson",
}
EXPECTED_BACKGROUND_FAMILIES = tuple(sorted(set(BACKGROUND_FAMILY.values())))
BASE_COLUMNS = [
    "source_uid", "group_id", "source_fold", "event_uid", "sample_class",
    "process_or_mode", "class_label", "auxiliary_qcd",
    "physical_evaluation_eligible", "resolved_selection_contribution_weight",
    "r_hh_125_125",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> str:
    return str(value).strip()


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes"}


def load_registry(plan_path: Path, authorization_path: Path) -> pd.DataFrame:
    plan = pd.read_csv(plan_path, sep="\t", dtype=str, keep_default_na=False)
    auth = pd.read_csv(authorization_path, sep="\t", dtype=str, keep_default_na=False)
    require(len(plan) == 464 and len(auth) == 464, "source registry count changed")
    keys = ["production_row_index", "source_uid"]
    merged = plan.merge(
        auth[keys + ["physical_weight_application_authorized"]],
        on=keys, how="inner", validate="one_to_one",
    )
    merged["auxiliary"] = merged["auxiliary_qcd"].map(truthy)
    merged["authorized"] = merged["physical_weight_application_authorized"].map(truthy)
    primary = merged.loc[(~merged.auxiliary) & merged.authorized].copy()
    primary["source_fold"] = pd.to_numeric(primary.optimized_fold_k5, errors="raise").astype(int)
    primary["broad_eligible_rows"] = pd.to_numeric(primary.broad_eligible_rows, errors="raise").astype(int)
    require(len(primary) == 441, "primary source count changed")
    require(int(primary.broad_eligible_rows.sum()) == 1_042_397, "primary row count changed")
    require(set(primary.source_fold) == set(range(5)), "fold coverage changed")
    return primary.sort_values("production_row_index", key=lambda x: pd.to_numeric(x)).reset_index(drop=True)


def feature_names(protocol: dict[str, Any], variant: str) -> list[str]:
    blind = list(protocol["features"]["mass_plane_blind"])
    append = list(protocol["features"]["mass_aware_append"])
    exclusions = list(protocol["features"]["mass_plane_blind_exclusions"])
    aware = blind + append
    validate_feature_contract(aware, blind, exclusions)
    require(len(aware) == 34 and len(blind) == 30, "feature cardinality changed")
    return aware if variant == "global_mass_aware" else blind


def read_sources(registry: pd.DataFrame, features: list[str], access_log: list[dict[str, Any]], role: str) -> pd.DataFrame:
    columns = list(dict.fromkeys(BASE_COLUMNS + features))
    parts: list[pd.DataFrame] = []
    for source in registry.itertuples(index=False):
        path = Path(clean(source.final_resolved_path))
        require(path.is_file(), f"missing source table: {path}")
        schema = set(pq.read_schema(path).names)
        require(not (set(columns) - schema), f"missing columns in {source.source_uid}: {sorted(set(columns)-schema)}")
        frame = pd.read_parquet(path, columns=columns)
        require(len(frame) == int(source.broad_eligible_rows), f"row closure failed: {source.source_uid}")
        require(frame.source_uid.astype(str).eq(clean(source.source_uid)).all(), "source UID mismatch")
        require(pd.to_numeric(frame.source_fold).astype(int).eq(int(source.source_fold)).all(), "fold mismatch")
        require(~frame.auxiliary_qcd.astype(bool).any(), "auxiliary QCD entered primary population")
        require(frame.physical_evaluation_eligible.astype(bool).all(), "physical eligibility mismatch")
        access_log.append({"role": role, "source_uid": clean(source.source_uid), "fold": int(source.source_fold), "rows": len(frame), "path": str(path), "opened_after_selection_freeze": role == "held_outer"})
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    for column in features + ["resolved_selection_contribution_weight", "r_hh_125_125"]:
        result[column] = pd.to_numeric(result[column], errors="raise")
        require(np.isfinite(result[column].to_numpy(float)).all(), f"nonfinite column: {column}")
    result["class_label"] = pd.to_numeric(result.class_label, errors="raise").astype(np.int8)
    result["source_fold"] = pd.to_numeric(result.source_fold, errors="raise").astype(np.int8)
    require(result.event_uid.astype(str).is_unique, f"duplicate event UID in {role}")
    return result


def hierarchical_weights(frame: pd.DataFrame) -> np.ndarray:
    require(len(frame) > 0, "empty weighting population")
    classes = frame.sample_class.astype(str).to_numpy()
    processes = frame.process_or_mode.astype(str).to_numpy()
    groups = frame.group_id.astype(str).to_numpy()
    strata = np.empty(len(frame), dtype=object)
    signal = classes == "signal"
    background = classes == "background"
    require(np.all(signal | background) and signal.any() and background.any(), "class coverage changed")
    require(set(processes[signal]) == {"ggf_hh4b", "vbf_hh4b"}, "signal mode coverage changed")
    unknown = sorted(set(processes[background]) - set(BACKGROUND_FAMILY))
    require(not unknown, f"unknown background process: {unknown}")
    strata[signal] = [f"signal::{value}" for value in processes[signal]]
    strata[background] = [f"background::{BACKGROUND_FAMILY[value]}" for value in processes[background]]
    require(set(value.split("::", 1)[1] for value in strata[background]) == set(EXPECTED_BACKGROUND_FAMILIES), "background family coverage changed")
    weights = np.zeros(len(frame), dtype=np.float64)
    for class_name, class_mask in (("signal", signal), ("background", background)):
        class_strata = sorted(set(strata[class_mask]))
        class_fraction = 0.5
        for stratum in class_strata:
            stratum_mask = strata == stratum
            stratum_groups = sorted(set(groups[stratum_mask]))
            source_fraction = class_fraction / len(class_strata) / len(stratum_groups)
            for group in stratum_groups:
                mask = stratum_mask & (groups == group)
                weights[mask] = source_fraction / int(mask.sum())
    weights *= len(weights) / weights.sum()
    validate_training_weights(weights)
    require(abs(float(weights.mean()) - 1.0) < 1e-12, "training-weight normalization failed")
    return weights


def estimator(protocol: dict[str, Any], candidate: dict[str, Any]) -> XGBClassifier:
    params = dict(protocol["implementation"]["fixed_parameters"])
    params.update({key: value for key, value in candidate.items() if key != "id"})
    params["random_state"] = int(protocol["implementation"]["random_seed"])
    return XGBClassifier(**params)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, required=True, choices=range(5))
    parser.add_argument("--variant", required=True, choices=("global_mass_aware", "global_mass_plane_blind"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    args = parser.parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    require(protocol["study_id"] == "hh4b_bdt_apples_to_apples_v1", "protocol identity changed")
    require(protocol["sealed_data"]["bdt_validation_payloads_opened"] == 0 and protocol["sealed_data"]["bdt_test_payloads_opened"] == 0, "sealed-data contract changed")
    require(xgboost.__version__ == protocol["implementation"]["xgboost"], "XGBoost version mismatch")
    require(sklearn.__version__ == protocol["implementation"]["sklearn"], "sklearn version mismatch")
    features = feature_names(protocol, args.variant)
    registry = load_registry(args.plan, args.authorization)
    development_registry = registry.loc[registry.source_fold != args.outer_fold]
    outer_registry = registry.loc[registry.source_fold == args.outer_fold]
    access_log: list[dict[str, Any]] = []

    development = read_sources(development_registry, features, access_log, "outer_development")
    require(len(development) == 1_042_397 - protocol["folds"]["primary_fold_rows"][args.outer_fold], "development row count changed")
    X = development[features].to_numpy(np.float32)
    y = development.class_label.to_numpy(np.int8)
    development_weights = hierarchical_weights(development)
    candidates = protocol["hyperparameter_candidates"]
    require(len(candidates) == 8, "candidate count changed")
    inner_folds = [fold for fold in range(5) if fold != args.outer_fold]
    trial_rows = []
    inner_predictions: dict[str, np.ndarray] = {}
    for candidate in candidates:
        oof = np.full(len(development), np.nan, dtype=np.float32)
        fold_aucs = []
        for inner_fold in inner_folds:
            held = development.source_fold.to_numpy() == inner_fold
            fit = ~held
            require(held.any() and fit.any() and not np.any(held & fit), "inner partition failure")
            fit_weights = hierarchical_weights(development.loc[fit].reset_index(drop=True))
            model = estimator(protocol, candidate)
            model.fit(X[fit], y[fit], sample_weight=fit_weights)
            oof[held] = model.predict_proba(X[held])[:, 1].astype(np.float32)
            fold_aucs.append(float(roc_auc_score(y[held], oof[held], sample_weight=development_weights[held])))
        require(np.isfinite(oof).all(), "incomplete inner-OOF predictions")
        pooled_auc = float(roc_auc_score(y, oof, sample_weight=development_weights))
        trial_rows.append({"candidate_id": candidate["id"], "pooled_inner_oof_weighted_roc_auc": pooled_auc, "inner_fold_aucs": fold_aucs, "max_depth": candidate["max_depth"], "n_estimators": candidate["n_estimators"]})
        inner_predictions[candidate["id"]] = oof
    trial_rows.sort(key=lambda row: (-row["pooled_inner_oof_weighted_roc_auc"], row["max_depth"], row["n_estimators"], row["candidate_id"]))
    selected_id = trial_rows[0]["candidate_id"]
    selected_candidate = next(candidate for candidate in candidates if candidate["id"] == selected_id)
    selected_oof = inner_predictions[selected_id]
    threshold = select_fixed_efficiency_threshold(selected_oof, y, development_weights, float(protocol["operating_point"]["target_signal_efficiency"]))
    selection_freeze = {"outer_fold": args.outer_fold, "variant": args.variant, "selected_candidate_id": selected_id, "threshold": threshold, "selection_source": "pooled_inner_oof_development_only", "outer_payloads_opened_before_freeze": 0, "validation_payloads_opened": 0, "test_payloads_opened": 0}
    atomic_json(args.output_dir / "selection_freeze.json", selection_freeze)

    final_model = estimator(protocol, selected_candidate)
    final_model.fit(X, y, sample_weight=development_weights)
    final_model.save_model(args.output_dir / "final_model.json")
    inner_table = development[["event_uid", "source_uid", "group_id", "source_fold", "class_label"]].copy()
    inner_table["development_training_weight"] = development_weights
    inner_table["selected_candidate_inner_oof_score"] = selected_oof
    inner_table.to_parquet(args.output_dir / "selected_inner_oof_predictions.parquet", index=False)
    del inner_predictions, X, development

    held_outer = read_sources(outer_registry, features, access_log, "held_outer")
    require(len(held_outer) == protocol["folds"]["primary_fold_rows"][args.outer_fold], "held-outer row count changed")
    outer_score = final_model.predict_proba(held_outer[features].to_numpy(np.float32))[:, 1].astype(np.float32)
    output = held_outer[["event_uid", "source_uid", "group_id", "source_fold", "sample_class", "process_or_mode", "class_label", "resolved_selection_contribution_weight", "r_hh_125_125"]].copy()
    output["variant"] = args.variant
    output["score"] = outer_score
    output["score_threshold"] = threshold
    output["bdt_selected"] = outer_score >= threshold
    output["historical_rhh34_selected"] = historical_rhh34_mask(output.r_hh_125_125.to_numpy(float), np.ones(len(output), dtype=bool))
    output.to_parquet(args.output_dir / "held_outer_predictions.parquet", index=False)
    pd.DataFrame(trial_rows).to_json(args.output_dir / "hyperparameter_trials.json", orient="records", indent=2)
    pd.DataFrame(access_log).to_csv(args.output_dir / "payload_access_log.tsv", sep="\t", index=False)
    gain = final_model.get_booster().get_score(importance_type="gain")
    importance = pd.DataFrame([{"feature": name, "gain": float(gain.get(f"f{index}", 0.0))} for index, name in enumerate(features)])
    importance.to_csv(args.output_dir / "feature_importance_gain.tsv", sep="\t", index=False)
    receipt = {
        "status": "pass", "study_id": protocol["study_id"], "outer_fold": args.outer_fold,
        "variant": args.variant, "features": features, "candidate_count": len(candidates),
        "inner_fold_count": 4, "selected_candidate_id": selected_id, "score_threshold": threshold,
        "development_rows": len(inner_table), "held_outer_rows": len(output),
        "held_outer_payloads_opened_before_selection_freeze": 0,
        "validation_payloads_opened": 0, "test_payloads_opened": 0,
        "protocol_sha256": sha256(args.protocol), "feature_registry_sha256": sha256(REPO / "docs/analysis/hh4b_bdt_apples_to_apples_v1_features.tsv"),
        "python": platform.python_version(), "xgboost": xgboost.__version__, "sklearn": sklearn.__version__,
        "wall_seconds": time.time() - started, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "artifacts": {},
    }
    for path in sorted(args.output_dir.iterdir()):
        if path.name != "receipt.json":
            receipt["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    atomic_json(args.output_dir / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "outer_fold", "variant", "selected_candidate_id", "wall_seconds", "max_rss_kib")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
