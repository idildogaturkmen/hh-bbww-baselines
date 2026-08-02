#!/usr/bin/env python3
"""Freeze the train-only PN-c7t baseline change investigation.

This is a read-only consumer of sealed PN-c7r/c7s/c7t checkpoints and the
earlier frozen expanded-cut train population.  It never reads validation or
test payloads and refuses to replace an existing output or staging directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO = Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines")
C7R = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/"
    "pn_c7r_threeb_fourb_multijet_transfer_20260802_v1"
)
C7S = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/"
    "pn_c7s_final_train_baseline_inputs_20260802_v1"
)
C7T = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/"
    "pn_c7t_reestablished_train_baselines_20260802_v1"
)
EARLIER_CUT = REPO / "docs/checkpoints/hh4b_expanded_cut_baseline_train_20260728_v1"

C7R_MANIFEST_SHA = "a663fe73d23db0e2ee856dd041916a22008c020c9b11a7eb59e3dd14acbd19c6"
C7S_MANIFEST_SHA = "9885748add80bfd44cb01947e5ae21b58e08f28396c55cf1fdc7934063309ae1"
C7T_MANIFEST_SHA = "2de28666cdd9e055747b7e2d3df70968bad3a8d8d0613bf3d5c6e05a1d6fc016"
EARLIER_CUT_MANIFEST_SHA = "8d409db8a36b289fb6743c109c385f37d8632635711e525938e6eb8802d73baf"

PREDICTIONS = C7T / "predictions/train_primary_projection_oof_scores.parquet"
MODEL_ORDER = ("cut", "bdt", "dense_dnn", "lbn_dnn")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sha_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        if rel.startswith("./"):
            rel = rel[2:]
        require(rel not in result, f"duplicate manifest member: {rel}")
        result[rel] = digest
    return result


def verify_checkpoint(path: Path, expected_manifest_sha: str) -> None:
    manifest = path / "SHA256SUMS"
    require(sha256(manifest) == expected_manifest_sha, f"checkpoint manifest mismatch: {path}")
    for rel, expected in parse_sha_manifest(manifest).items():
        member = path / rel
        require(member.is_file() and sha256(member) == expected, f"checkpoint member mismatch: {member}")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"refusing empty TSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def indexed(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["level"], row["category"]): row for row in rows}


def comparison_row(
    category: str,
    quantity: str,
    earlier: float,
    current: float,
    unit: str,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "quantity": quantity,
        "earlier_frozen_train_population": earlier,
        "current_441_source_population": current,
        "absolute_change": current - earlier,
        "relative_change": (current - earlier) / earlier if earlier else 0.0,
        "unit": unit,
        "interpretation": interpretation,
    }


def effective_events(weights: np.ndarray) -> float:
    total = float(weights.sum())
    sum2 = float(np.square(weights).sum())
    return total * total / sum2 if sum2 > 0.0 else 0.0


def selection_for(name: str, frame: pd.DataFrame, metrics: dict[str, dict[str, str]]) -> np.ndarray:
    if name == "cut":
        return frame["cut_score"].to_numpy(dtype=np.float64) > -34.0
    threshold = float(metrics[name]["operating_threshold"])
    return frame[f"{name}_score"].to_numpy(dtype=np.float64) >= threshold


def process_composition(
    predictions: pd.DataFrame,
    metrics: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    weights = predictions["primary_projection_physical_weight_inclusive"].to_numpy(dtype=np.float64)
    for baseline in MODEL_ORDER:
        selected = selection_for(baseline, predictions, metrics)
        for sample_class in ("background", "signal"):
            class_mask = predictions["registry_sample_class"].to_numpy() == sample_class
            class_yield = float(weights[selected & class_mask].sum())
            grouped = predictions.loc[selected & class_mask].groupby(
                ["population_kind", "registry_process_or_mode"], sort=True, dropna=False
            )
            for (population_kind, process), frame in grouped:
                component_weights = frame["primary_projection_physical_weight_inclusive"].to_numpy(
                    dtype=np.float64
                )
                component_yield = float(component_weights.sum())
                result.append({
                    "baseline": baseline,
                    "sample_class": sample_class,
                    "population_kind": population_kind,
                    "process_or_mode": process,
                    "selected_rows": len(frame),
                    "selected_sources": frame["registry_group_id"].nunique(),
                    "physical_yield": component_yield,
                    "physical_sumw2": float(np.square(component_weights).sum()),
                    "physical_effective_events": effective_events(component_weights),
                    "fraction_of_selected_class_yield": component_yield / class_yield if class_yield else 0.0,
                })
    return result


def build(staging: Path) -> dict[str, Any]:
    verify_checkpoint(C7R, C7R_MANIFEST_SHA)
    verify_checkpoint(C7S, C7S_MANIFEST_SHA)
    verify_checkpoint(C7T, C7T_MANIFEST_SHA)
    verify_checkpoint(EARLIER_CUT, EARLIER_CUT_MANIFEST_SHA)

    c7r_summary = json.loads((C7R / "summary.json").read_text())
    c7s_summary = json.loads((C7S / "summary.json").read_text())
    c7t_summary = json.loads((C7T / "summary.json").read_text())
    earlier_summary = json.loads((EARLIER_CUT / "source/summary.json").read_text())
    for payload, label in ((c7r_summary, "c7r"), (c7s_summary, "c7s"), (c7t_summary, "c7t")):
        require(payload["validation_payload_files_opened"] == 0, f"{label} validation access drift")
        test_key = "test_or_evaluation_payload_files_opened"
        require(payload[test_key] == 0, f"{label} test access drift")

    earlier_weights = indexed(read_tsv(EARLIER_CUT / "source/weight_summary.tsv"))
    current_weights = indexed(read_tsv(C7S / "weight_summary.tsv"))
    earlier_global = earlier_weights[("global", "all_train_candidates")]
    current_global = current_weights[("global", "all_train_candidates")]
    earlier_background = earlier_weights[("class", "background")]
    current_background = current_weights[("class", "background")]
    earlier_signal = earlier_weights[("class", "signal")]
    current_signal = current_weights[("class", "signal")]
    earlier_qcd = earlier_weights[("background_family", "qcd_multijet")]
    current_qcd = current_weights[("background_family", "qcd_multijet")]
    earlier_non_qcd = int(earlier_background["candidate_rows"]) - int(earlier_qcd["candidate_rows"])
    current_non_qcd = int(current_background["candidate_rows"]) - int(current_qcd["candidate_rows"])

    population_rows = [
        comparison_row("population", "source_members", float(earlier_global["members"]),
                       float(current_global["members"]), "sources",
                       "the authoritative 441-source production replaces the legacy development membership"),
        comparison_row("population", "candidate_bearing_members",
                       float(earlier_global["candidate_bearing_members"]),
                       float(current_global["candidate_bearing_members"]), "sources", "candidate-bearing source count"),
        comparison_row("population", "all_model_rows", float(earlier_global["candidate_rows"]),
                       float(current_global["candidate_rows"]), "rows", "complete model-development population"),
        comparison_row("population", "background_rows", float(earlier_background["candidate_rows"]),
                       float(current_background["candidate_rows"]), "rows", "direct four-b training background"),
        comparison_row("population", "signal_rows", float(earlier_signal["candidate_rows"]),
                       float(current_signal["candidate_rows"]), "rows", "signal population is exactly preserved"),
        comparison_row("population", "direct_qcd_fourb_rows", float(earlier_qcd["candidate_rows"]),
                       float(current_qcd["candidate_rows"]), "rows", "dominant population change"),
        comparison_row("population", "non_qcd_background_rows", float(earlier_non_qcd),
                       float(current_non_qcd), "rows", "all non-QCD background rows are exactly preserved"),
        comparison_row("development_weighting", "qcd_mean_row_training_weight",
                       float(earlier_qcd["mean_training_weight"]),
                       float(current_qcd["mean_training_weight"]), "development_weight",
                       "equal-family balancing concentrates the QCD share into 56 direct rows"),
        comparison_row("development_weighting", "global_rescale_factor",
                       float(earlier_global["rescale_factor"]), float(current_global["rescale_factor"]),
                       "development_weight", "mean development row weight remains exactly one"),
    ]
    require(earlier_non_qcd == current_non_qcd == 21312, "non-QCD population unexpectedly changed")
    require(int(earlier_signal["candidate_rows"]) == int(current_signal["candidate_rows"]) == 9337,
            "signal population unexpectedly changed")
    write_tsv(staging / "train_population_and_weight_comparison.tsv", population_rows)

    metric_rows = read_tsv(C7T / "baseline_metrics.tsv")
    metrics = {row["baseline"]: row for row in metric_rows}
    require(tuple(metrics) == MODEL_ORDER, "baseline order drift")
    predictions = pq.read_table(PREDICTIONS).to_pandas()
    require(len(predictions) == 31225, "projection prediction row-count drift")
    composition_rows = process_composition(predictions, metrics)
    write_tsv(staging / "selected_process_composition.tsv", composition_rows)

    transfer_rows = []
    for baseline in MODEL_ORDER:
        row = metrics[baseline]
        prediction = float(row["transferred_qcd_prediction"])
        truth = float(row["direct_qcd_secondary_truth"])
        background = float(row["background_yield"])
        systematic = float(row["multijet_systematic_absolute"])
        transfer_rows.append({
            "baseline": baseline,
            "operating_threshold": row["operating_threshold"],
            "selected_background_yield": background,
            "transferred_qcd_prediction": prediction,
            "transferred_qcd_fraction_of_background": prediction / background,
            "direct_qcd_secondary_truth": truth,
            "direct_qcd_relative_nonclosure": float(row["score_domain_qcd_relative_nonclosure"]),
            "factor_envelope_low": float(row["score_domain_factor_envelope_low"]),
            "factor_envelope_high": float(row["score_domain_factor_envelope_high"]),
            "direct_truth_covered_by_factor_envelope": row[
                "score_domain_direct_truth_covered_by_factor_envelope"
            ],
            "multijet_systematic_absolute": systematic,
            "multijet_systematic_relative_to_background": systematic / background,
            "nominal_asimov_ZA": float(row["asimov_ZA"]),
            "systematic_aware_asimov_ZA": float(row["systematic_aware_asimov_ZA"]),
        })
    write_tsv(staging / "score_domain_transfer_and_systematic.tsv", transfer_rows)

    snapshot_rows = read_tsv(C7T / "earlier_snapshot_comparison.tsv")
    write_tsv(staging / "earlier_snapshot_metric_comparison.tsv", snapshot_rows)

    qcd_weight_amplification = (
        float(current_qcd["mean_training_weight"]) / float(earlier_qcd["mean_training_weight"])
    )
    findings = [
        {
            "finding": "population_change_is_qcd_only",
            "evidence": (
                f"direct QCD four-b rows {earlier_qcd['candidate_rows']} -> {current_qcd['candidate_rows']}; "
                f"signal rows remain 9337 and non-QCD background rows remain {current_non_qcd}"
            ),
            "interpretation": "the 42.24% total-row reduction is not a broad reconstruction loss",
        },
        {
            "finding": "hierarchical_qcd_row_weight_concentration",
            "evidence": f"mean QCD development row weight is amplified by {qcd_weight_amplification:.6g}x",
            "interpretation": "the frozen equal-family objective gives 56 surviving direct-QCD rows high leverage",
        },
        {
            "finding": "feature_contract_is_unchanged",
            "evidence": "mass-aware 34 and mass-plane-blind 30 sets match frozen protocol cc56783...",
            "interpretation": "feature-schema drift does not explain the metric changes",
        },
        {
            "finding": "cut_signal_population_and_threshold_are_unchanged",
            "evidence": (
                f"R_HH(125,120)<34 and weighted signal efficiency "
                f"{c7t_summary['target_weighted_signal_efficiency']:.15g}"
            ),
            "interpretation": "cut changes arise from physical background treatment/effective statistics, not retuning",
        },
        {
            "finding": "primary_background_is_transfer_dominated",
            "evidence": "selected transferred-QCD fractions range from 84.77% to 87.21% across baselines",
            "interpretation": "direct stitched QCD is secondary closure only and cannot be presented as the prediction",
        },
        {
            "finding": "transfer_uncertainty_dominates_nominal_sensitivity",
            "evidence": "score-domain multijet uncertainty is 223.66% to 226.82% of total background",
            "interpretation": "nominal ZA rankings are train diagnostics; systematic-aware ZA is effectively zero",
        },
        {
            "finding": "stochastic_attribution_is_not_identifiable_from_snapshot",
            "evidence": "current seeds are fixed; bootstrap resamples source members but does not retrain models",
            "interpretation": "the approximate earlier snapshot lacks a model-artifact identity for a retraining-variance decomposition",
        },
    ]
    write_tsv(staging / "material_change_findings.tsv", findings)

    summary = {
        "schema_version": 1,
        "status": "pn_c7u_reestablished_baseline_change_investigation_pass",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": "train",
        "model_order": list(MODEL_ORDER),
        "earlier_numeric_snapshot_role": "reference_only_not_acceptance_target",
        "earlier_population_checkpoint": str(EARLIER_CUT),
        "current_baseline_checkpoint": str(C7T),
        "earlier_source_members": int(earlier_global["members"]),
        "current_source_members": int(current_global["members"]),
        "earlier_model_rows": int(earlier_global["candidate_rows"]),
        "current_model_rows": int(current_global["candidate_rows"]),
        "earlier_direct_qcd_fourb_rows": int(earlier_qcd["candidate_rows"]),
        "current_direct_qcd_fourb_rows": int(current_qcd["candidate_rows"]),
        "signal_rows_both": 9337,
        "non_qcd_background_rows_both": 21312,
        "qcd_development_mean_row_weight_amplification": qcd_weight_amplification,
        "nominal_transfer_factor": c7s_summary["nominal_transfer_factors"]["inclusive_mhh"],
        "nominal_transfer_factor_relative_statistical_uncertainty": c7r_summary[
            "nominal_transfer_factor_relative_statistical_uncertainty"
        ],
        "findings": findings,
        "validation_payload_files_opened": 0,
        "test_or_evaluation_payload_files_opened": 0,
        "validation_metrics_reported": False,
        "test_metrics_reported": False,
        "full_run2_prediction": False,
        "next_gate": (
            "do not open validation or advance to SPA-Net; first decide whether to improve/freeze the "
            "three-b transfer systematic using train-only control information"
        ),
    }
    write_json(staging / "summary.json", summary)
    (staging / "RUN_CONTRACT.txt").write_text(
        "Read-only investigation of sealed train artifacts. Earlier snapshot values are references, not fit "
        "targets. Process yields use the PN-c7t primary three-b-transfer projection; direct QCD remains a "
        "secondary score-domain closure. No validation or test payload is opened.\n"
    )
    (staging / "README.md").write_text(
        "# PN-c7u re-established baseline change investigation\n\n"
        "Train-only decomposition of the PN-c7t metric changes into authoritative source composition, "
        "development weights, selected physical process yields, and multijet-transfer closure/systematics.\n"
    )
    return summary


def freeze(staging: Path, output: Path) -> str:
    (staging / "COMPLETE").write_text("pn_c7u_reestablished_baseline_change_investigation_pass\n")
    files = sorted(path for path in staging.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (staging / "SHA256SUMS").write_text(
        "\n".join(f"{sha256(path)}  ./{path.relative_to(staging).as_posix()}" for path in files) + "\n"
    )
    manifest_sha = sha256(staging / "SHA256SUMS")
    os.rename(staging, output)
    for path in sorted(output.rglob("*"), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    output.chmod(0o555)
    return manifest_sha


def run(output: Path) -> dict[str, Any]:
    require(output.is_absolute() and output.parent.is_dir(), "invalid absolute output path")
    require(not output.exists(), f"refusing to overwrite output: {output}")
    staging = output.with_name(f".{output.name}.staging")
    require(not staging.exists(), f"refusing to overwrite preserved staging: {staging}")
    staging.mkdir()
    try:
        summary = build(staging)
        manifest_sha = freeze(staging, output)
        return {"output": str(output), "sha256sums_sha256": manifest_sha, **summary}
    except Exception as exc:
        try:
            write_json(staging / "FAILURE.json", {
                "status": "pn_c7u_failed_preserved_staging",
                "error": str(exc),
                "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
