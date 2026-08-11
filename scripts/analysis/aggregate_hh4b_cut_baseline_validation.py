#!/usr/bin/env python3
"""Aggregate the one-time fixed-cut HH->4b validation source outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


CATEGORIES = ("exact3tag", "ge4tag")
SAMPLE_CLASSES = ("signal", "background")
COUNT_FIELDS = (
    "total_rows",
    "selected_rows",
    "negative_weight_rows",
    "selected_negative_weight_rows",
)
SUM_FIELDS = (
    "total_signed_yield",
    "selected_signed_yield",
    "total_sumw2",
    "selected_sumw2",
)
METRICS = (
    "signal_physical_efficiency",
    "background_physical_efficiency",
    "background_rejection",
    "signal_selected_signed_yield",
    "background_selected_signed_yield",
    "signal_selected_sumw2",
    "background_selected_sumw2",
    "signal_selected_effective_events",
    "background_selected_effective_events",
    "signal_finite_mc_relative_uncertainty",
    "background_finite_mc_relative_uncertainty",
    "signal_over_background",
    "asimov_significance_stat_only",
)


class AggregateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateError(message)


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


def effective_events(signed_yield: float, sumw2: float) -> float:
    return 0.0 if sumw2 <= 0.0 else signed_yield * signed_yield / sumw2


def relative_mc_uncertainty(signed_yield: float, sumw2: float) -> float:
    require(signed_yield != 0.0, "zero signed yield has undefined relative MC uncertainty")
    require(sumw2 >= 0.0, "negative sumw2")
    return math.sqrt(sumw2) / abs(signed_yield)


def asimov_significance(signal: float, background: float) -> float:
    require(signal >= 0.0 and background > 0.0, "invalid Asimov yields")
    value = 2.0 * ((signal + background) * math.log1p(signal / background) - signal)
    return math.sqrt(max(value, 0.0))


def combine_summaries(
    summaries: Iterable[Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    inputs = list(summaries)
    require(inputs, "cannot combine an empty summary collection")
    output: dict[str, dict[str, Any]] = {}
    for sample_class in SAMPLE_CLASSES:
        output[sample_class] = {
            field: sum(int(item[sample_class][field]) for item in inputs)
            for field in COUNT_FIELDS
        }
        output[sample_class].update(
            {
                field: math.fsum(float(item[sample_class][field]) for item in inputs)
                for field in SUM_FIELDS
            }
        )
    return output


def performance_row(scope: str, summary: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "selection_id": "fixed_nominal_deployment_cut",
        "evaluation_role": "one_time_validation_of_master_frozen_nominal_cut",
        "scope": scope,
        "outer_fold": "validation",
        "sqrt_s_tev": 13.0,
        "luminosity_fb_inverse": 138.0,
        "luminosity_pb_inverse_internal": 138000.0,
        "yield_interpretation": "Run-2 expected-yield projection",
    }
    for sample_class in SAMPLE_CLASSES:
        values = summary[sample_class]
        for field in (*COUNT_FIELDS, *SUM_FIELDS):
            row[f"{sample_class}_{field}"] = values[field]
        total_yield = float(values["total_signed_yield"])
        selected_yield = float(values["selected_signed_yield"])
        total_rows = int(values["total_rows"])
        selected_rows = int(values["selected_rows"])
        require(total_yield > 0.0 and total_rows > 0, f"empty {sample_class} validation denominator")
        row[f"{sample_class}_physical_efficiency"] = selected_yield / total_yield
        row[f"{sample_class}_raw_efficiency"] = selected_rows / total_rows
        row[f"{sample_class}_total_effective_events"] = effective_events(
            total_yield, float(values["total_sumw2"])
        )
        row[f"{sample_class}_selected_effective_events"] = effective_events(
            selected_yield, float(values["selected_sumw2"])
        )
        row[f"{sample_class}_finite_mc_relative_uncertainty"] = relative_mc_uncertainty(
            selected_yield, float(values["selected_sumw2"])
        )
    background_efficiency = float(row["background_physical_efficiency"])
    signal = float(row["signal_selected_signed_yield"])
    background = float(row["background_selected_signed_yield"])
    require(background_efficiency > 0.0 and signal >= 0.0 and background > 0.0, "invalid validation metric yields")
    row["background_rejection"] = 1.0 / background_efficiency
    row["signal_over_background"] = signal / background
    row["asimov_significance_stat_only"] = asimov_significance(signal, background)
    row["systematics_included"] = False
    row["official_cms_result"] = False
    row["validation_payloads_opened"] = 1
    row["test_payloads_opened"] = 0
    return row


def source_summary_for_table(payload: Mapping[str, Any], category: str) -> dict[str, Any]:
    values = payload["categories"][category]
    return {
        "source_uid": payload["source_uid"],
        "production_row_index": payload["production_row_index"],
        "group_id": payload["group_id"],
        "sample_class": payload["sample_class"],
        "process_or_mode": payload["process_or_mode"],
        "scope": category,
        **{field: values[field] for field in (*COUNT_FIELDS, *SUM_FIELDS)},
        "validation_payloads_opened": 1,
        "test_payloads_opened": 0,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(rows, f"cannot write empty TSV: {path}")
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-access-manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--validation-return-checkpoint", type=Path, required=True)
    parser.add_argument("--train-performance-root", type=Path, required=True)
    parser.add_argument("--validation-worker", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    return_checkpoint = args.validation_return_checkpoint.resolve()
    require(return_checkpoint.is_dir() and not return_checkpoint.is_symlink(), "validation return checkpoint is invalid")
    sums = return_checkpoint / "SHA256SUMS"
    require(sums.is_file() and not sums.is_symlink(), "validation return checkpoint lacks SHA256SUMS")
    sums_check = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"], cwd=return_checkpoint,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    require(sums_check.returncode == 0, f"validation return checkpoint checksum failure: {sums_check.stdout}{sums_check.stderr}")
    frozen_return_root = return_checkpoint / "evidence/validation_returns"
    args.validation_return_audit = frozen_return_root / "validation_return_audit.json"
    args.source_output_root = frozen_return_root / "source_outputs"

    for path in [
        args.source_access_manifest,
        args.authorization,
        args.validation_return_audit,
        args.validation_worker,
        args.train_performance_root / "tables/train_performance_metrics.tsv",
        args.train_performance_root / "train_performance_summary.json",
    ]:
        require(path.is_file() and not path.is_symlink(), f"missing/nonregular input: {path}")
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    require(authorization.get("status") == "authorized_one_time_cut_baseline_validation", "authorization status changed")
    require(clean(authorization.get("repository_head")), "authorization repository head is missing")
    require(
        len(args.expected_head) == 40
        and all(character in "0123456789abcdef" for character in args.expected_head),
        "execution repository head is invalid",
    )
    require(authorization.get("test_payloads_opened") == 0, "test was opened")
    require(
        authorization["authorized_sha256"]["source_access_manifest"]
        == sha256(args.source_access_manifest),
        "authorized source access manifest changed",
    )
    require(
        authorization["authorized_sha256"]["validation_source_worker"]
        == sha256(args.validation_worker),
        "authorized validation worker changed",
    )
    require(
        authorization["authorized_sha256"]["validation_aggregator"]
        == sha256(Path(__file__).resolve()),
        "authorized validation aggregator changed",
    )
    return_freeze = json.loads(
        (return_checkpoint / "artifact_checkpoint_freeze.json").read_text(encoding="utf-8")
    )
    require(
        return_freeze.get("status") == "pass_validation_returns_artifact_checkpoint_frozen"
        and return_freeze.get("role") == "validation_returns",
        "validation return checkpoint freeze status changed",
    )
    require(
        return_freeze.get("implementation_sha256")
        == authorization["authorized_sha256"].get("validation_artifact_freezer"),
        "validation return checkpoint was not produced by the master-authorized freezer",
    )
    return_audit = json.loads(args.validation_return_audit.read_text(encoding="utf-8"))
    require(
        return_audit.get("status")
        == "pass_complete_one_time_cut_baseline_validation_return_audit",
        "validation complete-return audit did not pass",
    )
    require(return_audit.get("repository_head") == args.expected_head, "validation return-audit head changed")
    require(return_audit.get("clean_history_jobs") == 116, "validation history closure changed")
    require(return_audit.get("physical_validation_sources_opened_once") == 116, "validation source-open closure changed")
    require(return_audit.get("source_payload_reruns") == 0, "validation source rerun detected")
    require(return_audit.get("validation_evaluation_cycles") == 1, "validation evaluation-cycle count changed")
    require(return_audit.get("validation_payloads_opened") == 1, "validation open count changed")
    require(return_audit.get("test_payloads_opened") == 0, "test was opened in return audit")
    require(return_audit.get("nominal_thresholds") == authorization.get("nominal_thresholds"), "return-audit nominal thresholds changed")

    access = pd.read_csv(args.source_access_manifest, sep="\t", keep_default_na=False)
    physical = access.loc[access["physical_evaluation_eligible"].map(truthy)].copy()
    auxiliary = access.loc[~access["physical_evaluation_eligible"].map(truthy)].copy()
    require(len(access) == 121 and len(physical) == 116 and len(auxiliary) == 5, "validation source partition changed")
    require(auxiliary["auxiliary_qcd"].map(truthy).all(), "nonphysical source is not auxiliary QCD")

    expected_summary_paths = {
        args.source_output_root / f"source_{int(row.production_row_index):04d}_summary.json"
        for row in physical.itertuples(index=False)
    }
    expected_distribution_paths = {
        args.source_output_root / f"source_{int(row.production_row_index):04d}_distributions.tsv"
        for row in physical.itertuples(index=False)
    }
    expected_attempt_paths = {
        args.source_output_root
        / f"source_{int(row.production_row_index):04d}_VALIDATION_OPEN_DO_NOT_RERUN.json"
        for row in physical.itertuples(index=False)
    }
    require(set(args.source_output_root.glob("source_*_summary.json")) == expected_summary_paths, "validation summary file set mismatch")
    require(set(args.source_output_root.glob("source_*_distributions.tsv")) == expected_distribution_paths, "validation distribution file set mismatch")
    require(
        set(args.source_output_root.glob("source_*_VALIDATION_OPEN_DO_NOT_RERUN.json"))
        == expected_attempt_paths,
        "validation do-not-rerun marker set mismatch",
    )

    auth_sha = sha256(args.authorization)
    source_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    distributions = []
    category_class: dict[tuple[str, str], list[dict[str, Any]]] = {
        (category, sample_class): []
        for category in CATEGORIES
        for sample_class in SAMPLE_CLASSES
    }

    for row in physical.sort_values("production_row_index").itertuples(index=False):
        index = int(row.production_row_index)
        summary_path = args.source_output_root / f"source_{index:04d}_summary.json"
        distribution_path = args.source_output_root / f"source_{index:04d}_distributions.tsv"
        attempt_path = (
            args.source_output_root
            / f"source_{index:04d}_VALIDATION_OPEN_DO_NOT_RERUN.json"
        )
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        require(
            attempt.get("status")
            == "validation_source_open_attempt_durable_do_not_rerun",
            f"source attempt marker failed: {index}",
        )
        require(attempt.get("source_uid") == row.source_uid, f"attempt UID changed: {index}")
        require(attempt.get("repository_head") == args.expected_head, f"attempt head changed: {index}")
        require(
            attempt.get("authorization_repository_head")
            == authorization.get("repository_head"),
            f"attempt authorization head changed: {index}",
        )
        require(attempt.get("authorization_sha256") == auth_sha, f"attempt authorization changed: {index}")
        require(attempt.get("rerun_forbidden_even_if_downstream_bookkeeping_fails") is True, f"attempt rerun gate changed: {index}")
        require(attempt.get("test_payloads_opened") == 0, f"attempt test count changed: {index}")
        require(payload.get("status") == "pass_one_time_fixed_nominal_validation_source_evaluation", f"source status failed: {index}")
        require(payload.get("repository_head") == args.expected_head, f"source head changed: {index}")
        require(
            payload.get("authorization_repository_head")
            == authorization.get("repository_head"),
            f"source authorization head changed: {index}",
        )
        require(payload.get("source_uid") == row.source_uid, f"source UID changed: {index}")
        require(payload.get("authorization_sha256") == auth_sha, f"source authorization changed: {index}")
        require(payload.get("source_payload_opened_once") is True, f"source not opened exactly once: {index}")
        require(payload.get("source_payload_rerun_performed") is False, f"source rerun detected: {index}")
        require(payload.get("cut_scan_performed") is False, f"cut scan detected: {index}")
        require(payload.get("threshold_adjustment_performed") is False, f"threshold change detected: {index}")
        require(payload.get("family_adjustment_performed") is False, f"family change detected: {index}")
        require(
            payload.get("nominal_thresholds")
            == authorization.get("nominal_thresholds"),
            f"source nominal thresholds changed: {index}",
        )
        require(payload.get("validation_payloads_opened") == 1, f"validation count changed: {index}")
        require(payload.get("test_payloads_opened") == 0, f"test count changed: {index}")
        require(payload.get("distribution_sha256") == sha256(distribution_path), f"distribution SHA changed: {index}")
        require(payload.get("source_open_attempt_marker") == attempt_path.name, f"attempt marker name changed: {index}")
        require(payload.get("source_open_attempt_marker_sha256") == sha256(attempt_path), f"attempt marker SHA changed: {index}")

        for category in CATEGORIES:
            table_row = source_summary_for_table(payload, category)
            source_rows.append(table_row)
            category_class[(category, row.sample_class)].append(table_row)
        frame = pd.read_csv(distribution_path, sep="\t", keep_default_na=False)
        require(len(frame) == 1360, f"distribution row count changed: {index}")
        require(set(frame["source_uid"].astype(str)) == {row.source_uid}, f"distribution UID changed: {index}")
        require(frame["validation_payloads_opened"].eq(1).all(), f"distribution validation count changed: {index}")
        require(frame["test_payloads_opened"].eq(0).all(), f"distribution test count changed: {index}")
        distributions.append(frame)
        inventory_rows.append(
            {
                "production_row_index": index,
                "source_uid": row.source_uid,
                "sample_class": row.sample_class,
                "process_or_mode": row.process_or_mode,
                "summary_path": str(summary_path),
                "summary_sha256": sha256(summary_path),
                "distribution_path": str(distribution_path),
                "distribution_sha256": sha256(distribution_path),
                "do_not_rerun_marker_path": str(attempt_path),
                "do_not_rerun_marker_sha256": sha256(attempt_path),
                "validation_payloads_opened": 1,
                "test_payloads_opened": 0,
            }
        )

    category_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for category in CATEGORIES:
        category_summaries[category] = {}
        for sample_class in SAMPLE_CLASSES:
            items = category_class[(category, sample_class)]
            require(items, f"empty validation category/class: {category}/{sample_class}")
            category_summaries[category][sample_class] = {
                field: sum(int(item[field]) for item in items) for field in COUNT_FIELDS
            }
            category_summaries[category][sample_class].update(
                {
                    field: math.fsum(float(item[field]) for item in items)
                    for field in SUM_FIELDS
                }
            )
    combined = combine_summaries(category_summaries.values())
    performance = [
        performance_row(category, category_summaries[category]) for category in CATEGORIES
    ] + [performance_row("combined", combined)]

    all_distributions = pd.concat(distributions, ignore_index=True)
    identity = [
        "category_id",
        "sample_class",
        "selection_stage",
        "selection_id",
        "variable",
        "axis_label_latex",
        "bin_index",
        "bin_low_inclusive",
        "bin_high_exclusive",
    ]
    additive = [
        "rows",
        "signed_yield",
        "sumw2",
        "underflow_rows_distribution_total",
        "overflow_rows_distribution_total",
        "underflow_signed_yield_distribution_total",
        "overflow_signed_yield_distribution_total",
        "underflow_sumw2_distribution_total",
        "overflow_sumw2_distribution_total",
    ]
    aggregate_distributions = (
        all_distributions.groupby(identity, sort=True, observed=True)[additive]
        .sum()
        .reset_index()
    )
    require(len(aggregate_distributions) == 2720, "aggregate validation distribution row count mismatch")
    aggregate_distributions["validation_payloads_opened"] = 1
    aggregate_distributions["test_payloads_opened"] = 0

    train = pd.read_csv(
        args.train_performance_root / "tables/train_performance_metrics.tsv",
        sep="\t",
        keep_default_na=False,
    )
    train_pooled = train.loc[
        train["outer_fold"].astype(str).eq("pooled")
        & train["selection_id"].isin(["nested_outer_oof", "fixed_nominal_deployment_cut"])
    ]
    require(len(train_pooled) == 6, "train pooled comparison coverage changed")
    validation_lookup = {row["scope"]: row for row in performance}
    comparison_rows = []
    for train_row in train_pooled.sort_values(["selection_id", "scope"]).to_dict("records"):
        validation_row = validation_lookup[train_row["scope"]]
        for metric in METRICS:
            train_value = float(train_row[metric])
            validation_value = float(validation_row[metric])
            comparison_rows.append(
                {
                    "scope": train_row["scope"],
                    "train_selection_id": train_row["selection_id"],
                    "train_evaluation_role": train_row["evaluation_role"],
                    "validation_selection_id": "fixed_nominal_deployment_cut",
                    "validation_evaluation_role": "one_time_validation_of_master_frozen_nominal_cut",
                    "metric": metric,
                    "train_value": train_value,
                    "validation_value": validation_value,
                    "validation_minus_train": validation_value - train_value,
                    "validation_over_train": (
                        validation_value / train_value if train_value != 0.0 else "undefined"
                    ),
                    "like_for_like_fixed_cut": train_row["selection_id"]
                    == "fixed_nominal_deployment_cut",
                    "validation_payloads_opened": 1,
                    "test_payloads_opened": 0,
                }
            )
    require(len(comparison_rows) == 78, "validation/train comparison row count mismatch")

    output = args.output_root.resolve()
    require(not output.exists() and output.parent.is_dir(), f"invalid existing output root: {output}")
    build = output.parent / f".{output.name}.build.{os.getpid()}"
    require(not build.exists(), f"build root exists: {build}")
    build.mkdir()
    try:
        (build / "tables").mkdir()
        (build / "figure_data").mkdir()
        (build / "manifests").mkdir()
        write_tsv(build / "tables/validation_performance_metrics.tsv", performance)
        write_tsv(build / "tables/validation_source_metrics.tsv", source_rows)
        write_tsv(build / "manifests/validation_source_output_inventory.tsv", inventory_rows)
        write_tsv(build / "figure_data/validation_vs_train_metrics.tsv", comparison_rows)
        aggregate_distributions.to_csv(
            build / "figure_data/validation_pre_post_nominal_variable_distributions.tsv",
            sep="\t",
            index=False,
            lineterminator="\n",
        )
        summary = {
            "schema_version": 1,
            "status": "pass_one_time_hh4b_cut_baseline_validation_aggregation",
            "repository_head": args.expected_head,
            "master_train_only_checkpoint_commit": authorization[
                "master_train_only_checkpoint_commit"
            ],
            "validation_authorization_sha256": auth_sha,
            "validation_complete_return_audit_sha256": sha256(
                args.validation_return_audit
            ),
            "validation_sources_opened": 116,
            "validation_source_payloads_each_opened_exactly_once": True,
            "validation_payload_reruns": 0,
            "auxiliary_qcd_validation_sources_excluded_and_unopened": 5,
            "cut_scan_performed": False,
            "threshold_adjustment_performed": False,
            "family_adjustment_performed": False,
            "nominal_selection_changed": False,
            "pooled_metrics": {
                row["scope"]: {metric: row[metric] for metric in METRICS}
                for row in performance
            },
            "validation_payloads_opened": 1,
            "test_payloads_opened": 0,
            "official_cms_result": False,
            "yield_interpretation": "Run-2 expected-yield projection",
        }
        summary_path = build / "validation_performance_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        files = sorted(path for path in build.rglob("*") if path.is_file())
        manifest = {
            "schema_version": 1,
            "status": "pass_validation_artifact_manifest",
            "repository_head": args.expected_head,
            "inputs": {
                "source_access_manifest": sha256(args.source_access_manifest),
                "authorization": auth_sha,
                "validation_complete_return_audit": sha256(
                    args.validation_return_audit
                ),
                "validation_worker": sha256(args.validation_worker),
                "train_performance_summary": sha256(
                    args.train_performance_root / "train_performance_summary.json"
                ),
                "train_performance_metrics": sha256(
                    args.train_performance_root / "tables/train_performance_metrics.tsv"
                ),
            },
            "outputs": {
                str(path.relative_to(build)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in files
            },
            "validation_payloads_opened": 1,
            "test_payloads_opened": 0,
        }
        manifest_path = build / "manifests/artifact_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checksum_files = sorted(path for path in build.rglob("*") if path.is_file())
        (build / "SHA256SUMS").write_text(
            "".join(
                f"{sha256(path)}  ./{path.relative_to(build).as_posix()}\n"
                for path in checksum_files
            ),
            encoding="utf-8",
        )
        os.replace(build, output)
    except Exception:
        # Preserve the build directory as evidence; it contains no event payloads.
        raise

    print("CUT_BASELINE_ONE_TIME_VALIDATION_AGGREGATION=PASS")
    print("VALIDATION_SOURCES_OPENED_EXACTLY_ONCE=116")
    print("AUXILIARY_QCD_VALIDATION_SOURCES_UNOPENED=5")
    print("CUT_SCAN_PERFORMED=FALSE")
    print("NOMINAL_SELECTION_CHANGED=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=1")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
