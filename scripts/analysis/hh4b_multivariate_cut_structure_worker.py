#!/usr/bin/env python3
"""Run one outer-fold/category/structure nested-OOF cut job."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd

from hh4b_multivariate_cut_optimizer import (
    Configuration,
    OptimizationSettings,
    canonical_sha256,
    configuration_registry,
    load_amended_configurations,
    metric_to_dict,
    optimize_configuration,
    physical_yield_summary,
    selection_mask,
    structure_registry,
    support_gate,
    threshold_json,
    weighted_metric,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument(
        "--category-id", choices=["exact3tag", "ge4tag"], required=True
    )
    parser.add_argument("--structure-id", required=True)
    parser.add_argument("--expected-repository-head", required=True)
    parser.add_argument(
        "--execution-provenance",
        default=None,
        help=(
            "Optional frozen transfer-execution provenance JSON. "
            "When supplied, repository identity is validated from this "
            "manifest instead of requiring a Git checkout in the sandbox."
        ),
    )
    return parser.parse_args()


def load_table(input_root: Path, fold: int, category_id: str) -> pd.DataFrame:
    path = input_root / f"fold_{fold}" / f"{category_id}.parquet"
    require(path.is_file(), f"missing fold/category table: {path}")
    frame = pd.read_parquet(path)
    require(len(frame) > 0, f"empty fold/category table: {path}")
    require(
        pd.to_numeric(frame["source_fold"], errors="raise")
        .astype(int)
        .eq(fold)
        .all(),
        f"fold contamination in {path}",
    )
    tagged = pd.to_numeric(
        frame["candidate_tagged_jet_count"], errors="raise"
    ).astype(int)
    if category_id == "exact3tag":
        require((tagged == 3).all(), f"category contamination in {path}")
    else:
        require((tagged >= 4).all(), f"category contamination in {path}")
    return frame


def selected_uid_sha256(frame: pd.DataFrame, mask: np.ndarray) -> str:
    values = sorted(frame.loc[mask, "event_uid"].astype(str).tolist())
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    require(args.outer_fold in range(5), "outer fold must be 0..4")

    config_path = Path(args.config).resolve()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    config = json.loads(config_path.read_text())

    worker_path = Path(__file__).resolve()
    optimizer_path = worker_path.with_name(
        "hh4b_multivariate_cut_optimizer.py"
    )
    require(
        optimizer_path.is_file(),
        f"missing optimizer beside worker: {optimizer_path}",
    )

    execution_worker_sha256 = sha256(worker_path)
    execution_optimizer_sha256 = sha256(optimizer_path)

    execution_provenance_mode = "git_checkout"
    execution_provenance_sha256 = None

    if args.execution_provenance is not None:
        provenance_path = Path(args.execution_provenance).resolve()
        require(
            provenance_path.is_file(),
            f"missing execution provenance: {provenance_path}",
        )

        provenance = json.loads(provenance_path.read_text())

        require(
            provenance.get("schema_version") == 1,
            "execution provenance schema mismatch",
        )
        require(
            str(provenance.get("status", "")).startswith(
                "transfer_package_"
            ),
            "execution provenance status mismatch",
        )
        require(
            provenance.get("repository_branch")
            == config["repository"]["branch"],
            "execution provenance branch mismatch",
        )
        require(
            int(provenance.get("full_generated_event_accounting", -1))
            == 5_200_000,
            "execution provenance generated-event accounting mismatch",
        )
        require(
            int(provenance.get("training_generated_events", -1))
            == 3_799_873,
            "execution provenance training-event accounting mismatch",
        )
        require(
            int(provenance.get("fold_table_count", -1)) == 10,
            "execution provenance fold-table count mismatch",
        )
        require(
            provenance.get("production_search_budget")
            == "63_128_16_8",
            "execution provenance search-budget mismatch",
        )
        require(
            float(provenance.get("target_signal_efficiency", -1.0))
            == float(
                config["optimization"]["target_signal_efficiency"]
            ),
            "execution provenance target-efficiency mismatch",
        )
        require(
            int(provenance.get("validation_payloads_opened", -1)) == 0,
            "execution provenance indicates validation access",
        )
        require(
            int(provenance.get("test_payloads_opened", -1)) == 0,
            "execution provenance indicates test access",
        )
        require(
            provenance.get("worker_sha256")
            == execution_worker_sha256,
            "execution provenance worker hash mismatch",
        )
        require(
            provenance.get("optimizer_sha256")
            == execution_optimizer_sha256,
            "execution provenance optimizer hash mismatch",
        )

        actual_head = str(provenance.get("repository_head", ""))
        require(actual_head, "execution provenance lacks repository head")
        require(
            actual_head == args.expected_repository_head,
            "execution provenance repository-head mismatch",
        )

        execution_provenance_mode = "transferred_manifest"
        execution_provenance_sha256 = sha256(provenance_path)
    else:
        repo = Path(config["repository"]["path"]).resolve()

        actual_head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()

        require(
            actual_head == args.expected_repository_head,
            "repository head mismatch",
        )

    settings = OptimizationSettings(**config["optimization"])
    amendment_path = Path(config["inputs"]["category_endpoint_amendment"])
    configurations = load_amended_configurations(amendment_path)
    registry = structure_registry(configurations)
    key = (args.category_id, args.structure_id)
    require(key in registry, f"unknown category/structure: {key}")
    configuration: Configuration = registry[key]

    job_id = (
        f"outer{args.outer_fold}__{args.category_id}__"
        f"{args.structure_id.replace('::', '__').replace('/', '_')}"
    )
    job_root = output_root / job_id
    require(not job_root.exists(), f"job output exists: {job_root}")
    job_root.mkdir(parents=True)

    started = time.time()
    frames = {
        fold: load_table(input_root, fold, args.category_id)
        for fold in range(5)
    }
    all_uids = pd.concat(
        [frame["event_uid"].astype(str) for frame in frames.values()],
        ignore_index=True,
    )
    require(all_uids.is_unique, "event UID overlap across prepared folds")

    development_folds = sorted(set(range(5)) - {args.outer_fold})
    weight_column = f"comparison_weight_outer_fold_{args.outer_fold}"
    physical_weight_column = config["weights"]["physical_column"]

    for fold, frame in frames.items():
        weights = pd.to_numeric(frame[weight_column], errors="coerce")
        if fold == args.outer_fold:
            require(weights.isna().all(), "outer comparison sentinel changed")
        else:
            values = weights.to_numpy(dtype=float)
            require(np.isfinite(values).all(), "nonfinite development weight")
            require((values >= 0.0).all(), "negative development weight")
            frame[weight_column] = values

    inner_rows: list[dict[str, Any]] = []
    pooled_frames: list[pd.DataFrame] = []
    pooled_masks: list[np.ndarray] = []

    for inner_heldout in development_folds:
        inner_training_folds = [
            fold for fold in development_folds if fold != inner_heldout
        ]
        inner_training = pd.concat(
            [frames[fold] for fold in inner_training_folds],
            ignore_index=True,
        )
        inner_heldout_frame = frames[inner_heldout].copy()

        optimization = optimize_configuration(
            inner_training,
            configuration,
            weight_column,
            settings,
        )
        heldout_mask = selection_mask(
            inner_heldout_frame,
            configuration,
            optimization.thresholds,
        )
        heldout_metric = weighted_metric(
            inner_heldout_frame,
            heldout_mask,
            weight_column,
            settings.target_signal_efficiency,
        )
        heldout_support_pass, heldout_support_failures = support_gate(
            heldout_metric,
            settings,
            pooled=False,
        )
        pooled_frames.append(inner_heldout_frame)
        pooled_masks.append(heldout_mask)
        inner_rows.append(
            {
                "outer_fold": args.outer_fold,
                "inner_heldout_fold": inner_heldout,
                "inner_training_folds": ",".join(
                    str(fold) for fold in inner_training_folds
                ),
                "category_id": args.category_id,
                "structure_id": args.structure_id,
                "category_family_id": configuration.category_family_id,
                "thresholds_json": threshold_json(
                    optimization.thresholds
                ),
                "training_metric_json": json.dumps(
                    metric_to_dict(optimization.metric),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "heldout_metric_json": json.dumps(
                    metric_to_dict(heldout_metric),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "heldout_support_pass": heldout_support_pass,
                "heldout_support_failures": ",".join(
                    heldout_support_failures
                ),
                "optimizer_audit_json": json.dumps(
                    optimization.audit,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "selected_event_uid_sha256": selected_uid_sha256(
                    inner_heldout_frame, heldout_mask
                ),
            }
        )

    pooled_frame = pd.concat(pooled_frames, ignore_index=True)
    pooled_mask = np.concatenate(pooled_masks)
    require(
        pooled_frame["event_uid"].astype(str).is_unique,
        "pooled inner OOF event UIDs are not unique",
    )
    pooled_metric = weighted_metric(
        pooled_frame,
        pooled_mask,
        weight_column,
        settings.target_signal_efficiency,
    )
    pooled_support_pass, pooled_support_failures = support_gate(
        pooled_metric,
        settings,
        pooled=True,
    )
    all_inner_support_pass = all(row["heldout_support_pass"] for row in inner_rows)

    full_development = pd.concat(
        [frames[fold] for fold in development_folds],
        ignore_index=True,
    )
    refit = optimize_configuration(
        full_development,
        configuration,
        weight_column,
        settings,
    )
    outer_frame = frames[args.outer_fold]
    outer_mask = selection_mask(
        outer_frame,
        configuration,
        refit.thresholds,
    )
    outer_physical = physical_yield_summary(
        outer_frame,
        outer_mask,
        physical_weight_column,
    )

    result = {
        "schema_version": 1,
        "status": "pass_structure_job_complete",
        "job_id": job_id,
        "repository_head": actual_head,
        "execution_provenance_mode": execution_provenance_mode,
        "execution_provenance_sha256": execution_provenance_sha256,
        "execution_worker_sha256": execution_worker_sha256,
        "execution_optimizer_sha256": execution_optimizer_sha256,
        "input_root": str(input_root),
        "outer_fold": args.outer_fold,
        "development_folds": development_folds,
        "category_id": args.category_id,
        "structure_id": args.structure_id,
        "category_family_id": configuration.category_family_id,
        "cuts": [
            {"variable": cut.variable, "operator": cut.operator}
            for cut in configuration.cuts
        ],
        "target_signal_efficiency": settings.target_signal_efficiency,
        "pooled_inner_oof_metric": metric_to_dict(pooled_metric),
        "pooled_inner_oof_feasible": pooled_metric.feasible,
        "all_inner_support_pass": all_inner_support_pass,
        "pooled_support_pass": pooled_support_pass,
        "pooled_support_failures": pooled_support_failures,
        "refit_thresholds": refit.thresholds,
        "refit_development_metric": metric_to_dict(refit.metric),
        "refit_optimizer_audit": refit.audit,
        "outer_physical_evaluation": outer_physical,
        "outer_selected_event_uid_sha256": selected_uid_sha256(
            outer_frame, outer_mask
        ),
        "outer_fold_used_for_selection": False,
        "comparison_weight_column": weight_column,
        "physical_weight_column": physical_weight_column,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    result["canonical_payload_sha256"] = canonical_sha256(result)
    result["runtime_seconds"] = time.time() - started

    pd.DataFrame(inner_rows).to_csv(
        job_root / "inner_crossfit.tsv", sep="\t", index=False
    )
    (job_root / "structure_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    checksum_lines = []
    for path in sorted(p for p in job_root.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS":
            continue
        checksum_lines.append(
            f"{sha256(path)}  ./{path.relative_to(job_root)}"
        )
    (job_root / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")

    print("STRUCTURE_JOB=PASS")
    print(f"JOB_ID={job_id}")
    print(f"OUTER_FOLD={args.outer_fold}")
    print(f"CATEGORY_ID={args.category_id}")
    print(f"STRUCTURE_ID={args.structure_id}")
    print(
        "POOLED_INNER_OOF_SIGNAL_EFFICIENCY="
        f"{pooled_metric.signal_efficiency:.12g}"
    )
    print(
        "POOLED_INNER_OOF_BACKGROUND_EFFICIENCY="
        f"{pooled_metric.background_efficiency:.12g}"
    )
    print(f"POOLED_INNER_OOF_FEASIBLE={str(pooled_metric.feasible).upper()}")
    print(f"ALL_INNER_SUPPORT_PASS={str(all_inner_support_pass).upper()}")
    print(f"POOLED_SUPPORT_PASS={str(pooled_support_pass).upper()}")
    print("OUTER_FOLD_USED_FOR_SELECTION=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")
    print(f"OUTPUT_ROOT={job_root}")
    print("RESULT=HH4B_MULTIVARIATE_CUT_STRUCTURE_JOB_PASS")


if __name__ == "__main__":
    main()
