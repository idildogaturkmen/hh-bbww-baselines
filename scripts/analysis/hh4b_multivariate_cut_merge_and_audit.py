#!/usr/bin/env python3
"""Merge, audit, and select train-only nested-OOF HH→4b cut results."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import statistics
import subprocess

import pandas as pd


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
    parser.add_argument("--job-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-repository-head", required=True)
    return parser.parse_args()


def result_rank(payload: dict[str, Any]) -> tuple[Any, ...]:
    metric = payload["pooled_inner_oof_metric"]
    return (
        float(metric["background_efficiency"]),
        float(metric["signal_efficiency"])
        - float(payload["target_signal_efficiency"]),
        len(payload["cuts"]),
        payload["structure_id"],
    )


def add_physical(
    accumulator: dict[str, dict[str, float]],
    payload: dict[str, Any],
) -> None:
    for sample_class in ("signal", "background"):
        source = payload[sample_class]
        target = accumulator.setdefault(
            sample_class,
            {
                "total_rows": 0,
                "selected_rows": 0,
                "total_signed_yield": 0.0,
                "selected_signed_yield": 0.0,
                "total_sumw2": 0.0,
                "selected_sumw2": 0.0,
                "negative_weight_rows": 0,
                "selected_negative_weight_rows": 0,
            },
        )
        for key in (
            "total_rows",
            "selected_rows",
            "negative_weight_rows",
            "selected_negative_weight_rows",
        ):
            target[key] += int(source[key])
        for key in (
            "total_signed_yield",
            "selected_signed_yield",
            "total_sumw2",
            "selected_sumw2",
        ):
            target[key] += float(source[key])


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    repo = Path(config["repository"]["path"]).resolve()
    job_root = Path(args.job_root).resolve()
    output_root = Path(args.output_root).resolve()
    require(not output_root.exists(), f"output root exists: {output_root}")
    output_root.mkdir(parents=True)

    actual_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    require(actual_head == args.expected_repository_head,
            "repository head mismatch")

    result_paths = sorted(job_root.glob("*/structure_result.json"))
    require(len(result_paths) == 270, f"expected 270 results, got {len(result_paths)}")

    payloads = []
    seen_keys = set()
    for path in result_paths:
        checksum_path = path.parent / "SHA256SUMS"
        require(checksum_path.is_file(), f"missing checksums: {path.parent}")
        for line in checksum_path.read_text().splitlines():
            digest, relative = line.split(None, 1)
            relative_path = path.parent / relative.strip().removeprefix("./")
            require(relative_path.is_file(), f"missing job artifact: {relative_path}")
            require(sha256(relative_path) == digest,
                    f"checksum mismatch: {relative_path}")

        payload = json.loads(path.read_text())
        require(payload["status"] == "pass_structure_job_complete",
                f"job did not pass: {path}")
        require(payload["repository_head"] == actual_head,
                f"job head mismatch: {path}")
        require(payload["outer_fold_used_for_selection"] is False,
                f"outer leakage flag changed: {path}")
        require(payload["validation_payloads_opened"] == 0,
                f"validation opened: {path}")
        require(payload["test_payloads_opened"] == 0,
                f"test opened: {path}")
        key = (
            int(payload["outer_fold"]),
            payload["category_id"],
            payload["structure_id"],
        )
        require(key not in seen_keys, f"duplicate job key: {key}")
        seen_keys.add(key)
        payloads.append(payload)

    expected_keys = {
        (outer, category, structure["structure_id"])
        for outer in range(5)
        for category in ("exact3tag", "ge4tag")
        for structure in json.loads(
            Path(config["inputs"]["category_endpoint_amendment"]).read_text()
        )["category_program"]["structures"]
    }
    require(seen_keys == expected_keys, "job key coverage mismatch")

    selected_rows = []
    selected_payloads: dict[tuple[int, str], dict[str, Any]] = {}
    all_metrics_rows = []

    for payload in payloads:
        metric = payload["pooled_inner_oof_metric"]
        all_metrics_rows.append(
            {
                "outer_fold": payload["outer_fold"],
                "category_id": payload["category_id"],
                "structure_id": payload["structure_id"],
                "category_family_id": payload["category_family_id"],
                "cut_count": len(payload["cuts"]),
                "signal_efficiency": metric["signal_efficiency"],
                "background_efficiency": metric["background_efficiency"],
                "selected_signal_rows": metric["selected_signal_rows"],
                "selected_background_rows": metric["selected_background_rows"],
                "selected_background_neff": metric[
                    "selected_background_neff"
                ],
                "pooled_inner_oof_feasible": payload[
                    "pooled_inner_oof_feasible"
                ],
                "all_inner_support_pass": payload[
                    "all_inner_support_pass"
                ],
                "pooled_support_pass": payload["pooled_support_pass"],
                "eligible_for_family_selection": bool(
                    payload["pooled_inner_oof_feasible"]
                    and payload["all_inner_support_pass"]
                    and payload["pooled_support_pass"]
                ),
            }
        )

    for outer_fold in range(5):
        for category_id in ("exact3tag", "ge4tag"):
            candidates = [
                payload
                for payload in payloads
                if int(payload["outer_fold"]) == outer_fold
                and payload["category_id"] == category_id
                and payload["pooled_inner_oof_feasible"]
                and payload["all_inner_support_pass"]
                and payload["pooled_support_pass"]
            ]
            require(candidates,
                    f"no eligible structure for outer={outer_fold}, category={category_id}")
            candidates.sort(key=result_rank)
            chosen = candidates[0]
            selected_payloads[(outer_fold, category_id)] = chosen
            metric = chosen["pooled_inner_oof_metric"]
            selected_rows.append(
                {
                    "outer_fold": outer_fold,
                    "category_id": category_id,
                    "selected_structure_id": chosen["structure_id"],
                    "selected_category_family_id": chosen[
                        "category_family_id"
                    ],
                    "cut_count": len(chosen["cuts"]),
                    "pooled_inner_oof_signal_efficiency": metric[
                        "signal_efficiency"
                    ],
                    "pooled_inner_oof_background_efficiency": metric[
                        "background_efficiency"
                    ],
                    "pooled_selected_signal_rows": metric[
                        "selected_signal_rows"
                    ],
                    "pooled_selected_background_rows": metric[
                        "selected_background_rows"
                    ],
                    "pooled_selected_background_neff": metric[
                        "selected_background_neff"
                    ],
                    "refit_thresholds_json": json.dumps(
                        chosen["refit_thresholds"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "outer_selected_event_uid_sha256": chosen[
                        "outer_selected_event_uid_sha256"
                    ],
                }
            )

    metrics_frame = pd.DataFrame(all_metrics_rows).sort_values(
        ["outer_fold", "category_id", "background_efficiency", "structure_id"],
        kind="mergesort",
    )
    metrics_frame.to_csv(
        output_root / "all_structure_inner_oof_metrics.tsv",
        sep="\t",
        index=False,
    )
    selected_frame = pd.DataFrame(selected_rows).sort_values(
        ["outer_fold", "category_id"], kind="mergesort"
    )
    selected_frame.to_csv(
        output_root / "selected_structure_by_outer_fold_category.tsv",
        sep="\t",
        index=False,
    )

    physical_by_category: dict[str, dict[str, dict[str, float]]] = {}
    physical_combined: dict[str, dict[str, float]] = {}
    for (outer_fold, category_id), payload in selected_payloads.items():
        category_accumulator = physical_by_category.setdefault(category_id, {})
        add_physical(category_accumulator, payload["outer_physical_evaluation"])
        add_physical(physical_combined, payload["outer_physical_evaluation"])

    physical_rows = []
    for category_id, accumulator in [
        *sorted(physical_by_category.items()),
        ("combined", physical_combined),
    ]:
        for sample_class in ("signal", "background"):
            row = dict(accumulator[sample_class])
            row["category_id"] = category_id
            row["sample_class"] = sample_class
            physical_rows.append(row)
    pd.DataFrame(physical_rows).to_csv(
        output_root / "train_oof_physical_yield_summary.tsv",
        sep="\t",
        index=False,
    )

    stability_rows = []
    deployment = {}
    for category_id in ("exact3tag", "ge4tag"):
        category_selected = [
            selected_payloads[(fold, category_id)] for fold in range(5)
        ]
        counts = Counter(payload["structure_id"] for payload in category_selected)
        ordered_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        modal_structure, modal_count = ordered_counts[0]
        second_count = ordered_counts[1][1] if len(ordered_counts) > 1 else 0
        unique_modal = modal_count > second_count
        matching = [
            payload for payload in category_selected
            if payload["structure_id"] == modal_structure
        ]
        threshold_variables = sorted(matching[0]["refit_thresholds"])
        require(
            all(sorted(payload["refit_thresholds"]) == threshold_variables
                for payload in matching),
            f"threshold variable mismatch in {category_id}",
        )
        median_thresholds = {
            variable: float(statistics.median(
                float(payload["refit_thresholds"][variable])
                for payload in matching
            ))
            for variable in threshold_variables
        }
        deployment[category_id] = {
            "modal_structure_id": modal_structure,
            "modal_outer_fold_count": modal_count,
            "modal_outer_fold_frequency": modal_count / 5.0,
            "second_structure_outer_fold_count": second_count,
            "unique_modal": unique_modal,
            "matching_outer_folds": sorted(
                int(payload["outer_fold"]) for payload in matching
            ),
            "coordinatewise_median_thresholds": median_thresholds,
            "instability_modal_frequency_below_0p70": modal_count / 5.0 < 0.70,
            "instability_top_two_difference_below_0p15": (
                (modal_count - second_count) / 5.0 < 0.15
            ),
            "instability_no_unique_modal": not unique_modal,
        }
        for structure_id, count in ordered_counts:
            stability_rows.append(
                {
                    "category_id": category_id,
                    "structure_id": structure_id,
                    "outer_fold_count": count,
                    "outer_fold_frequency": count / 5.0,
                    "is_modal": structure_id == modal_structure,
                }
            )

    pd.DataFrame(stability_rows).to_csv(
        output_root / "family_stability.tsv",
        sep="\t",
        index=False,
    )
    (output_root / "deployment_rule_train_only.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "train_only_deployment_rule_prebootstrap",
                "repository_head": actual_head,
                "categories": deployment,
                "validation_payloads_opened": 0,
                "test_payloads_opened": 0,
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    summary = {
        "schema_version": 1,
        "status": "pass_full_train_nested_oof_cut_merge_prebootstrap",
        "repository_head": actual_head,
        "structure_jobs": len(payloads),
        "outer_folds": 5,
        "categories": 2,
        "structures_per_category": 27,
        "selected_structures": selected_rows,
        "physical_oof_by_category": physical_by_category,
        "physical_oof_combined": physical_combined,
        "deployment_rule": deployment,
        "bootstrap_completed": False,
        "selection_reoptimization_completed": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    (output_root / "multivariate_cut_full_train_merge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    checksum_lines = []
    for path in sorted(p for p in output_root.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS":
            continue
        checksum_lines.append(
            f"{sha256(path)}  ./{path.relative_to(output_root)}"
        )
    (output_root / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")

    print("FULL_TRAIN_NESTED_OOF_MERGE=PASS")
    print("STRUCTURE_JOBS=270/270")
    print("OUTER_FOLDS=5/5")
    print("CATEGORIES=2/2")
    print("SELECTED_STRUCTURES=10/10")
    print("OUTER_FOLD_USED_FOR_SELECTION=FALSE")
    print("BOOTSTRAP_COMPLETED=FALSE")
    print("SELECTION_REOPTIMIZATION_COMPLETED=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")
    print(f"OUTPUT_ROOT={output_root}")
    print("RESULT=HH4B_MULTIVARIATE_CUT_FULL_TRAIN_MERGE_PASS")


if __name__ == "__main__":
    main()
