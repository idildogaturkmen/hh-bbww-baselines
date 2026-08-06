#!/usr/bin/env python3
"""Run the six committed-code canary structure jobs twice and audit them."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import subprocess
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-repository-head", required=True)
    return parser.parse_args()


def run_once(
    *,
    pass_root: Path,
    config_path: Path,
    input_root: Path,
    expected_head: str,
    sentinel_rows: list[dict],
    worker: Path,
) -> dict[str, dict]:
    pass_root.mkdir(parents=True)
    results = {}
    for row in sentinel_rows:
        command = [
            sys.executable,
            str(worker),
            "--config",
            str(config_path),
            "--input-root",
            str(input_root),
            "--output-root",
            str(pass_root),
            "--outer-fold",
            "0",
            "--category-id",
            row["category_id"],
            "--structure-id",
            row["structure_id"],
            "--expected-repository-head",
            expected_head,
        ]
        subprocess.run(command, check=True)

    for path in sorted(pass_root.glob("*/structure_result.json")):
        payload = json.loads(path.read_text())
        key = payload["category_family_id"]
        require(key not in results, f"duplicate canary result: {key}")
        results[key] = payload
    require(len(results) == 6, f"expected 6 canary results, got {len(results)}")
    return results


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    require(not output_root.exists(), f"output root exists: {output_root}")
    output_root.mkdir(parents=True)

    config = json.loads(config_path.read_text())
    amendment = json.loads(
        Path(config["inputs"]["category_endpoint_amendment"]).read_text()
    )
    sentinel_ids = amendment["canary_v3_contract"][
        "sentinel_category_family_configurations"
    ]
    registry = {
        row["category_family_id"]: row
        for row in amendment["category_program"][
            "category_family_configurations"
        ]
    }
    require(len(sentinel_ids) == 6, "sentinel count changed")
    sentinel_rows = [registry[value] for value in sentinel_ids]

    worker = Path(__file__).with_name(
        "hh4b_multivariate_cut_structure_worker.py"
    )
    first = run_once(
        pass_root=output_root / "pass1",
        config_path=config_path,
        input_root=input_root,
        expected_head=args.expected_repository_head,
        sentinel_rows=sentinel_rows,
        worker=worker,
    )
    second = run_once(
        pass_root=output_root / "pass2",
        config_path=config_path,
        input_root=input_root,
        expected_head=args.expected_repository_head,
        sentinel_rows=sentinel_rows,
        worker=worker,
    )

    require(set(first) == set(second), "canary result keys changed across rerun")
    for key in first:
        require(
            first[key]["canonical_payload_sha256"]
            == second[key]["canonical_payload_sha256"],
            f"nondeterministic canary result: {key}",
        )

    target = float(config["optimization"]["target_signal_efficiency"])
    tolerance = float(
        config["production"]["canary_only_absolute_oof_tolerance"]
    )
    category_rows = []
    for category_id in ("exact3tag", "ge4tag"):
        category_payloads = [
            payload
            for payload in first.values()
            if payload["category_id"] == category_id
        ]
        require(len(category_payloads) == 3, f"sentinel coverage changed: {category_id}")
        best = max(
            category_payloads,
            key=lambda payload: payload["pooled_inner_oof_metric"][
                "signal_efficiency"
            ],
        )
        maximum = float(
            best["pooled_inner_oof_metric"]["signal_efficiency"]
        )
        require(
            maximum >= target - tolerance,
            f"{category_id} canary misses repaired tolerance",
        )
        require(
            any(payload["all_inner_support_pass"] for payload in category_payloads),
            f"no {category_id} sentinel passes inner support",
        )
        require(
            any(payload["pooled_support_pass"] for payload in category_payloads),
            f"no {category_id} sentinel passes pooled support",
        )
        category_rows.append(
            {
                "category_id": category_id,
                "target": target,
                "canary_tolerance": tolerance,
                "maximum_pooled_inner_oof_signal_efficiency": maximum,
                "passes_repaired_canary_acceptance": True,
                "diagnostic_best_structure_id": best["structure_id"],
            }
        )

    summary = {
        "schema_version": 1,
        "status": "pass_committed_code_canary_not_physics_result",
        "repository_head": args.expected_repository_head,
        "sentinel_jobs_per_pass": 6,
        "complete_passes": 2,
        "deterministic_rerun": True,
        "category_acceptance": category_rows,
        "production_target_unchanged": target,
        "production_target_tolerance": 0.0,
        "canary_physics_result": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    (output_root / "committed_code_canary_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print("COMMITTED_CODE_CANARY=PASS")
    print("SENTINEL_CONFIGURATIONS=6/6")
    print("COMPLETE_RERUNS=2/2")
    print("DETERMINISTIC_RERUN=PASS")
    print("EXACT3TAG_REPAIRED_CANARY_ACCEPTANCE=PASS")
    print("GE4TAG_REPAIRED_CANARY_ACCEPTANCE=PASS")
    print(f"PRODUCTION_TARGET_UNCHANGED={target}")
    print("PRODUCTION_TARGET_TOLERANCE=0")
    print("CANARY_PHYSICS_RESULT=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")
    print(f"OUTPUT_ROOT={output_root}")
    print("RESULT=HH4B_MULTIVARIATE_CUT_COMMITTED_CODE_CANARY_PASS")


if __name__ == "__main__":
    main()
