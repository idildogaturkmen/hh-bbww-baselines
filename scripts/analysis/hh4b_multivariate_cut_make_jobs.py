#!/usr/bin/env python3
"""Generate the exact 270-job manifest for the full nested-OOF cut scan."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd

from hh4b_multivariate_cut_optimizer import load_amended_configurations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--jobs-tsv", required=True)
    parser.add_argument("--queue-file", required=True)
    parser.add_argument("--expected-repository-head", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    configurations = load_amended_configurations(
        config["inputs"]["category_endpoint_amendment"]
    )

    rows = []
    for outer_fold in range(5):
        for configuration in sorted(
            configurations,
            key=lambda row: (row.category_id, row.structure_id),
        ):
            rows.append(
                {
                    "job_index": len(rows),
                    "outer_fold": outer_fold,
                    "category_id": configuration.category_id,
                    "structure_id": configuration.structure_id,
                    "category_family_id": configuration.category_family_id,
                    "config": str(Path(args.config).resolve()),
                    "input_root": str(Path(args.input_root).resolve()),
                    "output_root": str(Path(args.output_root).resolve()),
                    "expected_repository_head": args.expected_repository_head,
                }
            )

    if len(rows) != 270:
        raise RuntimeError(f"expected 270 jobs, got {len(rows)}")
    frame = pd.DataFrame(rows)
    jobs_path = Path(args.jobs_tsv).resolve()
    queue_path = Path(args.queue_file).resolve()
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(jobs_path, sep="\t", index=False)

    with queue_path.open("w") as handle:
        for row in rows:
            fields = [
                row["job_index"],
                row["outer_fold"],
                row["category_id"],
                row["structure_id"],
                row["config"],
                row["input_root"],
                row["output_root"],
                row["expected_repository_head"],
            ]
            handle.write("\t".join(str(value) for value in fields) + "\n")

    print("JOB_MANIFEST=PASS")
    print("EXPECTED_JOBS=270")
    print(f"ACTUAL_JOBS={len(rows)}")
    print("OUTER_FOLDS=5")
    print("CATEGORIES=2")
    print("STRUCTURES_PER_CATEGORY=27")
    print(f"JOBS_TSV={jobs_path}")
    print(f"QUEUE_FILE={queue_path}")
    print("RESULT=HH4B_MULTIVARIATE_CUT_JOB_MANIFEST_PASS")


if __name__ == "__main__":
    main()
