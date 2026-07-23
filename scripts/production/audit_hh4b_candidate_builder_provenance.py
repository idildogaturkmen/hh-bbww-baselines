#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq


EXPECTED_LOCAL_MEMBERS = 27

SIGNATURE_TOKENS = [
    "n_selected_bjets",
    "mbb1",
    "mbb2",
    "avg_mbb",
    "delta_mbb",
    "mhh",
    "drbb1",
    "drbb2",
    "pairing",
    "j1_pt",
    "j2_pt",
    "j3_pt",
    "j4_pt",
]

PROVENANCE_PATTERN = re.compile(
    r"candidate|builder|command|script|git|commit|"
    r"config|card|jet|pt|eta|btag|pair|mbb|mhh|"
    r"selection|threshold|delphes|parquet",
    re.IGNORECASE,
)

TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".sub",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".cfg",
    ".md",
}


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def read_tsv(
    path: Path,
) -> list[dict[str, str]]:
    require(
        path.is_file(),
        f"missing TSV {path}",
    )

    with path.open(
        newline="",
        errors="replace",
    ) as handle:
        return list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )


def flatten_json(
    value: Any,
    prefix: str = "",
) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            yield from flatten_json(
                child,
                child_prefix,
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = (
                f"{prefix}[{index}]"
            )

            yield from flatten_json(
                child,
                child_prefix,
            )

    else:
        yield prefix, value


def candidate_text_files(
    roots: list[Path],
) -> Iterator[Path]:
    seen: set[Path] = set()

    for root in roots:
        if not root.exists():
            continue

        if root.is_file():
            paths = [root]
        else:
            paths = root.rglob("*")

        for path in paths:
            if not path.is_file():
                continue

            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue

            try:
                if path.stat().st_size > 5_000_000:
                    continue
            except OSError:
                continue

            resolved = path.resolve()

            if resolved in seen:
                continue

            seen.add(resolved)
            yield resolved


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--store",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
    )

    arguments = parser.parse_args()

    repo = arguments.repo.resolve()
    store = arguments.store.resolve()
    manifest_path = arguments.manifest.resolve()
    outdir = arguments.outdir.resolve()

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    manifest = read_tsv(
        manifest_path
    )

    local_rows = [
        row
        for row in manifest
        if (
            row["availability"]
            == "local_candidate_parquet"
        )
    ]

    require(
        len(local_rows) == EXPECTED_LOCAL_MEMBERS,
        (
            f"expected {EXPECTED_LOCAL_MEMBERS} local members, "
            f"found {len(local_rows)}"
        ),
    )

    campaigns = sorted({
        row["source_campaign"].strip()
        for row in local_rows
        if row.get(
            "source_campaign",
            "",
        ).strip()
    })

    roots = [
        repo / "scripts",
        repo / "configs",
    ]

    for campaign in campaigns:
        for base in (
            store / "condor_submit",
            store / "condor_return",
        ):
            candidate = base / campaign

            if candidate.exists():
                roots.append(candidate)

    code_rows: list[dict[str, Any]] = []

    for path in candidate_text_files(
        roots
    ):
        try:
            text = path.read_text(
                errors="replace"
            )
        except OSError:
            continue

        matched = [
            token
            for token in SIGNATURE_TOKENS
            if token in text
        ]

        score = len(matched)

        if score == 0:
            continue

        relevant_lines = []

        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):
            if any(
                token in line
                for token in matched
            ):
                relevant_lines.append(
                    f"{line_number}:{line.strip()}"
                )

        code_rows.append({
            "path":
                str(path),
            "signature_score":
                score,
            "matched_tokens":
                ",".join(matched),
            "relevant_lines":
                " || ".join(
                    relevant_lines[:80]
                ),
        })

    code_rows.sort(
        key=lambda row: (
            -int(row["signature_score"]),
            row["path"],
        )
    )

    receipt_rows: list[dict[str, Any]] = []
    receipts_parsed = 0
    missing_receipts = 0

    for row in local_rows:
        receipt_raw = row.get(
            "receipt_path",
            "",
        ).strip()

        if not receipt_raw:
            missing_receipts += 1
            continue

        receipt_path = Path(
            receipt_raw
        )

        if not receipt_path.is_file():
            missing_receipts += 1
            continue

        try:
            document = json.loads(
                receipt_path.read_text()
            )
        except Exception as error:
            receipt_rows.append({
                "registry_index":
                    row["registry_index"],
                "receipt_path":
                    str(receipt_path),
                "key_path":
                    "__parse_error__",
                "value":
                    repr(error),
            })
            continue

        receipts_parsed += 1

        for key_path, value in flatten_json(
            document
        ):
            value_text = str(value)

            if not (
                PROVENANCE_PATTERN.search(
                    key_path
                )
                or PROVENANCE_PATTERN.search(
                    value_text
                )
            ):
                continue

            receipt_rows.append({
                "registry_index":
                    row["registry_index"],
                "receipt_path":
                    str(receipt_path),
                "key_path":
                    key_path,
                "value":
                    value_text[:4000],
            })

    parquet_metadata_rows: list[
        dict[str, Any]
    ] = []

    for row in local_rows:
        parquet_path = Path(
            row[
                "local_candidate_parquet"
            ]
        )

        require(
            parquet_path.is_file(),
            f"missing Parquet {parquet_path}",
        )

        parquet = pq.ParquetFile(
            parquet_path
        )

        raw_metadata = (
            parquet.metadata.metadata
            or {}
        )

        if not raw_metadata:
            parquet_metadata_rows.append({
                "registry_index":
                    row["registry_index"],
                "parquet_path":
                    str(parquet_path),
                "metadata_key":
                    "__none__",
                "metadata_value":
                    "",
            })
            continue

        for key, value in raw_metadata.items():
            key_text = key.decode(
                "utf-8",
                errors="replace",
            )

            value_text = value.decode(
                "utf-8",
                errors="replace",
            )

            parquet_metadata_rows.append({
                "registry_index":
                    row["registry_index"],
                "parquet_path":
                    str(parquet_path),
                "metadata_key":
                    key_text,
                "metadata_value":
                    value_text[:8000],
            })

    outdir.mkdir(
        parents=True
    )

    code_path = (
        outdir
        / "candidate_builder_code_candidates.tsv"
    )

    with code_path.open(
        "w",
        newline="",
    ) as handle:
        columns = [
            "path",
            "signature_score",
            "matched_tokens",
            "relevant_lines",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            code_rows
        )

    receipt_path = (
        outdir
        / "candidate_receipt_provenance_hits.tsv"
    )

    with receipt_path.open(
        "w",
        newline="",
    ) as handle:
        columns = [
            "registry_index",
            "receipt_path",
            "key_path",
            "value",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            receipt_rows
        )

    metadata_path = (
        outdir
        / "candidate_parquet_metadata.tsv"
    )

    with metadata_path.open(
        "w",
        newline="",
    ) as handle:
        columns = [
            "registry_index",
            "parquet_path",
            "metadata_key",
            "metadata_value",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            parquet_metadata_rows
        )

    score_counts = Counter(
        int(row["signature_score"])
        for row in code_rows
    )

    summary = {
        "schema_version": 1,
        "status": "pass",
        "local_members": len(local_rows),
        "source_campaigns": campaigns,
        "search_roots": [
            str(path)
            for path in roots
        ],
        "candidate_code_files":
            len(code_rows),
        "signature_score_counts":
            {
                str(key): value
                for key, value in sorted(
                    score_counts.items()
                )
            },
        "maximum_signature_score":
            (
                max(
                    (
                        int(row["signature_score"])
                        for row in code_rows
                    ),
                    default=0,
                )
            ),
        "receipts_parsed":
            receipts_parsed,
        "missing_receipts":
            missing_receipts,
        "receipt_provenance_hits":
            len(receipt_rows),
        "parquet_metadata_rows":
            len(parquet_metadata_rows),
        "candidate_builder_provenance_frozen":
            False,
        "reason":
            (
                "The highest-scoring code and receipt "
                "entries require semantic review."
            ),
        "code_candidates":
            str(code_path),
        "receipt_hits":
            str(receipt_path),
        "parquet_metadata":
            str(metadata_path),
    }

    summary_path = (
        outdir
        / "candidate_builder_provenance_audit.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "HH4B_CANDIDATE_BUILDER_PROVENANCE_AUDIT_VALID"
    )
    print(
        f"LOCAL_MEMBERS={len(local_rows)}"
    )
    print(
        f"SOURCE_CAMPAIGNS={len(campaigns)}"
    )
    print(
        f"CANDIDATE_CODE_FILES={len(code_rows)}"
    )
    print(
        "MAXIMUM_SIGNATURE_SCORE="
        f"{summary['maximum_signature_score']}"
    )
    print(
        f"RECEIPTS_PARSED={receipts_parsed}"
    )
    print(
        f"MISSING_RECEIPTS={missing_receipts}"
    )
    print(
        "RECEIPT_PROVENANCE_HITS="
        f"{len(receipt_rows)}"
    )
    print(
        "CANDIDATE_BUILDER_PROVENANCE_NOT_YET_FROZEN"
    )
    print(
        f"summary_json={summary_path}"
    )
    print(
        f"code_candidates={code_path}"
    )
    print(
        f"receipt_hits={receipt_path}"
    )


if __name__ == "__main__":
    main()
