#!/usr/bin/env python3
"""Freeze validation evidence for the successful HH4b ttbar retry2 canary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[2]

JOB_ID = "3655099.0"
MEMBER = "ttbar_100k_shard003"
SAMPLE = f"{MEMBER}_pythia8_delphes"
SEED = 105003

CAMPAIGN = (
    "hh4b_ttbar8_exact_regeneration_canary_retry2_20260727_v1"
)

RETURN_DIR = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/condor_return/"
    f"{CAMPAIGN}/members/{MEMBER}"
)

CANONICAL = (
    RETURN_DIR
    / "parquet"
    / f"{MEMBER}_pythia8_delphes_canonical72.parquet"
)

RECEIPT = RETURN_DIR / "receipts" / f"{MEMBER}_receipt.json"
FINAL_STATUS = (
    RETURN_DIR / "receipts" / f"{MEMBER}_final_status.txt"
)
RETURNED_CHECKSUMS = (
    RETURN_DIR / "checksums" / f"{MEMBER}_SHA256SUMS"
)

ROOT_AUDIT = RETURN_DIR / "audits" / "root_input_audit.json"
SCHEMA_AUDIT = RETURN_DIR / "audits" / "schema_audit.json"
ACCOUNTING = (
    RETURN_DIR / "audits" / "entry_candidate_accounting.json"
)
ENVIRONMENT = (
    RETURN_DIR / "logs" / f"{MEMBER}_environment.json"
)

SOURCE_ROOT = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/condor_return/"
    "hh4b_ttbar8_exact_regeneration_20260727_v1/"
    f"members/{MEMBER}/root/{MEMBER}_pythia8_delphes.root"
)

LEGACY = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/parquet/"
    f"{MEMBER}_hh4b_candidates.parquet"
)

PAYLOAD = (
    REPO
    / "outputs/agent_runs"
    / "hh4b_ttbar8_canary_retry1_prepare_20260727_v1"
    / "hh4b_ttbar8_exact_regeneration_canary_retry1_payload.tar.gz"
)

PREVALIDATION = (
    REPO
    / "outputs/agent_runs"
    / "hh4b_ttbar8_canary_retry2_prevalidation_20260728_v2"
    / "prevalidation_summary.json"
)

SUBMISSION_RECORD = (
    REPO
    / "outputs/agent_runs"
    / "hh4b_ttbar8_canary_retry2_submission_20260727_v1"
    / "submission_record.json"
)

CONTRACT_SUMMARY = (
    REPO
    / "docs/checkpoints"
    / "hh4b_ttbar8_canary_retry2_contract_20260727_v1"
    / "summary.json"
)

EXPECTED_ROOT_SHA = (
    "127c87510087d164038b7629ea9d5210008cc4cb1473aa9b069a5cd4b859629e"
)

EXPECTED_PAYLOAD_SHA = (
    "3cc059b75cb63d27b57c51dfbd6643397fcb794e8e184ce689b4d905f53fad97"
)

EXPECTED_ENV_SHA = (
    "6f47dbb493681b424342045f38cd514a56dba056e41ce2cce04790d99ce515cd"
)

EXPECTED_CANDIDATE_SHA = (
    "37fb80340351fb487b8bd142943a12c4ba73263ff6deca63d9ff8319991f5125"
)

PAYLOAD_MEMBERS = {
    "canonical_builder": (
        "payload/repo/scripts/delphes/"
        "reconstruct_hh4b_candidates_v2.py",
        "4d7eb8e3400ec50c1c254744e6f3a2b11bd033a4525c13ec22e85247ecf59c57",
    ),
    "isolated_writer": (
        "payload/repo/scripts/delphes/"
        "write_parquet_from_pickle.py",
        "abef7e4f5d1b82fe72837834b0b0794b59bb31ff16032b1b5a7f1519a6edbcfb",
    ),
    "reconstruction_policy": (
        "payload/repo/configs/production/"
        "hh4b_rich_v2_reconstruction_policy_v1.yaml",
        "4b2a951dba7ac8bd0e2e982e3447e998e07b86d63390c248600976b3f4912b68",
    ),
    "canonical_schema": (
        "payload/repo/schema/canonical72_columns.tsv",
        "e9a12517772f77e311e37b09bc27df372735fee8dd48b176a0a82ada01913421",
    ),
}


class ValidationError(RuntimeError):
    """Raised when a frozen validation condition fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValidationError(f"Expected JSON object: {path}")

    return value


def run_bash(command: str) -> str:
    result = subprocess.run(
        ["/usr/bin/bash", "-lc", command],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        raise ValidationError(
            f"Command failed ({result.returncode}): {command}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout.strip()


def payload_member_bytes(
    archive: Path,
    member_name: str,
) -> bytes:
    with tarfile.open(archive, "r:gz") as handle:
        extracted = handle.extractfile(member_name)

        if extracted is None:
            raise ValidationError(
                f"Payload member is missing: {member_name}"
            )

        return extracted.read()


def payload_member_sha(
    archive: Path,
    member_name: str,
) -> str:
    return hashlib.sha256(
        payload_member_bytes(archive, member_name)
    ).hexdigest()


def canonical_schema_rows(
    archive: Path,
) -> list[dict[str, str]]:
    raw = payload_member_bytes(
        archive,
        PAYLOAD_MEMBERS["canonical_schema"][0],
    )

    text = raw.decode("utf-8")

    return list(
        csv.DictReader(
            io.StringIO(text),
            delimiter="\t",
        )
    )


def find_event_key(columns: list[str]) -> str:
    for candidate in (
        "event",
        "event_id",
        "event_number",
        "entry",
    ):
        if candidate in columns:
            return candidate

    raise ValidationError(
        f"No recognized event-key column in: {columns}"
    )


def compare_series(
    left: pd.Series,
    right: pd.Series,
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValidationError(
            "Cannot compare columns with different row counts"
        )

    left_numeric = pd.api.types.is_numeric_dtype(left)
    right_numeric = pd.api.types.is_numeric_dtype(right)

    if left_numeric and right_numeric:
        left_values = left.to_numpy()
        right_values = right.to_numpy()

        integer_like = (
            pd.api.types.is_integer_dtype(left)
            and pd.api.types.is_integer_dtype(right)
        ) or (
            pd.api.types.is_bool_dtype(left)
            and pd.api.types.is_bool_dtype(right)
        )

        if integer_like:
            equal = left_values == right_values
            mode = "exact"
            max_abs_difference = 0.0
        else:
            equal = np.isclose(
                left_values.astype(float),
                right_values.astype(float),
                rtol=1.0e-7,
                atol=1.0e-7,
                equal_nan=True,
            )
            mode = "rtol_1e-7_atol_1e-7"

            if len(left_values):
                differences = np.abs(
                    left_values.astype(float)
                    - right_values.astype(float)
                )
                finite = differences[np.isfinite(differences)]
                max_abs_difference = (
                    float(finite.max())
                    if len(finite)
                    else 0.0
                )
            else:
                max_abs_difference = 0.0
    else:
        left_values = (
            left.astype("string")
            .fillna("<NA>")
            .to_numpy()
        )
        right_values = (
            right.astype("string")
            .fillna("<NA>")
            .to_numpy()
        )
        equal = left_values == right_values
        mode = "exact_string"
        max_abs_difference = None

    mismatch_count = int((~equal).sum())

    return {
        "comparison_mode": mode,
        "rows_compared": len(left),
        "mismatch_count": mismatch_count,
        "all_values_match": mismatch_count == 0,
        "max_abs_difference": max_abs_difference,
    }


def latex_escape(value: object) -> str:
    text = str(value)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }

    for source, replacement in replacements.items():
        text = text.replace(source, replacement)

    return text


def write_table_bundle(
    base: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValidationError(
            f"Cannot write an empty table bundle: {base}"
        )

    fields = list(rows[0])

    with base.with_suffix(".tsv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    with base.with_suffix(".md").open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "| " + " | ".join(fields) + " |\n"
        )
        handle.write(
            "| " + " | ".join("---" for _ in fields) + " |\n"
        )

        for row in rows:
            values = [
                str(row.get(field, "")).replace("|", r"\|")
                for field in fields
            ]
            handle.write(
                "| " + " | ".join(values) + " |\n"
            )

    with base.with_suffix(".tex").open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\\begin{tabular}{")
        handle.write("l" * len(fields))
        handle.write("}\n\\toprule\n")
        handle.write(
            " & ".join(latex_escape(field) for field in fields)
            + r" \\"
            + "\n\\midrule\n"
        )

        for row in rows:
            handle.write(
                " & ".join(
                    latex_escape(row.get(field, ""))
                    for field in fields
                )
                + r" \\"
                + "\n"
            )

        handle.write("\\bottomrule\n\\end{tabular}\n")


def parse_final_status(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value

    return result


def verify_required_files() -> list[Path]:
    required = [
        CANONICAL,
        RECEIPT,
        FINAL_STATUS,
        RETURNED_CHECKSUMS,
        ROOT_AUDIT,
        SCHEMA_AUDIT,
        ACCOUNTING,
        ENVIRONMENT,
        SOURCE_ROOT,
        LEGACY,
        PAYLOAD,
        PREVALIDATION,
        SUBMISSION_RECORD,
        CONTRACT_SUMMARY,
    ]

    for path in required:
        if not path.is_file():
            raise ValidationError(
                f"Required validation input is missing: {path}"
            )

        if path.stat().st_size == 0:
            raise ValidationError(
                f"Required validation input is empty: {path}"
            )

    return required


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    checkpoint.mkdir(parents=True, exist_ok=True)

    required_files = verify_required_files()

    if sha256_file(PAYLOAD) != EXPECTED_PAYLOAD_SHA:
        raise ValidationError("Frozen payload SHA256 mismatch")

    if sha256_file(SOURCE_ROOT) != EXPECTED_ROOT_SHA:
        raise ValidationError("Source ROOT SHA256 mismatch")

    if sha256_file(CANONICAL) != EXPECTED_CANDIDATE_SHA:
        raise ValidationError(
            "Canonical candidate SHA256 mismatch"
        )

    checksum_output = run_bash(
        f"cd {RETURN_DIR} && "
        f"sha256sum -c checksums/{MEMBER}_SHA256SUMS"
    )

    live = run_bash(
        f"condor_q -name lpcschedd4.fnal.gov "
        f"{JOB_ID} -af ClusterId 2>/dev/null || true"
    )

    if live:
        raise ValidationError(
            f"Completed retry2 unexpectedly remains live: {live}"
        )

    history_text = run_bash(
        f"condor_history -name lpcschedd4.fnal.gov "
        f"{JOB_ID} -limit 1 "
        "-af ClusterId ProcId JobStatus ExitCode "
        "ExitBySignal CompletionDate RemoteWallClockTime"
    )

    history_values = history_text.split()

    if len(history_values) != 7:
        raise ValidationError(
            f"Unexpected scheduler history: {history_text!r}"
        )

    expected_history = [
        "3655099",
        "0",
        "4",
        "0",
        "false",
    ]

    if history_values[:5] != expected_history:
        raise ValidationError(
            f"Scheduler completion mismatch: {history_values}"
        )

    prevalidation = load_json(PREVALIDATION)

    if (
        prevalidation.get("status")
        != "hh4b_ttbar8_exact_regeneration_canary_retry2_prevalidation_pass"
    ):
        raise ValidationError(
            "Retry2 prevalidation status is not pass"
        )

    if prevalidation.get("candidate_rows") != 38:
        raise ValidationError(
            "Retry2 prevalidation row count is not 38"
        )

    if (
        prevalidation.get("candidate_sha256")
        != EXPECTED_CANDIDATE_SHA
    ):
        raise ValidationError(
            "Retry2 prevalidation candidate SHA mismatch"
        )

    submission = load_json(SUBMISSION_RECORD)

    if (
        submission.get("status")
        != "hh4b_ttbar8_exact_regeneration_canary_retry2_submitted"
    ):
        raise ValidationError(
            "Retry2 submission record status mismatch"
        )

    if submission.get("retry2_jobs_submitted") != 1:
        raise ValidationError(
            "Retry2 submission count is not exactly one"
        )

    if submission.get("scaleout_jobs_submitted") != 0:
        raise ValidationError(
            "Scaleout was submitted before validation"
        )

    contract = load_json(CONTRACT_SUMMARY)

    if (
        contract.get("status")
        != "hh4b_ttbar8_exact_regeneration_canary_retry2_contract_frozen"
    ):
        raise ValidationError(
            "Retry2 contract checkpoint status mismatch"
        )

    schema_rows = canonical_schema_rows(PAYLOAD)
    expected_names = [
        row["column_name"]
        for row in schema_rows
    ]
    expected_types = {
        row["column_name"]: row["arrow_type"]
        for row in schema_rows
    }

    if len(expected_names) != 72:
        raise ValidationError(
            f"Canonical schema has {len(expected_names)} columns"
        )

    canonical_table = pq.read_table(CANONICAL)

    observed_types = {
        field.name: str(field.type)
        for field in canonical_table.schema
    }

    if canonical_table.column_names != expected_names:
        raise ValidationError(
            "Canonical column name/order mismatch"
        )

    if observed_types != expected_types:
        raise ValidationError(
            "Canonical Arrow logical-type mismatch"
        )

    canonical_frame = canonical_table.to_pandas()

    if len(canonical_frame) != 38:
        raise ValidationError(
            f"Canonical rows={len(canonical_frame)}, expected 38"
        )

    canonical_event_key = find_event_key(
        list(canonical_frame.columns)
    )

    if canonical_frame[canonical_event_key].duplicated().any():
        raise ValidationError(
            "Canonical event keys are not unique"
        )

    if set(canonical_frame["sample"].astype(str)) != {SAMPLE}:
        raise ValidationError(
            "Canonical sample identity mismatch"
        )

    for column in canonical_frame.select_dtypes(
        include=[np.number]
    ).columns:
        if not np.isfinite(
            canonical_frame[column].to_numpy()
        ).all():
            raise ValidationError(
                f"Nonfinite canonical values in {column}"
            )

    receipt = load_json(RECEIPT)
    status_pairs = parse_final_status(FINAL_STATUS)
    root_audit = load_json(ROOT_AUDIT)
    schema_audit = load_json(SCHEMA_AUDIT)
    accounting = load_json(ACCOUNTING)
    environment = load_json(ENVIRONMENT)

    receipt_expected = {
        "candidate_exists": True,
        "candidate_rows": 38,
        "candidate_sha256": EXPECTED_CANDIDATE_SHA,
        "environment_validated": True,
        "exit_status": 0,
        "input_root_entries": 10000,
        "input_root_sha256": EXPECTED_ROOT_SHA,
        "member": MEMBER,
        "outcome": "success",
        "payload_validated": True,
        "python_environment_sha256": EXPECTED_ENV_SHA,
        "root_validated": True,
        "schema_columns": 72,
        "schema_validated": True,
        "seed_provenance": SEED,
    }

    for key, expected in receipt_expected.items():
        if receipt.get(key) != expected:
            raise ValidationError(
                f"Receipt {key}={receipt.get(key)!r}, "
                f"expected {expected!r}"
            )

    status_expected = {
        "status": "success",
        "exit_status": "0",
        "member": MEMBER,
        "seed": str(SEED),
        "payload_validated": "true",
        "python_environment_validated": "true",
        "root_validated": "true",
        "schema_validated": "true",
        "candidate_exists": "true",
        "candidate_rows": "38",
    }

    for key, expected in status_expected.items():
        if status_pairs.get(key) != expected:
            raise ValidationError(
                f"Final status {key}="
                f"{status_pairs.get(key)!r}, "
                f"expected {expected!r}"
            )

    if root_audit.get("entries") != 10000:
        raise ValidationError(
            "ROOT audit entry count mismatch"
        )

    if root_audit.get("readable") is not True:
        raise ValidationError(
            "ROOT audit readable flag mismatch"
        )

    if not all(
        root_audit.get("required_branches", {}).values()
    ):
        raise ValidationError(
            "ROOT audit reports a missing branch"
        )

    schema_expected = {
        "arrow_types_exact": True,
        "column_count": 72,
        "column_names_and_order_exact": True,
        "parquet_readable": True,
    }

    for key, expected in schema_expected.items():
        if schema_audit.get(key) != expected:
            raise ValidationError(
                f"Schema audit {key} mismatch"
            )

    accounting_expected = {
        "candidate_rows": 38,
        "duplicate_candidate_keys": 0,
        "root_entries": 10000,
        "sample_identity": SAMPLE,
        "source_job": "3654710.0",
        "failed_retry1_job": "3654939.0",
    }

    for key, expected in accounting_expected.items():
        if accounting.get(key) != expected:
            raise ValidationError(
                f"Accounting {key} mismatch"
            )

    environment_expected = {
        "numpy": "1.26.4",
        "awkward": "2.6.4",
        "uproot": "5.3.7",
        "pandas": "2.2.2",
        "pyarrow": "15.0.0",
        "parquet_backend": "pyarrow",
    }

    for key, expected in environment_expected.items():
        if environment.get(key) != expected:
            raise ValidationError(
                f"Environment {key} mismatch"
            )

    if not str(
        environment.get("python", "")
    ).startswith("3.9."):
        raise ValidationError(
            "Retry2 Python version is not Python 3.9"
        )

    forbidden_products = [
        str(path)
        for path in RETURN_DIR.rglob("*")
        if path.is_file()
        and path.name.lower().endswith(
            (".root", ".lhe", ".lhe.gz", ".hepmc")
        )
    ]

    if forbidden_products:
        raise ValidationError(
            "Reconstruction-only retry returned generated "
            f"products: {forbidden_products}"
        )

    legacy_table = pq.read_table(LEGACY)
    legacy_frame = legacy_table.to_pandas()
    legacy_rows_raw = len(legacy_frame)

    if legacy_rows_raw != 39:
        raise ValidationError(
            f"Legacy rows={legacy_rows_raw}, expected 39"
        )

    legacy_event_key = find_event_key(
        list(legacy_frame.columns)
    )

    canonical_events = canonical_frame[
        canonical_event_key
    ].astype("int64")

    legacy_events = legacy_frame[
        legacy_event_key
    ].astype("int64")

    if canonical_events.duplicated().any():
        raise ValidationError(
            "Canonical event keys are duplicated"
        )

    if legacy_events.duplicated().any():
        raise ValidationError(
            "Legacy event keys are duplicated"
        )

    canonical_event_set = set(canonical_events)
    legacy_event_set = set(legacy_events)

    shared_event_keys = sorted(
        canonical_event_set & legacy_event_set
    )

    canonical_only_event_keys = sorted(
        canonical_event_set - legacy_event_set
    )

    legacy_only_event_keys = sorted(
        legacy_event_set - canonical_event_set
    )

    if len(canonical_event_set) != 38:
        raise ValidationError(
            "Canonical unique-event count is not 38"
        )

    if len(legacy_event_set) != 39:
        raise ValidationError(
            "Legacy unique-event count is not 39"
        )

    if len(shared_event_keys) != 3:
        raise ValidationError(
            "Observed legacy/canonical event overlap changed: "
            f"{len(shared_event_keys)}"
        )

    if len(canonical_only_event_keys) != 35:
        raise ValidationError(
            "Observed canonical-only event count changed: "
            f"{len(canonical_only_event_keys)}"
        )

    if len(legacy_only_event_keys) != 36:
        raise ValidationError(
            "Observed legacy-only event count changed: "
            f"{len(legacy_only_event_keys)}"
        )

    common_columns = [
        column
        for column in legacy_frame.columns
        if column in canonical_frame.columns
    ]

    excluded_profile_columns = {
        "sample",
        canonical_event_key,
        legacy_event_key,
    }

    profile_columns = [
        column
        for column in common_columns
        if column not in excluded_profile_columns
    ]

    if len(profile_columns) < 8:
        raise ValidationError(
            "Too few shared columns for legacy distribution "
            f"profiling: {profile_columns}"
        )

    profile_rows: list[dict[str, Any]] = []

    for column in profile_columns:
        canonical_series = canonical_frame[column]
        legacy_series = legacy_frame[column]

        canonical_numeric = (
            pd.api.types.is_numeric_dtype(canonical_series)
        )
        legacy_numeric = (
            pd.api.types.is_numeric_dtype(legacy_series)
        )

        if canonical_numeric and legacy_numeric:
            canonical_values = (
                pd.to_numeric(
                    canonical_series,
                    errors="coerce",
                )
                .to_numpy(dtype=float)
            )

            legacy_values = (
                pd.to_numeric(
                    legacy_series,
                    errors="coerce",
                )
                .to_numpy(dtype=float)
            )

            if not np.isfinite(canonical_values).all():
                raise ValidationError(
                    "Nonfinite canonical legacy-profile values "
                    f"in {column}"
                )

            if not np.isfinite(legacy_values).all():
                raise ValidationError(
                    "Nonfinite legacy profile values "
                    f"in {column}"
                )

            profile_rows.append(
                {
                    "column": column,
                    "profile_mode": "numeric_descriptive",
                    "canonical_rows": len(canonical_values),
                    "legacy_rows": len(legacy_values),
                    "canonical_mean":
                        float(np.mean(canonical_values)),
                    "legacy_mean":
                        float(np.mean(legacy_values)),
                    "canonical_std":
                        float(np.std(canonical_values)),
                    "legacy_std":
                        float(np.std(legacy_values)),
                    "canonical_min":
                        float(np.min(canonical_values)),
                    "legacy_min":
                        float(np.min(legacy_values)),
                    "canonical_median":
                        float(np.median(canonical_values)),
                    "legacy_median":
                        float(np.median(legacy_values)),
                    "canonical_max":
                        float(np.max(canonical_values)),
                    "legacy_max":
                        float(np.max(legacy_values)),
                    "canonical_unique":
                        int(canonical_series.nunique(
                            dropna=False
                        )),
                    "legacy_unique":
                        int(legacy_series.nunique(
                            dropna=False
                        )),
                }
            )
        else:
            profile_rows.append(
                {
                    "column": column,
                    "profile_mode": "categorical_descriptive",
                    "canonical_rows": len(canonical_series),
                    "legacy_rows": len(legacy_series),
                    "canonical_mean": "",
                    "legacy_mean": "",
                    "canonical_std": "",
                    "legacy_std": "",
                    "canonical_min": "",
                    "legacy_min": "",
                    "canonical_median": "",
                    "legacy_median": "",
                    "canonical_max": "",
                    "legacy_max": "",
                    "canonical_unique":
                        int(canonical_series.nunique(
                            dropna=False
                        )),
                    "legacy_unique":
                        int(legacy_series.nunique(
                            dropna=False
                        )),
                }
            )

    event_relationship_rows = [
        {
            "canonical_rows": len(canonical_frame),
            "legacy_rows": len(legacy_frame),
            "canonical_unique_event_keys":
                len(canonical_event_set),
            "legacy_unique_event_keys":
                len(legacy_event_set),
            "shared_event_keys":
                len(shared_event_keys),
            "canonical_only_event_keys":
                len(canonical_only_event_keys),
            "legacy_only_event_keys":
                len(legacy_only_event_keys),
            "rowwise_comparison_applicable": False,
            "interpretation":
                "non_event_aligned_validation_reference",
        }
    ]

    schema_comparison_rows = []

    canonical_type_map = {
        field.name: str(field.type)
        for field in canonical_table.schema
    }
    legacy_type_map = {
        field.name: str(field.type)
        for field in legacy_table.schema
    }

    for column in sorted(
        set(canonical_type_map) | set(legacy_type_map)
    ):
        schema_comparison_rows.append(
            {
                "column": column,
                "canonical_type":
                    canonical_type_map.get(column, ""),
                "legacy_type":
                    legacy_type_map.get(column, ""),
                "shared":
                    column in canonical_type_map
                    and column in legacy_type_map,
                "profiled":
                    column in profile_columns,
            }
        )

    artifact_rows = []

    for path in required_files:
        artifact_rows.append(
            {
                "role": path.name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "committed_event_product": False,
            }
        )

    protected_rows = []

    for role, (
        member_name,
        expected_sha,
    ) in PAYLOAD_MEMBERS.items():
        observed_sha = payload_member_sha(
            PAYLOAD,
            member_name,
        )

        if observed_sha != expected_sha:
            raise ValidationError(
                f"Protected payload member mismatch: {role}"
            )

        protected_rows.append(
            {
                "role": role,
                "payload_member": member_name,
                "sha256": observed_sha,
                "verified": True,
            }
        )

    scheduler_rows = [
        {
            "job_id": JOB_ID,
            "cluster_id": history_values[0],
            "process_id": history_values[1],
            "job_status": history_values[2],
            "exit_code": history_values[3],
            "exit_by_signal": history_values[4],
            "completion_date": history_values[5],
            "remote_wall_clock_seconds":
                history_values[6],
        }
    ]

    recovery_rows = [
        {
            "stage": "before_retry2",
            "canonical_ready_members": 19,
            "remaining_exact_regeneration_members": 8,
        },
        {
            "stage": "after_retry2_validation",
            "canonical_ready_members": 20,
            "remaining_exact_regeneration_members": 7,
        },
    ]

    write_table_bundle(
        checkpoint / "scheduler_completion",
        scheduler_rows,
    )
    write_table_bundle(
        checkpoint / "validation_artifact_inventory",
        artifact_rows,
    )
    write_table_bundle(
        checkpoint / "protected_artifact_audit",
        protected_rows,
    )
    write_table_bundle(
        checkpoint / "legacy_schema_comparison",
        schema_comparison_rows,
    )
    write_table_bundle(
        checkpoint / "legacy_event_relationship",
        event_relationship_rows,
    )
    write_table_bundle(
        checkpoint / "legacy_distribution_profile",
        profile_rows,
    )
    write_table_bundle(
        checkpoint / "ttbar27_recovery_progress",
        recovery_rows,
    )

    environment_record = {
        "python": environment.get("python"),
        **environment_expected,
        "portable_environment_sha256":
            EXPECTED_ENV_SHA,
    }

    (
        checkpoint / "environment.json"
    ).write_text(
        json.dumps(
            environment_record,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "timestamp_utc":
            datetime.now(timezone.utc).isoformat(),
        "status":
            "hh4b_ttbar8_exact_regeneration_canary_retry2_validation_pass",
        "job_id": JOB_ID,
        "member": MEMBER,
        "seed_provenance": SEED,
        "source_job": "3654710.0",
        "failed_retry1_job": "3654939.0",
        "retry_mode":
            "canonical72_reconstruction_only_from_returned_root",
        "scheduler_job_status": 4,
        "scheduler_exit_code": 0,
        "root_entries": 10000,
        "candidate_rows": 38,
        "candidate_sha256":
            EXPECTED_CANDIDATE_SHA,
        "canonical_columns": 72,
        "duplicate_candidate_keys": 0,
        "finite_numeric_values": True,
        "schema_names_order_exact": True,
        "schema_types_exact": True,
        "returned_checksums_passed": True,
        "environment_exact": True,
        "legacy_path": str(LEGACY),
        "legacy_sha256": sha256_file(LEGACY),
        "legacy_rows_raw": legacy_rows_raw,
        "legacy_unique_event_keys":
            len(legacy_event_set),
        "legacy_columns":
            len(legacy_frame.columns),
        "legacy_event_key": legacy_event_key,
        "canonical_event_key": canonical_event_key,
        "legacy_event_overlap_count":
            len(shared_event_keys),
        "canonical_only_event_count":
            len(canonical_only_event_keys),
        "legacy_only_event_count":
            len(legacy_only_event_keys),
        "legacy_event_keys_exact": False,
        "legacy_rowwise_comparison_applicable": False,
        "legacy_value_equality_tested": False,
        "legacy_shared_columns":
            len(common_columns),
        "legacy_columns_profiled":
            len(profile_columns),
        "legacy_profile_finite": True,
        "legacy_reference_mode":
            "schema_and_distribution_only_non_event_aligned",
        "legacy_comparison_use":
            "validation_only_nonblocking",
        "canonical_product_authoritative": True,
        "canonical_ready_members": 20,
        "remaining_exact_regeneration_members": 7,
        "retry2_jobs_submitted": 1,
        "scaleout_jobs_submitted": 0,
        "madgraph_rerun": False,
        "pythia_rerun": False,
        "delphes_rerun": False,
        "event_level_products_committed": 0,
        "sealed_test_members_opened": 0,
        "physical_normalization": False,
        "next_gate":
            "freeze_hh4b_ttbar7_exact_regeneration_scaleout_contract",
    }

    (
        checkpoint / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        checkpoint / "README.md"
    ).write_text(
        f"""# HH4b ttbar retry2 canary validation

Retry2 job `{JOB_ID}` completed successfully and reconstructed the
immutable 10,000-entry ROOT product for `{MEMBER}`.

## Canonical result

- Candidate rows: **38**
- Canonical columns: **72**
- Duplicate event keys: **0**
- Candidate SHA-256: `{EXPECTED_CANDIDATE_SHA}`
- Scheduler exit code: **0**
- Returned checksum manifest: **pass**
- Portable Python environment: **exact**

## Legacy comparison

The predeclared nonsealed legacy product was used only as a validation
reference:

`{LEGACY}`

The legacy and regenerated canonical products are not event-aligned.
The canonical product contains 38 unique event keys, while the legacy
validation file contains 39 unique event keys. They share only 3 event
keys; 35 keys are canonical-only and 36 are legacy-only.

This proves that row-by-row value equality is not an applicable
validation criterion for these products. The legacy file is retained
only as a nonblocking schema and descriptive-distribution reference.
Shared columns are profiled independently, and finite values are
required, but no event-level equality claim is made.

The canonical-72 product reconstructed from the immutable 10,000-entry
ROOT is authoritative. The legacy 15-column product remains excluded
from the expanded canonical-v2 dataset.

## Recovery state

This canary raises the ttbar recovery state from 19 to 20 canonical-ready
members. Seven members still require exact full-chain regeneration.

No MadGraph, Pythia, or Delphes rerun occurred in retry2. No scaleout
job was submitted. No event-level product is committed in this
checkpoint. No sealed test member was opened. Physical normalization
remains out of scope.

Next gate:
`freeze_hh4b_ttbar7_exact_regeneration_scaleout_contract`
""",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )
    print()
    print(
        "HH4B_TTBAR8_CANARY_RETRY2_VALIDATION_PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
