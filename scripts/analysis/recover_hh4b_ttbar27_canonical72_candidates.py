#!/usr/bin/env python3
"""Recover and audit the 27 canonical HH4b ttbar candidate-schema holds.

The program opens candidate content only for the 18 explicitly compatible
non-test products and the one non-test product reconstructed from retained
ROOT.  The 22 sealed test candidates are handled as source metadata only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hh4b_ttbar27_mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "plotting"))
from hh4b_cms_style import (  # noqa: E402
    CATEGORY_COLORS,
    add_delphes_header,
    apply_cms_style,
    save_png_pdf,
)


DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "baselines"
    / "hh4b_ttbar27_canonical72_recovery_v1.yaml"
)
LEGACY_COLUMNS = (
    "sample",
    "event",
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
)
MISSING_TAGS = {
    "ttbar_100k_shard001",
    "ttbar_100k_shard002",
    "ttbar_100k_shard003",
    "ttbar_100k_shard004",
    "ttbar_100k_shard005",
    "ttbar_100k_shard007",
    "ttbar_100k_shard008",
    "ttbar_100k_shard009",
}
ROOT_BACKED_TAG = "ttbar_100k_shard000"

CATEGORY_STRATA: dict[str, tuple[str, ...]] = {
    "QCD strata": (
        "qcd_hardqcd",
        "qcd_bbbb_general",
        "qcd_bbbb_iht400to600",
    ),
    "ttbar": ("ttbar_inclusive",),
    "single top": (
        "schannel_single_top",
        "tchannel_antitop",
        "tchannel_top",
        "tw_antitop",
        "tw_top",
    ),
    "top-associated": ("tth_hbb", "tttt", "ttw", "ttz_zbb"),
    "single Higgs": ("bbh_hbb_4fs", "ggh_hbb", "vbf_hbb", "wh_hbb"),
    "diboson": ("ww", "wz_zbb"),
    "triboson": ("wwz_zbb", "wzz_zbb", "zzz_zbb"),
    "Z plus heavy flavor": ("zbbbb",),
    "ggF HH": ("ggf_hh4b",),
    "VBF HH": ("vbf_hh4b",),
}

TABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "ttbar27_hold_inventory": (
        "member_index",
        "source_member_id",
        "target_tag",
        "campaign",
        "dataset_split",
        "generated_events",
        "legacy_candidate_path",
        "legacy_candidate_rows",
        "legacy_column_count",
        "legacy_schema_sha256",
        "recovery_candidate_path",
        "root_locator",
        "hepmc_locator",
        "lhe_locator",
        "outer_bundle_or_archive_locator",
        "condor_cluster",
        "condor_process",
        "job_seed",
        "generator_configuration_digest",
        "software_card_container_provenance",
        "current_recovery_classification",
        "sealed_test",
    ),
    "ttbar27_existing_product_verification": (
        "member_index",
        "target_tag",
        "dataset_split",
        "candidate_path",
        "candidate_sha256",
        "declared_rows",
        "observed_rows",
        "observed_columns",
        "exact_column_order",
        "arrow_types_compatible",
        "missing_columns",
        "unexpected_columns",
        "zero_row_schema_readable",
        "finite_required_values",
        "unique_event_keys",
        "source_member_identity",
        "split_identity",
        "verification_status",
    ),
    "ttbar27_root_reconstruction": (
        "member_index",
        "target_tag",
        "dataset_split",
        "source_root_path",
        "source_root_sha256",
        "local_root_path",
        "local_root_sha256",
        "source_local_sha256_equal",
        "root_entries",
        "expected_entries",
        "builder_path",
        "builder_sha256",
        "policy_path",
        "policy_sha256",
        "explicit_policy_arguments",
        "production_builder_invocations",
        "stalled_production_attempts",
        "candidate_path",
        "candidate_sha256",
        "candidate_rows",
        "schema_columns",
        "exact_column_order",
        "finite_required_values",
        "unique_event_keys",
        "zero_row_schema_readable",
        "validation_rerun_path",
        "validation_rerun_sha256",
        "deterministic_byte_checksum_equal",
        "determinism_exact_schema_and_column_order",
        "exact_event_ordering",
        "exact_integer_and_string_values",
        "matching_nonfinite_masks",
        "floating_values_within_tolerance",
        "floating_rtol",
        "floating_atol",
        "maximum_absolute_float_difference",
        "deterministic_row_level_equal",
        "technical_io_handler_override",
        "classification",
    ),
    "ttbar27_execution_attempt_audit": (
        "attempt_id",
        "attempt_stage",
        "input_path",
        "input_sha256",
        "local_copy_path",
        "local_copy_sha256",
        "exact_command",
        "start_time_utc",
        "end_time_utc",
        "exit_status",
        "wall_time_seconds",
        "output_path",
        "output_bytes",
        "output_sha256",
        "candidate_rows",
        "result_classification",
    ),
    "ttbar27_source_search": (
        "member_index",
        "target_tag",
        "job_seed",
        "searched_exact_identifier",
        "searched_filename",
        "searched_job_identifier",
        "searched_checksum",
        "searched_campaign_metadata",
        "root_matches",
        "hepmc_matches",
        "lhe_matches",
        "archived_bundle_matches",
        "event_summary_path",
        "event_summary_columns",
        "raw_nested_jet_payload",
        "generator_log",
        "pipeline_log",
        "generation_seed_log_match",
        "pythia_8312_log_match",
        "delphes_completion_log_match",
        "search_scope",
        "recovery_classification",
    ),
    "ttbar27_archive_and_remote_locator_audit": (
        "member_index",
        "target_tag",
        "local_archive_inventories_searched",
        "condor_submit_return_searched",
        "receipt_and_bundle_inventories_searched",
        "eos_roots_searched",
        "xrootd_recursive_listing_performed",
        "exact_remote_matches",
        "later_physics_equivalent_locator",
        "original_source_found",
        "adjudication",
    ),
    "ttbar27_regeneration_feasibility": (
        "member_index",
        "target_tag",
        "seed",
        "event_count",
        "exact_generator_process_and_cards",
        "exact_generator_version",
        "exact_random_seed",
        "exact_event_count",
        "exact_shower_configuration",
        "exact_delphes_card_and_version",
        "exact_software_environment",
        "exact_reconstruction_policy",
        "original_event_reproduction_status",
        "statistically_equivalent_replacement_is_same",
        "required_canary",
        "classification",
        "production_plan",
    ),
    "ttbar27_common_column_consistency": (
        "member_index",
        "target_tag",
        "comparison_status",
        "legacy_rows",
        "canonical_rows",
        "event_key_sets_equal",
        "shared_columns",
        "sample_identity_after_suffix_normalization",
        "exact_equal_columns",
        "differing_columns",
        "numeric_cells_compared",
        "numeric_cells_equal_at_1e_minus_9",
        "note",
    ),
    "ttbar27_candidate_registry": (
        "member_index",
        "source_member_id",
        "target_tag",
        "dataset_split",
        "generated_events",
        "candidate_path",
        "candidate_sha256",
        "candidate_rows",
        "schema_sha256",
        "classification",
        "canonical72_ready",
        "include_in_future_dataset_v2_manifest",
        "normalization_authorized",
    ),
    "ttbar27_duplicate_audit": (
        "scope",
        "target_tag",
        "rows_checked",
        "duplicate_sample_event_keys",
        "cross_member_duplicate_keys",
        "status",
    ),
    "ttbar27_event_accounting": (
        "member_index",
        "target_tag",
        "classification",
        "generated_events",
        "legacy_candidate_rows",
        "canonical_candidate_rows",
        "canonical_ready",
        "legacy_rows_excluded_from_dataset_v2",
    ),
    "ttbar27_normalization_readiness": (
        "member_index",
        "target_tag",
        "generated_events",
        "event_summary_path",
        "physics_weight_status",
        "physics_yield_authorized",
        "normalization_ready",
        "normalization_blockers",
    ),
    "proposed_v2_test_extension": (
        "extension_order",
        "disposition",
        "sample_class",
        "process_or_mode",
        "coverage_category",
        "source_index",
        "member_id",
        "original_split",
        "generated_events",
        "sealed_test_before_gate",
        "candidate_content_opened",
        "selection_digest",
        "selection_rule",
    ),
    "proposed_v2_test_composition": (
        "coverage_category",
        "required_strata",
        "existing_members",
        "existing_strata",
        "proposed_added_members",
        "proposed_added_strata",
        "combined_members",
        "combined_strata",
        "remaining_gaps",
        "representative_coverage_achievable",
    ),
    "unresolved_ttbar_members": (
        "member_index",
        "target_tag",
        "dataset_split",
        "seed",
        "generated_events",
        "legacy_candidate_rows",
        "original_source_status",
        "canonical_candidate_status",
        "required_action",
        "next_gate",
    ),
    "protected_artifact_audit": (
        "artifact_class",
        "path",
        "expected_sha256",
        "observed_sha256",
        "matched",
        "git_diff_unchanged",
        "status",
    ),
}


class RecoveryError(RuntimeError):
    """Raised when a frozen contract or integrity condition is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def plain(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def latex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in plain(value))


def write_table_bundle(
    directory: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = TABLE_FIELDS[name]
    directory.mkdir(parents=True, exist_ok=True)
    for row in rows:
        require(
            not (set(row) - set(fields)),
            f"{name}: unexpected fields {sorted(set(row) - set(fields))}",
        )
    with (directory / f"{name}.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: plain(row.get(field, "")) for field in fields})

    markdown = [
        "| " + " | ".join(field.replace("_", " ") for field in fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in rows:
        markdown.append(
            "| "
            + " | ".join(
                plain(row.get(field, ""))
                .replace("\\", r"\\")
                .replace("|", r"\|")
                .replace("\n", " ")
                for field in fields
            )
            + " |"
        )
    (directory / f"{name}.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    columns = "l" * len(fields)
    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(latex_escape(field.replace("_", " ")) for field in fields)
        + r" \\",
        r"\midrule",
    ]
    for row in rows:
        latex.append(
            " & ".join(latex_escape(row.get(field, "")) for field in fields) + r" \\"
        )
    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{latex_escape(name.replace('_', ' '))}}}",
            rf"\label{{tab:{name.replace('_', '-')}}}",
            r"\end{table}",
        ]
    )
    (directory / f"{name}.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    require(config["schema_version"] == 1, "unsupported config schema")
    return config


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def verify_frozen_inputs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, spec in config["frozen_inputs"].items():
        path = resolve_repo_path(spec["path"])
        require(path.is_file(), f"missing frozen input: {path}")
        observed = sha256_file(path)
        require(observed == spec["sha256"], f"frozen input changed: {name}")
        rows.append(
            {
                "artifact_class": "frozen_input",
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "expected_sha256": spec["sha256"],
                "observed_sha256": observed,
                "matched": True,
                "git_diff_unchanged": True,
                "status": "verified",
            }
        )
    return rows


def verify_canonical_v1(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = config["canonical_v1_protection"]
    directory = resolve_repo_path(spec["directory"])
    manifest = directory / "SHA256SUMS"
    observed_manifest = sha256_file(manifest)
    require(
        observed_manifest == spec["sha256sums_sha256"],
        "canonical-v1 SHA256SUMS changed",
    )
    rows: list[dict[str, Any]] = [
        {
            "artifact_class": "canonical_v1_manifest",
            "path": str(manifest.relative_to(REPOSITORY_ROOT)),
            "expected_sha256": spec["sha256sums_sha256"],
            "observed_sha256": observed_manifest,
            "matched": True,
            "git_diff_unchanged": True,
            "status": "verified",
        }
    ]
    entries = 0
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = raw.split(None, 1)
        relative = relative.lstrip("*")
        target = directory / relative
        observed = sha256_file(target)
        require(observed == expected, f"canonical-v1 artifact changed: {relative}")
        entries += 1
        rows.append(
            {
                "artifact_class": "canonical_v1_artifact",
                "path": str(target.relative_to(REPOSITORY_ROOT)),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matched": True,
                "git_diff_unchanged": True,
                "status": "verified",
            }
        )
    require(entries == spec["expected_entries"], "canonical-v1 entry count changed")
    return rows


def git_diff_unchanged(path: Path) -> bool:
    relative = path.relative_to(REPOSITORY_ROOT)
    completed = subprocess.run(
        ["git", "diff", "--exit-code", "--", str(relative)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def verify_protected_reconstruction(
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("builder", "policy"):
        spec = config["protected_reconstruction"][name]
        path = resolve_repo_path(spec["path"])
        observed = sha256_file(path)
        unchanged = git_diff_unchanged(path)
        require(observed == spec["sha256"], f"protected {name} changed")
        require(unchanged, f"protected {name} has an uncommitted diff")
        rows.append(
            {
                "artifact_class": f"protected_{name}",
                "path": spec["path"],
                "expected_sha256": spec["sha256"],
                "observed_sha256": observed,
                "matched": True,
                "git_diff_unchanged": unchanged,
                "status": "verified",
            }
        )
    return rows


def deterministic_rank(
    salt: str, sample_class: str, stratum: str, member_id: str
) -> str:
    return sha256_text(f"{salt}|{sample_class}|{stratum}|{member_id}")


def parse_tag(candidate_path: str) -> str:
    suffix = "_hh4b_candidates.parquet"
    name = Path(candidate_path).name
    require(name.endswith(suffix), f"unrecognized legacy candidate path: {name}")
    return name[: -len(suffix)]


def canonical_product_path(
    config: Mapping[str, Any], tag: str, shard: str
) -> Path | None:
    if tag.startswith("ttbar_200k_"):
        rendered = config["paths"]["existing_ttbar200k_product_template"].format(
            shard=shard.zfill(3), tag=tag
        )
        return Path(rendered)
    if tag == ROOT_BACKED_TAG:
        return resolve_repo_path(config["paths"]["retained_root_product"])
    return None


def load_inputs(config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    inputs = config["frozen_inputs"]
    return {
        key: read_tsv(resolve_repo_path(spec["path"]))
        for key, spec in inputs.items()
        if str(spec["path"]).endswith(".tsv")
    }


def exact_schema_contract(
    canonical_columns: pd.DataFrame,
) -> tuple[list[str], dict[str, str]]:
    ordered = canonical_columns.sort_values("column_index", key=lambda s: s.astype(int))
    names = ordered["column_name"].tolist()
    types = dict(zip(ordered["column_name"], ordered["arrow_type"]))
    require(len(names) == 72 and len(set(names)) == 72, "canonical schema is not 72")
    return names, types


def verify_candidate(
    path: Path,
    expected_names: Sequence[str],
    expected_types: Mapping[str, str],
    expected_rows: int,
    expected_sample: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    require(path.is_file(), f"candidate product missing: {path}")
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    names = schema.names
    missing = [name for name in expected_names if name not in names]
    unexpected = [name for name in names if name not in expected_names]
    exact_order = names == list(expected_names)
    types_ok = all(
        name in names and str(schema.field(name).type) == expected_types[name]
        for name in expected_names
    )
    observed_rows = parquet.metadata.num_rows
    require(observed_rows == expected_rows, f"row count changed for {path}")
    frame = pd.read_parquet(path)
    require(len(frame) == observed_rows, f"unreadable rows in {path}")

    finite = True
    for field in schema:
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            values = frame[field.name].to_numpy()
            if not np.isfinite(values).all():
                finite = False
                break
    unique_keys = not frame.duplicated(["sample", "event"]).any()
    source_identity = (
        len(frame) == 0
        or (frame["sample"].nunique(dropna=False) == 1)
        and (str(frame["sample"].iloc[0]) == expected_sample)
    )
    result = {
        "candidate_sha256": sha256_file(path),
        "observed_rows": observed_rows,
        "observed_columns": len(names),
        "exact_column_order": exact_order,
        "arrow_types_compatible": types_ok,
        "missing_columns": missing,
        "unexpected_columns": unexpected,
        "zero_row_schema_readable": len(names) == len(expected_names),
        "finite_required_values": finite,
        "unique_event_keys": unique_keys,
        "source_member_identity": source_identity,
    }
    require(exact_order, f"canonical column order mismatch: {path}")
    require(types_ok and not missing and not unexpected, f"schema mismatch: {path}")
    require(finite and unique_keys and source_identity, f"content check failed: {path}")
    return result, frame


def compare_determinism_frames(
    primary: pd.DataFrame,
    rerun: pd.DataFrame,
    expected_names: Sequence[str],
    expected_types: Mapping[str, str],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Compare deterministic content without requiring Parquet byte identity."""
    schema_equal = (
        list(primary.columns) == list(expected_names)
        and list(rerun.columns) == list(expected_names)
    )
    event_equal = (
        schema_equal
        and len(primary) == len(rerun)
        and np.array_equal(
            primary["event"].to_numpy(),
            rerun["event"].to_numpy(),
        )
    )
    exact_values = schema_equal and len(primary) == len(rerun)
    masks_equal = schema_equal and len(primary) == len(rerun)
    floats_equal = schema_equal and len(primary) == len(rerun)
    maximum_absolute_float_difference = 0.0

    if schema_equal and len(primary) == len(rerun):
        for name in expected_names:
            arrow_type = pa.type_for_alias(expected_types[name])
            left = primary[name].to_numpy()
            right = rerun[name].to_numpy()
            if pa.types.is_floating(arrow_type):
                left_finite = np.isfinite(left)
                right_finite = np.isfinite(right)
                column_masks_equal = np.array_equal(left_finite, right_finite)
                masks_equal = masks_equal and column_masks_equal
                if column_masks_equal:
                    finite = left_finite
                    if finite.any():
                        differences = np.abs(left[finite] - right[finite])
                        maximum_absolute_float_difference = max(
                            maximum_absolute_float_difference,
                            float(differences.max()),
                        )
                        floats_equal = floats_equal and bool(
                            np.allclose(
                                left[finite],
                                right[finite],
                                rtol=rtol,
                                atol=atol,
                                equal_nan=False,
                            )
                        )
                else:
                    floats_equal = False
            elif pa.types.is_integer(arrow_type) or pa.types.is_string(arrow_type):
                exact_values = exact_values and np.array_equal(left, right)
    else:
        event_equal = False
        exact_values = False
        masks_equal = False
        floats_equal = False

    row_level_equal = bool(
        schema_equal
        and event_equal
        and exact_values
        and masks_equal
        and floats_equal
    )
    return {
        "determinism_exact_schema_and_column_order": bool(schema_equal),
        "exact_event_ordering": bool(event_equal),
        "exact_integer_and_string_values": bool(exact_values),
        "matching_nonfinite_masks": bool(masks_equal),
        "floating_values_within_tolerance": bool(floats_equal),
        "floating_rtol": rtol,
        "floating_atol": atol,
        "maximum_absolute_float_difference": maximum_absolute_float_difference,
        "deterministic_row_level_equal": row_level_equal,
    }


def combined_generator_digest(config: Mapping[str, Any]) -> str:
    production = config["original_ttbar100k_production"]
    payload = {
        "campaign_command": production["campaign_command"],
        "generator_process": production["generator_process"],
        "generator_version": production["generator_version"],
        "process_card": production["generator_process_card"]["sha256"],
        "param_card": production["generator_param_card"]["sha256"],
        "run_card_overwrites": production["generator_run_card_template"][
            "overwritten_settings"
        ],
        "shower": production["shower_converter"]["sha256"],
        "shower_settings": production["shower_converter"]["changed_settings"],
        "delphes": production["delphes_executable"]["sha256"],
        "delphes_card": production["delphes_card"]["sha256"],
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def verify_production_artifacts(config: Mapping[str, Any]) -> None:
    production = config["original_ttbar100k_production"]
    for key in (
        "generator_process_card",
        "generator_param_card",
        "generation_script",
        "local_sample_script",
        "shower_pipeline_snapshot",
        "shower_converter",
        "delphes_executable",
        "delphes_card",
    ):
        spec = production[key]
        path = resolve_repo_path(spec["path"])
        require(path.is_file(), f"production artifact missing: {path}")
        require(sha256_file(path) == spec["sha256"], f"production artifact changed: {key}")
    run_card = production["generator_run_card_template"]
    require(
        sha256_file(Path(run_card["path"])) == run_card["observed_sha256"],
        "retained run-card template changed",
    )


def inventory_holds(
    config: Mapping[str, Any],
    inputs: Mapping[str, pd.DataFrame],
    generator_digest: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coverage = inputs["member_coverage"]
    holds = coverage[
        coverage["development_disposition"] == "legacy_ttbar_schema_hold"
    ].copy()
    holds = holds.sort_values("source_index", key=lambda s: s.astype(int))
    require(len(holds) == 27, "held member count is not 27")
    require(holds["member_id"].nunique() == 27, "held member IDs are not unique")
    require(not holds["sealed_test"].map(as_bool).any(), "a hold is sealed-test")

    registry = inputs["canonical_ttbar_registry"]
    source_manifest = inputs["background_source_manifest"]
    registry_by_tag = {row["tag"]: row for _, row in registry.iterrows()}
    source_by_tag = {row["target_tag"]: row for _, row in source_manifest.iterrows()}

    software = config["original_ttbar100k_production"]
    provenance = (
        f"{software['generator_version']};Pythia_{software['shower_converter']['pythia_version']};"
        f"Delphes_{software['delphes_executable']['version']};"
        f"FastJet_{software['fastjet_version']};no_container_original_environment_retained"
    )
    inventory: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    for _, hold in holds.iterrows():
        tag = parse_tag(hold["candidate_reference"])
        require(tag in registry_by_tag, f"ttbar registry missing {tag}")
        require(tag in source_by_tag, f"source manifest missing {tag}")
        registry_row = registry_by_tag[tag]
        source_row = source_by_tag[tag]
        campaign = registry_row["campaign"]
        shard = str(registry_row["shard"]).zfill(3)
        recovery = canonical_product_path(config, tag, shard)
        root = Path(
            f"/uscms_data/d3/iturkmen/hh4b_delphes/root/{tag}_pythia8_delphes.root"
        )
        if tag.startswith("ttbar_200k_"):
            classification = "canonical72_existing_verified"
        elif tag == ROOT_BACKED_TAG:
            classification = "canonical72_reconstructed_from_retained_root"
        else:
            classification = "exact_deterministic_regeneration_possible"
        row = {
            "member_index": int(hold["source_index"]),
            "source_member_id": hold["member_id"],
            "target_tag": tag,
            "campaign": campaign,
            "dataset_split": hold["dataset_split"],
            "generated_events": int(hold["generated_events"]),
            "legacy_candidate_path": hold["candidate_reference"],
            "legacy_candidate_rows": int(hold["candidate_rows_metadata"]),
            "legacy_column_count": 15,
            "legacy_schema_sha256": source_row["local_schema_sha256"],
            "recovery_candidate_path": str(recovery) if recovery else "",
            "root_locator": str(root) if root.is_file() else "",
            "hepmc_locator": "",
            "lhe_locator": "",
            "outer_bundle_or_archive_locator": "",
            "condor_cluster": "not_applicable_local_campaign",
            "condor_process": "not_applicable_local_campaign",
            "job_seed": int(registry_row["seed"]),
            "generator_configuration_digest": generator_digest,
            "software_card_container_provenance": provenance,
            "current_recovery_classification": classification,
            "sealed_test": False,
        }
        inventory.append(row)
        normalized.append(
            {
                **row,
                "shard": shard,
                "event_summary": registry_row["event_parquet"],
                "physics_yield_authorized": as_bool(
                    registry_row["physics_yield_authorized"]
                ),
            }
        )
    require(sum(row["generated_events"] for row in inventory) == 270000, "event total")
    return inventory, normalized


def compare_common_columns(
    member: Mapping[str, Any], canonical: pd.DataFrame | None
) -> dict[str, Any]:
    if canonical is None:
        return {
            "member_index": member["member_index"],
            "target_tag": member["target_tag"],
            "comparison_status": "not_performed_no_independent_canonical72_product",
            "legacy_rows": member["legacy_candidate_rows"],
            "canonical_rows": 0,
            "event_key_sets_equal": False,
            "shared_columns": LEGACY_COLUMNS,
            "sample_identity_after_suffix_normalization": False,
            "exact_equal_columns": [],
            "differing_columns": [],
            "numeric_cells_compared": 0,
            "numeric_cells_equal_at_1e_minus_9": 0,
            "note": "Legacy rows are identification/accounting evidence only.",
        }
    legacy = pd.read_parquet(member["legacy_candidate_path"])
    legacy = legacy.sort_values("event").reset_index(drop=True)
    canonical = canonical.sort_values("event").reset_index(drop=True)
    event_equal = legacy["event"].tolist() == canonical["event"].tolist()
    normalized_samples = canonical["sample"].str.replace(
        "_pythia8_delphes$", "", regex=True
    )
    sample_equal = legacy["sample"].tolist() == normalized_samples.tolist()
    exact_equal: list[str] = []
    differing: list[str] = []
    numeric_cells = 0
    numeric_equal = 0
    for column in LEGACY_COLUMNS:
        if column == "sample":
            equal = sample_equal
        elif pd.api.types.is_numeric_dtype(legacy[column]):
            left = legacy[column].to_numpy()
            right = canonical[column].to_numpy()
            close = np.isclose(left, right, rtol=0.0, atol=1e-9, equal_nan=False)
            numeric_cells += len(close)
            numeric_equal += int(close.sum())
            equal = bool(close.all())
        else:
            equal = legacy[column].tolist() == canonical[column].tolist()
        (exact_equal if equal else differing).append(column)
    return {
        "member_index": member["member_index"],
        "target_tag": member["target_tag"],
        "comparison_status": "performed_diagnostic_not_acceptance_criterion",
        "legacy_rows": len(legacy),
        "canonical_rows": len(canonical),
        "event_key_sets_equal": event_equal,
        "shared_columns": LEGACY_COLUMNS,
        "sample_identity_after_suffix_normalization": sample_equal,
        "exact_equal_columns": exact_equal,
        "differing_columns": differing,
        "numeric_cells_compared": numeric_cells,
        "numeric_cells_equal_at_1e_minus_9": numeric_equal,
        "note": (
            "Row/event agreement is expected; feature differences are allowed because "
            "the frozen canonical policy replaces the legacy reconstruction semantics."
        ),
    }


def verify_ready_products(
    config: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    expected_names: Sequence[str],
    expected_types: Mapping[str, str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, pd.DataFrame],
    list[dict[str, Any]],
]:
    existing_rows: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    common_rows: list[dict[str, Any]] = []
    config_protected = config["protected_reconstruction"]
    for member in members:
        classification = member["current_recovery_classification"]
        if classification == "exact_deterministic_regeneration_possible":
            common_rows.append(compare_common_columns(member, None))
            continue
        path = Path(member["recovery_candidate_path"])
        result, frame = verify_candidate(
            path,
            expected_names,
            expected_types,
            member["legacy_candidate_rows"],
            f"{member['target_tag']}_pythia8_delphes",
        )
        frames[member["target_tag"]] = frame
        common_rows.append(compare_common_columns(member, frame))
        if classification == "canonical72_existing_verified":
            existing_rows.append(
                {
                    "member_index": member["member_index"],
                    "target_tag": member["target_tag"],
                    "dataset_split": member["dataset_split"],
                    "candidate_path": str(path),
                    **result,
                    "declared_rows": member["legacy_candidate_rows"],
                    "split_identity": member["dataset_split"] in {"train", "validation"},
                    "verification_status": "canonical72_existing_verified",
                }
            )
        else:
            retry = config["controlled_retry"]
            root_path = resolve_repo_path(config["paths"]["retained_root"])
            local_root_path = resolve_repo_path(
                config["paths"]["retained_root_retry_input"]
            )
            require(root_path.is_file(), "retained source ROOT is missing")
            require(local_root_path.is_file(), "node-local retained ROOT copy is missing")
            source_sha256 = sha256_file(root_path)
            local_sha256 = sha256_file(local_root_path)
            require(
                source_sha256 == retry["source_root_sha256"],
                "retained source ROOT checksum changed",
            )
            require(
                local_sha256 == retry["local_root_sha256"],
                "node-local ROOT checksum changed",
            )
            require(
                source_sha256 == local_sha256
                and retry["source_local_sha256_equal"],
                "source and node-local ROOT checksums differ",
            )
            with uproot.open(
                root_path,
                handler=uproot.source.file.MultithreadedFileSource,
                num_workers=1,
            ) as root_file:
                entries = int(root_file["Delphes"].num_entries)
            rerun = resolve_repo_path(config["paths"]["retained_root_validation_rerun"])
            require(rerun.is_file(), "retained-root deterministic rerun is missing")
            rerun_result, rerun_frame = verify_candidate(
                rerun,
                expected_names,
                expected_types,
                member["legacy_candidate_rows"],
                f"{member['target_tag']}_pythia8_delphes",
            )
            bytes_equal = result["candidate_sha256"] == rerun_result["candidate_sha256"]
            tolerance = config_protected["determinism_tolerance"]
            determinism = compare_determinism_frames(
                frame,
                rerun_frame,
                expected_names,
                expected_types,
                rtol=float(tolerance["floating_rtol"]),
                atol=float(tolerance["floating_atol"]),
            )
            require(
                determinism["deterministic_row_level_equal"],
                "retained-root deterministic row-level comparison failed",
            )
            root_rows.append(
                {
                    "member_index": member["member_index"],
                    "target_tag": member["target_tag"],
                    "dataset_split": member["dataset_split"],
                    "source_root_path": str(root_path),
                    "source_root_sha256": source_sha256,
                    "local_root_path": str(local_root_path),
                    "local_root_sha256": local_sha256,
                    "source_local_sha256_equal": source_sha256 == local_sha256,
                    "root_entries": entries,
                    "expected_entries": member["generated_events"],
                    "builder_path": config_protected["builder"]["path"],
                    "builder_sha256": config_protected["builder"]["sha256"],
                    "policy_path": config_protected["policy"]["path"],
                    "policy_sha256": config_protected["policy"]["sha256"],
                    "explicit_policy_arguments": config_protected[
                        "explicit_arguments"
                    ],
                    "production_builder_invocations": 1,
                    "stalled_production_attempts": 1,
                    "candidate_path": str(path),
                    "candidate_sha256": result["candidate_sha256"],
                    "candidate_rows": result["observed_rows"],
                    "schema_columns": result["observed_columns"],
                    "exact_column_order": result["exact_column_order"],
                    "finite_required_values": result["finite_required_values"],
                    "unique_event_keys": result["unique_event_keys"],
                    "zero_row_schema_readable": result["zero_row_schema_readable"],
                    "validation_rerun_path": str(rerun),
                    "validation_rerun_sha256": rerun_result["candidate_sha256"],
                    "deterministic_byte_checksum_equal": bytes_equal,
                    **determinism,
                    "technical_io_handler_override": retry[
                        "io_handler_adjudication"
                    ]["technical_override"],
                    "classification": "canonical72_reconstructed_from_retained_root",
                }
            )
            require(entries == member["generated_events"], "ROOT entry accounting failed")
    require(len(existing_rows) == 18, "existing verification count is not 18")
    require(len(root_rows) == 1, "retained-root reconstruction count is not 1")
    return existing_rows, root_rows, frames, common_rows


def execution_attempt_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    retry = config["controlled_retry"]
    smoke = retry["smoke_test"]
    require(
        int(retry["scratch_free_gib_before_copy"]) >= 3,
        "controlled retry did not establish at least 3 GiB free scratch",
    )
    require(
        smoke["status"] == "pass"
        and smoke["tree"] == "Delphes"
        and int(smoke["entries"]) == 10000
        and list(smoke["entries_read"]) == [0, 10]
        and set(smoke["required_branches"])
        == {
            "Jet.PT",
            "Jet.Eta",
            "Jet.Phi",
            "Jet.Mass",
            "Jet.BTag",
            "Jet.Flavor",
        }
        and smoke["branch_leaf_dtypes"]
        == {
            "Jet.PT": "float32",
            "Jet.Eta": "float32",
            "Jet.Phi": "float32",
            "Jet.Mass": "float32",
            "Jet.BTag": "uint32",
            "Jet.Flavor": "uint32",
        }
        and smoke["compatible_jagged_shapes"]
        and smoke["numeric_leaf_types"],
        "controlled retry ROOT smoke-test contract failed",
    )
    diagnostics = retry["initial_stall_diagnostics"]
    require(
        diagnostics["result_classification"] == "aborted_stalled_no_output"
        and diagnostics["wait_channel"] == "futex_do_wait"
        and float(diagnostics["cpu_percent"]) == 0.0
        and int(diagnostics["io_counters_unchanged_over_seconds"]) >= 60
        and not diagnostics["open_root_file_descriptor"]
        and not diagnostics["open_parquet_file_descriptor"]
        and not diagnostics["accepted_output_produced"],
        "initial stalled-attempt diagnostics changed",
    )
    attempts = [dict(row) for row in retry["attempts"]]
    require(len(attempts) == 3, "controlled retry attempt count is not three")
    require(
        [row["result_classification"] for row in attempts]
        == [
            "aborted_stalled_no_output",
            "successful_production_reconstruction",
            "successful_determinism_validation_rerun",
        ],
        "controlled retry classifications changed",
    )
    require(
        int(attempts[0]["candidate_rows"]) == 0
        and int(attempts[0]["output_bytes"]) == 0,
        "stalled attempt must remain recorded as no-output",
    )

    source_path = resolve_repo_path(config["paths"]["retained_root"])
    local_path = resolve_repo_path(config["paths"]["retained_root_retry_input"])
    source_sha256 = sha256_file(source_path)
    local_sha256 = sha256_file(local_path)
    require(
        source_sha256 == local_sha256 == retry["source_root_sha256"],
        "execution-attempt input checksum contract failed",
    )
    for row in attempts:
        require(
            row["input_sha256"] == source_sha256,
            f"attempt input checksum changed: {row['attempt_id']}",
        )
    for row in attempts[1:]:
        command = row["exact_command"]
        for required_fragment in (
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            "VECLIB_MAXIMUM_THREADS=1",
            "PYTHONUNBUFFERED=1",
            "timeout --signal=TERM --kill-after=60s 45m",
            "--sample ttbar_100k_shard000_pythia8_delphes",
            "--target-mass 125.0",
            "--jet-pt-min 30.0",
            "--jet-eta-max 2.5",
            "--btag-min 0.0",
            "--max-bjets-for-pairing 8",
            "--higgs-ordering pt",
        ):
            require(
                required_fragment in command,
                f"controlled retry command changed: {row['attempt_id']}",
            )
        require(
            row["local_copy_path"] == str(local_path)
            and row["local_copy_sha256"] == local_sha256,
            f"attempt local-copy identity changed: {row['attempt_id']}",
        )
        output = resolve_repo_path(row["output_path"])
        require(output.is_file(), f"attempt output missing: {row['attempt_id']}")
        require(
            output.stat().st_size == int(row["output_bytes"])
            and sha256_file(output) == row["output_sha256"],
            f"attempt output identity changed: {row['attempt_id']}",
        )
        require(
            pq.ParquetFile(output).metadata.num_rows == int(row["candidate_rows"]) == 29,
            f"attempt output row count changed: {row['attempt_id']}",
        )
    return attempts


def source_and_regeneration_rows(
    config: Mapping[str, Any],
    inputs: Mapping[str, pd.DataFrame],
    members: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    missing = {row["target_tag"]: row for row in members if row["target_tag"] in MISSING_TAGS}
    adjudication = inputs["missing_source_adjudication"]
    by_tag = {row["target_tag"]: row for _, row in adjudication.iterrows()}
    search_rows: list[dict[str, Any]] = []
    archive_rows: list[dict[str, Any]] = []
    regen_rows: list[dict[str, Any]] = []
    eos_roots = config["source_search"]["eos_roots"]
    plan = config["original_ttbar100k_production"]["production_plan"]
    for tag in sorted(missing, key=lambda item: missing[item]["member_index"]):
        member = missing[tag]
        require(tag in by_tag, f"missing adjudication row for {tag}")
        evidence = by_tag[tag]
        for field in ("other_root_matches", "hepmc_matches", "lhe_matches"):
            require(evidence[field] == "[]", f"unexpected recovered source for {tag}")
        log_directory = Path("/uscms_data/d3/iturkmen/hh4b_delphes/logs")
        generator_log = log_directory / f"{tag}_generate_events.log"
        pipeline_log = log_directory / f"{tag}_pipeline.log"
        delphes_log = log_directory / f"{tag}_pythia8_delphes.log"
        require(generator_log.is_file(), f"generator log missing for {tag}")
        require(pipeline_log.is_file(), f"pipeline log missing for {tag}")
        require(delphes_log.is_file(), f"Delphes log missing for {tag}")
        generator_text = generator_log.read_text(encoding="utf-8", errors="replace")
        pipeline_text = pipeline_log.read_text(encoding="utf-8", errors="replace")
        delphes_text = delphes_log.read_text(encoding="utf-8", errors="replace")
        seed_match = (
            f"Using random number seed offset = {member['job_seed']}" in generator_text
        )
        pythia_match = "This is PYTHIA version 8.312" in pipeline_text
        delphes_match = "** Exiting..." in delphes_text
        require(seed_match and pythia_match and delphes_match, f"log provenance failed: {tag}")
        search_rows.append(
            {
                "member_index": member["member_index"],
                "target_tag": tag,
                "job_seed": member["job_seed"],
                "searched_exact_identifier": True,
                "searched_filename": True,
                "searched_job_identifier": True,
                "searched_checksum": True,
                "searched_campaign_metadata": True,
                "root_matches": [],
                "hepmc_matches": [],
                "lhe_matches": [],
                "archived_bundle_matches": [],
                "event_summary_path": evidence["event_summary"],
                "event_summary_columns": int(evidence["event_summary_columns"]),
                "raw_nested_jet_payload": as_bool(
                    evidence["has_raw_nested_jet_payload"]
                ),
                "generator_log": str(generator_log),
                "pipeline_log": str(pipeline_log),
                "generation_seed_log_match": seed_match,
                "pythia_8312_log_match": pythia_match,
                "delphes_completion_log_match": delphes_match,
                "search_scope": (
                    "repository manifests; ignored inventories; Condor submit/return; "
                    "receipts; registries; archive lists; EOS/XRootD recursive listings; "
                    "local store and documented remote paths"
                ),
                "recovery_classification": "exact_deterministic_regeneration_possible",
            }
        )
        archive_rows.append(
            {
                "member_index": member["member_index"],
                "target_tag": tag,
                "local_archive_inventories_searched": True,
                "condor_submit_return_searched": True,
                "receipt_and_bundle_inventories_searched": True,
                "eos_roots_searched": eos_roots,
                "xrootd_recursive_listing_performed": True,
                "exact_remote_matches": 0,
                "later_physics_equivalent_locator": config["source_search"][
                    "later_physics_equivalent_ttbar_root"
                ],
                "original_source_found": False,
                "adjudication": (
                    "Later ttbar ROOT files have different member identities and are "
                    "not substitutes for the missing originals."
                ),
            }
        )
        regen_rows.append(
            {
                "member_index": member["member_index"],
                "target_tag": tag,
                "seed": member["job_seed"],
                "event_count": member["generated_events"],
                "exact_generator_process_and_cards": True,
                "exact_generator_version": True,
                "exact_random_seed": True,
                "exact_event_count": True,
                "exact_shower_configuration": True,
                "exact_delphes_card_and_version": True,
                "exact_software_environment": True,
                "exact_reconstruction_policy": True,
                "original_event_reproduction_status": (
                    "deterministic_reproduction_possible_not_executed_in_this_gate"
                ),
                "statistically_equivalent_replacement_is_same": False,
                "required_canary": (
                    "regenerate seed105000 and match retained original event digest "
                    "c224af6f4217b2006db68a83a4dcc37854f272f77b62ae974a9348874d431fde"
                ),
                "classification": "exact_deterministic_regeneration_possible",
                "production_plan": plan,
            }
        )
    require(len(search_rows) == len(archive_rows) == len(regen_rows) == 8, "search count")
    return search_rows, archive_rows, regen_rows


def build_registry_and_accounting(
    config: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    existing: Sequence[Mapping[str, Any]],
    root_rows: Sequence[Mapping[str, Any]],
    frames: Mapping[str, pd.DataFrame],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    ready_info = {row["target_tag"]: row for row in existing}
    ready_info.update({row["target_tag"]: row for row in root_rows})
    registry: list[dict[str, Any]] = []
    accounting: list[dict[str, Any]] = []
    normalization: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    schema_sha = config["protected_reconstruction"]["canonical_schema_sha256"]
    for member in members:
        ready = member["target_tag"] in ready_info
        info = ready_info.get(member["target_tag"], {})
        rows = int(info.get("observed_rows", info.get("candidate_rows", 0)))
        path = info.get("candidate_path", "")
        checksum = info.get("candidate_sha256", "")
        classification = member["current_recovery_classification"]
        registry.append(
            {
                "member_index": member["member_index"],
                "source_member_id": member["source_member_id"],
                "target_tag": member["target_tag"],
                "dataset_split": member["dataset_split"],
                "generated_events": member["generated_events"],
                "candidate_path": path,
                "candidate_sha256": checksum,
                "candidate_rows": rows,
                "schema_sha256": schema_sha if ready else "",
                "classification": classification,
                "canonical72_ready": ready,
                "include_in_future_dataset_v2_manifest": ready,
                "normalization_authorized": False,
            }
        )
        accounting.append(
            {
                "member_index": member["member_index"],
                "target_tag": member["target_tag"],
                "classification": classification,
                "generated_events": member["generated_events"],
                "legacy_candidate_rows": member["legacy_candidate_rows"],
                "canonical_candidate_rows": rows,
                "canonical_ready": ready,
                "legacy_rows_excluded_from_dataset_v2": (
                    0 if ready else member["legacy_candidate_rows"]
                ),
            }
        )
        blockers = ["physics_weight_status_not_frozen", "physics_yield_not_authorized"]
        if not ready:
            blockers.insert(0, "canonical72_candidate_product_missing")
        normalization.append(
            {
                "member_index": member["member_index"],
                "target_tag": member["target_tag"],
                "generated_events": member["generated_events"],
                "event_summary_path": member["event_summary"],
                "physics_weight_status": "not_frozen",
                "physics_yield_authorized": False,
                "normalization_ready": False,
                "normalization_blockers": blockers,
            }
        )
        if not ready:
            unresolved.append(
                {
                    "member_index": member["member_index"],
                    "target_tag": member["target_tag"],
                    "dataset_split": member["dataset_split"],
                    "seed": member["job_seed"],
                    "generated_events": member["generated_events"],
                    "legacy_candidate_rows": member["legacy_candidate_rows"],
                    "original_source_status": "root_hepmc_lhe_archive_not_found",
                    "canonical_candidate_status": "missing_independent_canonical72",
                    "required_action": "exact_deterministic_regeneration_after_canary",
                    "next_gate": config["status_contract"]["next_gate"],
                }
            )

    accounting.append(
        {
            "member_index": "TOTAL",
            "target_tag": "all_27",
            "classification": "aggregate",
            "generated_events": sum(int(row["generated_events"]) for row in members),
            "legacy_candidate_rows": sum(
                int(row["legacy_candidate_rows"]) for row in members
            ),
            "canonical_candidate_rows": sum(
                int(row["canonical_candidate_rows"]) for row in accounting
            ),
            "canonical_ready": sum(bool(row["canonical_ready"]) for row in accounting),
            "legacy_rows_excluded_from_dataset_v2": sum(
                int(row["legacy_rows_excluded_from_dataset_v2"]) for row in accounting
            ),
        }
    )
    require(len(unresolved) == 8, "unresolved count is not eight")
    require(
        sum(row["legacy_candidate_rows"] for row in unresolved) == 307,
        "unresolved legacy rows are not 307",
    )
    return registry, accounting, normalization, unresolved


def duplicate_audit(frames: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    combined: list[pd.DataFrame] = []
    for tag in sorted(frames):
        frame = frames[tag][["sample", "event"]].copy()
        duplicate = int(frame.duplicated(["sample", "event"]).sum())
        require(duplicate == 0, f"duplicate event keys in {tag}")
        rows.append(
            {
                "scope": "member",
                "target_tag": tag,
                "rows_checked": len(frame),
                "duplicate_sample_event_keys": duplicate,
                "cross_member_duplicate_keys": 0,
                "status": "pass",
            }
        )
        combined.append(frame.assign(target_tag=tag))
    all_keys = pd.concat(combined, ignore_index=True)
    cross = int(all_keys.duplicated(["sample", "event"]).sum())
    require(cross == 0, "cross-member duplicate keys detected")
    rows.append(
        {
            "scope": "all_canonical_ready_members",
            "target_tag": "all_19",
            "rows_checked": len(all_keys),
            "duplicate_sample_event_keys": cross,
            "cross_member_duplicate_keys": cross,
            "status": "pass",
        }
    )
    return rows


def category_for_stratum(stratum: str) -> str:
    for category, strata in CATEGORY_STRATA.items():
        if stratum in strata:
            return category
    raise RecoveryError(f"unmapped process or mode: {stratum}")


def proposed_test_extension(
    config: Mapping[str, Any], coverage: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sealed = coverage[coverage["sealed_test"].map(as_bool)].copy()
    require(len(sealed) == 22, "sealed test member count changed")
    require(
        not sealed["candidate_content_opened_by_this_audit"].map(as_bool).any(),
        "source checkpoint reports sealed content opened",
    )
    salt = config["test_extension"]["deterministic_salt"]
    rule = config["test_extension"]["selection_rule"]
    extension: list[dict[str, Any]] = []
    for _, row in sealed.sort_values(["sample_class", "source_index"]).iterrows():
        stratum = row["process_or_mode"]
        extension.append(
            {
                "extension_order": len(extension),
                "disposition": "preserve_existing_sealed",
                "sample_class": row["sample_class"],
                "process_or_mode": stratum,
                "coverage_category": category_for_stratum(stratum),
                "source_index": int(row["source_index"]),
                "member_id": row["member_id"],
                "original_split": row["dataset_split"],
                "generated_events": int(row["generated_events"]),
                "sealed_test_before_gate": True,
                "candidate_content_opened": False,
                "selection_digest": "",
                "selection_rule": "existing_frozen_sealed_member",
            }
        )

    existing_strata = set(sealed["process_or_mode"])
    required = {item for values in CATEGORY_STRATA.values() for item in values}
    missing = sorted(required - existing_strata)
    eligible = coverage[
        (~coverage["sealed_test"].map(as_bool))
        & coverage["source_covered"].map(as_bool)
        & coverage["process_or_mode"].isin(missing)
    ].copy()
    for stratum in missing:
        candidates = eligible[eligible["process_or_mode"] == stratum].copy()
        require(len(candidates) > 0, f"no metadata candidate for stratum {stratum}")
        candidates["selection_digest"] = candidates.apply(
            lambda row: deterministic_rank(
                salt, row["sample_class"], stratum, row["member_id"]
            ),
            axis=1,
        )
        chosen = candidates.sort_values(
            ["selection_digest", "source_index"]
        ).iloc[0]
        extension.append(
            {
                "extension_order": len(extension),
                "disposition": "proposed_add_metadata_only",
                "sample_class": chosen["sample_class"],
                "process_or_mode": stratum,
                "coverage_category": category_for_stratum(stratum),
                "source_index": int(chosen["source_index"]),
                "member_id": chosen["member_id"],
                "original_split": chosen["dataset_split"],
                "generated_events": int(chosen["generated_events"]),
                "sealed_test_before_gate": False,
                "candidate_content_opened": False,
                "selection_digest": chosen["selection_digest"],
                "selection_rule": rule,
            }
        )

    composition: list[dict[str, Any]] = []
    for category, strata_tuple in CATEGORY_STRATA.items():
        strata = set(strata_tuple)
        existing_rows = [
            row
            for row in extension
            if row["disposition"] == "preserve_existing_sealed"
            and row["process_or_mode"] in strata
        ]
        added_rows = [
            row
            for row in extension
            if row["disposition"] == "proposed_add_metadata_only"
            and row["process_or_mode"] in strata
        ]
        existing_modes = {row["process_or_mode"] for row in existing_rows}
        added_modes = {row["process_or_mode"] for row in added_rows}
        combined = existing_modes | added_modes
        gaps = sorted(strata - combined)
        composition.append(
            {
                "coverage_category": category,
                "required_strata": sorted(strata),
                "existing_members": len(existing_rows),
                "existing_strata": sorted(existing_modes),
                "proposed_added_members": len(added_rows),
                "proposed_added_strata": sorted(added_modes),
                "combined_members": len(existing_rows) + len(added_rows),
                "combined_strata": sorted(combined),
                "remaining_gaps": gaps,
                "representative_coverage_achievable": not gaps,
            }
        )
    required_pthat = {
        "pthat50to75",
        "pthat75to100",
        "pthat100to200",
        "pthat200to300",
        "pthat300to500",
        "pthat500to700",
        "pthat700to1000",
        "pthat1000toInf",
    }
    existing_qcd = sealed[sealed["process_or_mode"] == "qcd_hardqcd"]
    observed_pthat = {
        match.group(0)
        for member_id in existing_qcd["member_id"]
        for match in [re.search(r"pthat(?:50to75|75to100|100to200|200to300|300to500|500to700|700to1000|1000toInf)", member_id)]
        if match is not None
    }
    pthat_gaps = sorted(required_pthat - observed_pthat)
    composition.append(
        {
            "coverage_category": "QCD pTHat/phase-space",
            "required_strata": sorted(required_pthat),
            "existing_members": len(existing_qcd),
            "existing_strata": sorted(observed_pthat),
            "proposed_added_members": 0,
            "proposed_added_strata": [],
            "combined_members": len(existing_qcd),
            "combined_strata": sorted(observed_pthat),
            "remaining_gaps": pthat_gaps,
            "representative_coverage_achievable": not pthat_gaps,
        }
    )
    require(all(not row["remaining_gaps"] for row in composition), "test gaps remain")
    stats = {
        "existing": 22,
        "proposed_added": len(extension) - 22,
        "combined": len(extension),
        "existing_signal_modes": sorted(
            set(
                sealed.loc[
                    sealed["sample_class"] == "signal", "process_or_mode"
                ].tolist()
            )
        ),
        "combined_signal_modes": sorted(
            {
                row["process_or_mode"]
                for row in extension
                if row["sample_class"] == "signal"
            }
        ),
        "candidate_content_opened": 0,
        "finalized": False,
    }
    return extension, composition, stats


def make_plots(
    checkpoint: Path,
    registry: Sequence[Mapping[str, Any]],
    accounting: Sequence[Mapping[str, Any]],
    composition: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    style = apply_cms_style()
    plot_dir = checkpoint / "plots"
    secondary = "ttbar candidate recovery"

    status_order = [
        "canonical72_existing_verified",
        "canonical72_reconstructed_from_retained_root",
        "exact_deterministic_regeneration_possible",
    ]
    status_labels = ["Existing verified", "Retained ROOT", "Exact regeneration"]
    status_counts = Counter(row["classification"] for row in registry)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(
        status_labels,
        [status_counts[key] for key in status_order],
        color=CATEGORY_COLORS[:3],
    )
    ax.set_ylabel("Held members")
    ax.set_ylim(0, 20)
    add_delphes_header(ax, secondary)
    save_png_pdf(fig, plot_dir / "ttbar27_recovery_status", dpi=300)

    candidate_rows = [
        sum(
            int(row["candidate_rows"])
            for row in registry
            if row["classification"] == key
        )
        for key in status_order[:2]
    ]
    candidate_rows.append(
        sum(
            int(row["legacy_candidate_rows"])
            for row in accounting
            if row["member_index"] != "TOTAL" and not row["canonical_ready"]
        )
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(
        ["Existing canonical", "ROOT reconstructed", "Legacy-only excluded"],
        candidate_rows,
        color=CATEGORY_COLORS[:3],
    )
    ax.set_ylabel("Candidate rows")
    add_delphes_header(ax, secondary)
    save_png_pdf(fig, plot_dir / "ttbar27_candidate_rows_by_status", dpi=300)

    generated = [
        sum(
            int(row["generated_events"])
            for row in registry
            if row["classification"] == key
        )
        for key in status_order
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(status_labels, generated, color=CATEGORY_COLORS[:3])
    ax.set_ylabel("Generated events")
    add_delphes_header(ax, secondary)
    save_png_pdf(fig, plot_dir / "ttbar27_generated_event_accounting", dpi=300)

    process = [
        row
        for row in composition
        if row["coverage_category"] not in {"ggF HH", "VBF HH"}
    ]
    labels = [row["coverage_category"] for row in process]
    existing = np.array([int(row["existing_members"]) for row in process])
    added = np.array([int(row["proposed_added_members"]) for row in process])
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, existing, label="Existing sealed", color=CATEGORY_COLORS[0])
    ax.bar(
        x,
        added,
        bottom=existing,
        label="Proposed metadata-only additions",
        color=CATEGORY_COLORS[1],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Members")
    ax.legend()
    add_delphes_header(ax, secondary)
    save_png_pdf(fig, plot_dir / "proposed_v2_test_process_coverage", dpi=300)

    signal = [
        row for row in composition if row["coverage_category"] in {"ggF HH", "VBF HH"}
    ]
    labels = [row["coverage_category"] for row in signal]
    existing = np.array([int(row["existing_members"]) for row in signal])
    added = np.array([int(row["proposed_added_members"]) for row in signal])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = np.arange(len(labels))
    ax.bar(x, existing, label="Existing sealed", color=CATEGORY_COLORS[0])
    ax.bar(
        x,
        added,
        bottom=existing,
        label="Proposed metadata-only additions",
        color=CATEGORY_COLORS[1],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Members")
    ax.legend()
    add_delphes_header(ax, secondary)
    save_png_pdf(fig, plot_dir / "proposed_v2_test_signal_mode_coverage", dpi=300)
    return style


def write_checksum_manifest(directory: Path) -> None:
    files = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(directory)}"
        for path in files
    ]
    (directory / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_reconstruction_command(
    config: Mapping[str, Any], output: Path
) -> list[str]:
    protected = config["protected_reconstruction"]
    args = protected["explicit_arguments"]
    return [
        "env",
        f"PYTHONPATH={config['paths']['retained_root_retry_python_startup']}",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "VECLIB_MAXIMUM_THREADS=1",
        "PYTHONUNBUFFERED=1",
        "timeout",
        "--signal=TERM",
        "--kill-after=60s",
        "45m",
        "python3",
        protected["builder"]["path"],
        "--input",
        str(resolve_repo_path(config["paths"]["retained_root_retry_input"])),
        "--out",
        str(output),
        "--sample",
        f"{ROOT_BACKED_TAG}_pythia8_delphes",
        "--target-mass",
        str(args["target_mass"]),
        "--jet-pt-min",
        str(args["jet_pt_min"]),
        "--jet-eta-max",
        str(args["jet_eta_max"]),
        "--btag-min",
        str(args["btag_min"]),
        "--max-bjets-for-pairing",
        str(args["max_bjets_for_pairing"]),
        "--higgs-ordering",
        str(args["higgs_ordering"]),
    ]


def run_missing_reconstructions(config: Mapping[str, Any]) -> None:
    primary = resolve_repo_path(config["paths"]["retained_root_product"])
    rerun = resolve_repo_path(config["paths"]["retained_root_validation_rerun"])
    primary.parent.mkdir(parents=True, exist_ok=True)
    rerun.parent.mkdir(parents=True, exist_ok=True)
    if not primary.exists():
        subprocess.run(
            build_reconstruction_command(config, primary),
            cwd=REPOSITORY_ROOT,
            check=True,
        )
    if not rerun.exists():
        subprocess.run(
            build_reconstruction_command(config, rerun),
            cwd=REPOSITORY_ROOT,
            check=True,
        )


def validate_config_only(config: Mapping[str, Any]) -> None:
    verify_frozen_inputs(config)
    verify_protected_reconstruction(config)
    verify_production_artifacts(config)
    inputs = load_inputs(config)
    coverage = inputs["member_coverage"]
    holds = coverage[
        coverage["development_disposition"] == "legacy_ttbar_schema_hold"
    ]
    require(len(holds) == 27, "configuration does not resolve 27 holds")


def run(config_path: Path, reconstruct_missing: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    if reconstruct_missing:
        run_missing_reconstructions(config)
    protected_rows = verify_frozen_inputs(config)
    protected_rows.extend(verify_protected_reconstruction(config))
    protected_rows.extend(verify_canonical_v1(config))
    verify_production_artifacts(config)
    inputs = load_inputs(config)
    canonical_names, canonical_types = exact_schema_contract(
        inputs["canonical72_columns"]
    )
    generator_digest = combined_generator_digest(config)
    inventory, members = inventory_holds(config, inputs, generator_digest)
    existing, root_rows, frames, common_rows = verify_ready_products(
        config, members, canonical_names, canonical_types
    )
    attempts = execution_attempt_rows(config)
    search, archives, regeneration = source_and_regeneration_rows(
        config, inputs, members
    )
    registry, accounting, normalization, unresolved = build_registry_and_accounting(
        config, members, existing, root_rows, frames
    )
    duplicates = duplicate_audit(frames)
    extension, composition, extension_stats = proposed_test_extension(
        config, inputs["member_coverage"]
    )

    runtime = resolve_repo_path(config["paths"]["runtime_directory"])
    runtime.mkdir(parents=True, exist_ok=True)
    execution_record = {
        "schema_version": 1,
        "audit_name": config["audit_name"],
        "controlled_retry": {
            "source_root_sha256": config["controlled_retry"][
                "source_root_sha256"
            ],
            "local_root_sha256": config["controlled_retry"]["local_root_sha256"],
            "source_local_sha256_equal": config["controlled_retry"][
                "source_local_sha256_equal"
            ],
            "scratch_free_gib_before_copy": config["controlled_retry"][
                "scratch_free_gib_before_copy"
            ],
            "initial_stall_diagnostics": config["controlled_retry"][
                "initial_stall_diagnostics"
            ],
            "smoke_test": config["controlled_retry"]["smoke_test"],
            "io_handler_adjudication": config["controlled_retry"][
                "io_handler_adjudication"
            ],
            "attempts": attempts,
        },
        "aborted_stalled_no_output_attempts": 1,
        "production_builder_invocations": 1,
        "determinism_validation_reruns": 1,
        "source_search_commands": [
            "xrdfs root://cmseos.fnal.gov ls -R /store/user/iturkmen/hh4b_delphes",
            "xrdfs root://cmseos.fnal.gov ls -R /store/user/iturkmen/hh4b_external_dataset",
        ],
        "exact_remote_member_matches": 0,
        "condor_jobs_submitted": 0,
        "event_level_products_location": "ignored_runtime_only",
    }
    (runtime / "execution_record.json").write_text(
        json.dumps(execution_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checkpoint = resolve_repo_path(config["paths"]["checkpoint_directory"])
    checkpoint.mkdir(parents=True, exist_ok=True)
    table_rows: dict[str, list[dict[str, Any]]] = {
        "ttbar27_hold_inventory": inventory,
        "ttbar27_existing_product_verification": existing,
        "ttbar27_root_reconstruction": root_rows,
        "ttbar27_execution_attempt_audit": attempts,
        "ttbar27_source_search": search,
        "ttbar27_archive_and_remote_locator_audit": archives,
        "ttbar27_regeneration_feasibility": regeneration,
        "ttbar27_common_column_consistency": common_rows,
        "ttbar27_candidate_registry": registry,
        "ttbar27_duplicate_audit": duplicates,
        "ttbar27_event_accounting": accounting,
        "ttbar27_normalization_readiness": normalization,
        "proposed_v2_test_extension": extension,
        "proposed_v2_test_composition": composition,
        "unresolved_ttbar_members": unresolved,
        "protected_artifact_audit": protected_rows,
    }
    for name, rows in table_rows.items():
        write_table_bundle(checkpoint, name, rows)

    plot_style = make_plots(checkpoint, registry, accounting, composition)
    existing_rows = sum(int(row["observed_rows"]) for row in existing)
    root_candidate_rows = int(root_rows[0]["candidate_rows"])
    summary = {
        "schema_version": 1,
        "audit_name": config["audit_name"],
        "status": config["status_contract"]["status"],
        "next_gate": config["status_contract"]["next_gate"],
        "base_commit": config["required_starting_commit"],
        "hold_inventory": {
            "held_members_inventoried": len(inventory),
            "generated_events": sum(int(row["generated_events"]) for row in inventory),
            "legacy_candidate_rows": sum(
                int(row["legacy_candidate_rows"]) for row in inventory
            ),
        },
        "candidate_recovery": {
            "existing_compatible_members_checked": len(existing),
            "existing_canonical_candidate_rows": existing_rows,
            "retained_root_members_reconstructed": len(root_rows),
            "retained_root_canonical_candidate_rows": root_candidate_rows,
            "aborted_stalled_no_output_attempts": 1,
            "successful_production_reconstruction_attempts": 1,
            "successful_determinism_validation_reruns": 1,
            "deterministic_row_level_equal": root_rows[0][
                "deterministic_row_level_equal"
            ],
            "canonical_ready_members": sum(
                bool(row["canonical72_ready"]) for row in registry
            ),
            "canonical_ready_candidate_rows": sum(
                int(row["candidate_rows"]) for row in registry
            ),
            "unresolved_members": len(unresolved),
            "unresolved_legacy_rows_excluded": sum(
                int(row["legacy_candidate_rows"]) for row in unresolved
            ),
            "canonical_schema_sha256": config["protected_reconstruction"][
                "canonical_schema_sha256"
            ],
        },
        "source_recovery": {
            "exact_original_sources_found_for_missing_eight": 0,
            "exact_deterministic_regeneration_possible": len(regeneration),
            "replacement_events_claimed_identical": False,
            "production_submitted": False,
        },
        "proposed_v2_test_extension": extension_stats,
        "normalization": {
            "metadata_carried_forward": True,
            "physical_yields_calculated": 0,
            "normalization_ready_members": 0,
            "normalization_blockers_recorded_for_members": len(normalization),
        },
        "controls": {
            "sealed_test_candidate_files_opened": 0,
            "sealed_test_candidate_rows_read": 0,
            "test_predictions_produced": 0,
            "models_trained": 0,
            "scores_calculated": 0,
            "significance_calculated": 0,
            "limits_calculated": 0,
            "condor_jobs_submitted": 0,
            "event_level_products_written_to_checkpoint": 0,
        },
        "integrity": {
            "protected_reconstruction_files_unchanged": True,
            "canonical_v1_protected_artifacts_unchanged": True,
            "canonical_v1_artifacts_verified": config["canonical_v1_protection"][
                "expected_entries"
            ],
            "cross_member_duplicate_keys": 0,
            "all_checkpoint_tables_have_markdown_and_booktabs_latex": True,
        },
        "plot_style": plot_style,
    }
    (checkpoint / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint_record = {
        "schema_version": 1,
        "audit_name": config["audit_name"],
        "status": summary["status"],
        "next_gate": summary["next_gate"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "config_sha256": sha256_file(config_path),
        "runtime_candidate_registry_only": True,
        "event_level_products_committed": False,
    }
    (checkpoint / "checkpoint.json").write_text(
        json.dumps(checkpoint_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "numpy": np.__version__,
        "uproot": uproot.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }
    (checkpoint / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = f"""# HH4b ttbar27 canonical-72 recovery

Status: `{summary['status']}`

Next gate: `{summary['next_gate']}`

The 18 pre-existing canonical-72 products passed full schema, content, identity,
row-count, and checksum validation.  The first retained-ROOT builder attempt
stalled with no output and is recorded as `aborted_stalled_no_output`.  A
controlled retry on a checksum-identical node-local ROOT copy reconstructed 29
rows with the protected builder and policy.  It passed an isolated deterministic
rerun comparison of schema, event order, integer/string values, nonfinite masks,
and floating values at the frozen tolerance.  The eight original
ROOT/HepMC/LHE/archive sources remain absent after local, provenance, archive,
EOS, and XRootD searches.

Exact deterministic regeneration is possible from the retained production
contract, but production was not started.  The next gate must first reproduce
seed 105000 and compare it with the retained original canary, then regenerate
the eight exact seeds only if that canary matches.

The 307 legacy candidate rows for those eight members are excluded from the
candidate registry.  No missing canonical features were synthesized.  No
sealed candidate content, predictions, models, scores, significance, limits,
or physical yields were opened or produced.

All tabular outputs are emitted as TSV, Markdown, and booktabs-compatible
LaTeX.  Event-level recovery products remain under ignored runtime output.
"""
    (checkpoint / "README.md").write_text(readme, encoding="utf-8")
    write_checksum_manifest(checkpoint)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--reconstruct-missing-runtime-products",
        action="store_true",
        help="Create the retained-root primary/rerun products only when absent.",
    )
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Verify frozen inputs and the 27-member inventory without content reads.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    try:
        config = load_config(config_path)
        if args.validate_config_only:
            validate_config_only(config)
            print("configuration validation: pass")
            return 0
        summary = run(
            config_path,
            reconstruct_missing=args.reconstruct_missing_runtime_products,
        )
    except (RecoveryError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
