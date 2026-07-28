#!/usr/bin/env python3
"""Train-only expanded HH4b cut-baseline runner.

This executable reads only the frozen train cache. It evaluates the
CMS-aligned R_HH(125,120) geometry and the frozen optimized R_HH<34
cut. Validation and final evaluation data are never opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0,
    str(REPOSITORY_ROOT / "scripts" / "analysis"),
)

from hh4b_bdt_v1_common import build_hierarchical_weights  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )
        require(
            bool(reader.fieldnames),
            f"{path}: missing TSV header",
        )
        return list(reader)


def write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    require(
        bool(rows),
        f"refusing to write empty TSV: {path}",
    )
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (REPOSITORY_ROOT / path).resolve()


def current_git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()


def cms_rhh_125_120(
    mbb1: Sequence[float],
    mbb2: Sequence[float],
) -> np.ndarray:
    mass1 = np.asarray(mbb1, dtype=np.float64)
    mass2 = np.asarray(mbb2, dtype=np.float64)

    if mass1.shape != mass2.shape:
        raise ValueError(
            "mbb1 and mbb2 must have identical shapes"
        )
    if not np.all(np.isfinite(mass1)):
        raise ValueError("mbb1 contains nonfinite values")
    if not np.all(np.isfinite(mass2)):
        raise ValueError("mbb2 contains nonfinite values")

    return np.sqrt(
        np.square(mass1 - 125.0)
        + np.square(mass2 - 120.0)
    )


def cms_region_labels(
    rhh: Sequence[float],
) -> np.ndarray:
    values = np.asarray(rhh, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError(
            "RHH values must be one-dimensional"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "RHH contains nonfinite values"
        )

    labels = np.full(
        values.shape,
        "outside_geometry",
        dtype=object,
    )
    labels[values < 30.0] = "cms_signal_region"
    labels[
        (values >= 30.0)
        & (values < 55.0)
    ] = "cms_control_region"
    return labels


def efficiency(
    selected: Sequence[bool],
    population: Sequence[bool],
    weights: Sequence[float] | None = None,
) -> float:
    selected_array = np.asarray(
        selected,
        dtype=bool,
    )
    population_array = np.asarray(
        population,
        dtype=bool,
    )

    if selected_array.shape != population_array.shape:
        raise ValueError(
            "selected and population masks differ in shape"
        )

    numerator = selected_array & population_array

    if weights is None:
        denominator = int(
            np.count_nonzero(population_array)
        )
        if denominator == 0:
            raise ValueError(
                "efficiency population is empty"
            )
        return float(
            np.count_nonzero(numerator)
            / denominator
        )

    weight_array = np.asarray(
        weights,
        dtype=np.float64,
    )
    if weight_array.shape != population_array.shape:
        raise ValueError("weight shape mismatch")
    if (
        not np.all(np.isfinite(weight_array))
        or np.any(weight_array <= 0.0)
    ):
        raise ValueError(
            "weights must be finite and positive"
        )

    denominator = float(
        np.sum(weight_array[population_array])
    )
    if denominator <= 0.0:
        raise ValueError(
            "weighted population has zero weight"
        )

    return float(
        np.sum(weight_array[numerator])
        / denominator
    )


def selection_metrics(
    selected: Sequence[bool],
    labels: Sequence[int],
    weights: Sequence[float],
    signal_modes: Sequence[str],
    background_families: Sequence[str],
) -> dict[str, Any]:
    selected_array = np.asarray(selected, dtype=bool)
    label_array = np.asarray(labels, dtype=np.int8)
    weight_array = np.asarray(weights, dtype=np.float64)
    mode_array = np.asarray(signal_modes, dtype=object)
    family_array = np.asarray(
        background_families,
        dtype=object,
    )

    shapes = {
        selected_array.shape,
        label_array.shape,
        weight_array.shape,
        mode_array.shape,
        family_array.shape,
    }
    if len(shapes) != 1:
        raise ValueError(
            "metric inputs differ in shape"
        )
    if set(np.unique(label_array)) != {0, 1}:
        raise ValueError(
            "metrics require both target classes"
        )

    signal = label_array == 1
    background = label_array == 0

    weighted_signal = efficiency(
        selected_array,
        signal,
        weight_array,
    )
    weighted_background = efficiency(
        selected_array,
        background,
        weight_array,
    )
    raw_signal = efficiency(
        selected_array,
        signal,
    )
    raw_background = efficiency(
        selected_array,
        background,
    )

    return {
        "selected_signal_rows": int(
            np.count_nonzero(
                selected_array & signal
            )
        ),
        "selected_background_rows": int(
            np.count_nonzero(
                selected_array & background
            )
        ),
        "weighted_signal_efficiency":
            weighted_signal,
        "weighted_background_efficiency":
            weighted_background,
        "weighted_background_rejection_fraction":
            1.0 - weighted_background,
        "weighted_inverse_background_efficiency": (
            1.0 / weighted_background
            if weighted_background > 0.0
            else None
        ),
        "raw_signal_efficiency": raw_signal,
        "raw_background_efficiency": raw_background,
        "raw_background_rejection_fraction":
            1.0 - raw_background,
        "raw_inverse_background_efficiency": (
            1.0 / raw_background
            if raw_background > 0.0
            else None
        ),
        "balanced_efficiency_proxy": (
            weighted_signal
            / math.sqrt(weighted_background)
            if weighted_background > 0.0
            else None
        ),
        "purity_proxy": (
            weighted_signal / weighted_background
            if weighted_background > 0.0
            else None
        ),
        "weighted_signal_mode_efficiencies": {
            str(mode): efficiency(
                selected_array,
                mode_array == mode,
                weight_array,
            )
            for mode in sorted(
                set(mode_array[signal])
            )
        },
        "weighted_background_family_efficiencies": {
            str(family): efficiency(
                selected_array,
                family_array == family,
                weight_array,
            )
            for family in sorted(
                set(family_array[background])
            )
        },
    }


def validate_protocol(
    protocol: Mapping[str, Any],
) -> None:
    if protocol.get("status") != (
        "hh4b_expanded_cut_bdt_protocol_v2_frozen"
    ):
        raise ValueError("protocol v2 is not frozen")

    cut = protocol["cut_baseline"]

    if cut["variable"] != "r_hh_125_120":
        raise ValueError(
            "cut variable is not r_hh_125_120"
        )
    if cut["cms_reference"][
        "signal_region"
    ]["threshold_GeV"] != 30.0:
        raise ValueError(
            "reference signal threshold is not 30"
        )
    if cut["cms_reference"][
        "control_region"
    ]["upper_GeV"] != 55.0:
        raise ValueError(
            "reference control upper edge is not 55"
        )
    if cut["optimized_nominal"][
        "threshold_GeV"
    ] != 34.0:
        raise ValueError(
            "optimized nominal threshold is not 34"
        )
    if protocol["controls"][
        "evaluation_access_authorized"
    ] is not False:
        raise ValueError(
            "evaluation access was authorized"
        )
    if protocol["controls"][
        "validation_cut_metrics_authorized_before_bdt_freeze"
    ] is not False:
        raise ValueError(
            "premature validation access was authorized"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--source-commit",
        required=True,
    )
    args = parser.parse_args()

    source_commit = args.source_commit.lower()

    require(
        re.fullmatch(
            r"[0-9a-f]{40}",
            source_commit,
        )
        is not None,
        "source commit must be a full SHA",
    )
    require(
        current_git_head() == source_commit,
        "runner source commit differs from HEAD",
    )

    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)

    require(
        config_path.is_file(),
        f"missing config: {config_path}",
    )
    require(
        not output_dir.exists(),
        f"refusing to overwrite: {output_dir}",
    )

    temporary_dir = output_dir.with_name(
        output_dir.name + "_incomplete"
    )
    require(
        not temporary_dir.exists(),
        f"temporary output exists: {temporary_dir}",
    )

    protocol = json.loads(
        config_path.read_text(encoding="utf-8")
    )
    validate_protocol(protocol)

    train_spec = protocol["cache"][
        "development_train_cache"
    ]
    train_cache = resolve_path(
        train_spec["path"]
    )

    require(
        train_cache.is_file(),
        f"missing train cache: {train_cache}",
    )
    require(
        sha256_file(train_cache)
        == train_spec["sha256"],
        "train cache SHA-256 mismatch",
    )

    manifest_spec = protocol[
        "development_manifest"
    ]
    manifest_path = resolve_path(
        manifest_spec["path"]
    )

    require(
        manifest_path.is_file(),
        f"missing manifest: {manifest_path}",
    )
    require(
        sha256_file(manifest_path)
        == manifest_spec["sha256"],
        "development manifest SHA-256 mismatch",
    )

    manifest = read_tsv(manifest_path)
    train_manifest = [
        row
        for row in manifest
        if row["final_split"] == "train"
    ]

    require(
        len(train_manifest) == 464,
        "train manifest members are not 464",
    )

    group_ids = [
        row["group_id"]
        for row in train_manifest
    ]
    require(
        len(set(group_ids)) == 464,
        "train group IDs are not unique",
    )

    member_index_by_group = {
        group_id: index
        for index, group_id in enumerate(
            group_ids
        )
    }

    signal_map = protocol["weighting"][
        "signal_mode_mapping"
    ]
    background_map = protocol["weighting"][
        "background_family_mapping"
    ]

    members = []

    for member_index, row in enumerate(
        train_manifest
    ):
        sample_class = row["sample_class"]
        process = row["process_or_mode"]

        if sample_class == "signal":
            require(
                process in signal_map,
                f"unmapped signal process: {process}",
            )
            stratum = signal_map[process]
        elif sample_class == "background":
            require(
                process in background_map,
                f"unmapped background process: {process}",
            )
            stratum = background_map[process]
        else:
            raise RuntimeError(
                f"unsupported class: {sample_class}"
            )

        members.append(
            {
                "member_index": member_index,
                "sample_class": sample_class,
                "stratum": stratum,
                "candidate_rows": int(
                    row[
                        "candidate_rows_metadata"
                    ]
                ),
            }
        )

    require(
        sum(
            member["candidate_rows"]
            for member in members
        )
        == 53_162,
        "train manifest rows are not 53162",
    )

    read_columns = [
        "mbb1",
        "mbb2",
        "r_hh_125_120",
        "registry_sample_class",
        "registry_process_or_mode",
        "registry_group_id",
        "registry_final_split",
        "registry_training_target",
    ]

    frame = pd.read_parquet(
        train_cache,
        columns=read_columns,
        engine="pyarrow",
    ).reset_index(drop=True)

    require(
        len(frame) == 53_162,
        "train cache rows are not 53162",
    )
    require(
        set(
            frame[
                "registry_final_split"
            ].astype(str)
        )
        == {"train"},
        "non-train rows were opened",
    )

    observed_groups = set(
        frame[
            "registry_group_id"
        ].astype(str)
    )
    require(
        not (
            observed_groups
            - set(member_index_by_group)
        ),
        "train cache contains unknown group IDs",
    )

    row_member_indices = (
        frame[
            "registry_group_id"
        ]
        .astype(str)
        .map(member_index_by_group)
        .to_numpy(dtype=np.int64)
    )

    weight_result = build_hierarchical_weights(
        row_member_indices,
        members,
        signal_modes=sorted(
            set(signal_map.values())
        ),
        background_families=sorted(
            set(background_map.values())
        ),
        tolerance=float(
            protocol["weighting"]["tolerance"]
        ),
    )
    weights = weight_result.weights

    require(
        np.isclose(
            float(np.mean(weights)),
            1.0,
            atol=1.0e-10,
            rtol=0.0,
        ),
        "mean row weight is not one",
    )

    labels = frame[
        "registry_training_target"
    ].to_numpy(dtype=np.int8)

    sample_classes = frame[
        "registry_sample_class"
    ].astype(str).to_numpy()

    processes = frame[
        "registry_process_or_mode"
    ].astype(str).to_numpy()

    signal_modes = np.asarray(
        [
            signal_map[process]
            if sample_class == "signal"
            else ""
            for sample_class, process in zip(
                sample_classes,
                processes,
            )
        ],
        dtype=object,
    )

    background_families = np.asarray(
        [
            background_map[process]
            if sample_class == "background"
            else ""
            for sample_class, process in zip(
                sample_classes,
                processes,
            )
        ],
        dtype=object,
    )

    stored_rhh = frame[
        "r_hh_125_120"
    ].to_numpy(dtype=np.float64)

    recomputed_rhh = cms_rhh_125_120(
        frame["mbb1"],
        frame["mbb2"],
    )

    maximum_formula_difference = float(
        np.max(
            np.abs(
                stored_rhh - recomputed_rhh
            )
        )
    )

    require(
        maximum_formula_difference <= 1.0e-9,
        "stored RHH disagrees with formula",
    )

    definitions = [
        (
            "cms_reference_rhh30",
            "published_reference_geometry",
            30.0,
        ),
        (
            "optimized_nominal_rhh34",
            "frozen_train_only_nominal",
            34.0,
        ),
        (
            "higher_purity_rhh31p5",
            "frozen_rounded_alternative",
            31.5,
        ),
        (
            "higher_efficiency_rhh35p5",
            "frozen_rounded_alternative",
            35.5,
        ),
    ]

    metric_rows = []
    metric_payload = {}

    for baseline, role, threshold in definitions:
        metrics = selection_metrics(
            stored_rhh < threshold,
            labels,
            weights,
            signal_modes,
            background_families,
        )

        metric_payload[baseline] = metrics

        metric_rows.append(
            {
                "baseline": baseline,
                "role": role,
                "variable": "r_hh_125_120",
                "operator": "less_than",
                "threshold_GeV": threshold,
                "selected_signal_rows":
                    metrics[
                        "selected_signal_rows"
                    ],
                "selected_background_rows":
                    metrics[
                        "selected_background_rows"
                    ],
                "weighted_signal_efficiency":
                    metrics[
                        "weighted_signal_efficiency"
                    ],
                "weighted_background_efficiency":
                    metrics[
                        "weighted_background_efficiency"
                    ],
                "weighted_background_rejection_fraction":
                    metrics[
                        "weighted_background_rejection_fraction"
                    ],
                "weighted_inverse_background_efficiency":
                    metrics[
                        "weighted_inverse_background_efficiency"
                    ],
                "raw_signal_efficiency":
                    metrics[
                        "raw_signal_efficiency"
                    ],
                "raw_background_efficiency":
                    metrics[
                        "raw_background_efficiency"
                    ],
                "raw_background_rejection_fraction":
                    metrics[
                        "raw_background_rejection_fraction"
                    ],
                "raw_inverse_background_efficiency":
                    metrics[
                        "raw_inverse_background_efficiency"
                    ],
                "balanced_efficiency_proxy":
                    metrics[
                        "balanced_efficiency_proxy"
                    ],
                "purity_proxy":
                    metrics["purity_proxy"],
                "weighted_signal_mode_efficiencies":
                    json.dumps(
                        metrics[
                            "weighted_signal_mode_efficiencies"
                        ],
                        sort_keys=True,
                    ),
                "weighted_background_family_efficiencies":
                    json.dumps(
                        metrics[
                            "weighted_background_family_efficiencies"
                        ],
                        sort_keys=True,
                    ),
                "physical_normalization": False,
                "status": "pass",
            }
        )

    regions = cms_region_labels(stored_rhh)
    region_rows = []

    populations: list[
        tuple[str, str, np.ndarray]
    ] = [
        (
            "class",
            "signal",
            labels == 1,
        ),
        (
            "class",
            "background",
            labels == 0,
        ),
    ]

    for mode in sorted(
        set(signal_modes[labels == 1])
    ):
        populations.append(
            (
                "signal_mode",
                str(mode),
                signal_modes == mode,
            )
        )

    for family in sorted(
        set(
            background_families[
                labels == 0
            ]
        )
    ):
        populations.append(
            (
                "background_family",
                str(family),
                background_families
                == family,
            )
        )

    for region in (
        "cms_signal_region",
        "cms_control_region",
        "outside_geometry",
    ):
        selected = regions == region

        for level, name, population in populations:
            region_rows.append(
                {
                    "region": region,
                    "population_level": level,
                    "population": name,
                    "population_rows": int(
                        np.count_nonzero(population)
                    ),
                    "selected_rows": int(
                        np.count_nonzero(
                            selected & population
                        )
                    ),
                    "raw_fraction": efficiency(
                        selected,
                        population,
                    ),
                    "weighted_fraction": efficiency(
                        selected,
                        population,
                        weights,
                    ),
                    "status": "pass",
                }
            )

    summary = {
        "schema_version": 1,
        "status":
            "hh4b_expanded_cut_baseline_train_only_pass",
        "source_commit": source_commit,
        "protocol_path": str(
            config_path.relative_to(
                REPOSITORY_ROOT
            )
        ),
        "protocol_sha256":
            sha256_file(config_path),
        "train_cache": {
            "path": str(
                train_cache.relative_to(
                    REPOSITORY_ROOT
                )
            ),
            "sha256": sha256_file(train_cache),
            "rows": len(frame),
            "manifest_members": 464,
            "candidate_bearing_members": int(
                frame[
                    "registry_group_id"
                ].nunique()
            ),
        },
        "cms_geometry": {
            "variable": "r_hh_125_120",
            "formula":
                "sqrt((mbb1-125)^2+(mbb2-120)^2)",
            "signal_region":
                "r_hh_125_120 < 30",
            "control_region":
                "30 <= r_hh_125_120 < 55",
            "outside_geometry":
                "r_hh_125_120 >= 55",
            "maximum_formula_difference":
                maximum_formula_difference,
            "exact_cms_reconstruction_claimed":
                False,
        },
        "cut_metrics": metric_payload,
        "weighting": {
            "mean_row_weight": float(
                np.mean(weights)
            ),
            "physical_normalization": False,
        },
        "controls": {
            "train_candidate_rows_read":
                len(frame),
            "validation_candidate_files_opened":
                0,
            "validation_candidate_rows_read":
                0,
            "evaluation_candidate_files_opened":
                0,
            "evaluation_candidate_rows_read":
                0,
            "models_trained": 0,
            "scores_calculated": 0,
            "physical_normalization_performed":
                False,
        },
        "next_gate": (
            "freeze_train_only_cut_baseline_"
            "then_derive_bdt_oof_models"
        ),
    }

    temporary_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_tsv(
        temporary_dir
        / "cut_baseline_metrics.tsv",
        metric_rows,
    )
    write_tsv(
        temporary_dir
        / "cms_region_accounting.tsv",
        region_rows,
    )
    write_tsv(
        temporary_dir
        / "member_weight_audit.tsv",
        weight_result.member_rows,
    )
    write_tsv(
        temporary_dir
        / "weight_summary.tsv",
        weight_result.summary_rows,
    )

    (
        temporary_dir / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    cms = metric_payload[
        "cms_reference_rhh30"
    ]
    nominal = metric_payload[
        "optimized_nominal_rhh34"
    ]

    (
        temporary_dir / "README.md"
    ).write_text(
        f"""# Expanded HH4b train-only cut baseline

| Baseline | Weighted signal efficiency | Weighted background efficiency |
|---|---:|---:|
| Reference RHH < 30 | {cms['weighted_signal_efficiency']:.8f} | {cms['weighted_background_efficiency']:.8f} |
| Optimized RHH < 34 | {nominal['weighted_signal_efficiency']:.8f} | {nominal['weighted_background_efficiency']:.8f} |

The variable is `r_hh_125_120`. These are development-balancing
metrics, not physical yields. Validation and final evaluation content
remained unopened.
""",
        encoding="utf-8",
    )

    artifacts = sorted(
        path
        for path in temporary_dir.iterdir()
        if (
            path.is_file()
            and path.name != "SHA256SUMS"
        )
    )

    (
        temporary_dir / "SHA256SUMS"
    ).write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in artifacts
        ) + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary_dir,
        output_dir,
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )
    print()
    print("CMS_RHH_125_120_FORMULA_VALIDATION_PASS")
    print("CMS_SIGNAL_CONTROL_OUTSIDE_GEOMETRY_PASS")
    print("TRAIN_ONLY_HIERARCHICAL_WEIGHTS_PASS")
    print("CMS_REFERENCE_RHH30_TRAIN_BASELINE_PASS")
    print("OPTIMIZED_RHH34_TRAIN_BASELINE_PASS")
    print("NO_VALIDATION_CANDIDATE_CONTENT_OPENED")
    print("NO_EVALUATION_CANDIDATE_CONTENT_OPENED")
    print("HH4B_EXPANDED_CUT_BASELINE_TRAIN_ONLY_PASS")


if __name__ == "__main__":
    main()
