#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from hh4b_plot_style import (
    apply_hh4b_paper_style,
    save_figure,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def read_tsv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(
        newline="",
        encoding="utf-8",
        errors="strict",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    require(fields, f"{path}: missing TSV header")
    return fields, rows


def float_array(
    table: Any,
    column: str,
) -> np.ndarray:
    source = table[column]

    require(
        source.null_count == 0,
        f"{column} contains null values",
    )

    values = np.asarray(
        source.combine_chunks().to_numpy(
            zero_copy_only=False,
        ),
        dtype=np.float64,
    )

    require(
        np.all(np.isfinite(values)),
        f"{column} contains nonfinite values",
    )

    return values


def string_array(
    table: Any,
    column: str,
) -> np.ndarray:
    source = table[column]

    require(
        source.null_count == 0,
        f"{column} contains null values",
    )

    return np.asarray(
        source.to_pylist(),
        dtype=object,
    )


def point_metrics(
    *,
    selected_signal: int,
    selected_background: int,
    mode_counts: dict[str, int],
    family_counts: dict[str, int],
    signal_denominator: int,
    background_denominator: int,
    mode_denominators: dict[str, int],
    family_denominators: dict[str, int],
    signal_modes: list[str],
    background_families: list[str],
    minimum_background: int,
    minimum_signal_mode: int,
) -> dict[str, Any]:
    mode_efficiencies = [
        mode_counts[mode]
        / mode_denominators[mode]
        for mode in signal_modes
    ]

    family_efficiencies = [
        family_counts[family]
        / family_denominators[family]
        for family in background_families
    ]

    balanced_signal_efficiency = float(
        np.mean(mode_efficiencies)
    )

    family_balanced_background_efficiency = float(
        np.mean(family_efficiencies)
    )

    raw_proxy = (
        selected_signal
        / math.sqrt(selected_background)
        if selected_background > 0
        else math.inf
    )

    balanced_proxy = (
        balanced_signal_efficiency
        / math.sqrt(
            family_balanced_background_efficiency
        )
        if family_balanced_background_efficiency > 0.0
        else math.inf
    )

    balanced_s_over_b = (
        balanced_signal_efficiency
        / family_balanced_background_efficiency
        if family_balanced_background_efficiency > 0.0
        else math.inf
    )

    eligible = (
        selected_background >= minimum_background
        and all(
            mode_counts[mode] >= minimum_signal_mode
            for mode in signal_modes
        )
        and math.isfinite(balanced_proxy)
    )

    return {
        "selected_signal":
            selected_signal,
        "selected_background":
            selected_background,
        "unweighted_signal_efficiency":
            selected_signal / signal_denominator,
        "unweighted_background_efficiency":
            selected_background / background_denominator,
        "ggf_signal_efficiency":
            mode_efficiencies[0],
        "vbf_signal_efficiency":
            mode_efficiencies[1],
        "balanced_signal_efficiency":
            balanced_signal_efficiency,
        "family_balanced_background_efficiency":
            family_balanced_background_efficiency,
        "background_family_efficiency_std":
            float(
                np.std(
                    family_efficiencies,
                    ddof=0,
                )
            ),
        "minimum_background_family_efficiency":
            min(family_efficiencies),
        "maximum_background_family_efficiency":
            max(family_efficiencies),
        "raw_count_proxy_s_over_sqrt_b":
            raw_proxy,
        "balanced_proxy_s_over_sqrt_b":
            balanced_proxy,
        "balanced_proxy_s_over_b":
            balanced_s_over_b,
        "eligible_for_ranking":
            eligible,
    }


def add_pareto_flags(
    rows: list[dict[str, Any]],
) -> None:
    signal = np.asarray([
        float(row["balanced_signal_efficiency"])
        for row in rows
    ])

    background = np.asarray([
        float(
            row[
                "family_balanced_background_efficiency"
            ]
        )
        for row in rows
    ])

    for index, row in enumerate(rows):
        dominated = np.any(
            (signal >= signal[index])
            & (background <= background[index])
            & (
                (signal > signal[index])
                | (background < background[index])
            )
        )

        row["pareto_frontier"] = bool(
            not dominated
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan every observed train-sample RHH "
            "transition around the HH4b fine plateau."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo = script_path.parents[2]

    config_path = args.config

    if not config_path.is_absolute():
        config_path = (
            repo / config_path
        ).resolve()

    require(
        config_path.is_file(),
        f"missing configuration: {config_path}",
    )

    config = json.loads(
        config_path.read_text(
            encoding="utf-8",
        )
    )

    require(
        config.get("schema_version") == 1,
        "unsupported configuration schema",
    )
    require(
        config.get("scan_stage")
        == "exact_observed_rhh_train_only",
        "configuration is not an exact train-only scan",
    )
    require(
        config.get("dataset_split") == "train",
        "exact optimization must use train only",
    )
    require(
        config.get("test_access_allowed") is False,
        "test access must remain disabled",
    )
    require(
        config.get(
            "physical_significance_authorized"
        )
        is False,
        "physical significance must remain disabled",
    )

    cache_path = Path(
        config["cache_path"]
    ).resolve()

    require(
        cache_path.is_file(),
        f"missing cache: {cache_path}",
    )
    require(
        sha256_file(cache_path)
        == config["cache_sha256"],
        "cache SHA-256 mismatch",
    )

    fine_summary_path = Path(
        config["fine_summary_path"]
    )

    if not fine_summary_path.is_absolute():
        fine_summary_path = (
            repo / fine_summary_path
        ).resolve()

    fine_plateau_path = Path(
        config["fine_plateau_path"]
    )

    if not fine_plateau_path.is_absolute():
        fine_plateau_path = (
            repo / fine_plateau_path
        ).resolve()

    require(
        sha256_file(fine_summary_path)
        == config["fine_summary_sha256"],
        "fine-summary SHA-256 mismatch",
    )
    require(
        sha256_file(fine_plateau_path)
        == config["fine_plateau_sha256"],
        "fine-plateau SHA-256 mismatch",
    )

    fine_summary = json.loads(
        fine_summary_path.read_text(
            encoding="utf-8",
        )
    )

    require(
        fine_summary.get("status")
        == "hh4b_train_only_fine_cut_scan_pass",
        "fine scan did not pass",
    )
    require(
        fine_summary.get(
            "validation_rows_evaluated"
        ) == 0,
        "fine scan used validation",
    )
    require(
        fine_summary.get(
            "test_candidate_files_opened"
        ) == 0,
        "fine scan opened test candidates",
    )

    _, fine_plateau_rows = read_tsv(
        fine_plateau_path
    )

    require(
        len(fine_plateau_rows) == 13,
        "fine plateau does not contain 13 points",
    )

    output_dir = Path(
        config["output_dir"]
    )

    if not output_dir.is_absolute():
        output_dir = (
            repo / output_dir
        ).resolve()

    require(
        not output_dir.exists(),
        f"refusing to overwrite {output_dir}",
    )

    temporary_dir = Path(
        str(output_dir) + "_incomplete"
    )

    require(
        not temporary_dir.exists(),
        f"temporary output exists: {temporary_dir}",
    )

    acceptance = config["fixed_acceptance"]

    required_columns = [
        "sample_class",
        "process_or_mode",
        "dataset_split",
        acceptance["jet_pt_variable"],
        acceptance["jet_eta_variable"],
        acceptance["rhh_variable"],
    ]

    table = pq.read_table(
        cache_path,
        columns=required_columns,
        filters=[
            ("dataset_split", "=", "train")
        ],
        use_threads=True,
    )

    expected = config["expected_rows"]

    require(
        table.num_rows == expected["total"],
        (
            f"expected {expected['total']} train rows, "
            f"read {table.num_rows}"
        ),
    )

    sample_class = string_array(
        table,
        "sample_class",
    )
    process = string_array(
        table,
        "process_or_mode",
    )
    split = string_array(
        table,
        "dataset_split",
    )

    require(
        set(split.tolist()) == {"train"},
        "non-train rows entered exact scan",
    )

    min_jet_pt = float_array(
        table,
        acceptance["jet_pt_variable"],
    )
    max_abs_eta = float_array(
        table,
        acceptance["jet_eta_variable"],
    )
    rhh = float_array(
        table,
        acceptance["rhh_variable"],
    )

    signal_mask = sample_class == "signal"
    background_mask = sample_class == "background"

    signal_denominator = int(
        np.count_nonzero(signal_mask)
    )
    background_denominator = int(
        np.count_nonzero(background_mask)
    )

    require(
        signal_denominator == expected["signal"],
        "signal denominator mismatch",
    )
    require(
        background_denominator
        == expected["background"],
        "background denominator mismatch",
    )

    signal_modes = list(config["signal_modes"])

    require(
        signal_modes
        == ["ggf_hh4b", "vbf_hh4b"],
        "unexpected signal-mode order",
    )

    mode_masks = {
        mode:
            signal_mask & (process == mode)
        for mode in signal_modes
    }

    mode_denominators = {
        mode:
            int(np.count_nonzero(mode_masks[mode]))
        for mode in signal_modes
    }

    require(
        mode_denominators["ggf_hh4b"]
        == expected["ggf_hh4b"],
        "ggF denominator mismatch",
    )
    require(
        mode_denominators["vbf_hh4b"]
        == expected["vbf_hh4b"],
        "VBF denominator mismatch",
    )

    background_families = sorted(
        set(process[background_mask].tolist())
    )

    require(
        len(background_families) == 22,
        (
            "expected 22 background families, "
            f"found {len(background_families)}"
        ),
    )

    family_masks = {
        family:
            background_mask
            & (process == family)
        for family in background_families
    }

    family_denominators = {
        family:
            int(
                np.count_nonzero(
                    family_masks[family]
                )
            )
        for family in background_families
    }

    fixed_acceptance = (
        min_jet_pt
        > float(
            acceptance["jet_pt_min_GeV"]
        )
    ) & (
        max_abs_eta
        < float(
            acceptance["jet_abs_eta_max"]
        )
    )

    accepted_indices = np.flatnonzero(
        fixed_acceptance
    )

    require(
        len(accepted_indices) > 0,
        "fixed acceptance selects no rows",
    )

    accepted_rhh = rhh[accepted_indices]

    order = np.argsort(
        accepted_rhh,
        kind="mergesort",
    )

    sorted_indices = accepted_indices[order]
    sorted_rhh = rhh[sorted_indices]
    sorted_sample_class = sample_class[
        sorted_indices
    ]
    sorted_process = process[sorted_indices]

    prefix_signal = np.cumsum(
        sorted_sample_class == "signal",
        dtype=np.int64,
    )

    prefix_background = np.cumsum(
        sorted_sample_class == "background",
        dtype=np.int64,
    )

    mode_prefix = {
        mode:
            np.cumsum(
                (
                    (sorted_sample_class == "signal")
                    & (sorted_process == mode)
                ),
                dtype=np.int64,
            )
        for mode in signal_modes
    }

    family_prefix = {
        family:
            np.cumsum(
                (
                    (sorted_sample_class == "background")
                    & (sorted_process == family)
                ),
                dtype=np.int64,
            )
        for family in background_families
    }

    window = config["exact_scan_window"]

    window_min = float(window["rhh_min"])
    window_max = float(window["rhh_max"])

    transition_values = np.unique(
        sorted_rhh[
            (sorted_rhh >= window_min)
            & (sorted_rhh <= window_max)
        ]
    )

    require(
        len(transition_values) >= 100,
        (
            "too few observed transitions in window: "
            f"{len(transition_values)}"
        ),
    )

    end_positions = (
        np.searchsorted(
            sorted_rhh,
            transition_values,
            side="right",
        )
        - 1
    )

    ranking = config["ranking"]
    ranking_metric = ranking["metric"]
    minimum_background = int(
        ranking["minimum_selected_background"]
    )
    minimum_signal_mode = int(
        ranking[
            "minimum_selected_events_per_signal_mode"
        ]
    )

    exact_rows: list[dict[str, Any]] = []

    previous_total = int(
        np.searchsorted(
            sorted_rhh,
            transition_values[0],
            side="left",
        )
    )

    for transition_index, (
        transition_value,
        end_position,
    ) in enumerate(
        zip(
            transition_values,
            end_positions,
        ),
        start=1,
    ):
        selected_signal = int(
            prefix_signal[end_position]
        )
        selected_background = int(
            prefix_background[end_position]
        )

        mode_counts = {
            mode:
                int(mode_prefix[mode][end_position])
            for mode in signal_modes
        }

        family_counts = {
            family:
                int(
                    family_prefix[family][end_position]
                )
            for family in background_families
        }

        metrics = point_metrics(
            selected_signal=selected_signal,
            selected_background=selected_background,
            mode_counts=mode_counts,
            family_counts=family_counts,
            signal_denominator=signal_denominator,
            background_denominator=background_denominator,
            mode_denominators=mode_denominators,
            family_denominators=family_denominators,
            signal_modes=signal_modes,
            background_families=background_families,
            minimum_background=minimum_background,
            minimum_signal_mode=minimum_signal_mode,
        )

        selected_total = (
            selected_signal + selected_background
        )

        events_added = (
            selected_total - previous_total
        )

        previous_total = selected_total

        exact_rows.append({
            "point_id":
                f"exact_rhh_{transition_index:05d}",
            "point_type":
                "observed_transition",
            "transition_index":
                transition_index,
            "rhh_transition_value":
                float(transition_value),
            "rhh_threshold_value":
                float(
                    np.nextafter(
                        transition_value,
                        np.inf,
                    )
                ),
            "events_added_at_transition":
                events_added,
            **metrics,
        })

    add_pareto_flags(exact_rows)

    eligible_rows = [
        row
        for row in exact_rows
        if bool(row["eligible_for_ranking"])
    ]

    require(
        eligible_rows,
        "no exact thresholds eligible for ranking",
    )

    best_row = max(
        eligible_rows,
        key=lambda row: float(
            row[ranking_metric]
        ),
    )

    best_metric = float(
        best_row[ranking_metric]
    )

    stable_fraction = float(
        ranking["stable_fraction_of_best"]
    )

    for row in exact_rows:
        fraction = (
            float(row[ranking_metric])
            / best_metric
        )

        row["fraction_of_best_proxy"] = fraction

        row["near_optimal"] = bool(
            row["eligible_for_ranking"]
            and fraction >= stable_fraction
        )

        row["stable_component"] = False

    best_index = exact_rows.index(best_row)

    stable_start = best_index
    stable_end = best_index

    while (
        stable_start > 0
        and bool(
            exact_rows[
                stable_start - 1
            ]["near_optimal"]
        )
    ):
        stable_start -= 1

    while (
        stable_end + 1 < len(exact_rows)
        and bool(
            exact_rows[
                stable_end + 1
            ]["near_optimal"]
        )
    ):
        stable_end += 1

    for index in range(
        stable_start,
        stable_end + 1,
    ):
        exact_rows[index][
            "stable_component"
        ] = True

    stable_rows = exact_rows[
        stable_start:
        stable_end + 1
    ]

    require(
        stable_rows,
        "stable component is empty",
    )

    high_purity_row = max(
        stable_rows,
        key=lambda row: float(
            row["balanced_proxy_s_over_b"]
        ),
    )

    high_efficiency_row = max(
        stable_rows,
        key=lambda row: float(
            row["balanced_signal_efficiency"]
        ),
    )

    def evaluate_rounded_threshold(
        label: str,
        threshold: float,
    ) -> dict[str, Any]:
        selected = (
            fixed_acceptance
            & (rhh < threshold)
        )

        selected_signal = int(
            np.count_nonzero(
                selected & signal_mask
            )
        )

        selected_background = int(
            np.count_nonzero(
                selected & background_mask
            )
        )

        mode_counts = {
            mode:
                int(
                    np.count_nonzero(
                        selected & mode_masks[mode]
                    )
                )
            for mode in signal_modes
        }

        family_counts = {
            family:
                int(
                    np.count_nonzero(
                        selected
                        & family_masks[family]
                    )
                )
            for family in background_families
        }

        metrics = point_metrics(
            selected_signal=selected_signal,
            selected_background=selected_background,
            mode_counts=mode_counts,
            family_counts=family_counts,
            signal_denominator=signal_denominator,
            background_denominator=background_denominator,
            mode_denominators=mode_denominators,
            family_denominators=family_denominators,
            signal_modes=signal_modes,
            background_families=background_families,
            minimum_background=minimum_background,
            minimum_signal_mode=minimum_signal_mode,
        )

        return {
            "point_id":
                (
                    "rounded_rhh_"
                    + str(threshold)
                    .replace(".", "p")
                ),
            "point_type":
                "rounded_anchor",
            "transition_index":
                "",
            "rhh_transition_value":
                "",
            "rhh_threshold_value":
                threshold,
            "events_added_at_transition":
                "",
            "label":
                label,
            **metrics,
            "pareto_frontier":
                "",
            "fraction_of_best_proxy":
                float(
                    metrics[ranking_metric]
                )
                / best_metric,
            "near_optimal":
                float(
                    metrics[ranking_metric]
                )
                / best_metric
                >= stable_fraction,
            "stable_component":
                "",
        }

    rounded_rows = [
        evaluate_rounded_threshold(
            anchor["label"],
            float(anchor["rhh_sr_max"]),
        )
        for anchor
        in config[
            "rounded_anchor_thresholds"
        ]
    ]

    rounded_34 = next(
        row
        for row in rounded_rows
        if math.isclose(
            float(row["rhh_threshold_value"]),
            34.0,
            abs_tol=1.0e-12,
        )
    )

    exact_change_from_rounded_34 = (
        100.0
        * (
            float(best_row[ranking_metric])
            / float(
                rounded_34[ranking_metric]
            )
            - 1.0
        )
    )

    shortlist_candidates = [
        (
            "exact_proxy_maximum",
            best_row,
        ),
        (
            "stable_high_purity",
            high_purity_row,
        ),
        (
            "stable_high_efficiency",
            high_efficiency_row,
        ),
    ]

    shortlist_candidates.extend(
        (
            str(row["label"]),
            row,
        )
        for row in rounded_rows
    )

    shortlist_rows: list[dict[str, Any]] = []
    seen_signatures = set()

    for role, row in shortlist_candidates:
        signature = (
            int(row["selected_signal"]),
            int(row["selected_background"]),
        )

        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)

        shortlist_rows.append({
            "shortlist_role":
                role,
            **row,
        })

    best_at_window_boundary = bool(
        best_index == 0
        or best_index
        == len(exact_rows) - 1
    )

    temporary_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    point_fields = [
        "point_id",
        "point_type",
        "transition_index",
        "rhh_transition_value",
        "rhh_threshold_value",
        "events_added_at_transition",
        "selected_signal",
        "selected_background",
        "unweighted_signal_efficiency",
        "unweighted_background_efficiency",
        "ggf_signal_efficiency",
        "vbf_signal_efficiency",
        "balanced_signal_efficiency",
        "family_balanced_background_efficiency",
        "background_family_efficiency_std",
        "minimum_background_family_efficiency",
        "maximum_background_family_efficiency",
        "raw_count_proxy_s_over_sqrt_b",
        "balanced_proxy_s_over_sqrt_b",
        "balanced_proxy_s_over_b",
        "eligible_for_ranking",
        "pareto_frontier",
        "fraction_of_best_proxy",
        "near_optimal",
        "stable_component",
    ]

    write_tsv(
        temporary_dir
        / "exact_rhh_scan_points.tsv",
        exact_rows,
        point_fields,
    )

    write_tsv(
        temporary_dir
        / "exact_rhh_scan_shortlist.tsv",
        shortlist_rows,
        [
            "shortlist_role",
            "label",
            *point_fields,
        ],
    )

    write_tsv(
        temporary_dir
        / "exact_rhh_rounded_anchors.tsv",
        rounded_rows,
        [
            "label",
            *point_fields,
        ],
    )

    shutil.copy2(
        config_path,
        temporary_dir
        / "scan_config_snapshot.json",
    )

    apply_hh4b_paper_style()

    formats = [
        str(value)
        for value
        in config["plotting"]["formats"]
    ]

    dpi = int(
        config["plotting"]["dpi"]
    )

    x = np.asarray([
        float(row["rhh_transition_value"])
        for row in exact_rows
    ])

    proxy = np.asarray([
        float(
            row[
                "balanced_proxy_s_over_sqrt_b"
            ]
        )
        for row in exact_rows
    ])

    purity = np.asarray([
        float(row["balanced_proxy_s_over_b"])
        for row in exact_rows
    ])

    signal_efficiency = np.asarray([
        float(row["balanced_signal_efficiency"])
        for row in exact_rows
    ])

    background_efficiency = np.asarray([
        float(
            row[
                "family_balanced_background_efficiency"
            ]
        )
        for row in exact_rows
    ])

    stable_min = float(
        stable_rows[0]["rhh_transition_value"]
    )

    stable_max = float(
        stable_rows[-1]["rhh_transition_value"]
    )

    plot_paths: list[Path] = []

    figure, axis = plt.subplots(
        figsize=(7.4, 5.2)
    )

    axis.step(
        x,
        proxy,
        where="post",
        linewidth=1.4,
    )

    axis.axvspan(
        stable_min,
        stable_max,
        alpha=0.15,
        label="99.5% stable component",
    )

    axis.axvline(
        34.0,
        linestyle="--",
        linewidth=1.2,
        label="Rounded threshold 34",
    )

    axis.scatter(
        [
            float(
                best_row[
                    "rhh_transition_value"
                ]
            )
        ],
        [
            float(best_row[ranking_metric])
        ],
        marker="o",
        s=90,
        facecolors="none",
        edgecolors="black",
        linewidths=1.4,
        label="Exact proxy maximum",
    )

    axis.set_xlabel(
        r"Observed $R_{HH}$ transition"
    )
    axis.set_ylabel(
        r"$\epsilon_S^{\mathrm{bal}}/"
        r"\sqrt{\epsilon_B^{\mathrm{fam}}}$"
    )
    axis.set_title(
        r"$HH\rightarrow b\bar b b\bar b$ "
        r"exact train-only $R_{HH}$ scan"
    )
    axis.legend(loc="best")
    axis.minorticks_on()

    plot_paths.extend(
        save_figure(
            figure,
            temporary_dir
            / "exact_rhh_balanced_proxy",
            formats,
            dpi=dpi,
        )
    )

    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(7.4, 5.2)
    )

    axis.step(
        x,
        purity,
        where="post",
        linewidth=1.4,
    )

    axis.axvspan(
        stable_min,
        stable_max,
        alpha=0.15,
    )

    axis.axvline(
        34.0,
        linestyle="--",
        linewidth=1.2,
    )

    axis.set_xlabel(
        r"Observed $R_{HH}$ transition"
    )
    axis.set_ylabel(
        r"$\epsilon_S^{\mathrm{bal}}/"
        r"\epsilon_B^{\mathrm{fam}}$"
    )
    axis.set_title(
        r"$HH\rightarrow b\bar b b\bar b$ "
        r"exact train-only purity proxy"
    )
    axis.minorticks_on()

    plot_paths.extend(
        save_figure(
            figure,
            temporary_dir
            / "exact_rhh_purity_proxy",
            formats,
            dpi=dpi,
        )
    )

    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(7.4, 5.2)
    )

    axis.step(
        x,
        signal_efficiency,
        where="post",
        linewidth=1.4,
        label=(
            r"Mode-balanced signal "
            r"$\epsilon_S^{\mathrm{bal}}$"
        ),
    )

    axis.step(
        x,
        background_efficiency,
        where="post",
        linewidth=1.4,
        label=(
            r"Family-balanced background "
            r"$\epsilon_B^{\mathrm{fam}}$"
        ),
    )

    axis.axvspan(
        stable_min,
        stable_max,
        alpha=0.15,
    )

    axis.set_xlabel(
        r"Observed $R_{HH}$ transition"
    )
    axis.set_ylabel("Efficiency")
    axis.set_title(
        r"$HH\rightarrow b\bar b b\bar b$ "
        r"exact train-only efficiencies"
    )
    axis.legend(loc="best")
    axis.minorticks_on()

    plot_paths.extend(
        save_figure(
            figure,
            temporary_dir
            / "exact_rhh_efficiencies",
            formats,
            dpi=dpi,
        )
    )

    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(7.2, 5.4)
    )

    color_norm = Normalize(
        vmin=float(np.min(x)),
        vmax=float(np.max(x)),
    )

    scatter = axis.scatter(
        background_efficiency,
        signal_efficiency,
        c=x,
        cmap="viridis",
        norm=color_norm,
        s=30,
        alpha=0.8,
    )

    axis.scatter(
        [
            float(
                best_row[
                    "family_balanced_background_efficiency"
                ]
            )
        ],
        [
            float(
                best_row[
                    "balanced_signal_efficiency"
                ]
            )
        ],
        marker="o",
        s=100,
        facecolors="none",
        edgecolors="black",
        linewidths=1.4,
        label="Exact proxy maximum",
    )

    colorbar = figure.colorbar(
        ScalarMappable(
            norm=color_norm,
            cmap="viridis",
        ),
        ax=axis,
        pad=0.02,
    )

    colorbar.set_label(
        r"Observed $R_{HH}$ transition"
    )

    axis.set_xlabel(
        r"Family-balanced background efficiency "
        r"$\epsilon_B^{\mathrm{fam}}$"
    )
    axis.set_ylabel(
        r"Mode-balanced signal efficiency "
        r"$\epsilon_S^{\mathrm{bal}}$"
    )
    axis.set_title(
        r"$HH\rightarrow b\bar b b\bar b$ "
        r"exact train-only Pareto trajectory"
    )
    axis.legend(loc="best")
    axis.minorticks_on()

    plot_paths.extend(
        save_figure(
            figure,
            temporary_dir
            / "exact_rhh_pareto",
            formats,
            dpi=dpi,
        )
    )

    plt.close(figure)

    require(
        len(plot_paths)
        == 4 * len(formats),
        "unexpected exact-scan plot count",
    )

    summary = {
        "schema_version": 1,
        "status":
            "hh4b_train_only_exact_rhh_scan_pass",
        "analysis_name":
            config["analysis_name"],
        "source_cache":
            str(cache_path),
        "source_cache_sha256":
            sha256_file(cache_path),
        "source_fine_summary":
            str(fine_summary_path),
        "source_fine_summary_sha256":
            sha256_file(fine_summary_path),
        "source_fine_plateau":
            str(fine_plateau_path),
        "source_fine_plateau_sha256":
            sha256_file(fine_plateau_path),
        "configuration":
            str(config_path),
        "configuration_sha256":
            sha256_file(config_path),
        "fixed_jet_pt_min_GeV":
            acceptance["jet_pt_min_GeV"],
        "fixed_jet_abs_eta_max":
            acceptance["jet_abs_eta_max"],
        "scan_window": {
            "rhh_min": window_min,
            "rhh_max": window_max,
        },
        "observed_transition_points":
            len(exact_rows),
        "eligible_transition_points":
            len(eligible_rows),
        "pareto_frontier_points":
            sum(
                bool(row["pareto_frontier"])
                for row in exact_rows
            ),
        "ranking_metric":
            ranking_metric,
        "stable_fraction_of_best":
            stable_fraction,
        "stable_component_points":
            len(stable_rows),
        "stable_component_transition_min":
            stable_min,
        "stable_component_transition_max":
            stable_max,
        "exact_best_point":
            best_row,
        "stable_high_purity_point":
            high_purity_row,
        "stable_high_efficiency_point":
            high_efficiency_row,
        "rounded_34_point":
            rounded_34,
        "exact_proxy_change_from_rounded_34_percent":
            exact_change_from_rounded_34,
        "best_at_scan_window_boundary":
            best_at_window_boundary,
        "shortlist_points":
            len(shortlist_rows),
        "plots": [
            str(output_dir / path.name)
            for path in plot_paths
        ],
        "plots_produced":
            True,
        "physical_background_normalization_frozen":
            False,
        "physical_significance_authorized":
            False,
        "physical_significance_computed":
            False,
        "validation_rows_evaluated":
            0,
        "test_members_considered":
            0,
        "test_candidate_files_opened":
            0,
        "test_candidate_rows_read":
            0,
        "next_gate": (
            "expand_exact_rhh_scan_window"
            if best_at_window_boundary
            else
            "freeze_train_only_efficiency_purity_shortlist"
        ),
    }

    summary_path = (
        temporary_dir
        / "exact_rhh_scan_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    checksum_paths = sorted(
        path
        for path in temporary_dir.iterdir()
        if path.is_file()
        and path.name != "SHA256SUMS"
    )

    (
        temporary_dir
        / "SHA256SUMS"
    ).write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in checksum_paths
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_dir.rename(output_dir)

    print(
        "HH4B_TRAIN_ONLY_EXACT_RHH_SCAN_PASS"
    )
    print(
        "HH4B_OBSERVED_TRANSITION_POINTS="
        f"{len(exact_rows)}"
    )
    print(
        "HH4B_EXACT_BEST_POINT="
        + json.dumps(
            {
                "point_id":
                    best_row["point_id"],
                "rhh_transition_value":
                    best_row[
                        "rhh_transition_value"
                    ],
                "rhh_threshold_value":
                    best_row[
                        "rhh_threshold_value"
                    ],
                ranking_metric:
                    best_row[ranking_metric],
            },
            sort_keys=True,
        )
    )
    print(
        "HH4B_STABLE_COMPONENT="
        + json.dumps({
            "points": len(stable_rows),
            "transition_min": stable_min,
            "transition_max": stable_max,
        })
    )
    print(
        "HH4B_EXACT_PROXY_CHANGE_FROM_"
        "ROUNDED34_PERCENT="
        f"{exact_change_from_rounded_34}"
    )
    print(
        "HH4B_BEST_AT_SCAN_WINDOW_BOUNDARY="
        f"{best_at_window_boundary}"
    )
    print("HH4B_VALIDATION_ROWS_EVALUATED=0")
    print("HH4B_TEST_CANDIDATE_FILES_OPENED=0")
    print(
        "HH4B_PHYSICAL_SIGNIFICANCE_COMPUTED=False"
    )
    print("HH4B_PLOTS_PRODUCED=True")
    print(
        "NEXT_GATE="
        + summary["next_gate"]
    )
    print(f"summary_json={output_dir / 'exact_rhh_scan_summary.json'}")


if __name__ == "__main__":
    main()
