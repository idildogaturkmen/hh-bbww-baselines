#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from hh4b_plot_style import (
    apply_hh4b_paper_style,
    save_figure,
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(
    path: Path,
) -> str:
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


def float_array(
    table: Any,
    column: str,
) -> np.ndarray:
    arrow_column = table[column]

    require(
        arrow_column.null_count == 0,
        f"{column} contains null values",
    )

    values = np.asarray(
        arrow_column.combine_chunks().to_numpy(
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
    arrow_column = table[column]

    require(
        arrow_column.null_count == 0,
        f"{column} contains null values",
    )

    return np.asarray(
        arrow_column.to_pylist(),
        dtype=object,
    )


def threshold_token(
    value: float,
) -> str:
    return (
        f"{value:.3f}"
        .rstrip("0")
        .rstrip(".")
        .replace(".", "p")
    )


def point_id(
    pt_min: float,
    eta_max: float,
    rhh_max: float,
) -> str:
    return (
        f"pt{threshold_token(pt_min)}"
        f"_eta{threshold_token(eta_max)}"
        f"_rhh{threshold_token(rhh_max)}"
    )


def is_close(
    left: float,
    right: float,
) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )


def add_pareto_flags(
    rows: list[dict[str, Any]],
) -> None:
    signal = np.asarray([
        float(row["balanced_signal_efficiency"])
        for row in rows
    ])

    background = np.asarray([
        float(row["family_balanced_background_efficiency"])
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


def make_heatmaps(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    pt_values: list[float],
    eta_values: list[float],
    rhh_values: list[float],
    best_point: dict[str, Any],
    reference_point: dict[str, Any],
    formats: list[str],
    dpi: int,
) -> list[Path]:
    written: list[Path] = []

    for eta_max in eta_values:
        matrix = np.full(
            (
                len(pt_values),
                len(rhh_values),
            ),
            np.nan,
            dtype=np.float64,
        )

        lookup = {
            (
                float(row["jet_pt_min_GeV"]),
                float(row["rhh_sr_max"]),
            ):
            float(
                row[
                    "balanced_proxy_s_over_sqrt_b"
                ]
            )
            for row in rows
            if is_close(
                float(row["jet_abs_eta_max"]),
                eta_max,
            )
        }

        for pt_index, pt_min in enumerate(pt_values):
            for rhh_index, rhh_max in enumerate(rhh_values):
                matrix[pt_index, rhh_index] = lookup[
                    (pt_min, rhh_max)
                ]

        figure, axis = plt.subplots(
            figsize=(7.2, 5.4)
        )

        mesh = axis.pcolormesh(
            rhh_values,
            pt_values,
            matrix,
            shading="nearest",
        )

        colorbar = figure.colorbar(
            mesh,
            ax=axis,
            pad=0.02,
        )

        colorbar.set_label(
            r"$\epsilon_{S}^{\mathrm{bal}}/"
            r"\sqrt{\epsilon_{B}^{\mathrm{fam}}}$"
        )

        axis.set_xlabel(
            r"$R_{HH}^{\mathrm{SR,max}}$"
        )

        axis.set_ylabel(
            r"$p_{\mathrm{T}}^{\min}$ [GeV]"
        )

        axis.set_title(
            r"$HH\rightarrow b\bar b b\bar b$ "
            rf"train-only coarse scan, "
            rf"$|\eta|^{{\max}}={eta_max:g}$"
        )

        if is_close(
            eta_max,
            float(reference_point["jet_abs_eta_max"]),
        ):
            axis.scatter(
                [
                    float(
                        reference_point[
                            "rhh_sr_max"
                        ]
                    )
                ],
                [
                    float(
                        reference_point[
                            "jet_pt_min_GeV"
                        ]
                    )
                ],
                marker="*",
                s=150,
                facecolors="none",
                edgecolors="black",
                linewidths=1.4,
                label="Reference",
            )

        if is_close(
            eta_max,
            float(best_point["jet_abs_eta_max"]),
        ):
            axis.scatter(
                [
                    float(
                        best_point["rhh_sr_max"]
                    )
                ],
                [
                    float(
                        best_point[
                            "jet_pt_min_GeV"
                        ]
                    )
                ],
                marker="o",
                s=90,
                facecolors="none",
                edgecolors="black",
                linewidths=1.4,
                label="Best eligible proxy",
            )

        handles, labels = axis.get_legend_handles_labels()

        if handles:
            axis.legend(
                handles,
                labels,
                loc="best",
            )

        axis.minorticks_on()

        written.extend(
            save_figure(
                figure,
                output_dir
                / (
                    "coarse_proxy_heatmap_"
                    f"eta{threshold_token(eta_max)}"
                ),
                formats,
                dpi=dpi,
            )
        )

        plt.close(figure)

    return written


def make_pareto_plot(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    eta_values: list[float],
    pt_values: list[float],
    best_point: dict[str, Any],
    reference_row: dict[str, Any],
    formats: list[str],
    dpi: int,
) -> list[Path]:
    figure, axis = plt.subplots(
        figsize=(7.2, 5.4)
    )

    normalization = Normalize(
        vmin=min(pt_values),
        vmax=max(pt_values),
    )

    color_map = plt.get_cmap("viridis")
    markers = ["o", "s", "^", "D"]

    for eta_index, eta_max in enumerate(eta_values):
        selected = [
            row
            for row in rows
            if is_close(
                float(row["jet_abs_eta_max"]),
                eta_max,
            )
        ]

        axis.scatter(
            [
                float(
                    row[
                        "family_balanced_background_efficiency"
                    ]
                )
                for row in selected
            ],
            [
                float(
                    row[
                        "balanced_signal_efficiency"
                    ]
                )
                for row in selected
            ],
            c=[
                float(row["jet_pt_min_GeV"])
                for row in selected
            ],
            cmap=color_map,
            norm=normalization,
            marker=markers[
                eta_index % len(markers)
            ],
            s=34,
            alpha=0.75,
            label=(
                rf"$|\eta|^{{\max}}="
                rf"{eta_max:g}$"
            ),
        )

    pareto_rows = [
        row
        for row in rows
        if bool(row["pareto_frontier"])
    ]

    pareto_rows.sort(
        key=lambda row: float(
            row[
                "family_balanced_background_efficiency"
            ]
        )
    )

    axis.plot(
        [
            float(
                row[
                    "family_balanced_background_efficiency"
                ]
            )
            for row in pareto_rows
        ],
        [
            float(
                row[
                    "balanced_signal_efficiency"
                ]
            )
            for row in pareto_rows
        ],
        linewidth=1.3,
        label="Pareto frontier",
    )

    axis.scatter(
        [
            float(
                reference_row[
                    "family_balanced_background_efficiency"
                ]
            )
        ],
        [
            float(
                reference_row[
                    "balanced_signal_efficiency"
                ]
            )
        ],
        marker="*",
        s=160,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Reference",
    )

    axis.scatter(
        [
            float(
                best_point[
                    "family_balanced_background_efficiency"
                ]
            )
        ],
        [
            float(
                best_point[
                    "balanced_signal_efficiency"
                ]
            )
        ],
        marker="o",
        s=100,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Best eligible proxy",
    )

    colorbar = figure.colorbar(
        ScalarMappable(
            norm=normalization,
            cmap=color_map,
        ),
        ax=axis,
        pad=0.02,
    )

    colorbar.set_label(
        r"$p_{\mathrm{T}}^{\min}$ [GeV]"
    )

    axis.set_xlabel(
        r"Family-balanced background efficiency "
        r"$\epsilon_{B}^{\mathrm{fam}}$"
    )

    axis.set_ylabel(
        r"Mode-balanced signal efficiency "
        r"$\epsilon_{S}^{\mathrm{bal}}$"
    )

    axis.set_title(
        r"$HH\rightarrow b\bar b b\bar b$ "
        r"train-only coarse-scan Pareto plane"
    )

    axis.legend(
        loc="best",
    )

    axis.minorticks_on()

    written = save_figure(
        figure,
        output_dir / "coarse_scan_pareto",
        formats,
        dpi=dpi,
    )

    plt.close(figure)
    return written


def make_profile_plot(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    parameter: str,
    x_label: str,
    output_name: str,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    values = sorted({
        float(row[parameter])
        for row in rows
    })

    best_values = []

    for value in values:
        matching = [
            row
            for row in rows
            if is_close(
                float(row[parameter]),
                value,
            )
            and bool(row["eligible_for_ranking"])
        ]

        require(
            bool(matching),
            (
                f"no eligible points for "
                f"{parameter}={value}"
            ),
        )

        best_values.append(
            max(
                float(
                    row[
                        "balanced_proxy_s_over_sqrt_b"
                    ]
                )
                for row in matching
            )
        )

    figure, axis = plt.subplots(
        figsize=(7.2, 5.0)
    )

    axis.plot(
        values,
        best_values,
        marker="o",
        linewidth=1.5,
    )

    axis.set_xlabel(x_label)

    axis.set_ylabel(
        r"Best "
        r"$\epsilon_{S}^{\mathrm{bal}}/"
        r"\sqrt{\epsilon_{B}^{\mathrm{fam}}}$"
    )

    axis.set_title(
        r"$HH\rightarrow b\bar b b\bar b$ "
        r"train-only coarse-scan profile"
    )

    axis.minorticks_on()

    written = save_figure(
        figure,
        output_dir / output_name,
        formats,
        dpi=dpi,
    )

    plt.close(figure)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a train-only coarse HH4b cut "
            "parameter scan using a frozen cache."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="JSON scan configuration.",
    )

    arguments = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo = script_path.parents[2]

    config_path = arguments.config

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
        == "coarse_train_only",
        "configuration is not a coarse train-only scan",
    )

    require(
        config.get(
            "physical_background_normalization_frozen"
        )
        is False,
        "physical background normalization state changed",
    )

    require(
        config.get(
            "physical_significance_authorized"
        )
        is False,
        "physical significance must remain disabled",
    )

    require(
        config.get("test_access_allowed")
        is False,
        "test access must remain disabled",
    )

    cache_path = Path(
        config["cache_path"]
    ).resolve()

    require(
        cache_path.is_file(),
        f"missing scan cache: {cache_path}",
    )

    observed_cache_sha256 = sha256_file(
        cache_path
    )

    expected_cache_sha256 = str(
        config["cache_sha256"]
    ).lower()

    require(
        observed_cache_sha256
        == expected_cache_sha256,
        (
            "scan cache checksum mismatch: "
            f"{observed_cache_sha256}"
        ),
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

    temporary_output_dir = Path(
        str(output_dir) + "_incomplete"
    )

    require(
        not temporary_output_dir.exists(),
        (
            "temporary output directory exists: "
            f"{temporary_output_dir}"
        ),
    )

    split = str(
        config["dataset_split"]
    )

    require(
        split == "train",
        "coarse optimization must use train only",
    )

    selection = config["selection"]

    required_columns = [
        "sample_class",
        "process_or_mode",
        "dataset_split",
        selection["jet_pt_variable"],
        selection["jet_eta_variable"],
        selection["rhh_variable"],
    ]

    table = pq.read_table(
        cache_path,
        columns=required_columns,
        filters=[
            ("dataset_split", "=", split)
        ],
        use_threads=True,
    )

    expected = config["expected_rows"]

    require(
        table.num_rows
        == int(expected["total"]),
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

    dataset_split = string_array(
        table,
        "dataset_split",
    )

    require(
        set(dataset_split.tolist())
        == {"train"},
        "non-train rows entered the coarse scan",
    )

    min_jet_pt = float_array(
        table,
        selection["jet_pt_variable"],
    )

    max_abs_eta = float_array(
        table,
        selection["jet_eta_variable"],
    )

    rhh = float_array(
        table,
        selection["rhh_variable"],
    )

    signal_mask = sample_class == "signal"
    background_mask = sample_class == "background"

    require(
        int(np.count_nonzero(signal_mask))
        == int(expected["signal"]),
        "signal train-row count mismatch",
    )

    require(
        int(np.count_nonzero(background_mask))
        == int(expected["background"]),
        "background train-row count mismatch",
    )

    signal_modes = list(
        config["signal_modes"]
    )

    require(
        signal_modes
        == ["ggf_hh4b", "vbf_hh4b"],
        f"unexpected signal modes: {signal_modes}",
    )

    mode_masks = {
        mode:
            signal_mask & (process == mode)
        for mode in signal_modes
    }

    for mode in signal_modes:
        require(
            int(
                np.count_nonzero(
                    mode_masks[mode]
                )
            )
            == int(expected[mode]),
            f"{mode} train-row count mismatch",
        )

    background_families = sorted(
        set(
            process[
                background_mask
            ].tolist()
        )
    )

    require(
        bool(background_families),
        "no background families found",
    )

    family_masks = {
        family:
            background_mask
            & (process == family)
        for family in background_families
    }

    mode_denominators = {
        mode:
            int(
                np.count_nonzero(
                    mode_masks[mode]
                )
            )
        for mode in signal_modes
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

    require(
        all(
            count > 0
            for count
            in family_denominators.values()
        ),
        "a cached background family has zero train rows",
    )

    thresholds = config["thresholds"]

    pt_values = [
        float(value)
        for value
        in thresholds["jet_pt_min_GeV"]
    ]

    eta_values = [
        float(value)
        for value
        in thresholds["jet_abs_eta_max"]
    ]

    rhh_values = [
        float(value)
        for value
        in thresholds["rhh_sr_max"]
    ]

    require(
        len(pt_values) == 16,
        "expected 16 coarse pT thresholds",
    )

    require(
        eta_values == [2.4, 2.5],
        "unexpected eta scan",
    )

    require(
        len(rhh_values) == 16,
        "expected 16 coarse RHH thresholds",
    )

    require(
        pt_values == sorted(set(pt_values)),
        "pT thresholds are not unique and sorted",
    )

    require(
        rhh_values == sorted(set(rhh_values)),
        "RHH thresholds are not unique and sorted",
    )

    scan_rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []

    ranking = config["ranking"]

    minimum_background = int(
        ranking["minimum_selected_background"]
    )

    minimum_signal_mode = int(
        ranking[
            "minimum_selected_events_per_signal_mode"
        ]
    )

    reference = config["reference_point"]

    for pt_min, eta_max, rhh_max in itertools.product(
        pt_values,
        eta_values,
        rhh_values,
    ):
        identifier = point_id(
            pt_min,
            eta_max,
            rhh_max,
        )

        selected = (
            (min_jet_pt > pt_min)
            & (max_abs_eta < eta_max)
            & (rhh < rhh_max)
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

        signal_efficiency = (
            selected_signal
            / int(expected["signal"])
        )

        background_efficiency = (
            selected_background
            / int(expected["background"])
        )

        mode_efficiencies: list[float] = []
        mode_selected_counts: dict[str, int] = {}

        for mode in signal_modes:
            selected_mode = int(
                np.count_nonzero(
                    selected
                    & mode_masks[mode]
                )
            )

            efficiency = (
                selected_mode
                / mode_denominators[mode]
            )

            mode_selected_counts[mode] = (
                selected_mode
            )

            mode_efficiencies.append(
                efficiency
            )

            mode_rows.append({
                "point_id":
                    identifier,
                "jet_pt_min_GeV":
                    pt_min,
                "jet_abs_eta_max":
                    eta_max,
                "rhh_sr_max":
                    rhh_max,
                "signal_mode":
                    mode,
                "selected_rows":
                    selected_mode,
                "total_rows":
                    mode_denominators[mode],
                "efficiency":
                    efficiency,
            })

        family_efficiencies: list[float] = []

        for family in background_families:
            selected_family = int(
                np.count_nonzero(
                    selected
                    & family_masks[family]
                )
            )

            efficiency = (
                selected_family
                / family_denominators[family]
            )

            family_efficiencies.append(
                efficiency
            )

            family_rows.append({
                "point_id":
                    identifier,
                "jet_pt_min_GeV":
                    pt_min,
                "jet_abs_eta_max":
                    eta_max,
                "rhh_sr_max":
                    rhh_max,
                "background_family":
                    family,
                "selected_rows":
                    selected_family,
                "total_rows":
                    family_denominators[family],
                "efficiency":
                    efficiency,
            })

        balanced_signal_efficiency = (
            float(
                np.mean(mode_efficiencies)
            )
        )

        family_balanced_background_efficiency = (
            float(
                np.mean(family_efficiencies)
            )
        )

        family_efficiency_std = float(
            np.std(
                family_efficiencies,
                ddof=0,
            )
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
            if (
                family_balanced_background_efficiency
                > 0.0
            )
            else math.inf
        )

        balanced_s_over_b = (
            balanced_signal_efficiency
            / family_balanced_background_efficiency
            if (
                family_balanced_background_efficiency
                > 0.0
            )
            else math.inf
        )

        eligible = (
            selected_background
            >= minimum_background
            and all(
                count >= minimum_signal_mode
                for count
                in mode_selected_counts.values()
            )
            and math.isfinite(balanced_proxy)
        )

        is_reference = all([
            is_close(
                pt_min,
                float(
                    reference[
                        "jet_pt_min_GeV"
                    ]
                ),
            ),
            is_close(
                eta_max,
                float(
                    reference[
                        "jet_abs_eta_max"
                    ]
                ),
            ),
            is_close(
                rhh_max,
                float(
                    reference[
                        "rhh_sr_max"
                    ]
                ),
            ),
        ])

        scan_rows.append({
            "point_id":
                identifier,
            "jet_pt_min_GeV":
                pt_min,
            "jet_abs_eta_max":
                eta_max,
            "rhh_sr_max":
                rhh_max,
            "selected_signal":
                selected_signal,
            "selected_background":
                selected_background,
            "unweighted_signal_efficiency":
                signal_efficiency,
            "unweighted_background_efficiency":
                background_efficiency,
            "ggf_signal_efficiency":
                mode_efficiencies[0],
            "vbf_signal_efficiency":
                mode_efficiencies[1],
            "balanced_signal_efficiency":
                balanced_signal_efficiency,
            "family_balanced_background_efficiency":
                family_balanced_background_efficiency,
            "background_family_efficiency_std":
                family_efficiency_std,
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
            "reference_point":
                is_reference,
        })

    require(
        len(scan_rows) == 512,
        (
            f"expected 512 coarse points, "
            f"produced {len(scan_rows)}"
        ),
    )

    add_pareto_flags(scan_rows)

    ranking_metric = str(
        ranking["metric"]
    )

    eligible_rows = [
        row
        for row in scan_rows
        if bool(
            row["eligible_for_ranking"]
        )
    ]

    require(
        bool(eligible_rows),
        "no points are eligible for ranking",
    )

    best_point = max(
        eligible_rows,
        key=lambda row: float(
            row[ranking_metric]
        ),
    )

    best_metric = float(
        best_point[ranking_metric]
    )

    require(
        math.isfinite(best_metric),
        "best ranking metric is nonfinite",
    )

    for row in scan_rows:
        row["fraction_of_best_proxy"] = (
            float(row[ranking_metric])
            / best_metric
            if math.isfinite(
                float(row[ranking_metric])
            )
            else ""
        )

        row["near_optimal_plateau"] = bool(
            row["eligible_for_ranking"]
            and float(
                row["fraction_of_best_proxy"]
            )
            >= float(
                ranking[
                    "plateau_fraction_of_best"
                ]
            )
        )

    reference_rows = [
        row
        for row in scan_rows
        if bool(row["reference_point"])
    ]

    require(
        len(reference_rows) == 1,
        (
            "expected exactly one reference point, "
            f"found {len(reference_rows)}"
        ),
    )

    reference_row = reference_rows[0]

    plateau_rows = [
        row
        for row in scan_rows
        if bool(row["near_optimal_plateau"])
    ]

    require(
        bool(plateau_rows),
        "near-optimal plateau is empty",
    )

    shortlist_candidates = [
        row
        for row in plateau_rows
        if bool(row["pareto_frontier"])
    ]

    if not shortlist_candidates:
        shortlist_candidates = plateau_rows

    shortlist_candidates.sort(
        key=lambda row: (
            -float(row[ranking_metric]),
            -float(
                row[
                    "balanced_signal_efficiency"
                ]
            ),
            float(
                row[
                    "family_balanced_background_efficiency"
                ]
            ),
        )
    )

    shortlist_rows = shortlist_candidates[
        : int(ranking["shortlist_size"])
    ]

    temporary_output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    table_fields = [
        "point_id",
        "jet_pt_min_GeV",
        "jet_abs_eta_max",
        "rhh_sr_max",
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
        "near_optimal_plateau",
        "reference_point",
    ]

    write_tsv(
        temporary_output_dir
        / "coarse_scan_points.tsv",
        scan_rows,
        table_fields,
    )

    write_tsv(
        temporary_output_dir
        / "coarse_scan_signal_mode_counts.tsv",
        mode_rows,
        [
            "point_id",
            "jet_pt_min_GeV",
            "jet_abs_eta_max",
            "rhh_sr_max",
            "signal_mode",
            "selected_rows",
            "total_rows",
            "efficiency",
        ],
    )

    write_tsv(
        temporary_output_dir
        / "coarse_scan_background_family_counts.tsv",
        family_rows,
        [
            "point_id",
            "jet_pt_min_GeV",
            "jet_abs_eta_max",
            "rhh_sr_max",
            "background_family",
            "selected_rows",
            "total_rows",
            "efficiency",
        ],
    )

    write_tsv(
        temporary_output_dir
        / "coarse_scan_shortlist.tsv",
        shortlist_rows,
        table_fields,
    )

    shutil.copy2(
        config_path,
        temporary_output_dir
        / "scan_config_snapshot.json",
    )

    apply_hh4b_paper_style()

    plot_formats = [
        str(value)
        for value
        in config["plotting"]["formats"]
    ]

    plot_dpi = int(
        config["plotting"]["dpi"]
    )

    written_plots: list[Path] = []

    written_plots.extend(
        make_heatmaps(
            scan_rows,
            temporary_output_dir,
            pt_values=pt_values,
            eta_values=eta_values,
            rhh_values=rhh_values,
            best_point=best_point,
            reference_point=reference,
            formats=plot_formats,
            dpi=plot_dpi,
        )
    )

    written_plots.extend(
        make_pareto_plot(
            scan_rows,
            temporary_output_dir,
            eta_values=eta_values,
            pt_values=pt_values,
            best_point=best_point,
            reference_row=reference_row,
            formats=plot_formats,
            dpi=plot_dpi,
        )
    )

    written_plots.extend(
        make_profile_plot(
            scan_rows,
            temporary_output_dir,
            parameter="jet_pt_min_GeV",
            x_label=(
                r"$p_{\mathrm{T}}^{\min}$ [GeV]"
            ),
            output_name=(
                "coarse_best_proxy_vs_ptmin"
            ),
            formats=plot_formats,
            dpi=plot_dpi,
        )
    )

    written_plots.extend(
        make_profile_plot(
            scan_rows,
            temporary_output_dir,
            parameter="rhh_sr_max",
            x_label=(
                r"$R_{HH}^{\mathrm{SR,max}}$"
            ),
            output_name=(
                "coarse_best_proxy_vs_rhh"
            ),
            formats=plot_formats,
            dpi=plot_dpi,
        )
    )

    expected_plot_count = (
        len(eta_values) + 3
    ) * len(plot_formats)

    require(
        len(written_plots)
        == expected_plot_count,
        (
            f"expected {expected_plot_count} plot files, "
            f"wrote {len(written_plots)}"
        ),
    )

    summary = {
        "schema_version": 1,
        "status":
            "hh4b_train_only_coarse_cut_scan_pass",
        "analysis_name":
            config["analysis_name"],
        "source_cache":
            str(cache_path),
        "source_cache_sha256":
            observed_cache_sha256,
        "configuration":
            str(config_path),
        "configuration_sha256":
            sha256_file(config_path),
        "dataset_split":
            split,
        "train_rows_evaluated":
            table.num_rows,
        "signal_train_rows":
            int(expected["signal"]),
        "background_train_rows":
            int(expected["background"]),
        "background_families":
            background_families,
        "background_family_count":
            len(background_families),
        "grid_points":
            len(scan_rows),
        "eligible_grid_points":
            len(eligible_rows),
        "pareto_frontier_points":
            sum(
                bool(row["pareto_frontier"])
                for row in scan_rows
            ),
        "near_optimal_plateau_points":
            len(plateau_rows),
        "shortlist_points":
            len(shortlist_rows),
        "ranking_metric":
            ranking_metric,
        "best_point":
            {
                key: best_point[key]
                for key in table_fields
            },
        "reference_point":
            {
                key: reference_row[key]
                for key in table_fields
            },
        "plots": [
            str(
                output_dir / path.name
            )
            for path in written_plots
        ],
        "plots_produced":
            True,
        "cms_branding_used":
            False,
        "latex_math_labels_used":
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
        "next_gate":
            "define_and_run_fine_cut_scan_from_train_plateau",
    }

    summary_path = (
        temporary_output_dir
        / "coarse_scan_summary.json"
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
        for path in temporary_output_dir.iterdir()
        if path.is_file()
        and path.name != "SHA256SUMS"
    )

    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in checksum_paths
    ]

    (
        temporary_output_dir
        / "SHA256SUMS"
    ).write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )

    temporary_output_dir.rename(
        output_dir
    )

    print(
        "HH4B_TRAIN_ONLY_COARSE_CUT_SCAN_PASS"
    )
    print("HH4B_GRID_POINTS=512")
    print(
        "HH4B_ELIGIBLE_GRID_POINTS="
        f"{len(eligible_rows)}"
    )
    print(
        "HH4B_PARETO_FRONTIER_POINTS="
        f"{summary['pareto_frontier_points']}"
    )
    print(
        "HH4B_NEAR_OPTIMAL_PLATEAU_POINTS="
        f"{len(plateau_rows)}"
    )
    print(
        "HH4B_BEST_POINT="
        + json.dumps(
            {
                "point_id":
                    best_point["point_id"],
                "jet_pt_min_GeV":
                    best_point[
                        "jet_pt_min_GeV"
                    ],
                "jet_abs_eta_max":
                    best_point[
                        "jet_abs_eta_max"
                    ],
                "rhh_sr_max":
                    best_point[
                        "rhh_sr_max"
                    ],
                ranking_metric:
                    best_point[
                        ranking_metric
                    ],
            },
            sort_keys=True,
        )
    )
    print(
        "HH4B_REFERENCE_PROXY="
        f"{reference_row[ranking_metric]}"
    )
    print("HH4B_VALIDATION_ROWS_EVALUATED=0")
    print("HH4B_TEST_CANDIDATE_FILES_OPENED=0")
    print(
        "HH4B_PHYSICAL_SIGNIFICANCE_COMPUTED=False"
    )
    print("HH4B_PLOTS_PRODUCED=True")
    print(
        "NEXT_GATE="
        "define_and_run_fine_cut_scan_from_train_plateau"
    )
    print(
        "summary_json="
        f"{output_dir / 'coarse_scan_summary.json'}"
    )


if __name__ == "__main__":
    main()
