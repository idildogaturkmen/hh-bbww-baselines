#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
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


def evaluate_point(
    *,
    pt_min: float,
    eta_max: float,
    rhh_max: float,
    min_jet_pt: np.ndarray,
    max_abs_eta: np.ndarray,
    rhh: np.ndarray,
    signal_mask: np.ndarray,
    background_mask: np.ndarray,
    signal_modes: list[str],
    mode_masks: dict[str, np.ndarray],
    mode_denominators: dict[str, int],
    background_families: list[str],
    family_masks: dict[str, np.ndarray],
    family_denominators: dict[str, int],
    total_signal: int,
    total_background: int,
    minimum_background: int,
    minimum_signal_mode: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
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

    mode_rows: list[dict[str, Any]] = []
    mode_efficiencies: list[float] = []
    mode_selected_counts: dict[str, int] = {}

    for mode in signal_modes:
        selected_mode = int(
            np.count_nonzero(
                selected & mode_masks[mode]
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

    family_rows: list[dict[str, Any]] = []
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

    signal_efficiency = (
        selected_signal
        / total_signal
    )

    background_efficiency = (
        selected_background
        / total_background
    )

    balanced_signal_efficiency = float(
        np.mean(mode_efficiencies)
    )

    family_balanced_background_efficiency = float(
        np.mean(family_efficiencies)
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

    row = {
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
    }

    return row, mode_rows, family_rows


def add_pareto_flags(
    rows: list[dict[str, Any]],
) -> None:
    signal = np.asarray([
        float(
            row[
                "balanced_signal_efficiency"
            ]
        )
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
                | (
                    background
                    < background[index]
                )
            )
        )

        row["pareto_frontier"] = bool(
            not dominated
        )


def make_heatmaps(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    prefix: str,
    display_stage: str,
    pt_values: list[float],
    eta_values: list[float],
    rhh_values: list[float],
    best_point: dict[str, Any],
    anchor_point: dict[str, Any],
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

        for pt_index, pt_min in enumerate(
            pt_values
        ):
            for rhh_index, rhh_max in enumerate(
                rhh_values
            ):
                matrix[
                    pt_index,
                    rhh_index,
                ] = lookup[
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
            f"{display_stage}, "
            rf"$|\eta|^{{\max}}={eta_max:g}$"
        )

        if is_close(
            eta_max,
            float(
                anchor_point[
                    "jet_abs_eta_max"
                ]
            ),
        ):
            axis.scatter(
                [
                    float(
                        anchor_point[
                            "rhh_sr_max"
                        ]
                    )
                ],
                [
                    float(
                        anchor_point[
                            "jet_pt_min_GeV"
                        ]
                    )
                ],
                marker="*",
                s=150,
                facecolors="none",
                edgecolors="black",
                linewidths=1.4,
                label="Coarse optimum",
            )

        if is_close(
            eta_max,
            float(
                best_point[
                    "jet_abs_eta_max"
                ]
            ),
        ):
            axis.scatter(
                [
                    float(
                        best_point[
                            "rhh_sr_max"
                        ]
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
                label="Fine proxy optimum",
            )

        handles, labels = (
            axis.get_legend_handles_labels()
        )

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
                    f"{prefix}_proxy_heatmap_"
                    f"eta{threshold_token(eta_max)}"
                ),
                formats,
                dpi=dpi,
            )
        )

        plt.close(figure)

    return written


def make_profile_plot(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    prefix: str,
    display_stage: str,
    parameter: str,
    xlabel: str,
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
            and bool(
                row["eligible_for_ranking"]
            )
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

    axis.set_xlabel(xlabel)

    axis.set_ylabel(
        r"Best "
        r"$\epsilon_{S}^{\mathrm{bal}}/"
        r"\sqrt{\epsilon_{B}^{\mathrm{fam}}}$"
    )

    axis.set_title(
        r"$HH\rightarrow b\bar b b\bar b$ "
        f"{display_stage} profile"
    )

    axis.minorticks_on()

    written = save_figure(
        figure,
        output_dir
        / f"{prefix}_best_proxy_vs_{parameter}",
        formats,
        dpi=dpi,
    )

    plt.close(figure)

    return written


def make_pareto_plot(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    prefix: str,
    display_stage: str,
    eta_values: list[float],
    pt_values: list[float],
    best_point: dict[str, Any],
    anchor_point: dict[str, Any],
    reference_point: dict[str, Any],
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

    for eta_index, eta_max in enumerate(
        eta_values
    ):
        selected = [
            row
            for row in rows
            if is_close(
                float(
                    row[
                        "jet_abs_eta_max"
                    ]
                ),
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
                float(
                    row[
                        "jet_pt_min_GeV"
                    ]
                )
                for row in selected
            ],
            cmap=color_map,
            norm=normalization,
            marker=markers[
                eta_index
                % len(markers)
            ],
            s=36,
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
        label="Fine-grid Pareto frontier",
    )

    for point, marker, label, size in (
        (
            reference_point,
            "*",
            "Reference",
            160,
        ),
        (
            anchor_point,
            "s",
            "Coarse optimum",
            100,
        ),
        (
            best_point,
            "o",
            "Fine proxy optimum",
            100,
        ),
    ):
        axis.scatter(
            [
                float(
                    point[
                        "family_balanced_background_efficiency"
                    ]
                )
            ],
            [
                float(
                    point[
                        "balanced_signal_efficiency"
                    ]
                )
            ],
            marker=marker,
            s=size,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label=label,
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
        f"{display_stage} Pareto plane"
    )

    axis.legend(
        loc="best",
    )

    axis.minorticks_on()

    written = save_figure(
        figure,
        output_dir
        / f"{prefix}_scan_pareto",
        formats,
        dpi=dpi,
    )

    plt.close(figure)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a configurable train-only HH4b "
            "cut-threshold grid scan."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
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
        config.get("dataset_split")
        == "train",
        "optimization must use train only",
    )

    require(
        str(config.get("scan_stage", ""))
        .endswith("_train_only"),
        "scan stage is not train-only",
    )

    require(
        config.get(
            "physical_background_normalization_frozen"
        )
        is False,
        "physical background normalization changed",
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

    require(
        observed_cache_sha256
        == str(
            config["cache_sha256"]
        ).lower(),
        "scan cache SHA-256 mismatch",
    )

    coarse_summary_path = Path(
        config["coarse_summary_path"]
    )

    if not coarse_summary_path.is_absolute():
        coarse_summary_path = (
            repo / coarse_summary_path
        ).resolve()

    require(
        coarse_summary_path.is_file(),
        (
            "missing coarse summary: "
            f"{coarse_summary_path}"
        ),
    )

    require(
        sha256_file(coarse_summary_path)
        == str(
            config["coarse_summary_sha256"]
        ).lower(),
        "coarse summary SHA-256 mismatch",
    )

    coarse_summary = json.loads(
        coarse_summary_path.read_text(
            encoding="utf-8",
        )
    )

    require(
        coarse_summary.get("status")
        == "hh4b_train_only_coarse_cut_scan_pass",
        "coarse scan did not pass",
    )

    require(
        coarse_summary.get(
            "validation_rows_evaluated"
        ) == 0,
        "coarse scan used validation",
    )

    require(
        coarse_summary.get(
            "test_candidate_files_opened"
        ) == 0,
        "coarse scan opened test candidates",
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
            (
                "dataset_split",
                "=",
                "train",
            )
        ],
        use_threads=True,
    )

    expected = config["expected_rows"]

    require(
        table.num_rows
        == int(expected["total"]),
        (
            f"expected {expected['total']} train "
            f"rows, read {table.num_rows}"
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
        "non-train rows entered scan",
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
    background_mask = (
        sample_class == "background"
    )

    total_signal = int(
        np.count_nonzero(signal_mask)
    )

    total_background = int(
        np.count_nonzero(background_mask)
    )

    require(
        total_signal
        == int(expected["signal"]),
        "signal train-row count mismatch",
    )

    require(
        total_background
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
            signal_mask
            & (process == mode)
        for mode in signal_modes
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

    for mode in signal_modes:
        require(
            mode_denominators[mode]
            == int(expected[mode]),
            f"{mode} row count mismatch",
        )

    background_families = sorted(
        set(
            process[
                background_mask
            ].tolist()
        )
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

    require(
        all(
            count > 0
            for count
            in family_denominators.values()
        ),
        "background family with zero train rows",
    )

    thresholds = config["thresholds"]

    pt_values = [
        float(value)
        for value in thresholds[
            "jet_pt_min_GeV"
        ]
    ]

    eta_values = [
        float(value)
        for value in thresholds[
            "jet_abs_eta_max"
        ]
    ]

    rhh_values = [
        float(value)
        for value in thresholds[
            "rhh_sr_max"
        ]
    ]

    require(
        pt_values == sorted(set(pt_values)),
        "pT thresholds are not unique and sorted",
    )

    require(
        eta_values == sorted(set(eta_values)),
        "eta thresholds are not unique and sorted",
    )

    require(
        rhh_values == sorted(set(rhh_values)),
        "RHH thresholds are not unique and sorted",
    )

    expected_grid_points = (
        len(pt_values)
        * len(eta_values)
        * len(rhh_values)
    )

    require(
        expected_grid_points == 783,
        (
            "expected a 783-point fine grid, "
            f"found {expected_grid_points}"
        ),
    )

    ranking = config["ranking"]

    minimum_background = int(
        ranking[
            "minimum_selected_background"
        ]
    )

    minimum_signal_mode = int(
        ranking[
            "minimum_selected_events_per_signal_mode"
        ]
    )

    scan_rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []

    for (
        pt_min,
        eta_max,
        rhh_max,
    ) in itertools.product(
        pt_values,
        eta_values,
        rhh_values,
    ):
        (
            result,
            result_modes,
            result_families,
        ) = evaluate_point(
            pt_min=pt_min,
            eta_max=eta_max,
            rhh_max=rhh_max,
            min_jet_pt=min_jet_pt,
            max_abs_eta=max_abs_eta,
            rhh=rhh,
            signal_mask=signal_mask,
            background_mask=background_mask,
            signal_modes=signal_modes,
            mode_masks=mode_masks,
            mode_denominators=mode_denominators,
            background_families=background_families,
            family_masks=family_masks,
            family_denominators=family_denominators,
            total_signal=total_signal,
            total_background=total_background,
            minimum_background=minimum_background,
            minimum_signal_mode=minimum_signal_mode,
        )

        scan_rows.append(result)
        mode_rows.extend(result_modes)
        family_rows.extend(result_families)

    require(
        len(scan_rows)
        == expected_grid_points,
        "scan point total mismatch",
    )

    add_pareto_flags(scan_rows)

    ranking_metric = str(
        ranking["metric"]
    )

    eligible_rows = [
        row
        for row in scan_rows
        if bool(row["eligible_for_ranking"])
    ]

    require(
        bool(eligible_rows),
        "no points eligible for ranking",
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
        "best metric is nonfinite",
    )

    for row in scan_rows:
        metric = float(
            row[ranking_metric]
        )

        row["fraction_of_best_proxy"] = (
            metric / best_metric
            if math.isfinite(metric)
            else ""
        )

        row["near_optimal_plateau"] = bool(
            row["eligible_for_ranking"]
            and float(
                row[
                    "fraction_of_best_proxy"
                ]
            )
            >= float(
                ranking[
                    "plateau_fraction_of_best"
                ]
            )
        )

    plateau_rows = [
        row
        for row in scan_rows
        if bool(
            row["near_optimal_plateau"]
        )
    ]

    require(
        bool(plateau_rows),
        "fine plateau is empty",
    )

    shortlist_candidates = [
        row
        for row in plateau_rows
        if bool(row["pareto_frontier"])
    ]

    if not shortlist_candidates:
        shortlist_candidates = (
            plateau_rows
        )

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
        : int(
            ranking["shortlist_size"]
        )
    ]

    def evaluate_named_point(
        specification: dict[str, Any],
    ) -> dict[str, Any]:
        result, _, _ = evaluate_point(
            pt_min=float(
                specification[
                    "jet_pt_min_GeV"
                ]
            ),
            eta_max=float(
                specification[
                    "jet_abs_eta_max"
                ]
            ),
            rhh_max=float(
                specification[
                    "rhh_sr_max"
                ]
            ),
            min_jet_pt=min_jet_pt,
            max_abs_eta=max_abs_eta,
            rhh=rhh,
            signal_mask=signal_mask,
            background_mask=background_mask,
            signal_modes=signal_modes,
            mode_masks=mode_masks,
            mode_denominators=mode_denominators,
            background_families=background_families,
            family_masks=family_masks,
            family_denominators=family_denominators,
            total_signal=total_signal,
            total_background=total_background,
            minimum_background=minimum_background,
            minimum_signal_mode=minimum_signal_mode,
        )

        result["label"] = specification[
            "label"
        ]

        return result

    reference_point = evaluate_named_point(
        config["reference_point"]
    )

    anchor_point = evaluate_named_point(
        config["anchor_point"]
    )

    require(
        anchor_point["point_id"]
        == coarse_summary[
            "best_point"
        ]["point_id"],
        (
            "fine anchor does not match "
            "the coarse optimum"
        ),
    )

    temporary_dir.mkdir(
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
    ]

    prefix = str(
        config["output_prefix"]
    )

    write_tsv(
        temporary_dir
        / f"{prefix}_scan_points.tsv",
        scan_rows,
        table_fields,
    )

    write_tsv(
        temporary_dir
        / f"{prefix}_scan_plateau.tsv",
        sorted(
            plateau_rows,
            key=lambda row: -float(
                row[ranking_metric]
            ),
        ),
        table_fields,
    )

    write_tsv(
        temporary_dir
        / f"{prefix}_scan_shortlist.tsv",
        shortlist_rows,
        table_fields,
    )

    write_tsv(
        temporary_dir
        / (
            f"{prefix}_scan_"
            "signal_mode_counts.tsv"
        ),
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
        temporary_dir
        / (
            f"{prefix}_scan_"
            "background_family_counts.tsv"
        ),
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

    named_points = []

    for label, point in (
        ("reference", reference_point),
        ("coarse_proxy_optimum", anchor_point),
        ("fine_proxy_optimum", best_point),
    ):
        named_points.append({
            "working_point":
                label,
            **{
                field: point[field]
                for field in table_fields
                if field in point
            },
        })

    write_tsv(
        temporary_dir
        / f"{prefix}_named_points.tsv",
        named_points,
        [
            "working_point",
            *table_fields,
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

    display_stage = str(
        config["display_stage"]
    )

    written_plots: list[Path] = []

    written_plots.extend(
        make_heatmaps(
            rows=scan_rows,
            output_dir=temporary_dir,
            prefix=prefix,
            display_stage=display_stage,
            pt_values=pt_values,
            eta_values=eta_values,
            rhh_values=rhh_values,
            best_point=best_point,
            anchor_point=anchor_point,
            formats=formats,
            dpi=dpi,
        )
    )

    written_plots.extend(
        make_pareto_plot(
            rows=scan_rows,
            output_dir=temporary_dir,
            prefix=prefix,
            display_stage=display_stage,
            eta_values=eta_values,
            pt_values=pt_values,
            best_point=best_point,
            anchor_point=anchor_point,
            reference_point=reference_point,
            formats=formats,
            dpi=dpi,
        )
    )

    written_plots.extend(
        make_profile_plot(
            rows=scan_rows,
            output_dir=temporary_dir,
            prefix=prefix,
            display_stage=display_stage,
            parameter="jet_pt_min_GeV",
            xlabel=(
                r"$p_{\mathrm{T}}^{\min}$ [GeV]"
            ),
            formats=formats,
            dpi=dpi,
        )
    )

    written_plots.extend(
        make_profile_plot(
            rows=scan_rows,
            output_dir=temporary_dir,
            prefix=prefix,
            display_stage=display_stage,
            parameter="jet_abs_eta_max",
            xlabel=(
                r"$|\eta|^{\max}$"
            ),
            formats=formats,
            dpi=dpi,
        )
    )

    written_plots.extend(
        make_profile_plot(
            rows=scan_rows,
            output_dir=temporary_dir,
            prefix=prefix,
            display_stage=display_stage,
            parameter="rhh_sr_max",
            xlabel=(
                r"$R_{HH}^{\mathrm{SR,max}}$"
            ),
            formats=formats,
            dpi=dpi,
        )
    )

    expected_plot_files = (
        len(eta_values) + 4
    ) * len(formats)

    require(
        len(written_plots)
        == expected_plot_files,
        (
            f"expected {expected_plot_files} "
            f"plot files, wrote "
            f"{len(written_plots)}"
        ),
    )

    proxy_change_from_coarse = (
        100.0
        * (
            float(
                best_point[
                    ranking_metric
                ]
            )
            / float(
                anchor_point[
                    ranking_metric
                ]
            )
            - 1.0
        )
    )

    proxy_change_from_reference = (
        100.0
        * (
            float(
                best_point[
                    ranking_metric
                ]
            )
            / float(
                reference_point[
                    ranking_metric
                ]
            )
            - 1.0
        )
    )

    summary = {
        "schema_version": 1,
        "status":
            "hh4b_train_only_fine_cut_scan_pass",
        "analysis_name":
            config["analysis_name"],
        "scan_stage":
            config["scan_stage"],
        "source_cache":
            str(cache_path),
        "source_cache_sha256":
            observed_cache_sha256,
        "source_coarse_summary":
            str(coarse_summary_path),
        "source_coarse_summary_sha256":
            sha256_file(
                coarse_summary_path
            ),
        "configuration":
            str(config_path),
        "configuration_sha256":
            sha256_file(config_path),
        "dataset_split":
            "train",
        "train_rows_evaluated":
            table.num_rows,
        "signal_train_rows":
            total_signal,
        "background_train_rows":
            total_background,
        "background_family_count":
            len(background_families),
        "grid_shape": {
            "jet_pt_min_GeV":
                len(pt_values),
            "jet_abs_eta_max":
                len(eta_values),
            "rhh_sr_max":
                len(rhh_values),
        },
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
                field: best_point[field]
                for field in table_fields
            },
        "coarse_anchor_point":
            anchor_point,
        "reference_point":
            reference_point,
        "proxy_change_from_coarse_percent":
            proxy_change_from_coarse,
        "proxy_change_from_reference_percent":
            proxy_change_from_reference,
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
            (
                "run_exact_observed_rhh_"
                "threshold_scan_around_fine_plateau"
            ),
    }

    summary_path = (
        temporary_dir
        / f"{prefix}_scan_summary.json"
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

    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in checksum_paths
    ]

    (
        temporary_dir
        / "SHA256SUMS"
    ).write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )

    temporary_dir.rename(
        output_dir
    )

    print(
        "HH4B_TRAIN_ONLY_FINE_CUT_SCAN_PASS"
    )
    print(
        "HH4B_GRID_POINTS="
        f"{len(scan_rows)}"
    )
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
        "HH4B_PROXY_CHANGE_FROM_COARSE_PERCENT="
        f"{proxy_change_from_coarse}"
    )
    print(
        "HH4B_PROXY_CHANGE_FROM_REFERENCE_PERCENT="
        f"{proxy_change_from_reference}"
    )
    print("HH4B_VALIDATION_ROWS_EVALUATED=0")
    print("HH4B_TEST_CANDIDATE_FILES_OPENED=0")
    print(
        "HH4B_PHYSICAL_SIGNIFICANCE_COMPUTED=False"
    )
    print("HH4B_PLOTS_PRODUCED=True")
    print(
        "NEXT_GATE="
        "run_exact_observed_rhh_"
        "threshold_scan_around_fine_plateau"
    )
    print(
        "summary_json="
        f"{output_dir / f'{prefix}_scan_summary.json'}"
    )


if __name__ == "__main__":
    main()
