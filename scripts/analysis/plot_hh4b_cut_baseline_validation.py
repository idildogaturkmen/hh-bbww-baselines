#!/usr/bin/env python3
"""Plot the frozen one-time validation result against frozen train estimates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


SCOPES = ("exact3tag", "ge4tag", "combined")
SERIES = (
    (
        "nested_train_oof",
        "nested_outer_oof",
        "Train: nested outer-OOF",
        "o",
        "#2166ac",
    ),
    (
        "fixed_train_diagnostic",
        "fixed_nominal_deployment_cut",
        "Train: fixed nominal diagnostic",
        "s",
        "#67a9cf",
    ),
    (
        "fixed_validation",
        "fixed_nominal_deployment_cut",
        "Validation: fixed nominal (one time)",
        "D",
        "#b2182b",
    ),
)


class PlotError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlotError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path: Path) -> None:
    require(path.is_dir() and not path.is_symlink(), f"bad checkpoint: {path}")
    result = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(result.returncode == 0, f"checkpoint checksum failure: {result.stdout}{result.stderr}")


def build_figure_rows(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> list[dict[str, Any]]:
    train_pooled = train.loc[train["outer_fold"].astype(str).eq("pooled")]
    rows: list[dict[str, Any]] = []
    for series_id, selection_id, label, marker, color in SERIES:
        source = validation if series_id == "fixed_validation" else train_pooled
        for scope in SCOPES:
            matches = source.loc[
                source["scope"].astype(str).eq(scope)
                & source["selection_id"].astype(str).eq(selection_id)
            ]
            require(len(matches) == 1, f"figure source rows={len(matches)} for {series_id}/{scope}")
            row = matches.iloc[0]
            for metric in (
                "signal_physical_efficiency",
                "background_physical_efficiency",
                "background_rejection",
                "signal_selected_signed_yield",
                "background_selected_signed_yield",
                "signal_over_background",
                "asimov_significance_stat_only",
            ):
                value = float(row[metric])
                require(np.isfinite(value), f"non-finite figure metric: {series_id}/{scope}/{metric}")
                rows.append(
                    {
                        "series_id": series_id,
                        "series_label": label,
                        "selection_id": selection_id,
                        "scope": scope,
                        "metric": metric,
                        "value": value,
                        "marker": marker,
                        "color": color,
                        "like_for_like_fixed_nominal": series_id
                        in {"fixed_train_diagnostic", "fixed_validation"},
                        "validation_payloads_opened": 1,
                        "test_payloads_opened": 0,
                    }
                )
    require(len(rows) == len(SERIES) * len(SCOPES) * 7, "figure-data row count mismatch")
    return rows


def draw(rows: list[dict[str, Any]], pdf: Path, png: Path) -> None:
    data = pd.DataFrame(rows)
    x = np.arange(len(SCOPES), dtype=float)
    offsets = (-0.18, 0.0, 0.18)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    labels = ["exactly 3 tags", r"$\geq 4$ tags", "combined"]

    for offset, (series_id, _, label, marker, color) in zip(offsets, SERIES):
        subset = data.loc[data["series_id"].eq(series_id)]
        signal = [
            float(
                subset.loc[
                    subset["scope"].eq(scope)
                    & subset["metric"].eq("signal_physical_efficiency"),
                    "value",
                ].iloc[0]
            )
            for scope in SCOPES
        ]
        rejection = [
            float(
                subset.loc[
                    subset["scope"].eq(scope)
                    & subset["metric"].eq("background_rejection"),
                    "value",
                ].iloc[0]
            )
            for scope in SCOPES
        ]
        axes[0].plot(
            x + offset,
            signal,
            marker=marker,
            color=color,
            linestyle="none",
            markersize=7,
            label=label,
        )
        axes[1].plot(
            x + offset,
            rejection,
            marker=marker,
            color=color,
            linestyle="none",
            markersize=7,
            label=label,
        )

    axes[0].set_ylabel(r"Signal efficiency $\epsilon_{S}$")
    axes[1].set_ylabel(r"Background rejection $1/\epsilon_{\mathrm{bkg}}$")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.grid(axis="y", alpha=0.3, linewidth=0.7)
        axis.set_xlim(-0.45, len(SCOPES) - 0.55)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
    )
    axes[0].text(
        0.0,
        1.07,
        "Delphes simulation",
        transform=axes[0].transAxes,
        fontsize=12,
        fontweight="bold",
    )
    axes[1].text(
        1.0,
        1.07,
        r"$\sqrt{s}=13$ TeV, 138 fb$^{-1}$ equivalent",
        transform=axes[1].transAxes,
        ha="right",
        fontsize=10,
    )
    fig.suptitle(
        "Frozen resolved HH→4b cut: train and one-time validation",
        fontsize=13,
        y=0.965,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.78, wspace=0.2)
    fig.savefig(pdf)
    fig.savefig(png, dpi=350)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-checkpoint", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    verify_checkpoint(args.train_checkpoint)
    verify_checkpoint(args.validation_checkpoint)
    require(args.authorization.is_file() and not args.authorization.is_symlink(), "validation authorization is missing")
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    require(authorization.get("status") == "authorized_one_time_cut_baseline_validation", "validation authorization status changed")
    require(
        authorization.get("authorized_sha256", {}).get("validation_plotter")
        == sha256(Path(__file__).resolve()),
        "validation plotter is not the master-authorized implementation",
    )
    validation_freeze = json.loads(
        (args.validation_checkpoint / "artifact_checkpoint_freeze.json").read_text(encoding="utf-8")
    )
    require(
        validation_freeze.get("status")
        == "pass_validation_performance_artifact_checkpoint_frozen"
        and validation_freeze.get("role") == "validation_performance",
        "validation performance checkpoint freeze status changed",
    )
    require(
        validation_freeze.get("implementation_sha256")
        == authorization.get("authorized_sha256", {}).get("validation_artifact_freezer"),
        "validation performance checkpoint was not produced by the master-authorized freezer",
    )
    train_root = args.train_checkpoint / "evidence/train_performance"
    validation_root = args.validation_checkpoint / "evidence/validation_performance"
    train_summary_path = train_root / "train_performance_summary.json"
    validation_summary_path = validation_root / "validation_performance_summary.json"
    train_table_path = train_root / "tables/train_performance_metrics.tsv"
    validation_table_path = validation_root / "tables/validation_performance_metrics.tsv"
    for path in [
        train_summary_path,
        validation_summary_path,
        train_table_path,
        validation_table_path,
    ]:
        require(path.is_file() and not path.is_symlink(), f"missing plot input: {path}")
    train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
    validation_summary = json.loads(validation_summary_path.read_text(encoding="utf-8"))
    require(train_summary.get("validation_payloads_opened") == 0, "train input used validation")
    require(train_summary.get("test_payloads_opened") == 0, "train input used test")
    require(
        validation_summary.get("status")
        == "pass_one_time_hh4b_cut_baseline_validation_aggregation",
        "validation summary status changed",
    )
    require(validation_summary.get("repository_head") == args.expected_head, "validation head changed")
    require(validation_summary.get("validation_payloads_opened") == 1, "validation open count changed")
    require(validation_summary.get("test_payloads_opened") == 0, "test input used")
    require(validation_summary.get("nominal_selection_changed") is False, "nominal selection changed")

    rows = build_figure_rows(
        pd.read_csv(train_table_path, sep="\t", keep_default_na=False),
        pd.read_csv(validation_table_path, sep="\t", keep_default_na=False),
    )
    final_output = args.output_root.resolve()
    require(not final_output.exists() and final_output.parent.is_dir(), f"output root exists/invalid: {final_output}")
    output = final_output.parent / f".{final_output.name}.build.{os.getpid()}"
    require(not output.exists(), f"validation figure build root exists: {output}")
    (output / "figures/pdf").mkdir(parents=True)
    (output / "figures/png").mkdir(parents=True)
    (output / "figure_data").mkdir(parents=True)
    (output / "manifests").mkdir(parents=True)
    stem = "figure17_validation_vs_train_oof"
    pdf = output / f"figures/pdf/{stem}.pdf"
    png = output / f"figures/png/{stem}.png"
    sidecar = output / f"figure_data/{stem}.tsv"
    draw(rows, pdf, png)
    pd.DataFrame(rows).to_csv(sidecar, sep="\t", index=False, lineterminator="\n")
    manifest = {
        "schema_version": 1,
        "status": "pass_cut_baseline_validation_publication_figure",
        "repository_head": args.expected_head,
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "inputs": {
            str(train_summary_path): sha256(train_summary_path),
            str(validation_summary_path): sha256(validation_summary_path),
            str(train_table_path): sha256(train_table_path),
            str(validation_table_path): sha256(validation_table_path),
        },
        "outputs": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (pdf, png, sidecar)
        },
        "official_cms_result": False,
        "validation_payloads_opened": 1,
        "test_payloads_opened": 0,
    }
    manifest_path = output / f"manifests/{stem}_provenance.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  ./{path.relative_to(output).as_posix()}\n"
            for path in checksum_files
        ),
        encoding="utf-8",
    )
    os.replace(output, final_output)
    print("CUT_BASELINE_VALIDATION_PUBLICATION_FIGURE=PASS")
    print("FIGURES=1_PDF_AND_1_PNG")
    print("FIGURE_DATA_SIDECARS=1")
    print("VALIDATION_PAYLOADS_OPENED=1")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
