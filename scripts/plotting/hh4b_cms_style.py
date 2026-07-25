#!/usr/bin/env python3
"""CMS-publication-inspired, explicitly nonofficial HH4b plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt


FEATURE_LATEX_LABELS: dict[str, str] = {
    "n_selected_jets": r"$N_{\mathrm{jet}}$",
    "n_selected_bjets": r"$N_{b\mathrm{-jet}}$",
    "ht_selected_jets": r"$H_{\mathrm{T}}^{\mathrm{jets}}$",
    "ht_selected_bjets": r"$H_{\mathrm{T}}^{b\mathrm{-jets}}$",
    "ht_candidate_jets": r"$H_{\mathrm{T}}^{HH\ \mathrm{jets}}$",
    "mhh": r"$m_{HH}$",
    "hh_pt": r"$p_{\mathrm{T}}^{HH}$",
    "abs_hh_eta": r"$|\eta_{HH}|$",
    "h1_pt": r"$p_{\mathrm{T}}^{H_1}$",
    "abs_h1_eta": r"$|\eta_{H_1}|$",
    "h2_pt": r"$p_{\mathrm{T}}^{H_2}$",
    "abs_h2_eta": r"$|\eta_{H_2}|$",
    "abs_h_delta_eta": r"$|\Delta\eta_{HH}|$",
    "abs_h_delta_phi": r"$|\Delta\phi_{HH}|$",
    "h_delta_r": r"$\Delta R_{HH}$",
    "h_pt_balance": r"$p_{\mathrm{T}}\ \mathrm{balance}$",
    "drbb1": r"$\Delta R_{bb}^{H_1}$",
    "drbb2": r"$\Delta R_{bb}^{H_2}$",
    "j1_pt": r"$p_{\mathrm{T}}^{j_1}$",
    "abs_j1_eta": r"$|\eta_{j_1}|$",
    "j1_mass": r"$m_{j_1}$",
    "j2_pt": r"$p_{\mathrm{T}}^{j_2}$",
    "abs_j2_eta": r"$|\eta_{j_2}|$",
    "j2_mass": r"$m_{j_2}$",
    "j3_pt": r"$p_{\mathrm{T}}^{j_3}$",
    "abs_j3_eta": r"$|\eta_{j_3}|$",
    "j3_mass": r"$m_{j_3}$",
    "j4_pt": r"$p_{\mathrm{T}}^{j_4}$",
    "abs_j4_eta": r"$|\eta_{j_4}|$",
    "j4_mass": r"$m_{j_4}$",
    "mbb1": r"$m_{bb}^{H_1}$",
    "mbb2": r"$m_{bb}^{H_2}$",
    "delta_mbb": r"$|m_{bb}^{H_1}-m_{bb}^{H_2}|$",
    "r_hh_125_125": r"$R_{HH}^{125,125}$",
}

CATEGORY_COLORS: tuple[str, ...] = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#56B4E9",
    "#CC79A7",
    "#F0E442",
    "#000000",
    "#999999",
    "#332288",
)


def apply_cms_style() -> dict[str, Any]:
    """Apply mplhep CMS styling when present, otherwise a fixed clean fallback."""

    mplhep_used = False
    try:
        import mplhep

        plt.style.use(mplhep.style.CMS)
        mplhep_used = True
    except (ImportError, AttributeError):
        plt.style.use("default")

    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10.0,
            "axes.labelsize": 11.0,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.top": False,
            "ytick.right": False,
            "legend.frameon": False,
            "text.usetex": False,
            "mathtext.fontset": "dejavusans",
        }
    )
    return {
        "mplhep_used": mplhep_used,
        "cms_inspired_style_applied": True,
        "latex_math_labels_applied": True,
        "official_cms_status_claimed": False,
    }


def add_delphes_header(ax: matplotlib.axes.Axes, secondary: str) -> None:
    """Add the required truthful simulation and collision-energy annotation."""

    ax.text(
        0.0,
        1.035,
        "Delphes simulation",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
    )
    ax.text(
        1.0,
        1.035,
        r"$\sqrt{s}=13\,\mathrm{TeV}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    ax.text(
        0.0,
        1.005,
        secondary,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )


def save_png_pdf(
    fig: matplotlib.figure.Figure,
    base_path: Path,
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Write paired publication raster/vector files and close the figure."""

    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png = base_path.with_suffix(".png")
    pdf = base_path.with_suffix(".pdf")
    try:
        fig.savefig(png, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    if not png.is_file() or png.stat().st_size == 0:
        raise RuntimeError(f"PNG was not written: {png}")
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise RuntimeError(f"PDF was not written: {pdf}")
    return png, pdf
