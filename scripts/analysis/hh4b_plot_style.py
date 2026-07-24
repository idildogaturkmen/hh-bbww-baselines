from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
from matplotlib.figure import Figure


def apply_hh4b_paper_style() -> None:
    """Apply a clean collider-paper visual style without experiment branding."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "DejaVu Sans",
            "Arial",
            "Liberation Sans",
        ],
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 1.2,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.titlepad": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False
    })


def save_figure(
    figure: Figure,
    base_path: Path,
    formats: Iterable[str],
    *,
    dpi: int = 180,
) -> list[Path]:
    """Save one figure in all requested formats."""
    base_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    written: list[Path] = []

    for file_format in formats:
        normalized = file_format.lower().lstrip(".")
        output = base_path.with_suffix(f".{normalized}")

        figure.savefig(
            output,
            dpi=dpi,
            bbox_inches="tight",
        )

        written.append(output)

    return written
