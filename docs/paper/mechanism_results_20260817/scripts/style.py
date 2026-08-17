"""Shared plotting style. Validated categorical palette (dataviz skill,
references/palette.md, light-mode categorical slots 1-6), run through
scripts/validate_palette.js before use (see VALIDATION_REPORT.json /
SOURCE_PROVENANCE.tsv for the exact validator invocations and results).

House rule for this package: never use mplhep's CMS wordmark/label helper
(hep.cms.label / hep.cms.text) anywhere -- this project is explicitly not an
official CMS reproduction and must never carry a "CMS" stamp. mplhep is used
only for its axis/tick/font conventions.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import mplhep as hep
    hep.style.use("CMS")
    # Overwrite anything in the CMS style sheet that implies a CMS house
    # font/branding beyond generic sans-serif tick/frame conventions.
    plt.rcParams["font.family"] = "sans-serif"
except Exception:  # pragma: no cover - fallback if mplhep unavailable
    plt.rcParams.update({
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "font.family": "sans-serif",
    })

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "axes.grid": False,
    "legend.frameon": False,
    "font.size": 11,
})

# Validated categorical palette (dataviz skill palette.md, slots 1-8, light mode)
BLUE = "#2a78d6"     # slot 1
ORANGE = "#eb6834"   # slot 2
AQUA = "#1baf7a"      # slot 3
YELLOW = "#eda100"   # slot 4
MAGENTA = "#e87ba4"  # slot 5
GREEN = "#008300"    # slot 6

MUTED_INK = "#52514e"
AXIS_INK = "#898781"
GRID_INK = "#e1e0d9"

# Track-B E0/E1/E2 (validated all-pairs, both CVD modes; see VALIDATION_REPORT.json)
ARM_COLOR = {"E0": BLUE, "E1": ORANGE, "E2": AQUA}
ARM_MARKER = {"E0": "o", "E1": "s", "E2": "^"}
ARM_LABEL = {
    "E0": "E0 (five-jet kinematics only)",
    "E1": "E1 (E0 + continuous SophonAK4 flavor)",
    "E2": "E2 (E1 + frozen pairwise attention bias)",
}

# Track-B mechanism-gain ratio series
GAIN_COLOR = {"E1_over_E0": BLUE, "E2_over_E1": ORANGE}
GAIN_LABEL = {
    "E1_over_E0": r"$R_B(E1)\,/\,R_B(E0)$ -- continuous-flavor gain",
    "E2_over_E1": r"$R_B(E2)\,/\,R_B(E1)$ -- added pairwise-bias gain",
}

# Track-A: canonical left-to-right order in fig 3 == the palette's own fixed
# slot order 1-6 (validated as a set; see VALIDATION_REPORT.json).
TRACKA_MODEL_COLOR = {
    "cut_baseline": BLUE,
    "bdt_control0": ORANGE,
    "dnn": AQUA,
    "cmstransformer": YELLOW,
    "fivejetpartnet": MAGENTA,
    "bdt_newc": GREEN,
}
TRACKA_MODEL_HATCH = {
    "cut_baseline": "///",
    "bdt_control0": "///",
    "dnn": None,
    "cmstransformer": None,
    "fivejetpartnet": "///",
    "bdt_newc": "///",
}

SUPPORT_TIER_ORDER = [
    "well_supported",
    "statistically_limited_but_reportable",
    "exploratory_not_primary_quantitative_claim",
]
SUPPORT_TIER_SHORT = {
    "well_supported": r"well-supported ($\geq$100 raw QCD)",
    "statistically_limited_but_reportable": "limited but reportable (10-99 raw QCD)",
    "exploratory_not_primary_quantitative_claim": "exploratory only (<10 raw QCD)",
}


def apply_tier_style(tier: str) -> dict:
    """Marker/line kwargs encoding the predeclared statistical-support tier.
    Color always carries model/arm identity; tier is encoded redundantly via
    marker fill + size + error-bar presence, never via color alone."""
    if tier == "well_supported":
        return dict(markersize=7, markerfacecolor="__color__", alpha=1.0, mew=1.3)
    if tier == "statistically_limited_but_reportable":
        return dict(markersize=6, markerfacecolor="__color__", alpha=0.62, mew=1.1)
    return dict(markersize=5, markerfacecolor="none", alpha=0.85, mew=1.4)


SIM_WATERMARK = "External jetfree-hh4b Delphes benchmark\nDelphes Simulation -- not CMS data, not a CMS reproduction"
TRACKA_WATERMARK = "Delphes Simulation -- Track A -- not CMS data, not a CMS reproduction"


def sim_watermark(ax, text=SIM_WATERMARK, loc="upper left"):
    xy = {"upper left": (0.02, 0.94, "left", "top"),
          "upper right": (0.97, 0.94, "right", "top"),
          "lower left": (0.02, 0.04, "left", "bottom"),
          "lower right": (0.97, 0.04, "right", "bottom")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3],
             fontsize=8.0, color=MUTED_INK, linespacing=1.4)
