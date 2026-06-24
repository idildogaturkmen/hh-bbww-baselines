#!/usr/bin/env python3
'''
Quick follow-up to evaluate H→bb mass scaling and event inspection candidates.

Changes from the original `scripts/evaluate_hbb_mass_scaling.py`:
- Add normalized mass metrics. (formula: `scaled_width68_gev = width68 * 125 / median`)
- Add event inspection candidates (see `pick_top` function, picks top n events based on a scoring column).
- Add summary markdown output to `Results Summaries/RESULTS_HBB_MASS_SCALING_AND_EVENT_INSPECTION.md`.

'''

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import mplhep as hep
    hep.style.use("CMS")
    HAS_MPLHEP = True
except Exception:
    HAS_MPLHEP = False


M_HIGGS = 125.0
PT_CUTS = [0.0, 30.0, 40.0, 50.0, 60.0]


def width68(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    q16, q84 = np.percentile(values, [16, 84])
    return 0.5 * (q84 - q16)


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "n": 0,
            "median": np.nan,
            "mean": np.nan,
            "width68_gev": np.nan,
            "width68_percent_of_median": np.nan,
            "scale_to_125_from_median": np.nan,
            "scaled_width68_gev": np.nan,
            "raw_frac_90_140": np.nan,
            "raw_tail_low_90": np.nan,
            "raw_tail_high_140": np.nan,
            "scaled_frac_90_140": np.nan,
            "scaled_tail_low_90": np.nan,
            "scaled_tail_high_140": np.nan,
            "normalized_tail_high_140_equiv": np.nan,
        }

    med = float(np.median(values))
    mean = float(np.mean(values))
    w68 = float(width68(values))
    scale = M_HIGGS / med if med > 0 else np.nan
    scaled = values * scale

    return {
        "n": int(len(values)),
        "median": med,
        "mean": mean,
        "width68_gev": w68,
        "width68_percent_of_median": 100.0 * w68 / med if med > 0 else np.nan,
        "scale_to_125_from_median": scale,
        "scaled_width68_gev": w68 * scale if np.isfinite(scale) else np.nan,

        # Original raw GeV-window metrics
        "raw_frac_90_140": float(np.mean((values >= 90.0) & (values <= 140.0))),
        "raw_tail_low_90": float(np.mean(values < 90.0)),
        "raw_tail_high_140": float(np.mean(values > 140.0)),

        # Harvey-requested fair comparison:
        # scale each distribution so the median is 125, then apply the same GeV thresholds.
        "scaled_frac_90_140": float(np.mean((scaled >= 90.0) & (scaled <= 140.0))),
        "scaled_tail_low_90": float(np.mean(scaled < 90.0)),
        "scaled_tail_high_140": float(np.mean(scaled > 140.0)),

        # Equivalent normalized high-tail condition:
        # m / median > 140 / 125
        "normalized_tail_high_140_equiv": float(np.mean((values / med) > (140.0 / 125.0))) if med > 0 else np.nan,
    }


def add_plot_label(ax):
    if HAS_MPLHEP:
        try:
            hep.cms.text("COLLIDE-1M Simulation", ax=ax)
        except Exception:
            ax.text(0.0, 1.01, "COLLIDE-1M Simulation", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=13, fontweight="bold")
    else:
        ax.text(0.0, 1.01, "COLLIDE-1M Simulation", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=13, fontweight="bold")

    ax.text(1.0, 1.01, "HH→bbWW*", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=12)


def pick_top(df, category, mask, score_col, n=10, ascending=False):
    sub = df.loc[mask].copy()
    if len(sub) == 0:
        return pd.DataFrame()
    sub["inspection_category"] = category
    sub["inspection_score"] = sub[score_col]
    return sub.sort_values(score_col, ascending=ascending).head(n)


def main():
    outdir = Path("outputs/hbb_mass_scaling_and_event_inspection")
    plotdir = Path("outputs/plots/hbb_mass_scaling_and_event_inspection")
    outdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    event_csv = Path("outputs/bjet_response_clipping/test_event_mbb_with_clipped_corrections.csv")
    if not event_csv.exists():
        raise FileNotFoundError(
            f"Missing {event_csv}. Run scripts/scan_bjet_response_clipping.py first."
        )

    df = pd.read_csv(event_csv)

    # Identify available mass versions.
    version_cols = {
        "uncorrected": "mbb_uncorrected",
    }
    for c in df.columns:
        if c.startswith("mbb_dnn_"):
            version_cols[c.replace("mbb_", "")] = c

    # Preferred order if present.
    preferred = [
        "uncorrected",
        "dnn_original",
        "dnn_cap_1p25",
        "dnn_cap_1p50",
        "dnn_cap_1p75",
        "dnn_cap_2p00",
        "dnn_cap_2p50",
    ]
    ordered_versions = [v for v in preferred if v in version_cols]
    ordered_versions += [v for v in version_cols if v not in ordered_versions]

    # Harvey-style normalized metrics.
    rows = []
    for cut in PT_CUTS:
        pass_mask = df["min_bjet_pt"].to_numpy() >= cut
        n_total = len(df)
        n_pass = int(np.sum(pass_mask))
        eff = n_pass / n_total if n_total else np.nan

        for version in ordered_versions:
            values = df.loc[pass_mask, version_cols[version]].to_numpy()
            row = {
                "min_bjet_pt_cut": cut,
                "version": version,
                "n_total": n_total,
                "n_pass": n_pass,
                "efficiency": eff,
            }
            row.update(summarize(values))
            rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics_path = outdir / "test_normalized_mass_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    # Event inspection candidates.
    inspect_cols = [
        c for c in [
            "event_local_index",
            "b1",
            "b2",
            "min_bjet_pt",
            "mbb_uncorrected",
            "mbb_dnn_original",
            "mbb_dnn_cap_1p25",
            "mbb_dnn_cap_1p50",
            "mbb_dnn_cap_2p00",
        ]
        if c in df.columns
    ]

    work = df.copy()
    if "mbb_dnn_original" in work.columns:
        work["dnn_shift"] = work["mbb_dnn_original"] - work["mbb_uncorrected"]
        work["dnn_ratio"] = work["mbb_dnn_original"] / np.maximum(work["mbb_uncorrected"], 1e-9)
    if "mbb_dnn_cap_1p25" in work.columns and "mbb_dnn_original" in work.columns:
        work["cap1p25_reduction"] = work["mbb_dnn_original"] - work["mbb_dnn_cap_1p25"]

    candidate_tables = []

    candidate_tables.append(
        pick_top(
            work,
            "lowest_min_bjet_pt",
            work["min_bjet_pt"].notna(),
            "min_bjet_pt",
            n=10,
            ascending=True,
        )
    )

    candidate_tables.append(
        pick_top(
            work,
            "near_30gev_threshold_below",
            (work["min_bjet_pt"] >= 20) & (work["min_bjet_pt"] < 30),
            "min_bjet_pt",
            n=10,
            ascending=False,
        )
    )

    candidate_tables.append(
        pick_top(
            work,
            "near_30gev_threshold_above",
            (work["min_bjet_pt"] >= 30) & (work["min_bjet_pt"] < 40),
            "min_bjet_pt",
            n=10,
            ascending=True,
        )
    )

    if "mbb_dnn_original" in work.columns:
        candidate_tables.append(
            pick_top(
                work,
                "already_high_uncorrected",
                work["mbb_uncorrected"] > 140,
                "mbb_uncorrected",
                n=10,
                ascending=False,
            )
        )

        candidate_tables.append(
            pick_top(
                work,
                "new_high_after_original_dnn",
                (work["mbb_uncorrected"] <= 140) & (work["mbb_dnn_original"] > 140),
                "dnn_shift",
                n=10,
                ascending=False,
            )
        )

        candidate_tables.append(
            pick_top(
                work,
                "largest_original_dnn_ratio",
                work["dnn_ratio"].replace([np.inf, -np.inf], np.nan).notna(),
                "dnn_ratio",
                n=10,
                ascending=False,
            )
        )

    if "mbb_dnn_cap_1p25" in work.columns and "mbb_dnn_original" in work.columns:
        candidate_tables.append(
            pick_top(
                work,
                "cap1p25_fixes_original_dnn_high_tail",
                (work["mbb_dnn_original"] > 140) & (work["mbb_dnn_cap_1p25"] <= 140),
                "cap1p25_reduction",
                n=10,
                ascending=False,
            )
        )

    candidates = pd.concat([x for x in candidate_tables if len(x)], ignore_index=True)
    keep_cols = ["inspection_category", "inspection_score"] + [
        c for c in inspect_cols + ["dnn_shift", "dnn_ratio", "cap1p25_reduction"]
        if c in candidates.columns
    ]
    candidates = candidates[keep_cols]
    candidates_path = outdir / "test_event_inspection_candidates.csv"
    candidates.to_csv(candidates_path, index=False)

    # Plots for quick meeting.
    no_cut = metrics[metrics["min_bjet_pt_cut"] == 0].copy()
    no_cut["version"] = pd.Categorical(no_cut["version"], categories=ordered_versions, ordered=True)
    no_cut = no_cut.sort_values("version")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(no_cut["version"].astype(str), no_cut["width68_percent_of_median"], marker="o")
    ax.set_ylabel("width68 / median [%]")
    ax.set_xlabel("Correction version")
    ax.set_title("Relative H→bb mass resolution")
    ax.tick_params(axis="x", rotation=35)
    add_plot_label(ax)
    fig.tight_layout()
    fig.savefig(plotdir / "test_relative_width68_percent.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(no_cut["version"].astype(str), no_cut["raw_tail_high_140"], marker="o", label="raw m_bb > 140")
    ax.plot(no_cut["version"].astype(str), no_cut["scaled_tail_high_140"], marker="o", label="scaled-to-125 m_bb > 140")
    ax.set_ylabel("High-tail fraction")
    ax.set_xlabel("Correction version")
    ax.set_title("Raw vs normalized high-mass tail")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    add_plot_label(ax)
    fig.tight_layout()
    fig.savefig(plotdir / "test_raw_vs_scaled_high_tail.png", dpi=200)
    plt.close(fig)

    # Summary markdown
    md = []
    md.append("# H→bb Mass Scaling and Event Inspection Study")
    md.append("")
    md.append("This quick follow-up adds Harvey-requested normalized mass metrics and event candidates for inspection.")
    md.append("")
    md.append("## Outputs")
    md.append("")
    md.append(f"- `{metrics_path}`")
    md.append(f"- `{candidates_path}`")
    md.append(f"- `{plotdir / 'test_relative_width68_percent.png'}`")
    md.append(f"- `{plotdir / 'test_raw_vs_scaled_high_tail.png'}`")
    md.append("")
    md.append("## Definitions")
    md.append("")
    md.append("- `width68_percent_of_median = 100 * width68 / median`")
    md.append("- `scale_to_125_from_median = 125 / median`")
    md.append("- `scaled_width68_gev = width68 * 125 / median`")
    md.append("- `scaled_tail_high_140` is computed after scaling each mass distribution so its median is 125 GeV.")
    md.append("")
    md.append("## No-cut metrics")
    md.append("")
    show_cols = [
        "version",
        "median",
        "width68_gev",
        "width68_percent_of_median",
        "scaled_width68_gev",
        "raw_tail_high_140",
        "scaled_tail_high_140",
        "raw_frac_90_140",
        "scaled_frac_90_140",
    ]
    md.append(no_cut[show_cols].to_markdown(index=False, floatfmt=".4f"))
    md.append("")
    md.append("## Event categories selected for inspection")
    md.append("")
    md.append(candidates["inspection_category"].value_counts().to_markdown())
    md.append("")

    summary_path = Path("Results Summaries") / "RESULTS_HBB_MASS_SCALING_AND_EVENT_INSPECTION.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(md) + "\n")

    print(f"Wrote {metrics_path}")
    print(f"Wrote {candidates_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote plots in {plotdir}")
    print()
    print("No-cut normalized metrics:")
    print(no_cut[show_cols].to_string(index=False))
    print()
    print("Event inspection candidate categories:")
    print(candidates['inspection_category'].value_counts().to_string())


if __name__ == "__main__":
    main()
