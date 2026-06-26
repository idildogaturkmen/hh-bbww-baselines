'''
Uses the best pT-binned m_bb correction derived from the signal calibration events to all events in the one-lepton pre-mass region, and produces a summary of metrics and plots comparing the uncorrected, global median scaled, and pT-binned scaled m_bb variants.
n_bins = 6
clip_min = 0.95
clip_max = 1.20
mass window = 80–150

'''
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


WINDOWS = [
    (80, 150),
    (90, 140),
    (95, 145),
    (100, 140),
    (100, 150),
    (105, 145),
]


LABELS = {
    "mbb_uncorrected": "Uncorrected",
    "mbb_global_median_scaled": "Global median scale",
    "mbb_pt_binned_scaled": "pT-binned scale",
    "signal": "HH→bbWW signal",
    "all_background": "All backgrounds",
    "HH_other": "Other HH",
    "single_higgs": "Single Higgs",
    "diboson": "Diboson",
    "ttbar": "ttbar",
    "ttV_ttH_tttt": "ttV/ttH/tttt",
}


MAIN_BACKGROUNDS = [
    "all_background",
    "single_higgs",
    "HH_other",
    "diboson",
    "ttbar",
    "ttV_ttH_tttt",
]


def asimov_z(s: float, b: float) -> float:
    if s <= 0 or b <= 0:
        return 0.0
    return float(np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s)))


def width68(x: np.ndarray) -> tuple[float, float, float]:
    q16, q84 = np.percentile(x, [16, 84])
    return float(q16), float(q84), float(0.5 * (q84 - q16))


def shape_metrics(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "q16": np.nan,
            "q84": np.nan,
            "width68": np.nan,
            "width68_over_median": np.nan,
            "frac_90_140": np.nan,
            "frac_100_150": np.nan,
            "frac_gt_160": np.nan,
        }

    q16, q84, w68 = width68(x)
    med = float(np.median(x))

    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": med,
        "q16": q16,
        "q84": q84,
        "width68": w68,
        "width68_over_median": float(w68 / med) if med > 0 else np.nan,
        "frac_90_140": float(np.mean((x >= 90) & (x <= 140))),
        "frac_100_150": float(np.mean((x >= 100) & (x <= 150))),
        "frac_gt_160": float(np.mean(x > 160)),
    }


def add_deterministic_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    key = (
        df["sample"].astype(str)
        + "_"
        + df["file_index"].astype(str)
        + "_"
        + df["event_in_file"].astype(str)
    )
    hashed = pd.util.hash_pandas_object(key, index=False).to_numpy()
    df["is_calib"] = (hashed % 2) == 0
    return df


def make_preselection(df: pd.DataFrame) -> pd.Series:
    return (
        df["pass_1lep_ge3jets"].astype(bool)
        & (df["n_jets"] >= 3)
        & (df["n_jets"] <= 5)
        & (df["b2_btag"] >= 1)
        & np.isfinite(df["mbb_top2_btag"])
        & np.isfinite(df["b1_pt"])
        & np.isfinite(df["b2_pt"])
    )


def make_quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)

    if len(edges) < 3:
        edges = np.linspace(np.min(x), np.max(x), min(n_bins, 4) + 1)

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def calibrate_pt_binned_table(
    df_pre: pd.DataFrame,
    n_bins: int,
    clip_min: float,
    clip_max: float,
    target_mass: float,
    min_signal_per_bin: int,
) -> pd.DataFrame:
    sig_calib = df_pre[
        (df_pre["process_group"] == "signal") & (df_pre["is_calib"])
    ].copy()

    global_median = float(np.median(sig_calib["mbb_uncorrected"]))
    global_scale = target_mass / global_median

    edges = make_quantile_bins(sig_calib["soft_b_pt"].to_numpy(), n_bins)

    rows = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]

        mask = (sig_calib["soft_b_pt"] >= lo) & (sig_calib["soft_b_pt"] < hi)
        x = sig_calib.loc[mask, "mbb_uncorrected"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]

        if len(x) >= min_signal_per_bin:
            median_mbb = float(np.median(x))
            raw_scale = target_mass / median_mbb if median_mbb > 0 else np.nan
        else:
            median_mbb = np.nan
            raw_scale = np.nan

        if np.isfinite(raw_scale):
            scale = float(np.clip(raw_scale, clip_min, clip_max))
        else:
            scale = float(np.clip(global_scale, clip_min, clip_max))

        rows.append(
            {
                "bin_index": i,
                "soft_b_pt_low": lo,
                "soft_b_pt_high": hi,
                "n_signal_calib": int(len(x)),
                "median_mbb_signal_calib": median_mbb,
                "raw_scale_factor": raw_scale,
                "scale_factor": scale,
                "was_clipped_low": bool(np.isfinite(raw_scale) and raw_scale < clip_min),
                "was_clipped_high": bool(np.isfinite(raw_scale) and raw_scale > clip_max),
                "global_signal_median_calib": global_median,
                "global_scale_factor": global_scale,
            }
        )

    return pd.DataFrame(rows)


def apply_pt_binned_table(df_pre: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
    mbb = df_pre["mbb_uncorrected"].to_numpy(dtype=float)
    soft = df_pre["soft_b_pt"].to_numpy(dtype=float)
    corrected = np.full(len(df_pre), np.nan)

    for _, row in table.iterrows():
        lo = row["soft_b_pt_low"]
        hi = row["soft_b_pt_high"]
        scale = row["scale_factor"]
        mask = (soft >= lo) & (soft < hi)
        corrected[mask] = mbb[mask] * scale

    return corrected


def scan_windows(df_pre: pd.DataFrame, mass_col: str, ht_min: float) -> pd.DataFrame:
    rows = []
    mass = df_pre[mass_col].to_numpy(dtype=float)
    ht = df_pre["ht"].to_numpy(dtype=float)
    is_signal = df_pre["process_group"].to_numpy() == "signal"

    for lo, hi in WINDOWS:
        mask = (mass >= lo) & (mass <= hi) & (ht >= ht_min)

        s = int(np.sum(mask & is_signal))
        b = int(np.sum(mask & ~is_signal))

        rows.append(
            {
                "mass_variant": mass_col,
                "window_low": lo,
                "window_high": hi,
                "ht_min": ht_min,
                "S_raw": s,
                "B_raw": b,
                "S_over_B": s / b if b > 0 else np.nan,
                "S_over_sqrt_SB": s / np.sqrt(s + b) if (s + b) > 0 else np.nan,
                "asimovZ_raw": asimov_z(s, b),
            }
        )

    return pd.DataFrame(rows)


def make_shape_rows(df_pre: pd.DataFrame, mass_cols: list[str]) -> pd.DataFrame:
    rows = []

    for mass_col in mass_cols:
        mass = df_pre[mass_col].to_numpy(dtype=float)
        is_signal = df_pre["process_group"].to_numpy() == "signal"
        is_calib = df_pre["is_calib"].to_numpy(dtype=bool)

        groups = {
            "signal_all": mass[is_signal],
            "signal_calib": mass[is_signal & is_calib],
            "signal_validation": mass[is_signal & ~is_calib],
            "all_background": mass[~is_signal],
        }

        for group_name, values in groups.items():
            row = shape_metrics(values)
            row["mass_variant"] = mass_col
            row["group"] = group_name
            rows.append(row)

        for proc, sub in df_pre.groupby("process_group"):
            row = shape_metrics(sub[mass_col].to_numpy(dtype=float))
            row["mass_variant"] = mass_col
            row["group"] = proc
            rows.append(row)

    return pd.DataFrame(rows)


def make_final_comparison_table(window_scan: pd.DataFrame, shape: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for mass_col in ["mbb_uncorrected", "mbb_global_median_scaled", "mbb_pt_binned_scaled"]:
        best = (
            window_scan[window_scan["mass_variant"] == mass_col]
            .sort_values("asimovZ_raw", ascending=False)
            .iloc[0]
        )

        sig_val = shape[
            (shape["mass_variant"] == mass_col)
            & (shape["group"] == "signal_validation")
        ].iloc[0]

        sig_all = shape[
            (shape["mass_variant"] == mass_col)
            & (shape["group"] == "signal_all")
        ].iloc[0]

        bkg = shape[
            (shape["mass_variant"] == mass_col)
            & (shape["group"] == "all_background")
        ].iloc[0]

        rows.append(
            {
                "mass_variant": mass_col,
                "label": LABELS[mass_col],
                "best_window": f"{int(best['window_low'])}-{int(best['window_high'])}",
                "S_raw": int(best["S_raw"]),
                "B_raw": int(best["B_raw"]),
                "S_over_B": best["S_over_B"],
                "S_over_sqrt_SB": best["S_over_sqrt_SB"],
                "asimovZ_raw": best["asimovZ_raw"],
                "signal_all_median": sig_all["median"],
                "signal_all_width68_over_median": sig_all["width68_over_median"],
                "signal_val_median": sig_val["median"],
                "signal_val_width68_over_median": sig_val["width68_over_median"],
                "signal_val_frac_90_140": sig_val["frac_90_140"],
                "signal_val_frac_100_150": sig_val["frac_100_150"],
                "bkg_median": bkg["median"],
                "bkg_width68_over_median": bkg["width68_over_median"],
                "bkg_frac_90_140": bkg["frac_90_140"],
                "bkg_frac_100_150": bkg["frac_100_150"],
                "bkg_frac_gt_160": bkg["frac_gt_160"],
            }
        )

    out = pd.DataFrame(rows)

    ref_z = float(out.loc[out["mass_variant"] == "mbb_uncorrected", "asimovZ_raw"].iloc[0])
    ref_width = float(
        out.loc[
            out["mass_variant"] == "mbb_uncorrected",
            "signal_val_width68_over_median",
        ].iloc[0]
    )

    out["delta_asimovZ_vs_uncorrected"] = out["asimovZ_raw"] - ref_z
    out["relative_asimovZ_improvement_percent"] = 100.0 * (
        out["asimovZ_raw"] / ref_z - 1.0
    )
    out["delta_signal_val_width68_over_median_vs_uncorrected"] = (
        out["signal_val_width68_over_median"] - ref_width
    )
    out["relative_signal_val_width_improvement_percent"] = 100.0 * (
        1.0 - out["signal_val_width68_over_median"] / ref_width
    )

    return out


def plot_scale_table(table: pd.DataFrame, outdir: Path) -> None:
    centers = []
    labels = []

    for _, row in table.iterrows():
        lo = row["soft_b_pt_low"]
        hi = row["soft_b_pt_high"]

        if np.isneginf(lo):
            center = hi
            label = f"<{hi:.1f}"
        elif np.isposinf(hi):
            center = lo
            label = f">{lo:.1f}"
        else:
            center = 0.5 * (lo + hi)
            label = f"{lo:.1f}-{hi:.1f}"

        centers.append(center)
        labels.append(label)

    plt.figure(figsize=(8, 5))
    plt.plot(centers, table["raw_scale_factor"], marker="o", label="Raw bin scale")
    plt.plot(centers, table["scale_factor"], marker="s", label="Applied clipped scale")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel(r"Softer selected b-jet $p_T$ [GeV]")
    plt.ylabel(r"$m_{bb}$ scale factor")
    plt.title("Chosen pT-binned correction: 6 bins, clip 0.95–1.20")
    plt.xticks(centers, labels, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "chosen_pt_binned_scale_factors.png", dpi=220)
    plt.close()


def plot_signal_validation_variants(df_pre: pd.DataFrame, mass_cols: list[str], outdir: Path) -> None:
    sig_val = df_pre[(df_pre["process_group"] == "signal") & (~df_pre["is_calib"])]
    bins = np.linspace(0, 260, 65)

    plt.figure(figsize=(9, 6))

    for col in mass_cols:
        x = sig_val[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        med = np.median(x)
        q16, q84, w68 = width68(x)
        label = f"{LABELS[col]}: median={med:.1f}, width68/med={w68/med:.3f}"
        plt.hist(x, bins=bins, histtype="step", density=True, linewidth=2, label=label)

    plt.axvline(90, linestyle="--", linewidth=1)
    plt.axvline(125, linestyle="-", linewidth=1)
    plt.axvline(150, linestyle="--", linewidth=1)
    plt.xlabel(r"$m_{bb}$ [GeV]")
    plt.ylabel("Normalized signal validation events")
    plt.title("Signal validation mass-shape comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "signal_validation_mbb_variant_overlay.png", dpi=220)
    plt.close()


def plot_signal_background_overlay(df_pre: pd.DataFrame, mass_col: str, outdir: Path) -> None:
    bins = np.linspace(0, 350, 71)

    plt.figure(figsize=(9, 6))

    for group in ["signal", "all_background", "single_higgs", "HH_other", "diboson", "ttbar", "ttV_ttH_tttt"]:
        if group == "all_background":
            x = df_pre.loc[df_pre["process_group"] != "signal", mass_col].to_numpy(dtype=float)
        else:
            x = df_pre.loc[df_pre["process_group"] == group, mass_col].to_numpy(dtype=float)

        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue

        plt.hist(
            x,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=2,
            label=f"{LABELS.get(group, group)} (n={len(x)})",
        )

    plt.axvline(80, linestyle=":", linewidth=1)
    plt.axvline(90, linestyle="--", linewidth=1)
    plt.axvline(125, linestyle="-", linewidth=1)
    plt.axvline(150, linestyle=":", linewidth=1)
    plt.xlabel(r"$m_{bb}$ [GeV]")
    plt.ylabel("Normalized events")
    plt.title(LABELS[mass_col])
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / f"{mass_col}_signal_vs_background_overlay.png", dpi=220)
    plt.close()


def plot_signal_fraction(df_pre: pd.DataFrame, mass_cols: list[str], outdir: Path) -> None:
    bins = np.linspace(0, 350, 36)
    is_signal = df_pre["process_group"].to_numpy() == "signal"

    plt.figure(figsize=(9, 6))

    for col in mass_cols:
        mass = df_pre[col].to_numpy(dtype=float)

        s_counts, edges = np.histogram(mass[is_signal], bins=bins)
        b_counts, _ = np.histogram(mass[~is_signal], bins=bins)
        denom = s_counts + b_counts

        frac = np.divide(
            s_counts,
            denom,
            out=np.zeros_like(s_counts, dtype=float),
            where=denom > 0,
        )
        centers = 0.5 * (edges[1:] + edges[:-1])

        plt.step(centers, frac, where="mid", linewidth=2, label=LABELS[col])

    plt.axvline(80, linestyle=":", linewidth=1)
    plt.axvline(90, linestyle="--", linewidth=1)
    plt.axvline(125, linestyle="-", linewidth=1)
    plt.axvline(150, linestyle=":", linewidth=1)
    plt.xlabel(r"$m_{bb}$ [GeV]")
    plt.ylabel(r"Raw $S/(S+B)$ per bin")
    plt.title("Signal fraction by mass variant")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "signal_fraction_by_mbb_variant.png", dpi=220)
    plt.close()


def plot_summary_bars(comp: pd.DataFrame, outdir: Path) -> None:
    labels = comp["label"].tolist()
    x = np.arange(len(labels))

    plt.figure(figsize=(8, 5))
    plt.bar(x, comp["asimovZ_raw"])
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Best raw Asimov Z")
    plt.title("Best raw sensitivity proxy by mass variant")
    plt.tight_layout()
    plt.savefig(outdir / "comparison_best_asimovZ.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(x, comp["signal_val_width68_over_median"])
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Signal validation width68 / median")
    plt.title("Signal validation mass resolution by mass variant")
    plt.tight_layout()
    plt.savefig(outdir / "comparison_signal_validation_width.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(x, comp["S_over_B"])
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Raw S/B in best window")
    plt.title("Raw purity by mass variant")
    plt.tight_layout()
    plt.savefig(outdir / "comparison_best_SoverB.png", dpi=220)
    plt.close()


def write_summary_md(
    outdir: Path,
    comp: pd.DataFrame,
    table: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    best = comp.sort_values("asimovZ_raw", ascending=False).iloc[0]
    unc = comp[comp["mass_variant"] == "mbb_uncorrected"].iloc[0]
    glob = comp[comp["mass_variant"] == "mbb_global_median_scaled"].iloc[0]
    pt = comp[comp["mass_variant"] == "mbb_pt_binned_scaled"].iloc[0]

    lines = []
    lines.append("# Final m_bb correction summary before classifier/DNN\n")
    lines.append("## Selection\n")
    lines.append("- Channel: one lepton, 3 <= n_jets <= 5, second b-tag flag = 1.")
    lines.append("- Correction calibrated on half of the signal events using a deterministic split.")
    lines.append("- Signal validation metrics are computed on the other half of signal events.")
    lines.append("- Backgrounds are not used to derive the correction.")
    lines.append("- All yields and significances are raw MC-count diagnostics, not luminosity-normalized.\n")

    lines.append("## Chosen correction\n")
    lines.append(f"- Softer-b-jet pT-binned correction with n_bins = {args.n_bins}.")
    lines.append(f"- Applied scale factor clipped to [{args.clip_min}, {args.clip_max}].")
    lines.append(f"- Target mass = {args.target_mass} GeV.\n")

    lines.append("## Main result\n")
    lines.append(
        f"- Best variant by raw Asimov Z: **{best['label']}**, "
        f"window {best['best_window']} GeV."
    )
    lines.append(
        f"- Uncorrected: S = {int(unc['S_raw'])}, B = {int(unc['B_raw'])}, "
        f"S/B = {unc['S_over_B']:.4f}, Z = {unc['asimovZ_raw']:.3f}."
    )
    lines.append(
        f"- Global median scaled: S = {int(glob['S_raw'])}, B = {int(glob['B_raw'])}, "
        f"S/B = {glob['S_over_B']:.4f}, Z = {glob['asimovZ_raw']:.3f}."
    )
    lines.append(
        f"- pT-binned scaled: S = {int(pt['S_raw'])}, B = {int(pt['B_raw'])}, "
        f"S/B = {pt['S_over_B']:.4f}, Z = {pt['asimovZ_raw']:.3f}."
    )
    lines.append(
        f"- Relative raw-Z improvement of pT-binned over uncorrected: "
        f"{pt['relative_asimovZ_improvement_percent']:.2f}%."
    )
    lines.append(
        f"- Signal-validation width68/median changed from "
        f"{unc['signal_val_width68_over_median']:.3f} to "
        f"{pt['signal_val_width68_over_median']:.3f}."
    )
    lines.append(
        f"- Relative signal-validation width improvement: "
        f"{pt['relative_signal_val_width_improvement_percent']:.2f}%.\n"
    )

    lines.append("## Correction table\n")
    lines.append(table.to_markdown(index=False))
    lines.append("\n## Compact comparison table\n")
    compact = comp[
        [
            "label",
            "best_window",
            "S_raw",
            "B_raw",
            "S_over_B",
            "asimovZ_raw",
            "signal_val_median",
            "signal_val_width68_over_median",
            "bkg_frac_100_150",
            "relative_asimovZ_improvement_percent",
            "relative_signal_val_width_improvement_percent",
        ]
    ].copy()
    lines.append(compact.to_markdown(index=False))

    (outdir / "final_mbb_correction_summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skim",
        default="outputs/all_background_reco_skim/all_processes_reco_skim.parquet",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/final_mbb_correction_summary",
    )
    parser.add_argument("--n-bins", type=int, default=6)
    parser.add_argument("--clip-min", type=float, default=0.95)
    parser.add_argument("--clip-max", type=float, default=1.20)
    parser.add_argument("--target-mass", type=float, default=125.0)
    parser.add_argument("--min-signal-per-bin", type=int, default=25)
    parser.add_argument("--ht-min", type=float, default=100.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    plotdir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.skim)
    df = add_deterministic_split(df)

    pre_mask = make_preselection(df)
    df_pre = df[pre_mask].copy()

    df_pre["soft_b_pt"] = np.minimum(df_pre["b1_pt"], df_pre["b2_pt"])
    df_pre["mbb_uncorrected"] = df_pre["mbb_top2_btag"]

    sig_calib = df_pre[
        (df_pre["process_group"] == "signal") & (df_pre["is_calib"])
    ]

    global_median = float(np.median(sig_calib["mbb_uncorrected"]))
    global_scale = args.target_mass / global_median
    df_pre["mbb_global_median_scaled"] = df_pre["mbb_uncorrected"] * global_scale

    table = calibrate_pt_binned_table(
        df_pre=df_pre,
        n_bins=args.n_bins,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        target_mass=args.target_mass,
        min_signal_per_bin=args.min_signal_per_bin,
    )
    df_pre["mbb_pt_binned_scaled"] = apply_pt_binned_table(df_pre, table)

    mass_cols = [
        "mbb_uncorrected",
        "mbb_global_median_scaled",
        "mbb_pt_binned_scaled",
    ]

    # Outputs
    table.to_csv(outdir / "chosen_pt_binned_correction_table.csv", index=False)

    window_scan = pd.concat(
        [scan_windows(df_pre, col, args.ht_min) for col in mass_cols],
        ignore_index=True,
    )
    window_scan.to_csv(outdir / "final_mass_window_scan.csv", index=False)

    shape = make_shape_rows(df_pre, mass_cols)
    shape.to_csv(outdir / "final_shape_metrics.csv", index=False)

    comp = make_final_comparison_table(window_scan, shape)
    comp.to_csv(outdir / "final_mbb_correction_comparison.csv", index=False)

    compact = comp[
        [
            "label",
            "best_window",
            "S_raw",
            "B_raw",
            "S_over_B",
            "asimovZ_raw",
            "signal_val_median",
            "signal_val_width68_over_median",
            "bkg_frac_100_150",
            "relative_asimovZ_improvement_percent",
            "relative_signal_val_width_improvement_percent",
        ]
    ].copy()
    compact.to_csv(outdir / "final_mbb_correction_compact_table.csv", index=False)

    try:
        compact.to_latex(
            outdir / "final_mbb_correction_compact_table.tex",
            index=False,
            float_format="%.4f",
        )
    except Exception as exc:
        print("[warning] Could not write LaTeX table:", exc)

    keep_cols = [
        "sample",
        "process_group",
        "file_index",
        "event_in_file",
        "is_calib",
        "n_jets",
        "n_leptons",
        "ht",
        "met",
        "b1_pt",
        "b2_pt",
        "soft_b_pt",
        "mbb_uncorrected",
        "mbb_global_median_scaled",
        "mbb_pt_binned_scaled",
    ]
    df_pre[keep_cols].to_parquet(
        outdir / "onelep_pre_mbb_final_mass_variants.parquet",
        index=False,
    )

    plot_scale_table(table, plotdir)
    plot_signal_validation_variants(df_pre, mass_cols, plotdir)
    for col in mass_cols:
        plot_signal_background_overlay(df_pre, col, plotdir)
    plot_signal_fraction(df_pre, mass_cols, plotdir)
    plot_summary_bars(comp, plotdir)

    write_summary_md(outdir, comp, table, args)

    print("\nPre-mass selected events:", len(df_pre))
    print("Signal:", int((df_pre["process_group"] == "signal").sum()))
    print("Background:", int((df_pre["process_group"] != "signal").sum()))
    print("Signal calibration:", int(((df_pre["process_group"] == "signal") & df_pre["is_calib"]).sum()))
    print("Signal validation:", int(((df_pre["process_group"] == "signal") & ~df_pre["is_calib"]).sum()))

    print("\n=== Chosen correction table ===")
    print(table.to_string(index=False))

    print("\n=== Final compact comparison ===")
    print(compact.to_string(index=False))

    print("\nWrote outputs to:", outdir)
    print("Main summary:", outdir / "final_mbb_correction_summary.md")
    print("Main table:", outdir / "final_mbb_correction_compact_table.csv")
    print("Plots:", plotdir)


if __name__ == "__main__":
    main()
