'''
Bin by bin scan of pT-binned m_bb correction scale factors, with clipping to avoid extreme corrections in sparse bins.
The scan is performed over a range of clip_min and clip_max values, and the best configuration is selected based on the Asimov Z significance of the signal in the baseline region.
The output is a CSV file with the scan results, as well as plots of the best Asimov Z vs clip range, and the signal validation width vs best Asimov Z.

'''
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_WINDOWS = [
    (80, 150),
    (90, 140),
    (95, 145),
    (100, 140),
    (100, 150),
    (105, 145),
]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_clip_pairs(s: str) -> list[tuple[float, float]]:
    pairs = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        lo, hi = item.split(":")
        pairs.append((float(lo), float(hi)))
    return pairs


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


def calibrate_table(
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


def apply_table(df_pre: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
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


def window_scan_rows(
    df_pre: pd.DataFrame,
    mass_values: np.ndarray,
    mass_variant: str,
    windows: list[tuple[int, int]],
    ht_min: float,
) -> list[dict]:
    rows = []

    for lo, hi in windows:
        mask = (
            (mass_values >= lo)
            & (mass_values <= hi)
            & (df_pre["ht"].to_numpy(dtype=float) >= ht_min)
        )

        is_signal = df_pre["process_group"].to_numpy() == "signal"

        s = int(np.sum(mask & is_signal))
        b = int(np.sum(mask & ~is_signal))

        rows.append(
            {
                "mass_variant": mass_variant,
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

    return rows


def add_shape_prefix(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def summarize_variant(
    df_pre: pd.DataFrame,
    mass_values: np.ndarray,
    mass_variant: str,
    windows: list[tuple[int, int]],
    ht_min: float,
) -> dict:
    scan_rows = window_scan_rows(df_pre, mass_values, mass_variant, windows, ht_min)
    best = sorted(scan_rows, key=lambda r: r["asimovZ_raw"], reverse=True)[0]

    is_signal = df_pre["process_group"].to_numpy() == "signal"
    is_calib = df_pre["is_calib"].to_numpy(dtype=bool)

    sig_all = shape_metrics(mass_values[is_signal])
    sig_calib = shape_metrics(mass_values[is_signal & is_calib])
    sig_val = shape_metrics(mass_values[is_signal & ~is_calib])
    bkg = shape_metrics(mass_values[~is_signal])

    row = {
        "mass_variant": mass_variant,
        **{f"best_{k}": v for k, v in best.items() if k != "mass_variant"},
        **add_shape_prefix("signal_all", sig_all),
        **add_shape_prefix("signal_calib", sig_calib),
        **add_shape_prefix("signal_val", sig_val),
        **add_shape_prefix("bkg", bkg),
    }

    return row, scan_rows


def plot_summary(summary: pd.DataFrame, outdir: Path) -> None:
    plotdir = outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)

    pt = summary[summary["mass_variant"] == "mbb_pt_binned_scaled"].copy()
    if pt.empty:
        return

    pt["clip_label"] = pt.apply(
        lambda r: f"{r['clip_min']:.2f}-{r['clip_max']:.2f}",
        axis=1,
    )

    # Plot best Asimov Z by clip range for each n_bins.
    plt.figure(figsize=(11, 6))

    labels = sorted(pt["clip_label"].unique())
    x = np.arange(len(labels))

    for n_bins, sub in pt.groupby("n_bins"):
        y = []
        for label in labels:
            match = sub[sub["clip_label"] == label]
            y.append(match["best_asimovZ_raw"].max() if len(match) else np.nan)

        plt.plot(x, y, marker="o", label=f"n_bins={n_bins}")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Best raw Asimov Z over scanned mass windows")
    plt.xlabel("clip_min-clip_max")
    plt.title("pT-binned correction scan")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotdir / "best_asimovZ_vs_clip_range.png", dpi=200)
    plt.close()

    # Plot signal validation width vs best Asimov.
    plt.figure(figsize=(8, 6))
    for n_bins, sub in pt.groupby("n_bins"):
        plt.scatter(
            sub["signal_val_width68_over_median"],
            sub["best_asimovZ_raw"],
            label=f"n_bins={n_bins}",
        )

    plt.xlabel("Signal validation width68 / median")
    plt.ylabel("Best raw Asimov Z")
    plt.title("Mass-shape vs signal-extraction tradeoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotdir / "asimovZ_vs_signal_validation_width.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skim",
        default="outputs/all_background_reco_skim/all_processes_reco_skim.parquet",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/pt_binned_clip_scan",
    )
    parser.add_argument(
        "--n-bins-list",
        default="4,5,6,8,10",
        help="Comma-separated pT-bin counts to test.",
    )
    parser.add_argument(
        "--clip-pairs",
        default=(
            "0.75:1.50,"
            "0.80:1.40,"
            "0.80:1.30,"
            "0.85:1.35,"
            "0.85:1.30,"
            "0.85:1.25,"
            "0.90:1.30,"
            "0.90:1.25,"
            "0.90:1.20,"
            "0.95:1.20,"
            "0.95:1.15"
        ),
        help="Comma-separated clip ranges like 0.85:1.30,0.90:1.25",
    )
    parser.add_argument("--target-mass", type=float, default=125.0)
    parser.add_argument("--min-signal-per-bin", type=int, default=25)
    parser.add_argument("--ht-min", type=float, default=100.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_bins_list = parse_int_list(args.n_bins_list)
    clip_pairs = parse_clip_pairs(args.clip_pairs)

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

    print("Pre-mass selected events:", len(df_pre))
    print("Signal:", int((df_pre["process_group"] == "signal").sum()))
    print("Background:", int((df_pre["process_group"] != "signal").sum()))
    print("Signal calibration events:", int(((df_pre["process_group"] == "signal") & df_pre["is_calib"]).sum()))
    print("Signal validation events:", int(((df_pre["process_group"] == "signal") & ~df_pre["is_calib"]).sum()))
    print("Global calibration median:", global_median)
    print("Global scale:", global_scale)

    all_summary_rows = []
    all_window_rows = []
    all_table_rows = []

    # Reference rows: uncorrected and global median scaled.
    for mass_variant in ["mbb_uncorrected", "mbb_global_median_scaled"]:
        mass_values = df_pre[mass_variant].to_numpy(dtype=float)
        row, scan_rows = summarize_variant(
            df_pre,
            mass_values,
            mass_variant,
            DEFAULT_WINDOWS,
            args.ht_min,
        )
        row.update(
            {
                "tag": "reference",
                "n_bins": 0,
                "clip_min": np.nan,
                "clip_max": np.nan,
                "n_clipped_low": 0,
                "n_clipped_high": 0,
                "scale_min": np.nan,
                "scale_max": np.nan,
                "scale_range": np.nan,
                "max_adjacent_scale_jump": np.nan,
            }
        )
        all_summary_rows.append(row)

        for scan_row in scan_rows:
            scan_row.update(
                {
                    "tag": "reference",
                    "n_bins": 0,
                    "clip_min": np.nan,
                    "clip_max": np.nan,
                }
            )
            all_window_rows.append(scan_row)

    # pT-binned scan.
    for n_bins in n_bins_list:
        for clip_min, clip_max in clip_pairs:
            tag = f"nbins{n_bins}_clip{clip_min:.2f}_{clip_max:.2f}".replace(".", "p")
            print("Scanning", tag)

            table = calibrate_table(
                df_pre,
                n_bins=n_bins,
                clip_min=clip_min,
                clip_max=clip_max,
                target_mass=args.target_mass,
                min_signal_per_bin=args.min_signal_per_bin,
            )

            corrected = apply_table(df_pre, table)

            row, scan_rows = summarize_variant(
                df_pre,
                corrected,
                "mbb_pt_binned_scaled",
                DEFAULT_WINDOWS,
                args.ht_min,
            )

            scales = table["scale_factor"].to_numpy(dtype=float)
            adjacent_jumps = np.abs(np.diff(scales)) if len(scales) > 1 else np.array([0.0])

            row.update(
                {
                    "tag": tag,
                    "n_bins": n_bins,
                    "clip_min": clip_min,
                    "clip_max": clip_max,
                    "n_clipped_low": int(table["was_clipped_low"].sum()),
                    "n_clipped_high": int(table["was_clipped_high"].sum()),
                    "scale_min": float(np.min(scales)),
                    "scale_max": float(np.max(scales)),
                    "scale_range": float(np.max(scales) - np.min(scales)),
                    "max_adjacent_scale_jump": float(np.max(adjacent_jumps)),
                }
            )
            all_summary_rows.append(row)

            for scan_row in scan_rows:
                scan_row.update(
                    {
                        "tag": tag,
                        "n_bins": n_bins,
                        "clip_min": clip_min,
                        "clip_max": clip_max,
                    }
                )
                all_window_rows.append(scan_row)

            table = table.copy()
            table["tag"] = tag
            table["n_bins"] = n_bins
            table["clip_min"] = clip_min
            table["clip_max"] = clip_max
            all_table_rows.append(table)

    summary = pd.DataFrame(all_summary_rows)
    windows = pd.DataFrame(all_window_rows)
    correction_tables = pd.concat(all_table_rows, ignore_index=True) if all_table_rows else pd.DataFrame()

    summary.to_csv(outdir / "clip_scan_summary.csv", index=False)
    windows.to_csv(outdir / "clip_scan_all_windows.csv", index=False)
    correction_tables.to_csv(outdir / "clip_scan_correction_tables.csv", index=False)

    plot_summary(summary, outdir)

    print("\n=== Best overall by raw Asimov Z ===")
    print(
        summary.sort_values("best_asimovZ_raw", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    print("\n=== Best pT-binned only by raw Asimov Z ===")
    pt = summary[summary["mass_variant"] == "mbb_pt_binned_scaled"]
    print(
        pt.sort_values("best_asimovZ_raw", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    print("\n=== Best pT-binned with gentler correction sanity filter ===")
    filtered = pt[
        (pt["scale_range"] <= 0.50)
        & (pt["max_adjacent_scale_jump"] <= 0.25)
        & (pt["signal_val_width68_over_median"] <= 0.36)
    ]
    if len(filtered):
        print(
            filtered.sort_values("best_asimovZ_raw", ascending=False)
            .head(20)
            .to_string(index=False)
        )
    else:
        print("No pT-binned config passed the sanity filter.")

    print("\nWrote:")
    print(outdir / "clip_scan_summary.csv")
    print(outdir / "clip_scan_all_windows.csv")
    print(outdir / "clip_scan_correction_tables.csv")
    print(outdir / "plots")


if __name__ == "__main__":
    main()