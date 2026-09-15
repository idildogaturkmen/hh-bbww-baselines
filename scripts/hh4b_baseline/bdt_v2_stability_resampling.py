'''
Purpose: Evaluate the stability of BDT v2 categories using Poisson bootstrap resampling.
This script reads in the BDT v2 test scores and evaluates the stability of various selections and categories by performing Poisson bootstrap resampling. It computes various metrics such as Asimov significance, effective number of background events, and influential background events. 
The results are saved to CSV files and a Markdown interpretation file is generated summarizing the findings.
Usage:
    python scripts/hh4b_baseline/bdt_v2_stability_resampling.py --scores <path_to_scores_csv> --outdir <output_directory> [options]
Options:
    --n-bootstrap: Number of bootstrap replicates to generate (default: 2000)       
'''

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def asimov_z(s: float, b: float) -> float:
    if s <= 0 or b <= 0:
        return 0.0
    return math.sqrt(max(0.0, 2.0 * ((s + b) * math.log(1.0 + s / b) - s)))


def neff(w) -> float:
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w)]
    if len(w) == 0 or np.sum(w * w) <= 0:
        return 0.0
    return float(np.sum(w) ** 2 / np.sum(w * w))


def two_b_over_s(s: float, b: float) -> float:
    if s <= 0:
        return float("inf")
    return float(2.0 * b / s)


def summarize_nominal(df: pd.DataFrame, mask: np.ndarray, selection: str, category_type: str) -> dict:
    sub = df.loc[mask].copy()
    sub = sub[np.isfinite(sub["physics_weight"])].copy()

    sig = sub["is_signal"].to_numpy(dtype=bool)
    bkg = ~sig

    sw = float(sub.loc[sig, "physics_weight"].sum())
    bw = float(sub.loc[bkg, "physics_weight"].sum())

    bkg_weights = sub.loc[bkg, "physics_weight"].to_numpy(dtype=float)

    if bw > 0 and len(bkg_weights) > 0:
        max_single_bkg_weight = float(np.max(bkg_weights))
        max_single_bkg_frac = float(max_single_bkg_weight / bw)
    else:
        max_single_bkg_weight = 0.0
        max_single_bkg_frac = 0.0

    if bw > 0 and bkg.sum() > 0:
        by_sample = (
            sub.loc[bkg]
            .groupby(["sample", "process_group"], dropna=False)["physics_weight"]
            .sum()
            .reset_index()
            .sort_values("physics_weight", ascending=False)
        )
        top_sample = str(by_sample.iloc[0]["sample"])
        top_sample_group = str(by_sample.iloc[0]["process_group"])
        top_sample_b_w = float(by_sample.iloc[0]["physics_weight"])
        top_sample_frac = float(top_sample_b_w / bw)

        by_group = (
            sub.loc[bkg]
            .groupby("process_group", dropna=False)["physics_weight"]
            .sum()
            .reset_index()
            .sort_values("physics_weight", ascending=False)
        )
        top_group = str(by_group.iloc[0]["process_group"])
        top_group_b_w = float(by_group.iloc[0]["physics_weight"])
        top_group_frac = float(top_group_b_w / bw)
    else:
        top_sample = ""
        top_sample_group = ""
        top_sample_b_w = 0.0
        top_sample_frac = 0.0
        top_group = ""
        top_group_b_w = 0.0
        top_group_frac = 0.0

    return {
        "selection": selection,
        "category_type": category_type,
        "raw_signal": int(sig.sum()),
        "raw_background": int(bkg.sum()),
        "S_w": sw,
        "B_w": bw,
        "S_over_B": float(sw / bw) if bw > 0 else 0.0,
        "S_over_SplusB": float(sw / (sw + bw)) if (sw + bw) > 0 else 0.0,
        "S_over_sqrtSplusB": float(sw / math.sqrt(sw + bw)) if (sw + bw) > 0 else 0.0,
        "asimov_Z_A": asimov_z(sw, bw),
        "N_eff_bkg": neff(bkg_weights),
        "twoB_over_S": two_b_over_s(sw, bw),
        "max_single_bkg_weight": max_single_bkg_weight,
        "max_single_bkg_frac": max_single_bkg_frac,
        "top_sample": top_sample,
        "top_sample_group": top_sample_group,
        "top_sample_B_w": top_sample_b_w,
        "top_sample_frac_B_w": top_sample_frac,
        "top_process_group": top_group,
        "top_process_group_B_w": top_group_b_w,
        "top_process_group_frac_B_w": top_group_frac,
    }


def poisson_bootstrap_selection(
    df: pd.DataFrame,
    mask: np.ndarray,
    selection: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    sub = df.loc[mask].copy()
    sub = sub[np.isfinite(sub["physics_weight"])].copy()

    sig_w = sub.loc[sub["is_signal"].astype(bool), "physics_weight"].to_numpy(dtype=float)
    bkg_w = sub.loc[~sub["is_signal"].astype(bool), "physics_weight"].to_numpy(dtype=float)

    rows = []

    for i in range(n_bootstrap):
        if len(sig_w) > 0:
            sig_mult = rng.poisson(1.0, size=len(sig_w))
            sw = float(np.sum(sig_mult * sig_w))
        else:
            sw = 0.0

        if len(bkg_w) > 0:
            bkg_mult = rng.poisson(1.0, size=len(bkg_w))
            bw = float(np.sum(bkg_mult * bkg_w))
        else:
            bw = 0.0

        rows.append(
            {
                "selection": selection,
                "replicate": i,
                "S_w": sw,
                "B_w": bw,
                "S_over_B": float(sw / bw) if bw > 0 else 0.0,
                "S_over_sqrtSplusB": float(sw / math.sqrt(sw + bw)) if (sw + bw) > 0 else 0.0,
                "asimov_Z_A": asimov_z(sw, bw),
                "twoB_over_S": two_b_over_s(sw, bw),
            }
        )

    return pd.DataFrame(rows)


def bootstrap_summary(reps: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for selection, g in reps.groupby("selection"):
        z = g["asimov_Z_A"].to_numpy(dtype=float)
        sb = g["S_over_B"].to_numpy(dtype=float)
        bw = g["B_w"].to_numpy(dtype=float)
        sw = g["S_w"].to_numpy(dtype=float)

        z_p16, z_p50, z_p84 = np.quantile(z, [0.16, 0.50, 0.84])
        sb_p16, sb_p50, sb_p84 = np.quantile(sb, [0.16, 0.50, 0.84])
        bw_p16, bw_p50, bw_p84 = np.quantile(bw, [0.16, 0.50, 0.84])
        sw_p16, sw_p50, sw_p84 = np.quantile(sw, [0.16, 0.50, 0.84])

        z_rel_half_width = float((z_p84 - z_p16) / (2.0 * z_p50)) if z_p50 > 0 else float("inf")
        b_rel_half_width = float((bw_p84 - bw_p16) / (2.0 * bw_p50)) if bw_p50 > 0 else float("inf")

        rows.append(
            {
                "selection": selection,
                "Z_A_boot_p16": float(z_p16),
                "Z_A_boot_p50": float(z_p50),
                "Z_A_boot_p84": float(z_p84),
                "Z_A_boot_rel_half_width": z_rel_half_width,
                "S_over_B_boot_p16": float(sb_p16),
                "S_over_B_boot_p50": float(sb_p50),
                "S_over_B_boot_p84": float(sb_p84),
                "S_w_boot_p16": float(sw_p16),
                "S_w_boot_p50": float(sw_p50),
                "S_w_boot_p84": float(sw_p84),
                "B_w_boot_p16": float(bw_p16),
                "B_w_boot_p50": float(bw_p50),
                "B_w_boot_p84": float(bw_p84),
                "B_w_boot_rel_half_width": b_rel_half_width,
            }
        )

    return pd.DataFrame(rows)


def influential_background_events(df: pd.DataFrame, mask: np.ndarray, selection: str, top_n: int = 20) -> pd.DataFrame:
    sub = df.loc[mask].copy()
    sub = sub[(sub["is_signal"] == 0) & np.isfinite(sub["physics_weight"])].copy()

    if len(sub) == 0:
        return pd.DataFrame()

    total_b = float(sub["physics_weight"].sum())
    sub["selection"] = selection
    sub["fraction_of_B_w"] = sub["physics_weight"] / total_b if total_b > 0 else 0.0

    keep = [
        "selection",
        "event_key",
        "sample",
        "process_group",
        "physics_weight",
        "fraction_of_B_w",
        "bdt_score",
        "category_hmass_70_190",
        "category_hmass_90_160",
        "category_vbf_like",
        "category_boosted_1ak8_2b",
    ]

    keep = [c for c in keep if c in sub.columns]

    return sub.sort_values("physics_weight", ascending=False).head(top_n)[keep]


def build_masks(df: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    score = df["bdt_score"].to_numpy(dtype=float)

    masks = []

    masks.append(("cut_all_resolved_4b", "cut_baseline", np.ones(len(df), dtype=bool)))

    if "category_hmass_70_190" in df.columns:
        masks.append(
            (
                "cut_hmass_70_190",
                "cut_baseline",
                df["category_hmass_70_190"].astype(bool).to_numpy(),
            )
        )

    if "category_hmass_90_160" in df.columns:
        masks.append(
            (
                "cut_hmass_90_160",
                "cut_baseline",
                df["category_hmass_90_160"].astype(bool).to_numpy(),
            )
        )

    if "category_vbf_like" in df.columns:
        masks.append(
            (
                "cut_vbf_like",
                "cut_baseline",
                df["category_vbf_like"].astype(bool).to_numpy(),
            )
        )

    # Cumulative BDT thresholds.
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.74, 0.76]:
        masks.append(
            (
                f"bdt_score_ge_{thr:.2f}",
                "bdt_cumulative",
                score >= thr,
            )
        )

    # Non-overlapping BDT bins.
    bins = [
        (0.30, 0.40),
        (0.40, 0.50),
        (0.50, 0.60),
        (0.60, 0.68),
        (0.68, 0.74),
        (0.74, None),
    ]

    for lo, hi in bins:
        if hi is None:
            mask = score >= lo
            name = f"bdt_bin_{lo:.2f}_inf"
        else:
            mask = (score >= lo) & (score < hi)
            name = f"bdt_bin_{lo:.2f}_{hi:.2f}"

        masks.append((name, "bdt_nonoverlap_bin", mask))

    return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        default="outputs/hh4b_baseline/bdt_v2/hh4b_resolved_bdt_v2_test_scores.csv",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/hh4b_baseline/bdt_v2_stability",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--min-neff", type=float, default=5.0)
    parser.add_argument("--min-raw-bkg", type=int, default=20)
    parser.add_argument("--min-raw-sig", type=int, default=5)
    parser.add_argument("--max-single-frac", type=float, default=0.30)
    parser.add_argument("--max-sample-frac", type=float, default=0.70)
    parser.add_argument("--max-z-rel-half-width", type=float, default=0.60)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.scores)
    df = df[np.isfinite(df["physics_weight"])].copy().reset_index(drop=True)

    print("[INFO] Loaded:", args.scores)
    print("[INFO] Rows:", len(df))
    print("[INFO] Signal/background:")
    print(df["is_signal"].value_counts().rename(index={0: "background", 1: "signal"}).to_string())

    masks = build_masks(df)

    nominal_rows = []
    replicate_tables = []
    influence_tables = []

    for selection, category_type, mask in masks:
        print(f"[INFO] Resampling {selection} ({category_type}), selected rows = {int(np.sum(mask))}")

        nominal_rows.append(summarize_nominal(df, mask, selection, category_type))

        reps = poisson_bootstrap_selection(
            df=df,
            mask=mask,
            selection=selection,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        )
        replicate_tables.append(reps)

        infl = influential_background_events(df, mask, selection, top_n=20)
        if len(infl) > 0:
            influence_tables.append(infl)

    nominal = pd.DataFrame(nominal_rows)
    reps_all = pd.concat(replicate_tables, ignore_index=True)
    boot = bootstrap_summary(reps_all)

    summary = nominal.merge(boot, on="selection", how="left")

    summary["passes_stability"] = (
        (summary["raw_signal"] >= args.min_raw_sig)
        & (summary["raw_background"] >= args.min_raw_bkg)
        & (summary["N_eff_bkg"] >= args.min_neff)
        & (summary["max_single_bkg_frac"] <= args.max_single_frac)
        & (summary["top_sample_frac_B_w"] <= args.max_sample_frac)
        & (summary["Z_A_boot_rel_half_width"] <= args.max_z_rel_half_width)
    )

    # Useful ranking: stable first, then high bootstrap median Z, then high purity.
    ranked = summary.sort_values(
        [
            "passes_stability",
            "Z_A_boot_p50",
            "asimov_Z_A",
            "S_over_B",
            "N_eff_bkg",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    recommended = ranked[ranked["passes_stability"]].copy()

    if len(influence_tables) > 0:
        influence = pd.concat(influence_tables, ignore_index=True)
    else:
        influence = pd.DataFrame()

    nominal.to_csv(outdir / "bdt_v2_stability_nominal_summary.csv", index=False)
    reps_all.to_csv(outdir / "bdt_v2_stability_bootstrap_replicates.csv", index=False)
    boot.to_csv(outdir / "bdt_v2_stability_bootstrap_summary.csv", index=False)
    summary.to_csv(outdir / "bdt_v2_stability_summary.csv", index=False)
    ranked.to_csv(outdir / "bdt_v2_stability_ranked_categories.csv", index=False)
    recommended.to_csv(outdir / "bdt_v2_stability_recommended_categories.csv", index=False)
    influence.to_csv(outdir / "bdt_v2_stability_top_influential_background_events.csv", index=False)

    config = {
        "scores": args.scores,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "stability_requirements": {
            "min_neff": args.min_neff,
            "min_raw_bkg": args.min_raw_bkg,
            "min_raw_sig": args.min_raw_sig,
            "max_single_frac": args.max_single_frac,
            "max_sample_frac": args.max_sample_frac,
            "max_z_rel_half_width": args.max_z_rel_half_width,
        },
    }

    with open(outdir / "bdt_v2_stability_config.json", "w") as f:
        json.dump(config, f, indent=2)

    note = []
    note.append("# BDT v2 stability and resampling study\n\n")
    note.append("This study keeps all backgrounds in the nominal evaluation and uses Poisson bootstrap resampling to test finite-MC stability.\n\n")
    note.append("The goal is not to remove difficult backgrounds, but to check whether an apparent BDT improvement is robust or dominated by a few high-weight events.\n\n")
    note.append("## Stability requirements\n\n")
    note.append("```json\n")
    note.append(json.dumps(config["stability_requirements"], indent=2))
    note.append("\n```\n\n")
    note.append("## Top ranked categories\n\n")
    note.append(ranked.head(10).to_markdown(index=False))
    note.append("\n\n")
    if len(recommended) > 0:
        note.append("## Recommended stable categories\n\n")
        note.append(recommended.head(10).to_markdown(index=False))
        note.append("\n")
    else:
        note.append("## Recommended stable categories\n\n")
        note.append("No category passed all stability requirements. This means the current BDT regions are useful diagnostically but should not yet be claimed as robust optimized categories.\n")

    (outdir / "bdt_v2_stability_interpretation.md").write_text("".join(note), encoding="utf-8")

    print("\n[INFO] Top ranked categories:")
    cols = [
        "selection",
        "category_type",
        "passes_stability",
        "raw_signal",
        "raw_background",
        "S_w",
        "B_w",
        "S_over_B",
        "asimov_Z_A",
        "Z_A_boot_p16",
        "Z_A_boot_p50",
        "Z_A_boot_p84",
        "Z_A_boot_rel_half_width",
        "N_eff_bkg",
        "max_single_bkg_frac",
        "top_sample",
        "top_sample_frac_B_w",
    ]
    print(ranked[cols].head(30).to_string(index=False))

    print("\n[INFO] Recommended stable categories:")
    if len(recommended) > 0:
        print(recommended[cols].head(20).to_string(index=False))
    else:
        print("No categories passed all stability requirements.")

    print("\n[INFO] Top influential background events:")
    if len(influence) > 0:
        print(influence.head(50).to_string(index=False))
    else:
        print("No influential-background table.")

    print("\n[INFO] Wrote outputs to:", outdir)


if __name__ == "__main__":
    main()
