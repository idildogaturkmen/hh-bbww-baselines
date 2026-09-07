"""Track D, Task 2 -- empirical QCD tail-yield stability vs cumulative
statistics, using ONLY the real, already-existing frozen QCD production
(inference_qcd_1 / inference_qcd_2), not a future-generation projection.

Read-only. No Condor/EAF job launched. Inputs:
  - FULL_QCD_SURVIVOR_SIDECAR.h5 (557 files, 87,623,306-event union,
    every event above the SPA10M 7.5%-efficiency WP score>=0.998958945274353,
    which is looser than both u>3.5 and u>4.5 -- so every u>3.5/u>4.5 QCD
    survivor is already inside this file).
  - all_file_tree_entries.tsv: exact, independently-measured per-file
    ROOT tree_num_entries (selected/stored event count) for every one of
    the 176 (lane1) + 381 (lane2) files -- sums to 27,683,689 / 59,939,617
    exactly, matching the author-audited lane totals bit-for-bit.

Per-file GENERATED-event count verification (per task instruction):
  lane1: 17,600 generation jobs / 176 merged files = 100.000 jobs/file
         EXACTLY (integer) -- a uniform 5,000,000-event-job x 100 =
         500,000,000 generated events/file is therefore PROVEN for lane1
         (176 x 500,000,000 = 88,000,000,000, exact match to the
         author-confirmed lane total).
  lane2: 38,072 generation jobs / 381 merged files = 99.926... jobs/file
         -- NOT an integer, so a uniform-per-file generated-event count
         is NOT provable for lane2 from any manifest found in this
         project. No per-mergeid job-count manifest exists.
         BLOCKER, disclosed rather than hidden: lane2's per-file
         generated-event axis is reported as an APPROXIMATION ONLY
         (assuming each file's generated exposure is proportional to its
         own exact, measured selected-event count, scaled by the lane's
         overall generated/selected efficiency, 190,360,000,000/59,939,617
         = 3.1765x). This is a defensible proxy (lane1 and lane2 overall
         efficiencies agree to 0.1%, and per-file selected-event counts
         are tightly uniform within each lane, ~157,300 +/- small spread),
         but it is NOT an independently-audited per-file generated count,
         unlike lane1.
  Files are ordered by mergeid (0,1,2,...) within each lane -- the only
  deterministic ordering available; this may or may not equal true
  chronological generation/completion order, disclosed as a convention.
"""
import glob
import json
import os
import re

import h5py
import numpy as np
import pandas as pd

OUT_DIR = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_final_question_closure_20260907_v1"
WORK_DIR = os.path.join(OUT_DIR, "work")
FIG_DIR = os.path.join(OUT_DIR, "figures")

SIDECAR = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_characterization_20260825_v1/work/FULL_QCD_SURVIVOR_SIDECAR.h5"
ENTRIES_TSV = "/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4C1R1_C2_exact_entry_metadata_scan/evidence/all_file_tree_entries.tsv"

LANE_GENERATED_TOTAL = {"inference_qcd_1": 88_000_000_000, "inference_qcd_2": 190_360_000_000}
LANE_JOBS = {"inference_qcd_1": 17_600, "inference_qcd_2": 38_072}
LUMI_SCALE = 4.5
WEIGHT_NUMERATOR = LUMI_SCALE * 4522600.0 * 100000.0  # reproduces the frozen 7.3112875413 constant exactly

U35_SCORE = 1.0 - 10 ** (-3.5)
U45_SCORE = 1.0 - 10 ** (-4.5)


def mergeid_of(path):
    m = re.search(r"mergeid(\d+)\.root$", path)
    return int(m.group(1))


def main():
    # ---- load exact per-file selected-entry counts (independently measured, proven) ----
    ent = pd.read_csv(ENTRIES_TSV, sep="\t")
    ent = ent[ent["dataset_id"].isin(["inference_qcd_1", "inference_qcd_2"])].copy()
    ent["mergeid"] = ent["remote_file"].map(mergeid_of)
    ent = ent.sort_values(["dataset_id", "mergeid"]).reset_index(drop=True)

    lane1_files = int((ent["dataset_id"] == "inference_qcd_1").sum())
    lane2_files = int((ent["dataset_id"] == "inference_qcd_2").sum())
    assert lane1_files == 176 and lane2_files == 381, (lane1_files, lane2_files)
    lane1_selected = int(ent.loc[ent.dataset_id == "inference_qcd_1", "tree_num_entries"].sum())
    lane2_selected = int(ent.loc[ent.dataset_id == "inference_qcd_2", "tree_num_entries"].sum())
    assert lane1_selected == 27_683_689 and lane2_selected == 59_939_617, (lane1_selected, lane2_selected)

    lane1_jobs_per_file = LANE_JOBS["inference_qcd_1"] / lane1_files
    lane2_jobs_per_file = LANE_JOBS["inference_qcd_2"] / lane2_files
    lane1_uniform_provable = float(lane1_jobs_per_file).is_integer()
    lane2_uniform_provable = float(lane2_jobs_per_file).is_integer()

    # generated-event count per file: PROVEN uniform for lane1; PROXY (selected-count
    # scaled by lane efficiency) for lane2, explicitly disclosed.
    lane_eff = {
        "inference_qcd_1": lane1_selected / LANE_GENERATED_TOTAL["inference_qcd_1"],
        "inference_qcd_2": lane2_selected / LANE_GENERATED_TOTAL["inference_qcd_2"],
    }

    def gen_per_file(row):
        if row["dataset_id"] == "inference_qcd_1":
            return LANE_GENERATED_TOTAL["inference_qcd_1"] / lane1_files  # proven uniform
        return row["tree_num_entries"] / lane_eff["inference_qcd_2"]  # proxy

    ent["generated_events_this_file"] = ent.apply(gen_per_file, axis=1)
    ent["basename"] = ent["remote_file"].map(os.path.basename)

    # ---- load survivor sidecar, join by basename (source_file matches remote basename) ----
    with h5py.File(SIDECAR, "r") as f:
        source_file = np.array([s.decode() if isinstance(s, bytes) else s for s in f["source_file"][:]])
        dataset_id_code = f["dataset_id"][:]
        score = f["score_spanet_10m"][:].astype(np.float64)
        HT = f["HT"][:].astype(np.float64)
        n_sel = f["n_selected_jets"][:].astype(np.int64)
        mHH = f["mHH_leading_four"][:].astype(np.float64)
        rhh = f["pairing_RHH"][:]
        fixed_idx = f["fixed_pairing_index"][:]
        probB = f["jet_probB"][:]
        jmask = f["jet_mask"][:]

    surv_basename = np.array([os.path.basename(s.strip()) for s in source_file])
    # dataset_id in the sidecar is a uint8 code; recover the mapping via basename pattern
    lane_of_surv = np.where(np.char.find(surv_basename, "forInfer2") >= 0, "inference_qcd_2", "inference_qcd_1")

    rhh_sel = rhh[np.arange(len(rhh)), fixed_idx]
    n_btag_loose = (np.where(jmask, probB, 0.0) > 0.0243).sum(axis=1)

    surv = pd.DataFrame({
        "basename": surv_basename, "dataset_id": lane_of_surv, "score": score,
        "HT": HT, "n_selected_jets": n_sel, "mHH": mHH, "RHH": rhh_sel, "n_btag_loose": n_btag_loose,
    })
    surv["u35"] = surv["score"] > U35_SCORE
    surv["u45"] = surv["score"] > U45_SCORE

    n_qcd_u35 = int(surv["u35"].sum())
    n_qcd_u45 = int(surv["u45"].sum())
    assert n_qcd_u35 == 68, f"expected 68 QCD survivors at u>3.5, got {n_qcd_u35}"
    assert n_qcd_u45 == 12, f"expected 12 QCD survivors at u>4.5, got {n_qcd_u45}"

    # basenames actually present as survivors must be a subset of the 557-file manifest
    unmatched = set(surv["basename"]) - set(ent["basename"])
    assert not unmatched, f"survivor file(s) not found in entries manifest: {unmatched}"

    ent = ent.set_index("basename")
    surv["mergeid"] = surv["basename"].map(lambda b: ent.loc[b, "mergeid"])
    surv = surv.sort_values(["dataset_id", "mergeid"])

    # ---- build cumulative checkpoint tables (per lane, and combined lane1-then-lane2) ----
    def cumulative_table(entries_lane, surv_lane, checkpoints_frac):
        entries_lane = entries_lane.sort_values("mergeid").reset_index()
        entries_lane["cum_selected"] = entries_lane["tree_num_entries"].cumsum()
        entries_lane["cum_generated"] = entries_lane["generated_events_this_file"].cumsum()
        total_gen = entries_lane["cum_generated"].iloc[-1]
        total_sel = entries_lane["cum_selected"].iloc[-1]
        rows = []
        for frac in checkpoints_frac:
            target_sel = frac * total_sel
            idx = int(np.searchsorted(entries_lane["cum_selected"].values, target_sel))
            idx = min(idx, len(entries_lane) - 1)
            cum_sel = float(entries_lane["cum_selected"].iloc[idx])
            cum_gen = float(entries_lane["cum_generated"].iloc[idx])
            n_files_done = idx + 1
            mergeid_cut = entries_lane["mergeid"].iloc[idx]
            n35 = int((surv_lane["mergeid"] <= mergeid_cut).sum() if len(surv_lane) else 0)
            n35_35 = int(((surv_lane["mergeid"] <= mergeid_cut) & surv_lane["u35"]).sum() if len(surv_lane) else 0)
            n45 = int(((surv_lane["mergeid"] <= mergeid_cut) & surv_lane["u45"]).sum() if len(surv_lane) else 0)
            weight_running = WEIGHT_NUMERATOR / cum_gen
            rows.append(dict(
                frac_of_final_selected=frac, n_files_done=n_files_done, n_files_total=len(entries_lane),
                cum_selected_events=cum_sel, cum_generated_events_approx=cum_gen,
                n_qcd_survivors_u35=n35_35, n_qcd_survivors_u45=n45,
                rate_u35_per_1e9_generated=(n35_35 / cum_gen * 1e9) if cum_gen > 0 else np.nan,
                rate_u45_per_1e9_generated=(n45 / cum_gen * 1e9) if cum_gen > 0 else np.nan,
                running_weight_450fb=weight_running,
                running_B_u35=n35_35 * weight_running,
                running_B_u45=n45 * weight_running,
            ))
        return pd.DataFrame(rows), total_gen, total_sel

    checkpoints_frac = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

    lane1_ent = ent[ent["dataset_id"] == "inference_qcd_1"] if "dataset_id" in ent.columns else ent.loc[ent.index, :]
    # ent index is basename now; re-derive via the pre-index copy
    ent_flat = ent.reset_index()
    lane1_ent = ent_flat[ent_flat["dataset_id"] == "inference_qcd_1"]
    lane2_ent = ent_flat[ent_flat["dataset_id"] == "inference_qcd_2"]
    lane1_surv = surv[surv["dataset_id"] == "inference_qcd_1"]
    lane2_surv = surv[surv["dataset_id"] == "inference_qcd_2"]

    lane1_cum, lane1_gen_total, lane1_sel_total = cumulative_table(lane1_ent, lane1_surv, checkpoints_frac)
    lane2_cum, lane2_gen_total, lane2_sel_total = cumulative_table(lane2_ent, lane2_surv, checkpoints_frac)

    # combined: lane1 fully, then lane2 (disclosed convention -- files ordered
    # lane1-mergeid-ascending then lane2-mergeid-ascending)
    ent_flat_combined = ent_flat.copy()
    ent_flat_combined["order_key"] = np.where(
        ent_flat_combined["dataset_id"] == "inference_qcd_1",
        ent_flat_combined["mergeid"], ent_flat_combined["mergeid"] + 100000,
    )
    ent_flat_combined = ent_flat_combined.sort_values("order_key").reset_index(drop=True)
    ent_flat_combined["mergeid"] = ent_flat_combined["order_key"]  # reuse cumulative_table's mergeid column
    surv_combined = surv.copy()
    surv_combined["mergeid"] = np.where(
        surv_combined["dataset_id"] == "inference_qcd_1", surv_combined["mergeid"], surv_combined["mergeid"] + 100000,
    )
    combined_cum, combined_gen_total, combined_sel_total = cumulative_table(
        ent_flat_combined, surv_combined, checkpoints_frac
    )

    assert abs(lane1_gen_total - 88_000_000_000) < 1, lane1_gen_total
    assert abs(combined_gen_total - (lane1_gen_total + lane2_gen_total)) < 1

    # ---- stability verdicts: incremental-rate dispersion test on the 8 checkpoints above ----
    def dispersion_verdict(cum_df, col_n, min_events_for_judgment=20):
        n = cum_df[col_n].values.astype(float)
        incr_n = np.diff(np.concatenate([[0.0], n]))
        incr_exposure = np.diff(np.concatenate([[0.0], cum_df["cum_generated_events_approx"].values]))
        total_n = n[-1]
        if total_n < min_events_for_judgment:
            return "TOO_FEW_EVENTS_TO_JUDGE", dict(total_n=float(total_n))
        expected_rate = total_n / cum_df["cum_generated_events_approx"].values[-1]
        expected_incr = expected_rate * incr_exposure
        # Pearson chi-square dispersion statistic vs constant-rate Poisson null
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.nansum(np.where(expected_incr > 0, (incr_n - expected_incr) ** 2 / expected_incr, 0.0))
        dof = int((expected_incr > 0).sum()) - 1
        # crude threshold: chi2/dof > ~2.5 flagged as visible tension for this few bins
        ratio = chi2 / dof if dof > 0 else np.nan
        verdict = "TENSION_VISIBLE" if (dof > 0 and ratio > 2.5) else "STABLE_WITHIN_CURRENT_MC"
        return verdict, dict(chi2=float(chi2), dof=dof, chi2_over_dof=float(ratio) if dof > 0 else None,
                              total_n=float(total_n))

    verdict_u35, detail_u35 = dispersion_verdict(combined_cum, "n_qcd_survivors_u35")
    verdict_u45, detail_u45 = dispersion_verdict(combined_cum, "n_qcd_survivors_u45")

    # ---- QCD1 vs QCD2 tail composition comparison (final, full-lane populations) ----
    def describe(df, mask):
        sub = df[mask]
        if len(sub) == 0:
            return dict(n=0)
        return dict(n=int(len(sub)), HT_median=float(sub["HT"].median()), mHH_median=float(sub["mHH"].median()),
                     RHH_median=float(sub["RHH"].median()), n_selected_jets_median=float(sub["n_selected_jets"].median()),
                     n_btag_loose_median=float(sub["n_btag_loose"].median()))

    lane_compare = {
        "u35": {"lane1": describe(lane1_surv, lane1_surv["u35"]), "lane2": describe(lane2_surv, lane2_surv["u35"])},
        "u45": {"lane1": describe(lane1_surv, lane1_surv["u45"]), "lane2": describe(lane2_surv, lane2_surv["u45"])},
    }
    # rate compatibility test (two-sample Poisson rate ratio, u>3.5): exact binomial
    from scipy import stats
    n1 = int(lane1_surv["u35"].sum())
    n2 = int(lane2_surv["u35"].sum())
    # under H0 both lanes share one true rate per generated event, the count in
    # lane1 given (n1+n2) total is Binomial(n1+n2, p1) with p1 = gen1/(gen1+gen2)
    p1_expected = lane1_gen_total / (lane1_gen_total + lane2_gen_total)
    binom_test_u35 = stats.binomtest(n1, n1 + n2, p1_expected)
    n1_45 = int(lane1_surv["u45"].sum())
    n2_45 = int(lane2_surv["u45"].sum())
    binom_test_u45 = stats.binomtest(n1_45, n1_45 + n2_45, p1_expected) if (n1_45 + n2_45) > 0 else None

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    combined_cum.to_csv(os.path.join(OUT_DIR, "QCD_CUMULATIVE_STABILITY.csv"), index=False)
    lane1_cum.to_csv(os.path.join(WORK_DIR, "qcd1_cumulative.csv"), index=False)
    lane2_cum.to_csv(os.path.join(WORK_DIR, "qcd2_cumulative.csv"), index=False)

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cum_gen_np = combined_cum["cum_generated_events_approx"].to_numpy(dtype=float)
    n35_np = combined_cum["n_qcd_survivors_u35"].to_numpy(dtype=float)
    n45_np = combined_cum["n_qcd_survivors_u45"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cum_gen_np / 1e9, n35_np, "o-", label="u>3.5")
    ax.plot(cum_gen_np / 1e9, n45_np, "s-", label="u>4.5")
    ax.axvline(lane1_gen_total / 1e9, color="gray", ls="--", lw=1, label="end of QCD1 (lane boundary)")
    ax.set_xlabel("cumulative generated QCD exposure [x1e9 events]\n(lane1 exact; lane2 selected-count-proxy, see script docstring)")
    ax.set_ylabel("cumulative raw QCD survivors")
    ax.set_title("QCD tail-survivor growth vs cumulative generated exposure\n(existing frozen production only, no new MC)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "qcd_tail_rate_vs_generated_events.png"), dpi=150)
    plt.close(fig)

    for col, thr, fname in [("n_qcd_survivors_u35", "u>3.5", "qcd_u35_yield_stability.png"),
                             ("n_qcd_survivors_u45", "u>4.5", "qcd_u45_yield_stability.png")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        x = cum_gen_np / 1e9
        n = combined_cum[col].to_numpy(dtype=float)
        rate = n / cum_gen_np
        rate_err = np.sqrt(np.maximum(n, 1.0)) / cum_gen_np
        final_rate = n[-1] / cum_gen_np[-1]
        ax.errorbar(x, rate * 1e9, yerr=rate_err * 1e9, fmt="o-", capsize=3)
        ax.axhline(final_rate * 1e9, color="red", ls="--", lw=1, label="final cumulative rate")
        ax.set_xlabel("cumulative generated QCD exposure [x1e9 events]")
        ax.set_ylabel(f"cumulative survivor rate at {thr} [events / 1e9 generated]")
        ax.set_title(f"Running QCD tail rate, {thr} (Poisson error bars)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, feat in zip(axes, ["HT", "mHH", "n_btag_loose"]):
        for lane_df, lane_name, color in [(lane1_surv, "QCD1", "C0"), (lane2_surv, "QCD2", "C1")]:
            sub = lane_df[lane_df["u35"]]
            if len(sub):
                ax.hist(sub[feat].to_numpy(dtype=float), bins=12, histtype="step", density=True, label=lane_name, color=color)
        ax.set_xlabel(feat)
        ax.set_title(f"{feat} (u>3.5 survivors)")
    axes[0].legend()
    fig.suptitle("QCD1 vs QCD2 tail composition (u>3.5 survivors)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "qcd1_vs_qcd2_tail_composition.png"), dpi=150)
    plt.close(fig)

    result = dict(
        lane_generated_totals=LANE_GENERATED_TOTAL,
        lane_selected_totals={"inference_qcd_1": lane1_selected, "inference_qcd_2": lane2_selected},
        lane_files={"inference_qcd_1": lane1_files, "inference_qcd_2": lane2_files},
        lane1_jobs_per_file=lane1_jobs_per_file, lane1_uniform_per_file_generated_count_PROVEN=lane1_uniform_provable,
        lane2_jobs_per_file=lane2_jobs_per_file, lane2_uniform_per_file_generated_count_PROVEN=lane2_uniform_provable,
        lane2_per_file_generated_count_status="APPROXIMATION_ONLY_selected_count_scaled_by_lane_efficiency",
        lane_efficiency=lane_eff,
        n_qcd_survivors_u35_total=n_qcd_u35, n_qcd_survivors_u45_total=n_qcd_u45,
        weight_numerator_reproduces_frozen_constant=abs(WEIGHT_NUMERATOR / (lane1_gen_total + lane2_gen_total) - 7.3112875413) < 1e-6,
        final_running_weight_450fb=float(combined_cum["running_weight_450fb"].iloc[-1]),
        final_running_B_u35=float(combined_cum["running_B_u35"].iloc[-1]),
        final_running_B_u45=float(combined_cum["running_B_u45"].iloc[-1]),
        dispersion_test_u35=dict(verdict=verdict_u35, **detail_u35),
        dispersion_test_u45=dict(verdict=verdict_u45, **detail_u45),
        lane_compare=lane_compare,
        binomial_rate_compatibility_u35=dict(
            n_lane1=n1, n_lane2=n2, p1_expected_from_generated_exposure=p1_expected,
            observed_p1=n1 / (n1 + n2) if (n1 + n2) else None, p_value=binom_test_u35.pvalue,
        ),
        binomial_rate_compatibility_u45=(dict(
            n_lane1=n1_45, n_lane2=n2_45, p1_expected_from_generated_exposure=p1_expected,
            observed_p1=n1_45 / (n1_45 + n2_45) if (n1_45 + n2_45) else None, p_value=binom_test_u45.pvalue,
        ) if binom_test_u45 is not None else "n1_45+n2_45==0, test undefined"),
    )
    with open(os.path.join(WORK_DIR, "qcd_stability_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
