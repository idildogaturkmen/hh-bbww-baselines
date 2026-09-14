#!/usr/bin/env python3
"""Computes the common-cohort, all-background ROC curve and the
signal-efficiency-vs-background-rejection curve for all four models
(Native 2M, Native 10M, ParT active20 2M, ZERO20 2M), directly from the
frozen per-event score arrays. No training, no inference, no bootstrap --
pure arithmetic over already-frozen .npz exports.

Hard requirement enforced before any curve is computed: all four models'
event_id AND process arrays must match exactly after sorting by event_id
(the same "matched cohort" identity check this whole project already uses
elsewhere, e.g. export_model_eval_events.py / evaluate_multi_model.py's
own event_id/process cross-checks) -- this script refuses (non-zero exit)
if they do not.

Signal = process==0. Background = process in {1 (QCD), 2 (ttbar)}
(all-background). AUC is computed with the SAME dependency-free,
tie-averaged rank statistic (bootstrap_utils.roc_auc) already used for
every other AUC number in this project, so the legend AUC values here are
numerically consistent with (not a second, differently-rounded
implementation of) the ones already published.

Background rejection (1/epsilon_B) is computed ONLY over the raw finite
support actually present in each model's own 206,642-event background
sample -- the curve stops at the last threshold with >=1 background event
surviving; no fitted/extrapolated tail beyond that point is produced. The
"sparse" boundary (n_bg_pass < 10, the same low-statistics threshold this
project's own rejection_at_fixed_efficiency tables already flag) is
recorded per-point so the plotting script can style it distinctly rather
than silently presenting a Poisson-limited point as equally precise.
"""
import argparse
import json
import sys

import numpy as np

TRACK_F_CODE = "/uscms_data/d3/iturkmen/hh4b_delphes/track_f_evaluation_readiness_protocol_20260909_v1/code"
sys.path.insert(0, TRACK_F_CODE)
from bootstrap_utils import roc_auc  # noqa: E402 -- the exact same AUC statistic used throughout this project

PROCESS_SIGNAL = 0
SPARSE_BG_THRESHOLD = 10  # matches the "low-statistics (Poisson-limited)" flag already used in this project's tables

MODELS = [
    ("native2M", "Native SPA-Net 2M",
     "/uscms_data/d3/iturkmen/hh4b_delphes/track_f_postproduction_pipeline_20260908_v1/exports/control_2m_native_eval.npz"),
    ("native10M", "Native SPA-Net 10M",
     "/uscms_data/d3/iturkmen/hh4b_delphes/track_f_postproduction_pipeline_20260908_v1/exports/native10m_eval_400k.npz"),
    ("ParT20_2M", "SPA-Net + ParT active20 2M",
     "/uscms_data/d3/iturkmen/hh4b_delphes/track_f_evaluation_readiness_protocol_20260909_v1/results/part2m_seed0_matched_20260911T011132Z/spa2m_part_active_seed0_val_eval.npz"),
    ("ZERO20_2M", "SPA-Net + ZERO20 2M",
     "/uscms_data/d3/iturkmen/hh4b_delphes/track_j_zero20_diagnostic_20260911_v1/results/zero20_seed0_20260912T013902Z/spa2m_part_zero20_seed0_val_eval.npz"),
]


def load_sorted(path):
    d = np.load(path)
    order = np.argsort(d["event_id"])
    return dict(event_id=d["event_id"][order], process=d["process"][order], score=d["score"][order])


def roc_curve(score, is_signal):
    """Empirical ROC over all unique score thresholds actually present in
    the data (no binning, no smoothing). Returns fpr, tpr arrays from
    (0,0) to (1,1), descending-score order collapsed at ties."""
    order = np.argsort(-score, kind="mergesort")
    s_sorted = score[order]
    sig_sorted = is_signal[order].astype(np.int64)
    bg_sorted = (~is_signal[order]).astype(np.int64)

    tp_cum = np.cumsum(sig_sorted)
    fp_cum = np.cumsum(bg_sorted)

    # collapse to one point per unique threshold (last occurrence of each
    # run of tied scores) so the plotted curve does not show a fake
    # staircase inside a block of exactly-tied scores
    distinct = np.r_[np.diff(s_sorted) != 0, True]
    tp_cum = tp_cum[distinct]
    fp_cum = fp_cum[distinct]

    n_sig = int(sig_sorted.sum())
    n_bg = int(bg_sorted.sum())
    tpr = tp_cum / n_sig
    fpr = fp_cum / n_bg
    # prepend the (0,0) origin
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    n_bg_pass = np.r_[0, fp_cum]
    n_sig_pass = np.r_[0, tp_cum]
    return fpr, tpr, n_sig_pass, n_bg_pass, n_sig, n_bg


def _thin_indices(idx, max_points):
    """Evenly-spaced (never averaged/fit) selection of at most max_points
    indices from idx, keeping its first and last entries exactly."""
    if len(idx) <= max_points:
        return idx
    pick = np.linspace(0, len(idx) - 1, max_points).round().astype(int)
    return np.unique(idx[pick])


def thin_step_curve(x, detail_mask, max_detail_points=1200, max_bulk_points=600):
    """Reduces an already-monotonic step curve to a size a vector SVG can
    render efficiently, WITHOUT smoothing/interpolating -- every kept point
    is a REAL point from the original curve, none averaged or fit. Two
    tiers, each independently thinned by even index spacing: `detail_mask`
    marks the scientifically sensitive tail (kept at up to max_detail_points,
    much finer than the bulk); everything else is thinned to at most
    max_bulk_points. Always keeps the curve's first and last point exactly.
    Returns the selected index array (sorted, unique)."""
    n = len(x)
    full_idx = np.where(detail_mask)[0]
    bulk_idx = np.where(~detail_mask)[0]
    full_idx = _thin_indices(full_idx, max_detail_points)
    bulk_idx = _thin_indices(bulk_idx, max_bulk_points)
    keep = np.unique(np.r_[full_idx, bulk_idx, 0, n - 1])
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    print("==== Loading all four models, sorting by event_id ====")
    data = {}
    for key, label, path in MODELS:
        data[key] = load_sorted(path)
        print(f"  {key}: n={len(data[key]['event_id'])}  {path}")

    print("\n==== Verifying all four event_id/process arrays match exactly ====")
    ref_key = MODELS[0][0]
    ref = data[ref_key]
    for key, label, path in MODELS[1:]:
        if not np.array_equal(ref["event_id"], data[key]["event_id"]):
            print(f"FATAL: event_id mismatch between {ref_key} and {key}", file=sys.stderr)
            sys.exit(2)
        if not np.array_equal(ref["process"], data[key]["process"]):
            print(f"FATAL: process mismatch between {ref_key} and {key}", file=sys.stderr)
            sys.exit(3)
    print(f"  VERIFIED: identical event_id and process arrays across all {len(MODELS)} models "
          f"(n={len(ref['event_id'])}).")

    is_signal = ref["process"] == PROCESS_SIGNAL
    n_signal = int(is_signal.sum())
    n_background = int((~is_signal).sum())
    print(f"  n_signal={n_signal} n_background={n_background} (process 1=QCD + 2=ttbar)")

    DETAIL_N_BG_PASS = 200  # fine-thinned (not full) resolution below this bg-survivor count; coarser above it

    out = dict(n_signal=n_signal, n_background=n_background, sparse_bg_threshold=SPARSE_BG_THRESHOLD,
               detail_n_bg_pass_threshold=DETAIL_N_BG_PASS, models={})
    for key, label, path in MODELS:
        score = data[key]["score"]
        auc = roc_auc(score, is_signal.astype(int))
        fpr, tpr, n_sig_pass, n_bg_pass, n_sig, n_bg = roc_curve(score, is_signal)
        assert n_sig == n_signal and n_bg == n_background

        n_full_before_thin = len(fpr)
        roc_keep = thin_step_curve(fpr, n_bg_pass < DETAIL_N_BG_PASS)
        roc_keep.sort()
        fpr_plot, tpr_plot = fpr[roc_keep], tpr[roc_keep]

        # background-rejection curve: only where n_bg_pass >= 1 (finite,
        # non-extrapolated rejection); the point where n_bg_pass first
        # reaches 0 (moving from high to low score / high to low
        # efficiency) marks where rejection becomes formally infinite --
        # not plotted, not extrapolated, recorded separately as the
        # "max achieved" boundary.
        finite = n_bg_pass >= 1
        eff_s_finite = tpr[finite]
        rejection_finite = n_bg / n_bg_pass[finite]
        n_bg_pass_finite = n_bg_pass[finite]
        sparse_finite = n_bg_pass_finite < SPARSE_BG_THRESHOLD

        # order by increasing signal efficiency for plotting
        ord_eff = np.argsort(eff_s_finite)
        eff_s_finite = eff_s_finite[ord_eff]
        rejection_finite = rejection_finite[ord_eff]
        n_bg_pass_finite = n_bg_pass_finite[ord_eff]
        sparse_finite = sparse_finite[ord_eff]

        rej_keep = thin_step_curve(eff_s_finite, n_bg_pass_finite < DETAIL_N_BG_PASS)
        rej_keep.sort()
        eff_s_finite, rejection_finite = eff_s_finite[rej_keep], rejection_finite[rej_keep]
        n_bg_pass_finite, sparse_finite = n_bg_pass_finite[rej_keep], sparse_finite[rej_keep]

        # The floor, not the ceiling, is the informative boundary: as the
        # threshold tightens (signal efficiency falls), background survivors
        # run out at SOME minimum signal efficiency, below which rejection
        # is formally undefined (would require extrapolating past the last
        # real background event) -- that floor is exactly what must not be
        # crossed by the plotted curve.
        min_eff_s_with_finite_rejection = float(eff_s_finite.min()) if len(eff_s_finite) else None

        out["models"][key] = dict(
            label=label, source_npz=path, auc_all_background=float(auc),
            roc=dict(fpr=fpr_plot.tolist(), tpr=tpr_plot.tolist(),
                     n_points_full_resolution=int(n_full_before_thin), n_points_plotted=int(len(fpr_plot)),
                     note="step function reduced for plotting: full resolution kept wherever n_bg_pass < "
                          f"{DETAIL_N_BG_PASS}, evenly (not smoothed/interpolated) thinned above that"),
            rejection=dict(signal_efficiency=eff_s_finite.tolist(),
                            rejection=rejection_finite.tolist(),
                            n_bg_pass=n_bg_pass_finite.tolist(),
                            is_sparse=sparse_finite.tolist()),
            min_signal_efficiency_with_finite_bg_rejection=min_eff_s_with_finite_rejection,
        )
        print(f"  {key}: AUC={auc:.6f}  minimum signal-eff with finite (non-extrapolated) bg rejection = "
              f"{min_eff_s_with_finite_rejection:.5f}  "
              f"(ROC points: {n_full_before_thin} -> {len(fpr_plot)} plotted)")

    with open(args.out_json, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
