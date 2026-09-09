"""Task 3 (+ cross-checks for Task 1/2): exact FP32 unique-value census
of the real, frozen, on-disk SPA-Net event-classification score archive
for u = -log10(1 - score) > 5.5.

Data source (read-only, pre-existing, nothing recomputed):
  val_signal_probs_400k.npy    (400000,) float32 -- P(signal) per event
  val_process_labels_400k.npy  (400000,) '<U6'   -- {'qcd','signal','ttbar'}
from the frozen SPA-Net 2M seed-0 development classification evaluation
(checkpoint sha256 dc39cf76f0d8e40d07228f240fe58b1179e8134cd4f96c5c786a7d528d31ea8d),
produced by evaluate_classification.py (read only, model.eval()+torch.no_grad(),
no gradient, no training). This is NOT the governing 10M checkpoint
(fd9ea100...) -- that evaluation (evaluate_classification_10M.py) never
called np.save and only persisted summary statistics (see the report
for the direct code citation). The 2M and 10M scripts are otherwise
methodologically identical (same HDF5_VAL, same process reconstruction,
same forward-pass code path), so this is the closest available real
per-event evidence for the FP32 quantization mechanism Harvey is asking
about, used here as a faithful stand-in with that limitation stated
explicitly throughout the report.

No Stage-C/holdout_B file is read. No model is invoked. No new
inference is run. This script only loads two pre-existing .npy files.
"""
import csv
import json
import struct
import sys

import numpy as np

SRC_DIR = ("/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/"
           "phase4AF_spanet_partial_events_population_correction_20260819_v1/"
           "spanet_2M_seed0_classification_evaluation_v1/work")
PROBS_PATH = f"{SRC_DIR}/val_signal_probs_400k.npy"
LABELS_PATH = f"{SRC_DIR}/val_process_labels_400k.npy"
CKPT_SHA256 = "dc39cf76f0d8e40d07228f240fe58b1179e8134cd4f96c5c786a7d528d31ea8d"

SPACING = 2.0 ** -24
U_CUT = 5.5
OUT_DIR = "/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-bdt/docs/checkpoints/track_b_harvey_fp32_score_precision_20260906_v1"


def f32_hex_be(x):
    return struct.pack(">f", np.float32(x)).hex()


def u_of(score_f32_as_f64):
    one_minus = 1.0 - score_f32_as_f64
    with np.errstate(divide="ignore"):
        return -np.log10(one_minus)


def population_report(name, scores_f32):
    """scores_f32: 1D numpy float32 array for this population."""
    scores_f64 = scores_f32.astype(np.float64)
    u = u_of(scores_f64)
    mask = u > U_CUT
    n_tail = int(mask.sum())
    n_total = int(len(scores_f32))

    sel = scores_f32[mask]
    uniq, counts = np.unique(sel, return_counts=True)
    order = np.argsort(-uniq)  # descending score (i.e. ascending 1-score)
    uniq = uniq[order]
    counts = counts[order]

    rows = []
    for val, cnt in zip(uniq, counts):
        v64 = np.float64(val)
        k = int(round((1.0 - v64) / SPACING)) if val < 1.0 else 0
        rows.append(dict(
            population=name,
            score_f32=repr(float(val)),
            score_f32_exact_decimal=f"{v64:.20f}".rstrip("0").rstrip("."),
            score_f32_hex_be=f32_hex_be(val),
            k_round=k,
            u=float(u_of(v64)) if val < 1.0 else float("inf"),
            raw_count=int(cnt),
        ))

    n_eq_one = int((scores_f32 == np.float32(1.0)).sum())
    next_after = np.float32(np.nextafter(np.float32(1.0), np.float32(0.0)))
    n_eq_nextafter = int((scores_f32 == next_after).sum())

    return dict(
        population=name,
        n_total=n_total,
        n_tail_u_gt_5p5=n_tail,
        n_unique_tail_values=int(len(uniq)),
        n_exact_score_eq_1p0=n_eq_one,
        n_exact_score_eq_nextafter_1p0=n_eq_nextafter,
        nextafter_1p0_f32=repr(float(next_after)),
        rows=rows,
    )


def main():
    probs = np.load(PROBS_PATH)
    labels = np.load(LABELS_PATH)
    assert probs.dtype == np.float32, probs.dtype
    assert len(probs) == len(labels) == 400_000

    is_signal = labels == "signal"
    is_qcd = labels == "qcd"
    is_ttbar = labels == "ttbar"
    is_all_bg = ~is_signal
    is_other_bg = is_all_bg & ~is_qcd & ~is_ttbar  # expect zero -- only qcd/ttbar exist in this cohort

    populations = {
        "signal": probs[is_signal],
        "all_background": probs[is_all_bg],
        "qcd": probs[is_qcd],
        "ttbar": probs[is_ttbar],
        "other_background": probs[is_other_bg],
    }

    census = {}
    csv_rows = []
    for name, arr in populations.items():
        rep = population_report(name, arr)
        census[name] = rep
        csv_rows.extend(rep["rows"])
        print(f"{name:16s} n_total={rep['n_total']:7d} n_tail(u>5.5)={rep['n_tail_u_gt_5p5']:6d} "
              f"n_unique_tail={rep['n_unique_tail_values']:3d} "
              f"n_score==1.0={rep['n_exact_score_eq_1p0']:5d} "
              f"n_score==nextafter(1,0)={rep['n_exact_score_eq_nextafter_1p0']:5d}")

    # Weighted background yield: this dev cohort has no per-event physical
    # cross-section/luminosity weight stored anywhere in the frozen
    # archive (raw event counts only) -- explicitly note as N/A rather
    # than fabricate a weight.
    for name in ("all_background", "qcd", "ttbar", "other_background"):
        census[name]["weighted_yield_note"] = (
            "NOT AVAILABLE: this development-cohort per-event archive carries "
            "no per-event physical weight/cross-section field; only raw event "
            "counts exist here. Weighted yields exist only in the separate, "
            "aggregate physical-normalization freeze (CANONICAL_FOUR_MODEL_"
            "NORMALIZED_RESULTS.csv), which does not carry per-event scores. "
            "Not fabricated."
        )

    # Focus counts at exactly k=2 (u~6.9237) and k=4 (u~6.6226), the two
    # lattice points nearest Harvey's reported spike locations (Task 2).
    def count_at_k(arr_f32, k_target):
        v = np.float32(1.0) - np.float32(k_target) * np.float32(SPACING)
        return int((arr_f32 == v).sum())

    spike_counts = {}
    for name, arr in populations.items():
        spike_counts[name] = dict(
            k2_u6p9237=count_at_k(arr, 2),
            k4_u6p6226=count_at_k(arr, 4),
        )

    with open(f"{OUT_DIR}/FP32_UNIQUE_SCORE_CENSUS.csv", "w", newline="") as f:
        fieldnames = ["population", "score_f32", "score_f32_exact_decimal", "score_f32_hex_be",
                      "k_round", "u", "raw_count"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in csv_rows:
            w.writerow({k: r[k] for k in fieldnames})

    summary = dict(
        source_note="Real frozen per-event archive for SPA-Net 2M seed-0 (NOT the governing "
                     "10M checkpoint -- see report). Read-only; no recomputation of scores.",
        source_probs_path=PROBS_PATH,
        source_labels_path=LABELS_PATH,
        source_checkpoint_sha256=CKPT_SHA256,
        u_cut=U_CUT,
        spacing_2pow_neg24=SPACING,
        per_population=census,
        spike_focus_counts=spike_counts,
    )
    with open(f"{OUT_DIR}/fp32_census_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSpike-focus exact counts (population: k=2 [u=6.9237], k=4 [u=6.6226]):")
    for name, d in spike_counts.items():
        print(f"  {name:16s} k=2: {d['k2_u6p9237']:5d}   k=4: {d['k4_u6p6226']:5d}")

    print(f"\nwrote {OUT_DIR}/FP32_UNIQUE_SCORE_CENSUS.csv")
    print(f"wrote {OUT_DIR}/fp32_census_summary.json")


if __name__ == "__main__":
    sys.exit(main())
