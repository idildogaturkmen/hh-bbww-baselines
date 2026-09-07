"""v3: exact-Harvey-threshold FP32-vs-raw-logit audit.

Pure post-processing. Sole input: the v2 package's
SPA10M_EVENT_LOGITS_400K.parquet. No model, checkpoint, HDF5, GPU, or
training is touched anywhere in this file (no torch/spanet/h5py
imports).

Harvey's literal thresholds: score > 0.9997 and score > 0.99997 (exact
'>' semantics, not '>='. Not the nearby rounded u=3.5/4.5 used in v2).
"""
import csv
import json
import math

import numpy as np
import pyarrow.parquet as pq

V2_DIR = ("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-bdt/docs/checkpoints/"
          "track_b_harvey_fp32_score_precision_20260906_v2_governing10m")
PARQUET_PATH = f"{V2_DIR}/SPA10M_EVENT_LOGITS_400K.parquet"
OUT_DIR = ("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-bdt/docs/checkpoints/"
           "track_b_harvey_fp32_score_precision_20260907_v3_exact_harvey_thresholds")

HARVEY_THRESHOLDS = [0.9997, 0.99997]
SUPPLEMENTARY_THRESHOLDS = [0.9999, 0.99999]  # Task 3, "only if no new inference" -- pure arithmetic/filters, so included


def score_to_u(score):
    """u = -log10(1-score), float64, from the decimal threshold itself."""
    score = np.float64(score)
    one_minus = 1.0 - score
    return -math.log10(one_minus), one_minus


def score_to_delta(score):
    """Delta = ln(score/(1-score)) -- the natural-log logit function,
    consistent with this project's Delta = z_signal - z_background and
    u_logit = softplus(Delta)/ln(10) (both defined via natural exp/log
    in v2's scripts/run_governing_10m_tail_inference.py)."""
    score = np.float64(score)
    return math.log(score / (1.0 - score))


def delta_to_u(delta):
    """Independent round-trip check: u implied by a Delta value, via the
    same stable softplus formula v2 uses -- softplus(Delta)/ln(10) ==
    -log10(1-sigmoid(Delta)) in exact arithmetic."""
    d = np.float64(delta)
    softplus = max(d, 0.0) + math.log1p(math.exp(-abs(d)))
    return softplus / math.log(10.0)


def task1_exact_thresholds():
    results = {}
    for thr in HARVEY_THRESHOLDS:
        u, one_minus = score_to_u(thr)
        delta = score_to_delta(thr)
        u_roundtrip = delta_to_u(delta)  # independent second derivation, must match u
        matches = bool(abs(float(u) - float(u_roundtrip)) < 1e-12)
        results[repr(thr)] = dict(
            score_threshold=thr,
            one_minus_score=float(one_minus),
            u=float(u),
            u_roundtrip_from_delta=float(u_roundtrip),
            u_matches_roundtrip_to_1e12=matches,
            delta_threshold_natural_log=float(delta),
        )
        print(f"score>{thr}: u={u!r}  delta={delta!r}  "
              f"roundtrip_u_from_delta={u_roundtrip!r}  "
              f"match={matches}")
    with open(f"{OUT_DIR}/work/task1_exact_thresholds.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def load_parquet():
    t = pq.read_table(PARQUET_PATH)
    d = t.to_pydict()
    for k in ("row_index", "z_background", "z_signal", "delta", "score_float32_production",
              "k_lattice", "u_prob32", "u_logit", "sigmoid_delta_float64"):
        d[k] = np.asarray(d[k])
    d["process_label"] = np.asarray(d["process_label"])
    d["score_float32_hex_be"] = np.asarray(d["score_float32_hex_be"])
    return d


def populations(d):
    proc = d["process_label"]
    return {
        "signal": proc == "signal",
        "qcd": proc == "qcd",
        "ttbar": proc == "ttbar",
        "all_background": proc != "signal",
        "full_cohort": np.ones(len(proc), dtype=bool),
    }


def task2_crosscheck(d, pops, thresholds):
    """Event-by-event: stored score_float32_production > threshold
    (PRODUCTION DECISION) vs delta > delta_threshold (RAW-LOGIT
    DECISION), exact '>' both sides."""
    score32 = d["score_float32_production"]  # already float64 column holding exact float32 values
    delta = d["delta"]
    row_index = d["row_index"]
    proc = d["process_label"]
    u_prob32 = d["u_prob32"]
    u_logit = d["u_logit"]

    all_results = {}
    all_disagreement_rows = []

    for thr in thresholds:
        delta_thr = score_to_delta(thr)
        pass_fp32 = score32 > thr          # exact '>' on the stored float32-derived value
        pass_logit = delta > delta_thr     # exact '>' on the raw logit margin

        for pop_name, mask in pops.items():
            n_pop = int(mask.sum())
            n_fp32 = int((pass_fp32 & mask).sum())
            n_logit = int((pass_logit & mask).sum())
            only_fp32 = pass_fp32 & ~pass_logit & mask
            only_logit = pass_logit & ~pass_fp32 & mask
            n_only_fp32 = int(only_fp32.sum())
            n_only_logit = int(only_logit.sum())
            n_disagree = n_only_fp32 + n_only_logit

            key = f"thr={thr}|pop={pop_name}"
            all_results[key] = dict(
                threshold=thr, population=pop_name, n_population=n_pop,
                n_pass_fp32=n_fp32, n_pass_logit=n_logit,
                n_only_fp32=n_only_fp32, n_only_logit=n_only_logit,
                n_disagree=n_disagree,
                disagreement_fraction_of_population=(n_disagree / n_pop) if n_pop else None,
            )
            print(f"thr={thr:>8} pop={pop_name:15s} n={n_pop:7d} "
                  f"pass_fp32={n_fp32:6d} pass_logit={n_logit:6d} "
                  f"only_fp32={n_only_fp32:4d} only_logit={n_only_logit:4d} disagree={n_disagree:4d}")

            for mask_d, which in ((only_fp32, "only_fp32_pass"), (only_logit, "only_logit_pass")):
                idx = np.nonzero(mask_d)[0]
                for i in idx:
                    all_disagreement_rows.append(dict(
                        threshold=thr, population_row=str(proc[i]), disagreement_type=which,
                        row_index=int(row_index[i]), process_label=str(proc[i]),
                        score_float32=float(score32[i]), delta=float(delta[i]),
                        u_prob32=float(u_prob32[i]), u_logit=float(u_logit[i]),
                    ))

    # de-duplicate disagreement rows across the population loop (a given
    # row_index appears once under its own process AND once under
    # all_background/full_cohort) -- keep only the population-agnostic
    # unique event list, one row per (threshold, row_index).
    seen = set()
    unique_disagreements = []
    for r in all_disagreement_rows:
        key = (r["threshold"], r["row_index"])
        if key in seen:
            continue
        seen.add(key)
        unique_disagreements.append(r)

    with open(f"{OUT_DIR}/work/task2_crosscheck_by_population.json", "w") as f:
        json.dump(all_results, f, indent=2)

    fieldnames = ["threshold", "row_index", "process_label", "disagreement_type",
                  "score_float32", "delta", "u_prob32", "u_logit"]
    with open(f"{OUT_DIR}/SPA10M_EXACT_THRESHOLD_DISAGREEMENTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in unique_disagreements:
            w.writerow({k: r[k] for k in fieldnames})

    print(f"\nwrote work/task2_crosscheck_by_population.json")
    print(f"wrote SPA10M_EXACT_THRESHOLD_DISAGREEMENTS.csv ({len(unique_disagreements)} unique disagreement rows)")
    return all_results, unique_disagreements


def task3_tail_census(d, pops, thresholds, json_name="task3_tail_census.json",
                       csv_name="SPA10M_EXACT_THRESHOLD_TAIL_CENSUS.csv"):
    score32 = d["score_float32_production"]
    delta = d["delta"]
    u_logit = d["u_logit"]

    census = {}
    for thr in thresholds:
        tail_mask = score32 > thr
        row = dict(threshold=thr)
        for pop_name in ("signal", "qcd", "ttbar", "all_background"):
            mask = pops[pop_name] & tail_mask
            row[f"n_{pop_name}"] = int(mask.sum())

        full_tail = pops["full_cohort"] & tail_mask
        n_tail = int(full_tail.sum())
        vals32 = score32[full_tail]
        uniq32, counts32 = np.unique(vals32, return_counts=True)
        n_exact_1 = int((vals32 == 1.0).sum())
        n_tied = int(counts32[counts32 > 1].sum())  # events sharing a value with >=1 other event
        max_multiplicity = int(counts32.max()) if len(counts32) else 0

        row.update(dict(
            n_total_tail=n_tail,
            n_distinct_float32_probabilities=int(len(uniq32)),
            n_distinct_delta=int(len(np.unique(delta[full_tail]))),
            n_distinct_u_logit=int(len(np.unique(u_logit[full_tail]))),
            n_exact_score_eq_1p0=n_exact_1,
            fraction_events_in_fp32_tie=(n_tied / n_tail) if n_tail else None,
            max_multiplicity_one_stored_probability=max_multiplicity,
        ))
        census[repr(thr)] = row
        print(f"thr={thr}: n_tail={n_tail} n_distinct_prob32={len(uniq32)} "
              f"n_distinct_delta={row['n_distinct_delta']} n_distinct_u_logit={row['n_distinct_u_logit']} "
              f"n_exact_1={n_exact_1} frac_tied={row['fraction_events_in_fp32_tie']} "
              f"max_mult={max_multiplicity}")

    with open(f"{OUT_DIR}/work/{json_name}", "w") as f:
        json.dump(census, f, indent=2)

    fieldnames = ["threshold", "n_signal", "n_qcd", "n_ttbar", "n_all_background", "n_total_tail",
                  "n_distinct_float32_probabilities", "n_distinct_delta", "n_distinct_u_logit",
                  "n_exact_score_eq_1p0", "fraction_events_in_fp32_tie",
                  "max_multiplicity_one_stored_probability"]
    with open(f"{OUT_DIR}/{csv_name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for thr in thresholds:
            w.writerow({k: census[repr(thr)][k] for k in fieldnames})

    print(f"\nwrote work/{json_name}")
    print(f"wrote {csv_name}")
    return census


def task3_supplementary(d, pops):
    """0.9999 / 0.99999 supplementary table -- pure filtering/counting on
    the already-loaded parquet columns, no new inference. Uses distinct
    filenames from the main Harvey-threshold census so the two calls to
    task3_tail_census() do not clobber each other's output files."""
    return task3_tail_census(d, pops, SUPPLEMENTARY_THRESHOLDS,
                              json_name="task3_supplementary_census.json",
                              csv_name="SPA10M_SUPPLEMENTARY_0P9999_0P99999.csv")


def main():
    task1 = task1_exact_thresholds()
    d = load_parquet()
    pops = populations(d)
    task2, disagreements = task2_crosscheck(d, pops, HARVEY_THRESHOLDS)
    task3 = task3_tail_census(d, pops, HARVEY_THRESHOLDS)

    print("\n--- supplementary 0.9999/0.99999 table (no new inference, same parquet) ---")
    task3_supp = task3_supplementary(d, pops)

    master = dict(task1=task1, task2=task2, task3=task3, task3_supplementary=task3_supp)
    with open(f"{OUT_DIR}/work/master_results.json", "w") as f:
        json.dump(master, f, indent=2)
    print("\nwrote work/master_results.json")


if __name__ == "__main__":
    main()
