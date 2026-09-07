"""Post-processing only: direct event-by-event cross-check of
u_prob32 > threshold vs u_logit > threshold, using ONLY the
already-produced SPA10M_EVENT_LOGITS_400K.parquet. No model, no
checkpoint, no HDF5 touched. No inference of any kind.
"""
import csv
import json

import numpy as np
import pyarrow.parquet as pq

OUT_DIR = "/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-bdt/docs/checkpoints/track_b_harvey_fp32_score_precision_20260906_v2_governing10m"
PARQUET_PATH = f"{OUT_DIR}/SPA10M_EVENT_LOGITS_400K.parquet"


def main():
    t = pq.read_table(PARQUET_PATH)
    d = t.to_pydict()
    row_index = np.asarray(d["row_index"])
    process_label = np.asarray(d["process_label"])
    u_prob32 = np.asarray(d["u_prob32"])
    u_logit = np.asarray(d["u_logit"])
    delta = np.asarray(d["delta"])
    score32 = np.asarray(d["score_float32_production"])

    results = {}
    disagreement_rows = []

    for thr in (3.5, 4.5):
        pass_prob = u_prob32 > thr
        pass_logit = u_logit > thr

        n_pass_prob = int(pass_prob.sum())
        n_pass_logit = int(pass_logit.sum())
        only_prob = pass_prob & ~pass_logit
        only_logit = pass_logit & ~pass_prob
        n_only_prob = int(only_prob.sum())
        n_only_logit = int(only_logit.sum())
        n_agree = int((pass_prob == pass_logit).sum())
        n_disagree = n_only_prob + n_only_logit

        for mask, which in ((only_prob, "only_u_prob32"), (only_logit, "only_u_logit")):
            idx = np.nonzero(mask)[0]
            for i in idx:
                disagreement_rows.append(dict(
                    threshold=thr,
                    disagreement_type=which,
                    row_index=int(row_index[i]),
                    process_label=str(process_label[i]),
                    u_prob32=float(u_prob32[i]),
                    u_logit=float(u_logit[i]),
                    delta=float(delta[i]),
                    score_float32=float(score32[i]),
                ))

        results[str(thr)] = dict(
            threshold=thr,
            n_pass_u_prob32=n_pass_prob,
            n_pass_u_logit=n_pass_logit,
            n_only_u_prob32=n_only_prob,
            n_only_u_logit=n_only_logit,
            n_agree=n_agree,
            n_disagree=n_disagree,
            n_total=int(len(row_index)),
        )
        print(f"threshold={thr}: n_pass_u_prob32={n_pass_prob} n_pass_u_logit={n_pass_logit} "
              f"n_only_u_prob32={n_only_prob} n_only_u_logit={n_only_logit} n_disagree={n_disagree}")

    with open(f"{OUT_DIR}/work/threshold_crosscheck_result.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"{OUT_DIR}/SPA10M_THRESHOLD_CROSSCHECK_DISAGREEMENTS.csv", "w", newline="") as f:
        fieldnames = ["threshold", "disagreement_type", "row_index", "process_label",
                      "u_prob32", "u_logit", "delta", "score_float32"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(disagreement_rows)

    print(f"\nwrote work/threshold_crosscheck_result.json")
    print(f"wrote SPA10M_THRESHOLD_CROSSCHECK_DISAGREEMENTS.csv ({len(disagreement_rows)} disagreement rows)")


if __name__ == "__main__":
    main()
