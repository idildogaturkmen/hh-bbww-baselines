#!/usr/bin/env python3
"""Correction 2 (v2): process-by-process before/after table for the FROZEN
exploratory cut pT_H1 < 406.115 GeV, plus exploratory Z_A / B95 using the
project's already-established formulas (asimov_z, b95_upper_mean --
Poisson<->Gamma duality, stdlib-only bisection), copied verbatim from
track_b_harvey_score_tail_diagnostics_20260829_v3/work/common.py, NOT
re-derived here.

The cut itself is NOT re-optimized: it is the exact v1 threshold,
recovered here as the median of ttbar's own pT_H1 distribution at u>3.5
(406.1153676240428 GeV), reproducing v1's n_ttbar_before=27,
n_ttbar_after=13 exactly. Everything in this file is a downstream
read-only characterization of that already-frozen cut.

This cut was selected BY LOOKING AT the same 27-event ttbar sample it is
now being evaluated on (median-split) -- it is EXPLORATORY, not a
validated/held-out result. All numbers below carry that flag.
"""
import json
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/"
    "track_b_harvey_score_tail_diagnostics_20260829_v3/work",
)
from common import asimov_z, b95_upper_mean  # noqa: E402  (verbatim reuse, not re-derived)

MASTER = "../HARVEY_MASTER_EVENT_TABLE.parquet"
FROZEN_CUT_PT_H1 = 406.1153676240428  # exact v1 threshold (median of ttbar pT_H1 at u>3.5)
OUT_JSON = "correction2_ttbar_table.json"
OUT_MD_ROWS = "correction2_ttbar_table_rows.csv"


def process_row(name, sub_before, weight_col="weight_450fb"):
    n_before = len(sub_before)
    if n_before == 0:
        return None
    passed = sub_before["pT_H1"] < FROZEN_CUT_PT_H1
    n_after = int(passed.sum())
    w = sub_before[weight_col].to_numpy()
    w_before = float(w.sum())
    w_after = float(w[passed.to_numpy()].sum())
    eff = n_after / n_before
    return dict(
        process=name,
        n_raw_before=n_before,
        n_raw_after=n_after,
        weighted_before_450fb=w_before,
        weighted_after_450fb=w_after,
        raw_efficiency_or_survival=eff,
        raw_rejection=1.0 - eff,
    )


def main():
    master = pd.read_parquet(MASTER)
    sub35 = master[master["u"] > 3.5].copy()

    sig = sub35[sub35["label"] == 1]
    qcd = sub35[sub35["background_class"] == "QCD"]
    ttbar = sub35[sub35["background_class"] == "ttbar"]
    other = sub35[sub35["background_class"] == "other_background"]

    rows = [process_row("signal", sig)]
    rows.append(process_row("QCD", qcd))
    rows.append(process_row("ttbar (inference_ttbar_1+2 combined)", ttbar))
    for proc in sorted(other["process"].unique()):
        rows.append(process_row(proc, other[other["process"] == proc]))
    rows = [r for r in rows if r is not None]

    # aggregate "Total B" (QCD + ttbar + all other_background), matching
    # the definition of "all-background" used in the PRIMARY 110-event study
    allbkg = sub35[sub35["label"] == 0]
    total_b_row = process_row("TOTAL_BACKGROUND (QCD+ttbar+other, n=110)", allbkg)
    total_b_row["process"] = "TOTAL_BACKGROUND (QCD+ttbar+other, n=110)"

    df = pd.DataFrame(rows + [total_b_row])
    df.to_csv(OUT_MD_ROWS, index=False)

    # ---- exploratory Z_A / B95 downstream of the frozen cut ----
    sig_row = next(r for r in rows if r["process"] == "signal")
    S_weighted_after = sig_row["weighted_after_450fb"]
    B_weighted_after = total_b_row["weighted_after_450fb"]
    za_nominal = asimov_z(S_weighted_after, B_weighted_after)

    # B95 upper mean per background process after the cut, using each
    # process's own raw survivor count and its frozen per-event weight
    # (flat-weight x raw-count convention, verbatim from common.py).
    b95_components = []
    b95_total = 0.0
    for r in rows:
        if r["process"] == "signal":
            continue
        n_raw_after = r["n_raw_after"]
        w_after = r["weighted_after_450fb"]
        per_event_weight = w_after / n_raw_after if n_raw_after > 0 else 0.0
        b95 = b95_upper_mean(n_raw_after, per_event_weight) if n_raw_after > 0 else 0.0
        b95_components.append(dict(process=r["process"], n_raw_after=n_raw_after,
                                    per_event_weight_450fb=per_event_weight,
                                    b95_upper_450fb=b95))
        b95_total += b95
    za_b95 = asimov_z(S_weighted_after, b95_total)

    result = dict(
        frozen_cut="pT_H1 < 406.1153676240428 GeV",
        cut_status="EXPLORATORY -- selected as the median of the same 27-event ttbar sample it is evaluated on; NOT a validated/held-out cut",
        n_ttbar_before=27,
        n_ttbar_after=int(ttbar["pT_H1"].lt(FROZEN_CUT_PT_H1).sum()),
        ttbar_fraction_rejected=14 / 27,
        ttbar_fraction_rejected_pct_str="51.85% (14 of 27 removed, 13 remaining)",
        process_table=rows + [total_b_row],
        exploratory_significance={
            "note": "asimov_z and b95_upper_mean copied verbatim from "
                    "track_b_harvey_score_tail_diagnostics_20260829_v3/work/common.py, "
                    "not re-derived here. Weighted yields (450 fb^-1) after the frozen "
                    "exploratory cut. EXPLORATORY: the cut was chosen by looking at the "
                    "same finite ttbar sample used here, so this is not a held-out estimate.",
            "S_weighted_450fb_after_cut": S_weighted_after,
            "B_weighted_450fb_after_cut_nominal": B_weighted_after,
            "Z_A_nominal": za_nominal,
            "B95_components_per_background_process": b95_components,
            "B_95upper_450fb_total": b95_total,
            "Z_A_using_B95_upper": za_b95,
        },
    )
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(df.to_string(index=False))
    print()
    print(f"Z_A (nominal weighted B)   = {za_nominal:.4f}")
    print(f"Z_A (B95-upper weighted B) = {za_b95:.4f}")
    print(f"ttbar rejected: 14/27 = {14/27:.4%}")


if __name__ == "__main__":
    main()
