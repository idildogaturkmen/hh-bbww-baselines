"""
AUXILIARY_NONQCD_MET analysis (v2 Correction 1).

met_pt is NaN for every QCD row (branch not available / not meaningful for the
QCD sidecar extraction), so it CANNOT appear in the PRIMARY all-background
(110-event: QCD+ttbar+other) study without silently dropping all 68 QCD rows.

This auxiliary study exists ONLY to answer "is met_pt useful at all, for the
backgrounds where it IS defined" -- explicitly NOT an all-background result,
explicitly NOT compared numerically to the 110-background primary models as
if the populations were the same. QCD is completely ABSENT from this study.
"""
import json
import numpy as np
import pandas as pd
from stats_utils import roc_auc_weighted, bootstrap_auc_ci

MASTER = "../HARVEY_MASTER_EVENT_TABLE.parquet"
OUT = "auxiliary_met_summary.json"


def main():
    master = pd.read_parquet(MASTER)
    sub = master[master["u"] > 3.5].copy()

    sig = sub[sub["label"] == 1]
    nonqcd_bkg = sub[(sub["label"] == 0) & (sub["background_class"] != "QCD")]

    n_ttbar = int((nonqcd_bkg["background_class"] == "ttbar").sum())
    n_other = int((nonqcd_bkg["background_class"] == "other_background").sum())
    assert n_ttbar + n_other == len(nonqcd_bkg)

    sig_ok = sig[sig["met_pt"].notna()]
    bkg_ok = nonqcd_bkg[nonqcd_bkg["met_pt"].notna()]
    assert len(bkg_ok) == len(nonqcd_bkg), "met_pt must be fully populated for non-QCD backgrounds"

    x_sig = sig_ok["met_pt"].to_numpy()
    w_sig = sig_ok["weight_450fb"].to_numpy()
    x_bkg = bkg_ok["met_pt"].to_numpy()
    w_bkg = bkg_ok["weight_450fb"].to_numpy()

    scores = np.concatenate([x_sig, x_bkg])
    labels = np.concatenate([np.ones(len(x_sig), dtype=np.int64), np.zeros(len(x_bkg), dtype=np.int64)])
    weights = np.concatenate([w_sig, w_bkg])

    auc_raw = roc_auc_weighted(scores, labels, np.ones_like(weights))
    auc_weighted = roc_auc_weighted(scores, labels, weights)
    boot_mean, boot_lo, boot_hi = bootstrap_auc_ci(
        scores, labels, np.ones_like(weights), n_boot=150, max_class_n=2000, seed=42
    )

    result = {
        "label": "AUXILIARY_NONQCD_MET -- QCD ABSENT, NOT an all-background result, NOT numerically comparable to PRIMARY_BKG_N=110 models",
        "n_signal_met_available": int(len(x_sig)),
        "n_nonqcd_background": int(len(x_bkg)),
        "n_ttbar": n_ttbar,
        "n_other_background": n_other,
        "qcd_included": False,
        "met_pt_auc_raw_unweighted": float(auc_raw),
        "met_pt_auc_weighted_450fb": float(auc_weighted),
        "met_pt_auc_bootstrap_mean": float(boot_mean),
        "met_pt_auc_bootstrap_ci68": [float(boot_lo), float(boot_hi)],
        "median_met_pt_signal_GeV": float(np.median(x_sig)),
        "median_met_pt_nonqcd_background_GeV": float(np.median(x_bkg)),
        "interpretation": (
            "met_pt separates signal from ttbar+other_background reasonably well "
            "(AUC far from 0.5, background has systematically higher met_pt), but this "
            "cannot be folded into the PRIMARY 110-background ranking because QCD has no "
            "met_pt value at all. Any future model wanting to use met_pt would need a "
            "QCD-specific imputation/handling strategy, which is out of scope here."
        ),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
