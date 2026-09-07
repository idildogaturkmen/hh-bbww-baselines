#!/usr/bin/env python3
"""Merge signal + QCD + non-QCD tail extractions into one master
Parquet table, with the frozen 450 fb^-1 analysis weight attached per
process (reused, not re-derived, from track_b_four_model_physical_normalization_freeze_20260823_v1)."""
import numpy as np
import pandas as pd

WORK = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1/work"

SIGNAL_WEIGHT_450FB = 0.000473265
WEIGHTS_450FB = {
    "QCD": 7.3112875413134075,
    "inference_ttbar_1": 2.3214135411747647,
    "inference_ttbar_2": 2.3214135411747647,
    "ZJetsToQQ": 1.41031125,
    "TTbarZ": 0.38655,
    "ZZ": 1.078461607362026,
    "ZH": 1.1418,
    "ttH": 0.76065,
    "SingleHiggs": 2.1861,
    "SingleTop": 4.884600426532971,
    "TTbarW": 0.335385,
    "TW": 6.227102637699448,
    "VBFH": 1.7019,
    "WW": 3.72725937405907,
    "WminusH": 0.4794309588619177,
    "WplusH": 0.7559236066249192,
    "ZW": 2.9552364505296116,
}


def main():
    sig = pd.read_parquet(f"{WORK}/signal_tail.parquet")
    qcd = pd.read_parquet(f"{WORK}/qcd_tail.parquet")
    nonqcd = pd.read_parquet(f"{WORK}/nonqcd_tail.parquet")

    sig["process"] = "signal"
    sig["label"] = 1
    qcd["label"] = 0
    nonqcd["label"] = 0

    sig["weight_450fb"] = SIGNAL_WEIGHT_450FB
    qcd["weight_450fb"] = qcd["process"].map(WEIGHTS_450FB)
    nonqcd["weight_450fb"] = nonqcd["process"].map(WEIGHTS_450FB)
    assert qcd["weight_450fb"].notna().all()
    assert nonqcd["weight_450fb"].notna().all()

    # background_class: QCD vs ttbar vs "other" (ZJetsToQQ/SingleTop/TTbarW/TW/TTbarZ/ttH/etc.)
    def bkg_class(row):
        if row["process"] == "signal":
            return "signal"
        if row["process"] == "QCD":
            return "QCD"
        if row["process"] in ("inference_ttbar_1", "inference_ttbar_2"):
            return "ttbar"
        return "other_background"

    common_cols = sorted(set(sig.columns) & set(qcd.columns) & set(nonqcd.columns))
    all_cols = sorted(set(sig.columns) | set(qcd.columns) | set(nonqcd.columns))
    for df in (sig, qcd, nonqcd):
        for c in all_cols:
            if c not in df.columns:
                df[c] = np.nan

    master = pd.concat([sig[all_cols], qcd[all_cols], nonqcd[all_cols]], ignore_index=True)
    master["background_class"] = master.apply(bkg_class, axis=1)
    master["event_uid"] = master["process"].astype(str) + "|" + master["file_index"].astype(str) + "|" + master["entry_index"].astype(str)

    # sanity checks
    assert master["event_uid"].is_unique, "duplicate event_uid found -- STOP"
    assert (master["u"] > 3.5).all(), "a row with u<=3.5 leaked into the master table -- STOP"
    n_u35 = len(master)
    n_u45 = (master["u"] > 4.5).sum()
    print(f"master table: n_total(u>3.5)={n_u35}, n(u>4.5)={n_u45}")
    print(master.groupby(["background_class"])["u"].agg(["count"]))
    print(master[master["u"] > 4.5].groupby(["background_class"])["u"].agg(["count"]))

    out = f"{WORK}/../HARVEY_MASTER_EVENT_TABLE.parquet"
    master.to_parquet(out, index=False)
    print(f"wrote {out}, shape={master.shape}")
    return master


if __name__ == "__main__":
    main()
