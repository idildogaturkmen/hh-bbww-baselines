'''
Once I forbid tiny unstable tails, which recoMET BDT is actually best?

Reevaluate the BDTs trained on recoMET, but now only consider thresholds that are stable between validation and test sets.

'''

from pathlib import Path
import pandas as pd
import numpy as np

folders = {
    "recoMET_leptonMET": "outputs/bdt_weighted_baseline_lepemu_rough_augmented_recoMET",
    "recoMET_simple_topology": "outputs/bdt_weighted_baseline_lepemu_rough_topology_recoMET",
    "recoMET_hadronic_w_top": "outputs/bdt_weighted_baseline_lepemu_rough_hadronic_w_top_recoMET",
}

min_neff_values = [10, 25, 50]
min_s_raw = 10
min_b_raw = 25

rows = []

for model, folder in folders.items():
    path = Path(folder) / "bdt_threshold_scan_rough_weighted.csv"
    if not path.exists():
        print("Missing:", path)
        continue

    scan = pd.read_csv(path)

    val = scan[scan["split"] == "val"].copy()
    test = scan[scan["split"] == "test"].copy()

    merged = val.merge(
        test,
        on="threshold",
        suffixes=("_val", "_test"),
    )

    for min_neff in min_neff_values:
        good = merged[
            (merged["background_neff_val"] >= min_neff)
            & (merged["background_neff_test"] >= min_neff)
            & (merged["S_raw_val"] >= min_s_raw)
            & (merged["S_raw_test"] >= min_s_raw)
            & (merged["B_raw_val"] >= min_b_raw)
            & (merged["B_raw_test"] >= min_b_raw)
        ].copy()

        if len(good) == 0:
            rows.append({
                "model": model,
                "min_neff_valtest": min_neff,
                "status": "no_threshold_passes",
            })
            continue

        # Pick threshold by validation Z only, while requiring test stability.
        best = good.sort_values("asimovZ_weighted_val", ascending=False).iloc[0]

        rows.append({
            "model": model,
            "min_neff_valtest": min_neff,
            "status": "ok",
            "threshold": best["threshold"],

            "val_S_raw": best["S_raw_val"],
            "val_B_raw": best["B_raw_val"],
            "val_S_weighted": best["S_weighted_val"],
            "val_B_weighted": best["B_weighted_val"],
            "val_S_over_B": best["S_over_B_weighted_val"],
            "val_Z": best["asimovZ_weighted_val"],
            "val_bkg_neff": best["background_neff_val"],

            "test_S_raw": best["S_raw_test"],
            "test_B_raw": best["B_raw_test"],
            "test_S_weighted": best["S_weighted_test"],
            "test_B_weighted": best["B_weighted_test"],
            "test_S_over_B": best["S_over_B_weighted_test"],
            "test_Z": best["asimovZ_weighted_test"],
            "test_bkg_neff": best["background_neff_test"],
        })

out = pd.DataFrame(rows)
out_path = "outputs/bdt_recoMET_stable_thresholds.csv"
out.to_csv(out_path, index=False)

print(out.to_string(index=False))
print("\nWrote:", out_path)
