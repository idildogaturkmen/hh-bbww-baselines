#!/usr/bin/env python3

from pathlib import Path
import os
import pandas as pd
import numpy as np

REPO = Path(os.environ["HH4B_REPO"])

IN_DIR = REPO / "outputs/tables/hh4b_deepspanet_twohead_qcdplus_qcdplus_btag4_v1_2026_07_14"
OUT_DIR = REPO / "outputs/tables/deepspanet_twohead_qcdplus_diagnostics_2026_07_14"
OUT_DIR.mkdir(parents=True, exist_ok=True)

scan = pd.read_csv(IN_DIR / "threshold_scan.csv")

criteria = [
    ("raw_best_no_stability_cut", scan),
    ("min_10_bkg_rows", scan[scan["n_background_test_rows"] >= 10]),
    ("_stability_cut", scan),
    ("min_10min_25_bkg_rows", scan[scan["n_background_test_rows"] >= 25]),
    ("min_50_bkg_rows", scan[scan["n_background_test_rows"] >= 50]),
    ("min_10_neff_bkg", scan[scan["neff_background_rows"] >= 10]),
    ("min_25_neff_bkg", scan[scan["neff_background_rows"] >= 25]),
]

rows = []
for name, df in criteria:
    if len(df) == 0:
        rows.append({"criterion": name, "available": False})
        continue
    best = df.sort_values("S_over_sqrtB", ascending=False).iloc[0].to_dict()
    best["criterion"] = name
    best["available"] = True
    rows.append(best)

out = pd.DataFrame(rows)
cols = ["criterion", "available"] + [c for c in out.columns if c not in ["criterion", "available"]]
out = out[cols]

out.to_csv(OUT_DIR / "deepspanet_twohead_stability_summary.csv", index=False)
(OUT_DIR / "deepspanet_twohead_stability_summary.md").write_text(out.to_markdown(index=False) + "\n")

print(out.to_string(index=False))
print("Wrote:", OUT_DIR / "deepspanet_twohead_stability_summary.md")
