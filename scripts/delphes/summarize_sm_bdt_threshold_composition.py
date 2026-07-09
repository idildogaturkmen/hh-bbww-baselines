#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd

REPO = Path(os.environ["HH4B_REPO"])
OUTDIR = REPO / "outputs/tables/sm_normalized_hh4b_bdt_2026_07_09"

scan_path = OUTDIR / "bdt_threshold_scan.csv"
test_path = OUTDIR / "bdt_testset_by_sample.csv"

print("Existing threshold scan:")
scan = pd.read_csv(scan_path)
print(scan.sort_values("S_over_sqrtB", ascending=False).head(15).to_string(index=False))

print("\nExisting test-set by sample:")
by_sample = pd.read_csv(test_path)
print(by_sample.to_string(index=False))

print("\nRecommendation:")
print("Use BDT > 0.84–0.87 as the safer working region for now.")
print("BDT > 0.91 has better S/B but only 3 background test rows, so it is too MC-stat limited.")
