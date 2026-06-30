'''
Scans the recoMET BDT input features to find the best cuts on lepton MET features.

'''

import numpy as np
import pandas as pd
from pathlib import Path

def asimov_z(s, b):
    if b <= 0:
        return np.sqrt(2*s) if s > 0 else 0.0
    if s <= 0:
        return 0.0
    return float(np.sqrt(2*((s+b)*np.log(1+s/b)-s)))

def neff(w):
    w = np.asarray(w, dtype=float)
    den = np.sum(w*w)
    return float(np.sum(w)**2 / den) if den > 0 else 0.0

path = "outputs/lepton_met_features_lepemu_rough_recoMET/onelep_pre_mbb_classifier_input_with_lepton_met_features.parquet"
outdir = Path("outputs/lepton_met_features_lepemu_rough_recoMET")
df = pd.read_parquet(path)

df = df[
    df["has_physics_weight"]
    & np.isfinite(df["physics_weight_nominal"])
].copy()

base = (
    (df["ht"] >= 100)
    & (df["mbb_pt_binned_scaled"] >= 80)
    & (df["mbb_pt_binned_scaled"] <= 150)
)

rows = []
for mt_min in [0, 20, 30, 40, 50, 60, 80, 100]:
    for reliso_max in [None, 0.4, 0.2, 0.1]:
        mask = base.copy()

        if mt_min > 0:
            mask &= df["mt_lep_met"] >= mt_min

        if reliso_max is not None:
            mask &= df["leading_lepton_reliso"] <= reliso_max

        sub = df[mask]
        sig = sub[sub["target"] == 1]
        bkg = sub[sub["target"] == 0]

        sw = float(sig["physics_weight_nominal"].sum())
        bw = float(bkg["physics_weight_nominal"].sum())

        rows.append({
            "mt_min": mt_min,
            "reliso_max": reliso_max if reliso_max is not None else "none",
            "S_raw": len(sig),
            "B_raw": len(bkg),
            "S_weighted": sw,
            "B_weighted": bw,
            "S_over_B": sw / bw if bw > 0 else np.inf,
            "Z": asimov_z(sw, bw),
            "background_neff": neff(bkg["physics_weight_nominal"]),
            "top_backgrounds": ", ".join(
                bkg.groupby("process_group")["physics_weight_nominal"]
                .sum()
                .sort_values(ascending=False)
                .head(4)
                .index
                .tolist()
            ),
        })

out = pd.DataFrame(rows).sort_values(["Z", "S_over_B"], ascending=False)
out_path = outdir / "recoMET_lepton_met_cut_scan.csv"
out.to_csv(out_path, index=False)

print(out.head(30).to_string(index=False))
print("\nWrote:", out_path)
