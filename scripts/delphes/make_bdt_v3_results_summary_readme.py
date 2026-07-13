#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

REPO = Path.cwd()

OUTDIR = REPO / "outputs/summaries"
OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY = OUTDIR / "bdt_v3_qcdplus_mass_aware_vs_topology_only_results_2026_07_13.md"

PATHS = {
    "mass_aware_auc": REPO / "outputs/tables/bdt_v3_qcdplus_categories_2026_07_13/category_audit_auc_summary.csv",
    "mass_aware_yields": REPO / "outputs/tables/bdt_v3_qcdplus_categories_2026_07_13/category_yields.csv",
    "mass_aware_comp": REPO / "outputs/tables/bdt_v3_qcdplus_categories_2026_07_13/category_background_composition.csv",

    "topology_auc": REPO / "outputs/tables/bdt_v3_qcdplus_topology_only_categories_2026_07_13/category_audit_auc_summary.csv",
    "topology_yields": REPO / "outputs/tables/bdt_v3_qcdplus_topology_only_categories_2026_07_13/category_yields.csv",
    "topology_comp": REPO / "outputs/tables/bdt_v3_qcdplus_topology_only_categories_2026_07_13/category_background_composition.csv",

    "inclusive_comparison": REPO / "outputs/tables/bdt_v3_mass_aware_vs_topology_only_2026_07_13/mass_aware_vs_topology_only_inclusive_cat012.csv",
    "category_comparison": REPO / "outputs/tables/bdt_v3_mass_aware_vs_topology_only_2026_07_13/mass_aware_vs_topology_only_category_summary.csv",
    "mass_sculpting": REPO / "outputs/tables/bdt_v3_mass_sculpting_comparison_2026_07_13/background_mass_sculpting_CAT012.csv",
}

missing = [str(p) for p in PATHS.values() if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def nice_float(x, digits=4):
    try:
        return f"{float(x):.{digits}g}"
    except Exception:
        return x


def compact_auc(path, label):
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "model": label,
            "classifier": r["classifier"],
            "negative class": r["negative_class"],
            "unweighted AUC": round(float(r["unweighted_auc"]), 4),
            "physics-weighted AUC": round(float(r["physics_weighted_auc"]), 4),
            "train rows": int(r["n_train_task"]),
            "test rows": int(r["n_test_task"]),
        })
    return pd.DataFrame(rows)


def compact_yields(path, label):
    df = pd.read_csv(path)
    keep = [
        "category",
        "n_signal_test_rows",
        "n_background_test_rows",
        "signal_events_450fb",
        "background_events_450fb",
        "S_over_B",
        "S_over_sqrtB",
        "S_over_10pctB",
        "neff_background",
    ]
    out = df[keep].copy()
    out.insert(0, "model", label)

    for c in ["signal_events_450fb", "background_events_450fb"]:
        out[c] = out[c].map(lambda x: round(float(x), 3))
    for c in ["S_over_B", "S_over_sqrtB", "S_over_10pctB", "neff_background"]:
        out[c] = out[c].map(lambda x: round(float(x), 6))

    return out


def top_background_components(path, label, n=3):
    df = pd.read_csv(path)
    rows = []
    for cat, sub in df.groupby("category"):
        sub = sub.sort_values("background_events_450fb", ascending=False).head(n)
        for _, r in sub.iterrows():
            rows.append({
                "model": label,
                "category": cat,
                "background sample": r["analysis_sample"],
                "test rows": int(r["n_test_rows"]),
                "yield at 450/fb": round(float(r["background_events_450fb"]), 3),
                "fraction of category B": round(float(r["fraction_of_category_background"]), 4),
                "neff sample": round(float(r["neff_sample"]), 3),
            })
    return pd.DataFrame(rows)


def inclusive_table(path):
    df = pd.read_csv(path)
    out = df.copy()
    for c in ["signal_events_450fb", "background_events_450fb"]:
        out[c] = out[c].map(lambda x: round(float(x), 3))
    for c in ["S_over_B", "S_over_sqrtB", "S_over_10pctB"]:
        out[c] = out[c].map(lambda x: round(float(x), 6))
    return out


def mass_sculpting_table(path):
    df = pd.read_csv(path)
    keep = [
        "run",
        "category",
        "yield_450fb",
        "neff",
        "mbb1_median",
        "mbb2_median",
        "avg_mbb_median",
        "mhh_median",
        "r_hh_median",
        "avg_mbb_distance_from_125",
    ]
    out = df[keep].copy()
    for c in keep:
        if c not in ["run", "category"]:
            out[c] = out[c].map(lambda x: round(float(x), 3))
    return out


mass_auc = compact_auc(PATHS["mass_aware_auc"], "mass-aware BDT-v3")
topo_auc = compact_auc(PATHS["topology_auc"], "topology-only BDT-v3")
auc = pd.concat([mass_auc, topo_auc], ignore_index=True)

mass_yields = compact_yields(PATHS["mass_aware_yields"], "mass-aware BDT-v3")
topo_yields = compact_yields(PATHS["topology_yields"], "topology-only BDT-v3")
yields = pd.concat([mass_yields, topo_yields], ignore_index=True)

mass_comp = top_background_components(PATHS["mass_aware_comp"], "mass-aware BDT-v3")
topo_comp = top_background_components(PATHS["topology_comp"], "topology-only BDT-v3")
comp = pd.concat([mass_comp, topo_comp], ignore_index=True)

inclusive = inclusive_table(PATHS["inclusive_comparison"])
sculpt = mass_sculpting_table(PATHS["mass_sculpting"])

text = f"""# BDT-v3 qcdplus results summary: mass-aware vs topology-only

Date: 2026-07-13  
Branch: `delphes-hh4b-production`

## Purpose

This note summarizes the current BDT-v3 qcdplus baseline for the HH→4b Delphes analysis before moving to DNN, LBN-DNN, and SPA-Net.

Two BDT versions are compared:

1. **Mass-aware BDT-v3 qcdplus**
   - Uses candidate mass variables such as `mbb1`, `mbb2`, `avg_mbb`, `delta_mbb`, `r_hh`, `r_hh_125_125`, `r_hh_125_120`, `mhh`, and jet masses.
   - This is the strongest current BDT baseline.

2. **Topology-only BDT-v3 qcdplus**
   - Removes direct Higgs-candidate mass variables.
   - Used as a mass-sculpting control.

Both use the same enlarged-QCD sample set and the same train/test split logic.

## Main conclusion

The mass-aware BDT-v3 is the strongest current baseline. It gives higher AUC and higher expected sensitivity than the topology-only BDT. However, the mass-aware BDT also pulls background events closer to the Higgs-like mass region, so it is mass-sculpting. The topology-only BDT is weaker but provides an important robustness control.

## AUC summary

{auc.to_markdown(index=False)}

## Exclusive category yields

{yields.to_markdown(index=False)}

## Inclusive CAT0+CAT1+CAT2 comparison

The region CAT0+CAT1+CAT2 corresponds approximately to the best supported inclusive BDT-v3 rectangle: `qcd_score >= 0.800` and `top_score >= 0.500`.

{inclusive.to_markdown(index=False)}

## Top background components by category

Only the top three background components per category are shown.

{comp.to_markdown(index=False)}

## Background mass-sculpting check for CAT0-CAT2

This table focuses on background events because mass sculpting is mainly a background-modeling concern.

{sculpt.to_markdown(index=False)}

## Interpretation

### Mass-aware BDT-v3

The mass-aware BDT-v3 gives the best current sensitivity. CAT0 has the highest purity, but it is MC-stat limited. CAT1 is the most defensible tight category. The inclusive CAT0+CAT1+CAT2 region gives the best supported signal-enriched region.

### Topology-only BDT-v3

The topology-only BDT-v3 has lower AUC and lower expected sensitivity. This confirms that direct mass information is important for the current BDT performance. However, topology-only separation is nonzero, so event topology does contain useful discrimination power.

### Mass sculpting

The mass-aware BDT pulls background candidate masses closer to the Higgs-like region than the topology-only BDT, especially in CAT0 and CAT1. This is expected because the mass-aware BDT directly uses mass variables. This does not invalidate the mass-aware BDT, but it means any final paper-style study should report both mass-aware performance and topology-only control results.

## Current baseline choice

For future comparisons, use:

- **Main baseline:** mass-aware BDT-v3 qcdplus
- **Robustness/control baseline:** topology-only BDT-v3 qcdplus
- **Main signal-enriched region:** CAT0+CAT1+CAT2
- **Most defensible tight exclusive category:** CAT1
- **Diagnostic only:** CAT0, unless more MC statistics are produced

## Next analysis steps

1. Train DNN-v3 using the same mass-aware and topology-only feature sets.
2. Compare DNN-v3 to BDT-v3 using AUC, category yields, S/B, S/sqrt(B), background composition, and mass sculpting.
3. Train LBN-DNN after ordinary DNN.
4. Move to SPA-Net only after the tabular baselines are frozen and understood.
"""

SUMMARY.write_text(text)
print("Wrote", SUMMARY)
