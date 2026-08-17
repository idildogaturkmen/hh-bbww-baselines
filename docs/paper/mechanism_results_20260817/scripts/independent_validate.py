#!/usr/bin/env python3
"""Independent validation pass (Task 8).

Re-reads the RAW frozen source files directly -- NOT MECHANISM_RESULTS_DATA.json,
NOT the generated .tex tables, NOT the generated figures -- and independently
recomputes every quantity this package plots or tabulates, in a fresh code
path. Then cross-checks the independently-recomputed numbers against:
  (a) MECHANISM_RESULTS_DATA.json (the extraction script's output)
  (b) the 4 generated .tex tables (regex-parsed back out)

If anything disagrees beyond floating-point rounding, this script reports a
MISMATCH and exits non-zero. Per the governing instruction, a mismatch means
STOP and report -- this script does not silently "fix" anything; it only
writes VALIDATION_REPORT.json describing exactly what was checked and what,
if anything, disagreed.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from frozen_paths import (
    TRACK_A_EVALUATION_RESULTS_JSON,
    TRACK_A_SUPPORTED_REJECTION_RANGE_TSV,
    TRACK_B_HOLDOUT_A_WORKING_POINTS_TSV,
    TRACK_B_HOLDOUT_A_SEED_STABILITY_TSV,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "MECHANISM_RESULTS_DATA.json").read_text())

TOL_REL = 1e-6    # full-precision JSON-vs-JSON cross-check: must be essentially exact
TSV_TOL_REL = 1.5e-3  # recomputed-vs-source-TSV cross-check: R_B mean/std are stored
                       # to 3dp (rounding floor ~1.2e-4 rel.) but raw_qcd_mean is stored
                       # to only 2dp, which at this dataset's smallest values (e.g. 4.67)
                       # gives a rounding floor of ~1.1e-3 relative -- this tolerance
                       # absorbs that known text-rounding floor and nothing more (a
                       # genuine data error here would be orders of magnitude larger)
TABLE_TOL_REL = 2e-2  # .tex table cross-check: absorbs the table's own 1-2dp rounding
                        # on small-magnitude cells (e.g. "raw QCD (mean)" 2.67 -> "2.7")

# Confirmed by reading the actual source scripts that produced each track's
# seed-stability numbers (not assumed): Track-B's phase4L/work/make_plots.py
# uses np.nanstd(..., axis=0) -- default ddof=0, population std. Track-A's
# post_training_evaluation_20260815_v2/scripts/evaluate_full.py uses
# np.std(..., ddof=1) explicitly -- sample std -- everywhere. This is a real,
# pre-existing difference in each track's own established convention, not
# unified here (that would mean altering a frozen number); this script's own
# recomputation must match each track's actual convention to be a valid
# cross-check, so the two tracks use different ddof below on purpose.

checks = []


def record(name, ok, detail):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})


def close(a, b, tol=TOL_REL):
    if b == 0:
        return abs(a - b) < 1e-9
    return abs(a - b) / abs(b) < tol


# ---------------------------------------------------------------------------
# 1. Track-B: independently re-read the two raw TSVs, recompute tiers, recompute
#    the mechanism-gain ratios from scratch, cross-check vs MECHANISM_RESULTS_DATA.json
# ---------------------------------------------------------------------------

def tier(raw_qcd):
    if raw_qcd >= 100:
        return "well_supported"
    if raw_qcd >= 10:
        return "statistically_limited_but_reportable"
    return "exploratory_not_primary_quantitative_claim"


def check_trackb():
    with open(TRACK_B_HOLDOUT_A_SEED_STABILITY_TSV) as f:
        seed_rows = list(csv.DictReader(f, delimiter="\t"))
    with open(TRACK_B_HOLDOUT_A_WORKING_POINTS_TSV) as f:
        pt_rows = list(csv.DictReader(f, delimiter="\t"))

    seed_agg = {(r["arm"], float(r["eps_S_target"])): r for r in seed_rows}
    by_key = {}
    for r in pt_rows:
        by_key.setdefault((r["arm"], float(r["eps_S_target"])), []).append(r)

    n_checked = 0
    for (arm, eps_s), rows in by_key.items():
        rows = sorted(rows, key=lambda r: int(r["seed"]))
        assert len(rows) == 3
        rbs = [float(r["R_B"]) for r in rows]
        raws = [int(r["raw_surviving_qcd"]) for r in rows]
        mean_rb = sum(rbs) / 3
        # ddof=0 (population std, divide by N): matches phase4L/work/make_plots.py's
        # np.nanstd(..., axis=0) default, confirmed by reading that script directly.
        std_rb = (sum((x - mean_rb) ** 2 for x in rbs) / 3) ** 0.5
        mean_raw = sum(raws) / 3

        agg = seed_agg[(arm, eps_s)]
        ok_mean = close(mean_rb, float(agg["R_B_mean"]), tol=TSV_TOL_REL)
        ok_std = close(std_rb, float(agg["R_B_std"]), tol=TSV_TOL_REL)
        ok_raw = close(mean_raw, float(agg["raw_qcd_mean"]), tol=TSV_TOL_REL)
        record(f"trackb_seed_agg_recompute[{arm},eps_S={eps_s}]",
                ok_mean and ok_std and ok_raw,
                f"recomputed mean={mean_rb:.6f} (file={agg['R_B_mean']}), "
                f"std={std_rb:.6f} (file={agg['R_B_std']}), "
                f"raw_mean={mean_raw:.6f} (file={agg['raw_qcd_mean']})")

        # every individual seed's own support_label must match the tier() function
        for r in rows:
            expected = tier(int(r["raw_surviving_qcd"]))
            ok = expected == r["support_label"]
            record(f"trackb_per_seed_support_label[{arm},eps_S={eps_s},seed={r['seed']}]",
                    ok, f"raw_qcd={r['raw_surviving_qcd']} expected={expected} file={r['support_label']}")
        n_checked += 1

    record("trackb_n_working_points_checked", n_checked == 18, f"n={n_checked} (expected 18 = 3 arms x 6 eps_S)")

    # mechanism-gain ratios, fully independent recomputation
    eps_grid = sorted({float(r["eps_S_target"]) for r in pt_rows})
    for eps_s in eps_grid:
        for num_arm, den_arm, key in [("E1", "E0", "E1_over_E0"), ("E2", "E1", "E2_over_E1")]:
            num_rows = sorted(by_key[(num_arm, eps_s)], key=lambda r: int(r["seed"]))
            den_rows = sorted(by_key[(den_arm, eps_s)], key=lambda r: int(r["seed"]))
            ratios = [float(n["R_B"]) / float(d["R_B"]) for n, d in zip(num_rows, den_rows)]
            mean_ratio = sum(ratios) / 3
            std_ratio = (sum((x - mean_ratio) ** 2 for x in ratios) / 2) ** 0.5
            ratio_of_means = (
                float(seed_agg[(num_arm, eps_s)]["R_B_mean"]) / float(seed_agg[(den_arm, eps_s)]["R_B_mean"])
            )

            data_row = next(r for r in DATA["track_b"]["mechanism_gain"] if r["eps_S"] == eps_s)[key]
            ok1 = close(mean_ratio, data_row["mean_of_per_seed_ratios"])
            ok2 = close(std_ratio, data_row["std_of_per_seed_ratios"], tol=1e-4)
            ok3 = close(ratio_of_means, data_row["ratio_of_seed_means"])
            record(f"trackb_gain_recompute[{key},eps_S={eps_s}]", ok1 and ok2 and ok3,
                    f"mean_of_ratios={mean_ratio:.6f} (data={data_row['mean_of_per_seed_ratios']:.6f}), "
                    f"ratio_of_means={ratio_of_means:.6f} (data={data_row['ratio_of_seed_means']:.6f})")


# ---------------------------------------------------------------------------
# 2. Track-A: independently re-read EVALUATION_RESULTS.json + support TSV
# ---------------------------------------------------------------------------

def check_tracka():
    with open(TRACK_A_EVALUATION_RESULTS_JSON) as f:
        ev = json.load(f)
    with open(TRACK_A_SUPPORTED_REJECTION_RANGE_TSV) as f:
        support_rows = list(csv.DictReader(f, delimiter="\t"))

    # model ordering at eps_S = 0.40 (point estimates), independently sorted
    refs = ev["references"]
    entries = {
        "cut_baseline": refs["CUT_BASELINE_historical_RHH_lt_34"]["R_B_rejection_factor_1_over_epsilon_B"],
        "bdt_control0": refs["BDT_CONTROL_0"]["rb_eps_s_0.40_recomputed_here"],
        "dnn": ev["models"]["dnn"]["seed_to_seed_variation"]["rb_eps_S=0.4_mean"],
        "cmstransformer": ev["models"]["cmstransformer"]["seed_to_seed_variation"]["rb_eps_S=0.4_mean"],
        "fivejetpartnet": refs["FiveJetParTNet"]["rb_eps_s_0.40_frozen_reference"],
        "bdt_newc": refs["BDT_NEW_C"]["rb_eps_s_0.40_recomputed_here"],
    }
    recomputed_order = sorted(entries, key=lambda k: entries[k])
    expected_order = ["cut_baseline", "bdt_control0", "dnn", "cmstransformer", "fivejetpartnet", "bdt_newc"]
    record("tracka_model_ordering_eps_S_0.40", recomputed_order == expected_order,
            f"recomputed order={recomputed_order}")

    data_models = {m["short"]: m["R_B"] for m in DATA["track_a"]["fig3_models"]}
    for k, v in entries.items():
        ok = close(v, data_models[k])
        record(f"tracka_fig3_value_cross_check[{k}]", ok, f"raw_source={v} data_json={data_models[k]}")

    # seed-to-seed aggregation, independent recompute, both models, all eps_S
    for model_key in ["dnn", "cmstransformer"]:
        per_seed = ev["models"][model_key]["per_seed"]
        stv = ev["models"][model_key]["seed_to_seed_variation"]
        for eps_s in ev["efficiency_grid"]["epsilon_S"]:
            gk = f"eps_S={eps_s}"
            vals = [per_seed[s]["rb_grid"][gk]["rejection"] for s in per_seed]
            m = sum(vals) / len(vals)
            sd = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
            ok_m = close(m, stv[f"rb_{gk}_mean"])
            ok_s = close(sd, stv[f"rb_{gk}_std"], tol=1e-4)
            record(f"tracka_seed_agg_recompute[{model_key},{gk}]", ok_m and ok_s,
                    f"recomputed mean={m:.6f} (file={stv[f'rb_{gk}_mean']:.6f}), "
                    f"std={sd:.6f} (file={stv[f'rb_{gk}_std']:.6f})")

    # support flags: every eps_S-grid row for both models must be True (all 3 seeds)
    n_supportable_checked = 0
    for model_key in ["dnn", "cmstransformer"]:
        for seed, block in ev["models"][model_key]["per_seed"].items():
            for gk, row in block["rb_grid"].items():
                expected_flag = (
                    row["raw_background_rows"] >= 100
                    and row["qcd_raw_rows"] >= 10
                    and row["qcd_Neff"] >= 10
                )
                ok = expected_flag == row["scientifically_supportable"]
                record(f"tracka_support_gate_recompute[{model_key},seed={seed},{gk}]", ok,
                        f"raw_bkg={row['raw_background_rows']}, qcd_raw={row['qcd_raw_rows']}, "
                        f"qcd_Neff={row['qcd_Neff']:.3f}, recomputed={expected_flag}, "
                        f"file={row['scientifically_supportable']}")
                n_supportable_checked += 1
    record("tracka_n_support_flags_checked", n_supportable_checked == 36,
            f"n={n_supportable_checked} (expected 36 = 2 models x 3 seeds x 6 eps_S)")

    # fixed-epsilon_B tail rows (primary seed only) -- independent gate recompute
    n_tail = 0
    for r in support_rows:
        if r["working_point_type"] != "fixed_epsilon_B":
            continue
        raw_bkg, qcd_raw, qcd_neff = int(r["raw_background_rows"]), int(r["qcd_raw_rows"]), float(r["qcd_Neff"])
        expected_flag = raw_bkg >= 100 and qcd_raw >= 10 and qcd_neff >= 10
        file_flag = r["scientifically_supportable"] == "True"
        ok = expected_flag == file_flag
        record(f"tracka_eps_b_tail_gate_recompute[{r['model']},eps_B={r['target_epsilon_B']}]", ok,
                f"raw_bkg={raw_bkg}, qcd_raw={qcd_raw}, qcd_Neff={qcd_neff:.3f}, "
                f"recomputed={expected_flag}, file={file_flag}")
        n_tail += 1
    record("tracka_n_eps_b_tail_checked", n_tail == 8, f"n={n_tail} (expected 8 = 2 models x 4 eps_B)")

    # no unsupported model/point was assigned a fabricated uncertainty:
    # every fig3 entry with kind != measured_this_benchmark must have R_B_err None
    for m in DATA["track_a"]["fig3_models"]:
        if m["kind"] != "measured_this_benchmark":
            ok = m["R_B_err"] is None
            record(f"tracka_no_fabricated_uncertainty[{m['short']}]", ok,
                    f"kind={m['kind']} R_B_err={m['R_B_err']}")
        else:
            ok = m["R_B_err"] is not None
            record(f"tracka_measured_model_has_uncertainty[{m['short']}]", ok, f"R_B_err={m['R_B_err']}")


# ---------------------------------------------------------------------------
# 3. Cross-check generated .tex tables against MECHANISM_RESULTS_DATA.json
# ---------------------------------------------------------------------------

def check_tex_tables():
    trackb_tex = (ROOT / "TRACKB_MECHANISM_TABLE.tex").read_text()
    row_re = re.compile(
        r"^([\d.]+) & (E\d) & ([\d.]+) & \$([\d.]+) \\pm ([\d.]+)\$ & ([\d.]+) & \\(\w+) \\\\$",
        re.MULTILINE,
    )
    n = 0
    for m in row_re.finditer(trackb_tex):
        eps_s, arm, auc, rb_mean, rb_std, raw_qcd, supp_tag = m.groups()
        w = next(w for w in DATA["track_b"]["working_points"]
                  if w["arm"] == arm and close(w["eps_S"], float(eps_s), tol=1e-3))
        ok = (close(float(rb_mean), w["R_B_mean"], tol=TABLE_TOL_REL)
              and close(float(rb_std), w["R_B_std"], tol=TABLE_TOL_REL)
              and close(float(raw_qcd), w["raw_qcd_mean"], tol=TABLE_TOL_REL))
        record(f"tex_table_A_row[{arm},eps_S={eps_s}]", ok,
                f"tex R_B={rb_mean}+/-{rb_std} raw_qcd={raw_qcd}; "
                f"json R_B={w['R_B_mean']:.3f}+/-{w['R_B_std']:.3f} raw_qcd={w['raw_qcd_mean']:.3f}")
        n += 1
    record("tex_table_A_rows_found", n == 18, f"n={n} (expected 18)")

    tracka_tex = (ROOT / "TRACKA_MECHANISM_TABLE.tex").read_text()
    for m in DATA["track_a"]["fig3_models"]:
        val_str = f"{m['R_B']:.2f}"
        ok = val_str in tracka_tex
        record(f"tex_table_B_value_present[{m['short']}]", ok, f"looked for '{val_str}'")


def check_paper_draft_prose():
    """Cross-check every percentage figure quoted in the paper draft's prose
    against EVALUATION_RESULTS.json / MECHANISM_RESULTS_DATA.json directly.
    This check exists because manual transcription of a JSON figure into
    prose caught two real slips during drafting (one in this package's own
    first draft, one inherited from v2's own RESULTS_SUMMARY.md prose) --
    both fixed before freezing; this makes the catch permanent/automated."""
    draft = (ROOT / "MECHANISM_RESULTS_PAPER_DRAFT.md").read_text()

    gain_by_eps = {row["eps_S"]: row for row in DATA["track_b"]["mechanism_gain"]}
    for eps_s, expect_e0e1, expect_e1e2 in [
        (0.6, "+23%", "-32%"), (0.5, "+37%", "-39%"), (0.4, "+54%", "-44%"), (0.1, "+79%", "+29%"),
    ]:
        row = gain_by_eps[eps_s]
        actual_e0e1 = row["E1_over_E0"]["pct_change_ratio_of_means"]
        actual_e1e2 = row["E2_over_E1"]["pct_change_ratio_of_means"]
        rounded_e0e1 = f"{actual_e0e1:+.0f}%"
        rounded_e1e2 = f"{actual_e1e2:+.0f}%"
        record(f"paper_draft_prose_gain_pct[eps_S={eps_s},E1/E0]", rounded_e0e1 == expect_e0e1,
                f"prose claims {expect_e0e1}, recomputed {rounded_e0e1} (raw {actual_e0e1:.2f}%)")
        record(f"paper_draft_prose_gain_pct[eps_S={eps_s},E2/E1]", rounded_e1e2 == expect_e1e2,
                f"prose claims {expect_e1e2}, recomputed {rounded_e1e2} (raw {actual_e1e2:.2f}%)")
        for pct_str in (expect_e0e1, expect_e1e2):
            record(f"paper_draft_prose_string_present[{pct_str}]", pct_str in draft,
                    f"looked for literal '{pct_str}' in MECHANISM_RESULTS_PAPER_DRAFT.md")

    paired = DATA["track_a"]["paired_bootstrap"]["by_epsilon_s"]
    dnn_gt_at_010 = paired["eps_S=0.1"]["fraction_replicates_dnn_gt_transformer"]
    expect_str = f"{dnn_gt_at_010*100:.1f}%"
    record("paper_draft_prose_eps010_dnn_fraction", expect_str in draft,
            f"expected literal '{expect_str}' (from EVALUATION_RESULTS.json="
            f"{dnn_gt_at_010}) present in MECHANISM_RESULTS_PAPER_DRAFT.md")
    record("paper_draft_prose_eps010_dnn_fraction_not_source_typo", "63.6%" not in draft,
            "the source v2 RESULTS_SUMMARY.md prose's own transcription slip "
            "('63.6%', vs. its own table/JSON value 0.641=64.1%) must not appear in this package's prose")


def main() -> int:
    check_trackb()
    check_tracka()
    check_tex_tables()
    check_paper_draft_prose()

    n_ok = sum(1 for c in checks if c["ok"])
    n_total = len(checks)
    mismatches = [c for c in checks if not c["ok"]]

    report = {
        "n_checks": n_total,
        "n_passed": n_ok,
        "n_mismatches": len(mismatches),
        "status": "ALL_CHECKS_PASS" if not mismatches else "MISMATCH_FOUND_STOP",
        "checks": checks,
    }
    (ROOT / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2))

    print(f"{n_ok}/{n_total} checks passed.")
    if mismatches:
        print("STOP -- mismatches found, not silently corrected:", file=sys.stderr)
        for c in mismatches:
            print(f"  [{c['check']}] {c['detail']}", file=sys.stderr)
        return 1
    print("All checks pass. VALIDATION_REPORT.json written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
