"""v3 Task 6: independent adversarial self-check.

Re-derives the headline numbers a SECOND, independent way (pure-Python
loops instead of vectorized numpy filtering; `decimal`-module threshold
conversion instead of `math`), checks '>' vs '>=' semantics explicitly,
re-verifies score==1.0 handling, and re-hashes the untouched v2 package
to prove no modification. No model/checkpoint/HDF5 touched.
"""
import ast
import decimal
import hashlib
import json
import math

import numpy as np
import pyarrow.parquet as pq

V2_DIR = ("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-bdt/docs/checkpoints/"
          "track_b_harvey_fp32_score_precision_20260906_v2_governing10m")
PARQUET_PATH = f"{V2_DIR}/SPA10M_EVENT_LOGITS_400K.parquet"
OUT_DIR = ("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-bdt/docs/checkpoints/"
           "track_b_harvey_fp32_score_precision_20260907_v3_exact_harvey_thresholds")

THRESHOLDS = [0.9997, 0.99997]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decimal_threshold_check(thr):
    """Independent high-precision (50-digit decimal) recomputation of
    u and Delta, cross-checked against the float64 math.log/log10 path
    used in exact_threshold_analysis.py."""
    decimal.getcontext().prec = 50
    p = decimal.Decimal(str(thr))
    one_minus = decimal.Decimal(1) - p
    u_decimal = -(one_minus.ln() / decimal.Decimal(10).ln())
    delta_decimal = (p / one_minus).ln()

    u_float64 = -math.log10(1.0 - thr)
    delta_float64 = math.log(thr / (1.0 - thr))

    return dict(
        threshold=thr,
        u_decimal_50digit=str(u_decimal),
        u_float64=u_float64,
        u_agrees_to_1e10=bool(abs(float(u_decimal) - u_float64) < 1e-10),
        delta_decimal_50digit=str(delta_decimal),
        delta_float64=delta_float64,
        delta_agrees_to_1e10=bool(abs(float(delta_decimal) - delta_float64) < 1e-10),
    )


def python_loop_recount(score32_list, delta_list, thr, delta_thr):
    """Second, independent (non-vectorized) pass-count implementation:
    a plain Python for-loop over lists, not numpy boolean masking."""
    n_fp32 = 0
    n_logit = 0
    n_both = 0
    n_fp32_ge_only = 0  # score32 == thr exactly (to check > vs >= sensitivity)
    for s, dl in zip(score32_list, delta_list):
        p = s > thr
        l = dl > delta_thr
        if s == thr:
            n_fp32_ge_only += 1
        if p:
            n_fp32 += 1
        if l:
            n_logit += 1
        if p and l:
            n_both += 1
    return dict(n_pass_fp32=n_fp32, n_pass_logit=n_logit, n_pass_both=n_both,
                n_events_exactly_equal_to_threshold=n_fp32_ge_only)


def main():
    results = {}

    # ---- 1. decimal cross-check of threshold conversions ----
    results["decimal_threshold_crosscheck"] = {repr(t): decimal_threshold_check(t) for t in THRESHOLDS}
    for t, r in results["decimal_threshold_crosscheck"].items():
        print(f"thr={t}: u agrees(1e-10)={r['u_agrees_to_1e10']}  delta agrees(1e-10)={r['delta_agrees_to_1e10']}")

    # ---- 2. independent pure-Python-loop recount (signal population only, dominant case) ----
    t = pq.read_table(PARQUET_PATH)
    d = t.to_pydict()
    score32 = np.asarray(d["score_float32_production"])
    delta = np.asarray(d["delta"])
    proc = np.asarray(d["process_label"])
    is_signal = proc == "signal"

    score32_sig = score32[is_signal].tolist()
    delta_sig = delta[is_signal].tolist()

    loop_checks = {}
    for thr in THRESHOLDS:
        delta_thr = math.log(thr / (1.0 - thr))
        loop_result = python_loop_recount(score32_sig, delta_sig, thr, delta_thr)
        # vectorized reference for comparison
        vec_pass_fp32 = int((score32[is_signal] > thr).sum())
        vec_pass_logit = int((delta[is_signal] > delta_thr).sum())
        loop_checks[repr(thr)] = dict(
            loop=loop_result,
            vectorized_pass_fp32=vec_pass_fp32,
            vectorized_pass_logit=vec_pass_logit,
            loop_matches_vectorized_fp32=bool(loop_result["n_pass_fp32"] == vec_pass_fp32),
            loop_matches_vectorized_logit=bool(loop_result["n_pass_logit"] == vec_pass_logit),
        )
        print(f"thr={thr}: loop_fp32={loop_result['n_pass_fp32']} vec_fp32={vec_pass_fp32} "
              f"match={loop_result['n_pass_fp32'] == vec_pass_fp32} | "
              f"loop_logit={loop_result['n_pass_logit']} vec_logit={vec_pass_logit} "
              f"match={loop_result['n_pass_logit'] == vec_pass_logit} | "
              f"n_exactly_eq_threshold={loop_result['n_events_exactly_equal_to_threshold']}")
    results["independent_loop_recount_signal"] = loop_checks

    # ---- 3. '>' vs '>=' semantics: explicitly show whether it matters here ----
    ge_vs_gt = {}
    for thr in THRESHOLDS:
        n_gt = int((score32 > thr).sum())
        n_ge = int((score32 >= thr).sum())
        ge_vs_gt[repr(thr)] = dict(n_strictly_greater=n_gt, n_greater_or_equal=n_ge,
                                    n_exactly_equal_to_threshold=n_ge - n_gt,
                                    gt_and_ge_identical=bool(n_gt == n_ge))
        print(f"thr={thr}: n(>)={n_gt} n(>=)={n_ge} identical={n_gt == n_ge}")
    results["gt_vs_ge_semantics"] = ge_vs_gt

    # ---- 4. score==1.0 handling re-verified ----
    n_eq1 = int((score32 == 1.0).sum())
    u_prob32 = np.asarray(d["u_prob32"])
    n_inf_u = int((~np.isfinite(u_prob32)).sum())
    results["score_eq_1_handling"] = dict(
        n_score_eq_1p0=n_eq1, n_u_prob32_nonfinite=n_inf_u,
        matches=bool(n_eq1 == n_inf_u),
    )
    print(f"score==1.0 count={n_eq1}, u_prob32 non-finite count={n_inf_u}, matches={n_eq1 == n_inf_u}")

    # ---- 5. re-verify v2 package untouched ----
    v2_sha_now = sha256_file(PARQUET_PATH)
    with open(f"{V2_DIR}/SHA256SUMS") as f:
        v2_sha_recorded = None
        for line in f:
            if "SPA10M_EVENT_LOGITS_400K.parquet" in line:
                v2_sha_recorded = line.split()[0]
                break
    results["v2_package_untouched"] = dict(
        sha256_now=v2_sha_now, sha256_recorded_in_v2_SHA256SUMS=v2_sha_recorded,
        match=bool(v2_sha_now == v2_sha_recorded),
    )
    print(f"v2 parquet sha256 unchanged: {v2_sha_now == v2_sha_recorded}")

    # ---- 6. no-new-inference check over this package's own scripts, via AST ----
    # (AST-based, not naive substring grep: a substring scan would false-positive
    # on this very script, since it must literally name the forbidden module
    # names below to check for them. Parsing actual import statements is both
    # more rigorous and immune to that self-reference problem.)
    import glob
    forbidden_modules = {"torch", "spanet", "h5py", "pytorch_lightning"}
    findings = {}
    for path in sorted(glob.glob(f"{OUT_DIR}/scripts/*.py")):
        with open(path) as f:
            tree = ast.parse(f.read(), filename=path)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        findings[path] = sorted(imported & forbidden_modules)
    results["no_new_inference_ast_import_check"] = findings
    any_hits = any(v for v in findings.values())
    print(f"AST-based forbidden-import check over this package's own scripts: any hits = {any_hits}")
    for path, hits in findings.items():
        print(f"  {path}: imports={hits if hits else 'none forbidden'}")
    results["no_new_inference_confirmed"] = not any_hits
    results["note_on_prior_naive_grep_false_positive"] = (
        "An earlier naive substring grep over this same directory flagged this "
        "script itself, because it literally contains the forbidden-token "
        "strings as data (the token list used to search for them), not as "
        "executable imports. Replaced with this AST-based import check, which "
        "only looks at actual `import`/`from ... import` statements and is not "
        "subject to that self-reference false positive."
    )

    with open(f"{OUT_DIR}/work/adversarial_self_check.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_DIR}/work/adversarial_self_check.json")


if __name__ == "__main__":
    main()
