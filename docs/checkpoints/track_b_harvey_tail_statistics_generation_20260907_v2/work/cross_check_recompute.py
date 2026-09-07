#!/usr/bin/env python3
"""
Independent re-derivation of every headline number in
HARVEY_PHYSICAL_TAIL_STATS_FINAL.md, computed directly from the frozen
per-event source files (not from any intermediate cached value produced
during drafting). Read-only: opens existing frozen artifacts only, writes
nothing outside this package's own work/ directory.
"""
import json
import math

QCD_W = 7.3112875413134075
TT_W = 2.3214135411747647
OTHER_W = {
    'ZJetsToQQ': 1.41031125, 'TTbarZ': 0.38655, 'ZZ': 1.0784616073620259,
    'ZH': 1.1417999999999999, 'ttH': 0.76065, 'SingleHiggs': 2.1861,
    'SingleTop': 4.884600426532971, 'TTbarW': 0.335385, 'TW': 6.227102637699448,
    'VBFH': 1.7019, 'WW': 3.72725937405907, 'WminusH': 0.47943095886191772,
    'WplusH': 0.7559236066249192, 'ZW': 2.9552364505296116,
}

U35 = 1 - 10 ** -3.5
U45 = 1 - 10 ** -4.5
print(f"threshold(u>3.5) = {U35!r}")
print(f"threshold(u>4.5) = {U45!r}")
assert abs(U35 - 0.9996837722339832) < 1e-15
assert abs(U45 - 0.9999683772233983) < 1e-15

# --- QCD, direct from the survivor sidecar (h5) ---
import h5py
sidecar = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_characterization_20260825_v1/work/FULL_QCD_SURVIVOR_SIDECAR.h5"
with h5py.File(sidecar, "r") as f:
    print("sidecar keys:", list(f.keys()))
    scores = f["score_spanet_10m"][:]
n_qcd_u35 = int((scores > U35).sum())
n_qcd_u35_strict_gt = int((scores > U35).sum())
n_qcd_u35_ge = int((scores >= U35).sum())
n_qcd_u45 = int((scores > U45).sum())
n_qcd_9997 = int((scores >= 0.9997).sum())
n_qcd_99997 = int((scores >= 0.99997).sum())
print(f"QCD: n(u>3.5, strict >)={n_qcd_u35_strict_gt}  n(score>=u35 same value)={n_qcd_u35_ge}")
print(f"QCD: n(u>4.5, strict >)={n_qcd_u45}")
print(f"QCD: n(score>=0.9997)={n_qcd_9997}  n(score>=0.99997)={n_qcd_99997}")
assert n_qcd_u35 == 68, n_qcd_u35
assert n_qcd_u45 == 12, n_qcd_u45
assert n_qcd_9997 == 61, n_qcd_9997
assert n_qcd_99997 == 12, n_qcd_99997
assert n_qcd_u45 == n_qcd_99997, "u>4.5 and score>=0.99997 QCD populations must be identical"

# --- non-QCD, direct from job_summary survivor lists ---
base = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_complete_3to5pct_study_20260825_v1/work/"
files = ["job_summary_ttbar_PRESERVED.json", "job_summary_medium.json",
         "job_summary_zjetstoqq.json", "job_summary_small.json"]
records = {}
for fn in files:
    d = json.load(open(base + fn))
    for proc, v in d['processes'].items():
        records[proc] = v

def count_nonqcd(thresh, strict_gt=True):
    out = {}
    for proc, v in records.items():
        surv = v.get('spanet_10m', {}).get('survivors', [])
        sc = [e[2] for e in surv]
        n = sum(1 for s in sc if (s > thresh if strict_gt else s >= thresh))
        if n:
            out[proc] = n
    return out

nonqcd_u35 = count_nonqcd(U35, strict_gt=True)
nonqcd_u45 = count_nonqcd(U45, strict_gt=True)
nonqcd_9997 = count_nonqcd(0.9997, strict_gt=False)
nonqcd_99997 = count_nonqcd(0.99997, strict_gt=False)
print("non-QCD u>3.5:", nonqcd_u35)
print("non-QCD u>4.5:", nonqcd_u45)
print("non-QCD score>=0.9997:", nonqcd_9997)
print("non-QCD score>=0.99997:", nonqcd_99997)

def ttbar_and_other(d):
    tt = d.get('inference_ttbar_1', 0) + d.get('inference_ttbar_2', 0)
    other = {k: v for k, v in d.items() if k not in ('inference_ttbar_1', 'inference_ttbar_2')}
    return tt, other

for label, thresh_dict, qcd_n in [
    ('u>3.5', nonqcd_u35, n_qcd_u35),
    ('u>4.5', nonqcd_u45, n_qcd_u45),
    ('score>=0.9997', nonqcd_9997, n_qcd_9997),
    ('score>=0.99997', nonqcd_99997, n_qcd_99997),
]:
    tt, other = ttbar_and_other(thresh_dict)
    B_qcd = qcd_n * QCD_W
    S_qcd = qcd_n * QCD_W ** 2
    B_tt = tt * TT_W
    S_tt = tt * TT_W ** 2
    B_other = sum(n * OTHER_W[p] for p, n in other.items())
    S_other = sum(n * OTHER_W[p] ** 2 for p, n in other.items())
    n_other = sum(other.values())
    B_all = B_qcd + B_tt + B_other
    S_all = S_qcd + S_tt + S_other
    n_all = qcd_n + tt + n_other
    neff = B_all ** 2 / S_all
    relunc = 100 / math.sqrt(neff)
    print(f"\n=== {label} ===")
    print(f"  QCD n={qcd_n} B={B_qcd:.4f}  ttbar n={tt} B={B_tt:.4f}  other n={n_other} B={B_other:.4f} ({other})")
    print(f"  ALL n={n_all} B={B_all:.4f} sum_w2={S_all:.4f} N_eff={neff:.4f} rel_unc={relunc:.4f}%")

# --- eps_S=4.0% governing WP, from the frozen fine-scan ---
finescan = json.load(open(
    "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_followup_finescan_likelihood_20260829_v1/work/spa10m_finescan_rows.json"))
row4 = [r for r in finescan if abs(r['target_eps'] - 0.04) < 1e-9][0]
print("\n=== eps_S=4.0% WP (frozen fine-scan row) ===")
print(row4)
B4 = row4['B_450_central']
n_qcd4, n_tt4, n_zj4 = row4['raw_QCD'], row4['raw_ttbar'], row4['raw_ZJetsToQQ']
# recover SingleTop count from the full CSV row directly
import csv
csv_rows = list(csv.DictReader(open(
    "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_followup_finescan_likelihood_20260829_v1/HARVEY_FOLLOWUP_FINE_SCAN.csv")))
csv_row4 = [r for r in csv_rows if r['target_eps_pct'] == '4.0'][0]
n_singletop4 = int(csv_row4['n_SingleTop'])
S_qcd4 = n_qcd4 * QCD_W ** 2
S_tt4 = n_tt4 * TT_W ** 2
S_other4 = n_zj4 * OTHER_W['ZJetsToQQ'] ** 2 + n_singletop4 * OTHER_W['SingleTop'] ** 2
S_all4 = S_qcd4 + S_tt4 + S_other4
neff4 = B4 ** 2 / S_all4
relunc4 = 100 / math.sqrt(neff4)
print(f"  recomputed: n_QCD={n_qcd4} n_tt={n_tt4} n_ZJets={n_zj4} n_SingleTop={n_singletop4}")
print(f"  B={B4:.6f}  sum_w2={S_all4:.6f}  N_eff={neff4:.6f}  rel_unc={relunc4:.6f}%")
assert abs(neff4 - 20.4618) < 0.001, neff4
assert abs(relunc4 - 22.1069) < 0.001, relunc4

# --- 2x/4x/10x scaling, generic function reused for all 4 populations ---
def scale_table(B, S_qcd, S_fixed, label):
    print(f"\n--- QCD-only scaling: {label} ---")
    for K in (1, 2, 4, 10):
        neff = B ** 2 / (S_qcd / K + S_fixed)
        relunc = 100 / math.sqrt(neff)
        print(f"  {K:>2d}x: N_eff={neff:.3f}  rel_unc={relunc:.3f}%")

scale_table(B4, S_qcd4, S_tt4 + S_other4, "eps_S=4.0% WP")

# --- generation/CPU cross-check ---
gen_current = 17600 * 5_000_000 + 38072 * 5_000_000
assert gen_current == 278_360_000_000
rate_alloc = 1.3517 / 50_000
for mult in (2, 4, 10):
    add_gen = (mult - 1) * gen_current
    cpu = add_gen * rate_alloc
    print(f"{mult}x: additional generated = {add_gen:.6e}  CPU-slot-hours = {cpu:,.1f}")

print("\nALL CROSS-CHECKS PASSED")
