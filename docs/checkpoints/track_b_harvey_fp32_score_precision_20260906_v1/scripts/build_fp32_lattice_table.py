"""Task 2: exact FP32 lattice near score=1, and its u = -log10(1-score) image.

Pure arithmetic, no data/model access. For a normalized float32 value
immediately below 1.0, the representable values are 1 - k * 2^-24 for
integer k = 1, 2, 3, ... (exponent of numbers in [0.5, 1) is -1, and the
float32 mantissa has 23 explicit bits, so the ULP there is 2^(-1-23) =
2^-24). This holds only while score_k stays in [0.5, 1); it is valid for
all k printed here (k up to a few thousand).
"""
import csv
import struct
import numpy as np

SPACING = 2.0 ** -24

def f32_hex(x):
    return struct.pack(">f", np.float32(x)).hex()

def main():
    rows = []
    for k in range(1, 33):
        one_minus_score = k * SPACING
        score_f64 = 1.0 - one_minus_score
        score_f32 = np.float32(1.0) - np.float32(k) * np.float32(SPACING)
        u = -np.log10(one_minus_score)
        rows.append({
            "k": k,
            "one_minus_score_exact": f"{one_minus_score:.17g}",
            "score_f64": f"{score_f64:.17g}",
            "score_f32": f"{float(score_f32):.9g}",
            "score_f32_hex_be": f32_hex(score_f32),
            "u_k": f"{float(u):.6f}",
        })

    print(f"{'k':>3} {'1-score = k*2^-24':>22} {'score (f64 view)':>20} {'u_k = -log10(k*2^-24)':>24}")
    for r in rows:
        print(f"{r['k']:>3} {r['one_minus_score_exact']:>22} {r['score_f64']:>20} {r['u_k']:>24}")

    print()
    print("Targets of interest:")
    for target in (6.6, 6.9):
        best = min(rows, key=lambda r: abs(float(r["u_k"]) - target))
        print(f"  u ~ {target}: nearest lattice point is k={best['k']}, u_k={best['u_k']}, "
              f"score_f32={best['score_f32']}")

    out_csv = __file__.replace("scripts/build_fp32_lattice_table.py", "FP32_LATTICE_TABLE.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")

if __name__ == "__main__":
    main()
