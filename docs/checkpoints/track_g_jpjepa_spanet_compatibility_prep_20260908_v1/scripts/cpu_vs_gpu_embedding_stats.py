"""
Track G CPU/GPU numerics audit -- independent recomputation of CPU-vs-GPU
jet_embedding statistics from the raw per-jet HDF5 arrays (not from
dual_canary_report.json's own aggregate numbers), plus outlier/structure
analysis and a signal-vs-noise downstream-impact comparison.

See ../CPU_GPU_NUMERICS_AUDIT.md sections 1 and 4-5.

No GPU or torch required -- only h5py/numpy, reading already-produced
HDF5 embedding files from the 2026-08-25 EAF dual canary.

Usage:
    python3 cpu_vs_gpu_embedding_stats.py \\
        --cpu-h5  .../canary_eaf/cpu_reference/train_0000_cpu_dual.h5 \\
        --gpu-h5  .../canary_eaf/run1/train_0000.h5 \\
        --gpu-h5-2 .../canary_eaf/run2/train_0000.h5   # optional, for determinism check
"""
import argparse

import h5py
import numpy as np


def load(path, key="jpjepa_mini"):
    with h5py.File(path, "r") as f:
        emb = f[f"EMBEDDINGS/{key}"][:]
        row = f["JOIN_KEY/native_hdf5_row_index"][:]
        slot = f["JOIN_KEY/jet_slot"][:]
        pt = f["SANITY/jet_pt"][:]
        nc = f["SANITY/n_constituents_used"][:]
    return np.array([f"{r}_{s}" for r, s in zip(row, slot)]), emb, pt, nc


def compare(a, b, label):
    diff = a - b
    abs_diff = np.abs(diff)
    row_l2_diff = np.linalg.norm(diff, axis=1)
    row_l2_a = np.linalg.norm(a, axis=1)
    row_l2_b = np.linalg.norm(b, axis=1)
    rel_l2 = row_l2_diff / np.maximum(row_l2_a, 1e-12)
    cos = (a * b).sum(axis=1) / (row_l2_a * row_l2_b + 1e-12)
    print(f"--- {label} ---")
    print(f"  max_abs_diff={abs_diff.max():.6g}  mean_abs_diff={abs_diff.mean():.6g}")
    print(f"  min_cosine={cos.min():.8f}  mean_cosine={cos.mean():.8f}")
    print(f"  max_rel_l2={rel_l2.max():.6g}  mean_rel_l2={rel_l2.mean():.6g}  median_rel_l2={np.median(rel_l2):.6g}")
    return dict(abs_diff=abs_diff, row_l2_diff=row_l2_diff, row_l2_a=row_l2_a, rel_l2=rel_l2, cos=cos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-h5", required=True)
    ap.add_argument("--gpu-h5", required=True)
    ap.add_argument("--gpu-h5-2", default=None, help="optional second GPU run, for run-to-run determinism check")
    args = ap.parse_args()

    k_cpu, e_cpu, pt_cpu, nc_cpu = load(args.cpu_h5)
    k_gpu, e_gpu, pt_gpu, nc_gpu = load(args.gpu_h5)
    assert (k_cpu == k_gpu).all(), "join key order mismatch between CPU and GPU files"
    assert (pt_cpu == pt_gpu).all(), "jet_pt mismatch -- identity broken"
    print(f"n jets: {len(k_cpu)} (identity cross-checked via native_hdf5_row_index+jet_slot+jet_pt)\n")

    r = compare(e_cpu, e_gpu, "CPU vs GPU")

    if args.gpu_h5_2:
        k_g2, e_g2, pt_g2, _ = load(args.gpu_h5_2)
        assert (k_cpu == k_g2).all()
        compare(e_gpu, e_g2, "GPU run1 vs run2 (determinism check)")

    idx_sorted = np.argsort(-r["rel_l2"])
    print("\nTop-10 worst jets by rel_l2:")
    for i in idx_sorted[:10]:
        print(f"  key={k_cpu[i]} jet_pt={pt_cpu[i]:.2f} n_const={nc_cpu[i]} "
              f"rel_l2={r['rel_l2'][i]:.5f} abs_diff_max={r['abs_diff'][i].max():.5f}")

    print(f"\nCorrelation rel_l2 vs jet_pt:          {np.corrcoef(r['rel_l2'], pt_cpu)[0,1]:.4f}")
    print(f"Correlation rel_l2 vs n_constituents:  {np.corrcoef(r['rel_l2'], nc_cpu.astype(float))[0,1]:.4f}")
    print(f"Correlation rel_l2 vs embedding norm:  {np.corrcoef(r['rel_l2'], r['row_l2_a'])[0,1]:.4f}")

    # per-dimension concentration
    frac_top1 = (r["abs_diff"].max(axis=1) ** 2) / (np.sum(r["abs_diff"] ** 2, axis=1) + 1e-20)
    print(f"\nMedian fraction of squared-diff-energy in single worst dim: {np.median(frac_top1):.4f}")
    per_dim_mean_abs = r["abs_diff"].mean(axis=0)
    top_dims = np.argsort(-per_dim_mean_abs)[:5]
    print(f"Top-5 dims by mean abs diff: {list(top_dims)} -> {per_dim_mean_abs[top_dims].round(5).tolist()}")
    print(f"Per-dim mean_abs_diff distribution: min={per_dim_mean_abs.min():.3g} "
          f"median={np.median(per_dim_mean_abs):.3g} max={per_dim_mean_abs.max():.3g}")

    # signal-vs-noise
    mean_emb = e_cpu.mean(axis=0)
    intra_spread = np.linalg.norm(e_cpu - mean_emb, axis=1)
    print(f"\nCPU/GPU absolute L2 diff per jet: mean={r['row_l2_diff'].mean():.4f} "
          f"median={np.median(r['row_l2_diff']):.4f} max={r['row_l2_diff'].max():.4f}")
    print(f"Intra-file per-jet distance to file-mean embedding: mean={intra_spread.mean():.4f} "
          f"median={np.median(intra_spread):.4f}")


if __name__ == "__main__":
    main()
