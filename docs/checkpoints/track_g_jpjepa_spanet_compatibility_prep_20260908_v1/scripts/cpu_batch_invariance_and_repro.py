"""
Track G CPU/GPU numerics audit -- CPU-only reproducibility + batch-size
invariance check (see ../CPU_GPU_NUMERICS_AUDIT.md section 2).

No GPU is used or required. Reproduces the frozen
track_b_part_jpjepa_dual_embedding_production_20260825_v2/diagnostics/cpu_layers.npz
jet_embedding bit-for-bit using the exact frozen, sha256-verified
extract_shard.py, checkpoint, and manifest, then tests batch-size
invariance (bs=200 single batch vs bs=64 chunked vs bs=1 fully serial)
on the same 200 fixed jets.

Prerequisites:
  - A CPU-only torch environment (torch==2.8.0+cpu, numpy, awkward,
    uproot, fsspec-xrootd all suffice -- see CANARY_RUNBOOK.md section 1
    for how to build one from scratch if needed).
  - A valid X.509 grid proxy with IHEP EOS read access (for
    build_shard_tensors' XRootD read of the source ROOT file).
  - The v2 production bundle tarball extracted somewhere local:
      /eos/uscms/store/user/iturkmen/hh4b_delphes/part_jpjepa_dual_production_bundle_20260825_v2/part_jpjepa_dual_production_bundle_20260825_v2.tar.gz
    (sha256-verified against SHA256SUMS in that same directory before
    use; this script does not re-verify it -- do that once, manually,
    before trusting a fresh extraction).

Usage:
    tar xzf part_jpjepa_dual_production_bundle_20260825_v2.tar.gz -C /path/to/extract
    python3 cpu_batch_invariance_and_repro.py \\
        --bundle-dir /path/to/extract/bundle \\
        --manifest /path/to/extract/manifest/SHARD_MANIFEST.json \\
        --checkpoint /eos/uscms/store/user/iturkmen/hh4b_delphes/part_jpjepa_dual_production_bundle_20260825_v2/jpjepa_mini_pretrained.ckpt \\
        --cpu-layers-npz /eos/uscms/store/user/iturkmen/hh4b_delphes/track_b_part_jpjepa_dual_embedding_production_20260825_v2/diagnostics/cpu_layers.npz \\
        --out out/cpu_batch_invariance.npz
"""
import argparse
import json
import sys

import numpy as np
import torch

N_SUBSET = 200
SHARD_ID = "train_0000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True, help="extracted bundle/ directory (contains code/, code/jpjepa_src/)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cpu-layers-npz", required=True, help="frozen reference artifact to reproduce")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bundle_code = args.bundle_dir.rstrip("/") + "/code"
    jpjepa_src = bundle_code + "/jpjepa_src"
    sys.path.insert(0, bundle_code)
    from extract_shard import build_shard_tensors, build_jpjepa_reordered_features, load_jpjepa_mini_encoder  # noqa: E402

    manifest = json.load(open(args.manifest))
    shard = next(s for s in manifest["shards"] if s["shard_id"] == SHARD_ID)
    built = build_shard_tensors(shard)
    feat_jp = build_jpjepa_reordered_features(built["feat_natural"])[:N_SUBSET]
    lv = built["lv"][:N_SUBSET]
    mask = built["mask"][:N_SUBSET]
    identity_row = np.array([r["native_hdf5_row_index"] for r in built["rows"][:N_SUBSET]])
    identity_slot = np.array([r["jet_slot"] for r in built["rows"][:N_SUBSET]])

    model = load_jpjepa_mini_encoder(args.checkpoint, jpjepa_src, device="cpu")

    def run_batched(bs):
        """Mirrors extract_shard.py's run_jpjepa chunking exactly."""
        n = feat_jp.shape[0]
        out = torch.zeros(n, 128)
        with torch.no_grad():
            for lo in range(0, n, bs):
                hi = min(lo + bs, n)
                f = torch.from_numpy(feat_jp[lo:hi])
                v = torch.from_numpy(lv[lo:hi])
                m = torch.from_numpy(mask[lo:hi])
                _, outs = model.forward(f, v, m)
                out[lo:hi] = outs[-1][0][0]
        return out.numpy()

    ref = np.load(args.cpu_layers_npz)
    assert (ref["identity_native_row"] == identity_row).all()
    assert (ref["identity_jet_slot"] == identity_slot).all()

    emb_bs200 = run_batched(200)
    ref_emb = ref["jet_embedding"]
    repro_max_abs = float(np.abs(emb_bs200 - ref_emb).max())
    print(f"[reproduce {args.cpu_layers_npz} @ bs=200] max_abs_diff = {repro_max_abs:.10g} "
          f"({'BIT-IDENTICAL' if repro_max_abs == 0.0 else 'DIFFERS'})")

    emb_bs64 = run_batched(64)
    emb_bs1 = run_batched(1)
    d_64 = float(np.abs(emb_bs200 - emb_bs64).max())
    d_1 = float(np.abs(emb_bs200 - emb_bs1).max())
    print(f"[CPU bs=200 vs bs=64] max_abs_diff = {d_64:.10g} ({'BIT-IDENTICAL' if d_64 == 0.0 else 'DIFFERS'})")
    print(f"[CPU bs=200 vs bs=1 ] max_abs_diff = {d_1:.10g} ({'BIT-IDENTICAL' if d_1 == 0.0 else 'DIFFERS'})")

    np.savez(args.out, emb_bs200=emb_bs200, emb_bs64=emb_bs64, emb_bs1=emb_bs1, ref_emb=ref_emb,
             identity_row=identity_row, identity_slot=identity_slot,
             repro_max_abs=repro_max_abs, d_bs200_vs_bs64=d_64, d_bs200_vs_bs1=d_1)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
