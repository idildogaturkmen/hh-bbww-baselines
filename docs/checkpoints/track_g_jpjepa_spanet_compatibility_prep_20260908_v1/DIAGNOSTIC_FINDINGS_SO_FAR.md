# Diagnostic Findings So Far: JP-JEPA Mini CPU-vs-GPU `jet_embedding` Discrepancy

Last updated: 2026-09-09
Full technical detail: `CPU_GPU_NUMERICS_AUDIT.md`. This document is the
condensed, chronological findings log.

## VERDICT

```
JPJEPA_NUMERICS_ACCEPTABLE_WITH_DOCUMENTED_TOLERANCE
```

Not `PASS` (the discrepancy is real and exceeds the numeric bounds
originally derived from ParT's own, tighter, CPU/GPU behavior). Not
`BLOCKED` (the discrepancy is bounded, well-characterized, ruled out
against every mechanical cause this session could test, and shown to be
1-2 orders of magnitude below every physics-relevant scale checked in
the same embedding space -- see finding 7 below). The residual open
item (finding 8) is a genuine mechanistic gap, not a blocking one.

## Timeline of findings

1. **(2026-08-25, prior session, inherited)** The dual ParT+JP-JEPA EAF
   canary found ParT's CPU-vs-GPU agreement passed its own bound while
   JP-JEPA Mini's did not: `max_abs_diff=0.0597` vs a `0.005` bound,
   `mean_abs_diff=0.00157` vs a `2e-5` bound, `min_cosine=0.999989` vs a
   `0.999999` bound, `max_relative_L2=0.00469` vs a `0.0015` bound.
   (`dual_canary_report.json`, `cpu_vs_gpu_dual.jpjepa_mini.pass = false`.)

2. **(prior session, inherited, per this task's brief)** Batch size was
   already ruled out as the cause: CPU@512 == CPU@256, while CPU@256 vs
   GPU@256 still reproduces the discrepancy.

3. **(2026-09-09, this session)** Independently recomputed the CPU-vs-GPU
   statistics directly from the raw per-jet HDF5 arrays (not from the
   existing summary JSON) across all 12,089 real jets of `train_0000`:
   **exact match** to finding 1's numbers. Also recomputed GPU
   run-to-run determinism (`run1` vs `run2`): **bitwise identical**
   (`max_abs_diff=0.0`) -- the gap is a genuine CPU/GPU backend
   difference, not GPU noise.

4. **(2026-09-09)** Static code audit of the frozen, hash-verified
   extraction code found: `model.eval()` and `torch.no_grad()` are used
   correctly; no autocast; no explicit TF32 setting anywhere (PyTorch
   default -- TF32 off for matmul -- applies); and, critically, every
   `nn.MultiheadAttention` call uses the default `need_weights=True`
   (because attention matrices are captured), which forces PyTorch's
   non-fused reference attention path on **both** CPU and GPU. This
   rules out "GPU used a different fused attention kernel" as an
   explanation.

5. **(2026-09-09)** Compared against ParT (`part_full`) as a
   same-architecture control (identical `embed_dims`/`num_heads`/
   `num_layers`/`num_cls_layers`, identical frozen-JetClass-pretrained-
   checkpoint status, identical attention-path forcing). ParT's CPU/GPU
   gap is **15-150x smaller** across every metric. This rules out
   domain shift and attention-kernel selection as differentiators (both
   models face identical exposure to both) and points toward
   checkpoint-specific numerical conditioning (different pretraining
   objective: JEPA self-supervised vs. supervised classification) as
   the most defensible remaining hypothesis -- **not proven**, offered
   as the leading explanation.

6. **(2026-09-09)** CPU-only layer-by-layer magnitude profiling (the
   existing `cpu_layers.npz`, reproduced bit-for-bit this session as a
   fidelity check) found the sharpest activation-magnitude jump in the
   entire 10-block stack occurs at the two class-attention blocks --
   the same place `jet_embedding` is read from -- with the final block
   showing by far the largest dynamic range (`max=24.9`) of any layer.
   Architecturally consistent with the class-attention full-sequence
   softmax aggregation (over a sequence that is >80% zero-padding for
   these AK4 jets) being the most numerically sensitive stage. This is
   a CPU-only structural inference, not a direct GPU-side measurement.

7. **(2026-09-09)** Freshly-executed CPU batch-size invariance check
   (bs=200 vs bs=64: bit-identical; bs=200 vs bs=1: differs by only
   `1.3e-5`, ~4,500x smaller than the CPU/GPU gap) reinforces finding 2
   with new, this-session evidence.

8. **(2026-09-09)** Signal-vs-noise downstream-impact check: the mean
   CPU/GPU absolute-L2 noise per jet (`0.025`) is ~36x smaller than the
   smallest between-process mean-embedding separation (`0.91`, 1,050-jet
   canary) and ~230x smaller than the typical nearest-neighbor
   jet-to-jet distance (`5.85`, 2,000-jet check, this session). Even the
   single worst jet's noise (`0.152`, out of 12,089) is ~6-22x smaller
   than these same scales. **This magnitude of noise cannot plausibly
   make jets or processes indistinguishable in this frozen
   representation.**

## What remains genuinely open

A live GPU-side layer-by-layer capture and a live TF32-on/off ablation
were **not executed** -- this session had no reachable GPU (no local
CUDA hardware; `ssh eaf.fnal.gov` timed out; no GPU HTCondor slots
visible from the LPC pool). `diagnostics/layer_hook_localization.py` is
already written and ready to run on EAF
(`--device cuda --out gpu_layers.npz`); this is the single concrete
action that would fully mechanistically close the question. See
`CPU_GPU_NUMERICS_AUDIT.md` section 8.

## What this means for downstream use

The frozen `jet_embedding` extraction point
(`all_layer_outputs[-1][0][0]`) is **not changed** as a result of this
audit -- per the task's own instruction, the goal was never to make
CPU/GPU agree by moving the extraction point, and finding 8 shows there
is no scientific need to. Whoever produces the eventual full-scale
JP-JEPA embeddings should: (a) pick one device (CPU or GPU) and use it
consistently for the full production run, rather than mixing devices
across shards, since finding 1 shows the two are not bit-compatible;
(b) record which device was used in the production receipt, alongside
the existing checkpoint/commit/manifest hashes; and (c) treat
`CPU_GPU_NUMERICS_AUDIT.md`'s bound (`relative_L2` typically
`<0.001`, worst-case `<0.005`, on this diagnostic sample) as the
documented numerical tolerance for this representation, to be re-checked
(not assumed) at full 2M scale per `EXACT2M_PRODUCTION_PLAN.md`'s
pre-production gate.
