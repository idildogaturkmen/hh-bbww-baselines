# CPU-vs-GPU Numerics Audit: JP-JEPA Mini `jet_embedding`

Status: **EXECUTED** (bounded, on already-fixed jets and already-produced
GPU artifacts -- no new JP-JEPA production, no SPA-Net training)
Date: 2026-09-09
Verdict: see `STATUS.md` / `DIAGNOSTIC_FINDINGS_SO_FAR.md`
(`JPJEPA_NUMERICS_ACCEPTABLE_WITH_DOCUMENTED_TOLERANCE`)

## 0. What this resumes and why

The 2026-08-25 dual ParT+JP-JEPA EAF canary
(`track_b_part_jpjepa_dual_embedding_production_20260825_v2/canary_eaf/dual_canary_report.json`)
found that ParT's CPU-vs-GPU `jet_embedding` agreement passed its own
bound (`part_full`, `pass: true`) while JP-JEPA Mini's did not
(`jpjepa_mini`, `pass: false`, `max_abs_diff=0.0597` vs a
`0.005` bound). This package's own `EXACT2M_PRODUCTION_PLAN.md`
inherited that as an open item. This task's brief states that a
subsequent (prior, not captured in this checkpoint package's own docs)
investigation already ruled out batch size as the cause (CPU@512 ==
CPU@256; CPU@256 vs GPU@256 still shows the gap). This document does
**not** redo the feature-compatibility audit or the 1,050-jet Mini
canary (both already done, `FEATURE_COMPATIBILITY_MATRIX.md`,
`CANARY_RUNBOOK.md`) -- it is a narrowly-scoped follow-up on this one
open numerical question.

## 1. Environment reality check for this session (read before the rest)

This session runs on an LPC interactive node (`cmslpc342.fnal.gov`)
with **no CUDA hardware** (`nvidia-smi` not present), **no GPU-bearing
HTCondor slots visible from the LPC pool**
(`condor_status -constraint 'TotalGpus > 0'` returns empty), and **no
reachable SSH route to EAF** (`ssh eaf.fnal.gov` times out; the prior
session's GPU runs were executed interactively from an EAF Jupyter
terminal, `/home/iturkmen/dual_canary/...`, which is not scriptable
from here). **No new GPU forward pass was executed in this session.**

Rather than skip the numerical audit or fabricate a GPU run, this audit
does two things instead, both fully executed and verified in this
session:

1. **Independently re-derives** CPU-vs-GPU statistics directly from the
   raw per-jet HDF5 arrays of the prior session's real EAF GPU runs
   (two independent runs, `run1`/`run2`) and the real CPU tiny-test run
   -- not by re-reading `dual_canary_report.json`'s own summary numbers.
2. **Freshly executes** everything that genuinely does not require a
   GPU: reproducing the frozen CPU layer-by-layer capture bit-for-bit,
   a CPU-only batch-size-invariance check, and a static code audit of
   every setting the task asked about (`eval()`/`no_grad`, dtype,
   autocast, TF32, attention backend, determinism flags).

The one item this audit could **not** execute is a live GPU-side
layer-by-layer capture or a live TF32/SDP-backend ablation -- see
section 6.

## 2. Frozen identity (re-verified this session, not assumed)

| Item | Value | Re-verified this session? |
|---|---|---|
| `jetparticle-jepa` pinned commit | `c68509eead1866c2c86714147023f5e8312634c4` | already verified, `UPSTREAM_PROVENANCE.md` |
| `jpjepa_mini_pretrained.ckpt` SHA256 | `1484e704d9f872c17eed6ead713f9ac39c6866dd74d5ebb45c23c9af7b37e3e9` | **yes**, `sha256sum` rerun this session |
| `SHARD_MANIFEST.json` SHA256 | `35049c136a28dfaec377cb718a7d2e75abc5c7d931450fd8746606c453b32333` | **yes** |
| `extract_shard.py` SHA256 | `9b6df35f1abff830932f098f982bd4d50897e1698822b0a6864425875b04ee58` | **yes**, extracted fresh from the bundle tarball and hashed |
| `particle_transformer.py` (jpjepa_src) SHA256 | `80ea294feb1981e3aa47f566657a10bd7a3f80720b79bf49a3f2a941b64e559e` | **yes**, matches `UPSTREAM_PROVENANCE.md`'s upstream hash exactly (byte-identical vendored copy) |
| Shard / jets compared | `train_0000`, 12,089 real jets (full shard) for the final-embedding comparison; the same fixed 200 jets (by `native_hdf5_row_index`+`jet_slot`) as the existing `cpu_layers.npz` for the layer-by-layer / batch-invariance work | identity cross-checked via join keys and `jet_pt` equality across all files compared |

## 3. Runtime settings actually used (static code audit of the frozen, hash-verified `extract_shard.py` / `jpjepa_src/particle_transformer.py`)

| Setting | Finding |
|---|---|
| `model.eval()` | present (`extract_shard.py:319`) |
| `torch.no_grad()` | present (`extract_shard.py:328`, and in the diagnostics' `layer_hook_localization.py`) |
| dtype | float32 throughout; no `.half()`/`.double()`/autocast anywhere |
| `torch.autocast` | **not used** anywhere in the extraction path |
| TF32 (`torch.backends.cuda.matmul.allow_tf32`, `torch.set_float32_matmul_precision`) | **never set explicitly** -- PyTorch's own default applies (matmul TF32 has been **disabled by default** since PyTorch 1.12; CPU is unaffected by TF32 regardless of any setting) |
| Attention backend | `nn.MultiheadAttention` is called as `self.attn(x, x, x, key_padding_mask=..., attn_mask=...)` with **`need_weights` left at its default value of `True`**, in *every* `Block.forward` call, in *both* the JP-JEPA source (`particle_transformer.py:429`) and the vendored ParT source (`weaver_core_part/ParticleTransformer.py:898,904`) -- because `attn_matrix` is captured and returned as part of `all_outputs`. Per PyTorch's own documented fast-path conditions, the fused (flash/memory-efficient) SDPA kernel is only selected when `need_weights=False`; with it `True`, PyTorch falls back to the explicit reference math path (batched matmul + softmax) on **both CPU and GPU**. This rules out "GPU picked a different fused attention kernel than CPU" as an explanation, for both models symmetrically. |
| `torch.use_deterministic_algorithms(...)` | **never called** -- PyTorch defaults apply |

**Conclusion of this section**: nothing in the frozen extraction code
enables autocast, TF32, or a fused attention kernel. Whatever explains
the CPU/GPU gap is *not* an explicit precision/kernel choice made by
this project's own code -- it is either PyTorch/cuBLAS's own default
non-associative floating-point behavior on GPU vs CPU, or something
checkpoint-specific (section 5).

## 4. Final `jet_embedding` CPU-vs-GPU statistics (independently recomputed)

Recomputed directly from `EMBEDDINGS/jpjepa_mini` in
`canary_eaf/cpu_reference/train_0000_cpu_dual.h5` (CPU, bs=512) vs
`canary_eaf/run1/train_0000.h5` (GPU, EAF A100 80GB PCIe MIG 4g.40gb,
torch 2.8.0+cu128, bs=256), all 12,089 real jets of `train_0000`,
identity cross-checked (join key + `jet_pt` equality, exact match for
every row). Reproducing script: `scripts/cpu_vs_gpu_embedding_stats.py`.

| Metric | Value | Matches `dual_canary_report.json`? |
|---|---|---|
| `max_abs_diff` | `0.0597229` | yes, exactly |
| `mean_abs_diff` | `0.00156997` | yes, exactly |
| `min_cosine_similarity` | `0.99998915` | yes, exactly |
| `mean_cosine_similarity` | `0.99999970` | (not previously reported) |
| `max_relative_L2` | `0.00469365` | yes, exactly |
| `mean_relative_L2` | `0.000812204` | (not previously reported) |
| `median_relative_L2` | `0.000737979` | (not previously reported) |

**GPU run-to-run determinism** (`run1` vs `run2`, same GPU, same
bs=256): `max_abs_diff = 0.0` -- **bitwise identical**. The GPU side is
internally, run-to-run deterministic for this fixed shape/batch/hardware
combination even without `torch.use_deterministic_algorithms(True)`.
The CPU-vs-GPU gap is therefore a genuine backend difference, not
run-to-run GPU noise.

**Outlier structure -- no small set of pathological jets dominates.**
`relative_L2` is smooth across all 12,089 jets: median `0.00074`, p99
`0.00203`, p99.9 `0.00304`, max `0.00469` (only ~6x the median at the
single worst jet). Correlation of `relative_L2` with `jet_pt` is `0.10`,
with constituent multiplicity `-0.09`, with embedding norm `-0.07` --
all weak. The 10 worst jets span a wide range of `pT` (33-450 GeV) and
multiplicity (3-26 constituents) with no visible pattern. This is
consistent with **generic float32 backend noise**, not a specific
pathological input regime (e.g. extreme multiplicity, or the
`d0`/`dz`-`tanh`-saturation tail already characterized in
`FEATURE_COMPATIBILITY_MATRIX.md` section 3).

**Per-dimension structure -- mild, not extreme, concentration.** The
median fraction of a jet's total squared CPU/GPU difference carried by
its single worst output dimension is `0.114` (i.e. no one dimension
dominates for a typical jet). Averaged over all 12,089 jets, dimension
`123` is the most affected (`mean_abs_diff=0.0054`, ~4x the per-dimension
median of `0.0013`); dims `60`, `78`, `45` are mildly elevated
(~2.5-3x median). No dimension is either exactly-zero-diff or
wildly-outlier.

## 5. ParT (`part_full`) as a same-architecture control -- and why it does NOT explain the gap away

`ParT_full.pt` and `jpjepa_mini_pretrained.ckpt` are, architecturally,
**the same hyperparameters**: `embed_dims=[128,512,128]`, `num_heads=8`,
`num_layers=8` (main blocks), `num_cls_layers=2` (class-attention
blocks) -- verified from `canonical_part_jet_encoder.py`'s own
docstring for ParT and `EXTRACTION_POINT_PROOF.md` section 3 for
JP-JEPA Mini. Both are **official, frozen, non-fine-tuned,
JetClass-pretrained checkpoints** applied out-of-domain to the same
AK4 jets (ParT: "official JetClass-supervised Particle Transformer
checkpoint... FROZEN: no gradient updates, no fine-tuning",
`docs/analysis_contracts/SPANET_PART_RESOURCE_AWARE_COMPARISON.md.save`;
JP-JEPA: self-supervised JEPA-pretrained on JetClass,
`UPSTREAM_PROVENANCE.md`). Both go through the identical
`need_weights=True`-forced attention path (section 3). Yet ParT's own
CPU-vs-GPU agreement (`eaf_gpu_validation_20260825/CPU_GPU_COMPARISON.txt`,
same shard, same identity scheme) is **systematically and substantially
tighter**:

| Metric | ParT (`part_full`) | JP-JEPA Mini | Ratio |
|---|---|---|---|
| `max_abs_diff` | `0.00340` | `0.0597` | **17.5x** |
| `mean_abs_diff` | `1.04e-05` | `0.00157` | **150x** |
| `min_cosine` (as `1 - cos`) | `4.4e-7` | `1.1e-5` | **~25x** |
| `max_relative_L2` | `0.00097` | `0.00469` | **4.8x** |
| `median_relative_L2` | `5.5e-5` | `0.00074` | **13x** |

**This rules out domain shift and attention-kernel selection as
differentiators** -- both models face identical exposure to both, yet
only one shows a large gap. The most defensible remaining explanation
is **checkpoint-specific numerical conditioning**: JP-JEPA Mini was
pretrained with a self-supervised JEPA (representation-prediction)
objective, ParT with supervised JetClass classification. Different
training objectives produce different weight-magnitude and
activation-scale statistics, which can make one network's forward pass
more sensitive to the same order of GPU/CPU non-associative
floating-point noise than another's, independent of architecture. This
is a plausible, code-and-data-grounded hypothesis, **not a proven
causal mechanism** -- see section 8 (open question / email draft).

## 6. Layer-by-layer: what could and could not be established

**Could not be established this session**: a literal GPU-side
layer-by-layer capture. `diagnostics/layer_hook_localization.py`
(prior session) is fully prepared for exactly this (`--device cuda
--out gpu_layers.npz`) but was never run against a GPU
(`diagnostics/gpu_bs512/` is an empty directory -- prepared, not
executed), and this session has no GPU to run it on (section 1). **This
is the one concrete gap this audit leaves open.**

**What this session did establish, CPU-only**: the frozen CPU capture
(`diagnostics/cpu_layers.npz`, 200 fixed jets, all 10 layer outputs --
8 main blocks + 2 class-attention blocks) was **reproduced bit-for-bit**
in a fresh CPU venv this session (`max_abs_diff = 0.0` vs the frozen
artifact; script: `scripts/cpu_batch_invariance_and_repro.py`),
confirming this session's code/checkpoint/data combination is faithful
to the one that produced it. Its per-layer activation-magnitude profile:

| Layer | Shape | `mean(|x|)` | `std` | `max(|x|)` |
|---|---|---|---|---|
| `layer_00` (main 1) | `(128,200,128)` | 0.44 | 0.65 | 7.4 |
| `layer_01` | | 0.66 | 0.84 | 7.7 |
| `layer_02` | | 0.85 | 1.08 | 8.4 |
| `layer_03` (peak, main) | | **1.15** | 1.46 | 9.8 |
| `layer_04` | | 1.11 | 1.37 | 9.8 |
| `layer_05` | | 1.00 | 1.26 | 8.1 |
| `layer_06` | | 0.69 | 0.90 | 6.1 |
| `layer_07` (last main) | | 0.27 | 0.38 | 4.5 |
| `layer_08` (cls-attn 1) | `(1,200,128)` | **0.79** (2.97x `layer_07`) | 1.12 | 8.5 |
| `layer_09` (cls-attn 2 = `jet_embedding`) | `(1,200,128)` | **1.50** (1.90x `layer_08`) | **2.68** | **24.9** |

Activation magnitude rises through the first four main blocks, decays
through the rest of the main stack, then **jumps sharply at both
class-attention blocks** -- and `layer_09` (the frozen extraction
point) has by far the largest dynamic range (`max=24.9`, vs `<10` for
every main block) and largest spread (`std=2.68`) of any layer captured.

This is architecturally consistent with the class-attention mechanism
being the most numerically sensitive stage: it is a softmax-weighted
aggregation over the **full 128-slot sequence**, of which AK4 jets
typically populate fewer than 20 real constituents (mean `17.8`,
`FEATURE_COMPATIBILITY_MATRIX.md` section 5) -- i.e. **over 80% padding**.
With so few real tokens competing for attention weight, a small
per-constituent logit perturbation can shift softmax weights among them
non-negligibly, and this is the *last* stage of the network, with no
subsequent block to average the effect back down. This is offered as
the leading **architectural hypothesis** for where CPU/GPU divergence
is most likely amplified -- it is inferred from CPU-only dynamic-range
structure, not measured as a direct CPU-vs-GPU per-layer diff, and
should be labeled as such.

## 7. Batch-size invariance -- freshly executed, extends the prior finding

Using the fixed 200 jets, the exact frozen `extract_shard.py`, and a
fresh CPU-only venv (`torch==2.8.0+cpu`) this session:

| Comparison | `max_abs_diff` | Result |
|---|---|---|
| bs=200 (single batch) vs the frozen `cpu_layers.npz` artifact | `0.0` | **bit-identical** (environment/code fidelity check) |
| bs=200 vs bs=64 (chunked) | `0.0` | **bit-identical** |
| bs=200 vs bs=1 (fully serial) | `1.335e-05` | **differs** (small) |

The bs=200-vs-bs=64 bit-identical result is consistent with, and adds a
freshly-executed data point to, this task's stated prior finding that
CPU@512 == CPU@256 (batch chunking among "normal-sized" batches does
not matter on CPU). The one nonzero effect found -- fully-serial
(bs=1) inference differing from batched inference by up to `1.34e-05`
-- is **~4,500x smaller** than the CPU-vs-GPU `max_abs_diff` (`0.0597`),
so it does not explain the CPU/GPU gap; if anything it is a useful,
freshly-measured upper bound on how large a pure batch-chunking effect
can possibly be for this architecture, reinforcing rather than
undermining "batch size is not the cause."

## 8. TF32 / deterministic-SDP-backend ablation: NOT executed, why, and what would close it

The task asks to "explicitly test whether disabling TF32 / selecting
deterministic SDP/attention backends materially changes the
discrepancy." **This was not executed** -- it requires a GPU, and none
was reachable this session (section 1). What this audit substitutes
(section 3's static code audit) is real evidence but not a substitute
for the actual ablation: it shows the frozen code never *explicitly
enables* TF32 or a fused attention kernel, and that PyTorch's own
*default* is TF32-off-for-matmul and reference-math-attention (because
`need_weights=True`) -- but it cannot rule out, e.g., cuDNN-side TF32
paths, a PyTorch-version-specific default change, or non-associative
cuBLAS GEMM algorithm selection being the dominant residual driver
without literally running the two one-line ablations on the GPU that
originally produced `run1`/`run2`:

```python
torch.backends.cuda.matmul.allow_tf32 = False   # (should already be default)
# ... vs ...
torch.backends.cuda.matmul.allow_tf32 = True
```

and comparing `jet_embedding` under each. **This remains the single
concrete follow-up action that would fully close this question** rather
than bound it. `diagnostics/layer_hook_localization.py` is ready to run
as-is on EAF (`--device cuda --out gpu_layers.npz`) by anyone with GPU
access; a one-line TF32 toggle added around its `model.forward(...)`
call would answer the TF32 question directly, at zero additional
engineering cost, in well under a minute of GPU time.

## 9. Is this large enough to be scientifically ambiguous for downstream SPA-Net use? (signal-vs-noise)

The relevant question is not "does this pass an arbitrary bound copied
from ParT's own tighter behavior" but whether CPU/GPU backend noise at
this magnitude can plausibly blur the physics content SPA-Net would
need to extract from this frozen, non-trainable 128-d input feature.

Per-jet **absolute** L2 diff between CPU and GPU (not relative): mean
`0.025`, median `0.023`, p99 `0.061`, max `0.152` (over all 12,089 jets
of `train_0000`). Compare against the embedding-space scales already
measured in this package and freshly computed this session:

| Reference scale | Value | Ratio to mean CPU/GPU noise (`0.025`) | Ratio to max CPU/GPU noise (`0.152`) |
|---|---|---|---|
| Smallest between-process mean-embedding separation (QCD vs global mean, 1,050-jet canary, `CANARY_RUNBOOK.md`) | `0.91` | noise is **36x** smaller | noise is **6x** smaller |
| Typical nearest-neighbor jet-to-jet distance (2,000 QCD jets, this session) | `5.85` (median) | noise is **230x** smaller | noise is **38x** smaller |
| Closest 1st-percentile nearest-neighbor jet pair (same, hardest case) | `3.33` | noise is **131x** smaller | noise is **22x** smaller |

**Even the single worst CPU/GPU-divergent jet out of 12,089 sits ~22x
below the closest 1%-tile nearest-neighbor jet-to-jet distance within
the same process**, and the typical (mean) CPU/GPU noise is two to three
orders of magnitude below every physics scale checked. CPU/GPU backend
noise at this magnitude cannot plausibly make two jets, or two
processes, indistinguishable in this frozen representation.

## 10. Bottom line

- **What is settled**: the discrepancy is not caused by batch size
  (freshly re-confirmed, CPU-only, this session, extending the prior
  finding), not caused by GPU run-to-run nondeterminism (bitwise
  identical across two independent GPU runs), not caused by an explicit
  autocast/TF32/fused-attention-kernel choice in this project's own
  code (none exists), and not caused by a small set of pathological
  jets or a domain-shift effect unique to JP-JEPA (ParT faces the same
  domain shift with a ~15-150x tighter CPU/GPU agreement).
- **What is characterized but not proven**: the residual gap is most
  likely ordinary GPU-vs-CPU float32 non-associativity, most likely
  amplified by the class-attention blocks' full-sequence,
  heavily-padded softmax aggregation (CPU-only structural evidence,
  section 6), and most likely differs from ParT's much tighter behavior
  because of checkpoint-specific weight/activation conditioning tied to
  JP-JEPA's self-supervised pretraining objective (section 5) -- neither
  of these is confirmed by a direct GPU-side measurement.
- **What remains genuinely open**: a live GPU-side layer-by-layer trace
  and a live TF32-ablation, both blocked purely by this session's lack
  of GPU access, not by any conceptual gap in the audit design. Both are
  fully prepared to run in minutes by anyone with EAF GPU access.
- **Downstream impact**: quantitatively bounded and small relative to
  every physics scale checked (section 9).

See `STATUS.md` and `DIAGNOSTIC_FINDINGS_SO_FAR.md` for the formal
verdict, and `EMAIL_DRAFT_NOT_SENT.md` for the drafted (unsent) question
to upstream.
