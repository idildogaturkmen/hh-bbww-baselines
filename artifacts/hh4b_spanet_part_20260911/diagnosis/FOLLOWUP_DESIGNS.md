# Controlled follow-up trainings — DESIGNED, NOT LAUNCHED

**Every command below is `AUTHORIZED_TO_RUN = False`. Nothing in this
document was executed. No GPU was touched by this diagnosis.**

## A. ZERO20 — width/control ablation

**Question it answers:** is the harm caused merely by *widening* the
input embedding layer and the resulting change to the optimization
problem (more parameters, a different effective learning-rate-to-
parameter-count ratio, different init-scale interactions), independent of
what the 20 extra columns actually contain?

**Why it's ranked #1** (see `ROOT_CAUSE_DIAGNOSIS.md` for the full
argument): H8's inference-only counterfactuals already show the *trained*
ParT2M checkpoint's performance is statistically unchanged whether its 20
ParT columns hold their real values, are zeroed, or are scrambled — i.e.
the final policy barely depends on ParT content. That is exactly what
ZERO20 would be expected to show at the *training* level if the harm is a
pure widening/architecture-construction effect. ZERO20 is the single
experiment most likely to convert that inference-time observation into a
trained, decisive answer.

**Design:**
- Exact same 27-input architecture, options, seed (0), optimizer,
  schedule, loss scales as the real ParT2M run (`training_spanet_2M_
  part_active_seed0_20260910_result.json`'s `options_snapshot`,
  byte-identical except input paths — see `audit/H5_architecture_input_
  contract.md`).
- The 20 `part_emb_*` columns are **identically zero from the first
  gradient step** (not eval-time zeroing of an already-trained network,
  the actual gap H8 cannot close).
- Data: copy `spa2m_part_active_{train,val}.h5` (already built, 2.28GB +
  457MB) and overwrite the 20 `part_emb_*` columns with `0.0` — no new
  ParT extraction, no new join, no re-derivation of native features.
  `INPUTS/Source/MASK`/`TARGETS`/`CLASSIFICATIONS` copied verbatim.

**Estimated cost:**
| step | estimate | basis |
|---|---|---|
| data build (copy + zero 20 cols) | ~10–15 min | active27 file sizes (2.74GB combined), local disk, no network/extraction |
| additional storage | ~2.74GB | one new train+val H5 pair, same shape as active27 |
| GPU training (50 epochs, 2M events, seed 0) | **~1.8–2.4 GPU-hours** | directly measured: ParT2M's own `total_fit_wall_s=6579s` (1.83h), native's `total_fit_wall_s=8577s` (2.38h) on the same A100 MIG 4g.40gb slice — GPU cost is dominated by the fixed 8-layer/128-dim transformer encoder, not input width (7 vs 27 vs 135 all sit behind a tiny first linear layer) |
| GPU peak memory | ~1.6GB reserved (per ParT2M's own measured `gpu_peak_reserved_bytes`) — negligible on an 80GB-class card |

**Not authorized:**
```
# AUTHORIZED_TO_RUN=False
python3 build_zero20_variant.py \
    --active-train-h5 spa2m_part_active_train.h5 --active-val-h5 spa2m_part_active_val.h5 \
    --out-train-h5 spa2m_part_zero20_train.h5 --out-val-h5 spa2m_part_zero20_val.h5
# ^ build_zero20_variant.py does not exist yet; ~30 lines, h5py copy + zero-fill the 20 part_emb_* columns

# AUTHORIZED_TO_RUN=False
python3 launch_2M_seed0_training.py \
    --event-yaml part_augmented_active_hh4b.yaml \
    --hdf5-train spa2m_part_zero20_train.h5 --hdf5-val spa2m_part_zero20_val.h5 \
    --run-name spanet_2M_zero20_seed0 --seed 0
# ^ same launcher already used for ParT2M / native, only the input H5 changes
```

## B. ALL128 — feature-selection sensitivity ablation

**Question it answers:** did the active20 variance-based pruning
specifically (H4's confirmed defect: it discarded dims 29/61/82/121 —
demonstrably more discriminative than anything it kept) cause harm beyond
what a differently-selected/normalized ParT representation would?

**Design:**
- Same 2,000,000/400,000-event cohort, same architecture/training
  protocol, **all 128 frozen ParT dimensions** (native 7 + ParT 128 = 135
  input features), eps-clamped standardization, **TRAIN-only** statistics
  — using the *already-implemented* `--variant all128` mode of
  `build_augmented_input.py` (confirmed present and complete by reading
  the script; only ever exercised for `--variant active` so far).

**Do existing joined135 artifacts permit this without rerunning ParT
extraction? YES — confirmed, not assumed:** `joined_train.h5` /
`joined_val.h5` (`track_f_postproduction_pipeline_20260908_v1/logs/
join_v2_20260910T205544Z/`) already contain the full native-7 +
ParT-128 = 135-wide tensor for every one of the 2,000,000/400,000 events,
built once from the raw ParT shards and never touched since. `PART_TRAIN_
STATS_FROZEN.json` already has TRAIN-only mean/std/min/max/quantiles for
**all 128** dims (independently re-verified in H3, not just the 20
retained ones — this session recomputed the retained 20; the file itself
already reports all 128). `build_augmented_input.py --variant all128`
reads exactly these two existing artifacts and writes a new
eps-clamped-standardized 135-wide H5 — **zero new shard reads, zero new
xrdcp, zero re-extraction.**

**Estimated cost:**
| step | estimate | basis |
|---|---|---|
| data build (`--variant all128`, read joined135 + write standardized 135-wide H5) | ~20–40 min | active27's own build (27-wide output) took ~7 min wall (`BUILD_RECEIPT_active_{train,val}.json` timestamps, 16:36→16:43) reading the same 12.9GB joined input; all128's *output* is ~5× wider (135 vs 27 cols) so write-bound time is the main driver, ballparked at 3–5× |
| additional storage | **~12.9GB** (train ~10.8GB + val ~2.1GB, same shape as the already-existing joined135 files) | new standardized copy, separate from the raw joined135 files (kept) |
| GPU training (50 epochs, 2M events, seed 0) | **~1.8–2.4 GPU-hours** | same reasoning as ZERO20 — width 135 vs 27 is negligible against the fixed transformer stack |
| GPU peak memory | ~1.6–1.8GB reserved (extrapolated; still negligible) | |

**Not authorized:**
```
# AUTHORIZED_TO_RUN=False
python3 build_augmented_input.py \
    --variant all128 \
    --joined-train-h5 joined_train.h5 --joined-val-h5 joined_val.h5 \
    --train-stats PART_TRAIN_STATS_FROZEN.json --eps 1e-3 \
    --out-h5 spa2m_part_all128_train.h5 --out-yaml part_augmented_all128_hh4b.yaml \
    --out-receipt BUILD_RECEIPT_all128_train.json
# (and the matching --split val invocation)
# ^ this exact tool already exists and is already tested for --variant active;
#   --variant all128 has never been executed.

# AUTHORIZED_TO_RUN=False
python3 launch_2M_seed0_training.py \
    --event-yaml part_augmented_all128_hh4b.yaml \
    --hdf5-train spa2m_part_all128_train.h5 --hdf5-val spa2m_part_all128_val.h5 \
    --run-name spanet_2M_all128_seed0 --seed 0
```

## Ranking by information gained per GPU-hour

Both cost essentially the same GPU-hours (~2h; compute is dominated by
the fixed encoder, not input width) and both need a similar order of
human/wall-clock overhead, so the ranking comes down to information value
and how directly each result would be interpretable:

1. **ZERO20 (higher priority).** Directly and cleanly isolates the
   leading hypothesis from this diagnosis (H7's epoch-0-present,
   assignment-specific gap; H8's finding that the trained network barely
   uses its ParT content) from any question about *which* ParT dimensions
   are informative. A clean, single-variable ablation: same architecture,
   same width, content held at a known constant. **If ZERO20 reproduces
   most of the ΔAUC/Δreconstruction harm, the mechanism is a widening/
   optimization-dynamics effect** — independent of ParT (or JP-JEPA, or
   any future embedding) content, with direct implications for how *any*
   frozen-embedding integration should be done project-wide (e.g. a
   smaller initial-embedding width for the auxiliary block, a warmup
   schedule, or per-block learning rates). **If it does not**, that
   pattern is ruled out and the content/selection-based hypotheses (H4)
   rise in priority. Either outcome is decisive and actionable.
2. **ALL128 (second priority, lower marginal value given what H4/H8
   already show).** Tests whether *dropping specific informative
   dimensions* (H4's confirmed defect) is the harm's driver, independent
   of the width question — but ALL128 simultaneously changes *both* the
   feature content (all 128 vs 20) *and* the width (135 vs 27), so a
   positive result (ALL128 recovers performance) would be genuinely
   informative, while a negative result (ALL128 does not recover
   performance) would be ambiguous between "content wasn't the issue" and
   "widening to 135 is even worse than widening to 27" — it cannot cleanly
   separate those without ZERO20's result as a reference point. **Run
   ZERO20 first; its result sharpens what an ALL128 result would mean.**

Both remain **unauthorized**. Nothing above should be launched from this
diagnosis alone — see `ROOT_CAUSE_DIAGNOSIS.md`'s verdict for what would
need to happen before either is authorized.
