# Exact-2M JP-JEPA Embedding Production Plan (PREPARED, NOT RUN)

Status: PLAN ONLY. No production launched by this task or this package.

## 0. Important context this plan must not ignore

This task discovered that **a separate, already-authorized effort
elsewhere in this project's EOS namespace already targets exactly
this** -- a full-2M, dual ParT+JP-JEPA embedding production. See
`STATUS.md` for the full discovery narrative. Concretely:

- `part_jpjepa_dual_production_bundle_20260825_v2` (2026-08-25):
  bundles `extract_shard.py` (`process_one_shard_dual`), a 1,100-shard
  manifest (880 train + 220 val, matching the frozen native SPA2M
  cohort exactly), both checkpoints (`ParT_full.pt`,
  `jpjepa_mini_pretrained.ckpt`), and an explicit authorization note
  (`docs/JPJEPA_AUTHORIZATION_UPDATE.md`) declaring JP-JEPA
  "scientifically authorized for inference" as of that date.
- A newer, **currently active** attempt
  (`track_b_part_producer_v3_ihep_auth_fix_20260908_v1`, dated today)
  is running this exact production against the exact same manifest,
  currently gated on an IHEP XRootD authentication issue that a status
  file in that directory records as fixed as of today (a GPU canary
  reportedly passed on EAF after a proxy refresh).
- As of the most recent status this task read (dated today), the
  **ParT** side of this dual production stands at 747/880 train and
  7/220 val shard-pairs complete; the JP-JEPA side's completion count
  was not independently re-verified by this task (see `STATUS.md`).

**This plan therefore does not propose a new, from-scratch production
design.** Its job is to (a) confirm, via this package's independent
audit, that the *scientific* content of that existing plan (feature
construction, extraction point, checkpoint, jet cohort) is correct --
which `FEATURE_COMPATIBILITY_MATRIX.md`, `EXTRACTION_POINT_PROOF.md`,
and `CANARY_RUNBOOK.md` now do -- and (b) record, as the task
instructions require, what an exact-2M production plan looks like,
including where it deliberately diverges from or has an open question
against the existing effort's own choices. It is not a substitute for
resuming/monitoring the already-in-flight effort, which belongs to
whoever owns that EOS namespace and is outside this package's scope
(no production was touched or launched by this task).

## 1. Event cohort (identical to the native SPA2M / ParT cohort)

- **880 train shards / 2,000,000 events**, **220 val shards / 400,000
  events** -- the same manifest already frozen for the ParT production
  (`manifest_train_2M.tsv`, `SHARD_MANIFEST.json`,
  `SHARD_MANIFEST.sha256 = 35049c136a28dfaec377cb718a7d2e75abc5c7d931450fd8746606c453b32333`
  per the existing bundle's own fail-closed check in `extract_shard.py`).
  **JP-JEPA production must use this identical, already-hashed
  manifest** -- not regenerate one -- so that event/jet identity is
  provably shared with both the native SPA-Net cohort and the ParT
  embeddings.
- Process mix: 48.34% signal (H3VAR) / 43.62% QCD / 8.04% ttbar,
  identical across train/val (per the existing manifest's own summary).
- Join key: `(source_file_rank, source_entry_index_in_file, jet_slot)`
  -> `native_hdf5_row_index`, the same triple already used to join
  native SPA-Net rows to ParT embedding rows. JP-JEPA embeddings must
  be written keyed the same way.

## 2. Jet selection, ordering, and masks (identical to native/ParT)

`jet_pt > 30 GeV`, `|jet_eta| < 2.5`, sorted pT-descending, top 10 kept
(zero-mask beyond the real count) -- verified identical in
`build_hdf5_2M.py`, `materialize_windowed_tensors_full.py`, and
`extract_shard.py` (all three independently re-derive the same
algorithm; see the Explore-agent-sourced discovery reproduced in
`STATUS.md`). **No change proposed.** Constituent-to-jet assignment via
`part_label`; constituent order is storage order (verified
architecturally irrelevant, `FEATURE_COMPATIBILITY_MATRIX.md` section 4).

## 3. Feature construction

Exactly `PREPROCESSING_SPEC.md` + `FEATURE_COMPATIBILITY_MATRIX.md`:
17 natural-order `pf_features`, `(px,py,pz,E)` lorentz vectors, `pf_mask`,
`maxlen=128`, deta/dphi **derived** from `part_eta`/`part_phi` minus the
assigned AK4 jet's `jet_eta`/`jet_phi` (never the native `part_deta`/
`part_dphi` branches -- section 2 of the matrix document). This matches
what `extract_shard.py`'s `build_shard_tensors` already does.

## 4. Extraction point

`jet_embedding = all_layer_outputs[-1][0][0]` from `part_mini(
pretrained_weights="jpjepa_mini_pretrained.ckpt", num_classes=None)`,
verified in `EXTRACTION_POINT_PROOF.md`. This matches `extract_shard.py`'s
`run_jpjepa` exactly (`outs[-1][0][0]`).

## 5. Checkpoint and code provenance to freeze

- `jpjepa_mini_pretrained.ckpt`, SHA256
  `1484e704d9f872c17eed6ead713f9ac39c6866dd74d5ebb45c23c9af7b37e3e9`
  (independently re-verified by this task against the existing EOS copy).
- `jetparticle-jepa` pinned at `c68509eead1866c2c86714147023f5e8312634c4`
  (independently verified by this task against upstream `git
  ls-remote`, see `UPSTREAM_PROVENANCE.md`).
- Extraction code: this task recommends **reusing** the existing,
  already-hash-tracked `extract_shard.py` /
  `build_jpjepa_reordered_features` / `load_jpjepa_mini_encoder` /
  `run_jpjepa` functions (source-hashed in every shard receipt already),
  rather than introducing a second, independently-written
  implementation into the production path -- this package's own
  `scripts/run_jpjepa_mini_canary.py` was deliberately written
  independently *for cross-checking purposes only* (see
  `CANARY_RUNBOOK.md`) and is not proposed as a production replacement.

## 6. Chunking / idempotency / auth-safety

Already designed into the existing pipeline and should be kept:

- One shard = one source ROOT file window (`SHARD_MANIFEST.json` entry),
  processed independently -- naturally chunked and resumable (a failed
  shard does not corrupt others).
- `process_one_shard_dual` **fails closed**: raises and writes nothing
  on any non-finite embedding or row-count mismatch (`extract_shard.py:438-453`)
  -- never writes a partial/bad shard.
- The auth failures seen today (`[3010] ... Operation not permitted`
  against IHEP XRootD) are an X.509/token issue external to the
  extraction logic itself; today's status record in that namespace
  reports a proxy refresh resolved it for a GPU canary. This plan does
  not add new auth-handling logic beyond what already exists, since the
  existing effort already appears to have addressed it operationally
  (out of scope for this package to verify further).

## 7. EOS namespace: resolved (2026-09-09) -- historical dual bundle vs. the currently-live ParT-only production are two distinct things

**Update, 2026-09-09** (part of the CPU/GPU numerics resumption, see
`CPU_GPU_NUMERICS_AUDIT.md`): the schema-level ambiguity this section
previously "flagged rather than resolved" has now been checked directly
against the actual files on disk, read-only, this task -- not inferred
from directory names or status prose. Verdict below. **No file in any
namespace was altered, written to, or reused by this check.**

- **Historical dual ParT+JP-JEPA artifacts** (2026-08-25, NOT live):
  `part_jpjepa_dual_production_bundle_20260825_v2/` (the code/checkpoint
  bundle) and `track_b_part_jpjepa_dual_embedding_production_20260825_v2/`
  (its canary outputs, including the CPU/GPU diagnostics this task
  reused). Directly inspecting the HDF5 schema of
  `canary_eaf/{cpu_reference,run1,run2}/train_0000*.h5` confirms these
  files **do** carry both `EMBEDDINGS/part_full` **and**
  `EMBEDDINGS/jpjepa_mini` as sibling datasets, exactly as
  `write_shard_output_dual` is designed to produce. This bundle's own
  production shard loop was never run to completion at 2M scale (only
  the 12,089-jet `train_0000` dual canary exists here) -- it is a
  frozen historical artifact, not an active job.
- **The currently-live production** is
  `track_b_part_full_embedding_production_exact_2M_20260824_v1/shards/`
  (the actual growing 2M-scale shard output) and today's
  `track_b_part_producer_v3_ihep_auth_fix_20260908_v1/bundle_v3_canary/`
  (today's resumption attempt after the IHEP auth fix). **Directly
  inspecting the HDF5 schema of a real shard from each of these two
  directories** (`shards/train/*.h5` and
  `bundle_v3_canary/train/train_0870.h5`) shows **only**
  `EMBEDDINGS/part_full` -- **no `jpjepa_mini` key exists in either**.
  The currently-running production is confirmed **ParT-only**, despite
  living in a directory tree whose sibling (`track_b_part_jpjepa_dual_...`)
  and code heritage (`extract_shard.py`'s `process_one_shard_dual`) are
  dual-capable. It is simply not being invoked in dual mode right now.
- **Practical resolution**: the task's original instruction (embeddings
  into a namespace "completely separate ... from ParT") and the existing
  effort's combined-namespace design (section 7's original discussion,
  preserved below) both remain live, unreconciled options for a *future*
  full JP-JEPA production -- **that decision is still not made here**,
  and still not authorized to be made unilaterally by this package. What
  *is* now resolved is the factual question "does the live ParT job's
  output already contain JP-JEPA embeddings that could be silently
  reused or overwritten by mistake": **no, verified by direct schema
  read, not assumption.** It is therefore safe to state plainly: nothing
  about JP-JEPA readiness work (this package, or any future resumption)
  should touch `track_b_part_full_embedding_production_exact_2M_20260824_v1/`
  or `track_b_part_producer_v3_ihep_auth_fix_20260908_v1/` -- they are
  pure ParT, and altering or reusing their files under a mistaken
  assumption that JP-JEPA data is already present there would corrupt
  the live ParT production for no reason.

### Original discussion (2026-09-08, preserved for context)

The task instructions for this package (Task 11) ask for embeddings to
be produced into a namespace "completely separate EOS namespace from
ParT." **The existing, already-authorized (but not currently live in
dual mode) production design instead writes ParT and JP-JEPA embeddings
into the same shard HDF5 files** (one file per shard,
`EMBEDDINGS/part_full` and `EMBEDDINGS/jpjepa_mini` as sibling datasets,
per `write_shard_output_dual`) when run in dual mode, in a shared
namespace.

**This plan does not resolve this tension** for any future production.
It is recorded here as an explicit decision point for whoever owns and
continues the actual production:

- The combined-namespace design has real advantages already realized
  (one manifest, one fail-closed shard loop, one join key, one GPU pass
  reading the source file once for both encoders -- see
  `extract_shard.py`'s docstring reasoning) and is already
  hash-frozen and was exercised (at 12,089-jet canary scale, not 2M
  scale -- see above).
- A fully separate namespace, as this task's instructions describe,
  would mean re-deriving a second, ParT-independent production path --
  and, now that it's confirmed the live production is ParT-only with no
  JP-JEPA content to lose, this would not discard any existing JP-JEPA
  production progress (there is none at 2M scale to discard).

No production decision is made here either way; `SPANET_JPJEPA_2M_PLAN.md`
is written to be agnostic to which storage layout is ultimately used
(it consumes a `native_hdf5_row_index`-keyed JP-JEPA embedding array
regardless of which file it physically lives in).

## 8. Pre-production gate (mirrors the frozen ParT gate)

Before any JP-JEPA embedding is trusted for SPA-Net training, per this
project's own established discipline
(`docs/analysis_contracts/SPANET_PART_RESOURCE_AWARE_COMPARISON.md.save`'s
training gate, adapted):

- [ ] 880/880 train JP-JEPA shard entries valid and finite
- [ ] 220/220 val JP-JEPA shard entries valid and finite
- [ ] native-event <-> JP-JEPA-event identity closure passes
      (`native_hdf5_row_index` join, same scheme as ParT)
- [ ] native-jet <-> JP-JEPA-jet identity closure passes
- [ ] JP-JEPA embedding dimension is exactly 128 (Mini)
- [ ] no non-finite embeddings present (this task's canary already
      shows 0/1050 non-finite at small scale -- full-scale re-check
      still required, since a heavy-tailed d0/dz saturation event or
      an extreme-multiplicity jet at 2M scale could in principle differ
      from the 1,050-jet canary sample)
- [ ] checkpoint SHA256 matches `1484e704d9f872c17eed6ead713f9ac39c6866dd74d5ebb45c23c9af7b37e3e9`
      exactly at production time (re-verify, do not assume)
- [ ] `jetparticle-jepa` pinned commit re-verified as
      `c68509eead1866c2c86714147023f5e8312634c4` at production time
- [ ] relevant code/config/checkpoint hashes frozen and recorded in a
      `SHA256SUMS`/receipt, per this project's standing convention
- [ ] **(added 2026-09-09, see `CPU_GPU_NUMERICS_AUDIT.md`)** the full
      2M-scale production uses a single, consistently-recorded device
      (CPU or GPU) throughout -- CPU and GPU forward passes are **not**
      bit-compatible for JP-JEPA Mini (`relative_L2` typically `<0.001`,
      worst case `<0.005` on the diagnostic sample audited); mixing
      devices across shards within one production would silently
      introduce this same magnitude of noise as a shard-to-shard
      artifact rather than a documented, uniform tolerance

**None of these gate items were run at full 2M scale by this task.**
The canary (`CANARY_RUNBOOK.md`) satisfies the *feasibility* half of
this list at O(1000)-jet scale only.
