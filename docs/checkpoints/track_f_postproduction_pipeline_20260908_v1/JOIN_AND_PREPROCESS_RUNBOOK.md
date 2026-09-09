# Join + preprocess runbook — native SPA2M ⊕ frozen ParT

**Gated on `POSTFLIGHT_RUNBOOK.md` passing (`OVERALL_POSTFLIGHT_PASS=true`).
Not run against real production data this session.** Every script below
was dry-run tested against a synthetic H5 of the same schema in this
session (see receipt.json's `dry_run_verification` section) — the join and
preprocessing *logic* is verified end-to-end, only the real 1,100-shard
population is untouched.

## Event identity — explicit provenance, never positional guessing

The join key is `JOIN_KEY/{native_hdf5_row_index, jet_slot}`, written into
every ParT shard at production time. This is **not** a coincidental
row-count match: `SPANET_PART_COHORT_RECONCILIATION.md` proved the ParT
shards were built by reading the *same* governing source files at the
*same* `(offset, quota)` windows as `build_hdf5_2M.py` used for the native
H5 — so `native_hdf5_row_index` genuinely addresses the identical row in
`production_2M_{train,val}.h5`. Every join step below independently
re-verifies this rather than trusting it:
- every `(row, slot)` pair is unique within its own shard,
- every row falls inside that shard's own manifest-declared window
  (never another shard's),
- every covered `(row, slot)` is cross-checked against the **native file's
  own `MASK`** — a ParT embedding landing on a native "not a real jet" slot
  is a hard failure, not a warning.

Order is preserved by writing each shard's rows into the *identity-derived*
`[row_start, row_end)` slice of the output tensor — never by concatenating
shards in whatever order they happen to be listed.

## Step 1 — full-population join

Reuses `streaming_native_part_join.py`
(`track_b_part_postproduction_tools_20260826_v2/code/`) unchanged —
already tested at 5/20/50/150-shard bounded scale with peak RSS bounded by
the single largest shard, not total population size. This package's
wrapper (`scripts/join/run_full_join.sh`) adds: a hard require that
postflight passed, native-H5 checksum re-verification, local shard sync,
and a mandatory 100%-coverage gate (`verify_join_coverage.py`) — the
underlying join script alone will silently accept partial coverage, which
is fine for its own dev/test use but not for this pipeline's final output.

```bash
PKG=/uscms_data/d3/iturkmen/hh4b_delphes/track_f_postproduction_pipeline_20260908_v1

bash "$PKG/scripts/join/run_full_join.sh" \
    "$PKG/logs/postflight_<UTC_from_postflight_step>" \
    "$PKG/logs/join_$(date -u +%Y%m%dT%H%M%SZ)"
```

Produces, per split, `joined_<split>.h5` (`joined_features` shape
`(N,10,135)` = 7 native + 128 ParT, `mask`, `part_coverage_mask`) plus
`JOIN_RECEIPT_<split>.json` and a combined `JOIN_RECEIPT_COMBINED.json`.
**`OVERALL_JOIN_PASS=true` in that combined receipt is the only valid
signal to proceed.**

## Step 2 — TRAIN-ONLY embedding diagnostics (never val)

The prior `PREPROCESSING_DECISION.md` (2026-09-07, std<1e-3 threshold, 20
retained dims) was computed from an **incomplete 31-shard / 336,528-jet
sample** and is explicitly not a frozen scientific result. This step
recomputes everything from the **complete TRAIN population only**.

```bash
python3 "$PKG/scripts/preprocess/compute_train_embedding_stats.py" \
    --joined-train-h5 "$PKG/logs/join_<UTC>/joined_train.h5" \
    --out "$PKG/logs/join_<UTC>/PART_TRAIN_STATS_FROZEN.json"
```

Reports, per dimension: mean/std/min/max/quantiles, non-finite count,
embedding-norm distribution, and a **threshold scan** across
`{1e-1,1e-2,5e-3,1e-3,1e-4,1e-5}` plus a data-driven gap threshold (largest
log-scale jump between consecutive sorted stds) — printed for review, not
auto-applied. Compare this scan's counts to the prior provisional 20-dim
decision before picking `--active-threshold` in Step 3; if the full
population's gap sits somewhere other than ~1e-3, **do not silently keep
1e-3** — that is exactly the "silently canonize" failure mode the task
explicitly warned against.

## Step 3 — build both model-input variants

Both variants are built from the **same** `PART_TRAIN_STATS_FROZEN.json`
(train-only, frozen, applied unchanged to val) so neither result can be
attributed to different preprocessing:

```bash
# Variant A: resource-aware active-dimension version. --active-threshold
# is REQUIRED and must be the value you justified from Step 2's scan (the
# command below uses 1e-3 as an EXAMPLE, matching the prior provisional
# choice for continuity -- re-justify, do not copy blindly):
for SPLIT in train val; do
  python3 "$PKG/scripts/preprocess/build_augmented_input.py" \
      --joined-h5 "$PKG/logs/join_<UTC>/joined_${SPLIT}.h5" \
      --native-h5 /uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AF_spanet_partial_events_population_correction_20260819_v1/production_hdf5_build_v1/work/production_2M_${SPLIT}.h5 \
      --split "${SPLIT}" \
      --train-stats "$PKG/logs/join_<UTC>/PART_TRAIN_STATS_FROZEN.json" \
      --variant active --active-threshold 1e-3 \
      --event-yaml-template /uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4S_official_spanet_integration_canary_20260814_v1/event_config/trackb_hh4b.yaml \
      --out-h5 "$PKG/logs/join_<UTC>/spa2m_part_active_${SPLIT}.h5" \
      --out-yaml "$PKG/logs/join_<UTC>/part_augmented_active_hh4b.yaml" \
      --out-receipt "$PKG/logs/join_<UTC>/BUILD_RECEIPT_active_${SPLIT}.json"
done

# Variant B: all-128 sensitivity/control version, eps-clamped:
for SPLIT in train val; do
  python3 "$PKG/scripts/preprocess/build_augmented_input.py" \
      --joined-h5 "$PKG/logs/join_<UTC>/joined_${SPLIT}.h5" \
      --native-h5 /uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AF_spanet_partial_events_population_correction_20260819_v1/production_hdf5_build_v1/work/production_2M_${SPLIT}.h5 \
      --split "${SPLIT}" \
      --train-stats "$PKG/logs/join_<UTC>/PART_TRAIN_STATS_FROZEN.json" \
      --variant all128 --eps 1e-3 \
      --event-yaml-template /uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4S_official_spanet_integration_canary_20260814_v1/event_config/trackb_hh4b.yaml \
      --out-h5 "$PKG/logs/join_<UTC>/spa2m_part_all128_${SPLIT}.h5" \
      --out-yaml "$PKG/logs/join_<UTC>/part_augmented_all128_hh4b.yaml" \
      --out-receipt "$PKG/logs/join_<UTC>/BUILD_RECEIPT_all128_${SPLIT}.json"
done
```

**Schema note (matches the native H5's own layout, verified by direct
h5py inspection, not one packed tensor):** each output H5 carries one
named `(N,10)` dataset per feature — `INPUTS/Source/{pt,eta,phi,mass,
probB,probC,probL}` (copied unchanged) plus `INPUTS/Source/part_emb_NNN`
per retained dimension (standardized) — and `TARGETS/*`,
`CLASSIFICATIONS/*` copied **verbatim from the original native H5**
(the joined intermediate never carries these groups at all, since
`streaming_native_part_join.py` only ever touches `INPUTS/Source/*`).
Masked jet slots are written as exact `0.0` in every ParT column,
matching native zero-padding. **The native `production_2M_{train,val}.h5`
files themselves are never opened for writing, only read.**

Every build writes a receipt with schema, SHA256 (input joined H5, input
native H5, train-stats source, output H5), event counts, and the full
output feature-name list, per the task's explicit requirement.

## Verification performed this session (synthetic, non-destructive)

A 500-event synthetic joined+native H5 pair (100/128 dims forced
near-constant) was built in the scratchpad and run through
`compute_train_embedding_stats.py` (correctly identified 28 active
dimensions at the data-driven gap) and `build_augmented_input.py` in both
variants (correct output shapes: 35-wide for `active`, 135-wide for
`all128`; `TARGETS`/`CLASSIFICATIONS` correctly copied; zero non-finite
values). See receipt.json for exact commands/output.
