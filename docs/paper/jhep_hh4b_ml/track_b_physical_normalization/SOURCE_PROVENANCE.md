# Source Provenance

This archive is copied from a frozen, internally-hashed source-of-truth
package (`track_b_four_model_physical_normalization_freeze`, status
`FOUR_MODEL_PHYSICAL_NORMALIZATION_FREEZE_PASS`). Internal filesystem
paths from the originating analysis workspace are intentionally not
reproduced here; provenance is instead carried by content hashes, which
are independently reproducible from the numbers in this archive.

## Chain of custody (generic description)

1. **Raw pass-count scans.** For each of the four model artifacts
   (identified below by SHA256), the number of stored SM inference
   events passing each target-efficiency threshold, and the number of
   stored QCD inference events passing the same threshold (applied to
   each model's own score), were counted directly from the frozen
   SM and QCD inference populations. No event-level data is included in
   this archive — only the resulting integer pass counts and the
   derived physical quantities.
2. **Common-grid completion.** The initial pass-count scan for the BDT
   models covered a subset of the 17 target efficiencies; the
   remaining working points were computed in a supplemental pass using
   the identical counting procedure and are labeled
   `SUPPLEMENTAL_EXACT_COMMON_GRID_COMPLETION` in the `provenance`
   column, as distinct from the originally-scanned rows
   (`SUPPORTED_FROZEN_EXACT_COUNT`). This distinction is preserved
   throughout this archive and must not be collapsed or reinterpreted
   as "part of the original blind recount" (see §F of the archival
   record for this distinction's origin).
3. **Independent statistical audit.** All 68 canonical rows (17 target
   efficiencies x 4 models) were independently recomputed by a second,
   separately-coded implementation (50-digit decimal arithmetic,
   independently-derived Garwood-interval identity) and cross-checked
   against the primary implementation, a raw-count source table, and a
   third independent red-team audit pass. Maximum observed disagreement
   across all cross-checks was at the level of floating-point rounding
   (roughly $10^{-8}$ to $10^{-15}$ relative) — i.e., no numerical
   disagreement between independent implementations.
4. **Freeze.** The four exact model artifacts, the normalization
   constants, and all 68 resulting rows were then frozen into the
   canonical package referenced above, with a `SHA256SUMS` manifest
   covering every file in it.

## Model identity (exact, hash-verified)

| Model | SHA256 |
|---|---|
| Step2j BDT-K | `46591e1059374aae646675a121885bd3943e1f13d5fdb6b47844cbf9a0546035` |
| Step2j BDT-KF | `1ba187760b831e9452df537958a32828c1816715134b218d37c1d2360508df33` |
| SPA-Net 2M primary checkpoint | `dc39cf76f0d8e40d07228f240fe58b1179e8134cd4f96c5c786a7d528d31ea8d` |
| SPA-Net 10M primary checkpoint | `fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5` |

No model or checkpoint binary is included in this archive. See
`MODEL_FREEZE.md` for each model's architecture, training population,
and checkpoint-selection rule.

## What is and is not included here

**Included:** normalization constants and equations, per-working-point
pass counts and derived physical quantities (all 68 canonical rows plus
17 supplemental cross-check rows in the full common-grid table), the
finite-MC reporting policy, the Yang-Li external-reference comparison,
and two summary figures.

**Not included:** any event-level data (ROOT/HDF5/NPZ/Parquet), any
model or checkpoint binary, any large per-event score array, internal
filesystem paths, internal correspondence, or any advisor-specific
communication material.

## Independence from this repository's other Track-B material

This archive stands on its own: the numbers in it were not
recomputed, re-derived, or altered for publication here. Any
recomputation from the raw CSV in this archive (e.g. regenerating the
included figures) will reproduce the same values, since they are
computed directly from `CANONICAL_FOUR_MODEL_NORMALIZED_RESULTS.csv`.
