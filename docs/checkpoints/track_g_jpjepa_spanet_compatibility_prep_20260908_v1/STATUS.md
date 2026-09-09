# Track G Status: JP-JEPA / SPA-Net Compatibility Preparation

Package: `track_g_jpjepa_spanet_compatibility_prep_20260908_v1`
Last updated: 2026-09-09
Overall gate: **COMPATIBILITY-PREPARATION COMPLETE, WITH DOCUMENTED
NUMERICAL TOLERANCE** -- not a production authorization.

This file is the single top-level status record for this package
(referenced by every other document in it). It was missing from the
package as of the 2026-09-08 freeze despite being referenced
throughout; it is created now, 2026-09-09, as part of resuming this
track to close the CPU-vs-GPU numerics question. Nothing below
retroactively changes what was already done on 2026-09-08 -- it records
it, plus what was added today.

## What this package is, and is not

This is a **compatibility-preparation and diagnostics** package for
using JP-JEPA Mini's frozen `jet_embedding` as a SPA-Net input feature,
by analogy with the already-frozen ParT contract
(`docs/analysis_contracts/SPANET_PART_RESOURCE_AWARE_COMPARISON.md.save`).
It is explicitly **not**:
- a JP-JEPA production run (only a 1,050-jet canary and a 12,089-jet
  CPU/GPU diagnostic subset were ever processed, both far below 2M
  scale),
- a SPA-Net training run,
- an authorization to launch either of the above,
- a modification to the live, currently-running ParT-only exact-2M
  production (`track_b_part_full_embedding_production_exact_2M_20260824_v1`,
  `track_b_part_producer_v3_ihep_auth_fix_20260908_v1`) -- confirmed
  untouched, see `EXACT2M_PRODUCTION_PLAN.md` section 7.

## Deliverables and their status

| Document | Status | Summary |
|---|---|---|
| `UPSTREAM_PROVENANCE.md` | DONE (2026-09-08) | `jetparticle-jepa` pinned at `c68509e`, source SHA256s recorded, dependency gap noted |
| `PREPROCESSING_SPEC.md` | DONE (2026-09-08) | Exact upstream feature/preprocessing pipeline reproduced with line citations |
| `FEATURE_COMPATIBILITY_MATRIX.md` | DONE (2026-09-08) | Every required JP-JEPA input verified derivable from real Yang-Li ntuples; caught and corrected a native-branch trap (`part_deta`/`part_dphi` are NOT jet-relative) |
| `EXTRACTION_POINT_PROOF.md` | DONE (2026-09-08) | `jet_embedding = all_layer_outputs[-1][0][0]` verified line-by-line against upstream source; pre-final-LayerNorm subtlety documented |
| `CANARY_RUNBOOK.md` | DONE (2026-09-08) | 1,050 real jets (350 QCD/350 signal/350 ttbar), all finite, masking verified correct, no near-constant dimensions |
| `EXACT2M_PRODUCTION_PLAN.md` | DONE (2026-09-08), namespace ambiguity **resolved 2026-09-09** | Plan-only; discovered a pre-existing, already-authorized dual ParT+JP-JEPA production effort elsewhere in this project; namespace question now settled by direct schema inspection (section 7) |
| `CPU_GPU_NUMERICS_AUDIT.md` | **DONE (2026-09-09, this update)** | Bounded numerical audit of the CPU-vs-GPU `jet_embedding` discrepancy first found in the 2026-08-25 dual canary. See verdict below. |
| `EMAIL_DRAFT_NOT_SENT.md` | **DONE (2026-09-09, this update)** | Drafted, **not sent**, per instruction |
| `work/canary_report.json` | DONE (2026-09-08) | Machine-readable 1,050-jet canary output |
| `work/cpu_vs_gpu_numerics_audit.json` | **DONE (2026-09-09)** | Machine-readable numerics audit output |
| `work/cpu_batch_invariance.npz` | **DONE (2026-09-09)** | Raw arrays backing the CPU batch-invariance check |

## CPU-vs-GPU numerics verdict

```
JPJEPA_NUMERICS_ACCEPTABLE_WITH_DOCUMENTED_TOLERANCE
```

Full reasoning: `CPU_GPU_NUMERICS_AUDIT.md`; findings-only summary:
`DIAGNOSTIC_FINDINGS_SO_FAR.md`. In brief: the discrepancy is real
(`max_abs_diff=0.060`, `mean_abs_diff=0.00157`, `min_cosine=0.999989`
on the final `jet_embedding`, 12,089 real jets), exceeds the numeric
bounds originally derived from ParT's own tighter CPU/GPU behavior, and
is **not fully mechanistically explained** (no live GPU access this
session to run the one remaining ablation, section 8 of the audit
document) -- but it is bounded, well-characterized, shown not to be
driven by batch size, GPU nondeterminism, a handful of pathological
jets, or an explicit precision/kernel choice in this project's own
code, and shown to be 1-2 orders of magnitude smaller than every
physics-relevant scale in the same embedding space (nearest-neighbor
jet distance, process-mean separation). It does not block using this
frozen representation as a downstream SPA-Net input feature, provided
this tolerance is documented wherever that comparison is made -- hence
"ACCEPTABLE_WITH_DOCUMENTED_TOLERANCE", not a clean "PASS".

## Explicit non-actions (this session, 2026-09-09)

Per this task's explicit scope limits, none of the following were done:
- No full-scale (2M/400K) JP-JEPA production launched.
- No SPA-Net training, holdout_B evaluation, or Stage C work performed.
- No change to the live ParT EAF job
  (`track_b_part_full_embedding_production_exact_2M_20260824_v1`,
  `track_b_part_producer_v3_ihep_auth_fix_20260908_v1`) -- read-only
  inspection only (`find`, `h5py` reads, `sha256sum`), verified via
  `git status`-equivalent (no EOS write commands issued).
- No re-run of the already-complete feature-compatibility audit or
  1,050-jet Mini canary.
- No email sent (`EMAIL_DRAFT_NOT_SENT.md` is a draft only).
- No architecture change and no different embedding-extraction point
  chosen to make CPU/GPU agree -- `all_layer_outputs[-1][0][0]` remains
  the frozen extraction point exactly as upstream's `inference.py` uses
  it (`EXTRACTION_POINT_PROOF.md`).

## Known open item (not blocking, but not closed)

A live GPU-side layer-by-layer capture (`diagnostics/layer_hook_localization.py
--device cuda --out gpu_layers.npz`, already written and ready) and a
live TF32-on/off ablation were **not executed** -- this session had no
reachable GPU (no local CUDA hardware, no SSH route to EAF, no GPU
HTCondor slots visible from the LPC pool). This is the single concrete
action that would fully mechanistically close the CPU/GPU question
rather than bound it; see `CPU_GPU_NUMERICS_AUDIT.md` section 8 for the
exact one-line ablation to run and where.
