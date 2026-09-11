# ROOT_CAUSE_DIAGNOSIS — SPA2M native vs SPA2M+ParT active20 harm

**Package:** `track_i_part_harm_diagnosis_20260911_v1` (additive; nothing
under any Track F/B/G package or the live `exact_2M` production was
modified, moved, or renamed — see `RESULT_FREEZE.md` for the provenance
freeze of the governing result this diagnosis was run against).

**Scope actually performed:** provenance freeze; independent, mostly
full-population re-verification of 8 named root-cause hypotheses;
inference-only counterfactuals on the frozen checkpoint; design (not
launch) of two follow-up trainings; a verified (not assumed) JP-JEPA
blocking assessment. **No training or GPU production was launched. No
frozen file was modified.**

## Final verdict

# **HARM_ROOT_CAUSE_NARROWED_DIAGNOSTIC_TRAINING_REQUIRED**

The mechanism has been narrowed to two specific, quantified, non-mutually-
exclusive candidates — (a) a width/optimization-dynamics effect from
widening the input embedding layer, independent of ParT content, and
(b) a confirmed defect in the variance-based feature-selection method that
discarded genuinely more-discriminative ParT dimensions — with converging
evidence (H7, H8) that (a) is the larger contributor. But **no candidate
has been isolated by an actual controlled training**, which is what would
be needed to reach `HARM_ROOT_CAUSE_IDENTIFIED`. Nothing found rises to
`HARM_UNEXPLAINED_DO_NOT_PROCEED` — every hypothesis this task named was
either cleanly ruled out with full-population evidence, or converted into
a specific, quantified, actionable finding.

## Evidence summary

| # | hypothesis | verdict | key evidence |
|---|---|---|---|
| H1 | Native-feature preservation | **RULED OUT** | 2.4M events, 7 features × MASK × TARGETS × CLASSIFICATIONS, **zero** mismatches, full population |
| H2 | Jet-slot identity | **RULED OUT** (spot-check, not exhaustive) | 6-shard, ~50k-embedding independent `SANITY/jet_pt` cross-check: exact match everywhere; join script's own all-1,100-shard internal check: zero exceptions |
| H3 | Preprocessing numerics | **Transform correct**, one finding carried forward | Recomputed stats agree with frozen file to 1e-6; zero bad transforms / zero nonzero padding across ~500M cells checked; **but** un-clamped outlier z-scores up to **19σ** reach the network |
| H4 | Feature-selection (variance ≠ usefulness) | **CONFIRMED** | Dropped dims 29/61/82/121 are *more* signal- and Higgs-jet-discriminative (AUC dev up to 0.34) than any of the 20 variance-retained dims (max 0.29) |
| H5 | Architecture/input contract | **RULED OUT** | 63/63 `options_snapshot` fields identical except the 3 intended file paths; 654/654 checkpoint tensors identical except the 8 shape-diffs fully explained by `n_features: 7→27` |
| H6 | Checkpoint/weight diagnosis | **Healthy**, one asymmetry noted | No NaN/Inf/saturation anywhere; ParT-associated residual weights are 5–12× smaller than native flavor-tag weights (network structurally under-weights ParT channels) |
| H7 | Training-history diagnosis | **Immediate, assignment-specific, widening gap** | Assignment-loss ratio 2.25× at epoch 0 → 3.66× at epoch 49 (classification ratio flat ~1.3×); ParT2M-only instability spikes (epoch 11: val loss 2.06, cls-acc crashes to 0.577) |
| H8 | Inference-only counterfactuals | **Decisive negative result** | Zeroing/permuting-across-events/permuting-across-slots the 20 ParT columns on the **frozen, already-trained** checkpoint changes AUC by ≤0.0007 and reconstruction by ≤0.010 — the trained network is not, at inference time, meaningfully using its own ParT inputs |

## Synthesis — what actually happened

Put together, H1/H2/H5 close off every "something got corrupted or
mis-wired" explanation with strong, mostly full-population evidence: the
data reaching the network is exactly the intended data, in exactly the
intended slots, through exactly the intended (and otherwise
hyperparameter-identical) architecture. The harm is real, not an artifact
of a plumbing bug.

H7 shows the harm is not something that develops through training (no
late overfitting story) — it is present from the very first logged epoch,
disproportionately on the **assignment** task versus classification (the
same disproportion the frozen eval result itself shows: Δreconstruction
−0.371 vs ΔAUC −0.025), and it *widens* rather than narrows with more
epochs. This is the signature of a harder optimization problem from the
start, not a slowly-accumulating defect.

H8 then supplies the sharpest single piece of evidence in this whole
diagnosis: the checkpoint that resulted from that harder optimization
problem does not, at the end, actually depend on the *content* of its
ParT inputs. Zero them, scramble them across events, scramble them across
jet slots — no change outside noise. Combined with H6's finding that the
network's own learned weights lean away from the ParT channels (5–12×
smaller than the native flavor-tag weights), the most parsimonious
reading is: **the harm is predominantly a consequence of widening the
input embedding layer and the resulting change to training dynamics for
the assignment head — not a consequence of the ParT embedding's specific
information content being actively misleading.**

H4 is a real, independently confirmed, second effect layered on top: the
variance-based selection that produced the 20 retained dimensions
provably discarded some of the *more* useful raw dimensions (29, 61, 82,
121 among them). This does not contradict the H7/H8 picture — a
correctly-selected 20-dim (or full 128-dim) ParT block could still suffer
some width-driven optimization harm, just plausibly less of it, and/or
recover some of what a poorly-selected block loses. H3's ~19σ outlier
z-scores are a plausible proximate mechanism for *why* the wider input
specifically destabilizes training (co-timed with H7's ParT-only
instability epochs) rather than simply adding harmless extra capacity.

None of this is provable without a controlled training — inference-time
evidence (H8) cannot distinguish "the final policy ignores ParT" from
"the training trajectory was harmed by ParT's presence/instability along
the way, converging to a policy that then ignores it." That is exactly
the gap `FOLLOWUP_DESIGNS.md`'s ZERO20 ablation is designed to close, at
~2 GPU-hours.

## JP-JEPA blocking assessment — verified, not assumed

**Your prior (JP-JEPA production should remain blocked until this ParT
integration diagnosis is resolved) is CONFIRMED for the downstream
SPA-Net integration path, and separable for raw embedding extraction.**

Read `SPA2M_JPJEPA_INTEGRATION_CONTRACT.md`
(`track_g_jpjepa_exact2m_production_ready_20260910_v1/`) directly rather
than inferring from the package name. It states, in its own words:

- The feature composition "**mirrors the already-frozen ParT contract
  exactly in structure**": 7 native + 128-d frozen embedding = 135
  features/jet, "slots in exactly where ParT's 128-d embedding currently
  sits."
- The join key is explicitly **the same** `(native_hdf5_row_index,
  jet_slot)` scheme "already used to join native SPA-Net rows to ParT
  embedding rows... no new join logic is needed or should be invented."
- The eventual normalization/pruning step follows **the same
  TRAIN-only-statistics discipline** as ParT's, applied "eventually" once
  JP-JEPA embeddings exist — the contract explicitly (and correctly)
  forbids assuming ParT's specific 20-dim pruning result transfers, but
  it does **not** revise the *method* (population-wide variance
  thresholding) that H4 just showed discards more-useful dimensions than
  it keeps. Unless that method is fixed first, a "fresh" JP-JEPA pruning
  done the same way would very plausibly reproduce H4's defect on JP-JEPA's
  own embeddings.
- The same "native 7 + frozen 128-d block bolted onto SPA-Net's first
  embedding layer" architecture pattern is what H7/H8 implicate as the
  probable dominant harm mechanism (width/training-dynamics, largely
  independent of embedding *identity*) — this applies to JP-JEPA's
  integration exactly as much as it applied to ParT's, regardless of
  whether JP-JEPA's own 128 dimensions turn out to be more or less useful
  than ParT's.

**What is NOT blocked by this diagnosis:** raw JP-JEPA embedding
*extraction/production* (`GPU_CANARY_RUNBOOK.md`,
`FULL_PRODUCTION_RUNBOOK.md`) is already independently gated by its own
`AUTHORIZED_TO_RUN = False` and a GPU-resource precondition ("ParT being
idle"), unrelated to this diagnosis, and produces data (frozen per-jet
embeddings) that exists *before* any of the join/normalize/prune/augment-
SPA-Net-input steps this diagnosis is about. There is no scientific reason
extraction itself needs to wait.

**What SHOULD remain blocked:** any step that joins JP-JEPA embeddings
into an augmented SPA-Net input and trains on it — until (a) ZERO20 (and
ideally ALL128) have run and the width-vs-content question from this
diagnosis is resolved, and (b) whatever feature-selection method JP-JEPA
uses is checked against H4's finding (i.e. not blind population-wide
variance thresholding) — should not proceed. This is a direct, textual
confirmation of your prior, not an inference from similarity alone.

## Proposed next-command packet

**All commands below remain `AUTHORIZED_TO_RUN = False`.** Full design,
cost estimates, and exact (unauthorized) command lines are in
`FOLLOWUP_DESIGNS.md`.

1. **ZERO20** (recommended first; ~2 GPU-hours, ~2.7GB storage, ~10–15 min
   data-build, no new ParT extraction). Directly tests the leading
   hypothesis from H7/H8.
2. **ALL128** (recommended second, informed by ZERO20's result; ~2
   GPU-hours, ~12.9GB storage, ~20–40 min data-build, **confirmed** to
   need zero new ParT extraction — `joined_train.h5`/`joined_val.h5`
   already hold the full 135-wide tensor, and `build_augmented_input.py
   --variant all128` already exists, untested only because unexecuted).
   Tests H4's confirmed feature-selection defect.
3. Only after both: revisit whether a fixed (non-variance-only) selection
   method and/or a narrower/warmed-up embedding-layer construction let a
   ParT- or JP-JEPA-augmented SPA-Net avoid the harm this Track F result
   measured. **None of that redesign work is performed or authorized by
   this package.**

## What this package explicitly did NOT do

- Did not launch training, GPU production, or any modification to
  `exact_2M`/`ParT_full.pt`/any Track F or Track B frozen artifact.
- Did not modify any Track F preregistration, evaluation code, schema, or
  result file. `RESULT_FREEZE.md`/`SHA256SUMS` prove the governing result
  is untouched.
- Did not authorize ZERO20 or ALL128 — both remain designed-only.
- Did not touch JP-JEPA production authorization, code, or state.
- Did not perform an exhaustive (all-1,100-shard) H2 re-verification —
  flagged explicitly as a spot-check limitation, not overstated as
  exhaustive.
