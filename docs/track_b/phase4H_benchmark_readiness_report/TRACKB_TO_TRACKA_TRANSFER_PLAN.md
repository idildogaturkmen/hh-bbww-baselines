# Track B -> Track A Representation-Transfer Plan

Status: plan only. No step below has been executed. Track B events must never
enter Track A's physical evaluation at any stage of this sequence.

## Sequence

1. **High-statistics Track-B pairwise Transformer benchmark**
   Train and evaluate the genuinely pairwise-aware Transformer described in
   `TRACKB_CMS_TAIL_EXPERIMENT_PROPOSAL.md` on Track B's `training_qcd`
   development/holdout splits and `gghh_kl1` signal. Output: a frozen set of
   constituent-level embeddings / attention weights and a raw-count ROC/
   rejection characterization at Track-B statistics (tens of millions of QCD
   events), unavailable at Track-A's finite support.

2. **Native SPA-Net, Track A**
   Train SPA-Net natively on Track A's own finite-statistics HH->4b sample,
   using Track A's own truth jet-to-Higgs assignment labels (Track A's
   provenance is out of scope for this Track-B audit; its own readiness must
   be established independently). This step does not depend on Track B and
   proceeds on Track A's own timeline.

3. **SPA-Net + frozen ParT embeddings**
   Take the ParT-style constituent encoder trained in step 1 on Track-B
   statistics, freeze its weights, and use its embeddings as additional
   per-jet or per-particle features feeding Track A's SPA-Net. The rationale
   for pretraining the encoder on Track B is exactly the statistics gap
   documented in this audit: Track A's own sample cannot support learning a
   general-purpose particle/jet representation at the scale Track B's
   training_qcd (up to ~63M events, tight_exact4 alone ~103k raw events)
   provides.

4. **SPA-Net + frozen JP-JEPA embeddings**
   Same pattern as step 3, substituting a frozen JP-JEPA encoder (also
   pretrained at Track-B statistics) for the ParT encoder. Per
   `TRACKB_MODEL_INPUT_READINESS.tsv`, Track B's constituent branches are
   generically sufficient for this, contingent on independently confirming
   JP-JEPA's specific input contract (open item, not a Track-B blocker).

5. **Optional fine-tuning**
   If steps 3/4 show the frozen embeddings are informative but not optimal,
   a bounded, explicitly-labeled fine-tuning pass of the (previously frozen)
   encoder weights on Track A data only may be considered -- but this changes
   the encoder from "Track-B-pretrained, Track-A-frozen" to "Track-B-
   pretrained, Track-A-finetuned," and must be labeled as such in any
   downstream comparison table. This step requires its own separate
   authorization; it is not implied by authorization of steps 1-4.

6. **Track-A final physics comparison**
   The paper-facing BDT/DNN/SPA-Net/ParT/JP-JEPA comparison is performed
   entirely on Track A's own physically-normalized samples and yields. Track B
   never contributes events, weights, or yields to this step -- its role
   ends at step 4/5, contributing only pretrained/frozen representations.

## Isolation guarantees carried from this audit

- Track B events are never mixed into Track A's event samples or Run-2
  expected yields (unchanged from the Phase-4 handoff's non-negotiable
  constraint).
- Any embedding or representation crossing from Track B into Track A is
  frozen at the point of transfer (steps 3-4) unless a separately authorized
  fine-tuning pass (step 5) is explicitly labeled.
- Track B's unresolved training-QCD physical weight (Section 7 of
  `TRACKB_BENCHMARK_READINESS.md`) does not block steps 1-5: representation
  learning and raw-count benchmarking do not require physical normalization.
  It does block ever citing a Track-B yield or significance number inside the
  Track-A physics comparison in step 6.
- Track B's qualified statistical-independence finding (Section 6 of the
  readiness report -- author-documented at the generation-provenance level,
  not event-ID-verified) is sufficient justification for using training_qcd
  for representation pretraining, but is not sufficient to justify a
  quantitative closure claim; no such claim is made anywhere in this plan.
