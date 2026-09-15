# 2026-08-17: Current Status -- Track A / Track B Mechanism Results

Read-only synthesis of already-frozen Track-A and Track-B artifacts. No model
was trained, retrained, or evaluated to produce this entry. Full detail,
figures, and tables: `docs/paper/mechanism_results_20260817/`. Full
Track-B phase-by-phase chronology: `docs/track_b/phase4W_historical_reconciliation_20260817_v1/`.
Track-A mechanism interpretation: `docs/track_a/mechanism_interpretation_20260817_v1/`.

## Track A

- Completed: dense DNN and CMS-inspired five-jet pairwise event Transformer,
  5-fold OOF $\times$ 3-seed benchmark.
- Authoritative evaluation: `post_training_evaluation_20260815_v2` (**not**
  `v1` -- v2 documents and independently re-verifies a corrected ordering,
  a corrected cut-baseline metric identity, and a corrected paired-CI
  claim relative to v1; 25/25 independent checks pass).
- Track-A supported rejection: **$R_B \approx 100$-$170$** (the fixed-$\epsilon_S$
  grid, 0.10-0.60, is fully within the support gate for both models, all 3
  seeds; the fixed-$\epsilon_B$ tail fails the gate at $\epsilon_B \leq 10^{-3}$).
- The pairwise-attention Transformer gives a **modest, working-point-dependent
  gain** over the DNN: favored at loose-to-moderate $\epsilon_S$ (0.50-0.60),
  a coin flip at 0.25, disfavored at 0.10 -- not a uniform win.
- Continuous flavor information is **unavailable in Track A** (Delphes binary
  b-tag flag only; no continuous jet-flavor discriminator materialized
  anywhere in the current Track-A pipeline).
- Finite QCD support blocks any $R_B \sim 1000$-class claim on this sample.

## Track B

- `holdout_A`: **consumed** (opened exactly once, Phase-4M, 2026-08-13; not
  reopened by any artifact referenced here).
- `holdout_B`: **sealed** (no materialized tensor/event directory found
  anywhere in the Track-B tree).
- **E1 (continuous-flavor) result fully reproduces** on `holdout_A`: E1 > E0
  at every in-region working point, all 3 seeds, no exceptions.
- **E2 (added pairwise-attention-bias) does not improve over E1** -- a
  reproducible degradation at `eps_S >= 0.25`; one boundary-case seed
  discrepancy at `eps_S=0.20`, reported not suppressed.
- Official SPA-Net (`Alexanders101/SPANet`) integration is **CPU-validated**
  end to end (event topology, HDF5 conversion, forward/backward/gradient
  finiteness all confirmed via the real training-step code).
- The Phase-4U EAF GPU canary is **prepared but not yet executed** (as of
  the most recent inspected artifact, Phase-4V, 2026-08-15).
- Native official SPA-Net **full-scale training has not yet been run**.
- Frozen pretrained canonical ParT remains **blocked**: checkpoint
  domain/kinematic-regime mismatch ($R{=}0.8$, 500-1000 GeV vs. Track-B's
  $R{=}0.4$, $p_T{>}30$ GeV) and unverified JetClass normalization constants.
- JP-JEPA: author response received; pretrained weights/inference code may
  become available shortly (not yet in hand as of this entry).

## Next

- Execute the existing, already-prepared Phase-4U EAF GPU canary.
- If it passes, freeze/launch official SPA-Net development-only training.
- Integrate representation encoders (ParT, SophonAK4-embedding, JP-JEPA if/
  when available) only under their own separately-frozen contracts.
- `holdout_A` remains spent; `holdout_B` remains sealed until the principal
  model families (including SPA-Net) reach the same frozen,
  adjudication-ready state E0/E1/E2 reached before `holdout_A` was opened.
