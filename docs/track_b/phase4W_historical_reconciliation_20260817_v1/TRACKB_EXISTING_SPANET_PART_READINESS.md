# Existing SPA-Net / ParT / Constituent Readiness (Read-Only Summary)

Transcribed from Phase-4P/4Q/4R/4S/4T/4U/4V. No new gate was run, no
new code was executed, no holdout was opened, by this reconciliation.

## Registry snapshot (M01-M10, per `phase4R/MODEL_REGISTRY.tsv`, as
amended in status only by 4T's addendum -- the registry file itself was
never edited after 4R)

| Model | What it is | Status (as of 2026-08-15) | Blocking reason |
|---|---|---|---|
| M06 | Official `Alexanders101/SPANet`, single-head | `TRAINING_NEEDED` | CPU integration fully verified (4S); GPU gate blocked on infrastructure (no Condor GPU slot, 4T); EAF GPU canary prepared but not yet executed (4U) |
| M07 | Official `Alexanders101/SPANet`, two-head | `TRAINING_NEEDED` | Same as M06 |
| M08 | Official SPA-Net + frozen canonical-ParT embeddings | `BLOCKED_EXTERNAL_ARTIFACT` | ParT checkpoint disqualified on two independent grounds: jet-radius/kinematic-regime mismatch (R=0.8, 500-1000 GeV vs. Track-B's R=0.4, pT>30 GeV) and unverified normalization constants |
| M09 | Official SPA-Net + frozen JP-JEPA embeddings | `BLOCKED_EXTERNAL_ARTIFACT` | JP-JEPA has no public code or weights; not evaluated in 4T (condition unmet) |
| M10 | Project's own `SpaNetInspiredPairwiseSetTransformer` | engineering canary only | Explicitly excluded from the "SPA-Net" baseline role by the naming-gate correction in 4Q; one completed 3-epoch pilot run exists (pipeline-stability demonstration only, not a benchmark result) |

## Truth/assignment readiness (Phase-4P, confirmed at scale by 4Q)

- Truth identifiability: **100%** clean on a canary (every event yields
  exactly 2 Higgs blocks x 2 direct b-quark daughters).
- Stage-1 four-unique-jet matchable ceiling: **~48%** (47.5%-48.9%,
  cross-validated on 426,921 signal events / 4 files in 4Q), well below
  the published SPA-Net paper's 77-89% plausibility range. This is a
  disclosed caveat that must travel with any future assignment-training
  result, not a blocker to building the adapter itself.
- Deterministic assignment adapter: **READY, VALIDATED** -- implemented
  and run on 1,008,376 real events (421,921 signal + 470,455 QCD +
  116,000 ttbar), 17/17 integrity checks pass (4Q).

## Official SPA-Net integration (Phase-4S, CPU)

- Package `Alexanders101/SPANet` (BSD-3-Clause), pinned commit
  `46c68051fbc6178b932b8eec562c792886769511`, installed in a new isolated
  environment (`/tmp/iturkmen_track_b_official_spanet_env_20260814_v1/`);
  neither of the project's two existing frozen environments was touched.
- Event topology (both Higgs-pair and within-Higgs b-jet exchange
  symmetries) verified via the package's own real YAML parser, not
  assumed.
- HDF5 conversion of the already-open, already-leakage-checked 8-file
  Phase-4Q pilot (1,008,376 events) verified byte-exact against the
  official `JetReconstructionDataset` loader (row counts, assignment-valid
  mask, classification labels, train/val disjointness, and event totals
  all matched exactly).
- Direct-API forward/backward/gradient-finiteness check: **passed
  completely** for single-head, two-head, and strict-assignment-only
  configurations (all losses finite, all gradients finite, 40-50s each on
  CPU).
- CLI-level 1-epoch training canary: **did not complete** in either
  single-head or two-head configuration within its timeout budget.
  Diagnosed cause: Lightning `Trainer`/`DataLoader` overhead compounded by
  disclosed shared-machine CPU contention (stray orphaned processes from
  earlier in the project's own session history, plus an unrelated
  concurrently-running Track-A job) -- not a code or model defect. The
  independent direct-API check already demonstrates the actual
  forward/backward/gradient computation is correct.

## GPU readiness (Phase-4T, 4U, 4V)

- Phase-4T: no GPU-capable Condor slot was found for this account, and no
  LPC-documented GPU submission convention exists -- **GPU gate BLOCKED
  on infrastructure**, explicitly not a finding about the model or the
  integration (already independently verified correct on CPU by 4S).
- Phase-4U: prepared (environment bootstrap, CUDA verification script,
  GPU-instrumented forward/backward canary, CLI canary scripts) a
  development-only GPU integration canary on a newly-available FNAL EAF
  A100 (`1g.10gb` MIG slice) session, intended to unblock 4T's
  Condor-specific finding via a different execution venue. **As of the
  most recent artifact this reconciliation could inspect (Phase-4V,
  2026-08-15), this canary had not yet been executed**:
  `phase4V.../MODEL_RESOURCE_SUMMARY.csv` states explicitly "no GPU job
  has executed for any model as of 2026-08-15." No `SHA256SUMS` exists in
  the 4U directory (consistent with an in-progress/unexecuted phase).
- Phase-4V: forward-only resource ledger; confirms the same "no GPU job
  executed yet" state and explicitly refuses to fabricate GPU-hour figures
  for it.

## ParT and JP-JEPA (unchanged, multiple phases)

- **ParT (canonical, pretrained checkpoint)**: BLOCKED. Two independent,
  unresolved reasons, confirmed again in 4T with a pinned checkpoint hash:
  domain mismatch (R=0.8, 500-1000 GeV boosted-jet tagger vs. Track-B's
  R=0.4, pT>30 GeV resolved b-jets) and unverified JetClass normalization
  constants. A from-scratch-trained ParT-*architecture* encoder on
  Track-B's own data remains structurally available as a fallback, but is
  explicitly a different, separately-named approach -- not a substitute
  for "frozen pretrained ParT."
- **SophonAK4 embeddings**: BLOCKED. Only the final softmax classification
  output is confirmed present in the bundled ONNX checkpoint; no
  intermediate/embedding output is confirmed extractable. A specific
  missing artifact was identified (an embedding-mode ONNX export, or the
  raw PyTorch checkpoint, neither of which exists in any location audited
  by this project) -- this path cannot proceed until one is obtained from
  the authors.
- **JP-JEPA**: no public code or weights exist; status unchanged since
  first flagged, `NOT_CURRENTLY_PUBLICLY_AVAILABLE`.

## Naming-gate restrictions currently in force (carried forward from 4Q,
extended by 4T)

No arm of the SPA-Net/ParT/JP-JEPA study may be called "frozen pretrained
ParT," "frozen pretrained SophonAK4 embeddings," or bare "SPA-Net"/"native
SPA-Net" unless it is genuinely the official, unmodified
`Alexanders101/SPANet` codebase or a genuinely-resolved checkpoint/adapter
chain. The project's own custom architecture (M10) must always carry its
own distinct name and must never be silently presented as the paper's
SPA-Net baseline.
