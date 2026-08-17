# Current Next-Step Decision (Read-Only Reconciliation, Not a New Authorization)

This document states what the existing frozen record already implies is
the next valid step. It does not itself authorize, launch, or execute
anything -- it is a reading of `phase4Q`'s and `phase4S`'s own
already-written "single next authorized action" language, reconciled
against what `phase4U`/`4V` show has and has not happened since.

## What remains before the next scientifically valid *performance*
experiment

1. **A completed CLI-level 1-epoch official-SPA-Net training canary.**
   Phase-4S's direct-API check already proves the official model's
   forward/backward/gradient computation is correct; what has not yet
   succeeded is a real end-to-end CLI run under the official
   `Trainer`/`DataLoader` stack, which timed out twice on CPU under shared-
   machine contention. Phase-4Q/4S both state explicitly that scaling to
   the full 880-file development partition and the full
   `{0,1,2}`-seed/50-epoch/9-run official-SPA-Net benchmark should happen
   **only after** this canary passes.
2. **Execution of the already-prepared Phase-4U EAF GPU canary.** This is
   the concrete, already-written attempt to obtain that canary on a
   different execution venue (an FNAL EAF A100 MIG slice) after Phase-4T
   found no matching GPU resource in the Condor pool. As of the most
   recent artifact this reconciliation could inspect
   (`phase4V.../MODEL_RESOURCE_SUMMARY.csv`, 2026-08-15), this had **not
   yet been run** -- the scripts, environment bootstrap, and exact command
   list exist (`phase4U.../EAF_EXACT_COMMANDS.md`) but require a human to
   execute them interactively in the EAF Jupyter session; they are not
   something this read-only reconciliation phase can or should run.

Both M08 (ParT-embedding arm) and M09 (JP-JEPA-embedding arm) remain
independently `BLOCKED_EXTERNAL_ARTIFACT` for reasons unrelated to GPU
availability (checkpoint/domain-mismatch and no-public-artifact,
respectively) and are not affected by completing the GPU canary.

## Answering the 8 questions

1. **E0 -> E1**: consistent, reproducible improvement (development +22%
   to +31% to +61% in `R_B` across `eps_S=0.60/0.40/0.25`; fully
   reproduces on `holdout_A`, zero seed-level sign flips at any in-region
   working point).
2. **E1 -> E2**: consistent degradation on development (-31% to -42% to
   -38% across the same three working points, all seeds agree); partially
   reproduces on `holdout_A` (agrees at `eps_S>=0.25`; one seed flips at
   `eps_S=0.20`, exactly at E2's raw-QCD=10 support-tier boundary).
3. **Was the E2 pairwise benefit reproducible?** No -- there was no E2
   benefit to reproduce. E2 underperformed E1 on development, and that
   underperformance mostly reproduced on `holdout_A` (with one
   boundary-case exception, reported not suppressed). E2's tail numbers
   (`eps_S<=0.20`) are individually large in places but sit on 1-11 raw
   QCD events with 33%-64% relative seed spread -- numerically larger,
   not statistically distinguishable, and the frozen 4L document already
   says so explicitly.
4. **Empirically supported rejection range**: `eps_B=1e-2` (`R_B=100`) is
   well-supported (raw QCD>=100) for every arm/seed on development;
   `eps_B=1e-3` (`R_B=1000`) is reportable-but-limited; `eps_B=1e-4` is
   exploratory only. On `holdout_A`, `eps_S` down to `0.20` is in-region;
   `eps_S=0.10` is explicitly exploratory there too.
5. **Holdout_A adjudication conclusion**: E1 is the strongest arm, and its
   advantage over E0 is a robust, reproduced finding; E2 does not show a
   reproducible benefit over E1 and is not being carried forward as the
   preferred event-Transformer arm on this evidence. `holdout_A` is now
   spent for E0/E1/E2 adjudication.
6. **SPA-Net/ParT/constituent work already completed**: truth-matching and
   assignment-adapter readiness (100% truth ID, ~48% matchable ceiling,
   validated adapter); official SPA-Net package installed and
   CPU-integration-verified end to end (topology, HDF5, forward/backward/
   gradients); GPU execution blocked on Condor infrastructure, with an EAF
   GPU path prepared but not yet run; ParT and SophonAK4-embedding arms
   both independently blocked on artifact/compatibility grounds unrelated
   to GPU; JP-JEPA blocked on public-availability grounds. None of this
   used `holdout_A`, `holdout_B`, `holdout_Q`, or `inference_qcd`.
7. **What remains before the next valid performance experiment**: a
   completed CLI-level 1-epoch official-SPA-Net canary, most concretely
   via running the already-prepared but not-yet-executed Phase-4U EAF GPU
   commands.
8. **Is the next valid step (a) native SPA-Net training, (b) SPA-Net+ParT
   frozen-embedding training, (c) another technical readiness gate, (d)
   final holdout_B evaluation, or something else?**

   **(c), specifically: execute the already-prepared Phase-4U EAF GPU
   canary to obtain the one still-missing CLI-level 1-epoch completion.**
   Full native SPA-Net training (a) is the step immediately *after* that
   canary passes, per Phase-4Q's and Phase-4S's own explicit sequencing --
   it is not itself authorized yet, since the canary hasn't completed.
   (b) remains independently blocked (ParT checkpoint disqualified on
   domain/normalization grounds, unrelated to GPU access) and is not next
   regardless of GPU outcome. (d), opening `holdout_B`, is not next: this
   project's own stated rationale for holding `holdout_A`/`holdout_B`
   open ("we want the principal model families frozen before the
   one-time holdout adjudication") implies the SPA-Net family should
   reach the same frozen, adjudication-ready state E0/E1/E2 reached before
   `holdout_A` was opened -- and SPA-Net training (native, at minimum)
   has not happened yet at all, let alone been frozen. Opening `holdout_B`
   now would adjudicate an incomplete set of model families.

## Explicit non-recommendation

This document does not recommend *when* or *whether* to run the Phase-4U
EAF commands -- that remains a decision for whoever holds execution
authorization, informed by this reconciliation, exactly as
`DEVELOPMENT_MODEL_FREEZE_DECISION.md` (Phase-4L) declined to make the
E1-vs-E2 "which arm goes to holdout_A" call on its own behalf. No new
computation was run, and no holdout was opened, to produce this document.
