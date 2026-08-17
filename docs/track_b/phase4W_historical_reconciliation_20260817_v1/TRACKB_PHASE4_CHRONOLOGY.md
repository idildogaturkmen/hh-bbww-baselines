# Track-B Phase-4 Chronology (Read-Only Historical Reconciliation)

Run 2026-08-17. **This phase opens no new data.** Every claim below traces
to an already-frozen artifact under
`/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/`,
re-read and re-checksummed (not re-derived) this phase. `holdout_A` was
already opened and spent by Phase-4M on 2026-08-13 -- this phase does not
reopen it, recompute it, or treat it as available. `holdout_B`,
`inference_qcd_1`, `inference_qcd_2` remain sealed/unopened (verified this
phase: no materialized `holdout_B` tensor directory exists anywhere in the
tree, only file-list seal/manifest artifacts from Phase-4I/4J). No model
was trained, retrained, or retuned by this phase. No split, feature,
threshold, or hyperparameter was changed.

## Trigger for this reconciliation

A prior request in this session asked for a *new* pre-holdout,
development-only E0/E1/E2 adjudication, to be written to a directory named
`phase4M_e0e1e2_development_adjudication_20260817_v1`, under the stated
assumption that `holdout_A` had not yet been opened. Before writing
anything, the existing Phase-4 root was inspected read-only. That
inspection found `phase4M_holdout_a_adjudication_20260813_v1` already on
disk, dated 2026-08-13, containing a complete, checksum-verified,
already-adjudicated `holdout_A` result -- and found the campaign had
continued for four more lettered phases (`N` through `V`) into SPA-Net/ParT
feasibility work and resource accounting, through 2026-08-15. Writing a new
"pre-holdout" adjudication under the letter `M` would have collided with
that existing, already-consumed phase and produced a false record
(asserting `holdout_A opened = false` when it is demonstrably `true`). That
request was not carried out. This document is the read-only reconciliation
requested instead.

## Chronology

See `TRACKB_PHASE4_CHRONOLOGY.tsv` for the full machine-readable table
(phase, date, purpose, data partitions accessed, models involved,
whether performance was evaluated, decision, artifact path, SHA
provenance status). Narrative summary by stage:

### Stage 1 -- Contract freeze (4I, 4J, 4K)

Population/support definition (4I), development/`holdout_B` file-split
contract with `holdout_B` sealed but not opened (4J), and the model
architecture + training + evaluation contract for E0/E1/E2 (4K). All three
verified this phase: `SHA256SUMS` present and 100% clean (24/24, 23/23,
15/15 files respectively).

### Stage 2 -- Development training and one-time holdout_A adjudication (4L, 4M)

4L trained E0/E1/E2, 3 seeds each (9 runs), on `internal_train`, evaluated
only on `internal_val` (development). E1 (kinematics + continuous Sophon
`probB/C/L`) was the strongest arm on every adequately-supported working
point; E2 (+pairwise attention bias) consistently underperformed E1 at
`eps_S>=0.25`. `SHA256SUMS`: 922/922 clean.

4M then opened `holdout_A` **exactly once**, reusing the 9 already-trained
4L checkpoints unmodified (checkpoint hashes and script hashes both
re-verified and locked in `PROTOCOL_LOCK/` before the sample was opened).
E1>E0 fully reproduced on `holdout_A` at every in-region working point.
E1>E2 partially reproduced: 2 of 3 seeds still favored E1 at
`eps_S=0.20`, but seed 0 flipped (`E2>E1`) exactly at the point where E2's
raw surviving-QCD count was 10 -- the boundary of the
"statistically limited but reportable" support tier. This was reported as
a genuine discrepancy, not suppressed or used to retune anything.
`holdout_A` is recorded as spent for arm/model adjudication going forward.
`SHA256SUMS`: 324/324 clean.

### Stage 3 -- Independent verification and transfer audit (4N, 4O)

4N drafted a LaTeX appendix from the already-frozen 4L/4M numbers (no new
data). 4O then independently re-derived all 36 (arm, `eps_S`, sample)
`R_B`/AUC figures directly from the raw per-seed `eval_*.json` files,
bypassing the pre-aggregated summary tables entirely, and found one
genuine arithmetic error in 4L's prose (`MECHANISM_ABLATION_RESULT.md`
stated "+33%" at `eps_S=0.40` for the E0->E1 improvement; the correct
figure, independently confirmed twice, is "+31%" / +30.9%). This was
corrected via a scoped erratum file (`PHASE4L_MECHANISM_ABLATION_ERRATUM.md`)
-- the original 4L file and its `SHA256SUMS` entry were left untouched, per
this project's freeze discipline. All other 4L/4M figures (the E1-vs-E2
percentages, the `eps_S=0.20` seed-0 flip, AUC values, support labels) were
independently reconfirmed exact. 4O also established that E1's mechanism
(continuous Sophon flavor input) **cannot transfer to Track A**, which has
only a binary `Jet.BTag` flag and no continuous jet-flavor discriminator
anywhere in its pipeline; the kinematic/pairwise-attention mechanism (E0,
and half of E2) does transfer directly. `SHA256SUMS`: 22/22 clean.

### Stage 4 -- SPA-Net/ParT feasibility and the official-baseline correction (4P, 4Q, 4R)

4P gated four candidate architectures on truth-matching and checkpoint
compatibility: native SPA-Net classification READY_NOW; native SPA-Net
assignment READY_AFTER_ADAPTER (with a ~48% event-level matchable-fraction
ceiling caveat, well below the published SPA-Net paper's 77-89% range);
ParT-embedding and SophonAK4-embedding arms both BLOCKED (domain
mismatch/unverified normalization for ParT; no confirmed embedding output
for SophonAK4).

4Q initially, incorrectly, treated its own custom
`SpaNetInspiredPairwiseSetTransformer` architecture as if it were "native
SPA-Net" and proposed launching it at full scale. This was caught and
corrected the same day: per an explicit governing naming requirement, the
paper's SPA-Net baseline must be the **official** `Alexanders101/SPANet`
codebase, not a same-session custom reimplementation. The custom model was
renamed and demoted to a separately-labeled engineering canary (used only
to demonstrate the assignment-label adapter trains stably end-to-end; 3
epochs, one seed, explicitly not a benchmark result). `holdout_Q` was
declared but never opened. `SHA256SUMS`: 43/43 clean.

4R froze a model registry (M01-M10) and a machine-readable result schema
for all future reporting, with no new data access. `SHA256SUMS`: 16/16
clean.

### Stage 5 -- Official SPA-Net integration and GPU readiness (4S, 4T, 4U, 4V)

4S installed the official `Alexanders101/SPANet` package (pinned commit,
isolated environment) and verified it against the already-open Phase-4Q
pilot data: event-topology symmetries, HDF5 conversion (byte-exact vs. the
official loader), and forward/backward/gradient-finiteness (all three of
single-head/two-head/strict-assignment-only) all passed via a direct-API
check on CPU. A CLI-level 1-epoch training canary did not complete in
either configuration within its timeout budget -- diagnosed as
Trainer/DataLoader/system-contention overhead, not a code or model defect;
the direct-API check already independently demonstrates correctness.
`SHA256SUMS`: 36/36 clean.

4T checked the GPU and ParT-compatibility gates for registry rows
M06/M07/M08: GPU gate BLOCKED for lack of any matching Condor GPU slot
(infrastructure, not a model finding); ParT gate BLOCKED (domain mismatch
+ unverified normalization, now with a pinned checkpoint hash). The
registry itself (`phase4R/MODEL_REGISTRY.tsv`) was left unmodified; this
was an addendum only. `SHA256SUMS`: 7/7 clean.

4U prepared (but, as of the most recent artifacts inspected this phase,
had not yet executed) a development-only, resource-instrumented GPU
canary for M06/M07 on a newly-available FNAL EAF A100 MIG slice, intended
to unblock 4T's Condor-specific GPU-gate finding via a different execution
venue. **No `SHA256SUMS` file exists in this directory**, and
`phase4V_compute_resource_accounting_20260815_v1/MODEL_RESOURCE_SUMMARY.csv`
explicitly states "no GPU job has executed for any model as of
2026-08-15." This reconciliation did not attempt to execute the prepared
EAF commands (that would be new computation, out of scope for a read-only
reconciliation) and did not find independent evidence in the tree that
they were run since.

4V built a forward-only compute/resource ledger from already-existing
machine-readable timing records (4L-4U), explicitly recording
"not_measured"/"N/A" rather than inventing figures for untracked jobs
(4A-4K never wrapped their jobs in a timing harness). **No `SHA256SUMS`
file exists in this directory either.**

## Provenance-status note on 4U/4V

Every other phase in this chronology (4I through 4T) carries a
`SHA256SUMS` manifest that this reconciliation independently re-verified
as 100% clean. `4U` and `4V` do not have one. This is disclosed as a gap
in this reconciliation's own provenance coverage, not asserted as a
problem with those phases' content (both are explicitly scoped,
in-progress/accounting-only phases whose own text already discloses their
incompleteness). Nothing in `4U`/`4V` was relied upon for any
performance claim in this document -- only for the "GPU canary not yet
executed" and "resource-ledger methodology" facts, both stated directly in
prose by those phases themselves.
