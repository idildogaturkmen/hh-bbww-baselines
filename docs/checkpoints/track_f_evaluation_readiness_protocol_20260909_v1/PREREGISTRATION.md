# Pre-registered evaluation protocol — native SPA2M vs SPA2M+ParT vs native SPA10M

**Status: FROZEN BEFORE RESULTS.** Written 2026-09-09, before SPA2M+ParT ("TEST") has been trained or scored.
No ParT2M number of any kind has been seen by whoever writes or approves this document. Every metric,
statistic, and decision threshold below is fixed now and must not be revised after TEST results exist, except
by an explicit, dated, publicly-visible AMENDMENT section appended (never edited in place) to this file, stating
what changed and why — exactly the discipline this project already applies everywhere else (e.g.
`hh4b_bdt_validation_protocol_amendment_20260727_v1`).

This document governs analysis only. It does not authorize, launch, or presuppose any production or training
run. `SPANET_PART_COMPARISON_CONTRACT_v2.md` (2026-09-07) remains the authoritative source for which models are
compared and under what fixed conditions; this document is strictly downstream of it and does not alter it.

## 0. The three model arms

| arm | role | status as of 2026-09-09 | events | checkpoint |
|---|---|---|---|---|
| **native SPA2M** | CONTROL (primary comparator) | frozen, already trained (epoch 49/50 selected) | 2,000,000 train / 400,000 val | `sha256=dc39cf76f0d8e40d07228f240fe58b1179e8134cd4f96c5c786a7d528d31ea8d` |
| **SPA2M+ParT** | TEST (primary subject) | **not yet trained** — gated on the 4-item gate in `SPANET_PART_COMPARISON_CONTRACT_v2.md` (`FULL_2M_TRAINING_READY_NOW = NO` as of last check) | same 2,000,000 / 400,000 | none yet; ParT checkpoint frozen at `sha256=61e752f80d7c237d4b18b97705df416a8518dd9e3d5a78a8bbdeebadd787fec0` |
| **native SPA10M** | SECONDARY comparator (scale reference for Story B only) | frozen, already trained and evaluated | 10,000,000 train / 400,000 val (**same val cohort as SPA2M**, per `evaluate_classification_10M.py`'s own docstring) | `sha256=fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5` |

SPA2M+ParT input width is **27** (7 native + 20 retained/standardized ParT dims out of 128), per
`PREPROCESSING_DECISION.md` — not the 128/135 figure used in earlier documents (that figure is superseded, not
edited, per project convention). The retained-dimension index list and exact mean/std constants are frozen in
*method* only (variance threshold `1e-3`, `eps=1e-3`) and remain provisional in *exact values* pending
full-population re-characterization (gate item 3). This protocol treats that as a precondition of TEST existing
at all, not something it re-litigates.

**No fresh native-only control is planned or needed** — both native arms already exist as frozen checkpoints
(`SPANET_PART_COMPARISON_CONTRACT_v2.md`, "No new native-only control is planned for either comparison").

## 1. Cohort and identity discipline

- All CONTROL vs TEST statistics are computed on the **identical 400,000-event validation cohort**
  (`production_2M_val.h5`, this project's "holdout_A"), matched **event-for-event by `event_id`
  (`native_hdf5_row_index`)**. Any script implementing this protocol must hard-fail (not warn) if the CONTROL
  and TEST event_id arrays, once sorted, are not byte-identical, or if their `process` (signal/qcd/ttbar) labels
  disagree for any shared event_id. This is already implemented and enforced in
  `evaluate_matched_2m.py` (`track_f_postproduction_pipeline_20260908_v1/scripts/eval/`) and is carried forward
  unchanged (see `code/README.md` in this package).
- native SPA10M's val cohort is the **same** 400,000 events (its own evaluation script docstring says so
  explicitly), which in principle permits *paired* statistics against it too — but only if a per-event score
  array (event_id-indexed) for native SPA10M is produced. As of this writing, only an **aggregate** result
  (`classification_evaluation_10M_result.json`: ROC AUC + working-point table, no per-event array) exists. This
  protocol therefore predeclares two paths for the native-SPA10M comparison (§4.2) rather than choosing after
  seeing which one looks more favorable.
- `holdout_B` / Stage C is never read by anything this protocol authorizes. Every number this protocol produces
  is a **matched development/validation result**, never labeled "final test performance."

## 2. PRIMARY metrics (predeclared, exhaustive — nothing added post hoc)

One designated **primary endpoint** (§3) drives the GO/NO-GO rule; all primary metrics below are still computed,
reported, and published in full regardless of which one is designated primary, so nothing is cherry-picked out
of the report — only the *decision rule* privileges one endpoint, to avoid a multiple-comparisons fishing
expedition at the decision-making step itself.

1. **Exact-event HH reconstruction efficiency** — fraction of matchable events (`assignment_defined = True`,
   i.e. truth Higgs→jet-pair assignment exists) for which the model's **full predicted assignment (both h1 AND
   h2) exactly matches truth, with no partial credit**. Equivalent to this project's established
   `exact_event_pairing_accuracy` convention (`pn_c7x_spanet_evaluation.py`, `ASSIGNMENT_METRICS`).
2. **Higgs assignment/pairing accuracy** — accuracy at the level of **individual Higgs candidates** (h1 and h2
   scored separately, 2 opportunities per matchable event, partial credit given). Equivalent to this project's
   established `per_higgs_pairing_accuracy` convention. **Distinct from #1** — a model can score higher on #2
   than #1 whenever it gets exactly one of the two Higgs candidates right more often than it gets both right
   simultaneously.

   > **AMENDMENT NOTICE (2026-09-10):** the "h1 and h2 scored separately" framing above, and both items' citations
   > to `pn_c7x_spanet_evaluation.py`, were corrected before any ParT2M result existed. Item 1 and item 2 remain
   > the intended metrics, but "h1"/"h2" are NOT independently addressable labels — see
   > **"Amendment 2026-09-10 — H<->H permutation symmetry correction"** under **## Amendments** at the very end of
   > this document for the corrected, symmetry-invariant definitions and the corrected provenance claim. The original
   > text above is left exactly as first written, not edited, per this document's own amendment discipline.
3. **All-background AUC** — ROC AUC, signal vs. (QCD ∪ ttbar) combined, on the full 400,000-event cohort.
4. **QCD AUC** — ROC AUC, signal vs. QCD only.
5. **ttbar AUC** — ROC AUC, signal vs. ttbar only.
6. **Background rejection at fixed signal efficiency**, `R = 1/epsB`, at **epsS ∈ {10%, 7.5%, 5%, 4%, 3%}**,
   each computed for all-background, QCD-only, and ttbar-only, using **per-model exact order-statistic
   thresholds** (each model's own score distribution supplies its own threshold at each target `epsS`; this is
   the established project convention, not a shared/control-derived threshold — see
   `rejection_at_fixed_efficiency()`).

Metrics #1–#2 require the input-array schema extension in §5.1 (the existing `evaluate_matched_2m.py` schema
carries only a single event-level `assignment_correct` flag, which alone gives #1 but not #2 — see
`schemas/model_eval_events.schema.json`).

## 3. Designated primary endpoint (drives §6's decision rule only)

**`delta_AUC_all_background`** — the paired-bootstrap difference (TEST − CONTROL) in all-background AUC, 95% CI,
`n_boot = 10,000`, `seed = 0` (fixed, not re-rolled). Chosen because (a) all-background AUC is this project's
own most-used top-line classification summary throughout Track B, and (b) unlike a single fixed-`epsS` rejection
number, it is not itself a member of a 5-point scan that would otherwise invite picking the most favorable
`epsS` after the fact.

**Co-primary confirmatory endpoints** (both must also be checked before declaring an improvement signal, per
§6 — this prevents a lucky AUC fluctuation alone from triggering a GO):

- `delta_rejection` at **epsS = 5%** (the middle rank of the 5-point scan), all-background, paired bootstrap,
  same `n_boot`/`seed`.
- McNemar exact test on paired exact-event reconstruction correctness (metric #1), α = 0.05.

## 4. SECONDARY / robustness metrics (predeclared)

1. **Mass resolution** — `std(higgs_mass_1)`, `std(higgs_mass_2)` under each model's own predicted assignment,
   signal events only, finite values only.
2. **Score-vs-mass correlation** — Pearson and Spearman correlation of the classification score against
   `higgs_mass_1`/`higgs_mass_2`, signal events only. A large positive correlation is a **diagnostic flag, not
   automatically disqualifying** — it is reported and discussed, not thresholded, because mass-adjacent scoring
   is physically expected to some degree in an HH→4b search; this protocol does not predeclare a mass-decorrelation
   requirement that was never part of the governing training contract.
3. **≤4 vs ≥5 jet-multiplicity strata** — AUC (all-background) computed separately for `n_jets ≤ 4` and
   `n_jets ≥ 5`, using the existing per-multiplicity stratification machinery
   (`stratify_by_jet_multiplicity()`), which already implements `eq4`, `eq5`, `ge5`, `ge6`; this protocol adds
   the coarser `le4` bucket explicitly (`n_jets ≤ 4`, which in this event topology is equivalent to `eq4` since
   HH→4b requires ≥4 real jets for a defined assignment — verified, not assumed, by a hard assertion in
   `code/evaluate_multi_model.py` that `n_jets.min() >= 4` whenever `assignment_defined`).
4. **Finer jet-multiplicity/topology strata** — the full `eq4/eq5/ge5/ge6` breakdown plus `extra_jet_activity`
   bins (`0,1,2,3+` selected jets beyond the 4 used in the Higgs pairs), reusing
   `PAIRING_BIN_CONTRACTS["extra_jet_activity"]`'s bin edges (`pn_c7x_spanet_evaluation.py`) for consistency
   with the rest of the project rather than inventing new bin edges.
5. **Raw tight-tail survivor counts** — for every `epsS`-scan working point (§2.6), the raw `n_bg_pass` count is
   reported as its own first-class output (not only the derived rejection ratio), together with the finite
   interval from §7.3/§7.4 whenever `n_bg_pass < 10` — this is the exact sparse-tail disclosure discipline
   already established project-wide (`finite_support_warnings`) and is mandatory, not optional, output.
6. **Runtime, peak GPU memory, parameter count** — `total_fit_wall_s`, `gpu_peak_reserved_bytes`,
   `gpu_peak_allocated_bytes`, `host_peak_rss_kib_self`, `gpu_device_name` (from each model's own training-result
   JSON), plus the architecturally-exact parameter-count delta: `ΔW = 16 × n_retained_part_dims` (first-`Source`-
   embedding-layer scalars only; `initial_embedding_dim = 16` is unchanged) — **not a gating criterion**, purely
   descriptive resource accounting, exactly as the predecessor plan already scopes it.

## 5. Data contracts

### 5.1 Per-event arrays (native SPA2M, SPA2M+ParT)

Extends `evaluate_matched_2m.py`'s existing `.npz` schema (unchanged fields kept; two NEW required fields added
to support primary metric #2 without which #1 and #2 would silently collapse to the same number). Formal schema:
`schemas/model_eval_events.schema.json`. New fields:

    higgs1_assignment_correct   bool   predicted jet-pair for Higgs candidate 1 matches truth (matchable events only)
    higgs2_assignment_correct   bool   predicted jet-pair for Higgs candidate 2 matches truth (matchable events only)

`assignment_correct` (existing field) MUST equal `higgs1_assignment_correct AND higgs2_assignment_correct`
whenever `assignment_defined` — enforced by hard assertion in `code/evaluate_multi_model.py`, not silently
trusted.

> **AMENDMENT NOTICE (2026-09-10):** the three fields described above (`assignment_correct`,
> `higgs1_assignment_correct`, `higgs2_assignment_correct`) are **superseded** — see
> **"Amendment 2026-09-10 — H<->H permutation symmetry correction"** under **## Amendments** at the very end of
> this document. `schemas/model_eval_events.schema.json`
> and `code/evaluate_multi_model.py` were both corrected before any ParT2M result existed; this section's original
> text is left unedited, per this document's own amendment discipline, but must not be used as the schema
> reference — use the schema file itself, which is current.

### 5.2 Aggregate-only result (native SPA10M, until/unless a per-event array is produced)

Formal schema: `schemas/model_eval_aggregate.schema.json`, matching the existing
`classification_evaluation_10M_result.json` structure (ROC AUC dict + working-point list with per-process
pass/total counts) — documented, not reinvented.

### 5.3 Comparison result / decision output

`schemas/comparison_result.schema.json` (the full multi-arm evaluation output) and
`schemas/go_no_go_decision.schema.json` (the frozen decision applied to it) — see §6, §8.

## 6. Statistics (implemented — reused, one function genuinely new)

All of the following are already implemented and were **not written for this task** except where marked NEW;
they are copied forward verbatim (hash-recorded, see `code/README.md`) from
`track_f_postproduction_pipeline_20260908_v1/scripts/eval/bootstrap_utils.py`:

- **Paired bootstrap delta-AUC / delta-rejection** — `paired_bootstrap_delta()`: resamples event indices once
  per replicate and applies the SAME resampled index set to both TEST and CONTROL scores before differencing,
  which is what makes it *paired* (controls for shared per-event variance) rather than two independent
  bootstraps subtracted. Returns observed delta, bootstrap mean/std, percentile CI, and an `excludes_zero` flag.
- **McNemar's test** — `mcnemar_test()`: exact two-sided binomial McNemar on paired event-level correctness
  (used for primary metric #1), preferred over the chi-square approximation precisely because reconstruction
  accuracy near ~0.49 with a modest discordant-pair count is exactly the low-count regime where the exact test
  is appropriate; the continuity-corrected chi-square statistic is still reported for cross-reference. Pure
  stdlib (`math.comb`), so it degrades gracefully in any environment.
- **Exact Poisson count interval** — `exact_poisson_count_interval()`: Garwood exact interval for a raw
  survivor count, used for §4.5's tight-tail counts (`n_pass < 10`). Falls back to a flagged normal
  approximation only if `scipy` is unavailable.
- **Exact binomial interval (NEW, additive)** — `exact_binomial_interval()` in
  `code/bootstrap_utils_ext.py` (new file; does not edit the existing `bootstrap_utils.py`). Clopper-Pearson
  exact interval for a *rate* (`k` successes of `n` trials — e.g. achieved `epsB` itself, not just the raw
  count), complementing the existing Poisson-on-the-count interval with the binomial-on-the-rate interval the
  task explicitly asks for. Uses `scipy.stats.beta` quantiles (`Beta(k, n-k+1)` / `Beta(k+1, n-k)` construction),
  with the same flagged normal-approximation fallback pattern as `exact_poisson_count_interval()` for
  environment robustness. **This is the one genuinely new statistical function in this package** — everything
  else is reuse.

## 7. GO/NO-GO decision rule — frozen, exact, evaluated by `code/decide_go_no_go.py`

Let (all from the primary/co-primary endpoints in §3, single-seed unless stated otherwise):

- `sig_class` = True iff the 95% CI of `delta_AUC_all_background` excludes zero **and** is positive.
- `sig_reco` = True iff (McNemar exact p < 0.05 **and** `test_better_than_control` True) **or** the 95% CI of
  paired-bootstrap `delta_exact_event_HH_reconstruction_efficiency` excludes zero and is positive.
- `harm_class` = True iff the 95% CI of `delta_AUC_all_background` excludes zero **and** is negative.
- `harm_reco` = True iff McNemar exact p < 0.05 **and** `test_better_than_control` False.
- `underpowered` = True iff **neither** `sig_class`/`harm_class` nor `sig_reco`/`harm_reco` resolved **and**
  the `delta_AUC_all_background` 95% CI half-width exceeds `2 × |observed delta|` (the CI is too wide relative
  to the point estimate to distinguish a null from an underpowered null).

### 7.1 First decision: more ParT2M seeds?

| condition | decision | rationale |
|---|---|---|
| `(sig_class OR sig_reco) AND NOT (harm_class OR harm_reco)` | **GO — run 2 additional seeds (3 total)** | a single seed cannot establish robustness; any real signal must survive seed-to-seed replication before it justifies further resource commitment |
| `underpowered` | **INCONCLUSIVE_UNDERPOWERED — run exactly 1 additional seed**, then re-apply this rule with the 2-seed pooled estimate; do not loop indefinitely | bounded, predeclared response to a genuinely ambiguous single-seed result — not an open-ended "keep trying until significant" license |
| `NOT sig_class AND NOT sig_reco AND NOT underpowered` | **NO-GO — Story C, controlled negative result; do not run further ParT2M seeds** | a well-powered null is a real, reportable finding, not a reason to keep sampling for a flip |
| `harm_class OR harm_reco` (regardless of any `sig_*`) | **NO-GO — investigate before any further seed**, report the harm finding explicitly | a statistically supported regression must be understood, not run past |

### 7.2 Second decision: authorize ParT10M `additional8M` production?

Only reachable after the multi-seed branch above completes (≥2 of 3 seeds evaluated). Let `n_agree` = number of
seeds (of the ≥2 additional run) whose own single-seed `sig_class OR sig_reco` (same direction, no `harm_*`)
matches the original seed's direction, and `median_delta_AUC` = median of `delta_AUC_all_background` point
estimates across all evaluated seeds.

**Authorize** iff `n_agree >= 2 (of 3 total seeds)` **AND** `median_delta_AUC >= 0.005`.

The `0.005` floor is not arbitrary: this project's own Aug-21 development snapshot
(`docs/track_b/development_snapshot_20260821/`) already established that **native** 2M→10M scaling alone moves
all-background classification AUC by an amount the project itself characterized as "essentially flat to
slightly down" (ttbar AUC moved ~0.002, the largest observed native-scaling shift). Authorizing a ~9.7-day,
8M-additional-event production (`SPANET_PART_COMPARISON_CONTRACT_v2.md`'s own resource estimate) requires the
ParT signal to be distinguishably larger than that already-measured native-scaling noise floor, not merely
nonzero. `0.005` is set at roughly 2.5× that floor.

**Do not authorize** otherwise — report the multi-seed result, the reason authorization was withheld
(disagreement vs. sub-floor magnitude, stated explicitly), and stop. This is a resource-aware, deliberately
conservative bar, consistent with this project's established resource-aware framing throughout the 2M-primary
decision.

## 8. Story definitions (falsifiable, evaluated mechanically, not narratively, by `code/decide_go_no_go.py`)

**Story A — genuine improvement over native SPA2M.**
Supported iff `sig_class OR sig_reco` (§7) is True at least once. Graded, not binary:
*suggestive* (true on the initial single seed only) → *confirmed* (`n_agree >= 2/3`, §7.2's replication bar,
regardless of magnitude) → *strong* (confirmed **and** `median_delta_AUC >= 0.005`, i.e. also clears the 10M
authorization floor).

**Story B — ParT2M approaches/beats native SPA10M (representation > scale).**
Evaluated against native SPA10M's all-background AUC (`auc_all_background_spanet10m`, from the aggregate
result). Two predeclared paths, chosen by data availability alone (not by which gives a nicer answer):
- **Paired path** (used only if a per-event, event_id-matched native-SPA10M score array exists): paired
  bootstrap `delta = AUC_ParT2M − AUC_native10M`; *"approaches"* = 95% CI includes zero; *"beats"* = 95% CI
  excludes zero and positive.
- **Unpaired fallback path** (used otherwise, and explicitly labeled `UNPAIRED_COMPARISON_WIDER_UNCERTAINTY`
  in every output that uses it): compare the ParT2M point estimate (with its own single-arm bootstrap CI,
  resampling ParT2M events only) against the native-SPA10M point estimate as reported in
  `classification_evaluation_10M_result.json`; *"approaches"* = ParT2M's point estimate is within its own CI
  half-width of native-SPA10M's point estimate; *"beats"* = ParT2M's CI lower bound exceeds native-SPA10M's
  point estimate.

**Story C — no gain; controlled negative result.**
Supported iff `NOT sig_class AND NOT sig_reco AND NOT underpowered` (§7.1's explicit NO-GO/well-powered-null
branch). When this is the outcome, the report must state, as **candidate, not proven, explanations** (per §9,
no causal claim beyond what the data shows): (i) domain mismatch — ParT's frozen representation was learned on
JetClass (generator-level simulated jets from `e+e-`/`pp` samples spanning top/W/Z/Higgs/QCD single-jet
tagging), never fine-tuned on this project's Delphes-simulated HH→4b sample or task; and/or (ii) the aggressive
108-of-128-dimension drop (`PREPROCESSING_DECISION.md`) may have discarded genuinely informative variation along
with the floating-point-noise dimensions it was designed to remove. Both are stated as open questions this
result cannot itself adjudicate, not as conclusions.

## 9. Literature/terminology guardrails (see `LITERATURE_TERMINOLOGY_NOTES.md` for full citations)

- ParT (Qu, Li, Qian, *Particle Transformer for Jet Tagging*, ICML 2022, arXiv:2202.03772) reported
  state-of-the-art performance **on the JetClass single-jet tagging benchmark, against ParticleNet** — a
  different task (single-jet multi-class tagging) on a different sample (JetClass) than this project's task
  (HH→4b multi-jet event reconstruction/classification on a Delphes hh4b sample). **This project's results,
  whatever they turn out to be, must never be described as inheriting or reproducing ParT's SOTA claim** — at
  most, they test whether ParT's *learned representation* transfers usefully as an auxiliary per-jet feature to
  a materially different downstream task and sample.
- SPANet (Shmakov, Fenton, Ho, Hsu, Whiteson, Baldi, SciPost Phys. 12, 178 (2022), arXiv:2106.03898; predecessor
  arXiv:2010.09206) is the **permutationless set-assignment / jet-reconstruction architecture** used throughout
  this project — a different role from ParT's (constituent-level per-jet classification/embedding). The two are
  not competing architectures being benchmarked against each other; ParT's frozen embeddings are used as an
  *input feature* to SPA-Net, not as a replacement for it.
- **No "state of the art" claim is permitted anywhere in this project's own SPA2M/SPA2M+ParT/SPA10M results**
  unless a *direct, comparable, published benchmark* on the *same task and sample family* supports it. None is
  known to this protocol's authors as of 2026-09-09. Any future draft asserting SOTA must cite the specific
  comparable benchmark or the claim must be removed before publication.

## 10. What this protocol explicitly does NOT do

- Does not launch, request, or presuppose any training, inference, or production run.
- Does not touch, re-derive, or override any number in any already-frozen result
  (`SPANET_PART_COMPARISON_CONTRACT_v2.md`, `classification_evaluation_10M_result.json`,
  `classification_evaluation_result.json`, or anything under `docs/track_b/development_snapshot_20260821/`).
- Does not authorize `additional8M` production by itself — §7.2 states the *rule* that would authorize it once
  real, multi-seed ParT2M evidence exists; this document alone satisfies none of that rule's conditions.
- Does not decide the 10M/ParT secondary comparison's own evaluation plan beyond referencing it as conditional
  in `SPANET_PART_COMPARISON_CONTRACT_v2.md`; SPA10M+ParT does not exist and is out of scope here.

## Amendments

### Amendment 2026-09-10 — H<->H permutation symmetry correction

**Status at the time of this amendment: SPA2M+ParT (TEST) had still not been trained or scored. No ParT2M
result of any kind had been seen by whoever wrote or approved this correction.** This is the last appropriate
moment to fix a genuine protocol bug through a dated amendment rather than silently editing the frozen original
text — accordingly, §2 items 1–2 and §5.1 above are left completely unedited (annotated with pointers to this
section only); this amendment is the sole place the correction is recorded.

**What was wrong.** §2 item 2 and §5.1 (2026-09-09 original text) described "Higgs candidate 1" and "Higgs
candidate 2" as if they were independently addressable, comparable labels — the per-event schema required
precomputed `higgs1_assignment_correct`/`higgs2_assignment_correct` booleans, implicitly assuming a canonical,
symmetry-safe H1-vs-H2 identity that does not exist. **The two Higgs bosons in HH→4b are identical particles.**
This project's own SPA-Net event topology
(`track_b_phase4_preflight_20260812/phase4S_official_spanet_integration_canary_20260814_v1/event_config/trackb_hh4b.yaml`)
declares this explicitly:

```
EVENT:
  h1: [b1, b2]
  h2: [b3, b4]
PERMUTATIONS:
    EVENT:
      - [ h1, h2 ]
    h1:
      - [ b1, b2 ]
    h2:
      - [ b3, b4 ]
```

`h1`/`h2` are an explicit, declared `PERMUTATIONS.EVENT` symmetry group (interchangeable), and each candidate's
two jets are a further, separately declared inner symmetry. Neither a trained model's output-slot order nor any
truth-storage convention is guaranteed to track a fixed physical H1-vs-H2 identity across events — this is
precisely what "symmetry preserving attention" (the network's own name) means: its training loss is already
symmetrized over exactly this permutation group, so the network has no incentive to learn, and no mechanism
that would force, a consistent slot-to-physical-Higgs mapping. Comparing "predicted h1" directly to "truth h1"
without resolving this would silently mis-score any event where the model's (equally valid) output happens to
land in the swapped slot order relative to however truth was stored for that event — biasing
`higgs_assignment_pairing_accuracy` downward in a slot-convention-dependent way that has nothing to do with the
model's actual reconstruction quality.

**Authoritative confirmation, not just a theoretical concern.** The official SPANet library's own validation
code, which computes `validation_average_jet_accuracy` — the exact metric this project already uses for
checkpoint selection throughout (`spanet/network/jet_reconstruction/jet_reconstruction_validation.py`, present
in this project's own `phase4Y_spanet_exact_v23_release_integration_20260817_v1/env/spanet_repo_v23exact/`) —
explicitly enumerates **every** event-level truth permutation, scores accuracy under each, and takes the
**argmax** before computing any downstream metric:

```python
for i, permutation in enumerate(event_permutation_group):
    ...
chosen_permutations = self.event_permutation_tensor[jet_accuracies.argmax(0)].T
```

This is exactly the "try all permutations, take the best" principle this correction applies — not an invented
convention, but the same one the SPA-Net authors themselves use for the metric this project already relies on.

**Corrected, symmetry-invariant definition** (implemented in `code/evaluate_multi_model.py`'s
`match_higgs_pairs()`):

    direct           = I(P1 = T1) + I(P2 = T2)
    swapped          = I(P1 = T2) + I(P2 = T1)
    n_correct_higgs  = max(direct, swapped)
    per_higgs_pairing_accuracy       = n_correct_higgs / 2      (primary metric #2)
    exact_event_hh_reconstruction_efficiency = (n_correct_higgs == 2)   (primary metric #1)

where each predicted/truth pair (`P1,P2,T1,T2`) is compared as an **unordered set of two jet indices** (never
an ordered tuple), so the inner `[b1,b2]`/`[b3,b4]` jet-exchange symmetry is handled by construction, and the
outer `[h1,h2]` exchange is handled by the direct-vs-swapped max. Proven, not just implemented: see
`tests/test_symmetry_invariance.py`, 20/20 checks passing, explicitly proving (task-required properties):
swapping predicted H1/H2 leaves both metrics unchanged; swapping truth H1/H2 leaves both metrics unchanged;
swapping the two jets inside any pair (predicted or truth side, independently or together) leaves both metrics
unchanged; one correct pair gives exactly 0.5 per-Higgs accuracy and `exact_event_correct = False`; both correct
gives 1.0 and `True`; neither correct gives 0.

**A second, independent, already-existing correct precedent was found — and the original provenance claim is
corrected.** §2's original text cited `pn_c7x_spanet_evaluation.py` as establishing the
`per_higgs_pairing_accuracy` convention this package claimed to be "equivalent to." Inspecting that file (as
instructed) confirms this citation was **wrong**:

```python
"learned_exact_event_pairing_accuracy": learned_accuracy,
"learned_per_higgs_pairing_accuracy": learned_accuracy,
"learned_per_jet_partner_accuracy": learned_accuracy,
```

All three metric names are literal aliases of the same variable (`learned_accuracy = ratio(learned_count,
match_count)`, itself derived from `matchable & (learned == truth)` on a **single discrete integer partition
label** — a different, simpler classification task, not SPA-Net's own per-particle jet-pair assignment output).
That script never computed a genuinely distinct partial-credit metric; it used one name three times. **The
"equivalent to this project's established convention" claim in the original §2 item 2 is withdrawn.** No such
convention was actually established by `pn_c7x_spanet_evaluation.py`.

However, a **different, already-existing, and correctly symmetry-safe implementation of exactly this metric**
was found during this audit:
`paper_exports/track_b_harvey_tail_characterization_20260825_v1/work/part6_pairing_accuracy_and_mass.py`
(written 2026-08-25, predating this package). It independently implements the identical principle — comparing
`{frozenset(pred pair A), frozenset(pred pair B)}` against `{frozenset(truth b1,b2), frozenset(truth b3,b4)}` as
2-element sets of frozensets, with `n_pairs_correct` (0/1/2) and `per_event_pair_correct_rate = n_pairs_correct
/ 2.0` — and even asserts the same `both_pairs_correct == full_correct` invariant this correction relies on.
`code/evaluate_multi_model.py`'s new `pred_b1..pred_b4`/`truth_b1..truth_b4` field names, and the -1 sentinel
for undefined truth, are deliberately adopted verbatim from this script (and its sibling
`part2_spanet_assignment.py`, which independently proves raw predicted-index extraction — `pred.assignments[0]`,
`pred.assignments[1]` — is already working, proven-feasible code against the real, frozen CONTROL checkpoint) —
**this is the corrected provenance citation**, replacing `pn_c7x_spanet_evaluation.py`.

**Schema and code changes made** (edited directly, not amendment-annotated — these are tooling artifacts of an
as-yet-unused, pre-result package, not scientific-decision prose; see task instructions' own distinction between
"append an amendment" for this document and "update" for schema/code):

- `schemas/model_eval_events.schema.json`: `assignment_correct`/`higgs1_assignment_correct`/
  `higgs2_assignment_correct` removed from the schema entirely (no longer read); `pred_b1,pred_b2,pred_b3,pred_b4`
  and `truth_b1,truth_b2,truth_b3,truth_b4` added as required raw jet-index fields.
- `code/evaluate_multi_model.py`: `load_model_eval()` now requires the raw index fields (hard-asserts
  `assignment_defined == (truth_b1 != -1)`, that truth pairs use 4 distinct jets wherever defined, and counts —
  without crashing on — any degenerate/repeated-jet predicted pair, matching `part6`'s own
  `n_degenerate_predicted_pairs` diagnostic). New `match_higgs_pairs()` computes `n_correct_higgs`/
  `exact_event_correct` centrally; `reconstruction_metrics()`, the McNemar call, and the paired-bootstrap
  exact-event-efficiency delta all consume its output instead of any precomputed boolean.
- **A second, independent bug was found and fixed while rewriting this code**: the pre-amendment
  `paired_bootstrap_delta_exact_event_efficiency` computation passed a `correct_arr` keyword that was fixed to
  TEST's own correctness array regardless of whether `paired_bootstrap_delta` was internally evaluating the
  "test" or "control" side of the paired difference (the closure ignored its own `scores` argument) — so the
  reported delta was always exactly `test_mean − test_mean = 0`, with a degenerate zero-width confidence
  interval, never actually comparing control to test. This silently meant one of the two co-primary
  confirmatory endpoints (§3) was completely non-functional (always reporting a fake, tight, uninformative zero)
  rather than simply absent — worse than a visible gap. Fixed as part of this same rewrite (the closure now uses
  its own `scores` argument, which `paired_bootstrap_delta` already passes correctly); verified non-degenerate:
  a synthetic replay gives `observed_delta=0.232`, 95% CI `(0.205, 0.256)`, `bootstrap_std_delta=0.013` — real
  variance, no longer a constant zero.
- `tests/fixtures/make_synthetic_fixtures.py`: regenerated to the new schema. Deliberately randomizes, per
  event and independently for predicted vs. truth, both the outer `[h1,h2]` slot assignment and each pair's
  inner jet order — so the fixture data itself carries **no** consistent slot convention, exactly mirroring the
  real symmetric SPA-Net output. This is a standing integration-level regression guard: had `match_higgs_pairs()`
  silently depended on a consistent convention, this fixture would have exposed it immediately.
- `tests/test_symmetry_invariance.py` (new): the 6 task-required unit-test properties, 20/20 passing.
- `tests/test_bootstrap_threshold_recompute.py` (new): the secondary-audit verification below, 3/3 passing.
- `tests/run_pipeline_smoke_test.py`: now runs both new test modules first; full suite **42/42 passing**.

**Secondary audit: paired-bootstrap threshold recomputation.** Verified, not assumed: `rejection_at_fixed_
efficiency()`'s signal-efficiency threshold is derived from its own `scores`/`labels` arguments on every call,
and every call site inside a bootstrap replicate loop (`paired_bootstrap_delta`'s `statistic_fn(scores, labels,
idx, ...)`) passes the **already-resampled** arrays (`scores[idx]`, `labels[idx]`, and — critically —
`process_full[idx]`) — so the threshold, the signal subsample it is computed from, and the background pass/fail
counts are all recomputed fresh on every replicate, never frozen from the nominal sample. Confirmed empirically
in `test_bootstrap_threshold_recompute.py`: resampling the same underlying scores 200 times yields 44 distinct
threshold values (std ≈ 0.052, not 0), and a full `paired_bootstrap_delta` run for rejection-at-`epsS=10%`
produces a nonzero `bootstrap_std_delta` (≈7.4). **Finding: no bug. No amendment to the statistics methodology
was needed here** — this is recorded as an audited-and-confirmed-correct item precisely so it is visibly
distinct from an unchecked assumption.

**Secondary audit: exporter readiness.** `part2_spanet_assignment.py` already proves predicted-index extraction
(`pred.assignments[0]`/`[1]` → `pred_b1..pred_b4`) is working, low-risk code against the real, frozen CONTROL
checkpoint **today**, with no dependency on TEST existing. However, **no concrete script currently exists
anywhere in this project that assembles a complete, schema-conformant `.npz`** (predicted indices + truth
indices + score + process + masses + n_jets, for CONTROL or TEST) — `part2`/`part3`/`part6` each compute pieces
of this in isolation, and `evaluate_matched_2m.py`'s own docstring already noted "this script does not launch
inference." **Recommendation, not performed by this amendment:** write and dry-run such an exporter against
CONTROL now (the checkpoint already exists; this requires no training and does not touch TEST), so that
schema-conformance is proven in practice before TEST exists, rather than the exporter being written and tuned
only after seeing preliminary TEST output — which would itself undermine the "frozen before results" discipline
this whole protocol is built on. This is a real, open dependency; it does not block this document's own
correctness, but it does gate when real evaluation can actually run.

**No GO/NO-GO threshold, effect-size floor, or story definition in §7–§8 was changed.** This amendment corrects
a metric-definition/implementation bug and a provenance citation; it does not touch, and was not motivated by,
any concern about the decision rule's conservatism.
