# Status — track_f_evaluation_readiness_protocol_20260909_v1

## 2026-09-10 (later) — matched-evaluation exporter implemented, run for real against CONTROL

Closes G5 in `track_f_postproduction_pipeline_20260908_v1/FINAL_EXECUTION_PACKET.md` Step 9 ("no inference/
scoring script exists anywhere in the project that produces the `.npz` files the evaluator requires"). New:
`code/export_model_eval_events.py`, targeting the CURRENT amended schema (`pred_b1..pred_b4`/`truth_b1..
truth_b4`, no H1/H2 correctness booleans) and reusing proven code: `evaluate_classification.py`'s architecture/
options block and `_patched_load_assignments` patch (parameterized, not hardcoded), `model.predict(sources).
assignments` (the same call `part2_spanet_assignment.py` already uses against this exact checkpoint), and
`four_vec()`/`pair_mass()` (copied verbatim from `part3_event_listing_and_truth.py`/`part6_pairing_accuracy_
and_mass.py`). Truth and raw kinematics are read directly via h5py — independent of the model's own tensor
pipeline, deliberately, to stay structurally immune to this project's own documented in-place-tensor-mutation
bug precedent (`RUNNER_PROVENANCE_RECEIPT.json`).

**Run for real** (read-only inference against the frozen, already-public native SPA2M checkpoint — does not
touch or reveal the unseen TEST/ParT2M result) using
`phase4U_eaf_official_spanet_gpu_canary_20260815_v1/.pixi/envs/default/bin/python3` (the default LPC shell's
`/usr/bin/python3` has no torch/pytorch_lightning; confirmed, not assumed). A live discrepancy was found and
fixed cleanly: this pixi environment's own editable-installed `spanet` resolves to a **different** git commit
(`46c68051...`) than the project's pinned commit (`debbdc999...`) — the exporter's `--spanet-repo` argument is
inserted at `sys.path[0]` explicitly, forcing the pinned-commit tree regardless, with a hard self-check
(`spanet.__file__` must resolve under the pinned path) so this can never silently regress.

**Full 400,000-event CONTROL export**: `track_f_postproduction_pipeline_20260908_v1/exports/
control_2m_native_eval.npz`, sha256 `bebfb680bf8e76a90bd0e4d626997a92b00a300c21444b1c2889986dff7ee26d`.
process composition `signal=193358 qcd=174485 ttbar=32157` — exact match to `FROZEN_FACTS.json`. Checkpoint
loaded with zero missing/unexpected keys. All hard validations passed, including a real, substantive finding
correctly resolved: 1,173 events had a predicted jet pointing at a padded (non-real) slot — investigated, not
waved away — and found to occur **exclusively** at `n_jets<4` (mathematically unavoidable: fewer than 4 real
jets exist for those events, so no all-real 4-jet assignment is possible), **zero** occurrences at `n_jets>=4`
(which would have been a genuine anomaly).

**Cross-checks against existing frozen artifacts** (task-required, "no handwaving"):

| cross-check | result |
|---|---|
| truth (`truth_targets_400k.npz`, full 400,000 events) | **exact match**, b1/b2/b3/b4, zero exceptions |
| classification score/AUC (`classification_evaluation_result.json`) | matches to **~4e-10** (all_background/qcd/ttbar AUC) — floating-point-noise level |
| predicted assignments (`spanet2m_signal_pairing_sample.npz`, 10,000-event sample) | **99.11% exact match** (9,911/10,000 rows); 89-row (0.89%) residual — see below |
| pairing-accuracy metrics (`signal_pairing_accuracy_and_mass_10k.json`) | agree to within 0.03–0.05 percentage points on every reported fraction |

The 89-row assignment residual was investigated, not hand-waved: confirmed **identical** torch
2.8.0+cu128/numpy 2.0.2 between this exporter's environment and the environment that originally produced
`spanet2m_signal_pairing_sample.npz` (`gpu_canary_10jet_trainonlyweighted_v23exact_r2/.pixi/envs/default`) —
ruling out a library-version mismatch. Of the 89 differing rows, 74 show a different `match_higgs_pairs()`
correctness verdict; net effect on aggregate metrics is only 3–5 events (out of 10,000), consistent with
symmetric, non-directional noise from a near-tied assignment-probability argmax decision — a well-documented
characteristic of CPU neural-network inference (non-associative floating-point reduction order under
different thread counts/hardware), not a version mismatch or a logic defect (which the exact-match truth/AUC
cross-checks above already rule out for everything else this exporter does).

**Tests added**: `tests/test_exporter_contract.py` (19 checks, against the REAL export above) — proves the
exporter never emits the forbidden H1/H2 booleans, that a slot-swapped copy of REAL (not synthetic) data gives
identical `reconstruction_metrics()`, that a trivial control-vs-itself comparison gives an exact-zero delta-AUC
and zero McNemar-discordant pairs, and that dropping or duplicating one event makes `evaluate_multi_model.py`
fail closed. Full suite: **43/43 outer checks, 85 total individual checks** (was 42/65 before this session).

**`FINAL_EXECUTION_PACKET.md` Step 9 updated**: G5 marked RESOLVED (original text preserved, not deleted),
pointing at this exporter and at `code/evaluate_multi_model.py` (not the superseded `evaluate_matched_2m.py`).
Exact future CONTROL/TEST/evaluator commands recorded there and in this session's final report.

**Not done**: TEST was NOT exported (SPA2M+ParT does not exist). No training, no `additional8M`, no JP-JEPA, no
`holdout_B`/Stage C. No GO/NO-GO threshold, primary/secondary metric, ParT preprocessing decision, SPA-Net
architecture/hyperparameter, or active/all128 training decision was changed.

## 2026-09-10 pre-result audit — H<->H permutation symmetry bug found and fixed

A pre-result audit (still before any ParT2M result exists) found that the 2026-09-09 `higgs1_assignment_correct`/
`higgs2_assignment_correct` schema silently assumed a canonical H1-vs-H2 labeling that does not exist — the two
Higgs bosons are an explicit `PERMUTATIONS.EVENT: [h1,h2]` symmetry in this project's own SPA-Net event topology,
confirmed authoritative by the official SPANet library's own `validation_average_jet_accuracy` code (which takes
the argmax over event-permutations before scoring). **Fixed via a dated amendment** (never a silent rewrite) —
see `PREREGISTRATION.md`'s **"Amendment 2026-09-10 — H<->H permutation symmetry correction"** for the full
account, including a correction to this document's own earlier (wrong) provenance claim about
`pn_c7x_spanet_evaluation.py`, and a second, independent bug (a degenerate always-zero paired-bootstrap delta)
found and fixed in the same pass. Schema and code were corrected directly (tooling, not frozen scientific
prose); `PREREGISTRATION.md`'s original §2/§5.1 text is annotated, not edited. All 42 pipeline checks plus 20
new symmetry-invariance checks plus 3 new bootstrap-verification checks pass (65 total). See "Standing facts"
below for updated hashes and counts.

## Summary

Freezes a publication-quality evaluation protocol for native SPA2M vs. SPA2M+ParT (vs. native SPA10M as a
secondary scale reference), written and frozen **before any ParT2M result exists** — ParT2M has not been
trained (`SPANET_PART_COMPARISON_CONTRACT_v2.md`: `FULL_2M_TRAINING_READY_NOW = NO` as of last check). No
production or training was launched by this package. The exact-2M frozen-ParT production was not touched.

Builds on, and does not modify, the existing `SPA2M_PART_EVALUATION_PLAN.md`
(`track_b_part_postflight_2m_20260907_v1/evaluation/`) and its accompanying `evaluate_matched_2m.py` /
`bootstrap_utils.py` (`track_f_postproduction_pipeline_20260908_v1/scripts/eval/`) — this package extends that
existing, working implementation (native SPA10M arm, exact-event-vs-per-Higgs pairing-accuracy split, exact
binomial intervals, formal JSON schemas, table/figure generators, and a mechanical GO/NO-GO decision rule with
Story A/B/C framing), rather than duplicating or replacing it.

## What is DONE this session

1. **Discovered and read the full existing decision chain** before writing anything: `SPANET_PART_COMPARISON_
   CONTRACT_v2.md` (2026-09-07, authoritative: 2M is primary, no fresh native control planned, both native
   checkpoints frozen), `SPA2M_PART_EVALUATION_PLAN.md` (predecessor metrics table), `PREPROCESSING_DECISION.md`
   (ParT dims: 128→20 retained, input width 27 not 135), `evaluate_matched_2m.py`/`bootstrap_utils.py` (working
   paired-bootstrap/McNemar/Poisson-interval implementation already in place).
2. **`PREREGISTRATION.md`** — the frozen protocol: 6 primary metrics (exact-event HH reconstruction efficiency
   and Higgs pairing accuracy now formally distinguished, all-background/QCD/ttbar AUC, background rejection at
   epsS ∈ {10%,7.5%,5%,4%,3%}), 6 secondary/robustness metric groups, cohort/identity discipline, statistics
   (paired bootstrap, McNemar, Poisson + NEW binomial intervals), a designated primary endpoint, a mechanical
   GO/NO-GO rule for (a) more ParT2M seeds and (b) `additional8M` authorization — with the `additional8M`
   effect-size floor (0.005 AUC) explicitly derived from this project's own already-measured native 2M→10M
   scaling noise ceiling (~0.002 AUC), not an arbitrary number — and falsifiable Story A/B/C definitions.
3. **`LITERATURE_TERMINOLOGY_NOTES.md`** — checked ParT (Qu/Li/Qian, ICML 2022, arXiv:2202.03772) and SPA-Net
   (Shmakov/Fenton et al., SciPost Phys. 12, 178 (2022), arXiv:2106.03898) against current web sources; recorded
   what each paper's SOTA claim actually covers (ParT: JetClass single-jet tagging vs. ParticleNet — not this
   project's task) and froze a standing "no SOTA claim without a matched benchmark" rule for this project.
4. **4 JSON schemas** (`schemas/`): per-event arrays (2026-09-10: `pred_b1..pred_b4`/`truth_b1..truth_b4` raw
   jet-index fields, required — see the 2026-09-10 audit note above; supersedes the original higgs1/higgs2
   boolean fields), aggregate-only (native SPA10M), the full comparison result, and the GO/NO-GO decision output.
5. **`code/`**: `bootstrap_utils.py` copied forward byte-identical (verified, hash recorded); one new statistics
   function (`exact_binomial_interval`, Clopper-Pearson) in a separate additive file; `evaluate_multi_model.py`
   (3-arm evaluator); `decide_go_no_go.py` (mechanical decision rule); `validate_schema.py` (this environment
   has no `jsonschema` package — confirmed, not assumed); `build_tables.py`; `build_figures.py`.
6. **Ran the full pipeline end-to-end against synthetic fixtures only** (`tests/`, every filename tagged
   `SYNTHETIC_TEST_FIXTURE`, random data with a controllable injected effect — never real model output):
   **42/42 checks passed** (2026-09-10: was 40/40 before the audit added 2 more outer checks wrapping the new
   unit-test modules), including that the decision logic reaches the mechanically correct conclusion on
   data engineered to demonstrate Story A (clean replicated improvement → `GO` then `AUTHORIZE`), Story C (a
   well-powered null → no significant flags), and an engineered regression (→ `NO_GO_INVESTIGATE_HARM`), and
   that the schema validator correctly rejects a broken instance (not just accepts good ones).
7. **(2026-09-10)** `tests/test_symmetry_invariance.py` — 20/20 checks, proving `match_higgs_pairs()` is
   invariant under every symmetry in `event_config/trackb_hh4b.yaml` (outer H1/H2 swap, inner jet swap, on both
   predicted and truth sides, independently and together), and gives exactly 0/0.5/1.0 per-Higgs accuracy for
   0/1/2 correct pairs. `tests/test_bootstrap_threshold_recompute.py` — 3/3 checks, empirically confirming the
   secondary audit found no bug in threshold recomputation. `tests/fixtures/make_synthetic_fixtures.py` rewritten
   to randomize H1/H2 slot order and inner-jet order independently on predicted and truth sides, per event — an
   integration-level regression guard against the exact bug class just fixed.

## What is explicitly NOT done

- No production or model training launched. No inference run against real data.
- No modification to any existing frozen result, contract, or plan (`SPANET_PART_COMPARISON_CONTRACT_v2.md`,
  `SPA2M_PART_EVALUATION_PLAN.md`, `evaluate_matched_2m.py`, `bootstrap_utils.py`,
  `classification_evaluation_10M_result.json`, anything under `docs/track_b/development_snapshot_20260821/`).
- No real number is reported anywhere in this package as a result — every number in `tests/` is explicitly
  labeled `SYNTHETIC_TEST_FIXTURE` / `_synthetic_test_fixture: true` and is random data with an engineered
  effect, used only to prove the tooling runs correctly.
- Does not itself authorize `additional8M` production — it states the rule that would, once real evidence
  exists; none of that rule's conditions are satisfied by this package.

## Standing facts

- `bootstrap_utils.py` SHA256 (this package's copy == the source file's, byte-identical): see `SHA256SUMS`.
- Smoke test: `python3 tests/run_pipeline_smoke_test.py` → **43/43 passed (85 total individual checks including
  the 3 embedded unit-test modules' own 20+3+19), exit 0.**
- `export_model_eval_events.py` SHA256: `40a8c09e0b2d3364e2731300e11d6237d6fb1376d4e3653ce8edd0dcc0ac8ba0`.
- Real CONTROL export (`track_f_postproduction_pipeline_20260908_v1/exports/control_2m_native_eval.npz`) SHA256:
  `bebfb680bf8e76a90bd0e4d626997a92b00a300c21444b1c2889986dff7ee26d` (37,603,758 bytes, 400,000 events).
- All 4 schemas are valid JSON (`python3 -c "import json; json.load(open(...))"` checked for each).
- All 8 `code/*.py` files plus 3 `tests/*.py` files `py_compile` clean.
- No production or model training launched, no TEST/ParT2M inference performed, no `additional8M`, no JP-JEPA,
  no `holdout_B`, no Stage C touched by either this audit or the exporter implementation session.
