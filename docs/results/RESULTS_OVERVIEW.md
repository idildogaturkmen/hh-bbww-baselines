# Results overview

The strongest, most defensible results across the full project, grounded in
committed frozen artifacts. Every quantitative claim below cites the exact
file it came from. Results are grouped by project stage (see
`docs/project_overview/PROJECT_STORY.md` for the narrative connecting them).
Nothing here is strengthened beyond what its source document states — where a
result is a train-only estimate, a projection, or blocked/incomplete, that
status is stated explicitly.

## 1. bbWW baseline (2026-05 → 2026-06)

**Source:** `docs/methods/ml_baselines/multiclass_dnn_comparison.md`.

A tabular multiclass DNN and four LBN-style variants were compared against a
corrected-recoMET BDT reference on the one-lepton HH→bbWW-like selection.

| Model | Test HH weighted AUC |
|---|---:|
| Tabular multiclass DNN | 0.611 |
| LBN v0 full | 0.625 |
| LBN v0 obj+aux no-pair | 0.618 |

The best LBN variant (obj+aux, no explicit pair features) reached a broad-region
test significance (Z ≈ 0.00664, background N_eff ≈ 60.1) comparable to, but
not clearly exceeding, the broad corrected-recoMET BDT reference (Z ≈ 0.00676,
N_eff ≈ 58.7). **Caveat, stated in the source document itself:** normalization
was still rough at this stage and no systematics or fit model were included —
these numbers are diagnostic, not final.

## 2. HH→4b cut-based baseline (2026-08)

**Source:** `HH4B_CUT_BASELINE_FINAL_STATUS.md` (root, frozen write target of
`scripts/analysis/build_hh4b_cut_baseline_final_report.py`).

Train-only performance (pooled 5-fold nested outer-OOF, Run-2 138 fb⁻¹-equivalent
projection, no systematics):

| Selection | εS | εB | S/B | stat-only Z_A |
|---|---:|---:|---:|---:|
| nested outer-OOF / combined | 0.6104 | 0.2855 | 3.81e-6 | 0.02698 |
| historical R_HH<34 / combined | 0.5904 | 0.2772 | 3.80e-6 | 0.02648 |

The frozen optimized cut shows **no material stat-only significance
improvement** over the historical simple cut (bootstrap median Z_A difference
spans zero: 95% interval [−8.19e-4, 6.18e-4]).

**Status: validation BLOCKED.** The one accepted validation campaign
(cluster `3795859`) failed at the infrastructure level (`ModuleNotFoundError`
in all 116 jobs) before any validation event was opened. Zero validation
payloads were opened; no validation performance number exists. This is
reported by the source document as a scientific non-result, not a negative
result — treat the cut baseline's out-of-sample performance as **not yet
measured**, not as poor.

## 3. HH→4b BDT baseline

**Source:** `docs/checkpoints/hh4b_bdt_model_choice_20260727_v1/README.md`
(not moved this stage — see §9).

| Model | Train-only weighted OOF AUC |
|---|---:|
| `global_v1_mass_aware` (primary) | 0.7737 |
| `categorized_cms_inspired_mass_aware` (secondary, low m_HH) | 0.7537 |
| `categorized_cms_inspired_mass_aware` (secondary, high m_HH) | 0.8053 |

The categorized secondary model's nominal-point improvement over the global
primary (bootstrap median difference −0.00048, 95% interval
[−0.0135, 0.0133]) **does not survive** a source-member bootstrap stability
check — the source document itself concludes the improvement is "not stable
under this diagnostic," and the simpler global model remains primary.

HH→4b dense-DNN and LBN-style comparisons follow the same protocol
(`scripts/train_tabular_dnn_baseline.py`, `scripts/train_lbn_fourvector_dnn_v0.py`,
`scripts/train_tabular_multiclass_dnn_cmsstyle.py`,
`docs/provenance/analysis_notes/hh4b_model_comparison_and_reporting_contract.md`)
but no equivalently frozen, dated HH→4b-specific DNN/LBN result document was
found during this stage's audit. **Do not cite bbWW-era DNN/LBN numbers
(§1) as HH→4b results** — they are a different channel and selection.

## 4. Physical normalization (2026-07-29 → 07-31)

**Source:** `docs/checkpoints/hh4b_physical_normalization_*` (41 dated
checkpoints, not moved this stage — see §9), summarized narratively in
`docs/provenance/analysis_notes/hh4b_physical_normalization_provenance.md`.

This is provenance/process work — establishing cross-section, luminosity, and
generator-event-weight conventions used to convert raw simulated yields into
Run-2 (138 fb⁻¹-equivalent) physical yields — rather than a single headline
number. It is the numerical foundation the cut-baseline (§2) and BDT (§3)
physical-yield tables above depend on, and should be cited as such in the
paper's methods section rather than as an independent result.

## 5. SPA-Net native scaling: 2M vs. 10M (2026-08-21, refined 2026-09-11, reconstruction added 2026-09-14)

**Source:** `artifacts/hh4b_spanet_part_20260911/README.md`, refining the
earlier development pass at
`docs/track_b/development_snapshot_20260821/spanet_10m_scaling/README.md`
(not moved this stage — see §9). The reconstruction column was computed
directly from the frozen `native10m_eval_400k.npz` export
(`artifacts/hh4b/pretrained_jet_representations/zero20_20260914/code/
compute_native10m_reconstruction.py`), hard-gated on exact reproduction of
the already-published Native 2M reconstruction numbers before being
trusted (confirmed, bit-for-bit).

| Model | All-background AUC | QCD AUC | ttbar AUC | Exact-event reco. | Per-Higgs reco. |
|---|---:|---:|---:|---:|---:|
| Native SPA-Net (2M) | 0.969281 | 0.968155 | 0.975392 | 0.866588 | 0.895299 |
| Native SPA-Net (10M) | 0.969272 | 0.967756 | 0.977502 | 0.872229* | 0.898838* |

Classification AUC has **saturated, not improved**, with 5x more training
data. Reconstruction shows a small **point-estimate** improvement
(+0.0056 exact-event, +0.0035 per-Higgs) — \*no paired-bootstrap CI was
computed for this delta (out of scope for a compact provenance addition),
so it is reported as a raw point estimate only, not claimed as
statistically significant. This is a matched, checksummed, hash-verified
result (`artifacts/hh4b_spanet_part_20260911/diagnosis/SHA256SUMS`;
Native10M reconstruction cross-check:
`artifacts/hh4b/pretrained_jet_representations/zero20_20260914/metrics/
native10m_reconstruction_crosscheck.json`).

## 6. QCD-tail / numerical-reliability studies (2026-09-06 → 09-07)

**Source:** `docs/checkpoints/track_b_harvey_final_question_closure_20260907_v1/`
and its predecessor packages (not moved this stage — see §9). All checks run
against the checksum-verified frozen SPA-Net 10M governing checkpoint.

- **Fifth-jet causal test (executed, not merely observed):** every one of
  88,854 signal events with ≥5 selected jets changes score when jets ranked
  ≥5 are masked out (0% unaffected); median Δu_logit = −0.090, mean = −0.148;
  58.1% of events are net-helped by the extra jet(s), 41.9% net-hurt.
- **QCD tail-rate stability (empirical, using existing production, not a
  projection):** stable across the two existing QCD lanes at u>3.5 (p=0.90);
  too few raw events (12) at u>4.5 to judge either way — stated as such, not
  forced to a verdict.
- **FP32 score-quantization spikes (u≈6.6, u≈6.9):** kinematically
  unremarkable — no special mode found; spike-bin events sit smoothly on the
  same kinematic trend as neighboring populated bins.

These results bound the numerical and statistical reliability of the
governing SPA-Net checkpoint's tail behavior, which the ParT active20 result
(§7) and the ZERO20 diagnostic (§8) build on.

## 7. Frozen ParT "active20" augmentation — a harm result, diagnosed (2026-09-11)

**Source:** `artifacts/hh4b_spanet_part_20260911/README.md` and
`artifacts/hh4b_spanet_part_20260911/diagnosis/ROOT_CAUSE_DIAGNOSIS.md`.

| Metric | Native SPA-Net (2M) | + ParT active20 (2M) | Δ (paired 95% CI) |
|---|---:|---:|---|
| All-background AUC | 0.969281 | 0.944222 | −0.0251 [−0.0255, −0.0246] |
| Exact-event HH reconstruction rate | 0.866588 | 0.496015 | −0.3706 [−0.3739, −0.3672] |

Appending 20 TRAIN-only, variance-selected, frozen ParT embedding dimensions
**significantly degraded** both classification and HH-reconstruction. A
read-only root-cause audit ruled out native-feature corruption, jet-slot
misalignment, preprocessing bugs, unintended hyperparameter/architecture
drift, and checkpoint corruption, narrowing the cause to (1) changed
optimization dynamics from the wider input embedding and (2) a variance-only
feature-selection rule that discarded several more discriminative embedding
dimensions than it kept.

**This is a matched development/validation result, not a final blind-test
result** — the source README states this explicitly. Treat it as informative
about mechanism, not as a final performance claim.

## 8. ZERO20 — a controlled width ablation, FINAL result (2026-09-14)

**Source:** `artifacts/hh4b/pretrained_jet_representations/zero20_20260914/`
(final, 10,000-replicate paired bootstrap both directions; see that
bundle's `README.md`, `metrics/`, and `tables/` for full statistics), and
`docs/studies/pretrained_jet_representations/zero20_width_control.md` for
the narrative.

The direct follow-up ablation to §7: 20 **zeroed** (not ParT-derived) extra
input dimensions, isolating "wider input embedding" from "ParT features
specifically" as the cause of the §7 harm result.

| Metric | Native (2M) | + ParT active20 (2M) | + ZERO20 (2M) | Δ ZERO20−Native (95% CI) |
|---|---:|---:|---:|---|
| All-background AUC | 0.969281 | 0.944222 | 0.969116 | −0.00017 [−0.00029, −0.00004] |
| Exact-event HH reconstruction | 0.866588 | 0.496015 | 0.866201 | −0.00039 [−0.00153, 0.00075] |

**ZERO20 is native-like, ParT active20 is not.** The AUC difference has a
95% CI that technically excludes zero — a large-cohort-size effect at
n=400,000 paired events — but is ~30× below this project's own
pre-registered 0.005-AUC practical-effect floor (§7.2 of the underlying
evaluation protocol, itself set from a previously measured ~0.002-AUC
native-scaling noise ceiling). Reconstruction is not statistically
resolvable at all (McNemar p=0.52). **Mechanically applying the
pre-registered causal-branch decision rule to these final numbers gives
Branch B on both metrics independently: this falsifies the pure-width
explanation for the §7 harm.** The cause is tied to the specific selected
ParT values and/or their preprocessing/optimization interaction, not
input width — see the source documents for the full reasoning and ranked,
unauthorized follow-up proposals
(`docs/studies/pretrained_jet_representations/next_experiments.md`).

A full ROC comparison of all four models on the exact common 400k cohort
(`.../zero20_20260914/plots/roc_all_background_four_models.svg`, with a
zoomed inset where Native 2M/10M/ZERO20 overlap) and a background-rejection
curve (`.../background_rejection_four_models.svg`, log-scale, stopping at
each model's own last real background survivor — no extrapolated tail)
make the same Branch B pattern visible directly, not just in tabulated
statistics.

**This is a matched development/validation result, not a final blind-test
result.**

## 9. Whole-project model landscape

Every model this project has trained, in one table, **with its dataset and
evaluation contract stated explicitly** — because the numbers are not on a
common scale and must not be read as a single ranking. In particular, the
bbWW/HH→4b cut-baseline/BDT rows report a **physical-yield-projected
significance** (Run 2, 138 fb⁻¹-equivalent, train-only, no systematics),
while the SPA-Net-era rows (native scaling, ParT active20, ZERO20) report
**raw AUC/reconstruction accuracy on a fixed, matched 2,000,000/400,000-event
development cohort** — a different quantity, on a different (though
overlapping-generation) simulated sample, under a contract with no
Run-2-luminosity projection at all. **No single bar chart correctly
represents both families at once** — this project deliberately does not
produce one; see the citation caveats already stated in §1 and §3 above,
which this table repeats for completeness rather than overriding.

| Model | Channel | Dataset / evaluation contract | Headline number |
|---|---|---|---|
| Tabular multiclass DNN | HH→bbWW | bbWW one-lepton selection, test region Z-significance, no systematics | Z≈0.0055 (§1) |
| LBN v0 (obj+aux, no-pair) | HH→bbWW | same as above | Z≈0.00664, N_eff≈60.1 (§1) |
| Corrected-recoMET BDT (reference) | HH→bbWW | same as above | Z≈0.00676, N_eff≈58.7 (§1) |
| HH→4b cut baseline (nested outer-OOF) | HH→4b | train-only, 5-fold pooled OOF, Run-2 138 fb⁻¹-eq. projected significance | Z_A=0.02698 (§2); **validation blocked, not measured** |
| HH→4b cut baseline (historical R_HH<34) | HH→4b | same contract as above | Z_A=0.02648 (§2) |
| HH→4b BDT `global_v1_mass_aware` (primary) | HH→4b | train-only weighted OOF AUC, same physical-yield convention | AUC=0.7737 (§3) |
| HH→4b BDT categorized (secondary) | HH→4b | same contract; bootstrap-unstable improvement, not adopted | AUC=0.7537 / 0.8053 by category (§3) |
| Native SPA-Net (2M) | HH→4b | matched 2M-train/400k-val cohort, raw AUC + exact-event reconstruction, no luminosity projection | AUC=0.969281, reco=0.866588 (§5, §7) |
| Native SPA-Net (10M) | HH→4b | same contract, 5× training data | AUC=0.969272 (§5) |
| SPA-Net + ParT active20 (2M) | HH→4b | same contract as native SPA-Net 2M | AUC=0.944222, reco=0.496015 — **harmed** (§7) |
| SPA-Net + ZERO20 (2M) | HH→4b | same contract as native SPA-Net 2M | AUC=0.969116, reco=0.866201 — **native-like** (§8) |

A separate, external, non-project reference significance point (retained
in the paper-preparation provenance as `EXTERNAL_REFERENCE_NOT_OUR_MODEL`,
never one of this project's own models) exists for cross-checking the
physical-yield background-budget methodology the cut/BDT rows above use;
it is not listed as a project result here and should not be cited as one.

## 10. A note on completeness

Three directories holding evidence cited above by their current paths —
`docs/checkpoints/` (BDT model choice, physical normalization, Harvey
reliability packages), `metadata/` (simulation-stage production bookkeeping),
and `Results Summaries/` (bbWW-era write-ups) — were candidates for
relocation into `docs/provenance/` in this reorganization stage but were
**not moved**, because a fresh dependency check found live script/config
consumers of those exact paths (see the root `README.md`'s provenance policy
section for specifics). The citations above point at their current,
unmoved locations and remain valid.
