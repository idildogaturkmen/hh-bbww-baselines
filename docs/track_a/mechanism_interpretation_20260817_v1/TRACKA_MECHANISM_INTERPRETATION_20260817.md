# Track-A Mechanism Interpretation Freeze

Written from **already-measured artifacts only** -- `post_training_evaluation_20260815_v2/` (the
validated, corrected evaluation pass; `v1` is preserved unchanged but superseded by v2's own
self-documented arithmetic/labeling corrections, see `post_training_evaluation_20260815_v2/
CORRECTIONS_FROM_V1.md` and its 25/25-check, 0-mismatch `VALIDATION_REPORT.json`) and this
benchmark's frozen pre-training contracts and incident documentation. **No new training, no Track-B
event, no CMS collision data, no git write.**

## 0. Naming discipline (hard rule, restated)

The custom model trained in this benchmark is the **CMS-inspired five-jet pairwise event
Transformer** (`CMSInspiredFiveJetPairwiseEventTransformer`, `model_literature_freeze_20260813_v1/
models/cms_inspired_fivejet_pairwise_event_transformer.py`). It is **not** SPA-Net, **not** canonical
ParT, and **not** an official CMS reproduction -- its architecture matches only the *documented*
facts of the CMS HIG-24-010 public description (five leading b-tagged jets as tokens, pairwise
jet-pair features modifying self-attention, eight Transformer blocks, a separate 3-hidden-layer
event-feature encoder, a common 3-hidden-layer decoder); every other hyperparameter (hidden width,
head count, feedforward width, dropout, the exact pairwise-fusion arithmetic) is explicitly tagged
`PROJECT_CHOICE_NOT_CMS` in the model source and was never claimed to match CMS's actual,
undocumented choices. Its measured performance here is **not** described as CMS-equivalent anywhere
in this document.

## 1. What CMS HIG-24-010 reports, as a reference only -- not a target this benchmark attempts to hit

Per the frozen literature contract, the public CMS description reports **approximately two orders of
magnitude of additional background rejection at ~40% signal efficiency, and three orders at ~10%**,
for its full analysis chain. This number is **not commensurable** with anything measured in this
package: Track A differs from HIG-24-010 in simulation (Delphes/Pythia8 vs. full CMS
simulation+reconstruction), tagger (binary Delphes b-tag vs. CMS's continuous flavor taggers),
background model (a single inclusive-QCD Delphes sample vs. CMS's full multi-process,
data-driven-corrected background model), sample support (this benchmark's QCD statistics run out at
`qcd_Neff<10` well before reaching two-orders-of-magnitude rejection, section 4), model
implementation (an independently-implemented, undocumented-hyperparameter approximation, not CMS's
actual trained network), and physical normalization (this project's own Run-2-labeled 138 fb⁻¹
convention, not CMS's official luminosity/systematic treatment). **This benchmark does not attempt,
and this document does not claim, to reproduce or approach that CMS number.**

## 2. Measured comparison -- point estimates, ordering, and confidence

| Model | Status | AUC | $R_B(\epsilon_S{=}0.40)$ |
|---|---|---:|---:|
| Cut baseline (historical $R_{HH}<34$) | measured, single fixed operating point | -- | 1.9496 |
| BDT `CONTROL_0` (34-feat.) | measured (recomputed here, matches frozen ref. to 1e-11 rel.) | 0.7883 | 13.206 |
| Dense DNN (this benchmark) | measured, 5-fold OOF x 3 seeds | 0.7889 $\pm$ 0.0013 | 13.29 $\pm$ 0.26 |
| CMS-inspired transformer (this benchmark) | measured, 5-fold OOF x 3 seeds | 0.7983 $\pm$ 0.0031 | 14.00 $\pm$ 0.30 |
| `FiveJetParTNet` | cited, frozen Phase-3 reference, **not retrained here** | 0.7989 | 14.154 |
| BDT `NEW_C` (55-feat., +5th jet+pairwise) | measured (recomputed here, matches frozen ref. exactly) | 0.7992 | 14.707 |
| Categorized BDT | **UNAVAILABLE** -- see section 3 | -- | -- |
| LBN (Lorentz Boost Network) | **UNAVAILABLE** -- see section 3 | -- | -- |

Point-estimate ordering at $\epsilon_S{=}0.40$: $1.9496 \ll 13.206 < 13.293 < \mathbf{14.004} <
14.154 < 14.707$. The CMS-inspired transformer sits **above** `CONTROL_0` and the dense DNN, but
**below** both `FiveJetParTNet` and `NEW_C`. Only the two newly-trained models (DNN, transformer)
have a bootstrap confidence interval computed anywhere in this package; `CONTROL_0`, `NEW_C`, and
`FiveJetParTNet` are frozen point estimates with no interval of their own. The transformer's 68% CI
`[12.77, 14.92]` contains all three reference point estimates; the DNN's 68% CI `[12.26, 14.22]`
contains `CONTROL_0` and `FiveJetParTNet` but not `NEW_C`. This is a one-sided
interval-contains-point-estimate relationship, not a claim that the references have their own
overlapping interval. **No comparison to `CONTROL_0`/`NEW_C`/`FiveJetParTNet` is available at any
$\epsilon_S$ other than 0.40** -- that is the only point at which those three models' numbers were
computed in any frozen artifact this package draws from; this document does not reconstruct or
estimate their values at other grid points.

## 3. Unavailable comparisons -- reported as unavailable, not reconstructed

- **Categorized BDT**: three real candidates were found and individually audited (population/policy
  detail in `post_training_evaluation_20260815_v2/EVALUATION_RESULTS.json` ->
  `references.CATEGORIZED_BDT`). Two were contractually specified but **never trained**. One
  (`pn_c7w_nested_categorized_bdt_20260803_v1`) **was** trained and evaluated, but on a
  physical-significance-optimization population of ~31,225 rows, structurally different from this
  benchmark's 1,042,397-row TRAIN-only classification population -- excluded on
  population-comparability grounds, not performance grounds.
- **LBN (Lorentz Boost Network)**: contractually specified (`lbn_dnn`,
  `hh4b_train_fold_and_baseline_benchmark_contract_20260805_v1`, intending this same population/
  folds/weights) but **never trained** -- no checkpoint or output artifact exists anywhere in the
  project.

Both are marked `UNAVAILABLE` rather than estimated, extrapolated, or silently omitted.

## 4. Three distinct limitations, kept explicitly separate

**(a) Classifier-capacity limitation.** The dense DNN (14,849 parameters) underperforms both the
CMS-inspired transformer (426,697 parameters, ~29x larger) and `NEW_C` (which has explicit
engineered 5th-jet/pairwise features) at $\epsilon_S{=}0.40$. This is consistent with, but not proof
of, a capacity/inductive-bias gap -- the DNN was never given pairwise jet information at all (its
34-feature `CONTROL_0` input has no 5th jet, no explicit jet-pair invariant beyond the two Higgs
candidates' own masses), while the transformer's pairwise-attention-bias mechanism and the BDT's
engineered pairwise features both encode that information directly.

**(b) Finite-QCD-support limitation.** Both newly-trained models hit the **same** support floor at
the **same** working points (`qcd_Neff<10` starting at $\epsilon_B{=}0.001$, section 5) -- this
ceiling is a property of the shared underlying QCD Monte Carlo sample's finite raw statistics at
tight background efficiency, **not** a property of either classifier. A better classifier cannot
purchase a scientifically supportable rejection number beyond where the background sample itself runs
out of effective events; it can only reach a given $\epsilon_B$ target with a higher $\epsilon_S$ (a
genuine classifier improvement) or the same $\epsilon_S$ with a lower raw event count surviving (not
by itself informative about support unless `qcd_Neff` is also tracked, which it is here).

**(c) Representation limitation.** Track A's own b-tagging input is Delphes' binary b-tag flag, not
a continuous flavor-tagger score; its background model is a single inclusive-QCD Delphes sample, not
CMS's full data-driven-corrected multi-process background. The CMS-inspired transformer's
*architecture* (pairwise-attention-bias injection over five jet tokens) is present and functioning in
Track A -- its measured effect (section 2, section 6) is a real, internally-consistent mechanism
result -- but Track A **cannot test** whether *continuous* flavor information or a richer background
model would add further separation on top of that mechanism. That question is out of scope for Track
A by construction, not because of a flaw in this benchmark; it is the exact question named in
`TRACKA_TO_TRACKB_QUESTION_HANDOFF.md`.

## 5. Maximum empirically supportable background-rejection range

Full grid in `TRACKA_SUPPORTED_REJECTION_RANGE.tsv` (primary seed 20260811, both models). Support
gate (unchanged, project-standard): `raw_background_rows>=100 AND qcd_raw_rows>=10 AND qcd_Neff>=10`.

| Slicing | Deepest scientifically supportable point | $R_B$ at that point | `qcd_Neff` there |
|---|---|---:|---:|
| Fixed $\epsilon_B$ | $\epsilon_B{=}0.01$ (both models) | $\approx 100.0$ | DNN 41.7 / transformer 50.9 |
| Fixed $\epsilon_S$ | $\epsilon_S{=}0.10$ (both models, the loosest grid point tested at this tightness) | DNN 172.7 / transformer 167.2 | DNN 19.8 / transformer 28.3 |

**Beyond $\epsilon_B{=}0.001$ ($R_B\approx1000$), `qcd_Neff` falls below 10 for both models
(DNN 2.19, transformer 6.56) -- these numbers are computable but explicitly `scientifically_
supportable: False` and are not reported as physics claims.** The same is true, more severely, at
$\epsilon_B{=}0.0001$ ($R_B\approx10{,}000$, `qcd_Neff` 1.1-2.0). **A larger numerical tail score at
these unsupported points is not converted into a sensitivity or ranking claim anywhere in this
document** -- per the earlier paired-bootstrap table (section 6), no working point beyond
$\epsilon_S{=}0.10$ is even evaluated for the DNN-vs-transformer comparison, and no fixed-$\epsilon_B$
point beyond 0.01 is quoted as a supported rejection number for either model.

## 6. CMS-inspired transformer vs. DNN at fixed $\epsilon_S$ -- every measurable point

Joint (same-resampled-event-multiset), source-group-stratified bootstrap, 2000 replicates, primary
seed. $\Delta R_B = R_B^{\rm transformer} - R_B^{\rm DNN}$ (from
`post_training_evaluation_20260815_v2/RESULTS_SUMMARY.md` section 5, cross-checked in that
package's `VALIDATION_REPORT.json`):

| $\epsilon_S$ | $\Delta R_B$ | 68% CI | 95% CI | Frac. favoring transformer |
|---:|---:|---|---|---:|
| 0.60 | +0.309 | [0.150, 0.449] | [0.023, 0.609] | 0.985 |
| 0.585957 | +0.294 | [0.143, 0.458] | [0.017, 0.639] | 0.983 |
| 0.50 | +0.508 | [0.226, 0.790] | [-0.011, 1.100] | 0.972 |
| 0.40 | +0.652 | [0.114, 1.169] | [-0.321, 1.728] | 0.885 |
| 0.25 | +0.066 | [-2.026, 1.980] | [-4.067, 4.269] | 0.489 |
| 0.10 | **-5.550** | [-28.701, 12.878] | [-51.739, 32.789] | 0.357 |

**No single model dominates across the full tested range.** The transformer is favored with a
68%-CI-excluding-zero margin at $\epsilon_S \in \{0.60, 0.585957, 0.50\}$, favored only at the 68%
(not 95%) level at $\epsilon_S{=}0.40$, a statistical coin flip at $\epsilon_S{=}0.25$, and
**disfavored** (63.6% of replicates favor the DNN) at $\epsilon_S{=}0.10$ -- though that interval
still includes zero at both 68% and 95% confidence, so this is not itself a statistically significant
reversal, only a directional one. This ordering flip between $\epsilon_S{=}0.40$ and
$\epsilon_S{=}0.10$ was explicitly tested for, not discovered incidentally, and is reported as a
genuine working-point-dependent result rather than resolved in either model's favor overall.

## 7. What a failure to reach CMS-level sensitivity would NOT mean (and does not, here, since no such comparison was attempted)

Track A has never computed a CMS-comparable expected-signal-strength limit -- that remains blocked on
4b signal-region Monte Carlo statistics for reasons documented elsewhere in this project
(`track_a_post_pilot_statistical_adjudication_20260814_v2/`), unrelated to classifier choice. Stated
here as a standing caution for any future reader: **even if such a comparison were attempted and Track
A fell short of CMS's public expected limit, that alone would not diagnose a classifier failure.**
CMS's complete sensitivity result folds in trigger efficiency, flavor-tagging performance, mass
regression, a fully validated data-driven background model, signal-region categorization, and the
full systematic-uncertainty treatment -- none of which this benchmark's classifier-only,
Delphes-simulation, single-inclusive-QCD-background comparison attempts to reproduce. A rejection-only
mechanism study like this one is not positioned to make, and does not make, any claim about final
analysis sensitivity.

## 8. Bottom line

The CMS-inspired five-jet pairwise event Transformer's pairwise-attention mechanism produces a real,
measured, mostly-positive but working-point-dependent improvement over a plain dense DNN on Track A's
own TRAIN population, and sits between the `CONTROL_0` and `NEW_C`/`FiveJetParTNet` tabular/jet-level
baselines in point estimate. The comparison is honest about three separable limitations (classifier
capacity, finite QCD support, and representation richness) and about which comparisons simply do not
exist yet (categorized BDT, LBN) rather than substituting an estimate for either. The deepest
scientifically supportable background-rejection claim from this benchmark is
$R_B \approx 100$-$170$, not the $10^2$-$10^3$ CMS quotes for its full public analysis, and this gap
is not attributed to the classifier by this document.
