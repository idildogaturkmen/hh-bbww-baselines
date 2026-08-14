# Representation-Aware Learning for Resolved HH -> 4b Reconstruction

Status: working manuscript sections, 2026-08-14. This document contains only
simulation studies, public literature, and public-repository results. It contains
no CMS collision data, CMS-internal material, or claim of reproducing a CMS
analysis. Bracketed text marks results or details that must be frozen before a
submission draft is prepared.

## Candidate title

**Representation-Aware Symmetry-Preserving Learning for Resolved Higgs-Boson
Pair Production in the Four-B-Jet Final State**

## Abstract

The reconstruction of Higgs-boson pair production in the four-bottom-quark
final state combines a combinatorial jet-assignment problem with rejection of
an overwhelmingly large multijet background. We study which input
representations and relational mechanisms provide robust gains in simulated
resolved HH -> 4b events. Two complementary simulation samples are used. A
locally generated 5.2-million-event Delphes campaign supplies a broad,
process-diverse baseline and an independent test of statistical limitations. A
high-statistics external sample released with the jet-free HH -> 4b study of
Yang and Li supplies the background support needed for controlled rejection
measurements in a fixed four-tight-tag population.

In the external sample, adding continuous SophonAK4 B/C/L flavor information
to a kinematics-only event classifier increases QCD rejection at 40% signal
efficiency from \(283.1 \pm 27.2\) to \(370.6 \pm 26.8\) in development data
and from \(285.7 \pm 43.9\) to \(439.5 \pm 38.4\) in a previously untouched holdout. The
relative improvements are 30.9% and 53.8%, respectively, and the ordering
reproduces across all three seeds and all working points with adequate raw-QCD
support. A frozen pairwise-attention augmentation does not provide an
additional improvement in this configuration. These values are raw-count
classifier rejections conditional on the selected external-simulation
population, not physically normalized sensitivities or a reproduction of the
CMS signal region.

[Insert the frozen same-population DNN-versus-pairwise-transformer result and
the native-SPA-Net-versus-representation-enhanced-SPA-Net result here. The
abstract must not claim either comparison before its blind protocol is
complete.]

## 1. Introduction

Higgs-boson pair production provides direct sensitivity to the structure of
the Higgs potential and, in particular, to the trilinear Higgs self-coupling.
The HH -> 4b decay mode has the largest branching fraction among experimentally
accessible HH final states, but it also faces a severe QCD multijet background.
Modern resolved analyses therefore depend on three coupled capabilities:
identifying bottom-flavored jets, assigning four reconstructed jets to two
Higgs candidates, and separating the resulting event from background [1].

Machine-learning models approach these tasks with different inductive biases.
Boosted decision trees and dense neural networks can exploit compact,
physics-motivated variables efficiently. SPA-Net incorporates permutation and
resonance symmetries directly into set assignment [4,5]. Particle-cloud models
such as the Particle Transformer learn representations from jet constituents
[6], while the jet-free framework of Yang and Li performs event-level inference
from all reconstructed particles and provides continuous SophonAK4 flavor
outputs [2,3].

The motivating observation from the initial Delphes studies was that larger
neural architectures did not automatically outperform a mass-aware BDT. On the
early selected sample, the strongest neural SPA-Net configuration improved on
the first SPA-Net but remained below the BDT, and its tightest working region
was limited by a handful of simulated background events. The expanded
5.2-million-event campaign confirmed that high-score QCD-tail support, rather
than training-set size alone, is a central experimental constraint. This
motivates a high-statistics benchmark in which representation changes can be
tested without interpreting an empty or nearly empty background tail as a
physics sensitivity.

This study is organized around three questions:

1. Does continuous learned flavor information improve an otherwise fixed
   event classifier, and does the improvement reproduce on untouched data?
2. Does an explicitly pairwise relational mechanism improve over a dense
   baseline when the population, folds, features, seeds, and optimization
   protocol are held fixed?
3. Can a symmetry-preserving assignment network benefit from frozen learned
   jet representations relative to native jet inputs?

The first question has a completed frozen result. The second is undergoing a
source-grouped out-of-fold comparison. The third is the intended novel model
study and remains conditional on a domain-compatible, reproducible embedding
interface. The usefulness of Sophon information is already part of the Yang-Li
study; its role here is benchmark validation and experimental control, not a
claim of discovering the value of Sophon tagging.

## 2. Samples and Event Populations

### 2.1 Locally generated Delphes sample

The local simulation campaign contains 5,000,000 generated background events
and 200,000 generated signal events distributed across 630 source members.
The campaign uses process-dependent hard-scattering configurations: signal and
several background processes use MadGraph5_aMC@NLO, while dedicated hard-QCD
production uses Pythia settings. Events are showered and hadronized with
Pythia 8 and processed with a CMS-inspired Delphes fast simulation [7-9]. Jets
are reconstructed with FastJet using the anti-kT algorithm [10,11]. The
background inventory includes QCD multijet, top-quark, Z+heavy-flavor,
single-Higgs, and smaller electroweak processes. Signal samples include
gluon-fusion and vector-boson-fusion HH -> 4b production.

Source members, rather than individual reconstructed events, define the data
splits. The expanded manifest contains 464 training members, 121 validation
members, and 45 final-evaluation members. No source member crosses a split. The
final-evaluation population remains closed to models developed under this
protocol.

Earlier feasibility samples used small-radius jets with R=0.5. The production
workflow subsequently introduced a CMS-convention AK4/AK8 card with
small-radius R=0.4 jets and R=0.8 large-radius jets. The final paper must name
the exact card, source registry, reconstruction version, and hashes used by
each reported model rather than combining the two detector configurations.

Although the generated sample is sizeable for model development, successive
selection and high-score requirements leave sparse QCD support in the most
signal-like region. Consequently, this sample is used for process-diverse
baseline comparisons, transfer tests, and explicit demonstrations of
finite-simulation limitations. It is not used to extrapolate unsupported tail
efficiencies.

### 2.2 External high-statistics HH -> 4b sample

The high-statistics benchmark uses the external simulation released with the
calibratable jet-free HH -> 4b framework of Yang and Li [2,3]. The distributed
training lanes contain 62,941,859 QCD events, 69,750,093 variable-mass
h3 -> h1 h2 -> 4b signal events, and 11,599,999 top-pair events. Separate
inference-QCD productions contain 27,683,689 and 59,939,617 stored events; the
SM-like `gghh_kl1` signal lane contains 2,538,851 stored events.

The QCD sample is intentionally conditioned rather than inclusive. Its
generation and storage apply a hard-scattering threshold, generator-level
kinematic filtering, and a reconstructed four-jet trigger-like filter. Results
therefore describe this released phase space and must not be interpreted as an
inclusive QCD measurement.

The completed ablation uses `training_qcd` and `gghh_kl1` within a frozen
`tight_exact4` population. In this population, all four of the first four
selected jets satisfy `jet_sophonAK4_probB > 0.643`. The 400 training-QCD files
were deterministically divided by file identity into 240 development files,
80 `holdout_A` files, and 80 sealed `holdout_B` files. These contain 61,948,
20,501, and 20,772 `tight_exact4` QCD events, respectively. File hashes and
split disjointness were checked before training.

`holdout_A` was opened once after the model family, checkpoints, working-point
grid, support labels, and reproduction criterion were frozen. It is now spent
for that adjudication. `holdout_B` and both inference-QCD productions remain
unopened for the present model-selection sequence.

## 3. Models

### 3.1 Conventional baselines

Cut-based selections provide an interpretable reference. Gradient-boosted
decision trees use reconstructed Higgs-candidate masses, angular separations,
jet and event transverse momenta, and related topology variables. Dense neural
networks use a matched feature contract and provide a non-tree baseline. The
final comparison must distinguish mass-aware models from mass-plane-blind
diagnostics because explicit mass inputs can improve rejection while also
increasing the risk of mass sculpting.

### 3.2 Frozen flavor and pairwise-attention ablation

Three event classifiers isolate the effects of flavor information and pairwise
attention:

| Model | Inputs and mechanism |
|---|---|
| E0 | Event classifier using frozen jet and event kinematics only |
| E1 | E0 plus continuous per-jet SophonAK4 B/C/L flavor information |
| E2 | E1 plus the frozen pairwise-attention augmentation |

E1 tests the value of continuous learned flavor information beyond kinematics.
E2 tests whether the particular frozen pairwise mechanism adds information on
top of both kinematics and flavor. It does not test whether all possible
pairwise-attention architectures are ineffective.

### 3.3 Symmetry-preserving assignment

SPA-Net treats the selected jets as an unordered set and constructs assignments
while respecting the exchange symmetries of the two Higgs candidates and their
daughter jets. The native configuration receives reconstructed per-jet
features. The proposed representation-enhanced configuration replaces or
augments those features with frozen constituent-derived jet embeddings while
holding the assignment heads, training data, data splits, event budget, and
evaluation protocol fixed.

The embedding source must be frozen before this comparison is authorized. A
canonical JetClass-pretrained Particle Transformer is not assumed to be
domain-compatible: its large-radius, high-transverse-momentum pretraining
domain differs from the softer R=0.4 resolved jets used here. Acceptable final
options are a validated domain-matched checkpoint, a frozen AK4 embedding from
the released framework with clarified usage terms, or an explicitly labeled
in-domain pretraining study. An out-of-domain checkpoint may be retained as a
diagnostic, but not silently presented as the nominal representation model.

## 4. Evaluation Protocol

For the high-statistics raw-count benchmark, background rejection is

\[
R_B = \frac{1}{\epsilon_B},
\]

where the score threshold is selected at a fixed signal efficiency
`epsilon_S`. The predeclared grid contains signal-efficiency working points
from 0.60 to 0.20. A comparison is considered adequately supported only when
at least 10 raw QCD events survive for each relevant model and seed. Results at
or near this boundary are reported with an explicit finite-count warning.

Every learned-model comparison uses three fixed seeds. Development data may be
used for fitting and protocol decisions. A holdout is opened only once, after
the complete comparison is frozen. The model definition, features, source
groups, threshold rule, seed list, uncertainty estimator, failure policy, and
software hashes must be recorded before that opening.

For the local Delphes benchmark, source-grouped five-fold out-of-fold
predictions are pooled before comparing models. No event used to fit a model
contributes to that model's prediction sample. [Insert the exact uncertainty
construction from the frozen evaluation report.]

## 5. Frozen Flavor-Ablation Result

At `epsilon_S=0.40`, the frozen rejection measurements are:

| Evaluation sample | E0: kinematics | E1: + continuous B/C/L | E2: + pairwise attention | E1 relative to E0 |
|---|---:|---:|---:|---:|
| Development | \(283.1 \pm 27.2\) | \(370.6 \pm 26.8\) | \(216.3 \pm 18.0\) | +30.9% |
| Untouched `holdout_A` | \(285.7 \pm 43.9\) | \(439.5 \pm 38.4\) | \(244.6 \pm 19.3\) | +53.8% |

Continuous SophonAK4 B/C/L information therefore produces a substantial and
reproducible gain over kinematics alone. E1 exceeds E0 for every seed at every
adequately supported working point. The approximately 31% development gain
does not disappear on unseen data; the holdout gain is approximately 54%.

The E2 augmentation does not provide an additional improvement in this frozen
setup. E1 exceeds E2 for all seeds at `epsilon_S` values 0.60, 0.50, 0.40, and
0.25. At `epsilon_S=0.20`, one seed changes ordering exactly at the predeclared
raw-QCD support boundary of 10 events. This point is evidence of limited tail
resolution, not evidence for selecting or excluding a seed.

These rejection values are numerically in the several-hundred range, but the
comparison to modern experimental analyses is only one of scale. The benchmark
is conditional on `tight_exact4` in a filtered external simulation. It is not a
CMS SR4b reconstruction, a data-driven background estimate, or a significance
calculation. The physically meaningful conclusion is the controlled ordering
of the frozen representations within the stated population.

## 6. Architecture and Representation Results to Complete

### 6.1 Dense network versus pairwise event transformer

[Insert the pooled five-fold, three-seed comparison only after every completed
run has passed finite-output and fold-closure checks. Report performance at the
same supported working points, paired differences, seed stability, surviving
raw background, and whether the three-hour execution ceiling left any planned
run incomplete. Do not substitute a single best seed or a training metric.]

### 6.2 Native SPA-Net versus representation-enhanced SPA-Net

[Insert assignment purity, event-classification rejection, and data-efficiency
curves for native SPA-Net and the frozen representation-enhanced model. The
comparison must use identical eligible events and assignment labels. Report the
fraction of signal events excluded because four distinct selected jets cannot
be matched to the four Higgs daughter quarks. Freeze any mass- or boost-binned
eligibility study before using it to change the training population.]

### 6.3 Final blind evaluation

[Open only the sample named in the final frozen contract. Report model-family
adjudication separately from a physically normalized analysis. A normalized
yield or significance may be quoted only after the QCD weighting, control
region, and systematic-uncertainty contracts are closed.]

## 7. Discussion

The completed ablation separates two effects that are often combined in an
end-to-end classifier. Continuous learned flavor information is robustly useful
in both development and untouched data. In contrast, the tested event-level
pairwise augmentation does not improve on the flavor-enriched classifier. The
negative E2 result is scientifically useful because it constrains where the
observed gain originates: in this implementation, the dominant improvement is
the jet representation rather than the added relational block.

This conclusion also sharpens the proposed SPA-Net comparison. A convincing
novel result is not that Sophon scores help, which the source study already
motivates, but whether a domain-compatible frozen constituent representation
improves symmetry-preserving assignment, classification, or data efficiency
relative to native SPA-Net under a controlled protocol. A null result remains
publishable if the embedding interface is valid, the statistical support is
adequate, and the model comparison is frozen before evaluation.

The local and external samples answer different questions. The local Delphes
campaign provides process diversity, transparent generation provenance, and a
lower-statistics transfer environment. The Yang-Li sample provides the raw QCD
support needed to resolve efficiencies at the few-per-mille scale. Agreement
between them is not expected numerically because their generators, filters,
detector configurations, tagging definitions, and selected populations differ.

## 8. Limitations

1. All results in this draft use simulation. No CMS collision data are used.
2. The high-statistics rejection result is conditional on a filtered
   `tight_exact4` population and is not a CMS SR4b reproduction.
3. The training-QCD physical normalization for the external sample is not yet
   closed. Raw-count rejection is authorized; a yield or significance claim is
   not.
4. File-level split disjointness is verified for the external QCD samples, but
   event-level duplicate checks are impossible because distributed event
   identifiers are absent.
5. Tail results are limited by the raw number of surviving QCD events. The
   `epsilon_S=0.20` seed flip occurs at the minimum supported count.
6. Pretrained constituent encoders can suffer substantial jet-radius,
   transverse-momentum, detector, and preprocessing domain shifts. Compatibility
   must be established rather than inferred from architecture names.
7. The uncertainty construction associated with the displayed `\pm` values
   must be stated explicitly in the submission version from the frozen
   evaluator record.

## 9. Reproducibility and Data Availability

The analysis code, public-facing dataset contracts, split inventories, frozen
result note, and compact summary table are maintained in this repository. Large
event files, generated intermediates, credentials, and CMS-internal material
are not redistributed. The external simulation and its upstream analysis code
are attributed to Yang and Li. Use of released model artifacts remains subject
to clarification of the applicable license or author permission where the
upstream repository does not provide an explicit license.

The development result and one-time `holdout_A` adjudication are frozen.
`holdout_B`, `inference_qcd_1`, and `inference_qcd_2` remain unopened for the
present comparison sequence. Their status must be updated in the paper only
after a separately frozen and authorized final evaluation.

## 10. Submission Completion Checklist

- Replace every bracketed placeholder with a frozen result or remove the claim.
- State the exact uncertainty estimator for every interval or error bar.
- Add the final DNN-versus-transformer pooled out-of-fold table.
- Freeze the embedding source and domain-compatibility statement.
- Add native versus representation-enhanced SPA-Net assignment and
  classification results.
- Close or explicitly exclude physical normalization and systematic
  uncertainties.
- Verify every dataset count, split, seed, model hash, and support label against
  the archived execution reports.
- Confirm that the final public artifact contains no CMS-internal text, plots,
  paths, or collision-data products.

## Public References Used in This Draft

1. CMS Collaboration, "Improved results on Higgs boson pair production in the
   4b final state," CMS-HIG-24-010, arXiv:2604.27044,
   <https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-24-010/index.html>.
2. T. Yang and C. Li, "Potential of di-Higgs observation via a calibratable
   jet-free HH -> 4b framework," arXiv:2508.15048,
   <https://arxiv.org/abs/2508.15048>.
3. PKU-Hep-Group, `jetfree-hh4b` public analysis repository,
   <https://github.com/pku-hep-group/jetfree-hh4b>.
4. A. Shmakov et al., "SPANet: Generalized Permutationless Set Assignment for
   Particle Physics using Symmetry Preserving Attention," arXiv:2106.03898,
   <https://arxiv.org/abs/2106.03898>.
5. C.-W. Chiang et al., "Deep Learning to Improve the Sensitivity of Di-Higgs
   Searches in the 4b Channel," arXiv:2401.14198,
   <https://arxiv.org/abs/2401.14198>.
6. H. Qu, C. Li, and S. Qian, "Particle Transformer for Jet Tagging,"
   arXiv:2202.03772, <https://arxiv.org/abs/2202.03772>.
7. J. Alwall et al., "The automated computation of tree-level and
   next-to-leading order differential cross sections, and their matching to
   parton shower simulations," arXiv:1405.0301,
   <https://arxiv.org/abs/1405.0301>.
8. T. Sjostrand et al., "An Introduction to PYTHIA 8.2," arXiv:1410.3012,
   <https://arxiv.org/abs/1410.3012>.
9. J. de Favereau et al., "DELPHES 3: a modular framework for fast simulation
   of a generic collider experiment," arXiv:1307.6346,
   <https://arxiv.org/abs/1307.6346>.
10. M. Cacciari, G. P. Salam, and G. Soyez, "FastJet user manual,"
    arXiv:1111.6097, <https://arxiv.org/abs/1111.6097>.
11. M. Cacciari, G. P. Salam, and G. Soyez, "The anti-kT jet clustering
    algorithm," arXiv:0802.1189, <https://arxiv.org/abs/0802.1189>.

## Repository Evidence for Frozen Statements

- `docs/track_b/phase4L4M_frozen_result/TRACKB_PHASE4L4M_FROZEN_RESULT.md`
- `docs/track_b/phase4L4M_frozen_result/trackb_phase4l4m_eps040_summary.tsv`
- `docs/track_b/phase4H_benchmark_readiness_report/TRACKB_BENCHMARK_READINESS.md`
- `docs/checkpoints/hh4b_full_5m_background_and_signal_coverage_20260727_v1/README.md`
- `docs/checkpoints/hh4b_full_5m_200k_expanded_manifest_split_20260728_v1/README.md`
