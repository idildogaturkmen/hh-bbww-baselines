# Track B and Track A integration plan

## Document status

This document defines the scientific and technical relationship between the
existing CMS-style HH→4b analysis, called **Track A**, and the external
variable-mass full-event representation study, called **Track B**.

Track B is designed to complement Track A. It does not replace the validated
Track A reconstruction, normalization, control-region, background, or
baseline-model program.

The associated released-model validation checkpoint is:

`docs/checkpoints/track_b_phase2_released_model_validation_20260801_v1`

## 1. Central scientific question

The central research question is:

> Can large-scale variable-mass four-b pretraining and full-event
> particle-level representations improve reconstruction, classification,
> sample efficiency, or robustness for fixed-mass Standard Model
> HH→bbbb analyses?

The study separates two hypotheses.

### 1.1 Representation hypothesis

A particle-level full-event representation may retain information lost when
an event is reduced to a small collection of reconstructed jets and
high-level variables.

### 1.2 Pretraining hypothesis

Learning a broad variable-mass two-resonance task may provide transferable
structure that improves fixed-mass HH performance, especially when Track A
training statistics are limited.

These hypotheses must be tested independently. A stronger architecture alone
does not establish a pretraining benefit. Pretrained and randomly initialized
versions of the same architecture must be compared.

## 2. Role of Track A

Track A is the primary CMS-style fixed-mass HH→4b analysis.

It supplies:

- fixed-mass Standard Model ggF and VBF HH samples;
- the approximately five-million-event background program;
- CMS-inspired resolved selections;
- three-b control and four-b signal-region studies;
- physical normalization and luminosity projections;
- cut-based, BDT, dense-DNN, LBN, and SPA-Net baselines;
- QCD-tail and rare-background treatment;
- detector-level Delphes reconstruction;
- analysis-level significance and mass-resolution metrics.

Track A remains the authoritative target analysis and the source of
physics-facing performance metrics.

Track A must remain independently usable even if Track B transfer produces
no improvement.

## 3. Role of Track B

Track B studies the external IHEP jet-free HH4b dataset and released models.

It supplies:

- a large variable-mass two-resonance signal grid;
- full-event particle inputs;
- a released three-model ONNX ensemble;
- 136 signal mass-grid classes;
- QCD and ttbar background classes;
- a mass-sensitive event representation;
- an official DCB-based mass-response procedure;
- a candidate pretrained representation for Track A.

Track B must first reproduce the public release faithfully. Transfer to
Track A begins only after unbiased evaluation on independent HH events.

## 4. Complementarity between the tracks

| Scientific question | Track A contribution | Track B contribution |
|---|---|---|
| Can HH be separated from realistic backgrounds? | Fixed-mass signal and full background mixture | Alternative particle-level representation |
| Can the Higgs pair be reconstructed? | Jet pairing and SPA-Net assignment | Two-dimensional mass-grid response and DCB peaks |
| Does more expressive input help? | Object-level and assignment-aware baselines | Full-event particle inputs |
| Does pretraining help? | Fixed-mass target task and from-scratch controls | Variable-mass pretrained initialization |
| Is the result robust? | QCD tails, process composition, and 3b/4b regions | Mass-domain and representation-domain tests |
| Does sensitivity improve? | Weighted yields and expected significance | Candidate representation evaluated on Track A |

Track B succeeds scientifically only when its contribution is evaluated on
the same Track A task, event splits, process mixture, and statistical budget.

## 5. Track B Phase 1 — dataset and model contract

Status: **complete**

Completed requirements:

- public repository and exact commit identified;
- exact ONNX input and output shapes identified;
- all 19 particle features reproduced;
- clipping, scaling, zero padding, masking, and truncation reproduced;
- 138 output classes identified;
- 136 signal mass-grid classes reconstructed;
- QCD and ttbar class indices identified;
- official inclusive signal discriminant identified;
- dataset schema and process conventions inspected;
- memory-safe temporary processing established.

## 6. Track B Phase 2 — released-model reproduction

Status: **control-sample validation complete; independent HH pending**

Completed:

- all three released ONNX models executed successfully;
- exact preprocessing contract reproduced;
- signal, QCD, and ttbar canaries passed;
- all 136 signal classes observed in a spread signal audit;
- training-to-inference QCD and ttbar behavior checked;
- fourteen minor-background processes audited;
- ZH and ZZ raw mass-grid localization demonstrated;
- official ZH DCB compatibility demonstrated;
- official ZZ DCB compatibility demonstrated;
- IHEP endpoint access validated;
- visible IHEP dataset namespace audited.

Remaining requirement:

- evaluate the released model on independent HH events not used in training.

This requirement is blocked only by the missing exact external sample path.

## 7. Independent-HH evaluation gate

Before matched-statistics training begins, the independent HH sample must
pass all of the following stages.

### 7.1 Access and schema

Verify:

- XRootD access;
- highest valid ROOT tree cycle;
- required branch intersection;
- `pass_selection`;
- `pass_4j3b_selection`;
- particle-vector consistency;
- finite preprocessing inputs;
- particle multiplicity and truncation rate.

### 7.2 Small inference canary

Run a spread 64-event sample through all three released models.

Require:

- output shape `[N, 138]`;
- finite predictions;
- softmax normalization;
- reproducible ensemble averaging;
- bounded memory;
- no persistent large output.

### 7.3 Spread pilot

Process at least 1,000 selected events distributed across the file or sample.

Record:

- mean signal, QCD, and ttbar probabilities;
- score quantiles;
- fixed signal-score efficiencies;
- model agreement;
- signal-grid marginals;
- leading mass-grid cells;
- particle truncation;
- 4j3b efficiency.

Sequential first-N sampling is not acceptable.

### 7.4 Official mass reconstruction

Construct a temporary official-format `Events` tree containing:

- `pass_selection`;
- `pass_4j3b_selection`;
- `score_0` through `score_137`.

Run the official DCB fitter and report:

- optimizer success;
- physical primary peak rate;
- valid non-sentinel secondary peak rate;
- physical secondary peak rate;
- localization near `(125,125)` GeV;
- sentinel and out-of-range rates;
- mass bias and resolution.

### 7.5 Independent discrimination

Measure:

- HH versus QCD ROC;
- HH versus ttbar ROC;
- HH versus combined-background ROC;
- fixed operating points;
- process-specific metrics;
- process-macro metrics;
- bootstrap uncertainties.

External training signal must not substitute for an independent signal test
sample.

## 8. Track B Phase 3 — matched-statistics model study

Status: **not started**

The purpose is to separate gains from architecture, representation,
pretraining, and training statistics.

### 8.1 Frozen event budgets

Models will be compared at several fixed training budgets:

- small-data regime;
- medium-data regime;
- full available Track A regime.

Exact event counts will be frozen before training.

### 8.2 Required models

At minimum compare:

1. Track A cut baseline;
2. Track A BDT;
3. Track A dense DNN;
4. Track A LBN;
5. Track A SPA-Net;
6. full-event architecture trained from scratch on Track A;
7. identical architecture initialized from Track B;
8. frozen Track B representation with a Track A head;
9. fine-tuned Track B representation;
10. optional SPA-Net and Track B hybrid.

### 8.3 Fair-comparison requirements

All comparisons must freeze:

- train, validation, and test identities;
- process composition;
- selection requirements;
- event budgets;
- random seeds or seed ensembles;
- optimization budgets;
- early-stopping rules;
- metric definitions;
- background aggregation conventions.

Parameter count, training time, and memory cost must be recorded.

A pretrained model must not be compared only against a weaker architecture.

## 9. Track B Phase 4 — transfer to Track A

Status: **not started**

Transfer proceeds from the least invasive test to the most adaptive.

### 9.1 Zero-shot application

Apply the released Track B model directly to compatible Track A particle
inputs without retraining.

Measure:

- Track A HH versus background separation;
- localization near `(125,125)` GeV;
- score calibration;
- domain shift between external and Track A samples.

Zero-shot failure may indicate domain mismatch rather than an unusable
representation.

### 9.2 Frozen representation

Freeze the pretrained representation and train only a lightweight Track A
classification or reconstruction head.

This tests whether useful HH information is already encoded.

### 9.3 Fine-tuning

Compare:

- full fine-tuning;
- partial layer unfreezing;
- discriminative learning rates;
- fine-tuning with classification only;
- fine-tuning with classification and mass objectives.

### 9.4 Random-initialization control

Train the identical architecture from random initialization on the same
Track A split.

The pretrained-versus-random difference is the primary measure of
pretraining benefit.

### 9.5 Hybrid assignment and classification

Candidate studies include:

- Track B embeddings concatenated with SPA-Net event features;
- pretrained particle embeddings supplied to an assignment head;
- joint classification and Higgs-assignment objectives;
- mass-grid probabilities used as auxiliary features;
- multi-task classification, assignment, and mass reconstruction.

## 10. Track B Phase 5 — domain robustness

Status: **not started**

### 10.1 Process-domain robustness

Evaluate separately on:

- QCD;
- ttbar;
- Z plus heavy flavor;
- single-Higgs processes;
- associated vector-boson production;
- diboson and triboson processes;
- top-associated rare backgrounds.

Report event-weighted and process-macro results separately.

### 10.2 Selection-domain robustness

Compare:

- loose preselection;
- three-b control region;
- four-b signal region;
- high-score tails;
- resolved categories;
- possible future boosted categories.

### 10.3 Mass-domain robustness

Evaluate:

- fixed `(125,125)` Standard Model HH;
- variable-mass external signals;
- ZH control events;
- ZZ control events;
- mass points near and far from the training-grid center.

### 10.4 Detector and reconstruction robustness

Study sensitivity to:

- particle multiplicity;
- 256-particle truncation;
- jet radius and object definitions;
- b-tagging working points;
- detector smearing;
- particle-category mismodeling;
- Track A versus external Delphes configurations.

### 10.5 Background-tail robustness

Track A QCD-tail studies remain authoritative for significance estimates.

Track B must be evaluated in the specific background regions that dominate
the final high-score statistical and systematic uncertainty.

## 11. Evaluation metrics

No single metric is sufficient.

### 11.1 Classification metrics

Report:

- AUROC;
- background rejection at fixed signal efficiency;
- signal efficiency at fixed background efficiency;
- precision-recall behavior;
- process-specific metrics;
- process-macro metrics;
- bootstrap uncertainty;
- score calibration.

### 11.2 Reconstruction metrics

Report:

- truth-assignment accuracy where available;
- signal-grid localization;
- DCB primary-fit success;
- physical peak rates;
- sentinel and out-of-range rates;
- reconstructed mass bias;
- reconstructed mass resolution;
- `(125,125)` localization on Track A HH.

### 11.3 Physics metrics

On Track A report:

- weighted signal yield;
- weighted background yield;
- signal-to-background ratio;
- Asimov significance;
- significance with background uncertainty;
- working-point stability;
- sensitivity to low-effective-statistics background tails.

Unweighted model metrics and physically weighted physics metrics must remain
separate.

## 12. External weighting policy

The two external `gen_weight` entries remain semantically unresolved.

Until authoritative definitions are received:

- released-model evaluation remains unweighted;
- external weights are not used for physical yields;
- ROC and response measurements are labeled unweighted;
- Track A normalization does not inherit an assumed external convention.

After clarification, both unweighted and weighted results will be retained.

## 13. Track A comparison table

The final study should contain at least:

| Method | Input | Pretraining | Assignment | Classification | Mass response | Track A significance |
|---|---|---|---|---|---|---|
| Cut baseline | high-level objects | no | analytic | yes | yes | yes |
| BDT | high-level variables | no | fixed | yes | indirect | yes |
| Dense DNN | high-level variables | no | fixed | yes | indirect | yes |
| LBN | object four-vectors | no | fixed | yes | indirect | yes |
| SPA-Net | jets or objects | no | learned | yes | yes | yes |
| Full-event scratch | particles | no | optional | yes | yes | yes |
| Track B frozen | particles | yes | lightweight | yes | yes | yes |
| Track B fine-tuned | particles | yes | optional | yes | yes | yes |
| Hybrid SPA-Net | particles and objects | yes | learned | yes | yes | yes |

Comparisons must state when methods receive different information.

## 14. Scientific decision gates

### Gate A — released-model reproduction

Pass condition:

- exact public model and preprocessing execute successfully;
- expected signal and background behavior is reproduced.

Status: **passed**

### Gate B — unbiased external HH evaluation

Pass condition:

- independent HH events demonstrate meaningful classification and mass
  response without training-sample reuse.

Status: **blocked on external path**

### Gate C — matched-statistics representation study

Pass condition:

- full-event and Track A models are compared on identical data budgets and
  frozen splits.

Status: **not started**

### Gate D — pretraining benefit

Pass condition:

- pretrained initialization outperforms the identical randomly initialized
  architecture with uncertainty quantified.

Status: **not started**

### Gate E — Track A physics benefit

Pass condition:

- any improvement survives realistic Track A backgrounds, normalization,
  background-tail limitations, and uncertainty treatment;
- at least one physics-facing metric improves.

Status: **not started**

## 15. Interpretation rules

The study must not claim:

- pretraining benefit from architecture-only comparisons;
- unbiased signal performance from external training signal;
- physical yields using undefined external weights;
- improved significance from AUROC alone;
- robust performance from inclusive QCD and ttbar alone;
- successful reconstruction from optimizer convergence alone.

Negative transfer is a valid scientific result.

## 16. Reproducibility and storage rules

- Pin every external repository commit.
- Record model hashes, configurations, and sample paths.
- Record deterministic or seeded sampling.
- Keep train, validation, and test identities disjoint.
- Use `/tmp` for large external-file canaries.
- Store only compact metadata during exploratory validation.
- Do not alter Track A data under
  `/uscms_data/d3/iturkmen/hh4b_delphes`.
- Do not modify certificate identity or `~/.globus`.
- Do not begin a full persistent external-dataset transfer until
  `eosquota` visibly reports 5 TB.

## 17. Immediate implementation sequence

While waiting for the independent HH path:

1. publish this integration plan;
2. implement a reusable preprocessing contract;
3. add tests for features, padding, masking, truncation, and class mapping;
4. implement a configuration-driven remote-file canary;
5. implement ensemble-score summaries;
6. implement temporary official-format ROOT output;
7. implement DCB-result validation;
8. leave independent-signal execution disabled until a valid path is
   supplied.

After the path arrives:

1. run the independent-HH access and schema canary;
2. run the 64-event inference canary;
3. run the spread pilot;
4. run official DCB reconstruction;
5. run independent ROC evaluation;
6. close Phase 2;
7. freeze Phase 3 matched-statistics contracts.

## 18. Final research outcome

The completed program answers three progressively stronger questions:

1. **Reproduction:** Can the external released model be faithfully
   reproduced?
2. **Transfer:** Does its representation or pretraining improve fixed-mass
   Track A performance?
3. **Physics impact:** Does the improvement survive realistic backgrounds
   and increase HH→4b sensitivity?

Track A supplies the realistic target analysis and physics interpretation.
Track B supplies the alternative representation and pretraining hypothesis.
Their combination forms a controlled transfer-learning study rather than two
unrelated analyses.
