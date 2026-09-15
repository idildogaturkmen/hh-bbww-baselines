# Track B: IHEP jet-free HH4b reproduction and transfer master status

Date: 2026-08-01

## 1. Executive status

Track B studies the released variable-mass, all-particle
HH4b framework associated with:

- paper: `arXiv:2508.15048`
- public repository: `pku-hep-group/jetfree-hh4b`
- pinned repository commit:
  `e8387f7e49a7f27c65b2f4a1f75cacf01cd45392`

The central research question is:

> Can large-scale variable-mass four-b pretraining and
> full-event particle representations improve fixed-mass
> Standard Model HH to 4b reconstruction and classification
> when transferred to the existing Track A samples?

The intended Track B phases are:

1. dataset and released-model contract;
2. released-model reproduction;
3. matched-statistics model studies;
4. transfer to Track A;
5. detector-domain and robustness studies.

Current status:

- released data layout: audited;
- representative remote files: inspected;
- released model contract: audited;
- released model contract: audited;
- released three-model inference: reproduced;
- resonance-localization canaries: performed;
- Track A raw-EFlow transfer canary: performed;
- Track A adapter preprocessing parity: passed;
- exact matched-card signal comparison: blocked only by
  the unavailable IHEP pileup artifact;
- larger deterministic raw-EFlow characterization:
  ready after correcting a stale selection-count assertion.

## 2. Supersession and interpretation notice

This document supersedes any earlier statement that the
Track A raw-EFlow adapter reproduces the official IHEP
particle representation exactly.

The official IHEP representation is based on:

`RunPUPPI/PuppiParticles -> ParticleFlowCandidate`

Track A contains only:

- `EFlowTrack`
- `EFlowPhoton`
- `EFlowNeutralHadron`

It does not contain:

- `ParticleFlowCandidate`;
- a candidate-level PUPPI collection;
- an equivalent PUPPI-named candidate branch.

Therefore, the existing Track A inference is correctly
described as:

> raw-EFlow-to-PUPPI-model domain transfer

It is not an exact IHEP detector/input reproduction.

This correction is scientifically important. The observed
weak signal-versus-QCD transfer and high reconstructed
signal-mass response cannot presently be interpreted as an
intrinsic failure of the released model.

## 3. Relationship between Track A and Track B

### Track A

Track A is the locally produced fixed-mass Standard Model
HH to 4b analysis and detector-simulation campaign.

Protected storage root:

`/uscms_data/d3/iturkmen/hh4b_delphes`

Representative files used in Track B transfer tests:

| Process | File |
|---|---|
| Signal | `/uscms_data/d3/iturkmen/hh4b_delphes/root/ggf_hh4b_ak4ak8_10k_pythia8_delphes.root` |
| QCD | `/uscms_data/d3/iturkmen/hh4b_delphes/root/qcd_bbbb_iht600plus_20000_pythia8_delphes.root` |
| Top | `/uscms_data/d3/iturkmen/hh4b_delphes/root/ttbar_200k_shard000_pythia8_delphes.root` |

SHA-256 receipts:

| Process | SHA-256 |
|---|---|
| Signal | `10f7c11b558482a945d3b566b44190ffd6a78e7bfb50aef44f45ca5e88c1571c` |
| QCD | `c89dc72a52be4f47def38534b7dcd63cb3c024f372bf85e59190371b941e3459` |
| Top | `c74ad8091105a4d122c02fe92b385451b445449f7fd203dc974b08127988adeb` |

A preserved signal generator-level source also exists:

`/uscms_data/d3/iturkmen/hh4b_delphes/hepmc/ggf_hh4b_ak4ak8_10k_pythia8.hepmc`

This HepMC source is the basis of the planned
same-generator-event matched-card comparison.

### Track B

Track B uses the released IHEP variable-mass jet-free
dataset and ensemble models to study:

- direct released-model reproduction;
- mass-grid response;
- signal-versus-background discrimination;
- transfer to fixed-mass SM HH;
- sensitivity to particle representation;
- eventual pretraining and fine-tuning studies.

## 4. The 144-million-event training dataset

### 4.1 Published event composition

The released paper describes a training dataset with
approximately 144 million selected events:

| Component | Raw generation | Selected training events | Training role |
|---|---:|---:|---|
| Variable-mass signal | 500 million matrix-element events | 70 million | 136 signal mass classes |
| QCD multijet | 200 billion unfiltered events | 63 million | QCD background class |
| Top-pair | 40 million events | 12 million | top background class |
| Total | — | 144 million as reported | 138-class training |

The rounded component values sum arithmetically to
145 million. The paper nevertheless reports the complete
sample as `1.44 x 10^8`, or 144 million. This document
preserves the authors' reported total and treats the
component values as rounded figures.

### 4.2 Signal generation

The signal process is:

`h3 -> h1 h2 -> b bbar b bbar`

within a variable-mass two-Higgs-doublet-model setup.

The two daughter-scalar masses cover:

`40 GeV <= m_h1,m_h2 <= 200 GeV`

in 10 GeV intervals.

The nominal grid has:

`16 x 16 = 256`

ordered mass combinations.

Exchange symmetry,

`(m_h1,m_h2) <-> (m_h2,m_h1)`,

reduces the distinct signal classes to the upper or lower
triangular set:

`16 x 17 / 2 = 136`

The 70-million-event signal sample is retained after the
union of the released `4j3b` and `4j2b` trigger paths.

### 4.3 QCD generation

The QCD sample required approximately:

`2 x 10^11`

unfiltered events.

Dedicated generator-level preselection and Delphes-level
filtering were used to populate the relevant resolved
four-jet phase space.

Approximately 63 million QCD events survive the union of
the released `4j3b` and `4j2b` trigger paths and enter
training.

The very large QCD statistics are not incidental. The paper
reports a substantial loss in discrimination when training
statistics are reduced by a factor of ten.

### 4.4 Top generation

The raw top-pair sample contains approximately 40 million
events.

Approximately 12 million enter training.

Unlike signal and QCD, the training top sample retains
events using the released `3b` or `2b` flavor requirements
without additionally imposing the leading-four-jet
transverse-momentum thresholds or the HT condition. This
choice preserves a larger top training sample.

### 4.5 Labels

The released network has 138 outputs.

Zero-based inference convention used in the Track B code:

| Output indices | Meaning |
|---|---|
| `0-135` | 136 variable-mass signal cells |
| `136` | QCD |
| `137` | top-pair |

Some prose in the paper describes the final classes using
one-based numbering. Track B uses the actual zero-based ONNX
output indices.

The aggregate signal probability is:

`Psignal = sum(output[0:136])`

with:

- `PQCD = output[136]`
- `Pttbar = output[137]`

### 4.6 Training configuration

Published training configuration:

- architecture: Particle Transformer;
- trainable parameters: approximately 9 million;
- ensemble members: 3;
- independent random initializations: 3;
- optimizer: Ranger, combining RAdam and Lookahead;
- initial learning rate: `2 x 10^-3`;
- precision: automatic mixed precision;
- hardware: four Nvidia A100 GPUs;
- epochs: 80;
- nominal batch size: 512;
- samples used per training epoch: `2.56 x 10^6`;
- samples used per validation epoch: `6.4 x 10^5`;
- approximate training time: 92 hours per model;
- event-level training weights: none.

No explicit event reweighting is applied to signal, QCD, or
top. The authors report that reweighting reduced effective
training statistics and degraded performance.

## 5. Remote IHEP dataset namespace

Endpoint:

`root://cceos.ihep.ac.cn:1094`

Base path:

`/eos/ihep/cms/store/user/coli/datasets/hh4b`

Primary split directories:

- `training`
- `inference`

Observed remote file-inventory footprint:

| Split | Observed size |
|---|---:|
| Training | approximately 878.324 GiB |
| Inference | approximately 709.058 GiB |
| Combined | approximately 1587.382 GiB, or 1.704 TB decimal |

These storage totals describe the available remote ROOT
inventory. They are not themselves event-count estimates.

The remote training signal directory contains 500 merged
ROOT files.

Important training datasets include:

- `HH4b_2HDM_H3VAR_H1H2_40to200_merged_ntuple`
- `QCD_DelphesHH4JTrig_merged_ntuple`
- `TTbar_ntuple`

Important inference datasets include:

- independent QCD samples;
- independent top samples;
- `ZZ_ntuple`;
- `ZH`-related samples;
- inclusive SM backgrounds;
- signal and resonance-control samples.

The training and inference samples were designed to be
statistically orthogonal.

## 6. File-level inventory and pilot design

A recursive remote inventory was completed without
downloading the complete dataset.

Inventory artifacts included:

- `dataset_summary.tsv`
- `extension_summary.tsv`
- `file_manifest.tsv`
- `largest_files.tsv`
- `pilot_candidates.tsv`
- `recursive_ls_l.txt`

The original persistent audit location was:

`/uscms_data/d3/iturkmen/hh4b_external_dataset/ihep_jetfree_hh4b_file_inventory_20260730_v1`

The intended future EOS namespace is:

`/store/user/iturkmen/hh4b_external_dataset/ihep_jetfree_hh4b_20260730_v1`

A representative transfer pilot was designed using:

- 5 signal files;
- 4 QCD files;
- 2 top files;

for approximately 1.4 million events.

The full persistent transfer remains prohibited until
`eosquota` visibly reports the requested 5 TB allocation.

## 7. Sequential representative-file schema audit

Five representative remote files were downloaded one at a
time to temporary storage, verified, inspected with Uproot,
and removed before the next file.

Representative files:

| Label | Dataset | Bytes |
|---|---|---:|
| Training signal | `HH4b_2HDM_H3VAR_H1H2_40to200_merged_ntuple` | 920264712 |
| Training QCD | `QCD_DelphesHH4JTrig_merged_ntuple` | 955591242 |
| Training top | `TTbar_ntuple` | 391691049 |
| Inference QCD | `QCD_DelphesHH4JTrig_forInfer2_merged_ntuple` | 698774117 |
| Inference ZZ | `ZZ_ntuple` | 690055895 |

Observed tree summaries:

| Label | Main entries | Branches | Schema prefix |
|---|---:|---:|---|
| Training signal | 134856 | 87 | `b195d1b29676` |
| Training QCD | 154925 | 87 | `b195d1b29676` |
| Training top | 58000 | 98 | `a23d1e371f50` |
| Inference QCD | 113269 | 87 | `b195d1b29676` |
| Inference ZZ | 122974 | 98 | `a23d1e371f50` |

Some ROOT files contain multiple tree cycles. Older cycles
were observed, for example:

- training top: 56196 entries in an older cycle;
- inference ZZ: 122606 entries in an older cycle.

Two principal schema families were therefore observed:

1. merged training-signal/QCD-style schema;
2. selected-background/resonance-style schema.

The schema contains the required particle-level,
jet-level, fatjet-level, generator-level, and selection
variables.

## 8. Released repository and model contract

Pinned public repository commit:

`e8387f7e49a7f27c65b2f4a1f75cacf01cd45392`

Released models:

| Model | Size | SHA-256 |
|---|---:|---|
| `model0.onnx` | approximately 35 MiB | `7b7de7ba1bfbbf00117b251661320afabe166c74c8687e3673d97590a3745786` |
| `model1.onnx` | approximately 35 MiB | `5514875d359fd4d6516e3367eb3264e6b4fea21f82755fc45851d06f788d429f` |
| `model2.onnx` | approximately 35 MiB | `2d02df9702bfa13f8c20b467087121abab2556fa654e12084b73cee968f3566e` |

ONNX input contract:

| Input | Shape |
|---|---|
| `pf_features` | `[N,19,256]` |
| `pf_vectors` | `[N,4,256]` |
| `pf_mask` | `[N,1,256]` |

Output contract:

`[N,138]`

Each event is truncated or padded to 256 particles.

The official preprocessing helper is:

`delphes/ana/OrtHelperSophonHH.h`

## 9. Official feature ordering

The 19 `pf_features` channels are:

1. transformed `log(pt)`;
2. transformed `log(E)`;
3. transformed `log(pt / global_sum_pt)`;
4. transformed `log(E / global_sum_E)`;
5. transformed delta-R to the global particle sum;
6. charge;
7. charged-hadron indicator;
8. neutral-hadron indicator;
9. photon indicator;
10. electron indicator;
11. muon indicator;
12. `tanh(d0)`;
13. clipped `d0` error;
14. `tanh(dz)`;
15. clipped `dz` error;
16. delta-eta to the global particle sum;
17. delta-phi to the global particle sum;
18. eta;
19. phi.

Four-vector channels:

1. px;
2. py;
3. pz;
4. energy.

Mask convention:

- 1 for valid particles;
- 0 for padding.

The first 256 particles in the stored collection order are
retained.

## 10. Mass-grid audit

A deterministic audit sampled:

- 12 evenly spaced training-signal files;
- 36,864 events in total.

Results:

- all 136 signal classes were observed;
- individual files contain broad mixtures of mass classes;
- files are not narrowly partitioned by one mass cell;
- random or evenly spread file sampling is appropriate for
  representative pilot construction.

Class 100 in the zero-based triangular ordering corresponds
to the `(125,125)` GeV cell.

## 11. Released-model reproduction canaries

### 11.1 Isolated inference

A memory-safe isolated inference canary successfully:

- checked out the pinned repository;
- validated all three ONNX files;
- loaded the models;
- constructed released-contract tensors;
- ran CPU ONNX Runtime inference;
- validated output shapes and normalization;
- removed all temporary inputs and outputs.

### 11.2 Background-domain canary

One-file, 96-event canaries were performed for training and
independent background datasets.

Representative ensemble results:

| Dataset | Target top-1 | Target mean probability |
|---|---:|---:|
| Training QCD | 100.000% | 0.893333 |
| QCD `forInfer` | 97.917% | 0.887904 |
| QCD `forInfer2` | 98.958% | 0.951289 |
| Training top | 82.292% | 0.717601 |
| Top `forInfer` | 86.458% | 0.771425 |
| Top `forInfer2` | 80.208% | 0.733782 |

The background transfer was therefore broadly stable at
canary scale, with a moderate shift in the QCD `forInfer2`
extension sample.

These are diagnostic canaries, not precision measurements.

### 11.3 Independent ZZ response

For a 96-event independent ZZ canary, the ensemble means
were approximately:

- aggregate signal probability: 0.356683;
- QCD probability: 0.505418;
- top probability: 0.137899.

This established nontrivial resonance sensitivity but did
not by itself reproduce the final mass fit.

### 11.4 Resonance localization

Completed checks included:

- raw ZZ localization;
- ZH localization;
- official ZH double-sided Crystal Ball fitting;
- minor-background audits.

The official ZZ DCB continuation was interrupted by the
IHEP service outage.

The released model's strong response to ZZ and ZH is
expected: the 136 signal classes describe generic
`h1 h2 -> 4b` structures rather than only SM HH.

## 12. Track A three-model raw-EFlow transfer canary

The first Track A transfer canary used:

- 64 selected signal events;
- 64 selected QCD events;
- 64 selected top events;
- three released models;
- 192 events total.

Aggregate mean probabilities:

| Process | Psignal | PQCD | Pttbar |
|---|---:|---:|---:|
| Signal | 0.485417 | 0.132269 | 0.382314 |
| QCD | 0.417303 | 0.192551 | 0.390146 |
| Top | 0.165994 | 0.127321 | 0.706685 |

Interpretation:

- model execution transferred technically;
- top discrimination transferred strongly;
- signal and QCD remained poorly separated;
- the fixed-mass signal response was shifted toward high
  signal-mass classes.

This is an unweighted domain-transfer canary.

It is not a physics-normalized result.

## 13. Raw-DZ versus zero-DZ ablation

The same 192 events were evaluated twice:

1. Track A raw DZ values;
2. identical inputs with only the transformed DZ channel
   zeroed.

### Signal

| Quantity | Raw DZ | Zero DZ |
|---|---:|---:|
| Mean Psignal | 0.485417 | 0.399879 |
| Mean PQCD | 0.132269 | 0.168975 |
| Mean Pttbar | 0.382314 | 0.431146 |
| Aggregate winners: signal/QCD/top | 34/5/25 | 24/7/33 |

Mean signal-probability change:

`-0.085538`

Winner changed for:

`11/64 = 17.188%`

### QCD

Mean signal-probability change:

`-0.045997`

Winner changed for:

`8/64 = 12.5%`

### Top

Mean signal-probability change:

`-0.023330`

Winner changed for:

`4/64 = 6.25%`

Interpretation:

- the released model genuinely uses DZ information;
- the effect is largest for the b-rich signal sample;
- DZ does not explain the high-mass localization;
- zeroing DZ does not repair signal-versus-QCD transfer.

## 14. Fixed-mass signal localization in the first canary

For raw DZ:

- mean class-100 `(125,125)` probability: 0.001123;
- mean probability in the 115-135 GeV region: 0.010043;
- mean fraction of aggregate Psignal in that region:
  0.022951;
- signal-class argmax in the region: 1 of 64.

Dominant signal-class cells included:

- class 134: `(185,195)` GeV;
- class 132: `(175,195)` GeV;
- class 122: `(155,165)` GeV.

Zeroing DZ did not materially correct this behavior.

The high-mass response is therefore presently treated as a
domain-shift observation, not as a released-model failure.

## 15. Track A particle-collection provenance

A direct ROOT branch audit established for signal, QCD,
and top:

- exact official PF collection: false;
- candidate-level PUPPI collection: false;
- raw EFlow collections: true.

Result:

`TRACK_A_RAW_EFLOW_ONLY`

The Track A cards merge the collections in this order:

1. tracks;
2. photons;
3. neutral hadrons.

The active Track A production card identified in current
production provenance was:

`cards/delphes/delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl`

A legacy card also appears in earlier production and
regeneration provenance:

`cards/delphes/delphes_card_CMS_lpc.tcl`

Per-file card provenance should therefore not be assumed
homogeneous without checking the relevant campaign.

## 16. Raw-EFlow adapter feature-parity audit

The Python adapter was compared against an independent
literal implementation of the released preprocessing
formulas.

Audit sample:

- 16 signal events;
- 16 QCD events;
- 16 top events;
- 48 total;
- 3 events with more than 256 particles.

Results:

| Quantity | Maximum difference |
|---|---:|
| Features | `0.000e+00` |
| Four-vectors | `0.000e+00` |
| Mask | `0.000e+00` |
| Global sums | `4.547e-13` |

Result:

`TRACK_A_RAW_EFLOW_ADAPTER_FEATURE_PARITY_PASS`

This excludes the following as explanations for the
observed transfer behavior:

- channel-order bugs;
- transformation bugs;
- padding bugs;
- mask bugs;
- 256-particle truncation bugs;
- vectorized-versus-reference construction differences.

It does not establish raw-EFlow/PUPPI detector equivalence.

## 17. Corrected Track A kinematic selection provenance

A stale assertion previously expected the first 4096 events
to produce the count vector:

`(1069,3090,1297)`

for signal, QCD, and top.

A dedicated provenance audit established that the released
kinematic four-jet conditions:

- base jets: `pT > 30 GeV`;
- acceptance: `|eta| < 2.5`;
- leading thresholds: `75,60,45,40 GeV`;
- `HT > 330 GeV`;

produce:

| Process | Passing events | Index SHA-256 |
|---|---:|---|
| Signal | 1111 | `90acb426589d2beae386f1ee0966b065b3df4a2913e1388baca992bc94a2c531` |
| QCD | 3041 | `8848e022ddd61e6115ce8e4d371337826d5081b8789aa360a74eab0bf60b191a` |
| Top | 1323 | `6f0b202dda52d78ee701bbdeda870118c4a35add161e23beee8ab098c130a9c5` |

Result:

`OFFICIAL_SELECTION_COUNTS_ESTABLISHED_PRIOR_VARIANT_UNRESOLVED`

No tested nearby threshold configuration reproduced all
three older counts.

The signal count 1069 is reproduced by changing only the
HT threshold to 340 GeV, but the corresponding QCD and top
counts are 3041 and 1260. It therefore does not explain the
old three-process vector.

Many text-search matches for `3090` were unrelated Pythia
generation progress messages.

The larger characterization must use the corrected count
vector and index hashes.

Important qualification:

This is the kinematic `4j` component. The complete released
`4j3b` trigger additionally requires at least three
SophonAK4-tagged jets. The current raw-EFlow transfer does
not reproduce that official b-tag domain.

## 18. Matched-card same-generator-event feasibility

The cleanest detector-domain diagnostic is:

> Process the preserved Track A signal HepMC through the
> pinned IHEP Delphes/PUPPI card and compare it to the
> existing Track A detector output for the same generator
> events.

Available generator sources:

| Process | Same-event generator source |
|---|---|
| Signal | HepMC and LHE available |
| QCD | not found |
| Top | not found |

Result:

`PARTIAL_GENERATOR_SOURCE_COVERAGE`

A signal-only same-event comparison remains scientifically
valid and useful.

QCD and top must not be silently replaced with regenerated
nonidentical events in a comparison labeled same-event.

## 19. Pinned IHEP Delphes card

Card:

`delphes/cards/delphes_card_CMS_JetClassII_lite.tcl`

SHA-256:

`8088b8e3939a8776c7ac79b8f102fd810c9aaab27d553df1c85afe663fb53578`

Relevant card contract:

- Delphes 3.5-style simulation;
- mean pileup: 50;
- required pileup file: `MinBias_100k.pileup`;
- PUPPI particle processing;
- active output:
  `RunPUPPI/PuppiParticles`;
- TreeWriter branch:
  `ParticleFlowCandidate`;
- vertex output;
- PUPPI-based jet collections.

## 20. Delphes runtime readiness

The available Track A Delphes executable initially failed
in the login-shell runtime because of incompatible or
missing C++ libraries.

A validated runtime was established using:

`/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh`

Validated components:

- GCC/G++ 13.1.0;
- ROOT 6.32.02;
- compatible `libstdc++.so.6`;
- `GLIBCXX_3.4.30`;
- `GLIBCXX_3.4.31`;
- `libtbb.so.12`;
- `libvdt.so`;
- ROOT shared libraries.

Result:

`DELPHES_LCG_RUNTIME_READY`

The corresponding Python environment also provides:

- Python 3.11.9;
- NumPy 1.26.4;
- Awkward 2.6.4;
- Uproot 5.3.7;
- ONNX Runtime 1.17.3;
- CPU execution provider.

Result:

`TRACK_B_ONNXRUNTIME_PYTHON_READY`

No additional Python installation is required.

## 21. Exact pileup blocker

The pinned card requires:

`MinBias_100k.pileup`

No local or CVMFS copy was found.

Available conversion utilities:

- `hepmc2pileup`;
- `root2pileup`;
- `stdhep2pileup`;
- standard `converter_card.tcl`.

Missing authoritative assets:

- final `MinBias_100k.pileup`;
- matching minimum-bias HepMC source;
- matching minimum-bias ROOT source;
- `generatePileUp.cmnd`;
- exact Pythia process settings;
- exact random seed;
- complete reproduction contract.

Result:

`EXACT_IHEP_PILEUP_STILL_EXTERNAL`

The exact matched-card signal canary is therefore externally
blocked, not locally or technically blocked.

Acceptable resume inputs are:

1. the exact pileup file and checksum;
2. the authoritative minimum-bias HepMC source and checksum;
3. a complete generation and conversion contract.

An arbitrary generic `MinBias.pileup` file must not be
substituted in a test labeled exact IHEP reproduction.

## 22. IHEP endpoint status

Remote endpoint:

`root://cceos.ihep.ac.cn:1094`

The endpoint became unreachable before the IHEP annual
cluster-maintenance window of July 31-August 1.

The failure occurred before the XRootD protocol stage,
consistent with a service/network outage.

Repeated probes were stopped to avoid unnecessary load.

The next connectivity check should be one bounded
exact-file `xrdfs stat` after the maintenance period.

## 23. Storage policy

Track A is protected and must remain intact.

No Track B full transfer is authorized to d3.

Observed d3 status during recent audits:

- approximately 195 GiB used;
- 200 GiB soft quota;
- 220 GiB hard limit.

Temporary work should continue under `/tmp`.

Persistent external-dataset storage is intended for EOS
only after the quota visibly reports 5 TB.

No recursive `xrdcp -r` transfer should be used.

Persistent EOS transfers should not use overwrite forcing.

## 24. Git checkpoints

Track B branch:

`track-b-sophon-transfer`

Known checkpoints:

### Initial transfer checkpoint

Commit:

`9306212a161281b372d02ec128dd97e8068c6a87`

Document:

`docs/checkpoints/track_b_sophon_transfer_canary_20260731_v1.md`

Any claim in that initial checkpoint implying exact
raw-EFlow equivalence to official PUPPI is superseded by
this master document.

### External pileup blocker checkpoint

Commit:

`628c8bf65ad64c9f5998b64a04c4b5bbf52b2ec9`

Document:

`docs/checkpoints/track_b_ihep_pileup_external_blocker_20260731_v1.md`

Result:

`TRACK_B_IHEP_PILEUP_BLOCKER_CHECKPOINT_PUSHED`

## 25. Current scientific interpretation

What is established:

1. The released three-model ensemble can be loaded and run.
2. The 138-output class contract is understood.
3. The 136-class signal mass grid is covered by the
   training data.
4. Direct IHEP background canaries behave sensibly.
5. Resonant processes produce signal-like responses.
6. Track A top transfer is strong.
7. Track A signal-versus-QCD transfer is weak.
8. Track A signal localization is shifted high.
9. Raw DZ affects the response but does not cause the
   localization shift.
10. The Python raw-EFlow adapter correctly implements the
    released formulas for the records supplied to it.
11. Track A lacks the official PUPPI candidate collection.
12. Particle representation and detector-card mismatch are
    the leading unresolved causes.
13. The exact matched-card test is blocked only by the
    unavailable pileup artifact.

What is not established:

- exact IHEP reproduction on Track A;
- physics-normalized Track A performance;
- weighted full-sample significance;
- failure of the released model on fixed-mass HH;
- equivalence of raw EFlow and PUPPI;
- exact same-event background comparisons;
- final transfer-learning benefit.

## 26. Immediate next gate

The immediate runnable Track B gate is:

> Larger deterministic raw-EFlow transfer characterization,
> using the corrected official kinematic four-jet selection.

Contract:

- scan first 4096 events per process;
- require canonical counts:
  `(1111,3041,1323)`;
- require the canonical passing-index hashes recorded above;
- select 256 evenly spread passing events per process;
- process 768 events total;
- run all three released models;
- compute ensemble probabilities;
- compute signal-versus-QCD AUC;
- compute a deterministic bootstrap interval;
- report per-model performance;
- report ensemble disagreement;
- report fixed-mass signal localization;
- report truncated-versus-nontruncated behavior.

The result must continue to be labeled:

> unweighted raw-EFlow-to-PUPPI-model domain-transfer
> characterization

It must not be labeled:

- exact IHEP reproduction;
- full `4j3b` reproduction;
- physics-normalized performance.

## 27. Subsequent gates

After the corrected characterization:

1. checkpoint its deterministic inputs and metrics;
2. wait for the authoritative pileup artifact;
3. run the same-HepMC matched-card signal canary;
4. compare official PUPPI and Track A raw-EFlow response;
5. determine whether localization moves toward `(125,125)`;
6. restore IHEP connectivity and rerun the pending ZZ DCB;
7. transfer only the bounded pilot after storage approval;
8. reproduce released-model metrics at larger statistics;
9. design matched-statistics Track B models;
10. evaluate pretraining and fine-tuning on Track A;
11. compare against Track A cut, BDT, DNN, LBN, SPA-Net,
    and representation-learning baselines.

## 28. Source references

Public paper:

`https://arxiv.org/abs/2508.15048`

Pinned public repository:

`https://github.com/pku-hep-group/jetfree-hh4b/tree/e8387f7e49a7f27c65b2f4a1f75cacf01cd45392`

Key released sources:

- `README.md`
- `delphes/cards/delphes_card_CMS_JetClassII_lite.tcl`
- `delphes/ana/makeNtuplesHH4bAllObjectsOptionalSel.C`
- `delphes/ana/OrtHelperSophonHH.h`
- `delphes/ana/InferSophonHH.C`
- `delphes/models/HH4b/model0.onnx`
- `delphes/models/HH4b/model1.onnx`
- `delphes/models/HH4b/model2.onnx`
- `ana_scripts/dcb_fit/dcb_fit_single_job.py`

## Repository-state receipt for this document

Track B branch SHA before this document:

`628c8bf65ad64c9f5998b64a04c4b5bbf52b2ec9`

Track A working-repository SHA observed when this document
was prepared:

`4647ffc50cc5436b73430897680613e76c327d94`

These hashes identify the two separate repository states
and must not be interpreted as one shared branch history.

## 29. Final status

Track B is not stalled globally.

Only the exact IHEP matched-card detector reproduction is
externally blocked.

Released-model studies, raw-EFlow domain characterization,
documentation, deterministic metric extraction, and later
matched-statistics design can continue independently.

## 2026-08-01 V4 direct-probability update

The deterministic raw-EFlow transfer characterization was
completed using the direct normalized ONNX probabilities.

Machine-readable interpretation:

    SECOND_SOFTMAX_APPLIED=0
    DOMAIN_LABEL=raw-EFlow-to-PUPPI-model
    NEXT_MILESTONE=frozen-score transfer baseline

V3 applied a second softmax to an output node that was
already a softmax. All V3 quantitative probability, AUC,
bootstrap, and localization results are superseded.

Valid V4 headline results:

- signal-QCD direct Psignal AUC: `0.596954`;
- signal-QCD conditional-discriminant AUC: `0.655045`;
- conditional-discriminant 95 percent bootstrap interval:
  `[0.605251,0.703328]`;
- top-versus-nontop Pttbar AUC: `0.792793`;
- top 95 percent bootstrap interval:
  `[0.759590,0.825756]`;
- conditional posterior in the 115-135 GeV region:
  `0.023308`;
- mean conditional posterior masses:
  `(128.109,172.646)` GeV;
- signal argmax in the 115-135 GeV region: `3/256`.

The full deterministic receipt is:

`docs/checkpoints/track_b_raw_eflow_transfer_characterization_v4_20260801_v1.md`

Track B SHA before this update:

`85e1a9e1ec95f84466c6a8c91ebc44d6f9555753`

Track A source-repository SHA observed during preparation:

`6835f69a0ef6cb40b82e39f91f7d2f7036c1ee0d`
