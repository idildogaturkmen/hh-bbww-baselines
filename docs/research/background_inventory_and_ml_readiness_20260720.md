# Background Inventory and ML Readiness

**Snapshot date:** 2026-07-20
**Branch:** `delphes-hh4b-production`
**Purpose:** Define which simulated backgrounds are currently usable for
pipeline testing, learning-curve studies, final-workflow training, and
statistical interpretation.

## 1. Compatibility rule

Samples are not considered mutually mergeable merely because they represent
the same physics process.

A final combined ML sample must have compatible:

- hard-process and decay definitions;
- generator and shower configuration;
- detector card and object definitions;
- feature and candidate schemas;
- event-weight interpretation;
- train/validation/test assignment;
- source lineage and checksum records;
- overlap-removal policy.

Samples that fail these criteria may still be retained for exploratory
development, but must not be silently mixed into final training or inference.

## 2. Tier A: final-workflow-compatible samples

### 2.1 Adaptive QCD HardQCD sample

Status: **full cross-layer valid**

- Generated events: 2,510,000
- Shards: 261
- Train events: 1,811,373
- Validation events: 514,084
- Sealed test events: 184,543
- Number of disjoint pTHat strata: 8
- Delphes card SHA-256:
  `1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c`
- Registry SHA-256:
  `f0209b8df13e606c73e693dd8f78be11805c947486119825fe1cfe7ba46cb6e6`
- Whole-shard split assignment: yes
- Test used for training: no
- Cross-layer failures: none
- Bootstrap unit: whole shard within pTHat stratum
- Bootstrap replicates: 2000

Physical weighting is performed after combining all accepted shards within
each disjoint pTHat stratum. Shards and waves are never normalized
independently.

The sample is usable for:

- candidate-level BDT and dense-DNN development;
- QCD component of event-level SPA-Net classification;
- raw and weighted learning curves;
- high-score-tail occupancy studies;
- background-statistics and effective-sample-size studies.

The sample is not yet authorized to expand automatically to 5M. The planned
additional 2,490,000 events require a separate allocation and authorization
decision.

### 2.2 Exact-lineage frozen-v2 inclusive ttbar

Status: **40K unsealed train/validation events validated**

| Shard | Seed | Split | Generated events | Candidate rows | Validation |
|---:|---:|---|---:|---:|---|
| 0 | 714000 | train | 10,000 | 39 | local frozen-v2 ROOT and canonical Parquet validation |
| 1 | 714001 | train | 10,000 | 34 | full end-to-end receipt/EOS validation |
| 2 | 714002 | train | 10,000 | 31 | full cross-layer validation |
| 3 | 714003 | validation | 10,000 | 43 | full cross-layer validation |
| 4 | 714004 | test | 10,000 | sealed | not processed or inspected |

Unsealed totals:

- Train generated events: 30,000
- Train candidate rows: 104
- Validation generated events: 10,000
- Validation candidate rows: 43
- Combined unsealed candidate acceptance: 147 / 40,000 = 0.003675

Canonical schema hashes:

- ROOT:
  `cee4ccc1ce6126eb7f03d05f84f40ff2dbf8d8f68672b885cd4faab80dfbf3f7`
- Event summary:
  `0259586415e02c7755767f5da07236b30ff1ee9afe953a0ff5100a18abe7efcd`
- Candidate table:
  `d44e5af4a4ec3f191e88f52b54c2e8afe64cbdc78431bad500a000de56b600c5`

The sample is usable for:

- validating generator-to-Delphes-to-Parquet execution;
- checking candidate acceptance;
- validating candidate schemas and feature quality;
- leave-one-shard-out out-of-fold development;
- small pipeline and learning-curve checkpoints.

The sample is not large enough for a final DNN or SPA-Net result.

The generator cross section near 512 pb is provisional generator-level
metadata and must not be treated as final higher-order normalization.

## 3. Tier B: provisional legacy ttbar

Status: **usable for exploratory studies; not compatible with final
frozen-v2 training**

- Generated events: 300,000
- Shards: 30
- Seeds: disjoint
- Canonical candidate source: manifest-listed per-shard candidate Parquets
- Candidate rows: 1,112
- Candidate raw efficiency: 0.0037067
- rHH < 80 rows: 838
- rHH < 50 rows: 597
- mHH > 600 rows: 204
- mHH > 800 rows: 92
- Generator process: `p p > t t~`
- Explicit decay override found: no
- Detector-card compatibility with frozen-v2: not proven
- Normalization: provisional generator-level, not final higher order

The 100K merged candidate output is consistent with its per-shard inputs.
The 200K merged output is inconsistent and is not canonical. It is preserved
for forensics only.

Allowed uses:

- approximate candidate-rate studies;
- exploratory architecture debugging;
- early kinematic comparisons;
- development of plotting and evaluation code.

Disallowed uses:

- merging with frozen-v2 samples for final training;
- final expected-yield calculations;
- final model comparison;
- final score-tail interpretation.

## 4. COLLIDE-1M status: excluded from the canonical Delphes study

COLLIDE-1M is not part of the canonical frozen-v2 Delphes dataset described
in this document.

It must not be mixed with the frozen-v2 QCD, ttbar, signal, or other Delphes
samples for:

- model training;
- validation or test evaluation;
- learning-curve comparisons;
- physical normalization;
- expected-yield calculations;
- significance estimates;
- score-tail uncertainty estimates.

The reasons are that COLLIDE-1M does not share the currently frozen generator,
detector-card, weighting, lineage, and whole-shard split definitions.

A future use is permitted only as a clearly separate external-domain
robustness benchmark. Such a study must train and evaluate the canonical
Delphes result first, keep all COLLIDE events outside the canonical splits,
and report the COLLIDE comparison as domain transfer rather than as added
training statistics.

## 5. ML feature denylist

The following fields must never be classifier inputs:

- sample or process label;
- campaign name;
- file or source path;
- shard ID;
- seed;
- event identifier;
- dataset split;
- fold identifier;
- generator cross section;
- event or normalization weight;
- payload, source, or card checksum;
- target labels and truth-only quantities unavailable at inference.

Weights may be used for training objectives or evaluation only under a
documented policy.

## 6. Split and evaluation policy

- Split assignment is performed by whole independent shard.
- Training preprocessing is fitted on training folds only.
- Validation is used for architecture choice, early stopping, and threshold
  selection.
- Test remains sealed until the model, features, preprocessing, and analysis
  thresholds are frozen.
- Out-of-fold predictions must be generated only by a model that did not
  train on the predicted event.
- Production-size learning curves must use nested event subsets.

## 7. Current readiness conclusion

The current final-workflow-compatible background base is:

- QCD: 2.51M fully audited events;
- ttbar: 30K train plus 10K validation events;
- ttbar sealed test: 10K, unopened.

This is sufficient for:

- constructing the canonical training registry;
- implementing out-of-fold BDT and DNN pipelines;
- validating the two-head SPA-Net training interface;
- measuring initial learning curves.

It is not sufficient for:

- claiming a final DNN or SPA-Net advantage;
- stable characterization of the most signal-like ttbar score tail;
- final 5M-scale conclusions;
- opening the sealed test set.

## 8. Scale-up gates

### 100K checkpoint

Required evidence:

- successful new-generation preflight;
- exact process and decay specification;
- stable acceptance across shards;
- no seed or event overlap;
- matching schemas;
- complete receipts and checksums.

### 500K checkpoint

Required evidence:

- first useful out-of-fold BDT/DNN/SPA-Net comparison;
- stable train-validation behavior;
- raw and weighted score-tail counts;
- effective background sample size;
- learning-curve improvement beyond seed/fold variation.

### 1M checkpoint

Required evidence:

- model performance versus candidate count;
- high-score-bin MC uncertainty;
- calibration and background-category stability;
- decision record authorizing or rejecting continuation toward 5M.

### 5M checkpoint

The 5M expansion is authorized only if the 1M study demonstrates that model
generalization or score-tail uncertainty remains simulation-statistics
limited.

The scientific question is not whether more events can be generated. It is
whether physically independent simulation continues to improve model
generalization and reduce uncertainty in the signal-like background tail.
