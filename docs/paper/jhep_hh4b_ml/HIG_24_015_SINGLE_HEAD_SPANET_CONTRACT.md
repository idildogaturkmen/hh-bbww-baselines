# Train-only single-head SPA-Net contract

Status: frozen before the complete signal-truth extraction and before any
single-head training on 2026-08-03.

## Scope and supported interpretation

This is a Delphes-simulation, train-only source-group out-of-fold study.  It
contains no observed data and must not open validation or test payloads.  The
primary physical QCD prediction remains the frozen exactly-$3b$ to
$\geq4b$ transfer; direct-$\geq4b$ QCD is secondary closure only.  Nominal and
shared-multijet-nuisance results are kept separate and neither is a Run-2
expected sensitivity.

The frozen c7s tables expose exactly the four jets in the reconstructed HH
candidate, their original Delphes jet indices, and the original selected-jet
multiplicity.  They do not expose a padded variable-length list of every
selected jet.  The supported model is therefore explicitly a **candidate-four
pairing SPA-Net baseline**: it learns the perfect pairing of the four frozen
candidate jets, not jet selection from all reconstructed jets.  Every input
mask has four real entries.  Original `n_selected_jets`,
`n_extra_selected_jets`, and their source-member distributions are reported as
diagnostics but are not reinterpreted as hidden model inputs.

Legacy SPA-Net-style files in `scripts/delphes/` and `data/spanet_hbb/` are not
numerical evidence for this stage.  They use older samples, random row splits,
validation/test partitions, or the different HH-to-bbWW assignment problem.
Only their general software concepts were inspected.  No legacy validation or
test payload was opened.

## Truth and ambiguity contract

Truth is recovered only for the 87 frozen train-signal source members in
`exact_441_train_root_source_registry.tsv`.  Each remote bundle must first pass
its frozen size plus Adler-32 contract or its frozen SHA-256 contract.  Archive
members are exact, path-traversal-safe registry values.  No background ROOT
payload is required, because background assignment loss is always masked.

For each signal candidate row:

1. resolve the two unique generator-level $H\to b\bar b$ daughter-index sets,
   collapsing duplicate Higgs copies that lead to the same two daughters;
2. enumerate every candidate-jet pair compatible with each daughter set using
   both $b$ permutations and $\Delta R(b,j)<0.4$ for both matches;
3. enumerate disjoint two-pair partitions of all four candidate jets and
   collapse the two Higgs ordering and the ordering within each Higgs;
4. label the event only when exactly one symmetry-distinct perfect partition
   remains.

Zero compatible partitions are `unmatched`.  More than one daughter-set
decomposition or more than one compatible perfect partition is `ambiguous`.
Rows with fewer than two unique $H\to b\bar b$ daughter sets are
`truth_unavailable`.  None of these statuses is repaired or assigned a guessed
label.  Assignment loss is enabled only for `matchable` signal rows and is
masked for unmatched/ambiguous signal and every background row.

The first independently checksum-verified canary, `ggf_shard_000`, contains 58
frozen c7s signal rows.  This rule labels 37 as matchable and 21 as unmatched,
with no ambiguous rows; the three perfect-partition labels occur 7, 5, and 25
times.  The complete 87-source audit, not this canary, will determine the final
published fractions.

Local ROOT reads use uproot's explicit `MemmapSource`.  In the frozen execution
environment, uproot 5.6.9's default fsspec local source waits indefinitely for
a range future even though the same verified file reads correctly through
`MemmapSource`; this behavior is tested and the extractor never silently falls
back to an unverified source.

## Frozen assignment architecture

The input of each jet is the ordered numerical vector
`log1p(pt)`, `eta`, `sin(phi)`, `cos(phi)`, and `log1p(mass)`.  Flavor labels are
forbidden.  The first normalization smoke gate established that all four
candidate `btag` columns are identically one in all 30,705 development rows,
whereas every primary transferred-QCD row has one promoted candidate with
`btag=0`.  `btag` is therefore excluded before any fit: it is unlearnable in the
training support and would introduce a construction-induced domain shift in the
primary projection.  Normalization means and standard deviations are fit on the
current source-group training partition only; any remaining constant feature
fails closed.

The frozen table contains six negative jet masses, all compatible with
floating-point reconstruction noise: the minimum is
$-1.1681\times10^{-6}\,\mathrm{GeV}$.  This was discovered by the first smoke
run, before any fit.  Values in $[-2\times10^{-6},0)\,\mathrm{GeV}$ are clipped
to zero before `log1p`; a mass below that frozen tolerance fails closed.  No
other kinematic feature is clipped.

There is no positional encoding.  A shared linear projection maps each jet to
64 dimensions, followed by two Transformer encoder layers with four attention
heads, feed-forward width 256, GELU activation, pre-layer normalization, and
dropout 0.05.  A shared pair scorer consumes the symmetric features
`h_i+h_j`, `abs(h_i-h_j)`, and `h_i*h_j`, uses a 64-unit GELU hidden layer, and
returns one scalar.  Each of the three four-jet perfect-matching logits is the
sum of its two pair scores.  This construction is jet-permutation equivariant,
invariant to $b$ interchange within a Higgs, and invariant to interchange of
the two Higgs candidates.

Training is fixed rather than architecture-selected: 24 epochs, batch size
512, AdamW with learning rate $10^{-3}$ and weight decay $10^{-4}$, weighted
cross-entropy over matchable signal, and no early stopping.  Signal development
weights are normalized to unit mean within each fit.  Torch deterministic
algorithms and fixed CPU seeds are required.  The seed is derived only from the
base seed 20260803 and the declared outer/inner fold IDs.  The implementation
must record loss curves, parameter count, environment versions, wall time, and
repeated-run inference median and interquartile range.

## Source-group OOF and downstream classification

The exact c7s `registry_oof_fold` is the five-fold outer split.  For each outer
fold, four inner fits supply source-group OOF scores on the outer-training
partition.  The untouched outer fold is evaluated once.  No source group may
occur in a fit and its held partition.

The assignment encoder has one and only one trainable task head.  Event
classification is a logically separate c7t-parameter XGBoost classifier trained
after the encoder.  Its inputs are only the learned permutation-invariant mean
and maximum jet embeddings plus sorted assignment probabilities, maximum
assignment probability, and assignment entropy.  It does not consume truth
labels or the frozen geometric pairing label.  XGBoost uses the frozen global
c7t values: 400 trees, learning rate 0.03, depth 5, subsample and column sample
0.75, minimum child weight 7, gamma 0.1, and unit L1/L2 regularization.

The operating target is frozen to the exact c7t cut-baseline weighted signal
efficiency, 0.6156418377159854.  Each outer fold's numerical classifier
threshold is obtained only from its inner source-group OOF scores.  Architecture,
features, epochs, optimizer, downstream-classifier parameters, and target
efficiency are not selected using outer-held results.

## Metrics and uncertainty

Assignment metrics are recomputed on complete source-member bootstrap draws:
exact event pairing, per-Higgs pairing, per-jet partner assignment, matchable
efficiency, ambiguity and unmatched fractions, and mass residuals before and
after learned pairing.  Under the candidate-four perfect-matching contract, an
incorrect perfect matching shares no pair and no jet partner with the truth;
therefore event, per-Higgs, and per-jet accuracies are numerically identical.
They remain separately named and tabulated so this structural degeneracy is
explicit.  Pairing accuracy is also binned in frozen $m_{HH}$, Higgs-$p_T$,
selected-jet multiplicity, and extra-jet activity bins.

The frozen geometric comparator is decoded from each authoritative row's exact
`pairing` value, one of the three canonical perfect matchings.  Candidate jets
are stored in descending $p_T$ order, so `(j1,j2)(j3,j4)` is not a universal
baseline pairing.  The decoded pairing must reproduce the stored `mbb1` and
`mbb2` masses numerically before it can be compared with truth or the learned
assignment.

Classification reports weighted and unweighted AUC, ROC, signal/background
efficiency and rejection, yields, $S/B$, $S/\sqrt{B}$, effective statistics,
nominal $Z_A$, shared-nuisance systematics-aware $Z_A$, transferred-QCD
fraction, direct-QCD closure/nonclosure, and process composition.  Every
resampleable primary value uses the already frozen 1000-replica common
source-member registry with SHA-256
`37218e5531018ea2a79473bfbee43df657b309cf06d89751509e74b586d9dc29`.
All comparison replicas are aligned by replica ID.

Each summary records the nominal full-OOF value, bootstrap mean, median,
standard deviation, 16th/84th and 2.5th/97.5th percentiles, valid count, and
invalid reasons.  The paper convention is the asymmetric median 68% interval.
Single-head-minus-inclusive-BDT paired intervals are mandatory for weighted
AUC, nominal and systematics-aware $Z_A$, $S/B$, and background effective
statistics; the report states whether each interval includes zero.  Undefined
replicas remain undefined.  ROC, rejection, threshold-scan, binned-pairing, and
mass-resolution figures carry source-bootstrap bands or asymmetric bars, and
all plot-data tables include interval bounds, valid counts, and support flags.

## Stage gate

The stage may be called complete only after the full truth inventory, nested OOF
predictions, common-bootstrap summaries and paired differences, uncertainty
figures, visual inspection, targeted tests, environment/model manifests, and an
immutable external checkpoint all pass.  ROOT, Parquet, model weights, replica
payloads, logs, caches, and reference PDFs remain outside Git.  Validation/test
opening, the two-head model, and the unified comparison remain separately gated.
