# HIG-24-015 train-only two-head SPA-Net contract

Status: frozen before PN-c7y model fitting.

## Scope and inputs

- Delphes simulation and the frozen c7s train split only.
- No validation, test, observed data, or Run-2 expected-sensitivity claim.
- Source groups, five outer folds, physical weights, exactly-$3b\rightarrow{\geq4b}$
  multijet projection, and direct-$\geq4b$ closure are inherited unchanged.
- The c7x candidate-four truth checkpoint is the only assignment-label source.
- Exactly four frozen candidate jets are encoded. The c7s table does not expose
  every selected jet as a variable-length tensor, so no all-selected-jet claim is made.
- Inputs are log1p(pt), eta, sin(phi), cos(phi), and log1p(mass).
  Candidate b-tag flags are excluded because they are constant for the development
  candidate four and do not describe the promoted exactly-$3b$ jet.

## Architecture

- Shared permutation-equivariant candidate-four transformer encoder:
  hidden width 64, two encoder layers, four attention heads, feed-forward width
  256, GELU activation, dropout 0.05, and final layer normalization.
- Assignment head: one shared symmetric pair scorer. The three logits are sums
  of scores for the three perfect four-jet matchings. Jet order, within-Higgs
  daughter order, and Higgs-candidate interchange symmetry are preserved.
- Classification head: a trainable MLP on concatenated mean and maximum pooled
  jet embeddings. Pooling makes the event score permutation invariant.
- The classifier is a genuine second neural-network head. No downstream XGBoost
  or other logically separate classifier defines the two-head result.

## Losses and optimization

- Weighted binary cross entropy is applied to every supported train row.
- Weighted three-way assignment cross entropy is applied only to uniquely
  matchable signal rows. Background, unmatched signal, and ambiguous signal are
  masked from the assignment loss.
- Combined loss is classification BCE plus lambda_assignment times assignment CE.
- Predefined assignment-loss weights are 0.25, 1.0, and 4.0.
- AdamW, learning rate 0.001, weight decay 0.0001, batch size 512, and 24 epochs
  are fixed. CPU deterministic algorithms and source-derived seeds are required.

## Nested loss-weight and operating-point selection

- For each outer source fold, all three weights are compared using the other four
  source folds as inner OOF folds.
- The frozen selection utility is one half inner weighted classification AUC plus
  one half inner exact pairing accuracy.
- Exact pairing accuracy is evaluated only on inner-held uniquely matchable signal.
- Maximum utility wins; exact ties select the smaller predefined weight.
- The outer held fold is not used for loss-weight selection.
- The event-score threshold is selected from the chosen weight's inner OOF scores
  at the frozen c7t weighted signal efficiency 0.6156418377159854.
- The chosen weight and threshold are then frozen for the corresponding outer fit.

## Evaluation and uncertainty

- Five deterministic source-group outer folds produce full train-only OOF scores
  and assignments for development, physical-projection, and direct-closure rows.
- Assignment metrics, classification metrics, loss tradeoffs, parameter count,
  repeated inference timing, training timing, gradient/loss-balance diagnostics,
  loss-weight stability, and fold stability are recorded.
- The exact common 1,000-replica c7t complete-source-member draw registry is used.
- Every resampleable primary value reports nominal, mean, median, standard
  deviation, p16, p84, p2.5, p97.5, and valid/invalid replica accounting.
- Replica-aligned paired differences are required for two-head minus inclusive BDT
  and two-head minus single-head SPA-Net. Pairing differences are required against
  single-head SPA-Net and the frozen geometric pairing where meaningful.
- Undefined systematics-aware replicas remain invalid and are never replaced by
  zero or interpolated.
- Applicable paper figures use 68% source-bootstrap bands or asymmetric error bars.

## Secondary categorized study

- Categorized two-head SPA-Net is a separate secondary result.
- It reuses the frozen nested category-construction algorithm from c7w on the
  two-head event score.
- Category count and boundaries are selected with inner source folds only.
- It is not part of the architectural definition above.
- Its primary metrics and comparisons require the same paired source-member
  uncertainty contract before the stage can be called complete.
