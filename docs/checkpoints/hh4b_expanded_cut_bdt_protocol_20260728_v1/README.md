# HH4b expanded cut and BDT baseline protocol

This checkpoint freezes the common execution protocol for the expanded
5M-background/200k-signal candidate-level cut and BDT baselines.

All baselines use the same immutable 63,386-row development cache:
53,162 train rows and 10,224 validation rows. The 45-member final
evaluation population remains closed.

The cut baseline is fixed externally at R_HH < 34 GeV. No threshold is
selected from the new train or validation population.

Four BDT components are frozen: a global mass-aware model, a global
explicit mass-plane-blind ablation, and low-/high-mHH categorized
mass-aware models separated at mHH = 450 GeV. Their hyperparameters
come from the previously completed grouped train-only cross-validation
studies.

Development weights are not physical normalization. Train and
validation are balanced independently by class, then signal mode or
background family, then candidate-bearing source member.

BDT operating thresholds will be derived from five-fold grouped
out-of-fold train predictions. The primary target is the weighted train
signal efficiency of the externally frozen cut. Validation is evaluated
exactly once after every model and operating threshold is frozen.

The global mass-aware BDT is the a-priori nominal baseline. Validation
results cannot trigger model refitting, threshold changes, feature
changes, or hyperparameter reoptimization.

No model, score, threshold, validation prediction, evaluation access,
scheduler action, or physical normalization occurred in this freeze.

The next gate is the single authorized expanded cut/BDT baseline
execution.
