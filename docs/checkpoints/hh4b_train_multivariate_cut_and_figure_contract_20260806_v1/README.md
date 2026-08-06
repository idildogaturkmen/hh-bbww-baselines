# Multivariate cut and paper-figure contract

This checkpoint supersedes the one-dimensional RHH-only contract for the
nominal optimized-cut baseline while preserving both the fixed RHH34 comparator
and the RHH-only scan as documented sub-studies.

## Search scope

- 5,200,000 generated events remain in global accounting.
- 3,799,873 generated training events define the train role.
- 1,042,397 authorized primary resolved training rows enter optimization.
- 128,675 auxiliary-QCD resolved rows are excluded.
- Validation and test remain sealed.

The frozen candidate space has 54 interpretable families formed from three
mutually exclusive mass selections, two candidate-b-tag requirements, and a
small predeclared pool of optional kinematic cuts. No family has more than four
continuous thresholds.

The search uses a deterministic empirical-quantile beam stage followed by exact
one-coordinate event-boundary refinement. It does not claim a globally
exhaustive Cartesian scan.

## Statistical policy

The primary endpoint minimizes inner-OOF background efficiency at signal
efficiency at least 0.585957. A statistical-only Asimov optimization is reported
as a secondary diagnostic and is not the nominal choice before the nuisance
model is frozen.

The final train-only metric uncertainty uses the existing 2,000-replica paired
source-group draw plan. A separate 200-replica full reoptimization diagnostic
measures cut-selection stability.

## Paper figures

No existing figure is moved in this checkpoint. Current figure files and
references are inventoried. The final migration must use `git mv`, update all
references atomically, and pass a reference audit.

New paper-quality cut figures are generated only after the multivariate result,
bootstrap, stability, checksum, and result-freeze gates all pass.

Existing figure files inventoried: 87
Existing figure references inventoried: 108
