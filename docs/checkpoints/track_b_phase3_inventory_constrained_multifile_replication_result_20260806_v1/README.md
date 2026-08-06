# Track B Phase 3 inventory-constrained replication result

This checkpoint freezes the successful `40J7Q-R1` seven-file Sophon
replication and compares it with the published four-file canary.

## Evidence

The combined evidence contains 11 files and 1,152 unweighted events:

- QCD: three files, 320 events
- ordinary ttbar: three files, 320 events
- Z+jets: three files, 320 events
- ZZ: two files, 192 events

All seven replication files passed size and Adler-32 identity checks, reached
128 valid events, completed all three model subprocesses, and were removed
from temporary storage. There were zero preprocessing failures.

## Diagnostics

The review recomputes file-level ensemble response statistics directly from
the event rows and reproduces the published per-file summaries. It freezes
file-to-file ranges, population standard deviations, threshold-fraction
ranges, model-agreement ranges, and canary-versus-replication comparisons.

No arbitrary stability acceptance threshold is introduced. The values are
descriptive unweighted released-model response diagnostics, not efficiencies,
calibrated probabilities, expected yields, AUC, significance, or sensitivity.

## Boundaries

Track-B physical weighting remains blocked. The full 44-file balanced,
46-file tail, and 90-selection-row studies remain unauthorized. No SPA-Net,
ParT, or JP-JEPA model is run or trained.

## Next gate

`40J7S_review_and_commit_inventory_constrained_replication_result`
