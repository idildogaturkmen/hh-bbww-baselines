# Train-fold and baseline benchmark contract

This checkpoint accepts the deterministic five-fold source-group map for the
resolved HH to 4b benchmark and freezes the order and governance of the model
comparison.

## Fold decision

The optimized map passes the physics-critical balance gates:

- total broad rows: max/min = 1.023325;
- signal broad rows: max/min = 1.081991;
- background broad rows: max/min = 1.026255;
- assignment-matchable rows: max/min = 1.009916;
- zero source-group leakage;
- every process represented by at least five source groups populates all folds.

Only the 23 train-only auxiliary-QCD source groups miss their provisional
row-balance gate, with max/min = 1.360775 versus the provisional threshold of
1.20. This is accepted as a documented waiver because the auxiliary population
is classification-only, receives no physical event weights, and is excluded
from physical yields and significance. Its learning contribution will be
controlled by the same frozen hierarchical comparison-weight or sampler
contract for every trainable classifier.

## Equal-comparison principle

Equal comparison means that every model uses the same immutable generated-event
accounting, train/validation/test roles, broad event universe, source-group fold
map, weight definitions, validation opening, test opening, operating-point
selection, and paired bootstrap. It does not require every architecture to use
identical representations.

## Baseline order

1. fixed legacy geometric cut, including R_HH < 34;
2. nested-OOF optimized cuts;
3. global 34-feature mass-aware BDT;
4. 30-feature mass-plane-blind BDT diagnostic;
5. historical low/high-m_HH categorized BDT with m_HH = 450 GeV;
6. separate dual-specialist categorized-BDT extension;
7. dense DNN;
8. LBN-DNN;
9. resolved SPA-Net studies only after the conventional baselines are frozen.

The SPA-Net stage will test the techniques in arXiv:2412.03819 through explicit
ablations: symmetric target assignment, partial-event target detection,
target-mass loss reweighting, and an event-classification head.

## Safety state

- validation payloads opened: 0;
- test payloads opened: 0;
- physical luminosity weights authorized: false;
- models trained during this freeze: 0.

## Next gate

Freeze the exact common geometric reconstruction, candidate feature schema,
hierarchical comparison-weight table, and primary physical-weight authorization.
Then run the cut, BDT, categorized-BDT, DNN, and LBN baseline ladder before
SPA-Net.
