# HH4b card-only configuration and convention freeze

The source-aware review showed that pilot and scale-out campaigns had matching process and
run-card fingerprints but different Pythia fingerprints because generator logs were mixed
with actual Pythia cards.

PN-c2b removes that ambiguity by constructing fingerprints only from:

- `proc_card_mg5.dat`;
- `run_card.dat`, excluding representative event count, run tag, and random seed;
- `pythia8_card_default.dat`, excluding operational event-count, output-file, input-file,
  and random-seed settings;
- `pythia_card_default.dat`.

Generator and Pythia logs are never part of configuration equivalence.

A process label is frozen as configuration-equivalent only when all its standard campaigns
collapse to one card-only fingerprint. This does not authorize sharing a physical cross
section or normalization denominator.

The gate also freezes 13 TeV beam provenance where exact run-card energies agree with the
frozen path tag, records explicit matrix-element or Pythia decay controls, and records
generator phase-space cuts. Numeric branching fractions, filter efficiencies,
normalization denominators, QCD stitching rules, and external cross sections remain
unauthorized.
