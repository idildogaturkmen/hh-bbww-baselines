# HH4b source-aware generator-configuration review

The first generator-provenance parser deliberately favored recall. Its output showed why a
second, source-aware review is required: a Pythia-card `lhaid` setting was classified as a
run-card setting, while valid Pythia and beam-energy fields were not recovered.

This gate reparses the checksum-frozen provenance archive according to exact source type:

- `proc_card_mg5.dat` supplies process commands;
- `run_card.dat` supplies beam, event-count, PDF, matching, and generation-cut settings;
- Pythia cards and logs supply shower and forced-decay settings;
- JSON and text provenance supply QCD sampling and overlap evidence.

Campaign-equivalence fingerprints exclude volatile event-count and random-seed settings but
retain process, generation-cut, PDF, matching, shower, and decay configuration.

All equivalence groups and normalization conventions remain candidates. This gate does not
authorize cross sections, branching fractions, filter efficiencies, denominators, QCD
stitching, physical weights, yields, or model thresholds.
