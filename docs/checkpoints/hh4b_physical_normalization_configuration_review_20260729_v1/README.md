# HH4b source-aware generator-configuration review

This checkpoint corrects the intentionally broad PN-c1 parse by interpreting each file according to its provenance type. Run-card assignments are read only from `run_card.dat`; Pythia assignments are read only from Pythia cards and logs; hard-process commands are read only from `proc_card_mg5.dat`.

The review produces normalized configuration fingerprints and candidate campaign-equivalence groups while removing volatile event-count and random-seed fields from equivalence hashes. Equivalence is not yet authorized.

Beam-energy, forced-decay, generator-cross-section, denominator, and QCD sampling evidence remain candidates. No external theory cross section, branching fraction, filter efficiency, denominator, event weight, yield, or threshold is assigned.
