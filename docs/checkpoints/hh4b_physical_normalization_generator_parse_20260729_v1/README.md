# HH4b generator-provenance parse

This checkpoint parses the 547 checksum-verified provenance files from all 53 EOS process-campaign bundles. It records candidate process definitions, run-card settings, Pythia settings, generator versions, generator cross sections, event counts, possible weight sums, and QCD phase-space information.

All parsed quantities remain candidates. No generator cross section is promoted to an authoritative reference cross section, and no event-count candidate is promoted to a normalization denominator without a weight-strategy review.

The unresolved 270k-event local legacy ttbar campaign is excluded from the authoritative physical-normalization source policy. This does not change model-training membership or modify candidate data. Provenance-complete EOS ttbar campaigns remain included pending review.

No candidate Parquet, validation content, final-evaluation content, ROOT, HepMC, or LHE payload is opened. No physical yield or model threshold is calculated.
