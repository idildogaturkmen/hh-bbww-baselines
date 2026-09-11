# HH4b generator-provenance parse policy

Generator metadata and externally recommended theory normalizations are distinct layers.

This gate parses only the checksum-frozen production provenance:

- generator process definitions;
- run-card settings and beam energies;
- Pythia settings and forced-decay candidates;
- generator-version candidates;
- generator-reported cross-section candidates;
- generated-event and possible sum-of-weight candidates;
- QCD importance-sampling and phase-space candidates.

Parsed values are not automatically authoritative. Generator-reported cross sections may
be used as production provenance or for custom filtered samples only after process, decay,
filter, and overlap conventions are reviewed. They are not silently substituted for
LHCHWG, TOP++, or other externally reviewed predictions.

Likewise, generated event counts are not automatically used as normalization denominators.
The campaign must first be shown to be positive-weight and unweighted, or an explicit
signed sum of generator weights must be recovered.

The local 270k-event legacy ttbar campaign is excluded from the authoritative physical
source policy because its exact process and run cards remain unresolved. The model-training
population is unchanged. The provenance-complete EOS ttbar campaigns remain available for
physical normalization after review.

No physical weights or yields are produced in this gate.
