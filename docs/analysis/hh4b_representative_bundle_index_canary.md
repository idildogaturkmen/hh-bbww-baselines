# HH4b representative-bundle index canary

This gate validates archive schemas before indexing every production campaign. It covers
ggF HH4b, VBF HH4b, inclusive ttbar, heavy-flavor QCD, importance-sampled hard QCD,
and ttZ with a Z-to-bb label.

One bundle is present in temporary scratch at a time. Each is byte-size verified,
SHA-256 hashed, indexed through archive headers, and removed. Archive members are never
extracted. ROOT, Parquet, HepMC, and LHE payload contents are not opened.

Small regular non-payload members with provenance-like names are recorded as candidates
for a later explicit extraction allowlist. No cross section, branching fraction,
generator-weight denominator, threshold, or physical yield is assigned in this gate.
