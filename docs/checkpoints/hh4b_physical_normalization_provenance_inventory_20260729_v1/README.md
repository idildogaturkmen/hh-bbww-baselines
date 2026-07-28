# HH4b physical-normalization provenance inventory

This checkpoint inventories the exact source population and creates a fail-closed
normalization registry template. It intentionally assigns no cross sections and computes
no physical yields.

The inventory follows a CMS-like separation of responsibilities:

- HH signal and minor backgrounds: theory-normalized simulation after process-matched
  generator provenance and higher-order reference cross sections are reviewed.
- ttbar: theory-normalized simulation with a separately documented differential-modeling
  review.
- QCD multijet: lower-b-tag control-region prediction is the intended primary treatment;
  direct QCD simulation normalization is a secondary projection only after overlap and
  stitching closure.

The source registry contains 630 indivisible members, 5,000,000 generated background
events, and 200,000 generated signal events. Candidate parquet files, validation content,
and final-evaluation content were not opened.

## Blocking rule

No process is authorized for physical weighting until its generator definition, reference
cross section, branching-fraction convention, filter efficiency, normalization denominator,
and overlap policy are resolved with source paths and checksums.

## Next gate

Recover and checksum generator artifacts campaign by campaign without opening candidate
parquet content.
