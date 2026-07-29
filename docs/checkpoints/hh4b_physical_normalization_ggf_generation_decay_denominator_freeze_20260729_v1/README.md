# HH4b ggF generation, decay, and denominator freeze

This checkpoint freezes the two nonstandard ggF HH campaigns as 13 TeV LO HEFT samples generated with the process `p p > h h, (h > b b~), (h > b b~)`. The production payload is identical across the four submit descriptions and is matched by one checksum-frozen representative provenance JSON from each campaign.

The samples contain 100 shards of 1000 unweighted events. Their effective generator numerator is one and their full-campaign denominators are 20,000 and 80,000. Split-specific independent renormalization is forbidden.

Because the process card forces both Higgs bosons to `bb`, a future registry based on an inclusive `pp -> HH` reference cross section must multiply `BR(H -> bb)^2` exactly once. No numerical branching fraction or reference cross section is assigned here.

The generated kinematics are explicitly classified as an LO HEFT ML-training approximation, not precision ggF HH modeling. No ROOT, HepMC, LHE, Parquet, candidate, validation, or evaluation payload is opened.
