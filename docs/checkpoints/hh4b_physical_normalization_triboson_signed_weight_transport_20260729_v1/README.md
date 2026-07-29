# HH4b triboson signed-weight transport audit

The standard LHE-weight canary found signed nominal weights in `wwz_zbb`. The representative WWZ source LHE contains 46,800 events while each reconstructed WWZ ROOT shard contains 5,200 events, so the full source-LHE sum cannot be used directly as the denominator of a reconstructed campaign.

This checkpoint verifies and opens only the 15 ROOT source bundles already mapped to the six WWZ, WZZ, and ZZZ pilot/scaleout campaigns. From each Delphes tree it reads exactly one nominal event-weight branch and records signed sums, squared sums, and sign counts. It verifies that campaign ROOT-entry totals equal the manifest-generated event totals.

Candidate Parquet content is not opened. ROOT-level denominator values remain candidates until per-event sign transport into the candidate weight layer is proven. No normalization denominator, reference cross section, physical event weight, yield, model, or threshold is authorized.
