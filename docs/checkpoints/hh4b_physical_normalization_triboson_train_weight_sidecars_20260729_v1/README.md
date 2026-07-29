# HH4b train-only triboson ROOT event-weight sidecars

This checkpoint materializes one immutable sidecar row per Delphes ROOT event for the 12 train-only WWZ, WZZ, and ZZZ source shards. Each sidecar stores the ROOT event index, exact nominal generator weight, and sign from `Event/Event.Weight`.

The existing train candidate Parquet files are checksum-verified and opened only to read their `event` indices. Every candidate index is proven unique, in range, and exactly joinable to its source event-weight sidecar. Source candidate files are never modified.

The three validation candidate files remain unopened and receive no sidecars. Their denominators and physical-weight application remain unauthorized.

For the three train scaleout campaigns, the complete sidecar sums reproduce the PN-c3c signed ROOT sums, so those campaign denominators are frozen. External cross sections and physical weights are still unauthorized.
