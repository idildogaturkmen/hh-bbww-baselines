# HH4b hard-QCD importance denominator and stitching freeze

The remaining hard-QCD sample consists of three physical production extensions plus one
10k test-fill shard. The three physical campaigns cover the same eight exclusive generator
intervals:

\[
[50,75), [75,100), [100,200), [200,300),
[300,500), [500,700), [700,1000), [1000,\infty).
\]

The campaigns are not independent physics processes. Within each interval, their shards
must be pooled as repeated Monte Carlo extensions. Assigning the full Pythia `sigmaGen`
cross section independently to every campaign or every shard would multiply the predicted
yield.

For an event with nominal Pythia generator weight \(w_{is}\) in shard \(s\), inside bin
\(b\), PN-c4e freezes the generator-level cross-section contribution as

\[
\left(\frac{N_s}{N_b}\right)
\left(\frac{\sigma^{\mathrm{Pythia}}_s}{\sum_j w_{js}}\right)
w_{is}.
\]

Here \(N_b\) is the sum of generated events across all three production extensions in that
bin. Summing this expression over every event in the bin gives the event-count-weighted
mean of the independent shard `sigmaGen` estimates and does not duplicate the cross
section.

The test-fill shard is excluded from nominal production. Direct stitched hard-QCD remains
a secondary closure or projection. It is not summed with the overlapping
`qcd_bbbb_general` or `qcd_bbbb_iht400to600` families. The primary CMS-style multijet
architecture remains the lower-b-tag simulation pseudo-data transfer.

This gate reads only previously checksum-frozen text metadata. It opens no event payload or
candidate content and assigns no external cross section, physical event weight, or yield.

The fixed v2 implementation explicitly distinguishes exact shard-level generator metadata
from identity-only receipts. A shard without `sigma_gen_pb`, `sum_event_weights`,
`sum_squared_event_weights`, and its weight range remains unresolved. Such rows are written
to `unresolved_qcd_normalization_issues.tsv`; they are never converted to empty floats and
never enter a coefficient or stitching closure calculation.
