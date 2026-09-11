# HH4b triboson signed-weight transport audit

PN-c3b established that five representative standard samples have uniform-positive LHE
weights, while the `wwz_zbb` representative has signed weights. Its source LHE has 46,800
events, whereas each reconstructed WWZ shard has 5,200 events. Therefore the full LHE sum
is not the denominator of a single reconstructed shard or campaign.

PN-c3c audits every ROOT source bundle used by the six standard triboson campaigns:

- WWZ pilot and scaleout;
- WZZ pilot and scaleout;
- ZZZ pilot and scaleout.

There are 15 source bundles in total. Each bundle is transferred sequentially to temporary
non-workspace scratch, checked against the frozen size and SHA-256 from the source map, and
removed after use. Only the mapped Delphes ROOT member is extracted. The auditor opens the
Delphes tree and reads exactly one nominal event-record weight branch. With uproot's split
branch naming it explicitly prefers `Event/Event.Weight` over the distinct
`Weight/Weight.Weight` collection.

For each shard and campaign, the checkpoint records positive, negative, and zero counts,
the signed sum of nominal ROOT weights, the sum of squared weights, extrema, and the branch
type. Campaign ROOT-entry totals must reproduce the frozen manifest event totals.

Candidate Parquet files are not opened. Even a complete ROOT-level signed sum remains a
denominator candidate until the same per-event sign is proven to reach the candidate
physical-weight layer. This checkpoint does not authorize denominators, cross sections,
physical weights, yields, models, or thresholds.
