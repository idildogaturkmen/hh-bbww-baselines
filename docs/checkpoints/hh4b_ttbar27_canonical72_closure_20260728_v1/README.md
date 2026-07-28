# HH4b ttbar27 canonical-72 closure

This checkpoint closes the 27-member canonical-schema hold population.

The closed registry contains all 27 intended non-test ttbar members,
representing 270,000 generated events and 1,020 canonical candidate
rows. It combines 18 previously verified canonical products, one
retained-ROOT reconstruction, the shard003 exact-regeneration canary,
and seven reconstruction-only scaleout recoveries.

All 27 candidate products were reopened independently. Their candidate
checksums, row counts, canonical 72-column names, column order, Arrow
types, sample identities, event-key uniqueness, and finite numeric
content passed. No legacy candidate rows were admitted.

The parent source-coverage checkpoint remains fixed at 5,000,000
generated background events and 200,000 generated signal events across
630 source members. The sealed test population remains closed.

No reconstruction, generation, Delphes, scheduler, model, scoring,
sealed-test-content, or physical-normalization action occurred.

The next gate is to freeze the complete expanded candidate manifest,
metadata-grouped development split, and common feature-cache contract.
