# HH4b expanded development cache repair overlay

This checkpoint freezes the non-destructive repair overlay required to
materialize the complete 63,386-row expanded-v2 development cache.

The 585-member development population contains 261 exact canonical-72
candidate products, 218 legitimate zero-candidate members, 97 ggF
candidate products with the complete canonical-72 schema plus three
provenance columns, and nine VBF candidate products stored remotely on
EOS.

The ggF products are projected to the canonical 72 columns in memory.
Their provenance columns are not used as model inputs and the source
Parquets are not rewritten. Zero-candidate members are represented as
empty canonical-72 tables inside the cache only.

The nine authoritative VBF candidate Parquets were copied from their
frozen EOS locations to ignored local runtime storage. Their frozen
SHA-256 identities, candidate row counts, canonical columns, Arrow
types, finite numeric values, and event-key uniqueness passed. No event
generation, detector simulation, or candidate reconstruction occurred.

No member from the 45-member evaluation population was opened. No
model was trained, no score was calculated, and no physical
normalization was performed.

The next gate is to commit this repair-overlay checkpoint and
materialize the complete common development cache.
