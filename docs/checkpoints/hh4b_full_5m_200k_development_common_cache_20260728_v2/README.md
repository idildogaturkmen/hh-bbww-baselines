# HH4b expanded-v2 common development cache

This checkpoint validates the complete common candidate and feature
cache for the expanded-v2 development population.

The cache was constructed from the frozen 585-member repair overlay:
261 exact canonical products, 218 zero-candidate members, 97 ggF
products projected from canonical-72 plus three provenance columns,
and nine authoritative VBF candidate products recovered from EOS.

The resulting cache contains 63,386 candidate rows: 53,162 train rows
and 10,224 validation rows. It contains 51,803 background rows and
11,583 signal rows. All nonempty candidate identities and row counts
were verified against the frozen manifest.

The cache retains the canonical 72 columns, adds nine deterministic
absolute-value features, and adds eight registry metadata columns. The
three cache artifacts remain in ignored runtime storage; their exact
paths, SHA-256 identities, sizes, schemas, and row counts are recorded
in this checkpoint.

No evaluation candidate was opened. No model was trained, no score or
threshold was produced, no source candidate was modified, and no
physical normalization was performed.

The next gate is to commit this cache-validation checkpoint and freeze
the common balancing and model-execution contract for the cut and BDT
baselines.
