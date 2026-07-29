# HH4b hard-QCD importance denominator and stitching audit

This checkpoint audits all 261 hard-QCD source identities and the available exact generator metadata. It never substitutes blank or representative-only values for missing shard-level `sigmaGen`, `sum_event_weights`, `sum_squared_event_weights`, or weight ranges.

The three physical campaigns are independent Monte Carlo extensions inside eight exclusive pTHat intervals. The 10k test-fill shard is excluded from nominal production. A future frozen event coefficient for event `i` in shard `s` and bin `b` has the form

`(N_s / N_b) * (sigmaGen_s / sumw_s) * w_i`.

That coefficient is authorized only after every production shard in the relevant interval has exact shard-level metadata. When the current checkpoint finds identity-only receipt metadata, it records the missing fields and leaves denominators and stitching fail-closed instead of attempting numeric aggregation.

Direct hard-QCD MC remains secondary closure/projection only and must not be summed with `qcd_bbbb_general` or `qcd_bbbb_iht400to600`. The primary CMS-style multijet architecture remains the lower-b-tag simulation pseudo-data transfer.

No ROOT, HepMC, LHE, Parquet, candidate, validation, or evaluation payload is opened. No external reference cross section, physical event weight, or yield is assigned.
