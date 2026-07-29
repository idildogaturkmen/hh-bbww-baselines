# HH4b hard-QCD importance denominator and stitching freeze

This checkpoint freezes exact generator denominators for all 260 physical hard-QCD production shards and excludes the single 10k diagnostic test-fill shard. All eight exclusive pTHat intervals are complete and all generator-coefficient closure checks pass.

The three physical campaigns are Monte Carlo extensions of the same eight bins. An event `i` from shard `s` in bin `b` receives the frozen generator-level coefficient

`(N_s / N_b) * (sigmaGen_s / sumw_s) * w_is`.

This gives each bin one event-count-weighted Pythia `sigmaGen` estimate rather than multiplying the cross section by the number of shards or campaign extensions. The direct stitched hard-QCD sample remains secondary closure/projection only and is not summed with `qcd_bbbb_general` or `qcd_bbbb_iht400to600`. The primary CMS-style multijet strategy remains the lower-b-tag simulation pseudo-data transfer.

Of the 260 physical production shards, 172 have variable positive generator weights and require immutable per-event generator-weight transport; 88 are uniform positive. This checkpoint opens no ROOT, HepMC, LHE, Parquet, candidate, validation, or evaluation payload. It assigns no external reference cross section, physical event weight, yield, model, or threshold.
