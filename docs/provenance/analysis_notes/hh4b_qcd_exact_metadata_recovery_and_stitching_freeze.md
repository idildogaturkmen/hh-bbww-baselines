# HH4b exact hard-QCD metadata recovery and stitching freeze

PN-c4f recovered exact small-provenance generator metadata for all 257 previously unresolved physical hard-QCD shards. The recovery opened seven small provenance members per bundle, opened no event payload, and recovered 2,470,000 generated events. Among the recovered shards, 169 have variable positive generator weights and 88 have uniform positive weights.

The recovery merge combines those records with the three already-complete production shards and the exact diagnostic test-fill record. Older partial PN-c4a identity rows for the 257 recovered tags are preserved in a dedicated supersession audit table and replaced by the exact PN-c4f records after independent ledger identity validation. The merged text-only metadata registry therefore covers all 261 QCD bundles and all 260 physical production shards without assigning a denominator, cross section, physical event weight, yield, model, or threshold.

PN-c4e then applies the frozen generator-level coefficient

`(N_s / N_b) * (sigmaGen_s / sumw_s) * w_is`

inside each of the eight exclusive pTHat intervals. The three nominal campaigns are treated as Monte Carlo extensions of the same bins rather than independent full-cross-section processes. The 10k test-fill shard is excluded. Direct stitched hard-QCD remains secondary closure/projection only and is not summed with the overlapping `qcd_bbbb_general` or `qcd_bbbb_iht400to600` families. The primary CMS-style multijet treatment remains the lower-b-tag simulation pseudo-data transfer.

A successful freeze must authorize all 260 production denominators, close all eight bins, retain zero unresolved issues, and record the need for immutable per-event generator-weight sidecars for every variable-weight production shard.
