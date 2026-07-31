# Hard-QCD global Run-2 physical-weight contract freeze

This checkpoint permanently binds the special hard-QCD importance-sampling
normalization to the frozen 138 fb^-1 Run-2-equivalent luminosity contract.

The nominal hard-QCD population contains three production extensions, 260
shards, 2,500,000 events, and eight exclusive pTHat intervals. Extensions are
pooled inside each bin and must never receive independent full-bin cross
sections.

For production shard s and event i:

    c_s = (N_s / N_b) * (sigmaGen_s / sumw_s)
    W_i = 138000 pb^-1 * c_s * w_gen(i,s)

The existing per-shard cross-section coefficients close to the frozen
event-count-weighted cross section in every exclusive bin. The 172
variable-weight shards require per-event generator-weight transport; 88 shards
are uniform positive.

The 10,000-event diagnostic testfill has a nominal coefficient and yield of
zero. The 300,000 events in qcd_bbbb_general and qcd_bbbb_iht400to600 remain
part of the raw inventory but also have zero nominal contribution when
qcd_hardqcd is used.

The lower-b-tag simulation-pseudodata transfer remains the primary multijet
architecture. Direct hard-QCD MC remains secondary closure/projection only.

No event payload is opened and no coefficient is applied to an event here.
Event-level physical weights and physical yields remain unauthorized.

PN_C7C_HARD_QCD_GLOBAL_WEIGHT_CONTRACT_FREEZE_PASS
