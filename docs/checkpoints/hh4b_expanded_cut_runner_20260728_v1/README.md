# Expanded HH4b cut runner implementation

This checkpoint introduces a permanent, tested train-only runner for
the expanded HH4b cut baseline.

Protocol v1 is preserved unchanged. Protocol v2 supersedes its cut
definition because the prior exact train-only scan and the reviewed
mass-plane geometry use `r_hh_125_120`, whereas protocol v1 assigned
the optimized cut to `r_hh_125_125`.

The runner reports the reference signal and control regions and the
frozen optimized `r_hh_125_120 < 34` cut. It reuses the existing
hierarchical weight implementation from `hh4b_bdt_v1_common.py`.

The BDT models, features, and hyperparameters are unchanged.

No candidate data were opened and no model or score was produced during
this implementation commit.
