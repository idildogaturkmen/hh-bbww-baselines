# HH4b train four-b selected physical-yield closure

This checkpoint joins the immutable 30705-row train physical-weight sidecar
to the frozen four-b candidate payloads and calculates train-partition
physical-yield contributions.

It uses only previously frozen region definitions:

- baseline: all current four-b candidates;
- nominal optimized signal region: R_HH(125,120) < 34 GeV;
- CMS-reference signal region: R_HH(125,120) < 30 GeV;
- CMS-reference control annulus: 30 <= R_HH(125,120) < 55 GeV;
- CMS-reference outside region: R_HH(125,120) >= 55 GeV;
- frozen alternatives at 31.5 and 35.5 GeV;
- low/high mHH categories split at 450 GeV;
- inclusive >=4b, exactly-4b and >=5b categories.

Outputs include signed yields, sum of squared weights, effective event counts,
cancellation diagnostics, process/family/class breakdowns, S/B,
S/sqrt(B), and statistical-only Asimov significance.

Interpretation boundaries:

- These are train-partition contributions, not a full Run-2 selected-yield
  prediction, because validation and test/evaluation remain sealed.
- Direct stitched hard QCD is included only as a secondary closure/projection.
- The primary lower-b-tag simulation-pseudodata multijet transfer is not yet
  complete.
- No systematic-aware significance or expected limit is calculated.
- No model score is evaluated and no threshold is selected.
