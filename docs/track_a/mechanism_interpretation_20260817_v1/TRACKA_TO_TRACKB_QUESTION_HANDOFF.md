# Track-A to Track-B Question Handoff

This document names the exact, narrow question Track A's mechanism benchmark leaves open, for Track
B to answer independently on its own data. It authorizes nothing on Track B and reads no Track-B
event, holdout, or `inference_qcd` sample -- it only states the question in the form Track A's own
findings motivate.

## The question

> **"When sufficient QCD statistics and continuous flavor information are available, how much
> additional rejection comes from continuous flavor and explicit pairwise event structure?"**

## Why Track A cannot answer this itself

Per `TRACKA_MECHANISM_INTERPRETATION_20260817.md` section 4(b)-(c):

- **Finite QCD support (b)**: Track A's own inclusive-QCD Delphes sample runs out of effective
  statistics (`qcd_Neff<10`) at `epsilon_B=0.001` for both newly-trained models -- identically, not
  as a function of which classifier is used. No amount of classifier improvement changes where this
  floor sits; only more/better-targeted QCD generation or a different background-estimation strategy
  would.
- **Representation limitation (c)**: Track A's b-tagging input is Delphes' binary b-tag flag, not a
  continuous flavor-tagger score. The CMS-inspired transformer's pairwise-attention mechanism is
  present and measurably effective in Track A (`TRACKA_MECHANISM_INTERPRETATION_20260817.md`
  section 6: a real, working-point-dependent improvement over the dense DNN), but Track A has no way
  to test whether *continuous* flavor information -- which Track B's own training-QCD development
  population may carry -- would add further separation on top of that already-measured pairwise
  mechanism.

## What this handoff does NOT authorize

- No Track-B training is authorized by this document.
- No Track-B event, holdout (`holdout_A`, `holdout_B`), or `inference_qcd` sample is read, referenced,
  or estimated here.
- This does not imply Track A's mechanism result is invalid or superseded -- section 6 of the
  interpretation document stands on its own as a Track-A-internal finding, independent of whatever
  Track B eventually finds.

## What a future, separately-authorized Track-B study would need to freeze first

Consistent with this project's standing discipline (every generation/training action gets its own
execution contract before running): if this question is pursued on Track B, a dedicated execution
contract naming Track B's exact population, fold convention, feature definitions (continuous flavor
score vs. Track A's binary b-tag), and comparison protocol against this document's Track-A numbers
would need to be written and frozen **before** any Track-B training runs -- exactly as
`EXECUTION_CONTRACT.md` and `CONTINUATION_EXECUTION_CONTRACT.md` were for this benchmark.
