# Email paragraph for Harvey

At your literal thresholds (score > 0.9997, i.e. u=3.522878745...; and
score > 0.99997, i.e. u=4.522878745...), the compression/spikes you
noticed arise after the model, when the two-class softmax probability
gets rounded into a 32-bit float for storage -- the raw logit margin
underneath stays continuous and keeps almost every event distinguishable
(99.93% and 99.92% of surviving events retain a fully distinct logit
margin at the two thresholds respectively, even where 92-100% of them
share a stored probability with other events). The exact
threshold-crossing impact: at score > 0.9997, the float32-based count
and the raw-logit-based count agree exactly (9,170 = 9,170, zero
disagreements); at score > 0.99997, they differ by 2 events out of
6,368-6,370 (about 0.001% of signal) -- both real, boundary-adjacent
events whose true confidence is a hair above the line but whose stored
float32 score rounds a hair below it. All of this is measured on our
fixed 400,000-event numerical-audit cohort, where no QCD or ttbar event
reaches anywhere near either threshold; that is a statement about this
cohort, not a claim about the physical background yield in the much
larger production study, which is a separate exercise. None of this
says the model is calibrated, and it doesn't tell us 32-bit training was
the right choice -- it only tells us the stored probability's display
resolution runs out before the model's own discrimination does. For
visualizing or ranking events in this extreme tail, we'd recommend using
the raw logit margin (u_logit) instead of the stored probability; for
reproducing the existing frozen analysis, the existing FP32 thresholds
should stay exactly as they are.
